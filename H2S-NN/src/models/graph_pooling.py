from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch_geometric.utils import scatter, softmax

from src.models.pooling_utils import pool_edge_stalks_to_nodes


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class GraphPooling(nn.Module, ABC):
    """Graph-level readout for graph classification.

    Pools the shared sheaf-diffusion backbone's per-node/per-edge state into one
    logits vector per graph in a PyG-batched disjoint union of many small graphs.
    Swappable via build_graph_pooling() so pooling strategy is an ablation axis
    (mirrors HetSheaf's own Table 1 pooling ablation), not a single hardcoded design.
    """

    @abstractmethod
    def forward(
        self,
        x_star: torch.Tensor,
        h0: torch.Tensor,
        s_star: torch.Tensor,
        delta0_x_star: torch.Tensor,
        delta0_h0: torch.Tensor,
        edge_index: torch.Tensor,
        F_src: torch.Tensor,
        F_tgt: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Returns [num_graphs, num_classes] logits."""
        raise NotImplementedError


class MeanPool(GraphPooling):
    """Baseline floor: plain mean pooling over node embeddings, directly analogous
    to the legacy CSNN baseline's global_mean_pool (models/coopshv_model.py)."""

    def __init__(self, d: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.head = _build_mlp(d, hidden_dim, num_classes, dropout)

    def forward(
        self,
        x_star: torch.Tensor,
        h0: torch.Tensor,
        s_star: torch.Tensor,
        delta0_x_star: torch.Tensor,
        delta0_h0: torch.Tensor,
        edge_index: torch.Tensor,
        F_src: torch.Tensor,
        F_tgt: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        pooled = scatter(x_star, batch, dim=0, dim_size=num_graphs, reduce="mean")
        return self.head(pooled)


class _StackedDescriptorPool(GraphPooling):
    """Shared column-stacking + per-graph aggregation for GramPool/S2Pool.

    H2SNN's node/edge tensors are flat [N,d]/[E,d] (no separate channel dimension),
    but a basis-free descriptor needs more than one column or it degenerates to an
    uninformative constant. Stack existing per-node/per-edge quantities as columns
    instead of inventing new ones: node H = [x_star, h0, transported_edges] -> [N,d,3];
    edge D = [delta0_x_star, delta0_h0, s_star] -> [E,d,3] (delta0_x_star/delta0_h0
    are already computed by encode() for the smoothness loss — reused, not
    recomputed). Subclasses only differ in how a [*, d, 3] stack becomes a small
    [*, 3, 3] descriptor.

    The two stacks are gauge-covariant for different reasons. The edge stack needs no
    work: delta0_x_star, delta0_h0, and s_star all already live in the *same* edge
    stalk, so a gauge change moves all three by that edge's R_e. The node stack does
    need work, because the incident-edge column starts life in the edge stalks: it is
    built by `pool_edge_stalks_to_nodes`, which pulls each edge state back through its
    own endpoint restriction map before averaging. Without that pullback the third
    column would not co-transform with x_star/h0 and the descriptor would depend on the
    arbitrary choice of stalk bases.
    """

    def __init__(self, d: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        descriptor_dim = 3 * 3
        self.node_mlp = _build_mlp(descriptor_dim, hidden_dim, hidden_dim, dropout)
        self.edge_mlp = _build_mlp(descriptor_dim, hidden_dim, hidden_dim, dropout)
        self.head = _build_mlp(2 * hidden_dim, hidden_dim, num_classes, dropout)

    def _descriptor(self, stacked: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        x_star: torch.Tensor,
        h0: torch.Tensor,
        s_star: torch.Tensor,
        delta0_x_star: torch.Tensor,
        delta0_h0: torch.Tensor,
        edge_index: torch.Tensor,
        F_src: torch.Tensor,
        F_tgt: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        pooled_edges = pool_edge_stalks_to_nodes(
            s_star=s_star,
            edge_index=edge_index,
            F_src=F_src,
            F_tgt=F_tgt,
            num_nodes=x_star.size(0),
        )
        node_stack = torch.stack([x_star, h0, pooled_edges], dim=-1)  # [N, d, 3]
        edge_stack = torch.stack([delta0_x_star, delta0_h0, s_star], dim=-1)  # [E, d, 3]

        node_descriptor = self._descriptor(node_stack).flatten(start_dim=1)  # [N, 9]
        edge_descriptor = self._descriptor(edge_stack).flatten(start_dim=1)  # [E, 9]

        z = self.node_mlp(node_descriptor)  # [N, hidden_dim]
        w = self.edge_mlp(edge_descriptor)  # [E, hidden_dim]

        edge_batch = batch[edge_index[0]]
        tau0 = scatter(z, batch, dim=0, dim_size=num_graphs, reduce="mean")
        tau1 = scatter(w, edge_batch, dim=0, dim_size=num_graphs, reduce="mean")

        return self.head(torch.cat([tau0, tau1], dim=-1))


class GramPool(_StackedDescriptorPool):
    """O(d)-invariant descriptor: the plain Gram matrix H^T H of a view stack.

    Under an orthogonal stalk gauge H -> Q H the Gram is exactly invariant, with no
    pseudo-inverse and no whitening needed, and — unlike S2Pool's projector — it keeps
    norm/scale information rather than discarding it. It is in fact a *complete*
    invariant of the ordered view stack modulo O(d): H^T H = H'^T H' iff H' = Q H for
    some Q in O(d), so nothing but the gauge is lost.

    Note what the invariance does and does not assume. The requirement is on the *gauge
    group* (Q^T Q = I), not on the restriction maps: `use_orthogonal: True` produces
    conformal-orthogonal maps F = c*Q with a learned scale c, not orthogonal ones, and
    the argument holds for arbitrary F either way — see `pool_edge_stalks_to_nodes`.
    """

    def _descriptor(self, stacked: torch.Tensor) -> torch.Tensor:
        return torch.bmm(stacked.transpose(-2, -1), stacked)  # [*, 3, 3]


class S2Pool(_StackedDescriptorPool):
    """Basis-free descriptor per S2Pool_proposal.md: P = X^T (X X^T)^+ X.

    DEGENERATE — do not use as a baseline without fixing it first. The intent was a
    descriptor invariant to any invertible gauge, not merely orthogonal ones. What it
    actually computes is the constant identity matrix whenever the stack has full column
    rank, which is the normal case here (3 views, d >= 3): writing the thin SVD
    X = U S V^T gives

        P = V S U^T (U S^2 U^T)^+ U S V^T = V S S^-2 S V^T = V V^T = I_3 .

    So P carries no information about X at all, every graph receives the same logits,
    and the GL(d)-invariance the class advertises is invariance-by-constancy. Measured:
    max |P - I_3| = 3.6e-07 on random full-rank stacks, and |logits(batch A) -
    logits(batch B)| = 1.5e-08 where GramPool/SheafCovPool both differ by ~2e-01.

    This is why an invariance test alone never caught it — a constant function passes
    every invariance test. A meaningful descriptor here would have to keep the singular
    values (as GramPool's H^T H does) instead of dividing them out.
    """

    def _descriptor(self, stacked: torch.Tensor) -> torch.Tensor:
        gram = torch.bmm(stacked, stacked.transpose(-2, -1))  # [*, d, d]
        gram_pinv = torch.linalg.pinv(gram, rtol=1e-5)
        return torch.bmm(torch.bmm(stacked.transpose(-2, -1), gram_pinv), stacked)  # [*, 3, 3]


def _sym_inv_sqrt(mat: torch.Tensor, eps: float) -> torch.Tensor:
    """Inverse matrix square-root of a batch of symmetric PSD matrices [*, d, d]."""
    evals, evecs = torch.linalg.eigh(mat)
    inv_sqrt = evals.clamp_min(eps).rsqrt()
    return evecs @ torch.diag_embed(inv_sqrt) @ evecs.transpose(-2, -1)


class SheafPool(GraphPooling):
    """An ADAPTED SheafPool readout, not the published one — refer to it as such.

    HetSheaf's SheafPool (arXiv:2409.08036, Sec. 4) is reimplemented here on the H2SNN
    backbone as an apples-to-apples readout-form BASELINE. Because it is fed H2SNN's
    node *and* edge three-view stacks rather than the node-only inputs of the original,
    a win over it is evidence about the pooling mechanism on matched inputs, never
    evidence of superiority over the published method.

    It applies the five SheafPool stages — per-node whitening -> anchor alignment ->
    gauge-invariant attention -> stalk-space pooling + late receive-only token ->
    channel-wise energy — to the SAME stacked node/edge descriptors GramPool/S2Pool use.
    Inputs are held fixed, so a GramPool-vs-SheafPool comparison isolates the pooling
    MECHANISM (simple Gram energies vs whitening+alignment+attention), answering "is our
    trace-free Gram readout as good as SheafPool's sophisticated one?".
    """

    def __init__(self, d: int, hidden_dim: int, num_classes: int, dropout: float, eps: float = 1e-3) -> None:
        super().__init__()
        self.d = d
        self.C = 3  # number of stacked views used as stalk channels
        self.eps = eps
        self.node_anchor = nn.Parameter(torch.randn(d, self.C) * 0.1)
        self.edge_anchor = nn.Parameter(torch.randn(d, self.C) * 0.1)
        self.node_scorer = nn.Sequential(nn.Linear(self.C, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.edge_scorer = nn.Sequential(nn.Linear(self.C, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.node_token = nn.Parameter(torch.zeros(d, self.C))
        self.edge_token = nn.Parameter(torch.zeros(d, self.C))
        self.node_phi = nn.Linear(2 * self.C, self.C)
        self.edge_phi = nn.Linear(2 * self.C, self.C)
        self.node_ln = nn.LayerNorm(self.C)
        self.edge_ln = nn.LayerNorm(self.C)
        self.head = _build_mlp(2 * self.C, hidden_dim, num_classes, dropout)

    def _pool(
        self,
        stack: torch.Tensor,       # [M, d, C]
        group: torch.Tensor,       # [M] graph id per row
        num_graphs: int,
        anchor: torch.Tensor,
        scorer: nn.Module,
        token: torch.Tensor,
        phi: nn.Module,
        ln: nn.Module,
    ) -> torch.Tensor:
        if stack.size(0) == 0:
            return stack.new_zeros((num_graphs, self.C))
        eye = torch.eye(self.d, device=stack.device, dtype=stack.dtype)
        gram = torch.matmul(stack, stack.transpose(-2, -1)) + self.eps * eye  # [M,d,d]
        z = torch.matmul(_sym_inv_sqrt(gram, self.eps), stack)                # [M,d,C] whitened
        m = torch.matmul(z, anchor.t())                                       # [M,d,d]
        u_svd, _, vh_svd = torch.linalg.svd(m)                                # orthogonal Procrustes
        rot = torch.matmul(u_svd, vh_svd)                                     # [M,d,d]
        z_aligned = torch.matmul(rot.transpose(-2, -1), z)                    # [M,d,C]
        energies = torch.diagonal(torch.matmul(z.transpose(-2, -1), z), dim1=-2, dim2=-1)  # [M,C]
        scores = scorer(energies)                                             # [M,1]
        alpha = softmax(scores, group, num_nodes=num_graphs)                  # [M,1] per-graph
        pooled = scatter(alpha.unsqueeze(-1) * z_aligned, group, dim=0, dim_size=num_graphs, reduce="sum")  # [G,d,C]
        tok = token.unsqueeze(0).expand(num_graphs, -1, -1)                   # [G,d,C]
        updated = ln(phi(torch.cat([tok, pooled], dim=-1)))                   # [G,d,C]
        return torch.diagonal(torch.matmul(updated.transpose(-2, -1), updated), dim1=-2, dim2=-1)  # [G,C]

    def forward(
        self,
        x_star: torch.Tensor,
        h0: torch.Tensor,
        s_star: torch.Tensor,
        delta0_x_star: torch.Tensor,
        delta0_h0: torch.Tensor,
        edge_index: torch.Tensor,
        F_src: torch.Tensor,
        F_tgt: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        pooled_edges = pool_edge_stalks_to_nodes(
            s_star=s_star,
            edge_index=edge_index,
            F_src=F_src,
            F_tgt=F_tgt,
            num_nodes=x_star.size(0),
        )
        node_stack = torch.stack([x_star, h0, pooled_edges], dim=-1)          # [N, d, 3]
        edge_stack = torch.stack([delta0_x_star, delta0_h0, s_star], dim=-1)  # [E, d, 3]
        edge_batch = batch[edge_index[0]]
        g_node = self._pool(node_stack, batch, num_graphs, self.node_anchor, self.node_scorer, self.node_token, self.node_phi, self.node_ln)
        g_edge = self._pool(edge_stack, edge_batch, num_graphs, self.edge_anchor, self.edge_scorer, self.edge_token, self.edge_phi, self.edge_ln)
        return self.head(torch.cat([g_node, g_edge], dim=-1))


class SheafCovPool(GraphPooling):
    """Sheaf Covariance Pooling -- the finalized pooling method for the H2SNN paper.

    Gauge-invariant first- AND second-order pooling over per-node/per-edge
    Gram descriptors (no whitening). Supersedes the earlier THS-Pool
    hierarchical/coarsening line (relearn, injection, Galerkin coarsening,
    spectral descriptor), all of which tied or lost against a plain invariant
    readout on TU graphs -- see docs/sheafcovpool_design_log.md Sec. 0.

    Shares GramPool's key property: under an orthogonal stalk gauge the plain Gram
    H^T H is already exactly invariant, with no pseudo-inverse or whitening needed.
    (The condition is on the gauge group, not on the maps -- `use_orthogonal: True`
    yields conformal-orthogonal F = c*Q.) SheafPool's whitening step (Z = S^-1/2 H)
    requires enough stalk channels to whiten across and collapses to a near-constant
    descriptor on H2SNN's 3-view stacks (diagnosed via the Phase-0 readout sweep:
    ~0.26 macro-F1 on MUTAG, below majority baseline) -- this pool avoids that
    failure mode entirely.

    Both view stacks must be *sections in a common stalk* for that argument to apply.
    The edge stack already is. The node stack's incident-edge column is made one by
    `pool_edge_stalks_to_nodes`, which transports each edge state back through its own
    endpoint restriction map before averaging; summing raw edge coordinates from
    different edge stalks, as an earlier version did, is not a well-defined sheaf
    operation and silently broke the invariance this class claims.

    Adds two ingredients GramPool lacks: a learned projection of the Gram
    descriptor scored by gauge-invariant attention, and the ACROSS-NODE (and
    across-edge) COVARIANCE of that projection, not just its attention-weighted
    mean -- capturing the spread/distribution of node and edge states within a
    graph, which a mean-only readout cannot represent.
    """

    def __init__(self, d: int, hidden_dim: int, num_classes: int, dropout: float, feat_dim: int = 8) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        descriptor_dim = 3 * 3
        self.node_feat = _build_mlp(descriptor_dim, hidden_dim, feat_dim, dropout)
        self.edge_feat = _build_mlp(descriptor_dim, hidden_dim, feat_dim, dropout)
        self.node_scorer = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.edge_scorer = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        num_tri = feat_dim * (feat_dim + 1) // 2
        summary_dim = 2 * (feat_dim + num_tri)  # node + edge, each mean [F] + cov upper-tri [F(F+1)/2]
        self.head = _build_mlp(summary_dim, hidden_dim, num_classes, dropout)
        triu_row, triu_col = torch.triu_indices(feat_dim, feat_dim)
        self.register_buffer("triu_row", triu_row, persistent=False)
        self.register_buffer("triu_col", triu_col, persistent=False)

    def _pool_moments(
        self,
        u: torch.Tensor,       # [M, F]
        scorer: nn.Module,
        group: torch.Tensor,   # [M] graph id per row
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tri = self.triu_row.numel()
        if u.size(0) == 0:
            return u.new_zeros((num_graphs, self.feat_dim)), u.new_zeros((num_graphs, num_tri))
        scores = scorer(u)  # [M, 1]
        alpha = softmax(scores, group, num_nodes=num_graphs)  # [M, 1], sums to 1 per graph
        mu = scatter(alpha * u, group, dim=0, dim_size=num_graphs, reduce="sum")  # [G, F]
        centered = u - mu[group]
        weighted_outer = alpha.unsqueeze(-1) * centered.unsqueeze(-1) * centered.unsqueeze(-2)  # [M, F, F]
        cov = scatter(weighted_outer, group, dim=0, dim_size=num_graphs, reduce="sum")  # [G, F, F]
        cov_flat = cov[:, self.triu_row, self.triu_col]  # [G, F(F+1)/2]
        return mu, cov_flat

    def forward(
        self,
        x_star: torch.Tensor,
        h0: torch.Tensor,
        s_star: torch.Tensor,
        delta0_x_star: torch.Tensor,
        delta0_h0: torch.Tensor,
        edge_index: torch.Tensor,
        F_src: torch.Tensor,
        F_tgt: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        pooled_edges = pool_edge_stalks_to_nodes(
            s_star=s_star,
            edge_index=edge_index,
            F_src=F_src,
            F_tgt=F_tgt,
            num_nodes=x_star.size(0),
        )
        node_stack = torch.stack([x_star, h0, pooled_edges], dim=-1)          # [N, d, 3]
        edge_stack = torch.stack([delta0_x_star, delta0_h0, s_star], dim=-1)  # [E, d, 3]

        node_gram = torch.bmm(node_stack.transpose(-2, -1), node_stack).flatten(1)  # [N, 9]
        edge_gram = torch.bmm(edge_stack.transpose(-2, -1), edge_stack).flatten(1)  # [E, 9]

        u_node = self.node_feat(node_gram)  # [N, F]
        u_edge = self.edge_feat(edge_gram)  # [E, F]

        edge_batch = batch[edge_index[0]]
        mu_n, cov_n = self._pool_moments(u_node, self.node_scorer, batch, num_graphs)
        mu_e, cov_e = self._pool_moments(u_edge, self.edge_scorer, edge_batch, num_graphs)

        summary = torch.cat([mu_n, cov_n, mu_e, cov_e], dim=-1)
        return self.head(summary)


# Backward-compatible public alias for configurations and imports created while the
# method was briefly named SheafConvPool. The name is Sheaf *Covariance* Pooling: the
# module performs no convolution, and the across-node/edge covariance is exactly what
# distinguishes it from GramPool.
SheafConvPool = SheafCovPool


def build_graph_pooling(
    pooling_type: str,
    d: int,
    hidden_dim: int,
    num_classes: int,
    dropout: float,
    sheafcov_feat_dim: int = 8,
) -> GraphPooling:
    if pooling_type == "meanpool":
        return MeanPool(d=d, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
    if pooling_type == "grampool":
        return GramPool(d=d, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
    if pooling_type == "s2pool":
        return S2Pool(d=d, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
    if pooling_type == "sheafpool":
        return SheafPool(d=d, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
    if pooling_type in {"sheafcov", "sheafconv"}:
        return SheafCovPool(
            d=d,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            feat_dim=sheafcov_feat_dim,
        )
    raise ValueError(f"Unsupported pooling_type '{pooling_type}'.")
