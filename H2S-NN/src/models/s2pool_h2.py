"""THS-Pool (Typed Hierarchical Sheaf Pooling) — V1 (lean).

Select–Reduce–Connect pooling for the H2SNN backbone (graph classification):

  1. Descriptors : O(d)-invariant per-node sheaf energies (node/edge/face).
  2. Select      : type-aware Top-K per node type, score-gated.
  3. Reduce/Conn : induced typed sub-complex (edge survives iff both ends survive).
  4. Re-diffuse  : coarse complex re-entered through H2SNN.encode_from_stalk (V1 re-learns maps).
  5. Readout     : per-rank (node/edge/face), per-type/relation invariant summaries, per level.
  6. Classify    : concat level tokens -> MLP -> logits (cross-entropy only).

V1 re-learns the coarse sheaf (the geometry is discarded and rebuilt each level). V1.5/V2 (carry the
surviving maps / compose along dropped paths) are the planned sheaf-true extensions; see
docs/sheafcovpool_explainer.md and docs/sheafcovpool_design_log.md.

The multi-level loop lives in `run()`, driven by two callbacks the trainer supplies so this module does
not have to hold a reference to the backbone: `encode_from_stalk` and `build_faces`.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import torch
from torch import nn
from torch_geometric.utils import scatter

from src.models.coboundary_ops import CoboundaryOps

EPS = 1e-6


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class CoarseComplex(NamedTuple):
    h0: torch.Tensor          # [N_c, d] gated coarse node stalks
    node_type: torch.Tensor   # [N_c]
    edge_index: torch.Tensor  # [2, E_c]
    edge_type: torch.Tensor   # [E_c]
    batch: torch.Tensor       # [N_c]
    perm: torch.Tensor        # [N_c] kept fine-node ids
    F_src: torch.Tensor       # [E_c, d, d] surviving fine src maps (for V1.5 injection)
    F_tgt: torch.Tensor       # [E_c, d, d] surviving fine tgt maps


def _standardize_per_graph(feat: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Per-graph z-score each column of `feat` [M, F] using segment mean/std over `batch`.

    The mean/std are DETACHED and the std is floored: they are a normalizer for the downstream score,
    not quantities to back-propagate through. Back-propagating through 1/std blows up when a graph has
    near-zero feature variance (e.g., all nodes of a type identical) — the source of the ENZYMES NaN.
    """
    mean = scatter(feat, batch, dim=0, dim_size=num_graphs, reduce="mean")  # [G, F]
    centered = feat - mean[batch]
    var = scatter(centered * centered, batch, dim=0, dim_size=num_graphs, reduce="mean")  # [G, F]
    scale = (var[batch].sqrt() + EPS).clamp_min(1e-2).detach()
    return (feat - mean[batch].detach()) / scale


def _topk_per_group(score: torch.Tensor, group: torch.Tensor, ratio: float, num_groups: int) -> torch.Tensor:
    """Boolean keep-mask [M]: top ceil(ratio * size) elements within each group (>=1 per non-empty group)."""
    # order rows by (group asc, score desc within group) via two stable sorts
    order_by_score = torch.argsort(score, descending=True, stable=True)
    order_by_group = torch.argsort(group[order_by_score], stable=True)
    idx = order_by_score[order_by_group]  # rows sorted group-major, score-desc within group
    counts = torch.bincount(group, minlength=num_groups)  # [num_groups]
    offsets = torch.zeros(num_groups, dtype=torch.long, device=score.device)
    offsets[1:] = counts.cumsum(0)[:-1]
    within_rank = torch.arange(score.size(0), device=score.device) - offsets[group[idx]]
    k_per = (counts.float() * ratio).ceil().clamp(min=1).long()  # [num_groups]
    keep_sorted = within_rank < k_per[group[idx]]
    mask = torch.zeros(score.size(0), dtype=torch.bool, device=score.device)
    mask[idx[keep_sorted]] = True
    return mask


class S2PoolH2(nn.Module):
    def __init__(
        self,
        d: int,
        num_node_types: int,
        num_relations: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        num_levels: int = 2,
        ratio: float = 0.5,
        inject_maps: bool = False,
    ) -> None:
        super().__init__()
        self.d = d
        self.num_node_types = num_node_types
        self.num_relations = num_relations
        self.num_levels = num_levels
        self.ratio = ratio
        self.inject_maps = inject_maps  # V1.5: reuse surviving fine maps instead of re-learning

        # scaled-log-degree view parameters (MVPool's structure view)
        self.deg_a = nn.Parameter(torch.ones(1))
        self.deg_b = nn.Parameter(torch.zeros(1))
        # type-specific node scorer: 5 invariant features -> importance in [0,1]
        self.score_mlps = nn.ModuleList(
            [_build_mlp(5, hidden_dim, 1, dropout) for _ in range(num_node_types)]
        )
        # per-level readout: [T*4 node] + [R*2 edge] + [1 face] -> hidden
        level_in = num_node_types * 4 + num_relations * 2 + 1
        self.level_mlp = nn.Sequential(nn.Linear(level_in, hidden_dim), nn.GELU())
        self.classifier = _build_mlp(num_levels * hidden_dim, hidden_dim, num_classes, dropout)

    # ---- invariant quantities ------------------------------------------------
    def _edge_scale(self, F_map: torch.Tensor) -> torch.Tensor:
        """lambda of a scaled-orthogonal map R=lambda*O recovered as ||R||_F / sqrt(d)."""
        return F_map.reshape(F_map.size(0), -1).norm(dim=1) / (self.d**0.5)

    def _face_to_node(self, face_curl, face_groups, num_nodes, batch):
        """Scatter per-face curl energy onto member nodes (mean); return (node_face_curl [N], face_batch or None).

        Guards alignment: if concatenated face count != face_curl rows, returns zeros (face rank disabled)."""
        if not face_groups or face_curl.numel() == 0:
            return face_curl.new_zeros(num_nodes), None
        member = torch.cat([fg.face_node_index for fg in face_groups], dim=0)  # [F_tot, L]
        if member.size(0) != face_curl.size(0):
            return face_curl.new_zeros(num_nodes), None
        node_sum = face_curl.new_zeros(num_nodes)
        node_cnt = face_curl.new_zeros(num_nodes)
        for col in range(member.size(1)):
            node_sum.index_add_(0, member[:, col], face_curl)
            node_cnt.index_add_(0, member[:, col], torch.ones_like(face_curl))
        node_face_curl = node_sum / node_cnt.clamp_min(1.0)
        face_batch = batch[member[:, 0]]
        return node_face_curl, face_batch

    def _invariants(self, encoded, edge_index, node_type, batch, face_groups):
        num_nodes = encoded.x_star.size(0)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        src, tgt = edge_index[0], edge_index[1]
        num_base_edges = edge_index.size(1)

        node_energy = encoded.x_star.pow(2).sum(dim=1)                       # [N]
        eps_e = encoded.delta0_x_star[:num_base_edges].pow(2).sum(dim=1)     # [E]
        loglam_e = torch.log(
            0.5 * (
                self._edge_scale(encoded.F_src[:num_base_edges])
                + self._edge_scale(encoded.F_tgt[:num_base_edges])
            )
            + EPS
        )                                                                    # [E]
        face_curl = encoded.raw_face_curl.pow(2).sum(dim=1)                 # [F_tot]
        node_face_curl, face_batch = self._face_to_node(face_curl, face_groups, num_nodes, batch)

        # scatter edge quantities to incident nodes (both endpoints)
        def to_nodes(edge_val):
            s = edge_val.new_zeros(num_nodes)
            c = edge_val.new_zeros(num_nodes)
            s.index_add_(0, src, edge_val); s.index_add_(0, tgt, edge_val)
            c.index_add_(0, src, torch.ones_like(edge_val)); c.index_add_(0, tgt, torch.ones_like(edge_val))
            return s / c.clamp_min(1.0)

        node_eps = to_nodes(eps_e)
        node_loglam = to_nodes(loglam_e)
        degree = torch.bincount(torch.cat([src, tgt]), minlength=num_nodes).to(node_energy.dtype)
        return {
            "num_graphs": num_graphs, "node_energy": node_energy, "node_eps": node_eps,
            "node_face_curl": node_face_curl, "node_loglam": node_loglam, "degree": degree,
            "eps_e": eps_e, "loglam_e": loglam_e, "face_curl": face_curl, "face_batch": face_batch,
        }

    # ---- steps ---------------------------------------------------------------
    def _score(self, inv, node_type, batch):
        deg_view = torch.sigmoid(self.deg_a * torch.log(inv["degree"] + EPS) + self.deg_b)
        raw = torch.stack(
            [inv["node_energy"], inv["node_eps"], inv["node_face_curl"], inv["node_loglam"], deg_view],
            dim=1,
        )  # [N, 5]
        z = _standardize_per_graph(raw, batch, inv["num_graphs"])
        p = z.new_zeros(z.size(0))
        for t, mlp in enumerate(self.score_mlps):
            m = node_type == t
            if m.any():
                p[m] = torch.sigmoid(mlp(z[m]).squeeze(-1))
        return p

    def _induce(self, perm, p, encoded, edge_index, edge_type, node_type, batch, num_nodes):
        new_id = torch.full((num_nodes,), -1, dtype=torch.long, device=perm.device)
        new_id[perm] = torch.arange(perm.size(0), device=perm.device)
        h0_c = p[perm].unsqueeze(1) * encoded.x_star[perm]           # gated coarse stalks
        keep_e = (new_id[edge_index[0]] >= 0) & (new_id[edge_index[1]] >= 0)
        edge_index_c = new_id[edge_index[:, keep_e]]
        edge_type_c = edge_type[keep_e]
        num_base_edges = edge_index.size(1)
        return CoarseComplex(
            h0=h0_c, node_type=node_type[perm], edge_index=edge_index_c,
            edge_type=edge_type_c, batch=batch[perm], perm=perm,
            F_src=encoded.F_src[:num_base_edges][keep_e],
            F_tgt=encoded.F_tgt[:num_base_edges][keep_e],
        )

    def _readout(self, inv, node_type, edge_type, edge_index, batch):
        G, T, R = inv["num_graphs"], self.num_node_types, self.num_relations
        # node part: per (graph, type) mean of 4 invariant node features -> [G, T*4]
        node_feat = torch.stack(
            [inv["node_energy"], inv["node_eps"], inv["node_face_curl"], inv["node_loglam"]], dim=1
        )
        node_grp = batch * T + node_type
        node_part = scatter(node_feat, node_grp, dim=0, dim_size=G * T, reduce="mean").reshape(G, T * 4)
        # edge part: per (graph, relation) mean of [log1p(eps), loglam] -> [G, R*2]
        edge_batch = batch[edge_index[0]]
        edge_feat = torch.stack([torch.log1p(inv["eps_e"]), inv["loglam_e"]], dim=1)
        edge_grp = edge_batch * R + edge_type
        edge_part = scatter(edge_feat, edge_grp, dim=0, dim_size=G * R, reduce="mean").reshape(G, R * 2)
        # face part: per-graph mean of log1p(curl) -> [G, 1]
        if inv["face_batch"] is not None and inv["face_curl"].numel() > 0:
            face_part = scatter(
                torch.log1p(inv["face_curl"]), inv["face_batch"], dim=0, dim_size=G, reduce="mean"
            ).unsqueeze(1)
        else:
            face_part = node_part.new_zeros(G, 1)
        return self.level_mlp(torch.cat([node_part, edge_part, face_part], dim=1))  # [G, hidden]

    # ---- multi-level driver --------------------------------------------------
    def run(
        self,
        encoded,
        face_groups,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_type: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        encode_from_stalk: Callable,
        build_faces: Callable,
    ) -> torch.Tensor:
        tokens: list[torch.Tensor] = []
        for level in range(self.num_levels):
            inv = self._invariants(encoded, edge_index, node_type, batch, face_groups)
            tokens.append(self._readout(inv, node_type, edge_type, edge_index, batch))
            if level == self.num_levels - 1:
                break
            # select + induce -> coarse complex
            p = self._score(inv, node_type, batch)
            grp = batch * self.num_node_types + node_type
            keep = _topk_per_group(p, grp, self.ratio, num_graphs * self.num_node_types)
            perm = keep.nonzero(as_tuple=False).view(-1)
            coarse = self._induce(perm, p, encoded, edge_index, edge_type, node_type, batch, node_type.size(0))
            if coarse.edge_index.size(1) == 0 or perm.size(0) == 0:
                break  # nothing left to diffuse; stop early, use tokens so far
            faces_c, closure_c = build_faces(coarse.edge_index, coarse.edge_type, coarse.node_type, coarse.batch)
            injected = (coarse.F_src, coarse.F_tgt) if self.inject_maps else None
            encoded = encode_from_stalk(
                h0=coarse.h0, edge_index=coarse.edge_index, edge_type=coarse.edge_type,
                face_groups=faces_c, closure_edge_index=closure_c,
                num_nodes=coarse.h0.size(0), num_edges=coarse.edge_index.size(1), discovery=None,
                injected_maps=injected,
            )
            edge_index, edge_type, node_type, batch = (
                coarse.edge_index, coarse.edge_type, coarse.node_type, coarse.batch
            )
            face_groups = faces_c
        # pad to num_levels tokens (if we stopped early) so the classifier input dim is fixed
        while len(tokens) < self.num_levels:
            tokens.append(torch.zeros_like(tokens[-1]))
        return self.classifier(torch.cat(tokens, dim=1))


class GalerkinSheafPool(nn.Module):
    """THS-Pool, Galerkin redesign (stage 1: node-level, typed).

    Coarsen the node sheaf Laplacian by the energy-preserving Galerkin construction
    (docs/sheafcovpool_implementation.md, Appendix A):

        L0_c(z) = S^T · L0( S · z )        (matrix-free; reuses the *fine* L0)

    where S is a learnable, type-blocked soft assignment of fine nodes to coarse supernodes.
    By construction  z^T L0_c z = ||delta0(S z)||^2  (fine sheaf energy of the lifted signal),
    so the coarse operator is energy-consistent — the property that "inject" lacked. No backbone
    re-run, no per-level face DFS: the only cost is a fine-L0 matvec per coarse diffusion step.
    """

    def __init__(
        self,
        d: int,
        num_node_types: int,
        num_relations: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        clusters_per_type: int = 4,
        coarse_steps: int = 4,
        eta_max: float = 1.0,
        use_coarse: bool = True,
    ) -> None:
        super().__init__()
        self.d = d
        self.T = num_node_types
        self.C = clusters_per_type
        self.coarse_steps = coarse_steps
        self.eta_max = eta_max
        self.use_coarse = use_coarse  # False = fine-only ablation (matched readout, no Galerkin level)
        self.score_mlps = nn.ModuleList(
            [_build_mlp(5, hidden_dim, clusters_per_type, dropout) for _ in range(num_node_types)]
        )
        # diffusion step = eta_max*sigmoid(log_eta) / spectral_norm(L0_c), so step*lambda_max < 1
        # (stable regardless of graph scale). init sigmoid(0)=0.5.
        self.log_eta = nn.Parameter(torch.tensor(0.0))
        # readout: fine per-type energy [T] + fine edge-disagreement [1] + fine curl [1] + coarse per-type energy [T]
        self.readout = _build_mlp(2 * num_node_types + 2, hidden_dim, num_classes, dropout)

    # ---- invariants (compact) ------------------------------------------------
    def _invariants(self, encoded, edge_index, node_type, batch, num_graphs, face_groups):
        N = encoded.x_star.size(0)
        src, tgt = edge_index[0], edge_index[1]
        node_energy = encoded.x_star.pow(2).sum(dim=1)
        eps_e = encoded.delta0_x_star.pow(2).sum(dim=1)
        lam = 0.5 * (
            encoded.F_src.reshape(encoded.F_src.size(0), -1).norm(dim=1)
            + encoded.F_tgt.reshape(encoded.F_tgt.size(0), -1).norm(dim=1)
        ) / (self.d ** 0.5)
        loglam_e = torch.log(lam + EPS)
        face_curl = encoded.raw_face_curl.pow(2).sum(dim=1)
        # face -> node (guarded alignment) + face_batch
        node_curl = node_energy.new_zeros(N)
        face_batch = None
        if face_groups and face_curl.numel() > 0:
            member = torch.cat([fg.face_node_index for fg in face_groups], dim=0)
            if member.size(0) == face_curl.size(0):
                cnt = node_energy.new_zeros(N)
                for col in range(member.size(1)):
                    node_curl.index_add_(0, member[:, col], face_curl)
                    cnt.index_add_(0, member[:, col], torch.ones_like(face_curl))
                node_curl = node_curl / cnt.clamp_min(1.0)
                face_batch = batch[member[:, 0]]

        def to_nodes(val):
            s = val.new_zeros(N); c = val.new_zeros(N)
            s.index_add_(0, src, val); s.index_add_(0, tgt, val)
            c.index_add_(0, src, torch.ones_like(val)); c.index_add_(0, tgt, torch.ones_like(val))
            return s / c.clamp_min(1.0)

        degree = torch.bincount(torch.cat([src, tgt]), minlength=N).to(node_energy.dtype)
        z = _standardize_per_graph(
            torch.stack([node_energy, to_nodes(eps_e), node_curl, to_nodes(loglam_e), degree], dim=1),
            batch, num_graphs,
        )
        return {"z": z, "node_energy": node_energy, "eps_e": eps_e, "face_curl": face_curl,
                "face_batch": face_batch, "N": N, "src": src, "tgt": tgt}

    # ---- assignment + Galerkin operator -------------------------------------
    def _assign(self, z, node_type, batch, num_graphs):
        N = z.size(0)
        logits = z.new_zeros(N, self.C)
        for t, mlp in enumerate(self.score_mlps):
            m = node_type == t
            if m.any():
                logits[m] = mlp(z[m])
        assign = torch.softmax(logits, dim=1)                       # [N, C] within each node's C clusters
        base = (batch * self.T + node_type) * self.C                # [N] global offset of node's cluster block
        coarse_idx = base.unsqueeze(1) + torch.arange(self.C, device=z.device).unsqueeze(0)  # [N, C]
        K = num_graphs * self.T * self.C
        return assign, coarse_idx, K

    def _prolong(self, coarse_z, coarse_idx, assign):               # S @ coarse_z : [K,d] -> [N,d]
        return (assign.unsqueeze(-1) * coarse_z[coarse_idx]).sum(dim=1)

    def _restrict(self, fine_y, coarse_idx, assign, K):             # S^T @ fine_y : [N,d] -> [K,d]
        out = fine_y.new_zeros(K, fine_y.size(1))
        contrib = (assign.unsqueeze(-1) * fine_y.unsqueeze(1)).reshape(-1, fine_y.size(1))
        out.index_add_(0, coarse_idx.reshape(-1), contrib)
        return out

    def _L0c(self, z, coarse_idx, assign, K, enc, inv):             # S^T L0 (S z)
        # The coarse Galerkin operator is FIXED geometry: detach both the maps (trained by the fine
        # backbone) and the assignment. Otherwise back-prop through the unrolled iterated diffusion
        # multiplies Jacobians of the conformal maps (lambda ~ e^2) and overflows. The diffusion thus
        # applies a fixed linear operator to z; gradients still flow to `assign` and x* through the
        # coarse init/readout (z itself), just not through the operator's structure.
        a = assign.detach()
        Sz = self._prolong(z, coarse_idx, a)
        L0Sz = CoboundaryOps.L0(Sz, enc.F_src.detach(), enc.F_tgt.detach(), inv["src"], inv["tgt"], inv["N"])
        return self._restrict(L0Sz, coarse_idx, a, K)

    @torch.no_grad()
    def _spectral_norm(self, coarse_idx, assign, K, enc, inv, iters: int = 30):
        """Power-iteration estimate of lambda_max(L0_c); detached (used only to scale the step).
        30 iters for reliable convergence; the caller adds a safety margin so the step is stable
        even if this slightly under-estimates."""
        v = torch.randn(K, self.d, device=enc.x_star.device)
        v = v / (v.norm() + EPS)
        lam = v.new_ones(())
        for _ in range(iters):
            w = self._L0c(v, coarse_idx, assign, K, enc, inv)
            lam = w.norm()
            if lam < EPS:
                return v.new_ones(())
            v = w / lam
        return lam.clamp_min(1.0)

    # ---- readout -------------------------------------------------------------
    def _per_type_energy(self, state, type_ids, graph_ids, num_graphs):
        e = state.pow(2).sum(dim=1)
        return scatter(e, graph_ids * self.T + type_ids, dim=0, dim_size=num_graphs * self.T, reduce="mean").reshape(num_graphs, self.T)

    def forward(self, encoded, edge_index, node_type, batch, num_graphs, face_groups):
        inv = self._invariants(encoded, edge_index, node_type, batch, num_graphs, face_groups)

        # fine readout: per-type energy + edge disagreement + curl
        fine_e = self._per_type_energy(encoded.x_star, node_type, batch, num_graphs)      # [G, T]
        fine_eps = scatter(inv["eps_e"], batch[inv["src"]], dim=0, dim_size=num_graphs, reduce="mean").unsqueeze(1)
        if inv["face_batch"] is not None and inv["face_curl"].numel() > 0:
            fine_curl = scatter(inv["face_curl"], inv["face_batch"], dim=0, dim_size=num_graphs, reduce="mean").unsqueeze(1)
        else:
            fine_curl = fine_e.new_zeros(num_graphs, 1)

        # coarse Galerkin level (skipped in the fine-only ablation)
        if self.use_coarse:
            assign, coarse_idx, K = self._assign(inv["z"], node_type, batch, num_graphs)
            x_c = self._restrict(encoded.x_star, coarse_idx, assign, K)                    # S^T x* (grad flows here)
            # spectral-normalized step with a 2x safety margin so step*lambda_max(L0_c) <= eta/2 < 1,
            # stable across graph scales even if the power-iteration estimate is slightly low.
            lam = self._spectral_norm(coarse_idx, assign, K, encoded, inv)
            step = (self.eta_max * torch.sigmoid(self.log_eta)) / (2.0 * lam)
            for _ in range(self.coarse_steps):                                            # Galerkin diffusion
                x_c = x_c - step * self._L0c(x_c, coarse_idx, assign, K, encoded, inv)
            a = torch.arange(K, device=x_c.device)
            sup_type = (a // self.C) % self.T
            sup_graph = a // (self.T * self.C)
            coarse_e = self._per_type_energy(x_c, sup_type, sup_graph, num_graphs)         # [G, T]
        else:
            coarse_e = fine_e.new_zeros(num_graphs, self.T)
        # log-scale the (large, unbounded) energy features so the readout MLP stays well-conditioned
        token = torch.cat(
            [torch.log1p(fine_e), torch.log1p(fine_eps), torch.log1p(fine_curl), torch.log1p(coarse_e)],
            dim=1,
        )
        return self.readout(token)

    @torch.no_grad()
    def energy_check(self, encoded, edge_index, node_type, batch, num_graphs, face_groups):
        """Returns (lhs, rhs): z^T L0_c z  vs  ||delta0(S z)||^2 for a random coarse probe."""
        inv = self._invariants(encoded, edge_index, node_type, batch, num_graphs, face_groups)
        assign, coarse_idx, K = self._assign(inv["z"], node_type, batch, num_graphs)
        z = torch.randn(K, self.d, device=encoded.x_star.device)
        lhs = (z * self._L0c(z, coarse_idx, assign, K, encoded, inv)).sum()
        Sz = self._prolong(z, coarse_idx, assign)
        d0 = CoboundaryOps.delta0(Sz, encoded.F_src, encoded.F_tgt, inv["src"], inv["tgt"])
        rhs = d0.pow(2).sum()
        return float(lhs), float(rhs)


class SheafSpectralPool(nn.Module):
    """Terminal readout using gauge- & permutation-invariant SPECTRAL descriptors of the sheaf
    node-Laplacian L0, combined with local node/edge/face energies.

    The spectrum of L0 is invariant under a node gauge (L0 -> G L0 G^T, orthogonal conjugation) and
    node permutation, and it encodes GLOBAL topology + sheaf geometry (cycles, connectivity, the whole
    spectrum of the learned restriction maps) that purely local energy readouts cannot see. We probe it
    with a cheap heat-kernel-like diffusion-energy-decay signature: for random probes p, the retained
    energy ||D^k p||^2 / ||p||^2 at a few k (D = I - L0/lambda_max, spectrum in [0,1]) is a spectral
    fingerprint. Computed with fixed (detached) geometry so it is numerically stable; the backbone is
    trained through the local-energy part of the readout.
    """

    def __init__(self, d, num_node_types, num_relations, num_classes, hidden_dim, dropout,
                 times=(1, 2, 4, 8, 16), n_probes=4, use_spectral: bool = True) -> None:
        super().__init__()
        self.d = d
        self.T = num_node_types
        self.times = tuple(times)
        self.n_probes = n_probes
        self.use_spectral = use_spectral  # False = local-energy-only ablation (matched readout)
        in_dim = num_node_types + 2 + len(self.times)
        self.readout = _build_mlp(in_dim, hidden_dim, num_classes, dropout)

    def _per_type_energy(self, state, type_ids, graph_ids, num_graphs):
        e = state.pow(2).sum(dim=1)
        return scatter(e, graph_ids * self.T + type_ids, dim=0, dim_size=num_graphs * self.T, reduce="mean").reshape(num_graphs, self.T)

    @torch.no_grad()
    def _spectral_decay(self, encoded, edge_index, batch, num_graphs):
        N = encoded.x_star.size(0)
        src, tgt = edge_index[0], edge_index[1]
        Fs, Ft = encoded.F_src.detach(), encoded.F_tgt.detach()
        dev = encoded.x_star.device
        # lambda_max(L0) via power iteration (to normalize the diffusion operator to spectrum [0,1])
        v = torch.randn(N, self.d, device=dev)
        v = v / (v.norm() + EPS)
        lam = v.new_ones(())
        for _ in range(20):
            w = CoboundaryOps.L0(v, Fs, Ft, src, tgt, N)
            lam = w.norm()
            if lam < EPS:
                lam = v.new_ones(()); break
            v = w / lam
        lam = lam.clamp_min(1.0)
        maxk = max(self.times)
        acc = torch.zeros(num_graphs, len(self.times), device=dev)
        for _ in range(self.n_probes):
            p = torch.randn(N, self.d, device=dev)
            e0 = scatter(p.pow(2).sum(1), batch, dim=0, dim_size=num_graphs, reduce="sum").clamp_min(EPS)
            cur = p
            cols = []
            for kk in range(1, maxk + 1):
                cur = cur - (1.0 / lam) * CoboundaryOps.L0(cur, Fs, Ft, src, tgt, N)
                if kk in self.times:
                    ek = scatter(cur.pow(2).sum(1), batch, dim=0, dim_size=num_graphs, reduce="sum")
                    cols.append(ek / e0)
            acc = acc + torch.stack(cols, dim=1)
        return acc / self.n_probes  # [G, len(times)] in [0,1], detached

    def forward(self, encoded, edge_index, node_type, batch, num_graphs, face_groups):
        src = edge_index[0]
        fine_e = self._per_type_energy(encoded.x_star, node_type, batch, num_graphs)          # [G, T]
        fine_eps = scatter(encoded.delta0_x_star.pow(2).sum(1), batch[src], dim=0,
                           dim_size=num_graphs, reduce="mean").unsqueeze(1)                    # [G, 1]
        face_curl = encoded.raw_face_curl.pow(2).sum(1)
        fine_curl = fine_e.new_zeros(num_graphs, 1)
        if face_groups and face_curl.numel() > 0:
            member = torch.cat([fg.face_node_index for fg in face_groups], dim=0)
            if member.size(0) == face_curl.size(0):
                fine_curl = scatter(face_curl, batch[member[:, 0]], dim=0,
                                    dim_size=num_graphs, reduce="mean").unsqueeze(1)
        if self.use_spectral:
            spec = self._spectral_decay(encoded, edge_index, batch, num_graphs)               # [G, len(times)]
        else:
            spec = fine_e.new_zeros(num_graphs, len(self.times))
        token = torch.cat([torch.log1p(fine_e), torch.log1p(fine_eps), torch.log1p(fine_curl), spec], dim=1)
        return self.readout(token)
