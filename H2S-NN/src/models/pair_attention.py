"""`coupling: pair_attention` - the A4 test variant (plan §12.3, task card Part F, U4b).

Changes ONE thing relative to the final model: how influence strength is formed. Same `h0`, same
maps (D1/D1a), same tied facts and walk-sampled cells (D7-D8), same single block of K steps, same
head, same eta. The strength of every message comes from a GATv2-style score computed once from
`h0` (like the maps), normalised by a softmax over the receiving node's own self-loop, its facts
and its cells:

    s(u<-e) = a^T LeakyReLU(W_e [h0_u || h0_v || rel_emb(psi_e) || dir_emb(role of u on e)])
    s(u<-f) = a^T LeakyReLU(W_f [h0_u || path_latent_f || pos_emb(position of u in f)]) + log m_f
    s(u<-u) = a^T LeakyReLU(W_s [h0_u || h0_u || self_emb])
    alpha_u = softmax over {u} + {facts at u} + {cells containing u}
    (P x)_u = alpha_uu x_u + sum_e alpha_ue Q_{u<-v} x_v + sum_f alpha_uf T_u^T z_f
    x <- x - eta (x - P x)

`Q_{u<-v}` is the orthogonal transport `Delta_0` uses (no `c_e`), `T_u^T z_f` the D9 cell term
(`z_f` = mean of the transported states). No per-step weights, no nonlinearity, no attention
recomputed from `x_t`; gate heads, `w_f`, `c_e^2` and `gamma` do not enter. A bias-free linear
layer inside the nonlinearity and a single head follow the card; the GATv2 score form is
`e(h_i, h_j) = a^T LeakyReLU(W [h_i || h_j])` (Brody, Alon, Yahav, ICLR 2022, Eq. 7, VERIFIED-
SOURCE `references/GATv2.pdf`; the paper adds a bias before the nonlinearity in practice, footnote 5,
omitted here as the card specifies).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src.models.map_operators import EdgeMapPair, PathTransportOperator, checkpoint_if_grad

# GAT's LeakyReLU slope (Velickovic et al. 2018); GATv2 keeps it.
LEAKY_RELU_SLOPE = 0.2


class PairAttention(nn.Module):
    """Score networks of the three entry kinds (one shared `a`, one `W` per kind: their inputs
    have different widths), plus the relation / role / position / self embeddings."""

    def __init__(self, d: int, d_r: int, path_hidden: int, num_relations: int,
                 max_positions: int, negative_slope: float = LEAKY_RELU_SLOPE) -> None:
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, d_r)
        self.dir_emb = nn.Embedding(2, d_r)  # 0: source role (or shared role), 1: target role
        self.pos_emb = nn.Embedding(max_positions, d_r)
        self.self_emb = nn.Parameter(torch.randn(d_r))
        self.w_edge = nn.Linear(2 * d + 2 * d_r, d, bias=False)
        self.w_cell = nn.Linear(d + path_hidden + d_r, d, bias=False)
        self.w_self = nn.Linear(2 * d + d_r, d, bias=False)
        self.a = nn.Linear(d, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope)

    def _score(self, linear: nn.Linear, features: torch.Tensor) -> torch.Tensor:
        return self.a(self.activation(linear(features))).squeeze(-1)

    def edge_scores(self, h0: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor,
                    shared_roles: torch.Tensor) -> torch.Tensor:
        """`[2E]`: incidence `e` (receiver = stored source, sender = stored target, role 0), then
        incidence `E + e` (receiver = target, sender = source, role 1 unless the relation has
        shared roles)."""
        src, tgt = edge_index[0], edge_index[1]
        relation = self.rel_emb(edge_type)
        role_src = self.dir_emb(torch.zeros_like(edge_type))
        role_tgt = self.dir_emb((~shared_roles[edge_type]).long())
        receiver_src = torch.cat([h0[src], h0[tgt], relation, role_src], dim=-1)
        receiver_tgt = torch.cat([h0[tgt], h0[src], relation, role_tgt], dim=-1)
        return self._score(self.w_edge, torch.cat([receiver_src, receiver_tgt], dim=0))

    def cell_scores(self, h0: torch.Tensor, nodes: torch.Tensor, latent: torch.Tensor,
                    log_merged: torch.Tensor, checkpoint: bool) -> torch.Tensor:
        """`[P, L+1]`: score of every position of every cell (plus `log m_f`)."""
        num_cells, size = nodes.shape
        positions = torch.arange(size, device=nodes.device)

        def compute(h_in: torch.Tensor, latent_in: torch.Tensor) -> torch.Tensor:
            features = torch.cat([
                h_in[nodes],
                latent_in.unsqueeze(1).expand(num_cells, size, -1),
                self.pos_emb(positions).unsqueeze(0).expand(num_cells, size, -1),
            ], dim=-1)
            return self._score(self.w_cell, features)

        scores = checkpoint_if_grad(compute, h0, latent, enabled=checkpoint)
        return scores + log_merged.unsqueeze(-1)

    def self_scores(self, h0: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [h0, h0, self.self_emb.unsqueeze(0).expand(h0.size(0), -1)], dim=-1)
        return self._score(self.w_self, features)


def segment_softmax(scores: torch.Tensor, receivers: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax of `scores` within each receiver's segment (stable: the per-segment maximum is
    subtracted; it carries no gradient, the result does not depend on it)."""
    peak = scores.new_full((num_nodes,), float("-inf"))
    peak = peak.scatter_reduce(0, receivers, scores.detach(), reduce="amax", include_self=True)
    weights = torch.exp(scores - peak[receivers])
    total = scores.new_zeros(num_nodes).index_add(0, receivers, weights)
    return weights / total[receivers]


class AttentionOperator:
    """`P x` of the pair-attention update, with the attention weights kept for diagnostics.

    Duck-typed against `DiffusionOperator` where the model reads it (`eta`, `maps`, `gamma`,
    `path_weights`, `laplacian`, `listen`, `broadcast`): there is no Laplacian and no gates here.
    """

    laplacian = None
    listen = None
    broadcast = None
    gamma = None
    path_weights: tuple[torch.Tensor, ...] = ()

    def __init__(self, maps: EdgeMapPair, eta: torch.Tensor, edge_index: torch.Tensor,
                 alpha_self: torch.Tensor, alpha_edge: torch.Tensor,
                 transports: Sequence[PathTransportOperator],
                 alpha_cells: Sequence[torch.Tensor]) -> None:
        self.maps = maps
        self.eta = eta
        self.edge_index = edge_index
        self.alpha_self = alpha_self
        self.alpha_edge = alpha_edge  # [2E]: first E receivers = stored sources, then targets
        self.transports = tuple(transports)
        self.alpha_cells = tuple(alpha_cells)

    def propagate(self, x: torch.Tensor) -> torch.Tensor:
        """`(P x)_u = alpha_uu x_u + sum_e alpha_ue Q_{u<-v} x_v + sum_f alpha_uf T_u^T z_f`."""
        src, tgt = self.edge_index[0], self.edge_index[1]
        num_edges = src.numel()
        out = self.alpha_self.unsqueeze(-1) * x
        if num_edges:
            to_src = self.maps.src.apply_orthogonal_T(self.maps.tgt.apply_orthogonal(x[tgt]))
            to_tgt = self.maps.tgt.apply_orthogonal_T(self.maps.src.apply_orthogonal(x[src]))
            out = out.index_add(0, src, self.alpha_edge[:num_edges].unsqueeze(-1) * to_src)
            out = out.index_add(0, tgt, self.alpha_edge[num_edges:].unsqueeze(-1) * to_tgt)
        for transport, alpha in zip(self.transports, self.alpha_cells):
            size = transport.face_node_index.size(1)
            z = transport.transport_sum(x) / size
            back = transport.adjoint_prefix(z)  # [P, L+1, d]: T_i^T z_f
            out = out.index_add(0, transport.face_node_index.reshape(-1),
                                (alpha.unsqueeze(-1) * back).reshape(-1, x.size(-1)))
        return out

    def row_sums(self, num_nodes: int) -> torch.Tensor:
        """Sum of every receiver's attention weights (1 up to rounding); used by the tests."""
        src, tgt = self.edge_index[0], self.edge_index[1]
        total = self.alpha_self.clone()
        total = total.index_add(0, src, self.alpha_edge[:src.numel()])
        total = total.index_add(0, tgt, self.alpha_edge[src.numel():])
        for transport, alpha in zip(self.transports, self.alpha_cells):
            total = total.index_add(0, transport.face_node_index.reshape(-1), alpha.reshape(-1))
        return total


def build_attention_operator(
    attention: PairAttention,
    maps: EdgeMapPair,
    eta: torch.Tensor,
    h0: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    shared_roles: torch.Tensor,
    transports: Sequence[PathTransportOperator],
    latents: Sequence[torch.Tensor],
    log_merged: Sequence[torch.Tensor],
    checkpoint: bool,
) -> AttentionOperator:
    """Scores of all entries, one softmax per receiving node, and the operator."""
    num_nodes = h0.size(0)
    num_edges = edge_index.size(1)
    receivers = [torch.arange(num_nodes, device=h0.device),
                 edge_index[0], edge_index[1]]
    scores = [attention.self_scores(h0),
              attention.edge_scores(h0, edge_index, edge_type, shared_roles)]
    cell_shapes = []
    for transport, latent, log_m in zip(transports, latents, log_merged):
        nodes = transport.face_node_index
        scores.append(attention.cell_scores(h0, nodes, latent, log_m, checkpoint).reshape(-1))
        receivers.append(nodes.reshape(-1))
        cell_shapes.append(nodes.shape)
    # receivers of the edge block: E sources then E targets (matching `edge_scores`)
    receiver_index = torch.cat([receivers[0], receivers[1], receivers[2], *receivers[3:]])
    alpha = segment_softmax(torch.cat(scores), receiver_index, num_nodes)
    alpha_self = alpha[:num_nodes]
    alpha_edge = alpha[num_nodes:num_nodes + 2 * num_edges]
    cursor = num_nodes + 2 * num_edges
    alpha_cells = []
    for shape in cell_shapes:
        count = shape[0] * shape[1]
        alpha_cells.append(alpha[cursor:cursor + count].view(shape))
        cursor += count
    return AttentionOperator(maps, eta, edge_index, alpha_self, alpha_edge, transports, alpha_cells)


__all__ = ["AttentionOperator", "PairAttention", "build_attention_operator", "segment_softmax"]
