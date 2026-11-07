"""Relational context `c` (methodology spec v1.2, R10, Eq. 9; task card Part G, G1).

    c_v = concat over (relation r, role s) of mean_{u in N_{r,s}(v)} h0_u     (zeros if empty)

over the facts the model diffuses on (D7: one oriented 1-cell per fact). Role 0: `v` is the stored
source of the fact and `u` its target; role 1: `v` is the target and `u` the source. A shared-role
relation (rho(r) = r, or every relation under `ablation.direction_blind_maps`) has one role: both
incidences go to slot 0 and slot 1 stays zero, so `c` does not depend on the stored orientation.
Parameter-free; computed once per forward from `h0`, before any propagation (one sheaf, L2).
"""

from __future__ import annotations

import torch
from torch import nn


class RelationalContext(nn.Module):
    def __init__(self, num_relations: int, shared_role_relations: tuple[int, ...] = ()) -> None:
        super().__init__()
        self.num_relations = num_relations
        # Plain tuple, not a buffer: the state dict is the same with or without the context.
        self.shared_role_relations = tuple(shared_role_relations)

    def output_dim(self, d: int) -> int:
        return 2 * self.num_relations * d

    def forward(self, h0: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """`[N, 2R d]`; slot `2 r + s` holds the mean over the (r, s) neighbours."""
        num_nodes, d = h0.shape
        slots_per_node = 2 * self.num_relations
        src, tgt = edge_index[0], edge_index[1]
        shared = torch.zeros(self.num_relations, dtype=torch.bool, device=h0.device)
        if self.shared_role_relations:
            shared[list(self.shared_role_relations)] = True
        target_role = (~shared[edge_type]).long()
        slots = torch.cat([src * slots_per_node + 2 * edge_type,
                           tgt * slots_per_node + 2 * edge_type + target_role])
        values = torch.cat([h0[tgt], h0[src]])
        sums = h0.new_zeros((num_nodes * slots_per_node, d)).index_add(0, slots, values)
        counts = h0.new_zeros(num_nodes * slots_per_node).index_add(
            0, slots, h0.new_ones(slots.numel()))
        return (sums / counts.clamp_min(1.0).unsqueeze(-1)).view(num_nodes, slots_per_node * d)


__all__ = ["RelationalContext"]
