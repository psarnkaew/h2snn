"""Input projection shared by the homogeneous baselines.

GCN and GAT accept HGB's per-type feature list directly, because HGB's own models carry
one ``nn.Linear`` per node type (``HGB/NC/benchmark/methods/GNN/GNN.py:29``). ChebNetII,
NSD and CSNN do not -- they were published on homogeneous graphs and take a single dense
feature matrix.

Feeding them the flattened, right-zero-padded matrix from ``stack_node_features`` would be
a poor comparison: DBLP's 334-dim author features would occupy 8% of a 4231-dim input with
the rest structural zeros, and the resulting number would measure the adapter rather than
the model. Instead they get the *same* per-type projection HGB applies, and the projected
result is concatenated into one matrix.

This is an adaptation, not something the authors wrote, and adapters that use it record
``input_projection: per_type_linear`` on the run record.
"""

from __future__ import annotations

import torch
from torch import nn


class TypeProjectedInput(nn.Module):
    """One ``nn.Linear`` per node type into a shared width, concatenated.

    Mirrors HGB's ``fc_list`` construction, including its Xavier gain, so the homogeneous
    baselines see features prepared exactly as HGB prepares them for GCN and GAT.

    Handles the sparse identity blocks that :class:`~src.baselines.bundles.HomogeneousView`
    emits for attribute-less node types: ``nn.Linear`` cannot consume a sparse tensor, so
    those are projected with an explicit sparse matmul, which is the same operation.
    """

    def __init__(self, in_dims: list[int], hidden_dim: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(in_dim, hidden_dim, bias=True) for in_dim in in_dims]
        )
        for projection in self.projections:
            nn.init.xavier_normal_(projection.weight, gain=1.414)
        self.out_dim = hidden_dim

    def forward(self, features_list: list[torch.Tensor]) -> torch.Tensor:
        projected: list[torch.Tensor] = []
        for projection, feature in zip(self.projections, features_list):
            if feature.is_sparse:
                # torch.sparse.mm(sparse, dense) == Linear without the bias term, which
                # is then added back; nn.Linear itself rejects sparse input.
                dense = torch.sparse.mm(feature, projection.weight.t())
                projected.append(dense + projection.bias)
            else:
                projected.append(projection(feature))
        return torch.cat(projected, dim=0)


__all__ = ["TypeProjectedInput"]
