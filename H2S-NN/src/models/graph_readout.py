from __future__ import annotations

import torch
from torch import nn


class GraphReadout(nn.Module):
    """Graph classification head (plan §4.8): sum of `[x_K, h0]` over each graph's nodes,
    then BatchNorm1d, then an MLP.

    Sum pooling is permutation-invariant and, unlike mean or max, can be injective on
    multisets (Xu et al., ICLR 2019). BatchNorm keeps the summed features, whose scale grows
    with graph size, in a trainable range for the MLP.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, node_representation: torch.Tensor, batch: torch.Tensor, num_graphs: int
    ) -> torch.Tensor:
        pooled = node_representation.new_zeros((num_graphs, node_representation.size(-1)))
        pooled.index_add_(0, batch, node_representation)
        return self.mlp(self.norm(pooled))
