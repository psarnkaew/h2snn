from __future__ import annotations

import torch
from torch import nn

from src.configs.config import GRAPH_POOLINGS

# HetSheaf's GraphClassifier builds AttentionPooling(hidden, heads=2) (graph_classifier.py:44-50).
ATTENTION_POOLING_HEADS = 2


class AttentionPooling(nn.Module):
    """HetSheaf's attention pooling (`hetsheaf/graph_classification/pooling.py`, commit 6964e1e):
    scores Linear(h, h) -> Tanh -> Linear(h, heads, bias=False), softmax over each graph's nodes,
    weighted sum per head, heads averaged."""

    def __init__(self, hidden_channels: int, heads: int = ATTENTION_POOLING_HEADS) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, heads, bias=False),
        )
        self.heads = heads

    def forward(self, x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        from torch_geometric.utils import softmax

        weights = softmax(self.attention(x), batch, num_nodes=num_graphs, dim=0)  # [N, heads]
        weighted = x.unsqueeze(1) * weights.unsqueeze(-1)                      # [N, heads, h]
        out = x.new_zeros((num_graphs, self.heads, x.size(-1)))
        out.index_add_(0, batch, weighted)
        return out.mean(dim=1)


def pool_nodes(x: torch.Tensor, batch: torch.Tensor, num_graphs: int, pooling: str) -> torch.Tensor:
    """Global mean / max / add pooling of node rows per graph (PyG's global_*_pool semantics)."""
    from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

    pools = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool}
    return pools[pooling](x, batch, size=num_graphs)


class GraphReadout(nn.Module):
    """Graph classification head (spec v1.15 §3 readouts, §6 GC; J6): `[x*, h0]` of every node ->
    pooling over the graph's nodes (mean, max, add, or HetSheaf's attention pooling) -> one Linear
    layer. No BatchNorm and no MLP, as HetSheaf's GraphClassifier (pool, then `Linear(hidden,
    out)`, graph_classifier.py:24-50, 98-110) and every GC baseline there."""

    def __init__(self, input_dim: int, num_classes: int, pooling: str) -> None:
        super().__init__()
        if pooling not in GRAPH_POOLINGS:
            raise ValueError(f"graph pooling must be one of {GRAPH_POOLINGS}, got {pooling!r}.")
        self.pooling = pooling
        self.attention = AttentionPooling(input_dim) if pooling == "attention" else None
        self.decoder = nn.Linear(input_dim, num_classes)

    def forward(
        self, node_representation: torch.Tensor, batch: torch.Tensor, num_graphs: int
    ) -> torch.Tensor:
        if self.attention is not None:
            pooled = self.attention(node_representation, batch, num_graphs)
        else:
            pooled = pool_nodes(node_representation, batch, num_graphs, self.pooling)
        return self.decoder(pooled)
