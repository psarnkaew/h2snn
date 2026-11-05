from __future__ import annotations

import torch
from torch import nn

from src.models.map_operators import EdgeMapPair
from src.models.pooling_utils import (
    pool_edge_stalks_to_nodes,
    pool_edge_stalks_to_nodes_matrix_free,
)


class NodeReadout(nn.Module):
    def __init__(
        self,
        d: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x_star: torch.Tensor,
        h0: torch.Tensor,
        s_star: torch.Tensor,
        edge_index: torch.Tensor,
        F_src: torch.Tensor | None,
        F_tgt: torch.Tensor | None,
        target_indices: torch.Tensor,
        edge_maps: EdgeMapPair | None = None,
    ) -> torch.Tensor:
        """Pass dense `F_src`/`F_tgt` (v1) or `edge_maps` operators (D1), not both."""
        if edge_maps is None:
            if F_src is None or F_tgt is None:
                raise ValueError("NodeReadout needs dense F_src/F_tgt or edge_maps.")
            pooled_edges = pool_edge_stalks_to_nodes(
                s_star=s_star,
                edge_index=edge_index,
                F_src=F_src,
                F_tgt=F_tgt,
                num_nodes=x_star.size(0),
            )
        else:
            if F_src is not None or F_tgt is not None:
                raise ValueError("Pass either dense F_src/F_tgt or edge_maps, not both.")
            pooled_edges = pool_edge_stalks_to_nodes_matrix_free(
                s_star=s_star,
                edge_index=edge_index,
                maps=edge_maps,
                num_nodes=x_star.size(0),
            )

        node_representation = torch.cat([x_star, h0, pooled_edges], dim=-1)
        return self.mlp(node_representation[target_indices])
