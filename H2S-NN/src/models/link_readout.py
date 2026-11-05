from __future__ import annotations

import torch
from torch import nn

from src.models.pooling_utils import pool_edge_stalks_to_nodes


class LinkPredictionReadout(nn.Module):
    """Scores node pairs from coupled node and incident-edge cochain states.

    Three swappable decoders (selected by cfg.data.decoder_type), matching the
    three interchangeable decoders HetSheaf treats as a hyperparameter: dot-product,
    relation-specific bilinear (the ``distmult`` config key), and concat+MLP. Returns
    raw logits — sigmoid is applied by BCEWithLogitsLoss in the trainer, not here.

    num_relations > 1 only happens for datasets whose official benchmark pools
    several raw relation ids into one target (currently just Amazon: rel_0/rel_1
    both scored jointly, matching Simple-HGN's run_dist.py). distmult then holds
    one d x d matrix per relation, selected per-edge via `edge_relation` — dot and
    concat_mlp are relation-agnostic by construction (HGB itself only ever tries
    dot/DistMult decoders for link prediction) and ignore it.
    """

    def __init__(
        self,
        d: int,
        decoder_type: str,
        hidden_dim: int,
        dropout: float,
        num_relations: int = 1,
    ) -> None:
        super().__init__()
        self.decoder_type = decoder_type
        self.endpoint_projection = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
        )
        if decoder_type == "dot":
            pass
        elif decoder_type == "distmult":
            self.relation_matrices = nn.Parameter(
                torch.eye(d).unsqueeze(0).repeat(num_relations, 1, 1)
            )
        elif decoder_type == "concat_mlp":
            self.mlp = nn.Sequential(
                nn.Linear(2 * d, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unsupported decoder_type '{decoder_type}'.")

    def forward(
        self,
        x_star: torch.Tensor,
        s_star: torch.Tensor,
        edge_index: torch.Tensor,
        F_src: torch.Tensor,
        F_tgt: torch.Tensor,
        edge_label_index: torch.Tensor,
        edge_relation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index.size(1) != s_star.size(0):
            raise ValueError(
                "edge_index must span the same real-plus-closure edge domain as s_star."
            )
        incident_edges = pool_edge_stalks_to_nodes(
            s_star=s_star,
            edge_index=edge_index,
            F_src=F_src,
            F_tgt=F_tgt,
            num_nodes=x_star.size(0),
        )
        endpoint_state = self.endpoint_projection(
            torch.cat([x_star, incident_edges], dim=-1)
        )
        src = endpoint_state[edge_label_index[0]]
        dst = endpoint_state[edge_label_index[1]]

        if self.decoder_type == "dot":
            return (src * dst).sum(dim=-1)
        if self.decoder_type == "distmult":
            if edge_relation is None:
                edge_relation = edge_label_index.new_zeros(edge_label_index.size(1))
            relation_matrices = self.relation_matrices[edge_relation]  # [E, d, d]
            transformed = torch.bmm(src.unsqueeze(1), relation_matrices).squeeze(1)
            return (transformed * dst).sum(dim=-1)
        if self.decoder_type == "concat_mlp":
            return self.mlp(torch.cat([src, dst], dim=-1)).squeeze(-1)
        raise ValueError(f"Unsupported decoder_type '{self.decoder_type}'.")
