from __future__ import annotations

import torch
from torch import nn


class LinkPredictionReadout(nn.Module):
    """Scores node pairs from the node representation `x*` (spec 1.25 Z3; `[x*, h0]` only under
    `ablation.ego_head`, the 1.24 readout; plan §4.8).

    The endpoint projection `Linear(2d, d) + GELU` (unchanged from the submitted model) maps
    each node's representation to `d` dimensions; one of two decoders (selected by
    cfg.data.decoder_type, tuned in {dot, DistMult}: the LP head shared with every baseline,
    spec v1.22 W2), dot-product or relation-specific bilinear (the ``distmult`` config key), then
    scores the projected endpoints. Returns raw logits — sigmoid is applied by
    BCEWithLogitsLoss in the trainer, not here. (The former concat+MLP decoder was removed by spec
    v1.22 W2 / Part M3a-fix: no baseline has it and HGB uses only dot and DistMult.)

    num_relations > 1 only happens for datasets whose official benchmark pools
    several raw relation ids into one target (currently just Amazon: rel_0/rel_1
    both scored jointly, matching Simple-HGN's run_dist.py). distmult then holds
    one d x d matrix per relation, selected per-edge via `edge_relation` — dot is
    relation-agnostic by construction (HGB itself only ever tries dot/DistMult decoders for
    link prediction) and ignores it.
    """

    def __init__(
        self,
        d: int,
        decoder_type: str,
        num_relations: int = 1,
        input_dim: int | None = None,
    ) -> None:
        """`input_dim` is the representation width the model passes: `d` for `x*` (spec 1.25 Z3),
        `2d` for `[x*, h0]` under `ablation.ego_head`, `d` under the `h0_only_readout` ablation
        (`None` means `2d`; the model always passes it). (The former `hidden_dim` and `dropout` belonged
        to the removed concat+MLP decoder; removed by the legacy-removal card, LR-3.)"""
        super().__init__()
        self.decoder_type = decoder_type
        self.endpoint_projection = nn.Sequential(
            nn.Linear(2 * d if input_dim is None else input_dim, d),
            nn.GELU(),
        )
        if decoder_type == "dot":
            pass
        elif decoder_type == "distmult":
            self.relation_matrices = nn.Parameter(
                torch.eye(d).unsqueeze(0).repeat(num_relations, 1, 1)
            )
        else:
            raise ValueError(f"Unsupported decoder_type '{decoder_type}'.")

    def forward(
        self,
        node_representation: torch.Tensor,
        edge_label_index: torch.Tensor,
        edge_relation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        endpoint_state = self.endpoint_projection(node_representation)
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
        raise ValueError(f"Unsupported decoder_type '{self.decoder_type}'.")
