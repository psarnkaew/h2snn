"""Task heads shared by every baseline on link prediction and graph classification.

Most of the compared models were published as node classifiers only. Attaching whatever
head each adapter author felt like would make the tables measure decoders rather than
encoders, so every baseline gets the same head here and it is trained end to end with the
encoder.

The link decoders mirror HGB's own, which ships both ``Dot`` and ``DisMult`` and lets each
method choose (HGB/LP/benchmark/methods/GNN/GNN.py). Following that, ``decoder_type`` is a
searched hyperparameter for every link-prediction baseline and is selected on validation,
which also removes any asymmetry with H2SNN's own decoder choice.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool


class DotDecoder(nn.Module):
    """Relation-agnostic dot product. Mirrors HGB's ``Dot``."""

    def forward(
        self,
        relation_ids: torch.Tensor,  # noqa: ARG002  (kept for a uniform decoder call)
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return (source * target).sum(dim=-1)


class DistMultDecoder(nn.Module):
    """Per-relation bilinear form `uᵀ W_r v` with a full `d × d` matrix per relation, as HGB's ``DistMult``
    (`HGB/LP/benchmark/methods/baseline/GNN.py:10-20`: `W` of shape `(num_rel, dim, dim)`, `xavier_normal_` with
    gain 1.414) and as H2S-NN's decoder (card 8 C2b, author 2026-11-20). The diagonal form used before is a strict
    special case (symmetric in u and v, no cross-dimension terms).

    Scored one relation at a time (`(u @ W_r) · v` over the pairs of relation r): the same numbers as HGB's
    `bmm(bmm(u, W[r_id]), v)` without gathering one `d × d` matrix per pair.
    """

    def __init__(self, num_relations: int, dim: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.empty(num_relations, dim, dim))
        nn.init.xavier_normal_(self.weights, gain=1.414)  # HGB GNN.py:14

    def forward(
        self,
        relation_ids: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        scores = source.new_zeros(source.size(0))
        for relation in torch.unique(relation_ids).tolist():
            mask = relation_ids == relation
            scores = scores.index_put((mask.nonzero().view(-1),),
                                      ((source[mask] @ self.weights[relation]) * target[mask]).sum(dim=-1))
        return scores


def build_link_decoder(decoder_type: str, *, num_relations: int, dim: int) -> nn.Module:
    if decoder_type == "dot":
        return DotDecoder()
    if decoder_type == "distmult":
        return DistMultDecoder(num_relations=num_relations, dim=dim)
    raise ValueError(
        f"Unsupported decoder_type {decoder_type!r}; expected 'dot' or 'distmult'."
    )


class SharedLinkHead(nn.Module):
    """Score candidate edges from node embeddings produced by a baseline encoder."""

    def __init__(self, decoder_type: str, *, num_relations: int, dim: int) -> None:
        super().__init__()
        self.decoder_type = decoder_type
        self.decoder = build_link_decoder(
            decoder_type, num_relations=num_relations, dim=dim
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        relation_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source = embeddings[edge_index[0]]
        target = embeddings[edge_index[1]]
        if relation_ids is None:
            relation_ids = torch.zeros(
                edge_index.size(1), dtype=torch.long, device=edge_index.device
            )
        return self.decoder(relation_ids, source, target)


class SharedGraphHead(nn.Module):
    """Global mean pool followed by a linear classifier.

    Deliberately the plainest readout there is. H2SNN uses SheafCovPool, and the size of
    that difference is reported separately by the existing readout ablation on ENZYMES
    (paper/h2snn_manuscript.tex, tab:ablation-pooling) rather than being folded into these
    tables.
    """

    def __init__(self, dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, embeddings: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        pooled = global_mean_pool(embeddings, batch)
        return self.classifier(self.dropout(pooled))
