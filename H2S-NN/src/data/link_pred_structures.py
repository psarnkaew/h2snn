from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from src.data.fact_layer import FactLayer

from src.data.structures import FaceGroup


@dataclass(slots=True)
class PreparedLinkPredData:
    """Single transductive heterogeneous graph prepared for link prediction.

    Mirrors PreparedH2SData's shape/`.to(device)` pattern so `H2SNN.encode()` can be
    driven by either container. Edge indices are all in the global flattened index
    space (offset by node_slices), consistent with how flatten_edges already
    offsets message-passing edges.
    """

    x: torch.Tensor
    node_type: torch.Tensor
    node_type_names: tuple[str, ...]
    node_slices: Mapping[str, tuple[int, int]]
    feature_dims: Mapping[str, int]

    edge_index: torch.Tensor      # message-passing graph (train positives only)
    edge_type: torch.Tensor
    relation_names: tuple[str, ...]
    relation_tuples: tuple[tuple[str, str, str], ...]

    target_relation: tuple[str, str, str]

    train_pos_edge_index: torch.Tensor
    # Candidate pool to sample per-epoch train negatives from (e.g. LastFM ships ~33M
    # known non-edges). None means the dataset ships no candidate pool (AmazonBook) —
    # the task head generates negatives on the fly via torch_geometric.utils.negative_sampling
    # instead of subsampling a precomputed pool.
    train_neg_candidates: torch.Tensor | None
    val_pos_edge_index: torch.Tensor
    val_neg_edge_index: torch.Tensor
    test_pos_edge_index: torch.Tensor
    test_neg_edge_index: torch.Tensor
    known_pos_edge_index: torch.Tensor

    # Local index (0-based, into the pooled set of target relations — see
    # target_relation_ids in link_prediction_dataset.py) for every labeled edge
    # above. Size 1 / all-zeros for every dataset except Amazon, whose official
    # benchmark pools two relations (co-view, co-purchase) into one target —
    # LinkPredictionReadout's distmult decoder uses this to pick a per-relation
    # matrix, mirroring Simple-HGN's run_dist.py feeding relation id into the score.
    num_target_relations: int
    train_pos_edge_relation: torch.Tensor
    val_pos_edge_relation: torch.Tensor
    val_neg_edge_relation: torch.Tensor
    test_pos_edge_relation: torch.Tensor
    test_neg_edge_relation: torch.Tensor

    neg_sampling_mode: str
    neg_sampling_ratio: int
    num_src_nodes: int   # global-index bounds of the target relation's src/dst types
    num_dst_nodes: int
    src_node_offset: int
    dst_node_offset: int

    # Path cells, enumerated on the train-only message graph `edge_index`.
    face_groups: tuple[FaceGroup, ...]
    # rho, kept relations, unmatched edges (D7); None for data built without a fact layer.
    fact_layer: "FactLayer | None" = None
    # HGB get_test_neigh (spec v1.15 §6, J5): heads without 2-hop candidates get no negatives;
    # these heads (global ids) and the number of their test positives are reported per task.
    test_heads_without_negatives: torch.Tensor | None = None
    test_positives_without_negatives: int = 0

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))

    def to(self, device: torch.device | str) -> "PreparedLinkPredData":
        return PreparedLinkPredData(
            x=self.x.to(device),
            node_type=self.node_type.to(device),
            node_type_names=self.node_type_names,
            node_slices=self.node_slices,
            feature_dims=self.feature_dims,
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            relation_names=self.relation_names,
            relation_tuples=self.relation_tuples,
            target_relation=self.target_relation,
            train_pos_edge_index=self.train_pos_edge_index.to(device),
            train_neg_candidates=(
                self.train_neg_candidates.to(device)
                if self.train_neg_candidates is not None
                else None
            ),
            val_pos_edge_index=self.val_pos_edge_index.to(device),
            val_neg_edge_index=self.val_neg_edge_index.to(device),
            test_pos_edge_index=self.test_pos_edge_index.to(device),
            test_neg_edge_index=self.test_neg_edge_index.to(device),
            known_pos_edge_index=self.known_pos_edge_index.to(device),
            num_target_relations=self.num_target_relations,
            train_pos_edge_relation=self.train_pos_edge_relation.to(device),
            val_pos_edge_relation=self.val_pos_edge_relation.to(device),
            val_neg_edge_relation=self.val_neg_edge_relation.to(device),
            test_pos_edge_relation=self.test_pos_edge_relation.to(device),
            test_neg_edge_relation=self.test_neg_edge_relation.to(device),
            neg_sampling_mode=self.neg_sampling_mode,
            neg_sampling_ratio=self.neg_sampling_ratio,
            num_src_nodes=self.num_src_nodes,
            num_dst_nodes=self.num_dst_nodes,
            src_node_offset=self.src_node_offset,
            dst_node_offset=self.dst_node_offset,
            face_groups=tuple(group.to(device) for group in self.face_groups),
            fact_layer=self.fact_layer,
            test_heads_without_negatives=self.test_heads_without_negatives,
            test_positives_without_negatives=self.test_positives_without_negatives,
        )
