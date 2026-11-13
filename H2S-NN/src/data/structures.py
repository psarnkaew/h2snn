from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple

import torch

from src.data.fact_layer import FactLayer


@dataclass(slots=True)
class FaceGroup:
    template_name: str
    node_type_sequence: tuple[str, ...]
    face_node_index: torch.Tensor
    face_edge_index: torch.Tensor
    relation_index: torch.Tensor
    face_edge_sign: torch.Tensor
    # Merged weight m_f of each cell (D8: summed 1/r over the walks that drew it); None for
    # the DFS builders (every cell weight 1). Multiplies w_f in both L_P and D.
    path_weight: torch.Tensor | None = None
    # WISP (Part H): groups are per length, so cells differ in relation sequence; per-cell id into
    # `signature_table` (rows: t_0, r_1, d_1 + 1, t_1, ..., canonical orientation). None / () for
    # the schema and DFS builders (one signature per group).
    signature_index: torch.Tensor | None = None
    signature_table: tuple[tuple[int, ...], ...] = ()

    @property
    def num_faces(self) -> int:
        return int(self.face_node_index.size(0))

    @property
    def length(self) -> int:
        return int(self.face_edge_index.size(1))

    def to(self, device: torch.device | str) -> "FaceGroup":
        return FaceGroup(
            template_name=self.template_name,
            node_type_sequence=self.node_type_sequence,
            face_node_index=self.face_node_index.to(device),
            face_edge_index=self.face_edge_index.to(device),
            relation_index=self.relation_index.to(device),
            face_edge_sign=self.face_edge_sign.to(device),
            path_weight=None if self.path_weight is None else self.path_weight.to(device),
            signature_index=(None if self.signature_index is None
                             else self.signature_index.to(device)),
            signature_table=self.signature_table,
        )


def face_lengths_of(face_groups: "tuple[FaceGroup, ...]") -> tuple[int, ...]:
    """Distinct path lengths present in `face_groups`, ascending.

    The "concat" face encoder builds one input projection per length, so it needs this at
    construction time. Empty groups are skipped: they contribute no instances and would
    otherwise create a projection nothing ever uses.
    """
    return tuple(sorted({group.length for group in face_groups if group.num_faces > 0}))


class ScopeIndex(NamedTuple):
    """Scope of every node, fact and cell of a batch of scopes (Part MB, MB3): the per-scope PCG
    reads it to give each row of the solved system its scope."""

    node: torch.Tensor                 # [N] scope of each node
    fact: torch.Tensor                 # [E] scope of each fact
    cell: tuple[torch.Tensor, ...]     # per face group, [P_g] scope of each cell
    num_scopes: int

    def to(self, device: torch.device | str) -> "ScopeIndex":
        return ScopeIndex(self.node.to(device), self.fact.to(device),
                          tuple(c.to(device) for c in self.cell), self.num_scopes)


@dataclass(slots=True)
class PreparedH2SData:
    x: torch.Tensor
    y: torch.Tensor
    node_type: torch.Tensor
    node_type_names: tuple[str, ...]
    node_slices: Mapping[str, tuple[int, int]]
    feature_dims: Mapping[str, int]
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    relation_names: tuple[str, ...]
    relation_tuples: tuple[tuple[str, str, str], ...]
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    target_node_type: str
    target_indices: torch.Tensor
    target_y: torch.Tensor
    target_train_mask: torch.Tensor
    target_val_mask: torch.Tensor
    target_test_mask: torch.Tensor
    num_classes: int
    is_multilabel: bool
    face_groups: tuple[FaceGroup, ...]
    metrics: torch.Tensor
    # rho, kept relations, unmatched edges (D7); None for data built without a fact layer.
    fact_layer: FactLayer | None = None
    # Part MB (MB1): for a batch of scopes, each row's node id in the graph the scopes were cut
    # from; featureless types read their embedding row by this id. None = the whole graph (rows
    # are the node ids), the full-batch behaviour.
    global_node_id: torch.Tensor | None = None
    # Part MB (MB3): for a batch of scopes, the scope of every node, fact and cell (per-scope
    # PCG); None = one system (full batch).
    scope_index: ScopeIndex | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))

    def to(self, device: torch.device | str) -> "PreparedH2SData":
        return PreparedH2SData(
            x=self.x.to(device),
            y=self.y.to(device),
            node_type=self.node_type.to(device),
            node_type_names=self.node_type_names,
            node_slices=self.node_slices,
            feature_dims=self.feature_dims,
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            relation_names=self.relation_names,
            relation_tuples=self.relation_tuples,
            train_mask=self.train_mask.to(device),
            val_mask=self.val_mask.to(device),
            test_mask=self.test_mask.to(device),
            target_node_type=self.target_node_type,
            target_indices=self.target_indices.to(device),
            target_y=self.target_y.to(device),
            target_train_mask=self.target_train_mask.to(device),
            target_val_mask=self.target_val_mask.to(device),
            target_test_mask=self.target_test_mask.to(device),
            num_classes=self.num_classes,
            is_multilabel=self.is_multilabel,
            face_groups=tuple(group.to(device) for group in self.face_groups),
            metrics=self.metrics.to(device),
            fact_layer=self.fact_layer,
            global_node_id=(None if self.global_node_id is None
                            else self.global_node_id.to(device)),
            scope_index=None if self.scope_index is None else self.scope_index.to(device),
        )
