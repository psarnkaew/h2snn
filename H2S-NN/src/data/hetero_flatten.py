from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData


def ensure_feature_matrix(node_store: object, num_nodes: int) -> torch.Tensor:
    x = getattr(node_store, "x", None)
    if x is None:
        return torch.zeros((num_nodes, 0), dtype=torch.float32)
    return x.float()


def stack_node_features(data: "HeteroData") -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[str, ...],
    dict[str, tuple[int, int]],
    dict[str, int],
]:
    node_types = tuple(str(node_type) for node_type in data.node_types)
    feature_dims: dict[str, int] = {}
    node_slices: dict[str, tuple[int, int]] = {}
    padded_features: list[torch.Tensor] = []
    node_type_ids: list[torch.Tensor] = []
    max_dim = 0

    raw_features_by_type: dict[str, torch.Tensor] = {}
    offset = 0
    for node_type_id, node_type in enumerate(node_types):
        node_store = data[node_type]
        num_nodes = int(node_store.num_nodes)
        features = ensure_feature_matrix(node_store, num_nodes)
        raw_features_by_type[node_type] = features
        feature_dims[node_type] = int(features.size(1))
        max_dim = max(max_dim, int(features.size(1)))
        node_slices[node_type] = (offset, offset + num_nodes)
        node_type_ids.append(torch.full((num_nodes,), node_type_id, dtype=torch.long))
        offset += num_nodes

    for node_type in node_types:
        features = raw_features_by_type[node_type]
        if features.size(1) < max_dim:
            padding = torch.zeros((features.size(0), max_dim - features.size(1)))
            features = torch.cat([features, padding], dim=-1)
        padded_features.append(features)

    x = torch.cat(padded_features, dim=0)
    node_type_tensor = torch.cat(node_type_ids, dim=0)
    return x, node_type_tensor, node_types, node_slices, feature_dims


def flatten_edges(
    data: "HeteroData",
    node_slices: dict[str, tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], tuple[tuple[str, str, str], ...]]:
    edge_index_parts: list[torch.Tensor] = []
    edge_type_parts: list[torch.Tensor] = []
    relation_names: list[str] = []
    relation_tuples: list[tuple[str, str, str]] = []

    for relation_id, edge_type in enumerate(data.edge_types):
        src_type, relation_name, dst_type = edge_type
        relation_tuples.append((str(src_type), str(relation_name), str(dst_type)))
        relation_names.append(f"{src_type}__{relation_name}__{dst_type}")

        edge_index = data[edge_type].edge_index.long().clone()
        edge_index[0] += node_slices[str(src_type)][0]
        edge_index[1] += node_slices[str(dst_type)][0]
        edge_index_parts.append(edge_index)
        edge_type_parts.append(
            torch.full((edge_index.size(1),), relation_id, dtype=torch.long)
        )

    return (
        torch.cat(edge_index_parts, dim=1),
        torch.cat(edge_type_parts, dim=0),
        tuple(relation_names),
        tuple(relation_tuples),
    )
