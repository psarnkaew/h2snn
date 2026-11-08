from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.configs.config import TrainConfig
from src.data.fact_layer import FactLayer, apply_fact_layer, build_fact_layer
from src.data.path_sampler import build_schema_path_groups
from src.data.wisp_sampler import build_wisp_path_groups
from src.data.schemas import resolve_schemas
from src.data.hetero_flatten import flatten_edges, stack_node_features
from src.data.preprocessing import (
    compute_relation_diagnostics,
    deterministic_validation_split,
    enumerate_metapath_faces,
)
from src.data.structures import FaceGroup, PreparedH2SData

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData


def _load_hetero_data(config: TrainConfig) -> "HeteroData":
    dataset_name = config.data.dataset_name.lower()
    root = Path(config.data.root)

    if dataset_name == "acm":
        from torch_geometric.datasets import HGBDataset

        dataset = HGBDataset(root=str(root / "acm"), name="ACM")
        return dataset[0]
    if dataset_name == "dblp":
        from torch_geometric.datasets import HGBDataset

        dataset = HGBDataset(root=str(root / "dblp"), name="DBLP")
        return dataset[0]
    if dataset_name == "imdb":
        from torch_geometric.datasets import HGBDataset

        dataset = HGBDataset(root=str(root / "imdb"), name="IMDB")
        return dataset[0]
    raise ValueError(f"Unsupported dataset '{config.data.dataset_name}'.")


def _first_mask_column(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.dim() == 1:
        return mask.bool()
    return mask[:, 0].bool()


def _prepare_target_labels(
    data: "HeteroData",
    node_slices: dict[str, tuple[int, int]],
    target_node_type: str,
    val_ratio: float,
    split_seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    bool,
]:
    target_store = data[target_node_type]
    raw_target_y = target_store.y
    is_multilabel = raw_target_y.dim() == 2
    target_y = raw_target_y.float() if is_multilabel else raw_target_y.long()
    target_indices = torch.arange(
        node_slices[target_node_type][0],
        node_slices[target_node_type][1],
        dtype=torch.long,
    )

    train_mask = _first_mask_column(getattr(target_store, "train_mask", None))
    val_mask = _first_mask_column(getattr(target_store, "val_mask", None))
    test_mask = _first_mask_column(getattr(target_store, "test_mask", None))

    if train_mask is None:
        raise ValueError(f"Dataset '{target_node_type}' is missing train_mask.")
    if val_mask is None or val_mask.sum() == 0:
        train_mask, val_mask = deterministic_validation_split(
            train_mask=train_mask,
            val_ratio=val_ratio,
            seed=split_seed,
        )
    if test_mask is None:
        test_mask = torch.zeros_like(train_mask, dtype=torch.bool)

    total_nodes = max(end for _, end in node_slices.values())
    if is_multilabel:
        num_classes = int(target_y.size(1))
        global_y = torch.zeros((total_nodes, num_classes), dtype=target_y.dtype)
        global_y[target_indices] = target_y
    else:
        global_y = torch.full((total_nodes,), -1, dtype=torch.long)
        global_y[target_indices] = target_y
        num_classes = int(target_y.max().item() + 1)

    global_train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    global_val_mask = torch.zeros(total_nodes, dtype=torch.bool)
    global_test_mask = torch.zeros(total_nodes, dtype=torch.bool)
    global_train_mask[target_indices] = train_mask
    global_val_mask[target_indices] = val_mask
    global_test_mask[target_indices] = test_mask

    return (
        global_y,
        global_train_mask,
        global_val_mask,
        global_test_mask,
        target_indices,
        target_y,
        train_mask,
        num_classes,
        is_multilabel,
    )


def build_node_classification_paths(
    config: TrainConfig,
    fact_layer: FactLayer,
    node_slices,
    node_type: torch.Tensor,
    node_type_names: tuple[str, ...],
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    relation_tuples: tuple[tuple[str, str, str], ...],
    seed: int,
) -> tuple[FaceGroup, ...]:
    """Path cells of one draw: WISP (`path_source: wisp`, Part H: schema-free walks from every
    node of the target type, grouped by length), the walk sampler over the resolved schemas
    (D8; `seed` is the draw's seed), or the pre-D7 DFS (`path_sampler: dfs`, seeded by
    acm_split_seed)."""
    if config.data.path_source == "wisp":
        start, end = node_slices[config.data.target_node_type]
        return build_wisp_path_groups(
            edge_index=edge_index,
            edge_type=edge_type,
            node_type=node_type,
            self_inverse=fact_layer.self_inverse,
            anchors=torch.arange(start, end),
            walks_per_anchor=config.data.wisp_walks_per_anchor,
            max_length=config.data.wisp_max_length,
            seed=seed,
        )
    if config.data.path_sampler == "dfs":
        return enumerate_metapath_faces(
            node_slices=node_slices,
            edge_index=edge_index,
            edge_type=edge_type,
            relation_tuples=relation_tuples,
            templates=config.data.metapath_templates,
            exclude_identity_metapaths=config.data.exclude_identity_metapaths,
            split_seed=config.data.acm_split_seed,
        )
    schemas = resolve_schemas(config.data.metapath_templates, fact_layer, relation_tuples)

    def anchors_of(type_name: str) -> torch.Tensor:
        start, end = node_slices[type_name]
        return torch.arange(start, end)

    return build_schema_path_groups(
        edge_index=edge_index,
        edge_type=edge_type,
        node_type=node_type,
        node_type_names=node_type_names,
        schemas=schemas,
        self_inverse=fact_layer.self_inverse,
        tied=fact_layer.mode == "tied",
        anchors_of=anchors_of,
        default_walks=config.data.face_sample_k,
        seed=seed,
        sampler=config.data.path_sampler,
    )


def build_dataset(config: TrainConfig) -> PreparedH2SData:
    data = _load_hetero_data(config)
    if config.data.feature_normalization == "row":
        # D12 G0b diagnostic: PyG NormalizeFeatures per node type, before stacking (HetSheaf's transform).
        from torch_geometric.transforms import NormalizeFeatures

        data = NormalizeFeatures()(data)
    raw_x, node_type, node_type_names, node_slices, feature_dims = stack_node_features(data)
    edge_index, edge_type, relation_names, relation_tuples = flatten_edges(data, node_slices)
    # Fact layer (D7): rho from exact edge-set equality, one 1-cell per fact when tied.
    fact_layer = build_fact_layer(edge_index, edge_type, relation_names, raw_x.size(0),
                                  config.data.fact_layer)
    edge_index, edge_type = apply_fact_layer(fact_layer, edge_index, edge_type)
    relation_names = fact_layer.kept_names
    relation_tuples = tuple(relation_tuples[r] for r in fact_layer.kept)

    (
        y,
        train_mask,
        val_mask,
        test_mask,
        target_indices,
        target_y,
        target_train_mask,
        num_classes,
        is_multilabel,
    ) = _prepare_target_labels(
        data=data,
        node_slices=node_slices,
        target_node_type=config.data.target_node_type,
        val_ratio=config.data.acm_val_ratio,
        split_seed=config.data.acm_split_seed,
    )

    target_val_mask = val_mask[target_indices]
    target_test_mask = test_mask[target_indices]
    train_indices = target_indices[target_train_mask]
    train_labels = target_y[target_train_mask]

    face_groups = build_node_classification_paths(
        config=config,
        fact_layer=fact_layer,
        node_slices=node_slices,
        node_type=node_type,
        node_type_names=node_type_names,
        edge_index=edge_index,
        edge_type=edge_type,
        relation_tuples=relation_tuples,
        seed=config.seed,
    )

    # Reporting-only statistics; never concatenated onto node features. See
    # compute_relation_diagnostics for why appending a global vector to every node
    # would be a no-op for the type-specific encoder.
    if config.data.compute_relation_diagnostics:
        metrics = compute_relation_diagnostics(
            raw_x=raw_x,
            node_type=node_type,
            edge_index=edge_index,
            edge_type=edge_type,
            train_indices=train_indices,
            train_labels=train_labels,
            num_classes=num_classes,
            num_relations=len(relation_names),
            face_groups=face_groups,
            num_label_prop_steps=config.data.num_label_prop_steps,
        )
    else:
        metrics = torch.zeros(0)

    return PreparedH2SData(
        x=raw_x,
        y=y,
        node_type=node_type,
        node_type_names=node_type_names,
        node_slices=node_slices,
        feature_dims=feature_dims,
        edge_index=edge_index,
        edge_type=edge_type,
        relation_names=relation_names,
        relation_tuples=relation_tuples,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        target_node_type=config.data.target_node_type,
        target_indices=target_indices,
        target_y=target_y,
        target_train_mask=target_train_mask,
        target_val_mask=target_val_mask,
        target_test_mask=target_test_mask,
        num_classes=num_classes,
        is_multilabel=is_multilabel,
        face_groups=face_groups,
        metrics=metrics,
        fact_layer=fact_layer,
    )
