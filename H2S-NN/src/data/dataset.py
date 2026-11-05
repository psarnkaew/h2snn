from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import torch

from src.configs.config import TrainConfig
from src.data.hetero_flatten import flatten_edges, stack_node_features
from src.data.preprocessing import (
    compute_relation_diagnostics,
    deterministic_validation_split,
    enumerate_metapath_faces,
    label_propagation,
)
from src.data.node_stage1 import build_node_amp_ss_faces
from src.data.schema_structures import DiscoveryMetadata
from src.data.structures import PreparedH2SData

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


def build_dataset(
    config: TrainConfig,
    *,
    frozen_selector_manifest: Mapping[str, Any] | None = None,
) -> PreparedH2SData:
    data = _load_hetero_data(config)
    raw_x, node_type, node_type_names, node_slices, feature_dims = stack_node_features(data)
    edge_index, edge_type, relation_names, relation_tuples = flatten_edges(data, node_slices)

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

    discovery_metadata: DiscoveryMetadata | None = None
    selector_report = None

    if config.discovery.enabled and config.discovery.mode == "auto":
        # Run label propagation once here for train-only Stage-I schema signals.
        propagated_labels = label_propagation(
            edge_index=edge_index,
            train_indices=train_indices,
            train_labels=train_labels,
            num_nodes=raw_x.size(0),
            num_classes=num_classes,
            num_steps=config.data.num_label_prop_steps,
        )
        face_groups, discovery_metadata, closure_edge_index, selector_report = (
            build_node_amp_ss_faces(
                config=config,
                node_slices=node_slices,
                edge_index=edge_index,
                edge_type=edge_type,
                propagated_labels=propagated_labels,
                target_node_type=config.data.target_node_type,
                relation_tuples=relation_tuples,
                target_indices=target_indices,
                target_y=target_y,
                target_train_mask=target_train_mask,
                is_multilabel=is_multilabel,
                frozen_manifest=frozen_selector_manifest,
            )
        )
    else:
        face_groups, closure_edge_index = enumerate_metapath_faces(
            node_slices=node_slices,
            edge_index=edge_index,
            edge_type=edge_type,
            relation_tuples=relation_tuples,
            templates=config.data.metapath_templates,
            exclude_identity_metapaths=config.data.exclude_identity_metapaths,
            split_seed=config.data.acm_split_seed,
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
        closure_edge_index=closure_edge_index,
        metrics=metrics,
        discovery=discovery_metadata,
        selector_report=selector_report,
    )
