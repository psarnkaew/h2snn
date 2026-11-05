from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Mapping

import torch

from src.configs.config import MetaPathConfig
from src.data.face_geometry import (
    validate_face_orientation,
    validate_relation_sequence_constant,
)
from src.data.structures import FaceGroup


def deterministic_validation_split(
    train_mask: torch.Tensor,
    val_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_indices = train_mask.nonzero(as_tuple=False).view(-1)
    if train_indices.numel() == 0:
        return train_mask.clone(), torch.zeros_like(train_mask, dtype=torch.bool)

    num_val = max(1, int(round(train_indices.numel() * val_ratio)))
    if train_indices.numel() > 1:
        num_val = min(num_val, train_indices.numel() - 1)
    else:
        num_val = 0
    generator = torch.Generator(device=train_indices.device)
    generator.manual_seed(seed)
    permutation = train_indices[torch.randperm(train_indices.numel(), generator=generator)]
    val_indices = permutation[:num_val]
    remaining_indices = permutation[num_val:]

    new_train_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    new_val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    new_train_mask[remaining_indices] = True
    new_val_mask[val_indices] = True
    return new_train_mask, new_val_mask


def split_edge_index_train_val(
    edge_index: torch.Tensor,
    val_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Carve a validation edge set out of a positive edge_index (columns are edges).

    Mirrors deterministic_validation_split's reproducible-permutation approach, but
    splits edge columns instead of a boolean node mask — used for datasets (e.g.
    AmazonBook) that ship train positives only, with no validation split of their own.
    """
    num_edges = edge_index.size(1)
    if num_edges == 0:
        return edge_index.clone(), edge_index.new_zeros((2, 0))

    num_val = max(1, int(round(num_edges * val_ratio)))
    num_val = min(num_val, num_edges - 1) if num_edges > 1 else 0

    generator = torch.Generator(device=edge_index.device)
    generator.manual_seed(seed)
    permutation = torch.randperm(num_edges, generator=generator)

    val_columns = permutation[:num_val]
    train_columns = permutation[num_val:]
    return edge_index[:, train_columns], edge_index[:, val_columns]


def label_propagation(
    edge_index: torch.Tensor,
    train_indices: torch.Tensor,
    train_labels: torch.Tensor,
    num_nodes: int,
    num_classes: int,
    num_steps: int,
) -> torch.Tensor:
    device = edge_index.device
    propagated = torch.zeros((num_nodes, num_classes), device=device)
    if train_labels.dim() == 1:
        train_label_distribution = torch.nn.functional.one_hot(
            train_labels,
            num_classes=num_classes,
        ).float()
    else:
        train_label_distribution = train_labels.float()
        train_label_distribution = train_label_distribution / train_label_distribution.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)
    propagated[train_indices] = train_label_distribution

    if edge_index.numel() == 0 or num_steps <= 0:
        return propagated

    src_idx = edge_index[0]
    tgt_idx = edge_index[1]
    ones = torch.ones_like(tgt_idx, dtype=propagated.dtype)
    indegree = torch.zeros(num_nodes, device=device, dtype=propagated.dtype)
    indegree.index_add_(0, tgt_idx, ones)
    indegree = indegree.clamp_min(1.0)

    for _ in range(num_steps):
        updated = torch.zeros_like(propagated)
        updated.index_add_(0, tgt_idx, propagated[src_idx])
        updated = updated / indegree.unsqueeze(-1)
        updated[train_indices] = propagated[train_indices]
        row_sums = updated.sum(dim=-1, keepdim=True).clamp_min(1.0)
        propagated = updated / row_sums
        propagated[train_indices] = train_label_distribution

    return propagated


def compute_feature_atoms(
    raw_x: torch.Tensor,
    node_type: torch.Tensor,
    propagated_labels: torch.Tensor,
    train_indices: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if raw_x.size(1) == 0:
        return torch.ones((raw_x.size(0), 1), device=raw_x.device)

    standardised = torch.zeros_like(raw_x)
    unique_types = node_type.unique(sorted=True)
    for node_type_id in unique_types.tolist():
        mask = node_type == int(node_type_id)
        features = raw_x[mask]
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        standardised[mask] = (features - mean) / std

    prototypes = torch.zeros((num_classes, raw_x.size(1)), device=raw_x.device)
    for class_id in range(num_classes):
        if train_labels.dim() == 1:
            class_mask = train_labels == class_id
        else:
            class_mask = train_labels[:, class_id] > 0
        if class_mask.any():
            class_indices = train_indices[class_mask]
            prototypes[class_id] = standardised[class_indices].mean(dim=0)

    residual = standardised - propagated_labels @ prototypes
    return torch.softmax(residual, dim=-1)


def compute_structural_atoms(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    num_relations: int,
    face_groups: tuple[FaceGroup, ...],
) -> tuple[torch.Tensor, ...]:
    template_counts = torch.zeros((num_nodes, len(face_groups)), device=edge_index.device)
    for template_index, face_group in enumerate(face_groups):
        flat_nodes = face_group.face_node_index.reshape(-1)
        template_counts[:, template_index].index_add_(
            0,
            flat_nodes,
            torch.ones_like(flat_nodes, dtype=template_counts.dtype),
        )

    src_idx = edge_index[0]
    tgt_idx = edge_index[1]
    atoms: list[torch.Tensor] = []

    for relation_id in range(num_relations):
        relation_mask = edge_type == relation_id
        relation_src = src_idx[relation_mask]
        relation_tgt = tgt_idx[relation_mask]

        out_degree = torch.zeros(num_nodes, device=edge_index.device)
        in_degree = torch.zeros(num_nodes, device=edge_index.device)
        out_degree.index_add_(
            0,
            relation_src,
            torch.ones_like(relation_src, dtype=out_degree.dtype),
        )
        in_degree.index_add_(
            0,
            relation_tgt,
            torch.ones_like(relation_tgt, dtype=in_degree.dtype),
        )

        adjacency: dict[int, set[int]] = defaultdict(set)
        for source, target in zip(relation_src.tolist(), relation_tgt.tolist(), strict=False):
            adjacency[int(source)].add(int(target))

        two_hop = torch.zeros(num_nodes, device=edge_index.device)
        for source, first_hop_targets in adjacency.items():
            reached: set[int] = set()
            for middle in first_hop_targets:
                reached.update(adjacency.get(middle, set()))
            two_hop[source] = float(len(reached))

        descriptors = torch.cat(
            [
                out_degree.unsqueeze(-1),
                in_degree.unsqueeze(-1),
                two_hop.unsqueeze(-1),
                template_counts,
            ],
            dim=-1,
        )
        descriptors = torch.log1p(descriptors)
        mean = descriptors.mean(dim=0, keepdim=True)
        std = descriptors.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalised = (descriptors - mean) / std
        atoms.append(torch.softmax(normalised, dim=-1))

    return tuple(atoms)


def _agreement_score(source_atoms: torch.Tensor, target_atoms: torch.Tensor) -> torch.Tensor:
    return (source_atoms * target_atoms).sum(dim=-1).mean()


def _chance_baseline(source_atoms: torch.Tensor, target_atoms: torch.Tensor) -> torch.Tensor:
    source_mean = source_atoms.mean(dim=0)
    target_mean = target_atoms.mean(dim=0)
    return (source_mean * target_mean).sum()


def _normalise_signed_score(agreement: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-3
    baseline = baseline.clamp(min=epsilon, max=1.0 - epsilon)
    if agreement >= baseline:
        return (agreement - baseline) / (1.0 - baseline)
    return (agreement - baseline) / baseline


def _normalised_mutual_information(
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
) -> torch.Tensor:
    epsilon = 1e-8
    joint = torch.einsum("ec,ed->cd", source_labels, target_labels)
    joint = joint / joint.sum().clamp_min(epsilon)
    source_marginal = joint.sum(dim=1, keepdim=True)
    target_marginal = joint.sum(dim=0, keepdim=True)
    independent = source_marginal @ target_marginal
    mutual_information = (
        joint * (joint.clamp_min(epsilon).log() - independent.clamp_min(epsilon).log())
    ).sum()
    entropy = -(target_marginal * target_marginal.clamp_min(epsilon).log()).sum()
    return (mutual_information / entropy.clamp_min(epsilon)).clamp(min=0.0, max=1.0)


def compute_relation_metrics(
    propagated_labels: torch.Tensor,
    structural_atoms: tuple[torch.Tensor, ...],
    feature_atoms: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_relations: int,
) -> torch.Tensor:
    metrics: list[torch.Tensor] = []
    src_idx = edge_index[0]
    tgt_idx = edge_index[1]

    for relation_id in range(num_relations):
        relation_mask = edge_type == relation_id
        relation_src = src_idx[relation_mask]
        relation_tgt = tgt_idx[relation_mask]

        if relation_src.numel() == 0:
            metrics.extend(
                [
                    torch.tensor(0.0, device=edge_index.device),
                    torch.tensor(0.0, device=edge_index.device),
                    torch.tensor(0.0, device=edge_index.device),
                    torch.tensor(0.0, device=edge_index.device),
                ]
            )
            continue

        label_src = propagated_labels[relation_src]
        label_tgt = propagated_labels[relation_tgt]
        label_agreement = _agreement_score(label_src, label_tgt)
        label_baseline = _chance_baseline(label_src, label_tgt)

        struct_atoms = structural_atoms[relation_id]
        struct_src = struct_atoms[relation_src]
        struct_tgt = struct_atoms[relation_tgt]
        struct_agreement = _agreement_score(struct_src, struct_tgt)
        struct_baseline = _chance_baseline(struct_src, struct_tgt)

        feature_src = feature_atoms[relation_src]
        feature_tgt = feature_atoms[relation_tgt]
        feature_agreement = _agreement_score(feature_src, feature_tgt)
        feature_baseline = _chance_baseline(feature_src, feature_tgt)

        metrics.extend(
            [
                _normalise_signed_score(label_agreement, label_baseline),
                _normalise_signed_score(struct_agreement, struct_baseline),
                _normalise_signed_score(feature_agreement, feature_baseline),
                _normalised_mutual_information(label_src, label_tgt),
            ]
        )

    return torch.stack(metrics, dim=0)


def compute_relation_diagnostics(
    raw_x: torch.Tensor,
    node_type: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    train_indices: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    num_relations: int,
    face_groups: tuple[FaceGroup, ...],
    num_label_prop_steps: int,
) -> torch.Tensor:
    """Four training-only agreement statistics per relation type: `[4 * num_relations]`.

    A reporting artifact, not a model input. The vector is global — one value per
    (relation, statistic) shared by every node — so concatenating it onto node features
    would only re-parameterise the encoder's per-type bias. `scripts/h2snn_paper_studies.py`
    consumes it for the relation-metric table and heatmap.
    """
    propagated_labels = label_propagation(
        edge_index=edge_index,
        train_indices=train_indices,
        train_labels=train_labels,
        num_nodes=raw_x.size(0),
        num_classes=num_classes,
        num_steps=num_label_prop_steps,
    )
    structural_atoms = compute_structural_atoms(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=raw_x.size(0),
        num_relations=num_relations,
        face_groups=face_groups,
    )
    feature_atoms = compute_feature_atoms(
        raw_x=raw_x,
        node_type=node_type,
        propagated_labels=propagated_labels,
        train_indices=train_indices,
        train_labels=train_labels,
        num_classes=num_classes,
    )
    return compute_relation_metrics(
        propagated_labels=propagated_labels,
        structural_atoms=structural_atoms,
        feature_atoms=feature_atoms,
        edge_index=edge_index,
        edge_type=edge_type,
        num_relations=num_relations,
    )


def enumerate_metapath_faces(
    node_slices: Mapping[str, tuple[int, int]],
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    relation_tuples: tuple[tuple[str, str, str], ...],
    templates: tuple[MetaPathConfig, ...],
    exclude_identity_metapaths: bool,
    split_seed: int,
) -> tuple[tuple[FaceGroup, ...], torch.Tensor]:
    transition_lookup: dict[tuple[str, str], list[int]] = defaultdict(list)
    adjacency_by_transition: dict[tuple[str, str], dict[int, list[tuple[int, int]]]] = {}

    for relation_id, (src_type, _, dst_type) in enumerate(relation_tuples):
        transition_key = (src_type, dst_type)
        transition_lookup[transition_key].append(relation_id)
        adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
        relation_mask = edge_type == relation_id
        relation_edges = edge_index[:, relation_mask]
        relation_edge_ids = relation_mask.nonzero(as_tuple=False).view(-1)
        for offset, edge_id in enumerate(relation_edge_ids.tolist()):
            src = int(relation_edges[0, offset].item())
            dst = int(relation_edges[1, offset].item())
            adjacency[src].append((dst, int(edge_id)))
        adjacency_by_transition[transition_key] = adjacency

    face_groups: list[FaceGroup] = []
    closure_edges: list[list[int]] = []

    for template_index, template in enumerate(templates):
        path_nodes: list[list[int]] = []
        path_edges: list[list[int]] = []
        relation_rows: list[list[int]] = []
        transition_keys = list(zip(template.node_types[:-1], template.node_types[1:], strict=False))
        relation_ids: list[int] = []

        for transition_key in transition_keys:
            candidates = transition_lookup.get(transition_key, [])
            if len(candidates) != 1:
                raise ValueError(
                    "Expected exactly one relation for transition "
                    f"{transition_key}, found {len(candidates)}."
                )
            relation_ids.append(candidates[0])

        start_type = template.node_types[0]
        start_begin, start_end = node_slices[start_type]

        for start_node in range(start_begin, start_end):
            rng = random.Random(split_seed + 9973 * template_index + start_node)
            sampled_node_paths: list[list[int]] = []
            sampled_edge_paths: list[list[int]] = []

            def dfs(current_node: int, depth: int, node_path: list[int], edge_path: list[int]) -> None:
                if (
                    template.sample_per_anchor is not None
                    and len(sampled_node_paths) >= template.sample_per_anchor
                ):
                    return
                if depth == len(relation_ids):
                    # The revisit guard below already rejects node_path[0] == node_path[-1],
                    # so `exclude_identity_metapaths` is subsumed and no longer consulted:
                    # endpoint coincidence is excluded unconditionally because it degenerates
                    # the closure edge into a self-loop, which the transport construction
                    # cannot close (see src/data/face_geometry.py).
                    sampled_node_paths.append(node_path.copy())
                    sampled_edge_paths.append(edge_path.copy())
                    return

                adjacency = adjacency_by_transition[transition_keys[depth]]
                candidates = list(adjacency.get(current_node, []))
                rng.shuffle(candidates)
                for next_node, edge_id in candidates:
                    if next_node in node_path:
                        # Keep faces simple: one transport map per node position, so an
                        # instance may not revisit a node. Also drops backtracking walks
                        # (A -> P -> A -> ...), which carry no meta-path information.
                        continue
                    node_path.append(next_node)
                    edge_path.append(edge_id)
                    dfs(next_node, depth + 1, node_path, edge_path)
                    node_path.pop()
                    edge_path.pop()

            dfs(start_node, 0, [start_node], [])
            path_nodes.extend(sampled_node_paths)
            path_edges.extend(sampled_edge_paths)
            relation_rows.extend([relation_ids.copy() for _ in sampled_node_paths])

        if not path_nodes:
            continue

        face_node_index = torch.tensor(path_nodes, dtype=torch.long)
        face_edge_index = torch.tensor(path_edges, dtype=torch.long)
        relation_index = torch.tensor(relation_rows, dtype=torch.long)
        # DFS above only ever walks forward through adjacency_by_transition (built from
        # each relation's stored src->dst direction), so every boundary edge is traversed
        # in its natural orientation and the incidence sign is always +1. Asserted rather
        # than assumed, since that is a property of enumeration and not of the data.
        validate_face_orientation(
            face_node_index=face_node_index,
            face_edge_index=face_edge_index,
            edge_index=edge_index,
            template_name=template.name,
        )
        validate_relation_sequence_constant(
            relation_index=relation_index,
            template_name=template.name,
        )
        face_edge_sign = torch.ones_like(face_edge_index, dtype=torch.float32)
        closure_edge_index = torch.stack(
            [face_node_index[:, -1], face_node_index[:, 0]],
            dim=1,
        )
        closure_edges.extend(closure_edge_index.tolist())
        face_groups.append(
            FaceGroup(
                template_name=template.name,
                node_type_sequence=template.node_types,
                face_node_index=face_node_index,
                face_edge_index=face_edge_index,
                relation_index=relation_index,
                face_edge_sign=face_edge_sign,
                closure_edge_index=closure_edge_index,
            )
        )

    if closure_edges:
        closure_edge_tensor = torch.tensor(closure_edges, dtype=torch.long).t().contiguous()
    else:
        closure_edge_tensor = torch.empty((2, 0), dtype=torch.long)
    return tuple(face_groups), closure_edge_tensor
