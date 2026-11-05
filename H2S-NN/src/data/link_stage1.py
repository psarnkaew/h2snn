from __future__ import annotations

import random
from collections import defaultdict
from typing import Mapping, Sequence

import torch

from src.configs.config import TrainConfig
from src.data.schema_discovery import (
    allocate_face_quotas,
    build_schema_graph,
    materialize_sampled_schemas,
    schema_stats_from_samples,
)
from src.data.schema_sampling import (
    SchemaPathSample,
    SampledPath,
    canonical_schema_identity,
    sampled_support_hash,
    sample_schema_paths,
    truncate_schema_path_sample,
)
from src.data.selector_reporting import (
    build_coverage_diagnostic,
    candidate_statistics_from_samples,
    enrich_report,
    selected_schema_statistics,
    tensor_content_hash,
)
from src.data.schema_structures import (
    DiscoveryMetadata,
    SchemaCandidate,
    make_discovery_metadata,
)
from src.data.preprocessing import label_propagation
from src.data.stage1_selector import (
    SelectorReport,
    TrialSupport,
    derive_seed,
    make_selector_split,
    select_schema_features,
)
from src.data.structures import FaceGroup


def generate_link_schema_candidates(
    relation_tuples: tuple[tuple[str, str, str], ...],
    source_node_type: str,
    destination_node_type: str,
    min_length: int,
    max_length: int,
    require_source_start: bool,
    require_destination_end: bool,
    allow_repeated_node_types: bool,
    max_identities: int,
) -> tuple[SchemaCandidate, ...]:
    """Enumerate typed path schemas whose endpoints match the link task."""
    schema_adjacency = build_schema_graph(relation_tuples)
    identities: set[
        tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]
    ] = set()
    start_types = (
        [source_node_type] if require_source_start else sorted(schema_adjacency)
    )

    def dfs(
        current_type: str,
        node_types: list[str],
        relation_ids: list[int],
        relation_names: list[str],
    ) -> None:
        length = len(relation_ids)
        if length >= min_length:
            if not require_destination_end or current_type == destination_node_type:
                identities.add((tuple(node_types), tuple(relation_ids), tuple(relation_names)))
                if len(identities) > max_identities:
                    raise ValueError(
                        "AMPSS feasible link-schema count exceeds "
                        f"max_schema_identities guard: more than {max_identities}."
                    )
        if length == max_length:
            return
        for next_rel_id, next_rel_name, next_type in sorted(
            schema_adjacency.get(current_type, ()),
            key=lambda item: (item[0], item[1], item[2]),
        ):
            if not allow_repeated_node_types and next_type in node_types:
                continue
            dfs(
                next_type,
                node_types + [next_type],
                relation_ids + [next_rel_id],
                relation_names + [next_rel_name],
            )

    for start_type in sorted(start_types):
        dfs(start_type, [start_type], [], [])

    ordered = sorted(identities, key=lambda item: (item[0], item[1], item[2]))
    return tuple(
        SchemaCandidate(
            schema_id=index,
            name="__".join(node_types),
            node_types=node_types,
            relation_ids=relation_ids,
            relation_names=relation_names,
        )
        for index, (node_types, relation_ids, relation_names) in enumerate(ordered)
    )


def _anchors_from_slices(
    node_slices: Mapping[str, tuple[int, int]],
) -> dict[str, tuple[int, ...]]:
    return {
        node_type: tuple(range(begin, end))
        for node_type, (begin, end) in node_slices.items()
    }


def _known_tails_by_source(edge_index: torch.Tensor) -> dict[int, set[int]]:
    known: dict[int, set[int]] = defaultdict(set)
    for src, dst in edge_index.detach().cpu().t().tolist():
        known[int(src)].add(int(dst))
    return known


def _sample_selector_tail_negatives(
    reference_pos_edge_index: torch.Tensor,
    known_positive_edge_index: torch.Tensor,
    dst_range: tuple[int, int],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    dst_start, dst_end = dst_range
    known_by_src = _known_tails_by_source(known_positive_edge_index)
    sampled: list[tuple[int, int]] = []
    kept_indices: list[int] = []
    used_negatives: set[tuple[int, int]] = set()
    for reference_index, src_value in enumerate(
        reference_pos_edge_index[0].detach().cpu().tolist()
    ):
        src = int(src_value)
        known_tails = known_by_src.get(src, set())
        candidates = [
            dst for dst in range(dst_start, dst_end) if dst not in known_tails
        ]
        if not candidates:
            continue
        candidates = [
            destination
            for destination in candidates
            if (src, destination) not in used_negatives
        ]
        if not candidates:
            continue
        negative = (src, rng.choice(candidates))
        sampled.append(negative)
        used_negatives.add(negative)
        kept_indices.append(reference_index)
    if not sampled:
        empty_edges = reference_pos_edge_index.new_empty((2, 0))
        return empty_edges, reference_pos_edge_index.new_empty((0,))
    negatives = torch.tensor(
        sampled, dtype=torch.long, device=reference_pos_edge_index.device
    ).t().contiguous()
    kept = torch.tensor(
        kept_indices,
        dtype=torch.long,
        device=reference_pos_edge_index.device,
    )
    return negatives, kept


def _sample_selector_positives(
    train_pos_edge_index: torch.Tensor,
    max_positive_examples: int,
    seed: int,
) -> torch.Tensor:
    if train_pos_edge_index.size(1) <= max_positive_examples:
        return train_pos_edge_index
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(train_pos_edge_index.size(1), generator=generator)
    chosen = order[:max_positive_examples].sort().values
    return train_pos_edge_index[:, chosen]


def _build_fit_role_signal(
    *,
    edge_index: torch.Tensor,
    fit_positive_edge_index: torch.Tensor,
    num_nodes: int,
    num_steps: int,
) -> torch.Tensor:
    roles = torch.zeros((num_nodes, 2), dtype=torch.float32, device=edge_index.device)
    if fit_positive_edge_index.numel() == 0:
        return roles

    ones = torch.ones(
        fit_positive_edge_index.size(1),
        dtype=roles.dtype,
        device=roles.device,
    )
    roles[:, 0].index_add_(0, fit_positive_edge_index[0], ones)
    roles[:, 1].index_add_(0, fit_positive_edge_index[1], ones)
    role_mass = roles.sum(dim=-1)
    fit_nodes = role_mass.nonzero(as_tuple=False).view(-1)
    if fit_nodes.numel() == 0:
        return roles

    fit_roles = roles[fit_nodes] / role_mass[fit_nodes].unsqueeze(-1).clamp_min(1.0)
    return label_propagation(
        edge_index=edge_index,
        train_indices=fit_nodes,
        train_labels=fit_roles,
        num_nodes=num_nodes,
        num_classes=2,
        num_steps=num_steps,
    )


def _build_selector_edges(
    train_pos_edge_index: torch.Tensor,
    known_positive_edge_index: torch.Tensor,
    dst_range: tuple[int, int],
    max_examples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_positive = max(1, max_examples // 2)
    positives = _sample_selector_positives(
        train_pos_edge_index=train_pos_edge_index,
        max_positive_examples=max_positive,
        seed=derive_seed(seed, "selector_positive_subset"),
    )
    negatives, kept_positive_indices = _sample_selector_tail_negatives(
        reference_pos_edge_index=positives,
        known_positive_edge_index=known_positive_edge_index,
        dst_range=dst_range,
        seed=derive_seed(seed, "selector_negative_edges"),
    )
    if positives.numel() == 0 or negatives.numel() == 0:
        empty = train_pos_edge_index.new_empty((2, 0))
        return empty, torch.zeros(0), empty
    positives = positives[:, kept_positive_indices]
    edge_label_index = torch.cat([positives, negatives], dim=1)
    labels = torch.cat(
        [
            torch.ones(positives.size(1), dtype=torch.float32),
            torch.zeros(negatives.size(1), dtype=torch.float32),
        ]
    )
    return edge_label_index, labels, positives


def _remove_inner_selection_edges(
    *,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    relation_tuples: tuple[tuple[str, str, str], ...],
    source_node_type: str,
    destination_node_type: str,
    held_out_positive_edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove inner-selection target edges and stored reverse counterparts."""
    held_out_pairs = {
        (int(source), int(destination))
        for source, destination in held_out_positive_edge_index.detach().cpu().t().tolist()
    }
    if not held_out_pairs:
        return edge_index, edge_type
    keep: list[bool] = []
    for source, destination, relation_id in zip(
        edge_index[0].detach().cpu().tolist(),
        edge_index[1].detach().cpu().tolist(),
        edge_type.detach().cpu().tolist(),
        strict=True,
    ):
        relation = relation_tuples[int(relation_id)]
        direct_target = (
            relation[0] == source_node_type
            and relation[2] == destination_node_type
            and (int(source), int(destination)) in held_out_pairs
        )
        reverse_target = (
            relation[0] == destination_node_type
            and relation[2] == source_node_type
            and (int(destination), int(source)) in held_out_pairs
        )
        keep.append(not (direct_target or reverse_target))
    mask = torch.tensor(keep, dtype=torch.bool, device=edge_index.device)
    return edge_index[:, mask], edge_type[mask]


def _endpoint_schema_features(
    *,
    node: int,
    sample: SchemaPathSample,
    propagated_roles: torch.Tensor,
    attempts_per_anchor: int,
) -> torch.Tensor:
    role_dim = int(propagated_roles.size(1))
    features = torch.zeros(2 * (role_dim + 2), dtype=torch.float32)
    offset = 0
    for paths, counterpart_position in (
        (sample.forward_by_anchor.get(node, ()), -1),
        (sample.backward_by_anchor.get(node, ()), 0),
    ):
        count = len(paths)
        features[offset] = count / max(attempts_per_anchor, 1)
        if count > 0:
            counterparts = [path.nodes[counterpart_position] for path in paths]
            features[offset + 1] = len(set(counterparts)) / count
            counterpart_index = torch.tensor(counterparts, dtype=torch.long)
            evidence = propagated_roles[counterpart_index].float().mean(dim=0)
            features[offset + 2 : offset + 2 + role_dim] = evidence
        offset += role_dim + 2
    return features


def _direct_path_counts(
    *,
    source: int,
    destination: int,
    forward_paths: Sequence[SampledPath],
    backward_paths: Sequence[SampledPath],
) -> tuple[int, int]:
    forward_count = sum(1 for path in forward_paths if path.nodes[-1] == destination)
    backward_count = sum(1 for path in backward_paths if path.nodes[0] == source)
    return forward_count, backward_count


def _link_schema_feature_block(
    *,
    sample: SchemaPathSample,
    edge_label_index: torch.Tensor,
    propagated_roles: torch.Tensor,
    attempts_per_anchor: int,
) -> tuple[torch.Tensor, frozenset[int]]:
    role_dim = int(propagated_roles.size(1))
    endpoint_dim = 2 * (role_dim + 2)
    features = torch.zeros(
        (edge_label_index.size(1), 2 + 2 * endpoint_dim),
        dtype=torch.float32,
    )
    covered_examples: set[int] = set()
    for row, (source_value, destination_value) in enumerate(
        edge_label_index.detach().cpu().t().tolist()
    ):
        source = int(source_value)
        destination = int(destination_value)
        forward_paths = sample.forward_by_anchor.get(source, ())
        backward_paths = sample.backward_by_anchor.get(destination, ())
        direct_forward, direct_backward = _direct_path_counts(
            source=source,
            destination=destination,
            forward_paths=forward_paths,
            backward_paths=backward_paths,
        )
        features[row, 0] = direct_forward / max(attempts_per_anchor, 1)
        features[row, 1] = direct_backward / max(attempts_per_anchor, 1)
        if direct_forward + direct_backward > 0:
            covered_examples.add(row)
        source_features = _endpoint_schema_features(
            node=source,
            sample=sample,
            propagated_roles=propagated_roles,
            attempts_per_anchor=attempts_per_anchor,
        )
        destination_features = _endpoint_schema_features(
            node=destination,
            sample=sample,
            propagated_roles=propagated_roles,
            attempts_per_anchor=attempts_per_anchor,
        )
        features[row, 2 : 2 + endpoint_dim] = source_features
        features[row, 2 + endpoint_dim :] = destination_features
    return features, frozenset(covered_examples)


def _positive_pair_coverage(
    sample: SchemaPathSample,
    positive_edge_index: torch.Tensor,
) -> frozenset[int]:
    covered: set[int] = set()
    for unit, (source_value, destination_value) in enumerate(
        positive_edge_index.detach().cpu().t().tolist()
    ):
        source = int(source_value)
        destination = int(destination_value)
        forward_paths = sample.forward_by_anchor.get(source, ())
        backward_paths = sample.backward_by_anchor.get(destination, ())
        direct_forward, direct_backward = _direct_path_counts(
            source=source,
            destination=destination,
            forward_paths=forward_paths,
            backward_paths=backward_paths,
        )
        if direct_forward + direct_backward > 0:
            covered.add(unit)
    return frozenset(covered)


def _endpoint_node_coverage_sets(
    samples: Sequence[SchemaPathSample],
    positive_edge_index: torch.Tensor,
) -> tuple[tuple[frozenset[int], ...], int]:
    source_nodes = sorted(
        {int(value) for value in positive_edge_index[0].detach().cpu().tolist()}
    )
    destination_nodes = sorted(
        {int(value) for value in positive_edge_index[1].detach().cpu().tolist()}
    )
    units = [
        *((0, node) for node in source_nodes),
        *((1, node) for node in destination_nodes),
    ]
    unit_by_role_node = {value: index for index, value in enumerate(units)}
    coverage_sets = []
    for sample in samples:
        covered: set[int] = set()
        for path in sample.materialized_paths:
            source_unit = unit_by_role_node.get((0, int(path.nodes[0])))
            destination_unit = unit_by_role_node.get((1, int(path.nodes[-1])))
            if source_unit is not None:
                covered.add(source_unit)
            if destination_unit is not None:
                covered.add(destination_unit)
        coverage_sets.append(frozenset(covered))
    return tuple(coverage_sets), len(units)


def build_link_amp_ss_faces(
    config: TrainConfig,
    *,
    node_slices: Mapping[str, tuple[int, int]],
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    relation_tuples: tuple[tuple[str, str, str], ...],
    source_node_type: str,
    destination_node_type: str,
    train_pos_edge_index: torch.Tensor,
    known_positive_edge_index: torch.Tensor,
    dst_range: tuple[int, int],
    propagated_roles: torch.Tensor,
) -> tuple[
    tuple[FaceGroup, ...],
    DiscoveryMetadata | None,
    torch.Tensor,
    SelectorReport | None,
]:
    split_seed = derive_seed(
        config.discovery.selector_seed,
        "ampss_link_selector_split",
        config.data.dataset_name,
    )
    selector_sample_seed = derive_seed(
        config.discovery.selector_seed,
        "ampss_link_selector_path_sample",
        config.data.dataset_name,
    )
    materialization_seed = derive_seed(
        config.discovery.materialization_seed,
        "ampss_link_materialization_path_sample",
        config.data.dataset_name,
    )
    selector_seed = derive_seed(
        config.discovery.selector_seed,
        "ampss_link_selector_edges",
        config.data.dataset_name,
    )
    selector_edges, labels, selector_positive_edges = _build_selector_edges(
        train_pos_edge_index=train_pos_edge_index,
        known_positive_edge_index=known_positive_edge_index,
        dst_range=dst_range,
        max_examples=config.discovery.selector_max_examples,
        seed=selector_seed,
    )
    if labels.numel() == 0:
        return tuple(), None, edge_index.new_empty((2, 0)), None

    split = make_selector_split(
        labels=labels,
        task="binary",
        holdout_ratio=config.discovery.selector_holdout_ratio,
        seed=split_seed,
        min_examples_per_label=config.discovery.selector_min_examples_per_label,
    )
    fit_positive_indices = split.fit_indices[labels[split.fit_indices] > 0.5]
    selection_positive_indices = split.selection_indices[
        labels[split.selection_indices] > 0.5
    ]
    selector_edge_index, selector_edge_type = _remove_inner_selection_edges(
        edge_index=edge_index,
        edge_type=edge_type,
        relation_tuples=relation_tuples,
        source_node_type=source_node_type,
        destination_node_type=destination_node_type,
        held_out_positive_edge_index=selector_edges[:, selection_positive_indices],
    )
    fit_propagated_roles = _build_fit_role_signal(
        edge_index=selector_edge_index,
        fit_positive_edge_index=selector_edges[:, fit_positive_indices],
        num_nodes=int(propagated_roles.size(0)),
        num_steps=config.data.num_label_prop_steps,
    )
    candidates = generate_link_schema_candidates(
        relation_tuples=relation_tuples,
        source_node_type=source_node_type,
        destination_node_type=destination_node_type,
        min_length=config.discovery.min_schema_length,
        max_length=config.discovery.max_schema_length,
        require_source_start=config.discovery.require_target_start,
        require_destination_end=config.discovery.require_target_end,
        allow_repeated_node_types=config.discovery.allow_repeated_node_types,
        max_identities=config.discovery.max_schema_identities,
    )

    target_begin, target_end = node_slices[source_node_type]
    dst_begin, dst_end = node_slices[destination_node_type]
    target_nodes = set(range(target_begin, target_end))
    target_nodes.update(range(dst_begin, dst_end))
    samples = sample_schema_paths(
        candidates=candidates,
        edge_index=selector_edge_index,
        edge_type=selector_edge_type,
        anchor_nodes_by_type=_anchors_from_slices(node_slices),
        target_nodes=frozenset(target_nodes),
        attempts_per_anchor=config.discovery.selector_num_attempts,
        max_instances_per_schema=config.discovery.max_instances_per_schema,
        sample_seed=selector_sample_seed,
        max_viable_samples=config.discovery.max_schema_candidates,
        max_total_incidences=config.discovery.max_total_incidences,
    )

    incidence_budget = config.discovery.max_total_incidences
    viable_samples = tuple(
        sample
        for sample in samples
        if sample.unique_paths > 0
        and (
            incidence_budget is None
            or sample.schema.length + 1 <= incidence_budget
        )
    )
    if len(viable_samples) > config.discovery.max_schema_candidates:
        raise ValueError(
            "AMPSS viable link-schema count exceeds max_schema_candidates guard: "
            f"{len(viable_samples)} > {config.discovery.max_schema_candidates}."
        )

    prefix_cache: dict[
        tuple[int, int],
        tuple[SchemaPathSample, torch.Tensor, frozenset[int]],
    ] = {}

    def prefix_support(
        candidate_index: int,
        quota: int,
    ) -> tuple[SchemaPathSample, torch.Tensor, frozenset[int]]:
        key = (candidate_index, quota)
        cached = prefix_cache.get(key)
        if cached is not None:
            return cached
        truncated = truncate_schema_path_sample(
            viable_samples[candidate_index],
            quota,
        )
        block = _link_schema_feature_block(
            sample=truncated,
            edge_label_index=selector_edges,
            propagated_roles=fit_propagated_roles,
            attempts_per_anchor=config.discovery.selector_num_attempts,
        )[0]
        covered = _positive_pair_coverage(truncated, selector_positive_edges)
        value = (truncated, block, covered)
        prefix_cache[key] = value
        return value

    standalone_quotas = [
        allocate_face_quotas(
            [sample.unique_paths],
            [sample.schema.length],
            config.discovery.max_total_instances,
            config.discovery.max_total_incidences,
            [canonical_schema_identity(sample.schema)],
        )[0]
        for sample in viable_samples
    ]
    feature_blocks = tuple(
        prefix_support(index, quota)[1]
        for index, quota in enumerate(standalone_quotas)
    )
    coverage_sets = tuple(
        prefix_support(index, quota)[2]
        for index, quota in enumerate(standalone_quotas)
    )

    def subset_support(indices) -> TrialSupport | None:
        canonical_indices = tuple(
            sorted(
                (int(index) for index in indices),
                key=lambda index: canonical_schema_identity(
                    viable_samples[index].schema
                ),
            )
        )
        subset = [viable_samples[index] for index in canonical_indices]
        quotas = allocate_face_quotas(
            [sample.unique_paths for sample in subset],
            [sample.schema.length for sample in subset],
            config.discovery.max_total_instances,
            config.discovery.max_total_incidences,
            [canonical_schema_identity(sample.schema) for sample in subset],
        )
        if any(quota <= 0 for quota in quotas):
            return None
        supports = [
            prefix_support(index, quota)
            for index, quota in zip(canonical_indices, quotas, strict=True)
        ]
        truncated_samples = tuple(value[0] for value in supports)
        return TrialSupport(
            candidate_indices=canonical_indices,
            quotas=tuple(quotas),
            feature_blocks=tuple(value[1] for value in supports),
            coverage_sets=tuple(value[2] for value in supports),
            realized_incidence=float(
                sum(
                    quota * (sample.schema.length + 1)
                    for quota, sample in zip(quotas, subset, strict=True)
                )
            ),
            support_hash=sampled_support_hash(truncated_samples),
        )

    result = select_schema_features(
        candidate_identities=tuple(
            canonical_schema_identity(sample.schema) for sample in viable_samples
        ),
        feature_blocks=feature_blocks,
        coverage_sets=coverage_sets,
        incidence_masses=tuple(sample.incidence_cost for sample in viable_samples),
        labels=labels,
        split=split,
        task="binary",
        max_schemas=config.discovery.selected_schema_k,
        incidence_budget=incidence_budget,
        stop_tolerance=config.discovery.selector_stop_tolerance,
        tie_tolerance=config.discovery.selector_tie_tolerance,
        probe_l2=config.discovery.selector_probe_l2,
        probe_max_iter=config.discovery.selector_probe_max_iter,
        split_seed=split_seed,
        sample_seed=selector_sample_seed,
        candidate_identity_count=len(candidates),
        coverage_denominator=int(selector_positive_edges.size(1)),
        sampling_attempts=sum(sample.attempts for sample in samples),
        sampling_successes=sum(sample.successes for sample in samples),
        sampling_unique_paths=sum(sample.unique_paths for sample in samples),
        subset_support_fn=subset_support,
    )
    candidate_identities = tuple(
        canonical_schema_identity(sample.schema) for sample in viable_samples
    )
    candidate_statistics = candidate_statistics_from_samples(
        all_samples=samples,
        viable_samples=viable_samples,
        feature_blocks=feature_blocks,
        coverage_sets=coverage_sets,
        coverage_denominator=int(selector_positive_edges.size(1)),
        selected_indices=result.selected_indices,
        selection_steps=result.report.steps,
        incidence_budget=incidence_budget,
    )
    endpoint_coverage_sets, num_endpoint_units = _endpoint_node_coverage_sets(
        tuple(
            prefix_support(index, quota)[0]
            for index, quota in enumerate(standalone_quotas)
        ),
        selector_positive_edges,
    )
    diagnostic_endpoint_coverage_sets = list(endpoint_coverage_sets)
    if result.final_support is not None:
        selected_prefixes = tuple(
            prefix_support(index, quota)[0]
            for index, quota in zip(
                result.final_support.candidate_indices,
                result.final_support.quotas,
                strict=True,
            )
        )
        selected_endpoint_sets, selected_endpoint_units = _endpoint_node_coverage_sets(
            selected_prefixes,
            selector_positive_edges,
        )
        if selected_endpoint_units != num_endpoint_units:
            raise RuntimeError("Link endpoint coverage universe changed across quotas.")
        for index, covered in zip(
            result.final_support.candidate_indices,
            selected_endpoint_sets,
            strict=True,
        ):
            diagnostic_endpoint_coverage_sets[index] = covered
    coverage_diagnostics = (
        build_coverage_diagnostic(
            definition="link_positive_pairs_postquota",
            candidate_identities=candidate_identities,
            coverage_sets=(
                tuple(
                    (
                        dict(
                            zip(
                                result.final_support.candidate_indices,
                                result.final_support.coverage_sets,
                                strict=True,
                            )
                        ).get(index, covered)
                        if result.final_support is not None
                        else covered
                    )
                    for index, covered in enumerate(coverage_sets)
                )
            ),
            selected_indices=result.selected_indices,
            num_units=int(selector_positive_edges.size(1)),
        ),
        build_coverage_diagnostic(
            definition="link_endpoint_nodes_postquota_diagnostic_only",
            candidate_identities=candidate_identities,
            coverage_sets=diagnostic_endpoint_coverage_sets,
            selected_indices=result.selected_indices,
            num_units=num_endpoint_units,
        ),
    )
    selected_samples = tuple(viable_samples[index] for index in result.selected_indices)
    if not selected_samples:
        report = enrich_report(
            result.report,
            selector_unit_hash=tensor_content_hash(selector_positive_edges),
            selector_support_hash="",
            candidate_statistics=candidate_statistics,
            selected_statistics=(),
            coverage_diagnostics=coverage_diagnostics,
        )
        return tuple(), None, edge_index.new_empty((2, 0)), report

    final_samples = sample_schema_paths(
        candidates=tuple(sample.schema for sample in selected_samples),
        edge_index=edge_index,
        edge_type=edge_type,
        anchor_nodes_by_type=_anchors_from_slices(node_slices),
        target_nodes=frozenset(target_nodes),
        attempts_per_anchor=config.discovery.selector_num_attempts,
        max_instances_per_schema=config.discovery.max_instances_per_schema,
        sample_seed=materialization_seed,
    )
    if any(sample.unique_paths <= 0 for sample in final_samples):
        raise RuntimeError(
            "A link schema selected on the inner-fit graph has no support on the "
            "full outer-training graph."
        )
    selected_stats = schema_stats_from_samples(
        final_samples,
        propagated_labels=propagated_roles,
        num_target_nodes=max(dst_end - dst_begin, target_end - target_begin, 1),
    )
    face_groups, metadata, closure = materialize_sampled_schemas(
        edge_index=edge_index,
        edge_type=edge_type,
        samples=final_samples,
        stats=selected_stats,
        max_schema_length=config.discovery.max_schema_length,
        max_total_instances=config.discovery.max_total_instances,
        max_total_incidences=config.discovery.max_total_incidences,
    )
    if metadata is not None:
        units_by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
        for unit, pair in enumerate(
            selector_positive_edges.detach().cpu().t().tolist()
        ):
            units_by_pair[(int(pair[0]), int(pair[1]))].append(unit)
        materialized_coverage: list[set[int]] = [
            set() for _ in range(metadata.num_schemas)
        ]
        for group_index, face_group in enumerate(face_groups):
            schema_index = int(metadata.schema_ids[group_index].item())
            for nodes in face_group.face_node_index.detach().cpu().tolist():
                materialized_coverage[schema_index].update(
                    units_by_pair.get((int(nodes[0]), int(nodes[-1])), ())
                )
        metadata = make_discovery_metadata(
            num_schemas=metadata.num_schemas,
            face_groups=face_groups,
            schema_ids=metadata.schema_ids,
            coverage_sets=materialized_coverage,
            num_coverage_units=int(selector_positive_edges.size(1)),
        )
    report = enrich_report(
        result.report,
        selector_unit_hash=tensor_content_hash(selector_positive_edges),
        selector_support_hash=(
            result.final_support.support_hash
            if result.final_support is not None
            else ""
        ),
        candidate_statistics=candidate_statistics,
        selected_statistics=(
            selected_schema_statistics(
                face_groups=face_groups,
                discovery=metadata,
                selected_identities=result.report.selected_identities,
            )
            if metadata is not None
            else ()
        ),
        coverage_diagnostics=coverage_diagnostics,
    )
    return face_groups, metadata, closure, report
