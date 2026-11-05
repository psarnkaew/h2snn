from __future__ import annotations

from typing import Any, Mapping

import torch

from src.configs.config import TrainConfig
from src.data.preprocessing import label_propagation
from src.data.schema_discovery import (
    allocate_face_quotas,
    generate_schema_candidates,
    materialize_sampled_schemas,
    schema_stats_from_samples,
)
from src.data.schema_sampling import (
    SchemaPathSample,
    canonical_schema_identity,
    sampled_support_hash,
    sample_schema_paths,
    truncate_schema_path_sample,
)
from src.data.selector_reporting import (
    build_coverage_diagnostic,
    candidate_statistics_from_samples,
    enrich_report,
    materialized_face_hash,
    selected_schema_statistics,
    tensor_content_hash,
)
from src.data.schema_structures import DiscoveryMetadata, make_discovery_metadata
from src.data.stage1_selector import (
    SelectorReport,
    TrialSupport,
    derive_seed,
    make_selector_split,
    select_schema_features,
)
from src.data.structures import FaceGroup


def _anchors_from_slices(
    node_slices: Mapping[str, tuple[int, int]],
) -> dict[str, tuple[int, ...]]:
    return {
        node_type: tuple(range(begin, end))
        for node_type, (begin, end) in node_slices.items()
    }


def _node_schema_feature_block(
    sample: SchemaPathSample,
    target_nodes: torch.Tensor,
    propagated_fit_labels: torch.Tensor,
    num_classes: int,
    attempts_per_anchor: int,
) -> torch.Tensor:
    features = torch.zeros(
        (target_nodes.numel(), 2 * (num_classes + 2)),
        dtype=torch.float32,
    )
    for row, node_value in enumerate(target_nodes.tolist()):
        node = int(node_value)
        offset = 0
        for paths, counterpart_position in (
            (sample.forward_by_anchor.get(node, ()), -1),
            (sample.backward_by_anchor.get(node, ()), 0),
        ):
            count = len(paths)
            features[row, offset] = count / max(attempts_per_anchor, 1)
            if count > 0:
                counterparts = [path.nodes[counterpart_position] for path in paths]
                features[row, offset + 1] = len(set(counterparts)) / count
                counterpart_index = torch.tensor(counterparts, dtype=torch.long)
                evidence = propagated_fit_labels[counterpart_index].float().mean(dim=0)
                features[row, offset + 2 : offset + 2 + num_classes] = evidence
            offset += num_classes + 2
    return features


def build_node_amp_ss_faces(
    config: TrainConfig,
    *,
    node_slices: Mapping[str, tuple[int, int]],
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    relation_tuples: tuple[tuple[str, str, str], ...],
    target_node_type: str,
    target_indices: torch.Tensor,
    target_y: torch.Tensor,
    target_train_mask: torch.Tensor,
    is_multilabel: bool,
    propagated_labels: torch.Tensor,
    frozen_manifest: Mapping[str, Any] | None = None,
) -> tuple[
    tuple[FaceGroup, ...],
    DiscoveryMetadata | None,
    torch.Tensor,
    SelectorReport | None,
]:
    split_seed = derive_seed(
        config.discovery.selector_seed,
        "ampss_selector_split",
        config.data.dataset_name,
    )
    sample_seed = derive_seed(
        config.discovery.materialization_seed,
        "ampss_path_sample",
        config.data.dataset_name,
    )
    train_target_nodes = target_indices[target_train_mask]
    train_labels = target_y[target_train_mask]
    task = "multi_label" if is_multilabel else "single_label"
    split = make_selector_split(
        labels=train_labels,
        task=task,
        holdout_ratio=config.discovery.selector_holdout_ratio,
        seed=split_seed,
        min_examples_per_label=config.discovery.selector_min_examples_per_label,
    )

    num_classes = (
        int(target_y.size(1))
        if is_multilabel
        else int(target_y.max().item() + 1)
    )
    propagated_fit_labels = label_propagation(
        edge_index=edge_index,
        train_indices=train_target_nodes[split.fit_indices],
        train_labels=train_labels[split.fit_indices],
        num_nodes=max(node_slice[1] for node_slice in node_slices.values()),
        num_classes=num_classes,
        num_steps=config.data.num_label_prop_steps,
    )

    candidates = generate_schema_candidates(
        relation_tuples=relation_tuples,
        target_node_type=target_node_type,
        min_length=config.discovery.min_schema_length,
        max_length=config.discovery.max_schema_length,
        require_target_start=config.discovery.require_target_start,
        require_target_end=config.discovery.require_target_end,
        allow_repeated_node_types=config.discovery.allow_repeated_node_types,
        max_identities=config.discovery.max_schema_identities,
    )
    if frozen_manifest is not None:
        if frozen_manifest.get("protocol_version") != config.discovery.protocol_version:
            raise ValueError("Frozen AMPSS manifest protocol version is incompatible.")
        selected_identities = tuple(
            str(value) for value in frozen_manifest.get("selected_identities", ())
        )
        candidate_by_identity = {
            canonical_schema_identity(candidate): candidate
            for candidate in candidates
        }
        missing = [
            identity
            for identity in selected_identities
            if identity not in candidate_by_identity
        ]
        if missing:
            raise ValueError(
                "Frozen AMPSS manifest contains schemas absent from the current "
                f"schema graph: {missing}."
            )
        frozen_candidates = tuple(
            candidate_by_identity[identity] for identity in selected_identities
        )
        frozen_samples = sample_schema_paths(
            candidates=frozen_candidates,
            edge_index=edge_index,
            edge_type=edge_type,
            anchor_nodes_by_type=_anchors_from_slices(node_slices),
            target_nodes=frozenset(int(value) for value in target_indices.tolist()),
            attempts_per_anchor=config.discovery.selector_num_attempts,
            max_instances_per_schema=config.discovery.max_instances_per_schema,
            sample_seed=sample_seed,
        )
        if any(sample.unique_paths <= 0 for sample in frozen_samples):
            raise ValueError("Frozen AMPSS schema has no current materialized support.")
        frozen_stats = schema_stats_from_samples(
            frozen_samples,
            propagated_labels=propagated_labels,
            num_target_nodes=int(target_indices.numel()),
        )
        face_groups, metadata, closure = materialize_sampled_schemas(
            edge_index=edge_index,
            edge_type=edge_type,
            samples=frozen_samples,
            stats=frozen_stats,
            max_schema_length=config.discovery.max_schema_length,
            max_total_instances=config.discovery.max_total_instances,
            max_total_incidences=config.discovery.max_total_incidences,
        )
        if metadata is None:
            raise ValueError("Frozen AMPSS manifest materialized no face groups.")
        unit_by_node = {
            int(node): unit for unit, node in enumerate(target_indices.tolist())
        }
        realized_coverage: list[set[int]] = [
            set() for _ in range(metadata.num_schemas)
        ]
        for group_index, face_group in enumerate(face_groups):
            schema_index = int(metadata.schema_ids[group_index].item())
            for node in face_group.face_node_index.unique().tolist():
                unit = unit_by_node.get(int(node))
                if unit is not None:
                    realized_coverage[schema_index].add(unit)
        metadata = make_discovery_metadata(
            num_schemas=metadata.num_schemas,
            face_groups=face_groups,
            schema_ids=metadata.schema_ids,
            coverage_sets=realized_coverage,
            num_coverage_units=int(target_indices.numel()),
        )
        frozen_statistics = selected_schema_statistics(
            face_groups=face_groups,
            discovery=metadata,
            selected_identities=selected_identities,
        )
        current_face_hash = materialized_face_hash(frozen_statistics)
        if current_face_hash != str(
            frozen_manifest.get("final_materialization_hash", "")
        ):
            raise ValueError(
                "Frozen AMPSS manifest face hash does not match current materialization."
            )
        return face_groups, metadata, closure, None
    target_node_set = frozenset(int(value) for value in target_indices.tolist())
    samples = sample_schema_paths(
        candidates=candidates,
        edge_index=edge_index,
        edge_type=edge_type,
        anchor_nodes_by_type=_anchors_from_slices(node_slices),
        target_nodes=target_node_set,
        attempts_per_anchor=config.discovery.selector_num_attempts,
        max_instances_per_schema=config.discovery.max_instances_per_schema,
        sample_seed=sample_seed,
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
            "AMPSS viable schema count exceeds max_schema_candidates guard: "
            f"{len(viable_samples)} > {config.discovery.max_schema_candidates}."
        )

    prefix_cache: dict[
        tuple[int, int],
        tuple[SchemaPathSample, torch.Tensor, frozenset[int]],
    ] = {}
    target_begin = int(target_indices[0].item()) if target_indices.numel() > 0 else 0

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
        block = _node_schema_feature_block(
            sample=truncated,
            target_nodes=train_target_nodes,
            propagated_fit_labels=propagated_fit_labels,
            num_classes=num_classes,
            attempts_per_anchor=config.discovery.selector_num_attempts,
        )
        covered = frozenset(
            node - target_begin for node in truncated.target_participants
        )
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
        labels=train_labels,
        split=split,
        task=task,
        max_schemas=config.discovery.selected_schema_k,
        incidence_budget=incidence_budget,
        stop_tolerance=config.discovery.selector_stop_tolerance,
        tie_tolerance=config.discovery.selector_tie_tolerance,
        probe_l2=config.discovery.selector_probe_l2,
        probe_max_iter=config.discovery.selector_probe_max_iter,
        split_seed=split_seed,
        sample_seed=sample_seed,
        candidate_identity_count=len(candidates),
        coverage_denominator=int(target_indices.numel()),
        sampling_attempts=sum(sample.attempts for sample in samples),
        sampling_successes=sum(sample.successes for sample in samples),
        sampling_unique_paths=sum(sample.unique_paths for sample in samples),
        subset_support_fn=subset_support,
    )
    candidate_statistics = candidate_statistics_from_samples(
        all_samples=samples,
        viable_samples=viable_samples,
        feature_blocks=feature_blocks,
        coverage_sets=coverage_sets,
        coverage_denominator=int(target_indices.numel()),
        selected_indices=result.selected_indices,
        selection_steps=result.report.steps,
        incidence_budget=incidence_budget,
    )
    diagnostic_coverage_sets = list(coverage_sets)
    if result.final_support is not None:
        for index, covered in zip(
            result.final_support.candidate_indices,
            result.final_support.coverage_sets,
            strict=True,
        ):
            diagnostic_coverage_sets[index] = covered
    coverage_diagnostic = build_coverage_diagnostic(
        definition="node_target_participation_postquota",
        candidate_identities=tuple(
            canonical_schema_identity(sample.schema) for sample in viable_samples
        ),
        coverage_sets=diagnostic_coverage_sets,
        selected_indices=result.selected_indices,
        num_units=int(target_indices.numel()),
    )
    selected_samples = tuple(viable_samples[index] for index in result.selected_indices)
    if not selected_samples:
        report = enrich_report(
            result.report,
            selector_unit_hash=tensor_content_hash(target_indices),
            selector_support_hash="",
            candidate_statistics=candidate_statistics,
            selected_statistics=(),
            coverage_diagnostics=(coverage_diagnostic,),
        )
        return tuple(), None, edge_index.new_empty((2, 0)), report

    selected_stats = schema_stats_from_samples(
        selected_samples,
        propagated_labels=propagated_labels,
        num_target_nodes=int(target_indices.numel()),
    )
    if result.final_support is None:
        raise RuntimeError("Selected AMPSS schemas are missing their exact trial support.")
    quota_by_candidate = dict(
        zip(
            result.final_support.candidate_indices,
            result.final_support.quotas,
            strict=True,
        )
    )
    confirmed_quotas = tuple(
        quota_by_candidate[index] for index in result.selected_indices
    )
    face_groups, metadata, closure = materialize_sampled_schemas(
        edge_index=edge_index,
        edge_type=edge_type,
        samples=selected_samples,
        stats=selected_stats,
        max_schema_length=config.discovery.max_schema_length,
        max_total_instances=config.discovery.max_total_instances,
        max_total_incidences=config.discovery.max_total_incidences,
        confirmed_quotas=confirmed_quotas,
    )
    realized_incidence = sum(
        int(group.face_edge_index.numel() + group.closure_edge_index.size(0))
        for group in face_groups
    )
    if realized_incidence != int(result.final_support.realized_incidence):
        raise RuntimeError(
            "Node AMPSS selector support and final materialization incidence disagree."
        )
    if metadata is not None:
        unit_by_node = {
            int(node): unit for unit, node in enumerate(target_indices.tolist())
        }
        coverage_sets: list[set[int]] = [
            set() for _ in range(metadata.num_schemas)
        ]
        for group_index, face_group in enumerate(face_groups):
            schema_index = int(metadata.schema_ids[group_index].item())
            for node in face_group.face_node_index.unique().tolist():
                unit = unit_by_node.get(int(node))
                if unit is not None:
                    coverage_sets[schema_index].add(unit)
        metadata = make_discovery_metadata(
            num_schemas=metadata.num_schemas,
            face_groups=face_groups,
            schema_ids=metadata.schema_ids,
            coverage_sets=coverage_sets,
            num_coverage_units=int(target_indices.numel()),
        )
    report = enrich_report(
        result.report,
        selector_unit_hash=tensor_content_hash(target_indices),
        selector_support_hash=result.final_support.support_hash,
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
        coverage_diagnostics=(coverage_diagnostic,),
    )
    return face_groups, metadata, closure, report
