from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import torch

from src.data.schema_sampling import (
    SchemaPathSample,
    canonical_schema_identity,
)
from src.data.stage1_selector import (
    CandidateStatistics,
    CoverageDiagnostic,
    SelectedSchemaStatistics,
    SelectorReport,
)


def stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_content_hash(tensor: torch.Tensor) -> str:
    return stable_hash(tensor.detach().cpu().tolist())


def pairwise_jaccard(
    coverage_sets: Sequence[frozenset[int] | set[int]],
) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for left in coverage_sets:
        row: list[float] = []
        for right in coverage_sets:
            union = set(left) | set(right)
            row.append(
                1.0 if not union else len(set(left) & set(right)) / len(union)
            )
        rows.append(tuple(row))
    return tuple(rows)


def build_coverage_diagnostic(
    *,
    definition: str,
    candidate_identities: Sequence[str],
    coverage_sets: Sequence[frozenset[int] | set[int]],
    selected_indices: Sequence[int],
    num_units: int,
) -> CoverageDiagnostic:
    selected_sets = [coverage_sets[index] for index in selected_indices]
    selected_union: set[int] = set()
    for covered in selected_sets:
        selected_union.update(covered)
    return CoverageDiagnostic(
        definition=definition,
        num_units=num_units,
        candidate_identities=tuple(candidate_identities),
        candidate_coverage_fraction=tuple(
            len(covered) / max(num_units, 1) for covered in coverage_sets
        ),
        pairwise_jaccard=pairwise_jaccard(coverage_sets),
        selected_identities=tuple(
            candidate_identities[index] for index in selected_indices
        ),
        selected_pairwise_jaccard=pairwise_jaccard(selected_sets),
        selected_union_coverage_fraction=len(selected_union) / max(num_units, 1),
    )


def candidate_statistics_from_samples(
    *,
    all_samples: Sequence[SchemaPathSample],
    viable_samples: Sequence[SchemaPathSample],
    feature_blocks: Sequence[torch.Tensor],
    coverage_sets: Sequence[frozenset[int] | set[int]],
    coverage_denominator: int,
    selected_indices: Sequence[int],
    selection_steps: Sequence[object],
    incidence_budget: float | None,
) -> tuple[CandidateStatistics, ...]:
    viable_index_by_identity = {
        canonical_schema_identity(sample.schema): index
        for index, sample in enumerate(viable_samples)
    }
    selected_set = set(selected_indices)
    step_by_index = {
        int(step.candidate_index): int(step.step) for step in selection_steps
    }
    rows: list[CandidateStatistics] = []
    for candidate_index, sample in enumerate(all_samples):
        identity = canonical_schema_identity(sample.schema)
        viable_index = viable_index_by_identity.get(identity)
        if sample.unique_paths == 0:
            rejection_reason = "no_sampled_paths"
        elif (
            incidence_budget is not None
            and sample.schema.length + 1 > incidence_budget
        ):
            rejection_reason = "incidence_budget"
        else:
            rejection_reason = ""
        viable = viable_index is not None
        covered = coverage_sets[viable_index] if viable else frozenset()
        rows.append(
            CandidateStatistics(
                candidate_index=candidate_index,
                identity=identity,
                path_length=sample.schema.length,
                viable=viable,
                rejection_reason=rejection_reason,
                sampling_attempts=sample.attempts,
                sampling_successes=sample.successes,
                sampling_unique_paths=sample.unique_paths,
                pre_quota_instances=sample.unique_paths,
                pre_quota_incidences=sample.unique_paths * (sample.schema.length + 1),
                probe_feature_dim=(
                    int(feature_blocks[viable_index].size(1)) if viable else 0
                ),
                coverage_units=len(covered),
                coverage_fraction=len(covered) / max(coverage_denominator, 1),
                selected=viable_index in selected_set if viable else False,
                selection_step=(
                    step_by_index.get(viable_index) if viable else None
                ),
            )
        )
    return tuple(rows)


def selected_schema_statistics(
    *,
    face_groups: Sequence[object],
    discovery: object,
    selected_identities: Sequence[str],
) -> tuple[SelectedSchemaStatistics, ...]:
    num_schemas = int(discovery.num_schemas)
    face_counts = [0 for _ in range(num_schemas)]
    path_incidences = [0 for _ in range(num_schemas)]
    closure_incidences = [0 for _ in range(num_schemas)]
    face_tokens: list[list[object]] = [[] for _ in range(num_schemas)]
    for group_index, face_group in enumerate(face_groups):
        schema_index = int(discovery.schema_ids[group_index].item())
        face_counts[schema_index] += int(face_group.face_node_index.size(0))
        path_incidences[schema_index] += int(face_group.face_edge_index.numel())
        closure_incidences[schema_index] += int(
            face_group.closure_edge_index.size(0)
        )
        for nodes, edges in zip(
            face_group.face_node_index.detach().cpu().tolist(),
            face_group.face_edge_index.detach().cpu().tolist(),
            strict=True,
        ):
            face_tokens[schema_index].append(
                {
                    "template": face_group.template_name,
                    "nodes": nodes,
                    "edges": edges,
                }
            )

    coverage_counts = [0 for _ in range(num_schemas)]
    for schema_index in discovery.coverage_schema_index.detach().cpu().tolist():
        coverage_counts[int(schema_index)] += 1
    rows = []
    for schema_index in range(num_schemas):
        identity = (
            selected_identities[schema_index]
            if schema_index < len(selected_identities)
            else f"schema_{schema_index}"
        )
        rows.append(
            SelectedSchemaStatistics(
                schema_index=schema_index,
                identity=identity,
                materialized_faces=face_counts[schema_index],
                path_incidences=path_incidences[schema_index],
                closure_incidences=closure_incidences[schema_index],
                total_incidences=(
                    path_incidences[schema_index]
                    + closure_incidences[schema_index]
                ),
                coverage_units=coverage_counts[schema_index],
                coverage_fraction=coverage_counts[schema_index]
                / max(int(discovery.num_coverage_units), 1),
                face_set_hash=stable_hash(sorted(
                    face_tokens[schema_index],
                    key=lambda value: json.dumps(value, sort_keys=True),
                )),
            )
        )
    return tuple(rows)


def materialized_face_hash(
    statistics: Sequence[SelectedSchemaStatistics],
) -> str:
    return stable_hash(
        [
            {
                "schema_index": item.schema_index,
                "identity": item.identity,
                "face_set_hash": item.face_set_hash,
                "materialized_faces": item.materialized_faces,
                "total_incidences": item.total_incidences,
            }
            for item in statistics
        ]
    )


def enrich_report(
    report: SelectorReport,
    *,
    selector_unit_hash: str,
    candidate_statistics: Sequence[CandidateStatistics],
    selected_statistics: Sequence[SelectedSchemaStatistics],
    coverage_diagnostics: Sequence[CoverageDiagnostic],
    selector_support_hash: str = "",
) -> SelectorReport:
    from dataclasses import replace

    return replace(
        report,
        selector_unit_hash=selector_unit_hash,
        selector_support_hash=selector_support_hash,
        materialized_face_hash=materialized_face_hash(selected_statistics),
        candidate_statistics=tuple(candidate_statistics),
        selected_schema_statistics=tuple(selected_statistics),
        coverage_diagnostics=tuple(coverage_diagnostics),
    )
