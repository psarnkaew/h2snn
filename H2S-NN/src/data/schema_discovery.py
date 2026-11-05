from __future__ import annotations

import heapq
import random
from collections import defaultdict
from typing import Mapping, Sequence

import torch

from src.data.face_geometry import (
    validate_face_orientation,
    validate_relation_sequence_constant,
)
from src.data.schema_structures import (
    DiscoveryMetadata,
    SchemaCandidate,
    SchemaStats,
    make_discovery_metadata,
)
from src.data.schema_sampling import SchemaPathSample, canonical_schema_identity
from src.data.structures import FaceGroup

_MAX_STATS_SAMPLE = 5000  # max paths sampled per schema during stat estimation


# ─── Stage A: Schema Graph ────────────────────────────────────────────────────


def build_schema_graph(
    relation_tuples: tuple[tuple[str, str, str], ...],
) -> dict[str, list[tuple[int, str, str]]]:
    """Build type-graph adjacency: src_type -> [(relation_id, relation_name, dst_type)]."""
    adjacency: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for relation_id, (src_type, relation_name, dst_type) in enumerate(relation_tuples):
        adjacency[src_type].append((relation_id, relation_name, dst_type))
    return dict(adjacency)


# ─── Stage B: Candidate Schema Generation ────────────────────────────────────


def generate_schema_candidates(
    relation_tuples: tuple[tuple[str, str, str], ...],
    target_node_type: str,
    min_length: int,
    max_length: int,
    require_target_start: bool,
    require_target_end: bool,
    allow_repeated_node_types: bool,
    max_identities: int,
) -> tuple[SchemaCandidate, ...]:
    """Stream feasible typed schemas and fail as soon as the guard is exceeded."""
    schema_adjacency = build_schema_graph(relation_tuples)
    identities: set[
        tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]
    ] = set()

    start_types = (
        [target_node_type] if require_target_start else list(schema_adjacency.keys())
    )

    def dfs(
        current_type: str,
        node_types: list[str],
        rel_ids: list[int],
        rel_names: list[str],
    ) -> None:
        length = len(rel_ids)
        if length >= min_length:
            if not require_target_end or current_type == target_node_type:
                identities.add((tuple(node_types), tuple(rel_ids), tuple(rel_names)))
                if len(identities) > max_identities:
                    raise ValueError(
                        "AMPSS feasible schema count exceeds max_schema_identities "
                        f"guard: more than {max_identities}."
                    )
        if length == max_length:
            return
        for next_rel_id, next_rel_name, next_type in sorted(
            schema_adjacency.get(current_type, []),
            key=lambda item: (item[0], item[1], item[2]),
        ):
            if not allow_repeated_node_types and next_type in node_types:
                continue
            dfs(next_type, node_types + [next_type], rel_ids + [next_rel_id], rel_names + [next_rel_name])

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


# ─── Stage C: Schema Stats Estimation ────────────────────────────────────────


def _build_adjacency_by_relation(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_relations: int,
) -> dict[int, dict[int, list[tuple[int, int]]]]:
    """Build relation-keyed adjacency: rel_id -> src_node -> [(dst_node, edge_id)]."""
    adj: dict[int, dict[int, list[tuple[int, int]]]] = {
        r: defaultdict(list) for r in range(num_relations)
    }
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    rel_list = edge_type.tolist()
    for edge_id, (src, dst, rel) in enumerate(zip(src_list, dst_list, rel_list)):
        adj[int(rel)][int(src)].append((int(dst), edge_id))
    return adj


def compute_endpoint_label_discrepancy(
    instance_sample: list[tuple[int, ...]],
    propagated_labels: torch.Tensor,
) -> float:
    """Mean ||y_hat_{v0} - y_hat_{vl}||^2 over sampled path instances.

    Topology-native schema score: high value means the schema systematically
    connects heterophilic endpoints, creating large delta1 discrepancy signals.
    Uses only training-set-derived propagated labels — no test label leakage.
    """
    if not instance_sample:
        return 0.0
    start_nodes = torch.tensor([p[0] for p in instance_sample], dtype=torch.long)
    end_nodes = torch.tensor([p[-1] for p in instance_sample], dtype=torch.long)
    diff = propagated_labels[start_nodes] - propagated_labels[end_nodes]
    return float(diff.pow(2).sum(dim=-1).mean().item())


def estimate_schema_stats(
    candidates: tuple[SchemaCandidate, ...],
    node_slices: Mapping[str, tuple[int, int]],
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    propagated_labels: torch.Tensor,
    target_node_type: str,
    max_instances_per_anchor: int,
    split_seed: int,
) -> tuple[SchemaStats, ...]:
    """Sample path instances per candidate and compute structural + topology stats."""
    num_relations = int(edge_type.max().item()) + 1 if edge_type.numel() > 0 else 0
    adj = _build_adjacency_by_relation(edge_index, edge_type, num_relations)
    target_begin, target_end = node_slices[target_node_type]
    num_target = max(target_end - target_begin, 1)

    stats_list: list[SchemaStats] = []

    for candidate in candidates:
        sampled_paths: list[tuple[int, ...]] = []
        start_nodes_seen: set[int] = set()
        end_nodes_seen: set[int] = set()
        branching_totals: list[float] = []

        start_type = candidate.node_types[0]
        start_begin, start_end = node_slices[start_type]

        for anchor in range(start_begin, start_end):
            if len(sampled_paths) >= _MAX_STATS_SAMPLE:
                break

            rng = random.Random(split_seed + anchor * 7919)
            anchor_paths: list[tuple[int, ...]] = []

            def dfs_stats(current: int, depth: int, path: list[int]) -> None:
                if len(anchor_paths) >= max_instances_per_anchor:
                    return
                if depth == len(candidate.relation_ids):
                    anchor_paths.append(tuple(path))
                    return
                rel_id = candidate.relation_ids[depth]
                neighbors = list(adj[rel_id].get(current, []))
                if neighbors:
                    branching_totals.append(float(len(neighbors)))
                rng.shuffle(neighbors)
                for next_node, _ in neighbors:
                    if next_node in path:
                        # Match the face builder's simple-path rule so schema scores are
                        # computed over the same population of instances that become faces.
                        continue
                    path.append(next_node)
                    dfs_stats(next_node, depth + 1, path)
                    path.pop()
                    if len(anchor_paths) >= max_instances_per_anchor:
                        break

            dfs_stats(anchor, 0, [anchor])
            if anchor_paths:
                start_nodes_seen.add(anchor)
                end_nodes_seen.update(p[-1] for p in anchor_paths)
            sampled_paths.extend(anchor_paths)

        num_instances = len(sampled_paths)
        start_coverage = len(start_nodes_seen) / num_target
        end_coverage = len(end_nodes_seen) / num_target
        relation_diversity = len(set(candidate.relation_ids)) / max(candidate.length, 1)
        type_diversity = len(set(candidate.node_types)) / max(len(candidate.node_types), 1)
        avg_branching = sum(branching_totals) / max(len(branching_totals), 1)
        estimated_cost = float(num_instances * candidate.length)
        disc = compute_endpoint_label_discrepancy(sampled_paths, propagated_labels)

        stats_list.append(
            SchemaStats(
                schema_id=candidate.schema_id,
                num_instances=num_instances,
                start_coverage=start_coverage,
                end_coverage=end_coverage,
                relation_diversity=relation_diversity,
                type_diversity=type_diversity,
                avg_branching=avg_branching,
                estimated_cost=estimated_cost,
                endpoint_label_disc=disc,
                start_node_sample=frozenset(start_nodes_seen),
                end_node_sample=frozenset(end_nodes_seen),
            )
        )

    return tuple(stats_list)


# ─── Stage D: Topology-Aware Greedy Schema Pool Selection ────────────────────


def jaccard_endpoint_overlap(stats_a: SchemaStats, stats_b: SchemaStats) -> float:
    """Jaccard similarity of end-node sample sets — proxy for schema redundancy."""
    if not stats_a.end_node_sample or not stats_b.end_node_sample:
        return 0.0
    inter = len(stats_a.end_node_sample & stats_b.end_node_sample)
    union = len(stats_a.end_node_sample | stats_b.end_node_sample)
    return inter / union if union > 0 else 0.0


def topology_aware_score(
    stats: SchemaStats,
    selected_stats: list[SchemaStats],
    alpha_disc: float = 1.0,
    alpha_cov: float = 0.5,
    alpha_div: float = 0.3,
) -> float:
    """Deprecated DCV heuristic baseline.

    score = disc^alpha * cov^beta * div^gamma / (1 + max_redundancy).

    disc: endpoint label discrepancy — topology-native task signal.
    cov:  end_coverage — ensures the schema reaches diverse target nodes.
    div:  type_diversity — rewards cross-type heterogeneous paths.
    redundancy: max Jaccard overlap with already-selected schemas.

    Kept for ablation/backward compatibility. Main AMPSS Stage-I paths should
    prefer task-aligned probe objectives from `stage1_selector`.
    """
    disc = max(stats.endpoint_label_disc, 1e-6)
    cov = max(stats.end_coverage, 1e-6)
    div = max(stats.type_diversity, 1e-6)
    redundancy = max(
        (jaccard_endpoint_overlap(stats, s) for s in selected_stats),
        default=0.0,
    )
    base = (disc ** alpha_disc) * (cov ** alpha_cov) * (div ** alpha_div)
    return base / (1.0 + redundancy)


def select_schema_pool(
    candidates: tuple[SchemaCandidate, ...],
    stats: tuple[SchemaStats, ...],
    selected_schema_k: int,
) -> tuple[tuple[SchemaCandidate, ...], tuple[SchemaStats, ...], torch.Tensor]:
    """Deprecated greedy DCV heuristic selection of top-K diverse schemas.

    Returns (selected_schemas, selected_stats, topology_scores).
    topology_scores[i] is schema i's DCV score against all other selected
    schemas. Kept for the `dcv_heuristic` baseline path.
    """
    if not candidates:
        return (), (), torch.zeros(0, dtype=torch.float32)

    selected_cands: list[SchemaCandidate] = []
    selected_stats_list: list[SchemaStats] = []
    remaining = list(zip(candidates, stats))

    for _ in range(min(selected_schema_k, len(candidates))):
        if not remaining:
            break
        scores = [
            topology_aware_score(s, selected_stats_list)
            for _, s in remaining
        ]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_cand, best_st = remaining.pop(best_idx)
        selected_cands.append(best_cand)
        selected_stats_list.append(best_st)

    # Recompute scores of each schema against the rest of the pool (leave-one-out).
    final_scores: list[float] = []
    for i, st in enumerate(selected_stats_list):
        others = [s for j, s in enumerate(selected_stats_list) if j != i]
        final_scores.append(topology_aware_score(st, others))

    topology_scores = torch.tensor(final_scores, dtype=torch.float32)
    return tuple(selected_cands), tuple(selected_stats_list), topology_scores


# ─── Stage E: Face Group Materialization ─────────────────────────────────────


def schema_stats_from_samples(
    samples: tuple[SchemaPathSample, ...],
    propagated_labels: torch.Tensor,
    num_target_nodes: int,
) -> tuple[SchemaStats, ...]:
    stats: list[SchemaStats] = []
    for sample in samples:
        schema = sample.schema
        paths = [path.nodes for path in sample.materialized_paths]
        start_nodes = frozenset(sample.forward_by_anchor)
        end_nodes = frozenset(sample.backward_by_anchor)
        stats.append(
            SchemaStats(
                schema_id=schema.schema_id,
                num_instances=sample.unique_paths,
                start_coverage=len(start_nodes) / max(num_target_nodes, 1),
                end_coverage=len(end_nodes) / max(num_target_nodes, 1),
                relation_diversity=len(set(schema.relation_ids)) / max(schema.length, 1),
                type_diversity=len(set(schema.node_types)) / max(len(schema.node_types), 1),
                avg_branching=0.0,
                estimated_cost=sample.incidence_cost,
                endpoint_label_disc=compute_endpoint_label_discrepancy(paths, propagated_labels),
                start_node_sample=start_nodes,
                end_node_sample=end_nodes,
            )
        )
    return tuple(stats)


def allocate_face_quotas(
    counts: list[int],
    lengths: list[int],
    max_total_instances: int | None,
    max_total_incidences: int | None,
    identities: list[str] | None = None,
) -> list[int]:
    """Allocate deterministic proportional quotas with identity-stable tie breaks."""
    if len(counts) != len(lengths):
        raise ValueError("counts and lengths must align.")
    if any(count < 0 for count in counts):
        raise ValueError("face counts must be non-negative.")
    if any(length < 1 for length in lengths):
        raise ValueError("schema lengths must be positive.")
    if identities is not None and len(identities) != len(counts):
        raise ValueError("identities must align with counts when provided.")
    if max_total_instances is not None and max_total_instances < 0:
        raise ValueError("max_total_instances must be non-negative.")
    if max_total_incidences is not None and max_total_incidences < 0:
        raise ValueError("max_total_incidences must be non-negative.")
    if max_total_instances is None and max_total_incidences is None:
        return counts.copy()

    face_cap = sum(counts) if max_total_instances is None else max_total_instances
    incidence_cap = (
        sum(
            count * (length + 1)
            for count, length in zip(counts, lengths, strict=True)
        )
        if max_total_incidences is None
        else max_total_incidences
    )
    quotas = [0 for _ in counts]
    stable_identities = (
        identities
        if identities is not None
        else [f"{index:012d}" for index in range(len(counts))]
    )
    queue = [
        (1.0 / count, stable_identities[index], index)
        for index, count in enumerate(counts)
        if count > 0
    ]
    heapq.heapify(queue)
    total_faces = 0
    total_incidences = 0
    while queue and total_faces < face_cap:
        _, _, index = heapq.heappop(queue)
        incidence = lengths[index] + 1
        if total_incidences + incidence > incidence_cap:
            continue
        quotas[index] += 1
        total_faces += 1
        total_incidences += incidence
        if quotas[index] < counts[index]:
            heapq.heappush(
                queue,
                (
                    (quotas[index] + 1) / counts[index],
                    stable_identities[index],
                    index,
                ),
            )
    return quotas


def materialize_sampled_schemas(
    *,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    samples: tuple[SchemaPathSample, ...],
    stats: tuple[SchemaStats, ...],
    max_schema_length: int,
    max_total_instances: int | None,
    max_total_incidences: int | None = None,
    confirmed_quotas: Sequence[int] | None = None,
) -> tuple[tuple[FaceGroup, ...], DiscoveryMetadata | None, torch.Tensor]:
    if not samples:
        return tuple(), None, edge_index.new_empty((2, 0))

    if confirmed_quotas is None:
        quotas = allocate_face_quotas(
            [sample.unique_paths for sample in samples],
            [sample.schema.length for sample in samples],
            max_total_instances,
            max_total_incidences,
            [canonical_schema_identity(sample.schema) for sample in samples],
        )
    else:
        quotas = [int(value) for value in confirmed_quotas]
        if len(quotas) != len(samples):
            raise ValueError("Confirmed quotas must align with sampled schemas.")
        if any(
            quota < 0 or quota > sample.unique_paths
            for quota, sample in zip(quotas, samples, strict=True)
        ):
            raise ValueError("Confirmed quota is outside its sampled schema support.")
        total_faces = sum(quotas)
        total_incidences = sum(
            quota * (sample.schema.length + 1)
            for quota, sample in zip(quotas, samples, strict=True)
        )
        if max_total_instances is not None and total_faces > max_total_instances:
            raise ValueError("Confirmed quotas exceed max_total_instances.")
        if (
            max_total_incidences is not None
            and total_incidences > max_total_incidences
        ):
            raise ValueError("Confirmed quotas exceed max_total_incidences.")
    face_groups: list[FaceGroup] = []
    schema_ids: list[int] = []
    closure_parts: list[torch.Tensor] = []

    for selected_index, (sample, quota) in enumerate(zip(samples, quotas, strict=True)):
        selected_paths = sample.materialized_paths[:quota]
        if not selected_paths:
            continue
        schema = sample.schema
        path_nodes = [list(path.nodes) for path in selected_paths]
        path_edges = [list(path.edges) for path in selected_paths]
        face_node_index = torch.tensor(path_nodes, dtype=torch.long)
        face_edge_index = torch.tensor(path_edges, dtype=torch.long)
        relation_index = torch.tensor(
            [list(schema.relation_ids)] * len(path_nodes),
            dtype=torch.long,
        )
        validate_face_orientation(
            face_node_index=face_node_index,
            face_edge_index=face_edge_index,
            edge_index=edge_index,
            template_name=schema.name,
        )
        validate_relation_sequence_constant(
            relation_index=relation_index,
            template_name=schema.name,
        )
        closure_edge_index = torch.stack(
            [face_node_index[:, -1], face_node_index[:, 0]],
            dim=1,
        )
        face_groups.append(
            FaceGroup(
                template_name=schema.name,
                node_type_sequence=schema.node_types,
                face_node_index=face_node_index,
                face_edge_index=face_edge_index,
                relation_index=relation_index,
                face_edge_sign=torch.ones_like(face_edge_index, dtype=torch.float32),
                closure_edge_index=closure_edge_index,
            )
        )
        schema_ids.append(selected_index)
        closure_parts.append(closure_edge_index)

    if not face_groups:
        return tuple(), None, edge_index.new_empty((2, 0))

    schema_id_tensor = torch.tensor(schema_ids, dtype=torch.long)
    metadata = make_discovery_metadata(
        num_schemas=len(samples),
        face_groups=face_groups,
        schema_ids=schema_id_tensor,
    )
    closure = torch.cat(closure_parts, dim=0).t().contiguous()
    return tuple(face_groups), metadata, closure


def enumerate_discovered_faces(
    node_slices: Mapping[str, tuple[int, int]],
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    schemas: tuple[SchemaCandidate, ...],
    selected_stats: tuple[SchemaStats, ...],
    topology_scores: torch.Tensor,
    max_instances_per_anchor: int,
    max_instances_per_schema: int,
    exclude_identity_metapaths: bool,
    split_seed: int,
    max_schema_length: int,
    max_total_instances: int | None = None,
) -> tuple[tuple[FaceGroup, ...], DiscoveryMetadata, torch.Tensor]:
    """Materialize FaceGroups for each selected schema.

    `max_instances_per_schema` bounds each schema independently; `max_total_instances`
    additionally bounds the sum across the whole selected pool (None = unbounded), so
    total materialized-face memory/compute stays predictable even as `selected_schema_k`
    grows. Schemas are processed in order and later schemas are truncated or skipped
    first once the global budget is exhausted.

    Returns:
        face_groups: one FaceGroup per schema that produced ≥1 face.
        discovery_metadata: gates/features needed by the model at train time.
        closure_edge_tensor: [2, total_closure_edges] — closure edges for all groups.
    """
    num_relations = int(edge_type.max().item()) + 1 if edge_type.numel() > 0 else 0

    adj = _build_adjacency_by_relation(edge_index, edge_type, num_relations)
    max_cost = max((s.estimated_cost for s in selected_stats), default=1.0)

    face_groups: list[FaceGroup] = []
    schema_ids_list: list[int] = []       # index into `schemas` for each produced FaceGroup
    closure_edges_all: list[list[int]] = []

    total_instances_so_far = 0

    for schema_idx, (schema, stats) in enumerate(zip(schemas, selected_stats)):
        # Global cap reached: remaining schemas contribute no faces (same code
        # path as a schema whose DFS happens to find zero instances).
        if max_total_instances is not None and total_instances_so_far >= max_total_instances:
            continue

        schema_instance_budget = max_instances_per_schema
        if max_total_instances is not None:
            schema_instance_budget = min(
                schema_instance_budget, max_total_instances - total_instances_so_far
            )

        start_type = schema.node_types[0]
        start_begin, start_end = node_slices[start_type]

        path_nodes: list[list[int]] = []
        path_edges: list[list[int]] = []

        for anchor in range(start_begin, start_end):
            if len(path_nodes) >= schema_instance_budget:
                break
            rng = random.Random(split_seed + 9973 * schema_idx + anchor)
            anchor_nodes: list[list[int]] = []
            anchor_edges_list: list[list[int]] = []

            def dfs_faces(
                current: int, depth: int, n_path: list[int], e_path: list[int]
            ) -> None:
                if len(anchor_nodes) >= max_instances_per_anchor:
                    return
                if len(path_nodes) + len(anchor_nodes) >= schema_instance_budget:
                    return
                if depth == len(schema.relation_ids):
                    # `exclude_identity_metapaths` is subsumed by the revisit guard below,
                    # which rejects n_path[0] == n_path[-1] unconditionally.
                    anchor_nodes.append(n_path.copy())
                    anchor_edges_list.append(e_path.copy())
                    return
                rel_id = schema.relation_ids[depth]
                neighbors = list(adj[rel_id].get(current, []))
                rng.shuffle(neighbors)
                for next_node, edge_id in neighbors:
                    if next_node in n_path:
                        # Simple paths only — see src/data/face_geometry.py.
                        continue
                    n_path.append(next_node)
                    e_path.append(edge_id)
                    dfs_faces(next_node, depth + 1, n_path, e_path)
                    n_path.pop()
                    e_path.pop()
                    if len(anchor_nodes) >= max_instances_per_anchor:
                        break

            dfs_faces(anchor, 0, [anchor], [])
            path_nodes.extend(anchor_nodes)
            path_edges.extend(anchor_edges_list)

        total_instances_so_far += len(path_nodes)

        if not path_nodes:
            continue

        face_node_index = torch.tensor(path_nodes, dtype=torch.long)
        face_edge_index_t = torch.tensor(path_edges, dtype=torch.long)
        relation_index = torch.tensor(
            [list(schema.relation_ids)] * len(path_nodes), dtype=torch.long
        )
        # dfs_faces above only ever walks forward through adj[rel_id] (built from each
        # relation's stored src->dst direction), so every boundary edge is traversed in
        # its natural orientation and the incidence sign is always +1 — asserted here.
        validate_face_orientation(
            face_node_index=face_node_index,
            face_edge_index=face_edge_index_t,
            edge_index=edge_index,
            template_name=schema.name,
        )
        validate_relation_sequence_constant(
            relation_index=relation_index,
            template_name=schema.name,
        )
        face_edge_sign = torch.ones_like(face_edge_index_t, dtype=torch.float32)
        closure_edge_index = torch.stack(
            [face_node_index[:, -1], face_node_index[:, 0]], dim=1
        )
        closure_edges_all.extend(closure_edge_index.tolist())

        face_groups.append(
            FaceGroup(
                template_name=schema.name,
                node_type_sequence=schema.node_types,
                face_node_index=face_node_index,
                face_edge_index=face_edge_index_t,
                relation_index=relation_index,
                face_edge_sign=face_edge_sign,
                closure_edge_index=closure_edge_index,
            )
        )
        schema_ids_list.append(schema_idx)        # index within `schemas` (the selected pool)

    if closure_edges_all:
        closure_tensor = torch.tensor(closure_edges_all, dtype=torch.long).t().contiguous()
    else:
        closure_tensor = torch.empty((2, 0), dtype=torch.long)

    schema_id_tensor = torch.tensor(schema_ids_list, dtype=torch.long)
    discovery_metadata = make_discovery_metadata(
        num_schemas=len(schemas),
        face_groups=face_groups,
        schema_ids=schema_id_tensor,
    )

    return tuple(face_groups), discovery_metadata, closure_tensor
