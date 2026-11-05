from __future__ import annotations

import random
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from src.data.schema_structures import SchemaCandidate
from src.data.stage1_selector import derive_seed


@dataclass(frozen=True, slots=True)
class SampledPath:
    nodes: tuple[int, ...]
    edges: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SchemaPathSample:
    schema: SchemaCandidate
    forward_by_anchor: Mapping[int, tuple[SampledPath, ...]]
    backward_by_anchor: Mapping[int, tuple[SampledPath, ...]]
    materialized_paths: tuple[SampledPath, ...]
    target_participants: frozenset[int]
    attempts: int
    successes: int

    @property
    def unique_paths(self) -> int:
        return len(self.materialized_paths)

    @property
    def incidence_cost(self) -> float:
        # Each materialized face contributes all path-edge incidences and one
        # closure incidence to the current operator.
        return float(self.unique_paths * (self.schema.length + 1))


def canonical_schema_identity(schema: SchemaCandidate) -> str:
    node_part = ",".join(schema.node_types)
    relation_part = ",".join(str(value) for value in schema.relation_ids)
    return f"nodes=[{node_part}]|relations=[{relation_part}]"


def truncate_schema_path_sample(
    sample: SchemaPathSample,
    quota: int,
) -> SchemaPathSample:
    """Return the exact deterministic materialization prefix for one schema."""
    if quota < 0:
        raise ValueError("quota must be non-negative.")
    selected_paths = sample.materialized_paths[:quota]
    selected_set = set(selected_paths)
    forward_by_anchor = {
        anchor: tuple(path for path in paths if path in selected_set)
        for anchor, paths in sample.forward_by_anchor.items()
        if any(path in selected_set for path in paths)
    }
    backward_by_anchor = {
        anchor: tuple(path for path in paths if path in selected_set)
        for anchor, paths in sample.backward_by_anchor.items()
        if any(path in selected_set for path in paths)
    }
    participants = frozenset(
        node
        for path in selected_paths
        for node in path.nodes
        if node in sample.target_participants
    )
    return SchemaPathSample(
        schema=sample.schema,
        forward_by_anchor=forward_by_anchor,
        backward_by_anchor=backward_by_anchor,
        materialized_paths=selected_paths,
        target_participants=participants,
        attempts=sample.attempts,
        successes=sample.successes,
    )


def sampled_support_hash(samples: Sequence[SchemaPathSample]) -> str:
    payload = [
        {
            "identity": canonical_schema_identity(sample.schema),
            "paths": [
                {"nodes": path.nodes, "edges": path.edges}
                for path in sample.materialized_paths
            ],
        }
        for sample in samples
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relation_adjacencies(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
) -> tuple[
    dict[int, dict[int, tuple[tuple[int, int], ...]]],
    dict[int, dict[int, tuple[tuple[int, int], ...]]],
]:
    forward_lists: dict[int, dict[int, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    reverse_lists: dict[int, dict[int, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for edge_id, (src, dst, relation) in enumerate(
        zip(
            edge_index[0].detach().cpu().tolist(),
            edge_index[1].detach().cpu().tolist(),
            edge_type.detach().cpu().tolist(),
            strict=True,
        )
    ):
        forward_lists[int(relation)][int(src)].append((int(dst), edge_id))
        reverse_lists[int(relation)][int(dst)].append((int(src), edge_id))

    def freeze(
        source: dict[int, dict[int, list[tuple[int, int]]]],
    ) -> dict[int, dict[int, tuple[tuple[int, int], ...]]]:
        return {
            relation: {
                node: tuple(sorted(neighbors))
                for node, neighbors in by_node.items()
            }
            for relation, by_node in source.items()
        }

    return freeze(forward_lists), freeze(reverse_lists)


def _sample_forward_attempt(
    schema: SchemaCandidate,
    anchor: int,
    adjacency: Mapping[int, Mapping[int, tuple[tuple[int, int], ...]]],
    rng: random.Random,
) -> SampledPath | None:
    nodes = [anchor]
    edges: list[int] = []
    current = anchor
    for relation in schema.relation_ids:
        eligible = [
            item
            for item in adjacency.get(relation, {}).get(current, ())
            if item[0] not in nodes
        ]
        if not eligible:
            return None
        next_node, edge_id = eligible[rng.randrange(len(eligible))]
        nodes.append(next_node)
        edges.append(edge_id)
        current = next_node
    return SampledPath(tuple(nodes), tuple(edges))


def _sample_backward_attempt(
    schema: SchemaCandidate,
    anchor: int,
    reverse_adjacency: Mapping[int, Mapping[int, tuple[tuple[int, int], ...]]],
    rng: random.Random,
) -> SampledPath | None:
    reverse_nodes = [anchor]
    reverse_edges: list[int] = []
    current = anchor
    for relation in reversed(schema.relation_ids):
        eligible = [
            item
            for item in reverse_adjacency.get(relation, {}).get(current, ())
            if item[0] not in reverse_nodes
        ]
        if not eligible:
            return None
        previous_node, edge_id = eligible[rng.randrange(len(eligible))]
        reverse_nodes.append(previous_node)
        reverse_edges.append(edge_id)
        current = previous_node
    return SampledPath(
        nodes=tuple(reversed(reverse_nodes)),
        edges=tuple(reversed(reverse_edges)),
    )


def _path_priority(seed: int, identity: str, path: SampledPath) -> tuple[int, tuple[int, ...]]:
    priority = derive_seed(seed, "path_priority", identity, path.nodes, path.edges)
    return priority, path.edges


def sample_schema_paths(
    *,
    candidates: Sequence[SchemaCandidate],
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    anchor_nodes_by_type: Mapping[str, Sequence[int]],
    target_nodes: frozenset[int],
    attempts_per_anchor: int,
    max_instances_per_schema: int,
    sample_seed: int,
    max_viable_samples: int | None = None,
    max_total_incidences: int | None = None,
) -> tuple[SchemaPathSample, ...]:
    if attempts_per_anchor <= 0:
        raise ValueError("attempts_per_anchor must be positive.")
    if max_instances_per_schema <= 0:
        raise ValueError("max_instances_per_schema must be positive.")
    forward_adjacency, reverse_adjacency = _relation_adjacencies(edge_index, edge_type)
    output: list[SchemaPathSample] = []
    viable_count = 0

    for schema in candidates:
        identity = canonical_schema_identity(schema)
        forward_raw: dict[int, set[SampledPath]] = defaultdict(set)
        backward_raw: dict[int, set[SampledPath]] = defaultdict(set)
        all_paths: set[SampledPath] = set()
        attempts = 0
        successes = 0

        for direction, anchors in (
            ("forward", anchor_nodes_by_type.get(schema.node_types[0], ())),
            ("backward", anchor_nodes_by_type.get(schema.node_types[-1], ())),
        ):
            for anchor_value in anchors:
                anchor = int(anchor_value)
                for attempt in range(attempts_per_anchor):
                    attempts += 1
                    rng = random.Random(
                        derive_seed(
                            sample_seed,
                            "typed_walk",
                            identity,
                            direction,
                            anchor,
                            attempt,
                        )
                    )
                    if direction == "forward":
                        path = _sample_forward_attempt(schema, anchor, forward_adjacency, rng)
                    else:
                        path = _sample_backward_attempt(schema, anchor, reverse_adjacency, rng)
                    if path is None:
                        continue
                    successes += 1
                    all_paths.add(path)
                    if direction == "forward":
                        forward_raw[anchor].add(path)
                    else:
                        backward_raw[anchor].add(path)

        selected_paths = tuple(
            sorted(
                all_paths,
                key=lambda path: _path_priority(sample_seed, identity, path),
            )[:max_instances_per_schema]
        )
        selected_set = set(selected_paths)
        forward_by_anchor = {
            anchor: tuple(sorted(paths & selected_set, key=lambda path: (path.nodes, path.edges)))
            for anchor, paths in forward_raw.items()
            if paths & selected_set
        }
        backward_by_anchor = {
            anchor: tuple(sorted(paths & selected_set, key=lambda path: (path.nodes, path.edges)))
            for anchor, paths in backward_raw.items()
            if paths & selected_set
        }
        participants = frozenset(
            node
            for path in selected_paths
            for node in path.nodes
            if node in target_nodes
        )
        sampled = SchemaPathSample(
            schema=schema,
            forward_by_anchor=forward_by_anchor,
            backward_by_anchor=backward_by_anchor,
            materialized_paths=selected_paths,
            target_participants=participants,
            attempts=attempts,
            successes=successes,
        )
        output.append(sampled)
        if sampled.unique_paths > 0 and (
            max_total_incidences is None
            or sampled.schema.length + 1 <= max_total_incidences
        ):
            viable_count += 1
            if (
                max_viable_samples is not None
                and viable_count > max_viable_samples
            ):
                raise ValueError(
                    "AMPSS viable schema count exceeds max_schema_candidates "
                    f"guard: more than {max_viable_samples}."
                )
    return tuple(output)
