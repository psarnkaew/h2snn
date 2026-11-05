from __future__ import annotations

import random
from collections import defaultdict
from typing import DefaultDict

import torch

from src.data.face_geometry import (
    validate_face_orientation,
    validate_relation_sequence_constant,
)
from src.data.structures import FaceGroup
from src.data.stage1_selector import derive_seed


def _empty_faces(edge_index: torch.Tensor) -> tuple[tuple[FaceGroup, ...], torch.Tensor]:
    return tuple(), edge_index.new_empty((2, 0))


def _graph_starts(batch_ids: list[int]) -> dict[int, int]:
    starts: dict[int, int] = {}
    for node_id, graph_slot in enumerate(batch_ids):
        starts.setdefault(graph_slot, node_id)
    return starts


def _graph_seed_id(graph_slot: int, graph_ids: list[int]) -> int:
    if 0 <= graph_slot < len(graph_ids):
        return graph_ids[graph_slot]
    return graph_slot


def _template_name(key: tuple[int, ...]) -> str:
    """Stable typed-path name: N0_R1_N2_R3_N4 for node/edge alternating ids."""
    parts: list[str] = []
    for idx, value in enumerate(key):
        prefix = "N" if idx % 2 == 0 else "R"
        parts.append(f"{prefix}{value}")
    return "_".join(parts)


def _node_type_sequence_from_key(key: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(_node_type_name(type_id) for type_id in key[::2])


def _node_type_name(type_id: int) -> str:
    return f"node_type_{type_id}"


def _build_random_walk_face_groups(
    *,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    node_type: torch.Tensor,
    batch: torch.Tensor | None,
    graph_id: torch.Tensor | None,
    attempts_per_anchor: int,
    seed: int,
    min_path_length: int,
    max_path_length: int,
    selected_template_names: tuple[str, ...] | None,
    max_paths_per_schema: int | None,
    max_total_paths: int | None,
    max_total_incidences: int | None,
    allow_repeated_node_types: bool,
    max_template_identities: int | None,
) -> tuple[tuple[FaceGroup, ...], torch.Tensor]:
    device = edge_index.device
    num_nodes = int(node_type.numel())
    edge_index_cpu = edge_index.detach().cpu()
    edge_type_cpu = [int(value) for value in edge_type.detach().cpu().view(-1).tolist()]
    node_type_cpu = [int(value) for value in node_type.detach().cpu().view(-1).tolist()]
    batch_cpu = (
        [0 for _ in range(num_nodes)]
        if batch is None
        else [int(value) for value in batch.detach().cpu().view(-1).tolist()]
    )
    if graph_id is None:
        max_graph_slot = max(batch_cpu) if batch_cpu else 0
        graph_ids_cpu = list(range(max_graph_slot + 1))
    else:
        graph_ids_cpu = [int(value) for value in graph_id.detach().cpu().view(-1).tolist()]
    starts = _graph_starts(batch_cpu)
    selected_names = set(selected_template_names) if selected_template_names is not None else None

    adjacency: DefaultDict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge_id in range(edge_index.size(1)):
        src = int(edge_index_cpu[0, edge_id].item())
        dst = int(edge_index_cpu[1, edge_id].item())
        if batch_cpu[src] == batch_cpu[dst]:
            adjacency[src].append((dst, edge_id))
    for outgoing in adjacency.values():
        outgoing.sort()

    grouped: DefaultDict[
        tuple[int, ...],
        set[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    ] = defaultdict(set)
    for start_node in sorted(adjacency):
        graph_slot = batch_cpu[start_node]
        graph_seed = _graph_seed_id(graph_slot, graph_ids_cpu)
        local_start = start_node - starts[graph_slot]
        for attempt in range(attempts_per_anchor):
            rng = random.Random(
                derive_seed(seed, "graph_typed_walk", graph_seed, local_start, attempt)
            )
            current = start_node
            nodes = [start_node]
            edges: list[int] = []
            relations: list[int] = []
            for _ in range(max_path_length):
                eligible = [
                    item
                    for item in adjacency.get(current, [])
                    if batch_cpu[item[0]] == graph_slot
                    and item[0] not in nodes
                    and (
                        allow_repeated_node_types
                        or node_type_cpu[item[0]]
                        not in {node_type_cpu[node] for node in nodes}
                    )
                ]
                if not eligible:
                    break
                next_node, edge_id = eligible[rng.randrange(len(eligible))]
                nodes.append(next_node)
                edges.append(edge_id)
                relations.append(edge_type_cpu[edge_id])
                current = next_node
                if len(edges) < min_path_length:
                    continue
                key_parts = [node_type_cpu[nodes[0]]]
                for relation, path_node in zip(relations, nodes[1:], strict=True):
                    key_parts.extend([relation, node_type_cpu[path_node]])
                key = tuple(key_parts)
                if selected_names is None or _template_name(key) in selected_names:
                    if (
                        selected_names is None
                        and key not in grouped
                        and max_template_identities is not None
                        and len(grouped) >= max_template_identities
                    ):
                        raise ValueError(
                            "AMPSS graph schema enumeration reached guard + 1."
                        )
                    grouped[key].add((tuple(nodes), tuple(edges), tuple(relations)))

    face_groups: list[FaceGroup] = []
    closure_parts: list[torch.Tensor] = []
    total_paths_by_graph: DefaultDict[int, int] = defaultdict(int)
    total_incidences_by_graph: DefaultDict[int, int] = defaultdict(int)
    for key in sorted(grouped):
        paths = sorted(grouped[key])
        selected_paths = []
        schema_paths_by_graph: DefaultDict[int, int] = defaultdict(int)
        for path in paths:
            graph_slot = batch_cpu[path[0][0]]
            if (
                max_paths_per_schema is not None
                and schema_paths_by_graph[graph_slot] >= max_paths_per_schema
            ):
                continue
            if (
                max_total_paths is not None
                and total_paths_by_graph[graph_slot] >= max_total_paths
            ):
                continue
            path_incidences = len(path[1]) + 1
            if (
                max_total_incidences is not None
                and total_incidences_by_graph[graph_slot] + path_incidences
                > max_total_incidences
            ):
                continue
            selected_paths.append(path)
            schema_paths_by_graph[graph_slot] += 1
            total_paths_by_graph[graph_slot] += 1
            total_incidences_by_graph[graph_slot] += path_incidences
        if not selected_paths:
            continue
        paths = selected_paths
        face_node_index = torch.tensor([path[0] for path in paths], dtype=torch.long, device=device)
        face_edge_index = torch.tensor([path[1] for path in paths], dtype=torch.long, device=device)
        relation_index = torch.tensor([path[2] for path in paths], dtype=torch.long, device=device)
        name = _template_name(key)
        validate_face_orientation(
            face_node_index=face_node_index,
            face_edge_index=face_edge_index,
            edge_index=edge_index,
            template_name=name,
        )
        validate_relation_sequence_constant(relation_index=relation_index, template_name=name)
        closure = torch.stack([face_node_index[:, -1], face_node_index[:, 0]], dim=1)
        face_groups.append(
            FaceGroup(
                template_name=name,
                node_type_sequence=_node_type_sequence_from_key(key),
                face_node_index=face_node_index,
                face_edge_index=face_edge_index,
                relation_index=relation_index,
                face_edge_sign=torch.ones_like(face_edge_index, dtype=torch.float32),
                closure_edge_index=closure,
            )
        )
        closure_parts.append(closure)
    if not face_groups:
        return _empty_faces(edge_index)
    return tuple(face_groups), torch.cat(closure_parts, dim=0).t().contiguous()


def build_typed_path_face_groups(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    node_type: torch.Tensor,
    batch: torch.Tensor | None = None,
    graph_id: torch.Tensor | None = None,
    max_paths_per_anchor: int | None = 50,
    exclude_identity_paths: bool = True,
    seed: int = 0,
    min_path_length: int = 2,
    max_path_length: int = 2,
    selected_template_names: tuple[str, ...] | None = None,
    max_paths_per_schema: int | None = None,
    max_total_paths: int | None = None,
    max_total_incidences: int | None = None,
    allow_repeated_node_types: bool = True,
    max_template_identities: int | None = None,
    sampling_mode: str = "dfs",
) -> tuple[tuple[FaceGroup, ...], torch.Tensor]:
    """Materialize typed path faces for a PyG graph-classification batch.

    Each directed typed path becomes one face with the path edges as its boundary
    and a synthetic closure edge from the last node back to the first. Faces are
    grouped by typed path signature so the existing sequence edge-to-face sheaf
    can process them with the same ``FaceGroup`` contract used by node
    classification. Defaults preserve the original two-hop graph-classification
    behavior.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if edge_type.numel() != edge_index.size(1):
        raise ValueError("edge_type must contain one relation id per edge.")
    if max_paths_per_anchor is not None and max_paths_per_anchor <= 0:
        return _empty_faces(edge_index)
    if min_path_length < 1:
        raise ValueError("min_path_length must be >= 1.")
    if max_path_length < min_path_length:
        return _empty_faces(edge_index)
    if max_paths_per_schema is not None and max_paths_per_schema <= 0:
        return _empty_faces(edge_index)
    if max_total_paths is not None and max_total_paths <= 0:
        return _empty_faces(edge_index)
    if max_total_incidences is not None and max_total_incidences <= 0:
        return _empty_faces(edge_index)
    if max_template_identities is not None and max_template_identities <= 0:
        raise ValueError("max_template_identities must be positive when provided.")
    if sampling_mode not in {"dfs", "random_walk"}:
        raise ValueError("sampling_mode must be 'dfs' or 'random_walk'.")

    num_nodes = int(node_type.numel())
    num_edges = int(edge_index.size(1))
    if num_nodes == 0 or num_edges == 0:
        return _empty_faces(edge_index)
    if sampling_mode == "random_walk":
        return _build_random_walk_face_groups(
            edge_index=edge_index,
            edge_type=edge_type,
            node_type=node_type,
            batch=batch,
            graph_id=graph_id,
            attempts_per_anchor=(
                max_paths_per_anchor if max_paths_per_anchor is not None else 50
            ),
            seed=seed,
            min_path_length=min_path_length,
            max_path_length=max_path_length,
            selected_template_names=selected_template_names,
            max_paths_per_schema=max_paths_per_schema,
            max_total_paths=max_total_paths,
            max_total_incidences=max_total_incidences,
            allow_repeated_node_types=allow_repeated_node_types,
            max_template_identities=max_template_identities,
        )
    selected_names = set(selected_template_names) if selected_template_names is not None else None

    device = edge_index.device
    edge_index_cpu = edge_index.detach().cpu()
    edge_type_cpu = [int(value) for value in edge_type.detach().cpu().view(-1).tolist()]
    node_type_cpu = [int(value) for value in node_type.detach().cpu().view(-1).tolist()]

    if batch is None:
        batch_cpu = [0 for _ in range(num_nodes)]
    else:
        if batch.numel() != num_nodes:
            raise ValueError("batch must contain one graph id per node.")
        batch_cpu = [int(value) for value in batch.detach().cpu().view(-1).tolist()]

    if graph_id is None:
        max_graph_slot = max(batch_cpu) if batch_cpu else 0
        graph_ids_cpu = list(range(max_graph_slot + 1))
    else:
        graph_ids_cpu = [int(value) for value in graph_id.detach().cpu().view(-1).tolist()]

    starts = _graph_starts(batch_cpu)
    adjacency: DefaultDict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge_id in range(num_edges):
        src = int(edge_index_cpu[0, edge_id].item())
        dst = int(edge_index_cpu[1, edge_id].item())
        if not (0 <= src < num_nodes and 0 <= dst < num_nodes):
            raise ValueError("edge_index contains a node id outside node_type.")
        if batch_cpu[src] != batch_cpu[dst]:
            continue
        adjacency[src].append((dst, edge_id))
    for outgoing in adjacency.values():
        outgoing.sort(key=lambda item: (item[0], item[1]))

    grouped_nodes: DefaultDict[tuple[int, ...], list[list[int]]] = defaultdict(list)
    grouped_edges: DefaultDict[tuple[int, ...], list[list[int]]] = defaultdict(list)
    grouped_relations: DefaultDict[tuple[int, ...], list[list[int]]] = defaultdict(list)
    seen_template_keys: set[tuple[int, ...]] = set()

    for start_node in sorted(adjacency):
        graph_slot = batch_cpu[start_node]
        graph_seed = _graph_seed_id(graph_slot, graph_ids_cpu)
        local_start = start_node - starts[graph_slot]
        rng = random.Random(seed + 1_000_003 * (graph_seed + 1) + 9_176 * (local_start + 1))

        sampled_paths: list[
            tuple[
                tuple[int, ...],
                list[int],
                list[int],
                list[int],
            ]
        ] = []
        sampled_path_count = 0

        def dfs(current_node: int, node_path: list[int], edge_path: list[int], rel_path: list[int]) -> None:
            nonlocal sampled_path_count
            depth = len(edge_path)
            if depth >= min_path_length:
                # `exclude_identity_paths` is subsumed by the revisit guard below, which
                # rejects node_path[0] == node_path[-1] unconditionally: an endpoint repeat
                # makes the closure edge a self-loop, which the transport construction
                # cannot close (see src/data/face_geometry.py).
                key_parts: list[int] = [node_type_cpu[node_path[0]]]
                for rel_id, next_node in zip(rel_path, node_path[1:], strict=True):
                    key_parts.extend([rel_id, node_type_cpu[next_node]])
                key = tuple(key_parts)
                sampled_path_count += 1
                if selected_names is None or _template_name(key) in selected_names:
                    if (
                        selected_names is None
                        and key not in seen_template_keys
                        and max_template_identities is not None
                        and len(seen_template_keys) >= max_template_identities
                    ):
                        raise ValueError(
                            "AMPSS graph schema enumeration reached guard + 1."
                        )
                    seen_template_keys.add(key)
                    sampled_paths.append((key, node_path.copy(), edge_path.copy(), rel_path.copy()))
                if (
                    max_paths_per_anchor is not None
                    and sampled_path_count >= max_paths_per_anchor
                ):
                    return
            if depth >= max_path_length:
                return

            next_hop = list(adjacency.get(current_node, []))
            rng.shuffle(next_hop)
            for next_node, edge_id in next_hop:
                if batch_cpu[next_node] != graph_slot:
                    continue
                if next_node in node_path:
                    # Simple paths only — see src/data/face_geometry.py.
                    continue
                if (
                    not allow_repeated_node_types
                    and node_type_cpu[next_node]
                    in {node_type_cpu[node] for node in node_path}
                ):
                    continue
                node_path.append(next_node)
                edge_path.append(edge_id)
                rel_path.append(edge_type_cpu[edge_id])
                dfs(next_node, node_path, edge_path, rel_path)
                node_path.pop()
                edge_path.pop()
                rel_path.pop()
                if (
                    max_paths_per_anchor is not None
                    and sampled_path_count >= max_paths_per_anchor
                ):
                    break

        dfs(start_node, [start_node], [], [])

        sampled_paths.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

        for key, node_path, edge_path, relation_path in sampled_paths:
            grouped_nodes[key].append(node_path)
            grouped_edges[key].append(edge_path)
            grouped_relations[key].append(relation_path)

    if not grouped_nodes:
        return _empty_faces(edge_index)

    face_groups: list[FaceGroup] = []
    closure_parts: list[torch.Tensor] = []
    total_paths_by_graph: DefaultDict[int, int] = defaultdict(int)
    total_incidences_by_graph: DefaultDict[int, int] = defaultdict(int)
    for key in sorted(grouped_nodes):
        selected_indices: list[int] = []
        schema_paths_by_graph: DefaultDict[int, int] = defaultdict(int)
        for path_index, nodes in enumerate(grouped_nodes[key]):
            graph_slot = batch_cpu[nodes[0]]
            if (
                max_paths_per_schema is not None
                and schema_paths_by_graph[graph_slot] >= max_paths_per_schema
            ):
                continue
            if (
                max_total_paths is not None
                and total_paths_by_graph[graph_slot] >= max_total_paths
            ):
                continue
            path_incidences = len(grouped_edges[key][path_index]) + 1
            if (
                max_total_incidences is not None
                and total_incidences_by_graph[graph_slot] + path_incidences
                > max_total_incidences
            ):
                continue
            selected_indices.append(path_index)
            schema_paths_by_graph[graph_slot] += 1
            total_paths_by_graph[graph_slot] += 1
            total_incidences_by_graph[graph_slot] += path_incidences
        if not selected_indices:
            continue

        face_node_index = torch.tensor(
            [grouped_nodes[key][index] for index in selected_indices],
            dtype=torch.long,
            device=device,
        )
        face_edge_index = torch.tensor(
            [grouped_edges[key][index] for index in selected_indices],
            dtype=torch.long,
            device=device,
        )
        relation_index = torch.tensor(
            [grouped_relations[key][index] for index in selected_indices],
            dtype=torch.long,
            device=device,
        )
        validate_face_orientation(
            face_node_index=face_node_index,
            face_edge_index=face_edge_index,
            edge_index=edge_index,
            template_name=_template_name(key),
        )
        validate_relation_sequence_constant(
            relation_index=relation_index,
            template_name=_template_name(key),
        )
        face_edge_sign = torch.ones_like(face_edge_index, dtype=torch.float32, device=device)
        closure_edge_index = torch.stack(
            [face_node_index[:, -1], face_node_index[:, 0]],
            dim=1,
        )
        closure_parts.append(closure_edge_index)
        face_groups.append(
            FaceGroup(
                template_name=_template_name(key),
                node_type_sequence=_node_type_sequence_from_key(key),
                face_node_index=face_node_index,
                face_edge_index=face_edge_index,
                relation_index=relation_index,
                face_edge_sign=face_edge_sign,
                closure_edge_index=closure_edge_index,
            )
        )
    if not closure_parts:
        return _empty_faces(edge_index)
    closure_edge_tensor = torch.cat(closure_parts, dim=0).t().contiguous()
    return tuple(face_groups), closure_edge_tensor
