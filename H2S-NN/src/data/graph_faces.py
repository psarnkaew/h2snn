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
) -> tuple[FaceGroup, ...]:
    """Materialize typed path cells (link prediction, graph-classification batches).

    Each simple directed typed path, traversed along stored edges, becomes one path cell.
    Cells are grouped by typed path signature, so each group has one relation sequence and
    one length (the `FaceGroup` contract the path encoder and the transport rely on).
    Paths of `min_path_length..max_path_length` hops are sampled by a seeded DFS, at most
    `max_paths_per_anchor` per start node. `exclude_identity_paths` is subsumed by the
    simple-path rule (no repeated node, endpoints included).
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if edge_type.numel() != edge_index.size(1):
        raise ValueError("edge_type must contain one relation id per edge.")
    if max_paths_per_anchor is not None and max_paths_per_anchor <= 0:
        return tuple()
    if min_path_length < 1:
        raise ValueError("min_path_length must be >= 1.")
    if max_path_length < min_path_length:
        return tuple()

    num_nodes = int(node_type.numel())
    num_edges = int(edge_index.size(1))
    if num_nodes == 0 or num_edges == 0:
        return tuple()

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
                # rejects node_path[0] == node_path[-1] unconditionally: the transport assigns
                # one map per node position, so a repeated node would need two conflicting
                # maps (see src/data/face_geometry.py).
                key_parts: list[int] = [node_type_cpu[node_path[0]]]
                for rel_id, next_node in zip(rel_path, node_path[1:], strict=True):
                    key_parts.extend([rel_id, node_type_cpu[next_node]])
                key = tuple(key_parts)
                sampled_path_count += 1
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

    face_groups: list[FaceGroup] = []
    for key in sorted(grouped_nodes):
        face_node_index = torch.tensor(grouped_nodes[key], dtype=torch.long, device=device)
        face_edge_index = torch.tensor(grouped_edges[key], dtype=torch.long, device=device)
        relation_index = torch.tensor(grouped_relations[key], dtype=torch.long, device=device)
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
        face_groups.append(
            FaceGroup(
                template_name=_template_name(key),
                node_type_sequence=_node_type_sequence_from_key(key),
                face_node_index=face_node_index,
                face_edge_index=face_edge_index,
                relation_index=relation_index,
                face_edge_sign=face_edge_sign,
            )
        )
    return tuple(face_groups)
