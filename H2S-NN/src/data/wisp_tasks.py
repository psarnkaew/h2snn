"""WISP as the path source of link prediction and graph classification (spec v1.15 §3.2 (a), J1;
card Part M, M4).

Anchors (author, 2026-11-11): link prediction = every node of the two endpoint types of the target
relation, on the message graph (held-out facts removed in both orientations before the draw);
graph classification = every node of the graph. The sampler, its target measure and the cell
merge are those of node classification (`src/data/wisp_sampler.py`); only the anchor set
differs. Evaluation uses one fixed draw (seed of the run); training draws use `seed + epoch`.

Graph classification draws each graph from its own seed (`graph_draw_seed`; card 14: one batched
draw on the device, each graph reading its own counter stream), so the cells of a graph never
depend on which other graphs share its batch.
"""

from __future__ import annotations

import torch

from src.data.path_sampler import seed_value
from src.data.structures import FaceGroup
from src.data.wisp_sampler import (build_action_graph, build_wisp_path_groups, merge_wisp_cells,
                                   sample_wisp_walks)

# Seeds of per-graph draws: `draw_seed * GRAPH_SEED_STRIDE + graph_id` (distinct per graph and
# per draw seed for graph ids below the stride).
GRAPH_SEED_STRIDE = 1 << 24


def require_walks_per_anchor(walks_per_anchor: int, task: str) -> None:
    """r has no calibrated default outside node classification (spec v1.15 J2: tuned in
    {8, 16, 32, 64}); a draw with r unset (0) is refused."""
    if walks_per_anchor < 1:
        raise ValueError(
            f"data.wisp_walks_per_anchor must be set (>= 1) to draw WISP cells for {task} "
            f"(spec v1.15 J1, J2: r is tuned, no calibrated value); got {walks_per_anchor}.")


def link_prediction_anchors(node_slices: dict[str, tuple[int, int]],
                            target_relation: tuple[str, str, str]) -> torch.Tensor:
    """Every node of the target relation's head type and tail type (one range when equal)."""
    types = list(dict.fromkeys((target_relation[0], target_relation[2])))
    return torch.cat([torch.arange(*node_slices[t]) for t in types])


def build_link_wisp_path_groups(edge_index: torch.Tensor, edge_type: torch.Tensor,
                                node_type: torch.Tensor, self_inverse: tuple[int, ...],
                                node_slices: dict[str, tuple[int, int]],
                                target_relation: tuple[str, str, str], walks_per_anchor: int,
                                max_length: int, seed: int) -> tuple[FaceGroup, ...]:
    """One WISP draw on the message graph `edge_index` from both endpoint types."""
    require_walks_per_anchor(walks_per_anchor, "link prediction")
    return build_wisp_path_groups(
        edge_index=edge_index, edge_type=edge_type, node_type=node_type,
        self_inverse=self_inverse,
        anchors=link_prediction_anchors(node_slices, target_relation).to(edge_index.device),
        walks_per_anchor=walks_per_anchor, max_length=max_length, seed=seed)


def graph_draw_seed(draw_seed: int, graph_id: int) -> int:
    if not 0 <= graph_id < GRAPH_SEED_STRIDE:
        raise ValueError(f"graph id {graph_id} outside [0, {GRAPH_SEED_STRIDE}).")
    return int(draw_seed) * GRAPH_SEED_STRIDE + int(graph_id)


def build_graph_wisp_path_groups(edge_index: torch.Tensor, edge_type: torch.Tensor,
                                 node_type: torch.Tensor, batch: torch.Tensor,
                                 graph_id: torch.Tensor, self_inverse: tuple[int, ...],
                                 walks_per_anchor: int, max_length: int,
                                 draw_seed: int) -> tuple[FaceGroup, ...]:
    """WISP cells of a batch of disjoint graphs, on the batch's device: every node of each graph is
    an anchor, and each graph's walks read the counter stream of `graph_draw_seed(draw_seed, id)` at
    the node's index inside its graph, so they are those of the graph drawn alone (card 14: one
    vectorised draw on the batch's action graph, no loop over graphs; walks never leave their
    graph). The walks of all graphs are merged into one set of length groups (cells never cross
    graphs). Nodes of a graph are contiguous (PyG Batch)."""
    require_walks_per_anchor(walks_per_anchor, "graph classification")
    device = edge_index.device
    batch = batch.to(device).long()
    graph_id = graph_id.to(device).long().view(-1)
    num_nodes = int(node_type.numel())
    graph_seed(draw_seed)  # validates the seed range for every graph id below the stride
    if graph_id.numel() and bool(((graph_id < 0) | (graph_id >= GRAPH_SEED_STRIDE)).any()):
        raise ValueError(f"graph ids outside [0, {GRAPH_SEED_STRIDE}).")
    counts = torch.bincount(batch, minlength=graph_id.numel())
    first = torch.cumsum(counts, dim=0) - counts
    nodes = torch.arange(num_nodes, device=device)
    graph = build_action_graph(edge_index, edge_type, num_nodes, self_inverse)
    draw = sample_wisp_walks(graph, node_type, nodes, walks_per_anchor, max_length,
                             anchor_seed=int(draw_seed) * GRAPH_SEED_STRIDE + graph_id[batch],
                             anchor_index=nodes - first[batch])
    return merge_wisp_cells(draw, node_type, edge_index, walks_per_anchor)


def graph_seed(draw_seed: int) -> int:
    """`draw_seed * GRAPH_SEED_STRIDE`, checked so that every graph's seed fits int64."""
    seed_value(graph_draw_seed(draw_seed, GRAPH_SEED_STRIDE - 1))
    return int(draw_seed) * GRAPH_SEED_STRIDE


__all__ = ["GRAPH_SEED_STRIDE", "build_graph_wisp_path_groups", "build_link_wisp_path_groups",
           "graph_draw_seed", "link_prediction_anchors", "require_walks_per_anchor"]
