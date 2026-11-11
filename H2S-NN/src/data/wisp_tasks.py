"""WISP as the path source of link prediction and graph classification (spec v1.15 §3.2 (a), J1;
card Part M, M4).

Anchors (author, 2026-11-11): link prediction = every node of the two endpoint types of the target
relation, on the message graph (held-out facts removed in both orientations before the draw);
graph classification = every node of the graph. The sampler, its target measure and the cell
merge are those of node classification (`src/data/wisp_sampler.py`); only the anchor set
differs. Evaluation uses one fixed draw (seed of the run); training draws use `seed + epoch`.

Graph classification draws each graph on its own (its own action graph and generator), so the
cells of a graph never depend on which other graphs share its batch.
"""

from __future__ import annotations

from dataclasses import fields

import torch

from src.data.structures import FaceGroup
from src.data.wisp_sampler import (WispDraw, build_action_graph, build_wisp_path_groups,
                                   merge_wisp_cells, sample_wisp_walks)

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
        anchors=link_prediction_anchors(node_slices, target_relation),
        walks_per_anchor=walks_per_anchor, max_length=max_length, seed=seed)


def graph_draw_seed(draw_seed: int, graph_id: int) -> int:
    if not 0 <= graph_id < GRAPH_SEED_STRIDE:
        raise ValueError(f"graph id {graph_id} outside [0, {GRAPH_SEED_STRIDE}).")
    return int(draw_seed) * GRAPH_SEED_STRIDE + int(graph_id)


def _offset_draw(draw: WispDraw, node_offset: int, fact_offset: int) -> WispDraw:
    nodes = torch.where(draw.nodes >= 0, draw.nodes + node_offset, draw.nodes)
    facts = torch.where(draw.facts >= 0, draw.facts + fact_offset, draw.facts)
    return WispDraw(nodes, facts, draw.signs, draw.relations, draw.directions, draw.length,
                    draw.anchor + node_offset, draw.log_p, draw.log_q, draw.ratio,
                    draw.discarded_anchor + node_offset)


def _concat_draws(draws: list[WispDraw], max_length: int) -> WispDraw:
    if not draws:
        empty = torch.zeros(0, dtype=torch.long)
        return WispDraw(torch.zeros(0, max_length + 1, dtype=torch.long),
                        torch.zeros(0, max_length, dtype=torch.long),
                        torch.zeros(0, max_length, dtype=torch.long),
                        torch.zeros(0, max_length, dtype=torch.long),
                        torch.zeros(0, max_length, dtype=torch.long), empty, empty,
                        torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.float64),
                        torch.zeros(0, dtype=torch.float64), empty)
    names = [field.name for field in fields(WispDraw)]
    return WispDraw(**{name: torch.cat([getattr(d, name) for d in draws]) for name in names})


def build_graph_wisp_path_groups(edge_index: torch.Tensor, edge_type: torch.Tensor,
                                 node_type: torch.Tensor, batch: torch.Tensor,
                                 graph_id: torch.Tensor, self_inverse: tuple[int, ...],
                                 walks_per_anchor: int, max_length: int,
                                 draw_seed: int) -> tuple[FaceGroup, ...]:
    """WISP cells of a batch of disjoint graphs: every node of each graph is an anchor; each graph
    is drawn alone (its own action graph, generator seeded by `graph_draw_seed`), then the walks
    of all graphs are merged into one set of length groups (cells never cross graphs)."""
    require_walks_per_anchor(walks_per_anchor, "graph classification")
    batch = batch.cpu()
    node_type = node_type.cpu()
    edge_index = edge_index.cpu()
    edge_type = edge_type.cpu()
    num_graphs = int(graph_id.numel())
    edge_graph = batch[edge_index[0]] if edge_index.numel() else torch.zeros(0, dtype=torch.long)
    draws: list[WispDraw] = []
    for g in range(num_graphs):
        nodes = torch.nonzero(batch == g).flatten()
        node_offset = int(nodes.min()) if nodes.numel() else 0
        edges = torch.nonzero(edge_graph == g).flatten()
        if edges.numel() and not torch.equal(edges, torch.arange(int(edges[0]),
                                                                 int(edges[0]) + edges.numel())):
            raise ValueError("graph classification batches must store each graph's edges "
                             "contiguously (PyG Batch).")
        fact_offset = int(edges[0]) if edges.numel() else 0
        local_index = edge_index[:, edges] - node_offset
        graph = build_action_graph(local_index, edge_type[edges], int(nodes.numel()), self_inverse)
        generator = torch.Generator().manual_seed(graph_draw_seed(draw_seed, int(graph_id[g])))
        draw = sample_wisp_walks(graph, node_type[nodes], torch.arange(nodes.numel()),
                                 walks_per_anchor, max_length, generator)
        draws.append(_offset_draw(draw, node_offset, fact_offset))
    return merge_wisp_cells(_concat_draws(draws, max_length), node_type, edge_index,
                            walks_per_anchor)


__all__ = ["GRAPH_SEED_STRIDE", "build_graph_wisp_path_groups", "build_link_wisp_path_groups",
           "graph_draw_seed", "link_prediction_anchors", "require_walks_per_anchor"]
