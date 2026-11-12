"""Cells of the fixed-literature-schema row (spec v1.20 §6 U1; cards Part O O3, Part P P0b; a
comparison row of H2S-NN, not the method and not a baseline).

The main model with one change, the path source: instead of WISP's schema-free walks, cells are
drawn only along the listed meta-paths (`LITERATURE_METAPATHS`, HGB's published table, paths that
start at the target type).
- Budget (O-2): from every target node, the tuned r walks per anchor split evenly over the k listed
  paths (`literature_walk_split`, at least one each, summing to r).
- Hop rule (D-O3): within a listed path every hop follows WISP's rule with the action fixed by the
  list: a uniform fact of the listed (relation, direction) from the current node to a node not on
  the walk (`wisp_sampler.open_actions` / `unvisited_fact_step`, WISP's own step); if there is none
  the walk is discarded (a dead end, as in WISP). So P(walk) = prod_j 1 / |N_j^unvisited|.
- Mass (D-O4): each accepted walk of listed path pi carries m = 1 / (k r_pi), so every listed path
  has the same share and the total mass per target is at most 1, as `full`'s (each WISP walk carries
  1 / r); merged per cell by summing (`merge_length_group`).
- Cells use the WISP cell format (one group per length, D8 identity, per-cell signature id), so `g`
  reads a listed cell exactly as it reads a WISP cell (D-O2).
"""

from __future__ import annotations

import torch

from src.configs.config import literature_walk_split
from src.data.path_sampler import Walks, concat_walks
from src.data.schemas import ResolvedSchema
from src.data.structures import FaceGroup
from src.data.wisp_sampler import (
    ActionGraph,
    action_of,
    build_action_graph,
    merge_length_group,
    open_actions,
    unvisited_fact_step,
)


def literature_walk_plan(schemas: tuple[ResolvedSchema, ...], walks_per_anchor: int
                         ) -> tuple[tuple[ResolvedSchema, int], ...]:
    """(listed path, walks per anchor) in table order; the walks sum to r."""
    return tuple(zip(schemas, literature_walk_split(walks_per_anchor, len(schemas))))


def sample_listed_walks(graph: ActionGraph, hops: tuple[tuple[int, int], ...],
                        anchors: torch.Tensor, walks_per_anchor: int, mass: float,
                        generator: torch.Generator) -> Walks:
    """`walks_per_anchor` walks of one listed path from every anchor under WISP's hop rule with
    the action forced to the listed (relation, direction) of each hop; dead ends are discarded.
    One `torch.rand` per hop (the fact choice). Accepted walks carry `mass` each (float64)."""
    if walks_per_anchor <= 0:
        raise ValueError("walks_per_anchor must be positive.")
    current = anchors.long().repeat_interleave(walks_per_anchor)
    num = current.numel()
    length = len(hops)
    nodes = torch.full((num, length + 1), -1, dtype=torch.long)
    nodes[:, 0] = current
    facts = torch.full((num, length), -1, dtype=torch.long)
    signs, relations, directions = facts.clone(), facts.clone(), facts.clone()
    alive = torch.ones(num, dtype=torch.bool)
    for hop, (relation, direction) in enumerate(hops, start=1):
        u_fact = torch.rand(num, generator=generator, dtype=torch.float64)
        chosen = torch.full((num,), int(action_of(torch.tensor(relation),
                                                  torch.tensor(direction))), dtype=torch.long)
        open_, v_action, v_position, v_valid = open_actions(graph, current, nodes[:, :hop])
        alive = alive & (open_.gather(1, chosen.unsqueeze(1)).squeeze(1) > 0)  # dead end
        position, _ = unvisited_fact_step(graph, current, chosen, open_, v_action, v_position,
                                          v_valid, u_fact, alive)
        nxt = graph.other[position] if graph.other.numel() else current
        current = torch.where(alive, nxt, current)
        nodes[:, hop] = torch.where(alive, nxt, nodes[:, hop])
        for store, column in ((facts, graph.fact), (signs, graph.sign),
                              (relations, graph.relation), (directions, graph.direction)):
            store[:, hop - 1] = torch.where(alive, column[position], store[:, hop - 1])
    keep = alive
    return Walks(nodes[keep], facts[keep], signs[keep], relations[keep], directions[keep],
                 torch.full((int(keep.sum()),), float(mass), dtype=torch.float64))


def sample_literature_walks(graph: ActionGraph, schemas: tuple[ResolvedSchema, ...],
                            anchors: torch.Tensor, walks_per_anchor: int, seed: int
                            ) -> tuple[tuple[ResolvedSchema, Walks], ...]:
    """The accepted walks of every listed path (table order) of one draw (seed `seed`), each walk
    of path pi carrying m = 1 / (k r_pi)."""
    generator = torch.Generator().manual_seed(int(seed))
    k = len(schemas)
    return tuple((schema, sample_listed_walks(graph, schema.hops, anchors, r_pi,
                                              1.0 / (k * r_pi), generator))
                 for schema, r_pi in literature_walk_plan(schemas, walks_per_anchor))


def build_literature_path_groups(edge_index: torch.Tensor, edge_type: torch.Tensor,
                                 node_type: torch.Tensor, node_type_names: tuple[str, ...],
                                 schemas: tuple[ResolvedSchema, ...],
                                 self_inverse: tuple[int, ...], target_node_type: str,
                                 anchors: torch.Tensor, walks_per_anchor: int,
                                 seed: int) -> tuple[FaceGroup, ...]:
    """One draw of the row (seed `seed`): walks of every listed path from every anchor, merged
    per length into WISP-format groups (`literature_L<length>`)."""
    for schema in schemas:
        if schema.node_types[0] != target_node_type:
            raise ValueError(f"listed meta-path {schema.name!r} starts at {schema.node_types[0]!r}, "
                             f"not at the target type {target_node_type!r} (spec v1.19 §6).")
    type_ids = {name: i for i, name in enumerate(node_type_names)}
    if any(int(t) != type_ids[target_node_type] for t in node_type[anchors].unique()):
        raise ValueError("every anchor of the row is a node of the target type.")
    graph = build_action_graph(edge_index, edge_type, int(node_type.numel()), self_inverse)
    by_length: dict[int, list[Walks]] = {}
    for schema, walks in sample_literature_walks(graph, schemas, anchors, walks_per_anchor, seed):
        by_length.setdefault(schema.length, []).append(walks)
    groups: list[FaceGroup] = []
    for length in sorted(by_length):
        walks = concat_walks(by_length[length])
        if walks.nodes.size(0) == 0:
            continue
        groups.append(merge_length_group(walks, node_type, edge_index, f"literature_L{length}"))
    return tuple(groups)


__all__ = ["build_literature_path_groups", "literature_walk_plan", "sample_listed_walks",
           "sample_literature_walks"]
