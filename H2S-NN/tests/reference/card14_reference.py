"""Card 14: the functions the card rewrote, as they were at 66e9349 (test references only).

* `old_build_scope_batch` (+ `scope_members`, `_local`, `_merge_groups`): the per-target loop of
  `src/data/scopes.py`, verbatim; `old_cells_per_target` is the trainers' former
  `[cells.of(int(t)) for t in targets.tolist()]`.
* `old_sample_wisp_walks_chunked`: the per-chunk loop of `src/data/wisp_sampler.py`;
* `old_build_graph_wisp_path_groups` (+ `_offset_draw`, `_concat_draws`): the per-graph loop of
  `src/data/wisp_tasks.py`.

The two draw loops are verbatim except the random source: their per-chunk / per-graph CPU
`torch.Generator` became that chunk's / graph's seed, which `sample_wisp_walks` reads as its counter
stream (card 14 changed the random stream on purpose; the loops' offsets, seeds and concatenation are
what the vectorised versions must reproduce). They run on the CPU.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Sequence

import torch

from src.data.scopes import IncidenceIndex, ScopeBatch, ScopeMembers, incidence_index
from src.data.structures import FaceGroup, PreparedH2SData, ScopeIndex
from src.data.wisp_sampler import (ActionGraph, AnchorCells, WispDraw, WISP_CHUNK_ANCHORS,
                                   build_action_graph, chunk_seed, merge_wisp_cells,
                                   sample_wisp_walks)
from src.data.wisp_tasks import graph_draw_seed, require_walks_per_anchor


def old_cells_per_target(cells: AnchorCells, targets: torch.Tensor) -> list[tuple[FaceGroup, ...]]:
    """The trainers' former per-target selection of a target's cells."""
    return [cells.of(int(t)) for t in targets.tolist()]


def scope_members(graph: PreparedH2SData, target: int, cells: Sequence[FaceGroup],
                  index: IncidenceIndex | None = None) -> ScopeMembers:
    """Nodes and facts of `target`'s scope (`cells` = the target's own cells, graph ids); `index`
    = the incidence index of `graph` (built and cached if not given)."""
    if index is None:
        index = incidence_index(graph)
    incident = index.facts_of(target)
    cell_facts = [g.face_edge_index.reshape(-1) for g in cells if g.num_faces]
    cell_nodes = [g.face_node_index.reshape(-1) for g in cells if g.num_faces]
    facts = torch.unique(torch.cat([incident.to(torch.long), *cell_facts]))
    endpoints = graph.edge_index[:, facts].reshape(-1)
    nodes = torch.unique(torch.cat([torch.tensor([int(target)], dtype=torch.long),
                                    endpoints, *cell_nodes]))
    return ScopeMembers(nodes=nodes, facts=facts)


def _local(sorted_ids: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Positions of `ids` in the sorted id tensor (every id must be present)."""
    position = torch.searchsorted(sorted_ids, ids)
    if ids.numel() and (position.max() >= sorted_ids.numel()
                        or not bool(torch.equal(sorted_ids[position], ids))):
        raise ValueError("a cell or fact refers to a node or fact outside its scope.")
    return position


def _merge_groups(parts: list[tuple[FaceGroup, int]]) -> tuple[FaceGroup, torch.Tensor]:
    """Concatenate groups of one length (each already in batch ids) and their scope ids; the
    per-cell signature ids are re-indexed into the union of the signature tables."""
    first = parts[0][0]
    table: dict[tuple[int, ...], int] = {}
    signature_index = []
    for group, _ in parts:
        if group.signature_index is None:
            continue
        rows = [table.setdefault(tuple(r), len(table)) for r in group.signature_table]
        signature_index.append(torch.tensor(rows, dtype=torch.long)[group.signature_index])
    has_signatures = first.signature_index is not None
    weights = [g.path_weight for g, _ in parts]
    merged = FaceGroup(
        template_name=first.template_name,
        node_type_sequence=first.node_type_sequence,
        face_node_index=torch.cat([g.face_node_index for g, _ in parts]),
        face_edge_index=torch.cat([g.face_edge_index for g, _ in parts]),
        relation_index=torch.cat([g.relation_index for g, _ in parts]),
        face_edge_sign=torch.cat([g.face_edge_sign for g, _ in parts]),
        path_weight=None if any(w is None for w in weights) else torch.cat(weights),
        signature_index=torch.cat(signature_index) if has_signatures else None,
        signature_table=tuple(sorted(table, key=table.get)) if has_signatures else (),
    )
    scope = torch.cat([torch.full((g.num_faces,), s, dtype=torch.long) for g, s in parts])
    return merged, scope


def old_build_scope_batch(graph: PreparedH2SData, targets: torch.Tensor,
                      cells: Sequence[Sequence[FaceGroup]],
                      index: IncidenceIndex | None = None) -> ScopeBatch:
    """The disjoint union of the scopes of `targets` (node ids of `graph`, the graph given: N6);
    `cells[i]` are the cells of `targets[i]` (groups in `graph`'s node and fact ids); `index` = the
    incidence index of `graph` (built once and cached if not given)."""
    targets = torch.as_tensor(targets, dtype=torch.long).reshape(-1)
    if len(cells) != targets.numel():
        raise ValueError(f"{len(cells)} cell lists for {targets.numel()} targets.")
    position = torch.full((graph.num_nodes,), -1, dtype=torch.long)
    position[graph.target_indices] = torch.arange(graph.target_indices.numel())
    if bool((position[targets] < 0).any()):
        raise ValueError("every batch target must be a target node of the graph.")

    node_parts, fact_parts, edge_parts, node_scope, fact_scope, target_rows = [], [], [], [], [], []
    group_parts: dict[int, list[tuple[FaceGroup, int]]] = {}
    node_offset = fact_offset = 0
    if index is None:
        index = incidence_index(graph)
    for s, (target, own) in enumerate(zip(targets.tolist(), cells)):
        members = scope_members(graph, target, own, index)
        local_edges = _local(members.nodes, graph.edge_index[:, members.facts].reshape(-1))
        edge_parts.append(local_edges.reshape(2, -1) + node_offset)
        node_parts.append(members.nodes)
        fact_parts.append(members.facts)
        node_scope.append(torch.full((members.nodes.numel(),), s, dtype=torch.long))
        fact_scope.append(torch.full((members.facts.numel(),), s, dtype=torch.long))
        target_rows.append(node_offset + int(_local(members.nodes, torch.tensor([target]))[0]))
        for group in own:
            if not group.num_faces:
                continue
            relabelled = replace(
                group,
                face_node_index=_local(members.nodes, group.face_node_index.reshape(-1)
                                       ).reshape(group.face_node_index.shape) + node_offset,
                face_edge_index=_local(members.facts, group.face_edge_index.reshape(-1)
                                       ).reshape(group.face_edge_index.shape) + fact_offset)
            group_parts.setdefault(group.length, []).append((relabelled, s))
        node_offset += members.nodes.numel()
        fact_offset += members.facts.numel()

    global_node_id = torch.cat(node_parts)
    global_fact_id = torch.cat(fact_parts)
    groups, cell_scope = [], []
    for length in sorted(group_parts):
        merged, scope = _merge_groups(group_parts[length])
        groups.append(merged)
        cell_scope.append(scope)
    rows = torch.tensor(target_rows, dtype=torch.long)
    index = position[targets]
    data = replace(
        graph,
        x=graph.x[global_node_id],
        y=graph.y[global_node_id],
        node_type=graph.node_type[global_node_id],
        edge_index=torch.cat(edge_parts, dim=1),
        edge_type=graph.edge_type[global_fact_id],
        train_mask=graph.train_mask[global_node_id],
        val_mask=graph.val_mask[global_node_id],
        test_mask=graph.test_mask[global_node_id],
        target_indices=rows,
        target_y=graph.target_y[index],
        target_train_mask=graph.target_train_mask[index],
        target_val_mask=graph.target_val_mask[index],
        target_test_mask=graph.target_test_mask[index],
        face_groups=tuple(groups),
        metrics=torch.zeros(0),
        global_node_id=global_node_id,
        scope_index=ScopeIndex(torch.cat(node_scope), torch.cat(fact_scope), tuple(cell_scope),
                               int(targets.numel())),
    )
    return ScopeBatch(data=data, targets=targets, target_rows=rows,
                      global_node_id=global_node_id, global_fact_id=global_fact_id,
                      node_scope=torch.cat(node_scope), fact_scope=torch.cat(fact_scope),
                      cell_scope=tuple(cell_scope), num_scopes=int(targets.numel()))



def old_sample_wisp_walks_chunked(graph: ActionGraph, node_type: torch.Tensor, anchors: torch.Tensor,
                              walks_per_anchor: int, max_length: int, seed: int,
                              chunk_size: int = WISP_CHUNK_ANCHORS) -> WispDraw:
    """A mini-batch draw from `p` (`proposal: uniform`): `sample_wisp_walks` on consecutive chunks
    of `chunk_size` anchors, chunk `k` with its own generator seeded by `chunk_seed(seed, k)`. The
    result depends only on (seed, chunk size, anchor order). Full-batch draws do not use this."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    anchors = anchors.long().reshape(-1)
    if anchors.numel() == 0:
        raise ValueError("a draw needs at least one anchor.")
    parts = []
    for k, start in enumerate(range(0, anchors.numel(), chunk_size)):
        generator = chunk_seed(seed, k)  # card 14: the chunk's seed, read as a counter stream
        parts.append(sample_wisp_walks(graph, node_type, anchors[start:start + chunk_size],
                                       walks_per_anchor, max_length, generator))
    return WispDraw(*(torch.cat([getattr(part, field) for part in parts])
                      for field in WispDraw.__slots__))



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


def old_build_graph_wisp_path_groups(edge_index: torch.Tensor, edge_type: torch.Tensor,
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
        generator = graph_draw_seed(draw_seed, int(graph_id[g]))  # card 14: the graph's seed
        draw = sample_wisp_walks(graph, node_type[nodes], torch.arange(nodes.numel()),
                                 walks_per_anchor, max_length, generator)
        draws.append(_offset_draw(draw, node_offset, fact_offset))
    return merge_wisp_cells(_concat_draws(draws, max_length), node_type, edge_index,
                            walks_per_anchor)
