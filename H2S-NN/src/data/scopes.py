"""Scopes of targets and batches of scopes (spec v1.21 Lemma R, L5, R7; `docs/minibatch_design_v3B.md`
§3; decisions MB0, N6; card Part MB, MB1).

Scope of a target `t` on a given graph (the training graph, or an evaluation graph such as a test
window, N6):
    * `t`;
    * every fact of `t` (self-loop facts included) and its other endpoint;
    * every node and fact of `t`'s cells (the cells of the walks drawn from `t`).
Nothing else: no fact between two scope nodes is added unless it is a fact of `t` or lies on a cell
of `t` (no induced facts). The scope is a lower set of the incidence poset (every cell brings its
facts and nodes, every fact its two endpoints).

A batch is the **disjoint union** of its targets' scopes: a node in several scopes is copied into
each. The batch graph is an ordinary `PreparedH2SData` on which the unchanged model runs; it
carries `global_node_id` (batch row -> node id in the given graph), so featureless types read the
embedding row of the global node. `ScopeBatch` also records the scope of every node, fact and cell,
and the batch row of every target.

The facts of a target are read from an incidence index built once per graph (Part M3a A2; decisions
log, Part MB review): for every node the ids of its facts in ascending order, in CSR form (`rowptr`,
`fact_ids`; a self-loop fact is listed once). The index belongs to the graph given (N6: the training
graph and each evaluation graph have their own) and is cached on that graph's `edge_index` tensor.
The output is identical to the former O(E)-per-target comparison, which is kept as the test-only
reference `tests/scope_reference.py`.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, replace
from typing import Sequence

import torch

from src.data.structures import FaceGroup, PreparedH2SData, ScopeIndex
from src.data.wisp_sampler import AnchorCells


@dataclass(frozen=True, slots=True)
class ScopeMembers:
    """One target's scope in the given graph's ids (both sorted ascending)."""

    nodes: torch.Tensor
    facts: torch.Tensor


@dataclass(slots=True)
class ScopeBatch:
    data: PreparedH2SData                 # the batch graph; `target_indices` = the targets' rows
    targets: torch.Tensor                 # [S] node ids of the targets in the given graph
    target_rows: torch.Tensor             # [S] batch row of each target
    global_node_id: torch.Tensor          # [N_b] batch row -> node id in the given graph
    global_fact_id: torch.Tensor          # [E_b] batch fact -> fact id in the given graph
    node_scope: torch.Tensor              # [N_b] scope of each batch node
    fact_scope: torch.Tensor              # [E_b] scope of each batch fact
    cell_scope: tuple[torch.Tensor, ...]  # per batch group, [P_g] scope of each cell
    num_scopes: int


@dataclass(frozen=True, slots=True)
class IncidenceIndex:
    """CSR incidence of a graph: the facts of node `v` are `fact_ids[rowptr[v]:rowptr[v + 1]]`,
    ascending; a self-loop fact is listed once."""

    rowptr: torch.Tensor    # [N + 1]
    fact_ids: torch.Tensor  # [sum of the node degrees, self-loops counted once]

    def facts_of(self, node: int) -> torch.Tensor:
        node = int(node)
        if not 0 <= node < self.rowptr.numel() - 1:
            return self.fact_ids.new_zeros(0)
        return self.fact_ids[int(self.rowptr[node]):int(self.rowptr[node + 1])]


def build_incidence_index(edge_index: torch.Tensor, num_nodes: int) -> IncidenceIndex:
    """The incidence index of the facts `edge_index` ([2, E]) on `num_nodes` nodes (O(E log E))."""
    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)
    fact = torch.arange(src.numel(), dtype=torch.long, device=src.device)
    loop = src == dst
    nodes = torch.cat([src, dst[~loop]])
    facts = torch.cat([fact, fact[~loop]])
    size = max(int(num_nodes), int(nodes.max()) + 1 if nodes.numel() else 0)
    # (node, fact) pairs are distinct, so this key sorts by node, then by ascending fact id
    order = torch.argsort(nodes * max(src.numel(), 1) + facts)
    rowptr = torch.zeros(size + 1, dtype=torch.long, device=src.device)
    rowptr[1:] = torch.cumsum(torch.bincount(nodes, minlength=size), dim=0)
    return IncidenceIndex(rowptr=rowptr, fact_ids=facts[order])


# edge_index tensor id -> (weak reference to it, its version, num_nodes, index). An entry is used only
# while it refers to the same live tensor, unchanged in place, with the same number of nodes.
_INDEX_CACHE: dict[int, tuple[weakref.ref, int, int, IncidenceIndex]] = {}


def incidence_index(graph: PreparedH2SData) -> IncidenceIndex:
    """The incidence index of `graph` (the graph given, N6), built once and cached on its facts."""
    edge_index = graph.edge_index
    key = id(edge_index)
    entry = _INDEX_CACHE.get(key)
    if entry is not None:
        ref, version, num_nodes, index = entry
        if ref() is edge_index and version == edge_index._version and num_nodes == graph.num_nodes:
            return index
    index = build_incidence_index(edge_index, graph.num_nodes)
    _INDEX_CACHE[key] = (weakref.ref(edge_index, lambda _ref, key=key: _INDEX_CACHE.pop(key, None)),
                         edge_index._version, graph.num_nodes, index)
    return index


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
    nodes = torch.unique(torch.cat([torch.tensor([int(target)], dtype=torch.long,
                                                 device=graph.edge_index.device),
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
        signature_index.append(torch.tensor(rows, dtype=torch.long,
                                            device=group.signature_index.device)[group.signature_index])
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
    scope = torch.cat([torch.full((g.num_faces,), s, dtype=torch.long,
                                  device=g.face_node_index.device) for g, s in parts])
    return merged, scope


def _segments(start: torch.Tensor, count: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For segments `[start_i, start_i + count_i)`: (segment of every element, element positions)."""
    owner = torch.arange(count.numel(), device=count.device).repeat_interleave(count)
    offset = torch.cumsum(count, dim=0) - count
    within = torch.arange(owner.numel(), device=count.device) - offset[owner]
    return owner, start[owner] + within


def _select_rows(group: FaceGroup, rows: torch.Tensor) -> FaceGroup:
    return replace(
        group,
        face_node_index=group.face_node_index[rows],
        face_edge_index=group.face_edge_index[rows],
        relation_index=group.relation_index[rows],
        face_edge_sign=group.face_edge_sign[rows],
        path_weight=None if group.path_weight is None else group.path_weight[rows],
        signature_index=None if group.signature_index is None else group.signature_index[rows])


def _anchor_cells_by_scope(targets: torch.Tensor, cells: AnchorCells
                           ) -> tuple[list[FaceGroup], list[torch.Tensor]]:
    """The cells of every scope from cells merged per anchor: scope `s` owns the cells of anchor
    `targets[s]` (a target listed twice owns them twice). Per length: the rows sorted by (scope,
    row), as the per-target loop concatenated them. No loop over targets."""
    sorted_targets, order = torch.sort(targets, stable=True)
    groups, scopes = [], []
    for group, owner in zip(cells.groups, cells.anchors):  # one group per length
        owner = owner.to(targets.device)
        low = torch.searchsorted(sorted_targets, owner)
        count = torch.searchsorted(sorted_targets, owner, right=True) - low
        row, position = _segments(low, count)
        if row.numel() == 0:
            continue
        scope = order[position]
        perm = torch.argsort(scope * max(owner.numel(), 1) + row)
        groups.append(_select_rows(group, row[perm]))
        scopes.append(scope[perm])
    return groups, scopes


def _list_cells_by_scope(cells: Sequence[Sequence[FaceGroup]], device: torch.device
                         ) -> tuple[list[FaceGroup], list[torch.Tensor]]:
    """The per-target form (`cells[s]` = the groups of target `s`): groups of one length are
    concatenated in scope order (signature tables united, `_merge_groups`)."""
    group_parts: dict[int, list[tuple[FaceGroup, int]]] = {}
    for s, own in enumerate(cells):
        for group in own:
            if group.num_faces:
                group_parts.setdefault(group.length, []).append((group.to(device), s))
    groups, scopes = [], []
    for length in sorted(group_parts):
        merged, scope = _merge_groups(group_parts[length])
        groups.append(merged)
        scopes.append(scope.to(device))
    return groups, scopes


def build_scope_batch(graph: PreparedH2SData, targets: torch.Tensor,
                      cells: "AnchorCells | Sequence[Sequence[FaceGroup]]",
                      index: IncidenceIndex | None = None) -> ScopeBatch:
    """The disjoint union of the scopes of `targets` (node ids of `graph`, the graph given: N6), on
    `graph`'s device. `cells` = the draw merged per anchor (`AnchorCells`; target `targets[i]` owns
    the cells of that anchor), or `cells[i]` = the cells of `targets[i]` (groups in `graph`'s node
    and fact ids); `index` = the incidence index of `graph` (built once and cached if not given).

    Card 14: vectorised over targets. Scope membership is a sorted set of keys `scope * N + node`
    and `scope * E + fact`, so the batch rows are the members of scope 0 ascending, then scope 1,
    ... (the order of the former per-target loop, kept as `tests/reference/card14_reference.py`),
    and every local index is a `searchsorted` into those keys."""
    device = graph.edge_index.device
    targets = torch.as_tensor(targets, dtype=torch.long).reshape(-1).to(device)
    num_scopes = int(targets.numel())
    if isinstance(cells, AnchorCells):
        groups, cell_scopes = _anchor_cells_by_scope(targets, cells)
    else:
        if len(cells) != num_scopes:
            raise ValueError(f"{len(cells)} cell lists for {num_scopes} targets.")
        groups, cell_scopes = _list_cells_by_scope(cells, device)
    position = torch.full((graph.num_nodes,), -1, dtype=torch.long, device=device)
    position[graph.target_indices] = torch.arange(graph.target_indices.numel(), device=device)
    if bool((position[targets] < 0).any()):
        raise ValueError("every batch target must be a target node of the graph.")
    if index is None:
        index = incidence_index(graph)

    num_nodes = int(graph.num_nodes)
    num_facts = max(int(graph.edge_index.size(1)), 1)
    scope_ids = torch.arange(num_scopes, device=device)
    # facts: every fact of the target, every fact of its cells
    start = index.rowptr[targets]
    incident_scope, incident = _segments(start, index.rowptr[targets + 1] - start)
    fact_keys = [incident_scope * num_facts + index.fact_ids[incident]]
    for group, scope in zip(groups, cell_scopes):
        fact_keys.append((scope.unsqueeze(1) * num_facts + group.face_edge_index).reshape(-1))
    fact_key = torch.unique(torch.cat(fact_keys))
    fact_scope = fact_key // num_facts
    global_fact_id = fact_key % num_facts
    # nodes: the target, both endpoints of every scope fact, every node of its cells
    endpoints = graph.edge_index[:, global_fact_id].long()
    node_keys = [scope_ids * num_nodes + targets,
                 (fact_scope.unsqueeze(0) * num_nodes + endpoints).reshape(-1)]
    for group, scope in zip(groups, cell_scopes):
        node_keys.append((scope.unsqueeze(1) * num_nodes + group.face_node_index).reshape(-1))
    node_key = torch.unique(torch.cat(node_keys))
    node_scope = node_key // num_nodes
    global_node_id = node_key % num_nodes

    def node_row(scope: torch.Tensor, node: torch.Tensor) -> torch.Tensor:
        return torch.searchsorted(node_key, scope * num_nodes + node)

    batch_edges = node_row(fact_scope.unsqueeze(0), endpoints)
    rows = node_row(scope_ids, targets)
    relabelled = []
    for group, scope in zip(groups, cell_scopes):
        relabelled.append(replace(
            group,
            face_node_index=node_row(scope.unsqueeze(1), group.face_node_index),
            face_edge_index=torch.searchsorted(
                fact_key, scope.unsqueeze(1) * num_facts + group.face_edge_index)))
    cell_scope = tuple(cell_scopes)
    target_index = position[targets]
    data = replace(
        graph,
        x=graph.x[global_node_id],
        y=graph.y[global_node_id],
        node_type=graph.node_type[global_node_id],
        edge_index=batch_edges,
        edge_type=graph.edge_type[global_fact_id],
        train_mask=graph.train_mask[global_node_id],
        val_mask=graph.val_mask[global_node_id],
        test_mask=graph.test_mask[global_node_id],
        target_indices=rows,
        target_y=graph.target_y[target_index],
        target_train_mask=graph.target_train_mask[target_index],
        target_val_mask=graph.target_val_mask[target_index],
        target_test_mask=graph.target_test_mask[target_index],
        face_groups=tuple(relabelled),
        metrics=torch.zeros(0, device=device),
        global_node_id=global_node_id,
        scope_index=ScopeIndex(node_scope, fact_scope, cell_scope, num_scopes),
    )
    return ScopeBatch(data=data, targets=targets, target_rows=rows,
                      global_node_id=global_node_id, global_fact_id=global_fact_id,
                      node_scope=node_scope, fact_scope=fact_scope,
                      cell_scope=cell_scope, num_scopes=num_scopes)


__all__ = ["IncidenceIndex", "ScopeBatch", "ScopeMembers", "build_incidence_index", "build_scope_batch",
           "incidence_index", "scope_members"]
