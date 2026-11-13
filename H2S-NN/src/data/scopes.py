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
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch

from src.data.structures import FaceGroup, PreparedH2SData, ScopeIndex


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


def scope_members(graph: PreparedH2SData, target: int,
                  cells: Sequence[FaceGroup]) -> ScopeMembers:
    """Nodes and facts of `target`'s scope (`cells` = the target's own cells, graph ids)."""
    src, dst = graph.edge_index[0], graph.edge_index[1]
    incident = ((src == target) | (dst == target)).nonzero(as_tuple=True)[0]
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


def build_scope_batch(graph: PreparedH2SData, targets: torch.Tensor,
                      cells: Sequence[Sequence[FaceGroup]]) -> ScopeBatch:
    """The disjoint union of the scopes of `targets` (node ids of `graph`, the graph given: N6);
    `cells[i]` are the cells of `targets[i]` (groups in `graph`'s node and fact ids)."""
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
    for s, (target, own) in enumerate(zip(targets.tolist(), cells)):
        members = scope_members(graph, target, own)
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


__all__ = ["ScopeBatch", "ScopeMembers", "build_scope_batch", "scope_members"]
