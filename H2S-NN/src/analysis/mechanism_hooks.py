"""Read-only analysis hooks of the ablation plan v2 (`docs/ablation_plan_v2_2026-11-19.html`, "Face effect",
"Two-regime analysis"; card 5 U2). Each reads a trained model and an evaluation batch (full-batch data or a
mini-batch of scopes) and changes nothing in training.

(a) `energy_at_solution`: the spec §3 energy at the solve's state `Y* = (x*, s*, z*)` of an evaluation
    forward, with that forward's weights and maps and the solve's normalisation (`L_hat = D^{-1/2} L
    D^{-1/2}`, so `E = E_terms(D^{-1/2} Y*)` = 1/2 Y*^T L_hat Y*); per node `e_v` = the node-fact and
    node-cell terms containing `x_v`; the fact-cell part; `E_tilde = E / ||x*||^2`. `sum_v e_v + fact-cell
    = E` by construction. tau is not in E (the solve minimises 1/2 ||x - h0||^2 + tau E).
(b) Face knockout (F3): `evaluation.knockout_nodes` / `model.knockout_nodes`, applied in the model's
    evaluation forward (`H2SNN._knockout`); `knockout_forward` sets it for one forward and restores it.
(c) `transports_per_fact`: `Q_e = F_src^T F_tgt / c_e^2`, `||Q_e - I||_F` and the endpoints, in chunks.
(d) `cells_per_target`: per cell its anchor, node ids, relation sequence, schema key
    (`wisp_readouts.full_schema_keys`), `w_f` and `beta_f`. Neither `wisp_readouts.py` nor
    `schema_readout.py` returns per-cell rows (they aggregate per schema), so only that table is added.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any, NamedTuple

import torch

from src.models.sheaf_energy import EliminatedOperator, SheafEnergyOperator


class EnergyParts(NamedTuple):
    energy: torch.Tensor        # scalar E
    node_energy: torch.Tensor   # [N] e_v
    node_fact: torch.Tensor     # scalar, node-fact part
    node_cell: torch.Tensor     # scalar, node-cell part
    fact_cell: torch.Tensor     # scalar, fact-cell part


def energy_terms(op: SheafEnergyOperator, x: torch.Tensor, s: torch.Tensor, z: torch.Tensor) -> EnergyParts:
    """Every term of the three-level energy (`sheaf_energy.py` docstring) at `(x, s, z)`:
    1/2 a_e ||F_{v<|e} x_v - s_e||^2 per (fact, role), 1/2 beta_f ||T_{v,f} x_v - z_f||^2 per node-cell
    incidence, 1/2 beta_f ||R_{e,f} s_e - z_f||^2 per fact-cell incidence (maps through their orthogonal
    factor, a_e = c_e^2). Each node-fact / node-cell term goes to the one node it contains."""
    if op.node_terms:
        raise ValueError("energy_terms covers the three-level energy; an arm with extra node terms "
                         "(closure_only) is not supported.")
    a = op.a
    e_v = x.new_zeros(op.num_nodes)
    src_term = 0.5 * a * (op.maps.src.apply_orthogonal(x[op.src_idx]) - s).square().sum(-1)
    tgt_term = 0.5 * a * (op.maps.tgt.apply_orthogonal(x[op.tgt_idx]) - s).square().sum(-1)
    e_v.index_add_(0, op.src_idx, src_term)
    e_v.index_add_(0, op.tgt_idx, tgt_term)
    node_fact = src_term.sum() + tgt_term.sum()
    node_cell = x.new_zeros(())
    fact_cell = x.new_zeros(())
    for g, rows in zip(op.groups, op._cell_slices()):
        z_f = z[rows].unsqueeze(1)
        beta = g.beta.view(-1, 1)
        if op.node_cell:
            term = 0.5 * beta * (g.transport.apply(x) - z_f).square().sum(-1)        # [P, L+1]
            e_v.index_add_(0, g.face_node_index.reshape(-1), term.reshape(-1))
            node_cell = node_cell + term.sum()
        if op.fact_cell:
            term = 0.5 * beta * (g.fact_maps.apply(s) - z_f).square().sum(-1)         # [P, L]
            fact_cell = fact_cell + term.sum()
    return EnergyParts(node_fact + node_cell + fact_cell, e_v, node_fact, node_cell, fact_cell)


def _forward(model, data) -> Any:
    if model.training:
        raise ValueError("analysis hooks read evaluation forwards; call model.eval() first.")
    with torch.no_grad():
        return model.encode(data.x, data.node_type, data.edge_index, data.edge_type, data.face_groups,
                            data.num_nodes, epoch=-1, global_node_id=data.global_node_id,
                            scope_index=data.scope_index)


def _global(ids: torch.Tensor, global_node_id: torch.Tensor | None) -> torch.Tensor:
    return ids if global_node_id is None else global_node_id[ids]


def energy_at_solution(model, data) -> dict[str, Any]:
    """(a) E, e_v (with global node ids), the fact-cell part and E_tilde at the solution of one evaluation
    forward of `data` (full batch or a batch of scopes)."""
    previous = model.record_solve_state
    model.record_solve_state = True
    try:
        _forward(model, data)
        snap = model.last_solve_state
    finally:
        model.record_solve_state = previous
    with torch.no_grad():
        reduced: EliminatedOperator = snap.build(*snap.params)
        op = reduced.operator
        x, _, z = reduced.layout.split(snap.y_star)
        s = reduced.fact_states(snap.y_star)                        # every fact state, kept + eliminated
        inv_v, inv_e, inv_f = op.inverse_sqrt_degree()
        parts = energy_terms(op, inv_v.unsqueeze(-1) * x, inv_e.unsqueeze(-1) * s, inv_f.unsqueeze(-1) * z)
        x_norm2 = x.square().sum()
    return {"energy": parts.energy, "node_energy": parts.node_energy, "node_fact": parts.node_fact,
            "node_cell": parts.node_cell, "fact_cell": parts.fact_cell,
            "energy_tilde": parts.energy / x_norm2, "x_star": x,
            "node_ids": _global(torch.arange(snap.num_nodes, device=x.device), snap.global_node_id),
            "tau": snap.tau}


@contextlib.contextmanager
def knockout(model, nodes) -> Iterator[None]:
    """(b) F3: every cell through one of `nodes` (global ids) gets beta_f = 0 in the evaluation
    forwards inside the block (`H2SNN._knockout`); the previous setting is restored."""
    previous = model.knockout_nodes
    model.knockout_nodes = tuple(int(v) for v in nodes)
    try:
        yield
    finally:
        model.knockout_nodes = previous


def transports_per_fact(model, data, facts: torch.Tensor | None = None, chunk: int = 4096
                        ) -> dict[str, torch.Tensor]:
    """(c) `Q_e = F_src^T F_tgt / c_e^2` ([k, d, d]), `||Q_e - I||_F` and the two endpoint node ids (global)
    for `facts` (rows of `data.edge_index`; all facts when None), materialised `chunk` facts at a time."""
    state = _forward(model, data).operator
    if state is None:
        raise ValueError("the model has no maps (h0_only_readout).")
    maps = state.maps
    rows = (torch.arange(data.edge_index.size(1), device=data.edge_index.device) if facts is None
            else facts.to(data.edge_index.device).long())
    q_parts, dist_parts = [], []
    with torch.no_grad():
        for start in range(0, rows.numel(), chunk):
            part = rows[start:start + chunk]
            f_src = maps.src.select(part).materialize()
            f_tgt = maps.tgt.select(part).materialize()
            c2 = maps.src.scale.index_select(0, part).square().view(-1, 1, 1)
            q = f_src.transpose(-2, -1) @ f_tgt / c2
            eye = torch.eye(q.size(-1), dtype=q.dtype, device=q.device)
            q_parts.append(q)
            dist_parts.append((q - eye).flatten(1).norm(dim=1))
    ends = data.edge_index[:, rows]
    return {"facts": rows, "Q": torch.cat(q_parts) if q_parts else rows.new_zeros((0,)),
            "distance_to_identity": torch.cat(dist_parts) if dist_parts else rows.new_zeros((0,)),
            "src": _global(ends[0], data.global_node_id), "tgt": _global(ends[1], data.global_node_id)}


def cells_per_target(model, data) -> list[dict[str, Any]]:
    """(d) Per solved face group of one evaluation forward: for every cell its anchor (the end that is a
    target node; the start end if both are), node ids in order (global), relation sequence, schema key
    (`wisp_readouts.full_schema_keys`), `w_f` and `beta_f = w_f m_f`."""
    from src.analysis.wisp_readouts import full_schema_keys

    previous = model.record_solve_state
    model.record_solve_state = True
    try:
        state = _forward(model, data).operator
        snap = model.last_solve_state
    finally:
        model.record_solve_state = previous
    targets = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.edge_index.device)
    targets[data.target_indices] = True
    self_inverse = tuple(data.fact_layer.self_inverse) if data.fact_layer is not None else ()
    out = []
    for group, w in zip(snap.groups, state.path_weights):
        nodes = group.face_node_index
        start_is_target = targets[nodes[:, 0]]
        end_is_target = targets[nodes[:, -1]]
        anchor = torch.where(start_is_target, nodes[:, 0],
                             torch.where(end_is_target, nodes[:, -1], torch.full_like(nodes[:, 0], -1)))
        beta = w if group.path_weight is None else w * group.path_weight.to(w.dtype)
        out.append({"length": group.length,
                    "anchor": torch.where(anchor >= 0, _global(anchor.clamp_min(0), data.global_node_id), anchor),
                    "node_ids": _global(nodes, data.global_node_id),
                    "relations": group.relation_index,
                    "schema_keys": full_schema_keys(group, data.node_type, self_inverse),
                    "w": w.detach(), "beta": beta.detach()})
    return out


__all__ = ["EnergyParts", "cells_per_target", "energy_at_solution", "energy_terms", "knockout",
           "transports_per_fact"]
