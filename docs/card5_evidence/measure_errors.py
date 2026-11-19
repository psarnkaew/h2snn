"""Card 5 U7: measured errors of T5-T8 (the tests assert the tolerances; this prints the values for the report).
Same fixtures as `tests/test_card5_protocol_hooks.py`; synthetic data only. Run from H2S-NN/ in env E:

    python ../docs/card5_evidence/measure_errors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))

from tests.partMB_fixtures import restore_real_data, unreachable_real_data  # noqa: E402
from tests.test_card5_protocol_hooks import F64, _logits, _model  # noqa: E402


def mx(t: torch.Tensor) -> float:
    return float(t.abs().max()) if t.numel() else 0.0


def main() -> None:
    from src.analysis.mechanism_hooks import (cells_per_target, energy_at_solution, energy_terms, knockout,
                                              transports_per_fact)
    from tests.test_sheaf_energy import Fixture
    from tests.test_td1_maps import thm4_production

    out: dict[str, float] = {}
    # T5 fixture vs dense reference; Thm 4 section
    fx = Fixture(dtype=F64)
    op = fx.operator()
    L, _ = fx.dense_reference()
    y = torch.randn(op.layout.rows, fx.d, generator=torch.Generator().manual_seed(11), dtype=F64)
    parts = energy_terms(op, *op.layout.split(y))
    dense = 0.5 * float(y.reshape(-1) @ L @ y.reshape(-1))
    out["T5 |E_hook - 1/2 Y^T L Y| (fixture)"] = abs(float(parts.energy) - dense)
    out["T5 E (fixture, scale)"] = dense
    out["T5 |sum e_v + fact-cell - E| (fixture)"] = abs(float(parts.node_energy.sum() + parts.fact_cell - parts.energy))
    op4, labels, classes, _ = thm4_production(Fixture(d=8))
    sec = energy_terms(op4, *op4.layout.split(op4.initial_guess(classes[labels])))
    out["T5 max |e_v| at Thm 4 section"] = mx(sec.node_energy)
    out["T5 |E| at Thm 4 section"] = abs(float(sec.energy))
    # T5 model solution
    _, data, model = _model()
    res = energy_at_solution(model, data)
    model.record_solve_state = True
    _logits(model, data)
    snap = model.last_solve_state
    model.record_solve_state = False
    reduced = snap.build(*snap.params)
    sop = reduced.operator
    x, _, z = reduced.layout.split(snap.y_star)
    s = reduced.fact_states(snap.y_star)
    iv, ie, iff = sop.inverse_sqrt_degree()
    full = torch.cat([x * iv.unsqueeze(-1), s * ie.unsqueeze(-1), z * iff.unsqueeze(-1)])
    quad = 0.5 * float((full * sop.apply(full)).sum())
    out["T5 |E_hook - 1/2 Y^T L_hat Y| (model)"] = abs(float(res["energy"]) - quad)
    out["T5 E (model, scale)"] = quad
    out["T5 |sum e_v + fact-cell - E| (model)"] = abs(float(res["node_energy"].sum() + res["fact_cell"] - res["energy"]))
    # T6 knockout
    base = _logits(model, data)
    with knockout(model, ()):
        out["T6 max |logits(empty) - logits|"] = mx(_logits(model, data) - base)
    with knockout(model, range(data.num_nodes)):
        every = _logits(model, data)
    _, _, no_cells = _model(arm="no_path_cells", state=model.state_dict())
    out["T6 max |logits(all) - logits(no_path_cells)|"] = mx(every - _logits(no_cells, data))
    # T7 transports
    facts = torch.arange(min(50, data.edge_index.size(1)))
    tr = transports_per_fact(model, data, facts, chunk=16)
    with torch.no_grad():
        state = model.encode(data.x, data.node_type, data.edge_index, data.edge_type, data.face_groups,
                             data.num_nodes).operator
    xv = torch.randn(facts.numel(), model.d, generator=torch.Generator().manual_seed(2), dtype=F64)
    exp = state.maps.src.select(facts).apply_inverse(state.maps.tgt.select(facts).apply(xv))
    out["T7 max |Q x - F_src^-1 F_tgt x|"] = mx((tr["Q"] @ xv.unsqueeze(-1)).squeeze(-1) - exp)
    eye = torch.eye(model.d, dtype=F64).expand_as(tr["Q"])
    out["T7 max |Q^T Q - I|"] = mx(tr["Q"].transpose(-2, -1) @ tr["Q"] - eye)
    out["T7 median ||Q - I||_F (for scale)"] = float(tr["distance_to_identity"].median())
    # T8 cells per target
    rows = cells_per_target(model, data)
    with torch.no_grad():
        state = model.encode(data.x, data.node_type, data.edge_index, data.edge_type, data.face_groups,
                             data.num_nodes).operator
    betas = model.last_cell_betas
    out["T8 max |w_hook - w_model|"] = max(mx(r["w"] - w) for r, w in zip(rows, state.path_weights))
    out["T8 max |beta_hook - beta_model|"] = max(mx(r["beta"] - b) for r, b in zip(rows, betas))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unreachable_real_data()
    try:
        main()
    finally:
        restore_real_data()
