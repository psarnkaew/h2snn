"""H6 correction (independent review MAJOR-1): the pre-registered r rule for DBLP and IMDB
(preregistration.md, lines 49-52): the smallest r whose evaluation-draw incidence count
(sum cells x (length + 1), seed 42, l_max 4) reaches the dataset's schemas-run count (h0_data_facts:
DBLP 862,788, IMDB 143,052). No labels, no model. Incidences are non-decreasing in r in practice;
the search brackets by doubling/halving and then bisects, and checks r - 1 explicitly.
Run from H2S-NN/: python ../docs/WISP_evidence/h6_calibrate_r.py   Writes h6_calibrate_r.json"""
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path.cwd()))
from src.configs.config import TrainConfig  # noqa: E402
from src.data.dataset import build_dataset  # noqa: E402
from src.data.wisp_sampler import build_action_graph, build_wisp_path_groups  # noqa: E402

BUDGETS = {"dblp": 862_788, "imdb": 143_052}
out = {}
for name, budget in BUDGETS.items():
    raw = yaml.safe_load(Path(f"configs/experiments/{name}_h2snn.yaml").read_text(encoding="utf-8"))
    raw["device"] = "cpu"
    data = build_dataset(TrainConfig.from_dict(raw))  # schemas run: edges and fact layer
    start, end = data.node_slices[data.target_node_type]
    anchors = torch.arange(start, end)
    graph = build_action_graph(data.edge_index, data.edge_type, data.num_nodes,
                               data.fact_layer.self_inverse)
    curve = {}

    def incidences(r):
        if r not in curve:
            groups = build_wisp_path_groups(data.edge_index, data.edge_type, data.node_type,
                                            data.fact_layer.self_inverse, anchors, r, 4, 42,
                                            graph=graph)
            curve[r] = sum(g.num_faces * (g.length + 1) for g in groups)
            print(name, "r", r, "incidences", curve[r], flush=True)
        return curve[r]

    lo, hi = 1, 42
    while incidences(hi) < budget:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if incidences(mid) >= budget:
            hi = mid
        else:
            lo = mid
    r = hi if incidences(1) < budget else 1
    assert incidences(r) >= budget and (r == 1 or incidences(r - 1) < budget)
    out[name] = {"budget_schemas_incidences": budget, "r": r, "incidences_at_r": curve[r],
                 "curve": dict(sorted(curve.items()))}
(Path("..") / "docs" / "WISP_evidence" / "h6_calibrate_r.json").write_text(
    json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: (v["r"], v["incidences_at_r"]) for k, v in out.items()}))
