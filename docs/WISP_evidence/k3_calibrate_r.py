"""K3 (Part K; spec v1.6 E11): calibrate r per dataset and l_max with the H1/H6 incidence rule:
the smallest r whose evaluation-draw incidence count (sum cells x (length + 1), seed 42) reaches
the dataset's schemas-run budget (h0_data_facts.json: ACM 403,122, DBLP 862,788, IMDB 143,052).
No labels, no model. l_max = 4 is recomputed as a reproduction check (expected 42 / 65 / 12).
Reachability first: distinct cells <= all walks (revisits allowed), so sum_{l=2}^{l_max} (l + 1) *
(walks of length l from the anchors, by sparse powers of the fact adjacency) bounds the reachable
incidences; if the bound is below the budget the rule has no solution and no r is recorded (the
first run doubled r without end on DBLP l_max 2 and segfaulted). The search is also capped at
anchors x r <= 4M walks per draw.
Run from H2S-NN/, one dataset per process (memory; the first all-in-one run segfaulted during
DBLP with ~3 GB of host RAM free):
    python ../docs/WISP_evidence/k3_calibrate_r.py acm|dblp|imdb    Merges into k3_calibrate_r.json"""
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path.cwd()))
from src.configs.config import TrainConfig  # noqa: E402
from src.data.dataset import build_dataset  # noqa: E402
from src.data.wisp_sampler import build_action_graph, build_wisp_path_groups  # noqa: E402

BUDGETS = {"acm": 403_122, "dblp": 862_788, "imdb": 143_052}
LENGTHS = (2, 3, 4, 5, 6)
OUT = Path("..") / "docs" / "WISP_evidence" / "k3_calibrate_r.json"
saved = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"budgets": BUDGETS,
                                                                          "table": {}}
out = saved["table"]
for name in sys.argv[1:]:
    budget = BUDGETS[name]
    raw = yaml.safe_load(Path(f"configs/experiments/{name}_h2snn.yaml").read_text(encoding="utf-8"))
    raw["device"] = "cpu"
    raw["data"] = {**raw["data"], "path_source": "schemas"}  # edges and fact layer only
    data = build_dataset(TrainConfig.from_dict(raw))
    start, end = data.node_slices[data.target_node_type]
    anchors = torch.arange(start, end)
    graph = build_action_graph(data.edge_index, data.edge_type, data.num_nodes,
                               data.fact_layer.self_inverse)
    n, ei = data.num_nodes, data.edge_index
    loop = ei[0] == ei[1]
    rows = torch.cat([ei[0], ei[1][~loop]])
    cols = torch.cat([ei[1], ei[0][~loop]])
    adjacency = torch.sparse_coo_tensor(torch.stack([rows, cols]),
                                        torch.ones(rows.numel(), dtype=torch.float64), (n, n)).coalesce()
    counts, x = {}, torch.zeros(n, dtype=torch.float64)
    x[start:end] = 1.0
    for length in range(1, max(LENGTHS) + 1):
        x = torch.sparse.mm(adjacency.t(), x.unsqueeze(1)).squeeze(1)
        counts[length] = float(x.sum())
    cap = 4_000_000 // anchors.numel()
    out[name] = {}
    for max_length in LENGTHS:
        curve = {}
        bound = sum((length + 1) * counts[length] for length in range(2, max_length + 1))
        if bound < budget:
            out[name][max_length] = {"r": None, "reason": "unreachable", "upper_bound": bound}
            print(name, "l_max", max_length, "UNREACHABLE: upper bound", int(bound), "< budget",
                  budget, flush=True)
            continue

        def incidences(r):
            if r not in curve:
                groups = build_wisp_path_groups(data.edge_index, data.edge_type, data.node_type,
                                                data.fact_layer.self_inverse, anchors, r,
                                                max_length, 42, graph=graph)
                curve[r] = sum(g.num_faces * (g.length + 1) for g in groups)
            return curve[r]

        lo, hi = 0, 16
        while incidences(hi) < budget and hi * 2 <= cap:
            lo, hi = hi, hi * 2
        if incidences(hi) < budget:
            out[name][max_length] = {"r": None, "reason": f"not reached by r = {hi} (cap)",
                                     "upper_bound": bound, "incidences_at_cap": curve[hi]}
            print(name, "l_max", max_length, "NOT REACHED by r =", hi, curve[hi], flush=True)
            continue
        while hi - lo > 1:  # invariant: incidences(lo) < budget <= incidences(hi) (lo = 0 counts as below)
            mid = (lo + hi) // 2
            if incidences(mid) >= budget:
                hi = mid
            else:
                lo = mid
        r = hi
        assert incidences(r) >= budget and (r == 1 or incidences(r - 1) < budget)
        out[name][max_length] = {"r": r, "incidences_at_r": curve[r],
                                 "incidences_at_r_minus_1": curve.get(r - 1), "upper_bound": bound}
        print(name, "l_max", max_length, "r", r, "incidences", curve[r], "r-1:", curve.get(r - 1),
              flush=True)
    OUT.write_text(json.dumps({"budgets": BUDGETS, "table": out}, indent=1), encoding="utf-8")
