"""H1: (a) calibrate the default r on ACM (smallest r whose evaluation-draw incidences, seed 42,
l_max 4, reach the schemas run's 403,122; no labels, no model); (b) T-H10 cost shape: draw time at
r0, 2r0, 4r0 (median of 3, action index prebuilt and timed separately), linear fit R^2.
Run from H2S-NN/: python ../docs/WISP_evidence/h1_calibrate_cost.py   Writes h1_calibrate_cost.json"""
import json, sys, time
from pathlib import Path
import torch, yaml
sys.path.insert(0, str(Path.cwd()))
from src.configs.config import TrainConfig
from src.data.dataset import build_dataset
from src.data.wisp_sampler import build_action_graph, build_wisp_path_groups

BUDGET = 403_122
raw = yaml.safe_load(Path("configs/experiments/acm_h2snn.yaml").read_text(encoding="utf-8"))
raw["device"] = "cpu"
data = build_dataset(TrainConfig.from_dict(raw))  # schemas run: edges, fact layer
start, end = data.node_slices[data.target_node_type]
anchors = torch.arange(start, end)
t = time.perf_counter()
graph = build_action_graph(data.edge_index, data.edge_type, data.num_nodes, data.fact_layer.self_inverse)
index_seconds = time.perf_counter() - t

def draw(r, seed=42):
    return build_wisp_path_groups(data.edge_index, data.edge_type, data.node_type,
                                  data.fact_layer.self_inverse, anchors, r, 4, seed, graph=graph)

def incidences(groups):
    return sum(g.num_faces * (g.length + 1) for g in groups)

curve, r = {}, 32
while True:
    curve[r] = incidences(draw(r))
    print("r", r, "incidences", curve[r], flush=True)
    if curve[r] >= BUDGET:
        break
    r += 1
r0 = r
groups = draw(r0)
stats = {"cells_per_length": {g.length: g.num_faces for g in groups},
         "signatures_per_length": {g.length: len(g.signature_table) for g in groups},
         "incidences": incidences(groups), "mass_per_length": {g.length: float(g.path_weight.double().sum()) for g in groups}}
times = {}
for r in (r0, 2 * r0, 4 * r0):
    samples = []
    for rep in range(3):
        t = time.perf_counter(); draw(r, seed=100 + rep); samples.append(time.perf_counter() - t)
    times[r] = sorted(samples)[1]
    print("r", r, "seconds", round(times[r], 3), flush=True)
x = torch.tensor(list(times), dtype=torch.float64); y = torch.tensor(list(times.values()), dtype=torch.float64)
A = torch.stack([x, torch.ones_like(x)], 1)
coef = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
pred = A @ coef
r2 = 1 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
out = {"budget_schemas_incidences": BUDGET, "incidence_curve": curve, "r_default": r0, "draw_at_r_default": stats,
       "action_index_seconds": index_seconds, "draw_seconds_median_of_3": times,
       "linear_fit": {"slope_s_per_r": float(coef[0]), "intercept_s": float(coef[1]), "R2": r2},
       "anchors": int(anchors.numel())}
print(json.dumps(out, indent=1))
Path("../docs/WISP_evidence/h1_calibrate_cost.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
