"""H3 variance gate (reduced settings, pre-registered in preregistration.md):
fixed model = the 5-epoch ACM smoke checkpoint (path_source wisp, path_weight exp, full, uniform
proposal, seed 42); q from that model's w_f m_f on one uniform draw (seed 43), detached;
20 training draws per proposal at equal r (seeds 1000..1019, the same for both); 16 Rademacher
probe matrices z (fixed seed); V(q) = mean_draws mean_probes ||(L_P_hat^(i) - mean_q) z||^2 with
L_P_hat = sum_f w_f m_f L_f applied matrix-free (Welford over draws). Gate: V(adaptive) <= 0.8 V(uniform).
Run from H2S-NN/:  python ../docs/WISP_evidence/h3_variance_gate.py   Writes h3_variance_gate.json"""
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src.builders import build_experiment_components  # noqa: E402
from src.configs.config import TrainConfig  # noqa: E402
from src.data.wisp_sampler import (build_action_graph, build_wisp_path_groups,  # noqa: E402
                                   count_actions, proposal_from_mass, sample_wisp_walks)
from src.utils.misc import seed_everything  # noqa: E402

OUT = ROOT.parent / "docs" / "WISP_evidence" / "h3_variance_gate.json"
raw = yaml.safe_load((ROOT / "configs" / "experiments" / "acm_h2snn.yaml").read_text(encoding="utf-8"))
raw["data"] = {**raw["data"], "path_source": "wisp"}
raw.update(epochs=5, seed=42, device="cuda", exp_name="h3_smoke_acm_5ep", output_dir="experiments/wisp_h3")
cfg = TrainConfig.from_dict(raw)
seed_everything(cfg.seed, cfg.device)
data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
checkpoint = cfg.exp_dir / "checkpoints" / "last.pt"
if not checkpoint.exists():
    try:
        trainer.fit(data)
    finally:
        logger.close()
model.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])
device = torch.device("cuda")
model.to(device).eval()
dev = data.to(device)
self_inverse = data.fact_layer.self_inverse
graph = build_action_graph(data.edge_index, data.edge_type, data.num_nodes, self_inverse)
start, end = data.node_slices[data.target_node_type]
anchors = torch.arange(start, end)
r, max_length = cfg.data.wisp_walks_per_anchor, cfg.data.wisp_max_length


def draw(seed, proposal):
    groups = build_wisp_path_groups(data.edge_index, data.edge_type, data.node_type, self_inverse,
                                    anchors, r, max_length, seed, proposal=proposal, graph=graph)
    return groups, tuple(g.to(device) for g in groups)


with torch.no_grad():
    h0 = model.encode_nodes(dev.x, dev.node_type)
    base_cpu, base = draw(43, None)
    operator = model.build_operator(h0, dev.edge_index, dev.edge_type, base, dev.num_nodes)
    q = proposal_from_mass(base_cpu, operator.path_weights, data.node_type,
                           len(data.node_type_names), count_actions(data.edge_type, self_inverse),
                           max_length, self_inverse)
    generator = torch.Generator().manual_seed(7)
    probes = [(torch.randint(0, 2, (data.num_nodes, model.d), generator=generator) * 2 - 1)
              .float().to(device) for _ in range(16)]
    result = {"r": r, "draws": 20, "probes": len(probes), "checkpoint": str(checkpoint),
              "q_len": q.length[2:].tolist(), "log_w_of_checkpoint": dict(model.last_path_stats)}
    top = torch.topk(q.action.flatten(), 8)
    result["q_act_top"] = [[int(i) // q.action.size(1), int(i) % q.action.size(1), float(v)]
                           for v, i in zip(top.values, top.indices)]
    for name, proposal in (("uniform", None), ("adaptive", q)):
        mean = [torch.zeros_like(z) for z in probes]
        m2 = [torch.zeros_like(z) for z in probes]
        seconds = []
        for n, seed in enumerate(range(1000, 1020), start=1):
            t = time.perf_counter()
            _, groups = draw(seed, proposal)
            seconds.append(time.perf_counter() - t)
            op = model.build_operator(h0, dev.edge_index, dev.edge_type, groups, dev.num_nodes)
            for k, z in enumerate(probes):
                y = op.laplacian.apply_path(z)
                delta = y - mean[k]
                mean[k] += delta / n
                m2[k] += delta * (y - mean[k])
        variance = sum(float(m.sum()) / 20 for m in m2) / len(probes)
        walks = sample_wisp_walks(graph, data.node_type, anchors, r, max_length,
                                  torch.Generator().manual_seed(1000), proposal)
        result[name] = {"V": variance, "draw_seconds_median": sorted(seconds)[10],
                        "ratio_max": float(walks.ratio.max()), "ratio_mean": float(walks.ratio.mean()),
                        "accepted_walks": int(walks.ratio.numel())}
        print(name, json.dumps(result[name]), flush=True)
    result["reduction"] = 1.0 - result["adaptive"]["V"] / result["uniform"]["V"]
    result["gate_pass"] = result["adaptive"]["V"] <= 0.8 * result["uniform"]["V"]
print(json.dumps({k: result[k] for k in ("reduction", "gate_pass", "q_len")}, indent=1))
OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
