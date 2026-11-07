"""G3 cost gate (task card Part G; pre-registered in g1_g3_preregistration.md). Run from H2S-NN/:
    python ../docs/D12_evidence/g3_cost.py
ACM, configs/experiments/acm_h2snn.yaml, seed 42, CUDA, training mode: one forward + backward
(cross-entropy on the training nodes); 1 warm-up, then 3 repeats. Gate: filter_bank vs first_order
(geometry_context on in both) overhead <= 25 % time, <= 10 % peak memory. geometry_context off:
informational only. Writes docs/D12_evidence/g3_cost.json."""
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src.configs.config import TrainConfig  # noqa: E402
from src.data.dataset import build_dataset  # noqa: E402
from src.models.h2snn_model import build_model  # noqa: E402

OUT = ROOT.parent / "docs" / "D12_evidence" / "g3_cost.json"
base = yaml.safe_load((ROOT / "configs" / "experiments" / "acm_h2snn.yaml").read_text(encoding="utf-8"))
base["device"] = "cuda"
device = torch.device("cuda")
data = build_dataset(TrainConfig.from_dict(base)).to(device)
mask, labels = data.target_train_mask, data.target_y[data.target_train_mask]
results = {"hardware": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "num_nodes": int(data.num_nodes), "num_facts": int(data.edge_index.size(1)),
           "path_cells": int(sum(g.num_faces for g in data.face_groups)), "arms": {}}

for encoder, context in (("first_order", True), ("filter_bank", True),
                         ("first_order", False), ("filter_bank", False)):
    raw = {**base, "model": {**base["model"], "encoder": encoder, "geometry_context": context}}
    cfg = TrainConfig.from_dict(raw)
    torch.manual_seed(42)
    model = build_model(cfg, data).to(device).train()

    def step() -> None:
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(data).logits[mask], labels)
        loss.backward()

    step()  # warm-up
    torch.cuda.synchronize()
    times, peaks, resident = [], [], []
    for _ in range(3):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        resident.append(torch.cuda.memory_allocated())
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        peaks.append(torch.cuda.max_memory_allocated())
    key = f"{encoder}/context={'on' if context else 'off'}"
    results["arms"][key] = {
        "seconds": times, "seconds_mean": statistics.fmean(times),
        "peak_bytes": peaks, "peak_mib_max": max(peaks) / 2**20,
        "resident_before_step_mib": resident[0] / 2**20,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    print(key, json.dumps({k: v for k, v in results["arms"][key].items() if k != "peak_bytes"}), flush=True)
    del model
    torch.cuda.empty_cache()

for context in ("on", "off"):
    fo, fb = results["arms"][f"first_order/context={context}"], results["arms"][f"filter_bank/context={context}"]
    results[f"overhead_context_{context}"] = {
        "time_pct": 100 * (fb["seconds_mean"] / fo["seconds_mean"] - 1),
        "peak_memory_pct": 100 * (fb["peak_mib_max"] / fo["peak_mib_max"] - 1),
        "parameters_delta": fb["parameters"] - fo["parameters"],
    }
gate = results["overhead_context_on"]
results["gate"] = {"time_le_25pct": gate["time_pct"] <= 25.0, "memory_le_10pct": gate["peak_memory_pct"] <= 10.0}
results["context_cost_first_order"] = {
    "time_pct": 100 * (results["arms"]["first_order/context=on"]["seconds_mean"]
                       / results["arms"]["first_order/context=off"]["seconds_mean"] - 1),
    "peak_memory_pct": 100 * (results["arms"]["first_order/context=on"]["peak_mib_max"]
                              / results["arms"]["first_order/context=off"]["peak_mib_max"] - 1)}
OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
print(json.dumps({k: results[k] for k in ("overhead_context_on", "overhead_context_off", "gate",
                                          "context_cost_first_order")}, indent=1))
