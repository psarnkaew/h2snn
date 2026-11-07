"""G3b (a2, a4): DBLP defaults 2 epochs; every comparison switch 2 epochs on ACM
(pre-registered in g3b_preregistration.md). Run from H2S-NN/:
    python ../docs/D12_evidence/g3b_switches.py
Production entry point scripts/train_h2snn.py, one run at a time. Accuracy is logged only.
Writes docs/D12_evidence/g3b_switches.json."""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path.cwd()
EVIDENCE = ROOT.parent / "docs" / "D12_evidence"
RUN_ROOT = "experiments/d12_g3b"
CFG_DIR = ROOT / "experiments" / "d12_g3b_configs"
RUNS = {
    "dblp_defaults": ("dblp", {}, {}),
    "acm_first_order": ("acm", {"encoder": "first_order"}, {}),
    "acm_geometry_context_false": ("acm", {"geometry_context": False}, {}),
    "acm_symmetric": ("acm", {}, {"coupling": "symmetric"}),
    "acm_no_path_cells": ("acm", {}, {"no_path_cells": True}),
    "acm_endpoint_edges": ("acm", {}, {"endpoint_edges": True}),
    "acm_h0_only": ("acm", {}, {"h0_only_readout": True}),
    "acm_pair_attention": ("acm", {}, {"coupling": "pair_attention"}),
    "acm_identity_maps": ("acm", {}, {"identity_maps": True, "no_path_cells": True}),
}
report = {}
CFG_DIR.mkdir(parents=True, exist_ok=True)
(EVIDENCE / "g3b_logs").mkdir(exist_ok=True)
for name, (dataset, model_extra, ablation) in RUNS.items():
    raw = yaml.safe_load((ROOT / "configs" / "experiments" / f"{dataset}_h2snn.yaml").read_text(encoding="utf-8"))
    raw.update(epochs=2, log_every=1, seed=42, device="cuda", exp_name=f"g3b_{name}", output_dir=RUN_ROOT,
               ablation=ablation)
    raw["model"] = {**raw["model"], **model_extra}
    cfg_path = CFG_DIR / f"g3b_{name}.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    start = time.time()
    with open(EVIDENCE / "g3b_logs" / f"g3b_{name}.txt", "w", encoding="utf-8") as log:
        rc = subprocess.call([sys.executable, "-W", "ignore", "scripts/train_h2snn.py", "--config", str(cfg_path),
                              "--device", "cuda"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    run_dir = ROOT / RUN_ROOT / f"g3b_{name}"
    entry = {"rc": rc, "minutes": (time.time() - start) / 60,
             "best_test_metrics_written": (run_dir / "best_test_metrics.json").exists()}
    if (run_dir / "metrics.jsonl").exists():
        rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        entry["train_losses"] = [r["train"]["loss"] for r in rows]
        entry["all_finite"] = all(math.isfinite(v) for r in rows for split in ("train", "val", "test")
                                  for v in r[split].values() if isinstance(v, (int, float)))
        entry["val_macro_f1_logged_only"] = [r["val"]["macro_f1"] for r in rows]
        entry["diagnostic_keys"] = sorted(k for k in rows[0]["train"] if k in (
            "eta", "eta_lambda_max", "filter_norm_sum", "filter_gain", "lambda_max_A"))
        if "filter_gain" in rows[0]["train"]:
            entry["filter_gain"] = [r["train"]["filter_gain"] for r in rows]
    best = run_dir / "checkpoints" / "best.pt"
    if best.exists():
        state = torch.load(best, map_location="cpu", weights_only=False)["state_dict"]
        entry["path_lengths_in_model"] = sorted(int(k.split(".")[2]) for k in state
                                                if k.startswith("path_encoder.input_proj.") and k.endswith("weight"))
    entry["pass"] = rc == 0 and entry["best_test_metrics_written"] and entry.get("all_finite", False)
    report[name] = entry
    print(name, json.dumps({k: entry[k] for k in ("rc", "pass", "minutes")}), flush=True)
(EVIDENCE / "g3b_switches.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
