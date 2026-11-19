"""Card 5 U3 (spec 1.26 §6 "Search range under memory limits"): does H2S-NN `full` on HGB Freebase NC fit full batch
at the largest configuration (`d` 64, `l_max` 5, `r` 64; every other key at `configs/experiments/
freebase_h2snn_v3b.yaml`)? One training step and one evaluation forward on one GPU; prints
`torch.cuda.max_memory_allocated` / `max_memory_reserved`, the cell, incidence and fact-state counts, host peak RSS
and the wall time. No metric is computed or printed.

    python scripts/freebase_memory_probe.py [--config configs/experiments/freebase_h2snn_v3b.yaml] [--out probe.json]
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.builders import build_optimizer  # noqa: E402
from src.configs.config import TrainConfig  # noqa: E402
from src.data.dataset import build_dataset  # noqa: E402
from src.models.h2snn_model import build_model  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402

LARGEST = {"d": 64, "wisp_max_length": 5, "wisp_walks_per_anchor": 64}


def gib(value: int) -> float:
    return round(value / 2**30, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/freebase_h2snn_v3b.yaml")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw["data"] = {**raw["data"], "wisp_max_length": LARGEST["wisp_max_length"],
                   "wisp_walks_per_anchor": LARGEST["wisp_walks_per_anchor"], "training_mode": "full_batch"}
    raw["model"] = {**raw["model"], "d": LARGEST["d"]}
    raw.update(device="cuda", seed=0, exp_name="card5_freebase_probe", output_dir="experiments/card5_probe")
    cfg = TrainConfig.from_dict(raw)
    record = {"config": str(args.config), "largest": LARGEST, "training_mode": cfg.data.training_mode,
              "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                    cwd=ROOT).stdout.strip(),
              "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
              "gpu": torch.cuda.get_device_name(0), "gpu_total_gib": gib(torch.cuda.get_device_properties(0).total_memory)}
    start = time.time()
    seed_everything(cfg.seed, cfg.device)
    data = build_dataset(cfg)
    record["draw_seconds"] = round(time.time() - start, 1)
    record["nodes"] = int(data.num_nodes)
    record["facts"] = int(data.edge_index.size(1))
    record["cells_per_length"] = {int(g.length): int(g.num_faces) for g in data.face_groups}
    record["cells"] = sum(record["cells_per_length"].values())
    record["node_cell_incidences"] = sum(int(g.num_faces) * (int(g.length) + 1) for g in data.face_groups)
    record["fact_cell_incidences"] = sum(int(g.num_faces) * int(g.length) for g in data.face_groups)
    device = torch.device("cuda")
    model = build_model(cfg, data).to(device)
    record["parameters"] = sum(p.numel() for p in model.parameters())
    optimizer = build_optimizer(model, cfg)
    dev = data.to(device)
    torch.cuda.reset_peak_memory_stats()
    status = "ok"
    try:
        t = time.time()
        model.train()
        optimizer.zero_grad()
        logits = model(dev, epoch=0).logits
        mask = dev.target_train_mask
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], dev.target_y[mask].float())
                if dev.is_multilabel else torch.nn.functional.cross_entropy(logits[mask], dev.target_y[mask]))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        record["train_step_seconds"] = round(time.time() - t, 1)
        record["solver_after_train_step"] = dict(model.last_solver_stats)
        t = time.time()
        model.eval()
        with torch.no_grad():
            model(dev, epoch=-1)
        torch.cuda.synchronize()
        record["eval_forward_seconds"] = round(time.time() - t, 1)
    except torch.OutOfMemoryError as exc:
        status = f"OOM: {str(exc).splitlines()[0][:300]}"
    record["status"] = status
    record["max_memory_allocated_gib"] = gib(torch.cuda.max_memory_allocated())
    record["max_memory_reserved_gib"] = gib(torch.cuda.max_memory_reserved())
    record["host_peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)
    record["wall_seconds"] = round(time.time() - start, 1)
    print(json.dumps(record, indent=1), flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(record, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
