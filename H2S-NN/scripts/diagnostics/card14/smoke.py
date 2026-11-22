"""Card 14 (b): one-epoch smoke on the GPU: the trainer's own `fit` with `epochs: 1` from a saved trial
config (as the sweep's `train_trial`); checks a finite training loss and that the parameters moved.

Prints `SMOKE {json}` (loss, parameters moved, wall, peak GPU / host memory, outcome). A
`PCGConvergenceError` is recorded as the outcome (card 12's issue, not card 14's).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", type=Path, default=Path("/mnt/data/peeradon/card14_runs/smoke"))
    p.add_argument("--set", action="append", default=[], dest="sets")
    p.add_argument("--root", type=Path, default=ROOT, help="H2S-NN tree to import (default: this one)")
    args = p.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    import torch

    from src.builders import build_experiment_components
    from src.configs.config import TrainConfig
    from src.utils.misc import seed_everything

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for item in args.sets:
        key, _, value = item.partition("=")
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    raw.update(output_dir=str(args.out), exp_name=args.name, use_wandb=False, epochs=1)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=root).stdout.strip()
    print(f"smoke {args.name}: head={head} host={socket.gethostname()} "
          f"job={os.environ.get('SLURM_JOB_ID')} gpu={torch.cuda.get_device_name(0)}", flush=True)
    cfg = TrainConfig.from_dict(raw)
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed, cfg.device)
    data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    losses: list[float] = []

    def callback(epoch, train_metrics, val_metrics, test_metrics) -> None:
        losses.append(float(train_metrics["loss"]))
        print(f"smoke epoch {epoch}: train {json.dumps(train_metrics, default=str)} | val "
              f"{json.dumps(val_metrics, default=str)}", flush=True)

    start = time.time()
    outcome = "finished"
    try:
        trainer.fit(data, epoch_callback=callback)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        outcome = f"{type(exc).__name__}: {exc}"
    wall = time.time() - start
    after = model.state_dict()
    moved = sum(int(not torch.equal(before[k].to(after[k].device), after[k])) for k in before
                if before[k].is_floating_point())
    total = sum(int(v.is_floating_point()) for v in before.values())
    unmoved = [k for k in before if before[k].is_floating_point()
               and torch.equal(before[k].to(after[k].device), after[k])]
    result = {"name": args.name, "head": head, "job": os.environ.get("SLURM_JOB_ID"),
              "gpu": torch.cuda.get_device_name(0), "outcome": outcome, "train_loss": losses,
              "loss_finite": bool(losses) and all(math.isfinite(v) for v in losses),
              "params_moved": f"{moved}/{total}", "unmoved": unmoved, "wall_s": round(wall, 1),
              "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
              "host_peak_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    print("SMOKE " + json.dumps(result), flush=True)
    try:
        logger.close()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
