"""Card 2 D6b (amended 2026-11-18): nci1 log-weight diagnosis. H2S-NN GC nci1 at the D6.4 central point (d 32,
l_max 3, r 16, wd 5e-4, dropouts 0.45, pooling = the YAML's) with the given lr, up to 5 epochs or the spec §6
log-weight-range stop (> 30). Records `path_log_w_range` (max - min of log w_f over the path cells of one forward,
`src/models/h2snn_model.py:741`) of every training step and writes, per window of 50 optimizer steps, the largest
value, plus where the stop fired. No metric is read or written; W&B off.

    python ../docs/dryrun_evidence/d6b_nci1_logw.py --lr 1e-3 --out <json>        (from H2S-NN/, in a SLURM job)
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.cwd()))

import d6_dryrun as d6  # noqa: E402

WINDOW = 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch

    from src.builders import build_experiment_components
    from src.configs.config import TrainConfig
    from src.models import h2snn_model
    from src.models.anchored_solve import LogWeightRangeStopError
    from src.utils.misc import seed_everything

    d6.ARGS["point"] = "central"
    run_dir = args.out.parent / f"run_lr{args.lr:g}"
    raw, point = d6.patched_h2snn_raw("nci1", "graph_classification", run_dir)
    raw["optim"]["lr"] = point["optim.lr"] = args.lr
    raw["epochs"] = args.epochs
    raw["use_wandb"] = False
    cfg = TrainConfig.from_dict(raw)
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed, cfg.device)

    steps: list[float] = []          # largest path_log_w_range of the forwards of each optimizer step
    pending: list[float] = []        # forwards since the last step
    owner = next(c for c in vars(h2snn_model).values()
                 if isinstance(c, type) and "_encode_anchored" in vars(c))
    orig_encode = owner._encode_anchored

    def encode(self, *a, **k):
        try:
            return orig_encode(self, *a, **k)
        finally:
            if self.training and self.last_path_stats:
                pending.append(self.last_path_stats["path_log_w_range"])

    owner._encode_anchored = encode

    data, model, optimizer, _, logger, _, trainer = build_experiment_components(cfg)
    orig_step = trainer.optimizer.step

    def step(*a, **k):
        steps.append(max(pending) if pending else float("nan"))
        pending.clear()
        return orig_step(*a, **k)

    trainer.optimizer.step = step
    epoch_steps: list[int] = []
    orig_run_epoch = trainer._run_epoch

    def run_epoch(loader, epoch, train):
        out = orig_run_epoch(loader, epoch, train)
        if train:
            epoch_steps.append(len(steps))
        return out

    trainer._run_epoch = run_epoch
    stop = None
    t0 = time.perf_counter()
    try:
        trainer.fit(data)
    except LogWeightRangeStopError as exc:
        # exc.epoch -1 = an evaluation forward; otherwise the stop fired in the forward of training step at_step
        stop = {"value": round(exc.value, 4), "limit": exc.limit, "epoch": exc.epoch,
                "at_step": len(steps) + 1 if exc.epoch >= 0 else None}
    finally:
        logger.close()
    windows = [{"steps": f"{i + 1}-{min(i + WINDOW, len(steps))}", "max_range": round(max(steps[i:i + WINDOW]), 4)}
               for i in range(0, len(steps), WINDOW)]
    record = {"cell": "h2snn|graph_classification|nci1", "lr": args.lr, "point": point,
              "epochs_max": args.epochs, "steps_completed": len(steps), "steps_per_epoch_cumulative": epoch_steps,
              "stop": stop, "windows": windows, "first_step_range": round(steps[0], 4) if steps else None,
              "seconds": round(time.perf_counter() - t0, 1), "host": socket.gethostname(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
              "head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print("LOGW", json.dumps({k: record[k] for k in ("lr", "steps_completed", "stop")}))


if __name__ == "__main__":
    main()
