"""Part M smoke run (card M4): a few epochs of one config through the task's real trainer.

Checks function and cost only, never performance: finite losses, parameters move, PCG iterations
<= T_max (forward and backward), and records time and peak memory, tau and the log-weight range
of g per epoch. Metrics are written but are not a design signal (untuned).

    python scripts/partm_smoke.py --config configs/experiments/acm_pa_h2snn_v3b.yaml \
        --device cuda --epochs 3 --out ../docs/partM_evidence/m4_smoke_acm_pa.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import TrainConfig  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _finite(record: dict) -> bool:
    return all(math.isfinite(v) for v in record.values() if isinstance(v, float))


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["device"] = args.device
    payload["epochs"] = args.epochs
    payload["exp_name"] = f"partm_smoke_{args.config.stem}"
    cfg = TrainConfig.from_dict(payload)
    seed_everything(cfg.seed, cfg.device)

    from src.builders import build_experiment_components

    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
    build_seconds = time.perf_counter() - start
    before = {name: p.detach().clone().cpu() for name, p in model.named_parameters()}
    epochs: list[dict] = []
    epoch_times: list[float] = []
    clock = [time.perf_counter()]

    def callback(epoch: int, train: dict, val: dict, test: dict) -> None:
        now = time.perf_counter()
        epoch_times.append(now - clock[0])
        clock[0] = now
        epochs.append({"epoch": epoch, "train": train, "val": val, "test": test})

    error = None
    try:
        clock[0] = time.perf_counter()
        trainer.fit(data, epoch_callback=callback)
    except Exception as exc:  # recorded and re-raised below: a smoke failure is reported as is
        error = f"{type(exc).__name__}: {exc}"
    finally:
        logger.close()
    fit_seconds = sum(epoch_times)
    after = {name: p.detach().cpu() for name, p in model.named_parameters()}
    moved = {name: float((after[name] - before[name]).abs().max()) for name in before}
    t_max = cfg.model.solver_max_iterations
    pcg_keys = [k for k in (epochs[0]["train"] if epochs else {}) if k.endswith("_iters")]
    pcg_max = {k: max(e["train"].get(k, 0.0) for e in epochs) for k in pcg_keys}
    result = {
        "config": str(args.config).replace("\\", "/"),
        "device": cfg.device,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dataset": cfg.data.dataset_name,
        "task": cfg.task_type,
        "propagation": cfg.model.propagation,
        "arm": cfg.model.arm,
        "path_source": cfg.data.path_source,
        "wisp_max_length": cfg.data.wisp_max_length,
        "wisp_walks_per_anchor": cfg.data.wisp_walks_per_anchor,
        "d": cfg.model.d,
        "epochs_run": len(epochs),
        "error": error,
        "build_seconds": build_seconds,
        "fit_seconds": fit_seconds,
        "seconds_per_epoch": [round(t, 3) for t in epoch_times],
        "peak_memory_mb": (torch.cuda.max_memory_allocated(device) / 2**20
                           if device.type == "cuda" else None),
        "t_max": t_max,
        "pcg_iterations_max": pcg_max,
        "pcg_within_t_max": all(v <= t_max for v in pcg_max.values()),
        "all_metrics_finite": all(_finite(e[s]) for e in epochs for s in ("train", "val", "test")),
        "losses": [e["train"].get("loss") for e in epochs],
        "tau": [e["train"].get("tau") for e in epochs],
        "path_log_w_max": [e["train"].get("path_log_w_max") for e in epochs],
        "path_log_w_range": [e["train"].get("path_log_w_range") for e in epochs],
        "parameters_moved": sum(v > 0 for v in moved.values()),
        "parameters_total": len(moved),
        "parameters_not_moved": sorted(name for name, v in moved.items() if v == 0),
        "epochs": epochs,
        "note": "smoke run: function and cost only; metrics are untuned, not a design signal",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    summary = {k: result[k] for k in ("dataset", "epochs_run", "error", "fit_seconds",
                                      "peak_memory_mb", "pcg_iterations_max", "pcg_within_t_max",
                                      "all_metrics_finite", "losses", "tau", "path_log_w_max",
                                      "parameters_moved", "parameters_total")}
    print(json.dumps(summary, indent=1, default=float))
    if error is not None:
        sys.exit(1)


if __name__ == "__main__":
    main()
