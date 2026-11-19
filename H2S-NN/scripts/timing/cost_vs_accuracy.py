"""Card 7 V7 (author 2026-11-19; spec 1.27 C5): the cost-vs-accuracy figure on ogbn-mag. No training: it combines
the main-table test Macro-F1 of each method with its timing run (`time_methods.py`).

One point per main-table method: test Macro-F1 (mean over the main-table seeds, sample std as error bar) against time
per epoch, and against peak GPU memory. Inputs:
- `--timing`: the timing JSONs of ogbn-mag (`time_methods.py --dataset ogbn-mag`);
- `--metrics method=<summary json>`: the main-table summary of each method (H2S-NN's `runs_summary.json` or a
  baseline's `final_seeds_summary.json`; both keep `summary.metrics.test_macro_f1.{mean, std}`).
A method with only one of the two inputs, or without a time per epoch, is listed under `missing` and not plotted.
A timing run at the card 2 central point (not tuned) is flagged in the JSON and drawn with an open marker.

    python scripts/timing/cost_vs_accuracy.py --timing <dir>/*.json --metrics h2snn=<json> gcn=<json> ... \
        --out_dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DATASET = "ogbn-mag"
METRIC = "test_macro_f1"


def read_metric(path: Path) -> tuple[float | None, float | None]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entry = payload["summary"]["metrics"][METRIC]
    return entry.get("mean"), entry.get("std")


def combine(timing_paths: list[Path], metrics: dict[str, Path]) -> dict[str, Any]:
    timing: dict[str, dict[str, Any]] = {}
    for path in timing_paths:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if record["dataset"] != DATASET:
            continue
        if record["method"] in timing:
            raise ValueError(f"two timing records for {record['method']}")
        timing[record["method"]] = record
    points, missing = [], []
    for method in sorted(set(timing) | set(metrics)):
        t = timing.get(method)
        if t is None or method not in metrics or t.get("time_per_epoch_seconds") is None:
            missing.append({"method": method, "timing": t is not None, "metric": method in metrics,
                            "time_per_epoch": None if t is None else t.get("time_per_epoch_seconds")})
            continue
        mean, std = read_metric(metrics[method])
        if mean is None:
            missing.append({"method": method, "timing": True, "metric": False, "time_per_epoch": None})
            continue
        points.append({"method": method, "test_macro_f1_mean": mean, "test_macro_f1_std": std,
                       "time_per_epoch_seconds": t["time_per_epoch_seconds"],
                       "peak_gpu_memory_mb": t["peak_gpu_memory_mb"], "parameters": t["parameters"],
                       "configuration": t["configuration"], "flagged": bool(t["flagged"]), "gpu": t.get("gpu")})
    return {"dataset": DATASET, "metric": METRIC, "points": points, "missing": missing}


def figure(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, key, label in ((axes[0], "time_per_epoch_seconds", "time per epoch (s)"),
                           (axes[1], "peak_gpu_memory_mb", "peak GPU memory (MB)")):
        for p in result["points"]:
            ax.errorbar(p[key], p["test_macro_f1_mean"], yerr=p["test_macro_f1_std"] or 0.0,
                        fmt="o", mfc="none" if p["flagged"] else None, color="#4c6fbf", capsize=2, markersize=5)
            ax.annotate(p["method"], (p[key], p["test_macro_f1_mean"]), textcoords="offset points", xytext=(4, 3),
                        fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel(label)
    axes[0].set_ylabel("test Macro-F1")
    fig.suptitle(f"{result['dataset']}: accuracy vs cost (open marker: central point, not tuned)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing", type=Path, nargs="+", required=True)
    parser.add_argument("--metrics", nargs="+", required=True, help="method=<main-table summary json>")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = {k: Path(v) for k, v in (item.split("=", 1) for item in args.metrics)}
    result = combine(args.timing, metrics)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cost_vs_accuracy.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    figure(result, args.out_dir / "cost_vs_accuracy.pdf")
    print(f"{len(result['points'])} points, {len(result['missing'])} missing -> {args.out_dir}")


if __name__ == "__main__":
    main()
