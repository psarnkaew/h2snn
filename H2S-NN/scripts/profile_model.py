from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.profiling import count_parameters_by_module, profile_time_and_memory
from src.builders import build_optimizer
from src.configs.config import TrainConfig
from src.data.dataset import build_dataset
from src.models.h2snn_model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("experiments/acm_profiling/profiling_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    with args.config.open("r", encoding="utf-8") as handle:
        cfg = TrainConfig.from_dict(yaml.safe_load(handle))
    cfg.device = device

    data = build_dataset(cfg).to(device)
    model = build_model(cfg, data).to(device)
    optimizer = build_optimizer(model, cfg)

    param_breakdown = count_parameters_by_module(model)
    print("Parameter count by module:")
    for name, count in param_breakdown.items():
        print(f"  {name}: {count:,}")

    timing = profile_time_and_memory(
        model=model,
        data=data,
        optimizer=optimizer,
        device=device,
        num_iters=args.num_iters,
        warmup=args.warmup,
    )
    print(f"\nTiming/memory on {device} ({args.num_iters} iters, {args.warmup} warmup):")
    for phase, stats in timing.items():
        print(f"  {phase}: {stats}")

    report = {
        "dataset": cfg.data.dataset_name,
        "device": device,
        "num_iters": args.num_iters,
        "warmup": args.warmup,
        "parameters": param_breakdown,
        "timing": timing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
