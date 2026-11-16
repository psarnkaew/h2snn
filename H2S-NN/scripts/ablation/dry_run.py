"""Card 4 K6: one-epoch dry run of every ablation arm (configurations (a)) on ACM with a placeholder
configuration (no tuned `full` exists yet). Records status, wall time and peak GPU memory only; no
metric is read or written (the trainer's own log files stay under `experiments/`, outside git).

    python scripts/ablation/dry_run.py --full_config configs/experiments/acm_h2snn_v3b.yaml --out <json>
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms  # noqa: E402
from src.builders import build_experiment_components  # noqa: E402
from src.configs.config import TrainConfig  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output_dir", default="experiments/card4_k6_dry_run")
    args = parser.parse_args()
    full_raw = yaml.safe_load(args.full_config.read_text(encoding="utf-8"))
    results = {}
    for arm in arms.ARM_CHANGES:
        raw = arms.apply_change(full_raw, arm)
        raw.update(epochs=1, seed=1, exp_name=f"dry_{arm}", output_dir=args.output_dir,
                   log_every=1000)
        cfg = TrainConfig.from_dict(copy.deepcopy(raw))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        try:
            seed_everything(cfg.seed, cfg.device)
            data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
            try:
                trainer.fit(data)
            finally:
                logger.close()
            status = "ok"
        except Exception as exc:  # recorded per arm, then re-raised after the table is written
            status = f"failed: {type(exc).__name__}: {exc}"
            traceback.print_exc()
        results[arm] = {"status": status, "seconds": round(time.time() - start, 1),
                        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
        print(arm, json.dumps(results[arm]), flush=True)
    args.out.write_text(json.dumps({"config": str(args.full_config), "epochs": 1, "arms": results},
                                   indent=1), encoding="utf-8")
    failed = [a for a, r in results.items() if r["status"] != "ok"]
    if failed:
        raise SystemExit(f"arms failed: {failed}")


if __name__ == "__main__":
    main()
