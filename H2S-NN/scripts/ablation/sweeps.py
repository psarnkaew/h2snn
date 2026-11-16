"""Card 4 K6 (2): configurations (b), the five arms tuned on their own (pre-registration §1, B-2, B-4;
spec 1.23 X-6a): plain Bayes, 100 trials, seed 0, the validation metric, through the existing tuning
launcher `scripts/run_h2snn_wandb_tuning.py` (not edited). The base configuration is `full`'s tuning
base file + the arm's key + `tuning: {arm: <name>}` (the arm-tuning mode of `src/configs/config.py`).

Search space: §6's space (the launcher's `build_sweep_config`) restricted to the keys that act in the
arm. Removed keys (DERIVED from `h2snn_model.py`):
- `no_path_cells`: `wisp_max_length`, `wisp_walks_per_anchor` (cells are not in the solve and g's
  weights are unused, `h2snn_model.py` `cells_in_solve`); every other key acts.
- `endpoints`, `closure_only` (holonomy edges of the drawn cells), `instance_features` (cell features
  in the head), `S2` (cells, g, transports = I): every key acts; nothing removed.
The launcher builds its space itself and has no option to drop a key: for `no_path_cells` the
restricted space cannot be passed (stop item); its two removed keys would be sampled without effect.

    python scripts/ablation/sweeps.py --tuning_base configs/experiments/acm_h2snn_v3b.yaml --out_dir <dir>
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import arms  # noqa: E402
from src.configs.config import TrainConfig  # noqa: E402

REMOVED_KEYS: dict[str, tuple[str, ...]] = {
    "endpoints": (), "closure_only": (), "instance_features": (),
    "no_path_cells": ("wisp_max_length", "wisp_walks_per_anchor"), "S2": (),
}
N_TRIALS = 100
TUNING_SEED = 0


def launcher_space(base: TrainConfig) -> dict:
    from run_h2snn_wandb_tuning import build_sweep_config, default_monitor_for_task  # the launcher

    monitor = default_monitor_for_task(base.task_type).removeprefix("val_")
    return build_sweep_config(task_type=base.task_type, metric=monitor, epochs=base.epochs,
                              decoder_type=None, dataset_name=base.data.dataset_name)


def emit_configurations_b(tuning_base: Path, out_dir: Path) -> dict[str, dict]:
    base_raw = yaml.safe_load(tuning_base.read_text(encoding="utf-8"))
    dataset = str(base_raw["data"]["dataset_name"]).lower()
    out = {}
    for arm in arms.TUNED_ARMS:
        raw = arms.apply_change(base_raw, arm)
        raw["tuning"] = {"arm": arm}
        raw["exp_name"] = f"{base_raw.get('exp_name', dataset)}__ablation_b_{arm}"
        cfg = TrainConfig.from_dict(copy.deepcopy(raw))
        space = launcher_space(cfg)
        restricted = {k: v for k, v in space["parameters"].items() if k not in REMOVED_KEYS[arm]}
        sweep = {**space, "parameters": restricted}
        base_path = out_dir / dataset / f"tune_{arm}.yaml"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        record = {"arm": arm, "base_config": str(base_path), "n_trials": N_TRIALS,
                  "seed": TUNING_SEED, "method": sweep["method"], "metric": sweep["metric"],
                  "parameters": restricted, "removed_keys": list(REMOVED_KEYS[arm]),
                  "launcher_space_matches": sorted(restricted) == sorted(space["parameters"]),
                  "command": (f"python scripts/run_h2snn_wandb_tuning.py --config {base_path} "
                              f"--n_trials {N_TRIALS} --num_runs 1 --base_seed {TUNING_SEED}")}
        (out_dir / dataset / f"sweep_{arm}.json").write_text(json.dumps(record, indent=1),
                                                             encoding="utf-8")
        out[arm] = record
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning_base", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    records = emit_configurations_b(args.tuning_base, args.out_dir)
    print(json.dumps({arm: {k: r[k] for k in ("removed_keys", "launcher_space_matches", "command")}
                      for arm, r in records.items()}, indent=1))


if __name__ == "__main__":
    main()
