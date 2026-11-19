"""Card 7 V1 (ablation plan v2; built on card 4 K6 (3), (3b), (5)): the ablation run manifest.

Rows: the 7 rows of `arms.ARM_CHANGES` with configuration (a) and seeds 1-10, and the 4 arms of `arms.TUNED_ARMS`
with configuration (b) (one `tune` row each, then seeds 1-10 after the sweep). Nothing else (author scope rule).

Rows of `manifest.tsv` (one SLURM array task each, `ablation_array.sbatch`):
- `tune`  configuration (b) of a tuned arm on a dataset: the launcher command of `sweeps.py`;
- `runs`  seeds 1-10 of an arm through `scripts/run_h2snn_runs.py` (seed = base_seed + run index; every
  run has its own `exp_dir` `<exp_name>_run<i>` with `checkpoints/best.pt` and `config.yaml`,
  `run_h2snn_runs.py:112-113, 130-131`), for configuration (a) of every row and (b) of the four tuned
  arms. `full` (a) runs seeds 5-10 only: seeds 1-4 are the main table's runs, looked up (not re-run)
  at `<output_dir>/<main exp>/final_seeds/<main exp>_tune_final_seed<s>_final_seed<s>/`
  (`run_h2snn_wandb_tuning.py:410-411, 711-718`); `lookup.json` lists those folders.
A (b) `runs` row needs the arm's tuned configuration: `tuned_config()` writes it from the sweep's
`best_tuned_run0.json` (`sweep_params`, the launcher's `apply_sweep_overrides`) before the row runs.

Freebase: full batch if the largest configuration fits on one H100 94 GB, else mini-batch (plan v2; card 5 U3
sets it). The mode is an input (`--freebase_mode`) and is written into every Freebase configuration and the
manifest.

    python scripts/ablation/seed_runs.py --dataset acm --full_config <full tuned yaml> \
        --tuning_base configs/experiments/acm_h2snn_v3b.yaml --main_output_dir experiments \
        --out_dir <dir> [--freebase_mode full_batch|minibatch]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms  # noqa: E402
import sweeps  # noqa: E402

FIELDS = ("kind", "dataset", "arm", "configuration", "config", "base_seed", "num_runs", "exp_name",
          "command")
FREEBASE_MODES = ("full_batch", "minibatch")


def set_training_mode(path: Path, mode: str) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["data"] = {**raw.get("data", {}), "training_mode": mode}
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def runs_command(config: Path, base_seed: int, num_runs: int, exp_name: str) -> str:
    return (f"python scripts/run_h2snn_runs.py --config {config} --base_seed {base_seed} "
            f"--num_runs {num_runs} --exp_name {exp_name}")


def tuned_config_path(out_dir: Path, dataset: str, arm: str) -> Path:
    return out_dir / dataset / f"tuned_b_{arm}.yaml"


def tuned_config(tune_base: Path, best_tuned: Path, out: Path) -> Path:
    """The (b) configuration: the arm's tuning base + the sweep's best `sweep_params`."""
    sys.path.insert(0, str(sweeps.ROOT / "scripts"))
    from run_h2snn_wandb_tuning import apply_sweep_overrides  # the launcher's own merge

    raw = yaml.safe_load(tune_base.read_text(encoding="utf-8"))
    params = json.loads(best_tuned.read_text(encoding="utf-8"))["sweep_params"]
    merged = apply_sweep_overrides(copy.deepcopy(raw), params)
    out.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    return out


def build_manifest(dataset: str, full_config: Path, tuning_base: Path, main_output_dir: Path,
                   out_dir: Path, freebase_mode: str | None) -> list[dict]:
    configs_a = arms.emit_configurations_a(full_config, dataset, out_dir)
    sweeps_b = sweeps.emit_configurations_b(tuning_base, out_dir)
    if dataset == "freebase":
        if freebase_mode not in FREEBASE_MODES:
            raise ValueError(f"Freebase needs --freebase_mode in {FREEBASE_MODES} (card 2's dry run).")
        for path in list(configs_a.values()) + [Path(r["base_config"]) for r in sweeps_b.values()]:
            set_training_mode(path, freebase_mode)
    rows = []
    for arm, record in sweeps_b.items():
        rows.append({"kind": "tune", "dataset": dataset, "arm": arm, "configuration": "b",
                     "config": record["base_config"], "base_seed": sweeps.TUNING_SEED,
                     "num_runs": 1, "exp_name": f"{dataset}__ablation_b_{arm}",
                     "command": record["command"]})
    for arm, path in configs_a.items():
        first = 5 if arm == "full" else arms.SEEDS[0]
        exp = f"{dataset}__ablation_a_{arm}"
        rows.append({"kind": "runs", "dataset": dataset, "arm": arm, "configuration": "a",
                     "config": str(path), "base_seed": first, "num_runs": arms.SEEDS[-1] - first + 1,
                     "exp_name": exp, "command": runs_command(path, first, arms.SEEDS[-1] - first + 1,
                                                              exp)})
    for arm in arms.TUNED_ARMS:
        path = tuned_config_path(out_dir, dataset, arm)
        exp = f"{dataset}__ablation_b_{arm}_seeds"
        rows.append({"kind": "runs", "dataset": dataset, "arm": arm, "configuration": "b",
                     "config": str(path), "base_seed": 1, "num_runs": 10, "exp_name": exp,
                     "command": runs_command(path, 1, 10, exp)})
    full_raw = yaml.safe_load(full_config.read_text(encoding="utf-8"))
    main_exp = str(full_raw.get("exp_name"))
    lookup = {seed: str(main_output_dir / main_exp / "final_seeds"
                        / f"{main_exp}_tune_final_seed{seed}_final_seed{seed}")
              for seed in arms.REUSED_FULL_SEEDS}
    (out_dir / dataset).mkdir(parents=True, exist_ok=True)
    (out_dir / dataset / "lookup.json").write_text(json.dumps(
        {"full_seeds_1_4_from_main_table": lookup, "freebase_mode": freebase_mode}, indent=1),
        encoding="utf-8")
    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    lines = ["\t".join(FIELDS)] + ["\t".join(str(row[f]) for f in FIELDS) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--full_config", type=Path, required=True)
    parser.add_argument("--tuning_base", type=Path, required=True)
    parser.add_argument("--main_output_dir", type=Path, default=Path("experiments"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--freebase_mode", choices=FREEBASE_MODES, default=None)
    args = parser.parse_args()
    rows = build_manifest(args.dataset, args.full_config, args.tuning_base, args.main_output_dir,
                          args.out_dir, args.freebase_mode)
    write_manifest(rows, args.out_dir / args.dataset / "manifest.tsv")
    print(f"{len(rows)} rows -> {args.out_dir / args.dataset / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
