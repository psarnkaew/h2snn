"""Card 9 W1 (pre-registration v3, spec 1.29; built on card 7 V1 and card 4 K6 (3), (3b), (5)): the ablation run
manifest.

Rows: the 8 rows of `arms.ARM_CHANGES`, configuration (a) only (v3 D2: no row is tuned on its own, so there is no
`tune` row and no (b) row), seeds 1-5 (v3 D1). Nothing else (author scope rule).

Rows of `manifest.tsv` (one SLURM array task each, `ablation_array.sbatch`), all of kind `runs`: the seeds of a row
through `scripts/run_h2snn_runs.py` (seed = base_seed + run index; every run has its own `exp_dir`
`<exp_name>_run<i>` with `checkpoints/best.pt` and `config.yaml`, `run_h2snn_runs.py:112-113, 130-131`):
- every row except `full`: base seed 1, 5 runs (seeds 1-5);
- `full`: base seed 5, 1 run (seed 5). Seeds 1-4 are the main table's runs, looked up (not re-run) at
  `<output_dir>/<main exp>/final_seeds/<main exp>_tune_final_seed<s>_final_seed<s>/`
  (`run_h2snn_wandb_tuning.py:410-411, 711-718`); `lookup.json` lists those folders.

Freebase: full batch if the largest configuration fits on one H100 94 GB, else mini-batch (card 5 U3 sets it). The
mode is an input (`--freebase_mode`) and is written into every Freebase configuration and the manifest.

    python scripts/ablation/seed_runs.py --dataset acm --full_config <full tuned yaml> \
        --main_output_dir experiments --out_dir <dir> [--freebase_mode full_batch|minibatch]

`tuned_config` (the launcher's merge of a tuning base and a sweep's best parameters) is not used by the manifest any
more; it stays because `scripts/tsubame/run_row.py` uses it (the timing phase's optional H2S-NN tuned file).
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

FIELDS = ("kind", "dataset", "arm", "configuration", "config", "base_seed", "num_runs", "exp_name",
          "command")
FREEBASE_MODES = ("full_batch", "minibatch")
FULL_RUN_SEED = 5                    # `full` (a): seed 5 only; seeds 1-4 are the main table's (v3 §1)


def set_training_mode(path: Path, mode: str) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["data"] = {**raw.get("data", {}), "training_mode": mode}
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def runs_command(config: Path, base_seed: int, num_runs: int, exp_name: str) -> str:
    return (f"python scripts/run_h2snn_runs.py --config {config} --base_seed {base_seed} "
            f"--num_runs {num_runs} --exp_name {exp_name}")


def tuned_config(tune_base: Path, best_tuned: Path, out: Path) -> Path:
    """A tuning base + a sweep's best `sweep_params` (the launcher's own merge); used by `run_row.py`."""
    sys.path.insert(0, str(arms.ROOT / "scripts"))
    from run_h2snn_wandb_tuning import apply_sweep_overrides  # the launcher's own merge

    raw = yaml.safe_load(tune_base.read_text(encoding="utf-8"))
    params = json.loads(best_tuned.read_text(encoding="utf-8"))["sweep_params"]
    merged = apply_sweep_overrides(copy.deepcopy(raw), params)
    out.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    return out


def build_manifest(dataset: str, full_config: Path, main_output_dir: Path, out_dir: Path,
                   freebase_mode: str | None) -> list[dict]:
    configs_a = arms.emit_configurations_a(full_config, dataset, out_dir)
    if dataset == "freebase":
        if freebase_mode not in FREEBASE_MODES:
            raise ValueError(f"Freebase needs --freebase_mode in {FREEBASE_MODES} (card 5 U3).")
        for path in configs_a.values():
            set_training_mode(path, freebase_mode)
    rows = []
    last = arms.SEEDS[-1]
    for arm, path in configs_a.items():
        first = FULL_RUN_SEED if arm == "full" else arms.SEEDS[0]
        exp = f"{dataset}__ablation_a_{arm}"
        rows.append({"kind": "runs", "dataset": dataset, "arm": arm, "configuration": "a",
                     "config": str(path), "base_seed": first, "num_runs": last - first + 1,
                     "exp_name": exp, "command": runs_command(path, first, last - first + 1, exp)})
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
    parser.add_argument("--main_output_dir", type=Path, default=Path("experiments"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--freebase_mode", choices=FREEBASE_MODES, default=None)
    args = parser.parse_args()
    rows = build_manifest(args.dataset, args.full_config, args.main_output_dir, args.out_dir, args.freebase_mode)
    write_manifest(rows, args.out_dir / args.dataset / "manifest.tsv")
    print(f"{len(rows)} rows -> {args.out_dir / args.dataset / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
