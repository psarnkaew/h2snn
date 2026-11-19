"""W&B tuning for the baseline comparison cells.

Mirrors ``scripts/run_h2snn_wandb_tuning.py``: the same sweep engine
(:mod:`src.utils.wandb_campaign`), the same Bayes + Hyperband setup, the same
fingerprinted resumable state, and one fresh worker process per trial. Only the thing being
fitted differs, which comes from :mod:`src.baselines.campaign`.

Examples::

    # Check every node-classification cell resolves, without running anything.
    python scripts/run_baseline_tuning.py --task node_classification \\
        --models all --datasets all --validate_only

    # Tune one cell.
    python scripts/run_baseline_tuning.py --task link_prediction \\
        --models pshgcn --datasets lastfm --n_trials 30

    # Partition a graph-classification cell across machines. Units are
    # (repetition, fold) pairs, 50 per cell; never run the same index twice.
    python scripts/run_baseline_tuning.py --task graph_classification \\
        --models csnn --datasets mutag --unit_start 0 --unit_count 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Both entry points are spawned by absolute path -- run_baseline_tuning re-spawns itself
# once per trial (src/utils/wandb_campaign.py: run_unit_sweep) -- so sys.path[0] is
# scripts/, not the project root, and ``import src`` fails before argparse ever runs.
# Every other entry point in scripts/ carries the same three lines.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.campaign import (
    Cell,
    build_raw_base,
    build_sweep_config,
    build_units,
    default_epochs,
    fit_cell,
    monitor_for,
    resolve_device,
    state_path_for,
    summarize_cell,
)
from src.baselines.budget import default_budget
from src.baselines.progress import WandbEpochReport
from src.baselines.registry import DATASETS, models_for_task, unsupported_reason
from src.utils.wandb_campaign import (
    CampaignConfig,
    run_campaign_worker,
    run_unit_sweep,
)

ENTRYPOINT = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Not `required`: each trial's worker is re-spawned with only --worker_context (run_unit_sweep), which
    # argparse refused, so no baseline trial could run (card 2 D3); main() requires --task otherwise.
    parser.add_argument("--task", type=str, default=None, choices=sorted(DATASETS))
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/baselines"))
    parser.add_argument("--state_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true",
                        help="Card 2 D3: epoch budget 1 for every trial (launcher check, never a result); "
                        "W&B runs tagged dry-run.")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--unit_start", type=int, default=0)
    parser.add_argument("--unit_count", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default="h2snn_baseline")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sweep_id", type=str, default="")
    parser.add_argument("--validate_only", action="store_true")
    # Internal: the per-trial worker re-invokes this script with a context file.
    parser.add_argument("--worker_context", type=Path, default=None)
    parser.add_argument(
        "--final_seeds", type=int, nargs="+", default=None,
        help="Instead of tuning (card 2 D6.8, as H2S-NN's --final_seeds): train each selected cell's best trial of "
        "its finished seed-0 sweep (by the validation monitor) on these seeds (spec §6: 0 1 2 3 4), same budget, "
        "nothing re-tuned.")
    return parser.parse_args()


def resolve_cells(task: str, models: list[str], datasets: list[str]) -> list[Cell]:
    """Expand ``all`` and drop cells the registry marks unsupported."""

    known_models = [spec.name for spec in models_for_task(task)]
    chosen_models = known_models if models == ["all"] else models
    known_datasets = list(DATASETS[task])
    chosen_datasets = known_datasets if datasets == ["all"] else datasets

    for name in chosen_models:
        if name not in known_models:
            raise SystemExit(
                f"{name!r} has no {task} row. Known: {', '.join(known_models)}."
            )
    for name in chosen_datasets:
        if name not in known_datasets:
            raise SystemExit(
                f"{name!r} is not a {task} dataset. Known: {', '.join(known_datasets)}."
            )

    cells: list[Cell] = []
    for model in chosen_models:
        for dataset in chosen_datasets:
            reason = unsupported_reason(model, task, dataset)
            if reason:
                print(f"skip {model}/{dataset}: {reason}")
                continue
            cells.append(Cell(model=model, task=task, dataset=dataset))
    return cells


def train_trial(
    raw_base: dict[str, Any],
    *,
    params: dict[str, Any],
    allowed_parameters: Any,
    run_index: int,
    num_runs: int,
    seed: int,
    base_exp_name: str,
    output_dir: Path,
    monitor: str,
    trial_name: str,
    unit_label: str,
    unit_name: str | None,
    record_extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """One sweep trial: build the cell's bundle and fit the adapter on it.

    Runs in a fresh worker process, so everything it needs comes from ``raw_base`` --
    including the campaign's ``--device`` and budget, which is why they are put there
    rather than read from the parent's arguments.

    ``WandbEpochReport`` is what gives the sweep's Hyperband bracket something to prune
    on: without a per-epoch series, every trial reaches the end of its budget no matter
    how badly it was doing at epoch 20.
    """

    device = resolve_device(raw_base)
    record = fit_cell(
        raw_base,
        params,
        source="tuned",
        device=device,
        report=WandbEpochReport(monitor),
    )
    record.setdefault("device", device.type)
    # Card 2 D6.8: the trial's parameters travel with its record (as H2S-NN's `sweep_params`), so the
    # final-seeds step reruns the selected trial's values unchanged.
    record["sweep_params"] = dict(params)
    record["trial_name"] = trial_name
    record["unit"] = unit_name or f"{unit_label}{run_index}"
    if record_extra:
        record.update(record_extra)
    return record


def summarize_unit(records: list[dict[str, Any]], monitor: str, path: Path) -> dict[str, Any]:
    """One unit's selected trial: the best validation `monitor`, written to `path` (`best_tuned_<unit>.json`).

    `run_unit_sweep` calls `summarize(records, monitor, path)`; this launcher passed that monitor on as the task
    of `summarize_cell`, so every baseline sweep raised "Unsupported task_type 'macro_f1'" after its trials (card 2
    D3). Same selection as H2S-NN's `summarize_tuned_seed`; the cell summary over units stays `summarize_cell`.
    """
    from src.utils.run_reports import write_run_record

    best = select_best_trial(records, monitor)
    write_run_record(path, best)
    return best


def select_best_trial(records: list[dict[str, Any]], monitor: str) -> dict[str, Any]:
    """The trial with the best sweep objective (`sweep_objective`: `val_subset_macro_f1` on the 8 H2GB NC datasets,
    the whole-split validation `monitor` elsewhere; card 1f U6, option A); the first on a tie."""
    from src.utils.wandb_campaign import sweep_objective

    if not records:
        raise RuntimeError("No completed tuning records to summarize.")
    return max(records, key=lambda record: sweep_objective(record, monitor))


def run_final_seeds(cell: Cell, *, seeds: list[int], output_dir: Path, monitor: str, device: str | None,
                    epochs: int | None, patience: int | None) -> list[dict[str, Any]]:
    """Card 2 D6.8 (author decision 6): the baseline counterpart of H2S-NN's `--final_seeds` (spec §6: tune on seed 0,
    then the best configuration on seeds 0-4). The finished seed-0 sweep's trials (`tuning_trials_run0.json`, written
    by `run_unit_sweep`) -> the best by the validation `monitor` -> its `sweep_params` trained unchanged on each seed
    with the tuning budget (`build_raw_base` with the same `--epochs` / `--patience`); one record per seed
    (`final_seed<s>.json`) and the cell summary (`final_seeds_summary.json`). No W&B sweep, nothing re-tuned."""
    from src.utils.run_reports import write_run_record
    from src.utils.wandb_campaign import CampaignUnit

    cell_dir = output_dir / cell.token
    trials_path = cell_dir / "tuning_trials_run0.json"
    if not trials_path.exists():
        raise FileNotFoundError(f"{trials_path} is missing: run the seed-0 sweep of {cell.token} first.")
    trials = json.loads(trials_path.read_text(encoding="utf-8"))
    best = select_best_trial(trials, monitor)
    if "sweep_params" not in best:
        raise ValueError(f"{trials_path}: the selected trial has no sweep_params (a sweep recorded before card 2 "
                         "D6.8); its parameters are only in W&B.")
    params = dict(best["sweep_params"])
    records: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        unit = CampaignUnit(run_index=index, num_runs=len(seeds), seed=int(seed), label="final_seed",
                            name=f"final_seed{seed}")
        raw_base = build_raw_base(cell, unit, device=device, epochs=epochs, patience=patience)
        record = fit_cell(raw_base, dict(params), source="tuned", device=resolve_device(raw_base), report=None)
        record.update({"final_seed": int(seed), "sweep_params": params, "selected_trial": best.get("trial_name"),
                       "selected_from": str(trials_path), "monitor": monitor})
        write_run_record(cell_dir / f"final_seed{seed}.json", record)
        records.append(record)
    summarize_cell(records, cell.task, cell_dir / "final_seeds_summary.json")
    return records


def main() -> None:
    args = parse_args()

    if args.worker_context is not None:
        run_campaign_worker(args.worker_context, train_trial=train_trial, known_parameters=set())
        return
    if args.task is None:
        raise SystemExit("run_baseline_tuning.py: --task is required unless --worker_context is used.")

    if args.dry_run:  # card 2 D3: the explicit dry-run flag (one epoch)
        if args.epochs is not None and args.epochs != 1:
            raise SystemExit("--dry_run sets the epoch budget to 1; do not combine it with --epochs.")
        args.epochs = 1
        args.wandb_tags = list(dict.fromkeys([*(args.wandb_tags or ()), "dry-run"]))
    cells = resolve_cells(args.task, args.models, args.datasets)
    if not cells:
        raise SystemExit("No runnable cells selected.")

    monitor = monitor_for(args.task)
    if args.final_seeds is not None:  # card 2 D6.8: no sweep, the selected trial on each seed
        for cell in cells:
            run_final_seeds(cell, seeds=list(args.final_seeds), output_dir=args.output_dir, monitor=monitor,
                            device=args.device, epochs=args.epochs, patience=args.patience)
            print(f"{cell.token}: final seeds {list(args.final_seeds)} written to {args.output_dir / cell.token}")
        return
    epochs = args.epochs or default_epochs(args.task)
    state_dir = args.state_dir or (args.output_dir / "tuning_state")
    campaign = CampaignConfig(
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_group=args.wandb_group,
        wandb_tags=tuple(args.wandb_tags or ()),
        n_trials=args.n_trials,
        resume=args.resume,
        sweep_id=args.sweep_id,
    )

    all_units = build_units(args.task, base_seed=args.base_seed, num_runs=args.num_runs)
    stop = len(all_units) if args.unit_count is None else args.unit_start + args.unit_count
    units = all_units[args.unit_start : stop]
    if not units:
        raise SystemExit(
            f"--unit_start {args.unit_start} selects nothing; the task has "
            f"{len(all_units)} units."
        )

    print(
        f"task {args.task}: {len(cells)} cell(s) x {len(units)} unit(s) "
        f"x {args.n_trials} trials = {len(cells) * len(units) * args.n_trials} fits"
    )
    print(
        f"monitor {monitor} | epochs {epochs} | "
        f"patience {args.patience or default_budget(args.task).patience} | device {args.device or 'auto'} | "
        f"project {args.wandb_project}"
    )

    if args.validate_only:
        for cell in cells:
            sweep = build_sweep_config(
                cell,
                monitor=monitor,
                n_trials=args.n_trials,
                epochs=epochs,
            )
            print(f"  {cell.token}: {len(sweep['parameters'])} tunable parameters")
        print("validate_only: nothing was run.")
        return

    for cell in cells:
        sweep_config = build_sweep_config(
            cell,
            monitor=monitor,
            n_trials=args.n_trials,
            epochs=epochs,
        )
        records: list[dict[str, Any]] = []
        for unit in units:
            raw_base = build_raw_base(
                cell,
                unit,
                device=args.device,
                epochs=args.epochs,
                patience=args.patience,
            )
            best = run_unit_sweep(
                raw_base,
                campaign=campaign,
                unit=unit,
                sweep_config=sweep_config,
                base_exp_name=cell.token,
                output_dir=args.output_dir,
                task_type=args.task,
                base_config_fingerprint="",
                monitor=monitor,
                state_path=state_path_for(state_dir, cell, unit),
                worker_entrypoint=ENTRYPOINT,
                summarize=summarize_unit,
            )
            records.append(best)

        summary_path = args.output_dir / cell.token / "tuning_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = summarize_cell(records, args.task, summary_path)
        print(f"{cell.token}: {json.dumps(summary, indent=2)[:600]}")


if __name__ == "__main__":
    main()
