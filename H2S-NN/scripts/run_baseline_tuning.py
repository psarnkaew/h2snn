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
from src.baselines.budget import SHARED_PATIENCE
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
    parser.add_argument("--task", type=str, required=True, choices=sorted(DATASETS))
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/baselines"))
    parser.add_argument("--state_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--unit_start", type=int, default=0)
    parser.add_argument("--unit_count", type=int, default=None)
    parser.add_argument("--hyperband_min_iter", type=int, default=20)
    parser.add_argument("--hyperband_eta", type=int, default=3)
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
    record["trial_name"] = trial_name
    record["unit"] = unit_name or f"{unit_label}{run_index}"
    if record_extra:
        record.update(record_extra)
    return record


def main() -> None:
    args = parse_args()

    if args.worker_context is not None:
        run_campaign_worker(args.worker_context, train_trial=train_trial, known_parameters=set())
        return

    cells = resolve_cells(args.task, args.models, args.datasets)
    if not cells:
        raise SystemExit("No runnable cells selected.")

    monitor = monitor_for(args.task)
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
        f"patience {args.patience or SHARED_PATIENCE} | device {args.device or 'auto'} | "
        f"project {args.wandb_project}"
    )

    if args.validate_only:
        for cell in cells:
            sweep = build_sweep_config(
                cell,
                monitor=monitor,
                n_trials=args.n_trials,
                epochs=epochs,
                hyperband_min_iter=args.hyperband_min_iter,
                hyperband_eta=args.hyperband_eta,
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
            hyperband_min_iter=args.hyperband_min_iter,
            hyperband_eta=args.hyperband_eta,
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
                summarize=lambda recs, task, path: summarize_cell(recs, task, path),
            )
            records.append(best)

        summary_path = args.output_dir / cell.token / "tuning_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = summarize_cell(records, args.task, summary_path)
        print(f"{cell.token}: {json.dumps(summary, indent=2)[:600]}")


if __name__ == "__main__":
    main()
