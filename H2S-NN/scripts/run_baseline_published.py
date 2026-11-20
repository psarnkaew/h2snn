"""Run baseline cells at the authors' published hyperparameters -- no sweep.

A cell qualifies only if its authors published dataset-specific settings for that exact
dataset; :func:`src.baselines.published.load_published_params` returns ``None`` otherwise
and the cell belongs to ``scripts/run_baseline_tuning.py`` instead. Use ``--list`` to see
which is which.

Everything else matches the tuned path: the same units (5 seeds, or 5x10 folds for graph
classification), the same bundles, the same shared heads, and the same summary format.

Examples::

    python scripts/run_baseline_published.py --task node_classification --list
    python scripts/run_baseline_published.py --task link_prediction \\
        --models pshgcn --datasets lastfm amazon
    python scripts/run_baseline_published.py --task graph_classification \\
        --models csnn --datasets enzymes --unit_start 0 --unit_count 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    build_units,
    fit_cell,
    summarize_cell,
)
from src.baselines.cell_status import filled_cells
from src.baselines.registry import (
    DATASETS,
    get_adapter,
    models_for_task,
    unsupported_reason,
)
from src.utils.wandb_campaign import CampaignUnit


def seed_units(seeds: list[int]) -> list[CampaignUnit]:
    """One unit per seed (card 8 C5: `--final_seeds 0 1 2 3 4`)."""
    return [CampaignUnit(run_index=index, num_runs=len(seeds), seed=int(seed), label="run")
            for index, seed in enumerate(seeds)]


def run_published_seed(cell: Cell, unit: CampaignUnit, values: dict, *, device, args: argparse.Namespace) -> dict:
    """One seed of a published cell in its own W&B run (card 8 C7): per-epoch report, summary at the end; the record
    is written to `<output_dir>/<token>/published_seed<s>.json` (read by `scripts/tsubame/collect.py`)."""
    import time

    from src.baselines.campaign import monitor_for
    from src.baselines.progress import WandbEpochReport
    from src.baselines.wandb_runs import finish, open_run, write_summary
    from src.utils.run_reports import write_run_record

    started = time.time()
    raw_base = build_raw_base(cell, unit)
    config = {"raw_base": raw_base, "params": values, "source": "published"}
    run = open_run(project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode, kind="published",
                   token=cell.token, seed=unit.seed, config=config)
    monitor = monitor_for(cell.task)
    try:
        record = fit_cell(raw_base, values, source="published", device=device,
                          report=WandbEpochReport(monitor) if run is not None else None)
        record.update({"unit": unit.token, "published_seed": int(unit.seed), "monitor": monitor})
        write_summary(run, record, config=config, started=started)
    finally:
        finish(run)
    write_run_record(args.output_dir / cell.token / f"published_seed{unit.seed}.json", record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, required=True, choices=sorted(DATASETS))
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/baselines"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_runs", type=int, default=5)
    # card 8 C1: seeds 0-4 (spec §6); the perturbation manifest passes --base_seed 1 explicitly (unchanged)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--final_seeds", type=int, nargs="+", default=None,
                        help="card 8 C5: the seeds to run (one unit per seed), instead of --num_runs/--base_seed")
    # card 8 C7: one W&B run per seed (group <token>_published, tag published); W&B never fails a run
    parser.add_argument("--wandb_project", type=str, default="h2snn_baseline")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--state_dir", type=Path, default=None,
                        help="card 8 C5: where `published_state.json` records the finished seeds of each cell")
    parser.add_argument("--unit_start", type=int, default=0)
    parser.add_argument("--unit_count", type=int, default=None)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Report which cells carry published parameters, then exit.",
    )
    return parser.parse_args()


def published_cells(task: str, models: list[str], datasets: list[str]) -> list[Cell]:
    """Cells whose authors published settings for this exact dataset."""

    known_models = [spec.name for spec in models_for_task(task)]
    chosen_models = known_models if models == ["all"] else models
    chosen_datasets = list(DATASETS[task]) if datasets == ["all"] else datasets

    cells: list[Cell] = []
    for model in chosen_models:
        for dataset in chosen_datasets:
            if unsupported_reason(model, task, dataset):
                continue
            if (task, dataset, model) in filled_cells():  # card 8 C0: a filled cell is never run
                continue
            if get_adapter(model).published_params(task, dataset) is None:
                continue
            cells.append(Cell(model=model, task=task, dataset=dataset))
    return cells


def main() -> None:
    args = parse_args()

    if args.list:
        for task in ([args.task] if args.task else sorted(DATASETS)):
            cells = published_cells(task, ["all"], ["all"])
            print(f"{task}: {len(cells)} published cell(s)")
            for cell in cells:
                print(f"  {cell.model} / {cell.dataset}")
        return

    cells = published_cells(args.task, args.models, args.datasets)
    if not cells:
        raise SystemExit(
            "No published cells selected. These authors published dataset-specific "
            "settings only for some cells; run the rest through "
            "scripts/run_baseline_tuning.py."
        )

    import torch

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.final_seeds is not None:
        units = seed_units(list(args.final_seeds))
    else:
        all_units = build_units(args.task, base_seed=args.base_seed, num_runs=args.num_runs)
        stop = len(all_units) if args.unit_count is None else args.unit_start + args.unit_count
        units = all_units[args.unit_start : stop]

    print(f"{len(cells)} published cell(s) x {len(units)} unit(s) on {device}")

    for cell in cells:
        params = get_adapter(cell.model).published_params(args.task, cell.dataset)
        assert params is not None  # guaranteed by published_cells
        records = []
        for unit in units:
            records.append(run_published_seed(cell, unit, dict(params.values), device=device, args=args))
            print(f"  {cell.token} {unit.token}: done")

        summary_path = args.output_dir / cell.token / "published_summary.json"
        summary = summarize_cell(records, args.task, summary_path)
        if args.state_dir is not None:
            state = args.state_dir / cell.token / "published_state.json"
            state.parent.mkdir(parents=True, exist_ok=True)
            state.write_text(json.dumps({"seeds": [u.seed for u in units], "summary": str(summary_path)}, indent=2))
        print(f"{cell.token}: {json.dumps(summary['metrics'], indent=2)}")


if __name__ == "__main__":
    main()
