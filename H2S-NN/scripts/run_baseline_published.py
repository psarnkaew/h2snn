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
from src.baselines.registry import (
    DATASETS,
    get_adapter,
    models_for_task,
    unsupported_reason,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, required=True, choices=sorted(DATASETS))
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/baselines"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
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
    all_units = build_units(args.task, base_seed=args.base_seed, num_runs=args.num_runs)
    stop = len(all_units) if args.unit_count is None else args.unit_start + args.unit_count
    units = all_units[args.unit_start : stop]

    print(f"{len(cells)} published cell(s) x {len(units)} unit(s) on {device}")

    for cell in cells:
        params = get_adapter(cell.model).published_params(args.task, cell.dataset)
        assert params is not None  # guaranteed by published_cells
        records = []
        for unit in units:
            record = fit_cell(
                build_raw_base(cell, unit),
                dict(params.values),
                source="published",
                device=device,
            )
            record["unit"] = unit.token
            records.append(record)
            print(f"  {cell.token} {unit.token}: done")

        summary_path = args.output_dir / cell.token / "published_summary.json"
        summary = summarize_cell(records, args.task, summary_path)
        print(f"{cell.token}: {json.dumps(summary['metrics'], indent=2)}")


if __name__ == "__main__":
    main()
