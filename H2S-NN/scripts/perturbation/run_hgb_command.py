"""Card 7 amendment 4: a GCN or HAN perturbation run at HGB's published command (`baseline_configs.HGB_COMMANDS`).

The loop of `scripts/run_baseline_published.py` with the parameter set taken from the mapping instead of
`configs/baselines/` (adding GCN/HAN files there would turn the main-table cells into published ones): the same units
(`build_units`, seeds base_seed .. base_seed + num_runs - 1), the same `fit_cell` (source "published", so the HGB
epochs / patience in the parameter set are the budget, as for every published cell), the same summary
(`summarize_cell`), written to `<output_dir>/<model>_node_classification_<dataset>/hgb_command_summary.json`. Each
record carries `configuration: hgb_command` and the command's source. The bundle reads the default data root
`data`, so the run starts in the perturbed graph's folder (see `manifest.py`).

    python scripts/perturbation/run_hgb_command.py --model gcn --dataset acm --num_runs 5 --base_seed 1 \
        --output_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_configs as bc  # noqa: E402

TASK = "node_classification"


def run(model: str, dataset: str, *, num_runs: int, base_seed: int, output_dir: Path, device: str | None) -> dict:
    import torch

    from src.baselines.campaign import Cell, build_raw_base, build_units, fit_cell, summarize_cell
    from src.utils.run_reports import write_run_record

    entry = bc.HGB_COMMANDS[(model, dataset)]
    params = bc.hgb_command_params(model, dataset)
    cell = Cell(model=model, task=TASK, dataset=dataset)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    records = []
    for unit in build_units(TASK, base_seed=base_seed, num_runs=num_runs):
        record = fit_cell(build_raw_base(cell, unit), dict(params), source="published", device=torch_device)
        record.update({"unit": unit.token, "configuration": "hgb_command", "hgb_command": entry["command"],
                       "configuration_source": entry["source"], "sweep_params": params})
        write_run_record(output_dir / cell.token / f"hgb_command_{unit.token}.json", record)
        records.append(record)
        print(f"  {cell.token} {unit.token}: done")
    return summarize_cell(records, TASK, output_dir / cell.token / "hgb_command_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted({m for m, _ in bc.HGB_COMMANDS}))
    parser.add_argument("--dataset", required=True, choices=sorted({d for _, d in bc.HGB_COMMANDS}))
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=1)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    summary = run(args.model, args.dataset, num_runs=args.num_runs, base_seed=args.base_seed,
                  output_dir=args.output_dir, device=args.device)
    print(json.dumps(summary.get("metrics", {}), indent=1))


if __name__ == "__main__":
    main()
