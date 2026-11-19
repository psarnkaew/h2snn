"""Card 7 V3: the run manifest of the perturbation study (ablation plan v2, Q4; R3's perturbation axis).

Rows, seeds 1-5, never retuned:
- every targeted and random graph (`make_graphs.py`'s `graph.json`): H2S-NN `full` and `no_path_cells`, and the
  baselines GCN, HAN, SeHGNN, Hetero2Net, HetSheaf;
- the targeted graphs only: H2S-NN S2 (R3).
Configurations: each method's configuration tuned on the original graph, read from the main-table output
(H2S-NN: `full`'s tuned file per dataset, `--full_config`; `no_path_cells` and S2 are its configurations (a) by
`scripts/ablation/arms.py`; a baseline: its seed-0 sweep `<baseline_output>/<model>_node_classification_<dataset>/
tuning_trials_run0.json`). A baseline without a tuned configuration uses its published configuration only when
`--published_config` is given (listed in `missing.json` either way); without either it has no row (a finding).
Launchers (existing fixed-configuration paths, not edited):
- H2S-NN: `scripts/run_h2snn_runs.py --config <file with data.root = the graph's data root> --base_seed 1
  --num_runs 5`;
- tuned baseline: `scripts/run_baseline_tuning.py --final_seeds 1 2 3 4 5` (card 2 D6.8: the best trial of the
  seed-0 sweep, whose trials file is copied into the perturbation output folder first);
- published baseline: `scripts/run_baseline_published.py --num_runs 5 --base_seed 1` (seeds 1-5).
Both baseline launchers build their bundles from the default data root `data`, so their rows run from the graph's
folder (`cd <...>/<kind>_h<level>`), where `data/` is the perturbed graph: only the data directory changes.
Unreached levels (`graph.json` `reached: false`) get no rows unless `--include_unreached` (listed in the summary).

    python scripts/perturbation/manifest.py --graphs data/NC --full_config acm=<yaml> dblp=<yaml> \
        --baseline_output experiments/baselines --out_dir <dir> [--published_config] [--include_unreached]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "ablation"))

import arms  # noqa: E402

H2SNN_ROWS: tuple[str, ...] = ("full", "no_path_cells")
H2SNN_TARGETED_ONLY: tuple[str, ...] = ("S2",)
BASELINES: tuple[str, ...] = ("gcn", "han", "sehgnn", "hetero2net", "hetsheaf")
SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
TASK = "node_classification"
FIELDS = ("graph_kind", "dataset", "level", "method", "configuration", "seeds", "graph_root", "cwd", "exp_name",
          "command")


def find_graphs(graphs_dir: Path) -> list[dict[str, Any]]:
    """Every `graph.json` under `graphs_dir` (`<...>/<DATASET>_pert/<kind>_h<level>/graph.json`), sorted."""
    records = []
    for path in sorted(Path(graphs_dir).glob("*_pert/*_h*/graph.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["folder"] = str(path.parent.resolve())
        # the graph's data root from where the graph is now (graph.json's own path may be another machine's)
        record["data_root"] = str(path.parent.resolve() / "data")
        records.append(record)
    return records


def h2snn_config(full_raw: dict[str, Any], row: str, graph: dict[str, Any], out: Path) -> Path:
    raw = arms.apply_change(full_raw, row)
    raw = copy.deepcopy(raw)
    raw["data"] = {**raw.get("data", {}), "root": graph["data_root"]}
    name = f"{graph['dataset']}__pert_{graph['kind']}_h{graph['level']:.2f}_{row}"
    raw["exp_name"] = name
    from src.configs.config import TrainConfig

    TrainConfig.from_dict(copy.deepcopy(raw))
    path = out / graph["dataset"] / f"{graph['kind']}_h{graph['level']:.2f}" / f"{row}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def baseline_configuration(model: str, dataset: str, baseline_output: Path) -> tuple[str | None, Path | None]:
    """("tuned", trials file) when the main-table sweep exists, ("published", None) when the adapter has
    published parameters for the cell, else (None, None)."""
    trials = baseline_output / f"{model}_{TASK}_{dataset}" / "tuning_trials_run0.json"
    if trials.exists():
        return "tuned", trials
    from src.baselines.registry import get_adapter

    if get_adapter(model).published_params(TASK, dataset) is not None:
        return "published", None
    return None, None


def build(graphs_dir: Path, full_configs: dict[str, Path], baseline_output: Path, out_dir: Path,
          published_config: bool, include_unreached: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: dict[str, Any] = {"baselines_without_configuration": [], "published_configuration_used": [],
                               "unreached_graphs_without_rows": []}
    runs_out = out_dir / "runs"
    script = PROJECT_ROOT / "scripts"
    for graph in find_graphs(graphs_dir):
        dataset, kind, level = graph["dataset"], graph["kind"], float(graph["level"])
        tag = f"{kind}_h{level:.2f}"
        if not graph["reached"] and not include_unreached:
            missing["unreached_graphs_without_rows"].append(f"{dataset}/{tag}")
            continue
        if dataset not in full_configs:
            raise ValueError(f"no --full_config for {dataset}")
        full_raw = yaml.safe_load(full_configs[dataset].read_text(encoding="utf-8"))
        h2snn_rows = H2SNN_ROWS + (H2SNN_TARGETED_ONLY if kind == "targeted" else ())
        base = {"graph_kind": kind, "dataset": dataset, "level": f"{level:.2f}", "graph_root": graph["data_root"],
                "seeds": " ".join(map(str, SEEDS))}
        for row in h2snn_rows:
            config = h2snn_config(full_raw, row, graph, out_dir / "configs")
            exp = f"{dataset}__pert_{tag}_{row}"
            rows.append({**base, "method": f"h2snn_{row}", "configuration": "a" if row != "full" else "tuned",
                         "cwd": str(PROJECT_ROOT), "exp_name": exp,
                         "command": (f"python {script / 'run_h2snn_runs.py'} --config {config} --base_seed {SEEDS[0]} "
                                     f"--num_runs {len(SEEDS)} --exp_name {exp} "
                                     f"--output_dir {runs_out / dataset / tag}")})
        for model in BASELINES:
            source, trials = baseline_configuration(model, dataset, baseline_output)
            out = runs_out / dataset / tag / "baselines"
            token = f"{model}_{TASK}_{dataset}"
            if source == "tuned":
                command = (f"mkdir -p {out / token} && cp {trials} {out / token}/ && "
                           f"python {script / 'run_baseline_tuning.py'} --task {TASK} --models {model} "
                           f"--datasets {dataset} --final_seeds {' '.join(map(str, SEEDS))} --output_dir {out}")
            elif source is None or not published_config:
                if f"{model}/{dataset}" not in missing["baselines_without_configuration"]:
                    missing["baselines_without_configuration"].append(f"{model}/{dataset}")
                continue
            else:
                if f"{model}/{dataset}" not in missing["published_configuration_used"]:
                    missing["published_configuration_used"].append(f"{model}/{dataset}")
                command = (f"python {script / 'run_baseline_published.py'} --task {TASK} --models {model} "
                           f"--datasets {dataset} --num_runs {len(SEEDS)} --base_seed {SEEDS[0]} --output_dir {out}")
            rows.append({**base, "method": model, "configuration": source, "cwd": graph["folder"],
                         "exp_name": token, "command": command})
    return rows, missing


def write(rows: list[dict[str, Any]], missing: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.tsv"
    lines = ["\t".join(FIELDS)] + ["\t".join(str(r[f]) for f in FIELDS) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["method"]] = counts.get(r["method"], 0) + 1
    (out_dir / "missing.json").write_text(json.dumps({**missing, "rows": len(rows), "rows_per_method": counts,
                                                      "runs": len(rows) * len(SEEDS)}, indent=1), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=Path, required=True, help="the data root's NC folder (holds *_pert/)")
    parser.add_argument("--full_config", nargs="+", required=True, help="dataset=<full's tuned yaml> ...")
    parser.add_argument("--baseline_output", type=Path, default=Path("experiments/baselines"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--published_config", action="store_true",
                        help="baselines without a tuned configuration use their published one")
    parser.add_argument("--include_unreached", action="store_true")
    args = parser.parse_args()
    full = {k: Path(v) for k, v in (item.split("=", 1) for item in args.full_config)}
    rows, missing = build(args.graphs, full, args.baseline_output, args.out_dir, args.published_config,
                          args.include_unreached)
    path = write(rows, missing, args.out_dir)
    print(f"{len(rows)} rows -> {path}; {json.dumps(missing)}")


if __name__ == "__main__":
    main()
