"""Card 7 V3 + amendments 1, 3, 4, 7 (author 2026-11-19): the run manifest of the perturbation study (ablation plan
v2, Q4; R3's perturbation axis).

Graphs: ACM and DBLP (IMDB is not used: its graphs may stay on disk, they get no rows), the levels of
`make_graphs.LEVELS` (0.20-0.60; other folders on disk, e.g. Phase A's 0.80, get no rows), targeted and random,
reached only (an unreached graph gets rows only with `--include_unreached`). Every excluded graph is listed in
`missing.json`.
Rows, seeds 1-5, never retuned:
- every graph: H2S-NN `full` and `no_path_cells`, and the baselines GCN, HAN, SeHGNN, Hetero2Net, HetSheaf, CSNN;
- the targeted graphs only: H2S-NN S2 (R3).
Configurations:
- H2S-NN: `full`'s tuned file per dataset (`--full_config`); `no_path_cells` and S2 are its configurations (a) by
  `scripts/ablation/arms.py`; launcher `scripts/run_h2snn_runs.py --config <file with data.root = the graph's data
  root> --base_seed 1 --num_runs 5`;
- baselines, fixed per method (`baseline_configs.CONFIG_SOURCE`):
  - GCN, HAN: HGB's published command mapped onto our keys, `scripts/perturbation/run_hgb_command.py`;
  - SeHGNN, Hetero2Net: published, `scripts/run_baseline_published.py --num_runs 5 --base_seed 1`;
  - HetSheaf, CSNN: the main-table seed-0 sweep `<baseline_output>/<model>_node_classification_<dataset>/
    tuning_trials_run0.json` (copied into the run folder), `scripts/run_baseline_tuning.py --final_seeds 1 2 3 4 5`;
    without it the row stays in the manifest with configuration "needs tuned config" and no command.
  A published / HGB-command source that the adapter cannot serve raises (no silent fallback).
  The baseline launchers build their bundles from the default data root `data`, so their rows run from the graph's
  folder (`cwd` = `<...>/<kind>_h<level>`), where `data/` is the perturbed graph.
Original-graph reference point (amendment 8, author 2026-11-19): the six baselines on the original ACM and DBLP
graphs (`<...>/<DATASET>_pert/original/`, written by `make_graphs.py --original`), seeds 1-5, through the same checked
launcher and with the same configuration source as their perturbation rows (HetSheaf / CSNN "needs tuned config"
without a sweep): 6 x 2 x 5 = 60 runs, `graph_kind` `original`, empty `level`. The H2S-NN reference (`full`,
`no_path_cells`, S2) has no rows: the figure reuses the ablation configuration (a) runs at seeds 1-5 on the original
graphs (`figure.py --h2snn_reference_table`). A dataset without its original folder is listed under
`original_graphs_missing`. Main-table results are never used as the reference.
Fingerprint check (amendment 3): every command runs through `scripts/perturbation/run_checked.py` with the graph's
`graph.json` and the generator's `edge_fingerprint` (column `fingerprint`); the run fails on a mismatch and logs
every check to `<run folder>/fingerprint_checks.jsonl`. Column `summary`: the summary file the row writes.

    python scripts/perturbation/manifest.py --graphs data/NC --full_config acm=<yaml> dblp=<yaml> \
        --baseline_output experiments/baselines --out_dir <dir> [--include_unreached]
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "perturbation"))

import arms  # noqa: E402
import baseline_configs as bc  # noqa: E402

H2SNN_ROWS: tuple[str, ...] = ("full", "no_path_cells")
H2SNN_TARGETED_ONLY: tuple[str, ...] = ("S2",)
BASELINES: tuple[str, ...] = ("gcn", "han", "sehgnn", "hetero2net", "hetsheaf", "csnn")
DATASETS: tuple[str, ...] = ("acm", "dblp")
LEVELS: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60)      # = make_graphs.LEVELS (checked by the tests)
SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
TASK = "node_classification"
FIELDS = ("graph_kind", "dataset", "level", "h", "h2_index", "method", "configuration", "seeds", "graph_root", "cwd",
          "exp_name", "fingerprint", "summary", "command")
CHECKER = PROJECT_ROOT / "scripts" / "perturbation" / "run_checked.py"


ORIGINAL = "original"
H2SNN_REFERENCE = "ablation configuration (a), seeds 1-5 (figure.py --h2snn_reference_table); no manifest rows"


def find_graphs(graphs_dir: Path, pattern: str = "*_pert/*_h*/graph.json") -> list[dict[str, Any]]:
    """Every `graph.json` under `graphs_dir` (`<...>/<DATASET>_pert/<kind>_h<level>/graph.json`; the original graphs
    with `pattern` `*_pert/original/graph.json`), sorted."""
    records = []
    for path in sorted(Path(graphs_dir).glob(pattern)):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["folder"] = str(path.parent.resolve())
        record["graph_json"] = str(path.resolve())
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


def baseline_configuration(model: str, dataset: str, baseline_output: Path) -> tuple[str, Path | None]:
    """(source, trials file or None): "hgb_command", "published", "tuned", or `bc.NEEDS_TUNED` (a tuned method whose
    seed-0 sweep is not there yet)."""
    source = bc.source_for(model)
    if source == "hgb_command":
        bc.hgb_command_params(model, dataset)                              # raises when not mapped
        return source, None
    if source == "published":
        from src.baselines.registry import get_adapter

        if get_adapter(model).published_params(TASK, dataset) is None:
            raise ValueError(f"{model}/{dataset}: no published configuration")
        return source, None
    trials = baseline_output / f"{model}_{TASK}_{dataset}" / "tuning_trials_run0.json"
    return ("tuned", trials) if trials.exists() else (bc.NEEDS_TUNED, None)


def checked(graph: dict[str, Any], log: Path, launcher: Path, args: str) -> str:
    return (f"python {CHECKER} --graph_json {graph['graph_json']} --expected_fingerprint {graph['edge_fingerprint']} "
            f"--log {log} -- {launcher} {args}")


def baseline_rows(graph: dict[str, Any], base: dict[str, Any], run_dir: Path, baseline_output: Path,
                  missing: dict[str, Any]) -> list[dict[str, Any]]:
    """The six baselines' rows on one graph (a perturbed graph or the original one): the configuration source fixed per
    method, every command through the fingerprint check, run from the graph's folder."""
    rows = []
    dataset = graph["dataset"]
    script = PROJECT_ROOT / "scripts"
    seeds = " ".join(map(str, SEEDS))
    for model in BASELINES:
        source, trials = baseline_configuration(model, dataset, baseline_output)
        missing["configuration_per_baseline"][f"{model}/{dataset}"] = source
        out = run_dir / "baselines"
        token = f"{model}_{TASK}_{dataset}"
        log = out / token / "fingerprint_checks.jsonl"
        if source == bc.NEEDS_TUNED:
            if f"{model}/{dataset}" not in missing["needs_tuned_config"]:
                missing["needs_tuned_config"].append(f"{model}/{dataset}")
            command, summary = "", ""
        elif source == "tuned":
            command = (f"mkdir -p {out / token} && cp {trials} {out / token}/ && " + checked(
                graph, log, script / "run_baseline_tuning.py",
                f"--task {TASK} --models {model} --datasets {dataset} --final_seeds {seeds} --output_dir {out}"))
            summary = str(out / token / "final_seeds_summary.json")
        elif source == "published":
            command = checked(graph, log, script / "run_baseline_published.py",
                              f"--task {TASK} --models {model} --datasets {dataset} --num_runs {len(SEEDS)} "
                              f"--base_seed {SEEDS[0]} --output_dir {out}")
            summary = str(out / token / "published_summary.json")
        else:
            command = checked(graph, log, script / "perturbation" / "run_hgb_command.py",
                              f"--model {model} --dataset {dataset} --num_runs {len(SEEDS)} "
                              f"--base_seed {SEEDS[0]} --output_dir {out}")
            summary = str(out / token / "hgb_command_summary.json")
        rows.append({**base, "method": model, "configuration": source, "cwd": graph["folder"],
                     "exp_name": token, "summary": summary, "command": command})
    return rows


def build(graphs_dir: Path, full_configs: dict[str, Path], baseline_output: Path, out_dir: Path,
          include_unreached: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: dict[str, Any] = {"needs_tuned_config": [], "configuration_per_baseline": {},
                               "unreached_graphs_without_rows": [], "excluded_graphs": [],
                               "original_graphs_missing": [], "h2snn_reference": H2SNN_REFERENCE}
    runs_out = out_dir / "runs"
    script = PROJECT_ROOT / "scripts"
    seeds = " ".join(map(str, SEEDS))
    for graph in find_graphs(graphs_dir):
        dataset, kind, level = graph["dataset"], graph["kind"], round(float(graph["level"]), 2)
        tag = f"{kind}_h{level:.2f}"
        if dataset not in DATASETS or level not in LEVELS:
            missing["excluded_graphs"].append(f"{dataset}/{tag}")
            continue
        if not graph["reached"] and not include_unreached:
            missing["unreached_graphs_without_rows"].append(f"{dataset}/{tag}")
            continue
        if not graph.get("edge_fingerprint"):
            raise ValueError(f"{graph['folder']}: graph.json has no edge_fingerprint (run make_graphs.py --backfill)")
        if dataset not in full_configs:
            raise ValueError(f"no --full_config for {dataset}")
        full_raw = yaml.safe_load(full_configs[dataset].read_text(encoding="utf-8"))
        h2snn_rows = H2SNN_ROWS + (H2SNN_TARGETED_ONLY if kind == "targeted" else ())
        run_dir = runs_out / dataset / tag
        base = {"graph_kind": kind, "dataset": dataset, "level": f"{level:.2f}", "h": f"{graph['h']:.4f}",
                "h2_index": f"{graph['h2_index']:.4f}", "graph_root": graph["data_root"], "seeds": seeds,
                "fingerprint": graph["edge_fingerprint"]}
        for row in h2snn_rows:
            config = h2snn_config(full_raw, row, graph, out_dir / "configs")
            exp = f"{dataset}__pert_{tag}_{row}"
            args = (f"--config {config} --base_seed {SEEDS[0]} --num_runs {len(SEEDS)} --exp_name {exp} "
                    f"--output_dir {run_dir}")
            rows.append({**base, "method": f"h2snn_{row}", "configuration": "a" if row != "full" else "tuned",
                         "cwd": str(PROJECT_ROOT), "exp_name": exp,
                         "summary": str(run_dir / exp / "runs_summary.json"),
                         "command": checked(graph, run_dir / exp / "fingerprint_checks.jsonl",
                                            script / "run_h2snn_runs.py", args)})
        rows += baseline_rows(graph, base, run_dir, baseline_output, missing)
    originals = {g["dataset"]: g for g in find_graphs(graphs_dir, f"*_pert/{ORIGINAL}/graph.json")}
    for dataset in DATASETS:
        graph = originals.get(dataset)
        if graph is None:
            missing["original_graphs_missing"].append(dataset)
            continue
        if graph.get("kind") != ORIGINAL or not graph.get("edge_fingerprint"):
            raise ValueError(f"{graph['folder']}: not an original graph with an edge fingerprint "
                             "(make_graphs.py --original)")
        base = {"graph_kind": ORIGINAL, "dataset": dataset, "level": "", "h": f"{graph['h']:.4f}",
                "h2_index": f"{graph['h2_index']:.4f}", "graph_root": graph["data_root"], "seeds": seeds,
                "fingerprint": graph["edge_fingerprint"]}
        rows += baseline_rows(graph, base, runs_out / dataset / ORIGINAL, baseline_output, missing)
    return rows, missing


def write(rows: list[dict[str, Any]], missing: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.tsv"
    lines = ["\t".join(FIELDS)] + ["\t".join(str(r[f]) for f in FIELDS) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["method"]] = counts.get(r["method"], 0) + 1
    runnable = [r for r in rows if r["command"]]
    (out_dir / "missing.json").write_text(json.dumps({**missing, "rows": len(rows), "rows_per_method": counts,
                                                      "runs": len(rows) * len(SEEDS), "runnable_rows": len(runnable),
                                                      "reference_rows": sum(r["graph_kind"] == ORIGINAL for r in rows),
                                                      "runnable_runs": len(runnable) * len(SEEDS)}, indent=1),
                                          encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=Path, required=True, help="the data root's NC folder (holds *_pert/)")
    parser.add_argument("--full_config", nargs="+", required=True, help="dataset=<full's tuned yaml> ...")
    parser.add_argument("--baseline_output", type=Path, default=Path("experiments/baselines"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--include_unreached", action="store_true")
    args = parser.parse_args()
    full = {k: Path(v) for k, v in (item.split("=", 1) for item in args.full_config)}
    rows, missing = build(args.graphs, full, args.baseline_output, args.out_dir, args.include_unreached)
    path = write(rows, missing, args.out_dir)
    print(f"{len(rows)} rows -> {path}; {json.dumps(missing)}")


if __name__ == "__main__":
    main()
