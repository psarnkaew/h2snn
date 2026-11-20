"""Card 9 W5 (pre-registration v3 §4.2 "Definitions" and "Expected"; spec 1.29 §6 "Timing and analysis-only
settings"): the drop, the gap and the rules E1, E2 of the perturbation study, per seed. No training.

Per-seed test Macro-F1 (seeds 1-5):
- H2S-NN rows (`h2snn_<row>`): the run records `best_test_metrics_run<i>.json` (key `seed`) of the row's runs,
  `<output_dir>/<exp>_run<i>/` beside the summary folder `<output_dir>/<exp>/runs_summary.json`
  (`scripts/run_h2snn_runs.py`: `exp_dir = output_dir / <exp>_run<i>`);
- baseline rows: the summary file of the row (`hgb_command_summary.json`, `published_summary.json` or
  `final_seeds_summary.json`, all written by `src/baselines/campaign.py: summarize_cell`) holds `units`, one record
  per seed with `test_macro_f1`; the seed is the record's `final_seed` (tuned: `run_baseline_tuning.py
  --final_seeds`), else unit `run<i>` = the i-th of the row's seeds (`build_units`: seed = base_seed + i). A summary
  without per-seed records raises `NoPerSeedValues` (card 9 stop condition; never read as a mean);
- the H2S-NN original-graph reference: the ablation configuration (a) runs, seeds 1-5 (the long table of
  `scripts/ablation/stats.py collect`), rows `full` and `no_path_cells`.

Definitions (fixed by v3):
- `h_max`: the highest targeted level reached per dataset (`graph.json` of `<DATASET>_pert/targeted_h<level>/`,
  `reached`), among the study's levels;
- `drop_m(s) = F1_m,random(h_max, s) - F1_m,targeted(h_max, s)`;
- `gap(G, s) = F1_full(G, s) - F1_no_path_cells(G, s)`.
Rules (each per-seed statement over seeds 1-5; `None` when a seed is missing on either side):
- **E1** per baseline m in {GCN, HAN, SeHGNN}, per dataset: `mean(drop_full) < mean(drop_m)` and
  `drop_full(s) < drop_m(s)` in >= 4 of 5 seeds; E1 for m holds when it holds on both ACM and DBLP (false as soon as
  one dataset is false). Hetero2Net, CSNN and HetSheaf: the same numbers, reported, no verdict;
- **E2** per dataset: `gap(targeted h_max, s) > gap(original, s)` in >= 4 of 5 seeds; holds on both ACM and DBLP.
A method with no runnable manifest row on a dataset (HetSheaf, CSNN without a tuned configuration) is `absent` there;
it never raises.

    python scripts/perturbation/rules.py --manifest <pert>/manifest.tsv --graphs data/NC \
        --h2snn_reference_table <stats collect csv> --out_dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

DATASETS: tuple[str, ...] = ("acm", "dblp")
LEVELS: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60)       # = manifest.LEVELS
SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
SEEDS_NEEDED = 4
METRIC = "test_macro_f1"
FULL, NO_CELLS = "h2snn_full", "h2snn_no_path_cells"
E1_BASELINES: tuple[str, ...] = ("gcn", "han", "sehgnn")
REPORTED_BASELINES: tuple[str, ...] = ("hetero2net", "csnn", "hetsheaf")
ORIGINAL = "original"


class NoPerSeedValues(ValueError):
    """A summary holds no per-seed test values (only a mean and std): card 9 stop condition."""


# ------------------------------------------------------------------------------------------------ inputs
def h_max(graphs_dir: Path) -> dict[str, float]:
    """The highest reached targeted level per dataset, among `LEVELS`."""
    out: dict[str, float] = {}
    for path in sorted(Path(graphs_dir).glob("*_pert/targeted_h*/graph.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        level = round(float(record["level"]), 2)
        if record.get("kind", "targeted") != "targeted" or not record.get("reached") or level not in LEVELS:
            continue
        dataset = str(record["dataset"])
        out[dataset] = max(out.get(dataset, level), level)
    return out


def _unit_seed(record: dict[str, Any], index: int, seeds: list[int]) -> int:
    if "final_seed" in record:
        return int(record["final_seed"])
    unit = str(record.get("unit", f"run{index}"))
    match = re.fullmatch(r"(?:run|final_seed)(\d+)", unit)
    if match is None:
        raise ValueError(f"unit {unit!r}: not run<i> or final_seed<s>")
    return int(match[1]) if unit.startswith("final_seed") else seeds[int(match[1])]


def per_seed(row: dict[str, str]) -> dict[int, float] | None:
    """Per-seed test Macro-F1 of one manifest row; `None` while the row has no summary yet."""
    summary = Path(row["summary"]) if row.get("summary") else None
    if summary is None or not summary.exists():
        return None
    seeds = [int(s) for s in str(row["seeds"]).split()]
    out: dict[int, float] = {}
    if row["method"].startswith("h2snn_"):
        exp = summary.parent.name
        for path in sorted(summary.parent.parent.glob(f"{exp}_run*/best_test_metrics_run*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            out[int(record["seed"])] = float(record[METRIC])
        if not out:
            raise NoPerSeedValues(f"{summary}: no run record {exp}_run<i>/best_test_metrics_run<i>.json")
        return out
    payload = json.loads(summary.read_text(encoding="utf-8"))
    units = payload.get("units") if isinstance(payload, dict) else None
    if not units:
        raise NoPerSeedValues(f"{summary}: no per-seed records (`units`); a mean and std are not per-seed values")
    for index, record in enumerate(units):
        if record.get(METRIC) is None:
            raise NoPerSeedValues(f"{summary}: unit {index} has no {METRIC}")
        out[_unit_seed(record, index, seeds)] = float(record[METRIC])
    return out


def read_manifest(manifest: Path) -> tuple[dict[tuple[str, str, float | None, str], dict[int, float]],
                                           dict[tuple[str, str], bool]]:
    """(values by (dataset, kind, level, method), runnable[(dataset, method)]): per-seed values of every row with a
    summary; whether the method has a runnable row on the dataset."""
    values: dict[tuple[str, str, float | None, str], dict[int, float]] = {}
    runnable: dict[tuple[str, str], bool] = {}
    with Path(manifest).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key_rm = (row["dataset"], row["method"])
            runnable[key_rm] = runnable.get(key_rm, False) or bool(row.get("command"))
            if not row.get("command"):
                continue
            level = None if row["graph_kind"] == ORIGINAL else round(float(row["level"]), 2)
            got = per_seed(row)
            if got is not None:
                values[(row["dataset"], row["graph_kind"], level, row["method"])] = got
    return values, runnable


def read_reference(table: Path) -> dict[tuple[str, str, float | None, str], dict[int, float]]:
    """H2S-NN `full` and `no_path_cells` on the original graphs: ablation configuration (a), seeds 1-5."""
    out: dict[tuple[str, str, float | None, str], dict[int, float]] = {}
    with Path(table).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["configuration"] != "a" or row["row"] not in ("full", "no_path_cells"):
                continue
            seed = int(row["seed"])
            if seed in SEEDS and row["dataset"] in DATASETS:
                key = (row["dataset"], ORIGINAL, None, f"h2snn_{row['row']}")
                out.setdefault(key, {})[seed] = float(row[METRIC])
    return out


# ------------------------------------------------------------------------------------------------ definitions
def difference(a: dict[int, float] | None, b: dict[int, float] | None) -> dict[int, float]:
    """`a - b` on the seeds 1-5 present in both."""
    if not a or not b:
        return {}
    return {s: a[s] - b[s] for s in SEEDS if s in a and s in b}


def drop(values: dict, dataset: str, method: str, level: float) -> dict[int, float]:
    return difference(values.get((dataset, "random", level, method)), values.get((dataset, "targeted", level, method)))


def gap(values: dict, dataset: str, kind: str, level: float | None) -> dict[int, float]:
    return difference(values.get((dataset, kind, level, FULL)), values.get((dataset, kind, level, NO_CELLS)))


def _mean(x: dict[int, float]) -> float | None:
    return sum(x.values()) / len(x) if x else None


def compare(lower: dict[int, float], higher: dict[int, float], need_mean: bool) -> dict[str, Any]:
    """Seeds where `lower(s) < higher(s)` (strict), and the rule: [mean(lower) < mean(higher) when `need_mean`] and
    the per-seed statement in >= 4 of 5 seeds; `holds` is `None` unless all 5 seeds are on both sides."""
    seeds = [s for s in SEEDS if s in lower and s in higher]
    count = sum(1 for s in seeds if lower[s] < higher[s])
    complete = len(seeds) == len(SEEDS)
    mean_ok = (_mean({s: lower[s] for s in seeds}) < _mean({s: higher[s] for s in seeds})) if seeds else None
    holds = None
    if complete:
        holds = bool(count >= SEEDS_NEEDED and (mean_ok if need_mean else True))
    return {"seeds": seeds, "count": count, "mean_lower": _mean({s: lower[s] for s in seeds}),
            "mean_higher": _mean({s: higher[s] for s in seeds}), "holds": holds}


def both(verdicts: dict[str, bool | None]) -> bool | None:
    """AND over the datasets: false as soon as one is false; `None` while one is undecided and none is false."""
    vals = [verdicts.get(d) for d in DATASETS]
    if any(v is False for v in vals):
        return False
    if any(v is None for v in vals):
        return None
    return True


def rules(values: dict, runnable: dict[tuple[str, str], bool], hmax: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {"h_max": hmax, "seeds": list(SEEDS), "seeds_needed": SEEDS_NEEDED, "drop": {}, "gap": {},
                           "E1": {}, "E1_reported": {}, "E2": {}, "absent": {}}
    for dataset in DATASETS:
        level = hmax.get(dataset)
        out["absent"][dataset] = sorted(m for m in (*E1_BASELINES, *REPORTED_BASELINES)
                                        if not runnable.get((dataset, m), False))
        out["drop"][dataset] = {m: (drop(values, dataset, m, level) if level is not None else {})
                                for m in (FULL, *E1_BASELINES, *REPORTED_BASELINES)}
        out["gap"][dataset] = {"targeted_h_max": gap(values, dataset, "targeted", level) if level is not None else {},
                               ORIGINAL: gap(values, dataset, ORIGINAL, None)}
    for method in (*E1_BASELINES, *REPORTED_BASELINES):
        per = {}
        for dataset in DATASETS:
            if method in out["absent"][dataset]:
                per[dataset] = {"status": "absent", "holds": None}
                continue
            drops = out["drop"][dataset]
            per[dataset] = {"status": "computed", **compare(drops[FULL], drops[method], need_mean=True)}
        if method in E1_BASELINES:
            out["E1"][method] = {"per_dataset": per, "holds": both({d: per[d]["holds"] for d in DATASETS})}
        else:                                                   # reported, no verdict
            for d in DATASETS:
                per[d].pop("holds", None)
            out["E1_reported"][method] = {"per_dataset": per}
    e2 = {d: compare(out["gap"][d][ORIGINAL], out["gap"][d]["targeted_h_max"], need_mean=False) for d in DATASETS}
    out["E2"] = {"per_dataset": e2, "holds": both({d: e2[d]["holds"] for d in DATASETS})}
    return out


def _f(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def markdown(result: dict[str, Any]) -> str:
    lines = ["# Perturbation rules E1, E2 (card 9 W5, pre-registration v3 §4.2)", "",
             f"h_max {result['h_max']}; seeds {result['seeds']}; a rule needs its statement in >= "
             f"{result['seeds_needed']} of 5 seeds (`—` = incomplete). drop_m(s) = F1 random(h_max) - F1 "
             "targeted(h_max); gap(G, s) = F1 full - F1 no_path_cells.", "",
             "| E1 | dataset | status | mean drop full | mean drop m | seeds drop_full < drop_m | holds |",
             "|---|---|---|---|---|---|---|"]
    for method, item in (*result["E1"].items(), *result["E1_reported"].items()):
        for dataset, r in item["per_dataset"].items():
            if r["status"] == "absent":
                lines.append(f"| {method} | {dataset} | absent | | | | |")
                continue
            lines.append(f"| {method} | {dataset} | computed | {_f(r['mean_lower'])} | {_f(r['mean_higher'])} | "
                         f"{r['count']}/{len(r['seeds'])} | {_f(r.get('holds')) if 'holds' in r else 'reported'} |")
        if "holds" in item:
            lines.append(f"| **{method}** | ACM and DBLP | | | | | **{_f(item['holds'])}** |")
    lines += ["", "| E2 | dataset | mean gap original | mean gap targeted h_max | seeds gap grows | holds |",
              "|---|---|---|---|---|---|"]
    for dataset, r in result["E2"]["per_dataset"].items():
        lines.append(f"| E2 | {dataset} | {_f(r['mean_lower'])} | {_f(r['mean_higher'])} | "
                     f"{r['count']}/{len(r['seeds'])} | {_f(r['holds'])} |")
    lines.append(f"| **E2** | ACM and DBLP | | | | **{_f(result['E2']['holds'])}** |")
    lines += ["", "Absent methods (no runnable row): " + "; ".join(f"{d}: {', '.join(m) or 'none'}"
                                                                    for d, m in result["absent"].items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True, help="the data root's NC folder (holds *_pert/)")
    parser.add_argument("--h2snn_reference_table", type=Path, required=True,
                        help="the ablation runs' long table (scripts/ablation/stats.py collect)")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    values, runnable = read_manifest(args.manifest)
    values.update(read_reference(args.h2snn_reference_table))
    result = rules(values, runnable, h_max(args.graphs))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "perturbation_rules.json").write_text(json.dumps(result, indent=1, default=str),
                                                          encoding="utf-8")
    (args.out_dir / "perturbation_rules.md").write_text(markdown(result), encoding="utf-8")
    print(f"E1 {[(m, r['holds']) for m, r in result['E1'].items()]}; E2 {result['E2']['holds']} -> {args.out_dir}")


if __name__ == "__main__":
    main()
