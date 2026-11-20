"""Card 9 W3 (pre-registration v3 §1, §3; spec 1.29 §6 "Ablation statistics"; built on card 7 V2): statistics of
the ablation table.

Input: one long table of test Macro-F1 per (dataset, row, configuration, seed), configuration (a) only and seeds 1-5
(v3 D1, D2). IMDB's value is the recorded `test_macro_f1` (the mean per-genre F1 of the IMDB protocol). `collect`
builds it from the run folders of `seed_runs.py` (`best_test_metrics_run<i>.json`, written by
`scripts/run_h2snn_runs.py`) and `full`'s seeds 1-4 of the main table (`lookup.json`).

Per comparison of `full` with a row X on a dataset (paired by seed):
- the seeds; mean +- sample std (ddof = 1) per row; the paired differences `full - X` by seed; their mean;
- `wins`: the number of seeds with `full - X > 0` (strict: a tie is not a win);
- `better` ("`full` better than X on the dataset", v3 §1): `mean_difference > 0 and wins >= 4`, computed only when all
  5 seeds are present on both sides; otherwise `None` (incomplete), never `False`.
No p-value, no multiplicity adjustment and no bootstrap interval is computed (v3 D1).

Claims (v3 §3; all configuration (a)):
- C2-cells `no_path_cells`: better on >= 3 of 4 datasets, else warning signal;
- C2 `endpoints`: >= 3 of 4, else kill rule;
- C2-w `uniform`: >= 3 of 4, else no claim that learned per-face weights help;
- C14 `instance_features`: >= 3 of 4;
- C15 S2: better on both IMDB and Freebase, else warning signal.
A claim's verdict is `None` while a dataset it needs is incomplete.
Reported only (no claim): S0 (`full` vs S0, the same numbers); the 2x2 (four mean paired differences and, per
dataset, `complementary` = mean of `full - no_path_cells` > mean of `S2 - S0`); the literature row with the rule with
roles swapped (`literature_better`: mean of `literature - full` > 0 and literature strictly higher in >= 4 of 5
seeds).

    python scripts/ablation/stats.py collect --runs <dir> [<dir> ...] --lookup <out>/<ds>/lookup.json ... --out t.csv
    python scripts/ablation/stats.py analyse --table t.csv --out_dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DATASETS: tuple[str, ...] = ("acm", "dblp", "imdb", "freebase")
METRIC = "test_macro_f1"
SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
WINS_NEEDED = 4                      # "at least 4 of the 5 seeds" (v3 §1)
CONFIGURATION = "a"                  # v3 D2: configuration (a) only


@dataclass(frozen=True)
class Claim:
    name: str
    row: str
    rule: str                       # "3of4" or "imdb_and_freebase"
    if_not: str


CLAIMS: tuple[Claim, ...] = (
    Claim("C2-cells", "no_path_cells", "3of4", "warning signal: stop and discuss"),
    Claim("C2", "endpoints", "3of4", "kill rule: no claim that the face weight depends on the middle node; stop and "
                                     "discuss before writing"),
    Claim("C2-w", "uniform", "3of4", "no claim that learned per-face weights help"),
    Claim("C14", "instance_features", "3of4", "no claim that diffusing over faces beats using them as head features"),
    Claim("C15", "S2", "imdb_and_freebase", "warning signal: drop the claim that the heterophily mechanism comes from "
                                            "the transports"),
)
REPORTED_ONLY: tuple[str, ...] = ("S0",)
LITERATURE = "literature"
TWO_BY_TWO: tuple[tuple[str, str, str], ...] = (
    ("face effect with sheaf", "full", "no_path_cells"),
    ("face effect without sheaf", "S2", "S0"),
    ("sheaf effect with faces", "full", "S2"),
    ("sheaf effect without faces", "no_path_cells", "S0"),
)


# ------------------------------------------------------------------------------------------------ statistics
def describe(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=np.float64)
    return {"n": int(arr.size), "mean": float(arr.mean()) if arr.size else math.nan,
            "std": float(arr.std(ddof=1)) if arr.size > 1 else math.nan}


def rule(mean_difference: float, wins: int) -> bool:
    """v3 §1: the mean paired difference is > 0 and the first side is strictly higher in >= 4 of the 5 seeds."""
    return bool(mean_difference > 0 and wins >= WINS_NEEDED)


def paired(table: dict, dataset: str, a: str, b: str) -> dict[str, Any] | None:
    """Paired comparison `a - b` (rows of configuration (a)) on `dataset` over seeds 1-5 present in both; `better`
    (the rule, `a` better than `b`) only when all 5 seeds are present on both sides, else `None`."""
    va = {s: v for s, v in table.get((dataset, a), {}).items() if s in SEEDS}
    vb = {s: v for s, v in table.get((dataset, b), {}).items() if s in SEEDS}
    seeds = sorted(set(va) & set(vb))
    if not seeds:
        return None
    diff = np.array([va[s] - vb[s] for s in seeds], dtype=np.float64)
    wins = int((diff > 0).sum())
    complete = set(SEEDS) <= set(va) and set(SEEDS) <= set(vb)
    mean_difference = float(diff.mean())
    return {"seeds": seeds, "a": describe(va[s] for s in seeds), "b": describe(vb[s] for s in seeds),
            "differences": diff.tolist(), "mean_difference": mean_difference, "wins": wins,
            "complete": complete, "better": rule(mean_difference, wins) if complete else None,
            "missing_seeds": {a: sorted(set(SEEDS) - set(va)), b: sorted(set(SEEDS) - set(vb))}}


def claim_result(table: dict, claim: Claim) -> dict[str, Any]:
    """`full` vs the claim's row on every dataset and the verdict (`None` while a needed dataset is incomplete)."""
    per_dataset = {d: paired(table, d, "full", claim.row) for d in DATASETS}
    better = {d: (r["better"] if r else None) for d, r in per_dataset.items()}
    needed = DATASETS if claim.rule == "3of4" else ("imdb", "freebase")
    complete = all(better[d] is not None for d in needed)
    if not complete:
        holds = None
    elif claim.rule == "3of4":
        holds = sum(1 for d in needed if better[d]) >= 3
    else:
        holds = all(better[d] for d in needed)
    return {"claim": claim.name, "row": claim.row, "configuration": CONFIGURATION, "rule": claim.rule,
            "per_dataset": per_dataset, "better": better, "datasets_better": sum(1 for v in better.values() if v),
            "holds": holds, "if_not": claim.if_not, "complete": complete}


def reported_only(table: dict, row: str) -> dict[str, Any]:
    return {"row": row, "configuration": CONFIGURATION,
            "per_dataset": {d: paired(table, d, "full", row) for d in DATASETS}}


def literature_row(table: dict) -> dict[str, Any]:
    """The literature row with the roles swapped: `literature - full`; `literature_better` by the same rule."""
    per_dataset = {}
    for d in DATASETS:
        r = paired(table, d, LITERATURE, "full")
        if r is not None:
            r["literature_better"] = r.pop("better")
        per_dataset[d] = r
    return {"row": LITERATURE, "configuration": CONFIGURATION, "difference": "literature - full",
            "per_dataset": per_dataset}


def two_by_two(table: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"differences": {}, "complementary": {}}
    for label, a, b in TWO_BY_TWO:
        item = out["differences"][f"{a} - {b}"] = {"label": label, "per_dataset": {}}
        for dataset in DATASETS:
            r = paired(table, dataset, a, b)
            item["per_dataset"][dataset] = None if r is None else {
                k: r[k] for k in ("seeds", "differences", "mean_difference")}
    faces = out["differences"]["full - no_path_cells"]["per_dataset"]
    plain = out["differences"]["S2 - S0"]["per_dataset"]
    for dataset in DATASETS:
        f, s = faces[dataset], plain[dataset]
        out["complementary"][dataset] = (None if f is None or s is None
                                         else bool(f["mean_difference"] > s["mean_difference"]))
    return out


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        if str(row.get("configuration", CONFIGURATION)) != CONFIGURATION:
            raise ValueError(f"configuration {row['configuration']!r}: pre-registration v3 has configuration (a) "
                             "only")
        key = (row["dataset"], row["row"])
        seed = int(row["seed"])
        if seed in table.setdefault(key, {}):
            raise ValueError(f"two values for {key} seed {seed}")
        table[key][seed] = float(row[METRIC])
    arms = {f"{d}|{r}": {"seeds": sorted(v), **describe(v.values())} for (d, r), v in sorted(table.items())}
    return {"metric": METRIC, "seeds": list(SEEDS), "wins_needed": WINS_NEEDED, "configuration": CONFIGURATION,
            "arms": arms, "claims": [claim_result(table, c) for c in CLAIMS],
            "reported_only": [reported_only(table, r) for r in REPORTED_ONLY],
            "literature": literature_row(table), "two_by_two": two_by_two(table)}


# ------------------------------------------------------------------------------------------------ input / output
KNOWN_ROWS = frozenset({"full", "no_path_cells", "S2", "S0", "uniform", "endpoints", "instance_features",
                        "literature"})
ABLATION_NAME = re.compile(r"^(?P<dataset>[a-z]+)__ablation_(?P<rest>.+)$")
RUN_NAME = re.compile(r"^(?P<dataset>[a-z]+)__ablation_a_(?P<row>.+)$")


def collect(run_dirs: list[Path], lookups: list[Path]) -> list[dict[str, Any]]:
    """Rows from the ablation run folders (`<exp>/<exp>_run<i>/best_test_metrics_run<i>.json`, `exp_name`
    `<dataset>__ablation_a_<row>`) and `full`'s main-table seeds 1-4 (`lookup.json`). Records whose name is not an
    ablation name are skipped; an ablation name of configuration (b), an unknown row or a seed outside 1-5 raises
    (pre-registration v3 has configuration (a), the 8 rows and seeds 1-5 only)."""
    out = []
    for root in run_dirs:
        for path in sorted(Path(root).rglob("best_test_metrics_run*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            base = re.sub(r"_run\d+$", "", str(record["exp_name"]))
            if ABLATION_NAME.match(base) is None:
                continue
            match = RUN_NAME.match(base)
            if match is None:
                raise ValueError(f"{path}: run name {base!r} is not `<dataset>__ablation_a_<row>`; pre-registration "
                                 "v3 has configuration (a) only (no (b) run is read)")
            if match["row"] not in KNOWN_ROWS:
                raise ValueError(f"{path}: row {match['row']!r} is not one of the 8 rows {sorted(KNOWN_ROWS)}")
            seed = int(record["seed"])
            if seed not in SEEDS:
                raise ValueError(f"{path}: seed {seed} is not one of seeds {SEEDS}")
            out.append({"dataset": match["dataset"], "row": match["row"], "configuration": CONFIGURATION,
                        "seed": seed, METRIC: float(record[METRIC]), "source": str(path)})
    for lookup in lookups:
        payload = json.loads(Path(lookup).read_text(encoding="utf-8"))
        for seed, folder in payload["full_seeds_1_4_from_main_table"].items():
            files = sorted(Path(folder).rglob("best_test_metrics_run*.json"))
            if len(files) != 1:
                raise FileNotFoundError(f"{folder}: expected one main-table record for seed {seed}, found "
                                        f"{len(files)}")
            record = json.loads(files[0].read_text(encoding="utf-8"))
            if int(record["seed"]) != int(seed):
                raise ValueError(f"{files[0]}: seed {record['seed']} != {seed}")
            out.append({"dataset": str(record["dataset"]), "row": "full", "configuration": "a",
                        "seed": int(seed), METRIC: float(record[METRIC]), "source": str(files[0])})
    return out


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value:.{digits}g}" if isinstance(value, float) else str(value)


def _diffs(r: dict[str, Any]) -> str:
    return ", ".join(f"{s}: {_fmt(v)}" for s, v in zip(r["seeds"], r["differences"]))


def markdown(result: dict[str, Any]) -> str:
    lines = ["# Ablation statistics (card 9 W3, pre-registration v3)", "",
             f"Metric `{result['metric']}`, configuration (a), seeds {result['seeds']}. Differences are `full - row` "
             f"by seed. `better` = mean difference > 0 and `full` strictly higher in >= {result['wins_needed']} of 5 "
             "seeds (a tie is no win); `—` = incomplete (not all 5 seeds on both sides). No p-value is computed.", "",
             "## Claims", "",
             "| claim | dataset | full mean ± sd | row mean ± sd | differences (seed: full - row) | mean diff | wins "
             "| better |", "|---|---|---|---|---|---|---|---|"]
    for claim in result["claims"]:
        for dataset, r in claim["per_dataset"].items():
            if r is None:
                lines.append(f"| {claim['claim']} ({claim['row']}) | {dataset} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {claim['claim']} ({claim['row']}) | {dataset} | {_fmt(r['a']['mean'])} ± {_fmt(r['a']['std'])} | "
                f"{_fmt(r['b']['mean'])} ± {_fmt(r['b']['std'])} | {_diffs(r)} | {_fmt(r['mean_difference'])} | "
                f"{r['wins']}/{len(r['seeds'])} | {_fmt(r['better'])} |")
        lines.append(f"| **{claim['claim']}** | rule {claim['rule']} | | | | | {claim['datasets_better']} datasets | "
                     f"**holds: {_fmt(claim['holds'])}** (if not: {claim['if_not']}) |")
    lines += ["", "## Reported only: S0 (no claim)", "",
              "| row | dataset | full mean ± sd | row mean ± sd | differences | mean diff | wins | better |",
              "|---|---|---|---|---|---|---|---|"]
    for item in result["reported_only"]:
        for dataset, r in item["per_dataset"].items():
            if r is not None:
                lines.append(f"| {item['row']} | {dataset} | {_fmt(r['a']['mean'])} ± {_fmt(r['a']['std'])} | "
                             f"{_fmt(r['b']['mean'])} ± {_fmt(r['b']['std'])} | {_diffs(r)} | "
                             f"{_fmt(r['mean_difference'])} | {r['wins']}/{len(r['seeds'])} | {_fmt(r['better'])} |")
    lines += ["", "## Literature row (reported; roles swapped: `literature - full`)", "",
              "| dataset | literature mean ± sd | full mean ± sd | differences | mean diff | wins | literature "
              "better |", "|---|---|---|---|---|---|---|"]
    for dataset, r in result["literature"]["per_dataset"].items():
        if r is not None:
            lines.append(f"| {dataset} | {_fmt(r['a']['mean'])} ± {_fmt(r['a']['std'])} | {_fmt(r['b']['mean'])} ± "
                         f"{_fmt(r['b']['std'])} | {_diffs(r)} | {_fmt(r['mean_difference'])} | "
                         f"{r['wins']}/{len(r['seeds'])} | {_fmt(r['literature_better'])} |")
    lines += ["", "## 2x2 (descriptive)", "", "| difference | dataset | differences | mean |", "|---|---|---|---|"]
    for name, item in result["two_by_two"]["differences"].items():
        for dataset, r in item["per_dataset"].items():
            if r is not None:
                lines.append(f"| {name} ({item['label']}) | {dataset} | {_diffs(r)} | {_fmt(r['mean_difference'])} |")
    lines += ["", "Complementary (mean of `full - no_path_cells` > mean of `S2 - S0`): "
              + ", ".join(f"{d} {_fmt(v)}" for d, v in result["two_by_two"]["complementary"].items())]
    return "\n".join(lines) + "\n"


def read_table(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["dataset", "row", "configuration", "seed", METRIC, "source"]  # configuration is always "a"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--runs", type=Path, nargs="+", required=True)
    c.add_argument("--lookup", type=Path, nargs="*", default=[])
    c.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("analyse")
    a.add_argument("--table", type=Path, required=True)
    a.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        rows = collect(args.runs, args.lookup)
        write_table(rows, args.out)
        print(f"{len(rows)} rows -> {args.out}")
        return
    result = analyse(read_table(args.table))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ablation_stats.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (args.out_dir / "ablation_stats.md").write_text(markdown(result), encoding="utf-8")
    print(f"wrote {args.out_dir / 'ablation_stats.json'} and ablation_stats.md")


if __name__ == "__main__":
    main()
