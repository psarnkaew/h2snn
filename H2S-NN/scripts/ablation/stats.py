"""Card 7 V2 (ablation plan v2, "กติกากลาง" and Q2): statistics of the ablation table.

Input: one long table of test Macro-F1 per (dataset, row, configuration, seed) — IMDB's value is the recorded
`test_macro_f1` (the mean per-genre F1 of the IMDB protocol). `collect` builds it from the run folders of
`seed_runs.py` (`best_test_metrics_run<i>.json`, written by `scripts/run_h2snn_runs.py`) and `full`'s seeds 1-4 of
the main table (`lookup.json`).

Per comparison of `full` with a row on a dataset (paired by seed):
- mean ± sample std per arm; the paired differences `full - row` by seed;
- Wilcoxon signed-rank, two-sided (`scipy.stats.wilcoxon`, its defaults);
- Holm over the datasets of **one claim** (m = 4), never over all claims;
- the median difference; a bootstrap 95 % CI of the mean difference (10,000 resamples, percentile, seed 0).

Claims (plan v2): a dataset passes when the Holm-adjusted p < 0.05 and the median difference > 0.
- C2-cells `no_path_cells`: holds on >= 3 of 4 datasets, else warning signal;
- C2 `endpoints`: >= 3 of 4, else kill rule;
- C14 `instance_features`: >= 3 of 4;
- C15 S2: both IMDB and Freebase, else warning signal.
Decided on configuration (b) (`full`'s tuned configuration vs the arm's own tuned one); (a) is reported beside it
with the same statistics. S0 and the literature row: reported only (no claim, no Holm). The 2x2 table: four
descriptive paired differences on configuration (a).

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
ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class Claim:
    name: str
    row: str
    rule: str                       # "3of4" or "imdb_and_freebase"
    if_not: str


CLAIMS: tuple[Claim, ...] = (
    Claim("C2-cells", "no_path_cells", "3of4", "warning signal"),
    Claim("C2", "endpoints", "3of4", "kill rule"),
    Claim("C14", "instance_features", "3of4", "—"),
    Claim("C15", "S2", "imdb_and_freebase", "warning signal"),
)
REPORTED_ONLY: tuple[str, ...] = ("S0", "literature")
TWO_BY_TWO: tuple[tuple[str, str, str], ...] = (
    ("face effect with sheaf", "full", "no_path_cells"),
    ("face effect without sheaf", "S2", "S0"),
    ("sheaf effect with faces", "full", "S2"),
    ("sheaf effect without faces", "no_path_cells", "S0"),
)


# ------------------------------------------------------------------------------------------------ statistics
def holm(p_values: dict[str, float | None]) -> dict[str, float | None]:
    """Holm step-down adjusted p over the given family (keys: datasets of one claim); `None` stays `None` and does
    not count in m."""
    present = sorted(((p, k) for k, p in p_values.items() if p is not None), key=lambda t: (t[0], t[1]))
    m = len(present)
    adjusted: dict[str, float | None] = {k: None for k in p_values}
    running = 0.0
    for i, (p, key) in enumerate(present):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


def wilcoxon_p(differences: np.ndarray) -> float | None:
    """Two-sided Wilcoxon signed-rank p (`scipy.stats.wilcoxon`, defaults); `None` when every difference is 0."""
    from scipy.stats import wilcoxon

    if differences.size == 0 or np.all(differences == 0):
        return None
    return float(wilcoxon(differences, alternative="two-sided").pvalue)


def bootstrap_ci(differences: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile 95 % CI of the mean paired difference (resampling seeds with replacement, fixed seed)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, differences.size, size=(resamples, differences.size))
    means = differences[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def describe(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=np.float64)
    return {"n": int(arr.size), "mean": float(arr.mean()) if arr.size else math.nan,
            "std": float(arr.std(ddof=1)) if arr.size > 1 else math.nan}


def paired(table: dict, dataset: str, a: tuple[str, str], b: tuple[str, str]) -> dict[str, Any] | None:
    """Paired comparison `a - b` on `dataset`; `a`, `b` = (row, configuration). Seeds present in both only."""
    va, vb = table.get((dataset, *a), {}), table.get((dataset, *b), {})
    seeds = sorted(set(va) & set(vb))
    if not seeds:
        return None
    diff = np.array([va[s] - vb[s] for s in seeds], dtype=np.float64)
    low, high = bootstrap_ci(diff)
    return {"seeds": seeds, "a": describe(va[s] for s in seeds), "b": describe(vb[s] for s in seeds),
            "differences": diff.tolist(), "mean_difference": float(diff.mean()),
            "median_difference": float(np.median(diff)), "bootstrap_ci95": [low, high],
            "p_raw": wilcoxon_p(diff),
            "unpaired_seeds": sorted(set(va) ^ set(vb))}


def claim_result(table: dict, claim: Claim, configuration: str) -> dict[str, Any]:
    """`full` vs the claim's row on every dataset, Holm over this claim's datasets only, and the verdict."""
    full_conf = "a"                 # `full`'s tuned configuration is its configuration (a) (it has no (b))
    per_dataset = {d: paired(table, d, ("full", full_conf), (claim.row, configuration)) for d in DATASETS}
    adjusted = holm({d: (r["p_raw"] if r else None) for d, r in per_dataset.items()})
    passes = {}
    for dataset, result in per_dataset.items():
        if result is None:
            passes[dataset] = None
            continue
        result["p_holm"] = adjusted[dataset]
        result["m"] = sum(1 for r in per_dataset.values() if r is not None and r["p_raw"] is not None)
        passes[dataset] = bool(result["p_holm"] is not None and result["p_holm"] < ALPHA
                               and result["median_difference"] > 0)
        result["passes"] = passes[dataset]
    complete = all(passes[d] is not None for d in DATASETS)
    if claim.rule == "3of4":
        holds = sum(1 for v in passes.values() if v) >= 3 if complete else None
    else:
        needed = ("imdb", "freebase")
        holds = all(passes[d] for d in needed) if all(passes[d] is not None for d in needed) else None
    return {"claim": claim.name, "row": claim.row, "configuration": configuration, "rule": claim.rule,
            "per_dataset": per_dataset, "passes": passes, "holds": holds,
            "if_not": claim.if_not, "complete": complete}


def reported_only(table: dict, row: str) -> dict[str, Any]:
    return {"row": row, "configuration": "a",
            "per_dataset": {d: paired(table, d, ("full", "a"), (row, "a")) for d in DATASETS}}


def two_by_two(table: dict) -> dict[str, Any]:
    out = {}
    for label, a, b in TWO_BY_TWO:
        out[f"{a} - {b}"] = {"label": label, "per_dataset": {}}
        for dataset in DATASETS:
            result = paired(table, dataset, (a, "a"), (b, "a"))
            out[f"{a} - {b}"]["per_dataset"][dataset] = None if result is None else {
                k: result[k] for k in ("seeds", "mean_difference", "median_difference", "bootstrap_ci95")}
    return out


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[tuple[str, str, str], dict[int, float]] = {}
    for row in rows:
        key = (row["dataset"], row["row"], row["configuration"])
        seed = int(row["seed"])
        if seed in table.setdefault(key, {}):
            raise ValueError(f"two values for {key} seed {seed}")
        table[key][seed] = float(row[METRIC])
    arms = {f"{d}|{r}|{c}": {"seeds": sorted(v), **describe(v.values())} for (d, r, c), v in sorted(table.items())}
    return {"metric": METRIC, "alpha": ALPHA, "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES,
                                                             "seed": BOOTSTRAP_SEED, "kind": "percentile"},
            "arms": arms,
            "claims_b": [claim_result(table, c, "b") for c in CLAIMS],
            "claims_a": [claim_result(table, c, "a") for c in CLAIMS],
            "reported_only": [reported_only(table, r) for r in REPORTED_ONLY],
            "two_by_two_a": two_by_two(table)}


# ------------------------------------------------------------------------------------------------ input / output
KNOWN_ROWS = frozenset({"full", "no_path_cells", "S2", "S0", "endpoints", "instance_features", "literature"})
RUN_NAME = re.compile(r"^(?P<dataset>[a-z]+)__ablation_(?P<conf>[ab])_(?P<row>.+?)(?P<seeds>_seeds)?$")


def collect(run_dirs: list[Path], lookups: list[Path]) -> list[dict[str, Any]]:
    """Rows from the ablation run folders (`<exp>/<exp>_run<i>/best_test_metrics_run<i>.json`, `exp_name`
    `<dataset>__ablation_<a|b>_<row>[_seeds]`) and `full`'s main-table seeds 1-4 (`lookup.json`)."""
    out = []
    for root in run_dirs:
        for path in sorted(Path(root).rglob("best_test_metrics_run*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            base = re.sub(r"_run\d+$", "", str(record["exp_name"]))
            match = RUN_NAME.match(base)
            # (b) seed runs carry `_seeds` (the `tune` rows' trial records are not ablation runs); seeds 1-10 only
            if (match is None or match["row"] not in KNOWN_ROWS or (match["conf"] == "b") != bool(match["seeds"])
                    or not 1 <= int(record["seed"]) <= 10):
                continue
            out.append({"dataset": match["dataset"], "row": match["row"], "configuration": match["conf"],
                        "seed": int(record["seed"]), METRIC: float(record[METRIC]), "source": str(path)})
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


def markdown(result: dict[str, Any]) -> str:
    lines = ["# Ablation statistics (card 7 V2, plan v2)", "",
             f"Metric `{result['metric']}`; Wilcoxon signed-rank two-sided; Holm per claim over its datasets; "
             f"bootstrap {result['bootstrap']['resamples']} resamples (percentile, seed "
             f"{result['bootstrap']['seed']}). Differences are `full - row`.", ""]
    for key, title in (("claims_b", "Claims, configuration (b) (decides)"),
                       ("claims_a", "Claims, configuration (a) (reported beside)")):
        lines += [f"## {title}", "",
                  "| claim | dataset | full mean ± sd | row mean ± sd | n | median diff | mean diff [95 % CI] | "
                  "p raw | p Holm | passes |", "|---|---|---|---|---|---|---|---|---|---|"]
        for claim in result[key]:
            for dataset, r in claim["per_dataset"].items():
                if r is None:
                    lines.append(f"| {claim['claim']} | {dataset} | — | — | 0 | — | — | — | — | — |")
                    continue
                lines.append(
                    f"| {claim['claim']} | {dataset} | {_fmt(r['a']['mean'])} ± {_fmt(r['a']['std'])} | "
                    f"{_fmt(r['b']['mean'])} ± {_fmt(r['b']['std'])} | {len(r['seeds'])} | "
                    f"{_fmt(r['median_difference'])} | {_fmt(r['mean_difference'])} "
                    f"[{_fmt(r['bootstrap_ci95'][0])}, {_fmt(r['bootstrap_ci95'][1])}] | {_fmt(r['p_raw'])} | "
                    f"{_fmt(r.get('p_holm'))} | {_fmt(r.get('passes'))} |")
            lines.append(f"| **{claim['claim']}** | rule {claim['rule']} | | | | | | | | **holds: "
                         f"{_fmt(claim['holds'])}** (if not: {claim['if_not']}) |")
        lines.append("")
    lines += ["## Reported only (configuration (a), no claim)", "",
              "| row | dataset | n | median diff | mean diff [95 % CI] | p raw |", "|---|---|---|---|---|---|"]
    for item in result["reported_only"]:
        for dataset, r in item["per_dataset"].items():
            if r is not None:
                lines.append(f"| {item['row']} | {dataset} | {len(r['seeds'])} | {_fmt(r['median_difference'])} | "
                             f"{_fmt(r['mean_difference'])} [{_fmt(r['bootstrap_ci95'][0])}, "
                             f"{_fmt(r['bootstrap_ci95'][1])}] | {_fmt(r['p_raw'])} |")
    lines += ["", "## 2x2 (configuration (a), descriptive)", "",
              "| difference | dataset | mean | median | 95 % CI |", "|---|---|---|---|---|"]
    for name, item in result["two_by_two_a"].items():
        for dataset, r in item["per_dataset"].items():
            if r is not None:
                lines.append(f"| {name} ({item['label']}) | {dataset} | {_fmt(r['mean_difference'])} | "
                             f"{_fmt(r['median_difference'])} | [{_fmt(r['bootstrap_ci95'][0])}, "
                             f"{_fmt(r['bootstrap_ci95'][1])}] |")
    return "\n".join(lines) + "\n"


def read_table(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["dataset", "row", "configuration", "seed", METRIC, "source"]
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
