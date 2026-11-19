"""Card 7 amendment 1 (author 2026-11-19): the perturbation figure (Q4): test Macro-F1 against h per method.

`collect` reads the manifest (`manifest.tsv`: h, H^2 and the summary file of every row) and each row's summary
(`summary.metrics.test_macro_f1.{mean, std}`, the format of H2S-NN's `runs_summary.json` and of the baseline
summaries); rows without a summary yet are listed under `missing`. The ceiling 1 - 1/C of each dataset is read from
the graphs' `graph.json` (`ceiling`, written by `make_graphs.py`).

`figure`: one panel per dataset and graph kind (targeted, random); x = the reached h of each graph, one line per
method (mean, error bar = std over seeds 1-5); a dashed vertical line at 1 - 1/C per dataset (the largest h edge
addition can reach, spec 1.27 / pre-registration v2 §4.2); the original graph as a reference point in both panels
(hollow markers at the original graph's h). No training.

Reference point (amendment 8, author 2026-11-19): the baselines' rows of kind `original` in the manifest (seeds 1-5,
same configuration source and checked launcher as their perturbation rows); H2S-NN `full`, `no_path_cells` and S2 from
the ablation configuration (a) runs at seeds 1-5 on the original graphs (`--h2snn_reference_table`: the long table of
`scripts/ablation/stats.py collect`; `full`'s seeds 1-4 there are the main table's runs, which are configuration (a)
by construction). A reference with fewer than the 5 seeds is listed under `missing`. Main-table summaries are never
read.

    python scripts/perturbation/figure.py --manifest <dir>/manifest.tsv --graphs data/NC --out_dir <dir> \
        [--h2snn_reference_table <stats collect csv>]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "timing"))

from cost_vs_accuracy import read_metric  # noqa: E402  (one reader of the summary format)

KINDS = ("targeted", "random")
ORIGINAL = "original"
REFERENCE_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
H2SNN_REFERENCE_ROWS: tuple[str, ...] = ("full", "no_path_cells", "S2")


def ceilings(graphs_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for path in sorted(Path(graphs_dir).glob("*_pert/*_h*/graph.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("ceiling") is None:
            continue
        if out.setdefault(record["dataset"], record["ceiling"]) != record["ceiling"]:
            raise ValueError(f"{path}: ceiling {record['ceiling']} differs from {out[record['dataset']]}")
    return out


def collect(manifest: Path) -> dict[str, Any]:
    """Points of the perturbed graphs (`points`), the baselines' original-graph points (`reference`, amendment 8) and
    the rows without a summary yet (`missing`)."""
    points, reference, missing = [], [], []
    with Path(manifest).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            summary = Path(row["summary"]) if row["summary"] else None
            original = row["graph_kind"] == ORIGINAL
            key = {"dataset": row["dataset"], "kind": row["graph_kind"],
                   "level": None if original else float(row["level"]), "method": row["method"]}
            if summary is None or not summary.exists():
                missing.append({**key, "configuration": row["configuration"]})
                continue
            mean, std = read_metric(summary)
            point = {**key, "h": float(row["h"]), "h2_index": float(row["h2_index"]), "mean": mean, "std": std}
            (reference if original else points).append(point)
    return {"points": points, "reference": reference, "missing": missing}


def h2snn_reference(table: Path, original_h: dict[str, float], datasets: tuple[str, ...] = ("acm", "dblp")
                    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Amendment 8: H2S-NN `full`, `no_path_cells`, S2 at the original graph's h from the ablation configuration (a)
    runs, seeds 1-5 (mean, sample std). Returns (reference points, missing)."""
    import statistics

    values: dict[tuple[str, str], dict[int, float]] = {}
    with Path(table).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["configuration"] != "a" or row["row"] not in H2SNN_REFERENCE_ROWS:
                continue
            seed = int(row["seed"])
            if seed in REFERENCE_SEEDS and row["dataset"] in datasets:
                values.setdefault((row["dataset"], row["row"]), {})[seed] = float(row["test_macro_f1"])
    points, missing = [], []
    for dataset in datasets:
        for arm in H2SNN_REFERENCE_ROWS:
            key = {"dataset": dataset, "kind": ORIGINAL, "level": None, "method": f"h2snn_{arm}"}
            seeds = values.get((dataset, arm), {})
            if sorted(seeds) != list(REFERENCE_SEEDS) or dataset not in original_h:
                missing.append({**key, "configuration": "a", "seeds_found": sorted(seeds)})
                continue
            scores = [seeds[s] for s in REFERENCE_SEEDS]
            points.append({**key, "h": original_h[dataset], "mean": statistics.fmean(scores),
                           "std": statistics.stdev(scores)})
    return points, missing


def original_h(graphs_dir: Path) -> dict[str, float]:
    """h of each dataset's original graph (`<DATASET>_pert/original/graph.json`, `make_graphs.py --original`)."""
    out = {}
    for path in sorted(Path(graphs_dir).glob(f"*_pert/{ORIGINAL}/graph.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        out[record["dataset"]] = float(record["h"])
    return out


def figure(result: dict[str, Any], ceiling: dict[str, float], path: Path) -> list[dict[str, Any]]:
    """Writes the figure; returns the ceiling lines drawn (dataset, kind, x, line style), read back from the axes.
    `result["reference"]`: the original-graph points, drawn in both panels of their dataset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = result.get("reference", [])
    datasets = sorted({p["dataset"] for p in result["points"]} | {p["dataset"] for p in reference} | set(ceiling))
    fig, axes = plt.subplots(len(KINDS), len(datasets), figsize=(3.6 * len(datasets), 5.6), squeeze=False,
                             sharey="col")
    methods = sorted({p["method"] for p in result["points"]} | {p["method"] for p in reference})
    colors = {m: plt.get_cmap("tab10")(i % 10) for i, m in enumerate(methods)}
    drawn: list[dict[str, Any]] = []
    for col, dataset in enumerate(datasets):
        for row, kind in enumerate(KINDS):
            ax = axes[row][col]
            for method in methods:
                pts = sorted((p for p in result["points"] if (p["dataset"], p["kind"], p["method"]) ==
                              (dataset, kind, method)), key=lambda p: p["h"])
                if pts:
                    ax.errorbar([p["h"] for p in pts], [p["mean"] for p in pts],
                                yerr=[p["std"] or 0.0 for p in pts], marker="o", markersize=3, capsize=2,
                                color=colors[method], label=method)
            for ref in (p for p in reference if p["dataset"] == dataset):
                ax.errorbar([ref["h"]], [ref["mean"]], yerr=[ref["std"] or 0.0], fmt="o", mfc="none",
                            color=colors[ref["method"]], capsize=2, markersize=5)
            if dataset in ceiling:
                line = ax.axvline(ceiling[dataset], linestyle="--", color="black", linewidth=0.8, label="1 - 1/C")
                drawn.append({"dataset": dataset, "kind": kind, "x": float(line.get_xdata()[0]),
                              "linestyle": line.get_linestyle()})
            ax.set_title(f"{dataset}, {kind}", fontsize=9)
            ax.set_xlabel("h (1 - MLH over M_2)")
            if col == 0:
                ax.set_ylabel("test Macro-F1")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return drawn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True, help="the data root's NC folder (holds *_pert/)")
    parser.add_argument("--h2snn_reference_table", type=Path, default=None,
                        help="amendment 8: the ablation runs' long table (scripts/ablation/stats.py collect)")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.manifest)
    ceiling = ceilings(args.graphs)
    if args.h2snn_reference_table is not None:
        points, missing = h2snn_reference(args.h2snn_reference_table, original_h(args.graphs))
        result["reference"] += points
        result["missing"] += missing
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "perturbation.json").write_text(json.dumps({**result, "ceiling": ceiling}, indent=1),
                                                    encoding="utf-8")
    figure(result, ceiling, args.out_dir / "perturbation.pdf")
    print(f"{len(result['points'])} points, {len(result['reference'])} reference points, "
          f"{len(result['missing'])} missing -> {args.out_dir}")


if __name__ == "__main__":
    main()
