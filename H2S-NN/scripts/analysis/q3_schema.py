"""Card 9 W4 (pre-registration v3 §4.1, spec 1.29 §5 C2b; built on card 7 V4): the schema read-out analysis. Reads
the per-run JSONs of `scripts/schema_readout.py` (c_sigma, I_sigma, rho_sigma per sub-path schema, computed by the
existing `src/analysis/schema_readout.py` functions on each run's best checkpoint and evaluation draw); trains nothing.

Inputs: `full`'s configuration-(a) runs of seeds 1-5 only (seeds 1-4 are the main-table runs, seed 5 the ablation
run). A record of any other seed (main-table seed 0) is not read; it is listed under `ignored`. Per dataset:
- **Top schemas:** length-2 sub-path schemas with mean coverage c_sigma >= 1 % over seeds 1-5, the ten largest by
  mean coverage (their c, I, rho per seed);
- **Stability (C2b):** the mean pairwise Kendall tau-b of the rho_sigma rankings of the top schemas (those present in
  every seed) over the 10 pairs of seeds 1-5; stable = mean tau-b >= 0.5 (`None` with fewer than the 5 seeds). No
  permutation null and no multiplicity adjustment (v3 D5). The mean tau-b of the c_sigma rankings beside it; fewer
  than 4 schemas -> "n/a";
- **Literature check:** per HGB published meta-path, the number of seeds (of 5) with rho_sigma > 1; `favour` = the
  count is 5 of 5 (only then does the paper say "the learned face weights favour sigma");
- **Heterophily link:** a scatter of rho_sigma against the label (edge) homophily of sigma's induced target-type
  graph G_sigma (length-2 schemas whose two ends are the target type; homophily from the reference code
  `scripts/heterophily/compute_heterophily.py` `analyse`, on the graph as its own driver loads it: Freebase
  symmetrised), all seeds, one marker per (seed, schema), per dataset; the per-seed Spearman in the JSON (fewer than
  3 such schemas in a seed -> "n/a"). No pooled single-number verdict;
- **Figures (PDF):** a bar chart of rho_sigma for the top schemas (mean over seeds, range as error bar), a scatter of
  c_sigma vs I_sigma with the diagonal, and the homophily scatter (when `--data_root` is given).
Wording (v3 §4.1): "the learned face weights", never "the sampler learns" (WISP's proposal is fixed; `g` weighs the
faces after they are drawn).

    python scripts/analysis/q3_schema.py --runs <out>/<dataset>/*.json ... --out_dir <dir> [--data_root data]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "ablation"))

COVERAGE_MIN = 0.01
TOP = 10
STABILITY_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
TAU_STABLE = 0.5
MIN_SCHEMAS = 4
MIN_SPEARMAN_POINTS = 3
ABLATION_DATASETS: tuple[str, ...] = ("acm", "dblp", "imdb", "freebase")

Signature = tuple[int, ...]


# ------------------------------------------------------------------------------------------------ input
def load_records(paths: list[Path], ignored: list[dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Run records by dataset (WISP runs only: the literature row's runs are not read-out inputs); one per seed;
    seeds 1-5 only (a record of another seed, e.g. main-table seed 0, is appended to `ignored`, never read)."""
    by_dataset: dict[str, dict[int, dict[str, Any]]] = {}
    for path in paths:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if "schemas" not in record or record.get("path_source", "wisp") not in ("wisp", ""):
            continue
        if int(record["seed"]) not in STABILITY_SEEDS:
            if ignored is not None:
                ignored.append({"run": str(path), "dataset": record.get("dataset"), "seed": int(record["seed"])})
            continue
        seeds = by_dataset.setdefault(str(record["dataset"]), {})
        if int(record["seed"]) in seeds:
            raise ValueError(f"{path}: a second run of {record['dataset']} seed {record['seed']}")
        seeds[int(record["seed"])] = record
    return {d: [r[s] for s in sorted(r)] for d, r in by_dataset.items()}


def length2(record: dict[str, Any]) -> dict[Signature, dict[str, Any]]:
    return {tuple(s["signature"]): s for s in record["schemas"] if s["span"] == 2}


def top_schemas(records: list[dict[str, Any]]) -> list[Signature]:
    """Length-2 schemas with mean coverage >= 1 % (absent in a run = 0), the ten largest by mean coverage."""
    tables = [length2(r) for r in records]
    keys = sorted({k for t in tables for k in t})
    mean_c = {k: float(np.mean([t[k]["c"] if k in t else 0.0 for t in tables])) for k in keys}
    eligible = [k for k in keys if mean_c[k] >= COVERAGE_MIN]
    return sorted(eligible, key=lambda k: (-mean_c[k], k))[:TOP]


# ------------------------------------------------------------------------------------------------ stability
def mean_pairwise_tau(matrix: np.ndarray) -> float:
    """Mean Kendall tau-b (`scipy.stats.kendalltau`, variant b) over every pair of rows (seeds)."""
    from scipy.stats import kendalltau

    taus = []
    for a in range(matrix.shape[0]):
        for b in range(a + 1, matrix.shape[0]):
            tau = kendalltau(matrix[a], matrix[b], variant="b").statistic
            taus.append(0.0 if math.isnan(tau) else float(tau))
    return float(np.mean(taus))


def stability(records: list[dict[str, Any]], schemas: list[Signature]) -> dict[str, Any]:
    """Mean pairwise Kendall tau-b of the rho rankings over seeds 1-5 (10 pairs); stable = mean >= 0.5, `None` when
    fewer than the 5 seeds are present. No permutation null (v3 D5)."""
    runs = [r for r in records if int(r["seed"]) in STABILITY_SEEDS]
    tables = [length2(r) for r in runs]
    present = [k for k in schemas if all(k in t and t[k]["rho"] is not None for t in tables)]
    out: dict[str, Any] = {"seeds": [int(r["seed"]) for r in runs], "schemas": [list(k) for k in present]}
    if len(present) < MIN_SCHEMAS or len(runs) < 2:
        return {**out, "status": "n/a", "reason": f"{len(present)} schemas, {len(runs)} seeds", "stable": None}
    rho = np.array([[t[k]["rho"] for k in present] for t in tables], dtype=np.float64)
    cov = np.array([[t[k]["c"] for k in present] for t in tables], dtype=np.float64)
    observed = mean_pairwise_tau(rho)
    complete = sorted(out["seeds"]) == list(STABILITY_SEEDS)
    return {**out, "status": "computed", "pairs": len(runs) * (len(runs) - 1) // 2, "mean_tau_b_rho": observed,
            "mean_tau_b_c": mean_pairwise_tau(cov), "complete": complete,
            "stable": bool(observed >= TAU_STABLE) if complete else None}


def literature_counts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs = [r for r in records if int(r["seed"]) in STABILITY_SEEDS]
    if not runs or runs[0].get("literature") is None:
        return []
    out = []
    for i, item in enumerate(runs[0]["literature"]):
        rhos = [r["literature"][i]["rho"] for r in runs]
        count = sum(1 for v in rhos if v is not None and v > 1.0)
        out.append({"listed": item["listed"], "relations": item["relations"], "seeds": len(runs),
                    "seeds_rho_above_1": count, "favour": count == len(STABILITY_SEEDS),
                    "rho_per_seed": rhos})
    return out


# ------------------------------------------------------------------------------------------------ heterophily link
def hop_edge_types(relations: str) -> list[tuple[str, int]]:
    """`paper -[paper__to__author +]-> author -[paper__to__author -]-> paper` -> [(relation, direction), ...]."""
    out = []
    for part in relations.split("-[")[1:]:
        name, mark = part.split("]")[0].rsplit(" ", 1)
        out.append((name, {"+": 1, "-": -1, "~": 0}[mark]))
    return out


def mirror_type(data, edge_type: tuple[str, str, str]) -> tuple[str, str, str] | None:
    """The stored edge type whose edges are `edge_type`'s reversed (HGB's reverse link, or Freebase's `rev_`)."""
    edges = data[edge_type].edge_index
    forward = set(zip(edges[0].tolist(), edges[1].tolist()))
    for other in data.edge_types:
        if (other[0], other[2]) != (edge_type[2], edge_type[0]):
            continue
        e = data[other].edge_index
        if e.size(1) == edges.size(1) and set(zip(e[1].tolist(), e[0].tolist())) == forward:
            return other
    return None


def schema_metapath(data, target: str, relations: str) -> tuple | None:
    """σ's G_σ as an M_2 meta-path of the loaded graph, or None when σ's ends are not both the target type or a
    hop does not resolve."""
    hops = hop_edge_types(relations)
    if len(hops) != 2:
        return None
    path = []
    for name, direction in hops:
        edge_type = tuple(name.split("__"))
        if edge_type not in data.edge_types:
            return None
        if direction < 0:
            edge_type = mirror_type(data, edge_type)
            if edge_type is None:
                return None
        path.append(edge_type)
    if path[0][0] != target or path[1][2] != target or path[0][2] != path[1][0]:
        return None
    return tuple(path)


def schema_homophily(data_root: Path, dataset: str) -> tuple[Any, str, dict[str, float]]:
    """(graph, target, {M_2 name: edge homophily}) from the reference `analyse` (fill seed 0, as the reference's
    primary value), on the graph as the reference driver loads it."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "heterophily"))
    import compute_heterophily as ref

    data = ref.load_hgb(data_root, dataset)
    if dataset == "freebase":
        data = ref.symmetrise_like_h2gb(data)
    result = ref.analyse(data, dataset, "q3", [0], 200, 30_000_000, 1e-6)
    return data, ref.target_type(data), {row.metapath: row.h_edge for row in result.rows}


def spearman_link(records: list[dict[str, Any]], data, target: str, homophily: dict[str, float]
                  ) -> list[dict[str, Any]]:
    from scipy.stats import spearmanr

    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "heterophily"))
    import compute_heterophily as ref

    out = []
    for record in records:
        points = []
        for schema in length2(record).values():
            if schema["rho"] is None:
                continue
            path = schema_metapath(data, target, schema["relations"])
            name = ref.metapath_name(path) if path is not None else None
            if name in homophily:
                points.append((schema["rho"], homophily[name], schema["name"]))
        entry = {"seed": int(record["seed"]), "points": len(points),
                 "schemas": [p[2] for p in points],
                 "values": [{"schema": p[2], "rho": p[0], "homophily": p[1]} for p in points]}
        if len(points) < MIN_SPEARMAN_POINTS:
            out.append({**entry, "status": "n/a"})
            continue
        stat = spearmanr([p[0] for p in points], [p[1] for p in points])
        out.append({**entry, "status": "computed", "spearman": float(stat.statistic)})
    return out


def homophily_scatter(dataset: str, link: list[dict[str, Any]], out_dir: Path) -> str:
    """rho_sigma against the homophily of G_sigma: every (seed, schema) point of `spearman_link`, one marker per seed
    (legend: the seed and its Spearman)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    markers = "osD^v<>"
    for i, entry in enumerate(link):
        xs = [v["homophily"] for v in entry["values"]]
        ys = [v["rho"] for v in entry["values"]]
        label = f"seed {entry['seed']}" + (f" (Spearman {entry['spearman']:.2f})" if entry["status"] == "computed"
                                           else " (n/a)")
        ax.scatter(xs, ys, s=14, marker=markers[i % len(markers)], alpha=0.7, label=label)
    ax.axhline(1.0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"label homophily of $G_\sigma$")
    ax.set_ylabel(r"$\rho_\sigma$ (learned face weights / coverage)")
    ax.set_title(f"{dataset}: learned face weights vs schema homophily", fontsize=9)
    ax.legend(fontsize=6)
    fig.tight_layout()
    path = out_dir / f"{dataset}_rho_vs_homophily.pdf"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ------------------------------------------------------------------------------------------------ figures
def figures(dataset: str, records: list[dict[str, Any]], schemas: list[Signature], out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tables = [length2(r) for r in records]
    names, means, lows, highs = [], [], [], []
    for k in schemas:
        values = [t[k]["rho"] for t in tables if k in t and t[k]["rho"] is not None]
        if not values:
            continue
        name = next(t[k]["name"] for t in tables if k in t)
        names.append(name)
        means.append(float(np.mean(values)))
        lows.append(max(0.0, float(np.mean(values) - min(values))))
        highs.append(max(0.0, float(max(values) - np.mean(values))))
    order = np.argsort(means)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(range(len(order)), [means[i] for i in order], yerr=[[lows[i] for i in order], [highs[i] for i in order]],
           color="#4c6fbf", capsize=3)
    ax.axhline(1.0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(order)), [names[i] for i in order], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel(r"$\rho_\sigma$ (mean, range over seeds)")
    ax.set_title(f"{dataset}: learned face weights' share / coverage, top length-2 schemas", fontsize=9)
    fig.tight_layout()
    bar = out_dir / f"{dataset}_rho_top.pdf"
    fig.savefig(bar)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    cs = [s["c"] for t in tables for s in t.values()]
    is_ = [s["I"] for t in tables for s in t.values()]
    ax.scatter(cs, is_, s=6, alpha=0.5, color="#4c6fbf")
    top = max(cs + is_ + [1e-9])
    ax.plot([0, top], [0, top], color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"coverage $c_\sigma$")
    ax.set_ylabel(r"learned share $I_\sigma$")
    ax.set_title(f"{dataset}: length-2 schemas, all seeds", fontsize=9)
    fig.tight_layout()
    scatter = out_dir / f"{dataset}_coverage_vs_share.pdf"
    fig.savefig(scatter)
    plt.close(fig)
    return [str(bar), str(scatter)]


# ------------------------------------------------------------------------------------------------ driver
def analyse(records_by_dataset: dict[str, list[dict[str, Any]]], out_dir: Path, data_root: Path | None,
            make_figures: bool = True, ignored: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"coverage_min": COVERAGE_MIN, "top": TOP, "seeds": list(STABILITY_SEEDS),
                              "tau_stable": TAU_STABLE, "ignored": list(ignored or []), "datasets": {}}
    for dataset, records in sorted(records_by_dataset.items()):
        records = [r for r in records if int(r["seed"]) in STABILITY_SEEDS]
        schemas = top_schemas(records)
        tables = [length2(r) for r in records]
        top = []
        for k in schemas:
            name = next(t[k]["name"] for t in tables if k in t)
            relations = next(t[k]["relations"] for t in tables if k in t)
            top.append({"signature": list(k), "name": name, "relations": relations,
                        "per_seed": {int(r["seed"]): ({f: t[k][f] for f in ("c", "I", "rho")} if k in t else None)
                                     for r, t in zip(records, tables)}})
        entry: dict[str, Any] = {"seeds": [int(r["seed"]) for r in records], "top_schemas": top,
                                 "stability": stability(records, schemas),
                                 "literature": literature_counts(records)}
        if make_figures:
            entry["figures"] = figures(dataset, records, schemas, out_dir)
        if data_root is not None:
            data, target, homophily = schema_homophily(data_root, dataset)
            entry["schema_homophily"] = homophily
            entry["spearman_rho_vs_homophily"] = spearman_link(records, data, target, homophily)
            if make_figures:
                entry["figures"].append(homophily_scatter(dataset, entry["spearman_rho_vs_homophily"], out_dir))
        result["datasets"][dataset] = entry
    (out_dir / "q3_schema.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out_dir / "q3_schema.md").write_text(markdown(result), encoding="utf-8")
    return result


def _f(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def markdown(result: dict[str, Any]) -> str:
    lines = ["# Q3 schema read-out (card 9 W4; pre-registration v3 §4.1)", "",
             f"`full`, configuration (a), seeds {result['seeds']}; stable = mean tau-b of the rho rankings >= "
             f"{result['tau_stable']} (no permutation null). rho_sigma = the learned face weights' share / coverage.",
             "", "| dataset | seeds | schemas | pairs | mean tau-b (rho) | stable | mean tau-b (c) |",
             "|---|---|---|---|---|---|---|"]
    for d, e in result["datasets"].items():
        s = e["stability"]
        if s["status"] != "computed":
            lines.append(f"| {d} | {s['seeds']} | {len(s['schemas'])} | | n/a ({s['reason']}) | | |")
            continue
        lines.append(f"| {d} | {s['seeds']} | {len(s['schemas'])} | {s['pairs']} | {_f(s['mean_tau_b_rho'])} | "
                     f"{_f(s['stable'])} | {_f(s['mean_tau_b_c'])} |")
    for d, e in result["datasets"].items():
        if e["literature"]:
            lines += ["", f"## {d}: literature meta-paths, seeds (of 5) with rho > 1", "",
                      "| meta-path | seeds | rho > 1 | learned face weights favour (5 of 5) |", "|---|---|---|---|"]
            lines += [f"| {x['listed']} | {x['seeds']} | {x['seeds_rho_above_1']} | {_f(x['favour'])} |"
                      for x in e["literature"]]
        if "spearman_rho_vs_homophily" in e:
            lines += ["", f"## {d}: Spearman(rho, homophily of G_sigma) per seed", "", "| seed | points | rho_s |",
                      "|---|---|---|"]
            lines += [f"| {x['seed']} | {x['points']} | {_f(x.get('spearman')) if x['status'] == 'computed' else 'n/a'} |"
                      for x in e["spearman_rho_vs_homophily"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="schema_readout.py run JSONs")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=None,
                        help="data root for the schema homophily (reference code); omit to skip that part")
    args = parser.parse_args()
    ignored: list[dict[str, Any]] = []
    records = load_records(args.runs, ignored)
    analyse(records, args.out_dir, args.data_root, ignored=ignored)


if __name__ == "__main__":
    main()
