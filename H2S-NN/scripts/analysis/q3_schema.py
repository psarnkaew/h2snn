"""Card 7 V4 (ablation plan v2 Q3; spec 1.27 §5 C2b, §6 "Schema read-out and coverage"): the schema read-out
analysis. Reads the per-run JSONs of `scripts/schema_readout.py` (c_sigma, I_sigma, rho_sigma per sub-path schema,
computed by the existing `src/analysis/schema_readout.py` functions on each run's best checkpoint and evaluation
draw); trains nothing.

Inputs: the main-table runs (seeds 0-4) and `full`'s ablation runs (configuration (a), seeds 1-10; seeds 1-4 are the
main table's). Per dataset:
- **Top schemas:** length-2 sub-path schemas with mean coverage c_sigma >= 1 % over the runs, the ten largest by
  mean coverage (their c, I, rho per seed);
- **Stability (C2b):** over seeds 1-10, the mean pairwise Kendall tau-b of the rho_sigma rankings of the top schemas
  (those present in every seed); permutation null: shuffle schema labels within each seed, 10,000 draws (fixed seed),
  p = (1 + #{null >= observed}) / (1 + draws); Holm over the 4 ablation datasets; stable = mean tau-b >= 0.5 and
  adjusted p < 0.05; the mean tau-b of the c_sigma rankings beside it; fewer than 4 schemas -> "n/a";
- **Literature check:** per HGB published meta-path, the number of seeds (1-10) with rho_sigma > 1;
- **Heterophily link:** per seed, Spearman between rho_sigma and the edge homophily of sigma's induced target-type
  graph G_sigma (length-2 schemas whose two ends are the target type; homophily from the reference code
  `scripts/heterophily/compute_heterophily.py` `analyse`, on the graph as its own driver loads it: Freebase
  symmetrised); fewer than 3 such schemas in a seed -> "n/a";
- **Figures (PDF):** a bar chart of rho_sigma for the top schemas (mean over seeds, range as error bar) and a scatter
  of c_sigma vs I_sigma with the diagonal.

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
STABILITY_SEEDS: tuple[int, ...] = tuple(range(1, 11))
PERMUTATIONS = 10_000
PERMUTATION_SEED = 0
TAU_STABLE = 0.5
ALPHA = 0.05
MIN_SCHEMAS = 4
MIN_SPEARMAN_POINTS = 3
ABLATION_DATASETS: tuple[str, ...] = ("acm", "dblp", "imdb", "freebase")

Signature = tuple[int, ...]


# ------------------------------------------------------------------------------------------------ input
def load_records(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """Run records by dataset (WISP runs only: the literature row's runs are not read-out inputs); one per seed."""
    by_dataset: dict[str, dict[int, dict[str, Any]]] = {}
    for path in paths:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if "schemas" not in record or record.get("path_source", "wisp") not in ("wisp", ""):
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


def permutation_p(matrix: np.ndarray, observed: float, draws: int = PERMUTATIONS,
                  seed: int = PERMUTATION_SEED) -> float:
    """Null: schema labels shuffled independently within each seed (row)."""
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(draws):
        shuffled = np.stack([row[rng.permutation(row.size)] for row in matrix])
        if mean_pairwise_tau(shuffled) >= observed - 1e-12:
            count += 1
    return (1 + count) / (1 + draws)


def stability(records: list[dict[str, Any]], schemas: list[Signature], draws: int = PERMUTATIONS
              ) -> dict[str, Any]:
    runs = [r for r in records if int(r["seed"]) in STABILITY_SEEDS]
    tables = [length2(r) for r in runs]
    present = [k for k in schemas if all(k in t and t[k]["rho"] is not None for t in tables)]
    out: dict[str, Any] = {"seeds": [int(r["seed"]) for r in runs], "schemas": [list(k) for k in present]}
    if len(present) < MIN_SCHEMAS or len(runs) < 2:
        return {**out, "status": "n/a", "reason": f"{len(present)} schemas, {len(runs)} seeds"}
    rho = np.array([[t[k]["rho"] for k in present] for t in tables], dtype=np.float64)
    cov = np.array([[t[k]["c"] for k in present] for t in tables], dtype=np.float64)
    observed = mean_pairwise_tau(rho)
    return {**out, "status": "computed", "mean_tau_b_rho": observed, "p_permutation": permutation_p(rho, observed,
                                                                                                    draws),
            "permutations": draws, "mean_tau_b_c": mean_pairwise_tau(cov)}


def literature_counts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs = [r for r in records if int(r["seed"]) in STABILITY_SEEDS]
    if not runs or runs[0].get("literature") is None:
        return []
    out = []
    for i, item in enumerate(runs[0]["literature"]):
        rhos = [r["literature"][i]["rho"] for r in runs]
        out.append({"listed": item["listed"], "relations": item["relations"], "seeds": len(runs),
                    "seeds_rho_above_1": sum(1 for v in rhos if v is not None and v > 1.0),
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
                 "schemas": [p[2] for p in points]}
        if len(points) < MIN_SPEARMAN_POINTS:
            out.append({**entry, "status": "n/a"})
            continue
        stat = spearmanr([p[0] for p in points], [p[1] for p in points])
        out.append({**entry, "status": "computed", "spearman": float(stat.statistic), "p": float(stat.pvalue)})
    return out


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
    ax.set_title(f"{dataset}: learned share / coverage, top length-2 schemas", fontsize=9)
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
            draws: int = PERMUTATIONS, make_figures: bool = True) -> dict[str, Any]:
    from stats import holm

    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"coverage_min": COVERAGE_MIN, "top": TOP, "permutations": draws,
                              "permutation_seed": PERMUTATION_SEED, "datasets": {}}
    for dataset, records in sorted(records_by_dataset.items()):
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
                                 "stability": stability(records, schemas, draws),
                                 "literature": literature_counts(records)}
        if data_root is not None:
            data, target, homophily = schema_homophily(data_root, dataset)
            entry["schema_homophily"] = homophily
            entry["spearman_rho_vs_homophily"] = spearman_link(records, data, target, homophily)
        if make_figures:
            entry["figures"] = figures(dataset, records, schemas, out_dir)
        result["datasets"][dataset] = entry
    family = {d: (e["stability"].get("p_permutation") if e["stability"]["status"] == "computed" else None)
              for d, e in result["datasets"].items() if d in ABLATION_DATASETS}
    adjusted = holm(family)
    for dataset, entry in result["datasets"].items():
        s = entry["stability"]
        if s["status"] != "computed":
            s["stable"] = None
            continue
        s["p_holm"] = adjusted.get(dataset)
        s["stable"] = bool(s["mean_tau_b_rho"] >= TAU_STABLE and s["p_holm"] is not None and s["p_holm"] < ALPHA)
    (out_dir / "q3_schema.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out_dir / "q3_schema.md").write_text(markdown(result), encoding="utf-8")
    return result


def _f(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def markdown(result: dict[str, Any]) -> str:
    lines = ["# Q3 schema read-out (card 7 V4; spec 1.27 C2b)", "",
             "| dataset | schemas | mean tau-b (rho) | p perm | p Holm | stable | mean tau-b (c) |",
             "|---|---|---|---|---|---|---|"]
    for d, e in result["datasets"].items():
        s = e["stability"]
        if s["status"] != "computed":
            lines.append(f"| {d} | {len(s['schemas'])} | n/a ({s['reason']}) | | | | |")
            continue
        lines.append(f"| {d} | {len(s['schemas'])} | {_f(s['mean_tau_b_rho'])} | {_f(s['p_permutation'])} | "
                     f"{_f(s.get('p_holm'))} | {_f(s['stable'])} | {_f(s['mean_tau_b_c'])} |")
    for d, e in result["datasets"].items():
        if e["literature"]:
            lines += ["", f"## {d}: literature meta-paths, seeds with rho > 1", "", "| meta-path | seeds | rho > 1 |",
                      "|---|---|---|"]
            lines += [f"| {x['listed']} | {x['seeds']} | {x['seeds_rho_above_1']} |" for x in e["literature"]]
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
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = parser.parse_args()
    analyse(load_records(args.runs), args.out_dir, args.data_root, args.permutations)


if __name__ == "__main__":
    main()
