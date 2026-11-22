"""Heterophily statistics of one (dataset, task, measure) (card 6, `docs/H2S-NN_heterophily_harness_card.md`).

    python scripts/heterophily_stats.py --dataset acm --task nc --measure h2 [--out-dir DIR]
    python scripts/heterophily_stats.py --collect [--out-dir DIR]        # stats.json + table.md from the records
    python scripts/heterophily_stats.py --dataset h-pokec --task nc --measure h2 --metapath-index I [--no-labeled-only]
    python scripts/heterophily_stats.py --dataset h-pokec --task nc --collect-h2-parts   # H2c-2: ℋ² from the parts

Measures: `h2` (ℋ², H2GB calcHomophily), `mlh` (Hetero²Net label_homophily on M₂), `mde` (Hetero²Net
attribute_homophily on M₂; HGB NC: reference and chunked values both, gate H0-eq), `task_h_edge` (LP target relation),
`gc_types` (GC type counts and type-mixing ratio), `plan` (path-instance counts per meta-path: the size bound of each
`G_P`, no measure), `hubs` (H2b: the hub of every meta-path, or none). One JSON record per call in `--out-dir/records/`: value(s), per-meta-path values, per-genre
values (IMDB), fill seeds and spread, labeled-only value, `M₂` used, code path and vendor commits, harness commit,
loader file:line, what symmetrisation did, peak host memory, wall time, job id and node. A value that cannot be
computed within the job's limits is written as `"value": null` with the reason (card, author 2026-11-17).

Formulas: `src/analysis/heterophily.py` (reference code called unchanged; conventions in its docstring).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.analysis import heterophily as het  # noqa: E402
from src.data.data_layout import REPO_DATA_ROOT  # noqa: E402

DEFAULT_OUT = ROOT.parent / "docs" / "heterophily_evidence"

HGB_NC = {"acm": ("ACM", "paper"), "dblp": ("DBLP", "author"), "imdb": ("IMDB", "movie"),
          "freebase": ("Freebase", "book")}
H2GB_NC = ("ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns")
#: local LP tasks: (Hetero²Net list key of the NC release it derives from, NC target type of that list)
LP_LOCAL = {"acm_pa": ("acm", "paper"), "acm_pp": ("acm", "paper"), "dblp_ap": ("dblp", "author"),
            "imdb_ma": ("imdb", "movie"),
            "freebase_bb": (None, None), "pubmed": (None, None), "lastfm": (None, None)}
LP_LARGE = {"ogbn-mag_pp": ("ogbn-mag", "paper"), "h-pokec_uu": (None, None)}
#: task-level H_edge (spec §1): the LP tasks whose two endpoint types carry labels, with the NC graph holding them
TASK_H_EDGE = {"freebase_bb": ("freebase", ("book", "and", "book")),
               "acm_pp": ("acm", ("paper", "cite", "paper")),  # card 15: both ends are labelled papers
               "ogbn-mag_pp": ("ogbn-mag", ("paper", "cites", "paper")),
               "h-pokec_uu": ("h-pokec", ("user", "has_friend", "user"))}
GC = {"mutag": "MUTAG", "proteins": "PROTEINS", "enzymes": "ENZYMES", "nci1": "NCI1"}

LOADERS = {
    "hgb": "src/data/dataset.py:31-52 (offline_hgb_dataset, src/data/offline.py:21: PyG HGBDataset)",
    "h2gb": "src/data/dataset.py:53-58 -> src/data/h2gb_nc.py:179 load_h2gb_nc (H2GB's preformat_*; seed 0)",
    "lp_local": "src/data/link_prediction_dataset.py:612 _load_local_hgb_link_data (.data: every relation's "
                "training facts; F17 node.dat)",
    "lp_large": "src/baselines/large_lp_minibatch.py:60 task_message_graph (held-out pairs removed) on "
                "src/data/h2gb_nc.py:179 load_h2gb_nc",
    "gc": "src/data/graph_classification_dataset.py:53 _load_tu_type_tensors on offline_tu_dataset "
          "(src/data/offline.py:46)",
}


class Deadline(Exception):
    pass


def _alarm(signum: int, frame: Any) -> None:
    raise Deadline()


# ----------------------------------------------------------------------------------------------- loading
def load_nc(dataset: str, data_root: Path) -> tuple[Any, str, str]:
    from src.data.data_layout import nc_root

    if dataset in HGB_NC:
        from src.data.offline import offline_hgb_dataset

        name, target = HGB_NC[dataset]
        return offline_hgb_dataset(nc_root(data_root), name)[0], target, LOADERS["hgb"]
    from src.data.h2gb_nc import ALL_H2GB_NC_SPECS, load_h2gb_nc

    return load_h2gb_nc(dataset, data_root, seed=0), ALL_H2GB_NC_SPECS[dataset].target_node_type, LOADERS["h2gb"]


def lp_config(task: str, data_root: Path) -> Any:
    from src.configs.config import TrainConfig

    return TrainConfig.from_dict({"data": {"dataset_name": task, "task_type": "link_prediction",
                                           "root": str(data_root)}})


def load_lp_training_graph(task: str, data_root: Path) -> tuple[Any, tuple[str, str, str], str]:
    if task in LP_LOCAL:
        from src.data.link_prediction_dataset import _load_local_hgb_link_data

        config = lp_config(task, data_root)
        return _load_local_hgb_link_data(config).data, tuple(config.data.target_relation), LOADERS["lp_local"]
    from src.baselines.large_lp_minibatch import read_task_files, task_message_graph
    from src.data.h2gb_nc import load_h2gb_nc
    from src.data.large_lp import LARGE_LP_TASKS

    spec = LARGE_LP_TASKS[task]
    data = load_h2gb_nc(spec.nc_dataset, data_root, seed=0)
    graph = task_message_graph(data, task, read_task_files(data_root, task))
    return graph, spec.relation, LOADERS["lp_large"]


# ----------------------------------------------------------------------------------------------- measures
def graph_summary(data: Any) -> dict[str, Any]:
    return {"node_types": {t: int(data[t].num_nodes) for t in data.node_types},
            "edge_types": {"-".join(e): int(data[e].edge_index.size(1)) for e in data.edge_types}}


def plan_counts(data: Any, target: str, list_key: str | None) -> dict[str, Any]:
    """Path instances per meta-path (sum over intermediate nodes of in-degree times out-degree): the bound on
    |E(G_P)| (equal when no node pair is joined by two instances), for H2GB's and Hetero²Net's sets."""
    def instances(steps_adj: list[Any]) -> int:
        a, b = steps_adj
        indeg = torch.bincount(a.storage.col(), minlength=a.size(1)).to(torch.float64)
        outdeg = torch.bincount(b.storage.row(), minlength=b.size(0)).to(torch.float64)
        return int((indeg * outdeg).sum())

    from torch_sparse import SparseTensor

    out: dict[str, Any] = {"hetero2net": {}, "h2gb": {}}
    paths, note = het.hetero2net_m2(list_key, data.edge_types, target)
    out["hetero2net_source"] = note
    for path in paths:
        out["hetero2net"][het.metapath_name(path)] = instances([het.step_adjacency(data, s, d) for s, d in path])
    for path in het.h2gb_metapaths(data.edge_types, target):
        adjs = []
        for e in path:
            ei = data[e].edge_index
            adjs.append(SparseTensor(row=ei[0], col=ei[1], sparse_sizes=(int(data[e[0]].num_nodes),
                                                                         int(data[e[2]].num_nodes))))
        out["h2gb"][het.metapath_name(path)] = instances(adjs)
    return out


def hub_survey(data: Any, target: str, list_key: str | None) -> dict[str, Any]:
    """H2b: the hub (or None) of every meta-path of both sets: a hub makes `G_P` complete (closed form)."""
    paths, note = het.hetero2net_m2(list_key, data.edge_types, target)
    return {"hetero2net_source": note,
            "hetero2net": {het.metapath_name(p): het.complete_gp_hub(data, p) for p in paths},
            "h2gb": {het.metapath_name(p): het.complete_gp_hub(data, p)
                     for p in het.h2gb_metapaths(data.edge_types, target)}}


def run(args: argparse.Namespace, record: dict[str, Any]) -> None:
    data_root = Path(args.data_root)
    measure, task, dataset = args.measure, args.task, args.dataset
    seeds = tuple(args.fill_seeds)
    if task == "gc":
        if measure != "gc_types":
            record.update(value=None, reason="GC: no heterophily value (spec 1.23)")
            return
        from src.data.data_layout import gc_root
        from src.data.graph_classification_dataset import _load_tu_type_tensors
        from src.data.offline import offline_tu_dataset

        name = GC[dataset]
        ds = offline_tu_dataset(gc_root(data_root), name)
        graphs = [ds[i] for i in range(len(ds))]
        node_types, _, n_nt, n_et = _load_tu_type_tensors(data_root, name, graphs)
        record["loader"] = LOADERS["gc"]
        record.update(het.gc_type_statistics(node_types, [g.edge_index for g in graphs], n_nt, n_et))
        record["value"] = record["type_mixing_ratio"]
        return

    if task == "nc":
        raw, target, loader = load_nc(dataset, data_root)
        list_key = dataset
    elif measure == "task_h_edge":
        if dataset not in TASK_H_EDGE:
            record.update(value=None, reason="not both endpoint types labeled (or no labels)")
            return
        nc_dataset, relation = TASK_H_EDGE[dataset]
        raw, target, loader = load_nc(nc_dataset, data_root)
        sym, record["symmetrisation"] = het.symmetrise_facts(raw)
        record.update(loader=loader + f" (labels of {nc_dataset}; full target relation: all task positives)",
                      relation="-".join(relation), labels_of=target)
        if dataset == "freebase_bb":
            record["relation_check"] = freebase_relation_check(raw, data_root)
        record.update(het.task_h_edge(sym[relation].edge_index, raw[target].y, seeds))
        return
    else:
        raw, relation, loader = load_lp_training_graph(dataset, data_root)
        record["target_relation"] = "-".join(relation)
        target = None
        list_key = None

    record["loader"] = loader
    sym, record["symmetrisation"] = het.symmetrise_facts(raw)
    record["graph"] = graph_summary(sym)

    if task == "lp":
        if measure not in ("mde", "hubs"):
            record.update(value=None, reason="LP: label measures are the task-level H_edge only (card scope)")
            return
        key, list_target = (LP_LOCAL | LP_LARGE)[dataset]
        endpoints = sorted({relation[0], relation[2]})
        record["per_endpoint_type"] = {}
        for t in endpoints:
            x = sym[t].get("x")
            if x is None:
                record["per_endpoint_type"][t] = {"value": None, "reason": "features removed (F17) or absent"}
                continue
            if measure == "hubs":
                record["per_endpoint_type"][t] = {"value": None,
                                                  "hubs": hub_survey(sym, t, key if t == list_target else None)}
                continue
            paths, note = het.hetero2net_m2(key if t == list_target else None, sym.edge_types, t)
            rec = het.mde_measure(sym, t, paths, x.float())
            rec.update(m2=[het.metapath_name(p) for p in paths], m2_source=note)
            record["per_endpoint_type"][t] = rec
        vals = None if measure == "hubs" else {t: r["value"] for t, r in record["per_endpoint_type"].items()}
        record["value"] = vals
        return

    y = raw[target].y
    record["target"] = target
    if measure == "plan":
        record.update(value=None, plan=plan_counts(sym, target, list_key))
        return
    if measure == "hubs":
        record.update(value=None, hubs=hub_survey(sym, target, list_key))
        return
    if measure == "h2":
        record.update(het.h2_measure(sym, target, y, seeds, labeled_only=not args.no_labeled_only,
                                     metapath_index=args.metapath_index))
        return
    paths, note = het.hetero2net_m2(list_key, sym.edge_types, target)
    record.update(m2=[het.metapath_name(p) for p in paths], m2_source=note)
    if measure == "mlh":
        record.update(het.mlh_measure(sym, target, paths, y, seeds))
    elif measure == "mde":
        x = sym[target].get("x")
        if x is None:
            record.update(value=None, reason="target type has no features")
            return
        record.update(het.mde_measure(sym, target, paths, x.float(), both=dataset in HGB_NC))
    else:
        raise ValueError(f"unknown measure {measure!r} for task {task!r}")


def freebase_relation_check(nc: Any, data_root: Path) -> dict[str, Any]:
    """The NC release's book-and-book facts vs the freebase_bb task's target relation (train + valid + test) as
    unordered pairs: the task-level H_edge is computed on the former with the NC labels."""
    from src.data.link_prediction_dataset import _load_local_hgb_link_data

    lp = _load_local_hgb_link_data(lp_config("freebase_bb", data_root))
    rel = ("book", "rel_0", "book")
    parts = [d[rel] for d in (lp.train_edge_index_by_relation, lp.valid_edge_index_by_relation,
                              lp.test_edge_index_by_relation) if rel in d]
    lp_index = torch.cat(parts, dim=1)

    def pairs(index: torch.Tensor) -> set[tuple[int, int]]:
        lo, hi = torch.minimum(index[0], index[1]), torch.maximum(index[0], index[1])
        return set(zip(lo.tolist(), hi.tolist()))

    nc_pairs, lp_pairs = pairs(nc["book", "and", "book"].edge_index), pairs(lp_index)
    book_raw = sorted(r for r, t in lp.raw_to_type_name.items() if t == "book")
    return {"nc_pairs": len(nc_pairs), "lp_pairs_train_valid_test": len(lp_pairs), "equal": nc_pairs == lp_pairs,
            "lp_book_local_ids_equal_raw_ids": book_raw == list(range(len(book_raw)))
            and all(lp.raw_to_local[r] == r for r in book_raw),
            "books_lp": len(book_raw), "books_nc": int(nc["book"].num_nodes)}


# ----------------------------------------------------------------------------------------------- collect
def _fmt(value: Any, digits: int = 4) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _cell(rec: dict[str, Any] | None, digits: int = 4) -> str:
    if rec is None:
        return "not run"
    if rec.get("value") is None:
        return f"– ({rec.get('reason', 'no value')})"
    text = _fmt(rec["value"], digits)
    if rec.get("spread"):
        text += f" [{_fmt(rec['spread'][0], digits)}, {_fmt(rec['spread'][1], digits)}]"
    if rec.get("labeled_only") is not None:
        text += f"; lab. {_fmt(rec['labeled_only'], digits)}"
    complete, total = _h2b_count(rec)
    if complete:
        text += f" (H2b closed form on {complete}/{total} G_P)"
    cliques = _h2c1_count(rec)
    if cliques:
        text += f" (H2c-1 clique closed form on {cliques}/{len(rec['per_metapath'])} G_P)"
    if rec.get("h2c_parts"):
        text += f" (H2c-2: {len(rec['h2c_parts'])} per-meta-path tasks; fill seed {rec.get('fill_seeds_used')})"
    return text


def _h2c1_count(rec: dict[str, Any]) -> int:
    """Meta-paths computed by the H2c-1 clique closed form."""
    return sum(1 for r in rec.get("per_metapath") or [] if r.get("gp_cliques"))


def _h2b_count(rec: dict[str, Any]) -> tuple[int, int]:
    """Meta-paths computed by the H2b closed form (complete `G_P`) and all meta-paths of a record."""
    if rec.get("h2b"):
        return len(rec["h2b"]["complete_gp"]), len(rec.get("h2gb_metapaths", []))
    rows = rec.get("per_metapath") or []
    return sum(1 for r in rows if r.get("gp_complete")), len(rows)


def _cost(rec: dict[str, Any] | None) -> str:
    return "" if rec is None or "wall_s" not in rec else f"{rec['wall_s'] / 3600:.2f} h / {rec['peak_rss_gb']:.1f} GB"


def collect(out_dir: Path) -> None:
    """`stats.json` (every record) and `table.md` (dataset × measure; '–' with the reason; cost per record)."""
    records = {}
    for path in sorted((out_dir / "records").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        records[(rec["task"], rec["dataset"], rec["measure"])] = rec
    (out_dir / "stats.json").write_text(json.dumps(list(records.values()), indent=1), encoding="utf-8")
    lines = ["# Heterophily statistics (card 6)", "",
             "Generated by `scripts/heterophily_stats.py --collect` from `records/`. Heterophily form (ℋ², MLH, task "
             "H_edge: 0 homophilic). `[a, b]` = range over the 5 fill seeds of unlabeled targets (primary value: "
             "first seed); `lab.` = labeled-only diagnostic. Cost = wall time / peak host memory of the record's job. "
             "ℋ² uses H2GB's meta-path rule, MLH and MDE Hetero²Net's `M₂`, so they may rest on different sets.", "",
             "## Node classification", "", "| Dataset | ℋ² | MLH | MDE | cost ℋ² | cost MLH | cost MDE |",
             "|---|---|---|---|---|---|---|"]
    for d in list(HGB_NC) + list(H2GB_NC):
        r = [records.get(("nc", d, m)) for m in ("h2", "mlh", "mde")]
        lines.append(f"| {d} | {_cell(r[0])} | {_cell(r[1])} | {_cell(r[2])} | "
                     f"{_cost(r[0])} | {_cost(r[1])} | {_cost(r[2])} |")
    lines += ["", "## Link prediction", "", "| Task | task-level H_edge | MDE per endpoint type (training graph) | "
              "cost H_edge | cost MDE |", "|---|---|---|---|---|"]
    for d in list(LP_LOCAL) + list(LP_LARGE):
        h, m = records.get(("lp", d, "task_h_edge")), records.get(("lp", d, "mde"))
        if m is None:
            mde = "not run"
        else:
            mde = "; ".join(f"{t}: {_cell(r)}" for t, r in m.get("per_endpoint_type", {}).items()) or _cell(m)
        lines.append(f"| {d} | {_cell(h)} | {mde} | {_cost(h)} | {_cost(m)} |")
    lines += ["", "## Graph classification (no heterophily value, spec 1.23)", "",
              "| Dataset | node types | edge types | type-mixing ratio (pooled) | mean per-graph ratio |",
              "|---|---|---|---|---|"]
    for d in GC:
        r = records.get(("gc", d, "gc_types"))
        if r is None:
            lines.append(f"| {d} | not run | | | |")
        else:
            lines.append(f"| {d} | {r['num_node_types']} | {r['num_edge_types']} | {_fmt(r['type_mixing_ratio'])} | "
                         f"{_fmt(r['mean_per_graph_ratio'])} |")
    (out_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------------------- driver
def git_head() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def record_path(out_dir: Path, args: argparse.Namespace) -> Path:
    if getattr(args, "metapath_index", None) is not None:
        name = f"{args.task}__{args.dataset}__{args.measure}__mp{args.metapath_index:02d}.json"
        return out_dir / "records" / "parts" / name
    return out_dir / "records" / f"{args.task}__{args.dataset}__{args.measure}.json"


def collect_h2_parts(out_dir: Path, task: str, dataset: str) -> Path:
    """H2c-2: the ℋ² record of `dataset` from its parts (`records/parts/`, one per meta-path index): `h2_collect`
    when every part has its rows; else value null with the reason and every part's time."""
    files = sorted((out_dir / "records" / "parts").glob(f"{task}__{dataset}__h2__mp*.json"))
    parts = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    first = parts[0]
    times = [{"metapath_index": int(f.stem.rsplit("mp", 1)[1]), "metapath": q.get("part", {}).get("metapath"),
              "job": q.get("job"), "node": q.get("node"), "harness_commit": q.get("harness_commit"),
              "wall_s": q.get("wall_s"), "peak_rss_gb": q.get("peak_rss_gb"),
              **({"reason": q["reason"]} if "part" not in q else {})} for f, q in zip(files, parts)]
    record: dict[str, Any] = {k: first.get(k) for k in ("dataset", "task", "harness_commit", "vendor_commits",
                                                        "loader", "symmetrisation", "graph", "target")}
    record.update(measure="h2", job=",".join(str(t["job"]) for t in times),
                  code="H2GB/calcHomophily.py calcHomophily (H2c-2: one meta-path per task, collected by h2_collect)",
                  fill_seeds_used=first.get("per_problem", {}).get("label", {}).get("seeds"), h2c_parts=times)
    failed = [t for t in times if "reason" in t]
    n = len(first.get("h2gb_metapaths") or [])
    if failed or len(parts) != n or n == 0:
        record.update(value=None, reason=f"H2c-2: {len(failed)} meta-path task(s) without a result "
                      f"({'; '.join(str(t['metapath_index']) + ': ' + t['reason'] for t in failed)}); "
                      f"{len(parts)} of {n} parts")
    else:
        record.update(het.h2_collect(parts))
    record.update(wall_s=max(t["wall_s"] or 0 for t in times), wall_s_sum=sum(t["wall_s"] or 0 for t in times),
                  peak_rss_gb=max(t["peak_rss_gb"] or 0 for t in times))
    path = out_dir / "records" / f"{task}__{dataset}__h2.json"
    path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    het.log(f"wrote {path}: value {record.get('value')}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset")
    parser.add_argument("--task", choices=["nc", "lp", "gc"])
    parser.add_argument("--measure", choices=["h2", "mlh", "mde", "task_h_edge", "gc_types", "plan", "hubs"])
    parser.add_argument("--data-root", default=str(REPO_DATA_ROOT))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fill-seeds", type=int, nargs="+", default=list(het.FILL_SEEDS))
    parser.add_argument("--deadline-h", type=float, default=11.5,
                        help="write a blank record with the reason when this wall time is reached")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--metapath-index", type=int, default=None,
                        help="H2c-2 (h2 only): that enumerated meta-path only; writes a part to records/parts/")
    parser.add_argument("--no-labeled-only", action="store_true", help="h2: skip the labeled-only diagnostic")
    parser.add_argument("--collect-h2-parts", action="store_true", help="H2c-2: the h2 record from its parts")
    args = parser.parse_args(argv)
    if args.collect:
        collect(args.out_dir)
        return 0
    if args.collect_h2_parts:
        if not (args.dataset and args.task):
            parser.error("--collect-h2-parts needs --dataset and --task")
        collect_h2_parts(args.out_dir, args.task, args.dataset)
        return 0
    if args.metapath_index is not None and args.measure != "h2":
        parser.error("--metapath-index is for --measure h2")
    if not (args.dataset and args.task and args.measure):
        parser.error("--dataset, --task and --measure are required (or --collect)")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads())))
    timer = het.Timer()
    record: dict[str, Any] = {
        "dataset": args.dataset, "task": args.task, "measure": args.measure,
        "harness_commit": git_head(), "job": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
        "threads": torch.get_num_threads(),
        "vendor_commits": {"hetero2net": het.vendor_commit("hetero2net"), "h2gb": het.vendor_commit("h2gb")},
        "code": {"h2": "H2GB/calcHomophily.py calcHomophily (H2b: closed form on a complete G_P"
                       + ("; H2c-2: one meta-path per task)" if args.metapath_index is not None else ")"),
                 "mlh": "hetero2net/homophily.py label_homophily (H2b: closed form on a complete G_P)",
                 "mde": "hetero2net/homophily.py attribute_homophily (D-a: mde_chunked above 2e6 edges; H2b: closed "
                        "form on a complete G_P; H2c-1: closed form on a disjoint union of cliques)",
                 "task_h_edge": "hetero2net/homophily.py label_homophily",
                 "gc_types": "src/analysis/heterophily.py gc_type_statistics",
                 "plan": "src/analysis/heterophily.py (instance counts)",
                 "hubs": "src/analysis/heterophily.py complete_gp_hub (H2b)"}[args.measure],
    }
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(args.deadline_h * 3600))
    try:
        run(args, record)
    except Deadline:
        record.update(value=None, reason=f"time limit: {args.deadline_h} h reached (job log has the last progress)")
    except MemoryError:
        record.update(value=None, reason="memory: allocation failed (MemoryError)")
    finally:
        signal.alarm(0)
    record.update(timer.stats())
    path = record_path(args.out_dir, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    het.log(f"wrote {path}: value {record.get('value')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
