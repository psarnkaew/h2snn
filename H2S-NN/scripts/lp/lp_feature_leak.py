"""Features-only leak measure on the HeaRT lists (task card Part J, J5.2, J5.3, J5.5; pre-registered
in `docs/lp_evidence/j5_preregistration.md`).

For a pair (x, y), x of an affected type with rule (source, aggregator, column map m) and y on the
other endpoint: s(x, y) = |supp(g_m(y)) & supp(f(x))| / |supp(g_m(y))| (0 when g_m(y) is empty):
the share of the candidate's contribution under the rule that the kept node's stored feature
already contains. Scored on each positive with its own 500 negatives; rank as HeaRT's evaluator
(`vendor/HeaRT/benchmarking/evalutors.py:108-112`): 0.5 (#neg >= pos + #neg > pos) + 1.

Ctrl-G (training-graph control) replaces supp(f(x)) by the union of supp(g_m(y')) over x's
**training** neighbours y', read here from the task's link.dat minus link.dat.valid (its own
parser; no stored feature of x is read). It is also the independent code path of the J5.3
bitwise check (`rule_on_training`).

Run from H2S-NN/: python scripts/lp/lp_feature_leak.py --tasks dblp_ap imdb_ma [--node-dat ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_feature_rules as fr  # noqa: E402  (rule definitions: sources and column maps)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def heart_rank(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """HeaRT's rank: mean of optimistic and pessimistic rank, 1-based."""
    ge = (neg >= pos[:, None]).sum(axis=1)
    gt = (neg > pos[:, None]).sum(axis=1)
    return 0.5 * (ge + gt) + 1.0


def metrics(rank: np.ndarray) -> dict:
    return {"mrr": float(np.mean(1.0 / rank)), "hits@10": float(np.mean(rank <= 10)),
            "lists": int(len(rank))}


def training_pairs(folder: Path, target: int) -> set[tuple[int, int]]:
    """Target-relation facts of link.dat that link.dat.valid does not list (raw ids)."""
    def rows(path: Path) -> set[tuple[int, int]]:
        out = set()
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                s, o, r = line.rstrip("\r\n").split("\t")[:3]
                if int(r) == target:
                    out.add((int(s), int(o)))
        return out
    return rows(folder / "link.dat") - rows(folder / "link.dat.valid")


def contribution(g: sp.csr_matrix, mapping: np.ndarray) -> sp.csr_matrix:
    """Presence of g_m(y) on x's columns (rows: y nodes in raw-id order)."""
    d = len(mapping)
    cols = np.flatnonzero(mapping >= 0)
    sub = g.tocsc()[:, mapping[cols]].tocoo()
    out = sp.csr_matrix((np.ones(sub.nnz), (sub.row, cols[sub.col])), shape=(g.shape[0], d))
    out.data[:] = 1.0
    out.sum_duplicates()
    out.data = (out.data != 0).astype(np.float64)
    out.eliminate_zeros()
    return out


def presence(x: sp.csr_matrix) -> sp.csr_matrix:
    p = x.tocsr().copy()
    p.data = (p.data != 0).astype(np.float64)
    p.eliminate_zeros()
    return p


def containment(x_pres: sp.csr_matrix, c: sp.csr_matrix, xi: np.ndarray,
                yi: np.ndarray) -> np.ndarray:
    inter = np.asarray(x_pres[xi].multiply(c[yi]).sum(axis=1)).ravel()
    size = np.asarray(c[yi].sum(axis=1)).ravel()
    return np.divide(inter, size, out=np.zeros_like(inter), where=size > 0)


def cosine(xa: sp.csr_matrix, xb: sp.csr_matrix, ai: np.ndarray, bi: np.ndarray) -> np.ndarray:
    num = np.asarray(xa[ai].multiply(xb[bi]).sum(axis=1)).ravel()
    na = np.sqrt(np.asarray(xa.multiply(xa).sum(axis=1)).ravel())[ai]
    nb = np.sqrt(np.asarray(xb.multiply(xb).sum(axis=1)).ravel())[bi]
    den = na * nb
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def chosen_rule(spec: fr.TaskSpec, type_id: int, source: str, aggregator: str):
    """(g, mapping, other type, release types) of one named candidate rule, from the release."""
    types = fr.read_nodes(spec.release)
    links = fr.read_links(spec.release / "link.dat")
    rel_types = fr.relation_types(spec.release / "info.dat")
    t_start, t_end = rel_types[spec.target]
    other = t_end if type_id == t_start else t_start
    sources = fr.vocabulary_sources(types, links, rel_types, type_id, other, spec.target,
                                    spec.keyword_type if other == 0 else None)
    g = sources[source]
    x = types[type_id].x
    identity = source in ("V1", "V1p") and g.shape[1] == x.shape[1]
    n = fr.neighbourhood(links[spec.target], rel_types[spec.target], types[type_id], types[other])
    agg = fr.aggregate(n, g, aggregator)
    mapping, amb, _, holds, _ = fr.evaluate_rule(x, agg, g, identity)
    return g, mapping, other, types, rel_types, int(holds.sum()), amb


def rule_on_training(train: set[tuple[int, int]], x_ids: np.ndarray, y_ids: np.ndarray,
                     x_is_start: bool, g: sp.csr_matrix, mapping: np.ndarray,
                     aggregator: str) -> np.ndarray:
    """The rule applied to each x node's training neighbours, node by node (dense, float64);
    a node with no training fact gets the zero vector. Independent of `aggregate`."""
    y_row = {int(r): i for i, r in enumerate(y_ids)}
    nbrs: dict[int, list[int]] = {}
    for s, o in train:
        xr, yr = (s, o) if x_is_start else (o, s)
        nbrs.setdefault(xr, []).append(y_row[yr])
    d = len(mapping)
    cols = np.flatnonzero(mapping >= 0)
    gc = g.tocsr()
    out = np.zeros((len(x_ids), d))
    for i, raw in enumerate(x_ids):
        ys = sorted(set(nbrs.get(int(raw), [])))
        if not ys:
            continue
        block = gc[ys][:, mapping[cols]].toarray()   # |N| x matched columns
        if aggregator == "or":
            vals = (block != 0).any(axis=0).astype(np.float64)
        elif aggregator == "sum":
            vals = block.sum(axis=0)
        elif aggregator == "mean":
            vals = block.sum(axis=0) / len(ys)
        elif aggregator == "max":
            vals = block.max(axis=0)
        else:
            raise ValueError(aggregator)
        out[i, cols] = vals
    return out


def measure(task: str, type_id: int, source: str, aggregator: str, node_dat_folder: Path,
            lp_root: Path, heart_subdir: str = "heart", with_cosine: bool = False) -> dict:
    spec = fr.task_spec(task, lp_root)
    g, mapping, other, release_types, rel_types, holds, amb = chosen_rule(spec, type_id, source,
                                                                          aggregator)
    stored = fr.read_nodes(node_dat_folder)
    t_start, _ = rel_types[spec.target]
    x_is_start = type_id == t_start
    x_ids = stored[type_id].ids
    y_ids = stored[other].ids
    if not (np.array_equal(x_ids, release_types[type_id].ids)
            and np.array_equal(y_ids, release_types[other].ids)):
        raise ValueError("node ids differ between the release and the task files")
    c = contribution(g, mapping)
    train = training_pairs(spec.folder, spec.target)
    ctrl = sp.csr_matrix(rule_on_training(train, x_ids, y_ids, x_is_start, g, mapping,
                                          aggregator))
    ctrl_pres = presence(ctrl)
    x_stored = stored[type_id].x
    out = {"task": task, "type": type_id, "rule": f"{source}/{aggregator}",
           "rule_nodes_holding_on_release": holds, "ambiguous_columns": amb,
           "stored_features": x_stored is not None, "splits": {}}
    heart = spec.folder / heart_subdir
    for split in ("valid", "test"):
        pos = np.load(heart / f"heart_{split}_pos.npy")
        neg = np.load(heart / f"heart_{split}_samples.npy")
        pairs = np.concatenate([pos[:, None, :], neg], axis=1).reshape(-1, 2)
        xs_raw, ys_raw = (pairs[:, 0], pairs[:, 1]) if x_is_start else (pairs[:, 1], pairs[:, 0])
        xi = np.searchsorted(x_ids, xs_raw)
        yi = np.searchsorted(y_ids, ys_raw)
        assert np.array_equal(x_ids[xi], xs_raw) and np.array_equal(y_ids[yi], ys_raw)
        shape = (len(pos), neg.shape[1] + 1)
        res = {}
        if x_stored is None:
            s = np.zeros(len(pairs))  # dropped type: no feature, every score ties
        else:
            s = containment(presence(x_stored), c, xi, yi)
        s = s.reshape(shape)
        res["scorer"] = metrics(heart_rank(s[:, 0], s[:, 1:]))
        sc = containment(ctrl_pres, c, xi, yi).reshape(shape)
        res["ctrl_g"] = metrics(heart_rank(sc[:, 0], sc[:, 1:]))
        if (with_cosine and x_stored is not None and stored[other].x is not None
                and x_stored.shape[1] == stored[other].x.shape[1]):  # a shared feature space
            cs = cosine(x_stored, stored[other].x, xi, yi).reshape(shape)
            res["cosine"] = metrics(heart_rank(cs[:, 0], cs[:, 1:]))
        res["diff_mrr"] = res["scorer"]["mrr"] - res["ctrl_g"]["mrr"]
        res["diff_hits@10"] = res["scorer"]["hits@10"] - res["ctrl_g"]["hits@10"]
        out["splits"][split] = res
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rules", type=Path, required=True,
                        help="JSON list of {task, type, source, aggregator}")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--lp-root", type=Path, default=PROJECT_ROOT / "data" / "lp_derived")
    parser.add_argument("--node-dat-root", type=Path, default=None,
                        help="folder holding <task>/node.dat to score (default: the task folder)")
    parser.add_argument("--heart-subdir", default="heart")
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    rules = json.loads(args.rules.read_text(encoding="utf-8"))
    results = []
    for rule in rules:
        if rule["task"] not in args.tasks:
            continue
        root = args.node_dat_root or args.lp_root
        res = measure(rule["task"], int(rule["type"]), rule["source"], rule["aggregator"],
                      root / rule["task"], args.lp_root, args.heart_subdir,
                      with_cosine=args.cosine)
        results.append(res)
        for split, r in res["splits"].items():
            line = (f"{res['task']} type {res['type']} {res['rule']} {split}: scorer MRR "
                    f"{r['scorer']['mrr']:.4f} Hits@10 {r['scorer']['hits@10']:.4f}; Ctrl-G MRR "
                    f"{r['ctrl_g']['mrr']:.4f} Hits@10 {r['ctrl_g']['hits@10']:.4f}; diff MRR "
                    f"{r['diff_mrr']:+.4f} Hits@10 {r['diff_hits@10']:+.4f}")
            if "cosine" in r:
                line += (f"; cosine MRR {r['cosine']['mrr']:.4f} Hits@10 "
                         f"{r['cosine']['hits@10']:.4f}")
            print(line, flush=True)
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
