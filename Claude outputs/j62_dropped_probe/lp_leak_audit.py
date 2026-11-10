"""One leak audit per link-prediction task (spec v1.10 F18; task card Part J, J6.2), as fixed in
`docs/lp_evidence/j6_preregistration.md`.

Fingerprint probe. For a column family F of an endpoint type X (non-binary columns: more than 2
distinct text values) and a cap c, R(x) = {(column, exact text value)} of x in F with the value not
a literal zero and held by at most c nodes of X in that column. On a HeaRT pair with x of type X
and y the other endpoint, s(x, y) = |R(x) & union of R(z) over z in N_tr(y)|, N_tr = training
neighbours through the target relation (link.dat minus link.dat.valid). Same-type tasks:
s(u, v) = shared(u, N_tr(v)) + shared(v, N_tr(u)).

Ctrl-G (training-graph control, feature-free): for every relation r incident to X in the training
message graph (validation target pairs removed in both orientations), g_r(x, y) =
sum over z in N_tr(y) of |N_r(x) & N_r(z)| (r = target: the bipartite L3 count); Ctrl-G per split
is the best r (highest MRR). Same-type tasks symmetrised.

Pass: every family, every cap in {5, 50, inf}, both splits: MRR - MRR_ctrl <= 0.01 and
Hits@10 - Hits@10_ctrl <= 0.02. Ranks as HeaRT's evaluator.

Run from H2S-NN/: python scripts/lp/lp_leak_audit.py [--tasks ...] [--positive-control]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lp_feature_leak import heart_rank, metrics  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPS = (5, 50, None)  # None = no cap
TOL_MRR, TOL_HITS = 0.01, 0.02
ZEROS = {"0", "0.0"}
CHUNK = 5_000  # pairs per block
MAX_NNZ = 2_000_000  # stored entries per block


# ------------------------------------------------------------------------- core (testable)
def fingerprint_matrix(values: np.ndarray, cap: int | None) -> sp.csr_matrix:
    """values: (n, k) array of exact text values. Rows: nodes; columns: (column, value) tokens
    that are not a literal zero and are held by at most `cap` nodes in that column."""
    n, k = values.shape
    rows, cols, offset = [], [], 0
    for j in range(k):
        uniq, inv, counts = np.unique(values[:, j], return_inverse=True, return_counts=True)
        keep = np.array([u not in ZEROS for u in uniq]) & (
            counts <= cap if cap is not None else True)
        sel = keep[inv]
        rows.append(np.flatnonzero(sel))
        cols.append(offset + inv[sel])
        offset += len(uniq)
    r = np.concatenate(rows) if rows else np.zeros(0, int)
    c = np.concatenate(cols) if cols else np.zeros(0, int)
    return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, max(offset, 1)))


def _rowdot_lazy(a: sp.csr_matrix, t: sp.csr_matrix, right: sp.csr_matrix, ai: np.ndarray,
                 bi: np.ndarray, binary: bool) -> np.ndarray:
    """Row-wise dot products a[ai[k]] . (t @ right)[bi[k]] (binarised if `binary`), computing
    t @ right only for the rows a block needs: the full two-hop matrix can be near dense
    (PubMed). Blocks of at most CHUNK pairs and about MAX_NNZ stored entries."""
    a, t, right = a.tocsr(), t.tocsr(), right.tocsr()
    # upper bound on the stored entries of (t @ right)[y], known before the product is formed,
    # so a block is shrunk before the product can exhaust memory (partitioning only)
    bound = t @ np.diff(right.indptr).astype(np.float64)
    out = np.empty(len(ai))
    lo = 0
    while lo < len(ai):
        hi = min(lo + CHUNK, len(ai))
        while hi - lo > 1:
            uy_ = np.unique(bi[lo:hi])
            if bound[uy_].sum() <= 4 * MAX_NNZ:
                break
            hi = lo + max(1, (hi - lo) // 2)
        while True:
            sl = slice(lo, hi)
            uy, inv = np.unique(bi[sl], return_inverse=True)
            b = (t[uy] @ right).tocsr()
            gathered = int(np.diff(b.indptr)[inv].sum())  # entries of b[inv], one row per pair
            if gathered <= MAX_NNZ or hi - lo <= 1:
                break
            hi = lo + max(1, (hi - lo) // 2)
        if binary:
            b.data[:] = 1.0
        out[sl] = np.asarray(a[ai[sl]].multiply(b[inv]).sum(axis=1)).ravel()
        lo = hi
    return out


def probe_scores(fp: sp.csr_matrix, t: sp.csr_matrix, xi: np.ndarray,
                 yi: np.ndarray) -> np.ndarray:
    """|R(x) & union R(z), z in N_tr(y)|; t: (Y x X) training target adjacency."""
    return _rowdot_lazy(fp, t, fp, xi, yi, binary=True)


def path_scores(a: sp.csr_matrix, t: sp.csr_matrix, xi: np.ndarray,
                yi: np.ndarray) -> np.ndarray:
    """sum over z in N_tr(y) of |N_r(x) & N_r(z)|; a: (X x W) adjacency of relation r."""
    return _rowdot_lazy(a, t, a, xi, yi, binary=False)


def rank_metrics(scores: np.ndarray, shape: tuple[int, int]) -> dict:
    s = scores.reshape(shape)
    return metrics(heart_rank(s[:, 0], s[:, 1:]))


def verdict(family: dict, ctrl: dict) -> dict:
    d_mrr = family["mrr"] - ctrl["mrr"]
    d_hits = family["hits@10"] - ctrl["hits@10"]
    return {"diff_mrr": d_mrr, "diff_hits@10": d_hits,
            "pass": bool(d_mrr <= TOL_MRR and d_hits <= TOL_HITS)}


# ------------------------------------------------------------------------- task data
@dataclass(frozen=True)
class AuditTask:
    folder: str          # under data/
    target: int
    same_type: bool


AUDIT_TASKS = {
    "acm_pa": AuditTask("lp_derived/acm_pa", 2, False),
    "dblp_ap": AuditTask("lp_derived/dblp_ap", 0, False),
    "imdb_ma": AuditTask("lp_derived/imdb_ma", 2, False),
    "freebase_bb": AuditTask("lp_derived/freebase_bb", 0, True),
    "pubmed": AuditTask("PubMed", 2, True),
    "lastfm": AuditTask("LastFM", 0, False),
}
#: IMDB movie's named non-binary attributes (release column -> info.dat name), J6.0.
IMDB_NAMED = {0: "num_critic_for_reviews", 1: "duration", 2: "director_facebook_likes",
              3: "actor_1_facebook_likes", 4: "actor_3_facebook_likes", 5: "gross",
              6: "num_voted_users", 7: "cast_total_facebook_likes", 8: "facenumber_in_poster",
              9: "num_user_for_reviews", 10: "budget", 11: "title_year",
              12: "actor_2_facebook_likes", 13: "imdb_score", 14: "aspect_ration",
              15: "movie_facebook_likes"}


def read_node_types(path: Path) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    ids: dict[int, list[int]] = {}
    with open(path, encoding="utf-8") as h:
        for line in h:
            p = line.split("\t", 3)
            ids.setdefault(int(p[2]), []).append(int(p[0]))
    sorted_ids = {t: np.array(sorted(v), dtype=np.int64) for t, v in ids.items()}
    type_of = {int(r): t for t, v in sorted_ids.items() for r in v}
    return sorted_ids, type_of


def read_type_columns(path: Path, type_id: int, ids: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Exact text values of the non-binary columns of one type (rows in raw-id order) and the
    file column index of each kept column."""
    rows: dict[int, list[str]] = {}
    with open(path, encoding="utf-8") as h:
        for line in h:
            p = line.rstrip("\r\n").split("\t")
            if int(p[2]) == type_id and len(p) == 4 and p[3]:
                rows[int(p[0])] = p[3].split(",")
    if not rows:
        return np.zeros((len(ids), 0), dtype=str), []
    width = len(next(iter(rows.values())))
    distinct = [set() for _ in range(width)]
    for vals in rows.values():
        for j, v in enumerate(vals):
            if len(distinct[j]) < 3:
                distinct[j].add(v)
    keep = [j for j in range(width) if len(distinct[j]) > 2]
    arr = np.array([[rows[int(r)][j] for j in keep] for r in ids])
    return arr, keep


def read_relations(folder: Path, type_of: dict[int, int]):
    rel: dict[int, list[tuple[int, int]]] = {}
    with open(folder / "link.dat", encoding="utf-8") as h:
        for line in h:
            s, o, r = line.split("\t")[:3]
            rel.setdefault(int(r), []).append((int(s), int(o)))
    valid = set()
    with open(folder / "link.dat.valid", encoding="utf-8") as h:
        for line in h:
            s, o, r = line.split("\t")[:3]
            valid.add((int(s), int(o)))
    types = {r: (type_of[p[0][0]], type_of[p[0][1]]) for r, p in rel.items()}
    return rel, valid, types


def adjacency(pairs, rows_ids, cols_ids, rows_first: bool) -> sp.csr_matrix:
    p = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    if not rows_first:
        p = p[:, ::-1]
    r = np.searchsorted(rows_ids, p[:, 0])
    c = np.searchsorted(cols_ids, p[:, 1])
    m = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(rows_ids), len(cols_ids)))
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


def audit_task(name: str, data_root: Path, positive_control: bool = False) -> dict:
    task = AUDIT_TASKS[name]
    folder = data_root / task.folder
    ids, type_of = read_node_types(folder / "node.dat")
    rel, valid, rtypes = read_relations(folder, type_of)
    s_type, d_type = rtypes[task.target]
    held = valid | {(b, a) for a, b in valid}
    tgt_types = {s_type, d_type}
    train = [p for p in rel[task.target] if p not in valid]
    out: dict = {"task": name, "target": task.target, "endpoint_types": [s_type, d_type],
                 "families": [], "ctrl_g": {}}
    # families: (type, label, column positions in the type's non-binary array)
    families = []
    endpoint_types = sorted(tgt_types)
    node_file = folder / "node.dat"
    if positive_control:
        rec = json.loads((folder / "feature_rule.json").read_text(encoding="utf-8"))
        node_file = Path(rec["release"]) / "node.dat"
    values_by_type = {}
    for t in endpoint_types:
        vals, keep = read_type_columns(node_file, t, ids[t])
        values_by_type[t] = vals
        if not keep:
            continue
        if name == "imdb_ma" and t == 0:
            rec = json.loads((folder / "feature_rule.json").read_text(encoding="utf-8"))
            removed = sorted(int(c) for c in rec["removed_columns"].get("0", {}))
            release_cols = [c for c in range(3489) if c not in removed]
            if positive_control:
                release_cols = list(range(3489))
            orig = [release_cols[j] for j in keep]
            if positive_control:
                cols = [i for i, c in enumerate(orig) if c in (3, 4, 7, 12)]
                families.append((t, "POSITIVE CONTROL: release columns 3,4,7,12", cols))
            else:
                for i, c in enumerate(orig):
                    if c in IMDB_NAMED:
                        families.append((t, f"col {c} {IMDB_NAMED[c]}", [i]))
                tok = [i for i, c in enumerate(orig) if c >= 148]
                families.append((t, f"keyword-token counts ({len(tok)} non-binary columns)", tok))
        elif not positive_control:
            families.append((t, f"type {t} block ({len(keep)} non-binary columns)",
                             list(range(len(keep)))))
    out["family_count"] = len(families)
    if not families:
        out["note"] = "no non-binary feature column on either endpoint type"
        return out
    heart = folder / "heart"
    lists = {}
    for split in ("valid", "test"):
        pos = np.load(heart / f"heart_{split}_pos.npy")
        neg = np.load(heart / f"heart_{split}_samples.npy")
        pairs = np.concatenate([pos[:, None, :], neg], axis=1).reshape(-1, 2)
        lists[split] = (pairs, (len(pos), neg.shape[1] + 1))

    def idx(t, raw):
        i = np.searchsorted(ids[t], raw)
        assert np.array_equal(ids[t][i], raw)
        return i

    # training target adjacency T (Y x X) for each endpoint type X
    def t_matrix(x_type):
        y_type = d_type if x_type == s_type else s_type
        if task.same_type:
            both = train + [(b, a) for a, b in train]
            return adjacency(both, ids[x_type], ids[x_type], True)
        return adjacency(train, ids[y_type], ids[x_type], s_type == y_type)

    # Ctrl-G relations incident to each endpoint type with families
    for x_type in sorted({f[0] for f in families}):
        tm = t_matrix(x_type)
        rel_scores = {}
        seen = set()
        for r, (a_t, b_t) in sorted(rtypes.items()):
            if x_type not in (a_t, b_t):
                continue
            w_type = b_t if a_t == x_type else a_t
            pairs = rel[r]
            if {a_t, b_t} == tgt_types:
                pairs = [p for p in pairs if p not in held]
            if a_t == b_t:
                pairs = pairs + [(b, a) for a, b in pairs]
            a_m = adjacency(pairs, ids[x_type], ids[w_type], a_t == x_type)
            key = (w_type, a_m.nnz, hash(a_m.indices.tobytes()), hash(a_m.indptr.tobytes()))
            if key in seen:
                continue
            seen.add(key)
            res = {}
            for split, (pairs_l, shape) in lists.items():
                if task.same_type:
                    u = idx(x_type, pairs_l[:, 0]); v = idx(x_type, pairs_l[:, 1])
                    g = path_scores(a_m, tm, u, v) + path_scores(a_m, tm, v, u)
                else:
                    col = 0 if s_type == x_type else 1
                    xi = idx(x_type, pairs_l[:, col])
                    yi = idx(d_type if col == 0 else s_type, pairs_l[:, 1 - col])
                    g = path_scores(a_m, tm, xi, yi)
                res[split] = rank_metrics(g, shape)
            rel_scores[f"r{r} (via type {w_type})"] = res
        best = {split: max(rel_scores.values(), key=lambda m: m[split]["mrr"])[split]
                for split in lists}
        best_name = {split: max(rel_scores, key=lambda k: rel_scores[k][split]["mrr"])
                     for split in lists}
        out["ctrl_g"][str(x_type)] = {"per_relation": rel_scores, "best": best,
                                      "best_relation": best_name}
        for f_type, label, cols in families:
            if f_type != x_type:
                continue
            vals = values_by_type[x_type][:, cols]
            fam = {"type": x_type, "family": label, "columns": len(cols), "caps": {}}
            for cap in CAPS:
                fp = fingerprint_matrix(vals, cap)
                cap_res = {}
                for split, (pairs_l, shape) in lists.items():
                    if task.same_type:
                        u = idx(x_type, pairs_l[:, 0]); v = idx(x_type, pairs_l[:, 1])
                        s = probe_scores(fp, tm, u, v) + probe_scores(fp, tm, v, u)
                    else:
                        col = 0 if s_type == x_type else 1
                        xi = idx(x_type, pairs_l[:, col])
                        yi = idx(d_type if col == 0 else s_type, pairs_l[:, 1 - col])
                        s = probe_scores(fp, tm, xi, yi)
                    m = rank_metrics(s, shape)
                    cap_res[split] = {**m, **verdict(m, best[split])}
                fam["caps"]["inf" if cap is None else str(cap)] = cap_res
            fam["pass"] = all(r["pass"] for c in fam["caps"].values() for r in c.values())
            out["families"].append(fam)
    out["pass"] = all(f["pass"] for f in out["families"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--tasks", nargs="+", default=list(AUDIT_TASKS), choices=list(AUDIT_TASKS))
    parser.add_argument("--positive-control", action="store_true",
                        help="imdb_ma only: probe the release movie columns 3,4,7,12")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    results = []
    for name in args.tasks:
        res = audit_task(name, args.data_root, args.positive_control)
        results.append(res)
        print(f"== {name}: families {res['family_count']}"
              + (f" ({res['note']})" if "note" in res else ""), flush=True)
        for x_type, c in res["ctrl_g"].items():
            for split in ("valid", "test"):
                b = c["best"][split]
                print(f"  Ctrl-G type {x_type} {split}: best {c['best_relation'][split]} "
                      f"MRR {b['mrr']:.4f} Hits@10 {b['hits@10']:.4f}")
        for f in res["families"]:
            worst = max(((cap, sp_, r) for cap, cr in f["caps"].items() for sp_, r in cr.items()),
                        key=lambda z: z[2]["diff_mrr"])
            print(f"  {'PASS' if f['pass'] else 'FAIL'} type {f['type']} {f['family']}: worst "
                  f"cap {worst[0]} {worst[1]} MRR {worst[2]['mrr']:.4f} (diff "
                  f"{worst[2]['diff_mrr']:+.4f}) Hits@10 {worst[2]['hits@10']:.4f} (diff "
                  f"{worst[2]['diff_hits@10']:+.4f})", flush=True)
        if res["families"]:
            print(f"  task {'PASS' if res['pass'] else 'FAIL'}", flush=True)
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
