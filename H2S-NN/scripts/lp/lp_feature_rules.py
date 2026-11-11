"""Which link-prediction endpoint features encode the target relation (spec v1.9 F12; task card
Part J, J5.1), with the candidate rules fixed in `docs/lp_evidence/j5_preregistration.md`.

A node type x on the target relation is tested against the other endpoint y: is f(x) = A over
y in N(x) of g(y), where N(x) are x's target-relation neighbours in the release (all facts),
A in {or, sum, mean, max} and g(y) one of the vocabulary sources:
- V1   y's own features (and V1p: their presence, g != 0);
- V2:r y's indicator row over a third type through the non-target relation r;
- V3   (IMDB) token counts over the names of y's keywords (V3p presence; `cf` = case-folded).
Where g lives in x's feature space (V1/V1p, equal dimension) the column map is the identity;
otherwise every column of x is matched to the aggregated vocabulary column equal to it on every
node ("mapped vocabulary").

Read only: nothing is written except the report the caller asks for.
Run from H2S-NN/: python scripts/lp/lp_feature_rules.py --tasks acm_pa dblp_ap imdb_ma
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_layout import REPO_DATA_ROOT, lp_derived_root  # noqa: E402
AGGREGATORS = ("or", "sum", "mean", "max")
MEAN_TOLERANCE = 1e-12  # pre-registered: rounding only, for fractional (mean) values


@dataclass
class NodeType:
    type_id: int
    ids: np.ndarray            # raw ids, sorted
    names: list[str]
    feature_text: list[str]    # the fourth column, as stored ("" when absent)
    x: sp.csr_matrix | None    # parsed float64 features (sparse), or None


@dataclass
class Rule:
    """f(x) = aggregator over the target-relation neighbours of the mapped vocabulary g."""

    node_type: int
    other_type: int
    source: str
    aggregator: str
    mapping: np.ndarray        # per column of x: vocabulary column, -1 unmatched, -2 zero column
    ambiguous: int = 0
    columns_matched: int = 0
    nodes_holding: int = 0
    nodes: int = 0
    max_abs_diff: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def exact(self) -> bool:
        # every column matched (or all-zero), so the mapping alone rebuilds the feature
        return (self.nodes_holding == self.nodes and self.ambiguous == 0 and self.nodes > 0
                and self.columns_matched == len(self.mapping))


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def read_nodes(folder: Path) -> dict[int, NodeType]:
    rows: dict[int, list[tuple[int, str, str]]] = {}
    with open(folder / "node.dat", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            text = parts[3] if len(parts) == 4 else ""
            rows.setdefault(int(parts[2]), []).append((int(parts[0]), parts[1], text))
    types = {}
    for type_id, items in rows.items():
        items.sort()
        texts = [t for _, _, t in items]
        x = None
        if all(texts):
            x = sp.vstack([sp.csr_matrix(np.array(t.split(","), dtype=np.float64))
                           for t in texts]).tocsr()
        elif any(texts):
            raise ValueError(f"type {type_id} has partial features")
        types[type_id] = NodeType(type_id, np.array([i for i, _, _ in items], dtype=np.int64),
                                  [n for _, n, _ in items], texts, x)
    return types


def read_links(path: Path) -> dict[int, np.ndarray]:
    by_rel: dict[int, list[tuple[int, int]]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            s, o, r = line.rstrip("\r\n").split("\t")[:3]
            by_rel.setdefault(int(r), []).append((int(s), int(o)))
    return {r: np.array(p, dtype=np.int64).reshape(-1, 2) for r, p in by_rel.items()}


def relation_types(info_path: Path) -> dict[int, tuple[int, int]]:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    types = info["link.dat"]["link type"]
    return {int(r): (int(v["start"]), int(v["end"])) for r, v in types.items()}


def biadjacency(pairs: np.ndarray, rows: NodeType, cols: NodeType) -> sp.csr_matrix:
    """Binary, de-duplicated rows x cols matrix of the (row raw id, col raw id) pairs."""
    r = np.searchsorted(rows.ids, pairs[:, 0])
    c = np.searchsorted(cols.ids, pairs[:, 1])
    if not (np.array_equal(rows.ids[r], pairs[:, 0]) and np.array_equal(cols.ids[c], pairs[:, 1])):
        raise ValueError("pair endpoints are not of the expected node types")
    m = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(rows.ids), len(cols.ids)))
    m.data[:] = 1.0
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


def neighbourhood(pairs: np.ndarray, rel_types: tuple[int, int], x_type: NodeType,
                  y_type: NodeType) -> sp.csr_matrix:
    """N: x_type x y_type, from the facts `pairs` of a relation with endpoint types rel_types."""
    start, _ = rel_types
    if start == x_type.type_id:
        return biadjacency(pairs, x_type, y_type)
    return biadjacency(pairs[:, ::-1], x_type, y_type)


def tokens(name: str, casefold: bool) -> list[str]:
    words = name.replace("_", " ").split()
    return [w.casefold() for w in words] if casefold else words


def vocabulary_sources(types: dict[int, NodeType], links: dict[int, np.ndarray],
                       rel_types: dict[int, tuple[int, int]], x_type: int, y_type: int,
                       target: int, keyword_type: int | None) -> dict[str, sp.csr_matrix]:
    """The pre-registered g(y) matrices (rows = y nodes in raw-id order)."""
    y = types[y_type]
    sources: dict[str, sp.csr_matrix] = {}
    if y.x is not None:
        sources["V1"] = y.x.copy()
        presence = y.x.copy()
        presence.data = (presence.data != 0).astype(np.float64)
        presence.eliminate_zeros()
        sources["V1p"] = presence
    for r, (s, o) in sorted(rel_types.items()):
        if r == target or s != y_type or r not in links:
            continue
        if {s, o} == set(rel_types[target]):
            continue  # the target's reverse relation: the target facts, not a third type
        sources[f"V2:r{r}"] = biadjacency(links[r], y, types[o])
    if keyword_type is not None:
        kw = types[keyword_type]
        rel = [r for r, (s, o) in rel_types.items() if s == y_type and o == keyword_type]
        if len(rel) != 1:
            raise ValueError("expected one relation from the vocabulary type to keywords")
        y_kw = biadjacency(links[rel[0]], y, kw)
        for casefold in (False, True):
            vocab: dict[str, int] = {}
            r_idx, c_idx = [], []
            for k, name in enumerate(kw.names):
                for t in tokens(name, casefold):
                    r_idx.append(k)
                    c_idx.append(vocab.setdefault(t, len(vocab)))
            kw_tok = sp.csr_matrix((np.ones(len(r_idx)), (r_idx, c_idx)),
                                   shape=(len(kw.names), len(vocab)))
            counts = (y_kw @ kw_tok).tocsr()
            suffix = "cf" if casefold else ""
            sources[f"V3{suffix}"] = counts
            presence = counts.copy()
            presence.data[:] = 1.0
            sources[f"V3p{suffix}"] = presence
    return sources


def aggregate(n: sp.csr_matrix, g: sp.csr_matrix, aggregator: str) -> sp.csr_matrix:
    """A over y in N(x) of g(y), as an x-by-vocabulary sparse float64 matrix."""
    n = n.tocsr().astype(np.float64)
    deg = np.asarray(n.sum(axis=1)).ravel()
    if aggregator == "sum":
        return (n @ g).tocsr()
    if aggregator == "or":
        presence = g.copy()
        presence.data = (presence.data != 0).astype(np.float64)
        out = (n @ presence).tocsr()
        out.data = (out.data > 0).astype(np.float64)
        out.eliminate_zeros()
        return out
    if aggregator == "mean":
        total = (n @ g).tocsr()
        # divide each stored sum by the degree once (sum / |N(x)|), as the rule says
        rows = np.repeat(np.arange(total.shape[0]), np.diff(total.indptr))
        total.data = total.data / deg[rows]
        return total
    if aggregator == "max":
        x_of = np.repeat(np.arange(n.shape[0]), np.diff(n.indptr))
        gathered = g.tocsr()[n.indices].tocoo()
        xs = x_of[gathered.row]
        key = xs.astype(np.int64) * g.shape[1] + gathered.col
        order = np.argsort(key, kind="stable")
        key, vals, xs = key[order], gathered.data[order], xs[order]
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        group_max = np.maximum.reduceat(vals, starts)
        group_count = np.diff(np.r_[starts, len(key)])
        gx = xs[starts]
        # a neighbour with no stored entry contributes 0 to the max
        group_max = np.where(group_count < deg[gx], np.maximum(group_max, 0.0), group_max)
        out = sp.csr_matrix((group_max, (gx, key[starts] % g.shape[1])),
                            shape=(n.shape[0], g.shape[1]))
        out.eliminate_zeros()
        return out
    raise ValueError(aggregator)


def _column_keys(m: sp.csc_matrix, decimals: int) -> list[bytes]:
    keys = []
    for j in range(m.shape[1]):
        lo, hi = m.indptr[j], m.indptr[j + 1]
        rows = m.indices[lo:hi]
        vals = np.round(m.data[lo:hi], decimals)
        order = np.argsort(rows)
        keys.append(rows[order].tobytes() + b"|" + vals[order].tobytes())
    return keys


def _tolerance(x: sp.spmatrix) -> float:
    return 0.0 if np.all(x.data == np.round(x.data)) else MEAN_TOLERANCE


def evaluate_rule(x: sp.csr_matrix, aggregated: sp.csr_matrix, g: sp.csr_matrix,
                  identity: bool) -> tuple[np.ndarray, int, int, np.ndarray, float]:
    """(mapping, ambiguous, columns matched, per-node holds, max |diff| on matched columns)."""
    tol = _tolerance(x)
    d = x.shape[1]
    if identity:
        mapping = np.arange(d)
        diff = abs(aggregated.tocsr() - x).tocsr()
        bad = diff > tol
        holds = np.asarray(bad.sum(axis=1)).ravel() == 0
        col_ok = np.asarray(bad.sum(axis=0)).ravel() == 0
        return mapping, 0, int(col_ok.sum()), holds, float(diff.max()) if diff.nnz else 0.0
    xs = sp.csc_matrix(x)
    xs.eliminate_zeros()
    agg = aggregated.tocsc()
    agg.eliminate_zeros()
    decimals = 9
    by_key: dict[bytes, list[int]] = {}
    for v, key in enumerate(_column_keys(agg, decimals)):
        by_key.setdefault(key, []).append(v)
    g_csc = g.tocsc()
    mapping = np.full(d, -1, dtype=np.int64)
    ambiguous = 0
    x_keys = _column_keys(xs, decimals)
    for j in range(d):
        if xs.indptr[j + 1] == xs.indptr[j]:
            mapping[j] = -2
            continue
        cands = by_key.get(x_keys[j], [])
        col = xs[:, j].toarray().ravel()
        cands = [v for v in cands if np.abs(agg[:, v].toarray().ravel() - col).max() <= tol]
        if not cands:
            continue
        mapping[j] = cands[0]
        if len(cands) > 1:
            first = g_csc[:, cands[0]]
            if any((g_csc[:, v] != first).nnz for v in cands[1:]):
                ambiguous += 1
    matched = int((mapping != -1).sum())
    # per node: matched columns equal by construction; unmatched columns use the vocabulary
    # column with the largest per-node agreement among those overlapping the column's support
    chosen = mapping.copy()
    unmatched = np.flatnonzero(mapping == -1)
    if len(unmatched):
        support = (agg != 0).astype(np.float64).tocsc()
        x_sup = xs[:, unmatched].copy()
        x_sup.data = (x_sup.data != 0).astype(np.float64)
        overlap = (x_sup.T @ support).toarray()
        for k, j in enumerate(unmatched):
            best, best_agree = -3, -1  # -3: no overlapping column, compared with zero
            col = xs[:, j].toarray().ravel()
            for v in np.argsort(-overlap[k])[:3]:
                if overlap[k, v] == 0:
                    continue
                agree = int((np.abs(agg[:, v].toarray().ravel() - col) <= tol).sum())
                if agree > best_agree:
                    best, best_agree = int(v), agree
            chosen[j] = best
    holds = np.ones(x.shape[0], dtype=bool)
    max_diff = 0.0
    for lo in range(0, d, 512):
        cols = np.arange(lo, min(lo + 512, d))
        rebuilt = np.zeros((x.shape[0], len(cols)))
        use = chosen[cols] >= 0
        if use.any():
            rebuilt[:, use] = agg[:, chosen[cols][use]].toarray()
        diff = np.abs(rebuilt - xs[:, cols].toarray())
        holds &= (diff <= tol).all(axis=1)
        on_mapped = mapping[cols] >= 0
        if on_mapped.any():
            max_diff = max(max_diff, float(diff[:, on_mapped].max()))
    return mapping, ambiguous, matched, holds, max_diff


def _row_key(x: sp.csr_matrix, i: int) -> bytes:
    lo, hi = x.indptr[i], x.indptr[i + 1]
    order = np.argsort(x.indices[lo:hi])
    return x.indices[lo:hi][order].tobytes() + b"|" + x.data[lo:hi][order].tobytes()


def identical_neighbourhood_test(n: sp.csr_matrix, x: sp.csr_matrix, seed: int = 0) -> dict:
    groups: dict[bytes, list[int]] = {}
    for i in range(n.shape[0]):
        nb = n.indices[n.indptr[i]:n.indptr[i + 1]]
        if len(nb):
            groups.setdefault(np.sort(nb).tobytes(), []).append(i)
    pairs = [(g[0], o) for g in groups.values() if len(g) > 1 for o in g[1:]]
    x = x.tocsr()
    x.eliminate_zeros()
    same = sum(_row_key(x, a) == _row_key(x, b) for a, b in pairs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.shape[0], (5000, 2))
    rp = [(a, b) for a, b in idx if a != b]
    rand = sum(_row_key(x, a) == _row_key(x, b) for a, b in rp)
    return {"identical_set_pairs": len(pairs), "identical_features": same,
            "random_pairs": len(rp), "random_identical": rand}


@dataclass
class TaskSpec:
    name: str
    folder: Path            # data/lp_derived/<task>
    release: Path
    target: int
    keyword_type: int | None  # IMDB: the type whose names give the V3 tokens


def task_spec(name: str, lp_root: Path) -> TaskSpec:
    folder = lp_root / name
    meta = json.loads((folder / "split_meta.json").read_text(encoding="utf-8"))
    release = Path(meta["source"])
    for fname, digest in meta["source_md5"].items():
        if md5(release / fname) != digest:
            raise ValueError(f"{release / fname} differs from the release recorded at build time")
    keyword = 3 if name == "imdb_ma" else None
    return TaskSpec(name, folder, release, int(meta["target_relation"]), keyword)


def identify(spec: TaskSpec) -> dict:
    types = read_nodes(spec.release)
    links = read_links(spec.release / "link.dat")
    rel_types = relation_types(spec.release / "info.dat")
    t_start, t_end = rel_types[spec.target]
    report: dict = {"task": spec.name, "target": spec.target, "types": {}}
    for type_id, nt in sorted(types.items()):
        entry: dict = {"nodes": len(nt.ids),
                       "dim": None if nt.x is None else int(nt.x.shape[1])}
        if nt.x is None:
            entry["status"] = "no features"
        elif type_id not in (t_start, t_end):
            entry["status"] = "not on the target relation, not affected"
        else:
            other = t_end if type_id == t_start else t_start
            n = neighbourhood(links[spec.target], rel_types[spec.target], nt, types[other])
            entry["other_type"] = other
            entry["identical_neighbourhood"] = identical_neighbourhood_test(n, nt.x)
            sources = vocabulary_sources(types, links, rel_types, type_id, other, spec.target,
                                         spec.keyword_type if other == 0 else None)
            candidates = []
            for src_name, g in sources.items():
                identity = src_name in ("V1", "V1p") and g.shape[1] == nt.x.shape[1]
                for agg_name in AGGREGATORS:
                    agg = aggregate(n, g, agg_name)
                    mapping, amb, matched, holds, max_diff = evaluate_rule(nt.x, agg, g, identity)
                    rule = Rule(type_id, other, src_name, agg_name, mapping, amb, matched,
                                int(holds.sum()), len(holds), max_diff)
                    candidates.append({
                        "source": src_name, "aggregator": agg_name, "identity_map": identity,
                        "vocabulary": int(g.shape[1]), "columns_matched": matched,
                        "columns": int(nt.x.shape[1]), "ambiguous_columns": amb,
                        "zero_columns": int((mapping == -2).sum()),
                        "nodes_holding": rule.nodes_holding, "nodes": rule.nodes,
                        "max_abs_diff_matched": max_diff, "exact": rule.exact,
                        "mapping_md5": hashlib.md5(mapping.tobytes()).hexdigest(),
                    })
            candidates.sort(key=lambda c: (-c["nodes_holding"], -c["columns_matched"]))
            entry["candidates"] = candidates
            exact = [c for c in candidates if c["exact"]]
            ident = entry["identical_neighbourhood"]
            ident_rate = ident["identical_features"] / max(ident["identical_set_pairs"], 1)
            rand_rate = ident["random_identical"] / max(ident["random_pairs"], 1)
            best = candidates[0]
            affected = bool(exact) or (ident["identical_set_pairs"] > 0 and ident_rate >= 0.95
                                       and rand_rate <= 0.05) or \
                best["nodes_holding"] > 0.5 * best["nodes"]
            entry["affected"] = affected
            entry["exact_rules"] = [f"{c['source']}/{c['aggregator']}" for c in exact]
            entry["status"] = ("affected, exact rule" if exact else
                               "affected, no exact rule (drop)" if affected else "not affected")
        report["types"][str(type_id)] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", nargs="+", default=["acm_pa", "dblp_ap", "imdb_ma"])
    parser.add_argument("--lp-root", type=Path, default=lp_derived_root(REPO_DATA_ROOT))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    reports = []
    for name in args.tasks:
        rep = identify(task_spec(name, args.lp_root))
        reports.append(rep)
        print(f"== {name} (target relation {rep['target']})", flush=True)
        for type_id, e in rep["types"].items():
            print(f"type {type_id}: nodes {e['nodes']}, dim {e['dim']}: {e['status']}"
                  + (f"; exact: {e['exact_rules']}" if e.get("exact_rules") else ""), flush=True)
            if "identical_neighbourhood" in e:
                print(f"  identical-neighbourhood test: {e['identical_neighbourhood']}")
                for c in e["candidates"][:6]:
                    print(f"  {c['source']:>7}/{c['aggregator']:<4} cols {c['columns_matched']}/"
                          f"{c['columns']} (zero {c['zero_columns']}, ambiguous "
                          f"{c['ambiguous_columns']}) nodes {c['nodes_holding']}/{c['nodes']} "
                          f"max|diff| {c['max_abs_diff_matched']:.3g}"
                          + (" EXACT" if c["exact"] else ""), flush=True)
    if args.json_out:
        args.json_out.write_text(json.dumps(reports, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
