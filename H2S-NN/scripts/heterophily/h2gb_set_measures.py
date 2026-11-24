"""MLH and MDE on H2GB's meta-path set (card 19, `docs/H2S-NN_card19_card.md`; spec 1.31 draft, option A).

    python scripts/heterophily/h2gb_set_measures.py --measure mlh --records DIR          # every nc__<ds>__h2.json
    python scripts/heterophily/h2gb_set_measures.py --measure mde --task nc --dataset acm [--metapath-index I]
    python scripts/heterophily/h2gb_set_measures.py --measure mde --task lp --dataset acm_pp [--metapath-index I]
    python scripts/heterophily/h2gb_set_measures.py --collect --task nc --dataset h-pokec  # join the per-path parts
    python scripts/heterophily/h2gb_set_measures.py --table                               # table.md + lp_table.md

Analysis only. Calls `src/analysis/heterophily.py` (as `het`) and card 6's `scripts/heterophily_stats.py` (loaders,
the same symmetrisation) unchanged; changes neither.

* MLH (no graph loaded): from card 6's ℋ² records, which store `calcHomophily`'s `h_edge` for every non-empty
  meta-path of H2GB's set per label problem and fill seed. MLH = 1 − mean `h_edge` over the rows of a fill seed
  (first seed: primary value); IMDB: mean over genres; labeled-only from the `labeled_only` rows.
* MDE: per H2GB meta-path P, the two-type graph `het.labeled_only_metapath_data(sym, P, target, all-True)` (its only
  meta-path is P, every target kept), skipped when its `G_P` is empty (`het._mini_gp_empty`, as `calcHomophily`
  skips it), else `het.mde_measure(mini, "tgt", [tgt→mid→tgt], x)` (reference / chunked / closed forms as card 6).
  MDE = mean over non-empty meta-paths. LP: per endpoint type with features, on the task's training graph.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.analysis import heterophily as het  # noqa: E402


def _stats_module() -> Any:
    """Card 6's `scripts/heterophily_stats.py` (not a package), loaded by path."""
    spec = importlib.util.spec_from_file_location("heterophily_stats", ROOT / "scripts" / "heterophily_stats.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stats = _stats_module()

CARD6_RECORDS = ROOT.parent / "docs" / "heterophily_evidence" / "records"
DEFAULT_OUT = ROOT.parent / "docs" / "heterophily_evidence" / "h2gb_set"
NC_ORDER = list(stats.HGB_NC) + list(stats.H2GB_NC)
LP_ORDER = ["acm_pp", "dblp_ap", "imdb_ma", "pubmed", "ogbn-mag_pp", "h-pokec_uu", "freebase_bb", "lastfm"]
#: LP gate sets (card G1): task -> endpoint type whose H2GB set must equal card 6's `m2` list
LP_GATES = {"dblp_ap": "paper", "imdb_ma": "movie"}
MINI_PATH = [[("tgt", "mid"), ("mid", "tgt")]]
STOP_EXIT = 3


class Deadline(Exception):
    pass


def _alarm(signum: int, frame: Any) -> None:
    raise Deadline()


class Stop(Exception):
    """A card stop condition (path enumeration or LP gate set differs from card 6)."""


def git_head() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return None if not vals else sum(vals) / len(vals)


# ----------------------------------------------------------------------------------------------- MDE: overlapping cliques
#: overlap closed form: subset occurrences allowed per level (memory bound; above it the path keeps D-a)
OVERLAP_MAX_OCC = 150_000_000
#: overlap closed form: rows of the per-subset feature sums held at once (float64)
OVERLAP_GROUP_ELEMENTS = 100_000_000


def overlap_targets(mini: Any) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """When step 2 of the mini graph is the transpose of step 1 (`G_P = A Aᵀ`, symmetric): the unique (target, mid)
    pairs sorted by target then mid, their per-target counts k, and the number of mid nodes. Else None."""
    (e1,), (e2,) = ([mini[e].edge_index for e in mini.edge_types if e[0] == s] for s in ("tgt", "mid"))
    n_mid = int(mini["mid"].num_nodes)
    key1 = torch.unique(e1[0] * n_mid + e1[1])
    key2 = torch.unique(e2[1] * n_mid + e2[0])
    if key1.numel() == 0 or not torch.equal(key1, key2):
        return None
    k = torch.bincount(key1 // n_mid, minlength=int(mini["tgt"].num_nodes))
    return key1, k, n_mid


def _level_subsets(key: torch.Tensor, k: torch.Tensor, n_mid: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Every s-subset S of every target's mid set: (target of each occurrence, subset id per occurrence); subset ids
    are 0..(distinct subsets − 1), equal for equal subsets."""
    from itertools import combinations

    mids = key % n_mid
    ptr = torch.zeros(k.numel() + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(k, 0)
    targets, rows = [], []
    for kk in torch.unique(k).tolist():
        if kk < s:
            continue
        who = (k == kk).nonzero().flatten()
        mat = mids[ptr[who].unsqueeze(1) + torch.arange(kk)]  # (n_k, kk), ascending
        comb = torch.tensor(list(combinations(range(kk), s)), dtype=torch.long)  # (C, s)
        rows.append(mat[:, comb].reshape(-1, s))
        targets.append(who.repeat_interleave(comb.size(0)))
    if not rows:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    sub = torch.cat(rows)
    if float(n_mid) ** s < 2 ** 62:
        code = torch.zeros(sub.size(0), dtype=torch.long)
        for j in range(s):
            code = code * n_mid + sub[:, j]
        _, inv = torch.unique(code, return_inverse=True)
    else:
        _, inv = torch.unique(sub, dim=0, return_inverse=True)
    return torch.cat(targets), inv


def overlap_clique_mde(mini: Any, x: torch.Tensor,
                       max_occ: int = OVERLAP_MAX_OCC) -> tuple[dict[str, Any] | None, str]:
    """Exact `attribute_homophily` / `mde_chunked` on `G_P = A Aᵀ` (two targets adjacent iff they share a mid node;
    a union of overlapping cliques, self-pairs kept) without building `G_P`. With F(p) the mid set of target p and
    n_S = #{q : S ⊆ F(q)}, inclusion–exclusion gives
      degree d_p = Σ_{∅≠S⊆F(p)} (−1)^{|S|+1} n_S,   Σ_{(r,c)∈G_P} y_r·y_c = Σ_S (−1)^{|S|+1} ‖Σ_{q ⊇ S} y_q‖²,
    and, `G_P` being symmetric (source degree = destination degree = d), with y = x/√d the energy sum is
    Σ ‖y_r − y_c‖²/4 = (2 Σ_{d_p>0} ‖x_p‖² − 2 Σ y_r·y_c)/4 over Σ d_p edges. Float64 throughout.
    Returns (row fields, "") or (None, reason) when it does not apply (then the path keeps its usual route)."""
    found = overlap_targets(mini)
    if found is None:
        return None, "step 2 is not the transpose of step 1"
    key, k, n_mid = found
    kmax = int(k.max())
    per_level = [int((torch.tensor([math.comb(kk, s) for kk in range(kmax + 1)], dtype=torch.float64)
                      * torch.bincount(k, minlength=kmax + 1).double()).sum()) for s in range(1, kmax + 1)]
    if max(per_level) > max_occ:
        return None, f"subset occurrences per level {max(per_level)} > {max_occ} (k max {kmax})"
    het.log(f"overlap closed form: k max {kmax}, occurrences per level {per_level}")
    n = int(k.numel())
    deg = torch.zeros(n, dtype=torch.long)
    for s in range(1, kmax + 1):
        tgt, inv = _level_subsets(key, k, n_mid, s)
        size = torch.bincount(inv)
        deg.index_add_(0, tgt, (1 if s % 2 else -1) * size[inv])
    edges = int(deg.sum())
    used = deg > 0
    x64 = x.to(torch.float64)
    sq = float(x64[used].square().sum())
    y = torch.zeros_like(x64)
    y[used] = x64[used] / deg[used].to(torch.float64).sqrt().unsqueeze(1)
    cross = 0.0
    block = max(1, OVERLAP_GROUP_ELEMENTS // max(1, x.size(1)))
    for s in range(1, kmax + 1):
        tgt, inv = _level_subsets(key, k, n_mid, s)
        order = torch.argsort(inv)
        tgt, inv = tgt[order], inv[order]
        n_groups = int(inv[-1]) + 1
        bounds = torch.searchsorted(inv, torch.arange(0, n_groups + block, block).clamp(max=n_groups))
        level = 0.0
        for g0, a, b in zip(range(0, n_groups, block), bounds[:-1].tolist(), bounds[1:].tolist()):
            sums = torch.zeros(min(block, n_groups - g0), x.size(1), dtype=torch.float64)
            sums.index_add_(0, inv[a:b] - g0, y[tgt[a:b]])
            level += float(sums.square().sum())
        cross += level if s % 2 else -level
    value = (2 * sq - 2 * cross) / 4 / edges
    return {"edges": edges, "value": value, "k_max": kmax, "occurrences_per_level": per_level,
            "code_path": "closed form (overlapping cliques: inclusion–exclusion over shared mid sets; card 19)"}, ""


# ----------------------------------------------------------------------------------------------- MLH
def mlh_from_h2_record(rec: dict[str, Any]) -> dict[str, Any]:
    """MLH (heterophily form) from one card 6 ℋ² record: per fill-seed index k, per label problem
    1 − mean `h_edge` over that run's rows (non-empty meta-paths), then the mean over problems (IMDB genres)."""
    per_problem = rec["per_problem"]
    names = list(per_problem)
    n_seeds = len(per_problem[names[0]]["seeds"])
    over_seeds = []
    for k in range(n_seeds):
        vals = []
        for p in names:
            rows = per_problem[p]["runs"][k]["per_metapath"]
            m = _mean([r["h_edge"] for r in rows])
            vals.append(None if m is None else 1 - m)
        over_seeds.append(_mean(vals))
    lab_vals = []
    for p in names:
        lab = per_problem[p].get("labeled_only")
        if lab:
            m = _mean([r.get("h_edge") for r in lab["per_metapath"]])
            lab_vals.append(None if m is None else 1 - m)
    paths = list(rec.get("h2gb_metapaths") or [])
    rows0 = per_problem[names[0]]["runs"][0]["per_metapath"]
    out: dict[str, Any] = {
        "value": over_seeds[0], "value_over_fill_seeds": over_seeds,
        "fill_seeds": {p: per_problem[p]["seeds"] for p in names},
        "spread": None if n_seeds < 2 else [min(over_seeds), max(over_seeds)],
        "labeled_only": _mean(lab_vals) if lab_vals else None,
        "h2gb_metapaths": paths, "paths": len(paths), "rows_first_seed": len(rows0),
        "empty_paths": len(paths) - len(rows0),
        "empty_paths_named": (rec.get("h2b") or {}).get("empty_gp_skipped"),
        "form": "heterophily: 1 - mean over H2GB's non-empty meta-paths (and genres) of calcHomophily h_edge",
    }
    if len(names) > 1:
        out["per_genre"] = {p: (lambda m: None if m is None else 1 - m)(
            _mean([r["h_edge"] for r in per_problem[p]["runs"][0]["per_metapath"]])) for p in names}
    return out


def run_mlh(records: Path, out_dir: Path, base: dict[str, Any]) -> list[Path]:
    written = []
    for path in sorted(records.glob("nc__*__h2.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        record = {**base, "dataset": rec["dataset"], "task": "nc", "measure": "mlh",
                  "source": f"docs/heterophily_evidence/records/{path.name} (card 6 job {rec.get('job')}, "
                            f"harness {rec.get('harness_commit')})",
                  "h2_value": rec.get("value"), **mlh_from_h2_record(rec)}
        written.append(write_record(out_dir, record))
    return written


# ----------------------------------------------------------------------------------------------- MDE
def card6_h2_paths(dataset: str) -> list[str]:
    rec = json.loads((CARD6_RECORDS / f"nc__{dataset}__h2.json").read_text(encoding="utf-8"))
    return list(rec["h2gb_metapaths"])


def type_sequence(name_or_path: Any) -> str:
    """Node-type sequence `a-b-a` of an H2GB relation path or a Hetero²Net step name/list."""
    if isinstance(name_or_path, str):
        return name_or_path
    path = [tuple(s) for s in name_or_path]
    if len(path[0]) == 2:
        return "-".join([path[0][0]] + [s[1] for s in path])
    return "-".join([path[0][0]] + [s[2] for s in path])


def mde_paths(data: Any, target: str, x: torch.Tensor, paths: list[list[tuple[str, str, str]]],
              chosen: list[int], both: bool, overlap: bool = False) -> list[dict[str, Any]]:
    """Per chosen H2GB meta-path: the mini-graph route (empty `G_P` skipped as `calcHomophily` does). `overlap`
    (`--overlap-cliques`): a `G_P = A Aᵀ` with no H2b hub and not disjoint cliques uses `overlap_clique_mde`
    instead of D-a (same value, no `G_P` built)."""
    everyone = torch.ones(int(data[target].num_nodes), dtype=torch.bool)
    rows = []
    for i in chosen:
        path = paths[i]
        name = het.metapath_name(path)
        het.log(f"MDE {target}: meta-path {i} {name}")
        mini = het.labeled_only_metapath_data(data, path, target, everyone)
        if het._mini_gp_empty(mini):
            rows.append({"metapath_index": i, "metapath": name, "empty": True, "value": None,
                         "code_path": "empty G_P: skipped (as calcHomophily)"})
            continue
        if overlap and not both and het.complete_gp_hub(mini, MINI_PATH[0]) is None \
                and het.clique_gp_mid(mini, MINI_PATH[0]) is None:
            fields, why = overlap_clique_mde(mini, x)
            if fields is not None:
                rows.append({"metapath_index": i, "metapath": name, "empty": False, **fields})
                continue
            het.log(f"overlap closed form not used: {why}")
        rec = het.mde_measure(mini, "tgt", MINI_PATH, x, both=both)
        row = dict(rec["per_metapath"][0])
        row.pop("metapath", None)
        row.pop("steps", None)
        rows.append({"metapath_index": i, "metapath": name, "empty": False, **row})
    return rows


def summarise_mde(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    return {"value": _mean([r["value"] for r in rows if not r["empty"]]),
            "form": "MDE, mean over H2GB's non-empty meta-paths (raw features; per-path mini graph)",
            "h2gb_metapaths": names, "paths": len(names),
            "empty_paths": sum(1 for r in rows if r["empty"]),
            "empty_paths_named": [r["metapath"] for r in rows if r["empty"]], "per_metapath": rows}


def run_mde(args: argparse.Namespace, record: dict[str, Any]) -> None:
    data_root = Path(args.data_root)
    chosen_of = (lambda n: [args.metapath_index] if args.metapath_index is not None else list(range(n)))
    if args.task == "nc":
        raw, target, loader = stats.load_nc(args.dataset, data_root)
        sym, record["symmetrisation"] = het.symmetrise_facts(raw)
        record.update(loader=loader, target=target, graph=stats.graph_summary(sym))
        paths = het.h2gb_metapaths(sym.edge_types, target)
        names = [het.metapath_name(p) for p in paths]
        card6 = card6_h2_paths(args.dataset)
        record["card6_h2_paths_equal"] = names == card6
        if names != card6:
            record.update(card6_h2_paths=card6)
            raise Stop(f"H2GB path enumeration differs from card 6 h2 record: {names} vs {card6}")
        x = sym[target].get("x")
        if x is None:
            record.update(value=None, reason="target type has no features", h2gb_metapaths=names)
            return
        rows = mde_paths(sym, target, x.float(), paths, chosen_of(len(paths)), both=args.dataset in stats.HGB_NC,
                         overlap=args.overlap_cliques)
        record.update(feature_dim=int(x.size(1)), **summarise_mde(rows, names))
        _part(args, record, len(paths))
        return

    raw, relation, loader = stats.load_lp_training_graph(args.dataset, data_root)
    sym, record["symmetrisation"] = het.symmetrise_facts(raw)
    record.update(loader=loader, target_relation="-".join(relation), graph=stats.graph_summary(sym))
    card6 = json.loads((CARD6_RECORDS / f"lp__{args.dataset}__mde.json").read_text(encoding="utf-8"))
    record["per_endpoint_type"] = {}
    for t in sorted({relation[0], relation[2]}):
        x = sym[t].get("x")
        if x is None:
            record["per_endpoint_type"][t] = {"value": None, "reason": "features removed (F17) or absent"}
            continue
        paths = het.h2gb_metapaths(sym.edge_types, t)
        names = [het.metapath_name(p) for p in paths]
        if LP_GATES.get(args.dataset) == t:
            old = card6["per_endpoint_type"][t]["m2"]
            new_seq = sorted(type_sequence(p) for p in paths)
            equal = new_seq == sorted(old)
            record["per_endpoint_type"][t] = {"gate_set": {"h2gb": names, "card6_m2": old, "equal": equal}}
            if not equal:
                raise Stop(f"LP gate set {args.dataset}/{t} differs from card 6 m2: {names} vs {old}")
        rows = mde_paths(sym, t, x.float(), paths, chosen_of(len(paths)), both=False, overlap=args.overlap_cliques)
        entry = record["per_endpoint_type"].get(t, {})
        entry.update(feature_dim=int(x.size(1)), **summarise_mde(rows, names))
        record["per_endpoint_type"][t] = entry
    record["value"] = {t: r.get("value") for t, r in record["per_endpoint_type"].items()}
    _part(args, record, None)


def _part(args: argparse.Namespace, record: dict[str, Any], n: int | None) -> None:
    if args.metapath_index is not None:
        record["part"] = {"metapath_index": args.metapath_index, "of": n}


def collect_parts(out_dir: Path, task: str, dataset: str, base: dict[str, Any]) -> Path:
    """The MDE record of `dataset` from its per-path parts (`records/parts/`, one per meta-path index)."""
    files = sorted((out_dir / "records" / "parts").glob(f"{task}__{dataset}__mde__mp*.json"))
    parts = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    first = parts[0]
    times = [{"metapath_index": int(f.stem.rsplit("mp", 1)[1]), "job": q.get("job"), "node": q.get("node"),
              "harness_commit": q.get("harness_commit"), "wall_s": q.get("wall_s"), "peak_rss_gb": q.get("peak_rss_gb"),
              **({"reason": q["reason"]} if q.get("reason") else {})} for f, q in zip(files, parts)]
    record: dict[str, Any] = {k: first.get(k) for k in ("dataset", "task", "measure", "loader", "symmetrisation",
                                                        "graph", "target", "target_relation", "vendor_commits")}
    record.update(base, job=",".join(str(t["job"]) for t in times), parts=times)
    failed = [t for t in times if "reason" in t]

    def join(entries: list[dict[str, Any]]) -> dict[str, Any]:
        names = entries[0]["h2gb_metapaths"]
        rows = sorted((r for e in entries for r in e["per_metapath"]), key=lambda r: r["metapath_index"])
        if [r["metapath_index"] for r in rows] != list(range(len(names))):
            raise ValueError("parts do not cover every meta-path index exactly once")
        return {"feature_dim": entries[0].get("feature_dim"), **summarise_mde(rows, names)}

    if failed:
        record.update(value=None, reason=f"{len(failed)} part(s) without a result: "
                      + "; ".join(f"{t['metapath_index']}: {t['reason']}" for t in failed))
    elif task == "nc":
        record.update(card6_h2_paths_equal=all(q.get("card6_h2_paths_equal") for q in parts), **join(parts))
    else:
        record["per_endpoint_type"] = {}
        for t, e in first["per_endpoint_type"].items():
            if e.get("value", 0) is None and "per_metapath" not in e:
                record["per_endpoint_type"][t] = e
                continue
            record["per_endpoint_type"][t] = join([q["per_endpoint_type"][t] for q in parts])
        record["value"] = {t: r.get("value") for t, r in record["per_endpoint_type"].items()}
    record.update(wall_s=max(t["wall_s"] or 0 for t in times), wall_s_sum=sum(t["wall_s"] or 0 for t in times),
                  peak_rss_gb=max(t["peak_rss_gb"] or 0 for t in times))
    path = out_dir / "records" / f"{task}__{dataset}__mde.json"
    path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    het.log(f"wrote {path}: value {record.get('value')}")
    return path


# ----------------------------------------------------------------------------------------------- tables
def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _f(v: Any, d: int = 4) -> str:
    if v is None:
        return "–"
    return f"{v:.3g}" if abs(v) < 1e-3 and v != 0 else f"{v:.{d}f}"


def table(out_dir: Path) -> Path:
    rec = out_dir / "records"
    lines = ["# Heterophily on H2GB's meta-path set (card 19, spec 1.31 option A)", "",
             "Generated by `scripts/heterophily/h2gb_set_measures.py --table`. ℋ² is copied from card 6. "
             "1.30 = card 6 (Hetero²Net lists / `get_metapaths`); 1.31 = H2GB's set (`calcHomophily` rule). "
             "MLH, ℋ²: heterophily form (first fill seed); MDE: raw features. Paths = H2GB meta-paths; empty = "
             "`G_P` empty (skipped, as `calcHomophily`).", "",
             "## Node classification", "",
             "| Dataset | ℋ² | MLH 1.30 | MLH 1.31 | MDE 1.30 | MDE 1.31 | paths | empty |",
             "|---|---|---|---|---|---|---|---|"]
    for d in NC_ORDER:
        h2 = _load(CARD6_RECORDS / f"nc__{d}__h2.json")
        m0, d0 = _load(CARD6_RECORDS / f"nc__{d}__mlh.json"), _load(CARD6_RECORDS / f"nc__{d}__mde.json")
        m1, d1 = _load(rec / f"nc__{d}__mlh.json"), _load(rec / f"nc__{d}__mde.json")
        mde1 = "not run" if d1 is None else _f(d1.get("value"))
        if d == "freebase":
            mde1 = "– (no features)"
        paths = (d1 or m1 or {}).get("paths", "")
        empty = (d1 or m1 or {}).get("empty_paths", "")
        lines.append(f"| {d} | {_f((h2 or {}).get('value'), 3)} | {_f((m0 or {}).get('value'))} | "
                     f"{_f((m1 or {}).get('value'))} | {_f((d0 or {}).get('value'))} | {mde1} | {paths} | {empty} |")
    lines += ["", "## Link prediction (appendix; training graph, per endpoint type with features)", "",
              "| Task | endpoint type | paths (empty) | MDE 1.30 (card 6 set) | MDE 1.31 |", "|---|---|---|---|---|"]
    for t in LP_ORDER:
        old, new = _load(CARD6_RECORDS / f"lp__{t}__mde.json"), _load(rec / f"lp__{t}__mde.json")
        for typ, o in (old or {}).get("per_endpoint_type", {}).items():
            n = (new or {}).get("per_endpoint_type", {}).get(typ)
            if o.get("value") is None and (n is None or n.get("value") is None):
                if t in ("freebase_bb", "lastfm") or o.get("reason"):
                    lines.append(f"| {t} | {typ} | – | – | – ({o.get('reason', 'no value')}) |")
                continue
            old_set = ", ".join(o.get("m2") or [])
            if n is None:
                lines.append(f"| {t} | {typ} | | {_f(o.get('value'))} ({old_set}) | not run |")
            else:
                lines.append(f"| {t} | {typ} | {n.get('paths')} ({n.get('empty_paths')}) | {_f(o.get('value'))} "
                             f"({old_set}) | {_f(n.get('value'))} |")
    path = out_dir / "table.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------------------------- gates
#: G2 (card): MLH on H2GB's set from the card 6 / Part J values, ±1e-4
G2_EXPECTED = {"acm": 0.2621, "dblp": 0.1278, "imdb": 0.3247, "freebase": 0.8464}
G1_NC = ("dblp", "imdb", "ieee-cis-g")
G1_RTOL = 1e-6


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def gates(out_dir: Path) -> dict[str, Any]:
    """G1 (MDE by the new route = card 6 MDE, rel. diff <= 1e-6) and G2 (MLH from the records) -> gates.json."""
    rec = out_dir / "records"
    out: dict[str, Any] = {"G1": [], "G2": []}
    for d in G1_NC:
        old, new = _load(CARD6_RECORDS / f"nc__{d}__mde.json"), _load(rec / f"nc__{d}__mde.json")
        row: dict[str, Any] = {"case": f"nc {d}", "card6": (old or {}).get("value"),
                               "card19": None if new is None else new.get("value")}
        row["rel_diff"] = None if None in (row["card6"], row["card19"]) else _rel(row["card19"], row["card6"])
        row["pass"] = row["rel_diff"] is not None and row["rel_diff"] <= G1_RTOL
        out["G1"].append(row)
    for task, t in LP_GATES.items():
        old, new = _load(CARD6_RECORDS / f"lp__{task}__mde.json"), _load(rec / f"lp__{task}__mde.json")
        o = (old or {}).get("per_endpoint_type", {}).get(t, {})
        n = (new or {}).get("per_endpoint_type", {}).get(t, {})
        row = {"case": f"lp {task} ({t})", "card6": o.get("value"), "card19": n.get("value"),
               "set_equal": (n.get("gate_set") or {}).get("equal")}
        row["rel_diff"] = None if None in (row["card6"], row["card19"]) else _rel(row["card19"], row["card6"])
        row["pass"] = bool(row["set_equal"]) and row["rel_diff"] is not None and row["rel_diff"] <= G1_RTOL
        out["G1"].append(row)
    for d, expected in G2_EXPECTED.items():
        m = _load(rec / f"nc__{d}__mlh.json")
        v = None if m is None else m.get("value")
        out["G2"].append({"dataset": d, "expected": expected, "card19": v,
                          "pass": v is not None and abs(v - expected) <= 1e-4})
    out["G1_pass"] = all(r["pass"] for r in out["G1"])
    out["G2_pass"] = all(r["pass"] for r in out["G2"])
    out["pass"] = out["G1_pass"] and out["G2_pass"]
    (out_dir / "gates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------------------------- driver
def record_path(out_dir: Path, task: str, dataset: str, measure: str, index: int | None) -> Path:
    if index is not None:
        return out_dir / "records" / "parts" / f"{task}__{dataset}__{measure}__mp{index:02d}.json"
    return out_dir / "records" / f"{task}__{dataset}__{measure}.json"


def write_record(out_dir: Path, record: dict[str, Any], index: int | None = None) -> Path:
    path = record_path(out_dir, record["task"], record["dataset"], record["measure"], index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    het.log(f"wrote {path}: value {record.get('value')}")
    return path


def base_record() -> dict[str, Any]:
    return {"harness_commit": git_head(), "job": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
            "threads": torch.get_num_threads(),
            "vendor_commits": {"hetero2net": het.vendor_commit("hetero2net"), "h2gb": het.vendor_commit("h2gb")},
            "script": "H2S-NN/scripts/heterophily/h2gb_set_measures.py (card 19)"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--measure", choices=["mlh", "mde"])
    parser.add_argument("--task", choices=["nc", "lp"], default="nc")
    parser.add_argument("--dataset")
    parser.add_argument("--records", type=Path, default=CARD6_RECORDS, help="mlh: card 6 records directory")
    parser.add_argument("--metapath-index", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--deadline-h", type=float, default=23.5)
    parser.add_argument("--overlap-cliques", action="store_true",
                        help="G_P = A Aᵀ paths: inclusion–exclusion closed form instead of D-a (author 2026-11-24)")
    parser.add_argument("--collect", action="store_true", help="join the per-path MDE parts of --dataset")
    parser.add_argument("--table", action="store_true", help="write table.md from the records")
    parser.add_argument("--gates", action="store_true", help="write gates.json (G1, G2) from the records")
    args = parser.parse_args(argv)
    if args.table:
        table(args.out_dir)
        return 0
    if args.gates:
        return 0 if gates(args.out_dir)["pass"] else 1
    timer = het.Timer()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads())))
    base = base_record()
    if args.collect:
        collect_parts(args.out_dir, args.task, args.dataset, base)
        return 0
    if args.measure == "mlh":
        run_mlh(args.records, args.out_dir, {**base, **timer.stats()})
        return 0
    if not (args.measure == "mde" and args.dataset):
        parser.error("--measure mde needs --dataset (or --measure mlh, --collect, --table)")
    if args.data_root is None:
        from src.data.data_layout import REPO_DATA_ROOT

        args.data_root = str(REPO_DATA_ROOT)
    record: dict[str, Any] = {**base, "dataset": args.dataset, "task": args.task, "measure": "mde",
                              "code": "per H2GB meta-path: labeled_only_metapath_data (all targets) + mde_measure "
                                      "(hetero2net attribute_homophily; D-a chunked; H2b/H2c-1 closed forms)"}
    record["overlap_cliques"] = bool(args.overlap_cliques)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(args.deadline_h * 3600))
    code = 0
    try:
        run_mde(args, record)
    except Stop as stop:
        record.update(value=None, reason=f"STOP: {stop}")
        code = STOP_EXIT
    except Deadline:
        record.update(value=None, reason=f"time limit: {args.deadline_h} h reached (job log has the last progress)")
    except MemoryError:
        record.update(value=None, reason="memory: allocation failed (MemoryError)")
    finally:
        signal.alarm(0)
    record.update(timer.stats())
    write_record(args.out_dir, record, args.metapath_index)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
