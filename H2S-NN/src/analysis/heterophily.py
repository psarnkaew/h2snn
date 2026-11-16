"""Heterophily harness (card 6, `docs/H2S-NN_heterophily_harness_card.md`; spec v1.24 §1 heterophily paragraphs).

Dataset descriptors only; no model is trained. Every value comes from the reference code, called unchanged:

* MLH and MDE: Hetero²Net `hetero2net/homophily.py` (vendor commit 4913394): `label_homophily` (`:16-24`),
  `attribute_homophily` (`:4-14`), `get_metapaths` (`:51-63`). `M₂` is Hetero²Net's predefined list where it has
  one (Hetero²Net Appendix B.1, as listed in spec §1: `HETERO2NET_LISTS`), else `get_metapaths`.
* ℋ²: H2GB `H2GB/calcHomophily.py` `calcHomophily` (vendor commit d3b49db) with its own `extract_metapath` rule and
  its own random fill of unlabeled targets (`:110-113`).

What this module adds around them (each a convention recorded in the output; none is a formula of a measure):

* `G_P` for the Hetero²Net measures (Hetero²Net's repository has no `G_P` builder): the construction of
  `calcHomophily` (`:119-150`): rows of `A₁` in chunks times `A₂`, the support of the product as the edge set, so a
  node pair appears once whatever its number of path instances, and self-pairs (u, u) are kept. A step `(src, dst)`
  of a Hetero²Net meta-path is the union of every stored relation from `src` to `dst` (Hetero²Net names steps by
  node types only).
* D-a (card): `attribute_homophily` loops over edges in Python; above `MDE_REFERENCE_MAX_EDGES` edges the harness
  uses `mde_chunked`, the same per-edge expression vectorised over chunks (gate H0-eq: equal to the reference to
  rtol 1e-5 on every HGB `G_P`). `label_homophily` is a mean over edges; above the same size it is called unchanged
  on edge chunks and the chunk means are combined weighted by chunk size (`LabelHomophily`).
* The random fill of the Hetero²Net label measures is `fill_like_h2gb`, a replica of `calcHomophily.py:110-113`
  (tested equal to the labels `calcHomophily` leaves in place).
* Labeled-only diagnostic: MLH on the `G_P` edges whose two endpoints are labeled; ℋ² by `calcHomophily` on
  `labeled_only_metapath_data`, a two-type graph whose only meta-path is the given one with both endpoints restricted
  to labeled targets (intermediate nodes unrestricted).

Forms: the reference functions return homophily; the reported MLH, ℋ² and task-level H_edge are heterophily
(`1 − mean_P h_P`, spec §1); each record keeps the per-meta-path homophily values as returned.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import resource
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import torch

#: D-a (card): the reference `attribute_homophily` (and a single `label_homophily` call) up to this many edges.
MDE_REFERENCE_MAX_EDGES = 2_000_000
#: rows of `A₁` per chunk of the `G_P` product (`calcHomophily`'s default `chunk_size`).
GP_CHUNK_ROWS = 200
#: edges per call of the chunked label computations.
EDGE_CHUNK = 1_000_000
#: feature elements gathered per chunk of the chunked MDE (bounds memory: 1.5e8 float32 = 0.6 GB per side).
MDE_CHUNK_ELEMENTS = 150_000_000
#: fill seeds of the unlabeled targets (spec §1: the fill seed is fixed and recorded, spread over 5 seeds); the first
#: one gives the primary value.
FILL_SEEDS = (0, 1, 2, 3, 4)
#: seconds between progress lines of the chunked `G_P` passes.
PROGRESS_EVERY_S = 300.0

#: Hetero²Net's predefined meta-path lists (Hetero²Net Appendix B.1 as listed in spec v1.24 §1), in Hetero²Net's
#: step form `(src, dst)` and the node-type names of the project's loaders (PyG HGBDataset; H2GB's MAG loader).
HETERO2NET_LISTS: dict[str, list[list[tuple[str, str]]]] = {
    "dblp": [[("author", "paper"), ("paper", "author")]],                                   # APA
    "acm": [[("paper", "author"), ("author", "paper")],                                     # PAP
            [("paper", "subject"), ("subject", "paper")],                                   # PSP
            [("paper", "term"), ("term", "paper")]],                                        # PTP
    "imdb": [[("movie", "director"), ("director", "movie")],                                # MDM
             [("movie", "actor"), ("actor", "movie")],                                      # MAM
             [("movie", "keyword"), ("keyword", "movie")]],                                 # MKM
    "ogbn-mag": [[("paper", "field_of_study"), ("field_of_study", "paper")],                # PFP
                 [("paper", "author"), ("author", "paper")]],                               # PAP
}

EdgeType = tuple[str, str, str]
Step = tuple[str, str]
ChunkSource = Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------------------------- reference code
_MODULES: dict[str, ModuleType] = {}


def _load(name: str, path: Any) -> ModuleType:
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        _MODULES[name] = module
    return _MODULES[name]


def hetero2net_homophily() -> ModuleType:
    """Hetero²Net's `hetero2net/homophily.py`, loaded from its file (never edited)."""
    from src.baselines.runtime.vendor_paths import vendor_path

    return _load("_card6_hetero2net_homophily", vendor_path("hetero2net") / "hetero2net" / "homophily.py")


def h2gb_calc_homophily() -> ModuleType:
    """H2GB's `H2GB/calcHomophily.py`, loaded from its file (never edited). Its tqdm bar goes to stderr; jobs set
    `TQDM_MININTERVAL` (one progress line per interval), tests set `TQDM_DISABLE`."""
    from src.baselines.runtime.vendor_paths import vendor_path

    return _load("_card6_h2gb_calcHomophily", vendor_path("h2gb") / "H2GB" / "calcHomophily.py")


def vendor_commit(name: str) -> str:
    import subprocess

    from src.baselines.runtime.vendor_paths import vendor_path

    done = subprocess.run(["git", "-C", str(vendor_path(name)), "rev-parse", "HEAD"], capture_output=True, text=True)
    return done.stdout.strip() or "unknown"


# ----------------------------------------------------------------------------------------------- meta-path sets
def hetero2net_m2(list_key: str | None, edge_types: Iterable[EdgeType], target: str) -> tuple[list[list[Step]], str]:
    """`M₂` of MLH and MDE: Hetero²Net's predefined list `list_key` if it has one, else `get_metapaths` (length 2,
    target to target, same-type relations skipped). `get_metapaths` names steps by node types, so two relations
    between the same types give the same step sequence more than once; a repeated sequence is kept once."""
    if list_key is not None and list_key in HETERO2NET_LISTS:
        return [list(p) for p in HETERO2NET_LISTS[list_key]], f"Hetero2Net predefined list '{list_key}' (App. B.1)"
    raw = hetero2net_homophily().get_metapaths(target, [tuple(e) for e in edge_types], [], [], 2, target)
    out: list[list[Step]] = []
    for path in raw:
        steps = [tuple(s) for s in path]
        if steps not in out:
            out.append(steps)  # type: ignore[arg-type]
    note = "get_metapaths (homophily.py:51-63)"
    if len(out) != len(raw):
        note += f"; {len(raw)} sequences returned, repeats kept once"
    return out, note


def metapath_name(path: Iterable) -> str:
    """`paper-author-paper` (Hetero²Net steps) or `paper-rev_writes-author-writes-paper` (H2GB relations)."""
    path = list(path)
    if len(path[0]) == 2:
        return "-".join([path[0][0]] + [s[1] for s in path])
    return "-".join([path[0][0]] + [f"{s[1]}-{s[2]}" for s in path])


# ----------------------------------------------------------------------------------------------- G_P
def step_relations(edge_types: Iterable[EdgeType], src: str, dst: str) -> list[EdgeType]:
    found = [tuple(e) for e in edge_types if e[0] == src and e[2] == dst]
    if not found:
        raise KeyError(f"no relation from {src!r} to {dst!r}")
    return found  # type: ignore[return-value]


def step_adjacency(data: Any, src: str, dst: str):
    """`SparseTensor` [n_src, n_dst] of the union of every relation from `src` to `dst` (support only)."""
    from torch_sparse import SparseTensor

    parts = [data[e].edge_index for e in step_relations(data.edge_types, src, dst)]
    index = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    adj = SparseTensor(row=index[0], col=index[1], value=None,
                       sparse_sizes=(int(data[src].num_nodes), int(data[dst].num_nodes)))
    return adj.coalesce()


def iter_gp(data: Any, path: list[Step], chunk_rows: int = GP_CHUNK_ROWS,
            label: str = "") -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Edges `(row, col)` of `G_P` in row chunks, built as `calcHomophily.py:119-150` builds them: rows of `A₁` in
    chunks times the rest of the path; the product's support is the edge set (one edge per node pair, self-pairs
    kept). `path` is a Hetero²Net step list `[(t0, t1), (t1, t2), ...]`. Logs progress every `PROGRESS_EVERY_S`."""
    adjs = [step_adjacency(data, s, d) for s, d in path]
    rest = adjs[1]
    for adj in adjs[2:]:
        rest = rest @ adj
    first = adjs[0]
    n_rows = first.size(0)
    start, last, edges = time.time(), time.time(), 0
    for i in range(0, n_rows, chunk_rows):
        end = min(i + chunk_rows, n_rows)
        row, col, _ = (first[i:end] @ rest).coo()
        edges += int(row.numel())
        if time.time() - last > PROGRESS_EVERY_S:
            last = time.time()
            log(f"G_P {label or metapath_name(path)}: rows {end}/{n_rows} ({100 * end / n_rows:.1f} %), "
                f"edges so far {edges}, {last - start:.0f} s")
        yield row + i, col


def gp_edges_capped(data: Any, path: list[Step], cap: int) -> torch.Tensor | None:
    """The whole edge index of `G_P` if it has at most `cap` edges, else None (stops as soon as `cap` is passed)."""
    rows, cols, total = [], [], 0
    for row, col in iter_gp(data, path):
        total += int(row.numel())
        if total > cap:
            return None
        rows.append(row)
        cols.append(col)
    if not rows:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.stack([torch.cat(rows), torch.cat(cols)])


def gp_edge_index(data: Any, path: list[Step]) -> torch.Tensor:
    index = gp_edges_capped(data, path, cap=2 ** 62)
    assert index is not None
    return index


def chunks_of(edge_index: torch.Tensor, size: int = EDGE_CHUNK) -> ChunkSource:
    """A re-iterable chunk source over a stored edge index."""
    def source() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for start in range(0, edge_index.size(1), size):
            yield edge_index[0, start:start + size], edge_index[1, start:start + size]
    return source


# ----------------------------------------------------------------------------------------------- labels
def labeled_mask(y: torch.Tensor) -> torch.Tensor:
    """Labeled targets: `y != -1` (single label); a non-zero row (multilabel: PyG stores IMDB's unlabeled movies as
    all-zero rows)."""
    return (y.sum(dim=1) > 0) if y.dim() == 2 else (y != -1)


def fill_like_h2gb(y: torch.Tensor, seed: int) -> torch.Tensor:
    """The labels `calcHomophily` computes on after `torch.manual_seed(seed)`: replica of `calcHomophily.py:110-113`
    applied to a clone (every -1 gets `torch.randint(0, max + 1)`). Tested equal to `calcHomophily`'s in-place fill."""
    label = y.clone()
    torch.manual_seed(seed)
    num_classes = label.max() + 1
    mask = label == -1
    label[mask] = torch.randint(0, num_classes, size=(mask.sum(),), device=label.device)
    return label


def genre_labels(y: torch.Tensor) -> list[torch.Tensor]:
    """IMDB per genre (spec §1 multilabel): `1(g ∈ Y_v)` for labeled movies, -1 for unlabeled ones."""
    keep = labeled_mask(y)
    out = []
    for g in range(y.size(1)):
        label = y[:, g].long().clone()
        label[~keep] = -1
        out.append(label)
    return out


def genre_fill_seed(fill_seed: int, genre: int) -> int:
    """Fill seed of one genre: distinct per genre, so an unlabeled movie is not filled identically in every genre."""
    return 10 * fill_seed + genre


def label_tasks(y: torch.Tensor, fill_seeds: Iterable[int] = FILL_SEEDS) -> list[dict[str, Any]]:
    """The single-label problems a label measure is computed on: one (single label) or one per genre (multilabel),
    each with its raw labels (-1 = unlabeled) and its fill seeds (only the first when every target is labeled: the
    fill is then a no-op)."""
    fill_seeds = list(fill_seeds)
    problems = [("label", y.long(), None)] if y.dim() == 1 else [
        (f"genre_{g}", lab, g) for g, lab in enumerate(genre_labels(y))]
    out = []
    for name, lab, genre in problems:
        seeds = fill_seeds if bool((lab == -1).any()) else fill_seeds[:1]
        seeds = [s if genre is None else genre_fill_seed(s, genre) for s in seeds]
        out.append({"name": name, "labels": lab, "seeds": seeds})
    return out


# ----------------------------------------------------------------------------------------------- MLH
@dataclass
class LabelHomophily:
    """Hetero²Net's `label_homophily` over a `G_P` given whole (one call) or in edge chunks (one call per chunk; the
    chunk means combined weighted by the chunk's edge count, which is the mean over all edges)."""

    weighted_sum: float = 0.0
    edges: int = 0
    calls: int = 0

    def add(self, row: torch.Tensor, col: torch.Tensor, y: torch.Tensor) -> None:
        if row.numel() == 0:
            return
        value = hetero2net_homophily().label_homophily(torch.stack([row, col]), y)
        self.weighted_sum += float(value) * int(row.numel())
        self.edges += int(row.numel())
        self.calls += 1

    @property
    def value(self) -> float | None:
        return None if self.edges == 0 else self.weighted_sum / self.edges


def label_homophily_chunked(edge_index: torch.Tensor, y: torch.Tensor, chunk: int = EDGE_CHUNK) -> float | None:
    acc = LabelHomophily()
    for row, col in chunks_of(edge_index, chunk)():
        acc.add(row, col, y)
    return acc.value


def _mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return None if not vals else sum(vals) / len(vals)


def mlh_measure(data: Any, target: str, paths: list[list[Step]], y: torch.Tensor,
                fill_seeds: Iterable[int] = FILL_SEEDS, max_single_call: int = MDE_REFERENCE_MAX_EDGES,
                chunk: int = EDGE_CHUNK) -> dict[str, Any]:
    """MLH (spec §1) on `paths`: per label problem (IMDB: per genre) and fill seed, `label_homophily` on each `G_P`
    with H2GB's fill; MLH = 1 − mean over meta-paths with a non-empty `G_P`; multilabel: mean over genres. Also the
    labeled-only diagnostic (edges with both endpoints labeled; raw labels) and, for multilabel, Hetero²Net's
    exact-match rule on the label vectors as loaded (reproduction field)."""
    problems = label_tasks(y, fill_seeds)
    keep = labeled_mask(y)
    any_unlabeled = not bool(keep.all())
    filled = {(p["name"], s): fill_like_h2gb(p["labels"], s) for p in problems for s in p["seeds"]}
    per_path = []
    for path in paths:
        accs = {key: LabelHomophily() for key in filled}
        lab_accs = {p["name"]: LabelHomophily() for p in problems} if any_unlabeled else {}
        exact = LabelHomophily() if y.dim() == 2 else None

        def feed(row: torch.Tensor, col: torch.Tensor) -> None:
            for key, acc in accs.items():
                acc.add(row, col, filled[key])
            if lab_accs:
                both = keep[row] & keep[col]
                for p in problems:
                    lab_accs[p["name"]].add(row[both], col[both], p["labels"])
            if exact is not None:
                exact.add(row, col, y)

        whole = gp_edges_capped(data, path, max_single_call)
        if whole is not None:
            mode = "single call"
            feed(whole[0], whole[1])
            edges = int(whole.size(1))
        else:
            mode = "chunked (weighted chunk means)"
            edges = 0
            for row, col in iter_gp(data, path):
                edges += int(row.numel())
                for start in range(0, row.numel(), chunk):
                    feed(row[start:start + chunk], col[start:start + chunk])
        per_path.append({
            "metapath": metapath_name(path), "steps": [list(s) for s in path], "edges": edges, "mode": mode,
            "h_edge_homophily": {f"{n}@{s}": acc.value for (n, s), acc in accs.items()},
            "labeled_only_h_edge_homophily": {n: acc.value for n, acc in lab_accs.items()} or None,
            "labeled_only_edges": {n: acc.edges for n, acc in lab_accs.items()} or None,
            "exact_match_h_edge_homophily": None if exact is None else exact.value,
        })
    n_seeds = len(problems[0]["seeds"])
    over_seeds = []
    for k in range(n_seeds):
        per_problem = [1 - m for m in (_mean(r["h_edge_homophily"][f"{p['name']}@{p['seeds'][k]}"] for r in per_path)
                                       for p in problems) if m is not None]
        over_seeds.append(_mean(per_problem))
    out: dict[str, Any] = {
        "value": over_seeds[0], "form": "heterophily: 1 - mean over M2 (and genres) of label_homophily",
        "fill_seeds": {p["name"]: p["seeds"] for p in problems}, "value_over_fill_seeds": over_seeds,
        "spread": None if len(over_seeds) < 2 else [min(over_seeds), max(over_seeds)],
        "per_metapath": per_path, "labeled_targets": int(keep.sum()), "targets": int(keep.numel()),
    }
    if any_unlabeled:
        per_problem = [_mean(r["labeled_only_h_edge_homophily"][p["name"]] for r in per_path) for p in problems]
        out["labeled_only"] = _mean(1 - m for m in per_problem if m is not None)
    if y.dim() == 2:
        out["per_genre"] = {}
        for p in problems:
            m = _mean(r["h_edge_homophily"][f"{p['name']}@{p['seeds'][0]}"] for r in per_path)
            out["per_genre"][p["name"]] = None if m is None else 1 - m
        exact_mean = _mean(r["exact_match_h_edge_homophily"] for r in per_path)
        out["reproduction_exact_match"] = {
            "rule": "Hetero2Net label_homophily on the multilabel vectors as loaded (all genres equal; unlabeled "
                    "movies = all-zero rows, no fill); reproduction field only, never the reported MLH",
            "homophily": exact_mean, "heterophily": None if exact_mean is None else 1 - exact_mean}
    return out


# ----------------------------------------------------------------------------------------------- MDE
def mde_reference(edge_index: torch.Tensor, x: torch.Tensor) -> float:
    return float(hetero2net_homophily().attribute_homophily(edge_index, x))


def mde_chunked(source: ChunkSource, x: torch.Tensor, chunk: int | None = None) -> tuple[float | None, int]:
    """D-a: `attribute_homophily` (`homophily.py:4-14`) vectorised: the same per-edge expression
    `‖x_r/√deg_src(r) − x_c/√deg_dst(c)‖² / 4` with `degree(...).clamp(min=1)` (degrees over the whole `G_P`, float32
    as the reference), energies in float32, mean over all edges (accumulated in float64; the reference takes the
    float32 mean of the float32 energies). `source()` yields `(row, col)` chunks; it is iterated twice (degrees, then
    energies). Returns (value, edges)."""
    n = x.size(0)
    if chunk is None:
        chunk = max(1024, MDE_CHUNK_ELEMENTS // max(1, x.size(1)))
    src_count = torch.zeros(n, dtype=torch.float64)
    dst_count = torch.zeros(n, dtype=torch.float64)
    edges = 0
    for row, col in source():
        src_count += torch.bincount(row, minlength=n).to(torch.float64)
        dst_count += torch.bincount(col, minlength=n).to(torch.float64)
        edges += int(row.numel())
    if edges == 0:
        return None, 0
    src_scale = src_count.to(torch.float32).clamp(min=1).sqrt().unsqueeze(1)
    dst_scale = dst_count.to(torch.float32).clamp(min=1).sqrt().unsqueeze(1)
    total = 0.0
    for row, col in source():
        for start in range(0, row.numel(), chunk):
            r, c = row[start:start + chunk], col[start:start + chunk]
            diff = x[r] / src_scale[r] - x[c] / dst_scale[c]
            energy = diff.norm(p=2, dim=1).square().to(torch.float32) / 4
            total += float(energy.sum(dtype=torch.float64))
    return total / edges, edges


def mde_measure(data: Any, target: str, paths: list[list[Step]], x: torch.Tensor, both: bool = False,
                max_reference_edges: int = MDE_REFERENCE_MAX_EDGES) -> dict[str, Any]:
    """MDE (spec §1) on `paths` with raw target features `x`: per `G_P` the reference `attribute_homophily` when
    `|E(G_P)| <= max_reference_edges`, else `mde_chunked` on the `G_P` row chunks (D-a); `both`: the chunked value as
    well wherever the reference runs (gate H0-eq). MDE = mean over meta-paths with a non-empty `G_P`."""
    per_path = []
    for path in paths:
        rec: dict[str, Any] = {"metapath": metapath_name(path), "steps": [list(s) for s in path]}
        whole = gp_edges_capped(data, path, max_reference_edges)
        if whole is not None:
            rec["edges"] = int(whole.size(1))
            if rec["edges"] == 0:
                rec.update(value=None, code_path="empty G_P")
            else:
                rec["reference"] = mde_reference(whole, x)
                rec.update(value=rec["reference"], code_path="hetero2net.attribute_homophily (reference)")
                if both:
                    rec["chunked"], _ = mde_chunked(chunks_of(whole), x)
        else:
            value, edges = mde_chunked(lambda p=path: iter_gp(data, p), x)
            rec.update(edges=edges, value=value, chunked=value, code_path="mde_chunked (D-a)")
        per_path.append(rec)
    return {"value": _mean(r["value"] for r in per_path), "form": "MDE, mean over M2 (raw features)",
            "feature_dim": int(x.size(1)), "feature_dtype": str(x.dtype), "per_metapath": per_path}


# ----------------------------------------------------------------------------------------------- ℋ²
def _calc_input(data: Any, target: str, labels: torch.Tensor) -> Any:
    """A HeteroData sharing `data`'s edge tensors, with the node counts and the target's labels (a clone:
    `calcHomophily` fills unlabeled targets in place)."""
    from torch_geometric.data import HeteroData

    out = HeteroData()
    for t in data.node_types:
        out[t].num_nodes = int(data[t].num_nodes)
    out[target].y = labels.clone()
    for e in data.edge_types:
        out[e].edge_index = data[e].edge_index
    return out


def calc_homophily(data: Any, target: str, labels: torch.Tensor, seed: int,
                   chunk_size: int = GP_CHUNK_ROWS) -> tuple[list[dict[str, Any]], torch.Tensor, list[list[EdgeType]]]:
    """H2GB's `calcHomophily` after `torch.manual_seed(seed)`. Returns its per-meta-path results (the meta-path read
    from its own printed `Results for metapath ...` line, in its order; a meta-path with an empty `G_P` has none),
    the labels it computed on (after its fill) and every meta-path it enumerated (its first printed line, `:106`).
    Values in H2GB's homophily form."""
    calc = h2gb_calc_homophily()
    graph = _calc_input(data, target, labels)
    buffer = io.StringIO()
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(buffer):
        h_edges, h_ins, h_adjs = calc.calcHomophily(graph, task_entity=target, chunk_size=chunk_size)
    printed = buffer.getvalue()
    paths = [ast.literal_eval(line.split("Results for metapath ", 1)[1].rsplit(" is:", 1)[0])
             for line in printed.splitlines() if line.startswith("Results for metapath ")]
    if len(paths) != len(h_adjs):
        raise RuntimeError("calcHomophily's printed meta-paths do not align with its results")
    rows = [{"metapath": metapath_name(p), "edge_types": [list(s) for s in p], "h_edge": float(a),
             "h_insensitive": float(b), "h_adj": float(c)} for p, a, b, c in zip(paths, h_edges, h_ins, h_adjs)]
    enumerated = [[tuple(s) for s in p] for p in ast.literal_eval(printed.splitlines()[0])]
    return rows, graph[target].y, enumerated  # type: ignore[return-value]


def h2gb_metapaths(edge_types: Iterable[EdgeType], target: str) -> list[list[EdgeType]]:
    """The meta-paths `calcHomophily` enumerates (`extract_metapath`, `:91-105`, nested, so not callable): read from
    its printed list on a probe graph with two nodes per type, two classes and identity relations (so no division
    by zero inside it) instead of re-implemented. Every length-2 sequence of stored relations from `target` back to
    `target`, same-type relations included."""
    from torch_geometric.data import HeteroData

    edge_types = [tuple(e) for e in edge_types]
    probe = HeteroData()
    for t in sorted({e[0] for e in edge_types} | {e[2] for e in edge_types} | {target}):
        probe[t].num_nodes = 2
    probe[target].y = torch.tensor([0, 1])
    for e in edge_types:
        probe[e].edge_index = torch.tensor([[0, 1], [0, 1]])
    _, _, enumerated = calc_homophily(probe, target, probe[target].y, seed=0)
    return enumerated


def labeled_only_metapath_data(data: Any, path: list[EdgeType], target: str, keep: torch.Tensor) -> Any:
    """A two-type graph whose only H2GB meta-path is `path` with both endpoints restricted to the targets in `keep`:
    type `tgt` = kept targets (re-indexed), type `mid` = every node of the intermediate type (also when that is the
    target type), relations `tgt -> mid` (first step, kept sources) and `mid -> tgt` (second step, kept
    destinations). Its `G_P` is the original `G_P` restricted to pairs of kept targets."""
    from torch_geometric.data import HeteroData

    (s1, r1, m1), (m2, r2, d2) = path
    if not (s1 == target and d2 == target and m1 == m2):
        raise ValueError(f"not a target-to-target length-2 meta-path: {path}")
    new_id = torch.full((int(data[target].num_nodes),), -1, dtype=torch.long)
    new_id[keep] = torch.arange(int(keep.sum()))
    e1 = data[tuple(path[0])].edge_index
    e1 = e1[:, keep[e1[0]]]
    e2 = data[tuple(path[1])].edge_index
    e2 = e2[:, keep[e2[1]]]
    out = HeteroData()
    out["tgt"].num_nodes = int(keep.sum())
    out["mid"].num_nodes = int(data[m1].num_nodes)
    out["tgt", r1, "mid"].edge_index = torch.stack([new_id[e1[0]], e1[1]])
    out["mid", r2, "tgt"].edge_index = torch.stack([e2[0], new_id[e2[1]]])
    return out


def h2_measure(data: Any, target: str, y: torch.Tensor, fill_seeds: Iterable[int] = FILL_SEEDS,
               labeled_only: bool = True) -> dict[str, Any]:
    """ℋ² (spec §1): per label problem (IMDB: per genre) and fill seed, `calcHomophily` on the whole graph;
    ℋ² = 1 − mean over its meta-paths of the adjusted homophily (multilabel: mean over genres). The fill
    `calcHomophily` applied is checked equal to `fill_like_h2gb`. Labeled-only diagnostic when targets are
    unlabeled: `calcHomophily` per enumerated meta-path on `labeled_only_metapath_data` (no fill needed)."""
    problems = label_tasks(y, fill_seeds)
    keep = labeled_mask(y)
    per_problem: dict[str, Any] = {}
    enumerated: list[list[EdgeType]] = []
    for p in problems:
        runs = []
        for seed in p["seeds"]:
            rows, used, enumerated = calc_homophily(data, target, p["labels"], seed)
            if not torch.equal(used, fill_like_h2gb(p["labels"], seed)):
                raise AssertionError("calcHomophily's fill differs from fill_like_h2gb")
            runs.append({"seed": seed, "per_metapath": rows,
                         "h2": None if not rows else 1 - sum(r["h_adj"] for r in rows) / len(rows)})
        lab = None
        if labeled_only and not bool(keep.all()):
            lab_rows = []
            for path in enumerated:
                mini = labeled_only_metapath_data(data, path, target, keep)
                rows, _, _ = calc_homophily(mini, "tgt", p["labels"][keep], seed=0)
                lab_rows.append({"metapath": metapath_name(path),
                                 "h_adj": rows[0]["h_adj"] if rows else None,
                                 "h_edge": rows[0]["h_edge"] if rows else None})
            vals = [r["h_adj"] for r in lab_rows if r["h_adj"] is not None]
            lab = {"h2": None if not vals else 1 - sum(vals) / len(vals), "per_metapath": lab_rows}
        per_problem[p["name"]] = {"seeds": p["seeds"], "runs": runs, "labeled_only": lab}
    n_seeds = len(problems[0]["seeds"])
    over_seeds = [_mean(per_problem[p["name"]]["runs"][k]["h2"] for p in problems) for k in range(n_seeds)]
    out: dict[str, Any] = {
        "value": over_seeds[0], "form": "heterophily: 1 - mean over H2GB's meta-paths (and genres) of adjusted "
        "homophily", "value_over_fill_seeds": over_seeds,
        "spread": None if len(over_seeds) < 2 else [min(over_seeds), max(over_seeds)],
        "h2gb_metapaths": [metapath_name(m) for m in enumerated], "per_problem": per_problem,
        "labeled_targets": int(keep.sum()), "targets": int(keep.numel()),
    }
    if labeled_only and not bool(keep.all()):
        out["labeled_only"] = _mean(per_problem[p["name"]]["labeled_only"]["h2"] for p in problems)
    if y.dim() == 2:
        out["per_genre"] = {p["name"]: per_problem[p["name"]]["runs"][0]["h2"] for p in problems}
    return out


# ----------------------------------------------------------------------------------------------- task-level H_edge
def task_h_edge(edge_index: torch.Tensor, y: torch.Tensor, fill_seeds: Iterable[int] = FILL_SEEDS) -> dict[str, Any]:
    """Task-level `H_edge` of an LP target relation (spec §1): `label_homophily` on the relation's facts (given
    symmetrised) with H2GB's fill; heterophily form; labeled-only diagnostic on edges with both endpoints labeled."""
    problems = label_tasks(y, fill_seeds)
    p = problems[0]
    over_seeds = []
    for seed in p["seeds"]:
        acc = LabelHomophily()
        for row, col in chunks_of(edge_index)():
            acc.add(row, col, fill_like_h2gb(p["labels"], seed))
        over_seeds.append(None if acc.value is None else 1 - acc.value)
    keep = labeled_mask(y)
    out: dict[str, Any] = {"value": over_seeds[0], "form": "heterophily: 1 - label_homophily",
                           "fill_seeds": p["seeds"], "value_over_fill_seeds": over_seeds,
                           "spread": None if len(over_seeds) < 2 else [min(over_seeds), max(over_seeds)],
                           "edges": int(edge_index.size(1)),
                           "self_loops": int((edge_index[0] == edge_index[1]).sum())}
    if not bool(keep.all()):
        both = keep[edge_index[0]] & keep[edge_index[1]]
        acc = LabelHomophily()
        for row, col in chunks_of(edge_index[:, both])():
            acc.add(row, col, p["labels"])
        out.update(labeled_only=None if acc.value is None else 1 - acc.value, labeled_only_edges=int(both.sum()))
    return out


# ----------------------------------------------------------------------------------------------- graphs
def symmetrise_facts(data: Any) -> tuple[Any, dict[str, str]]:
    """The graph the H2S-NN trainers walk, as a HeteroData: a relation with an exact reverse partner
    (`src/data/fact_layer.py` `inverse_relations_typed`: rho by exact edge-set equality) is kept as stored; a
    relation without one is a set of facts walked both ways, so its reverse is added: a same-type relation is made
    undirected in place (`to_undirected`), a cross-type relation gets `rev_<rel>` (H2GB's loader convention,
    `H2GB/datasets/mag_dataset.py:103-113`). Returns the graph and what was done per relation."""
    from torch_geometric.data import HeteroData
    from torch_geometric.utils import to_undirected

    from src.data.fact_layer import inverse_relations_typed

    sizes = {t: int(data[t].num_nodes) for t in data.node_types}
    rho = inverse_relations_typed({tuple(e): data[e].edge_index for e in data.edge_types}, sizes)
    out = HeteroData()
    for t in data.node_types:
        for key, value in data[t].items():
            out[t][key] = value
        out[t].num_nodes = sizes[t]
    record: dict[str, str] = {}
    for e in data.edge_types:
        e = tuple(e)
        index = data[e].edge_index
        name = "-".join(e)
        if rho[e] is not None:
            out[e].edge_index = index
            record[name] = "kept" if rho[e] == e else f"kept (reverse partner {'-'.join(rho[e])})"
        elif e[0] == e[2]:
            out[e].edge_index = to_undirected(index, num_nodes=sizes[e[0]])
            record[name] = "no reverse partner: made undirected (to_undirected)"
        else:
            out[e].edge_index = index
            rev = (e[2], "rev_" + e[1], e[0])
            if rev in data.edge_types:
                raise ValueError(f"{rev} exists but is not the exact reverse of {e}")
            out[rev].edge_index = index.flip(0)
            record[name] = f"no reverse partner: added {'-'.join(rev)}"
    return out, record


# ----------------------------------------------------------------------------------------------- GC
def type_mixing(node_type: torch.Tensor, edge_index: torch.Tensor) -> tuple[int, int]:
    """(stored edges joining nodes of different types, stored edges) of one graph."""
    if edge_index.numel() == 0:
        return 0, 0
    return int((node_type[edge_index[0]] != node_type[edge_index[1]]).sum()), int(edge_index.size(1))


def gc_type_statistics(node_types: list[torch.Tensor], edge_indices: list[torch.Tensor], num_node_types: int,
                       num_edge_types: int) -> dict[str, Any]:
    """GC table (spec 1.23): numbers of node and edge types and the type-mixing ratio (share of edges joining nodes
    of different types), pooled over all graphs; also the mean of the per-graph ratios (graphs with edges)."""
    cross = total = 0
    ratios = []
    for nt, ei in zip(node_types, edge_indices, strict=True):
        c, t = type_mixing(nt, ei)
        cross += c
        total += t
        if t:
            ratios.append(c / t)
    return {"num_node_types": int(num_node_types), "num_edge_types": int(num_edge_types), "graphs": len(node_types),
            "edges": total, "cross_type_edges": cross, "type_mixing_ratio": None if total == 0 else cross / total,
            "mean_per_graph_ratio": None if not ratios else sum(ratios) / len(ratios),
            "form": "heterogeneity statistic, never heterophily (spec 1.23)"}


@dataclass
class Timer:
    """Wall time and peak host memory of the process (ru_maxrss, kB on Linux) for a record."""

    start: float = field(default_factory=time.time)

    def stats(self) -> dict[str, float]:
        return {"wall_s": round(time.time() - self.start, 2),
                "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)}


__all__ = ["EDGE_CHUNK", "FILL_SEEDS", "GP_CHUNK_ROWS", "HETERO2NET_LISTS", "LabelHomophily", "MDE_REFERENCE_MAX_EDGES",
           "Timer", "calc_homophily", "chunks_of", "fill_like_h2gb", "gc_type_statistics", "genre_fill_seed",
           "genre_labels", "gp_edge_index", "gp_edges_capped", "h2_measure", "h2gb_calc_homophily", "h2gb_metapaths",
           "hetero2net_homophily", "hetero2net_m2", "iter_gp", "label_tasks", "label_homophily_chunked", "labeled_mask",
           "labeled_only_metapath_data", "mde_chunked", "mde_measure", "mde_reference", "metapath_name", "mlh_measure",
           "step_adjacency", "step_relations", "symmetrise_facts", "task_h_edge", "type_mixing", "vendor_commit"]
