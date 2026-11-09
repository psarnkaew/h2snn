"""MLH and the H^2 index of HGB node-classification graphs (task card Part J, unit J0).

Numbers come from the reference code wherever it can be called. The conventions and their
`file:line` sources are in `../docs/heterophily_evidence/j02_reference_code.md`.

- H2GB `calcHomophily` (`vendor/H2GB/H2GB/calcHomophily.py`, commit d3b49db) gives, per
  meta-path, the edge homophily and adjusted homophily of the meta-path-induced graph `G_P`,
  and its own meta-path list (printed by the function, parsed here).
- Hetero^2Net `label_homophily`, `label_homophily_matrix` and `get_metapaths`
  (`vendor/Hetero2Net/hetero2net/homophily.py`, commit 4913394) give multilabel edge homophily
  (IMDB), node homophily, and Hetero^2Net's meta-path set.

Own code exists only where neither reference can be called:
- `iter_metapath_chunks`: `G_P` exactly as `calcHomophily` builds it (the product sits inside that
  function). It feeds node homophily, the multilabel IMDB values and the size of `G_P`.
- `homophily_counts`: an independent float64 implementation, used only to cross-check the reference
  and for the labeled-only diagnostic.
- `symmetrise_like_h2gb`: H2GB's loader symmetrisation (`H2GB/datasets/mag_dataset.py:103-113`), for
  the flagged Freebase variant.

Homophily form: h (1 = every pair agrees). Heterophily form: H = 1 - h (H2GB Table 2).
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import functools
import importlib.util
import io
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Iterator

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.runtime.vendor_paths import vendor_path  # noqa: E402

EdgeType = tuple[str, str, str]
MetaPath = tuple[EdgeType, EdgeType]

#: HGB node-classification datasets: folder under the data root and PyG `HGBDataset` name.
HGB_DATASETS: dict[str, str] = {"acm": "ACM", "dblp": "DBLP", "imdb": "IMDB", "freebase": "Freebase"}

LOGGER = logging.getLogger("compute_heterophily")


# ---------------------------------------------------------------------------- references
def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1)
def reference_modules() -> tuple[ModuleType, ModuleType]:
    """(H2GB `calcHomophily` module, Hetero^2Net `homophily` module), loaded from their files."""
    h2gb = _load_module("_ref_h2gb_calcHomophily",
                        vendor_path("h2gb") / "H2GB" / "calcHomophily.py")
    hetero2net = _load_module("_ref_hetero2net_homophily",
                              vendor_path("hetero2net") / "hetero2net" / "homophily.py")
    return h2gb, hetero2net


@dataclass(slots=True)
class H2GBMetapathResult:
    metapath: MetaPath
    h_edge: float
    h_adj: float


def run_calc_homophily(data, target: str, seed: int, chunk_size: int) -> tuple[list[MetaPath],
                                                                           list[H2GBMetapathResult]]:
    """Call H2GB's `calcHomophily` and return (its meta-path list, its per-meta-path results).

    The function prints its meta-path list (`:106`) and one "Results for metapath ..." line per
    non-empty `G_P` (`:174`); both are parsed so the returned lists align with the reference's own
    enumeration. It fills labels equal to -1 with random classes **in place** (`:110-113`, global
    RNG): the RNG is seeded here and the function receives a clone, so the caller's label tensor
    (and every other reference to it) is never modified.
    """
    h2gb, _ = reference_modules()
    y_original = data[target].y
    data[target].y = y_original.clone()
    stdout, stderr = io.StringIO(), io.StringIO()
    torch.manual_seed(seed)
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            h_edges, _, h_adjs = h2gb.calcHomophily(data, task_entity=target, chunk_size=chunk_size)
    finally:
        data[target].y = y_original
    lines = stdout.getvalue().splitlines()
    metapaths = [_as_metapath(path) for path in ast.literal_eval(lines[0])]
    prefix = "Results for metapath "
    computed = [_as_metapath(ast.literal_eval(line[len(prefix):-len(" is:")]))
                for line in lines if line.startswith(prefix)]
    if len(computed) != len(h_edges) or len(computed) != len(h_adjs):
        raise RuntimeError("Could not align calcHomophily's printed meta-paths with its results.")
    results = [H2GBMetapathResult(path, float(he), float(ha))
               for path, he, ha in zip(computed, h_edges, h_adjs)]
    return metapaths, results


def _as_metapath(path: list) -> MetaPath:
    first, second = path
    return (tuple(first), tuple(second))  # type: ignore[return-value]


def hetero2net_metapaths(data, target: str) -> list[MetaPath]:
    """Hetero^2Net's length-2 target->target set (`get_metapaths`, `homophily.py:51-63`).

    It returns node-type pairs and skips same-type relations. Each pair is mapped to the unique stored
    edge type with those endpoint types; an ambiguous pair raises.
    """
    _, hetero2net = reference_modules()
    pairs = hetero2net.get_metapaths(target, data.edge_types, [], [], 2, target)
    resolved: list[MetaPath] = []
    for path in pairs:
        hops = []
        for src, dst in path:
            matches = [et for et in data.edge_types if et[0] == src and et[2] == dst]
            if len(matches) != 1:
                raise ValueError(f"Node-type pair {(src, dst)} matches {len(matches)} edge types.")
            hops.append(matches[0])
        resolved.append((hops[0], hops[1]))
    return resolved


# ------------------------------------------------------------------------------ own code
def iter_metapath_chunks(data, metapath: MetaPath, chunk_size: int) -> Iterator[torch.Tensor]:
    """`G_P` exactly as `calcHomophily` builds it (`:119-150`): row chunks of the coalesced product of
    the two stored relations, self-loops kept. Yields global `[2, E_chunk]` edge indices."""
    from torch_sparse import SparseTensor

    (src, _, mid), (_, _, dst) = metapath
    n_src, n_mid = data.num_nodes_dict[src], data.num_nodes_dict[mid]
    n_dst = data.num_nodes_dict[dst]
    first = data[metapath[0]].edge_index
    second = data[metapath[1]].edge_index
    adj_1 = SparseTensor(row=first[0], col=first[1], value=None, sparse_sizes=(n_src, n_mid))
    adj_2 = SparseTensor(row=second[0], col=second[1], value=None, sparse_sizes=(n_mid, n_dst))
    for start in range(0, n_src, chunk_size):
        row, col, _ = (adj_1[start:min(start + chunk_size, n_src)] @ adj_2).coo()
        yield torch.stack([row + start, col])


def metapath_graph(data, metapath: MetaPath, chunk_size: int) -> torch.Tensor:
    return torch.cat(list(iter_metapath_chunks(data, metapath, chunk_size)), dim=1)


@dataclass(slots=True)
class HomophilyCounts:
    """Float64 sums of one `G_P`. Equal labels: identical class (single-label) or identical label
    vector (multilabel, Hetero^2Net `homophily.py:20-21`)."""

    num_edges: int
    num_same: int
    node_same: torch.Tensor
    node_in: torch.Tensor
    class_in_degree: torch.Tensor | None

    @property
    def h_edge(self) -> float:
        return self.num_same / self.num_edges

    @property
    def h_node(self) -> float:
        """PyG `homophily(method='node')` convention: mean over all nodes, no in-edge -> 0."""
        per_node = torch.where(self.node_in > 0, self.node_same / self.node_in.clamp(min=1),
                               torch.zeros_like(self.node_same))
        return float(per_node.mean())

    @property
    def h_adj(self) -> float:
        """Adjusted homophily (Platonov et al.; H2GB `:170-173`): class shares are in-degree sums
        over all edges of `G_P`."""
        if self.class_in_degree is None:
            raise ValueError("Adjusted homophily is undefined for multilabel targets (J0.2 (d)).")
        shares = self.class_in_degree / self.num_edges
        expected = float((shares ** 2).sum())
        return (self.h_edge - expected) / (1.0 - expected)


def h2gb_metapaths(data, target: str) -> list[MetaPath]:
    """H2GB's enumeration (`calcHomophily.py:91-105`, nested, so not callable): every 2-hop sequence
    of stored edge types from `target` back to `target`, same-type relations included, in
    `data.edge_types` order. Tests check it equals the list `calcHomophily` prints."""
    return [(first, second) for first in data.edge_types if first[0] == target
            for second in data.edge_types if second[0] == first[2] and second[2] == target]


def h2gb_random_fill(y: torch.Tensor, seed: int) -> torch.Tensor:
    """The labels `calcHomophily` sees after `torch.manual_seed(seed)` (`:110-113`): -1 replaced
    by a random class. Same RNG calls in the same order, so the fill is identical (cross-checked)."""
    label = y.clone()
    torch.manual_seed(seed)
    num_classes = label.max() + 1
    mask = label == -1
    label[mask] = torch.randint(0, int(num_classes), size=(int(mask.sum()),), device=label.device)
    return label


def _labels_equal(y: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    if y.dim() > 1:
        return (y[row] == y[col]).all(dim=1)
    return y[row] == y[col]


def homophily_counts(data, metapath: MetaPath, y: torch.Tensor, chunk_size: int,
                     keep: torch.Tensor | None = None) -> HomophilyCounts:
    """Counts of `G_P`; `keep` (bool per target node) restricts `G_P` to pairs with both endpoints
    kept (labeled-only diagnostic, own convention, never a reported reference value)."""
    num_nodes = y.size(0)
    node_same = torch.zeros(num_nodes, dtype=torch.float64)
    node_in = torch.zeros(num_nodes, dtype=torch.float64)
    single = y.dim() == 1
    num_classes = int(y.max()) + 1 if single else 0
    class_in = torch.zeros(num_classes, dtype=torch.float64) if single else None
    num_edges = num_same = 0
    for chunk in iter_metapath_chunks(data, metapath, chunk_size):
        row, col = chunk
        if keep is not None:
            mask = keep[row] & keep[col]
            row, col = row[mask], col[mask]
        if single and bool((y[col] < 0).any()):
            raise ValueError("An unlabeled target is in G_P: fill (h2gb_random_fill) or restrict "
                             "(keep) first.")
        same = _labels_equal(y, row, col).to(torch.float64)
        num_edges += int(row.numel())
        num_same += int(same.sum())
        node_same.index_add_(0, col, same)
        node_in.index_add_(0, col, torch.ones_like(same))
        if class_in is not None:
            class_in.index_add_(0, y[col], torch.ones_like(same))
    return HomophilyCounts(num_edges, num_same, node_same, node_in, class_in)


def symmetrise_like_h2gb(data):
    """H2GB's loader convention (`H2GB/datasets/mag_dataset.py:103-113`): a same-type relation is
    made undirected; a cross-type relation gets a `rev_<rel>` reverse. Returns a new HeteroData."""
    from torch_geometric.data import HeteroData
    from torch_geometric.utils import to_undirected

    out = HeteroData()
    for node_type in data.node_types:
        for key, value in data[node_type].items():
            out[node_type][key] = value
        out[node_type].num_nodes = data[node_type].num_nodes
    for src, rel, dst in data.edge_types:
        edge_index = data[src, rel, dst].edge_index
        if src == dst:
            out[src, rel, dst].edge_index = to_undirected(edge_index,
                                                          num_nodes=data[src].num_nodes)
        else:
            out[src, rel, dst].edge_index = edge_index
            out[dst, "rev_" + rel, src].edge_index = edge_index.flip(0)
    return out


# ------------------------------------------------------------------------------- driver
@dataclass(slots=True)
class MetapathRow:
    metapath: str
    num_edges: int
    h_edge: float
    h_node: float
    h_adj: float | None
    source_h_edge: str
    source_h_node: str
    source_h_adj: str
    labeled_only_h_edge: float | None = None
    labeled_only_h_adj: float | None = None


@dataclass(slots=True)
class DatasetResult:
    dataset: str
    graph: str
    target: str
    num_targets: int
    num_labeled: int
    class_counts: dict[str, int]
    h2gb_metapaths: list[str]
    hetero2net_metapaths: list[str]
    rows: list[MetapathRow]
    mlh_h2gb_heterophily_mean: float
    mlh_h2gb_heterophily_max: float
    h2_index: float | None
    mlh_hetero2net_homophily: float | None
    fill_seeds: list[int] = field(default_factory=list)
    h2_index_over_seeds: list[float] = field(default_factory=list)
    mlh_mean_over_seeds: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def metapath_name(metapath: MetaPath) -> str:
    (src, rel_1, mid), (_, rel_2, dst) = metapath
    return f"{src} -{rel_1}-> {mid} -{rel_2}-> {dst}"


def load_hgb(data_root: Path, dataset: str):
    from torch_geometric.datasets import HGBDataset

    return HGBDataset(root=str(data_root / dataset), name=HGB_DATASETS[dataset])[0]


def target_type(data) -> str:
    targets = [nt for nt in data.node_types if "y" in data[nt]]
    if len(targets) != 1:
        raise ValueError(f"Expected one labeled node type, found {targets}.")
    return targets[0]


def labeled_mask(y: torch.Tensor) -> torch.Tensor:
    return y.sum(dim=1) > 0 if y.dim() > 1 else y >= 0


def class_counts(y: torch.Tensor) -> dict[str, int]:
    labeled = y[labeled_mask(y)]
    if y.dim() > 1:
        counts = {f"label_{k}": int(labeled[:, k].sum()) for k in range(y.size(1))}
        counts["distinct_label_sets"] = int(torch.unique(labeled, dim=0).size(0))
        return counts
    return {f"class_{k}": int(v) for k, v in enumerate(torch.bincount(labeled))}


def analyse(data, dataset: str, graph: str, seeds: list[int], chunk_size: int,
            node_ref_max_edges: int, cross_check_tol: float) -> DatasetResult:
    _, hetero2net = reference_modules()
    target = target_type(data)
    y = data[target].y
    multilabel = y.dim() > 1
    keep = labeled_mask(y)
    notes: list[str] = []
    het_paths = hetero2net_metapaths(data, target)

    has_unlabeled = bool((~keep).any())
    if multilabel:
        # calcHomophily cannot take a 2-D label tensor (J0.2 (d)).
        h2gb_paths = h2gb_metapaths(data, target)
        computed: dict[MetaPath, H2GBMetapathResult] = {}
        y_seen = y  # Hetero^2Net receives PyG's y: unlabeled movies are all-zero rows.
        notes.append("Multilabel: calcHomophily cannot run; h_edge and h_node from Hetero^2Net "
                     "(exact label-vector match) on G_P built as calcHomophily builds it, with "
                     "PyG's all-zero rows for unlabeled movies; H_adj and H^2 not computed "
                     "(undefined in both codebases).")
        seeds_used: list[int] = []
    else:
        seeds_used = seeds if has_unlabeled else seeds[:1]
        runs = [run_calc_homophily(data, target, seed, chunk_size) for seed in seeds_used]
        h2gb_paths, first_results = runs[0]
        if h2gb_paths != h2gb_metapaths(data, target):
            raise AssertionError("Own meta-path enumeration differs from calcHomophily's list.")
        computed = {r.metapath: r for r in first_results}
        y_seen = h2gb_random_fill(y, seeds_used[0])  # the labels calcHomophily saw (seed 0 run)
        if has_unlabeled:
            notes.append(f"{int((~keep).sum())} unlabeled targets: H2GB fills them with random "
                         f"classes (calcHomophily.py:110-113); fill seeds {seeds_used}; reported "
                         f"values use seed {seeds_used[0]}; H_node uses the same filled labels.")

    rows: list[MetapathRow] = []
    for path in h2gb_paths:
        counts = homophily_counts(data, path, y_seen, chunk_size)
        if counts.num_edges == 0:
            notes.append(f"{metapath_name(path)}: empty G_P, skipped (as calcHomophily does).")
            continue
        if counts.num_edges <= node_ref_max_edges:
            edge_index = metapath_graph(data, path, chunk_size)
            h_node = float(hetero2net.label_homophily_matrix(edge_index, y_seen).mean())
            src_node = "Hetero2Net label_homophily_matrix().mean()"
            if abs(h_node - counts.h_node) > cross_check_tol:
                raise AssertionError(f"h_node cross-check failed on {metapath_name(path)}.")
        else:
            edge_index = None
            h_node = counts.h_node
            src_node = f"own (G_P has {counts.num_edges} > {node_ref_max_edges} pairs)"
        if multilabel:
            if edge_index is None:
                raise MemoryError(f"G_P {metapath_name(path)} too large to materialise.")
            h_edge = float(hetero2net.label_homophily(edge_index, y_seen))
            if abs(h_edge - counts.h_edge) > cross_check_tol:
                raise AssertionError(f"h_edge cross-check failed on {metapath_name(path)}.")
            h_adj, src_edge, src_adj = None, "Hetero2Net label_homophily", "not computed"
        else:
            reference = computed[path]
            h_edge, h_adj = reference.h_edge, reference.h_adj
            src_edge = src_adj = "H2GB calcHomophily"
            # Own float64 code on the same labels must agree (validates G_P, the fill and h_adj).
            if abs(h_edge - counts.h_edge) > cross_check_tol or \
                    abs(h_adj - counts.h_adj) > cross_check_tol:
                raise AssertionError(f"H2GB cross-check failed on {metapath_name(path)}: "
                                     f"{h_edge} vs {counts.h_edge}, {h_adj} vs {counts.h_adj}.")
        labeled = homophily_counts(data, path, y, chunk_size, keep=keep) if has_unlabeled else None
        rows.append(MetapathRow(
            metapath=metapath_name(path), num_edges=counts.num_edges, h_edge=h_edge,
            h_node=h_node, h_adj=h_adj, source_h_edge=src_edge, source_h_node=src_node,
            source_h_adj=src_adj,
            labeled_only_h_edge=None if labeled is None or labeled.num_edges == 0 else labeled.h_edge,
            labeled_only_h_adj=None if labeled is None or labeled.num_edges == 0 or multilabel
            else labeled.h_adj,
        ))

    h_edges = [row.h_edge for row in rows]
    h_adjs = [row.h_adj for row in rows if row.h_adj is not None]
    het_names = {metapath_name(path) for path in het_paths}
    het_values = [row.h_edge for row in rows if row.metapath in het_names]
    result = DatasetResult(
        dataset=dataset, graph=graph, target=target, num_targets=int(y.size(0)),
        num_labeled=int(keep.sum()), class_counts=class_counts(y),
        h2gb_metapaths=[metapath_name(p) for p in h2gb_paths],
        hetero2net_metapaths=sorted(het_names), rows=rows,
        mlh_h2gb_heterophily_mean=1.0 - sum(h_edges) / len(h_edges),
        mlh_h2gb_heterophily_max=max(1.0 - h for h in h_edges),
        h2_index=None if multilabel else 1.0 - sum(h_adjs) / len(h_adjs),
        mlh_hetero2net_homophily=sum(het_values) / len(het_values) if het_values else None,
        fill_seeds=seeds_used, notes=notes,
    )
    if not multilabel and len(seeds_used) > 1:
        for paths_results in runs:
            _, per_seed = paths_results
            result.h2_index_over_seeds.append(1.0 - sum(r.h_adj for r in per_seed) / len(per_seed))
            result.mlh_mean_over_seeds.append(1.0 - sum(r.h_edge for r in per_seed) / len(per_seed))
    return result


#: Cited, never recomputed (spec v1.7 §1). H2GB Table 2 (KDD 2025), heterophily form; the MLH row's
#: aggregation is the mean (calcHomophily.py:185). Hetero^2Net Table II (TPAMI 2025), MLH homophily
#: form; its Table VII swaps the ACM and IMDB columns (j02 / preregistration).
CITED_H2GB_TABLE2: dict[str, dict[str, float]] = {
    "ogbn-mag": {"mlh_heterophily": 0.8731, "h2_index": 0.8773},
    "mag-year": {"mlh_heterophily": 0.7718, "h2_index": 0.9654},
    "IEEE-CIS-G": {"mlh_heterophily": 0.1352, "h2_index": 0.9846},
    "H-Pokec": {"mlh_heterophily": 0.3922, "h2_index": 0.9488},
}
CITED_HETERO2NET_TABLE2: dict[str, float] = {"dblp": 0.87, "acm": 0.60, "imdb": 0.24}


def write_markdown(results: list[DatasetResult], path: Path) -> None:
    lines = ["# Heterophily of the node-classification graphs (J0.5)", "",
             "Heterophily form H = 1 - h unless marked *homophily*. Conventions and their code "
             "sources: `j02_reference_code.md`. Computed values: "
             "`scripts/heterophily/compute_heterophily.py` (validated by the J0.4 gate). "
             "H_node follows PyG / Hetero^2Net: a target without in-edges in G_P counts as "
             "homophily 0, so H_node can exceed H_edge on sparse G_P.", "",
             "## Summary", "",
             "| Dataset | Graph | MLH, H2GB set, mean (H2GB) | MLH, max | H^2 index | "
             "MLH, Hetero^2Net set, *homophily* (ours / published) | Source |",
             "|---|---|---|---|---|---|---|"]
    for res in results:
        h2 = "not computed (multilabel)" if res.h2_index is None else f"{res.h2_index:.4f}"
        if res.h2_index_over_seeds:
            h2 += (f" (fill seeds: {min(res.h2_index_over_seeds):.4f}–"
                   f"{max(res.h2_index_over_seeds):.4f})")
        published = CITED_HETERO2NET_TABLE2.get(res.dataset)
        het = "—" if res.mlh_hetero2net_homophily is None else f"{res.mlh_hetero2net_homophily:.4f}"
        if published is not None:
            het += f" / {published:.2f} (Hetero^2Net Table II)"
        lines.append(f"| {res.dataset} | {res.graph} | {res.mlh_h2gb_heterophily_mean:.4f} | "
                     f"{res.mlh_h2gb_heterophily_max:.4f} | {h2} | {het} | computed |")
    for name, row in CITED_H2GB_TABLE2.items():
        lines.append(f"| {name} | H2GB loader | {row['mlh_heterophily']:.4f} | — | "
                     f"{row['h2_index']:.4f} | — | H2GB Table 2 (cited) |")
    lines += [""]
    for res in results:
        lines += [f"## {res.dataset} ({res.graph}); target `{res.target}`, "
                  f"{res.num_labeled}/{res.num_targets} labeled", "",
                  f"- MLH, H2GB set, **mean (H2GB's aggregation)**: {res.mlh_h2gb_heterophily_mean:.4f}; "
                  f"max: {res.mlh_h2gb_heterophily_max:.4f}",
                  f"- H^2 index (H2GB set, mean of H_adj): "
                  + ("not computed (multilabel)" if res.h2_index is None else f"{res.h2_index:.4f}"),
                  "- MLH, Hetero^2Net set, homophily form: "
                  + ("none (empty set)" if res.mlh_hetero2net_homophily is None
                     else f"{res.mlh_hetero2net_homophily:.4f}"),
                  f"- Class counts (labeled): {res.class_counts}", ""]
        if res.h2_index_over_seeds:
            lines.append(f"- Over fill seeds {res.fill_seeds}: H^2 "
                         + ", ".join(f"{v:.4f}" for v in res.h2_index_over_seeds)
                         + "; MLH " + ", ".join(f"{v:.4f}" for v in res.mlh_mean_over_seeds))
        lines += ["", "| Meta-path | pairs in G_P (loops incl.) | H_edge | H_node | H_adj | "
                  "labeled-only H_edge / H_adj (diagnostic) |", "|---|---|---|---|---|---|"]
        for row in res.rows:
            adj = "—" if row.h_adj is None else f"{1 - row.h_adj:.4f}"
            diag = "—" if row.labeled_only_h_edge is None else (
                f"{1 - row.labeled_only_h_edge:.4f} / "
                + ("—" if row.labeled_only_h_adj is None else f"{1 - row.labeled_only_h_adj:.4f}"))
            lines.append(f"| {row.metapath} | {row.num_edges:,} | {1 - row.h_edge:.4f} | "
                         f"{1 - row.h_node:.4f} | {adj} | {diag} |")
        lines += [""] + [f"- Note: {note}" for note in res.notes] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--datasets", nargs="+", default=list(HGB_DATASETS),
                        choices=list(HGB_DATASETS))
    parser.add_argument("--freebase-symmetrised", action="store_true",
                        help="Also report Freebase under H2GB's loader symmetrisation (flagged).")
    parser.add_argument("--fill-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--chunk-size", type=int, default=200, help="calcHomophily's default.")
    parser.add_argument("--node-ref-max-edges", type=int, default=30_000_000)
    parser.add_argument("--cross-check-tol", type=float, default=1e-6)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    results: list[DatasetResult] = []
    for dataset in args.datasets:
        data = load_hgb(args.data_root, dataset)
        graphs = [("as loaded by PyG HGBDataset", data)]
        if dataset == "freebase" and args.freebase_symmetrised:
            graphs.append(("symmetrised as H2GB's loaders do (flagged)", symmetrise_like_h2gb(data)))
        for graph_name, graph in graphs:
            LOGGER.info("%s, %s: start", dataset, graph_name)
            res = analyse(graph, dataset, graph_name, args.fill_seeds, args.chunk_size,
                          args.node_ref_max_edges, args.cross_check_tol)
            LOGGER.info("%s, %s: MLH(H2GB, mean, heterophily) %.6f, H2 %s, MLH(Hetero2Net, "
                        "homophily) %s", dataset, graph_name, res.mlh_h2gb_heterophily_mean,
                        "n/a" if res.h2_index is None else f"{res.h2_index:.6f}",
                        "n/a" if res.mlh_hetero2net_homophily is None
                        else f"{res.mlh_hetero2net_homophily:.6f}")
            for row in res.rows:
                LOGGER.info("  %s: |G_P| %d, h_edge %.6f, h_node %.6f, h_adj %s", row.metapath,
                            row.num_edges, row.h_edge, row.h_node,
                            "n/a" if row.h_adj is None else f"{row.h_adj:.6f}")
            results.append(res)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "computed": [asdict(r) for r in results],
        "cited": {"h2gb_table2_heterophily": CITED_H2GB_TABLE2,
                  "hetero2net_table2_mlh_homophily": CITED_HETERO2NET_TABLE2},
        "reference_commits": {name: subprocess.run(
            ["git", "-C", str(vendor_path(name)), "rev-parse", "HEAD"], capture_output=True,
            text=True, check=True).stdout.strip() for name in ("h2gb", "hetero2net")},
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.out_md is not None:
        write_markdown(results, args.out_md)


if __name__ == "__main__":
    main()
