"""Card 7 V3 (ablation plan v2, Q4): semi-synthetic graphs at heterophily h in {0.20, 0.30, 0.40, 0.50, 0.60}.

Levels (card 7 amendment 1, author 2026-11-19): 0.20-0.60 in steps of 0.1 on ACM and DBLP; 0.80 is dropped (an
intermediate node with n target neighbours caps h at (1 - 1/C) n/(n-1), the ceiling 1 - 1/C is recorded per graph
as `ceiling`); IMDB graphs may stay on disk but are not run (`manifest.py`).

h = 1 - MLH over the dataset's length-2 meta-paths M_2 (spec §1): the mean over M_2 of the edge heterophily of the
meta-path graph G_P, exactly as `scripts/heterophily/compute_heterophily.py` computes it. No second implementation:
the search evaluates h with that file's `homophily_counts` on its `h2gb_metapaths` (with its fill
`h2gb_random_fill` for unlabelled targets, primary fill seed 0; IMDB per genre with `genre_fill_seed`), and the
reported value of every written graph is that file's `analyse` (H2GB `calcHomophily`, cross-checked) on the graph
loaded back from the written files with the file's own loader `load_hgb`.

A *fact* is one stored link and its mirrored reverse link (HGB stores both directions as two link types: ACM
cite/ref, paper-author/author-paper, ...); adding or removing a fact adds or removes both lines. The *M_2 relations*
are the fact relations that appear in a meta-path of M_2 (ACM: all four; DBLP: author-paper only; IMDB: all three).

Rules (card 7 V3):
- **Raise h** (target above the current value): add facts of the M_2 relations that create different-class meta-path
  pairs. A fact (p, x) has the labelled target p as one endpoint. Draw p uniformly over the labelled targets, then a
  class c uniformly over all classes other than p's, then a labelled target q of class c uniformly; when x's type is
  the target type, x = q, otherwise x is a uniform neighbour of q in that relation (so p - x - q is a new
  different-class pair). Reading of the card's "partner drawn uniformly over all other classes" for relations whose
  other endpoint is not labelled. No duplicate fact and no self-loop. IMDB (multilabel): a "class" is a label set.
- **Lower h** (target below the current value): remove facts that create a different-class meta-path pair (both
  endpoints labelled and different for a target-target relation; for a target-x relation, x has a labelled
  neighbour of another class than p), in a seeded uniform order, at most 50 % of each relation's facts. Never add.
- One scalar alpha sets the count of every M_2 relation, `round(alpha * facts_r)` (relations keep their relative
  share of the change); alpha is found by bisection to reach |h - target| <= 0.01. A level that cannot be reached
  (removal: under the 50 % cap; addition: within `--max_add_ratio` times each relation's facts, an engineering bound
  the card does not set) is recorded as unreached, with the closest value; the cap is never raised.
- **Random control** for every targeted graph: the same number of facts added or removed per relation, drawn
  uniformly at random (additions: uniform endpoints of the relation's two types, new, no self-loop; removals: uniform
  facts of the relation).
- One fixed seed per (dataset, level, kind): `1000 * dataset index + 10 * level index + kind index`. The level index
  is fixed per level (`LEVEL_SEED_INDEX`): 0.20, 0.40, 0.60, 0.80 keep their Phase A indices 0-3 (so the graphs
  generated then are unchanged), 0.30 and 0.50 take 4 and 5.

Output: `<data_root>/NC/<DATASET>_pert/<kind>_h<level>/data/NC/<name>/raw/<NAME>/` in HGB's own file format (every
file of the release unchanged (hard-linked, else copied) except `link.dat`: the original lines minus the removed facts, then the added
facts). `<...>/<kind>_h<level>/data` is the data root of that graph: H2S-NN reads it with `data.root` set to it, and the
baseline launchers read it as `./data` from `<...>/<kind>_h<level>/` (their bundles' `data.root` default), so every
method uses its normal loader with only the data directory changed. Per graph `graph.json` (reached h, H^2, facts
added / removed per relation, seed, alpha, SHA-256 of `link.dat`, the edge-list fingerprint `edge_fingerprint`
(`fingerprint.py`; amendment 3: every run checks its loaded graph against it), C and the ceiling 1 - 1/C);
`reachability.json` / `.md` per run. `--backfill` adds the fingerprint, C and the ceiling to graphs written before
(their `link.dat` SHA-256 is checked unchanged).

    python scripts/perturbation/make_graphs.py --datasets acm dblp [--levels 0.30 0.50] [--data_root data] --out <json>
    python scripts/perturbation/make_graphs.py --datasets acm dblp --backfill --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "heterophily"))

import compute_heterophily as ref  # noqa: E402  (the reference functions; never re-implemented here)
import fingerprint as fp  # noqa: E402
from src.data.data_layout import REPO_DATA_ROOT, nc_root  # noqa: E402

LEVELS: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60)
#: seed index per level: Phase A's 0.20/0.40/0.60/0.80 keep 0-3 (their graphs stay as generated), new levels follow
LEVEL_SEED_INDEX: dict[float, int] = {0.20: 0, 0.40: 1, 0.60: 2, 0.80: 3, 0.30: 4, 0.50: 5}
KINDS: tuple[str, ...] = ("targeted", "random")
DATASET_INDEX: dict[str, int] = {"acm": 0, "dblp": 1, "imdb": 2}
TOLERANCE = 0.01
REMOVAL_CAP = 0.5
DEFAULT_MAX_ADD_RATIO = 4.0
FILL_SEED = 0                       # the reference's primary fill seed (`--fill-seeds` default starts at 0)
CHUNK_SIZE = 200                    # calcHomophily's default (compute_heterophily `--chunk-size`)
NODE_REF_MAX_EDGES = 30_000_000     # compute_heterophily `--node-ref-max-edges`
CROSS_CHECK_TOL = 1e-6              # compute_heterophily `--cross-check-tol`
BISECTION_STEPS = 40


def graph_seed(dataset: str, level: float, kind: str) -> int:
    return 1000 * DATASET_INDEX[dataset] + 10 * LEVEL_SEED_INDEX[round(float(level), 2)] + KINDS.index(kind)


def level_dir(data_root: Path, dataset: str, kind: str, level: float) -> Path:
    """`<data_root>/NC/<DATASET>_pert/<kind>_h<level>`; its `data/` sub-folder is the graph's data root."""
    return nc_root(data_root) / f"{ref.HGB_DATASETS[dataset]}_pert" / f"{kind}_h{level:.2f}"


# ------------------------------------------------------------------------------------------------ the graph
@dataclass
class FactRelation:
    forward: int                    # link type id whose (start, end) orients facts as (target, other)
    reverse: int | None             # mirrored link type (None: self-mirrored type)
    target_side: str                # "src": facts are (target, x); always oriented so
    other_type: str
    name: str

    @property
    def target_target(self) -> bool:
        return self.other_type == self.target_type

    target_type: str = ""


@dataclass
class Graph:
    dataset: str
    data: Any                       # PyG HeteroData from the reference loader
    raw_dir: Path
    target: str
    link_types: list[tuple[str, str, str]]          # link type id -> PyG edge type
    global_ids: dict[str, np.ndarray]               # node type -> global id per local index
    weight_token: str
    relations: list[FactRelation]
    facts: dict[str, np.ndarray]                    # relation name -> [n, 2] (target local, other local)
    labels: np.ndarray                              # class id per target (-1 unlabelled; IMDB: label-set id)
    num_nodes: dict[str, int] = field(default_factory=dict)


def _edges(data, edge_type) -> np.ndarray:
    return data[edge_type].edge_index.t().cpu().numpy().astype(np.int64)


def _keys(pairs: np.ndarray, width: int) -> np.ndarray:
    return pairs[:, 0].astype(np.int64) * width + pairs[:, 1].astype(np.int64)


def class_ids(y: torch.Tensor) -> np.ndarray:
    """Single label: the class (-1 unlabelled). Multilabel (IMDB): an id per distinct non-empty label set (-1 for
    the empty set, PyG's unlabelled rows)."""
    if y.dim() == 1:
        return y.cpu().numpy().astype(np.int64)
    rows = [tuple(r) for r in y.cpu().numpy().astype(np.int64).tolist()]
    ids: dict[tuple, int] = {}
    out = np.full(len(rows), -1, dtype=np.int64)
    for i, r in enumerate(rows):
        if any(r):
            out[i] = ids.setdefault(r, len(ids))
    return out


def load_graph(data_root: Path, dataset: str) -> Graph:
    data = ref.load_hgb(data_root, dataset)
    name = ref.HGB_DATASETS[dataset]
    raw_dir = nc_root(data_root) / dataset / "raw" / name
    target = ref.target_type(data)
    info = json.loads((raw_dir / "info.dat").read_text(encoding="utf-8"))
    n_types = {int(k): v for k, v in info["node.dat"]["node type"].items()}
    links = {int(k): v for k, v in info["link.dat"]["link type"].items()}
    link_types = list(data.edge_types)
    if len(link_types) != len(links):
        raise ValueError(f"{dataset}: {len(links)} link types in info.dat, {len(link_types)} in the loaded graph")
    for k, edge_type in enumerate(link_types):   # PyG keeps info.dat's link order (hgb_dataset.py process)
        if (n_types[int(links[k]["start"])], n_types[int(links[k]["end"])]) != (edge_type[0], edge_type[2]):
            raise ValueError(f"{dataset}: link type {k} {links[k]} does not match {edge_type}")
    global_ids: dict[str, list[int]] = {t: [] for t in n_types.values()}
    with (raw_dir / "node.dat").open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            global_ids[n_types[int(parts[2])]].append(int(parts[0]))
    weights = Counter()
    with (raw_dir / "link.dat").open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            weights[line.rstrip("\n").split("\t")[3]] += 1
            if i > 10_000:
                break
    num_nodes = {t: int(data[t].num_nodes) for t in data.node_types}
    # Fact relations: a link type and its mirrored type (equal edge sets up to reversal).
    used: set[int] = set()
    relations: list[FactRelation] = []
    facts: dict[str, np.ndarray] = {}
    m2 = {et for path in ref.h2gb_metapaths(data, target) for et in path}
    for k, edge_type in enumerate(link_types):
        if k in used:
            continue
        e = _edges(data, edge_type)
        width = max(num_nodes[edge_type[0]], num_nodes[edge_type[2]]) + 1
        mirror = None
        for j, other in enumerate(link_types):
            if j == k or j in used or (other[0], other[2]) != (edge_type[2], edge_type[0]):
                continue
            o = _edges(data, other)
            if o.shape == e.shape and np.array_equal(np.sort(_keys(o[:, ::-1], width)), np.sort(_keys(e, width))):
                mirror = j
                break
        if mirror is None:
            raise ValueError(f"{dataset}: link type {edge_type} has no mirrored type (a fact needs both)")
        used |= {k, mirror}
        if edge_type not in m2 and link_types[mirror] not in m2:
            continue
        forward = k if edge_type[0] == target else mirror
        fwd = link_types[forward]
        if fwd[0] != target:
            raise ValueError(f"{dataset}: M_2 relation {edge_type} does not touch the target type {target}")
        rel = FactRelation(forward=forward, reverse=mirror if forward == k else k, target_side="src",
                           other_type=fwd[2], name=f"{fwd[0]}-{fwd[1]}-{fwd[2]}", target_type=target)
        relations.append(rel)
        facts[rel.name] = _edges(data, fwd)
    return Graph(dataset=dataset, data=data, raw_dir=raw_dir, target=target, link_types=link_types,
                 global_ids={t: np.asarray(v, dtype=np.int64) for t, v in global_ids.items()},
                 weight_token=weights.most_common(1)[0][0], relations=relations, facts=facts,
                 labels=class_ids(data[target].y), num_nodes=num_nodes)


# ------------------------------------------------------------------------------------------------ h (reference)
def search_labels(graph: Graph) -> list[torch.Tensor]:
    """The labels the reference's primary value uses: single label with its fill (seed 0) when a target is
    unlabelled; IMDB one filled binary label per genre (`genre_fill_seed(0, g)`)."""
    y = graph.data[graph.target].y
    if y.dim() > 1:
        return [ref.h2gb_random_fill(y_g, ref.genre_fill_seed(FILL_SEED, g))
                for g, y_g in enumerate(ref.genre_labels(y))]
    if bool((y < 0).any()):
        return [ref.h2gb_random_fill(y, FILL_SEED)]
    return [y]


def with_facts(graph: Graph, facts: dict[str, np.ndarray]):
    """A shallow copy of the loaded graph with the M_2 relations' edges replaced by `facts` (both directions)."""
    data = graph.data.clone()
    for rel in graph.relations:
        pairs = torch.as_tensor(facts[rel.name], dtype=torch.long)
        data[graph.link_types[rel.forward]].edge_index = pairs.t().contiguous()
        data[graph.link_types[rel.reverse]].edge_index = pairs.flip(1).t().contiguous()
    return data


def heterophily(graph: Graph, facts: dict[str, np.ndarray], labels: list[torch.Tensor] | None = None) -> float:
    """h = 1 - mean over M_2 (and over genres for IMDB) of the edge homophily of G_P, by the reference's
    `homophily_counts` (empty G_P skipped, as calcHomophily does)."""
    data = with_facts(graph, facts)
    labels = search_labels(graph) if labels is None else labels
    values = []
    for y in labels:
        for path in ref.h2gb_metapaths(data, graph.target):
            counts = ref.homophily_counts(data, path, y, CHUNK_SIZE)
            if counts.num_edges:
                values.append(counts.h_edge)
    return 1.0 - float(np.mean(values))


def reference_values(graph_root: Path, dataset: str) -> dict[str, float]:
    """The reported values: the reference `analyse` on the graph loaded back from its files (`load_hgb`)."""
    data = ref.load_hgb(graph_root, dataset)
    result = ref.analyse(data, dataset, "perturbed", [FILL_SEED], CHUNK_SIZE, NODE_REF_MAX_EDGES, CROSS_CHECK_TOL)
    return {"h": result.mlh_h2gb_heterophily_mean, "h2_index": result.h2_index}


# ------------------------------------------------------------------------------------------------ candidates
def _neighbours(facts: np.ndarray, num_other: int) -> list[np.ndarray]:
    """Targets adjacent to each `other` node (facts are (target, other))."""
    order = np.argsort(facts[:, 1], kind="stable")
    sorted_facts = facts[order]
    bounds = np.searchsorted(sorted_facts[:, 1], np.arange(num_other + 1))
    return [sorted_facts[bounds[i]:bounds[i + 1], 0] for i in range(num_other)]


def _others_of_target(facts: np.ndarray, num_target: int) -> list[np.ndarray]:
    order = np.argsort(facts[:, 0], kind="stable")
    sorted_facts = facts[order]
    bounds = np.searchsorted(sorted_facts[:, 0], np.arange(num_target + 1))
    return [sorted_facts[bounds[i]:bounds[i + 1], 1] for i in range(num_target)]


def targeted_additions(graph: Graph, rel: FactRelation, count: int, rng: np.random.Generator
                       ) -> tuple[np.ndarray, np.ndarray]:
    """`count` new facts (p, x) of `rel`, each with its meta-path partner q (y(q) != y(p)); returns (facts, q)."""
    labels = graph.labels
    labelled = np.flatnonzero(labels >= 0)
    classes = np.unique(labels[labelled])
    by_class = {int(c): labelled[labels[labelled] == c] for c in classes}
    existing = graph.facts[rel.name]
    num_other = graph.num_nodes[rel.other_type]
    width = num_other + 1
    seen = set(_keys(existing, width).tolist())
    others_of = None if rel.target_target else _others_of_target(existing, graph.num_nodes[rel.target_type])
    out, partners = [], []
    attempts = 0
    while len(out) < count:
        attempts += 1
        if attempts > 50 * count + 10_000:
            raise RuntimeError(f"{graph.dataset} {rel.name}: could not draw {count} new different-class facts")
        p = int(labelled[int(rng.integers(labelled.size))])
        own_class = int(labels[p])
        other_classes = [int(c) for c in classes if int(c) != own_class]
        partner_class = other_classes[int(rng.integers(len(other_classes)))]
        pool = by_class[partner_class]
        q = int(pool[int(rng.integers(pool.size))])
        if rel.target_target:
            x = q
        else:
            neighbours = others_of[q]
            if neighbours.size == 0:
                continue
            x = int(neighbours[int(rng.integers(neighbours.size))])
        if rel.target_target and x == p:
            continue
        key = p * width + x
        if key in seen:
            continue
        seen.add(key)
        out.append((p, x))
        partners.append(q)
    return np.asarray(out, dtype=np.int64).reshape(-1, 2), np.asarray(partners, dtype=np.int64)


def removal_candidates(graph: Graph, rel: FactRelation, rng: np.random.Generator) -> np.ndarray:
    """Indices (into `graph.facts[rel.name]`) of the facts that create a different-class meta-path pair, in a
    seeded uniform order."""
    labels = graph.labels
    facts = graph.facts[rel.name]
    p_label = labels[facts[:, 0]]
    if rel.target_target:
        q_label = labels[facts[:, 1]]
        mask = (p_label >= 0) & (q_label >= 0) & (p_label != q_label)
    else:
        num_other = graph.num_nodes[rel.other_type]
        num_classes = int(labels.max()) + 1
        counts = np.zeros((num_other, num_classes), dtype=np.int64)
        lab = labels[facts[:, 0]]
        keep = lab >= 0
        np.add.at(counts, (facts[keep, 1], lab[keep]), 1)
        total = counts.sum(axis=1)
        own = np.where(p_label >= 0, counts[facts[:, 1], np.maximum(p_label, 0)], 0)
        mask = (p_label >= 0) & (total[facts[:, 1]] - own > 0)
    index = np.flatnonzero(mask)
    return index[rng.permutation(index.size)]


def random_additions(graph: Graph, rel: FactRelation, count: int, rng: np.random.Generator) -> np.ndarray:
    n_t, n_o = graph.num_nodes[rel.target_type], graph.num_nodes[rel.other_type]
    width = n_o + 1
    seen = set(_keys(graph.facts[rel.name], width).tolist())
    out = []
    while len(out) < count:
        p, x = int(rng.integers(n_t)), int(rng.integers(n_o))
        if (rel.target_target and p == x) or p * width + x in seen:
            continue
        seen.add(p * width + x)
        out.append((p, x))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


# ------------------------------------------------------------------------------------------------ search
@dataclass
class Perturbation:
    dataset: str
    kind: str
    level: float
    seed: int
    direction: str                                  # raise / lower / none
    alpha: float
    added: dict[str, np.ndarray]                    # relation -> [k, 2] new facts
    removed: dict[str, np.ndarray]                  # relation -> [k, 2] removed facts
    partners: dict[str, np.ndarray] = field(default_factory=dict)   # targeted additions: q per new fact
    search_h: float = float("nan")
    reached: bool = False
    bound_h: float | None = None                    # h at the bound (cap / addition bound) when unreached

    def facts(self, graph: Graph) -> dict[str, np.ndarray]:
        out = {}
        for rel in graph.relations:
            base = graph.facts[rel.name]
            if self.removed[rel.name].size:
                width = graph.num_nodes[rel.other_type] + 1
                drop = np.isin(_keys(base, width), _keys(self.removed[rel.name], width))
                base = base[~drop]
            out[rel.name] = np.concatenate([base, self.added[rel.name]]) if self.added[rel.name].size else base
        return out

    def counts(self) -> dict[str, dict[str, int]]:
        return {name: {"added": int(self.added[name].shape[0]), "removed": int(self.removed[name].shape[0])}
                for name in self.added}


def _empty(graph: Graph) -> dict[str, np.ndarray]:
    return {rel.name: np.zeros((0, 2), dtype=np.int64) for rel in graph.relations}


def targeted(graph: Graph, level: float, max_add_ratio: float = DEFAULT_MAX_ADD_RATIO,
             h0: float | None = None) -> Perturbation:
    seed = graph_seed(graph.dataset, level, "targeted")
    rng = np.random.default_rng(seed)
    h0 = heterophily(graph, graph.facts) if h0 is None else h0
    sizes = {rel.name: graph.facts[rel.name].shape[0] for rel in graph.relations}
    if abs(h0 - level) <= TOLERANCE:
        p = Perturbation(graph.dataset, "targeted", level, seed, "none", 0.0, _empty(graph), _empty(graph))
        p.search_h, p.reached = h0, True
        return p
    if h0 < level:
        streams, partners = {}, {}
        for rel in graph.relations:   # one ordered stream per relation: alpha selects a prefix
            streams[rel.name], partners[rel.name] = targeted_additions(
                graph, rel, int(round(max_add_ratio * sizes[rel.name])), rng)

        def build(alpha: float) -> Perturbation:
            take = {n: int(round(alpha * sizes[n])) for n in sizes}
            p = Perturbation(graph.dataset, "targeted", level, seed, "raise", alpha,
                             {n: streams[n][:take[n]] for n in sizes}, _empty(graph),
                             {n: partners[n][:take[n]] for n in sizes})
            return p
        low, high = 0.0, max_add_ratio
    else:
        order = {rel.name: removal_candidates(graph, rel, rng) for rel in graph.relations}

        def build(alpha: float) -> Perturbation:
            removed = {}
            for rel in graph.relations:
                n = sizes[rel.name]
                take = min(int(round(alpha * n)), int(REMOVAL_CAP * n), order[rel.name].size)
                removed[rel.name] = graph.facts[rel.name][np.sort(order[rel.name][:take])]
            return Perturbation(graph.dataset, "targeted", level, seed, "lower", alpha, _empty(graph), removed)
        low, high = 0.0, REMOVAL_CAP
    best: Perturbation | None = None
    for _ in range(BISECTION_STEPS):
        alpha = (low + high) / 2
        candidate = build(alpha)
        candidate.search_h = heterophily(graph, candidate.facts(graph))
        if best is None or abs(candidate.search_h - level) < abs(best.search_h - level):
            best = candidate
        if abs(candidate.search_h - level) <= TOLERANCE:
            break
        moved_too_far = candidate.search_h > level if h0 < level else candidate.search_h < level
        if moved_too_far:
            high = alpha
        else:
            low = alpha
    bound_h = None
    if abs(best.search_h - level) > TOLERANCE:          # the bound itself, in case the search never got there
        edge = build(max_add_ratio if h0 < level else REMOVAL_CAP)
        edge.search_h = bound_h = heterophily(graph, edge.facts(graph))
        if abs(edge.search_h - level) < abs(best.search_h - level):
            best = edge
    best.bound_h = bound_h
    best.reached = abs(best.search_h - level) <= TOLERANCE
    return best


def random_control(graph: Graph, target_graph: Perturbation) -> Perturbation:
    seed = graph_seed(graph.dataset, target_graph.level, "random")
    rng = np.random.default_rng(seed)
    added, removed = _empty(graph), _empty(graph)
    for rel in graph.relations:
        n_add = target_graph.added[rel.name].shape[0]
        n_remove = target_graph.removed[rel.name].shape[0]
        if n_add:
            added[rel.name] = random_additions(graph, rel, n_add, rng)
        if n_remove:
            facts = graph.facts[rel.name]
            removed[rel.name] = facts[np.sort(rng.choice(facts.shape[0], size=n_remove, replace=False))]
    p = Perturbation(graph.dataset, "random", target_graph.level, seed, target_graph.direction, target_graph.alpha,
                     added, removed)
    p.search_h = heterophily(graph, p.facts(graph))
    p.reached = target_graph.reached
    return p


# ------------------------------------------------------------------------------------------------ writing
def write_graph(graph: Graph, perturbation: Perturbation, out_dir: Path) -> Path:
    """HGB files of the perturbed graph under `out_dir/data/NC/<name>/raw/<NAME>/`; returns the graph's data root."""
    name = ref.HGB_DATASETS[graph.dataset]
    root = out_dir / "data"
    raw = nc_root(root) / graph.dataset / "raw" / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    raw.mkdir(parents=True)
    for path in graph.raw_dir.iterdir():   # unchanged release files: hard links (read-only use), else copies
        if path.is_file() and path.name != "link.dat":
            try:
                os.link(path, raw / path.name)
            except OSError:
                shutil.copy2(path, raw / path.name)
    removed_lines: set[tuple[int, int, int]] = set()
    added_lines: list[str] = []
    for rel in graph.relations:
        t_ids = graph.global_ids[rel.target_type]
        o_ids = graph.global_ids[rel.other_type]
        for p, x in perturbation.removed[rel.name].tolist():
            gp, gx = int(t_ids[p]), int(o_ids[x])
            removed_lines |= {(gp, gx, rel.forward), (gx, gp, rel.reverse)}
        for p, x in perturbation.added[rel.name].tolist():
            gp, gx = int(t_ids[p]), int(o_ids[x])
            added_lines += [f"{gp}\t{gx}\t{rel.forward}\t{graph.weight_token}",
                            f"{gx}\t{gp}\t{rel.reverse}\t{graph.weight_token}"]
    dropped = 0
    with (graph.raw_dir / "link.dat").open(encoding="utf-8") as src, \
            (raw / "link.dat").open("w", encoding="utf-8") as dst:
        for line in src:
            parts = line.rstrip("\n").split("\t")
            if (int(parts[0]), int(parts[1]), int(parts[2])) in removed_lines:
                dropped += 1
                continue
            dst.write(line)
        for line in added_lines:
            dst.write(line + "\n")
    if dropped != len(removed_lines):
        raise RuntimeError(f"{out_dir}: {len(removed_lines)} lines to remove, {dropped} found")
    return root


def raw_dir_of(root: Path, dataset: str) -> Path:
    return nc_root(root) / dataset / "raw" / ref.HGB_DATASETS[dataset]


def fingerprint(root: Path, dataset: str) -> str:
    """SHA-256 of the written `link.dat` (the file every vendor loader reads)."""
    return hashlib.sha256((raw_dir_of(root, dataset) / "link.dat").read_bytes()).hexdigest()


def class_ceiling(root: Path, dataset: str) -> dict[str, Any]:
    """C (PyG's `num_classes`: the classes listed in `info.dat`'s `label.dat` block) and the ceiling 1 - 1/C."""
    info = json.loads((raw_dir_of(root, dataset) / "info.dat").read_text(encoding="utf-8"))
    c = len(info["label.dat"]["node type"]["0"])
    return {"num_classes": c, "ceiling": 1.0 - 1.0 / c}


def edge_fields(root: Path, dataset: str) -> dict[str, Any]:
    edge = fp.from_raw(raw_dir_of(root, dataset))
    return {"edge_fingerprint": edge["sha256"], "edge_pairs": edge["pairs"], **class_ceiling(root, dataset)}


def backfill(data_root: Path, dataset: str, log=print) -> list[dict[str, Any]]:
    """Adds `edge_fingerprint`, `edge_pairs`, C and the ceiling to every graph of `dataset` already written; refuses a
    graph whose `link.dat` changed since `graph.json` was written."""
    records = []
    folder = nc_root(data_root) / f"{ref.HGB_DATASETS[dataset]}_pert"
    for path in sorted(folder.glob("*_h*/graph.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        root = path.parent / "data"
        if fingerprint(root, dataset) != record["link_sha256"]:
            raise RuntimeError(f"{path.parent}: link.dat changed since graph.json was written")
        record.update(edge_fields(root, dataset))
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        records.append(record)
        log(f"{dataset} {record['kind']} {record['level']:.2f}: edge fingerprint {record['edge_fingerprint'][:12]} "
            f"({record['edge_pairs']} pairs), C {record['num_classes']}")
    return records


def generate(data_root: Path, dataset: str, max_add_ratio: float = DEFAULT_MAX_ADD_RATIO,
             levels: tuple[float, ...] = LEVELS, log=print) -> list[dict[str, Any]]:
    graph = load_graph(data_root, dataset)
    h0 = heterophily(graph, graph.facts)
    original = reference_values(data_root, dataset)
    log(f"{dataset}: search h {h0:.4f}, reference h {original['h']:.4f}, H2 {original['h2_index']:.4f}, "
        f"M_2 relations {[r.name for r in graph.relations]}")
    records = []
    for level in levels:
        t = targeted(graph, level, max_add_ratio, h0)
        for p in (t, random_control(graph, t)):
            out_dir = level_dir(data_root, dataset, p.kind, level)
            root = write_graph(graph, p, out_dir)
            values = reference_values(root, dataset)
            record = {"dataset": dataset, "kind": p.kind, "level": level, "seed": p.seed,
                      "direction": p.direction, "alpha": p.alpha, "reached": bool(p.reached),
                      "search_h": p.search_h, "h": values["h"], "h_at_bound": p.bound_h, "h2_index": values["h2_index"],
                      "original_h": original["h"], "original_h2_index": original["h2_index"],
                      "facts": p.counts(), "facts_per_relation_before": {r.name: int(graph.facts[r.name].shape[0])
                                                                         for r in graph.relations},
                      "removal_cap": REMOVAL_CAP, "max_add_ratio": max_add_ratio, "tolerance": TOLERANCE,
                      "data_root": str(root), "link_sha256": fingerprint(root, dataset),
                      **edge_fields(root, dataset)}
            if abs(values["h"] - p.search_h) > 1e-5:
                raise AssertionError(f"{out_dir}: reference h {values['h']} != search h {p.search_h}")
            (out_dir / "graph.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
            records.append(record)
            log(f"{dataset} {p.kind} {level:.2f}: h {values['h']:.4f} H2 {values['h2_index']:.4f} "
                f"reached {p.reached} {p.direction} alpha {p.alpha:.4f} {p.counts()}")
    return records


def markdown(records: list[dict[str, Any]]) -> str:
    lines = ["# Perturbation reachability (card 7 V3)", "",
             "| dataset | level | kind | reached | h | H^2 | ceiling 1 - 1/C | direction | "
             "facts added / removed per relation | seed | edge fingerprint (pairs) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        counts = "; ".join(f"{n}: +{c['added']} / -{c['removed']}" for n, c in r["facts"].items())
        status = "yes" if r["reached"] else f"**unreached** (closest {r['h']:.4f})"
        ceiling = f"{r['ceiling']:.4f}" if r.get("ceiling") is not None else "-"
        edge = f"`{r['edge_fingerprint'][:12]}` ({r['edge_pairs']})" if r.get("edge_fingerprint") else "-"
        lines.append(f"| {r['dataset']} | {r['level']:.2f} | {r['kind']} | {status} | {r['h']:.4f} | "
                     f"{r['h2_index']:.4f} | {ceiling} | {r['direction']} | {counts} | {r['seed']} | {edge} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["acm", "dblp"], choices=["acm", "dblp"])
    parser.add_argument("--include_imdb", action="store_true")
    parser.add_argument("--data_root", type=Path, default=REPO_DATA_ROOT)
    parser.add_argument("--max_add_ratio", type=float, default=DEFAULT_MAX_ADD_RATIO)
    parser.add_argument("--levels", type=float, nargs="+", default=list(LEVELS), choices=list(LEVELS))
    parser.add_argument("--backfill", action="store_true",
                        help="add the edge fingerprint, C and the ceiling to graphs already written; generate nothing")
    parser.add_argument("--out", type=Path, required=True, help="reachability JSON (a .md is written beside it)")
    args = parser.parse_args()
    datasets = list(args.datasets) + (["imdb"] if args.include_imdb else [])
    records = []
    for dataset in datasets:
        if args.backfill:
            records += backfill(args.data_root, dataset)
        else:
            records += generate(args.data_root, dataset, args.max_add_ratio, tuple(args.levels))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=1), encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(records), encoding="utf-8")


if __name__ == "__main__":
    main()
