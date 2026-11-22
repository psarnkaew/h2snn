"""MAGNN (card 8 C3; decisions log 2026-11-20 "Card 8: R-GCN and MAGNN", "Card 8 amended: no MAGNN in GC").

Model code is HGB's (`vendor/HGB/{NC,LP}/benchmark/methods/MAGNN/model`), never edited: `MAGNN_nc_mb` for node
classification, `MAGNN_lp` (one layer per endpoint side) for link prediction.

* **Instances.** A meta-path instance is stored as MAGNN stores it (`utils/data.py: get_adjlist_pkl`): per centre node
  an adjacency row `"c n1 n2 ..."` (far ends, sorted, with multiplicity) and the instance array, each row the path
  reversed (far end first, centre last), rows sorted. HGB Freebase NC (published, `run_Freebase.py`) builds them with
  HGB's own `get_adjlist_pkl` on HGB's `data_loader` over our raw directory and the meta-paths of `meta1.json` /
  `meta2.json` (`run_Freebase.py:20-27`, edge ids and `num_edge_type` 72 as written); every other cell enumerates them
  here on our graph (`path_instances`, `magnn_lists`, same format), hop by hop over the distinct (u, v) pairs of a hop.
* **Meta-paths.** NC Freebase: MAGNN's lists (above). NC H2GB (probe cells): the symmetric length-2 schemas `t–X–t`
  over every neighbour type X of the target type t (card 8 C3 rule; one schema per X over every relation between t
  and X). LP: per endpoint type, the paths of HGB's published table that start at it
  (`_h2gformer_encoders.listed_paths`, `docs/predefined_metapaths.md`; IMDB actor: AMA, AMDMA; LastFM: MAGNN's own
  lists and edge ids, `run_LastFM.py:21-25, 28-31`); an endpoint type without a listed path takes the `t–X–t` rule on
  the message graph. Edge ids for the RotatE aggregator: relation r forward `2r`, reverse `2r + 1` (MAGNN's
  convention, e.g. DBLP `[0, 1]` = A→P, P→A); type-level hops of the rule get ids after the relations.
* **Sampling.** NC: HGB's `parse_minibatch` (`utils/tools.py`), unchanged. LP: `parse_side` re-implements
  `parse_adjlist_LastFM` (`LP/.../utils/tools.py:129-185`) for any endpoint types, because HGB's `parse_minibatch_LastFM`
  hard-codes `user`/`artist` offsets and `.to('cuda')` (`:210`): neighbour sampling with probability ∝ count^(3/4)
  per distinct neighbour, without replacement, `samples` neighbours (including its reassignment of `samples` inside the
  loop, as written), a node without neighbours gets itself, and the **training mask** (`use_masks`): for a positive
  training pair, instances whose first or last hop is a positive pair of the step are dropped (MAGNN's rule; on the
  schemas whose end hops are the target relation).
* **Held-out facts.** LP instances are enumerated on the message graph (held-out pairs removed in both orientations by
  the bundle); `assert_no_held_out` checks every hop of every instance against the held-out pairs and raises. Labels:
  NC enumeration reads no label at all; LP reads none either (link labels are the step's pairs, used only by the
  training mask, which removes them).
* **Training.** NC HGB: mini-batches of `batch_size` targets (MAGNN's runner), our selection (validation Macro-F1,
  patience) and evaluation; Adam for a published cell (`run_Freebase.py:83`), AdamW for a tuned one. NC H2GB: the shared
  index mini-batch loop (`minibatch.train_index_minibatch`). LP small: the shared LP loop with a per-step pair scorer
  (its `score_links` argument: only the step's endpoints are encoded, as MAGNN's runner), its
  decoder replaced by the shared head. LP large (probe cells): `fit_large_lp`, MAGNN's NC layer as the encoder of each
  sampled subgraph with the `t–X–t` rule inside the subgraph (no training mask; stated).
* **Empty batches (stated).** MAGNN's code fails when no target of a batch has any instance of a meta-path
  (`base_MAGNN.py:150` reshapes an empty tensor; HGB's README: "every node has at least one neighbour for every
  meta-path"). Only in that case each target of the batch gets one self-instance, MAGNN LP's own rule for a node
  without neighbours (`LP/.../utils/tools.py:178-182`); otherwise sampling is unchanged.
* **Vendor modules.** Every block pushes MAGNN's folder together with its HGB root (`hgb_nc` / `hgb_lp`):
  `pushd_sys_path` attributes a module under `HGB/*/benchmark/methods/` to that root, and drops on exit only the
  modules of the pushed roots, so pushing the folder alone would leave MAGNN's `utils` and `model` in `sys.modules`
  (found by card 8 C4: HAN's `import utils` then got MAGNN's).
* NC ACM, DBLP, IMDB hold published numbers and are not run; their MAGNN preprocessing (ACM's added self-loops and
  sub-graph, IMDB's full-batch `MAGNN_nc`) is not wired (registry reason). GC: not run (no meta-paths in the TU graphs).
"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import torch

from src.baselines import search_grids as grids
from src.baselines.adapters._magnn_args import run_arguments
from src.baselines.budget import budget_kwargs, resolve_budget
from src.baselines.protocol import LINK_PREDICTION, NODE_CLASSIFICATION, BaselineParams, UnsupportedCell
from src.baselines.published import load_published_params
from src.baselines.runtime import pushd_sys_path
from src.baselines.runtime._dgl_compat import resolve_dgl_device
from src.utils.misc import seed_everything

NAME = "magnn"
VENDOR_NC = "hgb_magnn_nc"
VENDOR_LP = "hgb_magnn_lp"
HGB_NC_DATASETS = frozenset({"freebase"})
NOT_WIRED_NC = frozenset({"acm", "dblp", "imdb"})
H2GB_DATASETS = frozenset({"ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns"})
SMALL_LP_DATASETS = frozenset({"lastfm", "pubmed", "acm_pa", "acm_pp", "dblp_ap", "imdb_ma", "freebase_bb"})
LARGE_LP_DATASETS = frozenset({"ogbn-mag_pp", "h-pokec_uu"})
#: Every H2GB NC and large-LP path reads the shared batch table (`budget.batch_plan`): H2GB NC (probe cells) through
#: `minibatch.minibatch_plan` and `train_index_minibatch`; large LP (probe cells) through
#: `large_lp_training.fit_large_lp` (`LP_POSITIVES_PER_STEP`, the protocol budget).
READS_BATCH_PLAN = True
#: Freebase: MAGNN's `num_edge_type` as written (`run_Freebase.py:81`).
FREEBASE_NUM_EDGE_TYPE = 72
#: LastFM (published): MAGNN's lists (`run_LastFM.py:21-25, 28-31`) over our relations rel_0 user-artist,
#: rel_1 user-user, rel_2 artist-tag (as `_h2gformer_encoders.LASTFM_PATHS`); MAGNN's edge ids 0 U->A, 1 A->U,
#: 2 A->T, 3 T->A, None = U-U (symmetric, no rotation); num_edge_type 4 (`run_LastFM.py:69`).
LASTFM_SIDES = (
    (  # user side: UAU, UATAU, UU
        ((("user", "rel_0", "artist"), ("artist", "rev_rel_0", "user")), [0, 1], True),
        ((("user", "rel_0", "artist"), ("artist", "rel_2", "tag"), ("tag", "rev_rel_2", "artist"),
          ("artist", "rev_rel_0", "user")), [0, 2, 3, 1], True),
        ((("user", "rel_1", "user"),), [None], False),
    ),
    (  # artist side: AUA, ATA, AUUA
        ((("artist", "rev_rel_0", "user"), ("user", "rel_0", "artist")), [1, 0], True),
        ((("artist", "rel_2", "tag"), ("tag", "rev_rel_2", "artist")), [2, 3], False),
        ((("artist", "rev_rel_0", "user"), ("user", "rel_1", "user"), ("user", "rel_0", "artist")),
         [1, None, 0], True),
    ),
)
LASTFM_NUM_EDGE_TYPE = 4
#: IMDB actor side of the HGB table (`docs/predefined_metapaths.md` §2: AMA, AMDMA), as node-type sequences.
TYPE_SEQUENCE_PATHS: dict[tuple[str, str], tuple[tuple[str, ...], ...]] = {
    ("imdb_ma", "actor"): (("actor", "movie", "actor"), ("actor", "movie", "director", "movie", "actor")),
}


# ================================================================================================ instances

def _unique_pairs(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct (u, v) pairs of a hop (HGB's sparse link matrices hold each pair once)."""
    if src.size == 0:
        return src.astype(np.int64), dst.astype(np.int64)
    pairs = np.unique(np.stack([src, dst], axis=1).astype(np.int64), axis=0)
    return pairs[:, 0], pairs[:, 1]


def path_instances(hops: list[tuple[np.ndarray, np.ndarray]], num_nodes: int) -> np.ndarray:
    """Every walk `v0 -> v1 -> ... -> vL` along the hops (each hop: source and destination global ids), `[m, L+1]`."""
    src, dst = _unique_pairs(*hops[0])
    paths = np.stack([src, dst], axis=1)
    for hop_src, hop_dst in hops[1:]:
        s, d = _unique_pairs(hop_src, hop_dst)
        order = np.argsort(s, kind="stable")
        s, d = s[order], d[order]
        indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        np.add.at(indptr, s + 1, 1)
        indptr = np.cumsum(indptr)
        last = paths[:, -1]
        counts = indptr[last + 1] - indptr[last]
        starts = np.repeat(indptr[last], counts)
        offsets = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
        paths = np.concatenate([np.repeat(paths, counts, axis=0), d[starts + offsets][:, None]], axis=1)
    return paths


def magnn_lists(paths: np.ndarray, centers: np.ndarray) -> tuple[dict[int, str], dict[int, np.ndarray]]:
    """MAGNN's (adjacency rows, instance arrays) of `paths` (`v0` = centre), for every node of `centers`."""
    length = paths.shape[1]
    rows = paths[:, ::-1]  # far end first, centre last (get_adjlist_pkl: tuple(reversed(path)))
    keys = [rows[:, j] for j in reversed(range(length))] + [rows[:, -1]]
    rows = rows[np.lexsort(keys)] if rows.size else rows
    adjlist: dict[int, str] = {}
    indices: dict[int, np.ndarray] = {}
    bounds = np.searchsorted(rows[:, -1], centers, side="left") if rows.size else np.zeros(len(centers), int)
    ends = np.searchsorted(rows[:, -1], centers, side="right") if rows.size else np.zeros(len(centers), int)
    for center, lo, hi in zip(centers.tolist(), bounds.tolist(), ends.tolist()):
        block = rows[lo:hi] if rows.size else np.empty((0, length), dtype=np.int64)
        indices[center] = block.astype(np.int64)
        adjlist[center] = " ".join(map(str, [center] + block[:, 0].tolist()))
    return adjlist, indices


def assert_no_held_out(paths: np.ndarray, held_out: set[tuple[int, int]]) -> None:
    """No hop of any instance is a held-out pair, in either orientation."""
    if not held_out or paths.size == 0:
        return
    for j in range(paths.shape[1] - 1):
        for u, v in zip(paths[:, j].tolist(), paths[:, j + 1].tolist()):
            if (u, v) in held_out or (v, u) in held_out:
                raise AssertionError(f"held-out fact ({u}, {v}) on a meta-path instance")


# ================================================================================================ graphs and schemas

class TypedGraph:
    """Global-id edges per relation (src type, name, dst type), node slices, the type of every node."""

    def __init__(self, node_types: tuple[str, ...], slices: dict[str, tuple[int, int]],
                 relations: tuple[tuple[str, str, str], ...], edges: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.node_types, self.slices, self.relations, self.edges = node_types, slices, relations, edges
        self.num_nodes = max(hi for _, hi in slices.values())
        self.type_mask = np.zeros(self.num_nodes, dtype=np.int64)
        for index, name in enumerate(node_types):
            lo, hi = slices[name]
            self.type_mask[lo:hi] = index

    @classmethod
    def from_global(cls, node_types: tuple[str, ...], slices: dict[str, tuple[int, int]],
                    relations: tuple[tuple[str, str, str], ...], edge_index: torch.Tensor,
                    edge_type: torch.Tensor) -> "TypedGraph":
        ei, et = edge_index.cpu().numpy(), edge_type.cpu().numpy()
        return cls(node_types, dict(slices), tuple(relations),
                   [(ei[0][et == r], ei[1][et == r]) for r in range(len(relations))])

    @classmethod
    def from_hetero(cls, data: Any) -> "TypedGraph":
        node_types = tuple(data.node_types)
        slices, start = {}, 0
        for name in node_types:
            slices[name] = (start, start + int(data[name].num_nodes))
            start += int(data[name].num_nodes)
        relations, edges = [], []
        for src_t, rel, dst_t in data.edge_types:
            index = data[src_t, rel, dst_t].edge_index
            relations.append((src_t, rel, dst_t))
            edges.append(((index[0] + slices[src_t][0]).numpy(), (index[1] + slices[dst_t][0]).numpy()))
        return cls(node_types, slices, tuple(relations), edges)

    def hop(self, hop: tuple[str, str, str]) -> tuple[tuple[np.ndarray, np.ndarray], int, bool]:
        """(edges, relation id, forward?) of a relation-level hop `(a, rel, b)` or `(a, rev_rel, b)`."""
        a, rel, b = hop
        if hop in self.relations:
            r = self.relations.index(hop)
            return self.edges[r], r, True
        if rel.startswith("rev_") and (b, rel[4:], a) in self.relations:
            r = self.relations.index((b, rel[4:], a))
            s, d = self.edges[r]
            return (d, s), r, False
        # The listed relation was dropped by the task (its exact reverse is kept): `dropped_hop` (card 18) -- the
        # direction follows the types, and on a same-type pair (ACM cited-by) the kept relation is walked backwards.
        from src.baselines.adapters._h2gformer_encoders import dropped_hop

        r, forward = dropped_hop(hop, self.relations)
        s, d = self.edges[r]
        return ((s, d), r, True) if forward else ((d, s), r, False)

    def type_pairs(self) -> list[tuple[str, str]]:
        return sorted({tuple(sorted((s, d))) for s, _, d in self.relations})

    def type_hop(self, a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
        """Every relation between types a and b, oriented a -> b (type-level adjacency)."""
        src, dst = [], []
        for (s_t, _, d_t), (s, d) in zip(self.relations, self.edges):
            if (s_t, d_t) == (a, b):
                src.append(s); dst.append(d)
            if (s_t, d_t) == (b, a) and a != b:
                src.append(d); dst.append(s)
            if (s_t, d_t) == (b, a) and a == b:  # same-type relation: both orientations
                src.extend([s, d]); dst.extend([d, s])
        if not src:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        return np.concatenate(src), np.concatenate(dst)


class Schema:
    """One meta-path of one centre type: its hops (edge arrays), MAGNN edge ids, and whether the training mask applies."""

    def __init__(self, name: str, hops: list[tuple[np.ndarray, np.ndarray]], etypes: list, use_mask: bool) -> None:
        self.name, self.hops, self.etypes, self.use_mask = name, hops, etypes, use_mask


def rule_schemas(graph: TypedGraph, center: str, *, id_base: int, target_pair: tuple[str, str] | None = None
                 ) -> list[Schema]:
    """The `t–X–t` schemas of type t = `center` over every neighbour type X (card 8 C3 rule)."""
    out = []
    for k, (a, b) in enumerate(graph.type_pairs()):
        if center not in (a, b):
            continue
        other = b if a == center else a
        out_hop = graph.type_hop(center, other)
        back_hop = graph.type_hop(other, center)
        if out_hop[0].size == 0:
            continue
        mask = target_pair is not None and {center, other} == set(target_pair)
        out.append(Schema(f"{center}-{other}-{center}", [out_hop, back_hop],
                          [id_base + 2 * k, id_base + 2 * k + 1], mask))
    return out


def listed_schemas(graph: TypedGraph, paths: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...],
                   target: tuple[str, str, str] | None) -> list[Schema]:
    """Relation-level listed paths (HGB's table) as schemas; the mask applies when an end hop is the target relation."""
    out = []
    for name, hops in paths:
        edges, ids = [], []
        for hop in hops:
            e, r, forward = graph.hop(hop)
            edges.append(e)
            ids.append(2 * r if forward else 2 * r + 1)
        rel_ids = [graph.hop(h)[1] for h in (hops[0], hops[-1])]
        mask = target is not None and graph.relations.index(target) in rel_ids
        out.append(Schema(name, edges, ids, mask))
    return out


def type_sequence_schemas(graph: TypedGraph, sequences: tuple[tuple[str, ...], ...], *, id_base: int,
                          target_pair: tuple[str, str] | None) -> list[Schema]:
    out = []
    pairs = graph.type_pairs()
    for seq in sequences:
        edges, ids = [], []
        for a, b in zip(seq, seq[1:]):
            k = pairs.index(tuple(sorted((a, b))))
            edges.append(graph.type_hop(a, b))
            ids.append(id_base + 2 * k + (0 if (a, b) == pairs[k] else 1))
        mask = target_pair is not None and ({seq[0], seq[1]} == set(target_pair) or {seq[-1], seq[-2]} ==
                                            set(target_pair))
        out.append(Schema("".join(t[0].upper() for t in seq), edges, ids, mask))
    return out


def lp_sides(task: str, graph: TypedGraph, target: tuple[str, str, str]
             ) -> tuple[list[list[Schema]], int]:
    """([head-side schemas, tail-side schemas], num_edge_type) of an LP task (module docstring, Meta-paths)."""
    if task == "lastfm":
        sides = []
        for side in LASTFM_SIDES:
            sides.append([Schema("-".join(h[0][0].upper() for h in hops) + hops[-1][2][0].upper(),
                                 [graph.hop(h)[0] for h in hops], list(ids), mask) for hops, ids, mask in side])
        return sides, LASTFM_NUM_EDGE_TYPE
    from src.baselines.adapters._h2gformer_encoders import DERIVED_SOURCE, listed_paths

    head_t, _, tail_t = target
    base = 2 * len(graph.relations)
    listed = listed_paths(task) if task in DERIVED_SOURCE else ()
    sides = []
    for t in (head_t, tail_t):
        starting = tuple(p for p in listed if p[1][0][0] == t)
        if starting:
            sides.append(listed_schemas(graph, starting, target))
        elif (task, t) in TYPE_SEQUENCE_PATHS:
            sides.append(type_sequence_schemas(graph, TYPE_SEQUENCE_PATHS[(task, t)], id_base=base,
                                               target_pair=(head_t, tail_t)))
        else:
            sides.append(rule_schemas(graph, t, id_base=base, target_pair=(head_t, tail_t)))
    return sides, base + 2 * len(graph.type_pairs())


def schema_lists(graph: TypedGraph, schemas: list[Schema], centers: np.ndarray,
                 held_out: set[tuple[int, int]] | None = None) -> tuple[list[dict], list[dict]]:
    adjlists, indices = [], []
    for schema in schemas:
        paths = path_instances(schema.hops, graph.num_nodes)
        if held_out is not None:
            assert_no_held_out(paths, held_out)
        a, i = magnn_lists(paths, centers)
        adjlists.append(a)
        indices.append(i)
    return adjlists, indices


# ================================================================================================ sampling (LP)

def parse_side(adjlist: dict[int, str], indices: dict[int, np.ndarray], nodes: list[int], samples: int | None,
               exclude: set[tuple[int, int]] | None, device: torch.device) -> tuple[Any, torch.Tensor, np.ndarray]:
    """`parse_adjlist_LastFM` + the graph build of `parse_minibatch_LastFM` (HGB LP MAGNN `utils/tools.py:129-210`) for
    any endpoint type, global ids; `exclude`: unordered positive pairs of the step (training mask) or None."""
    import dgl

    edges, nodes_seen, result = [], set(), []
    for node in nodes:
        row = list(map(int, adjlist[node].split(" ")))
        inst = indices[node]
        nodes_seen.add(row[0])
        if len(row) > 1:
            if samples is None:
                chosen = np.arange(len(row) - 1)
            else:
                unique, counts = np.unique(row[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row) - 1)  # reassigned inside the loop, as written (tools.py:161)
                chosen = np.sort(np.random.choice(len(row) - 1, samples, replace=False, p=p))
            picked = inst[chosen]
            neighbors = np.array([row[i + 1] for i in chosen])
            if exclude is not None:
                keep = np.array([not (_pair(a, b) in exclude or _pair(c, d) in exclude)
                                 for a, b, c, d in picked[:, [0, 1, -1, -2]].tolist()], dtype=bool)
                picked, neighbors = picked[keep], neighbors[keep]
            result.append(picked)
        else:
            neighbors = np.array([row[0]])
            result.append(np.array([[row[0]] * inst.shape[1]]))
        for dst in neighbors.tolist():
            nodes_seen.add(dst)
            edges.append((row[0], dst))
    if not edges:  # no instance in the whole batch: one self-instance per target (module docstring, Empty batches)
        width = next(iter(indices.values())).shape[1] if indices else 1
        result = [np.array([[node] * width]) for node in nodes]
        edges = [(node, node) for node in nodes]
    mapping = {old: new for new, old in enumerate(sorted(nodes_seen))}
    edges = [(mapping[a], mapping[b]) for a, b in edges]
    result_indices = np.vstack(result) if result else np.empty((0, 1), dtype=np.int64)
    graph = dgl.graph(([], []), num_nodes=len(nodes_seen))
    if edges:
        order = sorted(range(len(edges)), key=lambda i: edges[i])
        graph = dgl.graph((torch.tensor([edges[i][1] for i in order]), torch.tensor([edges[i][0] for i in order])),
                          num_nodes=len(nodes_seen))
        result_indices = result_indices[order]
    return graph.to(device), torch.as_tensor(result_indices, dtype=torch.long).to(device), \
        np.array([mapping[n] for n in nodes])


def _pair(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


# ================================================================================================ models

def _features(features_list: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [f.coalesce().to(device) if f.is_sparse else f.to(device) for f in features_list]


def build_nc_model(num_metapaths: int, num_edge_type: int, etypes_list: list, in_dims: list[int], hidden: int,
                   out_dim: int, heads: int, attn_vec_dim: int, rnn_type: str, dropout: float) -> torch.nn.Module:
    with pushd_sys_path(VENDOR_NC, "hgb_nc"):  # parent root too: its modules are dropped on exit (C4 fix)
        from model import MAGNN_nc_mb  # type: ignore[import-not-found]

        return MAGNN_nc_mb(num_metapaths, num_edge_type, etypes_list, in_dims, hidden, out_dim, heads, attn_vec_dim,
                           rnn_type, dropout)


def build_lp_model(num_metapaths: list[int], num_edge_type: int, etypes_lists: list, in_dims: list[int],
                   hidden: int, heads: int, attn_vec_dim: int, rnn_type: str, dropout: float) -> torch.nn.Module:
    with pushd_sys_path(VENDOR_LP, "hgb_lp"):  # parent root too (C4 fix)
        from model import MAGNN_lp  # type: ignore[import-not-found]

        # run_LastFM.py:69: out_dim = hidden_dim
        return MAGNN_lp(num_metapaths, num_edge_type, etypes_lists, in_dims, hidden, hidden, heads, attn_vec_dim,
                        rnn_type, dropout)


_VENDOR_FUNCTIONS: dict[str, Any] = {}


def nc_parse(adjlists: list, indices: list, idx_batch: np.ndarray, device: torch.device, samples: int | None):
    """HGB's `parse_minibatch` (NC `utils/tools.py:104-126`), unchanged."""
    import dgl

    if "parse_minibatch" not in _VENDOR_FUNCTIONS:  # imported once; the function keeps its own module globals
        with pushd_sys_path(VENDOR_NC, "hgb_nc"):  # parent root too (C4 fix)
            from utils.tools import parse_minibatch  # type: ignore[import-not-found]

            _VENDOR_FUNCTIONS["parse_minibatch"] = parse_minibatch
    g_list, indices_list, mapped_list = _VENDOR_FUNCTIONS["parse_minibatch"](adjlists, indices, idx_batch, device,
                                                                             samples)
    for k, (inst, mapped) in enumerate(zip(indices_list, mapped_list)):
        if inst.numel() == 0:  # no instance in the whole batch (module docstring, Empty batches)
            width = next(iter(indices[k].values())).shape[1] if isinstance(indices[k], dict) else \
                np.asarray(indices[k][0]).reshape(len(indices[k][0]), -1).shape[1]
            loops = torch.as_tensor(mapped, dtype=torch.long)
            g_list[k] = dgl.graph((loops, loops), num_nodes=int(g_list[k].num_nodes())).to(device)
            indices_list[k] = torch.as_tensor(np.repeat(np.asarray(idx_batch)[:, None], width, axis=1),
                                              dtype=torch.long, device=device)
    return g_list, indices_list, mapped_list


# ================================================================================================ adapter

def _optimizer(parameters: Any, args: dict[str, Any]) -> torch.optim.Optimizer:
    cls = torch.optim.Adam if args.get("optimizer") == "adam" else torch.optim.AdamW
    return cls(parameters, lr=float(args["lr"]), weight_decay=float(args["weight_decay"]))


class MAGNNAdapter:
    name = NAME

    def supports(self, task: str, dataset: str) -> bool:
        if task == NODE_CLASSIFICATION:
            return dataset in HGB_NC_DATASETS or dataset in H2GB_DATASETS
        if task == LINK_PREDICTION:
            return dataset in SMALL_LP_DATASETS or dataset in LARGE_LP_DATASETS
        return False

    def search_space(self, task: str, dataset: str) -> dict[str, Any]:
        # MAGNN's argparse keys (run_*.py): hidden-dim, num-heads, attn-vec-dim; samples and rnn-type stay at
        # MAGNN's 100 and RotatE0; the shared lr / weight-decay / dropout axes.
        space: dict[str, Any] = {"lr": grids.LR, "weight_decay": grids.WEIGHT_DECAY, "dropout": grids.DROPOUT,
                                 "hidden_dim": {"values": [32, 64, 128]}, "num_heads": {"values": [4, 8]},
                                 "attn_vec_dim": {"values": [64, 128]}}
        if task == NODE_CLASSIFICATION and dataset in H2GB_DATASETS:
            from src.configs.config import MINIBATCH_BATCH_SIZES

            space["batch_size"] = {"values": list(MINIBATCH_BATCH_SIZES)}
            return space
        if task == NODE_CLASSIFICATION and dataset in HGB_NC_DATASETS:
            space["batch_size"] = {"values": [8, 32, 128]}
            return space
        if task == LINK_PREDICTION:
            space["decoder_type"] = {"values": ["dot", "distmult"]}
            return space
        raise UnsupportedCell(NAME, task, dataset, "MAGNN runs NC and LP only")

    def published_params(self, task: str, dataset: str) -> BaselineParams | None:
        return load_published_params(NAME, task, dataset)

    def fit_task(self, bundle: Any, params: BaselineParams, *, task: str, seed: int,
                 device: torch.device) -> dict[str, float]:
        device = resolve_dgl_device(device)
        seed_everything(seed, device)
        np.random.seed(seed)  # MAGNN samples with numpy's global generator (tools.py)
        if task == NODE_CLASSIFICATION and bundle.dataset in HGB_NC_DATASETS:
            return self._fit_freebase(bundle, params, device=device)
        if task == NODE_CLASSIFICATION and bundle.dataset in H2GB_DATASETS:
            return self._fit_h2gb(bundle, params, seed=seed, device=device)
        if task == LINK_PREDICTION and bundle.dataset in LARGE_LP_DATASETS:
            return self._fit_large_lp(bundle, params, seed=seed, device=device)
        if task == LINK_PREDICTION:
            return self._fit_link_prediction(bundle, params, device=device)
        raise UnsupportedCell(NAME, task, bundle.dataset, "not wired (module docstring)")

    # ------------------------------------------------------------------------------------------ NC
    def _fit_freebase(self, bundle: Any, params: BaselineParams, *, device: torch.device) -> dict[str, float]:
        """`run_Freebase.py` with our split and evaluation: its meta-paths and instances (HGB's `get_adjlist_pkl` on
        HGB's `data_loader` over our raw directory), `MAGNN_nc_mb(len(etypes_list), 72, ...)`, batches of
        `batch_size` training targets, `neighbor_samples` per meta-path."""
        from src.baselines.bundles import bundle_data_root, hgb_raw_dir
        from src.baselines.runtime.vendor_paths import vendor_path

        args = run_arguments(dict(params.values), tuned_budget=None if params.source == "published"
                             else _budget(NODE_CLASSIFICATION, params, bundle.dataset))
        root = vendor_path(VENDOR_NC)
        etypes_list, paths = [], []
        for fn in ("meta1.json", "meta2.json"):  # run_Freebase.py:20-27, verbatim
            meta = json.loads((root / fn).read_text(encoding="utf-8"))
            for x in meta["node_0"][:2]:
                tmp = list(map(int, x["path"].split(",")))
                tmp = tmp + [-x - 1 for x in reversed(tmp)]
                etypes_list.append([x if x >= 0 else x + 36 for x in tmp])
                paths.append(list(map(int, x["path"].split(","))))
        raw = hgb_raw_dir(bundle.dataset, bundle_data_root(bundle))
        with pushd_sys_path(VENDOR_NC, "hgb_nc"):  # MAGNN's own `utils` first; `scripts` from HGB's root
            from scripts.data_loader import data_loader  # type: ignore[import-not-found]
            from utils.data import get_adjlist_pkl  # type: ignore[import-not-found]

            dl = data_loader(str(raw))
            adjlists, idxs = [], []
            for path in paths:  # utils/data.py:120-125 (load_Freebase_data)
                full = path + [-x - 1 for x in reversed(path)]
                adj, idx = get_adjlist_pkl(dl, full, symmetric=True)
                adjlists.append(adj)
                idxs.append(idx)
        data = bundle.data
        start, end = data.node_slices[data.target_node_type]
        if start != 0 or dl.nodes["shift"][0] != 0 or dl.nodes["count"][0] != end:
            raise ValueError("Freebase: the target type is not HGB's node type 0 at offset 0")
        view = bundle.homogeneous(feats_type=int(args["feats_type"]))
        model = build_nc_model(len(etypes_list), FREEBASE_NUM_EDGE_TYPE, etypes_list, view.in_dims,
                               int(args["hidden_dim"]), bundle.num_classes, int(args["num_heads"]),
                               int(args["attn_vec_dim"]), str(args["rnn_type"]), float(args["dropout_rate"])).to(device)
        features = _features(view.features_list, device)
        type_mask = view.node_type.cpu().numpy()
        samples = int(args["neighbor_samples"])

        def forward_rows(idx: torch.Tensor) -> torch.Tensor:
            batch = np.sort(idx.cpu().numpy())
            order = np.argsort(np.argsort(idx.cpu().numpy()))
            g_list, indices_list, mapped = nc_parse(adjlists, idxs, batch, device, samples)
            logits, _ = model((g_list, features, type_mask, indices_list, mapped))
            return logits[torch.as_tensor(order, device=logits.device)]

        record = _train_nc_minibatch(model, forward_rows, data, _optimizer(model.parameters(), args), args,
                                     report=params.report, device=device)
        record["params_source"] = params.source
        return record

    def _fit_h2gb(self, bundle: Any, params: BaselineParams, *, seed: int, device: torch.device) -> dict[str, float]:
        """Probe cells: MAGNN's NC model with the `t–X–t` rule on the H2GB graph, the shared index loop."""
        from src.baselines.minibatch import h2gb_graph, minibatch_plan, train_index_minibatch
        from src.data.h2gb_nc import ALL_H2GB_NC_SPECS

        args = run_arguments(dict(params.values), tuned_budget=None)
        data = h2gb_graph(bundle)
        target = ALL_H2GB_NC_SPECS[bundle.dataset].target_node_type
        graph = TypedGraph.from_hetero(data)
        schemas = rule_schemas(graph, target, id_base=0)
        lo, hi = graph.slices[target]
        adjlists, indices = schema_lists(graph, schemas, np.arange(lo, hi))
        features, in_dims = _h2gb_features(data)
        model = build_nc_model(len(schemas), 2 * len(graph.type_pairs()), [s.etypes for s in schemas], in_dims,
                               int(args["hidden_dim"]), int(data[target].y.max()) + 1, int(args["num_heads"]),
                               int(args["attn_vec_dim"]), str(args["rnn_type"]), float(args["dropout_rate"])).to(device)
        features = _features(features, device)
        samples = int(args["neighbor_samples"])

        def forward_rows(idx: torch.Tensor) -> torch.Tensor:
            local = idx.cpu().numpy()
            batch = np.sort(local) + lo
            order = np.argsort(np.argsort(local))
            g_list, indices_list, mapped = nc_parse(adjlists, indices, batch, device, samples)
            logits, _ = model((g_list, features, graph.type_mask, indices_list, mapped))
            return logits[torch.as_tensor(order, device=logits.device)]

        plan = minibatch_plan(bundle.dataset, int(params.get("batch_size", 256)))
        masks = {s: data[target][f"{s}_mask"].nonzero().view(-1) for s in ("train", "val", "test")}
        record = train_index_minibatch(model=model, forward_rows=forward_rows, labels=data[target].y.long().to(device),
                                       train_idx=masks["train"], val_idx=masks["val"], test_idx=masks["test"],
                                       optimizer=_optimizer(model.parameters(), args),
                                       criterion=torch.nn.functional.cross_entropy, plan=plan, base_seed=seed,
                                       report=params.report, run_seed=bundle.metadata.get("seed"))
        record["params_source"] = params.source
        record["batch_size"] = float(plan.batch_size)
        return record

    # ------------------------------------------------------------------------------------------ LP
    def lp_setup(self, bundle: Any, args: dict[str, Any], device: torch.device
                 ) -> tuple[torch.nn.Module, Callable[..., tuple[torch.Tensor, torch.Tensor]], list[list[Schema]]]:
        """(MAGNN_lp, encode(heads, tails, exclude) -> (head rows, tail rows), schemas) on the message graph."""
        data = bundle.data
        graph = TypedGraph.from_global(tuple(data.node_type_names), dict(data.node_slices), tuple(data.relation_tuples),
                                       data.edge_index, data.edge_type)
        target = tuple(data.target_relation)
        sides, num_edge_type = lp_sides(bundle.dataset, graph, target)
        held = torch.cat([data.val_pos_edge_index, data.test_pos_edge_index], dim=1).cpu().t().tolist()
        held_out = {(int(u), int(v)) for u, v in held}
        lists = []
        for side, t in zip(sides, (target[0], target[2])):
            lo, hi = graph.slices[t]
            lists.append(schema_lists(graph, side, np.arange(lo, hi), held_out))
        view = bundle.homogeneous(feats_type=int(args["feats_type"]))
        model = build_lp_model([len(s) for s in sides], num_edge_type, [[x.etypes for x in s] for s in sides],
                               view.in_dims, int(args["hidden_dim"]), int(args["num_heads"]),
                               int(args["attn_vec_dim"]), str(args["rnn_type"]), float(args["dropout_rate"])).to(device)
        features = _features(view.features_list, device)
        type_mask = view.node_type.cpu().numpy()
        samples = int(args["neighbor_samples"])

        def encode(heads: torch.Tensor, tails: torch.Tensor, exclude: set[tuple[int, int]] | None
                   ) -> tuple[torch.Tensor, torch.Tensor]:
            g_lists, idx_lists, mapped_lists = [[], []], [[], []], [[], []]
            for mode, (nodes, side) in enumerate(zip((heads, tails), sides)):
                nodes = nodes.cpu().tolist()
                for schema, adj, inst in zip(side, *lists[mode]):
                    g, idx, mapped = parse_side(adj, inst, nodes, samples, exclude if schema.use_mask else None, device)
                    g_lists[mode].append(g)
                    idx_lists[mode].append(idx)
                    mapped_lists[mode].append(mapped)
            [h_head, h_tail], _ = model((g_lists, features, type_mask, idx_lists, mapped_lists))
            return h_head, h_tail

        return model, encode, sides

    def _fit_link_prediction(self, bundle: Any, params: BaselineParams, *, device: torch.device) -> dict[str, float]:
        from src.baselines.heads import SharedLinkHead
        from src.baselines.training import train_link_prediction

        args = run_arguments(dict(params.values), tuned_budget=None)
        if "feats_type" not in params.values:
            args["feats_type"] = 3  # MAGNN LP feats-type 0 = all id vectors (run_LastFM.py:225) = our 3
        model, encode, _ = self.lp_setup(bundle, args, device)
        data = bundle.data.to(device)
        head = SharedLinkHead(str(params.get("decoder_type", "dot")),
                              num_relations=max(int(data.num_target_relations), 1),
                              dim=int(args["hidden_dim"])).to(device)
        optimizer = _optimizer(list(model.parameters()) + list(head.parameters()), args)

        def score_links(edges: torch.Tensor, relation: torch.Tensor, exclude: torch.Tensor | None) -> torch.Tensor:
            pairs = None if exclude is None else {_pair(int(u), int(v)) for u, v in exclude.cpu().t().tolist()}
            h_head, h_tail = encode(edges[0], edges[1], pairs)
            return head.decoder(relation.to(device), h_head, h_tail)

        record = train_link_prediction(model=model, head=head, embed=None, data=data, optimizer=optimizer,
                                       device=device, score_links=score_links,
                                       **budget_kwargs(LINK_PREDICTION, params))
        record["params_source"] = params.source
        return record

    def build_large_lp(self, d: Any, params: Any, device: torch.device, sampler: Any = None) -> tuple:
        """Probe cells: MAGNN's NC layer (`MAGNN_nc_mb`, out_dim = hidden) as the encoder of each sampled subgraph, the
        `t–X–t` rule inside the subgraph, seeds = the batch's first rows of the endpoint type (no training mask;
        stated). Sampler: H2GB's hgt sampler with the GCN yaml fan-out of the task's NC dataset (class C rule)."""
        from src.baselines.large_lp_training import LargeLPModel, h2gb_sampler
        from src.baselines.minibatch import batch_type_inputs, h2gb_input_dims

        values = params.values if isinstance(getattr(params, "values", None), dict) else dict(params)
        args = run_arguments(dict(values), tuned_budget=None)
        samples = int(args["neighbor_samples"])
        node_types, counts, dims = h2gb_input_dims(d.graph)
        sampler = sampler or h2gb_sampler("gcn", d.task, node_types, force_hgt=True)
        pairs = sorted({tuple(sorted((s_t, t_t))) for s_t, _, t_t in d.graph.edge_types})
        around = [p for p in pairs if d.node_type in p]
        hidden = int(args["hidden_dim"])
        encoder = build_nc_model(len(around), 2 * len(pairs), [[2 * pairs.index(p), 2 * pairs.index(p) + 1]
                                                               for p in around],
                                 dims, hidden, hidden, int(args["num_heads"]), int(args["attn_vec_dim"]),
                                 str(args["rnn_type"]), float(args["dropout_rate"]))

        def embed(batch: Any, train: bool) -> torch.Tensor:
            graph = TypedGraph.from_hetero(batch)
            if tuple(graph.node_types) != tuple(node_types):
                raise ValueError("batch node types differ from the graph's")
            lo = graph.slices[d.node_type][0]
            rows = int(batch[d.node_type].num_nodes)
            seeds = lo + np.arange(rows)
            schemas = [s for s in rule_schemas(graph, d.node_type, id_base=0)]
            # edge ids of the encoder: by type pair over the whole graph's pairs
            for schema in schemas:
                a, _, b = schema.name.split("-")
                k = pairs.index(tuple(sorted((a, b))))
                schema.etypes = [2 * k, 2 * k + 1]
            adjlists, indices = schema_lists(graph, schemas, seeds)
            inputs = [x.to(device) for x in batch_type_inputs(batch, node_types, counts)]
            g_list, idx_list, mapped = nc_parse(adjlists, indices, seeds, device, samples)
            out, _ = encoder((g_list, inputs, graph.type_mask, idx_list, mapped))
            return out

        return LargeLPModel(encoder, embed, hidden), sampler

    def _fit_large_lp(self, bundle: Any, params: BaselineParams, *, seed: int, device: torch.device
                      ) -> dict[str, float]:
        from src.baselines.large_lp_training import fit_large_lp

        record = fit_large_lp(bundle, params, seed=seed, device=device,
                              build=lambda d: self.build_large_lp(d, params, device))
        record["params_source"] = params.source
        return record


def _budget(task: str, params: BaselineParams, dataset: str) -> dict[str, int]:
    budget = resolve_budget(task, params, dataset)
    return {"epochs": budget.epochs, "patience": budget.patience}


def _h2gb_features(data: Any) -> tuple[list[torch.Tensor], list[int]]:
    feats, dims = [], []
    for name in data.node_types:
        x = data[name].get("x")
        if x is None:
            n = int(data[name].num_nodes)
            idx = torch.arange(n)
            x = torch.sparse_coo_tensor(torch.stack([idx, idx]), torch.ones(n), (n, n))
        feats.append(x.float() if not x.is_sparse else x)
        dims.append(int(x.size(1)))
    return feats, dims


def _train_nc_minibatch(model: torch.nn.Module, forward_rows: Callable[[torch.Tensor], torch.Tensor], data: Any,
                        optimizer: torch.optim.Optimizer, args: dict[str, Any], *, report: Any,
                        device: torch.device) -> dict[str, float]:
    """MAGNN's NC runner (`run_Freebase.py:91-150`): per epoch, shuffled batches of `batch_size` training targets, one
    step each; then the shared selection: validation Macro-F1 every epoch, patience, test at the best epoch."""
    from src.baselines.metrics import classification_metrics, node_classification_record
    from src.baselines.progress import prefixed_metrics, report_epoch
    from src.models.anchored_solve import check_finite_loss

    labels = data.target_y.to(device).long()
    split = {s: getattr(data, f"target_{s}_mask").nonzero().view(-1) for s in ("train", "val", "test")}
    batch_size = int(args["batch_size"])

    last_loss: dict[str, float] = {}  # card 8 C7: the last evaluation's loss, logged only

    def evaluate(index: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            logits = torch.cat([forward_rows(index[i:i + batch_size]) for i in range(0, index.numel(), batch_size)])
            last_loss["value"] = float(torch.nn.functional.cross_entropy(logits, labels[index.to(device)]))
        return classification_metrics(logits, labels[index.to(device)], is_multilabel=False)

    best, best_epoch, stale, best_val, best_test = float("-inf"), 0, 0, {}, {}
    for epoch in range(int(args["num_epochs"])):
        model.train()
        order = split["train"][torch.randperm(split["train"].numel())]
        losses: list[float] = []  # card 8 C7: train/loss, logged only
        for start in range(0, order.numel(), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(forward_rows(batch), labels[batch.to(device)])
            check_finite_loss(loss, epoch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        model.eval()
        val = evaluate(split["val"])
        extra = {"train/loss": sum(losses) / max(1, len(losses)), "val/loss": last_loss.get("value")}
        if val["macro_f1"] > best:  # MAGNN's runner evaluates the test set only at the end: test/* when improved
            best, best_epoch, stale, best_val, best_test = val["macro_f1"], epoch, 0, val, evaluate(split["test"])
            extra.update(prefixed_metrics("test", best_test))
        else:
            stale += 1
        report_epoch(report, epoch=epoch, monitor=val["macro_f1"], running_best=best, metrics=val, extra=extra)
        if stale >= int(args["patience"]):
            break
    return node_classification_record(val_metrics=best_val, test_metrics=best_test, best_epoch=best_epoch)


ADAPTER = MAGNNAdapter()

__all__ = ["ADAPTER", "MAGNNAdapter", "Schema", "TypedGraph", "assert_no_held_out", "lp_sides", "magnn_lists",
           "parse_side", "path_instances", "rule_schemas", "schema_lists"]
