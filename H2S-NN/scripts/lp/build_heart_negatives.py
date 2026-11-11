"""HeaRT hard negatives for the link-prediction tasks (spec v1.8 §6 and F8; task card J4.4).

HeaRT (Li et al., NeurIPS 2023 D&B; code Juanhui28/HeaRT c447cbf,
`heart_negatives/create_heart_negatives.py`), adapted as spec v1.8 fixes it. Every parameter and
its `file:line` source is in `../docs/lp_evidence/j4_preregistration.md`.

Per positive (a, b) of the validation file and of the test file, 500 negatives: 250 pairs (a, v)
that corrupt the destination, then 250 pairs (u, b) that corrupt the source (HeaRT `:220-226`).
- Candidates: every node of the target relation's type on the corrupted side (spec adaptation).
- Heuristics, each scored kept endpoint vs candidate, on training facts only:
  - same-type targets (pubmed, freebase_bb), C1: HeaRT's `calc_CN_metric` exactly as its code
    computes it (`:69-75`), which is CN(a, v) / deg(v) (verified by running those lines; see
    `j44_record.md`), on the loader's training message graph made homogeneous, symmetric, binary,
    without self-loops (`--cn-metric ra` gives textbook resource allocation, a variant);
  - cross-type targets (acm_pa, dblp_ap, imdb_ma, lastfm), C1: the degree-normalised L3 score of
    Kovács et al. 2019, Eq. (1), on the target relation's training facts;
  - every task: PPR by Andersen push, alpha 0.15, eps 5e-5 (`calc_ppr.py:19-110`), on the same
    homogeneous training graph;
  - C2: cosine feature similarity (`:82-91`) where both endpoint types share one feature space
    (pubmed; acm_pa, verified in J4.0).
- Mask (score -1): every target-relation positive of any split in either orientation with the
  kept endpoint, the kept endpoint itself and the other endpoint (HeaRT `:178-206`, stricter).
- Rank each heuristic as `rank_score_matrix` (`:94-116`), combine by min (`:124`, `:141-143`),
  select as `rank_and_merge_node` (`:145-162`) with `torch.topk`; when fewer than 250 candidates
  score above zero, the rest is drawn uniformly **without replacement** (C4) from the zero-score
  candidates not already selected, with `numpy.default_rng(seed)` consumed in positive order. A
  pool smaller than the need stops the run (card stop condition).

Positives follow the loader (`_load_local_hgb_link_data`): validation = the fixed `link.dat.valid`
(C3, J4.2), test = `link.dat.test`. Output, in raw HGB node ids, under `<task folder>/heart/`:
`heart_{valid,test}_pos.npy` (n x 2), `heart_{valid,test}_samples.npy` (n x 500 x 2, HeaRT's
layout), `meta.json`.

Run from H2S-NN/: python scripts/lp/build_heart_negatives.py [--tasks ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import TrainConfig  # noqa: E402
from src.data.data_layout import REPO_DATA_ROOT, lp_root  # noqa: E402
from src.data.link_prediction_dataset import (  # noqa: E402
    _LOCAL_HGB_LINK_DATASETS,
    _load_local_hgb_link_data,
    _load_local_hgb_nodes,
)

Relation = tuple[str, str, str]

#: HeaRT's parameters (create_heart_negatives.py:270, :123, :268-269, :273-274; hard_utils.py:14-18).
NUM_SAMPLES = 500
PPR_ALPHA = 0.15
PPR_EPS = 5e-5
SEED = 42


@dataclass(frozen=True, slots=True)
class HeartTask:
    dataset: str  # name in _LOCAL_HGB_LINK_DATASETS
    target: Relation
    pair_heuristic: str  # "cn" (same-type, C1) or "l3" (cross-type, C1)
    features: bool  # C2: both endpoint types share one feature space


HEART_TASKS: dict[str, HeartTask] = {
    "pubmed": HeartTask("pubmed", ("disease", "rel_2", "disease"), "cn", True),
    "freebase_bb": HeartTask("freebase_bb", ("book", "rel_0", "book"), "cn", False),
    # F17 (J6): author features are removed, so C2 (cosine in one shared space) no longer applies.
    "acm_pa": HeartTask("acm_pa", ("paper", "rel_2", "author"), "l3", False),
    "dblp_ap": HeartTask("dblp_ap", ("author", "rel_0", "paper"), "l3", False),
    "imdb_ma": HeartTask("imdb_ma", ("movie", "rel_2", "actor"), "l3", False),
    "lastfm": HeartTask("lastfm", ("user", "rel_0", "artist"), "l3", False),
}


class PoolTooSmall(RuntimeError):
    """C4: a positive's zero-score pool cannot fill its list without replacement (stop)."""


# ------------------------------------------------------------------------------ task data
@dataclass(slots=True)
class TaskData:
    task: HeartTask
    folder: Path
    type_sizes: dict[str, int]
    offsets: dict[str, int]
    train_by_relation: dict[Relation, torch.Tensor]  # local ids, the loader's training facts
    train_pos: np.ndarray  # target relation, local (src, dst)
    valid_pos: np.ndarray
    test_pos: np.ndarray
    features: dict[str, np.ndarray] = field(default_factory=dict)
    local_to_raw: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return sum(self.type_sizes.values())

    @property
    def same_type(self) -> bool:
        return self.task.target[0] == self.task.target[2]


def task_config(data_root: Path, task: HeartTask) -> TrainConfig:
    return TrainConfig.from_dict({"device": "cpu", "data": {
        "dataset_name": task.dataset, "task_type": "link_prediction", "root": str(data_root),
        "target_relation": list(task.target)}})


def load_task(data_root: Path, task: HeartTask) -> TaskData:
    """The loader's split (J4.2: validation = link.dat.valid) of the task's dataset."""
    spec = _LOCAL_HGB_LINK_DATASETS[task.dataset]
    folder = lp_root(data_root) / str(spec["folder"])
    split = _load_local_hgb_link_data(task_config(data_root, task))
    types = sorted(split.data.node_types)
    sizes = {t: int(split.data[t].num_nodes) for t in types}
    offsets, total = {}, 0
    for t in types:
        offsets[t] = total
        total += sizes[t]
    target = task.target

    def pos(edges: torch.Tensor) -> np.ndarray:
        return edges.t().numpy().astype(np.int64).reshape(-1, 2)

    nodes = _load_local_hgb_nodes(folder, spec["node_type_names"])
    local_to_raw = {t: np.zeros(sizes[t], dtype=np.int64) for t in types}
    for raw, local in nodes.raw_to_local.items():
        local_to_raw[nodes.raw_to_type_name[raw]][local] = raw
    data = TaskData(
        task=task, folder=folder, type_sizes=sizes, offsets=offsets,
        train_by_relation=split.train_edge_index_by_relation,
        train_pos=pos(split.train_edge_index_by_relation[target]),
        valid_pos=pos(split.valid_edge_index_by_relation[target]),
        test_pos=pos(split.test_edge_index_by_relation[target]),
        local_to_raw=local_to_raw,
    )
    if task.features:
        for t in {target[0], target[2]}:
            data.features[t] = split.data[t].x.numpy()
        widths = {t: x.shape[1] for t, x in data.features.items()}
        if len(set(widths.values())) != 1:
            raise ValueError(f"C2 needs one feature space; widths {widths}.")
    return data


# ------------------------------------------------------------------------------ heuristics
def heuristic_graph(data: TaskData) -> sp.csr_matrix:
    """The training message graph as HeaRT's `adj_t`: every relation's training facts (the
    loader's message graph), homogeneous over all node types, symmetric, binary, de-duplicated,
    self-loops removed (HeaRT drops them when reading, hard_utils.py:110-111)."""
    rows, cols = [], []
    for (src, _, dst), edges in data.train_by_relation.items():
        if edges.numel() == 0:
            continue
        rows.append(edges[0].numpy() + data.offsets[src])
        cols.append(edges[1].numpy() + data.offsets[dst])
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    keep = row != col
    row, col = row[keep], col[keep]
    n = data.num_nodes
    a = sp.coo_matrix((np.ones(2 * row.size, dtype=np.float32),
                       (np.concatenate([row, col]), np.concatenate([col, row]))), shape=(n, n))
    a = a.tocsr()
    a.data[:] = 1.0  # binary after summing duplicates
    a.eliminate_zeros()
    return a


def target_biadjacency(data: TaskData) -> sp.csr_matrix:
    """The target relation's training facts as a binary S x D matrix (L3's `a`)."""
    src, _, dst = data.task.target
    p = data.train_pos
    b = sp.coo_matrix((np.ones(len(p), dtype=np.float64), (p[:, 0], p[:, 1])),
                      shape=(data.type_sizes[src], data.type_sizes[dst])).tocsr()
    b.data[:] = 1.0
    return b


def candidate_columns(adj: sp.csr_matrix, candidates: np.ndarray) -> tuple[sp.csr_matrix,
                                                                             np.ndarray]:
    """(A[:, candidates], deg(candidates)) with deg = column sums of A (HeaRT `adj.sum(dim=0)`)."""
    deg = np.asarray(adj.sum(axis=0)).ravel()
    return adj[:, candidates].tocsr(), deg[candidates]


def cn_heart_rows(adj: sp.csr_matrix, sources: np.ndarray,
                  cand_cols: tuple[sp.csr_matrix, np.ndarray]) -> np.ndarray:
    """HeaRT's `calc_CN_metric` with metric RA, as its code computes it (`:69-75`):
    `cn = A @ (A * (1 / A.sum(dim=0)))`, where `*` scales column v by 1 / deg(v) (torch_sparse
    `mul`, col-wise), so cn[a, v] = CN(a, v) / deg(v). float32, like the reference."""
    columns, deg = cand_cols
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)  # nan_to_num of 1/0
    common = (adj[sources] @ columns).toarray()
    return (common * inv[None, :]).astype(np.float32)


def ra_rows(adj: sp.csr_matrix, sources: np.ndarray,
            cand_cols: tuple[sp.csr_matrix, np.ndarray]) -> np.ndarray:
    """Textbook resource allocation, sum over common neighbours w of 1 / deg(w) (variant)."""
    columns, _ = cand_cols
    deg = np.asarray(adj.sum(axis=0)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return (adj[sources] @ sp.diags(inv) @ columns).toarray().astype(np.float32)


def l3_rows(b: sp.csr_matrix, sources: np.ndarray, kept_side: str) -> np.ndarray:
    """Kovács et al. 2019, Eq. (1): p_XY = sum_{U,V} a_XU a_UV a_VY / sqrt(k_U k_V), on the
    bipartite graph `b` (S x D) of the target relation's training facts. `kept_side` "src":
    rows p[x, :] for kept sources x (candidates = all D nodes); "dst": rows p[:, y] for kept
    destinations y (candidates = all S nodes)."""
    k_s = np.asarray(b.sum(axis=1)).ravel()
    k_d = np.asarray(b.sum(axis=0)).ravel()
    is_s = sp.diags(np.divide(1.0, np.sqrt(k_s), out=np.zeros_like(k_s), where=k_s > 0))
    is_d = sp.diags(np.divide(1.0, np.sqrt(k_d), out=np.zeros_like(k_d), where=k_d > 0))
    bt = b.T.tocsr()
    if kept_side == "src":
        out = b[sources] @ is_d @ bt @ is_s @ b
    else:
        out = bt[sources] @ is_s @ b @ is_d @ bt
    return out.toarray().astype(np.float32)


def cosine_rows(x_kept: np.ndarray, x_cand: np.ndarray) -> np.ndarray:
    """HeaRT `calc_feat_sim` (`:82-91`): sklearn cosine similarity (zero vectors give 0)."""
    from sklearn.metrics.pairwise import cosine_similarity

    return cosine_similarity(x_kept, x_cand).astype(np.float32)


_PPR_KERNEL = None


def _ppr_kernel():
    """HeaRT's Andersen push (`calc_ppr.py:52-108`, after PyG's GDC), for one source, numba."""
    global _PPR_KERNEL
    if _PPR_KERNEL is None:
        import numba

        @numba.njit(cache=False)
        def push(inode, indptr, indices, out_degree, alpha, eps):
            alpha_eps = alpha * eps
            p = {inode: 0.0}
            r = {inode: alpha}
            q = [inode]
            while len(q) > 0:
                unode = q.pop()
                res = r[unode] if unode in r else 0.0
                if unode in p:
                    p[unode] += res
                else:
                    p[unode] = res
                r[unode] = 0.0
                for k in range(indptr[unode], indptr[unode + 1]):
                    vnode = indices[k]
                    _val = (1 - alpha) * res / out_degree[unode]
                    if vnode in r:
                        r[vnode] += _val
                    else:
                        r[vnode] = _val
                    res_vnode = r[vnode] if vnode in r else 0.0
                    if res_vnode >= alpha_eps * out_degree[vnode]:
                        if vnode not in q:
                            q.append(vnode)
            keys = np.empty(len(p), dtype=np.int64)
            vals = np.empty(len(p), dtype=np.float64)
            i = 0
            for key, val in p.items():
                keys[i] = key
                vals[i] = val
                i += 1
            return keys, vals

        _PPR_KERNEL = push
    return _PPR_KERNEL


def ppr_rows(adj: sp.csr_matrix, sources: np.ndarray, candidates: np.ndarray,
             alpha: float = PPR_ALPHA, eps: float = PPR_EPS) -> np.ndarray:
    """PPR from each source (HeaRT's row `ppr[source]`), read at the candidates. float32."""
    push = _ppr_kernel()
    indptr = adj.indptr.astype(np.int64)
    indices = adj.indices.astype(np.int64)
    out_degree = np.diff(indptr).astype(np.int64)
    position = np.full(adj.shape[0], -1, dtype=np.int64)
    position[candidates] = np.arange(candidates.size)
    out = np.zeros((sources.size, candidates.size), dtype=np.float32)
    for i, s in enumerate(sources):
        keys, vals = push(np.int64(s), indptr, indices, out_degree, alpha, eps)
        where = position[keys]
        hit = where >= 0
        out[i, where[hit]] = vals[hit]
    return out


# ----------------------------------------------------------------------- rank and select
def rank_score_matrix(row: np.ndarray) -> np.ndarray:
    """HeaRT `rank_score_matrix` (`:94-116`): positive scores ranked largest -> 1 (scipy
    `rankdata`, method 'min'), zeros -> max + 1, negatives (filtered) -> max + 2."""
    row = row.copy()
    if (row > 0).sum() > 0:
        ranks_row = rankdata(row[row > 0], method="min")
        ranks_row = ranks_row.max() - ranks_row + 1
        max_rank = ranks_row.max()
    else:
        ranks_row = []
        max_rank = 0
    row[row > 0] = ranks_row
    row[row == 0] = max_rank + 1
    row[row < 0] = max_rank + 2
    return row


def select_negatives(scores: list[np.ndarray], masked: np.ndarray, k: int,
                     rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    """HeaRT `rank_and_merge_node` (`:119-164`) with `agg = min`, over the candidates of one side;
    the random fill is drawn without replacement from zero-score candidates not already selected
    (C4). `scores` already hold -1 at masked candidates. Returns (candidate indices, number
    filled at random, size of the fill pool)."""
    zero = np.logical_and.reduce([s == 0 for s in scores])
    combined = np.min([rank_score_matrix(s) for s in scores], axis=0)
    max_greater_zero = int(combined.size - zero.sum() - masked.sum())
    if max_greater_zero >= k:
        return torch.topk(torch.from_numpy(-combined), k).indices.numpy(), 0, 0
    head = torch.topk(torch.from_numpy(-combined), max_greater_zero).indices.numpy()
    pool = np.setdiff1d(np.nonzero(zero)[0], head)
    need = k - max_greater_zero
    if pool.size < need:
        raise PoolTooSmall(f"need {need} random negatives, zero-score pool has {pool.size}")
    return np.concatenate([head, rng.choice(pool, size=need, replace=False)]), need, int(pool.size)


# ------------------------------------------------------------------------------- driver
@dataclass(slots=True)
class SideContext:
    kept_type: str
    cand_type: str
    kept_side: str  # "src" (corrupt the destination) or "dst" (corrupt the source)


def partners(data: TaskData) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Target-relation positives of every split, as (dst partners of each src, src partners of each
    dst); either orientation for a same-type target."""
    all_pos = np.concatenate([data.train_pos, data.valid_pos, data.test_pos])
    of_src: dict[int, set[int]] = {}
    of_dst: dict[int, set[int]] = {}
    for u, v in all_pos.tolist():
        of_src.setdefault(u, set()).add(v)
        of_dst.setdefault(v, set()).add(u)
        if data.same_type:
            of_src.setdefault(v, set()).add(u)
            of_dst.setdefault(u, set()).add(v)
    return of_src, of_dst


def side_scores(data: TaskData, adj: sp.csr_matrix, b: sp.csr_matrix | None,
                kept_local: np.ndarray, ctx: SideContext, cn_metric: str,
                cache: dict) -> list[np.ndarray]:
    """Score matrices (kept x candidates), one per heuristic, in HeaRT's order (CN, PPR, feat)."""
    kept_global = kept_local + data.offsets[ctx.kept_type]
    cand_global = np.arange(data.type_sizes[ctx.cand_type]) + data.offsets[ctx.cand_type]
    if data.task.pair_heuristic == "cn":
        if ctx.cand_type not in cache:
            cache[ctx.cand_type] = candidate_columns(adj, cand_global)
        pair = (cn_heart_rows if cn_metric == "heart" else ra_rows)(
            adj, kept_global, cache[ctx.cand_type])
    else:
        pair = l3_rows(b, kept_local, ctx.kept_side)
    out = [pair, ppr_rows(adj, kept_global, cand_global)]
    if data.task.features:
        out.append(cosine_rows(data.features[ctx.kept_type][kept_local],
                               data.features[ctx.cand_type]))
    return out


def build_split(data: TaskData, positives: np.ndarray, seed: int, cn_metric: str,
                batch: int = 256, adj: sp.csr_matrix | None = None,
                num_samples: int = NUM_SAMPLES) -> tuple[np.ndarray, dict]:
    """(n x num_samples x 2 local-id negatives in (src, dst) order, statistics) for one split.
    `num_samples` is HeaRT's 500 for every real task; tests use fewer on small graphs."""
    src_t, _, dst_t = data.task.target
    adj = heuristic_graph(data) if adj is None else adj
    b = target_biadjacency(data) if data.task.pair_heuristic == "l3" else None
    of_src, of_dst = partners(data)
    k = num_samples // 2
    rng = np.random.default_rng(seed)
    out = np.zeros((len(positives), 2 * k, 2), dtype=np.int64)
    stats = {"positives": int(len(positives)), "lists_with_random_fill": 0,
             "random_fill_entries": 0, "min_fill_pool": None, "min_eligible_candidates": None}
    sides = (SideContext(src_t, dst_t, "src"), SideContext(dst_t, src_t, "dst"))
    cache: dict = {}
    for start in range(0, len(positives), batch):
        chunk = positives[start:start + batch]
        per_side = []
        for ctx, column in zip(sides, (0, 1)):
            kept = np.unique(chunk[:, column])
            row_of = {int(n): i for i, n in enumerate(kept)}
            per_side.append((row_of, side_scores(data, adj, b, kept, ctx, cn_metric, cache)))
        for j, (a, c) in enumerate(chunk.tolist()):
            for side, (ctx, (row_of, mats)) in enumerate(zip(sides, per_side)):
                kept, other = (a, c) if side == 0 else (c, a)
                n_cand = data.type_sizes[ctx.cand_type]
                masked = np.zeros(n_cand, dtype=bool)
                masked[list((of_src if side == 0 else of_dst).get(kept, set()))] = True
                masked[other] = True
                if data.same_type:
                    masked[kept] = True
                eligible = int(n_cand - masked.sum())
                if eligible < k:
                    raise PoolTooSmall(f"positive {(a, c)}: {eligible} eligible candidates")
                if stats["min_eligible_candidates"] is None or \
                        eligible < stats["min_eligible_candidates"]:
                    stats["min_eligible_candidates"] = eligible
                scores = []
                for mat in mats:
                    row = mat[row_of[kept]].copy()
                    row[masked] = -1
                    scores.append(row)
                chosen, filled, pool = select_negatives(scores, masked, k, rng)
                if filled:
                    stats["lists_with_random_fill"] += 1
                    stats["random_fill_entries"] += filled
                    if stats["min_fill_pool"] is None or pool < stats["min_fill_pool"]:
                        stats["min_fill_pool"] = pool
                rows = out[start + j, side * k:(side + 1) * k]
                if side == 0:
                    rows[:, 0], rows[:, 1] = a, chosen
                else:
                    rows[:, 0], rows[:, 1] = chosen, c
    return out, stats


def to_raw(data: TaskData, local: np.ndarray) -> np.ndarray:
    src_t, _, dst_t = data.task.target
    raw = np.empty_like(local)
    raw[..., 0] = data.local_to_raw[src_t][local[..., 0]]
    raw[..., 1] = data.local_to_raw[dst_t][local[..., 1]]
    return raw


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_task(data_root: Path, name: str, seed: int = SEED, cn_metric: str = "heart",
               out_subdir: str = "heart", num_samples: int = NUM_SAMPLES) -> dict:
    task = HEART_TASKS[name]
    t0 = time.time()
    data = load_task(data_root, task)
    adj = heuristic_graph(data)
    out_dir = data.folder / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"task": name, "target": list(task.target), "seed": seed,
            "num_samples": num_samples, "per_side": num_samples // 2,
            "pair_heuristic": ("HeaRT calc_CN_metric as coded: CN(a,v)/deg(v)"
                               if task.pair_heuristic == "cn" and cn_metric == "heart" else
                               "textbook RA (variant)" if task.pair_heuristic == "cn" else
                               "L3 (Kovacs et al. 2019 Eq. 1) on target training facts"),
            "ppr": {"alpha": PPR_ALPHA, "eps": PPR_EPS}, "feature_cosine": task.features,
            "heuristic_graph": {"nodes": int(adj.shape[0]), "undirected_edges": int(adj.nnz // 2)},
            "target_training_facts": int(len(data.train_pos)),
            "random_fill": "numpy default_rng(seed), without replacement (C4)",
            "layout": "samples[i, :250] = (src_i, v), samples[i, 250:] = (u, dst_i); raw HGB ids"}
    for split, positives in (("valid", data.valid_pos), ("test", data.test_pos)):
        samples, stats = build_split(data, positives, seed, cn_metric, adj=adj,
                                     num_samples=num_samples)
        np.save(out_dir / f"heart_{split}_pos.npy", to_raw(data, positives))
        np.save(out_dir / f"heart_{split}_samples.npy", to_raw(data, samples))
        meta[split] = stats
    meta["md5"] = {p.name: _md5(p) for p in sorted(out_dir.glob("heart_*.npy"))}
    meta["seconds"] = round(time.time() - t0, 1)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=REPO_DATA_ROOT)
    parser.add_argument("--tasks", nargs="+", default=list(HEART_TASKS), choices=list(HEART_TASKS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cn-metric", choices=("heart", "ra"), default="heart",
                        help="Same-type pair heuristic: HeaRT's code (primary) or textbook RA.")
    parser.add_argument("--out-subdir", default="heart")
    args = parser.parse_args()
    for name in args.tasks:
        meta = build_task(args.data_root, name, args.seed, args.cn_metric, args.out_subdir)
        print(json.dumps({k: meta[k] for k in ("task", "pair_heuristic", "valid", "test",
                                                "seconds")}), flush=True)


if __name__ == "__main__":
    main()
