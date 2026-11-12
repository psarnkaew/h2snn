"""HGB link-prediction negatives (spec v1.16 §6, K1-K3; cards Part M M7, Part N N1-N2).

Port of HGB's ``LP/benchmark/scripts/data_loader.py`` and the way ``LP/benchmark/methods/
baseline/run_new.py`` uses it (THUDM/HGB master, last commit of both paths ``2ccb1d8``,
2023-03-18; read 2026-11-12 through a fetch agent: in Part M as code text, in Part N as a
structural summary because the repository declares no licence).

HGB's adjacency ``pos_links`` = every training fact plus its transpose, every test fact plus its
transpose, and every validation positive **in its stored orientation only** (``get_test_neigh`` /
``get_test_neigh_w_random`` add ``valid_pos`` as one ``coo_matrix`` without ``.T``; spec v1.16
K2). Here: ``both_ways`` = training and test facts, ``one_way`` = validation positives as stored.

* ``get_test_neigh`` (2-hop set): ``random.seed(1)``; a pair ``(h, t)`` is a candidate iff
  ``(A·A)[h, t] > 0``, ``A[h, t] == 0`` and ``h != t``, ``t`` of the tail type; heads ascending;
  ``random.choices(candidates, k = #test positives of h)``, candidates ascending (canonical CSR
  ``nonzero()`` order); a head without candidates gets **no** negative.
* ``get_test_neigh_w_random`` (random set): ``random.seed(1)``; the known neighbours of ``h`` are
  the nonzero columns of row ``h`` of ``A``; for each test positive in ``nonzero()`` order (head,
  then tail ascending) one tail ``randrange`` of the tail type, redrawn while it is a known
  neighbour of ``h``. One negative per test positive.
* ``get_valid_neg`` / ``get_train_neg``: one uniform tail of the tail type per positive, in the
  positives' order, known positives not excluded. ``run_new.py`` redraws ``get_train_neg`` every
  epoch (and ``get_valid_neg`` at every batch step; here validation negatives are drawn once per
  run, listed in the Part N report).

Only the rows of the test heads and the columns of the tail type of ``A·A`` are formed
(``A[heads] @ A[:, tails]``): the entries HGB reads, without the full ``N x N`` product.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

HGB_TEST_NEGATIVE_SEED = 1
NO_TWO_HOP_PATH_REASON = "no length-2 head→tail type path"


@dataclass(frozen=True, slots=True)
class HGBTestNegatives:
    """Test negatives of one target relation, and the heads that received none."""

    edge_index: torch.Tensor          # [2, M] global ids (head, tail)
    heads_without_candidates: torch.Tensor  # [H0] global ids, ascending
    positives_of_heads_without: int   # test positives whose head has no candidate


def _rows_cols(edge_index: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    edge_index = edge_index.reshape(2, -1)
    return (edge_index[0].cpu().numpy().astype(np.int64),
            edge_index[1].cpu().numpy().astype(np.int64))


def hgb_adjacency(both_ways: torch.Tensor, one_way: torch.Tensor,
                  num_nodes: int) -> sp.csr_matrix:
    """Binary ``pos_links``: ``both_ways`` facts in both orientations, ``one_way`` facts as
    stored (canonical CSR)."""
    r1, c1 = _rows_cols(both_ways)
    r2, c2 = _rows_cols(one_way)
    rows = np.concatenate([r1, c1, r2])
    cols = np.concatenate([c1, r1, c2])
    adjacency = sp.coo_matrix((np.ones(rows.size, dtype=np.int64), (rows, cols)),
                              shape=(num_nodes, num_nodes)).tocsr()
    adjacency.sum_duplicates()
    adjacency.data[:] = 1
    adjacency.sort_indices()
    return adjacency


def two_hop_candidates(adjacency: sp.csr_matrix, heads: np.ndarray,
                       dst_range: tuple[int, int]) -> list[np.ndarray]:
    """Per head (in the order given): ascending global ids ``t`` in ``dst_range`` with
    ``(A @ A)[h, t] > 0``, ``A[h, t] == 0`` and ``t != h``."""
    start, end = dst_range
    rows = adjacency[heads]                                  # [H, N]
    reach = (rows @ adjacency[:, start:end]).tocsr()          # [H, |T|]
    reach.data[:] = 1
    direct = rows[:, start:end].tocsr()
    candidates = (reach - direct).tocsr()
    candidates.data[candidates.data < 0] = 0
    candidates.eliminate_zeros()
    candidates.sort_indices()
    out: list[np.ndarray] = []
    for i, head in enumerate(heads.tolist()):
        cols = candidates.indices[candidates.indptr[i]:candidates.indptr[i + 1]] + start
        out.append(cols[cols != head])
    return out


def hgb_test_negatives(test_pos_edge_index: torch.Tensor, both_ways: torch.Tensor,
                       one_way: torch.Tensor, num_nodes: int, dst_range: tuple[int, int],
                       seed: int = HGB_TEST_NEGATIVE_SEED) -> HGBTestNegatives:
    """HGB ``get_test_neigh`` (the 2-hop set) for one target relation."""
    heads_all = test_pos_edge_index[0].cpu().numpy().astype(np.int64)
    heads, counts = np.unique(heads_all, return_counts=True)        # ascending, HGB's sorted()
    adjacency = hgb_adjacency(both_ways, one_way, num_nodes)
    candidates = two_hop_candidates(adjacency, heads, dst_range)
    rng = random.Random(seed)
    sampled_heads: list[int] = []
    sampled_tails: list[int] = []
    without: list[int] = []
    without_positives = 0
    for head, count, cands in zip(heads.tolist(), counts.tolist(), candidates):
        if cands.size == 0:
            without.append(head)
            without_positives += count
            continue
        tails = rng.choices(cands.tolist(), k=count)
        sampled_heads.extend([head] * count)
        sampled_tails.extend(tails)
    edge_index = torch.tensor([sampled_heads, sampled_tails], dtype=torch.long,
                              device=test_pos_edge_index.device).reshape(2, -1)
    return HGBTestNegatives(edge_index=edge_index,
                            heads_without_candidates=torch.tensor(without, dtype=torch.long),
                            positives_of_heads_without=int(without_positives))


def hgb_random_test_negatives(test_pos_edge_index: torch.Tensor, both_ways: torch.Tensor,
                              one_way: torch.Tensor, num_nodes: int,
                              dst_range: tuple[int, int],
                              seed: int = HGB_TEST_NEGATIVE_SEED) -> torch.Tensor:
    """HGB ``get_test_neigh_w_random`` (the random set): one negative per test positive, in the
    positives' ``nonzero()`` order (head, then tail ascending), redrawn while it is a known
    neighbour of the head."""
    start, end = dst_range
    adjacency = hgb_adjacency(both_ways, one_way, num_nodes)
    pairs = np.unique(test_pos_edge_index.cpu().numpy().astype(np.int64).T.reshape(-1, 2),
                      axis=0)                                       # row-major sorted, unique
    rng = random.Random(seed)
    heads, tails = [], []
    for head, _ in pairs.tolist():
        known = set(adjacency.indices[adjacency.indptr[head]:adjacency.indptr[head + 1]].tolist())
        if all(t in known for t in range(start, end)):
            raise ValueError(f"head {head} is a known neighbour of every tail: HGB's redraw loop "
                             "would not terminate.")
        tail = rng.randrange(start, end)
        while tail in known:
            tail = rng.randrange(start, end)
        heads.append(head)
        tails.append(tail)
    return torch.tensor([heads, tails], dtype=torch.long,
                        device=test_pos_edge_index.device).reshape(2, -1)


def hgb_uniform_tail_negatives(pos_edge_index: torch.Tensor, dst_range: tuple[int, int],
                               seed: int) -> torch.Tensor:
    """HGB ``get_valid_neg`` / ``get_train_neg``: one uniform tail of the tail type per
    positive, in the positives' order; known positives are not excluded."""
    rng = random.Random(seed)
    start, end = dst_range
    heads = pos_edge_index[0].tolist()
    tails = [rng.randrange(start, end) for _ in heads]
    return torch.tensor([heads, tails], dtype=torch.long,
                        device=pos_edge_index.device).reshape(2, -1)


def hgb_valid_negatives(valid_pos_edge_index: torch.Tensor, dst_range: tuple[int, int],
                        seed: int) -> torch.Tensor:
    """HGB ``get_valid_neg``."""
    return hgb_uniform_tail_negatives(valid_pos_edge_index, dst_range, seed)


def hgb_train_negatives(train_pos_edge_index: torch.Tensor, dst_range: tuple[int, int],
                        seed: int) -> torch.Tensor:
    """HGB ``get_train_neg`` (spec v1.16 K3): drawn every epoch by HGB's runner; seed = run seed
    + epoch here."""
    return hgb_uniform_tail_negatives(train_pos_edge_index, dst_range, seed)


__all__ = ["HGBTestNegatives", "HGB_TEST_NEGATIVE_SEED", "NO_TWO_HOP_PATH_REASON",
           "hgb_adjacency", "hgb_random_test_negatives", "hgb_test_negatives",
           "hgb_train_negatives", "hgb_uniform_tail_negatives", "hgb_valid_negatives",
           "two_hop_candidates"]
