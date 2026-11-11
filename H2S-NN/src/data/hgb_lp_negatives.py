"""HGB link-prediction negatives (spec v1.15 §6, J5; card Part M, M7).

Port of HGB's ``LP/benchmark/scripts/data_loader.py`` (THUDM/HGB, master; read 2026-11-12):

* ``get_test_neigh``: ``random.seed(1)``; the adjacency ``A`` holds every training fact, every
  test fact and every validation fact; ``A2 = A @ A`` is binarised, and a pair ``(h, t)`` is a
  candidate iff ``A2[h, t] > 0``, ``A[h, t] == 0`` and ``h != t`` (HGB subtracts ``A`` and the
  identity and keeps entries > 0). Candidates are filtered to the relation's head and tail types.
  Heads are visited in ascending id order; each head draws ``random.choices(candidates, k=n_h)``
  with ``n_h`` = its number of test positives; a head without candidates gets **no** negatives
  (no random fallback). Candidates of a head are in ascending id order, as the canonical CSR
  ``nonzero()`` HGB reads them from.
* ``get_valid_neg``: one uniformly random tail of the tail type per validation positive
  (``random.randrange`` over the tail type's id range), in the order of the positives; known
  positives are not excluded (HGB does not exclude them).

One deliberate difference, stated by the card and the spec ("symmetric adjacency of train +
validation + test"): HGB adds validation facts in their stored orientation only; here every
fact enters ``A`` in both orientations (decision recorded for the author in the Part M report).

Only the rows of the test heads and the columns of the tail type of ``A2`` are formed
(``A[heads] @ A[:, tails]``), which gives exactly the entries HGB reads without the full
``N x N`` product.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

HGB_TEST_NEGATIVE_SEED = 1


@dataclass(frozen=True, slots=True)
class HGBTestNegatives:
    """Test negatives of one target relation, and the heads that received none."""

    edge_index: torch.Tensor          # [2, M] global ids (head, tail)
    heads_without_candidates: torch.Tensor  # [H0] global ids, ascending
    positives_of_heads_without: int   # test positives whose head has no candidate


def symmetric_adjacency(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    """Binary ``A`` with every pair in both orientations (canonical CSR)."""
    rows = edge_index[0].cpu().numpy().astype(np.int64)
    cols = edge_index[1].cpu().numpy().astype(np.int64)
    data = np.ones(2 * rows.size, dtype=np.int64)
    adjacency = sp.coo_matrix((data, (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
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


def hgb_test_negatives(test_pos_edge_index: torch.Tensor, adjacency_edge_index: torch.Tensor,
                       num_nodes: int, dst_range: tuple[int, int],
                       seed: int = HGB_TEST_NEGATIVE_SEED) -> HGBTestNegatives:
    """HGB ``get_test_neigh`` for one target relation (see the module docstring).

    ``adjacency_edge_index``: every train, validation and test fact (any orientation; it is
    symmetrised). ``test_pos_edge_index``: the relation's test positives (global ids)."""
    heads_all = test_pos_edge_index[0].cpu().numpy().astype(np.int64)
    heads, counts = np.unique(heads_all, return_counts=True)        # ascending, as HGB's sorted()
    adjacency = symmetric_adjacency(adjacency_edge_index, num_nodes)
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
    device = test_pos_edge_index.device
    edge_index = torch.tensor([sampled_heads, sampled_tails], dtype=torch.long,
                              device=device).reshape(2, -1)
    return HGBTestNegatives(edge_index=edge_index,
                            heads_without_candidates=torch.tensor(without, dtype=torch.long),
                            positives_of_heads_without=int(without_positives))


def hgb_valid_negatives(valid_pos_edge_index: torch.Tensor, dst_range: tuple[int, int],
                        seed: int) -> torch.Tensor:
    """HGB ``get_valid_neg``: one uniform tail of the tail type per validation positive, in the
    positives' order; known positives are not excluded."""
    rng = random.Random(seed)
    start, end = dst_range
    heads = valid_pos_edge_index[0].tolist()
    tails = [rng.randrange(start, end) for _ in heads]
    return torch.tensor([heads, tails], dtype=torch.long,
                        device=valid_pos_edge_index.device).reshape(2, -1)


__all__ = ["HGBTestNegatives", "HGB_TEST_NEGATIVE_SEED", "hgb_test_negatives",
           "hgb_valid_negatives", "symmetric_adjacency", "two_hop_candidates"]
