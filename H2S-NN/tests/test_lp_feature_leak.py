"""J5.2 (task card Part J): the features-only scorer's rank, metrics and containment on known
answers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "lp"))

import lp_feature_leak as fl  # noqa: E402


class LeakMeasureTests(unittest.TestCase):
    def test_all_ties_rank_251_of_500(self) -> None:
        rank = fl.heart_rank(np.zeros(3), np.zeros((3, 500)))
        self.assertTrue(np.all(rank == 251.0))
        self.assertAlmostEqual(fl.metrics(rank)["mrr"], 1 / 251)
        self.assertEqual(fl.metrics(rank)["hits@10"], 0.0)

    def test_strictly_first_is_rank_1_and_one_tie_is_1_5(self) -> None:
        neg = np.zeros((2, 500))
        neg[1, 7] = 1.0
        rank = fl.heart_rank(np.array([1.0, 1.0]), neg)
        self.assertEqual(rank.tolist(), [1.0, 1.5])  # HeaRT: mean of optimistic and pessimistic

    def test_containment_is_the_share_of_the_contribution_inside_the_kept_feature(self) -> None:
        x = sp.csr_matrix(np.array([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=float))
        c = sp.csr_matrix(np.array([[1, 0, 1, 0], [0, 0, 0, 0]], dtype=float))
        s = fl.containment(x, c, np.array([0, 1, 0]), np.array([0, 0, 1]))
        self.assertEqual(s.tolist(), [0.5, 0.0, 0.0])  # empty contribution scores 0

    def test_contribution_follows_the_column_map(self) -> None:
        g = sp.csr_matrix(np.array([[0, 2, 0, 5]], dtype=float))
        mapping = np.array([3, -1, 1, -2])  # x col 0 <- vocab 3, col 2 <- vocab 1
        self.assertEqual(fl.contribution(g, mapping).toarray().tolist(), [[1, 0, 1, 0]])


if __name__ == "__main__":
    unittest.main()
