"""J6.2 (spec v1.10 F18): the fingerprint probe detects a column that is an attribute of the
other endpoint, passes an own random column, and the typed-path control counts paths exactly
(synthetic power check of `docs/lp_evidence/j6_preregistration.md`)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "lp"))

import lp_leak_audit as la  # noqa: E402

N_Y, PER_Y, N_NEG = 20, 4, 15  # each y has 4 x partners; one held out per y


def toy():
    rng = np.random.default_rng(0)
    n_x = N_Y * PER_Y
    partner = np.repeat(np.arange(N_Y), PER_Y)            # x -> its y
    held = np.arange(0, n_x, PER_Y)                        # first x of every y is held out
    train = [(y, x) for x, y in enumerate(partner) if x not in set(held)]
    t = sp.csr_matrix((np.ones(len(train)), ([p[0] for p in train], [p[1] for p in train])),
                      shape=(N_Y, n_x))                    # (Y x X)
    xi, yi = [], []
    for x in held:
        y = partner[x]
        negs = rng.choice([v for v in range(N_Y) if v != y], N_NEG, replace=False)
        xi += [x] * (N_NEG + 1)
        yi += [y] + list(negs)
    leak = np.array([[f"{partner[x] * 1000 + 17}"] for x in range(n_x)])  # the partner's attribute
    own = np.array([[f"{v}"] for v in rng.integers(1, 10**6, n_x)])     # the node's own value
    return t, np.array(xi), np.array(yi), leak, own, (len(held), N_NEG + 1)


class LeakAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t, self.xi, self.yi, self.leak, self.own, self.shape = toy()
        tie = la.heart_rank(np.zeros(self.shape[0]), np.zeros((self.shape[0], self.shape[1] - 1)))
        self.ctrl = la.metrics(tie)  # all-tie control: the toy graph carries no signal

    def run_family(self, values, cap):
        fp = la.fingerprint_matrix(values, cap)
        return la.rank_metrics(la.probe_scores(fp, self.t, self.xi, self.yi), self.shape)

    def test_partner_attribute_fails_the_criterion(self) -> None:
        for cap in la.CAPS:
            m = self.run_family(self.leak, cap)
            self.assertEqual(m["mrr"], 1.0)
            self.assertFalse(la.verdict(m, self.ctrl)["pass"], cap)

    def test_own_random_column_passes(self) -> None:
        for cap in la.CAPS:
            self.assertTrue(la.verdict(self.run_family(self.own, cap), self.ctrl)["pass"], cap)

    def test_cap_and_zeros_are_excluded_from_the_fingerprint(self) -> None:
        values = np.array([["0"], ["0.0"], ["7"], ["7"], ["7"], ["9"]])
        self.assertEqual(la.fingerprint_matrix(values, None).nnz, 4)   # zeros dropped
        self.assertEqual(la.fingerprint_matrix(values, 2).nnz, 1)      # '7' held by 3 > 2

    def test_typed_path_count_by_hand(self) -> None:
        # X = {0,1,2}, W = {0,1}; x0-w0, x1-w0, x1-w1, x2-w1. y0's training partners: x1, x2.
        a = sp.csr_matrix(np.array([[1, 0], [1, 1], [0, 1]], dtype=float))
        t = sp.csr_matrix(np.array([[0, 1, 1]], dtype=float))
        # x0 with y0: |N(x0)&N(x1)| + |N(x0)&N(x2)| = 1 + 0 = 1
        self.assertEqual(la.path_scores(a, t, np.array([0]), np.array([0])).tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()
