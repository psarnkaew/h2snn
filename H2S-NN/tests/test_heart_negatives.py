"""J4.4 (task card Part J, spec v1.8 F8): HeaRT negatives.

Hand values for each piece (HeaRT's CN-as-coded, textbook RA, L3, rank/merge, PPR push), then the
card's properties on synthetic releases of each kind of task (cross-type with features, same-type
with features and a self-inverse target, same-type split by unordered pair with self-loops):
none is a positive of any split in either orientation, every corrupted endpoint has the required
type, no duplicate within a list, same seed -> identical files, heuristics read training facts only.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import scipy.sparse as sp

from scripts.lp import build_derived_lp as bd
from scripts.lp import build_heart_negatives as hn
from tests.test_lp_derived import write_freebase_like, write_release
from tests.test_lp_valid_files import write_pubmed_like


def _sym(edges: list[tuple[int, int]], n: int) -> sp.csr_matrix:
    row = [u for u, v in edges] + [v for u, v in edges]
    col = [v for u, v in edges] + [u for u, v in edges]
    return sp.csr_matrix((np.ones(len(row), dtype=np.float32), (row, col)), shape=(n, n))


class PieceTests(unittest.TestCase):
    # Path 0-1-2-3 plus 1-3: degrees 1, 3, 2, 2.
    ADJ = _sym([(0, 1), (1, 2), (2, 3), (1, 3)], 4)

    def test_heart_cn_is_common_neighbours_over_candidate_degree(self) -> None:
        cols = hn.candidate_columns(self.ADJ, np.arange(4))
        got = hn.cn_heart_rows(self.ADJ, np.array([0, 2]), cols)
        # Rows of the verbatim HeaRT lines run on this graph (j44_heart_ra_semantics.txt).
        np.testing.assert_allclose(got, [[1.0, 0.0, 0.5, 0.5], [1.0, 1 / 3, 1.0, 0.5]], rtol=1e-6)

    def test_textbook_ra_differs(self) -> None:
        cols = hn.candidate_columns(self.ADJ, np.arange(4))
        got = hn.ra_rows(self.ADJ, np.array([0]), cols)
        np.testing.assert_allclose(got, [[1 / 3, 0.0, 1 / 3, 1 / 3]], rtol=1e-6)

    def test_l3_equals_kovacs_eq1_by_hand(self) -> None:
        # S = {x0, x1}, D = {y0, y1, y2}; facts x0-y0, x0-y1, x1-y1, x1-y2.
        b = sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64))
        r2, h = 1 / np.sqrt(2), 0.5
        np.testing.assert_allclose(hn.l3_rows(b, np.array([0]), "src"),
                                   [[r2 + h, r2 + 2 * h, h]], rtol=1e-6)
        np.testing.assert_allclose(hn.l3_rows(b, np.array([2]), "dst"), [[h, h + r2]], rtol=1e-6)

    def test_rank_score_matrix_matches_heart(self) -> None:
        got = hn.rank_score_matrix(np.array([0.5, 0, -1, 0.5, 0.2], dtype=np.float32))
        np.testing.assert_array_equal(got, [1, 3, 4, 1, 2])

    def test_select_top_k_and_fill_without_replacement(self) -> None:
        cn = np.array([0, 0.5, 0, 0, -1, 0.1], dtype=np.float32)
        ppr = np.array([0, 0, 0.3, 0, -1, 0], dtype=np.float32)
        masked = np.array([0, 0, 0, 0, 1, 0], dtype=bool)
        rng = np.random.default_rng(0)
        chosen, filled, _ = hn.select_negatives([cn, ppr], masked, 2, rng)
        self.assertEqual(set(chosen.tolist()), {1, 2})  # combined ranks [2, 1, 1, 2, 3, 2]
        self.assertEqual(filled, 0)
        # k = 4 > 3 candidates above zero: HeaRT takes the top 3 of the combined ranks (1, 2 and
        # one of the rank-2 ties 0, 3, 5) and fills from the zero-score candidates (0, 3) not
        # already taken, without replacement (C4).
        chosen, filled, _ = hn.select_negatives([cn.copy(), ppr.copy()], masked, 4, rng)
        self.assertEqual(len(set(chosen.tolist())), 4)  # no duplicate (C4)
        self.assertNotIn(4, chosen.tolist())
        self.assertTrue({1, 2} <= set(chosen.tolist()))
        self.assertEqual(filled, 1)
        with self.assertRaises(hn.PoolTooSmall):  # need 3, at most 2 zero-score candidates left
            hn.select_negatives([cn.copy(), ppr.copy()], masked, 6, rng)

    def test_fill_draws_without_replacement(self) -> None:
        """C4: 20 zero-score candidates, 20 needed: the fill must be a permutation (a draw with
        replacement repeats a node with probability 1 - 20!/20^20 > 0.99999998)."""
        zeros = [np.zeros(20, dtype=np.float32), np.zeros(20, dtype=np.float32)]
        chosen, filled, pool = hn.select_negatives(zeros, np.zeros(20, dtype=bool), 20,
                                                   np.random.default_rng(0))
        self.assertEqual(sorted(chosen.tolist()), list(range(20)))
        self.assertEqual((filled, pool), (20, 20))

    def test_ppr_push_approximates_exact_ppr(self) -> None:
        a = self.ADJ.toarray().astype(np.float64)
        p = a / a.sum(1, keepdims=True)
        alpha = hn.PPR_ALPHA
        exact = alpha * np.linalg.inv(np.eye(4) - (1 - alpha) * p)  # row s = PPR from s
        tight = hn.ppr_rows(self.ADJ, np.array([0, 2]), np.arange(4), eps=1e-9)
        np.testing.assert_allclose(tight, exact[[0, 2]], atol=1e-6)
        default = hn.ppr_rows(self.ADJ, np.array([0, 2]), np.arange(4))
        self.assertLess(float(np.abs(default - exact[[0, 2]]).max()), 1e-3)


class _SyntheticTaskMixin:
    """Generates negatives for one synthetic task and checks the J1/J4.4 properties."""

    name: str
    k_total = 10
    root: Path

    def _data(self):
        return hn.load_task(self.root, hn.HEART_TASKS[self.name])

    def _negatives(self, data, split: str, seed: int = 7):
        positives = data.valid_pos if split == "valid" else data.test_pos
        return hn.build_split(data, positives, seed, "heart", batch=16,
                              num_samples=self.k_total)[0]

    def test_no_negative_is_a_positive_of_any_split_either_orientation(self) -> None:
        data = self._data()
        positives = {tuple(p) for p in np.concatenate(
            [data.train_pos, data.valid_pos, data.test_pos]).tolist()}
        for split in ("valid", "test"):
            for u, v in self._negatives(data, split).reshape(-1, 2).tolist():
                self.assertNotIn((u, v), positives)
                if data.same_type:
                    self.assertNotIn((v, u), positives)
                    self.assertNotEqual(u, v)

    def test_no_duplicate_within_a_list_and_halves_keep_one_endpoint(self) -> None:
        data = self._data()
        positives = data.test_pos
        samples = self._negatives(data, "test")
        half = self.k_total // 2
        for i, (a, b) in enumerate(positives.tolist()):
            pairs = [tuple(p) for p in samples[i].tolist()]
            self.assertEqual(len(set(pairs)), len(pairs))
            self.assertTrue(all(p[0] == a for p in pairs[:half]))
            self.assertTrue(all(p[1] == b for p in pairs[half:]))

    def test_written_files_have_the_required_types_and_are_reproducible(self) -> None:
        task = hn.HEART_TASKS[self.name]
        meta = hn.build_task(self.root, self.name, seed=7, out_subdir="heart_small",
                             num_samples=self.k_total)
        folder = self._data().folder
        types = {int(line.split(b"\t")[0]): int(line.split(b"\t")[2])
                 for line in (folder / "node.dat").read_bytes().splitlines()}
        names = hn._LOCAL_HGB_LINK_DATASETS[task.dataset]["node_type_names"]
        type_id = {v: k for k, v in names.items()}
        for split in ("valid", "test"):
            samples = np.load(folder / "heart_small" / f"heart_{split}_samples.npy")
            self.assertTrue(all(types[u] == type_id[task.target[0]] for u in samples[..., 0].ravel()))
            self.assertTrue(all(types[v] == type_id[task.target[2]] for v in samples[..., 1].ravel()))
        again = hn.build_task(self.root, self.name, seed=7, out_subdir="heart_small_again",
                              num_samples=self.k_total)
        self.assertEqual(meta["md5"], again["md5"])

    def test_heuristics_read_training_facts_only(self) -> None:
        data = self._data()
        adj = hn.heuristic_graph(data).tocoo()
        edges = set(zip(adj.row.tolist(), adj.col.tolist()))
        src, _, dst = data.task.target
        for held in (data.valid_pos, data.test_pos):
            for u, v in held.tolist():
                gu, gv = u + data.offsets[src], v + data.offsets[dst]
                self.assertNotIn((gu, gv), edges)
                self.assertNotIn((gv, gu), edges)
        train = {tuple(p) for p in data.train_pos.tolist()}
        held = {tuple(p) for p in np.concatenate([data.valid_pos, data.test_pos]).tolist()}
        if data.task.pair_heuristic == "l3":
            b = hn.target_biadjacency(data).tocoo()
            self.assertEqual(set(zip(b.row.tolist(), b.col.tolist())), train)
        self.assertFalse(train & held)


class CrossTypeWithFeaturesTests(_SyntheticTaskMixin, unittest.TestCase):
    """acm_pa-like: paper -> author (L3 + PPR + cosine; two-dimensional shared features)."""

    name = "acm_pa"

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_release(self.root / "release")
        bd.build_task(self.root / "release", self.root / "lp_derived" / "acm_pa", 2, seed=42,
                      val_frac=0.09, test_frac=0.10)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_mask_and_scores_use_the_kept_endpoint(self) -> None:
        data = self._data()
        of_src, of_dst = hn.partners(data)
        a, b = data.test_pos[0].tolist()
        self.assertIn(b, of_src[a])
        self.assertIn(a, of_dst[b])


class SameTypeWithFeaturesTests(_SyntheticTaskMixin, unittest.TestCase):
    """pubmed-like: disease-disease, self-inverse (HeaRT CN + PPR + cosine)."""

    name = "pubmed"

    def setUp(self) -> None:
        from scripts.lp.write_valid_files import carve_rows

        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_pubmed_like(self.root / "PubMed")
        content, _ = carve_rows(self.root, "pubmed")
        (self.root / "PubMed" / "link.dat.valid").write_bytes(content)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class SameTypeByPairTests(_SyntheticTaskMixin, unittest.TestCase):
    """freebase_bb-like: book-book by unordered pair, self-loops in link.dat (HeaRT CN + PPR)."""

    name = "freebase_bb"

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_freebase_like(self.root / "release")
        bd.build_task_by_pair(self.root / "release", self.root / "lp_derived" / "freebase_bb", 0,
                              seed=42, val_frac=0.09, test_frac=0.10)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_self_loops_are_never_negatives(self) -> None:
        data = self._data()
        for split in ("valid", "test"):
            samples = self._negatives(data, split).reshape(-1, 2)
            self.assertFalse(bool((samples[:, 0] == samples[:, 1]).any()))


class PoolTooSmallTests(unittest.TestCase):
    def test_too_few_candidates_stop_the_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_release(root / "release")
            bd.build_task(root / "release", root / "lp_derived" / "acm_pa", 2, seed=42,
                          val_frac=0.09, test_frac=0.10)
            data = hn.load_task(root, hn.HEART_TASKS["acm_pa"])
            with self.assertRaises(hn.PoolTooSmall):  # 40 authors cannot give 250 per side
                hn.build_split(data, data.test_pos, 7, "heart")


if __name__ == "__main__":
    unittest.main()
