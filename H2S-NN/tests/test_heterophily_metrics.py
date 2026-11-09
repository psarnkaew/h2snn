"""J0.3 (task card Part J): MLH and the H^2 index against hand values and the reference code.

The reference functions (H2GB `calcHomophily`, Hetero^2Net `homophily.py`) are loaded from their
clones under `vendor/`; the tests are skipped only if a clone is absent.
"""

from __future__ import annotations

import unittest

import torch

from src.baselines.runtime.vendor_paths import VENDOR_ROOTS

_CLONES_PRESENT = all((VENDOR_ROOTS[name]).is_dir() for name in ("h2gb", "hetero2net")) and \
    (VENDOR_ROOTS["h2gb"] / "H2GB" / "calcHomophily.py").is_file() and \
    (VENDOR_ROOTS["hetero2net"] / "hetero2net" / "homophily.py").is_file()

PA = ("paper", "to", "author")
AP = ("author", "to", "paper")
CITE = ("paper", "cite", "paper")
PAP = (PA, AP)
PCP = (CITE, CITE)


def _toy():
    """Two node types, two meta-paths (P-A-P and P-cite-P-cite-P), labels y = [0, 0, 1, 0].

    Hand computation (pairs (u, v) of G_P, coalesced, self-loops kept, as calcHomophily builds it):
    - P-A-P, writes p0-a0, p1-a0, p1-a1, p2-a1, p3-a1: G_P = {00, 01, 10, 11, 12, 13, 21, 22, 23,
      31, 32, 33}, 12 pairs, 8 with equal labels -> h_edge = 2/3. In-degrees 2, 4, 3, 3; class
      shares (9/12, 3/12), sum of squares 5/8 -> h_adj = (2/3 - 5/8) / (3/8) = 1/9. Per-node
      in-agreement 1, 3/4, 1/3, 2/3 -> h_node = 11/16. Without the class-imbalance correction
      (uniform shares 1/2) h_adj would be 1/3.
    - P-cite-P-cite-P, cite 0<->2, 1<->3, 2<->3: G_P = {00, 03, 11, 12, 21, 22, 30, 33}, 8 pairs,
      6 equal -> h_edge = 3/4. In-degrees 2, 2, 2, 2; shares (6/8, 2/8), 5/8 -> h_adj = 1/3.
      Per-node 1, 1/2, 1/2, 1 -> h_node = 3/4.
    Aggregates: MLH (H2GB set, mean, heterophily) = 1 - 17/24 = 7/24; max = 1/3; H^2 =
    1 - (1/9 + 1/3) / 2 = 7/9; MLH (Hetero^2Net set = {P-A-P}, homophily) = 2/3.
    """
    from torch_geometric.data import HeteroData

    data = HeteroData()
    data["paper"].y = torch.tensor([0, 0, 1, 0])
    data["paper"].num_nodes = 4
    data["author"].num_nodes = 2
    writes = torch.tensor([[0, 1, 1, 2, 3], [0, 0, 1, 1, 1]])
    data[PA].edge_index = writes
    data[AP].edge_index = writes.flip(0)
    data[CITE].edge_index = torch.tensor([[0, 2, 1, 3, 2, 3], [2, 0, 3, 1, 3, 2]])
    return data


def _random_graph(seed: int, multilabel: bool = False, unlabeled: int = 0):
    from torch_geometric.data import HeteroData

    gen = torch.Generator().manual_seed(seed)

    def pairs(n_src: int, n_dst: int, count: int) -> torch.Tensor:
        edges = torch.stack([torch.randint(0, n_src, (count,), generator=gen),
                             torch.randint(0, n_dst, (count,), generator=gen)])
        return torch.unique(edges, dim=1)

    n_paper, n_author, n_term = 40, 25, 6
    data = HeteroData()
    if multilabel:
        y = (torch.rand(n_paper, 3, generator=gen) < 0.4).float()
    else:
        y = torch.randint(0, 3, (n_paper,), generator=gen)
        if unlabeled:
            y[torch.randperm(n_paper, generator=gen)[:unlabeled]] = -1
    data["paper"].y = y
    data["paper"].num_nodes = n_paper
    data["author"].num_nodes = n_author
    data["term"].num_nodes = n_term
    writes = pairs(n_paper, n_author, 80)
    has_term = pairs(n_paper, n_term, 60)
    data[PA].edge_index = writes
    data[AP].edge_index = writes.flip(0)
    data["paper", "to", "term"].edge_index = has_term
    data["term", "to", "paper"].edge_index = has_term.flip(0)
    data[CITE].edge_index = pairs(n_paper, n_paper, 50)
    return data


@unittest.skipUnless(_CLONES_PRESENT, "H2GB / Hetero2Net clones not present under vendor/")
class ToyGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        from scripts.heterophily import compute_heterophily as ch

        self.ch = ch
        self.data = _toy()

    def test_reference_h2gb_equals_hand_values(self) -> None:
        paths, results = self.ch.run_calc_homophily(self.data, "paper", seed=0, chunk_size=200)
        self.assertEqual(paths, [PAP, PCP])
        by_path = {r.metapath: r for r in results}
        self.assertEqual(by_path[PAP].h_edge, 2 / 3)
        self.assertEqual(by_path[PCP].h_edge, 3 / 4)
        # calcHomophily forms the adjustment in float32 (degree sums), so h_adj is float32-exact.
        self.assertAlmostEqual(by_path[PAP].h_adj, 1 / 9, places=6)
        self.assertAlmostEqual(by_path[PCP].h_adj, 1 / 3, places=6)

    def test_own_counts_equal_hand_values(self) -> None:
        pap = self.ch.homophily_counts(self.data, PAP, self.data["paper"].y, chunk_size=200)
        pcp = self.ch.homophily_counts(self.data, PCP, self.data["paper"].y, chunk_size=200)
        self.assertEqual((pap.num_edges, pcp.num_edges), (12, 8))
        self.assertAlmostEqual(pap.h_edge, 2 / 3, places=12)
        self.assertAlmostEqual(pcp.h_edge, 3 / 4, places=12)
        self.assertAlmostEqual(pap.h_node, 11 / 16, places=12)
        self.assertAlmostEqual(pcp.h_node, 3 / 4, places=12)

    def test_adjusted_homophily_applies_the_class_imbalance_correction(self) -> None:
        pap = self.ch.homophily_counts(self.data, PAP, self.data["paper"].y, chunk_size=200)
        pcp = self.ch.homophily_counts(self.data, PCP, self.data["paper"].y, chunk_size=200)
        self.assertAlmostEqual(pap.h_adj, 1 / 9, places=12)  # uniform shares would give 1/3
        self.assertAlmostEqual(pcp.h_adj, 1 / 3, places=12)  # uniform shares would give 1/2

    def test_reference_hetero2net_equals_hand_values(self) -> None:
        _, hetero2net = self.ch.reference_modules()
        y = self.data["paper"].y
        pap = self.ch.metapath_graph(self.data, PAP, chunk_size=200)
        pcp = self.ch.metapath_graph(self.data, PCP, chunk_size=200)
        self.assertAlmostEqual(hetero2net.label_homophily(pap, y), 2 / 3, places=6)
        self.assertAlmostEqual(hetero2net.label_homophily(pcp, y), 3 / 4, places=6)
        self.assertAlmostEqual(float(hetero2net.label_homophily_matrix(pap, y).mean()), 11 / 16,
                               places=6)
        self.assertAlmostEqual(float(hetero2net.label_homophily_matrix(pcp, y).mean()), 3 / 4,
                               places=6)

    def test_metapath_sets(self) -> None:
        self.assertEqual(self.ch.h2gb_metapaths(self.data, "paper"), [PAP, PCP])
        self.assertEqual(self.ch.hetero2net_metapaths(self.data, "paper"), [PAP])

    def test_aggregates(self) -> None:
        res = self.ch.analyse(self.data, "toy", "toy", seeds=[0], chunk_size=200,
                              node_ref_max_edges=10**6, cross_check_tol=1e-6)
        self.assertAlmostEqual(res.mlh_h2gb_heterophily_mean, 7 / 24, places=12)
        self.assertAlmostEqual(res.mlh_h2gb_heterophily_max, 1 / 3, places=12)
        self.assertAlmostEqual(res.h2_index, 7 / 9, places=6)
        self.assertAlmostEqual(res.mlh_hetero2net_homophily, 2 / 3, places=12)


@unittest.skipUnless(_CLONES_PRESENT, "H2GB / Hetero2Net clones not present under vendor/")
class RandomGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        from scripts.heterophily import compute_heterophily as ch

        self.ch = ch

    def test_own_equals_reference_on_a_random_graph(self) -> None:
        data = _random_graph(seed=0)
        y = data["paper"].y
        _, hetero2net = self.ch.reference_modules()
        paths, results = self.ch.run_calc_homophily(data, "paper", seed=0, chunk_size=7)
        self.assertEqual(paths, self.ch.h2gb_metapaths(data, "paper"))
        self.assertEqual(len(paths), 3)
        for result in results:
            own = self.ch.homophily_counts(data, result.metapath, y, chunk_size=7)
            edge_index = self.ch.metapath_graph(data, result.metapath, chunk_size=7)
            self.assertEqual(own.num_edges, edge_index.size(1))
            self.assertAlmostEqual(own.h_edge, result.h_edge, places=12)
            self.assertAlmostEqual(own.h_adj, result.h_adj, places=6)
            self.assertAlmostEqual(own.h_edge, hetero2net.label_homophily(edge_index, y), places=6)
            self.assertAlmostEqual(own.h_node,
                                   float(hetero2net.label_homophily_matrix(edge_index, y).mean()),
                                   places=6)

    def test_random_fill_matches_the_reference_and_labels_are_restored(self) -> None:
        data = _random_graph(seed=1, unlabeled=10)
        y_held = data["paper"].y  # a reference taken before the call (e.g. a shared graph's y)
        y_before = y_held.clone()
        _, results = self.ch.run_calc_homophily(data, "paper", seed=3, chunk_size=200)
        self.assertTrue(torch.equal(data["paper"].y, y_before))
        self.assertTrue(torch.equal(y_held, y_before))  # calcHomophily fills in place
        filled = self.ch.h2gb_random_fill(y_before, seed=3)
        self.assertTrue(bool((filled >= 0).all()))
        for result in results:
            own = self.ch.homophily_counts(data, result.metapath, filled, chunk_size=200)
            self.assertAlmostEqual(own.h_edge, result.h_edge, places=12)
            self.assertAlmostEqual(own.h_adj, result.h_adj, places=6)

    def test_unlabeled_target_without_fill_raises(self) -> None:
        data = _random_graph(seed=1, unlabeled=10)
        with self.assertRaises(ValueError):
            self.ch.homophily_counts(data, PAP, data["paper"].y, chunk_size=200)

    def test_exact_match_reproduction_row_equals_hetero2net(self) -> None:
        data = _random_graph(seed=2, multilabel=True)
        y = data["paper"].y
        _, hetero2net = self.ch.reference_modules()
        for path in self.ch.h2gb_metapaths(data, "paper"):
            own = self.ch.homophily_counts(data, path, y, chunk_size=200)
            edge_index = self.ch.metapath_graph(data, path, chunk_size=200)
            self.assertAlmostEqual(own.h_edge, hetero2net.label_homophily(edge_index, y), places=6)
            with self.assertRaises(ValueError):
                _ = own.h_adj


def _toy_multilabel(unlabeled: bool = False):
    """The toy graph with two genres, Y = [{0}, {0,1}, {1}, {0}] (multi-hot rows).

    Hand computation (G_P as in `_toy`):
    - Genre 0, y = [1, 1, 0, 1] (the complement of `_toy`'s partition): P-A-P h_edge 2/3,
      h_adj 1/9; P-cite-P h_edge 3/4, h_adj 1/3.
    - Genre 1, y = [0, 1, 1, 0]: P-A-P 6 of 12 pairs agree -> 1/2; in-degrees 2, 4, 3, 3 -> class
      shares 7/12 (nodes 1, 2), 5/12 -> sum of squares 37/72 -> h_adj = (1/2 - 37/72)/(35/72) =
      -1/35. P-cite-P 8 of 8 agree -> 1; shares 1/2, 1/2 -> h_adj = 1.
    - Per genre, heterophily: MLH = 1 - (17/24 + 3/4)/2 = 13/48; H^2 = 1 - (2/9 + 17/35)/2 =
      407/630.
    - Exact label-vector match (Hetero^2Net, the reproduction row): P-A-P 4/12, P-cite-P 6/8 ->
      MLH (heterophily) = 1 - 13/24 = 11/24, different from the per-genre 13/48.
    - Jaccard (Zhao et al. Def. 1): P-A-P 7/12, P-cite-P 7/8 -> mean 35/48 (homophily form).
    With `unlabeled`, paper 3 gets an all-zero row (PyG's unlabeled movie)."""
    data = _toy()
    y = torch.tensor([[1, 0], [1, 1], [0, 1], [1, 0]], dtype=torch.float)
    if unlabeled:
        y[3] = 0
    data["paper"].y = y
    return data


@unittest.skipUnless(_CLONES_PRESENT, "H2GB / Hetero2Net clones not present under vendor/")
class MultilabelPerGenreTests(unittest.TestCase):
    """J4.1 (a), (e): IMDB per genre, exact match only as the reproduction row, Jaccard appendix."""

    def setUp(self) -> None:
        from scripts.heterophily import compute_heterophily as ch

        self.ch = ch

    def _analyse(self, data):
        return self.ch.analyse(data, "toy", "toy", seeds=[0], chunk_size=200,
                               node_ref_max_edges=10**6, cross_check_tol=1e-6)

    def test_per_genre_reference_values_equal_hand_values(self) -> None:
        data = _toy_multilabel()
        genres = self.ch.genre_labels(data["paper"].y)
        self.assertEqual([g.tolist() for g in genres], [[1, 1, 0, 1], [0, 1, 1, 0]])
        expected = {0: {PAP: (2 / 3, 1 / 9), PCP: (3 / 4, 1 / 3)},
                    1: {PAP: (1 / 2, -1 / 35), PCP: (1.0, 1.0)}}
        for genre, y_g in enumerate(genres):
            _, results = self.ch.run_calc_homophily(data, "paper", seed=0, chunk_size=200,
                                                    labels=y_g)
            for r in results:
                self.assertAlmostEqual(r.h_edge, expected[genre][r.metapath][0], places=12)
                self.assertAlmostEqual(r.h_adj, expected[genre][r.metapath][1], places=6)

    def test_primary_values_are_per_genre_means_not_exact_match(self) -> None:
        res = self._analyse(_toy_multilabel())
        self.assertAlmostEqual(res.mlh_h2gb_heterophily_mean, 13 / 48, places=12)
        self.assertAlmostEqual(res.h2_index, 407 / 630, places=6)
        self.assertEqual([g["genre"] for g in res.per_genre], [0, 1])
        self.assertAlmostEqual(res.per_genre[1]["h2_index"], 1 - (-1 / 35 + 1) / 2, places=6)
        # The exact-match rule is kept only as the reproduction row.
        self.assertAlmostEqual(res.exact_match_mlh_h2gb_heterophily_mean, 11 / 24, places=6)
        self.assertAlmostEqual(res.mlh_hetero2net_homophily, 1 / 3, places=6)  # P-A-P only

    def test_jaccard_appendix_equals_hand_values(self) -> None:
        res = self._analyse(_toy_multilabel())
        by_path = {r["metapath"]: r for r in res.jaccard_rows}
        self.assertAlmostEqual(by_path[self.ch.metapath_name(PAP)]["jaccard"], 7 / 12, places=12)
        self.assertAlmostEqual(by_path[self.ch.metapath_name(PCP)]["jaccard"], 7 / 8, places=12)
        self.assertAlmostEqual(res.jaccard_homophily_labeled_only, 35 / 48, places=12)

    def test_unlabeled_movies_are_filled_per_genre_and_labeled_only_is_a_diagnostic(self) -> None:
        data = _random_graph(seed=2, multilabel=True)
        y = data["paper"].y
        unlabeled = ~self.ch.labeled_mask(y)
        self.assertGreater(int(unlabeled.sum()), 0)
        res = self._analyse(data)
        genres = self.ch.genre_labels(y)
        self.assertTrue(all(bool((g[unlabeled] == -1).all()) for g in genres))
        h_edge, h_adj, lo_edge = [], [], []
        for genre, y_g in enumerate(genres):
            _, results = self.ch.run_calc_homophily(
                data, "paper", seed=self.ch.genre_fill_seed(0, genre), chunk_size=200, labels=y_g)
            h_edge += [r.h_edge for r in results]
            h_adj += [r.h_adj for r in results]
            for r in results:
                lo_edge.append(self.ch.homophily_counts(
                    data, r.metapath, y_g, 200, keep=~unlabeled).h_edge)
        self.assertAlmostEqual(res.mlh_h2gb_heterophily_mean, 1 - sum(h_edge) / len(h_edge),
                               places=12)
        self.assertAlmostEqual(res.h2_index, 1 - sum(h_adj) / len(h_adj), places=12)
        self.assertAlmostEqual(res.mlh_labeled_only, 1 - sum(lo_edge) / len(lo_edge), places=12)
        self.assertGreater(abs(res.mlh_h2gb_heterophily_mean - res.mlh_labeled_only), 1e-3)
        self.assertTrue(torch.equal(data["paper"].y, y))


@unittest.skipUnless(_CLONES_PRESENT, "H2GB / Hetero2Net clones not present under vendor/")
class FillConventionTests(unittest.TestCase):
    """J4.1 (b): the primary value of a partly labeled single-label graph is calcHomophily's
    filled value at the first fill seed; the labeled-only value is only the diagnostic."""

    def test_primary_is_the_filled_reference_value(self) -> None:
        from scripts.heterophily import compute_heterophily as ch

        data = _random_graph(seed=1, unlabeled=10)
        res = ch.analyse(data, "toy", "toy", seeds=[3, 4], chunk_size=200,
                         node_ref_max_edges=10**6, cross_check_tol=1e-6)
        _, results = ch.run_calc_homophily(data, "paper", seed=3, chunk_size=200)
        self.assertAlmostEqual(res.h2_index, 1 - sum(r.h_adj for r in results) / len(results),
                               places=12)
        self.assertAlmostEqual(res.mlh_h2gb_heterophily_mean,
                               1 - sum(r.h_edge for r in results) / len(results), places=12)
        self.assertEqual(len(res.h2_index_over_seeds), 2)
        self.assertGreater(abs(res.h2_index - res.h2_index_labeled_only), 1e-3)

    def test_fully_labeled_graph_has_no_diagnostic_and_one_seed(self) -> None:
        from scripts.heterophily import compute_heterophily as ch

        res = ch.analyse(_toy(), "toy", "toy", seeds=[0, 1], chunk_size=200,
                         node_ref_max_edges=10**6, cross_check_tol=1e-6)
        self.assertIsNone(res.mlh_labeled_only)
        self.assertEqual(res.fill_seeds, [0])


@unittest.skipUnless(_CLONES_PRESENT, "H2GB / Hetero2Net clones not present under vendor/")
class TaskLevelHEdgeTests(unittest.TestCase):
    """J4.3: task-level H_edge of a same-type target relation (Freebase book-book)."""

    def test_hand_values_fill_and_labeled_only(self) -> None:
        from torch_geometric.data import HeteroData

        from scripts.heterophily import compute_heterophily as ch

        data = HeteroData()
        data["book"].y = torch.tensor([0, 1, 0, -1])
        data["book"].num_nodes = 4
        # Stored (0,1), (1,2), (0,0), (2,3); symmetrised: 00, 01, 10, 12, 21, 23, 32 (7 edges).
        data["book", "and", "book"].edge_index = torch.tensor([[0, 1, 0, 2], [1, 2, 0, 3]])
        out = ch.task_level_h_edge(data, ("book", "and", "book"), seeds=[5, 6],
                                   cross_check_tol=1e-6)
        self.assertEqual((out["edges"], out["self_loops"], out["labeled_edges"]), (7, 1, 5))
        # Labeled-only: of 00, 01, 10, 12, 21 only 00 agrees -> heterophily 4/5.
        self.assertAlmostEqual(out["labeled_only_h_edge_heterophily"], 4 / 5, places=12)
        filled = ch.h2gb_random_fill(data["book"].y, 5)
        edges = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
        agree = sum(int(filled[u] == filled[v]) for u, v in edges)
        self.assertAlmostEqual(out["h_edge_heterophily"], 1 - agree / 7, places=6)
        self.assertEqual(len(out["h_edge_heterophily_over_seeds"]), 2)
        self.assertTrue(torch.equal(data["book"].y, torch.tensor([0, 1, 0, -1])))


@unittest.skipUnless(_CLONES_PRESENT, "H2GB / Hetero2Net clones not present under vendor/")
class SymmetriseTests(unittest.TestCase):
    def test_same_type_undirected_cross_type_reversed(self) -> None:
        from torch_geometric.data import HeteroData

        from scripts.heterophily import compute_heterophily as ch

        data = HeteroData()
        data["book"].y = torch.tensor([0, 1, 0])
        data["book"].num_nodes = 3
        data["film"].num_nodes = 2
        data["book", "and", "book"].edge_index = torch.tensor([[0], [1]])
        data["book", "to", "film"].edge_index = torch.tensor([[0, 2], [1, 0]])
        out = ch.symmetrise_like_h2gb(data)
        self.assertEqual(sorted(map(tuple, out["book", "and", "book"].edge_index.t().tolist())),
                         [(0, 1), (1, 0)])
        self.assertTrue(torch.equal(out["film", "rev_to", "book"].edge_index,
                                    torch.tensor([[1, 0], [0, 2]])))
        self.assertEqual(ch.h2gb_metapaths(data, "book"),
                         [(("book", "and", "book"), ("book", "and", "book"))])
        self.assertIn((("book", "to", "film"), ("film", "rev_to", "book")),
                      ch.h2gb_metapaths(out, "book"))


if __name__ == "__main__":
    unittest.main()
