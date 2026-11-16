"""Card 6 H0: the heterophily harness (`src/analysis/heterophily.py`) on synthetic graphs only.

Hand graph (one meta-path paper-author-paper): papers P0..P3 with labels 0, 0, 1, 1 and features [1,0], [0,1], [1,1],
[2,0]; author A0 wrote P0, P1; author A1 wrote P1, P2, P3. G_P (support of the product, self-pairs kept):
P0 -> {P0, P1}, P1 -> {P0, P1, P2, P3}, P2 -> {P1, P2, P3}, P3 -> {P1, P2, P3}: 12 edges, degrees 2, 4, 3, 3.
* same-label edges 8 of 12: h_edge = 2/3, MLH (heterophily) = 1/3;
* calcHomophily: in-degree sums per class 6, 6 of 12 edges: adjust = (36 + 36) / 144 = 1/2,
  h_adj = (2/3 − 1/2) / (1 − 1/2) = 1/3, ℋ² = 2/3; class-insensitive h = 1/3;
* MDE: with v_i = x_i / √d_i, the 8 non-self edges give ‖v0−v1‖² = 3/4, ‖v1−v2‖² = 11/12 − 1/√3,
  ‖v1−v3‖² = 19/12, ‖v2−v3‖² = 2/3, each twice: sum = 47/6 − 2/√3; MDE = sum / 4 / 12.
Every test makes the real data root unreachable (tests never read real data).
"""

from __future__ import annotations

import math
import os
import unittest

os.environ.setdefault("TQDM_DISABLE", "1")

import torch  # noqa: E402
from torch_geometric.data import HeteroData  # noqa: E402

from src.analysis import heterophily as het  # noqa: E402
from tests.partMB_fixtures import restore_real_data, unreachable_real_data  # noqa: E402

PAP = [[("paper", "author"), ("author", "paper")]]
HAND_MDE = (47 / 6 - 2 / math.sqrt(3)) / 48


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def hand_graph(y: list[int] | None = None) -> HeteroData:
    data = HeteroData()
    data["paper"].num_nodes = 4
    data["paper"].x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    data["paper"].y = torch.tensor(y if y is not None else [0, 0, 1, 1])
    data["author"].num_nodes = 2
    writes = torch.tensor([[0, 1, 1, 2, 3], [0, 0, 1, 1, 1]])
    data["paper", "to", "author"].edge_index = writes
    data["author", "to", "paper"].edge_index = writes.flip(0)
    return data


def schema(edge_types: list[tuple[str, str, str]]) -> HeteroData:
    data = HeteroData()
    for t in sorted({e[0] for e in edge_types} | {e[2] for e in edge_types}):
        data[t].num_nodes = 2
    for e in edge_types:
        data[e].edge_index = torch.tensor([[0, 1], [1, 0]])
    return data


ACM_SCHEMA = [("paper", "cite", "paper"), ("paper", "ref", "paper"), ("paper", "to", "author"),
              ("author", "to", "paper"), ("paper", "to", "subject"), ("subject", "to", "paper"),
              ("paper", "to", "term"), ("term", "to", "paper")]


class HandGraphTests(unittest.TestCase):
    def test_gp_is_the_support_of_the_product_with_self_pairs(self) -> None:
        edges = het.gp_edge_index(hand_graph(), PAP[0])
        got = sorted(zip(edges[0].tolist(), edges[1].tolist()))
        want = sorted([(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)])
        self.assertEqual(got, want)

    def test_mlh_h2_mde_equal_the_hand_values(self) -> None:
        data = hand_graph()
        mlh = het.mlh_measure(data, "paper", PAP, data["paper"].y)
        self.assertAlmostEqual(mlh["value"], 1 / 3, places=6)
        self.assertEqual(mlh["per_metapath"][0]["edges"], 12)
        h2 = het.h2_measure(data, "paper", data["paper"].y)
        row = h2["per_problem"]["label"]["runs"][0]["per_metapath"][0]
        self.assertAlmostEqual(row["h_edge"], 2 / 3, places=6)
        self.assertAlmostEqual(row["h_adj"], 1 / 3, places=6)
        self.assertAlmostEqual(row["h_insensitive"], 1 / 3, places=6)
        self.assertAlmostEqual(h2["value"], 2 / 3, places=6)
        self.assertEqual(h2["value_over_fill_seeds"], [h2["value"]])  # fully labeled: one call, no fill
        mde = het.mde_measure(data, "paper", PAP, data["paper"].x, both=True)
        rec = mde["per_metapath"][0]
        self.assertAlmostEqual(rec["reference"], HAND_MDE, places=6)
        self.assertAlmostEqual(rec["chunked"], HAND_MDE, places=6)
        self.assertAlmostEqual(mde["value"], HAND_MDE, places=6)

    def test_fill_replica_equals_calc_homophily_in_place_fill(self) -> None:
        data = hand_graph([0, -1, 2, -1])
        for seed in (0, 1, 7):
            _, used, _ = het.calc_homophily(data, "paper", data["paper"].y, seed)
            self.assertTrue(torch.equal(used, het.fill_like_h2gb(data["paper"].y, seed)))
        self.assertEqual(data["paper"].y.tolist(), [0, -1, 2, -1])  # the caller's labels are never filled

    def test_unlabeled_targets_give_five_seeds_and_a_labeled_only_value(self) -> None:
        y = torch.tensor([0, 0, 1, -1])
        data = hand_graph(y.tolist())
        h2 = het.h2_measure(data, "paper", y)
        self.assertEqual(len(h2["value_over_fill_seeds"]), 5)
        # labeled-only: G_P restricted to P0..P2 is P0->{0,1}, P1->{0,1,2}, P2->{1,2}; computed by calcHomophily
        # on that graph built by hand
        manual = HeteroData()
        manual["paper"].num_nodes = 3
        manual["author"].num_nodes = 2
        writes = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
        manual["paper", "to", "author"].edge_index = writes
        manual["author", "to", "paper"].edge_index = writes.flip(0)
        rows, _, _ = het.calc_homophily(manual, "paper", torch.tensor([0, 0, 1]), seed=0)
        self.assertAlmostEqual(h2["labeled_only"], 1 - rows[0]["h_adj"], places=9)
        mlh = het.mlh_measure(data, "paper", PAP, y)
        same = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]  # of the 7 labeled pairs
        self.assertAlmostEqual(mlh["labeled_only"], 1 - len(same) / 7, places=6)

    def test_labeled_only_graph_keeps_intermediate_targets(self) -> None:
        # same-type meta-path paper-cites-paper-cites-paper: endpoints restricted, intermediates are all papers
        g = torch.Generator().manual_seed(3)
        data = HeteroData()
        data["paper"].num_nodes = 12
        data["paper", "cites", "paper"].edge_index = torch.randint(0, 12, (2, 30), generator=g)
        keep = torch.tensor([True, False] * 6)
        path = [("paper", "cites", "paper"), ("paper", "cites", "paper")]
        mini = het.labeled_only_metapath_data(data, path, "paper", keep)
        self.assertEqual(het.h2gb_metapaths(mini.edge_types, "tgt"), [[("tgt", "cites", "mid"), ("mid", "cites", "tgt")]])
        sub = het.gp_edge_index(mini, [("tgt", "mid"), ("mid", "tgt")])
        ids = keep.nonzero().view(-1)
        got = sorted(zip(ids[sub[0]].tolist(), ids[sub[1]].tolist()))
        full = het.gp_edge_index(data, [("paper", "paper"), ("paper", "paper")])
        want = sorted((a, b) for a, b in zip(full[0].tolist(), full[1].tolist()) if keep[a] and keep[b])
        self.assertEqual(got, want)


class MetapathSetTests(unittest.TestCase):
    def test_acm_uses_the_predefined_list_even_when_the_schema_offers_more(self) -> None:
        data = schema(ACM_SCHEMA + [("paper", "to", "venue"), ("venue", "to", "paper")])
        paths, note = het.hetero2net_m2("acm", data.edge_types, "paper")
        self.assertEqual(paths, [[("paper", "author"), ("author", "paper")], [("paper", "subject"), ("subject", "paper")],
                                 [("paper", "term"), ("term", "paper")]])
        self.assertIn("predefined", note)

    def test_lists_resolve_on_the_loader_schemas(self) -> None:
        loader_schemas = {
            "acm": ("paper", ACM_SCHEMA),
            "dblp": ("author", [("author", "to", "paper"), ("paper", "to", "term"), ("paper", "to", "venue"),
                                ("paper", "to", "author"), ("term", "to", "paper"), ("venue", "to", "paper")]),
            "imdb": ("movie", [("movie", "to", "director"), ("director", "to", "movie"), ("movie", ">actorh", "actor"),
                               ("actor", "to", "movie"), ("movie", "to", "keyword"), ("keyword", "to", "movie")]),
            "ogbn-mag": ("paper", [("author", "writes", "paper"), ("paper", "rev_writes", "author"),
                                   ("paper", "cites", "paper"), ("paper", "has_topic", "field_of_study"),
                                   ("field_of_study", "rev_has_topic", "paper")]),
        }
        names = {"acm": ["paper-author-paper", "paper-subject-paper", "paper-term-paper"],
                 "dblp": ["author-paper-author"], "imdb": ["movie-director-movie", "movie-actor-movie",
                                                          "movie-keyword-movie"],
                 "ogbn-mag": ["paper-field_of_study-paper", "paper-author-paper"]}
        for key, (target, edge_types) in loader_schemas.items():
            paths, _ = het.hetero2net_m2(key, edge_types, target)
            self.assertEqual([het.metapath_name(p) for p in paths], names[key], key)
            for path in paths:
                for s, d in path:
                    self.assertEqual(len(het.step_relations(edge_types, s, d)), 1, (key, s, d))

    def test_get_metapaths_fallback_skips_same_type_and_keeps_repeats_once(self) -> None:
        edge_types = [("paper", "cites", "paper"), ("paper", "w1", "author"), ("paper", "w2", "author"),
                      ("author", "r1", "paper"), ("author", "r2", "paper")]
        paths, note = het.hetero2net_m2(None, edge_types, "paper")
        self.assertEqual(paths, [[("paper", "author"), ("author", "paper")]])
        self.assertIn("repeats kept once", note)
        self.assertEqual(len(het.step_relations(edge_types, "paper", "author")), 2)  # step = union of relations

    def test_h2gb_rule_includes_a_same_type_relation(self) -> None:
        edge_types = [("paper", "cites", "paper"), ("paper", "to", "author"), ("author", "to", "paper")]
        h2gb = het.h2gb_metapaths(edge_types, "paper")
        self.assertIn([("paper", "cites", "paper"), ("paper", "cites", "paper")], h2gb)
        self.assertIn([("paper", "to", "author"), ("author", "to", "paper")], h2gb)
        hetero2net, _ = het.hetero2net_m2(None, edge_types, "paper")
        self.assertNotIn([("paper", "paper"), ("paper", "paper")], hetero2net)


class ChunkedEqualsReferenceTests(unittest.TestCase):
    def test_chunked_mde_equals_the_reference_function(self) -> None:
        g = torch.Generator().manual_seed(0)
        for n, e, dim in [(5, 9, 1), (37, 120, 3), (200, 900, 16)]:
            active = n - 3  # the last 3 nodes are isolated
            index = torch.randint(0, active, (2, e), generator=g)
            index = torch.cat([index, index[:, : e // 5]], dim=1)  # multi-edges
            index = torch.cat([index, torch.tensor([[0, 1], [0, 1]])], dim=1)  # self-loops
            x = torch.randn(n, dim, generator=g)
            reference = het.mde_reference(index, x)
            for chunk in (1, 7, 64, index.size(1) + 5):
                value, edges = het.mde_chunked(het.chunks_of(index, chunk), x, chunk=max(1, chunk // 2))
                self.assertEqual(edges, index.size(1))
                self.assertTrue(math.isclose(value, reference, rel_tol=1e-5), (n, chunk, value, reference))

    def test_chunked_label_homophily_equals_one_call(self) -> None:
        g = torch.Generator().manual_seed(1)
        index = torch.randint(0, 50, (2, 333), generator=g)
        y = torch.randint(0, 4, (50,), generator=g)
        one = het.hetero2net_homophily().label_homophily(index, y)
        for chunk in (1, 10, 100, 1000):
            self.assertAlmostEqual(het.label_homophily_chunked(index, y, chunk), one, places=6)

    def test_measures_take_the_chunked_path_above_the_threshold(self) -> None:
        data = hand_graph()
        mde = het.mde_measure(data, "paper", PAP, data["paper"].x, max_reference_edges=5)
        self.assertEqual(mde["per_metapath"][0]["code_path"], "mde_chunked (D-a)")
        self.assertAlmostEqual(mde["value"], HAND_MDE, places=6)
        mlh = het.mlh_measure(data, "paper", PAP, data["paper"].y, max_single_call=5, chunk=3)
        self.assertTrue(mlh["per_metapath"][0]["mode"].startswith("chunked"))
        self.assertAlmostEqual(mlh["value"], 1 / 3, places=6)


class MultilabelTests(unittest.TestCase):
    def test_imdb_per_genre_binary_then_mean_and_exact_match_separate(self) -> None:
        data = hand_graph()
        # genres of P0..P3; P3 unlabeled (all-zero row)
        y = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=torch.float)
        genres = het.genre_labels(y)
        self.assertEqual([g.tolist() for g in genres], [[1, 1, 0, -1], [0, 1, 1, -1], [1, 0, 1, -1]])
        mlh = het.mlh_measure(data, "paper", PAP, y)
        edges = het.gp_edge_index(data, PAP[0])
        lh = het.hetero2net_homophily().label_homophily
        per_genre = []
        for g, lab in enumerate(genres):
            filled = het.fill_like_h2gb(lab, het.genre_fill_seed(0, g))
            per_genre.append(1 - lh(edges, filled))
            self.assertAlmostEqual(mlh["per_genre"][f"genre_{g}"], per_genre[-1], places=6)
        self.assertAlmostEqual(mlh["value"], sum(per_genre) / 3, places=6)
        self.assertEqual(len(mlh["value_over_fill_seeds"]), 5)
        exact = lh(edges, y)  # all genres equal, as loaded
        self.assertAlmostEqual(mlh["reproduction_exact_match"]["homophily"], exact, places=6)
        self.assertNotAlmostEqual(mlh["value"], 1 - exact, places=3)
        h2 = het.h2_measure(data, "paper", y)
        self.assertEqual(sorted(h2["per_genre"]), ["genre_0", "genre_1", "genre_2"])
        self.assertAlmostEqual(h2["value"], sum(h2["per_genre"].values()) / 3, places=9)


class GraphTests(unittest.TestCase):
    def test_gc_type_mixing_hand_value(self) -> None:
        node_type = torch.tensor([0, 0, 1, 2])
        und = torch.tensor([[0, 1], [1, 2], [2, 3]]).t()
        index = torch.cat([und, und.flip(0)], dim=1)  # stored both ways
        self.assertEqual(het.type_mixing(node_type, index), (4, 6))
        stats = het.gc_type_statistics([node_type, torch.tensor([0, 0])], [index, torch.tensor([[0, 1], [1, 0]])], 3, 1)
        self.assertAlmostEqual(stats["type_mixing_ratio"], 4 / 8)
        self.assertAlmostEqual(stats["mean_per_graph_ratio"], (4 / 6 + 0) / 2)

    def test_symmetrise_adds_only_missing_reverses(self) -> None:
        data = HeteroData()
        data["a"].num_nodes = 3
        data["b"].num_nodes = 2
        data["a", "x", "b"].edge_index = torch.tensor([[0, 1], [1, 0]])
        data["b", "rx", "a"].edge_index = torch.tensor([[1, 0], [0, 1]])   # exact reverse of x
        data["a", "y", "b"].edge_index = torch.tensor([[2], [1]])          # no partner
        data["a", "z", "a"].edge_index = torch.tensor([[0], [2]])          # same type, no partner
        out, record = het.symmetrise_facts(data)
        self.assertTrue(record["a-x-b"].startswith("kept"))
        self.assertTrue(torch.equal(out["b", "rev_y", "a"].edge_index, torch.tensor([[1], [2]])))
        self.assertEqual(sorted(zip(*out["a", "z", "a"].edge_index.tolist())), [(0, 2), (2, 0)])
        self.assertNotIn(("b", "rev_x", "a"), out.edge_types)


if __name__ == "__main__":
    unittest.main()
