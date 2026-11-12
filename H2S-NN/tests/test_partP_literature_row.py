"""Part P, P0b (`docs/H2S-NN_partP_card.md`; spec v1.20 U1; decisions D-O3, D-O4, D-O6): the
fixed-literature-schema row's hop rule, cell mass and run directory.

(a) hop rule: on a hand graph (anchor P1, author A7 with papers P1, P2, P3) each P1-A7-P_j instance
    has probability 1/2 (WISP's rule: uniform fact to an unvisited node), against the hand value
    (Monte Carlo, 1e5 walks, 4-sigma) and against `sample_wisp_walks` on the same one-action graph
    (exact log p and Monte Carlo); a dead end discards the walk.
(b) mass: per anchor and listed path, the sum of m over its walks has mean (1/k) P(not discarded)
    (4-sigma); without dead ends the total per anchor is exactly 1 (float64, dyadic m).
(c) row vs `full` (WISP): total walk mass per anchor of both <= 1, and both equal 1 when neither has
    dead ends.
(d) the row's run directory differs from the `full` run's; a row is not built from a row.
Card mutations: m = 1/r_pi fails (b); D8's rejection walk fails (a).
"""

from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.data.literature_paths import sample_listed_walks, sample_literature_walks
from src.data.schemas import ResolvedSchema
from src.data.wisp_sampler import build_action_graph, sample_wisp_walks

# Papers P1, P2, P3 = 0, 1, 2; authors A7 = 3, A8 = 4. Relation 0 = writes, stored paper -> author.
PAP_HOPS = ((0, 1), (0, -1))


def _graph(edges: list[tuple[int, int]], num_nodes: int = 5):
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
    return edge_index, edge_type, build_action_graph(edge_index, edge_type, num_nodes, ())


A7_GRAPH = [(0, 3), (1, 3), (2, 3)]               # A7 wrote P1, P2, P3: no dead end from P1
DEAD_END_GRAPH = [(0, 3), (1, 3), (2, 3), (0, 4)]  # plus A8 who wrote only P1: P1-A8 dead-ends
NODE_TYPE = torch.tensor([0, 0, 0, 1, 1])


def _four_sigma(count: int, n: int, p: float) -> float:
    return abs(count - n * p) / math.sqrt(n * p * (1 - p))


class HopRuleTests(unittest.TestCase):
    N = 100_000

    def test_instance_probability_is_wisps(self) -> None:
        _, _, graph = _graph(A7_GRAPH)
        walks = sample_listed_walks(graph, PAP_HOPS, torch.tensor([0]), self.N, 1.0,
                                    torch.Generator().manual_seed(0))
        ends = walks.nodes[:, -1]
        self.assertTrue(bool((walks.nodes[:, 1] == 3).all()))
        self.assertFalse(bool((ends == 0).any()), "the visited P1 is never chosen")
        for paper in (1, 2):  # unconditional: every drawn walk is accepted here
            with self.subTest(paper=paper):
                self.assertLess(_four_sigma(int((ends == paper).sum()), self.N, 0.5), 4.0)
        # WISP on the same one-action graph (l_max 2: |A_j| = 1 at both hops)
        edge_index, edge_type, _ = _graph(A7_GRAPH)
        wisp_graph = build_action_graph(edge_index, edge_type, 5, ())
        draw = sample_wisp_walks(wisp_graph, NODE_TYPE, torch.tensor([0]), self.N, 2,
                                 torch.Generator().manual_seed(1))
        self.assertEqual(draw.nodes.size(0), self.N)
        self.assertTrue(torch.equal(draw.log_p, torch.full_like(draw.log_p, math.log(0.5))))
        for paper in (1, 2):
            with self.subTest(wisp_paper=paper):
                self.assertLess(_four_sigma(int((draw.nodes[:, 2] == paper).sum()), self.N, 0.5),
                                4.0)
        self.assertEqual({tuple(r) for r in walks.nodes.tolist()},
                         {tuple(r) for r in draw.nodes[:, :3].tolist()})

    def test_dead_end_discards(self) -> None:
        _, _, graph = _graph([(0, 4), (1, 3)])  # P1's only author A8 wrote only P1
        walks = sample_listed_walks(graph, PAP_HOPS, torch.tensor([0]), 1000, 1.0,
                                    torch.Generator().manual_seed(0))
        self.assertEqual(walks.nodes.size(0), 0)
        _, _, graph = _graph(DEAD_END_GRAPH)  # A7 (1/2, continues) or A8 (1/2, dead end)
        walks = sample_listed_walks(graph, PAP_HOPS, torch.tensor([0]), self.N, 1.0,
                                    torch.Generator().manual_seed(2))
        self.assertLess(_four_sigma(walks.nodes.size(0), self.N, 0.5), 4.0)
        self.assertTrue(bool((walks.nodes[:, 1] == 3).all()))


def _schemas(k: int) -> tuple[ResolvedSchema, ...]:
    return tuple(ResolvedSchema(f"PAP{i}", ("paper", "author", "paper"), PAP_HOPS, None)
                 for i in range(k))


def _mass_per_anchor(walks, num_nodes: int = 5) -> torch.Tensor:
    total = torch.zeros(num_nodes, dtype=torch.float64)
    total.index_add_(0, walks.nodes[:, 0], walks.weight)
    return total


class MassTests(unittest.TestCase):
    def test_expected_mass_is_share_times_survival(self) -> None:
        _, _, graph = _graph(DEAD_END_GRAPH)
        k, r = 2, 20_000  # r_pi = 10,000; P(not discarded) = 1/2 from P1
        for schema, walks in sample_literature_walks(graph, _schemas(k), torch.tensor([0]), r, 3):
            with self.subTest(path=schema.name):
                r_pi = r // k
                self.assertTrue(bool((walks.weight == 1.0 / (k * r_pi)).all()))
                mass = float(walks.weight.sum())
                sigma = math.sqrt(r_pi * 0.25) / (k * r_pi)
                self.assertLess(abs(mass - 0.5 / k) / sigma, 4.0)

    def test_total_mass_without_dead_ends_is_one(self) -> None:
        _, _, graph = _graph(A7_GRAPH)
        for k, r in ((2, 8), (4, 16), (1, 64), (2, 64)):  # m = 1/(k r_pi) dyadic: exact sums
            with self.subTest(k=k, r=r):
                parts = sample_literature_walks(graph, _schemas(k), torch.tensor([0, 1, 2]), r, 5)
                total = sum(_mass_per_anchor(walks) for _, walks in parts)
                self.assertTrue(torch.equal(total[:3], torch.ones(3, dtype=torch.float64)))
        parts = sample_literature_walks(graph, _schemas(3), torch.tensor([0, 1, 2]), 16, 5)
        total = sum(_mass_per_anchor(walks) for _, walks in parts)  # r_pi 6, 5, 5
        self.assertLessEqual(float((total[:3] - 1).abs().max()), 1e-15)


def _bipartite(papers: int = 6, authors: int = 4):
    edges = [(p, papers + a) for p in range(papers) for a in range(authors)]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
    node_type = torch.tensor([0] * papers + [1] * authors)
    return edge_index, edge_type, node_type, build_action_graph(edge_index, edge_type,
                                                                 papers + authors, ())


class RowVersusFullTests(unittest.TestCase):
    def _totals(self, edge_index, edge_type, node_type, graph, anchors, r):
        schemas = (ResolvedSchema("PAP", ("paper", "author", "paper"), PAP_HOPS, None),
                   ResolvedSchema("PAPA", ("paper", "author", "paper", "author"),
                                  PAP_HOPS + ((0, 1),), None))
        row = sum(_mass_per_anchor(w, node_type.numel())
                  for _, w in sample_literature_walks(graph, schemas, anchors, r, 9))
        draw = sample_wisp_walks(graph, node_type, anchors, r, 3, torch.Generator().manual_seed(9))
        full = torch.zeros(node_type.numel(), dtype=torch.float64)
        full.index_add_(0, draw.anchor, draw.ratio / r)  # WISP's m per walk: ratio / r
        return row[anchors], full[anchors]

    def test_equal_one_without_dead_ends(self) -> None:
        edge_index, edge_type, node_type, graph = _bipartite()
        anchors = torch.arange(6)
        row, full = self._totals(edge_index, edge_type, node_type, graph, anchors, 8)
        self.assertTrue(torch.equal(row, torch.ones(6, dtype=torch.float64)))
        self.assertTrue(torch.equal(full, torch.ones(6, dtype=torch.float64)))

    def test_at_most_one_with_dead_ends(self) -> None:
        edge_index, edge_type, graph = _graph(DEAD_END_GRAPH)
        anchors = torch.tensor([0, 1, 2])
        row, full = self._totals(edge_index, edge_type, NODE_TYPE, graph, anchors, 64)
        self.assertTrue(bool((row <= 1.0).all()) and bool((full <= 1.0).all()))
        self.assertLess(float(row[0]), 1.0)  # P1's dead end through A8


class RunDirectoryTests(unittest.TestCase):
    def test_row_has_its_own_run_directory(self) -> None:
        from src.configs.config import TrainConfig, fixed_literature_schema_config

        full = TrainConfig.from_dict({
            "device": "cpu", "exp_name": "acm_full_seed0", "output_dir": "runs",
            "data": {"dataset_name": "acm", "wisp_max_length": 3, "wisp_walks_per_anchor": 16},
            "model": {"propagation": "anchored", "arm": "full"},
            "ablation": {"coupling": "symmetric"}})
        row = fixed_literature_schema_config(full)
        self.assertNotEqual(row.exp_dir, full.exp_dir)
        self.assertEqual(row.output_dir, full.output_dir)
        self.assertEqual(row.exp_name, "acm_full_seed0__literature_row")
        with self.assertRaises(ValueError):
            fixed_literature_schema_config(row)


if __name__ == "__main__":
    unittest.main()
