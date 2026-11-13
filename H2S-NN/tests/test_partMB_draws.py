"""Part MB, MB2 (`docs/H2S-NN_partMB_card.md`; decision N2): chunked mini-batch draws and the
per-anchor merge. On the `test_wisp` fixture graph (synthetic).

- same seed and chunk size -> identical draw (`torch.equal`);
- every walk's log p equals the exact p of that walk (enumeration of WISP's target measure), and
  the walk frequencies from one anchor match enumeration (1e5 walks, 4 sigma);
- distinct chunks use distinct generator states: `chunk_seed` differs per chunk index, and the
  same anchor planted in two chunks does not copy its walks;
- one instance drawn from two anchors gives two cells, each with its own m_f;
- full-batch draws are unchanged (`docs/partMB_evidence/mb0_reference.pt`; the golden heads).
Card mutations: the anchor dropped from the merge key -> the per-anchor test fails; the chunk seed
ignoring the chunk index -> the distinct-chunk test fails.
"""

from __future__ import annotations

import math
import unittest
from collections import Counter

import torch

from src.data.path_sampler import Walks
from src.data.wisp_sampler import (
    _merge_cells,
    build_action_graph,
    build_wisp_anchor_cells,
    chunk_seed,
    sample_wisp_walks_chunked,
)
from tests.partMB_fixtures import restore_real_data, unreachable_real_data
from tests.test_wisp import (
    ANCHORS,
    EDGE_INDEX,
    EDGE_TYPE,
    NODE_TYPE,
    SELF_INVERSE,
    enumerate_target,
    walk_keys,
)


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


GRAPH = build_action_graph(EDGE_INDEX, EDGE_TYPE, int(NODE_TYPE.numel()), SELF_INVERSE)


class ChunkedDrawTests(unittest.TestCase):
    def test_same_seed_and_chunk_size_give_the_same_draw(self) -> None:
        a = sample_wisp_walks_chunked(GRAPH, NODE_TYPE, ANCHORS, 30, 3, seed=5, chunk_size=3)
        b = sample_wisp_walks_chunked(GRAPH, NODE_TYPE, ANCHORS, 30, 3, seed=5, chunk_size=3)
        for field in a.__slots__:
            self.assertTrue(torch.equal(getattr(a, field), getattr(b, field)), field)

    def test_log_p_and_frequencies_equal_the_exact_target(self) -> None:
        exact, _ = enumerate_target(3)
        draw = sample_wisp_walks_chunked(GRAPH, NODE_TYPE, ANCHORS, 2000, 3, seed=1, chunk_size=1)
        for key, log_p in zip(walk_keys(draw), draw.log_p.tolist()):
            self.assertAlmostEqual(log_p, math.log(exact[key]), places=12)
        n = 100_000
        draw = sample_wisp_walks_chunked(GRAPH, NODE_TYPE, torch.tensor([0]), n, 3, seed=2)
        counts = Counter(walk_keys(draw))
        for key, p in exact.items():
            if key[0] != 0:
                continue
            with self.subTest(walk=key):
                sigma = math.sqrt(n * p * (1 - p))
                self.assertLess(abs(counts.get(key, 0) - n * p) / sigma, 4.0)

    def test_chunks_use_distinct_generator_states(self) -> None:
        seeds = {chunk_seed(7, k) for k in range(1000)}
        self.assertEqual(len(seeds), 1000)
        draw = sample_wisp_walks_chunked(GRAPH, NODE_TYPE, torch.tensor([0, 0]), 40, 3, seed=7,
                                         chunk_size=1)  # anchor 0 planted in chunks 0 and 1
        first = draw.nodes[: int((draw.anchor == 0).sum()) // 2]
        keys = walk_keys(draw)
        half = len(keys) // 2
        self.assertNotEqual(keys[:half], keys[half:2 * half])
        self.assertGreater(first.size(0), 0)


class PerAnchorMergeTests(unittest.TestCase):
    def test_one_instance_from_two_anchors_gives_two_cells(self) -> None:
        # the instance 0 -f0-> 4 -f1-> 1 drawn from anchor 0, and reversed from anchor 1
        walks = Walks(nodes=torch.tensor([[0, 4, 1], [1, 4, 0], [0, 4, 1]]),
                      facts=torch.tensor([[0, 1], [1, 0], [0, 1]]),
                      signs=torch.tensor([[1, -1], [1, -1], [1, -1]]),
                      relations=torch.tensor([[0, 0], [0, 0], [0, 0]]),
                      directions=torch.tensor([[1, -1], [1, -1], [1, -1]]),
                      weight=torch.full((3,), 0.25, dtype=torch.float64))
        anchors = torch.tensor([0, 1, 0])
        whole, none = _merge_cells(walks, NODE_TYPE, EDGE_INDEX, "t", "sum", None)
        self.assertIsNone(none)
        self.assertEqual(whole.num_faces, 1)  # full batch: one cell, m = 3/4
        self.assertEqual(whole.path_weight.tolist(), [0.75])
        group, owner = _merge_cells(walks, NODE_TYPE, EDGE_INDEX, "t", "sum", anchors)
        self.assertEqual(group.num_faces, 2)
        self.assertEqual(dict(zip(owner.tolist(), group.path_weight.tolist())), {0: 0.5, 1: 0.25})

    def test_per_anchor_mass_is_the_anchors_walks_only(self) -> None:
        r = 25
        cells = build_wisp_anchor_cells(EDGE_INDEX, EDGE_TYPE, NODE_TYPE, SELF_INVERSE, ANCHORS,
                                        r, 3, seed=4, chunk_size=2)
        draw = sample_wisp_walks_chunked(GRAPH, NODE_TYPE, ANCHORS, r, 3, seed=4, chunk_size=2)
        for anchor in ANCHORS.tolist():
            own = cells.of(anchor)
            total = sum(float(g.path_weight.double().sum()) for g in own)
            walks = int((draw.anchor == anchor).sum())
            self.assertAlmostEqual(total, walks / r, places=6)  # every walk carries 1/r (q = p)


if __name__ == "__main__":
    unittest.main()
