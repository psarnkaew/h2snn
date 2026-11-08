"""WISP (task card Part H): sampler tests T-H1, T-H2 (q = p), T-H4, T-H9 (H1), extended in H2-H4.

Fixture: node types p (anchors), a, s; relations pa (p->a), ps (p->s), cite (p->p, a reciprocal
pair), co (a-a, self-inverse, with a self-loop); a parallel pair (two pa facts p2->a1), a dead end
(s1, reached only from p3) and a hub (s0, four papers). The exact walk measure is enumerated by an
independent reference below (its own half-edge list, no sampler code)."""
from __future__ import annotations

import math
import unittest
from collections import defaultdict

import torch

from src.data.wisp_sampler import (WispDraw, build_action_graph, build_wisp_path_groups,
                                   merge_wisp_cells, sample_wisp_walks)

NODE_TYPE = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])  # p0..p3, a0, a1, s0, s1
FACTS = [  # (src, dst, relation): 0 pa, 1 ps, 2 cite, 3 co (self-inverse)
    (0, 4, 0), (1, 4, 0), (2, 5, 0), (2, 5, 0), (3, 5, 0), (0, 5, 0),
    (0, 6, 1), (1, 6, 1), (2, 6, 1), (3, 6, 1), (3, 7, 1),
    (0, 1, 2), (1, 0, 2), (2, 3, 2),
    (4, 5, 3), (5, 5, 3),
]
SELF_INVERSE = (3,)
EDGE_INDEX = torch.tensor([[u for u, _, _ in FACTS], [v for _, v, _ in FACTS]])
EDGE_TYPE = torch.tensor([r for _, _, r in FACTS])
ANCHORS = torch.arange(4)


def reference_half_edges():
    """(node, other, fact, sign, relation, direction) per traversal, written independently."""
    half = []
    for f, (u, v, r) in enumerate(FACTS):
        if r in SELF_INVERSE:
            half.append((u, v, f, 1, r, 0))
            if u != v:
                half.append((v, u, f, -1, r, 0))
        else:
            half.append((u, v, f, 1, r, 1))
            half.append((v, u, f, -1, r, -1))
    return half


def enumerate_target(max_length: int):
    """Exact p over simple walks from each anchor, and the discarded (dead-end) mass."""
    by_node = defaultdict(list)
    for h in reference_half_edges():
        by_node[h[0]].append(h)
    walks, dead = {}, defaultdict(float)

    def extend(anchor, nodes, facts, prob, remaining):
        if remaining == 0:
            walks[(anchor, tuple(nodes), tuple(facts))] = walks.get(
                (anchor, tuple(nodes), tuple(facts)), 0.0) + prob
            return
        v = nodes[-1]
        options = defaultdict(list)
        for (_, other, f, _, r, d) in by_node[v]:
            if other not in nodes:
                options[(r, d)].append((other, f))
        if not options:
            dead[anchor] += prob
            return
        for edges in options.values():
            for other, f in edges:
                extend(anchor, nodes + [other], facts + [f],
                       prob / len(options) / len(edges), remaining - 1)

    for anchor in ANCHORS.tolist():
        for length in range(2, max_length + 1):
            extend(anchor, [anchor], [], 1.0 / (max_length - 1), length)
    return walks, dead


def walk_keys(draw: WispDraw):
    out = []
    for i in range(draw.length.numel()):
        length = int(draw.length[i])
        out.append((int(draw.anchor[i]), tuple(draw.nodes[i, :length + 1].tolist()),
                    tuple(draw.facts[i, :length].tolist())))
    return out


def dense_path_laplacian(groups, num_nodes: int) -> torch.Tensor:
    """Sum_f m_f L_f with the scalar cell Laplacian L_f = I - 11^T/|f| on the cell's nodes."""
    out = torch.zeros(num_nodes, num_nodes, dtype=torch.float64)
    for group in groups:
        size = group.face_node_index.size(1)
        block = torch.eye(size, dtype=torch.float64) - 1.0 / size
        for nodes, weight in zip(group.face_node_index.tolist(), group.path_weight.double().tolist()):
            index = torch.tensor(nodes)
            out[index.unsqueeze(1), index.unsqueeze(0)] += weight * block
    return out


def exact_path_laplacian(max_length: int) -> torch.Tensor:
    walks, _ = enumerate_target(max_length)
    out = torch.zeros(8, 8, dtype=torch.float64)
    for (_, nodes, _), prob in walks.items():
        size = len(nodes)
        index = torch.tensor(nodes)
        out[index.unsqueeze(1), index.unsqueeze(0)] += prob * (
            torch.eye(size, dtype=torch.float64) - 1.0 / size)
    return out


class WispSamplerTests(unittest.TestCase):
    MAX_LENGTH = 3

    def setUp(self) -> None:
        self.graph = build_action_graph(EDGE_INDEX, EDGE_TYPE, 8, SELF_INVERSE)

    def test_reference_fixture_has_the_required_structures(self) -> None:
        walks, dead = enumerate_target(self.MAX_LENGTH)
        self.assertGreater(sum(dead.values()), 0.0, "a dead end exists")
        self.assertGreater(len(walks), 50)
        self.assertEqual(self.graph.max_multiplicity, 2, "the parallel pa pair p2->a1")

    def test_TH1_exact_distribution_no_revisits_and_dead_end_mass(self) -> None:
        per_anchor = 50_000  # 4 anchors x 50k = 200k draws
        walks, dead = enumerate_target(self.MAX_LENGTH)
        draw = sample_wisp_walks(self.graph, NODE_TYPE, ANCHORS, per_anchor, self.MAX_LENGTH,
                                 torch.Generator().manual_seed(0))
        counts = defaultdict(int)
        for key in walk_keys(draw):
            nodes = key[1]
            self.assertEqual(len(set(nodes)), len(nodes), f"revisit in {key}")
            self.assertIn(key, walks, f"drawn walk not in the support: {key}")
            counts[key] += 1
        worst = 0.0
        for key, prob in walks.items():
            se = math.sqrt(prob * (1 - prob) / per_anchor)
            worst = max(worst, abs(counts[key] / per_anchor - prob) / se)
        self.assertLessEqual(worst, 4.0, f"max |freq - p| = {worst:.2f} s.e.")
        discarded = torch.bincount(draw.discarded_anchor, minlength=4).double() / per_anchor
        for anchor in ANCHORS.tolist():
            mass = dead[anchor]
            se = math.sqrt(max(mass * (1 - mass), 1e-12) / per_anchor)
            self.assertLessEqual(abs(float(discarded[anchor]) - mass), 4 * se + 1e-12,
                                 f"dead-end mass of anchor {anchor}")
        # exact log p of every drawn walk (float64) equals the enumeration
        for i, key in enumerate(walk_keys(draw)[:2000]):
            self.assertAlmostEqual(float(draw.log_p[i]), math.log(walks[key]), places=12)
        self.assertTrue(bool((draw.ratio == 1.0).all()), "q = p: every IS ratio is exactly 1")

    def test_TH2_unbiased_path_laplacian_for_q_equal_p(self) -> None:
        exact = exact_path_laplacian(self.MAX_LENGTH)
        draws = torch.stack([
            dense_path_laplacian(build_wisp_path_groups(
                EDGE_INDEX, EDGE_TYPE, NODE_TYPE, SELF_INVERSE, ANCHORS, 50, self.MAX_LENGTH,
                seed, graph=self.graph), 8)
            for seed in range(200)])
        mean = draws.mean(0)
        se = draws.std(0) / math.sqrt(draws.size(0))
        excess = (mean - exact).abs() - 4 * se
        self.assertLessEqual(float(excess.max()), 1e-6, f"worst entry beyond 4 s.e.: {excess.max()}")

    def test_TH4_identity_and_summed_merge(self) -> None:
        def draw_of(rows):
            n = len(rows)
            nodes = torch.full((n, 4), -1, dtype=torch.long)
            facts = torch.full((n, 3), -1, dtype=torch.long)
            signs, relations, directions = facts.clone(), facts.clone(), facts.clone()
            for i, (path, hops) in enumerate(rows):
                nodes[i, :len(path)] = torch.tensor(path)
                for j, (f, s) in enumerate(hops):
                    facts[i, j], signs[i, j] = f, s
                    relations[i, j] = FACTS[f][2]
                    directions[i, j] = 0 if FACTS[f][2] in SELF_INVERSE else s
            length = torch.tensor([len(h) for _, h in rows])
            ones = torch.ones(n, dtype=torch.float64)
            return WispDraw(nodes, facts, signs, relations, directions, length,
                            nodes[:, 0], ones.log(), ones.log(), ones, torch.zeros(0, dtype=torch.long))

        forward = ([0, 4, 1], [(0, 1), (1, -1)])   # p0 -pa-> a0 <-pa- p1
        backward = ([1, 4, 0], [(1, 1), (0, -1)])  # the reversal
        groups = merge_wisp_cells(draw_of([forward, backward]), NODE_TYPE, EDGE_INDEX, 4)
        self.assertEqual(groups[0].num_faces, 1, "a walk and its reversal are one cell")
        self.assertAlmostEqual(float(groups[0].path_weight[0]), 2 / 4, places=7)
        via_0_1 = ([0, 1, 4], [(11, 1), (1, 1)])   # p0 -cite(0->1)-> p1 -pa-> a0
        via_1_0 = ([0, 1, 4], [(12, -1), (1, 1)])  # p0 <-cite(1->0)- p1 -pa-> a0
        groups = merge_wisp_cells(draw_of([via_0_1, via_1_0]), NODE_TYPE, EDGE_INDEX, 4)
        self.assertEqual(groups[0].num_faces, 2, "different reciprocal facts, different cells")
        dropped = merge_wisp_cells(draw_of([forward, backward]), NODE_TYPE, EDGE_INDEX, 4,
                                   merge="drop_duplicates")
        self.assertAlmostEqual(float(dropped[0].path_weight[0]), 1 / 4, places=7,
                               msg="control: drop_duplicates keeps one weight")

    def test_TH9_determinism_and_draw_seeds(self) -> None:
        def draw(seed):
            return build_wisp_path_groups(EDGE_INDEX, EDGE_TYPE, NODE_TYPE, SELF_INVERSE, ANCHORS,
                                          20, self.MAX_LENGTH, seed, graph=self.graph)

        first, again, other = draw(42), draw(42), draw(43)
        self.assertEqual(len(first), len(again))
        for a, b in zip(first, again):
            for field in ("face_node_index", "face_edge_index", "relation_index", "path_weight",
                          "signature_index"):
                self.assertTrue(torch.equal(getattr(a, field), getattr(b, field)), field)
            self.assertEqual(a.signature_table, b.signature_table)
        differs = any(not torch.equal(a.path_weight, b.path_weight) or a.num_faces != b.num_faces
                      for a, b in zip(first, other)) or len(first) != len(other)
        self.assertTrue(differs, "a different seed gives a different draw")

    def test_TH9_evaluation_draw_ignores_model_and_ablation_settings(self) -> None:
        from types import SimpleNamespace

        from src.configs.config import TrainConfig
        from src.data.dataset import build_node_classification_paths

        def groups(extra):
            raw = {"data": {"dataset_name": "acm", "path_source": "wisp", "wisp_max_length": 3,
                            "wisp_walks_per_anchor": 20}, **extra}
            config = TrainConfig.from_dict(raw)
            return build_node_classification_paths(
                config=config, fact_layer=SimpleNamespace(self_inverse=SELF_INVERSE, mode="tied"),
                node_slices={"paper": (0, 4)}, node_type=NODE_TYPE, node_type_names=("paper", "a", "s"),
                edge_index=EDGE_INDEX, edge_type=EDGE_TYPE, relation_tuples=(), seed=config.seed)

        base = groups({})
        for extra in ({"model": {"encoder": "first_order"}}, {"model": {"geometry_context": False}},
                      {"ablation": {"no_path_cells": True}}, {"ablation": {"endpoint_edges": True}},
                      {"ablation": {"coupling": "symmetric"}}):
            with self.subTest(extra=extra):
                other = groups(extra)
                for a, b in zip(base, other):
                    self.assertTrue(torch.equal(a.face_node_index, b.face_node_index))
                    self.assertTrue(torch.equal(a.path_weight, b.path_weight))

    def test_wisp_config_rejections(self) -> None:
        from src.configs.config import TrainConfig, reject_never_tuned_settings

        for data in ({"path_source": "nope"}, {"path_source": "wisp", "wisp_max_length": 1},
                     {"path_source": "wisp", "wisp_walks_per_anchor": 0},
                     {"path_source": "wisp", "fact_layer": "untied"},
                     {"path_source": "wisp", "path_sampler": "uniform_instance"},
                     {"dataset_name": "amazon", "path_source": "wisp"}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                TrainConfig.from_dict({"data": {"dataset_name": "acm", **data}})
        for name in ("data.path_source", "data.wisp_max_length"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                reject_never_tuned_settings(None, {name: {"values": [1]}})
        reject_never_tuned_settings(None, {"data.wisp_walks_per_anchor": {"values": [16, 32]}})


if __name__ == "__main__":
    unittest.main()
