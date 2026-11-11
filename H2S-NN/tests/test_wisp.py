"""WISP (task card Part H): sampler tests T-H1, T-H2 (q = p), T-H4, T-H9 (H1), extended in H2-H4.

Fixture: node types p (anchors), a, s; relations pa (p->a), ps (p->s), cite (p->p, a reciprocal
pair), co (a-a, self-inverse, with a self-loop); a parallel pair (two pa facts p2->a1), a dead end
(s1, reached only from p3) and a hub (s0, four papers). The exact walk measure is enumerated by an
independent reference below (its own half-edge list, no sampler code)."""
from __future__ import annotations

import math
from pathlib import Path
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
                     {"path_source": "wisp", "wisp_walks_per_anchor": -1},
                     {"path_source": "wisp", "fact_layer": "untied"},
                     {"path_source": "wisp", "path_sampler": "uniform_instance"},
                     # spec v1.15 J1: the LP/GC length-2 walks were replaced by WISP
                     {"dataset_name": "lastfm", "path_source": "schemas"},
                     {"dataset_name": "mutag", "path_source": "schemas", "path_sampler": "walk"}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                TrainConfig.from_dict({"data": {"dataset_name": "acm", **data}})
        # Part I (I2): r = 0 means "the dataset's calibrated r" (spec v1.4 §6)
        self.assertEqual(TrainConfig.from_dict({"data": {
            "dataset_name": "acm", "path_source": "wisp",
            "wisp_walks_per_anchor": 0}}).data.wisp_walks_per_anchor, 42)
        with self.assertRaises(ValueError):
            reject_never_tuned_settings(None, {"data.path_source": {"values": [1]}})
        # spec v1.15 J2 (supersedes E11): l_max and r are tuned
        reject_never_tuned_settings(None, {"data.wisp_max_length": {"values": [2, 3, 4, 5]},
                                           "data.wisp_walks_per_anchor": {"values": [16, 32]}})



# ============================================================================= H2: weights
def _exp_system(seed: int):
    """test_cooperative.System with exp cell weights spanning e^-10 .. e^10 and no gamma."""
    from src.models.vertex_path_laplacian import PathCellTerm, VertexPathSheafLaplacian
    from tests.test_cooperative import System

    shared = seed % 2 == 0
    system = System(4000 + seed, shared=shared)
    generator = torch.Generator().manual_seed(seed)
    weights = []
    for nodes in system.paths.nodes:
        logits = torch.empty(nodes.size(0), dtype=torch.float64).uniform_(-10, 10,
                                                                          generator=generator)
        logits[0], logits[-1] = -10.0, 10.0
        weights.append(torch.exp(logits))
    system.weights = weights
    terms = tuple(PathCellTerm(t, w) for t, w in
                  zip(system.paths.transports(system.maps, shared), weights))
    system.laplacian = VertexPathSheafLaplacian(system.maps, system.paths.edge_index,
                                                system.num_nodes, terms, None, checkpoint=False)
    system.gamma = 1.0  # the dense reference multiplies L_P by gamma: 1 = no gamma
    return system


class WispWeightOperatorTests(unittest.TestCase):
    def test_TH5_props_2_3_5_with_exp_weights(self) -> None:
        from tests.test_cooperative import _random_gates
        from tests.test_filter_bank import _bank, _filter_matrix, _m_gain, _random_v

        generator = torch.Generator().manual_seed(7)
        worst_ref, worst_low, worst_high, worst_gain = 0.0, 0.0, 0.0, 0.0
        for seed in range(12):
            system = _exp_system(seed)
            production = system.production_delta_hat()
            dense = system.dense_delta_hat()
            worst_ref = max(worst_ref, float((production - dense).abs().max()))
            eigenvalues = torch.linalg.eigvalsh((production + production.T) / 2)
            worst_low = max(worst_low, float(-eigenvalues.min()))
            worst_high = max(worst_high, float(eigenvalues.max()) - 2.0)
            g, b = _random_gates(system)
            bank = _bank(system.d, 3, _random_v(3, system.d, generator))
            worst_gain = max(worst_gain,
                             _m_gain(_filter_matrix(system, g, b, bank), b / g, system.d))
        self.assertLessEqual(worst_ref, 1e-9, "production = dense reference with gamma absent")
        self.assertLessEqual(worst_low, 1e-9, "PSD")
        self.assertLessEqual(worst_high, 1e-9, "Prop. 3: lambda_max <= 2")
        self.assertLessEqual(worst_gain, 1.0 + 1e-9, "Prop. 5: filter gain <= 1")

    def test_TH5_kernel_preserved_with_exp_weights(self) -> None:
        """Prop. 2 on the D7 tree with sections: ker(Delta_0 + L_P) = ker Delta_0 for exp weights
        spanning e^-10 .. e^10 (production operator, gamma absent)."""
        from src.models.map_operators import PathTransportOperator
        from src.models.vertex_path_laplacian import PathCellTerm, VertexPathSheafLaplacian
        from tests.test_paths_and_facts import (SectionPreservationTests, _chain_operators,
                                                _moderate_scales)

        d, num_nodes = 3, 8
        for seed in range(3):
            for shared in (True, False):
                with self.subTest(seed=seed, shared=shared):
                    edge_index, groups = SectionPreservationTests()._tree(seed)
                    generator = torch.Generator().manual_seed(200 + seed)
                    maps = _moderate_scales(
                        _chain_operators(edge_index.size(1), k=2, d=d, seed=seed,
                                         dtype=torch.float64, shared=shared),
                        generator, shared)
                    terms = []
                    for nodes, edges, signs in groups:
                        logits = torch.empty(nodes.size(0), dtype=torch.float64).uniform_(
                            -10, 10, generator=generator)
                        logits[0], logits[-1] = -10.0, 10.0
                        transport = PathTransportOperator(maps.src, maps.tgt, nodes, edges,
                                                          edge_index, shared_scale=shared,
                                                          face_edge_sign=signs)
                        terms.append(PathCellTerm(transport, torch.exp(logits)))
                    full = VertexPathSheafLaplacian(maps, edge_index, num_nodes, tuple(terms),
                                                    None, checkpoint=False)
                    edges_only = VertexPathSheafLaplacian(maps, edge_index, num_nodes, (), None,
                                                          checkpoint=False)
                    basis = torch.eye(num_nodes * d, dtype=torch.float64).view(-1, num_nodes, d)
                    for name, lap in (("delta0", edges_only), ("full", full)):
                        matrix = torch.stack([lap.apply_normalized(e).reshape(-1)
                                              for e in basis], 1)
                        eigenvalues = torch.linalg.eigvalsh((matrix + matrix.T) / 2)
                        kernel = int((eigenvalues <= 1e-9).sum())
                        self.assertEqual(kernel, d, name)
                        self.assertGreater(float(eigenvalues[kernel]), 1e3 * 1e-9, name)


def _ladder_model(arm: str, geometry_context: bool = True, path_weight: str = "exp",
                  d: int = 4, path_weight_context: bool = False):
    from src.configs.config import TrainConfig
    from src.models.h2snn_model import H2SNN, ModelDims

    config = TrainConfig.from_dict({
        "device": "cpu",
        "model": {"d": d, "d_r": 2, "K": 2, "classifier_hidden_dim": 8, "face_mlp_hidden_dim": 8,
                  "edge_mlp_hidden_dim": 8, "geometry_context": geometry_context,
                  "path_weight": path_weight},
        "ablation": {"path_weight_input": arm, "path_weight_context": path_weight_context},
    })
    dims = ModelDims(in_channels=5, num_node_types=3, num_relations=4, num_classes=2,
                     face_lengths=(2, 3), self_inverse_relations=SELF_INVERSE, path_actions=True)
    torch.manual_seed(0)
    return H2SNN(config, dims).double().eval()


def _fixture_groups(max_length: int = 3):
    return build_wisp_path_groups(EDGE_INDEX, EDGE_TYPE, NODE_TYPE, SELF_INVERSE, ANCHORS, 30,
                                  max_length, 5)


ARMS = ("full", "endpoints", "schema", "uniform")


class WispLadderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.groups = _fixture_groups()
        self.group = next(g for g in self.groups if g.length == 3)

    def _weights(self, model, h0):
        """w_f of `self.group` through the production path (build_operator) from node states h0."""
        with torch.no_grad():
            operator = model.build_operator(h0, EDGE_INDEX, EDGE_TYPE, self.groups, 8)
        nonempty = [g for g in self.groups if g.num_faces]
        return operator.path_weights[nonempty.index(self.group)]

    def _weights_of(self, model, h0, group):
        self.group, previous = group, self.group
        try:
            return self._weights(model, h0)
        finally:
            self.group = previous

    @staticmethod
    def _h0(model, seed):
        return torch.randn(8, model.d, dtype=torch.float64,
                           generator=torch.Generator().manual_seed(seed))

    @staticmethod
    def _outside_neighbours(of, cell):
        """Neighbours of the nodes `of` that are not nodes of the cell."""
        of, cell = set(of), set(cell)
        out = set()
        for u, v in EDGE_INDEX.t().tolist():
            if u in of and v not in cell:
                out.add(v)
            if v in of and u not in cell:
                out.add(u)
        return out

    def test_TH6_exp_has_no_gamma_and_b_is_a_scalar_parameter(self) -> None:
        from src.builders import build_parameter_groups
        from src.configs.config import TrainConfig

        for arm in ARMS:
            with self.subTest(arm=arm):
                model = _ladder_model(arm)
                self.assertIsNone(model.theta_gamma)
                self.assertFalse(any("theta_gamma" in n for n, _ in model.named_parameters()))
                self.assertIsNone(model.gamma())
                b = model.path_gate.bias if model.path_gate is not None else model.path_log_weight
                cfg = TrainConfig.from_dict({"device": "cpu"})
                scalar = [g for g in build_parameter_groups(model, cfg) if g["weight_decay"] == 0.0]
                self.assertTrue(any(p is b for g in scalar for p in g["params"]),
                                "b in the scalar group")
                # created in float32 at construction (then cast): exactly log(0.5) in float32
                self.assertEqual(float(b.detach().reshape(-1)[0]),
                                 float(torch.tensor(math.log(0.5), dtype=torch.float32)))
        sigmoid = _ladder_model("full", path_weight="sigmoid_gamma")
        self.assertIsNotNone(sigmoid.theta_gamma)
        with self.assertRaises(ValueError):  # a gamma_init that exp would ignore is rejected
            TrainConfig.from_dict({"model": {"gamma_init": 2.0}})
        TrainConfig.from_dict({"model": {"gamma_init": 1.0}})  # the saved default round-trips
        with self.assertRaises(ValueError):
            TrainConfig.from_dict({"model": {"path_weight": "sigmoid_gamma",
                                             "path_weight_init": 0.3}})

    def test_saved_configs_round_trip_under_both_path_weights(self) -> None:
        import tempfile

        import yaml

        from src.configs.config import TrainConfig
        from src.utils.misc import save_config

        for model in ({}, {"path_weight": "sigmoid_gamma"}, {"path_weight_init": 0.3}):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                config = TrainConfig.from_dict({"device": "cpu", "model": model,
                                                "data": {"path_source": "wisp"}})
                path = Path(tmp) / "config.yaml"
                save_config(config, path)
                reloaded = TrainConfig.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
                self.assertEqual(reloaded.model, config.model)
                self.assertEqual(reloaded.data.path_source, "wisp")

    def test_TH6_exp_operator_has_no_gamma_multiplier(self) -> None:
        """The path part of D enters with exactly sum w_f m_f (1 - 1/|f|) s^2: no gamma, no constant."""
        system = _exp_system(3)
        self.assertIsNone(system.laplacian.gamma)
        path = sum(t.diagonal(system.num_nodes) for t in system.laplacian.path_terms)
        self.assertTrue(torch.equal(system.laplacian.degree(),
                                    system.laplacian.edge_degree() + path))

    def test_TH7_intermediate_and_its_neighbourhood_under_the_default_context(self) -> None:
        """Spec v1.4 (R12), under geometry_context: true (the default): perturbing an
        intermediate's h0 and the h0 of all its neighbours outside the instance leaves the
        endpoints, schema and uniform weights unchanged and changes full's."""
        cell = 0
        nodes = self.group.face_node_index[cell].tolist()
        for position in range(1, len(nodes) - 1):
            moved_nodes = {nodes[position]} | self._outside_neighbours([nodes[position]], nodes)
            for arm in ARMS:
                with self.subTest(position=position, arm=arm):
                    model = _ladder_model(arm)
                    self.assertIsNotNone(model.relational_context, "the default context is on")
                    h0 = self._h0(model, 2)
                    moved = h0.clone()
                    moved[sorted(moved_nodes)] += 1.0
                    before = self._weights(model, h0)[cell]
                    after = self._weights(model, moved)[cell]
                    if arm == "full":
                        self.assertNotEqual(float(before), float(after), "full sees the intermediate")
                    else:
                        self.assertEqual(float(before), float(after), f"{arm} must not see it")

    def test_TH7_a_node_outside_the_instance_reaches_no_arm_but_the_appendix(self) -> None:
        """g reads instance nodes only: a neighbour of the endpoints outside the cell changes no
        arm's weight. Control (spec v1.4 appendix pair, path_weight_context: true): it reaches
        full and endpoints through the endpoints' context c."""
        cell = 0
        nodes = self.group.face_node_index[cell].tolist()
        outside = sorted(self._outside_neighbours([nodes[0], nodes[-1]], nodes))
        self.assertTrue(outside, "fixture: an endpoint has a neighbour outside the cell")
        for arm in ARMS:
            for appendix in ((False, True) if arm in ("full", "endpoints") else (False,)):
                with self.subTest(arm=arm, path_weight_context=appendix):
                    model = _ladder_model(arm, path_weight_context=appendix)
                    h0 = self._h0(model, 5)
                    moved = h0.clone()
                    moved[outside] += 1.0
                    changed = float(self._weights(model, h0)[cell]) != float(
                        self._weights(model, moved)[cell])
                    self.assertEqual(changed, appendix, (arm, appendix))

    def test_TH7_positions_l_minus_1_and_l_separately(self) -> None:
        """Review MINOR-6: v_{l-1} (an intermediate) must not reach endpoints, v_l must; full sees
        both; schema and uniform see neither. On length-2 and length-3 cells."""
        for group in self.groups:
            if group.num_faces == 0:
                continue
            nodes = group.face_node_index[0].tolist()
            length = len(nodes) - 1
            for position, endpoints_sees in ((length - 1, False), (length, True)):
                for arm in ARMS:
                    with self.subTest(length=length, position=position, arm=arm):
                        model = _ladder_model(arm)
                        h0 = self._h0(model, 7)
                        moved = h0.clone()
                        moved[nodes[position]] += 1.0
                        changed = float(self._weights_of(model, h0, group)[0]) != float(
                            self._weights_of(model, moved, group)[0])
                        expected = {"full": True, "endpoints": endpoints_sees,
                                    "schema": False, "uniform": False}[arm]
                        self.assertEqual(changed, expected)

    def test_TH7_endpoint_perturbation_and_schema_equality(self) -> None:
        cell = 0
        start = int(self.group.face_node_index[cell, 0])
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = _ladder_model(arm)
                h0 = self._h0(model, 3)
                before = self._weights(model, h0)
                moved = h0.clone()
                moved[start] += 1.0
                after = self._weights(model, moved)
                changed = float(before[cell]) != float(after[cell])
                self.assertEqual(changed, arm in ("full", "endpoints"), arm)
                if arm == "schema":
                    signature = self.group.signature_index
                    nodes = self.group.face_node_index
                    same = [(i, j) for i in range(signature.numel())
                            for j in range(i + 1, signature.numel())
                            if int(signature[i]) == int(signature[j])
                            and not torch.equal(nodes[i], nodes[j])]
                    self.assertTrue(same, "fixture has two cells with one relation sequence")
                    for i, j in same:
                        self.assertEqual(float(before[i]), float(before[j]))

    def test_TH7_intermediate_reaches_endpoints_only_under_the_appendix_switch(self) -> None:
        """The v1.3 behaviour is the appendix pair: with path_weight_context: true an
        intermediate's h0 reaches the endpoints weight through c; with the context off entirely
        (geometry_context: false) g reads h0 either way and it does not."""
        cell = 0
        middle = int(self.group.face_node_index[cell, 1])
        for context, appendix, expect_change in ((True, True, True), (True, False, False),
                                                 (False, False, False)):
            with self.subTest(geometry_context=context, path_weight_context=appendix):
                model = _ladder_model("endpoints", geometry_context=context,
                                      path_weight_context=appendix)
                self.assertEqual(model.path_input_dim,
                                 model.geometry_dim if appendix else model.d)
                h0 = self._h0(model, 4)
                moved = h0.clone()
                moved[middle] += 1.0
                w = [self._weights(model, h)[cell] for h in (h0, moved)]
                self.assertEqual(float(w[0]) != float(w[1]), expect_change)

    def test_TH7_capacity_rule_endpoints_vs_full_within_5_percent(self) -> None:
        def g_params(model):
            modules = [m for m in (model.path_encoder, model.path_gate) if m is not None]
            return sum(p.numel() for m in modules for p in m.parameters())

        for context, appendix in ((True, False), (False, False), (True, True)):
            full = g_params(_ladder_model("full", geometry_context=context,
                                          path_weight_context=appendix))
            endpoints = g_params(_ladder_model("endpoints", geometry_context=context,
                                               path_weight_context=appendix))
            self.assertLessEqual(abs(endpoints - full) / full, 0.05,
                                 (context, appendix, full, endpoints))

    def test_path_weight_context_is_an_appendix_arm_never_tuned(self) -> None:
        from src.configs.config import TrainConfig, reject_never_tuned_settings

        bad = [({"model": {"geometry_context": False}}, {"path_weight_context": True}),
               ({}, {"path_weight_context": True, "path_weight_input": "schema"}),
               ({}, {"path_weight_context": True, "path_weight_input": "uniform"}),
               ({}, {"path_weight_context": True, "no_path_cells": True}),
               ({}, {"path_weight_context": "true"})]
        for extra, ablation in bad:
            with self.subTest(extra=extra, ablation=ablation), self.assertRaises(ValueError):
                TrainConfig.from_dict({**extra, "ablation": ablation})
        for arm in ("full", "endpoints"):  # the appendix pair loads
            TrainConfig.from_dict({"ablation": {"path_weight_context": True,
                                                "path_weight_input": arm}})
        with self.assertRaises(ValueError):
            reject_never_tuned_settings({"ablation": {"path_weight_context": True}})
        for name in ("ablation.path_weight_context", "path_weight_context"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                reject_never_tuned_settings({}, [name])
        reject_never_tuned_settings({"ablation": {"path_weight_context": False}})

    def test_TH8_length_neutral_initialisation(self) -> None:
        """Constant g (`uniform`): the expected total cell weight per anchor is equal across
        lengths on a fixture without dead ends (complete bipartite typed graph)."""
        left, right = torch.arange(5), torch.arange(5, 10)
        edge_index = torch.stack([left.repeat_interleave(5), right.repeat(5)])
        edge_type = torch.zeros(25, dtype=torch.long)
        node_type = torch.tensor([0] * 5 + [1] * 5)
        graph = build_action_graph(edge_index, edge_type, 10, ())
        model = _ladder_model("uniform")
        totals = []
        for seed in range(100):
            groups = build_wisp_path_groups(edge_index, edge_type, node_type, (), left, 40, 3,
                                            seed, graph=graph)
            row = {}
            for group in groups:
                w = model._path_weights(None, group, torch.zeros(1, dtype=torch.float64))
                row[group.length] = float((w * group.path_weight.double()).sum()) / 5
            totals.append([row.get(2, 0.0), row.get(3, 0.0)])
        totals = torch.tensor(totals, dtype=torch.float64)
        mean, se = totals.mean(0), totals.std(0) / math.sqrt(totals.size(0))
        self.assertLessEqual(float((mean[0] - mean[1]).abs()), 4 * float(se.norm()))
        expected = 0.5 / 2  # w = exp(b) = 0.5, length uniform over {2, 3}
        for k in range(2):
            self.assertLessEqual(abs(float(mean[k]) - expected), 4 * float(se[k]) + 1e-12)



# ============================================================================= I2: default flip
class WispDefaultTests(unittest.TestCase):
    """Spec v1.4 E7 (Part I, I2): node classification defaults to wisp with the dataset's
    incidence-matching r. Spec v1.15 J1 (Part M, M4): link prediction and graph classification
    default to wisp too, with r unset (no calibrated table; a draw needs it set)."""

    def test_default_path_source_and_r_per_dataset(self) -> None:
        from src.configs.config import TrainConfig

        for dataset, r in (("acm", 42), ("dblp", 65), ("imdb", 12)):
            with self.subTest(dataset=dataset):
                data = TrainConfig.from_dict({"data": {"dataset_name": dataset}}).data
                self.assertEqual((data.path_source, data.wisp_walks_per_anchor), ("wisp", r))
        for dataset in ("amazonbook", "lastfm", "mutag", "nci1"):
            with self.subTest(dataset=dataset):
                data = TrainConfig.from_dict({"data": {"dataset_name": dataset}}).data
                self.assertNotEqual(data.task_type, "node_classification")
                self.assertEqual((data.path_source, data.wisp_walks_per_anchor), ("wisp", 0))

    def test_explicit_values_are_kept(self) -> None:
        from src.configs.config import TrainConfig

        schemas = TrainConfig.from_dict({"data": {"dataset_name": "dblp",
                                                  "path_source": "schemas"}}).data
        self.assertEqual(schemas.path_source, "schemas")
        self.assertEqual([t.name for t in schemas.metapath_templates], ["APA", "APTPA"])
        explicit_r = TrainConfig.from_dict({"data": {"dataset_name": "imdb",
                                                     "wisp_walks_per_anchor": 30}}).data
        self.assertEqual(explicit_r.wisp_walks_per_anchor, 30)
        gc = TrainConfig.from_dict({"data": {"dataset_name": "mutag", "path_source": "wisp",
                                             "wisp_walks_per_anchor": 16}}).data
        self.assertEqual((gc.path_source, gc.wisp_walks_per_anchor), ("wisp", 16))
        with self.assertRaises(ValueError):  # spec v1.15 J1: the GC length-2 walk is gone
            TrainConfig.from_dict({"data": {"dataset_name": "mutag", "path_source": "schemas"}})
        with self.assertRaises(ValueError):  # an untied row must name the schema source
            TrainConfig.from_dict({"data": {"dataset_name": "acm", "fact_layer": "untied",
                                            "path_sampler": "dfs", "resample_paths": False}})

    def test_resolved_configs_round_trip(self) -> None:
        import tempfile

        import yaml

        from src.configs.config import TrainConfig
        from src.utils.misc import save_config

        for data in ({"dataset_name": "dblp"}, {"dataset_name": "acm", "path_source": "schemas"},
                     {"dataset_name": "mutag"}):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as tmp:
                config = TrainConfig.from_dict({"device": "cpu", "data": data})
                path = Path(tmp) / "config.yaml"
                save_config(config, path)
                reloaded = TrainConfig.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
                self.assertEqual(reloaded.data, config.data)

    def test_schemas_row_is_never_tuned_for_node_classification(self) -> None:
        from src.configs.config import reject_never_tuned_settings

        with self.assertRaises(ValueError):
            reject_never_tuned_settings({"data": {"dataset_name": "acm", "path_source": "schemas"}})
        reject_never_tuned_settings({"data": {"dataset_name": "acm"}})
        reject_never_tuned_settings({"data": {"dataset_name": "amazonbook",
                                              "path_source": "schemas"}})


# ============================================================================= H3: proposal
def _biased_proposal(max_length: int, near_degenerate: bool = False, seed: int = 0):
    """A strongly biased q: length mass mostly on the longest length; per-(type, action) weights
    spanning orders of magnitude, with zeros (no floors) and, if `near_degenerate`, 1e-300."""
    from src.data.wisp_sampler import Proposal, count_actions

    generator = torch.Generator().manual_seed(seed)
    length = torch.zeros(max_length + 1, dtype=torch.float64)
    length[2:] = 0.05
    length[max_length] = 1.0
    length = length / length[2:].sum()
    num_actions = count_actions(EDGE_TYPE, SELF_INVERSE)
    action = torch.rand(3, num_actions, generator=generator, dtype=torch.float64) ** 4
    action[action < 0.05] = 0.0
    if near_degenerate:
        action = torch.where(action > 0.5, torch.ones_like(action),
                             torch.full_like(action, 1e-300))
    return Proposal(length=length, action=action)


def enumerate_with_proposal(max_length: int, proposal):
    """Reference: for every simple walk, (anchor, nodes, facts) -> (p, q, per-step mixture), with
    q_act renormalised over the available actions (independent of the sampler)."""
    by_node = defaultdict(list)
    for h in reference_half_edges():
        by_node[h[0]].append(h)
    types = NODE_TYPE.tolist()
    out = {}

    def extend(anchor, nodes, facts, p, q, mix, remaining):
        if remaining == 0:
            out[(anchor, tuple(nodes), tuple(facts))] = (p, q, mix)
            return
        v = nodes[-1]
        options = defaultdict(list)
        for (_, other, f, _, r, d) in by_node[v]:
            if other not in nodes:
                options[(r, d)].append((other, f))
        if not options:
            return
        weights = {a: float(proposal.action[types[v], a[0] * 3 + a[1] + 1]) for a in options}
        total = sum(weights.values())
        for a, edges in options.items():
            p_a = 1.0 / len(options)
            q_a = weights[a] / total if total > 0 else 0.0
            for other, f in edges:
                p_step, q_step = p_a / len(edges), q_a / len(edges)
                extend(anchor, nodes + [other], facts + [f], p * p_step, q * q_step,
                       mix * (0.5 * p_step + 0.5 * q_step), remaining - 1)

    for anchor in ANCHORS.tolist():
        for length in range(2, max_length + 1):
            p_len = 1.0 / (max_length - 1)
            q_len = float(proposal.length[length])
            extend(anchor, [anchor], [], p_len, q_len, 0.5 * p_len + 0.5 * q_len, length)
    return out


class WispProposalTests(unittest.TestCase):
    MAX_LENGTH = 3

    def setUp(self) -> None:
        self.graph = build_action_graph(EDGE_INDEX, EDGE_TYPE, 8, SELF_INVERSE)

    def _draws(self, proposal, seeds, per_anchor=50):
        return [sample_wisp_walks(self.graph, NODE_TYPE, ANCHORS, per_anchor, self.MAX_LENGTH,
                                  torch.Generator().manual_seed(seed), proposal) for seed in seeds]

    def test_TH2_unbiased_under_a_strongly_biased_q_with_control(self) -> None:
        import dataclasses

        exact = exact_path_laplacian(self.MAX_LENGTH)
        proposal = _biased_proposal(self.MAX_LENGTH)
        draws = self._draws(proposal, range(300))
        weighted, unweighted = [], []
        for draw in draws:
            weighted.append(dense_path_laplacian(
                merge_wisp_cells(draw, NODE_TYPE, EDGE_INDEX, 50), 8))
            ones = dataclasses.replace(draw, ratio=torch.ones_like(draw.ratio))
            unweighted.append(dense_path_laplacian(
                merge_wisp_cells(ones, NODE_TYPE, EDGE_INDEX, 50), 8))
        for name, stack, should_pass in (("IS", weighted, True), ("no IS ratio", unweighted, False)):
            stack = torch.stack(stack)
            mean = stack.mean(0)
            se = stack.std(0) / math.sqrt(stack.size(0))
            excess = float(((mean - exact).abs() - 4 * se).max())
            if should_pass:
                self.assertLessEqual(excess, 1e-6, f"{name}: worst entry beyond 4 s.e.")
            else:
                self.assertGreater(excess, 0.0, "control: without the IS ratio the mean is biased")

    def test_TH3_ratio_is_walk_level_and_at_most_two_with_per_step_control(self) -> None:
        worst_control = 0.0
        for case, proposal in (("random", _biased_proposal(self.MAX_LENGTH, seed=1)),
                               ("random2", _biased_proposal(self.MAX_LENGTH, seed=2)),
                               ("near_degenerate", _biased_proposal(self.MAX_LENGTH, True, seed=3))):
            with self.subTest(case=case):
                reference = enumerate_with_proposal(self.MAX_LENGTH, proposal)
                draw = self._draws(proposal, [11], per_anchor=20_000)[0]
                self.assertTrue(bool((draw.ratio <= 2.0).all()), "every ratio <= 2 (exact)")
                self.assertTrue(bool(torch.isfinite(draw.ratio).all()))
                for i, key in enumerate(walk_keys(draw)):
                    p, q, _ = reference[key]
                    expected = p / (0.5 * p + 0.5 * q)
                    self.assertLessEqual(abs(float(draw.ratio[i]) - expected), 1e-12 * expected,
                                         key)
                worst_control = max(worst_control,
                                    max(p / mix for p, _, mix in reference.values()))
        self.assertGreater(worst_control, 2.0, "control: a per-step mixture exceeds 2")

    def test_proposal_from_mass_tables(self) -> None:
        from src.data.wisp_sampler import count_actions, proposal_from_mass

        groups = _fixture_groups()
        weights = tuple(torch.full((g.num_faces,), 2.0, dtype=torch.float64) for g in groups)
        q = proposal_from_mass(groups, weights, NODE_TYPE, 3, count_actions(EDGE_TYPE, SELF_INVERSE),
                               3, SELF_INVERSE)
        self.assertAlmostEqual(float(q.length[2:].sum()), 1.0, places=12)
        mass = {g.length: float(2.0 * g.path_weight.double().sum()) for g in groups}
        for length, value in mass.items():
            self.assertAlmostEqual(float(q.length[length]), value / sum(mass.values()), places=12)
        # every cell counted once per hop in each orientation
        total_action_mass = sum(2 * g.length * 2.0 * float(g.path_weight.double().sum())
                                for g in groups)
        self.assertAlmostEqual(float(q.action.sum()), total_action_mass, places=9)

    def test_TH9_evaluation_draw_identical_across_proposals_and_ladder_arms(self) -> None:
        from types import SimpleNamespace

        from src.configs.config import TrainConfig
        from src.data.dataset import build_node_classification_paths

        def groups(data_extra, ablation):
            raw = {"data": {"dataset_name": "acm", "path_source": "wisp", "wisp_max_length": 3,
                            "wisp_walks_per_anchor": 20, **data_extra},
                   "ablation": ablation}
            config = TrainConfig.from_dict(raw)
            return build_node_classification_paths(
                config=config, fact_layer=SimpleNamespace(self_inverse=SELF_INVERSE, mode="tied"),
                node_slices={"paper": (0, 4)}, node_type=NODE_TYPE,
                node_type_names=("paper", "a", "s"), edge_index=EDGE_INDEX, edge_type=EDGE_TYPE,
                relation_tuples=(), seed=config.seed)

        base = groups({}, {})
        for data_extra in ({"wisp_proposal": "adaptive"}, {}):
            for arm in ARMS:
                with self.subTest(data=data_extra, arm=arm):
                    other = groups(data_extra, {"path_weight_input": arm})
                    self.assertEqual(len(base), len(other))
                    for a, b in zip(base, other):
                        self.assertTrue(torch.equal(a.face_node_index, b.face_node_index))
                        self.assertTrue(torch.equal(a.path_weight, b.path_weight))




# ============================================================================= H4: read-outs
def _fixture_transports(groups, shared: bool = True):
    from src.models.map_operators import PathTransportOperator
    from tests.test_v2_model import _chain_operators

    maps = _chain_operators(EDGE_INDEX.size(1), k=2, d=3, seed=0, dtype=torch.float64,
                            shared=shared)
    return maps, [PathTransportOperator(maps.src, maps.tgt, g.face_node_index, g.face_edge_index,
                                        EDGE_INDEX, shared_scale=shared,
                                        face_edge_sign=g.face_edge_sign.double()) for g in groups]


class WispReadoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.groups = _fixture_groups()
        generator = torch.Generator().manual_seed(9)
        self.weights = [torch.rand(g.num_faces, generator=generator, dtype=torch.float64) + 0.1
                        for g in self.groups]

    def test_TH11_shares_sum_to_one(self) -> None:
        from src.analysis.wisp_readouts import i_pi, i_sigma, i_v_sigma, overlap_ratio

        pi = i_pi(self.groups, self.weights, NODE_TYPE, SELF_INVERSE)
        sigma = i_sigma(self.groups, self.weights, NODE_TYPE, SELF_INVERSE)
        self.assertLessEqual(abs(sum(pi.values()) - 1.0), 1e-10)
        self.assertLessEqual(abs(sum(sigma.values()) - 1.0), 1e-10)
        for v, table in i_v_sigma(self.groups, self.weights, NODE_TYPE, SELF_INVERSE).items():
            self.assertLessEqual(abs(sum(table.values()) - 1.0), 1e-10, v)
        self.assertGreater(len(sigma), len(pi), "sub-path schemas outnumber full schemas")
        ratio = overlap_ratio(self.groups, self.weights, NODE_TYPE, SELF_INVERSE)
        self.assertTrue(0.0 < ratio < 1.0, ratio)

    def test_TH11_pair_decomposition_reproduces_each_cell_energy(self) -> None:
        """Prop. K: 1/2 x^T L_f x (production PathCellTerm, one cell at a time) equals
        sum_{i<j} w_f / (2|f|) ||T_i x_i - T_j x_j||^2; the pair-form term with every pair kept
        equals the production term (apply and diagonal)."""
        from src.analysis.wisp_readouts import PairCellTerm, cell_energy_pairs
        from src.models.vertex_path_laplacian import PathCellTerm

        x = torch.randn(8, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
        maps, transports = _fixture_transports(self.groups)
        for group, transport, w in zip(self.groups, transports, self.weights):
            pairs = cell_energy_pairs(transport, w, x)
            for f in range(group.num_faces):
                from src.models.map_operators import PathTransportOperator

                one = PathTransportOperator(maps.src, maps.tgt,
                                            group.face_node_index[f:f + 1],
                                            group.face_edge_index[f:f + 1], EDGE_INDEX,
                                            shared_scale=True,
                                            face_edge_sign=group.face_edge_sign[f:f + 1].double())
                energy = 0.5 * float((x * PathCellTerm(one, w[f:f + 1]).apply(x, 8)).sum())
                self.assertLessEqual(abs(energy - float(pairs[f])), 1e-10 * max(1.0, abs(energy)))
            size = group.face_node_index.size(1)
            keep = torch.ones(group.num_faces, size, size, dtype=torch.float64) - torch.eye(
                size, dtype=torch.float64)
            pair_term = PairCellTerm(transport, w, keep)
            production = PathCellTerm(transport, w)
            self.assertLessEqual(float((pair_term.apply(x, 8) - production.apply(x, 8)).abs().max()),
                                 1e-12)
            self.assertLessEqual(float((pair_term.diagonal(8) - production.diagonal(8)).abs().max()),
                                 1e-12)

    def test_TH11_partial_pair_form_matches_its_kept_pair_energy(self) -> None:
        """With a random symmetric pair mask, 1/2 x^T L_keep x equals the kept pairs' energy
        sum_{kept i<j} w_f / (2|f|) ||T_i x_i - T_j x_j||^2, and the diagonal is
        sum_j keep_ij (w_f / |f|) s_i^2 (the form `delta_loss` uses when it drops a schema)."""
        from src.analysis.wisp_readouts import PairCellTerm

        x = torch.randn(8, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        generator = torch.Generator().manual_seed(4)
        _, transports = _fixture_transports(self.groups)
        for group, transport, w in zip(self.groups, transports, self.weights):
            size = group.face_node_index.size(1)
            upper = (torch.rand(group.num_faces, size, size, generator=generator) < 0.5).double()
            upper = torch.triu(upper, diagonal=1)
            keep = upper + upper.transpose(1, 2)
            self.assertGreater(float(keep.sum()), 0.0)
            y = transport.apply(x)
            expected = 0.0
            for i in range(size):
                for j in range(i + 1, size):
                    expected += float((keep[:, i, j] * w / (2 * size)
                                       * (y[:, i] - y[:, j]).square().sum(-1)).sum())
            term = PairCellTerm(transport, w, keep)
            energy = 0.5 * float((x * term.apply(x, 8)).sum())
            self.assertLessEqual(abs(energy - expected), 1e-10 * max(1.0, abs(expected)))
            per = (w / size).unsqueeze(1) * keep.sum(2) * transport.squared_scales()
            diagonal = torch.zeros(8, dtype=torch.float64).index_add_(
                0, group.face_node_index.reshape(-1), per.reshape(-1))
            self.assertLessEqual(float((term.diagonal(8) - diagonal).abs().max()), 1e-12)

    def test_TH11_schema_mask_drops_exactly_that_schema_in_both_orders(self) -> None:
        from src.analysis.wisp_readouts import pair_index, schema_keep

        for group in self.groups:
            pairs = pair_index(group, NODE_TYPE, SELF_INVERSE)
            size = group.face_node_index.size(1)
            for schema in sorted(set(pairs.schema))[:4]:
                keep = schema_keep(group, NODE_TYPE, SELF_INVERSE, schema)
                self.assertTrue(torch.equal(keep, keep.transpose(1, 2)), "symmetric")
                self.assertEqual(float(keep.diagonal(dim1=1, dim2=2).abs().sum()), 0.0)
                for c, i, j, key in zip(pairs.cell.tolist(), pairs.first.tolist(),
                                        pairs.second.tolist(), pairs.schema):
                    self.assertEqual(float(keep[c, i, j]), 0.0 if key == schema else 1.0)
            full = schema_keep(group, NODE_TYPE, SELF_INVERSE, None)
            self.assertEqual(float(full.sum()), group.num_faces * size * (size - 1))

    def test_TH11_delta_loss_of_an_absent_schema_is_zero(self) -> None:
        from src.analysis.wisp_readouts import delta_loss, pair_index

        model = _ladder_model("full")
        x = torch.randn(8, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
        labels = torch.tensor([0, 1, 1, 0])
        args = (model, x, NODE_TYPE, EDGE_INDEX, EDGE_TYPE, 8, [self.groups], ANCHORS, labels)
        self.assertEqual(delta_loss(*args, schema=(99, 99, 99), self_inverse=SELF_INVERSE), 0.0)
        present = pair_index(self.groups[0], NODE_TYPE, SELF_INVERSE).schema[0]
        self.assertNotEqual(delta_loss(*args, schema=present, self_inverse=SELF_INVERSE), 0.0,
                            "control: removing a present schema changes the loss")
        self.assertFalse(model.training, "the frozen model's mode is restored")

    def test_kendall_tau_and_permutation_null(self) -> None:
        from src.analysis.wisp_readouts import kendall_tau

        a = {i: float(i) for i in range(12)}
        tau, p = kendall_tau(a, a, permutations=200)
        self.assertAlmostEqual(tau, 1.0, places=12)
        self.assertLess(p, 0.05)
        tau, p = kendall_tau(a, {i: -float(i) for i in range(12)}, permutations=200)
        self.assertAlmostEqual(tau, -1.0, places=12)
        self.assertGreater(p, 0.9)

    def test_readouts_import_no_training_code(self) -> None:
        import subprocess
        import sys

        code = ("import sys; import src.analysis.wisp_readouts; "
                "print(sorted(m for m in sys.modules if m.startswith(('src.trainers', 'src.builders'))))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent.parent, check=True)
        self.assertEqual(out.stdout.strip(), "[]")



# ============================================================================= I5: review gaps
def _transported_maps():
    from tests.test_v2_model import _chain_operators

    return _chain_operators(EDGE_INDEX.size(1), k=2, d=3, seed=0, dtype=torch.float64,
                            shared=True)


def _dense_transported(maps, nodes, facts, signs, weights, d=3):
    """Dense [8d, 8d] sum_f w_f B_f^T (I - 11^T/|f| (x) I_d) B_f of one length's cells, through
    the production PathCellTerm (O(L) form) applied to the 8d basis vectors."""
    from src.models.map_operators import PathTransportOperator
    from src.models.vertex_path_laplacian import PathCellTerm

    transport = PathTransportOperator(maps.src, maps.tgt, nodes, facts, EDGE_INDEX,
                                      shared_scale=True, face_edge_sign=signs.double())
    term = PathCellTerm(transport, weights.double())
    basis = torch.eye(8 * d, dtype=torch.float64).reshape(8 * d, 8, d)
    return torch.stack([term.apply(e, 8).reshape(-1) for e in basis], dim=1)


def dense_transported_laplacian(groups, maps) -> torch.Tensor:
    out = torch.zeros(24, 24, dtype=torch.float64)
    for g in groups:
        if g.num_faces:
            out += _dense_transported(maps, g.face_node_index, g.face_edge_index,
                                      g.face_edge_sign, g.path_weight)
    return out


def exact_transported_laplacian(max_length: int, maps) -> torch.Tensor:
    """sum_omega p(omega) L_f(omega) with the transports of the walk's own facts and signs
    (independent enumeration; a hop's sign is +1 when the stored fact points along the walk)."""
    walks, _ = enumerate_target(max_length)
    by_length = defaultdict(list)
    for (_, nodes, facts), prob in walks.items():
        signs = [1 if (FACTS[f][0], FACTS[f][1]) == (nodes[j], nodes[j + 1]) else -1
                 for j, f in enumerate(facts)]
        by_length[len(facts)].append((nodes, facts, signs, prob))
    out = torch.zeros(24, 24, dtype=torch.float64)
    for rows in by_length.values():
        out += _dense_transported(maps, torch.tensor([r[0] for r in rows]),
                                  torch.tensor([r[1] for r in rows]),
                                  torch.tensor([r[2] for r in rows]),
                                  torch.tensor([r[3] for r in rows], dtype=torch.float64))
    return out


class WispTransportedUnbiasednessTests(unittest.TestCase):
    """T-H2 on the transported operator (review MINOR-10): E[L_P_hat] = L_P^target with each
    cell's own transports, for q = p and a strongly biased q; control: no IS ratio fails."""

    MAX_LENGTH = 3

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = build_action_graph(EDGE_INDEX, EDGE_TYPE, 8, SELF_INVERSE)
        cls.maps = _transported_maps()
        cls.exact = exact_transported_laplacian(cls.MAX_LENGTH, cls.maps)

    def _excess(self, matrices):
        stack = torch.stack(matrices)
        mean, se = stack.mean(0), stack.std(0) / math.sqrt(stack.size(0))
        return float(((mean - self.exact).abs() - 4 * se).max())

    def test_transported_TH2_for_q_equal_p(self) -> None:
        self.assertGreater(float(self.exact.abs().max()), 1e-3)
        draws = [dense_transported_laplacian(build_wisp_path_groups(
            EDGE_INDEX, EDGE_TYPE, NODE_TYPE, SELF_INVERSE, ANCHORS, 50, self.MAX_LENGTH, seed,
            graph=self.graph), self.maps) for seed in range(200)]
        self.assertLessEqual(self._excess(draws), 1e-6)

    def test_transported_TH2_under_a_biased_q_with_control(self) -> None:
        import dataclasses

        proposal = _biased_proposal(self.MAX_LENGTH)
        weighted, unweighted = [], []
        for seed in range(300):
            draw = sample_wisp_walks(self.graph, NODE_TYPE, ANCHORS, 50, self.MAX_LENGTH,
                                     torch.Generator().manual_seed(seed), proposal)
            weighted.append(dense_transported_laplacian(
                merge_wisp_cells(draw, NODE_TYPE, EDGE_INDEX, 50), self.maps))
            ones = dataclasses.replace(draw, ratio=torch.ones_like(draw.ratio))
            unweighted.append(dense_transported_laplacian(
                merge_wisp_cells(ones, NODE_TYPE, EDGE_INDEX, 50), self.maps))
        self.assertLessEqual(self._excess(weighted), 1e-6)
        self.assertGreater(self._excess(unweighted), 1e-3, "control: without IS the mean is biased")


class WispTrainerDrawTests(unittest.TestCase):
    """Review MINOR-5: under the adaptive proposal the training draws change, the evaluation cells
    do not, and every evaluation epoch sees the evaluation cells (through Trainer.fit)."""

    def _setup(self, overlap=None):
        from dataclasses import dataclass
        from types import SimpleNamespace

        from src.configs.config import TrainConfig
        from src.data.dataset import build_node_classification_paths
        from src.trainers.trainer import Trainer

        extra = {} if overlap is None else {"overlap_sampling": overlap}
        cfg = TrainConfig.from_dict({"device": "cpu", "epochs": 3, "data": {
            "dataset_name": "acm", "path_source": "wisp", "wisp_max_length": 3,
            "wisp_walks_per_anchor": 20, "wisp_proposal": "adaptive", **extra}})
        fact_layer = SimpleNamespace(self_inverse=SELF_INVERSE, mode="tied", summary=lambda: {})
        common = dict(fact_layer=fact_layer, node_slices={"paper": (0, 4)}, node_type=NODE_TYPE,
                      node_type_names=("paper", "a", "s"), edge_index=EDGE_INDEX,
                      edge_type=EDGE_TYPE, relation_tuples=())
        evaluation = build_node_classification_paths(config=cfg, seed=cfg.seed, **common)

        @dataclass
        class Prepared:
            face_groups: tuple
            target_val_mask: object = None
            target_test_mask: object = None

        prepared = Prepared(face_groups=evaluation)
        data = SimpleNamespace(to=lambda device: prepared, **common)
        snapshot = [(g.face_node_index.clone(), g.path_weight.clone()) for g in evaluation]

        class Model:
            last_path_weights = ()

            def to(self, device):
                return self

        model = Model()
        seen = {"train": [], "eval": []}

        class Recorder(Trainer):
            def _train_epoch(self, batch, epoch):
                seen["train"].append(batch.face_groups)
                generator = torch.Generator().manual_seed(epoch)
                # what a training forward leaves for the next adaptive draw: positive w_f
                model.last_path_weights = tuple(
                    torch.rand(g.num_faces, generator=generator, dtype=torch.float64) * 5 + 0.01
                    for g in batch.face_groups if g.num_faces)
                return {"loss": 0.0}

            def _eval_epoch(self, batch, mask, epoch=-1):
                seen["eval"].append(batch.face_groups)
                # macro_f1: the NC selection metric of spec v1.15 §6 (default since M9)
                return {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0}

        logger = SimpleNamespace(info=lambda *a, **k: None, log=lambda *a, **k: None,
                                 log_summary=lambda *a, **k: None)
        checkpointer = SimpleNamespace(save=lambda **k: None, save_if_best=lambda **k: False)
        trainer = Recorder(model=model, optimizer=None, scheduler=None, criterion=None, cfg=cfg,
                           logger=logger, checkpointer=checkpointer)
        return trainer, data, evaluation, snapshot, seen

    def test_adaptive_training_draws_leave_the_evaluation_cells_unchanged(self) -> None:
        trainer, data, evaluation, snapshot, seen = self._setup()
        trainer.fit(data)
        self.assertEqual(len(seen["train"]), 3)
        self.assertEqual(len(seen["eval"]), 6)  # validation and test, per epoch
        for groups in seen["eval"]:
            self.assertIs(groups, evaluation)
        for g, (nodes, weight) in zip(evaluation, snapshot):
            self.assertTrue(torch.equal(g.face_node_index, nodes))
            self.assertTrue(torch.equal(g.path_weight, weight))
        later = seen["train"][2]  # an adaptive draw (q from epoch 1's weights)
        self.assertFalse(all(a.face_node_index.shape == b.face_node_index.shape
                             and torch.equal(a.path_weight, b.path_weight)
                             for a, b in zip(later, evaluation)))
        ratios = torch.cat([g.path_weight for g in later]) * 20
        self.assertFalse(bool(torch.allclose(ratios, ratios.round())),
                         "the adaptive draw carries IS ratios other than 1 (q differs from p)")


class WispConfigValidationTests(unittest.TestCase):
    """Review MINOR-7/8: validation in __post_init__ (direct construction cannot bypass it),
    wisp_* keys rejected under schemas, and one rejection test per never-tuned key."""

    def test_direct_construction_is_validated(self) -> None:
        from src.configs.config import DataConfig, ModelConfig

        bad_model = [{"path_weight": "exp", "gamma_init": 2.0},
                     {"path_weight": "sigmoid_gamma", "path_weight_init": 0.3},
                     {"path_weight": "nope"}, {"path_weight_init": 0.0}]
        for kwargs in bad_model:
            with self.subTest(model=kwargs), self.assertRaises(ValueError):
                ModelConfig(**kwargs)
        ModelConfig(path_weight="exp", gamma_init=1.0)  # the saved default round-trips
        bad_data = [{"path_source": "nope"}, {"wisp_max_length": 1},
                    {"wisp_walks_per_anchor": -1}, {"wisp_proposal": "nope"},
                    {"path_source": "schemas", "wisp_max_length": 5},
                    {"path_source": "schemas", "wisp_walks_per_anchor": 30},
                    {"path_source": "schemas", "wisp_proposal": "adaptive"}]
        for kwargs in bad_data:
            with self.subTest(data=kwargs), self.assertRaises(ValueError):
                DataConfig(**kwargs)
        DataConfig(path_source="schemas")
        DataConfig(path_source="wisp", wisp_max_length=5, wisp_walks_per_anchor=30)

    def test_wisp_keys_are_rejected_under_schemas_through_from_dict(self) -> None:
        from src.configs.config import TrainConfig

        for extra in ({"wisp_max_length": 3}, {"wisp_walks_per_anchor": 30},
                      {"wisp_proposal": "adaptive"}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                TrainConfig.from_dict({"data": {"dataset_name": "acm", "path_source": "schemas",
                                                **extra}})
        with self.assertRaises(ValueError):  # the pinned GC DFS row is a schema row
            TrainConfig.from_dict({"data": {"dataset_name": "mutag", "path_source": "schemas",
                                            "path_sampler": "dfs", "fact_layer": "untied",
                                            "resample_paths": False, "wisp_max_length": 3}})

    def test_every_never_tuned_wisp_key_is_rejected(self) -> None:
        from src.configs.config import NEVER_TUNED_WISP_KEYS, reject_never_tuned_settings

        # spec v1.15 J2 (Part M, M10): l_max and r are tuned; the rest stays fixed
        expected = {"path_source", "path_weight", "path_weight_init",
                    "path_weight_input", "wisp_proposal", "path_weight_context"}
        self.assertEqual(set(NEVER_TUNED_WISP_KEYS), expected)
        for name in ("data.wisp_max_length", "data.wisp_walks_per_anchor"):
            reject_never_tuned_settings({}, [name])
        for key in sorted(expected):
            for name in (key, f"data.{key}", f"model.{key}", f"ablation.{key}"):
                with self.subTest(name=name):
                    with self.assertRaises(ValueError) as caught:
                        reject_never_tuned_settings({}, [name])
                    self.assertIn("wisp_proposal", str(caught.exception))
        reject_never_tuned_settings({}, ["model.d"])  # ordinary hyperparameters still pass
        for base in ({"model": {"path_weight": "sigmoid_gamma"}},
                     {"ablation": {"path_weight_input": "endpoints"}},
                     {"ablation": {"path_weight_context": True}},
                     {"data": {"dataset_name": "dblp", "path_source": "schemas"}}):
            with self.subTest(base=base), self.assertRaises(ValueError):
                reject_never_tuned_settings(base)



# ============================================================================= I6: overlap
class WispOverlapSamplingTests(unittest.TestCase):
    """Part I (I6): drawing epoch e+1's cells in a worker while epoch e trains changes nothing.
    For both proposals, every training draw's cells and m_f, every training logit and the final
    parameters are bit-identical with and without overlap (a stop condition if not)."""

    def _train(self, overlap: bool, proposal: str, epochs: int = 4):
        from types import SimpleNamespace

        from src.builders import build_optimizer
        from src.configs.config import TrainConfig
        from src.models.h2snn_model import build_model
        from src.trainers.trainer import build_trainer
        from tests.test_paths_and_facts import _prepared_small

        cfg = TrainConfig.from_dict({
            "device": "cpu", "seed": 7, "epochs": epochs, "log_every": 1000,
            "model": {"d": 4, "d_r": 2, "K": 2, "classifier_hidden_dim": 8,
                      "face_mlp_hidden_dim": 8, "edge_mlp_hidden_dim": 8},
            "data": {"dataset_name": "acm", "path_source": "wisp", "wisp_max_length": 3,
                     "wisp_walks_per_anchor": 10, "wisp_proposal": proposal,
                     "overlap_sampling": overlap}})
        torch.manual_seed(0)
        data = _prepared_small(cfg)
        torch.manual_seed(1)
        model = build_model(cfg, data)
        record = {"cells": [], "logits": []}

        def hook(module, args, output):
            if module.training:
                record["cells"].append([(g.face_node_index.clone(), g.path_weight.clone())
                                        for g in args[0].face_groups])
                record["logits"].append(output.logits.detach().clone())

        model.register_forward_hook(hook)
        logger = SimpleNamespace(info=lambda *a, **k: None, log=lambda *a, **k: None,
                                 log_summary=lambda *a, **k: None)
        checkpointer = SimpleNamespace(save=lambda **k: None, save_if_best=lambda **k: False)
        trainer = build_trainer(model=model, optimizer=build_optimizer(model, cfg), scheduler=None,
                                cfg=cfg, logger=logger, checkpointer=checkpointer)
        trainer.fit(data)
        return record, {k: v.clone() for k, v in model.state_dict().items()}, trainer

    def test_overlap_is_bit_identical_for_both_proposals(self) -> None:
        for proposal in ("uniform", "adaptive"):
            with self.subTest(proposal=proposal):
                off, off_state, off_trainer = self._train(False, proposal)
                on, on_state, on_trainer = self._train(True, proposal)
                self.assertEqual(len(off["logits"]), 4)
                self.assertEqual(len(on["logits"]), 4)
                for epoch, (a, b) in enumerate(zip(off["cells"], on["cells"])):
                    self.assertEqual(len(a), len(b))
                    for (na, wa), (nb, wb) in zip(a, b):
                        self.assertTrue(torch.equal(na, nb), (proposal, epoch))
                        self.assertTrue(torch.equal(wa, wb), (proposal, epoch))
                for epoch, (a, b) in enumerate(zip(off["logits"], on["logits"])):
                    self.assertTrue(torch.equal(a, b), (proposal, epoch))
                for name in off_state:
                    self.assertTrue(torch.equal(off_state[name], on_state[name]), name)
                # non-vacuity: the training draws differ across epochs, and (adaptive) q is used
                self.assertFalse(torch.equal(off["cells"][1][0][1], off["cells"][2][0][1])
                                 and torch.equal(off["cells"][1][0][0], off["cells"][2][0][0]))
                self.assertEqual(len(on_trainer.draw_wait_seconds), 4)
                if proposal == "adaptive":  # epochs >= 1 draw from 1/2 p + 1/2 q: ratios != 1
                    scaled = torch.cat([w for _, w in on["cells"][2]]).double() * 10
                    self.assertFalse(bool(torch.allclose(scaled, scaled.round())))
        _, _, adaptive = self._train(True, "adaptive", epochs=2)
        self.assertIsNotNone(adaptive._previous_draw)

    def test_overlap_switch_is_a_boolean_and_rejected_where_it_has_no_effect(self) -> None:
        from src.configs.config import TrainConfig

        with self.assertRaises(ValueError):
            TrainConfig.from_dict({"data": {"overlap_sampling": "yes"}})
        with self.assertRaises(ValueError):  # link prediction has its own trainer
            TrainConfig.from_dict({"data": {"dataset_name": "amazonbook", "overlap_sampling": False}})
        self.assertTrue(TrainConfig.from_dict({}).data.overlap_sampling)



# ============================================================================= K3: guards
class WispGuardTests(unittest.TestCase):
    """Part K, K3 (spec v1.6 E11, I7 review m1 and NIT): r is never tuned; the tuning base config
    carries the pre-registered WISP settings; path_weight_context is the ladder's appendix pair
    only; r resolves from the per-l_max calibration table; an unresolved path source is refused."""

    def test_r_is_in_the_search_space(self) -> None:
        """Spec v1.15 J2 supersedes E11 (r never tuned): r in {8, 16, 32, 64} is tuned."""
        from src.configs.config import reject_never_tuned_settings

        for name in ("data.wisp_walks_per_anchor", "wisp_walks_per_anchor"):
            with self.subTest(name=name):
                reject_never_tuned_settings({}, [name])

    def test_tuning_base_config_carries_the_pre_registered_wisp_settings(self) -> None:
        from src.configs.config import WISP_WALKS_PER_ANCHOR, reject_never_tuned_settings

        # spec v1.15 J2: l_max and r are tuned, so a base config may carry any value; the
        # proposal stays fixed (uniform).
        with self.assertRaises(ValueError):
            reject_never_tuned_settings({"data": {"dataset_name": "acm",
                                                  "wisp_proposal": "adaptive"}})
        for data in ({"dataset_name": "acm", "wisp_max_length": 5},
                     {"dataset_name": "acm", "wisp_walks_per_anchor": 30},
                     {"dataset_name": "dblp", "wisp_walks_per_anchor": 42}):
            with self.subTest(data=data):
                reject_never_tuned_settings({"data": data})
        for dataset in ("acm", "dblp", "imdb"):
            for r in (0, WISP_WALKS_PER_ANCHOR[dataset][4]):
                with self.subTest(dataset=dataset, r=r):
                    reject_never_tuned_settings({"data": {"dataset_name": dataset,
                                                          "wisp_walks_per_anchor": r}})
        # link prediction is not a WISP task: its base configs are not judged by these rules
        reject_never_tuned_settings({"data": {"dataset_name": "amazonbook"}})

    def test_path_weight_context_is_the_ladder_appendix_pair_only(self) -> None:
        from src.configs.config import TrainConfig

        for ablation in ({"endpoint_edges": True}, {"free_path_maps": True},
                         {"coupling": "pair_attention"}):
            with self.subTest(ablation=ablation), self.assertRaises(ValueError):
                TrainConfig.from_dict({"ablation": {**ablation, "path_weight_context": True}})
        for arm in ("full", "endpoints"):
            TrainConfig.from_dict({"ablation": {"path_weight_input": arm,
                                                "path_weight_context": True}})

    def test_r_resolves_from_the_table_for_its_l_max(self) -> None:
        from src.configs.config import WISP_WALKS_PER_ANCHOR, TrainConfig

        self.assertEqual({d: WISP_WALKS_PER_ANCHOR[d][4] for d in ("acm", "dblp", "imdb")},
                         {"acm": 42, "dblp": 65, "imdb": 12})
        expected = {"acm": [2, 3, 4, 5, 6], "dblp": [3, 4, 5, 6], "imdb": [2, 3, 4, 5, 6]}
        self.assertEqual({d: sorted(t) for d, t in WISP_WALKS_PER_ANCHOR.items()}, expected)
        for dataset, table in WISP_WALKS_PER_ANCHOR.items():
            for max_length, r in table.items():
                with self.subTest(dataset=dataset, max_length=max_length):
                    data = TrainConfig.from_dict({"data": {
                        "dataset_name": dataset, "wisp_max_length": max_length}}).data
                    self.assertEqual(data.wisp_walks_per_anchor, r)
        for dataset, max_length in (("acm", 7), ("dblp", 2)):  # no calibrated r for this support
            with self.subTest(dataset=dataset, max_length=max_length), self.assertRaises(ValueError):
                TrainConfig.from_dict({"data": {"dataset_name": dataset,
                                                "wisp_max_length": max_length}})
        explicit = TrainConfig.from_dict({"data": {"dataset_name": "acm", "wisp_max_length": 7,
                                                   "wisp_walks_per_anchor": 30}}).data
        self.assertEqual(explicit.wisp_walks_per_anchor, 30)

    def test_an_unresolved_path_source_is_refused_by_its_consumers(self) -> None:
        from src.configs.config import DataConfig, TrainConfig
        from src.data.dataset import build_node_classification_paths
        from src.models.h2snn_model import build_model

        from types import SimpleNamespace

        from tests.test_v2_model import _float64_pap

        config = TrainConfig()  # built directly: data.path_source is "" (unresolved)
        self.assertEqual(DataConfig().path_source, "")
        # real inputs, so a consumer without the check would silently build something
        with self.assertRaises(ValueError):
            build_model(config, _float64_pap())
        with self.assertRaises(ValueError):
            build_node_classification_paths(
                config=config, fact_layer=SimpleNamespace(self_inverse=SELF_INVERSE, mode="tied"),
                node_slices={"paper": (0, 4)}, node_type=NODE_TYPE,
                node_type_names=("paper", "a", "s"), edge_index=EDGE_INDEX, edge_type=EDGE_TYPE,
                relation_tuples=(), seed=0)


if __name__ == "__main__":
    unittest.main()
