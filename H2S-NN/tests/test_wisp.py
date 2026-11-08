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
                  d: int = 4):
    from src.configs.config import TrainConfig
    from src.models.h2snn_model import H2SNN, ModelDims

    config = TrainConfig.from_dict({
        "device": "cpu",
        "model": {"d": d, "d_r": 2, "K": 2, "classifier_hidden_dim": 8, "face_mlp_hidden_dim": 8,
                  "edge_mlp_hidden_dim": 8, "geometry_context": geometry_context,
                  "path_weight": path_weight},
        "ablation": {"path_weight_input": arm},
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

    def _weights(self, model, geometry):
        latent = model._path_latent(geometry, self.group)
        return model._path_weights(latent, self.group, geometry)

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

    def test_TH7_intermediate_perturbation(self) -> None:
        cell = 0
        middle = int(self.group.face_node_index[cell, 1])
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = _ladder_model(arm)
                geometry = torch.randn(8, model.geometry_dim, dtype=torch.float64,
                                       generator=torch.Generator().manual_seed(2))
                before = self._weights(model, geometry)[cell]
                moved = geometry.clone()
                moved[middle] += 1.0
                after = self._weights(model, moved)[cell]
                if arm == "full":
                    self.assertNotEqual(float(before), float(after), "full sees the intermediate")
                else:
                    self.assertEqual(float(before), float(after), f"{arm} must not see it")

    def test_TH7_endpoint_perturbation_and_schema_equality(self) -> None:
        cell = 0
        start = int(self.group.face_node_index[cell, 0])
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = _ladder_model(arm)
                geometry = torch.randn(8, model.geometry_dim, dtype=torch.float64,
                                       generator=torch.Generator().manual_seed(3))
                before = self._weights(model, geometry)
                moved = geometry.clone()
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

    def test_TH7_context_off_h0_perturbation_and_context_on_control(self) -> None:
        """(b) geometry_context false: perturbing an intermediate's h0 leaves the endpoints
        weight unchanged. Control: with the context on it changes (c of the endpoints contains
        the intermediate's h0: the only channel)."""
        cell = 0
        middle = int(self.group.face_node_index[cell, 1])
        for context, expect_change in ((False, False), (True, True)):
            with self.subTest(context=context):
                model = _ladder_model("endpoints", geometry_context=context)
                h0 = torch.randn(8, model.d, dtype=torch.float64,
                                 generator=torch.Generator().manual_seed(4))
                moved = h0.clone()
                moved[middle] += 1.0
                w = [self._weights(model, model.geometry_inputs(h, EDGE_INDEX, EDGE_TYPE))[cell]
                     for h in (h0, moved)]
                self.assertEqual(float(w[0]) != float(w[1]), expect_change)

    def test_TH7_capacity_rule_endpoints_vs_full_within_5_percent(self) -> None:
        def g_params(model):
            modules = [m for m in (model.path_encoder, model.path_gate) if m is not None]
            return sum(p.numel() for m in modules for p in m.parameters())

        for context in (True, False):
            full = g_params(_ladder_model("full", geometry_context=context))
            endpoints = g_params(_ladder_model("endpoints", geometry_context=context))
            self.assertLessEqual(abs(endpoints - full) / full, 0.05, (context, full, endpoints))

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


if __name__ == "__main__":
    unittest.main()
