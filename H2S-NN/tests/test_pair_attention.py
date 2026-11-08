"""`coupling: pair_attention`, the A4 test variant (plan §12.3, task card Part F U4b).

Test ids follow the card (T-F4 ... T-F7). Systems are built from explicit maps, signed
mixed-length cells and a real `PairAttention` module (float64), with an independent dense and
a loop-based reference.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest import mock

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from torch import nn

import src.models.pair_attention as pa
from src.configs.config import AblationConfig, TrainConfig, reject_never_tuned_settings
from src.models.map_operators import PathTransportOperator
from src.models.pair_attention import PairAttention, build_attention_operator
from tests.test_cooperative import (
    TreePaths,
    _balance_residual,
    _effective_weights,
    _gate_model,
    _step_matrix,
)
from tests.test_paths_and_facts import SectionPreservationTests, SignedPaths, _signed_dense_transports
from tests.test_v2_model import _chain_operators

F64_TOL = 1e-10
NUM_RELATIONS = 3
SHARED_RELATION = 2  # a relation with rho(r) = r: one shared role


class AttentionSystem:
    """Explicit maps + signed mixed-length cells + a randomly initialised `PairAttention`."""

    def __init__(self, seed: int, d: int = 4, num_nodes: int = 9, lengths=(2, 3, 4),
                 signed: bool = True, eta: float = 0.5, paths=None, edge_index=None,
                 merged=None) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.generator, self.d, self.num_nodes = generator, d, num_nodes
        if paths is None and edge_index is None:
            paths = SignedPaths(num_nodes, lengths, 3, generator, signed=signed)
        self.paths = paths
        self.edge_index = paths.edge_index if paths is not None else edge_index
        num_edges = self.edge_index.size(1)
        self.maps = _chain_operators(num_edges, k=2, d=d, seed=seed, dtype=torch.float64,
                                     shared=True)
        self.edge_type = torch.randint(0, NUM_RELATIONS, (num_edges,), generator=generator)
        self.shared_roles = torch.zeros(NUM_RELATIONS, dtype=torch.bool)
        self.shared_roles[SHARED_RELATION] = True
        self.h0 = torch.randn(num_nodes, d, generator=generator, dtype=torch.float64)
        self.hidden = 5
        groups = zip(paths.nodes, paths.edges, paths.signs) if paths is not None else ()
        self.transports, self.latents, self.log_merged, self.cells = [], [], [], []
        for index, (nodes, edges, signs) in enumerate(groups):
            self.transports.append(PathTransportOperator(
                self.maps.src, self.maps.tgt, nodes, edges, self.edge_index, shared_scale=True,
                face_edge_sign=signs))
            self.latents.append(torch.randn(nodes.size(0), self.hidden, generator=generator,
                                            dtype=torch.float64))
            m = (merged[index] if merged is not None else
                 0.3 + 2.0 * torch.rand(nodes.size(0), generator=generator, dtype=torch.float64))
            self.log_merged.append(m.log())
            self.cells.append((nodes, edges, signs))
        self.attention = PairAttention(d=d, d_r=3, path_hidden=self.hidden,
                                       num_relations=NUM_RELATIONS, max_positions=6).double()
        for parameter in self.attention.parameters():  # peaked, non-trivial scores
            nn.init.normal_(parameter, std=0.8, generator=generator)
        self.eta = torch.tensor(eta, dtype=torch.float64)

    def operator(self):
        return build_attention_operator(
            self.attention, self.maps, self.eta, self.h0, self.edge_index, self.edge_type,
            self.shared_roles, self.transports, self.latents, self.log_merged, checkpoint=False)

    # -- independent references -------------------------------------------------------------
    def reference_alpha(self):
        """Loop-based scores and softmax with the module's own layers, entry by entry."""
        a, act = self.attention, self.attention.activation
        n, num_edges = self.num_nodes, self.edge_index.size(1)

        def score(linear, features):
            return a.a(act(linear(features))).squeeze()

        entries = {u: [] for u in range(n)}  # receiver -> list of (key, raw score)
        for u in range(n):
            f = torch.cat([self.h0[u], self.h0[u], a.self_emb])
            entries[u].append((("self", u), score(a.w_self, f)))
        for e in range(num_edges):
            s, t = int(self.edge_index[0, e]), int(self.edge_index[1, e])
            r = int(self.edge_type[e])
            role_t = 0 if bool(self.shared_roles[r]) else 1
            fs = torch.cat([self.h0[s], self.h0[t], a.rel_emb.weight[r], a.dir_emb.weight[0]])
            ft = torch.cat([self.h0[t], self.h0[s], a.rel_emb.weight[r], a.dir_emb.weight[role_t]])
            entries[s].append((("edge", e, 0), score(a.w_edge, fs)))
            entries[t].append((("edge", e, 1), score(a.w_edge, ft)))
        for g, (nodes, _, _) in enumerate(self.cells):
            for f in range(nodes.size(0)):
                for i in range(nodes.size(1)):
                    u = int(nodes[f, i])
                    feat = torch.cat([self.h0[u], self.latents[g][f], a.pos_emb.weight[i]])
                    entries[u].append((("cell", g, f, i),
                                       score(a.w_cell, feat) + self.log_merged[g][f]))
        alpha = {}
        for u, items in entries.items():
            raw = torch.stack([s for _, s in items])
            weights = torch.softmax(raw, dim=0)
            for (key, _), w in zip(items, weights):
                alpha[key] = w
        return alpha

    def dense_p(self, operator) -> torch.Tensor:
        """`P` as an explicit `[N d, N d]` matrix from the operator's alpha, materialized `F`
        and the signed dense transports."""
        n, d, num_edges = self.num_nodes, self.d, self.edge_index.size(1)
        matrix = torch.zeros(n * d, n * d, dtype=torch.float64)
        eye = torch.eye(d, dtype=torch.float64)
        for u in range(n):
            matrix[u * d:(u + 1) * d, u * d:(u + 1) * d] += operator.alpha_self[u] * eye
        F_src, F_tgt = self.maps.src.materialize(), self.maps.tgt.materialize()
        for e in range(num_edges):
            s, t = int(self.edge_index[0, e]), int(self.edge_index[1, e])
            q_s_from_t = torch.linalg.solve(F_src[e], F_tgt[e])
            q_t_from_s = torch.linalg.solve(F_tgt[e], F_src[e])
            matrix[s * d:(s + 1) * d, t * d:(t + 1) * d] += operator.alpha_edge[e] * q_s_from_t
            matrix[t * d:(t + 1) * d, s * d:(s + 1) * d] += \
                operator.alpha_edge[num_edges + e] * q_t_from_s
        for (nodes, edges, signs), alpha in zip(self.cells, operator.alpha_cells):
            transports = _signed_dense_transports(self.maps, edges, signs)  # [P, n, d, d]
            size = nodes.size(1)
            for f in range(nodes.size(0)):
                for i in range(size):
                    u = int(nodes[f, i])
                    for j in range(size):
                        v = int(nodes[f, j])
                        matrix[u * d:(u + 1) * d, v * d:(v + 1) * d] += (
                            alpha[f, i] / size * transports[f, i].T @ transports[f, j])
        return matrix


def _loop_system(seed: int) -> AttentionSystem:
    """Edges with a self-loop and a repeated pair, no cells (ACM `cite` has self-loops)."""
    edge_index = torch.tensor([[0, 1, 2, 2, 3, 0, 4], [1, 2, 2, 0, 4, 1, 3]])
    return AttentionSystem(seed, num_nodes=5, edge_index=edge_index, lengths=())


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1.0))


@contextmanager
def _patched(*patches):
    for patch in patches:
        patch.start()
    try:
        yield
    finally:
        mock.patch.stopall()


# ----------------------------------------------------------------------------- T-F4
class AttentionWeightTests(unittest.TestCase):
    def test_TF4_every_row_is_positive_and_sums_to_one(self) -> None:
        systems = [AttentionSystem(seed, signed=seed % 2 == 0) for seed in range(8)]
        systems += [_loop_system(seed) for seed in range(4)]
        for system in systems:
            operator = system.operator()
            alphas = [operator.alpha_self, operator.alpha_edge, *operator.alpha_cells]
            self.assertTrue(all(bool((a > 0).all()) for a in alphas))
            self.assertLess(float((operator.row_sums(system.num_nodes) - 1).abs().max()), 1e-12)

    def test_TF4_alpha_matches_an_independent_loop_based_softmax(self) -> None:
        for system in (AttentionSystem(0), AttentionSystem(1, signed=False), _loop_system(2)):
            operator = system.operator()
            reference = system.reference_alpha()
            num_edges = system.edge_index.size(1)
            worst = 0.0
            for u in range(system.num_nodes):
                worst = max(worst, abs(float(operator.alpha_self[u] - reference[("self", u)])))
            for e in range(num_edges):
                worst = max(worst,
                            abs(float(operator.alpha_edge[e] - reference[("edge", e, 0)])),
                            abs(float(operator.alpha_edge[num_edges + e]
                                      - reference[("edge", e, 1)])))
            for g, alpha in enumerate(operator.alpha_cells):
                for f in range(alpha.size(0)):
                    for i in range(alpha.size(1)):
                        worst = max(worst, abs(float(alpha[f, i] - reference[("cell", g, f, i)])))
            self.assertLess(worst, 1e-12)

    def test_TF4_the_fixtures_have_loops_shared_roles_and_signed_hops(self) -> None:
        system = AttentionSystem(0)
        self.assertTrue(any(bool((s < 0).any()) for s in system.paths.signs))
        self.assertEqual(sorted({n.size(1) - 1 for n in system.paths.nodes}), [2, 3, 4])
        self.assertTrue(bool((system.edge_type == SHARED_RELATION).any()))
        loops = _loop_system(0)
        self.assertTrue(bool((loops.edge_index[0] == loops.edge_index[1]).any()))


# ----------------------------------------------------------------------------- T-F5
class SectionTests(unittest.TestCase):
    def _tree_system(self, seed: int, d: int = 3):
        edge_index, groups = SectionPreservationTests()._tree(seed)
        paths = TreePaths(edge_index, groups)
        system = AttentionSystem(seed, d=d, num_nodes=8, paths=paths)
        return system, edge_index

    def _section(self, system: AttentionSystem, edge_index: torch.Tensor) -> torch.Tensor:
        F_src, F_tgt = system.maps.src.materialize(), system.maps.tgt.materialize()
        x = torch.zeros(system.num_nodes, system.d, dtype=torch.float64)
        x[0] = torch.randn(system.d, dtype=torch.float64)
        known = {0}
        while len(known) < system.num_nodes:
            for e in range(edge_index.size(1)):
                u, v = int(edge_index[0, e]), int(edge_index[1, e])
                if u in known and v not in known:
                    x[v] = torch.linalg.solve(F_tgt[e], F_src[e] @ x[u])
                    known.add(v)
                elif v in known and u not in known:
                    x[u] = torch.linalg.solve(F_src[e], F_tgt[e] @ x[v])
                    known.add(u)
        return x

    def test_TF5_a_global_section_is_a_fixed_point(self) -> None:
        for seed in range(4):
            with self.subTest(seed=seed):
                system, edge_index = self._tree_system(seed)
                self.assertTrue(any(bool((s < 0).any()) for s in system.paths.signs))
                x = self._section(system, edge_index)
                operator = system.operator()
                self.assertLess(_rel(operator.propagate(x), x), F64_TOL)
                # control: a state that is not a section is moved
                y = torch.randn_like(x)
                self.assertGreater(_rel(operator.propagate(y), y), 1e-3)


# ----------------------------------------------------------------------------- T-F6
class NonExpansionTests(unittest.TestCase):
    def _max_norm_increase(self, eta: float, fixtures: int, steps: int = 50) -> float:
        model = _gate_model("pair_attention")
        worst = -1e9
        for seed in range(fixtures):
            system = AttentionSystem(500 + seed, eta=eta)
            operator = system.operator()
            x = torch.randn(system.num_nodes, system.d, generator=system.generator,
                            dtype=torch.float64)
            peak = float(x.norm(dim=-1).max())
            for _ in range(steps):
                x = model.diffuse(x, operator)
                now = float(x.norm(dim=-1).max())
                worst = max(worst, now - peak)
                peak = now
        return worst

    def test_TF6_max_node_norm_is_nonincreasing_for_eta_up_to_one(self) -> None:
        for eta in (0.2, 0.6, 1.0):
            self.assertLessEqual(self._max_norm_increase(eta, fixtures=50), 1e-12, f"eta={eta}")

    def test_TF6_control_unnormalized_exp_scores_expand(self) -> None:
        unnormalized = lambda scores, receivers, n: torch.exp(scores)  # noqa: E731
        with _patched(mock.patch.object(pa, "segment_softmax", unnormalized)):
            self.assertGreater(self._max_norm_increase(0.6, fixtures=50), 1e-3)


# ----------------------------------------------------------------------------- T-F7
class DenseReferenceTests(unittest.TestCase):
    def test_TF7_operator_equals_the_explicit_matrix(self) -> None:
        systems = [AttentionSystem(seed, signed=True) for seed in range(4)]
        systems += [AttentionSystem(10 + seed, lengths=(2, 3), signed=False) for seed in range(2)]
        systems += [_loop_system(seed) for seed in range(2)]
        for index, system in enumerate(systems):
            with self.subTest(fixture=index):
                operator = system.operator()
                dense = system.dense_p(operator)
                x = torch.randn(system.num_nodes, system.d, generator=system.generator,
                                dtype=torch.float64)
                got = operator.propagate(x).reshape(-1)
                self.assertLess(_rel(got, dense @ x.reshape(-1)), F64_TOL)

    def test_TF7_state_dict_differs_from_d11_only_by_attention_added_and_gates_removed(self) -> None:
        coop = _gate_model("cooperative")
        pair = _gate_model("pair_attention")
        coop_keys, pair_keys = set(coop.state_dict()), set(pair.state_dict())
        removed, added = coop_keys - pair_keys, pair_keys - coop_keys
        self.assertEqual(removed, {"listen_gate.weight", "listen_gate.bias",
                                   "broadcast_gate.weight", "broadcast_gate.bias"})
        self.assertTrue(added and all(key.startswith("pair_attention.") for key in added))
        self.assertEqual(added, {f"pair_attention.{k}" for k in PairAttention(
            d=4, d_r=2, path_hidden=8, num_relations=2, max_positions=5).state_dict()})
        self.assertIsNone(pair.listen_gate)
        self.assertIsNone(pair.broadcast_gate)
        for key in coop_keys & pair_keys:  # shapes of every shared parameter are unchanged
            self.assertEqual(coop.state_dict()[key].shape, pair.state_dict()[key].shape, key)


# ----------------------------------------------------------------------------- Prop. 4 control (T-F3b)
class Prop4ControlTests(unittest.TestCase):
    def test_TF3b_control_pair_attention_has_no_metric(self) -> None:
        """The same factorization check that the gated operator passes (tests.test_cooperative)
        fails for pair attention: free per-direction strengths admit no per-node metric."""
        model = _gate_model("pair_attention")
        worst = 0.0
        for seed in range(30):
            system = AttentionSystem(3000 + seed)
            matrix = _step_matrix(model, system.operator(), system.num_nodes, system.d)
            _, residual = _balance_residual(_effective_weights(matrix, system.num_nodes, system.d))
            worst = max(worst, residual)
        self.assertGreater(worst, 1e-3)


# ----------------------------------------------------------------------------- merged weights, config, groups
class MergedWeightTests(unittest.TestCase):
    def test_a_merged_cell_of_weight_two_equals_two_duplicate_unit_cells(self) -> None:
        """Softmax mass of a cell with merged weight m is m e^s (the `log m_f` term): one cell with
        m = 2 acts exactly like two identical cells with m = 1 each."""
        base = AttentionSystem(7, lengths=(2,))
        nodes, edges, signs = base.cells[0]
        keep = slice(0, 1)

        def with_cells(count: int, weight: float) -> AttentionSystem:
            system = AttentionSystem(7, lengths=(2,))
            node_rows = nodes[keep].repeat(count, 1)
            edge_rows = edges[keep].repeat(count, 1)
            sign_rows = signs[keep].repeat(count, 1)
            system.cells = [(node_rows, edge_rows, sign_rows)]
            system.transports = [PathTransportOperator(
                system.maps.src, system.maps.tgt, node_rows, edge_rows, system.edge_index,
                shared_scale=True, face_edge_sign=sign_rows)]
            system.latents = [base.latents[0][keep].repeat(count, 1)]
            system.log_merged = [torch.full((count,), weight, dtype=torch.float64).log()]
            return system

        merged, duplicated = with_cells(1, 2.0), with_cells(2, 1.0)
        x = torch.randn(base.num_nodes, base.d, generator=base.generator, dtype=torch.float64)
        self.assertLess(_rel(merged.operator().propagate(x), duplicated.operator().propagate(x)),
                        1e-12)

    def test_pair_attention_is_never_a_default_and_never_tuned(self) -> None:
        self.assertEqual(AblationConfig().coupling, "cooperative")
        self.assertEqual(TrainConfig.from_dict({}).ablation.coupling, "cooperative")
        with self.assertRaises(ValueError):
            reject_never_tuned_settings({"ablation": {"coupling": "pair_attention"}})
        with self.assertRaises(ValueError):
            reject_never_tuned_settings({}, search_parameters=["ablation.coupling"])
        reject_never_tuned_settings({"ablation": {"coupling": "cooperative"}})  # control
        for bad in ({"per_side_scale": True}, {"free_path_maps": True}, {"endpoint_edges": True},
                    {"holonomy_edges": True}, {"path_weight_input": "uniform"},
                    {"h0_only_readout": True}):
            with self.assertRaises(ValueError, msg=str(bad)):
                AblationConfig.from_dict({"coupling": "pair_attention", **bad})
        AblationConfig.from_dict({"coupling": "pair_attention", "no_path_cells": True})  # allowed

    def test_attention_parameters_share_the_fast_group_and_gate_heads_are_absent(self) -> None:
        from src.builders import build_optimizer
        from src.models.h2snn_model import build_model
        from tests.test_cooperative import _signal_fixture

        data = _signal_fixture()
        cfg = TrainConfig.from_dict({"device": "cpu",  # the D11 arm: eta in the fast group
                                     "model": {"d": 8, "d_r": 4, "K": 2, "encoder": "first_order",
                                               "path_weight": "sigmoid_gamma"},  # theta_gamma (Part H)
                                     "ablation": {"coupling": "pair_attention"},
                                     "data": {"resample_paths": False, "path_source": "schemas"}})
        model = build_model(cfg, data)
        optimizer = build_optimizer(model, cfg)
        fast = [g for g in optimizer.param_groups if g["lr"] > cfg.optim.lr]
        self.assertEqual(len(fast), 1)
        ids = {id(p) for p in fast[0]["params"]}
        expected = [model.theta_eta, model.theta_gamma, *model.pair_attention.parameters()]
        self.assertEqual(ids, {id(p) for p in expected})
        self.assertIsNone(model.listen_gate)
        out = model(data)  # a full forward and backward works, and the head is the D6 head
        out.logits.sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.pair_attention.parameters()))
        self.assertFalse(model.diagnostics_available)


if __name__ == "__main__":
    unittest.main()
