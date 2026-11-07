"""D12 (task card Part G): relational context (G1, T-G7) and the filter-bank encoder (G2, T-G1 ...
T-G6, T-G8). Float64 fixtures with path cells and signed hops."""
from __future__ import annotations

import unittest

import torch

from src.data.fact_layer import apply_fact_layer, build_fact_layer
from src.models.relational_context import RelationalContext
from src.models.sheaf_maps import _build_mlp, mlp_on_blocks
from tests.test_v2_model import _float64_pap, _v2_model


def _reference_context(h0: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor,
                       num_relations: int, shared: tuple[int, ...] = ()) -> torch.Tensor:
    """Loop reference of `c`: slot (r, role) of node v = mean over the other endpoints."""
    n, d = h0.shape
    sums = torch.zeros(n, 2 * num_relations, d, dtype=h0.dtype)
    counts = torch.zeros(n, 2 * num_relations, dtype=h0.dtype)
    for (u, v), r in zip(edge_index.t().tolist(), edge_type.tolist()):
        role_v = 0 if r in shared else 1
        sums[u, 2 * r] += h0[v]
        counts[u, 2 * r] += 1
        sums[v, 2 * r + role_v] += h0[u]
        counts[v, 2 * r + role_v] += 1
    return (sums / counts.clamp_min(1).unsqueeze(-1)).reshape(n, -1)


def _random_facts(seed: int, num_nodes: int = 12, num_edges: int = 30, num_relations: int = 3):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(num_nodes, (2, num_edges), generator=g),
            torch.randint(num_relations, (num_edges,), generator=g),
            torch.randn(num_nodes, 5, generator=g, dtype=torch.float64))


# ----------------------------------------------------------------------------- G1: context
class RelationalContextTests(unittest.TestCase):
    def test_context_equals_the_loop_reference(self) -> None:
        for seed in range(5):
            edge_index, edge_type, h0 = _random_facts(seed)
            c = RelationalContext(3)(h0, edge_index, edge_type)
            self.assertEqual(tuple(c.shape), (12, 2 * 3 * 5))
            self.assertLessEqual(float((c - _reference_context(h0, edge_index, edge_type, 3)).abs().max()),
                                 1e-12)
            # a node with no neighbour in a slot gets zeros there
        isolated = torch.zeros(1, 2 * 3 * 5, dtype=torch.float64)
        edge_index = torch.tensor([[0], [1]])
        c = RelationalContext(3)(torch.randn(3, 5, dtype=torch.float64), edge_index, torch.tensor([0]))
        self.assertTrue(torch.equal(c[2:3], isolated))

    def test_shared_role_relation_is_orientation_independent(self) -> None:
        edge_index, edge_type, h0 = _random_facts(7)
        flipped = edge_index.clone()
        on_zero = edge_type == 0
        flipped[:, on_zero] = edge_index.flip(0)[:, on_zero]
        shared = RelationalContext(3, (0,))
        self.assertLessEqual(float((shared(h0, edge_index, edge_type)
                                    - shared(h0, flipped, edge_type)).abs().max()), 1e-12)
        reference = _reference_context(h0, edge_index, edge_type, 3, shared=(0,))
        self.assertLessEqual(float((shared(h0, edge_index, edge_type) - reference).abs().max()), 1e-12)
        directed = RelationalContext(3)  # control: roles distinct, flipping moves c
        self.assertGreater(float((directed(h0, edge_index, edge_type)
                                  - directed(h0, flipped, edge_type)).abs().max()), 1e-6)

    def test_facts_only_no_double_counting_through_the_reverse_relation(self) -> None:
        """Stored edges in both directions (r and rho(r)); the tied fact layer keeps r. c over the
        facts equals the loop reference over the stored edges of r alone."""
        g = torch.Generator().manual_seed(3)
        writes = torch.stack([torch.randint(6, 12, (20,), generator=g), torch.randint(0, 6, (20,), generator=g)])
        writes = torch.unique(writes, dim=1)
        stored_index = torch.cat([writes, writes.flip(0)], dim=1)
        stored_type = torch.cat([torch.zeros(writes.size(1), dtype=torch.long),
                                 torch.ones(writes.size(1), dtype=torch.long)])
        layer = build_fact_layer(stored_index, stored_type, ("writes", "written_by"), 12, "tied")
        facts, fact_type = apply_fact_layer(layer, stored_index, stored_type)
        self.assertEqual(len(layer.kept), 1)
        h0 = torch.randn(12, 4, generator=g, dtype=torch.float64)
        c = RelationalContext(len(layer.kept))(h0, facts, fact_type)
        reference = _reference_context(h0, writes, torch.zeros(writes.size(1), dtype=torch.long), 1)
        self.assertEqual(tuple(c.shape), (12, 2 * 4))
        self.assertLessEqual(float((c - reference).abs().max()), 1e-12)


class FactoredFirstLayerTests(unittest.TestCase):
    def test_gathered_first_layer_equals_the_concatenation(self) -> None:
        g = torch.Generator().manual_seed(0)
        torch.manual_seed(0)
        mlp = _build_mlp(2 * 7 + 3 + 2, 11, 5).double()
        h = torch.randn(9, 7, generator=g, dtype=torch.float64)
        rel = torch.randn(4, 3, generator=g, dtype=torch.float64)
        src, tgt = torch.randint(9, (2, 25), generator=g)
        types = torch.randint(4, (25,), generator=g)
        side = torch.randn(25, 2, generator=g, dtype=torch.float64)
        blocks = ((h, src), (h, tgt), (rel, types), (side, None))
        factored = mlp_on_blocks(mlp, blocks, factored=True)
        concat = mlp_on_blocks(mlp, blocks, factored=False)
        self.assertLessEqual(float((factored - concat).abs().max()), 1e-12)
        self.assertTrue(torch.equal(concat, mlp(torch.cat([h[src], h[tgt], rel[types], side], -1))))


class ContextModelTests(unittest.TestCase):
    """T-G7: one neighbour's h0 reaches a node's maps and gates only under geometry_context."""

    def setUp(self) -> None:
        self.data = _float64_pap()
        ei = self.data.edge_index
        # a paper v with two distinct author neighbours u (perturbed) and w (edge kept)
        for v in range(int((self.data.node_type == 0).sum())):
            out = ei[1][ei[0] == v].unique()
            if out.numel() >= 2:
                self.v, self.u, self.w = v, int(out[0]), int(out[1])
                break
        self.edge = int(((ei[0] == self.v) & (ei[1] == self.w)).nonzero()[0])

    def _maps_and_gates(self, model, h0):
        geometry = model.geometry_inputs(h0, self.data.edge_index, self.data.edge_type)
        maps = model.node_to_edge(geometry, self.data.edge_index, self.data.edge_type,
                                  model.relation_embedding.weight).maps
        listen, broadcast = model.coupling_gates(geometry)
        return (maps.src.materialize()[self.edge], maps.tgt.materialize()[self.edge],
                listen[self.v], broadcast[self.v])

    def test_TG7_neighbour_reaches_maps_and_gates_only_with_context(self) -> None:
        for context in (True, False):
            with self.subTest(geometry_context=context):
                model = _v2_model(self.data, geometry_context=context).double().eval()
                with torch.no_grad():
                    h0 = model.encode_nodes(self.data.x, self.data.node_type)
                    moved = h0.clone()
                    moved[self.u] += 1.0
                    before = self._maps_and_gates(model, h0)
                    after = self._maps_and_gates(model, moved)
                for name, a, b in zip(("F_src", "F_tgt", "g", "b"), before, after):
                    if context:
                        self.assertGreater(float((a - b).abs().max()), 1e-12, name)
                    else:
                        self.assertTrue(torch.equal(a, b), name)

    def test_context_off_builds_the_d11_parameter_shapes(self) -> None:
        off = _v2_model(self.data, geometry_context=False)
        on = _v2_model(self.data)
        self.assertIsNone(off.relational_context)
        self.assertEqual(off.geometry_dim, off.d)
        num_relations = len(self.data.relation_names)
        self.assertEqual(on.geometry_dim, on.d * (1 + 2 * num_relations))
        self.assertEqual(off.listen_gate.in_features, off.d)
        self.assertEqual(on.listen_gate.in_features, on.geometry_dim)
        self.assertEqual(sorted(off.state_dict()), sorted(on.state_dict()))  # parameter-free context


# ----------------------------------------------------------------------------- G2: filter bank
def _materialize(fn, n: int, d: int) -> torch.Tensor:
    """Matrix of a linear map on `[n, d]` states, node-major vectorization (columns = basis)."""
    basis = torch.eye(n * d, dtype=torch.float64).view(-1, n, d)
    with torch.no_grad():
        return torch.stack([fn(e).reshape(-1) for e in basis], dim=1)


def _bank(d: int, K: int, V: torch.Tensor | None = None, eta0: float = 0.5):
    from src.models.filter_bank import ChebyshevFilterBank

    bank = ChebyshevFilterBank(d, K, eta0).double()
    bank.reset_parameters()
    if V is not None:
        with torch.no_grad():
            bank.V.copy_(V)
    return bank


def _filter_matrix(system, listen, broadcast, bank) -> torch.Tensor:
    """The production filter (matrix-free, production `A`), materialized column by column."""
    from src.models.h2snn_model import apply_operator

    operator = system.operator(listen, broadcast)
    return _materialize(lambda h: bank(lambda x: apply_operator(operator, x), h),
                        system.num_nodes, system.d)


def _production_a(system, listen, broadcast) -> torch.Tensor:
    from src.models.h2snn_model import apply_operator

    operator = system.operator(listen, broadcast)
    return _materialize(lambda x: apply_operator(operator, x), system.num_nodes, system.d)


def _dense_a(system, listen, broadcast) -> torch.Tensor:
    """Independent `A = G Delta_hat B` from the dense reference `Delta_hat` (no production code)."""
    rows = listen.repeat_interleave(system.d).view(-1, 1)
    columns = broadcast.repeat_interleave(system.d).view(1, -1)
    return rows * system.dense_delta_hat() * columns


def _m_gain(matrix: torch.Tensor, metric: torch.Tensor, d: int) -> float:
    root = metric.repeat_interleave(d).sqrt()
    return float(torch.linalg.matrix_norm(root.view(-1, 1) * matrix / root.view(1, -1), ord=2))


def _random_v(K: int, d: int, generator: torch.Generator) -> torch.Tensor:
    """Random blocks with sum ||V_k||_2 spread over [0.3, 5] (normalization active above 1)."""
    V = torch.randn(K + 1, d, d, generator=generator, dtype=torch.float64)
    target = 0.3 + 4.7 * float(torch.rand(1, generator=generator, dtype=torch.float64))
    return V * target / torch.linalg.matrix_norm(V, ord=2).sum()


class _Unnormalized:
    """Control: the same bank with `W_k = V_k` (normalization removed)."""

    def __init__(self, bank) -> None:
        self.bank = bank

    def __call__(self, apply_a, h):
        original = self.bank.weights
        self.bank.weights = lambda: self.bank.V
        try:
            return self.bank(apply_a, h)
        finally:
            self.bank.weights = original


class FilterBankOperatorTests(unittest.TestCase):
    FIXTURES = 50

    @staticmethod
    def _system(seed: int):
        from tests.test_cooperative import System, _random_gates

        system = System(3000 + seed, shared=seed % 2 == 0)
        listen, broadcast = _random_gates(system)
        return system, listen, broadcast

    def test_TG1_spectrum_of_A_is_real_and_in_0_2_max_gb(self) -> None:
        worst_imag, worst_low, worst_high = 0.0, 0.0, 0.0
        for seed in range(self.FIXTURES + 10):
            system, g, b = self._system(seed)
            eigenvalues = torch.linalg.eigvals(_production_a(system, g, b))
            bound = 2 * float((g * b).max())
            worst_imag = max(worst_imag, float(eigenvalues.imag.abs().max()))
            worst_low = max(worst_low, float(-eigenvalues.real.min()))
            worst_high = max(worst_high, float(eigenvalues.real.max()) - bound)
        self.assertLessEqual(worst_imag, 1e-9)
        self.assertLessEqual(worst_low, 1e-9)
        self.assertLessEqual(worst_high, 1e-9)

    def test_TG2_gain_in_metric_M_is_at_most_one_with_control(self) -> None:
        generator = torch.Generator().manual_seed(11)
        worst, control_exceeds = 0.0, 0
        for seed in range(self.FIXTURES):
            system, g, b = self._system(seed)
            K = 2 + seed % 3
            bank = _bank(system.d, K, _random_v(K, system.d, generator))
            metric = b / g
            worst = max(worst, _m_gain(_filter_matrix(system, g, b, bank), metric, system.d))
            raw = _materialize(lambda h, o=system.operator(g, b): _Unnormalized(bank)(
                lambda x: _apply(o, x), h), system.num_nodes, system.d)
            control_exceeds += _m_gain(raw, metric, system.d) > 1.0 + 1e-9
        self.assertLessEqual(worst, 1.0 + 1e-9)
        self.assertGreater(control_exceeds, 0, "control: without normalization the gain exceeds 1")

    def test_TG3_cooperation_through_the_filter_exact_zero_blocks(self) -> None:
        generator = torch.Generator().manual_seed(5)
        for seed in range(10):
            system, g, b = self._system(seed)
            n, d, K = system.num_nodes, system.d, 3
            bank = _bank(d, K, _random_v(K, d, generator))
            u = seed % n
            listen = g.clone()
            listen[u] = 0.0
            F_listen = _filter_matrix(system, listen, b, bank)
            broadcast = b.clone()
            broadcast[u] = 0.0
            F_broadcast = _filter_matrix(system, g, broadcast, bank)
            half = torch.full_like(g, 0.5)
            F_half = _filter_matrix(system, half, half, bank)
            rows, cols = slice(u * d, (u + 1) * d), slice(u * d, (u + 1) * d)
            others = [j for j in range(n) if j != u]
            for j in others:
                block = slice(j * d, (j + 1) * d)
                self.assertTrue(bool((F_listen[rows, block] == 0).all()), f"g_u = 0: block {u}<-{j}")
                self.assertTrue(bool((F_broadcast[block, cols] == 0).all()), f"b_v = 0: block {j}<-{u}")
            control = max(float(F_half[rows, slice(j * d, (j + 1) * d)].abs().max()) for j in others)
            self.assertGreater(control, 1e-8, "control: gates 0.5 couple u to the others")

    def test_TG5_lemma_N_diagonal_blocks_are_identity(self) -> None:
        checked = 0
        for seed in range(self.FIXTURES):
            system, _, _ = self._system(seed)
            n, d = system.num_nodes, system.d
            delta = system.production_delta_hat()
            edge_index = system.paths.edge_index
            excluded = set(edge_index[0][edge_index[0] == edge_index[1]].tolist())  # self-loop facts
            for nodes in system.paths.nodes:  # a node repeated inside one cell
                for row in nodes.tolist():
                    excluded |= {v for v in row if row.count(v) > 1}
            degree = system.laplacian.degree()
            for v in range(n):
                if v in excluded or float(degree[v]) == 0.0:
                    continue
                block = delta[v * d:(v + 1) * d, v * d:(v + 1) * d]
                self.assertLessEqual(float((block - torch.eye(d, dtype=torch.float64)).abs().max()),
                                     1e-12, f"fixture {seed}, node {v}")
                checked += 1
        self.assertGreater(checked, 100)

    def test_TG6_matrix_free_filter_equals_the_dense_reference(self) -> None:
        generator = torch.Generator().manual_seed(17)
        for seed in range(20):
            system, g, b = self._system(seed)
            n, d, K = system.num_nodes, system.d, 1 + seed % 5
            bank = _bank(d, K, _random_v(K, d, generator))
            shift = _dense_a(system, g, b) - torch.eye(n * d, dtype=torch.float64)
            weights = bank.weights().detach()
            powers = [torch.eye(n * d, dtype=torch.float64), shift]
            for _ in range(2, K + 1):
                powers.append(2 * shift @ powers[-1] - powers[-2])
            eye_n = torch.eye(n, dtype=torch.float64)
            reference = sum(powers[k] @ torch.kron(eye_n, weights[k].T.contiguous())
                            for k in range(K + 1))
            error = float((_filter_matrix(system, g, b, bank) - reference).abs().max())
            self.assertLessEqual(error, 1e-10, f"fixture {seed}, K = {K}")

    def test_gain_diagnostic_matches_the_exact_M_gain(self) -> None:
        from src.models.h2snn_model import apply_operator

        generator = torch.Generator().manual_seed(23)
        for seed in range(5):
            system, g, b = self._system(seed)
            bank = _bank(system.d, 3, _random_v(3, system.d, generator))
            exact = _m_gain(_filter_matrix(system, g, b, bank), b / g, system.d)
            operator = system.operator(g, b)
            estimate = bank.gain(lambda x: apply_operator(operator, x), b / g,
                                 (system.num_nodes, system.d), steps=500)
            self.assertLessEqual(estimate, exact + 1e-9)
            self.assertGreater(estimate, exact - 1e-4)


def _apply(operator, x):
    from src.models.h2snn_model import apply_operator

    return apply_operator(operator, x)


class FilterBankModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = _float64_pap()

    def _x_star(self, model) -> torch.Tensor:
        with torch.no_grad():
            return model.encode(self.data.x, self.data.node_type, self.data.edge_index,
                                self.data.edge_type, self.data.face_groups, self.data.num_nodes).x_star

    def test_TG4_d11_initialization_equals_first_order(self) -> None:
        import math

        for eta in (0.25, 0.5, 0.9):
            for K in (2, 5):
                with self.subTest(eta=eta, K=K):
                    bank = _v2_model(self.data, eta_init=eta, K=K).double().eval()
                    bank.filter_bank.reset_parameters()  # theta_k in float64
                    first = _v2_model(self.data, eta_init=eta, K=K,
                                      encoder="first_order").double().eval()
                    with torch.no_grad():
                        first.theta_eta.copy_(torch.tensor(math.log(eta / (1 - eta)),
                                                           dtype=torch.float64))
                    shared = {k: v for k, v in bank.state_dict().items() if k in first.state_dict()}
                    self.assertEqual(set(first.state_dict()) - set(shared), {"theta_eta"})
                    first.load_state_dict(shared, strict=False)
                    error = float((self._x_star(bank) - self._x_star(first)).abs().max())
                    self.assertLessEqual(error, 1e-10)

    def test_TG8_gradients_reach_every_V_k_gates_gamma_and_map_mlps(self) -> None:
        model = _v2_model(self.data).double().train()
        output = model(self.data)
        output.logits.square().sum().backward()
        V_grad = model.filter_bank.V.grad
        for k in range(V_grad.size(0)):
            self.assertGreater(float(V_grad[k].abs().sum()), 0.0, f"V_{k}")
        for name, parameter in (("listen gate", model.listen_gate.weight),
                                ("broadcast gate", model.broadcast_gate.weight),
                                ("gamma", model.theta_gamma),
                                ("reflector MLP", model.node_to_edge.reflector_mlp[0].weight),
                                ("edge-scale MLP", model.node_to_edge.edge_scale_mlp[0].weight)):
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)
        context_columns = model.node_to_edge.reflector_mlp[0].weight.grad[:, model.d:model.geometry_dim]
        self.assertGreater(float(context_columns.abs().sum()), 0.0, "map MLP columns reading c")

    def test_eta_only_under_first_order_and_V_in_the_normal_group(self) -> None:
        from src.builders import build_optimizer
        from tests.test_v2_model import _v2_config

        bank = _v2_model(self.data)
        self.assertIsNone(bank.theta_eta)
        self.assertIsNone(bank.eta())
        first = _v2_model(self.data, encoder="first_order")
        self.assertIsNone(first.filter_bank)
        optimizer = build_optimizer(bank, _v2_config())
        scalar = [g for g in optimizer.param_groups if g["weight_decay"] == 0.0]
        self.assertFalse(any(p is bank.filter_bank.V for g in scalar for p in g["params"]))
        self.assertTrue(any(p is bank.theta_gamma for g in scalar for p in g["params"]))

    def test_operator_diagnostics_under_the_filter_bank(self) -> None:
        model = _v2_model(self.data).double()
        with torch.no_grad():
            model.filter_bank.V.mul_(3.0)  # normalization active
        stats = model.operator_diagnostics(self.data.x, self.data.node_type, self.data.edge_index,
                                           self.data.edge_type, self.data.face_groups,
                                           self.data.num_nodes, steps=200)
        self.assertEqual(set(stats), {"lambda_max_A", "filter_norm_sum", "filter_gain"})
        self.assertLessEqual(stats["filter_norm_sum"], 1.0 + 1e-9)
        self.assertLessEqual(stats["filter_gain"], stats["filter_norm_sum"] + 1e-9)
        self.assertLessEqual(stats["lambda_max_A"], 2.0)
        first = _v2_model(self.data, encoder="first_order")
        self.assertEqual(set(first.operator_diagnostics(
            self.data.x.float(), self.data.node_type, self.data.edge_index, self.data.edge_type,
            self.data.face_groups, self.data.num_nodes)), {"eta_lambda_max"})

    def test_switches_are_never_tuned(self) -> None:
        from src.configs.config import reject_never_tuned_settings

        for raw in ({"model": {"encoder": "first_order"}}, {"model": {"geometry_context": False}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                reject_never_tuned_settings(raw)
        for name in ("model.encoder", "encoder", "model.geometry_context"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                reject_never_tuned_settings(None, {name: {"values": [1]}})
        reject_never_tuned_settings(None, {"model.encoder_dropout": {"values": [0.1]}})  # allowed
        reject_never_tuned_settings({"model": {"encoder": "filter_bank", "geometry_context": True}})


if __name__ == "__main__":
    unittest.main()
