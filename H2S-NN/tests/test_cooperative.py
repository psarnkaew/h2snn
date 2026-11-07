"""D10: cooperative coupling with a real spectrum (plan §10, task card Part E).

Test ids follow the card (T1-T6). Systems are built from explicit maps and signed path cells
(mixed lengths, mixed hop signs), with an independent dense reference.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.configs.config import TrainConfig
from src.models.h2snn_model import DiffusionOperator, H2SNN, ModelDims
from src.models.map_operators import EdgeMapPair, PathTransportOperator
from src.models.vertex_path_laplacian import PathCellTerm, VertexPathSheafLaplacian
from tests.test_paths_and_facts import (
    SectionPreservationTests,
    SignedPaths,
    _dense_cells,
    _moderate_scales,
    _signed_dense_transports,
)
from tests.test_v2_model import _chain_operators

F64_TOL = 1e-10
REPO = Path(__file__).resolve().parents[1]
GOLDEN_SCRIPT = REPO.parent / "docs" / "heads_evidence" / "v2_golden.py"
GOLDEN_REFERENCE = REPO.parent / "docs" / "heads_evidence" / "v2_golden.pt"
ETA = 0.5
GATE_PREACTIVATION = 4.59511985013459  # logit(0.99): sigmoid(GATE_PREACTIVATION * u), u in [-1, 1]


# ----------------------------------------------------------------------------- fixtures
def _gate_model(coupling: str = "cooperative", d: int = 4) -> H2SNN:
    """A real H2SNN (float64, eval, K = 1); only its gate heads and `diffuse` are used."""
    config = TrainConfig.from_dict({
        "device": "cpu",
        "model": {"d": d, "d_r": 2, "K": 1, "classifier_hidden_dim": 8, "face_mlp_hidden_dim": 8,
                  "edge_mlp_hidden_dim": 8, "layer_dropout": 0.0},
        "ablation": {"coupling": coupling},
    })
    dims = ModelDims(in_channels=5, num_node_types=2, num_relations=2, num_classes=2,
                     face_lengths=(2, 3, 4))
    return H2SNN(config, dims).double().eval()


def _span_gate_heads(model: H2SNN) -> None:
    """Heads reading column 0 of h0 so that gate = sigmoid(logit(0.99) * u) spans [0.01, 0.99]
    for u in [-1, 1]; column 1 drives the broadcast head with the opposite sign pattern."""
    for head, column, sign in ((model.listen_gate, 0, 1.0), (model.broadcast_gate, 1, 1.0)):
        if head is None:
            continue
        with torch.no_grad():
            head.weight.zero_()
            head.weight[0, column] = sign * GATE_PREACTIVATION
            head.bias.zero_()


class TreePaths:
    """The signed tree cells of `SectionPreservationTests._tree` in the interface `System` reads
    from `SignedPaths` (a tree has genuine sections, unlike random overlapping paths)."""

    def __init__(self, edge_index: torch.Tensor, groups) -> None:
        self.edge_index = edge_index
        self.nodes, self.edges, self.signs = ([g[i] for g in groups] for i in range(3))

    def transports(self, maps: EdgeMapPair, shared: bool) -> list[PathTransportOperator]:
        return [PathTransportOperator(maps.src, maps.tgt, nodes, edges, self.edge_index,
                                      shared_scale=shared, face_edge_sign=signs)
                for nodes, edges, signs in zip(self.nodes, self.edges, self.signs)]


class System:
    """Explicit maps + signed mixed-length path cells: the production Laplacian and an
    independent dense `Delta_hat = D^-1/2 (Delta_0 + gamma L_P) D^-1/2`."""

    def __init__(self, seed: int, shared: bool = True, d: int = 4, num_nodes: int = 9,
                 lengths: tuple[int, ...] = (2, 3, 4), gamma: float = 0.8,
                 maps_fn=None, paths=None) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.num_nodes, self.d, self.gamma = num_nodes, d, gamma
        self.paths = paths or SignedPaths(num_nodes, lengths, 3, generator, signed=True)
        edge_index = self.paths.edge_index
        maps = _chain_operators(edge_index.size(1), k=2, d=d, seed=seed, dtype=torch.float64,
                                shared=shared)
        self.maps = maps_fn(maps, generator, shared) if maps_fn else maps
        self.weights = [torch.rand(n.size(0), generator=generator, dtype=torch.float64) + 0.2
                        for n in self.paths.nodes]
        terms = tuple(PathCellTerm(t, w) for t, w in
                      zip(self.paths.transports(self.maps, shared), self.weights))
        self.laplacian = VertexPathSheafLaplacian(
            self.maps, edge_index, num_nodes, terms, torch.tensor(gamma, dtype=torch.float64),
            checkpoint=False)
        self.generator = generator

    # -- independent dense reference (no production operator code)
    def dense_delta_hat(self) -> torch.Tensor:
        n, d = self.num_nodes, self.d
        F_src, F_tgt = self.maps.src.materialize(), self.maps.tgt.materialize()
        edge_index = self.paths.edge_index
        delta0 = torch.zeros(n * d, n * d, dtype=torch.float64)
        degree = torch.zeros(n, dtype=torch.float64)
        for e in range(edge_index.size(1)):
            u, v = int(edge_index[0, e]), int(edge_index[1, e])
            block = torch.zeros(d, n * d, dtype=torch.float64)
            block[:, u * d:(u + 1) * d] -= F_src[e]
            block[:, v * d:(v + 1) * d] += F_tgt[e]
            delta0 += block.T @ block
            degree[u] += self.maps.src.scale[e] ** 2
            degree[v] += self.maps.tgt.scale[e] ** 2
        l_p = _dense_cells(self.maps, self.paths.nodes, self.paths.edges, self.paths.signs,
                           self.weights, n, d)
        for nodes, edges, signs, w in zip(self.paths.nodes, self.paths.edges, self.paths.signs,
                                          self.weights):
            size = nodes.size(1)
            squares = _signed_dense_transports(self.maps, edges, signs).square().sum(
                dim=(-2, -1)) / d
            part = self.gamma * (1 - 1 / size) * w.unsqueeze(1) * squares
            degree.index_add_(0, nodes.reshape(-1), part.reshape(-1))
        scale = torch.where(degree > 0, degree.clamp_min(1e-300).rsqrt(),
                            torch.zeros_like(degree)).repeat_interleave(d)
        return scale.view(-1, 1) * (delta0 + self.gamma * l_p) * scale.view(1, -1)

    def production_delta_hat(self) -> torch.Tensor:
        basis = torch.eye(self.num_nodes * self.d, dtype=torch.float64).view(-1, self.num_nodes,
                                                                             self.d)
        return torch.stack([self.laplacian.apply_normalized(b).reshape(-1) for b in basis], 1)

    def operator(self, listen, broadcast) -> DiffusionOperator:
        return DiffusionOperator(
            laplacian=self.laplacian, eta=torch.tensor(ETA, dtype=torch.float64), gamma=None,
            path_weights=(), maps=self.maps, listen=listen, broadcast=broadcast)


def _random_gates(system: System, low: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    g = low + (1 - low) * torch.rand(system.num_nodes, generator=system.generator,
                                     dtype=torch.float64)
    b = low + (1 - low) * torch.rand(system.num_nodes, generator=system.generator,
                                     dtype=torch.float64)
    return g, b


def _step_matrix(model: H2SNN, operator: DiffusionOperator, num_nodes: int, d: int) -> torch.Tensor:
    """`M = (I - Phi) / eta` with `Phi` the matrix of one production `diffuse` step: the
    effective operator `G Delta_hat B` (columns by basis vectors)."""
    basis = torch.eye(num_nodes * d, dtype=torch.float64).view(-1, num_nodes, d)
    columns = [(b - model.diffuse(b, operator)).reshape(-1) / float(operator.eta) for b in basis]
    return torch.stack(columns, dim=1)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1.0))


# ----------------------------------------------------------------------------- T1
class SymmetricIsThePreviousModelTests(unittest.TestCase):
    def test_T1_symmetric_reproduces_the_golden_outputs_bit_for_bit(self) -> None:
        spec = importlib.util.spec_from_file_location("v2_golden", GOLDEN_SCRIPT)
        golden = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(golden)  # type: ignore[union-attr]
        self.assertEqual(golden.build_config({}, ()).ablation.coupling, "symmetric")
        compared, mismatches, messages = golden.compare(GOLDEN_REFERENCE)
        self.assertEqual(mismatches, 0, messages[:5])
        self.assertEqual(compared, 844)

    def test_T1_control_the_default_model_is_not_the_golden_one(self) -> None:
        """Non-vacuity: with the default (cooperative) config the golden key set differs."""
        spec = importlib.util.spec_from_file_location("v2_golden", GOLDEN_SCRIPT)
        golden = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(golden)  # type: ignore[union-attr]
        reference = torch.load(GOLDEN_REFERENCE, weights_only=False)
        entry = reference["toy_acm"]
        data = golden._deserialize(entry["data"])
        config = TrainConfig.from_dict({"device": "cpu", "model": dict(golden.SMALL)})
        self.assertEqual(config.ablation.coupling, "cooperative")
        record = golden.run_case.__globals__["build_model"](config, data)
        extra = {k for k in record.state_dict() if k.startswith(("listen_gate", "broadcast_gate"))}
        self.assertEqual(extra, {"listen_gate.weight", "listen_gate.bias",
                                 "broadcast_gate.weight", "broadcast_gate.bias"})
        self.assertTrue(all(f"init/{k}" not in entry["record"] for k in extra))

    def test_modes_instantiate_the_right_heads(self) -> None:
        for coupling, listen, broadcast in (("cooperative", True, True),
                                            ("listen_only", True, False),
                                            ("symmetric", False, False)):
            model = _gate_model(coupling)
            self.assertEqual((model.listen_gate is not None, model.broadcast_gate is not None),
                             (listen, broadcast))
        for bad in ("directed", True):
            with self.assertRaises(ValueError):
                TrainConfig.from_dict({"ablation": {"coupling": bad}})
        for bad in (0.0, 1.0, -0.1):
            with self.assertRaises(ValueError):
                TrainConfig.from_dict({"model": {"coupling_gate_init": bad}})
        from src.configs.config import reject_never_tuned_settings

        for name in ("coupling", "ablation.coupling", "coupling_gate_init"):
            with self.assertRaises(ValueError):
                reject_never_tuned_settings(None, {name: {"values": [0.5]}})
        # the real search spaces and notebooks contain neither
        from scripts.run_h2snn_wandb_tuning import KNOWN_PARAMETER_NAMES, build_sweep_config

        self.assertFalse(any("coupling" in name for name in KNOWN_PARAMETER_NAMES))
        for task in ("node_classification", "link_prediction", "graph_classification"):
            sweep = build_sweep_config(task_type=task, metric="val_acc", epochs=50)
            self.assertFalse(any("coupling" in key for key in sweep["parameters"]))
        for notebook in sorted((REPO / "notebooks").glob("*tuning*.ipynb")):
            self.assertNotIn("coupling", notebook.read_text(encoding="utf-8"))

    def test_gates_start_at_the_configured_init(self) -> None:
        config = TrainConfig.from_dict({"device": "cpu", "model": {
            "d": 8, "coupling_gate_init": 0.9, "K": 1}})
        model = H2SNN(config, ModelDims(in_channels=5, num_node_types=2, num_relations=2,
                                        num_classes=2, face_lengths=(2,)))
        listen, broadcast = model.coupling_gates(torch.randn(50, 8))
        for gate in (listen, broadcast):
            self.assertLess(float((gate - 0.9).abs().max()), 0.05)
            self.assertGreater(float(gate.std()), 0.0)  # small random weights, not constant


# ----------------------------------------------------------------------------- T2
class GatedOperatorTests(unittest.TestCase):
    def test_T2_gated_step_equals_the_dense_reference(self) -> None:
        for shared in (True, False):
            for seed in range(3):
                system = System(seed, shared=shared)
                dense = system.dense_delta_hat()
                self.assertLess(_rel(system.production_delta_hat(), dense), F64_TOL)
                g, b = _random_gates(system, low=0.05)
                x = torch.randn(system.num_nodes, system.d, generator=system.generator,
                                dtype=torch.float64)
                ones = torch.ones_like(g)
                for mode, gates in (("cooperative", (g, b)), ("listen_only", (g, None)),
                                    ("symmetric", (None, None))):
                    with self.subTest(shared=shared, seed=seed, mode=mode):
                        model = _gate_model(mode)
                        G = torch.diag((ones if gates[0] is None else gates[0])
                                       .repeat_interleave(system.d))
                        B = torch.diag((ones if gates[1] is None else gates[1])
                                       .repeat_interleave(system.d))
                        reference = x.reshape(-1) - ETA * (G @ dense @ B @ x.reshape(-1))
                        got = model.diffuse(x, system.operator(*gates)).reshape(-1)
                        self.assertLess(_rel(got, reference), F64_TOL)

    def test_T2_the_fixtures_mix_lengths_and_signs(self) -> None:
        system = System(0)
        self.assertEqual(sorted({n.size(1) - 1 for n in system.paths.nodes}), [2, 3, 4])
        self.assertTrue(any(bool((s < 0).any()) for s in system.paths.signs))
        self.assertTrue(any(bool((s > 0).any()) for s in system.paths.signs))


# ----------------------------------------------------------------------------- T3
def _csnn_composed_operator(num_nodes: int, d: int, pairs: list[tuple[int, int]],
                            generator: torch.Generator) -> torch.Tensor:
    """CSNN's operator `(Delta_in)^T Delta_out` (VERIFIED-SOURCE: references/cooperative_sheaf_
    neural_network.pdf Eq. 6-8) with flat vector bundles: node-level conformal maps S_i, T_i
    (orthogonal times a positive scalar), undirected edges as two directed ones,
        L_out x_i = sum_j (S_i^T S_i x_i - T_i^T S_j x_j),
        (L_in)^T x_i = sum_j (T_i^T T_i x_i - T_i^T S_j x_j),
    each normalised by its block-diagonal (a scalar times I for conformal maps)."""
    def conformal():
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=generator, dtype=torch.float64))
        return (0.5 + torch.rand(1, generator=generator, dtype=torch.float64)) * q

    S = [conformal() for _ in range(num_nodes)]
    T = [conformal() for _ in range(num_nodes)]
    neighbours = {i: [] for i in range(num_nodes)}
    for a, b in pairs:
        neighbours[a].append(b)
        neighbours[b].append(a)
    L_out = torch.zeros(num_nodes * d, num_nodes * d, dtype=torch.float64)
    L_in_T = torch.zeros_like(L_out)
    for i in range(num_nodes):
        for j in neighbours[i]:
            si = slice(i * d, (i + 1) * d)
            sj = slice(j * d, (j + 1) * d)
            L_out[si, si] += S[i].T @ S[i]
            L_out[si, sj] -= T[i].T @ S[j]
            L_in_T[si, si] += T[i].T @ T[i]
            L_in_T[si, sj] -= T[i].T @ S[j]

    def normalise(L):
        diagonal = torch.stack([L[i * d, i * d] for i in range(num_nodes)])
        scale = torch.where(diagonal > 0, diagonal.clamp_min(1e-300).rsqrt(),
                            torch.zeros_like(diagonal)).repeat_interleave(d)
        return scale.view(-1, 1) * L * scale.view(1, -1)

    return normalise(L_in_T) @ normalise(L_out)


class SpectrumTests(unittest.TestCase):
    FIXTURES = 60

    def _gates_through_the_heads(self, model: H2SNN, system: System):
        """Gates from the production heads: h0 columns 0, 1 uniform in [-1, 1] so each gate
        spans [0.01, 0.99] (the heads are set to sigmoid(logit(0.99) u))."""
        h0 = torch.zeros(system.num_nodes, model.d, dtype=torch.float64)
        h0[:, :2] = 2 * torch.rand(system.num_nodes, 2, generator=system.generator,
                                   dtype=torch.float64) - 1
        return model.coupling_gates(h0)

    def _measure(self, model: H2SNN, fixtures: int):
        worst_imag, worst_ratio, margin = 0.0, 0.0, -1e9
        lowest, gate_low, gate_high = 0.0, 1.0, 0.0
        for seed in range(fixtures):
            system = System(1000 + seed, shared=seed % 2 == 0)
            listen, broadcast = self._gates_through_the_heads(model, system)
            gate_low = min(gate_low, float(torch.minimum(listen, broadcast).min()))
            gate_high = max(gate_high, float(torch.maximum(listen, broadcast).max()))
            matrix = _step_matrix(model, system.operator(listen, broadcast), system.num_nodes,
                                  system.d)
            eigenvalues = torch.linalg.eigvals(matrix)
            worst_imag = max(worst_imag, float(eigenvalues.imag.abs().max()
                                               / eigenvalues.abs().max().clamp_min(1e-300)))
            bound = 2.0 * float((listen * broadcast).max())
            margin = max(margin, float(eigenvalues.real.max()) - bound)
            lowest = min(lowest, float(eigenvalues.real.min()))
            worst_ratio = max(worst_ratio, float(eigenvalues.real.max()) / bound)
        return worst_imag, margin, worst_ratio, lowest, (gate_low, gate_high)

    def test_T3_real_spectrum_and_bound_with_gates(self) -> None:
        model = _gate_model("cooperative")
        _span_gate_heads(model)
        imag, margin, ratio, lowest, (gate_low, gate_high) = self._measure(model, self.FIXTURES)
        # the theory of plan §10.3: gates in (0, 1), spectrum real and inside [0, 2 max(g b)]
        self.assertGreater(gate_low, 0.0)
        self.assertLess(gate_high, 1.0)
        self.assertLess(gate_low, 0.02)  # the fixtures do reach small gates
        self.assertLessEqual(imag, 1e-9)
        self.assertLessEqual(margin, 1e-9)  # lambda_max <= 2 max(g b) + 1e-9
        self.assertGreaterEqual(lowest, -1e-9)  # PSD: no negative eigenvalue
        self.assertGreater(ratio, 0.05)  # the bound is not met by a trivially tiny spectrum

    def test_T3_negative_control_csnn_composed_operator_is_complex(self) -> None:
        worst = 0.0
        for seed in range(self.FIXTURES):
            system = System(1000 + seed)
            pairs = [(int(a), int(b)) for a, b in system.paths.edge_index.t().tolist()]
            matrix = _csnn_composed_operator(system.num_nodes, system.d, pairs, system.generator)
            eigenvalues = torch.linalg.eigvals(matrix)
            worst = max(worst, float(eigenvalues.imag.abs().max()))
        self.assertGreater(worst, 1e-3)


# ----------------------------------------------------------------------------- T4
class CooperativeBehaviourTests(unittest.TestCase):
    def _jacobian(self, system: System, listen, broadcast) -> torch.Tensor:
        model = _gate_model("cooperative")
        operator = system.operator(listen, broadcast)
        x0 = torch.randn(system.num_nodes * system.d, generator=system.generator,
                         dtype=torch.float64)
        return torch.autograd.functional.jacobian(
            lambda z: model.diffuse(z.view(system.num_nodes, system.d), operator).reshape(-1), x0)

    @staticmethod
    def _block(jacobian: torch.Tensor, row: int, column: int, d: int) -> torch.Tensor:
        return jacobian[row * d:(row + 1) * d, column * d:(column + 1) * d]

    def setUp(self) -> None:
        self.system = System(5)
        edge = self.system.paths.edge_index[:, 0]
        self.u, self.v = int(edge[0]), int(edge[1])
        self.n, self.d = self.system.num_nodes, self.system.d

    def test_T4_a_node_with_zero_listen_gate_is_unchanged_exactly(self) -> None:
        g = torch.full((self.n,), 0.7, dtype=torch.float64)
        b = torch.full((self.n,), 0.8, dtype=torch.float64)
        g[self.u] = 0.0
        model = _gate_model("cooperative")
        x = torch.randn(self.n, self.d, generator=self.system.generator, dtype=torch.float64)
        x1 = model.diffuse(x, self.system.operator(g, b))
        self.assertTrue(torch.equal(x1[self.u], x[self.u]))
        others = [w for w in range(self.n) if w != self.u]
        self.assertGreater(float((x1[others] - x[others]).abs().max()), 1e-8)  # control: others move

    def test_T4_a_node_with_zero_broadcast_gate_influences_nobody(self) -> None:
        g = torch.full((self.n,), 0.7, dtype=torch.float64)
        b = torch.full((self.n,), 0.8, dtype=torch.float64)
        b[self.v] = 0.0
        jacobian = self._jacobian(self.system, g, b)
        for w in range(self.n):
            if w != self.v:
                self.assertEqual(float(self._block(jacobian, w, self.v, self.d).abs().max()), 0.0)
        # control: all gates 0.5 -> the blocks of the neighbours of v are non-zero
        half = torch.full((self.n,), 0.5, dtype=torch.float64)
        control = self._jacobian(self.system, half, half)
        self.assertGreater(float(self._block(control, self.u, self.v, self.d).abs().max()), 1e-8)

    def test_T4_unequal_gates_make_the_jacobian_blocks_asymmetric(self) -> None:
        g = torch.full((self.n,), 0.8, dtype=torch.float64)
        b = torch.full((self.n,), 0.8, dtype=torch.float64)
        g[self.u], g[self.v] = 0.9, 0.3
        jacobian = self._jacobian(self.system, g, b)
        forward = self._block(jacobian, self.u, self.v, self.d)
        backward = self._block(jacobian, self.v, self.u, self.d).T
        self.assertGreater(float((forward - backward).abs().max()), 1e-8)
        half = torch.full((self.n,), 0.5, dtype=torch.float64)
        control = self._jacobian(self.system, half, half)
        symmetric = self._block(control, self.u, self.v, self.d) - self._block(
            control, self.v, self.u, self.d).T
        self.assertLess(float(symmetric.abs().max()), 1e-12)


# ----------------------------------------------------------------------------- T5
class FixedPointTests(unittest.TestCase):
    CUT = 1e-9  # eigenvalue cut for "zero", stated in docs/D10_evidence/pre_registration.md

    def test_T5_fixed_points_are_the_sections_seen_through_the_broadcast_gates(self) -> None:
        tree_tests = SectionPreservationTests()
        d, n = 3, 8
        for seed in range(4):
            edge_index, groups = tree_tests._tree(seed)
            paths = TreePaths(edge_index, groups)
            system = System(seed, shared=seed % 2 == 0, d=d, num_nodes=n, gamma=0.7,
                            maps_fn=_moderate_scales, paths=paths)
            model = _gate_model("cooperative", d=d)
            g = 0.1 + 0.9 * torch.rand(n, generator=system.generator, dtype=torch.float64)
            b = 0.1 + 0.9 * torch.rand(n, generator=system.generator, dtype=torch.float64)
            operator = system.operator(g, b)
            delta = system.production_delta_hat()
            values, vectors = torch.linalg.eigh(0.5 * (delta + delta.T))
            kernel = vectors[:, values <= self.CUT]
            self.assertEqual(kernel.size(1), d)  # a tree: one d-dimensional space of sections
            self.assertGreater(float(values[kernel.size(1)]), 1e3 * self.CUT)
            repeat_b = b.repeat_interleave(d)
            for column in range(kernel.size(1)):
                section = kernel[:, column]
                x = (section / repeat_b).view(n, d)  # x = B^-1 s
                residual = (x - model.diffuse(x, operator)).reshape(-1) / ETA  # G Delta B x
                self.assertLess(float(residual.abs().max() / x.abs().max()), self.CUT)
            # nothing else: the kernel of G Delta B has the same dimension (similar to
            # (GB)^{1/2} Delta (GB)^{1/2}), so B^-1 ker Delta is all of it
            scale = (g * b).sqrt().repeat_interleave(d)
            similar = scale.view(-1, 1) * delta * scale.view(1, -1)
            similar_values = torch.linalg.eigvalsh(0.5 * (similar + similar.T))
            self.assertEqual(int((similar_values <= self.CUT).sum()), d)
            matrix = _step_matrix(model, operator, n, d)
            rank = int((torch.linalg.svdvals(matrix) > 1e-7).sum())
            self.assertEqual(n * d - rank, d)  # control through the production step itself


# ----------------------------------------------------------------------------- T6
class GateGradientTests(unittest.TestCase):
    def test_T6_gradients_reach_both_gate_heads(self) -> None:
        from tests.test_paths_and_facts import _prepared_small

        config = TrainConfig.from_dict({
            "device": "cpu", "model": {"d": 4, "d_r": 2, "K": 2, "classifier_hidden_dim": 8,
                                       "face_mlp_hidden_dim": 8, "edge_mlp_hidden_dim": 8},
            "data": {"dataset_name": "acm", "fact_layer": "tied", "path_sampler": "walk",
                     "face_sample_k": 3,
                     "metapath_templates": [{"name": "PAP",
                                             "node_types": ["paper", "author", "paper"]}]}})
        data = _prepared_small(config)
        for mode, expected in (("cooperative", ("listen_gate", "broadcast_gate")),
                               ("listen_only", ("listen_gate",))):
            with self.subTest(mode=mode):
                config.ablation.coupling = mode
                torch.manual_seed(0)
                from src.models.h2snn_model import build_model

                model = build_model(config, data)
                model.train()
                output = model(data)
                (output.logits.square().sum() + output.x_star.square().mean()).backward()
                for name in expected:
                    head = getattr(model, name)
                    self.assertGreater(float(head.weight.grad.abs().max()), 0.0, name)
                    self.assertGreater(float(head.bias.grad.abs().max()), 0.0, name)
                self.assertEqual(sorted(model.last_coupling_stats),
                                 sorted(["g_mean", "g_min"] + (["b_mean", "b_min"]
                                        if mode == "cooperative" else [])))
                self.assertIn("g_mean", model.diffusion_scalars())


# ----------------------------------------------------------------------------- T-F1 (D11 learning fix)
def _signal_fixture():
    """Paper/author toy graph whose labels need diffusion: a paper's class is the argmax of its
    neighbour authors' mean features (first 3 columns), so the readout of h0 alone cannot
    reach it and gradients flow into eta and the gate heads."""
    from tests.test_factorized_maps import _random_pap_data
    from tests.test_v2_model import _with

    data = _random_pap_data()
    papers = int(data.target_indices.numel())
    mean = torch.zeros(papers, 3)
    count = torch.zeros(papers, 1)
    for u, v in data.edge_index.t().tolist():
        if u < papers <= v:
            mean[u] += data.x[v, :3]
            count[u] += 1
    labels = mean.div(count.clamp_min(1)).argmax(dim=1)
    y = data.y.clone()
    y[:papers] = labels
    return _with(data, target_y=labels, y=y)


def _train_toy(gate_init: float | None, grouped: bool, seed: int = 0, epochs: int = 50):
    """`epochs` full-batch steps; returns the moves of eta, mean |dg| and mean |db| (per node, so
    gates moving up and down do not cancel) and the losses."""
    import torch.nn.functional as F

    from src.builders import build_optimizer
    from src.models.h2snn_model import build_model

    data = _signal_fixture()
    model_cfg = {"d": 8, "d_r": 4, "K": 2, "classifier_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                 "edge_mlp_hidden_dim": 16, "layer_dropout": 0.0, "classifier_dropout": 0.0}
    if gate_init is not None:
        model_cfg["coupling_gate_init"] = gate_init
    cfg = TrainConfig.from_dict({"device": "cpu", "checkpoint_metric": "acc", "model": model_cfg,
                                 "data": {"resample_paths": False}})
    torch.manual_seed(seed)
    model = build_model(cfg, data)
    if grouped:
        optimizer = build_optimizer(model, cfg)
    else:  # the old setting: one group at the base rate for everything
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr,
                                      weight_decay=cfg.optim.weight_decay)

    def state() -> tuple[float, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            g, b = model.coupling_gates(model.encode_nodes(data.x, data.node_type))
            return float(model.eta()), g.clone(), b.clone()

    before = state()
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(data).logits[data.target_train_mask]
        loss = F.cross_entropy(logits, data.target_y[data.target_train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        optimizer.step()
        losses.append(float(loss))
    after = state()
    moves = [abs(after[0] - before[0]), float((after[1] - before[1]).abs().mean()),
             float((after[2] - before[2]).abs().mean())]
    return moves, losses


class LearningFixTests(unittest.TestCase):
    SEEDS = (0, 1, 2)

    def test_TF1_the_scalar_group_moves_eta_and_both_gate_heads(self) -> None:
        for seed in self.SEEDS:
            moves, losses = _train_toy(gate_init=None, grouped=True, seed=seed)  # defaults
            self.assertLess(losses[-1], 0.8 * losses[0], "the toy signal must be learnable")
            for name, move in zip(("eta", "g", "b"), moves):
                self.assertGreater(move, 0.05, f"{name} (seed {seed})")

    def test_TF1_control_old_setting_does_not_move(self) -> None:
        for seed in self.SEEDS:
            moves, losses = _train_toy(gate_init=0.95, grouped=False, seed=seed)
            self.assertLess(losses[-1], 0.8 * losses[0])  # it learns; the scalars cannot move
            for name, move in zip(("eta", "g", "b"), moves):
                self.assertLess(move, 0.05, f"{name} (seed {seed})")

    def test_TF1_group_contents_and_defaults(self) -> None:
        from src.builders import build_optimizer
        from src.models.h2snn_model import build_model

        data = _signal_fixture()
        cfg = TrainConfig.from_dict({"device": "cpu", "model": {"d": 8, "d_r": 4, "K": 2}})
        self.assertEqual(cfg.model.coupling_gate_init, 0.6)
        self.assertEqual(cfg.ablation.coupling, "cooperative")
        model = build_model(cfg, data)
        optimizer = build_optimizer(model, cfg)
        fast = [g for g in optimizer.param_groups if g["lr"] > cfg.optim.lr]
        self.assertEqual(len(fast), 1)
        ids = {id(p) for p in fast[0]["params"]}
        expected = [model.theta_eta, model.theta_gamma, *model.listen_gate.parameters(),
                    *model.broadcast_gate.parameters()]
        self.assertEqual(ids, {id(p) for p in expected})
        self.assertEqual(fast[0]["weight_decay"], 0.0)
        self.assertAlmostEqual(fast[0]["lr"], cfg.optim.lr * cfg.optim.scalar_lr_scale)


# ----------------------------------------------------------------------------- T-F2, T-F3, T-F3b (Prop. 4)
def _energy(system: System, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`E(x) = 1/2 <Bx, Delta_hat Bx>` with the production normalized operator."""
    y = b.unsqueeze(-1) * x
    return 0.5 * (y * system.laplacian.apply_normalized(y)).sum()


def _effective_weights(matrix: torch.Tensor, num_nodes: int, d: int) -> torch.Tensor:
    """`w[u, v]` = the scalar gain of the off-diagonal block `A[u, v]` (u listens to v):
    ||A_uv||_F / sqrt(d), which is the scalar factor when the block is a scalar times an
    orthogonal matrix and the root-mean-square gain otherwise."""
    blocks = matrix.view(num_nodes, d, num_nodes, d).permute(0, 2, 1, 3)
    weights = blocks.flatten(2).norm(dim=-1) / d ** 0.5
    weights.fill_diagonal_(0.0)
    return weights


def _balance_residual(weights: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Least-squares per-node metric from `log m_u - log m_v = log w(v<-u) - log w(u<-v)` over
    all unordered pairs with a connection; returns `(m, max |equation residual|)`."""
    n = weights.size(0)
    rows, rhs = [], []
    for u in range(n):
        for v in range(u + 1, n):
            if weights[u, v] > 0 and weights[v, u] > 0:
                row = torch.zeros(n, dtype=weights.dtype)
                row[u], row[v] = 1.0, -1.0
                rows.append(row)
                rhs.append(torch.log(weights[v, u]) - torch.log(weights[u, v]))
    matrix, target = torch.stack(rows), torch.stack(rhs).unsqueeze(-1)
    solution = torch.linalg.lstsq(matrix, target, driver="gelsd").solution
    residual = float((matrix @ solution - target).abs().max())
    return torch.exp(solution.squeeze(-1)), residual


def _symmetry_defect(matrix: torch.Tensor, metric: torch.Tensor, d: int) -> float:
    """`||M A - (M A)^T||_max / ||M A||_max` for the diagonal metric `M = diag(metric) x I_d`."""
    ma = metric.repeat_interleave(d).unsqueeze(-1) * matrix
    return float((ma - ma.T).abs().max() / ma.abs().max())


class MetricGradientFlowTests(unittest.TestCase):
    def test_TF2_metric_gradient_identity(self) -> None:
        model = _gate_model("cooperative")
        for shared in (True, False):
            for seed in range(4):
                with self.subTest(shared=shared, seed=seed):
                    system = System(seed, shared=shared)
                    g, b = _random_gates(system, low=0.05)
                    x = torch.randn(system.num_nodes, system.d, generator=system.generator,
                                    dtype=torch.float64, requires_grad=True)
                    gradient, = torch.autograd.grad(_energy(system, x, b), x)
                    lhs = (g / b).unsqueeze(-1) * gradient  # M^-1 grad E with M = B G^-1
                    step = (x.detach() - model.diffuse(x.detach(), system.operator(g, b))) / ETA
                    G = torch.diag(g.repeat_interleave(system.d))
                    B = torch.diag(b.repeat_interleave(system.d))
                    dense = (G @ system.dense_delta_hat() @ B @ x.detach().reshape(-1)).view_as(x)
                    self.assertLess(_rel(lhs, step), F64_TOL)
                    self.assertLess(_rel(lhs, dense), F64_TOL)
                    # controls: the swapped metric G B^-1 and the identity metric are other flows
                    self.assertGreater(_rel((b / g).unsqueeze(-1) * gradient, step), 1e-3)
                    self.assertGreater(_rel(gradient, step), 1e-3)

    def _energies(self, eta: float, fixtures: int, steps: int = 50) -> list[list[float]]:
        model = _gate_model("cooperative")
        histories = []
        for seed in range(fixtures):
            system = System(2000 + seed, shared=seed % 2 == 0)
            g, b = _random_gates(system, low=0.01)
            operator = system.operator(g, b)._replace(eta=torch.tensor(eta, dtype=torch.float64))
            x = torch.randn(system.num_nodes, system.d, generator=system.generator,
                            dtype=torch.float64)
            history = [float(_energy(system, x, b))]
            for _ in range(steps):
                x = model.diffuse(x, operator)
                history.append(float(_energy(system, x, b)))
            histories.append(history)
        return histories

    def test_TF3_energy_never_increases_for_eta_up_to_one_and_does_for_eta_three(self) -> None:
        for eta in (0.25, 0.7, 1.0):
            worst = max(max(later - earlier for earlier, later in zip(h, h[1:]))
                        for h in self._energies(eta, fixtures=50))
            self.assertLessEqual(worst, 1e-12, f"eta={eta}: largest one-step energy increase")
        increases = [max(later - earlier for earlier, later in zip(h, h[1:]))
                     for h in self._energies(3.0, fixtures=50)]
        self.assertGreater(max(increases), 1e-6, "control: eta = 3 must increase E somewhere")

    def _step_matrices(self, fixtures: int):
        model = _gate_model("cooperative")
        for seed in range(fixtures):
            system = System(3000 + seed, shared=seed % 2 == 0)
            g, b = _random_gates(system, low=0.02)
            yield system, g, b, _step_matrix(model, system.operator(g, b), system.num_nodes,
                                             system.d)

    def test_TF3b_prop4_the_d11_operator_factorizes_and_a_metric_exists(self) -> None:
        worst_residual, worst_symmetry = 0.0, 0.0
        for system, g, b, matrix in self._step_matrices(30):
            weights = _effective_weights(matrix, system.num_nodes, system.d)
            metric, residual = _balance_residual(weights)
            worst_residual = max(worst_residual, residual)
            worst_symmetry = max(worst_symmetry, _symmetry_defect(matrix, metric, system.d))
            ratio = metric / (b / g)  # the metric is the one the theory names, up to a constant
            self.assertLess(float((ratio / ratio[0] - 1).abs().max()), 1e-8)
        self.assertLess(worst_residual, 1e-10)
        self.assertLess(worst_symmetry, 1e-10)

    def test_TF3b_control_free_per_direction_weights_have_no_metric(self) -> None:
        """A free weight per direction breaks detailed balance, so the check is not vacuous."""
        worst = 0.0
        for system, g, b, matrix in self._step_matrices(30):
            n, d = system.num_nodes, system.d
            factors = 0.2 + 1.8 * torch.rand(n, n, generator=system.generator, dtype=torch.float64)
            free = (matrix.view(n, d, n, d) * factors[:, None, :, None]).reshape(n * d, n * d)
            _, residual = _balance_residual(_effective_weights(free, n, d))
            worst = max(worst, residual)
        self.assertGreater(worst, 1e-3)


if __name__ == "__main__":
    unittest.main()
