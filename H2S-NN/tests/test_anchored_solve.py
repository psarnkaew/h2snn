"""v3-B card U3 (`docs/H2S-NN_v3B_card.md` §2): the anchored solve.

T9  PCG (Jacobi, from the D-B7 initial guess) vs a dense solve of `(P^T P + tau L_hat) Y = P^T h`
    on the fixture, fp64, tol 1e-10: `|Y_pcg - Y_dense| <= 1e-8`; iterations < T_max = 100.
T10 implicit gradient (`AnchoredSolve`) vs central finite differences (fp64, eps = 1e-6, dense
    solves) on every parameter group (fact scales, TD-1 instance coefficients, subspaces `W_r`,
    frames, cell weights, theta_tau, h0; spec 1.25 Z2): relative error <= 1e-6 along three random directions per group.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.models.anchored_solve import (
    AnchoredSystem,
    PCGConvergenceError,
    SolveStats,
    anchored_solve,
    count_zero_rhs_scopes,
    pcg,
    segment_dot,
    solve_anchored,
    solver_tolerance,
)
from src.models.map_operators import orthonormal_subspaces, skew_from_coefficients
from src.models.sheaf_maps import TD1Factors, td1_edge_maps
from src.models.sheaf_energy import EliminatedOperator, SheafEnergyOperator, build_cell_group, materialize
from tests.test_sheaf_energy import Fixture

T9_TOL = 1e-8
T10_TOL = 1e-6
SOLVE_TOL = 1e-10
T_MAX = 100
EPS_FD = 1e-6
SCALE_EPS = 0.1
RANK = 2          # TD-1 rank of the d = 4 fixture (2 <= m <= d; m = 4 in the model)
SHARED = (1,)     # the fixture's relation 1 has rho(r) = r (tests/test_sheaf_energy.py)


# ----------------------------------------------------------------------------- parameterisation
class Parameters:
    """Leaf tensors of every group the operator depends on (fp64)."""

    def __init__(self, fixture: Fixture, seed: int = 7) -> None:
        g = torch.Generator().manual_seed(seed)
        E, d, m = fixture.edge_index.size(1), fixture.d, RANK
        R = int(fixture.edge_type.max()) + 1
        f64 = torch.float64
        self.fixture = fixture
        self.raw_scale = torch.randn(E, generator=g, dtype=f64)
        self.raw_skew = torch.randn(E, m * (m - 1) // 2, generator=g, dtype=f64)
        self.raw_subspace = torch.randn(R, d, m, generator=g, dtype=f64)
        self.raw_frame = 0.5 * torch.randn(R - len(SHARED), d, d, generator=g, dtype=f64)
        self.raw_beta = [0.5 * torch.randn(group.num_faces, generator=g, dtype=f64)
                         for group in fixture.groups]
        self.theta_tau = torch.tensor(0.3, dtype=f64)
        self.h0 = torch.randn(fixture.num_nodes, d, generator=g, dtype=f64)
        self.weight = torch.randn(fixture.num_nodes, d, generator=g, dtype=f64)  # loss weights

    def operator_params(self) -> tuple[torch.Tensor, ...]:
        return (self.raw_scale, self.raw_skew, self.raw_subspace, self.raw_frame, *self.raw_beta)

    def groups(self) -> dict[str, list[torch.Tensor]]:
        return {
            "fact_scales": [self.raw_scale],
            "instance_coefficients": [self.raw_skew],
            "subspaces": [self.raw_subspace],
            "frames": [self.raw_frame],
            "cell_weights": list(self.raw_beta),
            "theta_tau": [self.theta_tau],
            "h0": [self.h0],
        }


def _rotations(raw: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_exp(raw - raw.transpose(-2, -1))


def build_operator(fixture: Fixture, raw_scale: torch.Tensor, raw_skew: torch.Tensor,
                   raw_subspace: torch.Tensor, raw_frame: torch.Tensor, *raw_beta: torch.Tensor,
                   kept: torch.Tensor | None = None, node_cell: bool = True,
                   fact_cell: bool = True) -> EliminatedOperator:
    """TD-1 maps (spec 1.25 Z2) of the fixture: relation 1 has rho(r) = r (no frame, half angles)."""
    scale = torch.nn.functional.softplus(raw_scale) + SCALE_EPS
    generator = skew_from_coefficients(raw_skew, RANK)
    half = torch.where(fixture.edge_type == 1, 0.5, 1.0).to(generator.dtype).view(-1, 1, 1)
    rotation = torch.linalg.matrix_exp(generator * half) - torch.eye(RANK, dtype=generator.dtype)
    factors = TD1Factors(scale, rotation, orthonormal_subspaces(raw_subspace), _rotations(raw_frame))
    maps = td1_edge_maps(factors, fixture.edge_type, 2, SHARED)
    cells = [build_cell_group(maps, group, fixture.edge_index, torch.exp(beta))
             for group, beta in zip(fixture.groups, raw_beta)]
    operator = SheafEnergyOperator(maps, fixture.edge_index, fixture.num_nodes, cells,
                                   node_cell=node_cell, fact_cell=fact_cell)
    return EliminatedOperator(operator, kept)


def dense_solution(system: AnchoredSystem, h0: torch.Tensor) -> torch.Tensor:
    rows, d = system.operator.layout.rows, h0.size(-1)
    M = materialize(system.apply, rows, d, torch.float64)
    b = system.right_hand_side(h0).reshape(-1)
    return torch.linalg.solve(M, b).view(rows, d)


# ----------------------------------------------------------------------------- T9
class PCGTests(unittest.TestCase):
    def test_T9_pcg_matches_the_dense_solve(self) -> None:
        fixture = Fixture()
        params = Parameters(fixture)
        E = fixture.edge_index.size(1)
        for name, kept in (("reduced (E_shared)", None), ("full state", torch.arange(E))):
            for tau in (1.0, 9.0):
                with self.subTest(operator=name, tau=tau):
                    operator = build_operator(fixture, *params.operator_params(), kept=kept)
                    system = AnchoredSystem(operator, torch.tensor(tau, dtype=torch.float64))
                    y_dense = dense_solution(system, params.h0)
                    y, iterations, converged, residual = solve_anchored(
                        system, params.h0, None, SOLVE_TOL, T_MAX)
                    self.assertTrue(bool(converged.all()))
                    self.assertLess(int(iterations.max()), T_MAX)
                    self.assertLess(float((y - y_dense).abs().max()), T9_TOL)
                    # warm start from the solution: nothing left to do
                    _, warm_iterations, _, _ = solve_anchored(system, params.h0, y, SOLVE_TOL,
                                                              T_MAX)
                    self.assertEqual(int(warm_iterations.max()), 0)
        # schedule: 1e-3 for epochs 0..9, then 1e-4; evaluation (-1) uses the final tolerance
        self.assertEqual(solver_tolerance(0, 10, 1e-3, 1e-4), 1e-3)
        self.assertEqual(solver_tolerance(9, 10, 1e-3, 1e-4), 1e-3)
        self.assertEqual(solver_tolerance(10, 10, 1e-3, 1e-4), 1e-4)
        self.assertEqual(solver_tolerance(-1, 10, 1e-3, 1e-4), 1e-4)


# ----------------------------------------------------------------------------- T10
class ImplicitGradientTests(unittest.TestCase):
    def _loss(self, fixture: Fixture, params: Parameters, values: dict[str, list[torch.Tensor]],
              implicit: bool, stats: SolveStats | None = None) -> torch.Tensor:
        raw_scale = values["fact_scales"][0]
        raw_skew = values["instance_coefficients"][0]
        raw_subspace = values["subspaces"][0]
        raw_frame = values["frames"][0]
        betas = values["cell_weights"]
        theta = values["theta_tau"][0]
        h0 = values["h0"][0]
        operator_params = (raw_scale, raw_skew, raw_subspace, raw_frame, *betas)
        if implicit:
            y = anchored_solve(lambda *p: build_operator(fixture, *p), h0, theta, operator_params,
                               SOLVE_TOL, T_MAX, stats if stats is not None else SolveStats())
        else:
            with torch.no_grad():
                operator = build_operator(fixture, *operator_params)
                y = dense_solution(AnchoredSystem(operator, torch.exp(theta)), h0)
        x_star = y[:fixture.num_nodes]
        return (params.weight * x_star.square()).sum() + (params.weight * x_star).sum()

    def test_T10_implicit_gradient_matches_central_finite_differences(self) -> None:
        fixture = Fixture()
        params = Parameters(fixture)
        leaves = {name: [t.clone().requires_grad_(True) for t in tensors]
                  for name, tensors in params.groups().items()}
        stats = SolveStats()
        loss = self._loss(fixture, params, leaves, implicit=True, stats=stats)
        loss.backward()
        self.assertLess(stats.forward_iterations, T_MAX)
        self.assertLess(stats.backward_iterations, T_MAX)
        g = torch.Generator().manual_seed(13)
        worst: dict[str, float] = {}
        for name, tensors in leaves.items():
            grads = [t.grad for t in tensors]
            self.assertTrue(all(gr is not None for gr in grads), f"no gradient for {name}")
            self.assertGreater(float(sum(gr.norm() ** 2 for gr in grads)), 0.0, name)
            for _ in range(3):
                directions = [torch.randn(t.shape, generator=g, dtype=torch.float64)
                              for t in tensors]
                norm = torch.sqrt(sum(v.square().sum() for v in directions))
                directions = [v / norm for v in directions]
                analytic = float(sum((gr * v).sum() for gr, v in zip(grads, directions)))
                plus = {k: [t.detach().clone() for t in v] for k, v in leaves.items()}
                minus = {k: [t.detach().clone() for t in v] for k, v in leaves.items()}
                for t_plus, t_minus, v in zip(plus[name], minus[name], directions):
                    t_plus += EPS_FD * v
                    t_minus -= EPS_FD * v
                numeric = float((self._loss(fixture, params, plus, implicit=False)
                                 - self._loss(fixture, params, minus, implicit=False))
                                / (2.0 * EPS_FD))
                relative = abs(analytic - numeric) / max(abs(numeric), 1e-300)
                worst[name] = max(worst.get(name, 0.0), relative)
                self.assertLess(relative, T10_TOL,
                                f"{name}: analytic {analytic:.12g} vs FD {numeric:.12g}")
        self.worst = worst


# ----------------------------------------------------------------------------- card 12 unit P
# Zero right-hand-side rule in `pcg` (spec 1.30): a scope with ||b_s|| <= 1e-10 max_s ||b_s|| is
# solved by y_s = 0. Fixture: a block-diagonal SPD system (one dense block per scope, applied block
# by block so the scopes share no arithmetic), M = I + tau L, L PSD with spectrum [0.85, 5].
# The old function below is `pcg` at 2d822a4, copied verbatim (the reference for bit identity).

def _pcg_2d822a4(apply_m, b, y0, jacobi_inverse, tolerance, max_iterations, scope_rows=None,
                 num_scopes=1):
    expand = scope_rows if (scope_rows is not None and num_scopes > 1) else None

    def per_row(values: torch.Tensor) -> torch.Tensor:
        return (values[expand] if expand is not None else values.expand(b.size(0))).unsqueeze(-1)

    y = y0.clone()
    r = b - apply_m(y)
    zt = jacobi_inverse.unsqueeze(-1) * r
    p = zt.clone()
    rz = segment_dot(r, zt, scope_rows, num_scopes)
    b_norm = segment_dot(b, b, scope_rows, num_scopes).sqrt()
    threshold = tolerance * b_norm
    r_norm = segment_dot(r, r, scope_rows, num_scopes).sqrt()
    converged = r_norm <= threshold
    iterations = torch.zeros(num_scopes, dtype=torch.long, device=b.device)
    ones = torch.ones_like(rz)
    for _ in range(max_iterations):
        if bool(converged.all()):
            break
        active = ~converged
        q = apply_m(p)
        pq = segment_dot(p, q, scope_rows, num_scopes)
        alpha = torch.where(active, rz / torch.where(active & (pq != 0), pq, ones),
                            torch.zeros_like(rz))
        y = y + per_row(alpha) * p
        r = r - per_row(alpha) * q
        iterations = iterations + active.long()
        r_norm = segment_dot(r, r, scope_rows, num_scopes).sqrt()
        converged = converged | (r_norm <= threshold)
        zt = jacobi_inverse.unsqueeze(-1) * r
        rz_new = segment_dot(r, zt, scope_rows, num_scopes)
        still = active & ~converged
        beta = torch.where(still, rz_new / torch.where(still & (rz != 0), rz, ones),
                           torch.zeros_like(rz))
        p = zt + per_row(beta) * p
        rz = rz_new
    relative = r_norm / torch.where(b_norm > 0, b_norm, torch.ones_like(b_norm))
    return y, iterations, converged, relative


PCG_TOL = 1e-3       # the warm-up tolerance of the schedule (the PDNS failure was in epoch 1)
TINY_RHS = 3.4e-17   # ||b_s|| of the failing PDNS scope (card 11 B2c)
D_COLS = 16


def _spd_blocks(sizes: tuple[int, ...], dtype: torch.dtype, seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    blocks = []
    for n in sizes:
        q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
        spectrum = torch.linspace(0.85, 5.0, n, dtype=torch.float64)
        blocks.append(((q * spectrum) @ q.T).to(dtype))
    return blocks


class BlockOperator:
    """`L_hat = weight * blockdiag(blocks)` on node rows only (no cell rows); one block per scope."""

    def __init__(self, blocks: list[torch.Tensor], weight: torch.Tensor) -> None:
        self.blocks = blocks
        self.weight = weight
        self.sizes = [blk.size(0) for blk in blocks]
        self.num_nodes = sum(self.sizes)

        class _Layout:
            rows = self.num_nodes
        self.layout = _Layout()

    def apply_normalized(self, y: torch.Tensor) -> torch.Tensor:
        out, start = [], 0
        for blk, n in zip(self.blocks, self.sizes):
            out.append(blk @ y[start:start + n])
            start += n
        return self.weight * torch.cat(out)

    def initial_guess(self, h0: torch.Tensor) -> torch.Tensor:
        return h0.clone()

    def diagonal_scalars(self) -> torch.Tensor:
        return self.weight * torch.cat([torch.diagonal(blk) for blk in self.blocks])

    def scope_rows(self) -> torch.Tensor:
        return torch.cat([torch.full((n,), s, dtype=torch.long) for s, n in enumerate(self.sizes)])


def _system(sizes: tuple[int, ...], dtype: torch.dtype, tau: float = 1.0,
            seed: int = 0) -> tuple[AnchoredSystem, torch.Tensor]:
    operator = BlockOperator(_spd_blocks(sizes, dtype, seed), torch.tensor(1.0, dtype=dtype))
    return AnchoredSystem(operator, torch.tensor(tau, dtype=dtype)), operator.scope_rows()


def _rhs(sizes: tuple[int, ...], norms: tuple[float, ...], dtype: torch.dtype,
         seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    parts = []
    for n, norm in zip(sizes, norms):
        block = torch.randn(n, D_COLS, generator=g, dtype=torch.float64)
        parts.append((block * (norm / float(block.norm())) if norm > 0 else block * 0).to(dtype))
    return torch.cat(parts)


def _strict(result: tuple, max_iterations: int) -> tuple:
    """The `strict` check of `solve_anchored` on a raw `pcg` result."""
    if not bool(result[2].all()):
        raise PCGConvergenceError(f"PCG reached T_max = {max_iterations}")
    return result


class ZeroRightHandSideTests(unittest.TestCase):
    """Card 12 unit P tests 1-5 (the mutations of the card each make one of them fail)."""

    def _flush_denormals(self) -> None:
        # Why flush-to-zero (card 12 amendment 1): the old pcg fails on this fixture ONLY with
        # denormals flushed. On the failing PDNS scope (||b_s|| = 3.4e-17, card 11 B2c) the entries
        # are ~1e-19, so the per-scope dot products b_s^T b_s, r^T r, p^T q (~1e-38 and below)
        # underflow in fp32 when denormals are flushed: ||b_s|| becomes ~0 or exactly 0 and the
        # relative stopping test is unreachable (T_max). With denormals kept (CPU default) fp32
        # PCG is scale-invariant and the old code converges, so the failure is not reproduced.
        # The rule (||b_s|| <= rho * max) covers both the tiny and the exactly-zero case.
        if not torch.set_flush_denormal(True):
            self.skipTest("flush-denormal not supported on this CPU")
        self.addCleanup(torch.set_flush_denormal, False)

    # 1 ------------------------------------------------------------------------------------------
    def test_P1_zero_scope_gets_zero_and_other_scope_is_unchanged(self) -> None:
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                sizes = (40, 60)
                system, scope_rows = _system(sizes, dtype)
                b = _rhs(sizes, (1e-3, 0.0), dtype)
                y0 = torch.randn(sum(sizes), D_COLS, generator=torch.Generator().manual_seed(3),
                                 dtype=dtype)
                jinv = system.jacobi_inverse()
                y, iterations, converged, relative = pcg(system.apply, b, y0, jinv, PCG_TOL, T_MAX,
                                                         scope_rows, 2)
                self.assertTrue(bool((y[40:] == 0).all()))
                self.assertTrue(bool(converged.all()))
                self.assertEqual(int(iterations[1]), 0)
                self.assertEqual(float(relative[1]), 0.0)
                self.assertGreater(int(iterations[0]), 0)
                # scope 0 alone, old pcg, the same per-scope reductions (scope 1 empty): bit-identical
                block0 = system.operator.blocks[0]
                apply0 = lambda v: v + block0 @ v  # noqa: E731  (tau = weight = 1)
                y_old, it_old, conv_old, rel_old = _pcg_2d822a4(
                    apply0, b[:40], y0[:40], jinv[:40], PCG_TOL, T_MAX,
                    torch.zeros(40, dtype=torch.long), 2)
                self.assertTrue(torch.equal(y[:40], y_old))
                self.assertEqual(int(iterations[0]), int(it_old[0]))
                self.assertEqual(float(relative[0]), float(rel_old[0]))
                # as one scope (plain sum instead of index_add: other rounding): two PCG solutions
                # of the same system at tolerance PCG_TOL, so equal to <= cond(M) * PCG_TOL
                y_one, it_one, _, _ = _pcg_2d822a4(apply0, b[:40], y0[:40], jinv[:40], PCG_TOL,
                                                   T_MAX)
                self.assertEqual(int(it_one[0]), int(iterations[0]))
                self.assertLess(float((y[:40] - y_one).abs().max()),
                                10 * PCG_TOL * float(y_one.abs().max()))

    # 2 ------------------------------------------------------------------------------------------
    def test_P2_tiny_scope_does_not_raise_where_the_old_pcg_does(self) -> None:
        self._flush_denormals()
        sizes = (40, 60, 50)
        system, scope_rows = _system(sizes, torch.float32)
        b = _rhs(sizes, (1e-3, TINY_RHS, 1e-3), torch.float32)
        y0 = torch.zeros_like(b)
        jinv = system.jacobi_inverse()
        with self.assertRaises(PCGConvergenceError):   # the fixture reproduces the PDNS failure
            _strict(_pcg_2d822a4(system.apply, b, y0, jinv, PCG_TOL, T_MAX, scope_rows, 3), T_MAX)
        y, iterations, converged, relative = _strict(
            pcg(system.apply, b, y0, jinv, PCG_TOL, T_MAX, scope_rows, 3), T_MAX)
        self.assertEqual(iterations.tolist()[1], 0)
        self.assertTrue(bool((y[40:100] == 0).all()))
        self.assertEqual(float(relative[1]), 0.0)
        self.assertLess(int(iterations.max()), T_MAX)
        # through solve_anchored (strict=True), as the forward calls it
        h0 = b.clone()
        y_s, _, converged_s, _ = solve_anchored(system, h0, y0, PCG_TOL, T_MAX, scope_rows, 3)
        self.assertTrue(bool(converged_s.all()))
        self.assertTrue(torch.equal(y_s, y))

    # 3 ------------------------------------------------------------------------------------------
    def test_P3_all_zero_rhs(self) -> None:
        for num_scopes, sizes in ((1, (50,)), (3, (40, 60, 50))):
            with self.subTest(num_scopes=num_scopes):
                system, scope_rows = _system(sizes, torch.float32)
                b = torch.zeros(sum(sizes), D_COLS)
                y0 = torch.randn(sum(sizes), D_COLS, generator=torch.Generator().manual_seed(4))
                y, iterations, converged, relative = pcg(
                    system.apply, b, y0, system.jacobi_inverse(), PCG_TOL, T_MAX,
                    scope_rows if num_scopes > 1 else None, num_scopes)
                self.assertTrue(bool((y == 0).all()))
                self.assertTrue(bool(converged.all()))
                self.assertEqual(int(iterations.max()), 0)
                self.assertTrue(bool(torch.isfinite(relative).all()))
                self.assertEqual(float(relative.abs().max()), 0.0)

    # 4 ------------------------------------------------------------------------------------------
    def test_P4_no_trigger_is_bit_identical_to_the_old_pcg(self) -> None:
        cases = (("one scope", (70,), (1e-3,)),
                 ("three scopes", (40, 60, 50), (1e-3, 2e-6, 5.0)),
                 ("just above rho", (40, 60), (1.0, 2e-10)))
        for dtype in (torch.float32, torch.float64):
            for name, sizes, norms in cases:
                for warm in (False, True):
                    with self.subTest(dtype=dtype, case=name, warm=warm):
                        system, scope_rows = _system(sizes, dtype, tau=3.0)
                        b = _rhs(sizes, norms, dtype)
                        y0 = (torch.randn(b.shape, generator=torch.Generator().manual_seed(5),
                                          dtype=dtype) * 1e-3 if warm else torch.zeros_like(b))
                        S = len(sizes)
                        rows = scope_rows if S > 1 else None
                        args = (system.apply, b, y0, system.jacobi_inverse(), PCG_TOL, T_MAX,
                                rows, S)
                        new, old = pcg(*args), _pcg_2d822a4(*args)
                        for a, o in zip(new, old):
                            self.assertTrue(torch.equal(a, o))
                        self.assertEqual(count_zero_rhs_scopes(b, rows, S), 0)

    # 5 ------------------------------------------------------------------------------------------
    def _backward(self, sizes: tuple[int, ...], scope_weights: tuple[float, ...]):
        dtype = torch.float32
        blocks = _spd_blocks(sizes, dtype)
        raw_weight = torch.tensor(0.2, dtype=dtype, requires_grad=True)
        theta = torch.tensor(0.1, dtype=dtype, requires_grad=True)
        h0 = _rhs(sizes, (1.0,) * len(sizes), dtype, seed=6).requires_grad_(True)
        probe = BlockOperator(blocks, torch.tensor(1.0))
        scope_rows = probe.scope_rows()
        loss_weight = _rhs(sizes, scope_weights, dtype, seed=8)
        stats = SolveStats()
        y = anchored_solve(lambda w: BlockOperator(blocks, torch.exp(w)), h0, theta, (raw_weight,),
                           PCG_TOL, T_MAX, stats, scope_rows=scope_rows, num_scopes=len(sizes))
        loss = (loss_weight * y).sum()   # dL/dY = loss_weight: scope s gets ||dL/dY_s|| = weight s
        loss.backward()
        return stats, h0, raw_weight, theta

    def test_P5_backward_with_a_zero_gradient_scope(self) -> None:
        sizes = (40, 60, 50)
        stats, h0, raw_weight, theta = self._backward(sizes, (1.0, 0.0, 1.0))
        for t in (h0, raw_weight, theta):
            self.assertIsNotNone(t.grad)
            self.assertTrue(bool(torch.isfinite(t.grad).all()))
        self.assertTrue(bool((h0.grad[40:100] == 0).all()))
        self.assertEqual(stats.backward_zero_rhs_scopes, 1)
        self.assertEqual(stats.forward_zero_rhs_scopes, 0)
        self.assertEqual(stats.as_dict()["pcg_zero_rhs_scopes"], 1.0)
        self.assertEqual(stats.per_scope_forward[1] > 0, True)

    def test_P5b_backward_with_a_tiny_gradient_scope(self) -> None:
        """The PDNS case: one scope's dL/dY has norm 3.4e-17 (fp32, flushed denormals)."""
        self._flush_denormals()
        import src.models.anchored_solve as AS

        sizes = (40, 60, 50)
        original = AS.pcg
        AS.pcg = _pcg_2d822a4
        try:
            with self.assertRaises(PCGConvergenceError):   # old pcg: the backward raises
                self._backward(sizes, (1.0, TINY_RHS, 1.0))
        finally:
            AS.pcg = original
        stats, h0, raw_weight, theta = self._backward(sizes, (1.0, TINY_RHS, 1.0))
        for t in (h0, raw_weight, theta):
            self.assertTrue(bool(torch.isfinite(t.grad).all()))
        self.assertTrue(bool((h0.grad[40:100] == 0).all()))
        self.assertEqual(stats.backward_zero_rhs_scopes, 1)
        self.assertTrue(stats.backward_converged)

    def test_P5c_forward_counts_triggered_scopes(self) -> None:
        sizes = (40, 60, 50)
        blocks = _spd_blocks(sizes, torch.float64)
        h0 = _rhs(sizes, (1.0, 0.0, 1.0), torch.float64)
        stats = SolveStats()
        scope_rows = BlockOperator(blocks, torch.tensor(1.0)).scope_rows()
        y = anchored_solve(lambda: BlockOperator(blocks, torch.tensor(1.0, dtype=torch.float64)),
                           h0, torch.tensor(0.0, dtype=torch.float64), (), SOLVE_TOL, T_MAX,
                           stats, scope_rows=scope_rows, num_scopes=3)
        self.assertEqual(stats.forward_zero_rhs_scopes, 1)
        self.assertEqual(stats.per_scope_forward[1], 0)
        self.assertTrue(bool((y[40:100] == 0).all()))
        self.assertEqual(stats.as_dict()["pcg_zero_rhs_scopes"], 1.0)


if __name__ == "__main__":
    unittest.main()
