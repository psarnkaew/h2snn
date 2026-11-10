"""v3-B card U3 (`docs/H2S-NN_v3B_card.md` §2): the anchored solve.

T9  PCG (Jacobi, from the D-B7 initial guess) vs a dense solve of `(P^T P + tau L_hat) Y = P^T h`
    on the fixture, fp64, tol 1e-10: `|Y_pcg - Y_dense| <= 1e-8`; iterations < T_max = 100.
T10 implicit gradient (`AnchoredSolve`) vs central finite differences (fp64, eps = 1e-6, dense
    solves) on every parameter group (fact scales, reflectors, rotations, cell weights, theta_tau,
    h0): relative error <= 1e-6 along three random directions per group.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.models.anchored_solve import (
    AnchoredSystem,
    SolveStats,
    anchored_solve,
    solve_anchored,
    solver_tolerance,
)
from src.models.map_operators import EdgeMapPair, FactorizedMapOperator, normalize_reflectors
from src.models.sheaf_energy import EliminatedOperator, SheafEnergyOperator, build_cell_group, materialize
from tests.test_sheaf_energy import Fixture

T9_TOL = 1e-8
T10_TOL = 1e-6
SOLVE_TOL = 1e-10
T_MAX = 100
EPS_FD = 1e-6
SCALE_EPS = 0.1


# ----------------------------------------------------------------------------- parameterisation
class Parameters:
    """Leaf tensors of every group the operator depends on (fp64)."""

    def __init__(self, fixture: Fixture, seed: int = 7) -> None:
        g = torch.Generator().manual_seed(seed)
        E, d, k = fixture.edge_index.size(1), fixture.d, 2
        R = int(fixture.edge_type.max()) + 1
        f64 = torch.float64
        self.fixture = fixture
        self.raw_scale = torch.randn(E, generator=g, dtype=f64)
        self.raw_refl_src = torch.randn(E, k, d, generator=g, dtype=f64)
        self.raw_refl_tgt = torch.randn(E, k, d, generator=g, dtype=f64)
        self.raw_rot_src = 0.5 * torch.randn(R, d, d, generator=g, dtype=f64)
        self.raw_rot_tgt = 0.5 * torch.randn(R, d, d, generator=g, dtype=f64)
        self.raw_beta = [0.5 * torch.randn(group.num_faces, generator=g, dtype=f64)
                         for group in fixture.groups]
        self.theta_tau = torch.tensor(0.3, dtype=f64)
        self.h0 = torch.randn(fixture.num_nodes, d, generator=g, dtype=f64)
        self.weight = torch.randn(fixture.num_nodes, d, generator=g, dtype=f64)  # loss weights

    def operator_params(self) -> tuple[torch.Tensor, ...]:
        return (self.raw_scale, self.raw_refl_src, self.raw_refl_tgt, self.raw_rot_src,
                self.raw_rot_tgt, *self.raw_beta)

    def groups(self) -> dict[str, list[torch.Tensor]]:
        return {
            "fact_scales": [self.raw_scale],
            "reflectors": [self.raw_refl_src, self.raw_refl_tgt],
            "rotations": [self.raw_rot_src, self.raw_rot_tgt],
            "cell_weights": list(self.raw_beta),
            "theta_tau": [self.theta_tau],
            "h0": [self.h0],
        }


def _rotations(raw: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_exp(raw - raw.transpose(-2, -1))


def build_operator(fixture: Fixture, raw_scale: torch.Tensor, raw_refl_src: torch.Tensor,
                   raw_refl_tgt: torch.Tensor, raw_rot_src: torch.Tensor,
                   raw_rot_tgt: torch.Tensor, *raw_beta: torch.Tensor,
                   kept: torch.Tensor | None = None, node_cell: bool = True,
                   fact_cell: bool = True) -> EliminatedOperator:
    scale = torch.nn.functional.softplus(raw_scale) + SCALE_EPS
    maps = EdgeMapPair(
        src=FactorizedMapOperator(scale, normalize_reflectors(raw_refl_src, 1e-6),
                                  _rotations(raw_rot_src), fixture.edge_type),
        tgt=FactorizedMapOperator(scale, normalize_reflectors(raw_refl_tgt, 1e-6),
                                  _rotations(raw_rot_tgt), fixture.edge_type),
    )
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
        refl_src, refl_tgt = values["reflectors"]
        rot_src, rot_tgt = values["rotations"]
        betas = values["cell_weights"]
        theta = values["theta_tau"][0]
        h0 = values["h0"][0]
        operator_params = (raw_scale, refl_src, refl_tgt, rot_src, rot_tgt, *betas)
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


if __name__ == "__main__":
    unittest.main()
