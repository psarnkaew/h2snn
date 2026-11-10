"""v3-B mutation tests (Part L, unit L8c).

Each mutation is a plausible implementation error, applied by monkeypatching inside the test only
(no source file changes). For each one the named correctness test is run twice in a nested
`unittest` run: it must PASS without the mutation and FAIL under it. The failure message (the
failing number) is kept in `MUTATION_RESULTS` for the report.

  m1 FactCellTransport uses the forward (stored source-role) map on a reversed hop, no adjoint
     -> T1 (functoriality on the IMDB draw) fails.
  m2 R_{e,f} perturbed by a small random rotation (one per cell, angle scale 1e-2; UP, DOWN and
     apply consistent, so L stays symmetric) -> T1 fails and the flat-sheaf kernel drops below d.
  m3 node-cell term dropped inside `full` -> T2 (matrix-free vs dense) fails.
  m4 D_f = beta_f * 2 l_f (missing +1) -> T2 (L and D vs the dense reference) fails. T3 (diagonal
     blocks of L_hat = I) is run and recorded but cannot fail: `apply` takes its diagonal from the
     same `degree()`, so the wrong D cancels in L_hat (L8 report, DEVIATION L8c-a).
  m5 implicit backward without the tau factor -> U1 (unrolled vs implicit gradient) fails.
  m6 elimination with a wrong Schur weight (eliminated state gather / sqrt(D_e) instead of
     gather / D_e) -> T7 fails.
"""

from __future__ import annotations

import unittest
from unittest import mock

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

import src.models.sheaf_energy as sheaf_energy
import tests.test_sheaf_energy as test_sheaf_energy
from src.models.anchored_solve import AnchoredSolve, AnchoredSystem, PCGConvergenceError, pcg
from src.models.map_operators import _mix_rows
from src.models.sheaf_energy import (
    EliminatedOperator,
    FactCellTransport,
    SheafEnergyOperator,
)
from tests.test_sheaf_energy import Fixture
from tests.test_v3b_theory import flat_kernel_numbers

T1 = "tests.test_sheaf_energy.FunctorialityOnRealDrawTests.test_T1_fact_to_cell_maps_are_functorial_on_imdb"
T2 = "tests.test_sheaf_energy.OperatorTests.test_T2_matrix_free_equals_dense"
T3 = "tests.test_sheaf_energy.OperatorTests.test_T3_normalized_operator_is_symmetric_and_psd"
T7 = "tests.test_sheaf_energy.EliminationTests.test_T7_eliminated_solve_matches_the_full_state_solve"
U1 = "tests.test_v3b_theory.UnrollTests.test_U1_unrolled_gradient_equals_the_implicit_gradient"
K1 = "tests.test_v3b_theory.KernelTests.test_K1_flat_sheaf_kernel_is_the_constant_sections"

MUTATION_RESULTS: dict[str, dict[str, str]] = {}
_UNPATCHED: dict[str, tuple[str, str]] = {}
PERTURBATION_ANGLE = 1e-2


def run_named(test_id: str) -> tuple[str, str]:
    """Run one test in a nested unittest run: (`pass` | `fail` | `error` | `skip`, message)."""
    suite = unittest.defaultTestLoader.loadTestsFromName(test_id)
    result = unittest.TestResult()
    suite.run(result)
    if result.skipped:
        return "skip", result.skipped[0][1]
    for kind, records in (("fail", result.failures), ("error", result.errors)):
        if records:
            lines = [line for line in records[0][1].strip().splitlines() if line.strip()]
            return kind, lines[-1]
    return "pass", ""


def unpatched(test_id: str) -> tuple[str, str]:
    if test_id not in _UNPATCHED:
        _UNPATCHED[test_id] = run_named(test_id)
    return _UNPATCHED[test_id]


# ----------------------------------------------------------------------------- mutations
_original_build = sheaf_energy.build_cell_group


def _build_marking_reversed_hops(maps, group, edge_index, beta, free_maps=None):
    cell = _original_build(maps, group, edge_index, beta, free_maps)
    if isinstance(cell.fact_maps, FactCellTransport) and group.face_edge_sign is not None:
        cell.fact_maps._reversed = group.face_edge_sign < 0
    return cell


def _near_forward_map(self, j):
    """m1: the stored source-role map on every hop (a reversed hop should use the target role)."""
    reversed_hop = getattr(self, "_reversed", None)
    if reversed_hop is None or not bool(reversed_hop[:, j].any()):
        return self.transport._src[j]
    # on reversed rows `_tgt[j]` holds the stored source role
    return _mix_rows(self.transport._src[j], self.transport._tgt[j], reversed_hop[:, j])


def m1_patches() -> list:
    return [mock.patch.object(sheaf_energy, "build_cell_group", _build_marking_reversed_hops),
            mock.patch.object(test_sheaf_energy, "build_cell_group", _build_marking_reversed_hops),
            mock.patch.object(FactCellTransport, "_near", _near_forward_map)]


_original_up = FactCellTransport.up
_original_down = FactCellTransport.down
_original_apply = FactCellTransport.apply


def _rotation(self, like: torch.Tensor) -> torch.Tensor:
    """One small random rotation G_f per cell, fixed per transport instance."""
    G = getattr(self, "_perturbation", None)
    if G is None:
        g = torch.Generator().manual_seed(61)
        P, d = self.face_edge_index.size(0), like.size(-1)
        A = torch.randn(P, d, d, generator=g, dtype=torch.float64)
        G = torch.linalg.matrix_exp(PERTURBATION_ANGLE * (A - A.transpose(-2, -1)))
        G = G.to(dtype=like.dtype, device=like.device)
        self._perturbation = G
    return G


def _up_perturbed(self, s):
    G = _rotation(self, s)
    return (G @ _original_up(self, s).unsqueeze(-1)).squeeze(-1)


def _down_perturbed(self, z):
    G = _rotation(self, z)
    return _original_down(self, (G.transpose(-2, -1) @ z.unsqueeze(-1)).squeeze(-1))


def _apply_perturbed(self, s):
    G = _rotation(self, s)
    return (G.unsqueeze(1) @ _original_apply(self, s).unsqueeze(-1)).squeeze(-1)


def m2_patches() -> list:
    return [mock.patch.object(FactCellTransport, "up", _up_perturbed),
            mock.patch.object(FactCellTransport, "down", _down_perturbed),
            mock.patch.object(FactCellTransport, "apply", _apply_perturbed)]


_original_init = SheafEnergyOperator.__init__


def _init_without_node_cell(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.node_cell = False  # m3: `full` silently loses its node-cell term


def m3_patches() -> list:
    return [mock.patch.object(SheafEnergyOperator, "__init__", _init_without_node_cell)]


_original_degree = SheafEnergyOperator.degree


def _degree_missing_one(self):
    if self._degree is None:
        d_v, d_e, d_f = _original_degree(self)
        if self.groups and self.node_cell and self.fact_cell:
            d_f = self.beta * (2 * self.cell_length).to(self.beta.dtype)  # m4: 2 l_f, not 2 l_f + 1
        self._degree = (d_v, d_e, d_f)
    return self._degree


def m4_patches() -> list:
    return [mock.patch.object(SheafEnergyOperator, "degree", _degree_missing_one)]


def _backward_without_tau(ctx, grad_y):
    """Copy of `AnchoredSolve.backward` with the tau factor of the scalar dropped (m5)."""
    h0, theta_tau, y_star, *params = ctx.saved_tensors
    stats = ctx.stats
    with torch.no_grad():
        operator = ctx.build(*params)
        system = AnchoredSystem(operator, torch.exp(theta_tau))
        lam, iterations, converged, residual = pcg(
            system.apply, grad_y, torch.zeros_like(grad_y), system.jacobi_inverse(),
            ctx.tolerance, ctx.max_iterations, ctx.scope_rows, ctx.num_scopes)
        stats.backward_iterations = int(iterations.max())
        stats.backward_converged = bool(converged.all())
        if not stats.backward_converged:
            raise PCGConvergenceError("backward PCG reached T_max")
    needs = ctx.needs_input_grad
    grad_h0 = lam[:operator.num_nodes] if needs[1] else None
    grad_theta = None
    grad_params = [None] * ctx.num_params
    wanted = [needs[9 + i] for i in range(ctx.num_params)]
    if needs[2] or any(wanted):
        with torch.enable_grad():
            theta_g = theta_tau.detach().requires_grad_(bool(needs[2]))
            params_g = [p.detach().requires_grad_(w) for p, w in zip(params, wanted)]
            operator_g = ctx.build(*params_g)
            scalar = (lam.detach() * operator_g.apply_normalized(y_star.detach())).sum()  # no tau
            leaves = [theta_g] + [p for p, w in zip(params_g, wanted) if w]
            grads = torch.autograd.grad(scalar, leaves, allow_unused=True)
        if needs[2]:
            grad_theta = None if grads[0] is None else -grads[0]
        index = 1
        for i, w in enumerate(wanted):
            if w:
                g = grads[index]
                index += 1
                grad_params[i] = None if g is None else -g
    return (None, grad_h0, grad_theta, None, None, None, None, None, None, *grad_params)


def m5_patches() -> list:
    return [mock.patch.object(AnchoredSolve, "backward", staticmethod(_backward_without_tau))]


def _apply_normalized_wrong_schur(self, y):
    """Copy of `EliminatedOperator.apply_normalized` with the eliminated state weighted by
    D_e^{-1/2} instead of D_e^{-1} (m6)."""
    op = self.operator
    x, s_kept, z = self.layout.split(y)
    inv_v, inv_e, inv_f = self._scales()
    x_t = inv_v.unsqueeze(-1) * x
    z_t = inv_f.unsqueeze(-1) * z
    gather = self._fact_gather(x_t, z_t)
    s_t = inv_e.unsqueeze(-1) * gather  # m6: should be inv_e.square()
    if self.kept.numel():
        s_t = s_t.index_copy(0, self.kept, inv_e[self.kept].unsqueeze(-1) * s_kept)
    d_v, d_e, d_f = op.degree()
    o_x, o_s, o_z = op.off_diagonal(x_t, s_t, z_t, o_s=gather)
    l_x = d_v.unsqueeze(-1) * x_t - o_x
    for term in op.node_terms:
        l_x = l_x + term.apply(x_t, op.num_nodes)
    l_s = (d_e.unsqueeze(-1) * s_t - o_s)[self.kept]
    l_z = d_f.unsqueeze(-1) * z_t - o_z
    return self.layout.join(inv_v.unsqueeze(-1) * l_x,
                            inv_e[self.kept].unsqueeze(-1) * l_s,
                            inv_f.unsqueeze(-1) * l_z)


def m6_patches() -> list:
    return [mock.patch.object(EliminatedOperator, "apply_normalized", _apply_normalized_wrong_schur)]


# ----------------------------------------------------------------------------- the tests
class MutationTests(unittest.TestCase):
    def _check(self, mutation: str, patches: list, test_ids: tuple[str, ...],
               record_only: tuple[str, ...] = ()) -> None:
        """Every test in `test_ids` must pass unpatched and fail under the mutation; the tests in
        `record_only` are run under the mutation and their outcome is recorded, not asserted."""
        for test_id in record_only:
            for patch in patches:
                patch.start()
            try:
                status, message = run_named(test_id)
            finally:
                for patch in reversed(patches):
                    patch.stop()
            MUTATION_RESULTS.setdefault(mutation, {})[test_id.rsplit(".", 1)[-1]] = \
                f"{status} (recorded, not required): {message}"
        for test_id in test_ids:
            status, message = unpatched(test_id)
            if status == "skip":
                self.skipTest(f"{test_id} skipped: {message}")
            self.assertEqual(status, "pass", f"{test_id} must pass without {mutation}: {message}")
            for patch in patches:
                patch.start()
            try:
                status, message = run_named(test_id)
            finally:
                for patch in reversed(patches):
                    patch.stop()
            MUTATION_RESULTS.setdefault(mutation, {})[test_id.rsplit(".", 1)[-1]] = \
                f"{status}: {message}"
            self.assertIn(status, ("fail", "error"), f"{test_id} did not fail under {mutation}")
            if test_id != T1:  # the unpatched code is back (T1 reloads IMDB: checked once above)
                self.assertEqual(run_named(test_id)[0], "pass")

    def test_m1_forward_map_on_reversed_hop_breaks_T1(self) -> None:
        self._check("m1", m1_patches(), (T1,))

    def test_m2_perturbed_fact_cell_map_breaks_T1_and_the_flat_kernel(self) -> None:
        self._check("m2", m2_patches(), (T1, K1))
        patches = m2_patches()
        for patch in patches:
            patch.start()
        try:
            numbers = flat_kernel_numbers(Fixture())
        finally:
            for patch in reversed(patches):
                patch.stop()
        MUTATION_RESULTS["m2"]["flat kernel"] = (
            f"dim ker L_hat = {numbers['dim_ker_L_hat']} (d = 4); smallest eigenvalue "
            f"{numbers['next_eigenvalue']:.3e}")
        self.assertLess(numbers["dim_ker_L_hat"], 4)

    def test_m3_dropped_node_cell_term_breaks_T2(self) -> None:
        self._check("m3", m3_patches(), (T2,))

    def test_m4_missing_plus_one_in_cell_degree_breaks_T2(self) -> None:
        # T3 cannot see this mutation: `apply` takes its diagonal from `degree()`, so a wrong D
        # enters L and its normalisation alike and the diagonal blocks of L_hat stay I. Only the
        # independent dense reference of T2 detects it. T3's outcome is recorded (L8 report).
        self._check("m4", m4_patches(), (T2,), record_only=(T3,))

    def test_m5_backward_without_tau_breaks_the_unroll_check(self) -> None:
        self._check("m5", m5_patches(), (U1,))

    def test_m6_wrong_schur_weight_breaks_T7(self) -> None:
        self._check("m6", m6_patches(), (T7,))


if __name__ == "__main__":
    unittest.main()
