"""Anchored solve (spec v1.11 R13, D-B5; v3-B card U3).

    M Y* = P^T h0,   M = P^T P + tau L_hat,   P = node selector,   tau = exp(theta)

solved by preconditioned conjugate gradients with the Jacobi preconditioner
`J = diag(1 + tau on node blocks, tau on fact and cell blocks)` (exact after normalisation; on the
eliminated operator the scalar diagonal of the reduced `L_hat` replaces the 1), inner products and
stopping per scope (segment sums over a scope id per row; a full batch is one scope; converged
scopes are masked), a fixed tolerance schedule (1e-3 for the first ten epochs, then 1e-4) and
`T_max = 100`, which is a stop condition: the solver raises when a scope has not converged.

Backward (implicit, D-B5): with `g = dLoss/dY*`, solve `M lambda = g` by the same PCG, then
`dLoss/dtheta = -lambda^T (dM/dtheta) Y*` is the gradient of the scalar `tau(theta) lambda^T
L_hat(theta) Y*` with `lambda, Y*` detached, taken by autograd (so `dLoss/dtheta_tau =
-tau lambda^T L_hat Y*` comes for free), and `dLoss/dh0 = P lambda`. Memory is one state vector
times a handful of PCG vectors, independent of the iteration count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch


class AnchoredOperator(Protocol):
    """What the solver needs from `SheafEnergyOperator` / `EliminatedOperator`."""

    num_nodes: int
    layout: object

    def apply_normalized(self, y: torch.Tensor) -> torch.Tensor: ...

    def initial_guess(self, h0: torch.Tensor) -> torch.Tensor: ...


class PCGConvergenceError(RuntimeError):
    """`T_max` reached in some scope (card §3: a stop condition, reported, never tuned)."""


class StopConditionError(RuntimeError):
    """A stop condition of spec §6 fired (J10): the run stops and is reported; never a reason to
    change settings. Carries the condition, the value and the epoch (-1 = an evaluation forward)."""

    def __init__(self, condition: str, value: float, limit: float, epoch: int) -> None:
        self.condition, self.value, self.limit, self.epoch = condition, float(value), limit, epoch
        super().__init__(f"stop condition {condition}: {float(value):.6g} > {limit:g} at epoch "
                         f"{epoch}{' (evaluation forward)' if epoch < 0 else ''} (spec v1.15 §6)")


class TauStopError(StopConditionError):
    """`tau = exp(theta_tau) > tau_stop` (50; spec §3 learned tau, G6)."""

    def __init__(self, value: float, limit: float, epoch: int) -> None:
        super().__init__("tau", value, limit, epoch)


class LogWeightRangeStopError(StopConditionError):
    """Range `max - min` of the cell log-weights `log w_f = g(xi_f)` of one forward above
    `log_weight_range_stop` (30; spec §3 WISP instance weight)."""

    def __init__(self, value: float, limit: float, epoch: int) -> None:
        super().__init__("log_weight_range", value, limit, epoch)


@dataclass
class SolveStats:
    """Iteration counts of the solves of one forward / backward pair (filled in place)."""

    forward_iterations: int = 0
    backward_iterations: int = 0
    forward_residual: float = 0.0
    backward_residual: float = 0.0
    forward_converged: bool = True
    backward_converged: bool = True
    per_scope_forward: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return {
            "pcg_fwd_iters": float(self.forward_iterations),
            "pcg_bwd_iters": float(self.backward_iterations),
            "pcg_fwd_rel_residual": self.forward_residual,
            "pcg_bwd_rel_residual": self.backward_residual,
            "pcg_converged": float(self.forward_converged and self.backward_converged),
        }


def solver_tolerance(epoch: int, warmup_epochs: int, warmup_tolerance: float,
                     tolerance: float) -> float:
    """Fixed schedule (spec §6): `warmup_tolerance` for the first `warmup_epochs` epochs
    (0-indexed `epoch < warmup_epochs`), then `tolerance`. Never tuned."""
    if epoch < 0:
        return tolerance
    return warmup_tolerance if epoch < warmup_epochs else tolerance


def segment_dot(a: torch.Tensor, b: torch.Tensor, scope_rows: torch.Tensor | None,
                num_scopes: int) -> torch.Tensor:
    """`[S]` per-scope inner products `<a, b>_scope` (M1); one scope = the plain inner product."""
    row_dots = (a * b).sum(dim=-1)
    if scope_rows is None or num_scopes == 1:
        return row_dots.sum().reshape(1)
    out = row_dots.new_zeros(num_scopes)
    return out.index_add(0, scope_rows, row_dots)


def pcg(
    apply_m: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    y0: torch.Tensor,
    jacobi_inverse: torch.Tensor,
    tolerance: float,
    max_iterations: int,
    scope_rows: torch.Tensor | None = None,
    num_scopes: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched per-scope PCG (card §1).

    Returns `(Y, iterations [S], converged [S] bool, relative residual [S])`. `jacobi_inverse` is
    `[rows]` (`J^{-1}` is scalar per row), `scope_rows` `[rows]` the scope of each row (None: one
    scope). A converged scope receives `alpha = beta = 0` and stops changing; the loop ends when
    every scope has converged or after `max_iterations`.
    """
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


class AnchoredSystem:
    """`M = P^T P + tau L_hat` on the operator's stacked state, its right-hand side and Jacobi."""

    def __init__(self, operator: AnchoredOperator, tau: torch.Tensor) -> None:
        self.operator = operator
        self.tau = tau
        self.num_nodes = operator.num_nodes

    def apply(self, y: torch.Tensor) -> torch.Tensor:
        out = self.tau * self.operator.apply_normalized(y)
        out = out.clone()
        out[:self.num_nodes] += y[:self.num_nodes]
        return out

    def right_hand_side(self, h0: torch.Tensor) -> torch.Tensor:
        b = h0.new_zeros((self.operator.layout.rows, h0.size(-1)))  # type: ignore[attr-defined]
        b[:self.num_nodes] = h0
        return b

    def jacobi_inverse(self) -> torch.Tensor:
        """`J^{-1}` per row: `1 / (1 + tau d_v)` on nodes, `1 / (tau d)` elsewhere, with `d` the
        scalar diagonal of the (reduced) `L_hat` (1 for the full-state operator)."""
        rows = self.operator.layout.rows  # type: ignore[attr-defined]
        diag = getattr(self.operator, "diagonal_scalars", None)
        if diag is None:
            scalars = self.tau.new_ones(rows)
        else:
            scalars = diag()
        values = self.tau * scalars
        values = values.clone()
        values[:self.num_nodes] += 1.0
        # a row with d = 0 (isolated node: 1 + 0; a cell never has d = 0) stays finite
        return 1.0 / values


def solve_anchored(system: AnchoredSystem, h0: torch.Tensor, y0: torch.Tensor | None,
                   tolerance: float, max_iterations: int,
                   scope_rows: torch.Tensor | None = None, num_scopes: int = 1,
                   strict: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                 torch.Tensor]:
    """One forward solve `M Y = P^T h0` from `y0` (D-B7 initial guess when None)."""
    b = system.right_hand_side(h0)
    if y0 is None:
        y0 = system.operator.initial_guess(h0)
    y, iterations, converged, residual = pcg(system.apply, b, y0, system.jacobi_inverse(),
                                             tolerance, max_iterations, scope_rows, num_scopes)
    if strict and not bool(converged.all()):
        raise PCGConvergenceError(
            f"PCG reached T_max = {max_iterations} in {int((~converged).sum())} scope(s); "
            f"worst relative residual {float(residual.max()):.3e} (tolerance {tolerance:.1e}).")
    return y, iterations, converged, residual


class AnchoredSolve(torch.autograd.Function):
    """`Y* = AnchoredSolve.apply(build, h0, theta_tau, y0, tolerance, max_iterations, stats,
    scope_rows, num_scopes, *params)`.

    `build(*params)` rebuilds the (reduced) operator from the parameter tensors it depends on;
    the forward runs under no_grad, the backward recomputes `tau lambda^T L_hat Y*` with grad.
    """

    @staticmethod
    def forward(ctx, build: Callable[..., AnchoredOperator], h0: torch.Tensor,  # type: ignore[override]
                theta_tau: torch.Tensor, y0: torch.Tensor | None, tolerance: float,
                max_iterations: int, stats: SolveStats, scope_rows: torch.Tensor | None,
                num_scopes: int, *params: torch.Tensor) -> torch.Tensor:
        operator = build(*params)
        tau = torch.exp(theta_tau)
        system = AnchoredSystem(operator, tau)
        y, iterations, converged, residual = solve_anchored(
            system, h0, y0, tolerance, max_iterations, scope_rows, num_scopes, strict=True)
        stats.forward_iterations = int(iterations.max())
        stats.forward_converged = bool(converged.all())
        stats.forward_residual = float(residual.max())
        stats.per_scope_forward = iterations.tolist()
        ctx.build = build
        ctx.stats = stats
        ctx.tolerance = tolerance
        ctx.max_iterations = max_iterations
        ctx.scope_rows = scope_rows
        ctx.num_scopes = num_scopes
        ctx.num_params = len(params)
        ctx.save_for_backward(h0, theta_tau, y, *params)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):  # type: ignore[override]
        h0, theta_tau, y_star, *params = ctx.saved_tensors
        stats: SolveStats = ctx.stats
        with torch.no_grad():
            operator = ctx.build(*params)
            tau = torch.exp(theta_tau)
            system = AnchoredSystem(operator, tau)
            lam, iterations, converged, residual = pcg(
                system.apply, grad_y, torch.zeros_like(grad_y), system.jacobi_inverse(),
                ctx.tolerance, ctx.max_iterations, ctx.scope_rows, ctx.num_scopes)
            stats.backward_iterations = int(iterations.max())
            stats.backward_converged = bool(converged.all())
            stats.backward_residual = float(residual.max())
            if not stats.backward_converged:
                raise PCGConvergenceError(
                    f"backward PCG reached T_max = {ctx.max_iterations}; worst relative residual "
                    f"{float(residual.max()):.3e}.")
        needs = ctx.needs_input_grad
        grad_h0 = lam[:operator.num_nodes] if needs[1] else None
        grad_theta = None
        grad_params: list[torch.Tensor | None] = [None] * ctx.num_params
        wanted = [needs[9 + i] for i in range(ctx.num_params)]
        if needs[2] or any(wanted):
            with torch.enable_grad():
                theta_g = theta_tau.detach().requires_grad_(bool(needs[2]))
                params_g = [p.detach().requires_grad_(w) for p, w in zip(params, wanted)]
                operator_g = ctx.build(*params_g)
                scalar = torch.exp(theta_g) * (lam.detach()
                                               * operator_g.apply_normalized(y_star.detach())).sum()
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


def anchored_solve(build: Callable[..., AnchoredOperator], h0: torch.Tensor,
                   theta_tau: torch.Tensor, params: tuple[torch.Tensor, ...],
                   tolerance: float, max_iterations: int, stats: SolveStats,
                   y0: torch.Tensor | None = None, scope_rows: torch.Tensor | None = None,
                   num_scopes: int = 1) -> torch.Tensor:
    """Convenience wrapper around `AnchoredSolve.apply` (keyword arguments)."""
    return AnchoredSolve.apply(build, h0, theta_tau, y0, tolerance, max_iterations, stats,
                               scope_rows, num_scopes, *params)


__all__ = ["AnchoredSolve", "AnchoredSystem", "LogWeightRangeStopError", "PCGConvergenceError",
           "SolveStats", "StopConditionError", "TauStopError",
           "anchored_solve", "pcg", "segment_dot", "solve_anchored", "solver_tolerance"]
