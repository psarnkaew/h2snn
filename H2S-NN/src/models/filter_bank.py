"""Gated Chebyshev filter bank over the vertex-2-cell sheaf (spec v1.2 §3-§4; task card Part G, G2).

    x   = sum_{k=0..K} T_k(A - I) (h W_k),     W_k = V_k / max(1, sum_j ||V_j||_2)

`A = G Delta_hat B` is the D10/D11 operator, passed in as a function on `[N, d]` states; `T_k` are
Chebyshev polynomials and `W_k` (d x d) acts on stalk coordinates (row form `h W_k`). The sum is
evaluated by Clenshaw's recurrence with the vector-valued coefficients `h W_k`:

    b_K = h W_K,   b_k = h W_k + 2 L b_{k+1} - b_{k+2},   x = h W_0 + L b_1 - b_2,   L = A - I,

which equals the card's `T_0 = h, T_1 = L h, T_{k+1} = 2 L T_k - T_{k-1}` form term by term, with K
applications of `A` and nothing materialized. Init `V_k = theta_k I`, theta the Chebyshev
coefficients of `(1 - eta0 (1 + xi))^K`: the filter starts exactly at the first-order update
`(I - eta0 A)^K` (Prop. 7); `sum |theta_k| = 1` for eta0 in (0, 1), so the normalization is inactive
at the start. Gain <= sum_k ||W_k||_2 <= 1 in the metric `M = B G^-1` for every parameter value
(Prop. 5), provided the spectrum of `A` lies in [0, 2] (Prop. 3 with gates in (0, 1)).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from numpy.polynomial import chebyshev, polynomial
from torch import nn

ApplyA = Callable[[torch.Tensor], torch.Tensor]


def first_order_chebyshev_coefficients(eta: float, K: int) -> np.ndarray:
    """`[K + 1]` Chebyshev coefficients of `p(xi) = (1 - eta (1 + xi))^K` on [-1, 1] (float64)."""
    power_series = np.array([1.0])
    for _ in range(K):
        power_series = polynomial.polymul(power_series, np.array([1.0 - eta, -eta]))
    coefficients = np.zeros(K + 1)
    cheb = chebyshev.poly2cheb(power_series)
    coefficients[: cheb.size] = cheb
    return coefficients


def chebyshev_shift(apply_a: ApplyA, x: torch.Tensor) -> torch.Tensor:
    """`(A - I) x`: the argument of the Chebyshev polynomials (spectrum of A in [0, 2] -> [-1, 1])."""
    return apply_a(x) - x


class ChebyshevFilterBank(nn.Module):
    def __init__(self, d: int, K: int, eta0: float) -> None:
        super().__init__()
        if K < 0:
            raise ValueError(f"K must be non-negative, got {K}.")
        self.d, self.K, self.eta0 = d, K, eta0
        self.V = nn.Parameter(torch.empty(K + 1, d, d))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """`V_k = theta_k I` in the parameter's current dtype (the D11 initialization)."""
        theta = torch.as_tensor(first_order_chebyshev_coefficients(self.eta0, self.K),
                                dtype=self.V.dtype, device=self.V.device)
        with torch.no_grad():
            eye = torch.eye(self.d, dtype=self.V.dtype, device=self.V.device)
            self.V.copy_(theta.view(-1, 1, 1) * eye)

    def norm_sum(self) -> torch.Tensor:
        """`sum_k ||V_k||_2` (exact spectral norms of the d x d blocks)."""
        return torch.linalg.matrix_norm(self.V, ord=2).sum()

    def weights(self) -> torch.Tensor:
        """`W_k = V_k / max(1, sum_j ||V_j||_2)`, so `sum_k ||W_k||_2 <= 1`."""
        return self.V / torch.clamp(self.norm_sum(), min=1.0)

    @torch.no_grad()
    def diagnostic_norm_sum(self) -> float:
        """Logged `sum_k ||W_k||_2`: the model's `W_k` (its own dtype) cast to float64 before the
        spectral norms. A float32 SVD rounds an active cap to 1 +/- O(1e-6) (D12 G3b decision 1);
        the cap itself (`weights`) is unchanged."""
        return float(torch.linalg.matrix_norm(self.weights().double(), ord=2).sum())

    def forward(self, apply_a: ApplyA, h: torch.Tensor) -> torch.Tensor:
        weights = self.weights()

        def coefficient(k: int) -> torch.Tensor:
            return h @ weights[k]

        if self.K == 0:
            return coefficient(0)
        b_next, b_after = coefficient(self.K), torch.zeros_like(h)
        for k in range(self.K - 1, 0, -1):
            b_next, b_after = (coefficient(k) + 2 * chebyshev_shift(apply_a, b_next) - b_after,
                               b_next)
        return coefficient(0) + chebyshev_shift(apply_a, b_next) - b_after

    def adjoint(self, apply_a: ApplyA, y: torch.Tensor) -> torch.Tensor:
        """`F* y = sum_k (T_k(A - I) y) W_k^T`: the adjoint of the filter in the metric `M` when
        `A` is M-self-adjoint (Prop. 4); used by the gain diagnostic only."""
        weights = self.weights()
        t_prev, t_curr = y, chebyshev_shift(apply_a, y)
        output = t_prev @ weights[0].T
        if self.K >= 1:
            output = output + t_curr @ weights[1].T
        for k in range(2, self.K + 1):
            t_prev, t_curr = t_curr, 2 * chebyshev_shift(apply_a, t_curr) - t_prev
            output = output + t_curr @ weights[k].T
        return output

    @torch.no_grad()
    def gain(self, apply_a: ApplyA, metric: torch.Tensor, shape: tuple[int, int],
             steps: int = 30, seed: int = 0) -> float:
        """Power iteration of `F* F` in `<x, y>_M = sum_v M_v x_v . y_v`: a lower bound of the
        M-gain `||F||_M` that converges from below (diagnostic; Prop. 5 bounds it by
        `sum_k ||W_k||_2`)."""
        generator = torch.Generator(device=metric.device).manual_seed(seed)
        weight = metric.unsqueeze(-1)

        def m_norm(x: torch.Tensor) -> torch.Tensor:
            return (weight * x * x).sum().sqrt()

        x = torch.randn(shape, generator=generator, device=metric.device, dtype=metric.dtype)
        x = x / m_norm(x)
        for _ in range(steps):
            x = self.adjoint(apply_a, self.forward(apply_a, x))
            norm = m_norm(x)
            if float(norm) == 0.0:
                return 0.0
            x = x / norm
        return float(m_norm(self.forward(apply_a, x)) / m_norm(x))


__all__ = ["ChebyshevFilterBank", "chebyshev_shift", "first_order_chebyshev_coefficients"]
