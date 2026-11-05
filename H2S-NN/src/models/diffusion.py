from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn


def _softplus_inverse(x: float) -> float:
    """Returns y such that softplus(y) = x, so the learnable parameter
    initialises to the correct step size rather than softplus(x)."""
    return math.log(math.exp(x) - 1.0)


class HodgeDiffusion(nn.Module):
    def __init__(
        self,
        num_steps: int,
        eta_0: float,
        eta_1: float,
        learnable_eta: bool,
        layer_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.dropout = nn.Dropout(layer_dropout)
        if learnable_eta:
            # Store softplus_inverse so that _step_size(param) = softplus(softplus_inverse(eta)) = eta.
            self.eta_0 = nn.Parameter(torch.tensor(_softplus_inverse(eta_0)))
            self.eta_1 = nn.Parameter(torch.tensor(_softplus_inverse(eta_1)))
        else:
            self.register_buffer("eta_0", torch.tensor(float(eta_0)))
            self.register_buffer("eta_1", torch.tensor(float(eta_1)))
        self.learnable_eta = learnable_eta

    def _step_size(self, eta: torch.Tensor) -> torch.Tensor:
        if self.learnable_eta:
            return torch.nn.functional.softplus(eta)
        return eta

    def forward(
        self,
        x0: torch.Tensor,
        s0: torch.Tensor,
        apply_l0: Callable[[torch.Tensor], torch.Tensor],
        apply_l1: Callable[[torch.Tensor], torch.Tensor],
        num_steps: int | None = None,
        record_trajectory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        """`num_steps` overrides `self.num_steps` for illustrative rollouts (e.g.
        visualizing diffusion beyond the trained K) without touching the trained
        step sizes. `record_trajectory` returns the per-step node cochain history
        (including x0) for analysis; both are no-ops for the normal training/eval path."""
        x = x0
        s = s0
        steps = self.num_steps if num_steps is None else num_steps
        trajectory = [x0.detach()] if record_trajectory else None
        eta_0 = self._step_size(self.eta_0)
        eta_1 = self._step_size(self.eta_1)
        for _ in range(steps):
            x = x - eta_0 * apply_l0(x)
            s = s - eta_1 * apply_l1(s)
            x = self.dropout(x)
            s = self.dropout(s)
            if record_trajectory:
                trajectory.append(x.detach())
        return x, s, trajectory
