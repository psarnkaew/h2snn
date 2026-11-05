from __future__ import annotations

import torch
import torch.utils.checkpoint
from torch import nn

from src.models.fasthpp import (
    householder_product_sequential,
    householder_product_wy,
)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


class Orthogonal(nn.Module):
    def __init__(self, d: int, orthogonal_map: str) -> None:
        super().__init__()
        if orthogonal_map != "fasthpp":
            raise ValueError(
                f"Unsupported orthogonal_map '{orthogonal_map}'; expected 'fasthpp'."
            )
        self.d = d
        self.orthogonal_map = orthogonal_map
        self.parameter_count = self.num_parameters(d, orthogonal_map)

    @staticmethod
    def num_parameters(d: int, orthogonal_map: str) -> int:
        if orthogonal_map != "fasthpp":
            raise ValueError(
                f"Unsupported orthogonal_map '{orthogonal_map}'; expected 'fasthpp'."
            )
        return d * (d - 1) // 2

    @staticmethod
    def _householder_sequential(params: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Reference/fallback path implemented inside ``fasthpp.py``."""
        return householder_product_sequential(params, eps)

    @staticmethod
    def _householder_wy(params: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """FastHPP compact-WY builder for a materialized Householder product.

        Produces the same matrix as `_householder_sequential` to float64 roundoff (asserted in
        `tests/test_cuda_hardening.py`), but autograd retains `O(log d)` activations instead of
        `O(d)`. This trades compute for memory, and the trade only pays at larger `d`.

        Measured, one forward+backward, RTX 4060 8 GiB. Isolated over a batch of independent
        maps the peak is 1.36x lower at d=8, 1.98x at d=16, 3.19x at d=32. End to end:

            imdb d=8  K=4   3.34 -> 3.18 GiB   ( -5%)   649 -> 744 ms  (+15%)
            acm  d=8  K=1   8.40 -> 7.47 GiB   (-11%)   both thrash on this card
            imdb d=16 K=4   8.58 -> 6.85 GiB   (-20%)   both thrash on this card

        So: roughly 15% slower in compute (the clean d=8 reading), buying a memory cut that
        grows with `d`. Timings on rows that exceed the card are dominated by WDDM spilling and
        are not compute measurements.

        Adapted from the FastH reference implementation (`fasth/fasthpp.py`, MIT-licensed;
        Mathiasen et al., "What if Neural Networks had SVDs?", NeurIPS 2020), which in turn
        rests on the WY representation of Bischof & Van Loan (1987). Note that FastH targets a
        *single* large orthogonal matrix and optimises wall-clock GPU utilisation; here the
        batch is already ~1e6 tiny maps and the merge is used for its autograd-retention
        profile instead.

        Requires a square, power-of-two reflector count -- callers dispatch via
        `_householder_orgqr`.
        """
        return householder_product_wy(params, eps)

    @classmethod
    def _householder_orgqr(cls, params: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Build an orthogonal matrix from normalized Householder reflectors.

        Square power-of-two `d` -- every `d` any sweep uses -- takes the WY merge, and when
        gradients are being tracked the merge is checkpointed: its `log2(d)` intermediates are
        dropped and replayed during backward, taking retention from `O(batch * d^2 * log d)`
        down to `O(batch * d^2)`. Gradients are identical by construction, since checkpointing
        replays this same forward rather than substituting a hand-derived backward.

        Measured end to end (RTX 4060, one forward+backward), against the sequential builder:

            acm  d=8  K=1   8.40 -> 7.46 (WY) -> 5.97 GiB (WY + checkpoint)   -29%
            imdb d=16 K=4   8.58 -> 6.85 (WY) -> 5.65 GiB                     -34%

        Both of those rows exceed this 8 GiB card, so their timings measure WDDM spilling, not
        compute; see `tests`/`docs/roadA_memory_handoff.md` for the clean readings.
        """
        _, d, r = params.shape
        if not (torch.is_grad_enabled() and params.requires_grad):
            # Nothing to retain, so checkpointing would only add a redundant forward.
            return cls._householder_wy(params, eps)
        if not (d == r and _is_power_of_two(d)):
            # The public FastHPP function owns the exact non-power-of-two
            # fallback, so the active path never bypasses fasthpp.py.
            return cls._householder_wy(params, eps)
        return torch.utils.checkpoint.checkpoint(
            cls._householder_wy,
            params,
            eps,
            use_reentrant=False,
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        tril_indices = torch.tril_indices(
            row=self.d,
            col=self.d,
            offset=-1,
            device=params.device,
        )
        lifted = torch.zeros(
            (params.size(0), self.d, self.d),
            dtype=params.dtype,
            device=params.device,
        )
        lifted[:, tril_indices[0], tril_indices[1]] = params
        identity = torch.eye(
            self.d,
            dtype=params.dtype,
            device=params.device,
        ).unsqueeze(0)
        return self._householder_orgqr(lifted + identity)
