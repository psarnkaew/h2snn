"""Matrix-free sheaf coboundary on the observed edges (decision D1).

`(delta0 x)_e = F^e_tgt x_tgt - F^e_src x_src`, its adjoint, and `Delta_0 = delta0^T delta0`,
applied through map operators (`EdgeMapPair`); no `[num_edges, d, d]` tensor is formed.
"""

from __future__ import annotations

import torch

from src.models.map_operators import EdgeMapPair, checkpoint_if_grad


class CoboundaryOps:
    # Each public entry point runs under `checkpoint_if_grad`: autograd keeps only its input
    # (for `L0`, the [N, d] node cochain) instead of every [E, d] intermediate.
    @staticmethod
    def _delta0_mf(
        x: torch.Tensor,
        maps: EdgeMapPair,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
    ) -> torch.Tensor:
        return maps.tgt.apply(x[tgt_idx]) - maps.src.apply(x[src_idx])

    @staticmethod
    def _delta0_T_mf(
        s: torch.Tensor,
        maps: EdgeMapPair,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        output = torch.zeros((num_nodes, s.size(-1)), device=s.device, dtype=s.dtype)
        output.index_add_(0, tgt_idx, maps.tgt.apply_T(s))
        output.index_add_(0, src_idx, -maps.src.apply_T(s))
        return output

    @staticmethod
    def delta0_matrix_free(
        x: torch.Tensor,
        maps: EdgeMapPair,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """`(delta0 x)_e = F^e_tgt x_tgt - F^e_src x_src`."""
        return checkpoint_if_grad(
            lambda x_in: CoboundaryOps._delta0_mf(x_in, maps, src_idx, tgt_idx), x
        )

    @staticmethod
    def delta0_T_matrix_free(
        s: torch.Tensor,
        maps: EdgeMapPair,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Exact adjoint of `delta0_matrix_free`."""
        return checkpoint_if_grad(
            lambda s_in: CoboundaryOps._delta0_T_mf(s_in, maps, src_idx, tgt_idx, num_nodes), s
        )

    @staticmethod
    def L0_matrix_free(
        x: torch.Tensor,
        maps: EdgeMapPair,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        num_nodes: int,
        checkpoint: bool = True,
    ) -> torch.Tensor:
        """`delta0^T delta0 x` as one checkpointed block: nothing edge-sized is retained."""

        def l0(x_in: torch.Tensor) -> torch.Tensor:
            edge_values = CoboundaryOps._delta0_mf(x_in, maps, src_idx, tgt_idx)
            return CoboundaryOps._delta0_T_mf(edge_values, maps, src_idx, tgt_idx, num_nodes)

        return checkpoint_if_grad(l0, x, enabled=checkpoint)
