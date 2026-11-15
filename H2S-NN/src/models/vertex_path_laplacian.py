"""Path-cell terms used by the v3-B sheaf energy (`src/models/sheaf_energy.py`).

`PathCellTerm` is the 2-cell term of a face group through the forced transports
(`PathTransportOperator`), `HolonomyEdgeTerm` the closure edge of the arm `closure_only`, and
`PathTerm` / `IncidenceOperator` their interfaces. The v2 operator that also lived here
(`VertexPathSheafLaplacian`, the endpoint-edge and free-incidence ablations) was removed by the
legacy-removal card (v1/v2); the classes below are unchanged.
"""

from __future__ import annotations

from typing import Protocol

import torch

from src.models.map_operators import PathTransportOperator


class IncidenceOperator(Protocol):
    """Node cochain -> per-incidence values `[P, n, d]` in each cell's stalk, and adjoint."""

    face_node_index: torch.Tensor

    def apply(self, x: torch.Tensor) -> torch.Tensor: ...

    def apply_T(self, w: torch.Tensor, num_nodes: int) -> torch.Tensor: ...

    def squared_scales(self) -> torch.Tensor: ...


class PathCellTerm:
    """`sum_f w_f B_f^T (I - 1 1^T / |f|) B_f` over one face group (fixed `|f| = L + 1`)."""

    def __init__(self, incidence: IncidenceOperator, weights: torch.Tensor) -> None:
        num_paths, size = incidence.face_node_index.shape
        if weights.shape != (num_paths,):
            raise ValueError(f"weights must be [{num_paths}], got {tuple(weights.shape)}.")
        if size < 2:
            raise ValueError("a path cell needs at least two nodes.")
        self.incidence = incidence
        self.weights = weights
        self.cell_size = size

    def cell_values(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(y [P, n, d], z [P, d])`: transported node values and the 2-cell state."""
        y = self.incidence.apply(x)
        return y, y.mean(dim=1)

    def apply(self, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if isinstance(self.incidence, PathTransportOperator):
            return self._apply_transport(x, num_nodes)
        y, z = self.cell_values(x)
        residual = (y - z.unsqueeze(1)) * self.weights.view(-1, 1, 1)
        return self.incidence.apply_T(residual, num_nodes)

    def _apply_transport(self, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """O(L) form (D9), exact because `T_{v_i}^T T_{v_i} = s_i^2 I`:

            z_f = (sum_i T_{v_i} x_{v_i}) / n,   (L_f x)_{v_i} = w_f (s_i^2 x_{v_i} - T_{v_i}^T z_f).
        """
        transport = self.incidence
        nodes = transport.face_node_index
        z = transport.transport_sum(x) / self.cell_size
        pulled = transport.adjoint_prefix(z)
        own = x[nodes] * transport.squared_scales().unsqueeze(-1)
        residual = (own - pulled) * self.weights.view(-1, 1, 1)
        output = x.new_zeros((num_nodes, x.size(-1)))
        output.index_add_(0, nodes.reshape(-1), residual.reshape(-1, x.size(-1)))
        return output

    def diagonal(self, num_nodes: int) -> torch.Tensor:
        """Per-node scalar diagonal: `sum w_f (1 - 1/|f|) s_{v,f}^2`."""
        per_incidence = (
            (1.0 - 1.0 / self.cell_size)
            * self.weights.unsqueeze(1)
            * self.incidence.squared_scales()
        )
        output = per_incidence.new_zeros(num_nodes)
        output.index_add_(0, self.incidence.face_node_index.reshape(-1), per_incidence.reshape(-1))
        return output


class HolonomyEdgeTerm:
    """v1-style shortcut (ablation `path_mode: holonomy_edges`): one closure edge
    `(v_L -> v_0)` per path with `F_src = T_{v_L}` (the path holonomy) and `F_tgt = I`,
    i.e. `delta_f x = x_{v_0} - T_{v_L} x_{v_L}`, applied matrix-free."""

    def __init__(self, transport: "PathTransportOperator", weights: torch.Tensor) -> None:  # noqa: F821
        if weights.shape != (transport.face_node_index.size(0),):
            raise ValueError("weights must have one entry per path.")
        self.transport = transport
        self.first = transport.face_node_index[:, 0]
        self.last = transport.face_node_index[:, -1]
        self.weights = weights

    def apply(self, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
        delta = x[self.first] - self.transport.apply_holonomy(x[self.last])
        weighted = delta * self.weights.unsqueeze(-1)
        output = torch.zeros_like(x)
        output.index_add_(0, self.first, weighted)
        output.index_add_(0, self.last, -self.transport.apply_holonomy_T(weighted))
        return output

    def diagonal(self, num_nodes: int) -> torch.Tensor:
        output = self.weights.new_zeros(num_nodes)
        output.index_add_(0, self.first, self.weights)
        output.index_add_(0, self.last, self.weights * self.transport.squared_scales()[:, -1])
        return output


class PathTerm(Protocol):
    """A path-coupling term (2-cell, or an ablation's per-path edge): `apply` and `diagonal`."""

    def apply(self, x: torch.Tensor, num_nodes: int) -> torch.Tensor: ...

    def diagonal(self, num_nodes: int) -> torch.Tensor: ...
