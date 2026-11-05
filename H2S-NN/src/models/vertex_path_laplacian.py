"""Vertex-2-cell sheaf Laplacian (v2 decision D4, plan §4.2-4.4).

Each path-supported 2-cell `f` with nodes `v_0..v_L` couples to every node on it through the
node->2-cell maps `T_{v_i}` (ordered parallel transport, `PathTransportOperator`):

    y_{f,i} = T_{v_i} x_{v_i},     z_f = mean_i y_{f,i}  (= argmin_z sum_i ||y_{f,i} - z||^2)
    (L_P x)_v = sum_{f, i: v_i = v} w_f T_{v_i}^T (y_{f,i} - z_f)

so `L_f = w_f B_f^T (I - 1 1^T / |f|) B_f` with `B_f = blockdiag(T_{v_i})` stacked: the Hodge
0-Laplacian of the pullback sheaf on the star expansion (Prop. 2). The full operator is

    L = Delta_0 + gamma L_P,      x <- x - eta D^{-1/2} L D^{-1/2} x,

with `D` the per-node scalar degree built from the operator's own terms (Prop. 3):
edge part `sum_{incidences of v} c_{v<|e}^2`, path part `gamma sum_{f ∋ v} w_f (1 - 1/|f|) s_{v,f}^2`
(`s = 1` under per-edge scales). For a self-loop the edge part counts both incidences,
`c_src^2 + c_tgt^2`: that is Prop. 3's `D_e`, not the (non-scalar) diagonal block
`(F_tgt - F_src)^T (F_tgt - F_src)` of `L`, and it keeps `L <= 2D`. Nodes with `D_v = 0`
get a zero row.
"""

from __future__ import annotations

from typing import Protocol

import torch

from src.models.coboundary_ops import CoboundaryOps
from src.models.map_operators import EdgeMapPair, FactorizedMapOperator, checkpoint_if_grad


class IncidenceOperator(Protocol):
    """Node cochain -> per-incidence values `[P, n, d]` in each cell's stalk, and adjoint."""

    face_node_index: torch.Tensor

    def apply(self, x: torch.Tensor) -> torch.Tensor: ...

    def apply_T(self, w: torch.Tensor, num_nodes: int) -> torch.Tensor: ...

    def squared_scales(self) -> torch.Tensor: ...


class FreeIncidenceOperator:
    """Per-incidence orthogonal maps `G_{v,f}` (ablation `path_maps: free`).

    Same factorized family as the node->edge maps, `R_{position} H_k(u_{v,f})`, but each
    incidence `(v, f)` has its own map instead of the transport forced by functoriality.
    Row `p * n + i` of `maps` is the map of position `i` of path `p`.
    """

    def __init__(self, maps: FactorizedMapOperator, face_node_index: torch.Tensor) -> None:
        if face_node_index.dim() != 2:
            raise ValueError("face_node_index must be [P, n].")
        if maps.num_edges != face_node_index.numel():
            raise ValueError(
                f"free maps must have one row per incidence ({face_node_index.numel()}), "
                f"got {maps.num_edges}."
            )
        self.maps = maps
        self.face_node_index = face_node_index

    def squared_scales(self) -> torch.Tensor:
        return torch.ones(self.face_node_index.shape, dtype=self.maps.reflectors.dtype,
                          device=self.maps.reflectors.device)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        num_paths, n = self.face_node_index.shape
        values = x[self.face_node_index].reshape(num_paths * n, x.size(-1))
        return self.maps.apply_orthogonal(values).view(num_paths, n, x.size(-1))

    def apply_T(self, w: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if w.shape[:2] != self.face_node_index.shape:
            raise ValueError("w must be [P, n, d] matching face_node_index.")
        values = self.maps.apply_orthogonal_T(w.reshape(-1, w.size(-1)))
        output = w.new_zeros((num_nodes, w.size(-1)))
        output.index_add_(0, self.face_node_index.reshape(-1), values)
        return output


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
        y, z = self.cell_values(x)
        residual = (y - z.unsqueeze(1)) * self.weights.view(-1, 1, 1)
        return self.incidence.apply_T(residual, num_nodes)

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


class PathTerm(Protocol):
    """A path-coupling term (2-cell, or an ablation's per-path edge): `apply` and `diagonal`."""

    def apply(self, x: torch.Tensor, num_nodes: int) -> torch.Tensor: ...

    def diagonal(self, num_nodes: int) -> torch.Tensor: ...


def _safe_inverse_sqrt(degree: torch.Tensor) -> torch.Tensor:
    """`D^{-1/2}` with 0 where `D = 0` (no inf/NaN in forward or backward)."""
    positive = degree > 0
    safe = torch.where(positive, degree, torch.ones_like(degree))
    return torch.where(positive, safe.rsqrt(), torch.zeros_like(degree))


class VertexPathSheafLaplacian:
    """`L = Delta_0 + gamma sum_terms`, its scalar degree `D`, and `D^{-1/2} L D^{-1/2}`.

    Args:
        edge_maps: node->edge map operators of the observed edges (each side has `.scale`).
        edge_index: `[2, M]` observed edges (no closure edges).
        num_nodes: number of nodes `N`.
        path_terms: one term per face group (may be empty: `gamma` is then unused).
        gamma: scalar tensor `gamma >= 0`, or None when there are no path terms.
        checkpoint: run each matrix-free application under activation checkpointing.
    """

    def __init__(
        self,
        edge_maps: EdgeMapPair,
        edge_index: torch.Tensor,
        num_nodes: int,
        path_terms: tuple[PathTerm, ...],
        gamma: torch.Tensor | None,
        checkpoint: bool,
    ) -> None:
        if path_terms and gamma is None:
            raise ValueError("gamma is required when path terms are present.")
        self.edge_maps = edge_maps
        self.src_idx = edge_index[0]
        self.tgt_idx = edge_index[1]
        self.num_nodes = num_nodes
        self.path_terms = path_terms
        self.gamma = gamma
        self.checkpoint = checkpoint
        self._inverse_sqrt: torch.Tensor | None = None

    def edge_degree(self) -> torch.Tensor:
        """`sum_{incidences of v} c^2` (self-loops count both incidences)."""
        src_scale = self.edge_maps.src.scale
        tgt_scale = self.edge_maps.tgt.scale
        output = src_scale.new_zeros(self.num_nodes)
        output.index_add_(0, self.src_idx, src_scale.square())
        output.index_add_(0, self.tgt_idx, tgt_scale.square())
        return output

    def degree(self) -> torch.Tensor:
        """Scalar `D_v` of the full operator (edge part + gamma * path part)."""
        output = self.edge_degree()
        if self.path_terms:
            path = sum(term.diagonal(self.num_nodes) for term in self.path_terms)
            output = output + self.gamma * path
        return output

    def inverse_sqrt_degree(self) -> torch.Tensor:
        if self._inverse_sqrt is None:
            self._inverse_sqrt = _safe_inverse_sqrt(self.degree())
        return self._inverse_sqrt

    def apply_path(self, x: torch.Tensor) -> torch.Tensor:
        """`sum_terms term(x)` (without gamma)."""
        output = torch.zeros_like(x)
        for term in self.path_terms:
            output = output + checkpoint_if_grad(
                lambda x_in, t=term: t.apply(x_in, self.num_nodes), x, enabled=self.checkpoint
            )
        return output

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalized `L x`."""
        output = CoboundaryOps.L0_matrix_free(
            x=x,
            maps=self.edge_maps,
            src_idx=self.src_idx,
            tgt_idx=self.tgt_idx,
            num_nodes=self.num_nodes,
            checkpoint=self.checkpoint,
        )
        if self.path_terms:
            output = output + self.gamma * self.apply_path(x)
        return output

    def apply_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """`D^{-1/2} L D^{-1/2} x`; rows (and columns) of nodes with `D_v = 0` are zero."""
        scale = self.inverse_sqrt_degree().unsqueeze(-1)
        return scale * self.apply(scale * x)
