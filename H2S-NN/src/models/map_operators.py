"""Matrix-free node->edge restriction-map operators (v2 decision D1).

A factorized restriction map is

    F_{v<|e} = c_{v<|e} * R_{psi(e),side} * H_k(u_{v<|e}),
    H_k(u)   = H(u_1) H(u_2) ... H(u_k),   H(u) = I - 2 u u^T / ||u||^2,

with one positive scalar `c` and `k` reflection vectors per incidence and one orthogonal
`R` per (relation, side). The product is conformal-orthogonal, so the inverse is closed
form, `F^-1 = c^-1 H_k^T R^T`, and the v1 transport construction (and with it
`delta1 . delta0 = 0`) carries over unchanged.

The operators below store only the factors -- `E*(k*d + 1)` numbers per side plus the
shared `R` table -- and apply the maps to vectors. Nothing of shape `[E, d, d]` is built by
`apply`, `apply_T`, `apply_inverse` or `apply_inverse_T`; `materialize` exists for tests
and for small gathered subsets (the v1 face chunks) only.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import torch
import torch.nn.functional as F


def normalize_reflectors(raw: torch.Tensor, eps: float) -> torch.Tensor:
    """Unit Householder vectors along the last dim, with a finite-gradient guard.

    `u / sqrt(max(||u||^2, eps^2))` is exactly `u / ||u||` whenever `||u|| >= eps`, so the
    reflection is exactly orthogonal there. Inside the ball `||u|| < eps` the vector is
    shorter than one and the factor is not orthogonal. That is unavoidable for any guard
    that stays finite at `u = 0`: a reflection has determinant -1 and the identity +1, so
    a continuous path between them passes through a singular matrix. The guard keeps the
    forward and backward pass finite (no NaN) at and near `u = 0`; `eps` keeps the ball
    negligibly small.
    """
    if eps <= 0.0:
        raise ValueError(f"reflector eps must be positive, got {eps}.")
    squared = (raw * raw).sum(dim=-1, keepdim=True)
    return raw / torch.sqrt(squared.clamp_min(eps * eps))


class MapOperator(Protocol):
    """One side's restriction maps over `num_edges` edges, applied to `[E, ..., d]`."""

    @property
    def num_edges(self) -> int: ...

    def apply(self, x: torch.Tensor) -> torch.Tensor: ...

    def apply_T(self, y: torch.Tensor) -> torch.Tensor: ...

    def stored_elements(self) -> int: ...


class EdgeMapPair(NamedTuple):
    """Source-side and target-side restriction-map operators over the same edge list."""

    src: MapOperator
    tgt: MapOperator


def _broadcast_rows(values: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """View a per-edge tensor `[E, ...]` so it broadcasts against `like` `[E, *, d]`."""
    extra = like.dim() - values.dim()
    return values.reshape(values.shape + (1,) * extra)


class FactorizedMapOperator:
    """`F_e = c_e R_{g(e)} H(v_{e,1}) ... H(v_{e,k})` for `E` edges, one side.

    Args:
        scale: `[E]` positive conformal scale `c`.
        reflectors: `[E, k, d]` unit Householder vectors (see `normalize_reflectors`).
        rotations: `[G, d, d]` orthogonal frames, one per relation for this side.
        group_ids: `[E]` relation id of each edge, in `[0, G)`.
    """

    def __init__(
        self,
        scale: torch.Tensor,
        reflectors: torch.Tensor,
        rotations: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> None:
        if reflectors.dim() != 3:
            raise ValueError(f"reflectors must be [E, k, d], got {tuple(reflectors.shape)}.")
        num_edges, _, d = reflectors.shape
        if scale.shape != (num_edges,):
            raise ValueError(f"scale must be [{num_edges}], got {tuple(scale.shape)}.")
        if group_ids.shape != (num_edges,):
            raise ValueError(f"group_ids must be [{num_edges}], got {tuple(group_ids.shape)}.")
        if rotations.dim() != 3 or rotations.shape[1:] != (d, d):
            raise ValueError(f"rotations must be [G, {d}, {d}], got {tuple(rotations.shape)}.")
        self.scale = scale
        self.reflectors = reflectors
        self.rotations = rotations
        self.group_ids = group_ids
        self._layout: tuple[torch.Tensor, torch.Tensor, list[int]] | None = None

    @classmethod
    def from_raw(
        cls,
        raw_reflectors: torch.Tensor,
        raw_scale: torch.Tensor,
        rotations: torch.Tensor,
        group_ids: torch.Tensor,
        scale_eps: float,
        reflector_eps: float,
    ) -> "FactorizedMapOperator":
        """`c = softplus(raw_scale) + scale_eps`; `v = normalize_reflectors(raw_reflectors)`."""
        if scale_eps <= 0.0:
            raise ValueError(f"scale_eps must be positive, got {scale_eps}.")
        return cls(
            scale=F.softplus(raw_scale) + scale_eps,
            reflectors=normalize_reflectors(raw_reflectors, reflector_eps),
            rotations=rotations,
            group_ids=group_ids,
        )

    @property
    def num_edges(self) -> int:
        return int(self.reflectors.size(0))

    @property
    def d(self) -> int:
        return int(self.reflectors.size(-1))

    @property
    def householder_k(self) -> int:
        return int(self.reflectors.size(1))

    def stored_elements(self) -> int:
        """Floating-point elements held by this operator: `E*(k*d + 1) + G*d^2`.

        `group_ids` is the data's relation-type vector (shared with `edge_type`), not map
        storage, so it is not counted; neither is the lazily built grouping permutation.
        """
        return int(self.scale.numel() + self.reflectors.numel() + self.rotations.numel())

    def select(self, rows: torch.Tensor) -> "FactorizedMapOperator":
        """The maps of `rows` (any shape, flattened) as a new operator; factors only."""
        flat = rows.reshape(-1)
        return FactorizedMapOperator(
            scale=self.scale.index_select(0, flat),
            reflectors=self.reflectors.index_select(0, flat),
            rotations=self.rotations,
            group_ids=self.group_ids.index_select(0, flat),
        )

    # ------------------------------------------------------------------ application
    def _group_layout(self) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if self._layout is None:
            num_groups = int(self.rotations.size(0))
            perm = torch.argsort(self.group_ids, stable=True)
            inverse = torch.empty_like(perm)
            inverse[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
            sizes = torch.bincount(self.group_ids, minlength=num_groups).tolist()
            if len(sizes) != num_groups:
                raise ValueError(
                    f"group_ids reference relation {len(sizes) - 1}, but only {num_groups} "
                    "rotations exist."
                )
            self._layout = (perm, inverse, sizes)
        return self._layout

    def _rotate(self, y: torch.Tensor, transpose: bool) -> torch.Tensor:
        """`R_{g(e)} y_e` (or `R^T`), as one GEMM per relation; R is never gathered per edge."""
        perm, inverse, sizes = self._group_layout()
        parts = y.index_select(0, perm).split(sizes, dim=0)
        rotated = [
            part @ (rotation if transpose else rotation.transpose(-2, -1))
            for part, rotation in zip(parts, self.rotations)
        ]
        return torch.cat(rotated, dim=0).index_select(0, inverse)

    def _reflect(self, y: torch.Tensor, transpose: bool) -> torch.Tensor:
        """`H y` applies `H(v_k)` first; `H^T y` applies `H(v_1)` first (each is symmetric)."""
        k = self.householder_k
        order = range(k) if transpose else range(k - 1, -1, -1)
        middle = (1,) * (y.dim() - 2)
        for j in order:
            v = self.reflectors[:, j]
            v = v.reshape((v.size(0),) + middle + (v.size(-1),))
            y = y - 2.0 * v * (v * y).sum(dim=-1, keepdim=True)
        return y

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() < 2 or x.size(0) != self.num_edges or x.size(-1) != self.d:
            raise ValueError(
                f"expected [{self.num_edges}, ..., {self.d}] input, got {tuple(x.shape)}."
            )

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """`F_e x_e = c_e R (H x_e)` for `x` of shape `[E, d]` or `[E, n, d]`."""
        self._check_input(x)
        return _broadcast_rows(self.scale, x) * self._rotate(self._reflect(x, False), False)

    def apply_T(self, y: torch.Tensor) -> torch.Tensor:
        """`F_e^T y_e = c_e H^T (R^T y_e)`."""
        self._check_input(y)
        return _broadcast_rows(self.scale, y) * self._reflect(self._rotate(y, True), True)

    def apply_inverse(self, y: torch.Tensor) -> torch.Tensor:
        """`F_e^-1 y_e = c_e^-1 H^T (R^T y_e)` -- closed form, no solve."""
        self._check_input(y)
        return self._reflect(self._rotate(y, True), True) / _broadcast_rows(self.scale, y)

    def apply_inverse_T(self, y: torch.Tensor) -> torch.Tensor:
        """`F_e^-T y_e = c_e^-1 R (H y_e)`."""
        self._check_input(y)
        return self._rotate(self._reflect(y, False), False) / _broadcast_rows(self.scale, y)

    # ------------------------------------------------- dense reference (tests/subsets)
    def reflection_product(self) -> torch.Tensor:
        """Explicit `[E, d, d]` product `H(v_1) ... H(v_k)` from outer products."""
        eye = torch.eye(self.d, device=self.reflectors.device, dtype=self.reflectors.dtype)
        product = eye.expand(self.num_edges, self.d, self.d)
        for j in range(self.householder_k):
            v = self.reflectors[:, j]
            product = product @ (eye - 2.0 * v.unsqueeze(-1) * v.unsqueeze(-2))
        return product

    def _orthogonal_factor(self) -> torch.Tensor:
        """Explicit `[E, d, d]` orthogonal part `R_{g(e)} H_e`."""
        frames = self.rotations.index_select(0, self.group_ids)
        return frames @ self.reflection_product()

    def materialize(self) -> torch.Tensor:
        """Dense `[E, d, d]` maps. For tests and small gathered subsets only."""
        return self.scale.view(-1, 1, 1) * self._orthogonal_factor()

    def materialize_inverse(self) -> torch.Tensor:
        """Dense `[E, d, d]` closed-form inverses `c^-1 H^T R^T`. Tests/subsets only."""
        return self._orthogonal_factor().transpose(-2, -1) / self.scale.view(-1, 1, 1)

    def materialize_with_inverse(self) -> tuple[torch.Tensor, torch.Tensor]:
        """`(materialize(), materialize_inverse())` sharing one `R H` product."""
        orthogonal = self._orthogonal_factor()
        scale = self.scale.view(-1, 1, 1)
        return scale * orthogonal, orthogonal.transpose(-2, -1) / scale


class DenseMapOperator:
    """Explicit `[E, d, d]` maps behind the operator interface.

    Used for the v1 closure edges, whose maps (path holonomy and identity) are produced
    by the face transport per face instance, not per observed edge.
    """

    def __init__(self, maps: torch.Tensor) -> None:
        if maps.dim() != 3 or maps.size(-1) != maps.size(-2):
            raise ValueError(f"maps must be [E, d, d], got {tuple(maps.shape)}.")
        self.maps = maps

    @property
    def num_edges(self) -> int:
        return int(self.maps.size(0))

    def stored_elements(self) -> int:
        return int(self.maps.numel())

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(self.maps, x.unsqueeze(-1)).squeeze(-1)

    def apply_T(self, y: torch.Tensor) -> torch.Tensor:
        return torch.bmm(self.maps.transpose(-2, -1), y.unsqueeze(-1)).squeeze(-1)


class StackedMapOperator:
    """Row-wise concatenation of operators (e.g. observed edges, then closure edges)."""

    def __init__(self, parts: tuple[MapOperator, ...]) -> None:
        if not parts:
            raise ValueError("StackedMapOperator needs at least one part.")
        self.parts = parts
        self.sizes = [part.num_edges for part in parts]

    @property
    def num_edges(self) -> int:
        return int(sum(self.sizes))

    def stored_elements(self) -> int:
        return int(sum(part.stored_elements() for part in self.parts))

    def _map_parts(self, x: torch.Tensor, transpose: bool) -> torch.Tensor:
        if x.size(0) != self.num_edges:
            raise ValueError(f"expected {self.num_edges} rows, got {x.size(0)}.")
        chunks = x.split(self.sizes, dim=0)
        return torch.cat(
            [
                part.apply_T(chunk) if transpose else part.apply(chunk)
                for part, chunk in zip(self.parts, chunks)
            ],
            dim=0,
        )

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self._map_parts(x, transpose=False)

    def apply_T(self, y: torch.Tensor) -> torch.Tensor:
        return self._map_parts(y, transpose=True)


class PathTransportOperator:
    """Matrix-free parallel transport into each path's 2-cell stalk, gauge `A_f = I`.

    For a path `v_0 -e_1-> v_1 ... -e_L-> v_L` whose edges are stored along the path
    (`validate_face_orientation`), functoriality forces the node->2-cell maps

        T_{v_0} = I,   T_{v_i} = Q_1 Q_2 ... Q_i,   Q_j = (F^{e_j}_src)^-1 F^{e_j}_tgt.

    `apply` returns `T_{v_i} x_{v_i}` for every position of every path; `apply_T` is its
    exact adjoint (node cochain <- per-incidence values). Each `Q_j` is applied to all
    positions `i >= j` at once, so the cost is `L(L+1)/2` factorized map applications per
    path and nothing `d x d` is formed.

    Args:
        src: source-side factorized maps of the observed edges.
        tgt: target-side factorized maps of the same edges.
        face_node_index: `[P, L+1]` node ids along each path.
        face_edge_index: `[P, L]` edge ids along each path (indices into `src`/`tgt`).
    """

    def __init__(
        self,
        src: FactorizedMapOperator,
        tgt: FactorizedMapOperator,
        face_node_index: torch.Tensor,
        face_edge_index: torch.Tensor,
    ) -> None:
        if face_node_index.dim() != 2 or face_edge_index.dim() != 2:
            raise ValueError("face_node_index and face_edge_index must be two-dimensional.")
        if face_node_index.shape != (face_edge_index.size(0), face_edge_index.size(1) + 1):
            raise ValueError(
                "face_node_index must be [P, L+1] for face_edge_index [P, L], got "
                f"{tuple(face_node_index.shape)} and {tuple(face_edge_index.shape)}."
            )
        if src.num_edges != tgt.num_edges:
            raise ValueError("src and tgt operators must cover the same edges.")
        self.face_node_index = face_node_index
        self.length = int(face_edge_index.size(1))
        self._src = tuple(src.select(face_edge_index[:, j]) for j in range(self.length))
        self._tgt = tuple(tgt.select(face_edge_index[:, j]) for j in range(self.length))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """`[N, d]` node cochain -> `[P, L+1, d]` values `T_{v_i} x_{v_i}` in `F(f)`."""
        values = x[self.face_node_index]
        for j in range(self.length, 0, -1):
            tail = self._tgt[j - 1].apply(values[:, j:])
            tail = self._src[j - 1].apply_inverse(tail)
            values = torch.cat([values[:, :j], tail], dim=1)
        return values

    def apply_T(self, w: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Adjoint: `[P, L+1, d]` -> `[N, d]`, summing `T_{v_i}^T w_{f,i}` into node `v_i`."""
        if w.shape[:2] != self.face_node_index.shape:
            raise ValueError(
                f"w must be [P, L+1, d] matching {tuple(self.face_node_index.shape)}, "
                f"got {tuple(w.shape)}."
            )
        values = w
        for j in range(1, self.length + 1):
            tail = self._src[j - 1].apply_inverse_T(values[:, j:])
            tail = self._tgt[j - 1].apply_T(tail)
            values = torch.cat([values[:, :j], tail], dim=1)
        output = w.new_zeros((num_nodes, w.size(-1)))
        output.index_add_(0, self.face_node_index.reshape(-1), values.reshape(-1, w.size(-1)))
        return output
