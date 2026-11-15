"""Matrix-free node->edge restriction-map operators (decision D1).

A factorized restriction map is

    F_{v<|e} = c * R_{psi(e),side} * H_k(u_{v<|e}),
    H_k(u)   = H(u_1) H(u_2) ... H(u_k),   H(u) = I - 2 u u^T / ||u||^2,

with `k` reflection vectors per incidence, one orthogonal `R` per (relation, side) and a
positive scale `c` (one per edge, shared by both sides, under D1a; one per incidence under
the per-side ablation). The product is conformal-orthogonal, so the inverse is closed form,
`F^-1 = c^-1 H_k^T R^T`, and the path transport `PathTransportOperator` needs no solve.

The operators below store only the factors and apply the maps to vectors. Nothing of shape
`[E, d, d]` is built by `apply`, `apply_T`, `apply_inverse` or `apply_inverse_T`;
`materialize` exists for tests only.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Protocol

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from src.data.face_geometry import validate_face_orientation


def checkpoint_if_grad(
    function: Callable[..., torch.Tensor | tuple[torch.Tensor, ...]],
    *inputs: torch.Tensor,
    enabled: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Run `function` under activation checkpointing while autograd is recording.

    A matrix-free map application creates several `[E, d]` intermediates (reflection
    inputs, the relation-sorted copy, the pre-scale output). Autograd would keep all of
    them for every call -- on ACM (d=32, K=4) that alone exceeds 7 GiB. Checkpointing keeps
    only `inputs` and replays the deterministic forward during backward, so gradients are
    those of `function` itself. Same trade as the FastHPP merge in `orthogonal.py`.
    Tensors that `function` closes over (the map factors) still receive gradients.
    `enabled=False` (v2 `model.checkpointing: false`) runs `function` plainly.
    """
    if not enabled or not torch.is_grad_enabled():
        return function(*inputs)
    return torch.utils.checkpoint.checkpoint(function, *inputs, use_reentrant=False)


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
    """One side's restriction maps over `num_edges` edges, applied to `[E, d]` inputs.

    `FactorizedMapOperator` also accepts `[E, n, d]`; `DenseMapOperator` does not.
    """

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
            # v * (2 <v, y>) rather than (2 v) * <v, y>: same value (x2 is exact), but no
            # fresh [E, d] tensor `2 v` for autograd to keep.
            y = y - v * (2.0 * (v * y).sum(dim=-1, keepdim=True))
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

    def apply_orthogonal(self, x: torch.Tensor) -> torch.Tensor:
        """`R H x_e`: the map without its scale (D1a: shared scales cancel in transports)."""
        self._check_input(x)
        return self._rotate(self._reflect(x, False), False)

    def apply_orthogonal_T(self, y: torch.Tensor) -> torch.Tensor:
        """`H^T R^T y_e`, the transpose and the inverse of `apply_orthogonal`."""
        self._check_input(y)
        return self._reflect(self._rotate(y, True), True)

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

    Used by the dense map family. With `scale` (maps conformal-orthogonal, `F = c Q`) the inverse is the
    closed form `F^T / c^2`; no general solve is ever taken.
    """

    def __init__(self, maps: torch.Tensor, scale: torch.Tensor | None = None) -> None:
        if maps.dim() != 3 or maps.size(-1) != maps.size(-2):
            raise ValueError(f"maps must be [E, d, d], got {tuple(maps.shape)}.")
        if scale is not None and scale.shape != (maps.size(0),):
            raise ValueError(f"scale must be [{maps.size(0)}], got {tuple(scale.shape)}.")
        self.maps = maps
        self.scale = scale

    @property
    def num_edges(self) -> int:
        return int(self.maps.size(0))

    @property
    def d(self) -> int:
        return int(self.maps.size(-1))

    def stored_elements(self) -> int:
        return int(self.maps.numel() + (0 if self.scale is None else self.scale.numel()))

    @staticmethod
    def _matvec(maps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return torch.bmm(maps, x.unsqueeze(-1)).squeeze(-1)
        middle = (1,) * (x.dim() - 2)
        return (maps.reshape((maps.size(0),) + middle + maps.shape[1:]) @ x.unsqueeze(-1)).squeeze(-1)

    def _closed_form_scale(self, like: torch.Tensor) -> torch.Tensor:
        if self.scale is None:
            raise ValueError(
                "DenseMapOperator needs its conformal scale for the closed-form inverse."
            )
        return _broadcast_rows(self.scale, like)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self._matvec(self.maps, x)

    def apply_T(self, y: torch.Tensor) -> torch.Tensor:
        return self._matvec(self.maps.transpose(-2, -1), y)

    def apply_inverse(self, y: torch.Tensor) -> torch.Tensor:
        """`F^-1 y = F^T y / c^2`."""
        return self._matvec(self.maps.transpose(-2, -1), y) / self._closed_form_scale(y).square()

    def apply_inverse_T(self, y: torch.Tensor) -> torch.Tensor:
        """`F^-T y = F y / c^2`."""
        return self._matvec(self.maps, y) / self._closed_form_scale(y).square()

    def apply_orthogonal(self, x: torch.Tensor) -> torch.Tensor:
        """`Q x = F x / c`."""
        return self._matvec(self.maps, x) / self._closed_form_scale(x)

    def apply_orthogonal_T(self, y: torch.Tensor) -> torch.Tensor:
        """`Q^T y = F^T y / c`."""
        return self._matvec(self.maps.transpose(-2, -1), y) / self._closed_form_scale(y)

    def select(self, rows: torch.Tensor) -> "DenseMapOperator":
        flat = rows.reshape(-1)
        return DenseMapOperator(
            self.maps.index_select(0, flat),
            None if self.scale is None else self.scale.index_select(0, flat),
        )


def _mix_rows(
    first: FactorizedMapOperator | DenseMapOperator,
    second: FactorizedMapOperator | DenseMapOperator,
    take_second: torch.Tensor,
) -> FactorizedMapOperator | DenseMapOperator:
    """Row-wise choice between two operators over the same rows: row `p` is `second`'s map
    where `take_second[p]`, else `first`'s. Used for hops traversed against their stored
    direction, whose near side is the target role."""
    if isinstance(first, DenseMapOperator):
        pick = take_second.view(-1, 1, 1)
        scale = None if first.scale is None else torch.where(take_second, second.scale,
                                                             first.scale)
        return DenseMapOperator(torch.where(pick, second.maps, first.maps), scale)
    if first.rotations is second.rotations:
        rotations = first.rotations
        offset = 0
    else:
        rotations = torch.cat([first.rotations, second.rotations], dim=0)
        offset = int(first.rotations.size(0))
    return FactorizedMapOperator(
        scale=torch.where(take_second, second.scale, first.scale),
        reflectors=torch.where(take_second.view(-1, 1, 1), second.reflectors, first.reflectors),
        rotations=rotations,
        group_ids=torch.where(take_second, second.group_ids + offset, first.group_ids),
    )


class PathTransportOperator:
    """Matrix-free parallel transport into each path's 2-cell stalk, gauge `A_f = I`.

    For a simple path `v_0, e_1, v_1, ..., e_L, v_L` with hop signs `sigma_j`
    (`face_edge_sign`; +1: `e_j` stored `v_{j-1} -> v_j`, -1: stored `v_j -> v_{j-1}`;
    checked by `validate_face_orientation`), functoriality
    `T_{v_{j-1}} F_{v_{j-1}<|e}^-1 = T_{v_j} F_{v_j<|e}^-1` forces the node->2-cell maps

        T_{v_0} = I,   T_{v_i} = Q_1 Q_2 ... Q_i,   Q_j = F_{v_{j-1}<|e_j}^-1 F_{v_j<|e_j},

    i.e. `Q_j = (F_src)^-1 F_tgt` on +1 hops and `Q_j = (F_tgt)^-1 F_src` on -1 hops
    (`= Q_fwd^T` under D1a). Every method (`apply`, `apply_T`, holonomy, Horner sum, prefix
    recursion, `squared_scales`) goes through the same signed step.

    `apply` returns `T_{v_i} x_{v_i}` for every position of every path; `apply_T` is its
    exact adjoint (node cochain <- per-incidence values). Each `Q_j` is applied to all
    positions `i >= j` at once: `L(L+1)/2` applications of a `Q_j`, i.e. `L(L+1)`
    factorized map applications per path, and nothing `d x d` is formed. The 2-cell term
    (`PathCellTerm`) needs only `sum_i T_{v_i} x_{v_i}` (`transport_sum`, Horner, L steps)
    and `T_{v_i}^T z` (`adjoint_prefix`, L steps): 2L steps per path instead of `L(L+1)` (D9).

    With one scale per edge shared by both sides (D1a, `shared_scale=True`) the scales cancel,
    `Q_j = H_src^T R_src^T R_tgt H_tgt` is exactly orthogonal, and the operator applies only
    the orthogonal factors: no scale multiplication, and every inverse is a transpose.

    Args:
        src: source-side factorized maps of the observed edges.
        tgt: target-side factorized maps of the same edges.
        face_node_index: `[P, L+1]` node ids along each path.
        face_edge_index: `[P, L]` edge ids along each path (indices into `src`/`tgt`).
        edge_index: `[2, E]` observed edge list; used to check every hop against its sign
            and that paths are simple, which the formula assumes.
        shared_scale: the two sides carry the same scale on every edge (checked).
        face_edge_sign: `[P, L]` hop signs in {+1, -1}; None means all +1.

    `_src[j]` / `_tgt[j]` hold the maps at `v_{j-1}` / `v_j` of hop `j`: the stored source /
    target roles, swapped row-wise on -1 hops. Columns without a -1 hop use the plain
    selections, so all-+1 groups take exactly the unsigned code path.
    """

    def __init__(
        self,
        src: FactorizedMapOperator | DenseMapOperator,
        tgt: FactorizedMapOperator | DenseMapOperator,
        face_node_index: torch.Tensor,
        face_edge_index: torch.Tensor,
        edge_index: torch.Tensor,
        shared_scale: bool = False,
        face_edge_sign: torch.Tensor | None = None,
    ) -> None:
        if face_node_index.dim() != 2 or face_edge_index.dim() != 2:
            raise ValueError("face_node_index and face_edge_index must be two-dimensional.")
        if face_node_index.shape != (face_edge_index.size(0), face_edge_index.size(1) + 1):
            raise ValueError(
                "face_node_index must be [P, L+1] for face_edge_index [P, L], got "
                f"{tuple(face_node_index.shape)} and {tuple(face_edge_index.shape)}."
            )
        if src.num_edges != tgt.num_edges or edge_index.size(1) != src.num_edges:
            raise ValueError("src, tgt and edge_index must cover the same edges.")
        validate_face_orientation(
            face_node_index, face_edge_index, edge_index, template_name="PathTransportOperator",
            face_edge_sign=face_edge_sign,
        )
        if shared_scale and not torch.equal(src.scale, tgt.scale):
            raise ValueError(
                "shared_scale=True needs the same scale on both sides of every edge (D1a)."
            )
        self.shared_scale = shared_scale
        self.face_node_index = face_node_index
        self.length = int(face_edge_index.size(1))
        # All source selections first, then all target selections: the autograd graph (and
        # so the gradient summation order) of all-+1 groups is exactly the unsigned one.
        src_sel = [src.select(face_edge_index[:, j]) for j in range(self.length)]
        tgt_sel = [tgt.select(face_edge_index[:, j]) for j in range(self.length)]
        near, far = [], []
        for j in range(self.length):
            reversed_hop = None if face_edge_sign is None else face_edge_sign[:, j] < 0
            if reversed_hop is None or not bool(reversed_hop.any()):
                near.append(src_sel[j])
                far.append(tgt_sel[j])
            else:
                near.append(_mix_rows(src_sel[j], tgt_sel[j], reversed_hop))
                far.append(_mix_rows(tgt_sel[j], src_sel[j], reversed_hop))
        self._src = tuple(near)
        self._tgt = tuple(far)

    def _step(self, j: int, y: torch.Tensor) -> torch.Tensor:
        """`Q_j y = F_{v_{j-1}<|e_j}^-1 F_{v_j<|e_j} y` (orthogonal factors only when shared)."""
        if self.shared_scale:
            return self._src[j].apply_orthogonal_T(self._tgt[j].apply_orthogonal(y))
        return self._src[j].apply_inverse(self._tgt[j].apply(y))

    def _step_T(self, j: int, w: torch.Tensor) -> torch.Tensor:
        """`Q_j^T w = F_{v_j<|e_j}^T F_{v_{j-1}<|e_j}^-T w`."""
        if self.shared_scale:
            return self._tgt[j].apply_orthogonal_T(self._src[j].apply_orthogonal(w))
        return self._tgt[j].apply_T(self._src[j].apply_inverse_T(w))

    def squared_scales(self) -> torch.Tensor:
        """`[P, L+1]` squared conformal scale `s_i^2` of each `T_{v_i}` (`T^T T = s^2 I`).

        `s_i = prod_{j<=i} c_{v_j<|e_j} / c_{v_{j-1}<|e_j}`; exactly 1 under shared scales (D1a).
        """
        reference = self._src[0].scale if self.length else None
        if reference is None:
            return torch.ones(self.face_node_index.shape, dtype=torch.float32,
                              device=self.face_node_index.device)
        squares = [torch.ones_like(reference)]
        for j in range(self.length):
            if self.shared_scale:
                squares.append(squares[-1])
            else:
                squares.append(squares[-1] * (self._tgt[j].scale / self._src[j].scale).square())
        return torch.stack(squares, dim=1)

    def apply_holonomy(self, y: torch.Tensor) -> torch.Tensor:
        """`[P, d]` -> `T_{v_L} y = Q_1 Q_2 ... Q_L y` (path holonomy, gauge `A_f = I`)."""
        for j in range(self.length, 0, -1):
            y = self._step(j - 1, y)
        return y

    def apply_holonomy_T(self, w: torch.Tensor) -> torch.Tensor:
        """`[P, d]` -> `T_{v_L}^T w = Q_L^T ... Q_1^T w`."""
        for j in range(1, self.length + 1):
            w = self._step_T(j - 1, w)
        return w

    def transport_sum(self, x: torch.Tensor) -> torch.Tensor:
        """`[N, d]` -> `[P, d]` `sum_i T_{v_i} x_{v_i}` by Horner's rule (L steps, D9):

            S = x_{v_0} + Q_1 (x_{v_1} + Q_2 (x_{v_2} + ... + Q_L x_{v_L})).
        """
        values = x[self.face_node_index]
        total = values[:, self.length]
        for j in range(self.length, 0, -1):
            total = values[:, j - 1] + self._step(j - 1, total)
        return total

    def adjoint_prefix(self, z: torch.Tensor) -> torch.Tensor:
        """`[P, d]` -> `[P, L+1, d]` `t_i = T_{v_i}^T z` by `t_0 = z, t_i = Q_i^T t_{i-1}`
        (L steps, D9)."""
        prefix = [z]
        for j in range(1, self.length + 1):
            prefix.append(self._step_T(j - 1, prefix[-1]))
        return torch.stack(prefix, dim=1)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """`[N, d]` node cochain -> `[P, L+1, d]` values `T_{v_i} x_{v_i}` in `F(f)`."""
        values = x[self.face_node_index]
        for j in range(self.length, 0, -1):
            tail = self._step(j - 1, values[:, j:])
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
            tail = self._step_T(j - 1, values[:, j:])
            values = torch.cat([values[:, :j], tail], dim=1)
        output = w.new_zeros((num_nodes, w.size(-1)))
        output.index_add_(0, self.face_node_index.reshape(-1), values.reshape(-1, w.size(-1)))
        return output
