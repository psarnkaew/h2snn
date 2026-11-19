"""Matrix-free node->edge restriction-map operators (spec 1.25 Z2, TD-1).

A TD-1 restriction map (spec v1.25 §3) is, per fact `e` of relation `r`,

    F_{src<|e} = c_e I,          F_{tgt<|e} = c_e R_r exp(U_r C_e U_r^T)        (rho(r) != r)
    F_{src<|e} = c_e exp(-U_r C_e U_r^T / 2),  F_{tgt<|e} = c_e exp(+U_r C_e U_r^T / 2)   (rho(r) = r)

with `R_r` in SO(d) (one FastHPP frame per relation with rho(r) != r), `U_r` in R^{d x m} orthonormal
(m = 4), `C_e` in so(m) and one positive scale `c_e` per fact (D1a). The exponential is applied in
closed form, `exp(U C U^T) = I + U (exp(C) - I) U^T` (Lemma 1 of `docs/thm4_td1_proof.md`), so each
fact stores the m x m matrix `exp(C_e) - I` and its scale, and each relation its `U_r` and frame.
Every map is conformal-orthogonal, `F^-1 = F^T / c^2`, so `PathTransportOperator` needs no solve.

The operators below store only these factors and apply the maps to vectors; nothing of shape
`[E, d, d]` is built by `apply`, `apply_T`, `apply_inverse` or `apply_inverse_T`; `materialize`
exists for tests only.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Protocol

import torch
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


class MapOperator(Protocol):
    """One side's restriction maps over `num_edges` edges, applied to `[E, d]` inputs.

    `TD1MapOperator` also accepts `[E, n, d]`; `DenseMapOperator` does not.
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


def skew_from_coefficients(coefficients: torch.Tensor, rank: int) -> torch.Tensor:
    """`[..., m(m-1)/2]` coefficients -> `[..., m, m]` skew-symmetric `C` (upper triangle, row-major
    order of `torch.triu_indices(m, m, 1)`; `C[i, j] = k`, `C[j, i] = -k`)."""
    count = rank * (rank - 1) // 2
    if coefficients.size(-1) != count:
        raise ValueError(f"so({rank}) needs {count} coefficients, got {coefficients.size(-1)}.")
    rows, cols = torch.triu_indices(rank, rank, 1, device=coefficients.device)
    skew = coefficients.new_zeros(coefficients.shape[:-1] + (rank, rank))
    skew[..., rows, cols] = coefficients
    return skew - skew.transpose(-2, -1)


def orthonormal_subspaces(weights: torch.Tensor) -> torch.Tensor:
    """`[G, d, m]` learned matrices -> `[G, d, m]` orthonormal `U` = thin-QR Q with `sign(diag R) > 0`
    (spec 1.25 Z2: `U_r = thin-QR(W_r).Q`). The sign fix makes the map from `W` to `U` a function
    (QR is unique once `diag R > 0`)."""
    q, r = torch.linalg.qr(weights, mode="reduced")
    diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
    sign = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return q * sign.unsqueeze(-2)


class TD1MapOperator:
    """`F_e = c_e [R_{f(g(e))}] exp(U_{g(e)} C_e U_{g(e)}^T)` for `E` edges, one side (spec 1.25 Z2).

    Args:
        scale: `[E]` positive conformal scale `c_e`.
        rotation: `[E, m, m]` the side's `exp(.) - I` per edge (`exp(C_e) - I` on the target side of a
            relation with rho(r) != r; `exp(+-C_e/2) - I` on the two sides of rho(r) = r), or None when
            no group has an instance part.
        group_ids: `[E]` group (relation, or cell position) of each edge, in `[0, G)`.
        subspaces: `[G, d, m]` orthonormal `U_g`.
        frames: `[F, d, d]` orthogonal frames (F may be 0), or None.
        instance_groups: per group, whether its rows apply `exp(U C U^T)` (`rotation` rows are read
            only there).
        frame_index: per group, the row of `frames` applied after the instance part, or -1 (none).

    `apply(x) = c (R (x + U ((E - I)(U^T x))))`, the frame GEMM and the U projections one GEMM per
    group (rows permuted by group once per call; nothing is gathered per edge but the m x m blocks).
    """

    def __init__(
        self,
        scale: torch.Tensor,
        rotation: torch.Tensor | None,
        group_ids: torch.Tensor,
        subspaces: torch.Tensor,
        frames: torch.Tensor | None,
        instance_groups: tuple[bool, ...],
        frame_index: tuple[int, ...],
    ) -> None:
        if scale.dim() != 1:
            raise ValueError(f"scale must be [E], got {tuple(scale.shape)}.")
        num_edges = scale.size(0)
        if group_ids.shape != (num_edges,):
            raise ValueError(f"group_ids must be [{num_edges}], got {tuple(group_ids.shape)}.")
        if subspaces.dim() != 3:
            raise ValueError(f"subspaces must be [G, d, m], got {tuple(subspaces.shape)}.")
        num_groups, d, rank = subspaces.shape
        if len(instance_groups) != num_groups or len(frame_index) != num_groups:
            raise ValueError(f"instance_groups and frame_index need one entry per group ({num_groups}).")
        if any(instance_groups) and (rotation is None or rotation.shape != (num_edges, rank, rank)):
            raise ValueError(f"rotation must be [{num_edges}, {rank}, {rank}] when a group has an "
                             f"instance part, got {None if rotation is None else tuple(rotation.shape)}.")
        num_frames = 0 if frames is None else int(frames.size(0))
        if frames is not None and (frames.dim() != 3 or frames.shape[1:] != (d, d)):
            raise ValueError(f"frames must be [F, {d}, {d}], got {tuple(frames.shape)}.")
        if any(f >= num_frames or f < -1 for f in frame_index):
            raise ValueError(f"frame_index {frame_index} out of range for {num_frames} frames.")
        self.scale = scale
        self.rotation = rotation
        self.group_ids = group_ids
        self.subspaces = subspaces
        self.frames = frames
        self.instance_groups = tuple(bool(v) for v in instance_groups)
        self.frame_index = tuple(int(v) for v in frame_index)
        self._layout: tuple[torch.Tensor, torch.Tensor, list[int], tuple] | None = None

    @property
    def num_edges(self) -> int:
        return int(self.scale.size(0))

    @property
    def d(self) -> int:
        return int(self.subspaces.size(1))

    @property
    def rank(self) -> int:
        return int(self.subspaces.size(2))

    @property
    def trivial(self) -> bool:
        """No group applies an instance part or a frame: the orthogonal factor is `I` on every row."""
        return not any(self.instance_groups) and all(f < 0 for f in self.frame_index)

    def stored_elements(self) -> int:
        """Floating-point elements the maps of this side depend on: `E` scales, `E m^2` rotation
        entries (if any group has an instance part), `d m` per group with an instance part and `d^2`
        per distinct frame used. Target side of relations with rho(r) != r: `E (m^2 + 1) + R (d^2 + d m)`
        (card T1). `group_ids` (the data's relation vector) and the layout are not map storage."""
        used_frames = {f for f in self.frame_index if f >= 0}
        instance = sum(self.instance_groups)
        return int(self.num_edges
                   + (self.num_edges * self.rank ** 2 if instance else 0)
                   + instance * self.d * self.rank
                   + len(used_frames) * self.d ** 2)

    def select(self, rows: torch.Tensor) -> "TD1MapOperator":
        """The maps of `rows` (any shape, flattened) as a new operator; factors only."""
        flat = rows.reshape(-1)
        return TD1MapOperator(
            scale=self.scale.index_select(0, flat),
            rotation=None if self.rotation is None else self.rotation.index_select(0, flat),
            group_ids=self.group_ids.index_select(0, flat),
            subspaces=self.subspaces,
            frames=self.frames,
            instance_groups=self.instance_groups,
            frame_index=self.frame_index,
        )

    # ------------------------------------------------------------------ application
    def _group_layout(self) -> tuple[torch.Tensor, torch.Tensor, list[int], tuple]:
        if self._layout is None:
            num_groups = int(self.subspaces.size(0))
            perm = torch.argsort(self.group_ids, stable=True)
            inverse = torch.empty_like(perm)
            inverse[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
            sizes = torch.bincount(self.group_ids, minlength=num_groups).tolist()
            if len(sizes) != num_groups:
                raise ValueError(f"group_ids reference group {len(sizes) - 1}, but only {num_groups} "
                                 "subspaces exist.")
            blocks: tuple = ()
            if self.rotation is not None and any(self.instance_groups):
                blocks = tuple(self.rotation.index_select(0, perm).split(sizes, dim=0))
            self._layout = (perm, inverse, sizes, blocks)
        return self._layout

    @staticmethod
    def _rotate_block(part: torch.Tensor, subspace: torch.Tensor, block: torch.Tensor,
                      transpose: bool) -> torch.Tensor:
        """`part + U ((E - I) (U^T part))` (or with `(E - I)^T`) for the rows of one group."""
        projected = part @ subspace                                  # [n, ..., m]
        middle = (1,) * (part.dim() - 2)
        block = block.reshape((block.size(0),) + middle + block.shape[1:])
        if transpose:
            block = block.transpose(-2, -1)
        rotated = (block @ projected.unsqueeze(-1)).squeeze(-1)      # [n, ..., m]
        return part + rotated @ subspace.transpose(-2, -1)

    def _orthogonal(self, y: torch.Tensor, transpose: bool) -> torch.Tensor:
        """`R exp(A) y` per row (or `exp(A)^T R^T y`); exactly `y` when the side is trivial."""
        if self.trivial:
            return y
        perm, inverse, sizes, blocks = self._group_layout()
        parts = y.index_select(0, perm).split(sizes, dim=0)
        out = []
        for g, part in enumerate(parts):
            if part.size(0) == 0:
                out.append(part)
                continue
            frame = self.frames[self.frame_index[g]] if self.frame_index[g] >= 0 else None
            if not transpose:
                if self.instance_groups[g]:
                    part = self._rotate_block(part, self.subspaces[g], blocks[g], False)
                if frame is not None:
                    part = part @ frame.transpose(-2, -1)
            else:
                if frame is not None:
                    part = part @ frame
                if self.instance_groups[g]:
                    part = self._rotate_block(part, self.subspaces[g], blocks[g], True)
            out.append(part)
        return torch.cat(out, dim=0).index_select(0, inverse)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() < 2 or x.size(0) != self.num_edges or x.size(-1) != self.d:
            raise ValueError(
                f"expected [{self.num_edges}, ..., {self.d}] input, got {tuple(x.shape)}."
            )

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """`F_e x_e = c_e R exp(A_e) x_e` for `x` of shape `[E, d]` or `[E, n, d]`."""
        self._check_input(x)
        return _broadcast_rows(self.scale, x) * self._orthogonal(x, False)

    def apply_T(self, y: torch.Tensor) -> torch.Tensor:
        """`F_e^T y_e = c_e exp(A_e)^T R^T y_e`."""
        self._check_input(y)
        return _broadcast_rows(self.scale, y) * self._orthogonal(y, True)

    def apply_inverse(self, y: torch.Tensor) -> torch.Tensor:
        """`F_e^-1 y_e = c_e^-1 exp(A_e)^T R^T y_e` -- closed form, no solve."""
        self._check_input(y)
        return self._orthogonal(y, True) / _broadcast_rows(self.scale, y)

    def apply_inverse_T(self, y: torch.Tensor) -> torch.Tensor:
        """`F_e^-T y_e = c_e^-1 R exp(A_e) y_e`."""
        self._check_input(y)
        return self._orthogonal(y, False) / _broadcast_rows(self.scale, y)

    def apply_orthogonal(self, x: torch.Tensor) -> torch.Tensor:
        """`R exp(A) x_e`: the map without its scale (D1a: shared scales cancel in transports)."""
        self._check_input(x)
        return self._orthogonal(x, False)

    def apply_orthogonal_T(self, y: torch.Tensor) -> torch.Tensor:
        """`exp(A)^T R^T y_e`, the transpose and the inverse of `apply_orthogonal`."""
        self._check_input(y)
        return self._orthogonal(y, True)

    # ------------------------------------------------- dense reference (tests/subsets)
    def _orthogonal_factor(self) -> torch.Tensor:
        """Explicit `[E, d, d]` orthogonal part `R_{f(g(e))} exp(A_e)` (closed form)."""
        eye = torch.eye(self.d, device=self.scale.device, dtype=self.subspaces.dtype)
        factor = eye.expand(self.num_edges, self.d, self.d).clone()
        for g in range(int(self.subspaces.size(0))):
            rows = (self.group_ids == g).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            block = eye.expand(rows.numel(), self.d, self.d)
            if self.instance_groups[g]:
                u = self.subspaces[g]
                block = block + u @ self.rotation[rows] @ u.transpose(-2, -1)
            if self.frame_index[g] >= 0:
                block = self.frames[self.frame_index[g]] @ block
            factor[rows] = block
        return factor

    def materialize(self) -> torch.Tensor:
        """Dense `[E, d, d]` maps. For tests and small gathered subsets only."""
        return self.scale.view(-1, 1, 1) * self._orthogonal_factor()

    def materialize_inverse(self) -> torch.Tensor:
        """Dense `[E, d, d]` closed-form inverses `c^-1 exp(A)^T R^T`. Tests/subsets only."""
        return self._orthogonal_factor().transpose(-2, -1) / self.scale.view(-1, 1, 1)

    def materialize_with_inverse(self) -> tuple[torch.Tensor, torch.Tensor]:
        """`(materialize(), materialize_inverse())` sharing one orthogonal product."""
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


class IdentityMapOperator:
    """Arm "maps = I" (spec 1.23 X-11, card 4 K1: S2 = `full`, S0 = `no_path_cells`): every fact map
    is `F_e = c_e I`, so its orthogonal factor is the identity and every transport
    `T_{v,f} = Q_1 ... Q_i = I`. Only the per-edge scale `[E]` is stored (the fact weight
    `a_e = c_e^2` is unchanged); nothing `[E, d]` or `[E, d, d]` is built, and the orthogonal
    applications return their input unchanged, so the transports are exactly the identity.
    """

    def __init__(self, scale: torch.Tensor, d: int) -> None:
        if scale.dim() != 1:
            raise ValueError(f"scale must be [E], got {tuple(scale.shape)}.")
        if d < 1:
            raise ValueError(f"d must be positive, got {d}.")
        self.scale = scale
        self._d = int(d)

    @property
    def num_edges(self) -> int:
        return int(self.scale.size(0))

    @property
    def d(self) -> int:
        return self._d

    def stored_elements(self) -> int:
        """Only the scale: `E` elements."""
        return int(self.scale.numel())

    def select(self, rows: torch.Tensor) -> "IdentityMapOperator":
        return IdentityMapOperator(self.scale.index_select(0, rows.reshape(-1)), self._d)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() < 2 or x.size(0) != self.num_edges or x.size(-1) != self._d:
            raise ValueError(
                f"expected [{self.num_edges}, ..., {self._d}] input, got {tuple(x.shape)}."
            )

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """`F_e x_e = c_e x_e`."""
        self._check_input(x)
        return _broadcast_rows(self.scale, x) * x

    def apply_T(self, y: torch.Tensor) -> torch.Tensor:
        self._check_input(y)
        return _broadcast_rows(self.scale, y) * y

    def apply_inverse(self, y: torch.Tensor) -> torch.Tensor:
        self._check_input(y)
        return y / _broadcast_rows(self.scale, y)

    def apply_inverse_T(self, y: torch.Tensor) -> torch.Tensor:
        self._check_input(y)
        return y / _broadcast_rows(self.scale, y)

    def apply_orthogonal(self, x: torch.Tensor) -> torch.Tensor:
        """The orthogonal factor is `I`: `x` itself (exact)."""
        self._check_input(x)
        return x

    def apply_orthogonal_T(self, y: torch.Tensor) -> torch.Tensor:
        self._check_input(y)
        return y

    def materialize(self) -> torch.Tensor:
        """Dense `[E, d, d]` maps `c_e I`. Tests only."""
        eye = torch.eye(self._d, dtype=self.scale.dtype, device=self.scale.device)
        return self.scale.view(-1, 1, 1) * eye


def _mix_rows(
    first: TD1MapOperator | DenseMapOperator | IdentityMapOperator,
    second: TD1MapOperator | DenseMapOperator | IdentityMapOperator,
    take_second: torch.Tensor,
) -> TD1MapOperator | DenseMapOperator | IdentityMapOperator:
    """Row-wise choice between two operators over the same rows: row `p` is `second`'s map
    where `take_second[p]`, else `first`'s. Used for hops traversed against their stored
    direction, whose near side is the target role."""
    if isinstance(first, IdentityMapOperator):
        return IdentityMapOperator(torch.where(take_second, second.scale, first.scale), first.d)
    if isinstance(first, DenseMapOperator):
        pick = take_second.view(-1, 1, 1)
        scale = None if first.scale is None else torch.where(take_second, second.scale,
                                                             first.scale)
        return DenseMapOperator(torch.where(pick, second.maps, first.maps), scale)
    rank = first.rank
    if first.rotation is None and second.rotation is None:
        rotation = None
    else:
        zeros = None
        if first.rotation is None or second.rotation is None:
            like = first.rotation if first.rotation is not None else second.rotation
            zeros = like.new_zeros((first.num_edges, rank, rank))
        rotation = torch.where(take_second.view(-1, 1, 1),
                               second.rotation if second.rotation is not None else zeros,
                               first.rotation if first.rotation is not None else zeros)
    # The second operator's groups follow the first's (offset G); frames are concatenated likewise.
    num_groups = int(first.subspaces.size(0))
    first_frames = 0 if first.frames is None else int(first.frames.size(0))
    if first.frames is None and second.frames is None:
        frames = None
    elif second.frames is None or second.frames is first.frames:
        frames = first.frames
    elif first.frames is None:
        frames = second.frames
    else:
        frames = torch.cat([first.frames, second.frames], dim=0)
    shift = 0 if (second.frames is first.frames or first.frames is None) else first_frames
    second_index = tuple(-1 if f < 0 else f + shift for f in second.frame_index)
    subspaces = (first.subspaces if second.subspaces is first.subspaces
                 else torch.cat([first.subspaces, second.subspaces], dim=0))
    if subspaces is first.subspaces:  # shared U: the second's groups reuse the first's rows
        subspaces = torch.cat([first.subspaces, first.subspaces], dim=0)
    return TD1MapOperator(
        scale=torch.where(take_second, second.scale, first.scale),
        rotation=rotation,
        group_ids=torch.where(take_second, second.group_ids + num_groups, first.group_ids),
        subspaces=subspaces,
        frames=frames,
        instance_groups=first.instance_groups + second.instance_groups,
        frame_index=first.frame_index + second_index,
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
    `Q_j = F_src^T F_tgt / c_e^2` (TD-1: `R_r exp(A_e)`) is exactly orthogonal, and the operator applies only
    the orthogonal factors: no scale multiplication, and every inverse is a transpose.

    Args:
        src: source-side maps of the observed edges (`TD1MapOperator`, identity or dense).
        tgt: target-side maps of the same edges.
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
        src: TD1MapOperator | DenseMapOperator,
        tgt: TD1MapOperator | DenseMapOperator,
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
