"""Node->edge restriction maps (spec 1.25 Z2, TD-1; D1a per-fact scale).

`TD1NodeToEdgeSheaf` builds, per fact `e = (u -> v)` of relation `r`,

    rho(r) != r:   F_{u<|e} = c_e I,                    F_{v<|e} = c_e R_r exp(U_r C_e U_r^T)
    rho(r) == r:   F_{u<|e} = c_e exp(-U_r C_e U_r^T/2), F_{v<|e} = c_e exp(+U_r C_e U_r^T/2)

and returns them as matrix-free `TD1MapOperator`s (one per side). (The Householder family
`F = c R_{relation,side} H_k(u)` of spec 1.24 was replaced by spec 1.25 Z2.)
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.map_operators import (
    EdgeMapPair,
    IdentityMapOperator,
    TD1MapOperator,
    checkpoint_if_grad,
    orthonormal_subspaces,
    skew_from_coefficients,
)
from src.models.orthogonal import Orthogonal


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _relation_mask(relations: tuple[int, ...], num_relations: int,
                   device: torch.device) -> torch.Tensor:
    """`[num_relations]` bool, True for the listed relation ids."""
    mask = torch.zeros(num_relations, dtype=torch.bool, device=device)
    if relations:
        mask[list(relations)] = True
    return mask


def _validate_shared(shared_role_relations: tuple[int, ...], num_relations: int) -> tuple[int, ...]:
    shared = tuple(sorted(set(int(r) for r in shared_role_relations)))
    if any(r < 0 or r >= num_relations for r in shared):
        raise ValueError(
            f"shared_role_relations {shared} out of range for {num_relations} relations."
        )
    return shared


def validate_map_rank(rank: int, d: int) -> None:
    """`2 <= m <= d` (spec 1.25 Z2; m = 4 outside unit tests, enforced by the config)."""
    if isinstance(rank, bool) or not isinstance(rank, int) or not 2 <= rank <= d:
        raise ValueError(f"map_rank must be an integer in [2, d={d}], got {rank!r}.")


class TD1Factors(NamedTuple):
    """What the operators of one edge set are built from (the tensors the anchored solve
    differentiates): the shared scale `[E]`, the target side's `exp(.) - I` `[E, m, m]`, the
    orthonormal subspaces `[R, d, m]` and the frames `[R_f, d, d]` (rows of the relations with
    rho(r) != r, in ascending relation order)."""

    scale: torch.Tensor
    rotation: torch.Tensor
    subspaces: torch.Tensor
    frames: torch.Tensor


def td1_edge_maps(factors: TD1Factors, edge_type: torch.Tensor, num_relations: int,
                  shared_role_relations: tuple[int, ...], type_only: bool = False) -> EdgeMapPair:
    """The source and target operators of one edge set (spec 1.25 Z2):

    - relation with rho(r) != r: source `c_e I`, target `c_e R_r exp(A_e)` (frame row = the
      relation's rank among the relations with rho(r) != r);
    - relation with rho(r) = r: source `c_e exp(-A_e/2)`, target `c_e exp(A_e/2)`; `factors.rotation`
      holds `exp(C_e/2) - I` on these rows and the source reads it transposed
      (`exp(-A/2) = exp(A/2)^T`, no second block is stored);
    - `type_only` (N2c): `C_e = 0` on every row (frames alone)."""
    shared = set(int(r) for r in shared_role_relations)
    framed = [r for r in range(num_relations) if r not in shared]
    frame_row = {r: i for i, r in enumerate(framed)}
    tgt_frames = tuple(frame_row.get(r, -1) for r in range(num_relations))
    tgt_instance = tuple(not type_only for _ in range(num_relations))
    src_instance = tuple((r in shared) and not type_only for r in range(num_relations))
    tgt = TD1MapOperator(factors.scale, factors.rotation, edge_type, factors.subspaces,
                         factors.frames, tgt_instance, tgt_frames)
    src = TD1MapOperator(factors.scale, factors.rotation.transpose(-2, -1), edge_type,
                         factors.subspaces, factors.frames, src_instance,
                         tuple(-1 for _ in range(num_relations)))
    return EdgeMapPair(src=src, tgt=tgt)


class TD1NodeToEdgeSheafOutput(NamedTuple):
    maps: EdgeMapPair  # TD1MapOperator (IdentityMapOperator under maps = I) per side
    factors: "TD1Factors | None"  # None under maps = I


class TD1NodeToEdgeSheaf(nn.Module):
    """TD-1 node->edge maps (spec 1.25 Z2), returned as operators, not tensors.

        [k_e, s_e] = edge_mlp([h_u, h_v, rel_emb(r)])          one MLP, m(m-1)/2 + 1 outputs
        C_e        = skew(k_e) in so(m),   c_e = softplus(s_e) + scale_eps
        U_r        = thin-QR(W_r).Q  (sign(diag R) > 0),  R_r = FastHPP(theta_r)  (rho(r) != r only)

    Initialisation (card §0): the skew rows of the MLP's last layer (weights and bias) are zero, so
    `C_e = 0` and `Q_e = R_r` at init; the scale row keeps `nn.Linear`'s default initialisation (that
    of the former `edge_scale_mlp`); `R_r`'s FastHPP parameters are drawn N(0, 1/d) as before; `W_r`
    is N(0, 1).

    Shared roles (D7): for the relations in `shared_role_relations` (rho(r) = r, or every relation
    under `ablation.direction_blind_maps`) the MLP is evaluated in both input orders; the skew part
    is `(k(u, v) - k(v, u)) / 2` (antisymmetric in the endpoints) and the scale input
    `(s(u, v) + s(v, u)) / 2`, so the operator does not depend on the stored orientation; no frame.

    `identity_maps` (arm "maps = I", `ablation.fact_maps_identity: true`; spec 1.25 Z-17): `R_r = I`
    and `C_e = 0`, only `c_e` is learned. The frames, `W_r` and the skew rows are created (so every
    other parameter initialises exactly as in `full`) and then removed: they are not parameters of
    this arm.

    `type_only=True` in `forward` (N2c read-out, evaluation only): `C_e = 0`, i.e. the relation frames
    alone.
    """

    def __init__(
        self,
        d: int,
        d_r: int,
        num_relations: int,
        hidden_dim: int,
        map_rank: int,
        scale_eps: float,
        orthogonal_map: str,
        checkpoint: bool = True,
        shared_role_relations: tuple[int, ...] = (),
        identity_maps: bool = False,
    ) -> None:
        super().__init__()
        validate_map_rank(map_rank, d)
        if scale_eps <= 0.0:
            raise ValueError(f"scale_eps must be positive, got {scale_eps}.")
        self.d = d
        self.d_r = d_r
        self.num_relations = num_relations
        self.map_rank = map_rank
        self.num_skew = map_rank * (map_rank - 1) // 2
        self.scale_eps = scale_eps
        self.checkpoint = checkpoint
        # Plain tuples, not buffers: the state_dict is the same with or without shared roles.
        self.shared_role_relations = _validate_shared(shared_role_relations, num_relations)
        shared = set(self.shared_role_relations)
        self.framed_relations = tuple(r for r in range(num_relations) if r not in shared)
        self.orthogonal = Orthogonal(d=d, orthogonal_map=orthogonal_map)
        # One FastHPP frame per relation with rho(r) != r (spec 1.25 Z2: the second frame per
        # (relation, side) of 1.24 was a gauge). N(0, 1/d), the 1.24 initialisation (author, (a)).
        self.frame_params = nn.Parameter(
            torch.empty(len(self.framed_relations), self.orthogonal.parameter_count))
        nn.init.normal_(self.frame_params, mean=0.0, std=1.0 / math.sqrt(d))
        # W_r per relation, N(0, 1); U_r = thin-QR(W_r).Q.
        self.subspace_params = nn.Parameter(torch.empty(num_relations, d, map_rank))
        nn.init.normal_(self.subspace_params, mean=0.0, std=1.0)
        self.edge_mlp = _build_mlp(2 * d + d_r, hidden_dim, self.num_skew + 1)
        last = self.edge_mlp[-1]
        with torch.no_grad():  # C_e = 0 at init; the scale row keeps the default Linear init
            last.weight[:self.num_skew].zero_()
            last.bias[:self.num_skew].zero_()
        self.identity_maps = identity_maps
        if identity_maps:
            del self.frame_params
            del self.subspace_params
            scale_row = nn.Linear(hidden_dim, 1)
            with torch.no_grad():
                scale_row.weight.copy_(last.weight[self.num_skew:])
                scale_row.bias.copy_(last.bias[self.num_skew:])
            self.edge_mlp[-1] = scale_row

    # ------------------------------------------------------------------ factors
    def frames(self) -> torch.Tensor:
        """`[R_f, d, d]` frames `R_r` of the relations with rho(r) != r (ascending relation id)."""
        if not self.framed_relations:
            return self.frame_params.new_zeros((0, self.d, self.d))
        return self.orthogonal(self.frame_params)

    def subspaces(self) -> torch.Tensor:
        """`[R, d, m]` orthonormal `U_r`."""
        return orthonormal_subspaces(self.subspace_params)

    def operators(self, factors: TD1Factors, edge_type: torch.Tensor,
                  type_only: bool = False) -> EdgeMapPair:
        """The two sides' operators from `factors` (also the anchored solve's `build`)."""
        return td1_edge_maps(factors, edge_type, self.num_relations, self.shared_role_relations,
                             type_only)

    def _edge_outputs(self, h: torch.Tensor, relation_embeddings: torch.Tensor,
                      src_idx: torch.Tensor, tgt_idx: torch.Tensor,
                      edge_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(raw skew coefficients [E, m(m-1)/2] (or [E, 0] under maps = I), raw scale [E])`; on
        shared-role edges the skew part is antisymmetrised and the scale input symmetrised over both
        input orders."""
        raw = self.edge_mlp(torch.cat([h[src_idx], h[tgt_idx], relation_embeddings[edge_type]],
                                      dim=-1))
        skew, scale = raw[:, :-1], raw[:, -1]
        if self.shared_role_relations:
            shared = _relation_mask(self.shared_role_relations, self.num_relations,
                                    edge_type.device)[edge_type]
            rows = shared.nonzero(as_tuple=True)[0]
            if rows.numel():
                swapped = self.edge_mlp(torch.cat(
                    [h[tgt_idx[rows]], h[src_idx[rows]], relation_embeddings[edge_type[rows]]],
                    dim=-1))
                scale = scale.index_put((rows,), 0.5 * (scale[rows] + swapped[:, -1]))
                if skew.size(-1):
                    skew = skew.index_put((rows,), 0.5 * (skew[rows] - swapped[:, :-1]))
        return skew, scale

    def _scale(self, h: torch.Tensor, relation_embeddings: torch.Tensor, src_idx: torch.Tensor,
               tgt_idx: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """`c_e = softplus(s_e) + eps` (maps = I)."""
        _, raw = self._edge_outputs(h, relation_embeddings, src_idx, tgt_idx, edge_type)
        return F.softplus(raw) + self.scale_eps

    def _factors(self, h: torch.Tensor, relation_embeddings: torch.Tensor, src_idx: torch.Tensor,
                 tgt_idx: torch.Tensor, edge_type: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(c_e [E], exp(.) - I [E, m, m])` of the target side: `exp(C_e)` for rho(r) != r,
        `exp(C_e / 2)` for rho(r) = r (batched m x m `torch.linalg.matrix_exp`)."""
        skew, raw_scale = self._edge_outputs(h, relation_embeddings, src_idx, tgt_idx, edge_type)
        scale = F.softplus(raw_scale) + self.scale_eps
        generator = skew_from_coefficients(skew, self.map_rank)
        if self.shared_role_relations:
            shared = _relation_mask(self.shared_role_relations, self.num_relations,
                                    edge_type.device)[edge_type]
            half = torch.where(shared, 0.5, 1.0).to(generator.dtype)
            generator = generator * half.view(-1, 1, 1)
        eye = torch.eye(self.map_rank, dtype=generator.dtype, device=generator.device)
        return scale, torch.linalg.matrix_exp(generator) - eye

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        relation_embeddings: torch.Tensor,
        type_only: bool = False,
    ) -> TD1NodeToEdgeSheafOutput:
        src_idx = edge_index[0]
        tgt_idx = edge_index[1]
        # Checkpointed: autograd keeps h and the factors (the map storage itself), not the
        # [E, hidden] MLP activations, which are replayed in backward.
        if self.identity_maps:
            scale = checkpoint_if_grad(
                lambda h_in, rel_in: self._scale(h_in, rel_in, src_idx, tgt_idx, edge_type),
                h, relation_embeddings, enabled=self.checkpoint)
            return TD1NodeToEdgeSheafOutput(maps=EdgeMapPair(
                src=IdentityMapOperator(scale, self.d), tgt=IdentityMapOperator(scale, self.d)),
                factors=None)
        scale, rotation = checkpoint_if_grad(
            lambda h_in, rel_in: self._factors(h_in, rel_in, src_idx, tgt_idx, edge_type),
            h, relation_embeddings, enabled=self.checkpoint)
        factors = TD1Factors(scale=scale, rotation=rotation, subspaces=self.subspaces(),
                             frames=self.frames())
        return TD1NodeToEdgeSheafOutput(maps=self.operators(factors, edge_type, type_only),
                                        factors=factors)
