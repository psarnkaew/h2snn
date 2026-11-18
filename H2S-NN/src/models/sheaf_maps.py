"""Node->edge restriction maps (decision D1, with the D1a per-edge scale).

`FactorizedNodeToEdgeSheaf`: `F = c R_{relation,side} H_k(u)`, returned as matrix-free
operators. (The dense family `DenseConformalNodeToEdgeSheaf` ran only in the v2 model and was
removed by the legacy-removal card, B4.)
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.map_operators import (
    EdgeMapPair,
    FactorizedMapOperator,
    IdentityMapOperator,
    checkpoint_if_grad,
    normalize_reflectors,
)
from src.models.orthogonal import Orthogonal

# Numerical guard of the Householder normalization (see `normalize_reflectors`): below this
# norm a reflection vector is not normalized exactly. Not a modelling hyperparameter.
REFLECTOR_NORM_EPS = 1e-6


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


Block = tuple[torch.Tensor, "torch.Tensor | None"]


def mlp_on_blocks(mlp: nn.Sequential, blocks: tuple[Block, ...]) -> torch.Tensor:
    """`mlp` on the row-wise concatenation of the gathered blocks (`index` None: the block is
    already per row). Spec 1.25 Z1: the factored first layer existed only for `[h0, c]`."""
    return mlp(torch.cat([source if index is None else source[index]
                          for source, index in blocks], dim=-1))


def _relation_mask(relations: tuple[int, ...], num_relations: int,
                   device: torch.device) -> torch.Tensor:
    """`[num_relations]` bool, True for the listed relation ids."""
    mask = torch.zeros(num_relations, dtype=torch.bool, device=device)
    if relations:
        mask[list(relations)] = True
    return mask


def _symmetrize_shared(
    raw: torch.Tensor,
    apply_reversed: Callable[[torch.Tensor], torch.Tensor],
    edge_type: torch.Tensor,
    shared_role_relations: tuple[int, ...],
    num_relations: int,
) -> torch.Tensor:
    """`raw` with `(g(h_u, h_v) + g(h_v, h_u)) / 2` on the edges of shared-role relations;
    `apply_reversed(rows)` is `g` on the swapped inputs of those rows, `[len(rows)]`."""
    shared = _relation_mask(shared_role_relations, num_relations, edge_type.device)[edge_type]
    rows = shared.nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return raw
    averaged = 0.5 * (raw[rows] + apply_reversed(rows))
    return raw.index_put((rows,), averaged)


def _validate_shared(shared_role_relations: tuple[int, ...], num_relations: int) -> tuple[int, ...]:
    shared = tuple(sorted(set(int(r) for r in shared_role_relations)))
    if any(r < 0 or r >= num_relations for r in shared):
        raise ValueError(
            f"shared_role_relations {shared} out of range for {num_relations} relations."
        )
    return shared


class FactorizedNodeToEdgeSheafOutput(NamedTuple):
    maps: EdgeMapPair  # FactorizedMapOperator per side over the observed edges


class FactorizedNodeToEdgeSheaf(nn.Module):
    """Factorized node->edge maps (v2 decision D1), returned as operators, not tensors.

        F_{v<|e} = c_{v<|e} * R_{psi(e),side} * H_k(u_{v<|e})
        c        = softplus(scale_mlp(z)) + scale_eps
        u        = reflector_mlp(z),   z = [h_v, h_other, rel_emb(psi(e)), onehot(side)]

    `R_{psi,side}` is one FastHPP orthogonal matrix per (relation, side), shared by all
    edges of that relation. The two sides differ in their frame `R`, in the side code, and
    in the order of the endpoint states, so `F_src != F_tgt` and the edge transport
    `F_src^-1 F_tgt` is not the identity (a shared map would collapse path holonomy).

    `map_scale="per_edge"` (v2 decision D1a) replaces the per-incidence `c` by one scale per
    edge shared by both sides, `c_e = softplus(edge_scale_mlp([h_u, h_v, rel_emb])) + eps`
    from the ordered edge input; both operators then hold the same `[E]` tensor, and every
    edge transport `F_src^-1 F_tgt` is exactly orthogonal.

    Shared roles (D7, plan §9.1): for the relations in `shared_role_relations` (those with
    `rho(r) = r`, or every relation under `ablation.direction_blind_maps`) the two roles
    coincide: the target side uses the source frame `R_{psi,0}`, the side code is zero, and
    `c_e = softplus((g(h_u, h_v) + g(h_v, h_u)) / 2) + eps` is symmetrized over both input
    orders, so the operator does not depend on the stored orientation of these edges.
    """

    def __init__(
        self,
        d: int,
        d_r: int,
        num_relations: int,
        hidden_dim: int,
        householder_k: int,
        scale_eps: float,
        orthogonal_map: str,
        map_scale: str = "per_side",
        checkpoint: bool = True,
        shared_role_relations: tuple[int, ...] = (),
        identity_maps: bool = False,
    ) -> None:
        """The per-node geometry input is `h0` (width `d`; spec 1.25 Z1).
        `identity_maps` (arm "maps = I", `ablation.fact_maps_identity: true`; spec 1.23 X-11): the orthogonal
        factor `R H` is the identity; only the per-edge scale `c_e` (hence `a_e = c_e^2`) is learned.
        The frames and the reflector MLP are created (so every other parameter initialises exactly
        as in `full`) and then removed: they are not parameters of this arm."""
        super().__init__()
        if householder_k < 1:
            raise ValueError(f"householder_k must be >= 1, got {householder_k}.")
        if scale_eps <= 0.0:
            raise ValueError(f"scale_eps must be positive, got {scale_eps}.")
        if map_scale not in {"per_side", "per_edge"}:
            raise ValueError(f"map_scale must be 'per_side' or 'per_edge', got {map_scale!r}.")
        self.d = d
        self.d_r = d_r
        self.num_relations = num_relations
        self.householder_k = householder_k
        self.scale_eps = scale_eps
        self.map_scale = map_scale
        self.checkpoint = checkpoint
        # Plain tuple, not a buffer: the state_dict is the same with or without shared roles.
        self.shared_role_relations = _validate_shared(shared_role_relations, num_relations)
        self.orthogonal = Orthogonal(d=d, orthogonal_map=orthogonal_map)
        # Row 2 * relation + side holds the FastHPP *parameters* (strictly-lower Householder
        # entries) of R_{relation, side}, drawn N(0, 1/d) so the two sides start in
        # different frames. The std is an initialization choice, not a config key.
        self.rotation_params = nn.Parameter(
            torch.empty(2 * num_relations, self.orthogonal.parameter_count)
        )
        nn.init.normal_(self.rotation_params, mean=0.0, std=1.0 / math.sqrt(d))
        node_dim = d
        mlp_input_dim = 2 * node_dim + d_r + 2
        self.reflector_mlp = _build_mlp(mlp_input_dim, hidden_dim, householder_k * d)
        if map_scale == "per_side":
            self.scale_mlp = _build_mlp(mlp_input_dim, hidden_dim, 1)
        else:
            self.edge_scale_mlp = _build_mlp(2 * node_dim + d_r, hidden_dim, 1)
        self.identity_maps = identity_maps
        if identity_maps:
            if map_scale != "per_edge":
                raise ValueError("identity maps keep one scale per edge (D1a): map_scale per_edge.")
            del self.rotation_params
            del self.reflector_mlp

    def rotations(self) -> torch.Tensor:
        """`[num_relations, 2, d, d]`: `R_{psi, side}` with side 0 = source, 1 = target.

        A shared-role relation uses its source frame on both sides."""
        frames = self.orthogonal(self.rotation_params).view(
            self.num_relations, 2, self.d, self.d
        )
        if not self.shared_role_relations:
            return frames
        shared = _relation_mask(self.shared_role_relations, self.num_relations, frames.device)
        target = torch.where(shared.view(-1, 1, 1), frames[:, 0], frames[:, 1])
        return torch.stack([frames[:, 0], target], dim=1)

    def _side_code(self, like: torch.Tensor, edge_type: torch.Tensor, side: int) -> torch.Tensor:
        """`[M, 2]` one-hot side code; all zero on edges of shared-role relations."""
        side_code = like.new_zeros((edge_type.numel(), 2), dtype=like.dtype)
        if not self.shared_role_relations:
            side_code[:, side] = 1.0
            return side_code
        shared = _relation_mask(self.shared_role_relations, self.num_relations, edge_type.device)
        side_code[:, side] = (~shared[edge_type]).to(like.dtype)
        return side_code

    def _side_factors(
        self,
        h: torch.Tensor,
        relation_embeddings: torch.Tensor,
        self_idx: torch.Tensor,
        other_idx: torch.Tensor,
        edge_type: torch.Tensor,
        side: int,
        rotations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(c, unit u)` per incidence on one side: `z = [h_v, h_other, rel_emb, onehot(side)]`."""
        side_code = self._side_code(h, edge_type, side)
        # one input tensor shared by both MLPs (the D11 computation; same gradient order)
        inputs = torch.cat([h[self_idx], h[other_idx], relation_embeddings[edge_type], side_code],
                           dim=-1)
        raw_reflectors, raw_scale = self.reflector_mlp(inputs), self.scale_mlp(inputs)
        operator = FactorizedMapOperator.from_raw(
            raw_reflectors=raw_reflectors.view(-1, self.householder_k, self.d),
            raw_scale=raw_scale.squeeze(-1),
            rotations=rotations,
            group_ids=edge_type,
            scale_eps=self.scale_eps,
            reflector_eps=REFLECTOR_NORM_EPS,
        )
        return operator.scale, operator.reflectors

    def _side_reflectors(
        self,
        h: torch.Tensor,
        relation_embeddings: torch.Tensor,
        self_idx: torch.Tensor,
        other_idx: torch.Tensor,
        edge_type: torch.Tensor,
        side: int,
    ) -> torch.Tensor:
        """Unit reflectors only (per-edge scale mode); same input `z` as `_side_factors`."""
        side_code = self._side_code(h, edge_type, side)
        blocks = ((h, self_idx), (h, other_idx), (relation_embeddings, edge_type), (side_code, None))
        raw = mlp_on_blocks(self.reflector_mlp, blocks).view(
            -1, self.householder_k, self.d)
        return normalize_reflectors(raw, REFLECTOR_NORM_EPS)

    def _edge_scale(
        self,
        h: torch.Tensor,
        relation_embeddings: torch.Tensor,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """`c_e = softplus(edge_scale_mlp([h_src, h_tgt, rel_emb])) + eps`, one per edge;
        on shared-role edges the MLP output is averaged over both input orders."""
        blocks = ((h, src_idx), (h, tgt_idx), (relation_embeddings, edge_type))
        raw = mlp_on_blocks(self.edge_scale_mlp, blocks).squeeze(-1)
        if self.shared_role_relations:
            raw = _symmetrize_shared(
                raw,
                lambda rows: mlp_on_blocks(
                    self.edge_scale_mlp,
                    ((h, tgt_idx[rows]), (h, src_idx[rows]), (relation_embeddings, edge_type[rows]))
                ).squeeze(-1),
                edge_type, self.shared_role_relations, self.num_relations)
        return F.softplus(raw) + self.scale_eps

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        relation_embeddings: torch.Tensor,
    ) -> FactorizedNodeToEdgeSheafOutput:
        src_idx = edge_index[0]
        tgt_idx = edge_index[1]
        if self.identity_maps:
            scale = checkpoint_if_grad(
                lambda h_in, rel_in: self._edge_scale(h_in, rel_in, src_idx, tgt_idx, edge_type),
                h,
                relation_embeddings,
                enabled=self.checkpoint,
            )
            return FactorizedNodeToEdgeSheafOutput(maps=EdgeMapPair(
                src=IdentityMapOperator(scale, self.d), tgt=IdentityMapOperator(scale, self.d)))
        rotations = self.rotations()
        if self.map_scale == "per_edge":
            return self._forward_per_edge(
                h, src_idx, tgt_idx, edge_type, relation_embeddings, rotations
            )
        operators: list[FactorizedMapOperator] = []
        for side, (self_idx, other_idx) in enumerate(((src_idx, tgt_idx), (tgt_idx, src_idx))):
            # Checkpointed per side: autograd keeps h and the factors (the map storage
            # itself), not the [M, hidden] MLP activations, which are replayed in backward.
            side_rotations = rotations[:, side]
            scale, reflectors = checkpoint_if_grad(
                lambda h_in, rel_in, i=self_idx, j=other_idx, s=side, r=side_rotations: (
                    self._side_factors(h_in, rel_in, i, j, edge_type, s, r)
                ),
                h,
                relation_embeddings,
                enabled=self.checkpoint,
            )
            operators.append(
                FactorizedMapOperator(
                    scale=scale,
                    reflectors=reflectors,
                    rotations=side_rotations,
                    group_ids=edge_type,
                )
            )
        return FactorizedNodeToEdgeSheafOutput(
            maps=EdgeMapPair(src=operators[0], tgt=operators[1])
        )

    def _forward_per_edge(
        self,
        h: torch.Tensor,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        edge_type: torch.Tensor,
        relation_embeddings: torch.Tensor,
        rotations: torch.Tensor,
    ) -> FactorizedNodeToEdgeSheafOutput:
        # Checkpointed like the per-side factors: only h and the factors are kept for backward.
        scale = checkpoint_if_grad(
            lambda h_in, rel_in: self._edge_scale(h_in, rel_in, src_idx, tgt_idx, edge_type),
            h,
            relation_embeddings,
            enabled=self.checkpoint,
        )
        operators: list[FactorizedMapOperator] = []
        for side, (self_idx, other_idx) in enumerate(((src_idx, tgt_idx), (tgt_idx, src_idx))):
            reflectors = checkpoint_if_grad(
                lambda h_in, rel_in, i=self_idx, j=other_idx, s=side: (
                    self._side_reflectors(h_in, rel_in, i, j, edge_type, s)
                ),
                h,
                relation_embeddings,
                enabled=self.checkpoint,
            )
            # The same `scale` tensor object on both sides: one c_e per edge (D1a).
            operators.append(
                FactorizedMapOperator(
                    scale=scale,
                    reflectors=reflectors,
                    rotations=rotations[:, side],
                    group_ids=edge_type,
                )
            )
        return FactorizedNodeToEdgeSheafOutput(
            maps=EdgeMapPair(src=operators[0], tgt=operators[1])
        )
