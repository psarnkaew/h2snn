"""Node->edge restriction maps (decision D1, with the D1a per-edge scale).

`FactorizedNodeToEdgeSheaf`: `F = c R_{relation,side} H_k(u)`, returned as matrix-free
operators. `DenseConformalNodeToEdgeSheaf`: explicit `[M, d, d]` FastHPP maps with the
same scale law (the dense family of the map-family comparison).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.map_operators import (
    DenseMapOperator,
    EdgeMapPair,
    FactorizedMapOperator,
    checkpoint_if_grad,
    normalize_reflectors,
)
from src.models.orthogonal import Orthogonal
from src.utils.memory_guard import validate_node_edge_memory_budget

# Numerical guard of the Householder normalization (see `normalize_reflectors`): below this
# norm a reflection vector is not normalized exactly. Not a modelling hyperparameter.
REFLECTOR_NORM_EPS = 1e-6


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


def _symmetrize_shared(
    raw: torch.Tensor,
    mlp: nn.Module,
    h: torch.Tensor,
    src_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    relation: torch.Tensor,
    edge_type: torch.Tensor,
    shared_role_relations: tuple[int, ...],
    num_relations: int,
) -> torch.Tensor:
    """`raw` with `(g(h_u, h_v) + g(h_v, h_u)) / 2` on the edges of shared-role relations."""
    shared = _relation_mask(shared_role_relations, num_relations, edge_type.device)[edge_type]
    rows = shared.nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return raw
    reverse = torch.cat([h[tgt_idx[rows]], h[src_idx[rows]], relation[rows]], dim=-1)
    averaged = 0.5 * (raw[rows] + mlp(reverse).squeeze(-1))
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
    ) -> None:
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
        input_dim = 2 * d + d_r + 2
        self.reflector_mlp = _build_mlp(input_dim, hidden_dim, householder_k * d)
        if map_scale == "per_side":
            self.scale_mlp = _build_mlp(input_dim, hidden_dim, 1)
        else:
            self.edge_scale_mlp = _build_mlp(2 * d + d_r, hidden_dim, 1)

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
        side_code = like.new_zeros((edge_type.numel(), 2))
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
        h_self = h[self_idx]
        side_code = self._side_code(h_self, edge_type, side)
        inputs = torch.cat(
            [h_self, h[other_idx], relation_embeddings[edge_type], side_code], dim=-1
        )
        operator = FactorizedMapOperator.from_raw(
            raw_reflectors=self.reflector_mlp(inputs).view(-1, self.householder_k, self.d),
            raw_scale=self.scale_mlp(inputs).squeeze(-1),
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
        h_self = h[self_idx]
        side_code = self._side_code(h_self, edge_type, side)
        inputs = torch.cat(
            [h_self, h[other_idx], relation_embeddings[edge_type], side_code], dim=-1
        )
        raw = self.reflector_mlp(inputs).view(-1, self.householder_k, self.d)
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
        relation = relation_embeddings[edge_type]
        inputs = torch.cat([h[src_idx], h[tgt_idx], relation], dim=-1)
        raw = self.edge_scale_mlp(inputs).squeeze(-1)
        if self.shared_role_relations:
            raw = _symmetrize_shared(raw, self.edge_scale_mlp, h, src_idx, tgt_idx, relation,
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


class DenseConformalNodeToEdgeSheaf(nn.Module):
    """v2 dense map family (D4): `F = c * Q` with explicit `[M, d, d]` maps.

    `Q` is a per-edge, per-side FastHPP orthogonal map from per-relation MLPs over
    `[h_src, h_tgt, rel_emb]`; the scale `c` follows the same law as the factorized family
    (`map_scale`: one `c_e` per edge, or D1's per-side `c`). So dense and factorized differ
    only in how the orthogonal part is parameterized. Returned as `DenseMapOperator`s that
    carry `c`, so every inverse is the closed form `F^T / c^2`.

    Shared roles (D7): on edges of `shared_role_relations` the target map is the source MLP
    applied to the swapped input `[h_tgt, h_src, rel]` (one frame), the per-side scale has
    no side code, and the per-edge `c_e` is symmetrized over both input orders. The
    relation's `tgt_mlps` entry is then unused.
    """

    def __init__(
        self,
        d: int,
        d_r: int,
        num_relations: int,
        hidden_dim: int,
        scale_eps: float,
        orthogonal_map: str,
        map_scale: str,
        memory_guard_fraction: float,
        shared_role_relations: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        if scale_eps <= 0.0:
            raise ValueError(f"scale_eps must be positive, got {scale_eps}.")
        if map_scale not in {"per_side", "per_edge"}:
            raise ValueError(f"map_scale must be 'per_side' or 'per_edge', got {map_scale!r}.")
        self.num_relations = num_relations
        self.shared_role_relations = _validate_shared(shared_role_relations, num_relations)
        self.d = d
        self.scale_eps = scale_eps
        self.map_scale = map_scale
        self.memory_guard_fraction = memory_guard_fraction
        self.orthogonal = Orthogonal(d=d, orthogonal_map=orthogonal_map)
        self.param_dim = self.orthogonal.parameter_count
        input_dim = 2 * d + d_r
        self.src_mlps = nn.ModuleList(
            [_build_mlp(input_dim, hidden_dim, self.param_dim) for _ in range(num_relations)]
        )
        self.tgt_mlps = nn.ModuleList(
            [_build_mlp(input_dim, hidden_dim, self.param_dim) for _ in range(num_relations)]
        )
        if map_scale == "per_edge":
            self.edge_scale_mlp = _build_mlp(input_dim, hidden_dim, 1)
        else:
            self.scale_mlp = _build_mlp(input_dim + 2, hidden_dim, 1)

    @staticmethod
    def _project_by_relation(
        inputs: torch.Tensor, edge_type: torch.Tensor, mlps: nn.ModuleList
    ) -> torch.Tensor:
        output_dim = mlps[0][-1].out_features
        projected = inputs.new_zeros((inputs.size(0), output_dim))
        for relation_id, mlp in enumerate(mlps):
            relation_mask = edge_type == relation_id
            if relation_mask.any():
                projected[relation_mask] = mlp(inputs[relation_mask])
        return projected

    def _side_scale(self, h: torch.Tensor, relation_features: torch.Tensor,
                    self_idx: torch.Tensor, other_idx: torch.Tensor, side: int,
                    edge_type: torch.Tensor) -> torch.Tensor:
        side_code = h.new_zeros((self_idx.numel(), 2))
        if self.shared_role_relations:
            shared = _relation_mask(self.shared_role_relations, self.num_relations,
                                    edge_type.device)
            side_code[:, side] = (~shared[edge_type]).to(h.dtype)
        else:
            side_code[:, side] = 1.0
        inputs = torch.cat([h[self_idx], h[other_idx], relation_features, side_code], dim=-1)
        return F.softplus(self.scale_mlp(inputs).squeeze(-1)) + self.scale_eps

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        relation_embeddings: torch.Tensor,
    ) -> FactorizedNodeToEdgeSheafOutput:
        if h.is_cuda:
            validate_node_edge_memory_budget(
                num_edges=int(edge_index.size(1)),
                d=self.d,
                param_dim=self.param_dim,
                use_orthogonal=True,
                element_size=h.element_size(),
                total_device_bytes=torch.cuda.get_device_properties(h.device).total_memory,
                guard_fraction=self.memory_guard_fraction,
            )
        src_idx, tgt_idx = edge_index[0], edge_index[1]
        relation_features = relation_embeddings[edge_type]
        inputs = torch.cat([h[src_idx], h[tgt_idx], relation_features], dim=-1)
        Q_src = self.orthogonal(self._project_by_relation(inputs, edge_type, self.src_mlps))
        tgt_params = self._project_by_relation(inputs, edge_type, self.tgt_mlps)
        if self.shared_role_relations:
            shared = _relation_mask(self.shared_role_relations, self.num_relations,
                                    edge_type.device)[edge_type]
            rows = shared.nonzero(as_tuple=True)[0]
            if rows.numel():
                swapped = torch.cat([h[tgt_idx[rows]], h[src_idx[rows]],
                                     relation_features[rows]], dim=-1)
                tgt_params = tgt_params.index_put(
                    (rows,), self._project_by_relation(swapped, edge_type[rows], self.src_mlps)
                )
        Q_tgt = self.orthogonal(tgt_params)
        if self.map_scale == "per_edge":
            raw = self.edge_scale_mlp(inputs).squeeze(-1)
            if self.shared_role_relations:
                raw = _symmetrize_shared(raw, self.edge_scale_mlp, h, src_idx, tgt_idx,
                                         relation_features, edge_type,
                                         self.shared_role_relations, self.num_relations)
            c_src = F.softplus(raw) + self.scale_eps
            c_tgt = c_src
        else:
            c_src = self._side_scale(h, relation_features, src_idx, tgt_idx, 0, edge_type)
            c_tgt = self._side_scale(h, relation_features, tgt_idx, src_idx, 1, edge_type)
        return FactorizedNodeToEdgeSheafOutput(
            maps=EdgeMapPair(
                src=DenseMapOperator(Q_src * c_src.view(-1, 1, 1), c_src),
                tgt=DenseMapOperator(Q_tgt * c_tgt.view(-1, 1, 1), c_tgt),
            )
        )
