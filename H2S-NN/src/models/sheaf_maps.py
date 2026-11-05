from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from src.data.structures import FaceGroup
from src.models.map_operators import (
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


class NodeToEdgeSheafOutput(NamedTuple):
    F_src: torch.Tensor
    F_tgt: torch.Tensor
    delta0_h0: torch.Tensor
    # Conformal scale c of F_src = c*Q, or None when maps are unconstrained. Lets the face
    # transport invert F_src as F_src^T / c^2 on the gathered chunk instead of allocating a
    # full [num_edges, d, d] inverse or running a general solve.
    F_src_scale: torch.Tensor | None


class FaceTransport(NamedTuple):
    """Edge-to-face maps built by transport, plus the closure edge's own restriction maps.

    `face_maps` is ordered [path edges in traversal order..., closure edge], matching the
    `L+1` columns of a face group's `face_edge_index`.
    """

    face_maps: torch.Tensor  # [B, L+1, d, d]
    closure_F_src: torch.Tensor  # [B, d, d] — the path holonomy
    closure_F_tgt: torch.Tensor  # [B, d, d] — identity


class EdgeToFaceOutput(NamedTuple):
    """Face map caches plus the closure edges' restriction maps.

    `closure_F_src`/`closure_F_tgt` are ordered to match the global closure edge ids handed
    out in `SequenceEdgeToFaceSheaf.forward`, i.e. group order then instance order — the same
    order `PreparedH2SData.closure_edge_index` uses.
    """

    caches: tuple[FaceMapCache, ...]
    closure_F_src: torch.Tensor  # [total_faces, d, d]
    closure_F_tgt: torch.Tensor  # [total_faces, d, d]
    instance_weights: tuple[torch.Tensor, ...] | None


def _invert_path_maps(
    F_src_path: torch.Tensor,
    conformal_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Invert gathered `[B, L, d, d]` source maps.

    For a conformal-orthogonal map `F = c*Q` we have `F^-1 = Q^T / c = F^T / c^2`, so the
    inverse is a transpose and a divide — no solve, and nothing allocated at
    `[num_edges, d, d]` scale. Falls back to a batched inverse when the maps are
    unconstrained (`use_orthogonal=False`), which is why Road A prefers orthogonal maps.
    """
    if conformal_scale is None:
        return torch.linalg.inv(F_src_path)
    squared = (conformal_scale * conformal_scale).unsqueeze(-1).unsqueeze(-1)
    return F_src_path.transpose(-2, -1) / squared


def _gather_factorized_path_maps(
    maps: EdgeMapPair,
    path_edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense `[B, L, d, d]` (F_src, F_tgt, F_src^-1) for one face chunk's path edges.

    Only the gathered path edges of the chunk are materialized -- the v1 face code needs
    explicit matrices until D4 removes it -- never the observed-edge set as a whole. The
    inverse is the closed form `c^-1 H^T R^T` (what `_invert_path_maps` does for v1).
    """
    if not isinstance(maps.src, FactorizedMapOperator) or not isinstance(
        maps.tgt, FactorizedMapOperator
    ):
        raise TypeError("factorized face gathering needs FactorizedMapOperator maps.")
    d = maps.src.d
    shape = tuple(path_edge_index.shape) + (d, d)

    def gather(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_maps, src_inverses = maps.src.select(rows).materialize_with_inverse()
        tgt_maps = maps.tgt.select(rows).materialize()
        return src_maps.view(shape), tgt_maps.view(shape), src_inverses.view(shape)

    # Checkpointed so autograd keeps only the three outputs (what v1's index gather keeps),
    # not the 2k+3 intermediate [B*L, d, d] products of each materialization.
    return checkpoint_if_grad(gather, path_edge_index)


def compose_face_transport(
    A_f: torch.Tensor,
    F_src_path: torch.Tensor,
    F_tgt_path: torch.Tensor,
    F_src_path_inv: torch.Tensor | None = None,
) -> FaceTransport:
    """Build functorial edge-to-face maps so that `delta1 . delta0 == 0` holds exactly.

    A cellular sheaf on a 2-complex is a functor on the face poset, so the edge-to-face
    map must compose with the node-to-edge map. Emitting both from independent networks
    breaks that, and `delta1 . delta0 = 0` then has no non-degenerate solution on an open
    path. Instead we carry a transport to a base vertex along the boundary:

        T_{v0}  = A_f                                          (free, one per instance)
        T_{v_i} = T_{v_{i-1}} (F^{e_i}_src)^-1 F^{e_i}_tgt
        F_{e_i<|f} = T_{v_{i-1}} (F^{e_i}_src)^-1

    Each boundary term then collapses to `T_v x_v - T_u x_u`. The synthetic closure edge
    `eC = (v_L, v_0)` points *along* the cycle (so its incidence sign is `+1` like the rest)
    and its maps are *defined*, not learned:

        F^{eC}_tgt = I,  F^{eC}_src = T_{v0}^-1 T_{v_L},  F_{eC<|f} = T_{v0} = A_f

    which closes the telescoping sum at exactly zero for every `x`, with no assumption that
    the sheaf has trivial holonomy. `F^{eC}_src` reduces to the path holonomy
    `prod_i (F^{e_i}_src)^-1 F^{e_i}_tgt` and so carries no new parameters and does not
    depend on `A_f`.

    Args:
        A_f: [B, d, d] free per-instance map (the face stalk's frame).
        F_src_path: [B, L, d, d] source-side node-to-edge maps of the path edges.
        F_tgt_path: [B, L, d, d] target-side node-to-edge maps of the path edges.
        F_src_path_inv: optional [B, L, d, d] precomputed inverse of `F_src_path`. Pass the
            closed form when the maps are conformal-orthogonal (`c*Q -> Q^T / c`) to avoid a
            general solve; otherwise a batched inverse is taken here.
    """
    if F_src_path.shape != F_tgt_path.shape:
        raise ValueError(
            "F_src_path and F_tgt_path must have identical shape, got "
            f"{tuple(F_src_path.shape)} vs {tuple(F_tgt_path.shape)}."
        )
    if F_src_path.dim() != 4:
        raise ValueError(
            f"F_src_path must be [B, L, d, d], got {tuple(F_src_path.shape)}."
        )

    inverses = torch.linalg.inv(F_src_path) if F_src_path_inv is None else F_src_path_inv
    length = F_src_path.size(1)

    identity = (
        torch.eye(A_f.size(-1), device=A_f.device, dtype=A_f.dtype)
        .unsqueeze(0)
        .expand_as(A_f)
        .clone()
    )
    face_maps: list[torch.Tensor] = []
    transport = A_f  # T_{v0}
    holonomy = identity
    for position in range(length):
        src_inv = inverses[:, position]
        # F_{e<|f} = T_u (F^e_src)^-1, evaluated at the transport of this edge's source node.
        face_maps.append(torch.matmul(transport, src_inv))
        step = torch.matmul(src_inv, F_tgt_path[:, position])
        transport = torch.matmul(transport, step)
        holonomy = torch.matmul(holonomy, step)

    # Closure edge: stored (v_L, v_0), i.e. traversed along the cycle -> sign +1.
    face_maps.append(A_f)
    return FaceTransport(
        face_maps=torch.stack(face_maps, dim=1),
        closure_F_src=holonomy,
        closure_F_tgt=identity,
    )


@dataclass(slots=True)
class FaceMapCache:
    template_name: str
    face_edge_index: torch.Tensor
    relation_index: torch.Tensor
    face_edge_sign: torch.Tensor  # incidence sign eta(e,f) in {+1,-1}; always +1 today (§2.4)
    maps: torch.Tensor  # precomputed once in forward(); reused across all diffusion steps
    group_id: int  # index into the originating face_groups tuple (a group may span >1 chunk)
    instance_start: int  # face-instance offset of this chunk within its face group
    instance_end: int

    def __post_init__(self) -> None:
        if self.face_edge_index.dim() != 2:
            raise ValueError("face_edge_index must have shape [faces, incidences].")
        if self.face_edge_sign.shape != self.face_edge_index.shape:
            raise ValueError("face_edge_sign must align with face_edge_index.")
        if self.face_edge_index.numel() > 0 and int(
            self.face_edge_index.min().item()
        ) < 0:
            raise ValueError("face_edge_index must be non-negative.")
        if self.face_edge_sign.numel() > 0 and not bool(
            torch.all(self.face_edge_sign.abs() == 1)
        ):
            raise ValueError("face_edge_sign entries must be exactly +1 or -1.")
        if self.maps.dim() != 4 or self.maps.shape[:2] != self.face_edge_index.shape:
            raise ValueError("Face maps must have shape [faces, incidences, d, d].")
        if self.maps.size(-1) != self.maps.size(-2):
            raise ValueError("Face maps must be square.")
        if self.relation_index.dim() != 2:
            raise ValueError("relation_index must be two-dimensional.")
        if self.relation_index.size(0) != self.face_edge_index.size(0):
            raise ValueError("relation_index face count must align with the cache.")
        if self.instance_start < 0 or self.instance_end <= self.instance_start:
            raise ValueError("Face cache instance range must be non-empty and ordered.")
        if self.group_id < 0:
            raise ValueError("Face cache group_id must be non-negative.")
        if self.instance_end - self.instance_start != self.face_edge_index.size(0):
            raise ValueError("Face cache instance range must equal its number of rows.")


class NodeToEdgeSheaf(nn.Module):
    def __init__(
        self,
        d: int,
        d_r: int,
        num_relations: int,
        hidden_dim: int,
        use_orthogonal: bool,
        orthogonal_map: str,
        memory_guard_fraction: float = 0.20,
    ) -> None:
        super().__init__()
        self.d = d
        self.d_r = d_r
        self.num_relations = num_relations
        self.use_orthogonal = use_orthogonal
        self.memory_guard_fraction = memory_guard_fraction
        self.orthogonal = Orthogonal(d=d, orthogonal_map=orthogonal_map)
        self.param_dim = self.orthogonal.parameter_count
        output_dim = self.param_dim + 1 if use_orthogonal else d * d
        input_dim = 2 * d + d_r
        self.src_mlps = nn.ModuleList(
            [_build_mlp(input_dim, hidden_dim, output_dim) for _ in range(num_relations)]
        )
        self.tgt_mlps = nn.ModuleList(
            [_build_mlp(input_dim, hidden_dim, output_dim) for _ in range(num_relations)]
        )

    def _materialise_maps(
        self, raw_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (maps, conformal_scale). `conformal_scale` is None when unconstrained.

        The scale is clamped from **below** as well as above: the face transport inverts
        F_src as `F_src^T / c^2`, so an unbounded-small `c` would make the inverse — and
        therefore the composed edge-to-face maps — explode.
        """
        if self.use_orthogonal:
            params = raw_params[:, :-1]
            scales = torch.exp(raw_params[:, -1].clamp(min=-2.0, max=2.0))
            return self.orthogonal(params) * scales.view(-1, 1, 1), scales
        return raw_params.view(-1, self.d, self.d), None

    def _project_by_relation(
        self,
        inputs: torch.Tensor,
        edge_type: torch.Tensor,
        mlps: nn.ModuleList,
    ) -> torch.Tensor:
        output_dim = mlps[0][-1].out_features
        projected = torch.zeros(
            (inputs.size(0), output_dim),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        for relation_id, mlp in enumerate(mlps):
            relation_mask = edge_type == relation_id
            if relation_mask.any():
                projected[relation_mask] = mlp(inputs[relation_mask])
        return projected

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        relation_embeddings: torch.Tensor,
    ) -> NodeToEdgeSheafOutput:
        if h.is_cuda:
            total_device_bytes = torch.cuda.get_device_properties(h.device).total_memory
            validate_node_edge_memory_budget(
                num_edges=int(edge_index.size(1)),
                d=self.d,
                param_dim=self.param_dim,
                use_orthogonal=self.use_orthogonal,
                element_size=h.element_size(),
                total_device_bytes=total_device_bytes,
                guard_fraction=self.memory_guard_fraction,
            )
        src_idx = edge_index[0]
        tgt_idx = edge_index[1]
        relation_features = relation_embeddings[edge_type]
        inputs = torch.cat([h[src_idx], h[tgt_idx], relation_features], dim=-1)
        src_raw = self._project_by_relation(inputs, edge_type, self.src_mlps)
        tgt_raw = self._project_by_relation(inputs, edge_type, self.tgt_mlps)
        F_src, F_src_scale = self._materialise_maps(src_raw)
        F_tgt, _ = self._materialise_maps(tgt_raw)

        projected_src = torch.bmm(F_src, h[src_idx].unsqueeze(-1)).squeeze(-1)
        projected_tgt = torch.bmm(F_tgt, h[tgt_idx].unsqueeze(-1)).squeeze(-1)
        delta0_h0 = projected_tgt - projected_src
        return NodeToEdgeSheafOutput(
            F_src=F_src,
            F_tgt=F_tgt,
            delta0_h0=delta0_h0,
            F_src_scale=F_src_scale,
        )


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
        """`[num_relations, 2, d, d]`: `R_{psi, side}` with side 0 = source, 1 = target."""
        return self.orthogonal(self.rotation_params).view(
            self.num_relations, 2, self.d, self.d
        )

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
        side_code = h_self.new_zeros((h_self.size(0), 2))
        side_code[:, side] = 1.0
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
        side_code = h_self.new_zeros((h_self.size(0), 2))
        side_code[:, side] = 1.0
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
        """`c_e = softplus(edge_scale_mlp([h_src, h_tgt, rel_emb])) + eps`, one per edge."""
        inputs = torch.cat([h[src_idx], h[tgt_idx], relation_embeddings[edge_type]], dim=-1)
        return F.softplus(self.edge_scale_mlp(inputs).squeeze(-1)) + self.scale_eps

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
            )
            operators.append(
                FactorizedMapOperator(
                    scale=scale,
                    reflectors=reflectors,
                    rotations=side_rotations,
                    group_ids=edge_type,
                )
            )
        # delta0 is applied by the caller over the extended (observed + closure) edge set.
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
        )
        operators: list[FactorizedMapOperator] = []
        for side, (self_idx, other_idx) in enumerate(((src_idx, tgt_idx), (tgt_idx, src_idx))):
            reflectors = checkpoint_if_grad(
                lambda h_in, rel_in, i=self_idx, j=other_idx, s=side: (
                    self._side_reflectors(h_in, rel_in, i, j, edge_type, s)
                ),
                h,
                relation_embeddings,
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


class EdgeInitializer(nn.Module):
    def __init__(
        self,
        d: int,
        d_r: int,
        hidden_dim: int,
        use_delta0_init: bool,
    ) -> None:
        super().__init__()
        input_dim = 2 * d + d_r + (d if use_delta0_init else 0)
        self.use_delta0_init = use_delta0_init
        self.mlp = _build_mlp(input_dim, hidden_dim, d)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        relation_embeddings: torch.Tensor,
        delta0_h0: torch.Tensor | None,
    ) -> torch.Tensor:
        src_idx = edge_index[0]
        tgt_idx = edge_index[1]
        inputs = [h[src_idx], h[tgt_idx], relation_embeddings[edge_type]]
        if self.use_delta0_init and delta0_h0 is not None:
            inputs.append(delta0_h0)
        return self.mlp(torch.cat(inputs, dim=-1))


class SequenceEdgeToFaceSheaf(nn.Module):
    """Produces the free per-instance face map `A_f`; the rest of the face geometry follows
    by transport (`compose_face_transport`).

    Two encoders are available. `concat` (default) concatenates the path's node states in
    order with the schema's relation embeddings and applies one MLP per distinct path
    length. `lstm` is the superseded token-MLP + LSTM + attention-pool stack, kept so the
    sequence-encoder ablation is runnable.

    Why `concat` is the default: every face group has a *fixed* `L`, and the manual ACM/IMDB
    templates are `L=2` throughout, so the recurrence ran over a two-element sequence while
    its attention pool -- an order-agnostic weighted sum -- discarded part of the order it
    had just injected. Positional concatenation gives each slot its own weights, is a
    universal approximator over the ordered tuple, and drops the `O(L)` factor the LSTM
    carried in activation memory.
    """

    def __init__(
        self,
        d: int,
        d_r: int,
        d_h: int,
        hidden_dim: int,
        face_chunk_size: int = 4096,
        face_encoder: str = "concat",
        face_map_param: str = "full",
        face_lengths: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        if face_encoder not in {"concat", "lstm"}:
            raise ValueError(f"Unsupported face_encoder '{face_encoder}'.")
        if face_map_param not in {"full", "triangular"}:
            raise ValueError(f"Unsupported face_map_param '{face_map_param}'.")
        self.d = d
        self.d_r = d_r
        self.d_h = d_h
        self.hidden_dim = hidden_dim
        self.face_chunk_size = face_chunk_size
        self.face_encoder = face_encoder
        self.face_map_param = face_map_param
        self.face_lengths = tuple(face_lengths)
        # A_f enters the model only through A_f^T A_f, so a full d x d output carries
        # d(d-1)/2 gauge directions no gradient can see. "triangular" emits the identifiable
        # d(d+1)/2 instead; it is lossless because every PSD matrix is R^T R for upper
        # triangular R.
        self.map_output_dim = d * d if face_map_param == "full" else d * (d + 1) // 2

        if face_encoder == "concat":
            # Built eagerly: a projection created lazily on first forward would be
            # registered after build_optimizer() captured model.parameters(), and would
            # silently never receive updates.
            self.input_proj = nn.ModuleDict(
                {
                    str(length): nn.Linear((length + 1) * d + length * d_r, hidden_dim)
                    for length in self.face_lengths
                }
            )
            self.activation = nn.GELU()
        else:
            self.token_mlp = _build_mlp(2 * d + d_r, hidden_dim, d_h)
            self.lstm = nn.LSTM(input_size=d_h, hidden_size=d_h, batch_first=True)
            self.attention_vector = nn.Parameter(torch.randn(d_h))
        head_in = hidden_dim if face_encoder == "concat" else d_h
        self.face_latent_dim = head_in
        self.map_head = nn.Linear(head_in, self.map_output_dim)

    def _assemble_A_f(self, raw: torch.Tensor) -> torch.Tensor:
        """`[B, map_output_dim]` -> `[B, d, d]`.

        Under `triangular` the emitted values fill the **upper** triangle including the
        diagonal, so that `A_f^T A_f` ranges over exactly the PSD matrices (Cholesky).
        """
        if self.face_map_param == "full":
            return raw.view(-1, self.d, self.d)
        rows, cols = torch.triu_indices(self.d, self.d, offset=0, device=raw.device)
        maps = raw.new_zeros((raw.size(0), self.d, self.d))
        maps[:, rows, cols] = raw
        return maps

    def _face_map_from_path(
        self,
        node_states: torch.Tensor,
        relation_features: torch.Tensor,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an ordered path and return ``(A_f, instance_latent)``."""
        key = str(length)
        if key not in self.input_proj:
            available = ", ".join(sorted(self.input_proj)) or "none"
            raise KeyError(
                f"face encoder 'concat' has no input projection for path length {length} "
                f"(built for lengths: {available}). ModelDims.face_lengths must list every "
                "length the data materialises."
            )
        batch_size = node_states.size(0)
        inputs = torch.cat(
            [
                node_states.reshape(batch_size, -1),
                relation_features.expand(batch_size, -1),
            ],
            dim=-1,
        )
        hidden = self.activation(self.input_proj[key](inputs))
        return self._assemble_A_f(self.map_head(hidden)), hidden

    def forward(
        self,
        h: torch.Tensor,
        face_groups: tuple[FaceGroup, ...],
        relation_embeddings: torch.Tensor,
        F_src: torch.Tensor | None,
        F_tgt: torch.Tensor | None,
        F_src_scale: torch.Tensor | None,
        num_real_edges: int,
        closure_scale: float = 1.0,
        instance_weight_scorer: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
        factorized_maps: EdgeMapPair | None = None,
    ) -> EdgeToFaceOutput:
        """Build functorial edge-to-face maps and the closure edges' restriction maps.

        The LSTM+attention no longer emits one free map per (edge, face) incidence: a
        non-uniform per-edge rescaling destroys the boundary telescoping, so
        `delta1 . delta0 = 0` could not hold. Attention instead **pools** the path's hidden
        states into a single vector per instance, which produces the one free map `A_f`; the
        per-edge maps then follow by transport (see `compose_face_transport`).

        `face_edge_index` in the returned caches gains an `L+1`-th column holding the global
        edge id of that instance's closure edge, which lives at
        `num_real_edges + <running face-instance index>` — the same order in which
        `PreparedH2SData.closure_edge_index` concatenates per-group closure edges.

        Pass either dense `F_src`/`F_tgt` (+ `F_src_scale`) or `factorized_maps` (D1). In the
        factorized case only each chunk's gathered path edges are materialized.
        """
        if factorized_maps is None:
            if F_src is None or F_tgt is None:
                raise ValueError("Dense F_src and F_tgt are required without factorized_maps.")
        elif F_src is not None or F_tgt is not None or F_src_scale is not None:
            raise ValueError("Pass either dense F_src/F_tgt/F_src_scale or factorized_maps.")
        caches: list[FaceMapCache] = []
        closure_F_src: list[torch.Tensor] = []
        closure_F_tgt: list[torch.Tensor] = []
        weight_parts: list[list[torch.Tensor]] | None = (
            [[] for _ in face_groups] if instance_weight_scorer is not None else None
        )
        closure_cursor = 0

        for group_id, face_group in enumerate(face_groups):
            if face_group.num_faces == 0:
                continue

            for start in range(0, face_group.num_faces, self.face_chunk_size):
                end = min(start + self.face_chunk_size, face_group.num_faces)
                chunk_face_node_index = face_group.face_node_index[start:end]
                chunk_face_edge_index = face_group.face_edge_index[start:end]
                chunk_relation_index = face_group.relation_index[start:end]
                chunk_face_edge_sign = face_group.face_edge_sign[start:end]

                batch_size, length = chunk_face_edge_index.shape
                if self.face_encoder == "concat":
                    # The relation sequence is constant within a face group -- a FaceGroup
                    # materialises one schema, which fixes (r_1..r_L) -- so row 0 is the
                    # schema's identity and broadcasting it avoids a [B, L, d_r] gather.
                    # `validate_relation_sequence_constant` asserts this at build time.
                    schema_token = relation_embeddings[chunk_relation_index[0]].reshape(1, -1)
                    A_f, instance_latent = self._face_map_from_path(
                        node_states=h[chunk_face_node_index],
                        relation_features=schema_token,
                        length=length,
                    )
                else:
                    src_nodes = chunk_face_node_index[:, :-1]
                    tgt_nodes = chunk_face_node_index[:, 1:]
                    relation_features = relation_embeddings[chunk_relation_index]
                    tokens = torch.cat(
                        [h[src_nodes], h[tgt_nodes], relation_features], dim=-1
                    )
                    projected = self.token_mlp(tokens.reshape(batch_size * length, -1))
                    projected = projected.view(batch_size, length, self.d_h)
                    hidden_states, _ = self.lstm(projected)
                    logits = torch.einsum("blh,h->bl", hidden_states, self.attention_vector)
                    attention = torch.softmax(logits / self.d_h**0.5, dim=1)

                    # Attention-pooled instance summary -> the single free map A_f.
                    pooled = torch.einsum("blh,bl->bh", hidden_states, attention)
                    instance_latent = pooled
                    A_f = self._assemble_A_f(self.map_head(pooled))

                if instance_weight_scorer is not None:
                    learned_weight = instance_weight_scorer(group_id, instance_latent)
                    if learned_weight.shape != (batch_size,):
                        raise ValueError(
                            "instance_weight_scorer must return one scalar per face instance."
                        )
                    assert weight_parts is not None
                    weight_parts[group_id].append(learned_weight)

                if factorized_maps is None:
                    F_src_path = F_src[chunk_face_edge_index]
                    F_tgt_path = F_tgt[chunk_face_edge_index]
                    F_src_path_inv = _invert_path_maps(
                        F_src_path,
                        None if F_src_scale is None else F_src_scale[chunk_face_edge_index],
                    )
                else:
                    F_src_path, F_tgt_path, F_src_path_inv = _gather_factorized_path_maps(
                        factorized_maps, chunk_face_edge_index
                    )
                transport = compose_face_transport(
                    A_f=A_f,
                    F_src_path=F_src_path,
                    F_tgt_path=F_tgt_path,
                    F_src_path_inv=F_src_path_inv,
                )

                # Scaling the closure edge's restriction maps by `closure_scale` is absorbed
                # into its face map so the telescoping — and hence delta1.delta0 = 0 — is
                # preserved for any positive scale.
                maps = transport.face_maps
                group_closure_F_src = transport.closure_F_src
                group_closure_F_tgt = transport.closure_F_tgt
                if closure_scale <= 0.0:
                    raise ValueError(
                        "data.closure_scale must be > 0: the closure 1-cell is what makes the "
                        "face boundary a cycle, and its face map carries a 1/closure_scale "
                        "factor. To ablate the closure edge, set model.ablate_no_delta1 (which "
                        "drops faces and closure edges together) rather than scaling to zero."
                    )
                if closure_scale != 1.0:
                    group_closure_F_src = group_closure_F_src * closure_scale
                    group_closure_F_tgt = group_closure_F_tgt * closure_scale
                    maps = torch.cat(
                        [maps[:, :-1], (maps[:, -1:] / closure_scale)], dim=1
                    )

                closure_ids = torch.arange(
                    num_real_edges + closure_cursor,
                    num_real_edges + closure_cursor + batch_size,
                    device=chunk_face_edge_index.device,
                    dtype=chunk_face_edge_index.dtype,
                )
                closure_cursor += batch_size

                caches.append(
                    FaceMapCache(
                        template_name=face_group.template_name,
                        face_edge_index=torch.cat(
                            [chunk_face_edge_index, closure_ids.view(-1, 1)], dim=1
                        ),
                        relation_index=chunk_relation_index,
                        face_edge_sign=torch.cat(
                            [
                                chunk_face_edge_sign,
                                chunk_face_edge_sign.new_ones((batch_size, 1)),
                            ],
                            dim=1,
                        ),
                        maps=maps,
                        group_id=group_id,
                        instance_start=start,
                        instance_end=end,
                    )
                )
                closure_F_src.append(group_closure_F_src)
                closure_F_tgt.append(group_closure_F_tgt)

        if not closure_F_src:
            empty = (F_src if F_src is not None else h).new_zeros((0, self.d, self.d))
            return EdgeToFaceOutput(
                caches=tuple(),
                closure_F_src=empty,
                closure_F_tgt=empty,
                instance_weights=(
                    tuple(h.new_empty(0) for _ in face_groups)
                    if weight_parts is not None
                    else None
                ),
            )
        return EdgeToFaceOutput(
            caches=tuple(caches),
            closure_F_src=torch.cat(closure_F_src, dim=0),
            closure_F_tgt=torch.cat(closure_F_tgt, dim=0),
            instance_weights=(
                tuple(
                    torch.cat(parts, dim=0) if parts else h.new_empty(0)
                    for parts in weight_parts
                )
                if weight_parts is not None
                else None
            ),
        )

    def build_maps(self, cache: FaceMapCache) -> torch.Tensor:
        return cache.maps
