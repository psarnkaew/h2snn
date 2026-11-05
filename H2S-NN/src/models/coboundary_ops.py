from __future__ import annotations

from typing import Callable, NamedTuple

import torch

from src.models.map_operators import EdgeMapPair
from src.models.sheaf_maps import FaceMapCache


class Delta1Result(NamedTuple):
    grouped: tuple[torch.Tensor, ...]
    concatenated: torch.Tensor
    maps: tuple[torch.Tensor, ...]  # precomputed F_{e▷f} maps; reused in delta1_T to avoid redundant forward passes


class CoboundaryOps:
    @staticmethod
    def delta0(
        x: torch.Tensor,
        F_src: torch.Tensor,
        F_tgt: torch.Tensor,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
    ) -> torch.Tensor:
        source = torch.bmm(F_src, x[src_idx].unsqueeze(-1)).squeeze(-1)
        target = torch.bmm(F_tgt, x[tgt_idx].unsqueeze(-1)).squeeze(-1)
        return target - source

    @staticmethod
    def delta0_T(
        s: torch.Tensor,
        F_src: torch.Tensor,
        F_tgt: torch.Tensor,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        output = torch.zeros((num_nodes, s.size(-1)), device=s.device, dtype=s.dtype)
        target = torch.bmm(F_tgt.transpose(-2, -1), s.unsqueeze(-1)).squeeze(-1)
        source = torch.bmm(F_src.transpose(-2, -1), s.unsqueeze(-1)).squeeze(-1)
        output.index_add_(0, tgt_idx, target)
        output.index_add_(0, src_idx, -source)
        return output

    @staticmethod
    def _delta1_grouped(
        s: torch.Tensor,
        face_caches: tuple[FaceMapCache, ...],
        map_builder: Callable[[FaceMapCache], torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        grouped: list[torch.Tensor] = []
        maps_list: list[torch.Tensor] = []
        for cache in face_caches:
            maps = map_builder(cache)
            maps_list.append(maps)
            edge_states = s[cache.face_edge_index]
            per_edge = torch.matmul(maps, edge_states.unsqueeze(-1)).squeeze(-1)
            per_edge = per_edge * cache.face_edge_sign.unsqueeze(-1)
            values = per_edge.sum(dim=1)
            grouped.append(values)
        return tuple(grouped), tuple(maps_list)

    @staticmethod
    def delta1(
        s: torch.Tensor,
        face_caches: tuple[FaceMapCache, ...],
        map_builder: Callable[[FaceMapCache], torch.Tensor],
    ) -> Delta1Result:
        """Apply the raw face coboundary ``B``."""
        grouped, maps_list = CoboundaryOps._delta1_grouped(
            s,
            face_caches,
            map_builder,
        )
        if not grouped:
            empty = s.new_zeros((0, s.size(-1)))
            return Delta1Result(grouped=tuple(), concatenated=empty, maps=tuple())
        return Delta1Result(grouped=grouped, concatenated=torch.cat(grouped, dim=0), maps=maps_list)

    @staticmethod
    def scale_delta1_rows(
        delta1_result: Delta1Result,
        face_caches: tuple[FaceMapCache, ...],
        face_weights: tuple[torch.Tensor, ...] | None,
        *,
        power: float,
    ) -> Delta1Result:
        """Scale face rows while retaining the raw coboundary maps."""
        if face_weights is None:
            return delta1_result
        if power not in {0.5, 1.0}:
            raise ValueError("delta1 row-weight power must be 0.5 or 1.0.")

        scaled_groups: list[torch.Tensor] = []
        for cache, values in zip(
            face_caches,
            delta1_result.grouped,
            strict=True,
        ):
            if not 0 <= cache.group_id < len(face_weights):
                raise ValueError("Face cache group_id is outside face_weights.")
            weights = face_weights[cache.group_id][
                cache.instance_start : cache.instance_end
            ].to(device=values.device, dtype=values.dtype)
            if weights.numel() != values.size(0):
                raise ValueError("Face-weight slice does not align with cached face rows.")
            weights = weights.clamp_min(0.0)
            if power == 0.5:
                positive = weights > 0
                safe_inputs = torch.where(
                    positive,
                    weights,
                    torch.ones_like(weights),
                )
                scales = torch.where(
                    positive,
                    torch.sqrt(safe_inputs),
                    torch.zeros_like(weights),
                )
            else:
                scales = weights
            scaled_groups.append(values * scales.unsqueeze(-1))

        concatenated = (
            torch.cat(scaled_groups, dim=0)
            if scaled_groups
            else delta1_result.concatenated.new_zeros(
                (0, delta1_result.concatenated.size(-1))
            )
        )
        return Delta1Result(
            grouped=tuple(scaled_groups),
            concatenated=concatenated,
            maps=delta1_result.maps,
        )

    @staticmethod
    def delta1_T(
        delta1_result: Delta1Result,
        face_caches: tuple[FaceMapCache, ...],
        num_edges: int,
        d: int,
    ) -> torch.Tensor:
        if num_edges < 0 or d <= 0:
            raise ValueError("delta1_T requires num_edges >= 0 and d > 0.")
        if delta1_result.concatenated.dim() != 2:
            raise ValueError("delta1 concatenated values must have shape [faces, d].")
        if delta1_result.concatenated.size(1) != d:
            raise ValueError("delta1 value width does not match delta1_T stalk dimension.")
        if sum(values.size(0) for values in delta1_result.grouped) != int(
            delta1_result.concatenated.size(0)
        ):
            raise ValueError("Grouped and concatenated delta1 face counts disagree.")
        output = torch.zeros((num_edges, d), device=delta1_result.concatenated.device, dtype=delta1_result.concatenated.dtype)
        for cache, face_values, maps in zip(
            face_caches,
            delta1_result.grouped,
            delta1_result.maps,
            strict=True,
        ):
            expected_map_shape = (
                face_values.size(0),
                cache.face_edge_index.size(1),
                d,
                d,
            )
            if face_values.dim() != 2 or face_values.size(1) != d:
                raise ValueError("Each grouped delta1 value must have shape [faces, d].")
            if tuple(maps.shape) != expected_map_shape:
                raise ValueError(
                    "Face maps do not align with delta1 values/incidences: "
                    f"expected {expected_map_shape}, got {tuple(maps.shape)}."
                )
            if cache.face_edge_index.numel() > 0 and (
                int(cache.face_edge_index.min().item()) < 0
                or int(cache.face_edge_index.max().item()) >= num_edges
            ):
                raise ValueError("Face cache references an edge outside delta1_T output.")
            # Reuse precomputed maps from delta1 — avoids a redundant map_builder forward pass per Euler step.
            expanded_faces = face_values.unsqueeze(1).expand(-1, cache.face_edge_index.size(1), -1)
            contributions = torch.matmul(
                maps.transpose(-2, -1),
                expanded_faces.unsqueeze(-1),
            ).squeeze(-1)
            contributions = contributions * cache.face_edge_sign.unsqueeze(-1)
            output.index_add_(
                0,
                cache.face_edge_index.reshape(-1),
                contributions.reshape(-1, d),
            )
        return output

    @staticmethod
    def L0(
        x: torch.Tensor,
        F_src: torch.Tensor,
        F_tgt: torch.Tensor,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        return CoboundaryOps.delta0_T(
            CoboundaryOps.delta0(x, F_src, F_tgt, src_idx, tgt_idx),
            F_src,
            F_tgt,
            src_idx,
            tgt_idx,
            num_nodes,
        )

    @staticmethod
    def L1(
        s: torch.Tensor,
        F_src: torch.Tensor,
        F_tgt: torch.Tensor,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        num_nodes: int,
        face_caches: tuple[FaceMapCache, ...],
        map_builder: Callable[[FaceMapCache], torch.Tensor],
        num_edges: int,
        face_weights: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        down = CoboundaryOps.delta0(
            CoboundaryOps.delta0_T(s, F_src, F_tgt, src_idx, tgt_idx, num_nodes),
            F_src,
            F_tgt,
            src_idx,
            tgt_idx,
        )
        up = CoboundaryOps._upper_hodge_term(
            s, face_caches, map_builder, num_edges, face_weights
        )
        return down + up

    @staticmethod
    def _upper_hodge_term(
        s: torch.Tensor,
        face_caches: tuple[FaceMapCache, ...],
        map_builder: Callable[[FaceMapCache], torch.Tensor],
        num_edges: int,
        face_weights: tuple[torch.Tensor, ...] | None,
    ) -> torch.Tensor:
        """`delta1^T W delta1 s`, the face part of `L1`."""
        # delta1 builds maps once; delta1_T reuses them from the result — no redundant forward pass.
        raw_up_result = CoboundaryOps.delta1(s, face_caches, map_builder)
        up_result = CoboundaryOps.scale_delta1_rows(
            raw_up_result,
            face_caches,
            face_weights,
            power=1.0,
        )
        return CoboundaryOps.delta1_T(
            delta1_result=up_result,
            face_caches=face_caches,
            num_edges=num_edges,
            d=s.size(-1),
        )

    # ------------------------------------------------------------------------------
    # Matrix-free variants (v2 decision D1). Maps are operators (`EdgeMapPair`), so no
    # `[num_edges, d, d]` tensor is gathered or formed; the arithmetic matches the dense
    # methods above term by term.
    # ------------------------------------------------------------------------------
    @staticmethod
    def delta0_matrix_free(
        x: torch.Tensor,
        maps: EdgeMapPair,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
    ) -> torch.Tensor:
        """`(delta0 x)_e = F^e_tgt x_tgt - F^e_src x_src`."""
        return maps.tgt.apply(x[tgt_idx]) - maps.src.apply(x[src_idx])

    @staticmethod
    def delta0_T_matrix_free(
        s: torch.Tensor,
        maps: EdgeMapPair,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Exact adjoint of `delta0_matrix_free`."""
        output = torch.zeros((num_nodes, s.size(-1)), device=s.device, dtype=s.dtype)
        output.index_add_(0, tgt_idx, maps.tgt.apply_T(s))
        output.index_add_(0, src_idx, -maps.src.apply_T(s))
        return output

    @staticmethod
    def L0_matrix_free(
        x: torch.Tensor,
        maps: EdgeMapPair,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        return CoboundaryOps.delta0_T_matrix_free(
            CoboundaryOps.delta0_matrix_free(x, maps, src_idx, tgt_idx),
            maps,
            src_idx,
            tgt_idx,
            num_nodes,
        )

    @staticmethod
    def L1_matrix_free(
        s: torch.Tensor,
        maps: EdgeMapPair,
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        num_nodes: int,
        face_caches: tuple[FaceMapCache, ...],
        map_builder: Callable[[FaceMapCache], torch.Tensor],
        num_edges: int,
        face_weights: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """v1 `L1` with a matrix-free down term; the face (up) term is unchanged v1 code."""
        down = CoboundaryOps.delta0_matrix_free(
            CoboundaryOps.delta0_T_matrix_free(s, maps, src_idx, tgt_idx, num_nodes),
            maps,
            src_idx,
            tgt_idx,
        )
        up = CoboundaryOps._upper_hodge_term(
            s, face_caches, map_builder, num_edges, face_weights
        )
        return down + up
