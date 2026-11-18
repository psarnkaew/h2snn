"""Random TD-1 map factors for operator-level tests (spec 1.25 Z2; model change TD-1 card M2/M3).

`random_td1_factors` draws one shared scale per fact (log-uniform in [0.5, 2]), a random `C_e` in
so(m) per fact, orthonormal `U_r` per relation and frames `R_r` in SO(d) for the relations with
rho(r) != r; `random_td1_maps` turns them into the production operator pair (`td1_edge_maps`, the
builder's own function). Shared-role rows hold `exp(C_e / 2) - I`, as the builder stores them.
"""

from __future__ import annotations

import math

import torch

from src.models.map_operators import EdgeMapPair, skew_from_coefficients
from src.models.sheaf_maps import TD1Factors, td1_edge_maps


def random_rotation(count: int, d: int, generator: torch.Generator,
                    dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """`[count, d, d]` random matrices in SO(d) (QR of a Gaussian, first column flipped if det < 0)."""
    q, _ = torch.linalg.qr(torch.randn(count, d, d, generator=generator, dtype=torch.float64))
    sign = torch.sign(torch.linalg.det(q))
    q[:, :, 0] = q[:, :, 0] * sign.view(-1, 1)
    return q.to(dtype)


def random_subspaces(count: int, d: int, rank: int, generator: torch.Generator,
                     dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """`[count, d, m]` orthonormal columns."""
    q, _ = torch.linalg.qr(torch.randn(count, d, rank, generator=generator, dtype=torch.float64))
    return q.to(dtype)


def random_td1_factors(num_edges: int, edge_type: torch.Tensor, d: int, rank: int,
                       generator: torch.Generator, dtype: torch.dtype = torch.float64,
                       shared_role_relations: tuple[int, ...] = (), num_relations: int | None = None,
                       angle: float = 1.0, scale: torch.Tensor | None = None) -> TD1Factors:
    """Random factors; `angle` scales the skew coefficients (N(0, angle^2))."""
    num_relations = int(edge_type.max()) + 1 if num_relations is None else num_relations
    if scale is None:
        log_scale = math.log(0.5) + math.log(4.0) * torch.rand(num_edges, generator=generator,
                                                                dtype=torch.float64)
        scale = log_scale.exp()
    coefficients = angle * torch.randn(num_edges, rank * (rank - 1) // 2, generator=generator,
                                       dtype=torch.float64)
    generator_matrix = skew_from_coefficients(coefficients, rank)
    if shared_role_relations:
        shared = torch.zeros(num_relations, dtype=torch.bool)
        shared[list(shared_role_relations)] = True
        generator_matrix = generator_matrix * torch.where(shared[edge_type], 0.5, 1.0).view(-1, 1, 1)
    rotation = torch.linalg.matrix_exp(generator_matrix) - torch.eye(rank, dtype=torch.float64)
    framed = [r for r in range(num_relations) if r not in set(shared_role_relations)]
    return TD1Factors(scale=scale.to(dtype), rotation=rotation.to(dtype),
                      subspaces=random_subspaces(num_relations, d, rank, generator, dtype),
                      frames=random_rotation(len(framed), d, generator, dtype))


def random_td1_maps(num_edges: int, edge_type: torch.Tensor, d: int, rank: int,
                    generator: torch.Generator, dtype: torch.dtype = torch.float64,
                    shared_role_relations: tuple[int, ...] = (), num_relations: int | None = None,
                    **kwargs) -> EdgeMapPair:
    num_relations = int(edge_type.max()) + 1 if num_relations is None else num_relations
    factors = random_td1_factors(num_edges, edge_type, d, rank, generator, dtype,
                                 shared_role_relations, num_relations, **kwargs)
    return td1_edge_maps(factors, edge_type, num_relations, shared_role_relations)
