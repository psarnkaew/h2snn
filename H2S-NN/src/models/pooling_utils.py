from __future__ import annotations

import torch


def pool_edge_stalks_to_nodes(
    s_star: torch.Tensor,
    edge_index: torch.Tensor,
    F_src: torch.Tensor,
    F_tgt: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Transport each incident edge state into the node's own stalk, then mean-aggregate.

    Adding raw edge coordinates from *different* edge stalks straight into a node vector
    is not a well-defined sheaf operation, and it breaks the
    gauge invariance the Gram descriptors in `graph_pooling.py` rely on: under independent
    local gauges `x_v -> Q_v x_v`, `s_e -> R_e s_e`, the raw sum `sum_e R_e s_e` cannot be
    written as `Q_v (sum_e s_e)`, so the Gram entries mixing the node columns with the
    incident-edge column change with the choice of basis.

    Pulling each edge state back through its *endpoint-specific* restriction map first
    fixes that. With `F' = R_e F Q_v^T` we get

        (F')^T s'_e = (R_e F Q_v^T)^T R_e s_e = Q_v F^T R_e^T R_e s_e = Q_v F^T s_e ,

    so the summary co-transforms with the node stalk. This needs only the *gauge group* to
    be orthogonal (`R_e^T R_e = I`); `F` itself need not be — H2SNN's maps are
    conformal-orthogonal `F = cQ`. Using the adjoint `F^T` rather than the pseudo-inverse
    keeps the learned conformal incidence scale and matches the inner products the Hodge
    operators already use (`CoboundaryOps.delta0_T`).

    Both endpoints contribute with a `+` sign. This is an incident-edge *context* summary,
    not the coboundary adjoint `delta0_T`, which applies orientation-dependent incidence
    signs to represent a sheaf divergence.

    All of `s_star`, `edge_index`, `F_src`, and `F_tgt` must cover the same **extended**
    edge domain (real edges followed by closure 1-cells). An isolated node gets a zero
    summary; a self-loop counts as two incidences, once through each endpoint map.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}."
        )
    if F_src.dim() != 3 or F_tgt.dim() != 3:
        raise ValueError(
            "F_src and F_tgt must have shape [num_edges, d, d], got "
            f"{tuple(F_src.shape)} and {tuple(F_tgt.shape)}."
        )
    if F_src.shape != F_tgt.shape:
        raise ValueError(
            f"F_src and F_tgt must have the same shape, got {tuple(F_src.shape)} "
            f"and {tuple(F_tgt.shape)}."
        )

    num_edges = edge_index.size(1)
    if s_star.size(0) != num_edges or F_src.size(0) != num_edges:
        raise ValueError(
            "s_star, edge_index, F_src, and F_tgt must align over the same extended "
            f"edge domain (real + closure): got edge_index {num_edges}, s_star "
            f"{s_star.size(0)}, F_src {F_src.size(0)}."
        )

    d = s_star.size(-1)
    if F_src.size(1) != d or F_src.size(2) != d:
        raise ValueError(
            f"F_src/F_tgt must be [num_edges, {d}, {d}] to transport a stalk of "
            f"dimension {d}, got {tuple(F_src.shape)}."
        )

    # Same batched-adjoint pattern as CoboundaryOps.delta0_T, without the incidence signs.
    src_messages = torch.bmm(F_src.transpose(-2, -1), s_star.unsqueeze(-1)).squeeze(-1)
    tgt_messages = torch.bmm(F_tgt.transpose(-2, -1), s_star.unsqueeze(-1)).squeeze(-1)

    pooled_edges = torch.zeros((num_nodes, d), device=s_star.device, dtype=s_star.dtype)
    counts = torch.zeros((num_nodes, 1), device=s_star.device, dtype=s_star.dtype)
    ones = torch.ones((num_edges, 1), device=s_star.device, dtype=s_star.dtype)

    pooled_edges.index_add_(0, edge_index[0], src_messages)
    pooled_edges.index_add_(0, edge_index[1], tgt_messages)
    counts.index_add_(0, edge_index[0], ones)
    counts.index_add_(0, edge_index[1], ones)
    return pooled_edges / counts.clamp_min(1.0)
