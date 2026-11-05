from __future__ import annotations


def estimate_node_edge_live_bytes(
    *,
    num_edges: int,
    d: int,
    param_dim: int,
    use_orthogonal: bool,
    element_size: int,
) -> int:
    """Estimate the minimum simultaneous bytes for raw parameters and dense maps."""
    raw_dim = param_dim + 1 if use_orthogonal else d * d
    raw_elements = 2 * num_edges * raw_dim
    map_elements = 2 * num_edges * d * d
    return int((raw_elements + map_elements) * element_size)


def validate_node_edge_memory_budget(
    *,
    num_edges: int,
    d: int,
    param_dim: int,
    use_orthogonal: bool,
    element_size: int,
    total_device_bytes: int,
    guard_fraction: float,
) -> None:
    estimated_bytes = estimate_node_edge_live_bytes(
        num_edges=num_edges,
        d=d,
        param_dim=param_dim,
        use_orthogonal=use_orthogonal,
        element_size=element_size,
    )
    budget_bytes = int(total_device_bytes * guard_fraction)
    if estimated_bytes <= budget_bytes:
        return

    safe_d = 1
    for candidate_d in range(2, d):
        candidate_param_dim = candidate_d * (candidate_d - 1) // 2
        candidate_bytes = estimate_node_edge_live_bytes(
            num_edges=num_edges,
            d=candidate_d,
            param_dim=candidate_param_dim,
            use_orthogonal=use_orthogonal,
            element_size=element_size,
        )
        if candidate_bytes <= budget_bytes:
            safe_d = candidate_d
        else:
            break

    gib = 1024**3
    raise RuntimeError(
        "Unsafe node-to-edge CUDA allocation rejected before map materialisation: "
        f"edges={num_edges:,}, d={d}, estimated minimum live memory="
        f"{estimated_bytes / gib:.2f} GiB, device capacity={total_device_bytes / gib:.2f} GiB, "
        f"guard budget={guard_fraction:.0%} ({budget_bytes / gib:.2f} GiB). "
        f"Use d<={safe_d}, reduce the graph/batch size, or deliberately adjust "
        "model.node_edge_memory_guard_fraction after profiling."
    )
