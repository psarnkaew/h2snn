from __future__ import annotations

import torch

# Batched adaptation of fasth/fasthpp.py.
# Original implementation: Copyright (c) 2020 AlexanderMath, MIT License.
# The repository copy of that license is retained at fasth/LICENSE.


def householder_product_sequential(
    reflectors: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Exact fallback for reflector counts unsupported by the FastHPP merge.

    This function intentionally lives beside the FastHPP implementation so the
    active orthogonal parameterization has one implementation boundary.  It is
    used only for non-power-of-two dimensions; power-of-two square inputs use
    the compact-WY merge below.
    """
    if reflectors.dim() != 3:
        raise ValueError("reflectors must have shape [batch, d, count].")
    normalized = reflectors / torch.linalg.norm(
        reflectors,
        dim=-2,
        keepdim=True,
    ).clamp(min=eps)
    batch_size, d, count = normalized.shape
    output = torch.eye(
        d,
        count,
        dtype=normalized.dtype,
        device=normalized.device,
    ).unsqueeze(0).expand(batch_size, -1, -1)
    for column in range(count - 1, -1, -1):
        reflector = normalized[:, :, column].unsqueeze(-1)
        output = output - 2.0 * torch.bmm(
            reflector,
            torch.bmm(reflector.transpose(1, 2), output),
        )
    return output


def householder_product_wy(
    reflectors: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Materialize a batched Householder product with the FastHPP compact-WY merge.

    Args:
        reflectors: ``[B, d, d]`` tensor whose columns parameterize the ordered
            Householder reflectors.
        eps: Lower bound used while normalizing reflector columns.

    Returns:
        A ``[B, d, d]`` orthogonal matrix equal to the sequential reflector
        product up to floating-point roundoff.

    This is the batched PyTorch form of the merge in ``fasth/fasthpp.py``
    (Mathiasen et al., NeurIPS 2020).  The implementation deliberately exposes
    matrix materialization rather than a partial application because H2SNN
    reuses every learned restriction map in both coboundary directions.
    """
    if reflectors.dim() != 3:
        raise ValueError("reflectors must have shape [batch, d, d].")
    batch_size, d, count = reflectors.shape
    if d != count:
        return householder_product_sequential(reflectors, eps)
    if d <= 0:
        raise ValueError("stalk dimension must be positive.")
    if (d & (d - 1)) != 0:
        return householder_product_sequential(reflectors, eps)

    normalized = reflectors / torch.linalg.norm(
        reflectors,
        dim=-2,
        keepdim=True,
    ).clamp(min=eps)
    y = normalized.transpose(-1, -2).contiguous()
    w = -2.0 * y

    half = 1
    for _ in range(d.bit_length() - 1):
        block = half * 2
        y_blocks = y.view(batch_size, d // half, half, d)
        w_blocks = w.view(batch_size, d // half, half, d)
        lead = w_blocks[:, 0::2]
        trail = w_blocks[:, 1::2]
        inner = y_blocks[:, 0::2] @ trail.transpose(-1, -2)
        update = (lead.transpose(-1, -2) @ inner).transpose(-1, -2)
        w = torch.stack([lead, trail + update], dim=2).view(batch_size, d, d)
        half = block

    identity = torch.eye(d, dtype=reflectors.dtype, device=reflectors.device)
    return identity + w.transpose(-1, -2) @ y
