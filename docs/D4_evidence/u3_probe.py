"""U3 self-review numbers (run from H2S-NN/): T2 errors, T4 spectral gap, T5 extremes."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from tests import test_v2_model as t  # noqa: E402


def main() -> None:
    worst = 0.0
    for mode in ("per_edge", "per_side"):
        for seed in range(3):
            c = t._Complex(seed=seed, scale_mode=mode)
            op = c.laplacian(gamma=1.0)
            worst = max(worst, t._rel_err(t._matrix(op.apply_path, c.num_nodes, c.d),
                                          t._dense_path_part(c)))
    print(f"T2 worst relative error vs Kron-reduced reference: {worst:.3e}")

    for seed in range(3):
        c, _ = t._tree_complex(seed)
        for label, op in (("edges", c.laplacian(gamma=None, terms=False)),
                          ("transport", c.laplacian(gamma=3.0)),
                          ("free", c.laplacian(gamma=3.0, free=True))):
            m = t._matrix(op.apply_normalized, c.num_nodes, c.d)
            ev = torch.linalg.eigvalsh(0.5 * (m + m.T)).abs().sort().values
            k = int((ev < 1e-9).sum())
            below = float(ev[k - 1]) if k else float("nan")
            above = float(ev[k]) if k < ev.numel() else float("nan")
            print(f"T4 seed={seed} {label:9s} kernel={k}  largest-in-kernel={below:.2e}  "
                  f"smallest-outside={above:.2e}")

    lam_n, lam_u = [], []
    for seed in range(60):
        mode = "per_edge" if seed < 40 else "per_side"
        c = t._Complex(seed=100 + seed, scale_mode=mode, scale_values=(t.EPS, 50.0 + t.EPS))
        g = torch.Generator().manual_seed(seed)
        gamma = float(10 ** (4 * torch.rand(1, generator=g) - 2))
        op = c.laplacian(gamma=gamma)
        n = t._matrix(op.apply_normalized, c.num_nodes, c.d)
        u = t._matrix(op.apply, c.num_nodes, c.d)
        lam_n.append(float(torch.linalg.eigvalsh(0.5 * (n + n.T)).max()))
        lam_u.append(float(torch.linalg.eigvalsh(u).max()))
    print(f"T5 normalized lambda_max: max={max(lam_n):.16f} min={min(lam_n):.4f}")
    print(f"T5 unnormalized lambda_max: max={max(lam_u):.3e} fixtures>2: "
          f"{sum(v > 2 for v in lam_u)}/60")


if __name__ == "__main__":
    main()
