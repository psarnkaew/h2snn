"""U3 evidence: T5/T16 Delta_0 changes under flips (shared vs separate roles) and the T6
spectrum (dim ker, largest kernel eigenvalue, smallest eigenvalue outside the kernel), both
for the extreme per-side scales of `_chain_operators` (diagnostic) and the T6 fixture.
Run from H2S-NN/: python ../docs/D7_evidence/u3_checks.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch

from tests import test_paths_and_facts as T

case = T.SharedRoleTests()
rel0 = torch.tensor([0, 2, 4, 6, 8])
for family, scale in T.FAMILIES:
    shared = case._change(family, scale, (0,), rel0)
    control = case._change(family, scale, (), torch.tensor([0, 2]))
    blind = case._change(family, scale, (0, 1), torch.arange(9))
    blind_ctrl = case._change(family, scale, (0,), torch.tensor([1, 3]))
    print(f"{family:10s} {scale:8s} T5 shared {shared:.2e} | control separate {control:.3e} | "
          f"T16 blind all-flip {blind:.2e} | control directed {blind_ctrl:.3e}")


def spectra(moderate: bool) -> None:
    t = T.SectionPreservationTests()
    d, N, gamma = 3, 8, 0.7
    for shared in (True, False):
        for seed in range(3):
            ei, groups = t._tree(seed)
            generator = torch.Generator().manual_seed(100 + seed)
            maps = T._chain_operators(ei.size(1), k=2, d=d, seed=seed, dtype=torch.float64,
                                      shared=shared)
            if moderate:
                maps = T._moderate_scales(maps, generator, shared)
            w = [torch.rand(n.size(0), generator=generator, dtype=torch.float64) + 0.5
                 for n, _, _ in groups]
            L_P = T._dense_cells(maps, [g[0] for g in groups], [g[1] for g in groups],
                                 [g[2] for g in groups], w, N, d)
            Fs, Ft = maps.src.materialize(), maps.tgt.materialize()
            D0 = torch.zeros_like(L_P)
            deg = torch.zeros(N, dtype=torch.float64)
            for e in range(ei.size(1)):
                u, v = int(ei[0, e]), int(ei[1, e])
                B = torch.zeros(d, N * d, dtype=torch.float64)
                B[:, u * d:(u + 1) * d] -= Fs[e]
                B[:, v * d:(v + 1) * d] += Ft[e]
                D0 += B.T @ B
                deg[u] += maps.src.scale[e] ** 2
                deg[v] += maps.tgt.scale[e] ** 2
            for (nodes, edges, signs), ww in zip(groups, w):
                n = nodes.size(1)
                sq = T._signed_dense_transports(maps, edges, signs).square().sum(dim=(-2, -1)) / d
                deg.index_add_(0, nodes.reshape(-1),
                               (gamma * (1 - 1 / n) * ww.unsqueeze(1) * sq).reshape(-1))
            s = deg.rsqrt().repeat_interleave(d)
            norm = s.view(-1, 1) * s.view(1, -1)
            for name, M in (("Delta0", D0), ("Delta0+gL_P", D0 + gamma * L_P)):
                ev = torch.linalg.eigvalsh(norm * M)
                k = int((ev <= 1e-9).sum())
                print(f"  shared={shared} seed={seed} {name:12s}: dim ker={k}, "
                      f"max kernel ev={float(ev[k - 1]):.2e}, min non-kernel ev={float(ev[k]):.3e}, "
                      f"lambda_max={float(ev[-1]):.4f}")


print("T6 spectra, extreme per-side scales [0.1, 50] of _chain_operators (diagnostic):")
spectra(moderate=False)
print("T6 spectra, T6 fixture (scales log-uniform in [0.5, 2]):")
spectra(moderate=True)
