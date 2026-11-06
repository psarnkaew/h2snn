"""U3 T6 diagnosis: magnitude of L_P and of the section x vs the residual L_P x."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
from tests.test_paths_and_facts import SectionPreservationTests, _dense_cells
from tests.test_v2_model import _chain_operators

t = SectionPreservationTests()
d, N = 3, 8
for shared in (True, False):
    for seed in range(3):
        edge_index, groups = t._tree(seed)
        maps = _chain_operators(edge_index.size(1), k=2, d=d, seed=seed, dtype=torch.float64, shared=shared)
        Fs, Ft = maps.src.materialize(), maps.tgt.materialize()
        w = [torch.ones(n.size(0), dtype=torch.float64) for n, _, _ in groups]
        L = _dense_cells(maps, [g[0] for g in groups], [g[1] for g in groups], [g[2] for g in groups], w, N, d)
        x = torch.zeros(N, d, dtype=torch.float64); x[0] = torch.randn(d, dtype=torch.float64); known = {0}
        while len(known) < N:
            for e in range(edge_index.size(1)):
                u, v = int(edge_index[0, e]), int(edge_index[1, e])
                if u in known and v not in known: x[v] = torch.linalg.solve(Ft[e], Fs[e] @ x[u]); known.add(v)
                elif v in known and u not in known: x[u] = torch.linalg.solve(Fs[e], Ft[e] @ x[v]); known.add(u)
        r = (L @ x.reshape(-1)).abs().max()
        # natural scale of L_P x: |L| |x| summed over a row
        scale = (L.abs() @ x.reshape(-1).abs()).max()
        print(f"shared={shared} seed={seed}: |L_P|max={float(L.abs().max()):.3e} |x|max={float(x.abs().max()):.3e} "
              f"|L_P x|max={float(r):.3e}  relative to max_row(|L_P||x|)={float(r/scale):.3e}")
