"""Review response (finding 6): Freebase book-book task-level H_edge with and without self-loops,
same convention as compute_heterophily.task_level_h_edge (symmetrised, calcHomophily fill seeds
0-4; labelled-only diagnostic). Run from H2S-NN/."""
import sys
from pathlib import Path

import torch
from torch_geometric.utils import to_undirected

sys.path.insert(0, str(Path.cwd()))
from scripts.heterophily import compute_heterophily as ch  # noqa: E402

data = ch.load_hgb(Path("data"), "freebase")
y = data["book"].y
ei = to_undirected(data["book", "and", "book"].edge_index, num_nodes=data["book"].num_nodes)
for name, keep in (("self-loops kept (reported)", torch.ones(ei.size(1), dtype=torch.bool)),
                   ("self-loops dropped", ei[0] != ei[1])):
    r, c = ei[:, keep]
    filled = [1 - float((f[r] == f[c]).double().mean()) for f in
              (ch.h2gb_random_fill(y, s) for s in range(5))]
    lab = (y[r] >= 0) & (y[c] >= 0)
    lo = 1 - float((y[r][lab] == y[c][lab]).double().mean())
    print(f"{name}: edges {int(keep.sum())}; fill seed 0 {filled[0]:.4f} "
          f"(seeds {min(filled):.4f}-{max(filled):.4f}); labelled-only {lo:.4f} on {int(lab.sum())} edges")
