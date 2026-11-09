"""What does HeaRT's calc_CN_metric (create_heart_negatives.py:69-75, HeaRT c447cbf) compute? Its three lines,
verbatim, on a toy graph, against textbook RA and CN/deg(candidate)."""
import torch
from torch_sparse import SparseTensor
# path graph 0-1-2-3 plus 1-3 (undirected); degrees 1, 3, 2, 2
und = [(0, 1), (1, 2), (2, 3), (1, 3)]
ei = torch.tensor(und + [(b, a) for a, b in und]).t()
adj = SparseTensor.from_edge_index(ei, torch.ones(ei.size(1)), [4, 4])
non_weighted_adj = adj
# --- verbatim (create_heart_negatives.py:71-75)
reciprocal_degree = 1 / adj.sum(dim=0).to_dense().unsqueeze(0)
reciprocal_degree = torch.nan_to_num(reciprocal_degree)
weighted_adj = non_weighted_adj * reciprocal_degree
cn_scores = non_weighted_adj @ weighted_adj
# ---
A = adj.to_dense(); deg = A.sum(0)
ra = A @ torch.diag(1 / deg) @ A           # textbook RA: sum_w A[a,w] A[w,v] / deg(w)
cn_over_deg_v = (A @ A) / deg.unsqueeze(0)  # CN(a,v) / deg(v)
print("HeaRT code:\n", cn_scores.to_dense())
print("textbook RA:\n", ra)
print("CN/deg(v):\n", cn_over_deg_v)
print("HeaRT == RA:", torch.allclose(cn_scores.to_dense(), ra),
      "| HeaRT == CN/deg(v):", torch.allclose(cn_scores.to_dense(), cn_over_deg_v))
