"""D10 reality check (data): degree skew and label homophily on ACM, from the D7 walk cells.
Run from H2S-NN/: python ../docs/D10_evidence/reality_check.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
from src.configs.config import TrainConfig
from src.data.dataset import build_dataset

cfg = TrainConfig.from_dict({"data": {"dataset_name": "acm", "fact_layer": "tied", "path_sampler": "walk"}})
data = build_dataset(cfg)
names = list(data.relation_names)
print("ACM kept relations:", names)
paper0, paper1 = data.node_slices["paper"]
labels = torch.full((data.num_nodes,), -1, dtype=torch.long)
labels[data.target_indices] = data.target_y
for r, name in enumerate(names):
    m = data.edge_type == r
    ei = data.edge_index[:, m]
    # hub side = the non-paper endpoint (or either endpoint for paper-paper)
    deg = torch.bincount(ei.reshape(-1), minlength=data.num_nodes)
    other = [t for t in ("author", "subject", "term") if t in name]
    if other:
        s, e = data.node_slices[other[0]]
        d = deg[s:e].float()
        pd = deg[paper0:paper1].float()
        print(f"{name}: facts {int(m.sum())}; {other[0]} degree mean {d.mean():.1f} max {int(d.max())} p99 {d.kleene() if False else float(d.sort().values[int(0.99*len(d))]):.0f}; paper degree mean {pd.mean():.2f} max {int(pd.max())}; hub-to-paper fan-out ratio max/mean {float(d.max()/d.mean()):.1f}")
    else:
        d = deg[paper0:paper1].float()
        print(f"{name}: facts {int(m.sum())}; paper degree mean {d.mean():.2f} max {int(d.max())}")
for g in data.face_groups:
    a, b = g.face_node_index[:, 0], g.face_node_index[:, -1]
    ok = (labels[a] >= 0) & (labels[b] >= 0)
    same = (labels[a][ok] == labels[b][ok]).float()
    w = g.path_weight[ok]
    print(f"{g.template_name}: cells {g.num_faces}; endpoint label homophily (cells) {float(same.mean()):.3f}; weighted by m_f {float((same*w).sum()/w.sum()):.3f}; chance for 3 classes ~0.33")
