"""U5 smoke: tied + walk prepared data for ACM/DBLP/IMDB (cells, merged weights, time)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
from src.configs.config import TrainConfig
from src.data.dataset import build_dataset

for name in ("acm", "dblp", "imdb"):
    cfg = TrainConfig.from_dict({"data": {"dataset_name": name, "fact_layer": "tied", "path_sampler": "walk"}})
    t = time.perf_counter(); data = build_dataset(cfg); dt = time.perf_counter() - t
    print(f"{name}: facts {data.edge_index.size(1)} relations {list(data.relation_names)} build {dt:.1f}s")
    for g in data.face_groups:
        m = g.path_weight
        print(f"  {g.template_name} {g.node_type_sequence} cells={g.num_faces} L={g.length} "
              f"signs={sorted(set(g.face_edge_sign[0].tolist()))} m_f mean={float(m.mean()):.4f} "
              f"max={float(m.max()):.3f} sum={float(m.sum()):.1f}")
