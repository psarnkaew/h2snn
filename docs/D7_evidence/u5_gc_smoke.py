"""U5: graph classification with tied + walk, one forward/backward per dataset (no training)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
from src.configs.config import TrainConfig
from src.data.graph_classification_dataset import build_graph_classification_dataset
from src.models.h2snn_model import H2SNN
from src.trainers.graph_classification_trainer import GraphClassificationTrainer

for name in ("mutag", "proteins"):
    cfg = TrainConfig.from_dict({"device": "cpu", "data": {"dataset_name": name, "fact_layer": "tied", "path_sampler": "walk"}})
    fold = build_graph_classification_dataset(cfg)
    model = H2SNN(cfg, fold.dims)
    trainer = GraphClassificationTrainer(model, None, None, cfg, None, None)
    batch = next(iter(fold.train_loader))
    groups = trainer.batch_path_groups(batch, seed=cfg.seed)
    logits = trainer.forward_batch(batch)
    logits.square().sum().backward()
    print(f"{name}: relations {fold.dims.num_relations}, self-inverse {fold.dims.self_inverse_relations}, "
          f"shared roles {model.shared_role_relations}, batch edges {batch.edge_index.size(1)}, "
          f"groups {len(groups)}, cells {sum(g.num_faces for g in groups)}, logits {tuple(logits.shape)}, finite {bool(torch.isfinite(logits).all())}")
