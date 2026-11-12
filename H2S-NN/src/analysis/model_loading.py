from __future__ import annotations

from pathlib import Path

import torch
import yaml

from src.configs.config import TrainConfig
from src.data.dataset import build_dataset
from src.data.structures import FaceGroup, PreparedH2SData
from src.models.h2snn_model import H2SNN, build_model
from src.utils.checkpointer import Checkpointer


def load_trained_model(
    exp_dir: Path,
    device: str,
    tag: str = "best",
) -> tuple[H2SNN, PreparedH2SData, TrainConfig]:
    """Reconstructs a node-classification run from its saved `config.yaml` and
    restores a checkpoint's weights, for analysis/inference (not training).
    The returned data carry the run's evaluation draw (seed `cfg.seed`, `build_dataset`)."""
    with (exp_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        cfg = TrainConfig.from_dict(yaml.safe_load(handle))
    cfg.device = device

    data = build_dataset(cfg).to(device)
    model = build_model(cfg, data).to(device)

    checkpointer = Checkpointer(
        exp_dir=exp_dir,
        metric=cfg.checkpoint_metric,
        mode=cfg.checkpoint_mode,
    )
    checkpointer.load(model, tag=tag)
    model.eval()

    return model, data, cfg


def nonempty_groups(face_groups: tuple[FaceGroup, ...]) -> tuple[FaceGroup, ...]:
    """The groups the model computes (`H2SNN._encode_anchored` skips empty groups)."""
    return tuple(group for group in face_groups if group.num_faces > 0)


def evaluation_cell_weights(model: H2SNN, data: PreparedH2SData) -> tuple[torch.Tensor, ...]:
    """`w_f = exp(g(xi_f))` of every cell of `data.face_groups` (one tensor per non-empty group,
    in group order), computed by the model's own code (`H2SNN._cell_weights`) in eval mode without
    gradients. Evaluation only: the model's training flag is restored."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            h0 = model.encode_nodes(data.x, data.node_type)
            geometry = model.geometry_inputs(h0, data.edge_index, data.edge_type)
            _, weights, _ = model._cell_weights(h0, geometry, nonempty_groups(data.face_groups))
        return tuple(weights)
    finally:
        model.train(was_training)
