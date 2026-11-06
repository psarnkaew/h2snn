from __future__ import annotations

from pathlib import Path

import yaml

from src.configs.config import TrainConfig
from src.data.dataset import build_dataset
from src.data.structures import PreparedH2SData
from src.models.h2snn_model import H2SNN, build_model
from src.utils.checkpointer import Checkpointer


def load_trained_model(
    exp_dir: Path,
    device: str,
    tag: str = "best",
) -> tuple[H2SNN, PreparedH2SData, TrainConfig]:
    """Reconstructs a node-classification run from its saved `config.yaml` and
    restores a checkpoint's weights, for analysis/inference (not training)."""
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
