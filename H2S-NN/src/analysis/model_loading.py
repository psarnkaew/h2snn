from __future__ import annotations

from pathlib import Path

import yaml

from src.configs.config import TrainConfig
from src.data.dataset import build_dataset
from src.data.structures import PreparedH2SData
from src.models.h2snn_model import H2SNN, build_model
from src.utils.checkpointer import Checkpointer
from src.utils.discovery_artifacts import load_selector_manifest


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

    selector_manifest = (
        load_selector_manifest(exp_dir)
        if cfg.discovery.enabled and cfg.discovery.mode == "auto"
        else None
    )
    data = build_dataset(
        cfg,
        frozen_selector_manifest=selector_manifest,
    ).to(device)
    model = build_model(cfg, data).to(device)

    checkpointer = Checkpointer(
        exp_dir=exp_dir,
        metric=cfg.checkpoint_metric,
        mode=cfg.checkpoint_mode,
        protocol_version=cfg.discovery.protocol_version,
        selector_manifest=selector_manifest,
        require_selector_manifest=selector_manifest is not None,
    )
    checkpointer.load(model, tag=tag)
    model.eval()

    return model, data, cfg
