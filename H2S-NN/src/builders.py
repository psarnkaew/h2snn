from __future__ import annotations

import torch

from src.configs.config import TrainConfig
from src.data.dataset import build_dataset
from src.data.graph_classification_dataset import (
    GraphClassificationFold,
    build_graph_classification_dataset,
)
from src.data.link_pred_structures import PreparedLinkPredData
from src.data.link_prediction_dataset import build_link_prediction_dataset
from src.data.structures import PreparedH2SData
from src.models.h2snn_model import (
    H2SNN,
    build_graph_classification_model,
    build_link_prediction_model,
    build_model,
)
from src.trainers.graph_classification_trainer import (
    GraphClassificationTrainer,
    build_graph_classification_trainer,
)
from src.trainers.link_prediction_trainer import LinkPredictionTrainer, build_link_prediction_trainer
from src.trainers.trainer import Trainer, build_trainer
from src.utils.checkpointer import Checkpointer
from src.utils.logger import ExperimentLogger
from src.utils.misc import dataclass_to_dict


def build_logger(cfg: TrainConfig) -> ExperimentLogger:
    return ExperimentLogger(
        cfg.exp_dir,
        use_wandb=cfg.use_wandb,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        wandb_mode=cfg.wandb_mode,
        wandb_group=cfg.wandb_group,
        wandb_tags=cfg.wandb_tags,
        wandb_name=cfg.exp_name,
        wandb_log_artifacts=cfg.wandb_log_artifacts,
        config=dataclass_to_dict(cfg),
    )


def build_parameter_groups(model: H2SNN, cfg: TrainConfig) -> list[dict[str, object]]:
    """Two groups: the diffusion scalars `theta_eta`, `theta_gamma` and the coupling-gate heads
    (lr scaled by `optim.scalar_lr_scale`, no weight decay) and every other parameter."""
    scalars = list(model.scalar_parameters())
    scalar_ids = {id(parameter) for parameter in scalars}
    others = [p for p in model.parameters() if id(p) not in scalar_ids]
    return [
        {"params": others, "lr": cfg.optim.lr, "weight_decay": cfg.optim.weight_decay},
        {"params": scalars, "lr": cfg.optim.lr * cfg.optim.scalar_lr_scale, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: H2SNN,
    cfg: TrainConfig,
) -> torch.optim.Optimizer:
    groups = build_parameter_groups(model, cfg)
    if cfg.optim.name == "adam":
        return torch.optim.Adam(groups, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    if cfg.optim.name == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    raise ValueError(f"Unsupported optimizer '{cfg.optim.name}'.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if cfg.optim.scheduler == "none":
        return None
    if cfg.optim.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.optim.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 4), gamma=0.5)
    if cfg.optim.scheduler == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(5, cfg.epochs // 20),
        )
    raise ValueError(f"Unsupported scheduler '{cfg.optim.scheduler}'.")


def _checkpointer(cfg: TrainConfig) -> Checkpointer:
    return Checkpointer(exp_dir=cfg.exp_dir, metric=cfg.checkpoint_metric, mode=cfg.checkpoint_mode)


def build_experiment_components(
    cfg: TrainConfig,
) -> tuple[
    PreparedH2SData | PreparedLinkPredData | GraphClassificationFold,
    H2SNN,
    torch.optim.Optimizer,
    object | None,
    ExperimentLogger,
    Checkpointer,
    Trainer | LinkPredictionTrainer | GraphClassificationTrainer,
]:
    if cfg.task_type == "graph_classification":
        fold = build_graph_classification_dataset(cfg)
        model = build_graph_classification_model(cfg, fold.dims)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)
        logger = build_logger(cfg)
        checkpointer = _checkpointer(cfg)
        trainer = build_graph_classification_trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            logger=logger,
            checkpointer=checkpointer,
        )
        return fold, model, optimizer, scheduler, logger, checkpointer, trainer

    if cfg.task_type == "link_prediction":
        data = build_link_prediction_dataset(cfg)
        model = build_link_prediction_model(cfg, data)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)
        logger = build_logger(cfg)
        checkpointer = _checkpointer(cfg)
        trainer = build_link_prediction_trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            logger=logger,
            checkpointer=checkpointer,
        )
        return data, model, optimizer, scheduler, logger, checkpointer, trainer

    data = build_dataset(cfg)
    model = build_model(cfg, data)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    logger = build_logger(cfg)
    checkpointer = _checkpointer(cfg)
    if cfg.data.training_mode == "minibatch":  # Part MB (MB4): H2GB NC, or the N4 analysis flag
        from src.trainers.minibatch_trainer import MiniBatchNodeTrainer

        minibatch = MiniBatchNodeTrainer(model, optimizer, cfg, logger, checkpointer)
        return data, model, optimizer, scheduler, logger, checkpointer, minibatch  # type: ignore[return-value]
    trainer = build_trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        logger=logger,
        checkpointer=checkpointer,
    )
    return data, model, optimizer, scheduler, logger, checkpointer, trainer
