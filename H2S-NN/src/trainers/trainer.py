from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from src.configs.config import TrainConfig
from src.data.structures import PreparedH2SData
from src.trainers.ampss_metrics import build_ampss_metrics
from src.models.h2snn_model import H2SNNOutput
from src.utils.checkpointer import Checkpointer
from src.utils.logger import ExperimentLogger
from src.utils.misc import (
    compute_classification_metrics,
    compute_multilabel_classification_metrics,
)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | None,
        criterion: nn.Module,
        cfg: TrainConfig,
        logger: ExperimentLogger,
        checkpointer: Checkpointer,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.cfg = cfg
        self.logger = logger
        self.checkpointer = checkpointer
        self.device = torch.device(cfg.device)
        self._best_test_metrics: dict[str, float] | None = None
        self._best_epoch: int = -1

    def fit(
        self,
        data: PreparedH2SData,
        epoch_callback: Callable[
            [int, dict[str, float], dict[str, float], dict[str, float]], None
        ]
        | None = None,
    ) -> None:
        self.model.to(self.device)
        prepared_data = data.to(self.device)
        bad_epochs = 0
        for epoch in range(self.cfg.epochs):
            train_metrics = self._train_epoch(prepared_data, epoch)
            val_metrics = self._eval_epoch(prepared_data, prepared_data.target_val_mask, epoch)
            test_metrics = self._eval_epoch(prepared_data, prepared_data.target_test_mask, epoch)

            self.logger.log(epoch, train_metrics, val_metrics, test_metrics)
            self.checkpointer.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                tag="last",
                metrics=val_metrics,
                metadata={"test": test_metrics},
            )
            is_best = False
            checkpoint_eligible = self.cfg.is_full_operator_checkpoint_eligible(epoch)
            if checkpoint_eligible:
                is_best = self.checkpointer.save_if_best(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    metadata={"test": test_metrics},
                )
            if is_best:
                self._best_epoch = epoch
                bad_epochs = 0
                self._best_test_metrics = {
                    "epoch": epoch,
                    "best_epoch": epoch,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
            elif checkpoint_eligible:
                bad_epochs += 1

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["loss"])
            elif self.scheduler is not None:
                self.scheduler.step()

            if epoch_callback is not None:
                epoch_callback(epoch, train_metrics, val_metrics, test_metrics)

            if epoch % self.cfg.log_every == 0 or epoch == self.cfg.epochs - 1:
                self.logger.info(
                    "Checkpoint metric %s: %.4f",
                    self.cfg.checkpoint_metric,
                    val_metrics[self.cfg.checkpoint_metric.removeprefix("val_")],
                )

            if bad_epochs >= self.cfg.early_stopping_patience:
                self.logger.info(
                    "Early stopping at epoch %d after %d epochs without validation %s improvement.",
                    epoch,
                    bad_epochs,
                    self.cfg.checkpoint_metric,
                )
                break

        if self._best_test_metrics is not None:
            self.logger.info(
                "Best epoch %d | test_macro_f1: %.4f | test_micro_f1: %.4f | test_acc: %.4f",
                self._best_epoch,
                self._best_test_metrics.get("test_macro_f1", 0.0),
                self._best_test_metrics.get("test_micro_f1", 0.0),
                self._best_test_metrics.get("test_acc", 0.0),
            )
            out_path = self.cfg.exp_dir / "best_test_metrics.json"
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(self._best_test_metrics, handle, indent=2)
            self.logger.log_summary(self._best_test_metrics, prefix="best")

    def _train_epoch(self, data: PreparedH2SData, epoch: int) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(data, epoch=epoch)
        loss, metrics = self._compute_metrics(output, data, data.target_train_mask, epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
        self.optimizer.step()
        return metrics

    def _eval_epoch(
        self,
        data: PreparedH2SData,
        mask: torch.Tensor,
        epoch: int = -1,
    ) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            output = self.model(data, epoch=-1)
            _, metrics = self._compute_metrics(output, data, mask, -1)
        return metrics

    def _compute_metrics(
        self,
        output: H2SNNOutput,
        data: PreparedH2SData,
        target_mask: torch.Tensor,
        epoch: int = -1,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = output.logits[target_mask]
        labels = data.target_y[target_mask]
        if data.is_multilabel:
            classification_loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels.float(),
            )
        else:
            classification_loss = self.criterion(logits, labels)

        total_loss = classification_loss
        if data.is_multilabel:
            cls_metrics = compute_multilabel_classification_metrics(logits, labels)
        else:
            cls_metrics = compute_classification_metrics(logits, labels)
        metrics: dict[str, float] = {
            "loss": float(total_loss.item()),
            "acc": cls_metrics["acc"],
            "macro_f1": cls_metrics["macro_f1"],
            "micro_f1": cls_metrics["micro_f1"],
            "cls_loss": float(classification_loss.item()),
        }
        if data.discovery is not None:
            metrics.update(
                build_ampss_metrics(
                    output,
                    data.discovery,
                    data.face_groups,
                    self.cfg,
                )
            )
        return total_loss, metrics


def build_trainer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None,
    cfg: TrainConfig,
    logger: ExperimentLogger,
    checkpointer: Checkpointer,
) -> Trainer:
    criterion = nn.CrossEntropyLoss()
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        cfg=cfg,
        logger=logger,
        checkpointer=checkpointer,
    )
