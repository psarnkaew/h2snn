from __future__ import annotations

import json
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from src.configs.config import EVAL_DRAW_SEED_STRIDE, TrainConfig, pack_setting
from src.data.structures import PreparedH2SData
from src.models.h2snn_model import H2SNNOutput
from src.models.anchored_solve import check_finite_loss
from src.trainers.stop_conditions import logged_stop_conditions
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
        self._previous_draw = None  # the last training draw (CPU), for the adaptive proposal
        # Part I (I6): seconds the main thread spent obtaining each epoch's training draw (the
        # CPU build when not prefetched, else the wait for the prefetch, plus the device copy).
        self.draw_wait_seconds: list[float] = []
        # Card 23 (C4-C6; every one off by default, and then this trainer is that of b0f13fb).
        # C4: the evaluation draws k = 1..K-1 (device), built once in `fit`; with their warm starts.
        self.eval_draws = int(pack_setting(cfg, ("data", "eval_draws")))
        self._extra_eval_draws: tuple[PreparedH2SData, ...] = ()
        self._eval_warm_starts: dict[int, tuple] = {}
        # C5: theta_ema, one tensor per parameter (None until the first optimizer step).
        self.ema_decay = float(pack_setting(cfg, ("optim", "ema_decay")))
        self._ema: list[torch.Tensor] | None = None
        # C6: the last w values of the checkpoint metric.
        self.checkpoint_smoothing = int(pack_setting(cfg, ("checkpoint_smoothing",)))
        self._recent_scores: deque[float] = deque(maxlen=self.checkpoint_smoothing)

    def _draw_proposal(self, data: PreparedH2SData):
        """The adaptive proposal for the next training draw, or None (q = p). Main thread only:
        it reads the previous draw and the last training forward's w_f (Part H3)."""
        if (self.cfg.data.wisp_proposal == "adaptive" and self._previous_draw is not None
                and getattr(self.model, "last_path_weights", ())):
            # q from the previous epoch's path mass w_f m_f (detached; variance only).
            from src.data.wisp_sampler import count_actions, proposal_from_mass

            return proposal_from_mass(
                self._previous_draw, self.model.last_path_weights, data.node_type,
                len(data.node_type_names), count_actions(data.edge_type, data.fact_layer.self_inverse),
                self.cfg.data.wisp_max_length, data.fact_layer.self_inverse)
        return None

    def _draw_groups(self, data: PreparedH2SData, epoch: int, proposal) -> tuple:
        """CPU cells of epoch `epoch`'s training draw (seed `cfg.seed + epoch`). Pure: reads the
        CPU data and the given proposal and changes no trainer state (it may run in the worker)."""
        from src.data.dataset import build_node_classification_paths  # resampling only

        return build_node_classification_paths(
            config=self.cfg,
            fact_layer=data.fact_layer,
            node_slices=data.node_slices,
            node_type=data.node_type,
            node_type_names=data.node_type_names,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            relation_tuples=data.relation_tuples,
            seed=self.cfg.seed + epoch,
            proposal=proposal,
        )

    def build_eval_draws(self, data: PreparedH2SData, prepared_data: PreparedH2SData) -> None:
        """Card 23 C4: evaluation draws k = 1..K-1 (seed `cfg.seed + EVAL_DRAW_SEED_STRIDE * k`, the
        target measure p), built once on the CPU copy and moved to the device; draw 0 is
        `prepared_data` (seed `cfg.seed`). Nothing is built with K = 1."""
        from src.data.dataset import build_node_classification_paths  # resampling only

        draws = []
        for k in range(1, self.eval_draws):
            groups = build_node_classification_paths(
                config=self.cfg,
                fact_layer=data.fact_layer,
                node_slices=data.node_slices,
                node_type=data.node_type,
                node_type_names=data.node_type_names,
                edge_index=data.edge_index,
                edge_type=data.edge_type,
                relation_tuples=data.relation_tuples,
                seed=self.cfg.seed + EVAL_DRAW_SEED_STRIDE * k,
                proposal=None,
            )
            draws.append(replace(prepared_data,
                                 face_groups=tuple(group.to(self.device) for group in groups)))
        self._extra_eval_draws = tuple(draws)

    def training_draw(self, data: PreparedH2SData, prepared_data: PreparedH2SData,
                      epoch: int, pending: "Future | None" = None) -> PreparedH2SData:
        """Data for one training epoch: the evaluation draw (seed `cfg.seed`, built with the
        data) or, with `data.resample_paths`, a fresh draw with seed `cfg.seed + epoch` built on
        the CPU copy `data` (now, or prefetched by the worker: `pending`) and moved to the
        device (plan §9.2)."""
        if not self.cfg.data.resample_paths:
            return prepared_data
        groups = (self._draw_groups(data, epoch, self._draw_proposal(data)) if pending is None
                  else pending.result())
        self._previous_draw = groups
        return replace(prepared_data, face_groups=tuple(group.to(self.device) for group in groups))

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
        if getattr(data, "fact_layer", None) is not None:
            self.logger.info("fact layer: %s", data.fact_layer.summary())
        if self.eval_draws > 1:  # card 23 C4
            self.build_eval_draws(data, prepared_data)
        # Part I (I6): with resampling and `data.overlap_sampling`, one worker thread builds epoch
        # e+1's CPU cells while epoch e trains. Seeds and proposal inputs are those of the
        # sequential path, so cells, m_f and logits are bit-identical (test). The adaptive q needs
        # epoch e's training forward, so that draw starts after the step (overlaps evaluation).
        overlap = self.cfg.data.resample_paths and self.cfg.data.overlap_sampling
        adaptive = overlap and self.cfg.data.wisp_proposal == "adaptive"
        executor = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="h2snn-draw")
                    if overlap else None)
        try:
            with logged_stop_conditions(self.logger):  # spec §6 stop conditions (J10)
                self._run_epochs(data, prepared_data, epoch_callback, executor, adaptive)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    def _run_epochs(self, data, prepared_data, epoch_callback, executor, adaptive) -> None:
        bad_epochs = 0
        pending: Future | None = None
        last = self.cfg.epochs - 1
        for epoch in range(self.cfg.epochs):
            start = time.perf_counter()
            train_data = self.training_draw(data, prepared_data, epoch, pending)
            self.draw_wait_seconds.append(time.perf_counter() - start)
            pending = None
            if executor is not None and not adaptive and epoch < last:
                pending = executor.submit(self._draw_groups, data, epoch + 1, None)
            train_metrics = self._train_epoch(train_data, epoch)
            if executor is not None and adaptive and epoch < last:
                pending = executor.submit(self._draw_groups, data, epoch + 1,
                                          self._draw_proposal(data))
            with self.evaluation_weights():  # card 23 C5: theta_ema when the EMA is on
                if self._extra_eval_draws:  # card 23 C4: the mean of the K draws' probabilities
                    val_metrics, test_metrics = self._eval_epoch_averaged(prepared_data)
                else:
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
            with self.evaluation_weights():  # card 23 C5: the best checkpoint holds theta_ema
                if self.checkpoint_smoothing > 1:  # card 23 C6: the trailing mean selects
                    is_best = self.checkpointer.save_if_best(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=val_metrics,
                        metadata={"test": test_metrics},
                        score=self.smoothed_score(val_metrics),
                    )
                else:
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
                if self.checkpoint_smoothing > 1:  # card 23 C6: the score that selected the epoch
                    self._best_test_metrics["val_smoothed_score"] = self._recent_mean()
            else:
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
        check_finite_loss(loss, epoch)  # spec v1.16 K5: a non-finite loss fails the run
        loss.backward()
        if self.cfg.optim.grad_clip > 0:  # spec v1.15 §6: no clipping by default
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
        if self.ema_decay > 0.0 and self._ema is None:  # card 23 C5: theta_ema = theta at step 0
            self._ema = [param.detach().clone() for param in self.model.parameters()]
        self.optimizer.step()
        if self._ema is not None:
            self._update_ema()
        # tau, the solver statistics and log w_f after this epoch's update (the v2 operator
        # diagnostics were removed by the legacy-removal card).
        metrics.update(self.model.diffusion_scalars())  # type: ignore[operator]
        return metrics

    # ------------------------------------------------------------------ card 23 C4-C6
    def _update_ema(self) -> None:
        """C5: theta_ema <- beta theta_ema + (1 - beta) theta, every parameter, after a step."""
        beta = self.ema_decay
        with torch.no_grad():
            for ema, param in zip(self._ema, self.model.parameters()):  # type: ignore[arg-type]
                ema.mul_(beta).add_(param.detach(), alpha=1.0 - beta)

    @contextmanager
    def evaluation_weights(self):
        """C5: inside, the model holds theta_ema (training weights restored on exit); a no-op with the
        EMA off (or before the first step)."""
        if self._ema is None:
            yield
            return
        params = list(self.model.parameters())
        with torch.no_grad():
            saved = [param.detach().clone() for param in params]
            for param, ema in zip(params, self._ema):
                param.copy_(ema)
        try:
            yield
        finally:
            with torch.no_grad():
                for param, value in zip(params, saved):
                    param.copy_(value)

    def _recent_mean(self) -> float:
        return sum(self._recent_scores) / len(self._recent_scores)

    def smoothed_score(self, val_metrics: dict[str, float]) -> float:
        """C6: the mean of the checkpoint metric over this epoch and the w - 1 before it (fewer at the
        start: epochs max(0, e - w + 1)..e)."""
        key = self.cfg.checkpoint_metric.removeprefix("val_")
        self._recent_scores.append(float(val_metrics[key]))
        return self._recent_mean()

    def averaged_eval_logits(self, prepared_data: PreparedH2SData) -> torch.Tensor:
        """C4: one evaluation forward per draw (draw 0 = `prepared_data`), each warm-started from its
        own last solution as the single evaluation draw is; the mean of the K probability vectors
        (softmax, or sigmoid for multi-label), returned as logits of that mean (log p; logit p for
        multi-label), so the metrics and the loss below read the mean."""
        model = self.model
        self.model.eval()
        probabilities = []
        with torch.no_grad():
            for k, draw in enumerate((prepared_data, *self._extra_eval_draws)):
                model._warm_start, model._warm_start_key = self._eval_warm_starts.get(k, (None, ()))
                logits = model(draw, epoch=-1).logits
                self._eval_warm_starts[k] = (model._warm_start, model._warm_start_key)
                probabilities.append(torch.sigmoid(logits) if prepared_data.is_multilabel
                                     else torch.softmax(logits, dim=-1))
            mean = torch.stack(probabilities).mean(dim=0)
            if prepared_data.is_multilabel:
                return torch.logit(mean, eps=1e-7)
            return mean.clamp_min(torch.finfo(mean.dtype).tiny).log()

    def _eval_epoch_averaged(self, prepared_data: PreparedH2SData
                             ) -> tuple[dict[str, float], dict[str, float]]:
        """C4: validation and test metrics of one epoch from the K-draw mean (one set of forwards)."""
        logits = self.averaged_eval_logits(prepared_data)
        output = _LogitsOnly(logits)
        with torch.no_grad():
            _, val_metrics = self._compute_metrics(output, prepared_data, prepared_data.target_val_mask, -1)
            _, test_metrics = self._compute_metrics(output, prepared_data, prepared_data.target_test_mask, -1)
        return val_metrics, test_metrics

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
        return total_loss, metrics


class _LogitsOnly:
    """Card 23 C4: the averaged output as `_compute_metrics` reads it (logits only)."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


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
