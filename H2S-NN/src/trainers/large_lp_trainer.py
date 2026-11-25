"""H2S-NN trainer of the two large LP tasks (card 2 D4b): `LinkEpochLoop` with `LinkScopes` on the bundle's message
graph (`src/data/large_lp_h2snn.py`), with the protocol of the baselines' large-LP loop
(`src/baselines/large_lp_training.py: fit_large_lp`): HGB's per-epoch tail negatives, `LP_POSITIVES_PER_STEP`
positives per step, validation on all validation links every epoch, the 2-hop and random test sets evaluated when
validation AUROC improves, `cfg.epochs` / `cfg.early_stopping_patience` (40 / 40 for LP).

The loop, the scopes, the draws and the model are unchanged; this class wires them to the experiment's logger,
checkpointer and the tuning launcher's `epoch_callback`.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import torch

from src.configs.config import TrainConfig, reject_full_batch_nc_options
from src.data.large_lp_h2snn import LargeLinkH2SData
from src.trainers.minibatch_lp import LinkEpochLoop, LinkScopes


def link_scopes(model: torch.nn.Module, data: LargeLinkH2SData, cfg: TrainConfig) -> LinkScopes:
    """Scopes on the **message graph** (`data.graph`; held-out pairs are not facts of it). `r` and `l_max` are the
    config's (D1)."""
    return LinkScopes(model, data.graph, cfg.data.wisp_walks_per_anchor, cfg.data.wisp_max_length)


class _EpochLoop(LinkEpochLoop):
    """`LinkEpochLoop` whose validation also reports the full metric dict, evaluates the test sets when validation
    improves, and calls the trainer's per-epoch hooks."""

    def __init__(self, trainer: "LargeLinkTrainer", data: LargeLinkH2SData, scopes: LinkScopes,
                 epoch_callback: Callable | None) -> None:
        cfg = trainer.cfg
        super().__init__(scopes, trainer.optimizer, cfg.seed, cfg.epochs, cfg.early_stopping_patience,
                         data.dst_range, trainer.logger)
        self.trainer, self.data, self.epoch_callback = trainer, data, epoch_callback
        self._best = float("-inf")
        self._loss = float("nan")

    def metrics(self, pos: torch.Tensor, neg: torch.Tensor) -> dict[str, float]:
        from src.utils.misc import compute_link_prediction_metrics

        scores = self.scopes.evaluate(torch.cat([pos, neg], dim=1), self.base_seed)
        p, n = scores.split([pos.size(1), neg.size(1)])
        return compute_link_prediction_metrics(p.float().cpu(), n.float().cpu())

    def train_epoch(self, order, train_pos: torch.Tensor, epoch: int) -> float:
        self._loss = super().train_epoch(order, train_pos, epoch)
        return self._loss

    def validate(self, val_pos: torch.Tensor, val_neg: torch.Tensor) -> float:
        epoch = len(self.history)
        val = self.metrics(val_pos, val_neg)
        test: dict[str, float] = {}
        if val["auroc"] > self._best:
            self._best = val["auroc"]
            d = self.data
            test = {**{f"2hop_{k}": v for k, v in self.metrics(d.test_pos, d.test_2hop).items()},
                    **{f"random_{k}": v for k, v in self.metrics(d.test_pos, d.test_random).items()}}
            self.trainer.record_best(epoch, val, test)
        train = {"loss": self._loss}
        self.trainer.logger.log(epoch, train, val, test)
        if self.epoch_callback is not None:
            self.epoch_callback(epoch, train, val, test)
        return float(val["auroc"])


class LargeLinkTrainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, cfg: TrainConfig, logger,
                 checkpointer) -> None:
        reject_full_batch_nc_options(cfg, "LargeLinkTrainer")  # card 23 C4-C6
        self.model, self.optimizer, self.cfg = model, optimizer, cfg
        self.logger, self.checkpointer = logger, checkpointer
        self.device = torch.device(cfg.device)
        self._best_test_metrics: dict[str, Any] | None = None
        self.loop: _EpochLoop | None = None

    def record_best(self, epoch: int, val: dict[str, float], test: dict[str, float]) -> None:
        self._best_test_metrics = {"epoch": epoch, "best_epoch": epoch, **{f"val_{k}": v for k, v in val.items()},
                                   **{f"test_{k}": v for k, v in test.items()}}
        self.checkpointer.save_if_best(model=self.model, optimizer=self.optimizer, epoch=epoch, metrics=val,
                                       metadata={"test": test})

    def fit(self, data: LargeLinkH2SData, epoch_callback: Callable | None = None) -> None:
        """Train; a stop condition of spec §6 is logged and re-raised (`LinkEpochLoop.fit`)."""
        self.model.to(self.device)
        self.loop = _EpochLoop(self, data, link_scopes(self.model, data, self.cfg), epoch_callback)
        self.loop.fit(data.train_pos, data.val_pos, data.val_neg)
        if self._best_test_metrics is not None:
            with (self.cfg.exp_dir / "best_test_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(self._best_test_metrics, handle, indent=2)
            self.logger.log_summary(self._best_test_metrics, prefix="best")


__all__ = ["LargeLinkTrainer", "link_scopes"]
