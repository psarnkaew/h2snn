"""Mini-batch training and evaluation of H2S-NN for node classification (spec v1.21 §6 "Training
mode" and "Evaluation"; `docs/minibatch_design_v3B.md` §5-§6, §9; decisions MB1-MB3, MB6, N3, N5,
N6; card Part MB, MB4).

* Training order (N3): a seeded permutation of the training targets (seed from the base seed and
  the pass index), continued across epochs and reshuffled when exhausted.
* Epoch = `epoch_targets` training targets = `epoch_targets / B` optimizer steps (MB2); one step per
  batch; loss = mean over the batch's targets.
* Training draw: fresh per batch, seed from (base seed, epoch, batch index), `proposal: uniform`,
  chunked and merged per anchor (MB2), on the whole training graph; the batch is the disjoint union
  of the targets' scopes (MB1) and the unchanged model runs on it with per-scope PCG (MB3).
* Evaluation every `eval_period` epochs (and after the last): one draw per run from the base seed for
  every validation and test target, fixed for the run; the whole split in batches of
  EVAL_BATCH_TARGETS; the metric over the whole split (MB3, N5). The evaluation scopes are cut from
  the evaluation graph passed in (N6), else from the training graph.
* Stop conditions (tau, log-weight range per forward = per batch, non-finite g, T_max) raise and end
  the run (`logged_stop_conditions`); the solver's tolerance follows the epoch schedule.
"""

from __future__ import annotations

import json
from typing import Callable

import torch
import torch.nn as nn

from src.configs.config import EVAL_BATCH_TARGETS, TrainConfig
from src.data.scopes import build_scope_batch
from src.data.structures import PreparedH2SData
from src.data.wisp_sampler import AnchorCells, build_wisp_anchor_cells
from src.models.anchored_solve import check_finite_loss
from src.trainers.stop_conditions import logged_stop_conditions
from src.utils.checkpointer import Checkpointer
from src.utils.logger import ExperimentLogger
from src.utils.misc import compute_classification_metrics, compute_multilabel_classification_metrics

#: Seeds: pass p of the training order uses `base * SEED_EPOCHS + p`; the training draw of batch
#: b in epoch e uses `(base * SEED_EPOCHS + e) * SEED_BATCHES + b` (distinct for e < SEED_EPOCHS,
#: b < SEED_BATCHES).
SEED_EPOCHS = 100_003
SEED_BATCHES = 65_536


def pass_seed(base_seed: int, pass_index: int) -> int:
    if not 0 <= pass_index < SEED_EPOCHS:
        raise ValueError(f"pass index {pass_index} outside [0, {SEED_EPOCHS}).")
    return int(base_seed) * SEED_EPOCHS + int(pass_index)


def training_draw_seed(base_seed: int, epoch: int, batch_index: int) -> int:
    """The seed of the training draw of batch `batch_index` in epoch `epoch` (design §5)."""
    if not (0 <= epoch < SEED_EPOCHS and 0 <= batch_index < SEED_BATCHES):
        raise ValueError(f"(epoch, batch) = ({epoch}, {batch_index}) outside the seed range.")
    return (int(base_seed) * SEED_EPOCHS + int(epoch)) * SEED_BATCHES + int(batch_index)


def steps_per_epoch(epoch_targets: int, batch_size: int) -> int:
    if batch_size < 1 or epoch_targets % batch_size:
        raise ValueError(f"epoch_targets {epoch_targets} is not a multiple of B {batch_size}.")
    return epoch_targets // batch_size


class TrainingOrder:
    """Seeded permutation of the training targets, continued across epochs (N3)."""

    def __init__(self, targets: torch.Tensor, base_seed: int) -> None:
        if targets.numel() == 0:
            raise ValueError("no training targets.")
        self.targets = targets.long().cpu()
        self.base_seed = int(base_seed)
        self.pass_index = -1
        self.order = torch.empty(0, dtype=torch.long)
        self.position = 0

    def _reshuffle(self) -> None:
        self.pass_index += 1
        generator = torch.Generator().manual_seed(pass_seed(self.base_seed, self.pass_index))
        self.order = self.targets[torch.randperm(self.targets.numel(), generator=generator)]
        self.position = 0

    def next(self, batch_size: int) -> torch.Tensor:
        """The next `batch_size` targets of the continuing order (crossing a pass boundary)."""
        out = []
        needed = int(batch_size)
        while needed:
            if self.position >= self.order.numel():
                self._reshuffle()
            take = self.order[self.position:self.position + needed]
            self.position += take.numel()
            needed -= take.numel()
            out.append(take)
        return torch.cat(out)


class MiniBatchNodeTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, cfg: TrainConfig,
                 logger: ExperimentLogger, checkpointer: Checkpointer) -> None:
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.logger = logger
        self.checkpointer = checkpointer
        self.device = torch.device(cfg.device)
        self._best_test_metrics: dict[str, float] | None = None
        self._best_epoch = -1
        self._evaluation_cells: AnchorCells | None = None
        self.steps_taken = 0

    # ------------------------------------------------------------------ draws and batches
    def _cells(self, graph: PreparedH2SData, anchors: torch.Tensor, seed: int) -> AnchorCells:
        data = self.cfg.data
        return build_wisp_anchor_cells(
            graph.edge_index, graph.edge_type, graph.node_type, graph.fact_layer.self_inverse,
            torch.unique(anchors), data.wisp_walks_per_anchor, data.wisp_max_length, seed)

    def _batch(self, graph: PreparedH2SData, targets: torch.Tensor, cells: AnchorCells):
        batch = build_scope_batch(graph, targets, [cells.of(int(t)) for t in targets.tolist()])
        return batch, batch.data.to(self.device)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor, multilabel: bool) -> torch.Tensor:
        if multilabel:
            return nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        return nn.functional.cross_entropy(logits, labels)

    def _metrics(self, logits: torch.Tensor, labels: torch.Tensor,
                 multilabel: bool) -> dict[str, float]:
        loss = float(self._loss(logits, labels, multilabel))
        cls = (compute_multilabel_classification_metrics(logits, labels) if multilabel
               else compute_classification_metrics(logits, labels))
        return {"loss": loss, "acc": cls["acc"], "macro_f1": cls["macro_f1"],
                "micro_f1": cls["micro_f1"], "cls_loss": loss}

    # ------------------------------------------------------------------ training
    def train_epoch(self, graph: PreparedH2SData, order: TrainingOrder, epoch: int) -> dict:
        data = self.cfg.data
        self.model.train()
        losses = []
        for b in range(steps_per_epoch(data.epoch_targets, data.batch_size)):
            targets = order.next(data.batch_size)
            cells = self._cells(graph, targets, training_draw_seed(self.cfg.seed, epoch, b))
            _, batch = self._batch(graph, targets, cells)
            self.optimizer.zero_grad()
            logits = self.model(batch, epoch=epoch).logits
            loss = self._loss(logits, batch.target_y, batch.is_multilabel)  # mean over the batch
            check_finite_loss(loss, epoch)
            loss.backward()
            self.optimizer.step()
            self.steps_taken += 1
            losses.append(float(loss))
        return {"loss": sum(losses) / len(losses), "steps": float(len(losses))}

    # ------------------------------------------------------------------ evaluation
    def evaluation_cells(self, eval_graph: PreparedH2SData) -> AnchorCells:
        """One draw per run from the base seed for every validation and test target (fixed)."""
        if self._evaluation_cells is None:
            mask = eval_graph.target_val_mask | eval_graph.target_test_mask
            self._evaluation_cells = self._cells(eval_graph, eval_graph.target_indices[mask],
                                                 self.cfg.seed)
        return self._evaluation_cells

    def evaluate(self, eval_graph: PreparedH2SData, split_mask: torch.Tensor,
                 batch_size: int = EVAL_BATCH_TARGETS
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(target ids, logits, labels) of the whole split, in batches of `batch_size`."""
        targets = eval_graph.target_indices[split_mask]
        cells = self.evaluation_cells(eval_graph)
        self.model.eval()
        ids, logits, labels = [], [], []
        with torch.no_grad():
            for start in range(0, targets.numel(), batch_size):
                part = targets[start:start + batch_size]
                _, batch = self._batch(eval_graph, part, cells)
                logits.append(self.model(batch, epoch=-1).logits.cpu())
                labels.append(batch.target_y.cpu())
                ids.append(part)
        return torch.cat(ids), torch.cat(logits), torch.cat(labels)

    def evaluate_split(self, eval_graph: PreparedH2SData, split_mask: torch.Tensor) -> dict:
        _, logits, labels = self.evaluate(eval_graph, split_mask)
        return self._metrics(logits, labels, eval_graph.is_multilabel)

    # ------------------------------------------------------------------ loop
    def fit(self, graph: PreparedH2SData, eval_graph: PreparedH2SData | None = None,
            epoch_callback: Callable | None = None) -> None:
        self.model.to(self.device)
        eval_graph = graph if eval_graph is None else eval_graph
        order = TrainingOrder(graph.target_indices[graph.target_train_mask], self.cfg.seed)
        with logged_stop_conditions(self.logger):
            self._run(graph, eval_graph, order, epoch_callback)

    def _run(self, graph, eval_graph, order, epoch_callback) -> None:
        period = self.cfg.data.eval_period
        bad_epochs = 0
        last = self.cfg.epochs - 1
        for epoch in range(self.cfg.epochs):
            train_metrics = self.train_epoch(graph, order, epoch)
            if epoch % period and epoch != last:
                continue
            val_metrics = self.evaluate_split(eval_graph, eval_graph.target_val_mask)
            test_metrics = self.evaluate_split(eval_graph, eval_graph.target_test_mask)
            self.logger.log(epoch, train_metrics, val_metrics, test_metrics)
            self.checkpointer.save(model=self.model, optimizer=self.optimizer, epoch=epoch,
                                   tag="last", metrics=val_metrics, metadata={"test": test_metrics})
            if self.checkpointer.save_if_best(model=self.model, optimizer=self.optimizer,
                                              epoch=epoch, metrics=val_metrics,
                                              metadata={"test": test_metrics}):
                self._best_epoch = epoch
                bad_epochs = 0
                self._best_test_metrics = {
                    "epoch": epoch, "best_epoch": epoch,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()}}
            else:
                bad_epochs += period
            if epoch_callback is not None:
                epoch_callback(epoch, train_metrics, val_metrics, test_metrics)
            if bad_epochs >= self.cfg.early_stopping_patience:
                break
        if self._best_test_metrics is not None:
            with (self.cfg.exp_dir / "best_test_metrics.json").open("w", encoding="utf-8") as fh:
                json.dump(self._best_test_metrics, fh, indent=2)


__all__ = ["MiniBatchNodeTrainer", "SEED_BATCHES", "SEED_EPOCHS", "TrainingOrder", "pass_seed",
           "steps_per_epoch", "training_draw_seed"]
