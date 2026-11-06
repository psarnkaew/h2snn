from __future__ import annotations

import json

import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling

from src.configs.config import TrainConfig
from src.data.link_pred_structures import PreparedLinkPredData
from src.models.h2snn_model import H2SNN
from src.utils.checkpointer import Checkpointer
from src.utils.logger import ExperimentLogger
from src.utils.misc import compute_link_prediction_metrics


LINK_PREDICTION_METRIC_NAMES = ("auroc", "aupr", "mrr")


class LinkPredictionTrainer:
    """Single-forward-pass-per-epoch trainer for a transductive link-prediction graph.

    A sibling of Trainer (not a subclass): the loop shape stays the same
    (one epoch = train/val/test forward passes over one graph) but the readout is
    invoked per edge split with a different edge_label_index each time, and loss/
    metrics are BCE + AUROC/AUPR/MRR instead of cross-entropy + accuracy/F1 - different
    enough from node classification's masked-logits path that sharing Trainer's
    _compute_metrics would need an awkward branch rather than a clean reuse.
    """

    def __init__(
        self,
        model: H2SNN,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | None,
        cfg: TrainConfig,
        logger: ExperimentLogger,
        checkpointer: Checkpointer,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.logger = logger
        self.checkpointer = checkpointer
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = torch.device(cfg.device)
        self._cached_train_neg: torch.Tensor | None = None
        self._cached_train_neg_relation: torch.Tensor | None = None
        self._best_test_metrics: dict[str, float] | None = None
        self._best_epoch: int = -1

    def training_draw(self, data: PreparedLinkPredData, prepared_data: PreparedLinkPredData,
                      epoch: int) -> PreparedLinkPredData:
        """Data for one training epoch: the evaluation draw (seed `cfg.seed`) or, with
        `data.resample_paths`, a fresh draw on the message graph with seed `cfg.seed +
        epoch` (plan §9.2), built on the CPU copy and moved to the device."""
        if not self.cfg.data.resample_paths:
            return prepared_data
        from dataclasses import replace

        from src.data.link_prediction_dataset import build_link_path_groups  # resampling only

        groups = build_link_path_groups(self.cfg, data.fact_layer, data.edge_index,
                                        data.edge_type, data.node_type,
                                        seed=self.cfg.seed + epoch)
        return replace(prepared_data,
                       face_groups=tuple(group.to(self.device) for group in groups))

    def fit(
        self,
        data: PreparedLinkPredData,
        epoch_callback: "callable | None" = None,
    ) -> None:
        self.model.to(self.device)
        prepared_data = data.to(self.device)
        bad_epochs = 0
        if getattr(data, "fact_layer", None) is not None:
            self.logger.info("fact layer: %s", data.fact_layer.summary())
        for epoch in range(self.cfg.epochs):
            train_metrics = self._train_epoch(self.training_draw(data, prepared_data, epoch),
                                              epoch)
            val_metrics = self._eval_epoch(prepared_data, "val", epoch)
            test_metrics = self._eval_epoch(prepared_data, "test", epoch)

            self.logger.log(epoch, train_metrics, val_metrics, test_metrics)
            self.checkpointer.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                tag="last",
                metrics=val_metrics,
                metadata={"test": test_metrics},
            )
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
                    "Epoch %04d | val_auroc: %.4f | val_mrr: %.4f | test_auroc: %.4f | test_mrr: %.4f",
                    epoch,
                    val_metrics["auroc"],
                    val_metrics["mrr"],
                    test_metrics["auroc"],
                    test_metrics["mrr"],
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
                "Best epoch %d | test_auroc: %.4f | test_mrr: %.4f",
                self._best_epoch,
                self._best_test_metrics.get("test_auroc", 0.0),
                self._best_test_metrics.get("test_mrr", 0.0),
            )
            out_path = self.cfg.exp_dir / "best_test_metrics.json"
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(self._best_test_metrics, handle, indent=2)
            self.logger.log_summary(self._best_test_metrics, prefix="best")

    def _sample_negatives(
        self, data: PreparedLinkPredData, pos_edge_index: torch.Tensor
    ) -> torch.Tensor:
        num_needed = pos_edge_index.size(1) * data.neg_sampling_ratio
        if data.train_neg_candidates is not None:
            pool_size = data.train_neg_candidates.size(1)
            idx = torch.randint(0, pool_size, (num_needed,), device=data.train_neg_candidates.device)
            return data.train_neg_candidates[:, idx]

        local_known_pos = torch.stack(
            [
                data.known_pos_edge_index[0] - data.src_node_offset,
                data.known_pos_edge_index[1] - data.dst_node_offset,
            ]
        )
        neg_local = negative_sampling(
            edge_index=local_known_pos,
            num_nodes=(data.num_src_nodes, data.num_dst_nodes),
            num_neg_samples=num_needed,
        )
        offset = neg_local.clone()
        offset[0] += data.src_node_offset
        offset[1] += data.dst_node_offset
        return offset

    def _train_negatives(self, data: PreparedLinkPredData) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples train negatives per pooled target relation, tagging each with
        the relation it's scored against — matches Simple-HGN's run_dist.py, which
        draws each relation's negatives from its own positive count rather than
        pooling all relations into one untyped negative sample. A no-op split for
        every dataset except Amazon (num_target_relations == 1 everywhere else)."""
        if data.neg_sampling_mode == "fixed" and self._cached_train_neg is not None:
            return self._cached_train_neg, self._cached_train_neg_relation  # type: ignore[return-value]

        neg_parts: list[torch.Tensor] = []
        relation_parts: list[torch.Tensor] = []
        for local_idx in range(data.num_target_relations):
            mask = data.train_pos_edge_relation == local_idx
            pos_subset = data.train_pos_edge_index[:, mask]
            if pos_subset.size(1) == 0:
                continue
            neg_subset = self._sample_negatives(data, pos_subset)
            neg_parts.append(neg_subset)
            relation_parts.append(
                torch.full(
                    (neg_subset.size(1),), local_idx, dtype=torch.long, device=neg_subset.device
                )
            )
        neg_edge_index = torch.cat(neg_parts, dim=1)
        neg_edge_relation = torch.cat(relation_parts, dim=0)

        if data.neg_sampling_mode == "fixed":
            self._cached_train_neg = neg_edge_index
            self._cached_train_neg_relation = neg_edge_relation
        return neg_edge_index, neg_edge_relation

    def _compute(
        self,
        data: PreparedLinkPredData,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        pos_edge_relation: torch.Tensor,
        neg_edge_relation: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Message graph and path cells: `data.edge_index` / `data.face_groups`, built from
        # training positives only (src/data/link_prediction_dataset.py).
        encoded = self.model.encode(
            x=data.x,
            node_type=data.node_type,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            face_groups=data.face_groups,
            num_nodes=data.num_nodes,
        )
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_label_relation = torch.cat([pos_edge_relation, neg_edge_relation], dim=0)
        scores = self.model.link_readout(  # type: ignore[misc]
            node_representation=encoded.representation,
            edge_label_index=edge_label_index,
            edge_relation=edge_label_relation,
        )
        pos_scores, neg_scores = scores.split([pos_edge_index.size(1), neg_edge_index.size(1)])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        link_loss = self.criterion(scores, labels)

        total_loss = link_loss

        computed_link_metrics = compute_link_prediction_metrics(
            pos_scores,
            neg_scores,
            pos_edge_index=pos_edge_index,
            neg_edge_index=neg_edge_index,
        )
        # Keep the per-split metric schema stable for ExperimentLogger and the
        # W&B epoch callback. In particular, an empty/degenerate split should
        # still create train/{metric}, val/{metric}, and test/{metric} keys.
        link_metrics = {
            name: float(computed_link_metrics.get(name, 0.0))
            for name in LINK_PREDICTION_METRIC_NAMES
        }
        metrics: dict[str, float] = {
            "loss": float(total_loss.item()),
            "link_loss": float(link_loss.item()),
            **link_metrics,
        }
        return total_loss, metrics

    def _train_epoch(self, data: PreparedLinkPredData, epoch: int) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        neg_edge_index, neg_edge_relation = self._train_negatives(data)
        loss, metrics = self._compute(
            data,
            data.train_pos_edge_index,
            neg_edge_index,
            data.train_pos_edge_relation,
            neg_edge_relation,
            epoch,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
        self.optimizer.step()
        # Diffusion step and path-coupling strength after this epoch's update.
        metrics.update(self.model.diffusion_scalars())
        return metrics

    def _eval_epoch(
        self,
        data: PreparedLinkPredData,
        split: str,
        epoch: int = -1,
    ) -> dict[str, float]:
        self.model.eval()
        pos_edge_index = {"val": data.val_pos_edge_index, "test": data.test_pos_edge_index}[split]
        neg_edge_index = {"val": data.val_neg_edge_index, "test": data.test_neg_edge_index}[split]
        pos_edge_relation = {
            "val": data.val_pos_edge_relation,
            "test": data.test_pos_edge_relation,
        }[split]
        neg_edge_relation = {
            "val": data.val_neg_edge_relation,
            "test": data.test_neg_edge_relation,
        }[split]
        with torch.no_grad():
            _, metrics = self._compute(
                data, pos_edge_index, neg_edge_index, pos_edge_relation, neg_edge_relation, -1
            )
        return metrics


def build_link_prediction_trainer(
    model: H2SNN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None,
    cfg: TrainConfig,
    logger: ExperimentLogger,
    checkpointer: Checkpointer,
) -> LinkPredictionTrainer:
    return LinkPredictionTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        logger=logger,
        checkpointer=checkpointer,
    )
