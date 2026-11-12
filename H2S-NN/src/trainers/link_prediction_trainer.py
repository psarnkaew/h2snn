from __future__ import annotations

import json

import torch
import torch.nn as nn

from src.configs.config import TrainConfig
from src.data.link_pred_structures import PreparedLinkPredData
from src.models.h2snn_model import H2SNN
from src.utils.checkpointer import Checkpointer
from src.utils.logger import ExperimentLogger
from src.trainers.stop_conditions import logged_stop_conditions
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
                                        data.edge_type, data.node_type, data.node_slices,
                                        seed=self.cfg.seed + epoch)
        return replace(prepared_data,
                       face_groups=tuple(group.to(self.device) for group in groups))

    def fit(
        self,
        data: PreparedLinkPredData,
        epoch_callback: "callable | None" = None,
    ) -> None:
        """Train; a stop condition of spec §6 (tau, log-weight range) is logged and re-raised."""
        with logged_stop_conditions(self.logger):
            self._fit(data, epoch_callback)

    def _fit(
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
            test_metrics = self._eval_test_sets(prepared_data)

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
                    "Epoch %04d | val_auroc: %.4f | val_mrr: %.4f | test (2-hop, random) "
                    "auroc: %s, %s | mrr: %s, %s",
                    epoch,
                    val_metrics["auroc"],
                    val_metrics["mrr"],
                    test_metrics["2hop_auroc"],
                    test_metrics["random_auroc"],
                    test_metrics["2hop_mrr"],
                    test_metrics["random_mrr"],
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
                "Best epoch %d | test (2-hop, random) auroc: %s, %s | mrr: %s, %s",
                self._best_epoch,
                self._best_test_metrics.get("test_2hop_auroc"),
                self._best_test_metrics.get("test_random_auroc"),
                self._best_test_metrics.get("test_2hop_mrr"),
                self._best_test_metrics.get("test_random_mrr"),
            )
            out_path = self.cfg.exp_dir / "best_test_metrics.json"
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(self._best_test_metrics, handle, indent=2)
            self.logger.log_summary(self._best_test_metrics, prefix="best")

    def _train_negatives(self, data: PreparedLinkPredData,
                         epoch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """HGB's get_train_neg (spec v1.16 K3), the same function every baseline uses:
        one uniform tail of the tail type per training positive, known positives not excluded,
        seed run seed + epoch, tagged per target relation."""
        from src.data.hgb_lp_negatives import hgb_epoch_train_negatives

        return hgb_epoch_train_negatives(data, epoch)

    def _compute(
        self,
        data: PreparedLinkPredData,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        pos_edge_relation: torch.Tensor,
        neg_edge_relation: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
        # Message graph and path cells: `data.edge_index` / `data.face_groups`, built from
        # training positives only (src/data/link_prediction_dataset.py).
        encoded = self.model.encode(
            x=data.x,
            node_type=data.node_type,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            face_groups=data.face_groups,
            num_nodes=data.num_nodes,
            epoch=epoch,  # the anchored solver's fixed tolerance schedule; -1 = evaluation
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
        return total_loss, metrics, pos_scores.detach(), neg_scores.detach()

    def _train_epoch(self, data: PreparedLinkPredData, epoch: int) -> dict[str, float]:
        """One epoch = the training positives in shuffled batches of `data.link_batch_size`
        (8,192, HGB's LP default; spec v1.15 §6, J5), each with its share of the epoch's 1:1
        negatives; one optimizer step per batch, and for H2S-NN one solve on the whole message
        graph per step (the epoch's WISP draw is shared by its batches)."""
        self.model.train()
        neg_edge_index, neg_edge_relation = self._train_negatives(data, epoch)
        pos_edge_index, pos_edge_relation = data.train_pos_edge_index, data.train_pos_edge_relation
        batch = self.cfg.data.link_batch_size
        ratio = max(1, neg_edge_index.size(1) // max(1, pos_edge_index.size(1)))
        pos_order = torch.randperm(pos_edge_index.size(1), device=pos_edge_index.device)
        neg_order = torch.randperm(neg_edge_index.size(1), device=neg_edge_index.device)
        num_batches = max(1, -(-pos_edge_index.size(1) // batch))
        loss_sum, link_loss_sum, rows = 0.0, 0.0, 0
        pos_parts, neg_parts, pos_index_parts, neg_index_parts = [], [], [], []
        for step in range(num_batches):
            p = pos_order[step * batch:(step + 1) * batch]
            n = neg_order[step * batch * ratio:(step + 1) * batch * ratio]
            self.optimizer.zero_grad()
            loss, step_metrics, pos_scores, neg_scores = self._compute(
                data, pos_edge_index[:, p], neg_edge_index[:, n], pos_edge_relation[p],
                neg_edge_relation[n], epoch)
            loss.backward()
            if self.cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
            self.optimizer.step()
            size = int(p.numel() + n.numel())
            loss_sum += step_metrics["loss"] * size
            link_loss_sum += step_metrics["link_loss"] * size
            rows += size
            pos_parts.append(pos_scores)
            neg_parts.append(neg_scores)
            pos_index_parts.append(pos_edge_index[:, p])
            neg_index_parts.append(neg_edge_index[:, n])
        computed = compute_link_prediction_metrics(
            torch.cat(pos_parts), torch.cat(neg_parts),
            pos_edge_index=torch.cat(pos_index_parts, dim=1),
            neg_edge_index=torch.cat(neg_index_parts, dim=1))
        metrics = {"loss": loss_sum / max(1, rows), "link_loss": link_loss_sum / max(1, rows),
                   **{name: float(computed.get(name, 0.0))
                      for name in LINK_PREDICTION_METRIC_NAMES},
                   "optimizer_steps": float(num_batches)}
        # Diffusion step and path-coupling strength after this epoch's update.
        metrics.update(self.model.diffusion_scalars())
        if epoch % self.cfg.log_every == 0 and self.model.diagnostics_available:
            # first_order: eta * lambda_max of G Delta B (D10, power iteration, 30 steps: a lower
            # bound); filter_bank: lambda_max(A), sum_k ||W_k||_2 and the M-gain (D12).
            metrics.update(self.model.operator_diagnostics(
                data.x, data.node_type, data.edge_index, data.edge_type, data.face_groups,
                data.num_nodes,
            ))
            # Normalized energy per diffusion step, edge and path-cell parts (plan §12.6).
            metrics.update(self.model.energy_trace(
                data.x, data.node_type, data.edge_index, data.edge_type, data.face_groups,
                data.num_nodes,
            ))
        return metrics

    def _eval_test_sets(self, data: PreparedLinkPredData) -> dict[str, object]:
        """Test metrics on HGB's two test sets (spec v1.16 K1), one encode; a set without
        negatives (dblp_ap, imdb_ma 2-hop) gives None with its reason."""
        from src.utils.misc import lp_test_set_metrics

        self.model.eval()
        with torch.no_grad():
            encoded = self.model.encode(
                x=data.x, node_type=data.node_type, edge_index=data.edge_index,
                edge_type=data.edge_type, face_groups=data.face_groups,
                num_nodes=data.num_nodes, epoch=-1)

            def score(edge_index: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
                return self.model.link_readout(  # type: ignore[misc]
                    node_representation=encoded.representation,
                    edge_label_index=edge_index, edge_relation=relation)

            return lp_test_set_metrics(score, data)

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
            _, metrics, _, _ = self._compute(
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
