from __future__ import annotations

import json

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.configs.config import TrainConfig
from src.data.graph_faces import build_typed_path_face_groups
from src.data.wisp_tasks import build_graph_wisp_path_groups
from src.data.graph_classification_dataset import GraphClassificationFold
from src.models.h2snn_model import H2SNN
from src.models.anchored_solve import check_finite_loss
from src.trainers.stop_conditions import logged_stop_conditions
from src.utils.checkpointer import Checkpointer
from src.utils.logger import ExperimentLogger
from src.utils.misc import compute_graph_classification_metrics


class GraphClassificationTrainer:
    """Mini-batch trainer for graph classification.

    Unlike Trainer/LinkPredictionTrainer (one transductive graph, one forward pass
    per epoch), this iterates DataLoader batches of many small disjoint graphs per
    epoch. PyG's disjoint-union Batch means the shared encoder (encode()) needs no changes —
    a batch of many small graphs is one graph with several connected components to the
    matrix-free operators. Path cells are enumerated per batch (two-hop typed paths);
    only the readout needs the batch index to know graph boundaries.
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
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device(cfg.device)
        self._best_test_metrics: dict[str, float] | None = None
        self._best_epoch: int = -1

    def fit(
        self,
        fold: GraphClassificationFold,
        epoch_callback: "callable | None" = None,
    ) -> None:
        """Train; a stop condition of spec §6 (tau, log-weight range) is logged and re-raised."""
        with logged_stop_conditions(self.logger):
            self._fit(fold, epoch_callback)

    def _fit(
        self,
        fold: GraphClassificationFold,
        epoch_callback: "callable | None" = None,
    ) -> None:
        self.model.to(self.device)
        if getattr(fold, "fact_layer", None) is not None:
            self.logger.info("fact layer: %s", fold.fact_layer.summary())
        # Spec v1.15 §6 GC (HetSheaf's protocol): early stopping on the validation loss (patience
        # `early_stopping_patience`, strict improvement, as Lightning's EarlyStopping with
        # min_delta 0), and the reported test metrics are those of the final weights; the
        # validation Macro-F1 of those weights is what selects configurations. `best.pt` is still
        # written by `checkpoint_metric` (an artifact, not what is reported).
        bad_epochs = 0
        best_val_loss = float("inf")
        final_epoch = -1
        for epoch in range(self.cfg.epochs):
            train_metrics = self._run_epoch(fold.train_loader, epoch, train=True)
            val_metrics = self._run_epoch(fold.val_loader, epoch, train=False)
            test_metrics = self._run_epoch(fold.test_loader, epoch, train=False)

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
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                bad_epochs = 0
            else:
                bad_epochs += 1
            final_epoch = epoch
            # the record of the final weights (overwritten every epoch)
            self._best_test_metrics = {
                "epoch": epoch,
                "final_epoch": epoch,
                "best_checkpoint_epoch": self._best_epoch,
                "run_index": fold.run_index,
                "num_runs": fold.num_runs,
                "fold_index": fold.fold_index,
                "num_folds": fold.num_folds,
                "repetition_index": fold.repetition_index,
                "num_repetitions": fold.num_repetitions,
                "split_seed": fold.split_seed,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["loss"])
            elif self.scheduler is not None:
                self.scheduler.step()

            if epoch_callback is not None:
                epoch_callback(epoch, train_metrics, val_metrics, test_metrics)

            if epoch % self.cfg.log_every == 0 or epoch == self.cfg.epochs - 1:
                self.logger.info(
                    "Epoch %04d | val_auroc: %.4f | val_acc: %.4f | val_macro_f1: %.4f | test_auroc: %.4f | test_acc: %.4f | test_macro_f1: %.4f",
                    epoch,
                    val_metrics["auroc"],
                    val_metrics["acc"],
                    val_metrics["macro_f1"],
                    test_metrics["auroc"],
                    test_metrics["acc"],
                    test_metrics["macro_f1"],
                )

            if bad_epochs >= self.cfg.early_stopping_patience:
                self.logger.info(
                    "Early stopping at epoch %d after %d epochs without validation loss "
                    "improvement.",
                    epoch,
                    bad_epochs,
                )
                break

        if self._best_test_metrics is not None:
            self.logger.info(
                "Final epoch %d (final weights; run %d/%d, repetition %d/%d, fold %d/%d) | test_auroc: %.4f | test_acc: %.4f | test_macro_f1: %.4f | test_micro_f1: %.4f",
                final_epoch,
                fold.run_index,
                fold.num_runs,
                fold.repetition_index,
                fold.num_repetitions,
                fold.fold_index,
                fold.num_folds,
                self._best_test_metrics.get("test_auroc", 0.0),
                self._best_test_metrics.get("test_acc", 0.0),
                self._best_test_metrics.get("test_macro_f1", 0.0),
                self._best_test_metrics.get("test_micro_f1", 0.0),
            )
            if self.cfg.data.split_protocol in ("stratified_holdout", "hetsheaf_random"):
                out_path = self.cfg.exp_dir / f"best_test_metrics_run{fold.run_index}.json"
            else:
                out_path = self.cfg.exp_dir / f"best_test_metrics_fold{fold.fold_index}.json"
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(self._best_test_metrics, handle, indent=2)
            self.logger.log_summary(self._best_test_metrics, prefix="best")

    def _run_epoch(self, loader: DataLoader, epoch: int, train: bool) -> dict[str, float]:
        self.model.train(train)
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        loss_sum = 0.0
        num_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                logits = self.forward_batch(batch, seed=self.draw_seed(epoch, train),
                                            epoch=epoch if train else -1)
                labels = batch.y.view(-1)
                loss = self.criterion(logits, labels)

            if train:
                check_finite_loss(loss, epoch)  # spec v1.16 K5
                loss.backward()
                if self.cfg.optim.grad_clip > 0:  # spec v1.15 §6: no clipping by default
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.cfg.optim.grad_clip)
                self.optimizer.step()

            all_logits.append(logits.detach())
            all_labels.append(labels.detach())
            loss_sum += float(loss.item())
            num_batches += 1

        all_logits_t = torch.cat(all_logits, dim=0)
        all_labels_t = torch.cat(all_labels, dim=0)
        metrics = {
            "loss": loss_sum / max(1, num_batches),
            **compute_graph_classification_metrics(all_logits_t, all_labels_t),
        }
        if train:
            # Diffusion step and path-coupling strength after this epoch's updates.
            metrics.update(self.model.diffusion_scalars())
        return metrics

    def batch_path_groups(self, batch, seed: int):
        """Path cells of one batch: WISP (spec v1.15 J1; every node of each graph is an anchor;
        each graph drawn alone, so its cells do not depend on the other graphs of the batch), or,
        under `path_source: schemas`, the pinned pre-D7 DFS (`path_sampler: dfs`)."""
        if self.cfg.data.path_source == "wisp":
            graph_id = getattr(batch, "graph_id", None)
            if graph_id is None:
                raise ValueError("WISP graph-classification draws need each graph's `graph_id`.")
            return tuple(group.to(batch.edge_index.device) for group in
                         build_graph_wisp_path_groups(
                             edge_index=batch.edge_index,
                             edge_type=batch.edge_type,
                             node_type=batch.node_type,
                             batch=batch.batch,
                             graph_id=graph_id.view(-1),
                             self_inverse=self.model.self_inverse_relations,
                             walks_per_anchor=self.cfg.data.wisp_walks_per_anchor,
                             max_length=self.cfg.data.wisp_max_length,
                             draw_seed=seed,
                         ))
        if self.cfg.data.path_sampler == "dfs":
            return build_typed_path_face_groups(
                edge_index=batch.edge_index,
                edge_type=batch.edge_type,
                node_type=batch.node_type,
                batch=batch.batch,
                graph_id=getattr(batch, "graph_id", None),
                max_paths_per_anchor=self.cfg.data.face_sample_k,
                exclude_identity_paths=self.cfg.data.exclude_identity_metapaths,
                seed=seed,
            )
        raise ValueError(
            "graph classification path cells: path_source wisp, or path_source schemas with "
            "path_sampler dfs (baseline bundles); the length-2 walks were replaced by WISP (J1).")

    def draw_seed(self, epoch: int, train: bool) -> int:
        """Path-draw seed (plan §9.2): `seed + epoch` for training batches under
        `data.resample_paths`, else `seed`. Evaluation draws therefore repeat exactly across
        epochs; they depend on the evaluation batch composition, which is fixed because
        evaluation loaders do not shuffle (decision 13)."""
        if train and self.cfg.data.resample_paths:
            return self.cfg.seed + epoch
        return self.cfg.seed

    def forward_batch(self, batch, seed: int | None = None, epoch: int = -1) -> torch.Tensor:
        """Logits of one PyG batch: path cells, encoder, graph readout. `epoch` sets the anchored
        solver's fixed tolerance schedule (-1 = evaluation, the final tolerance)."""
        face_groups = self.batch_path_groups(batch, seed=self.cfg.seed if seed is None else seed)
        encoded = self.model.encode(
            x=batch.x,
            node_type=batch.node_type,
            edge_index=batch.edge_index,
            edge_type=batch.edge_type,
            face_groups=face_groups,
            num_nodes=batch.num_nodes,
            epoch=epoch,
        )
        return self.model.graph_readout(  # type: ignore[misc]
            encoded.representation, batch.batch, batch.num_graphs
        )


def build_graph_classification_trainer(
    model: H2SNN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None,
    cfg: TrainConfig,
    logger: ExperimentLogger,
    checkpointer: Checkpointer,
) -> GraphClassificationTrainer:
    return GraphClassificationTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        logger=logger,
        checkpointer=checkpointer,
    )
