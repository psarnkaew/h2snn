from __future__ import annotations

import json
from collections import defaultdict

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.configs.config import TrainConfig
from src.data.graph_faces import build_typed_path_face_groups
from src.data.graph_classification_dataset import GraphClassificationFold
from src.data.schema_structures import DiscoveryMetadata, make_discovery_metadata
from src.models.h2snn_model import H2SNN, H2SNNOutput
from src.models.s2pool_h2 import S2PoolH2, GalerkinSheafPool, SheafSpectralPool
from src.utils.checkpointer import Checkpointer
from src.utils.logger import ExperimentLogger
from src.utils.misc import compute_graph_classification_metrics
from src.trainers.ampss_metrics import build_ampss_metrics


class GraphClassificationTrainer:
    """Mini-batch trainer for graph classification.

    Unlike Trainer/LinkPredictionTrainer (one transductive graph, one forward pass
    per epoch), this iterates DataLoader batches of many small disjoint graphs per
    epoch. PyG's disjoint-union Batch means the shared sheaf-diffusion backbone
    (encode()) needs no changes at all — a batch of many small graphs is
    indistinguishable from one big graph with several connected components to the
    matrix-free delta0/delta1/L0/L1 operators. Only the final pooling step needs the
    batch index to know graph boundaries.
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
        self._discovery_schema_names: tuple[str, ...] = tuple()
        self._discovery_template: DiscoveryMetadata | None = None

    def fit(
        self,
        fold: GraphClassificationFold,
        epoch_callback: "callable | None" = None,
    ) -> None:
        self.model.to(self.device)
        self._discovery_schema_names = fold.discovery_schema_names
        self._discovery_template = fold.discovery.to(self.device) if fold.discovery is not None else None
        bad_epochs = 0
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
                    "Early stopping at epoch %d after %d epochs without validation %s improvement.",
                    epoch,
                    bad_epochs,
                    self.cfg.checkpoint_metric,
                )
                break

        if self._best_test_metrics is not None:
            self.logger.info(
                "Best epoch %d (run %d/%d, repetition %d/%d, fold %d/%d) | test_auroc: %.4f | test_acc: %.4f | test_macro_f1: %.4f | test_micro_f1: %.4f",
                self._best_epoch,
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
            if self.cfg.data.split_protocol == "stratified_holdout":
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
        discovery_sums: dict[str, float] = defaultdict(float)
        discovery_batches = 0
        num_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                discovery_enabled = (
                    self._discovery_template is not None
                    and bool(self._discovery_schema_names)
                )
                face_groups, closure_edge_index = build_typed_path_face_groups(
                    edge_index=batch.edge_index,
                    edge_type=batch.edge_type,
                    node_type=batch.node_type,
                    batch=batch.batch,
                    graph_id=getattr(batch, "graph_id", None),
                    max_paths_per_anchor=(
                        self.cfg.discovery.max_instances_per_anchor
                        if discovery_enabled
                        else self.cfg.data.face_sample_k
                    ),
                    exclude_identity_paths=(
                        self.cfg.discovery.exclude_identity_metapaths
                        if discovery_enabled
                        else self.cfg.data.exclude_identity_metapaths
                    ),
                    seed=(
                        self.cfg.discovery.materialization_seed
                        if discovery_enabled
                        else self.cfg.seed
                    ),
                    min_path_length=(
                        self.cfg.discovery.min_schema_length if discovery_enabled else 2
                    ),
                    max_path_length=(
                        self.cfg.discovery.max_schema_length if discovery_enabled else 2
                    ),
                    selected_template_names=(
                        self._discovery_schema_names if discovery_enabled else None
                    ),
                    max_paths_per_schema=(
                        self.cfg.discovery.max_instances_per_schema
                        if discovery_enabled
                        else None
                    ),
                    max_total_paths=(
                        self.cfg.discovery.max_total_instances
                        if discovery_enabled
                        else None
                    ),
                    max_total_incidences=(
                        self.cfg.discovery.max_total_incidences
                        if discovery_enabled
                        else None
                    ),
                    allow_repeated_node_types=(
                        self.cfg.discovery.allow_repeated_node_types
                        if discovery_enabled
                        else True
                    ),
                )
                face_groups, closure_edge_index, discovery = self._batch_discovery(
                    face_groups,
                    batch.edge_index,
                    batch.batch,
                    batch.num_graphs,
                )
                encoded = self.model.encode(
                    x=batch.x,
                    node_type=batch.node_type,
                    edge_index=batch.edge_index,
                    edge_type=batch.edge_type,
                    face_groups=face_groups,
                    closure_edge_index=closure_edge_index,
                    num_nodes=batch.num_nodes,
                    num_edges=batch.edge_index.size(1),
                    epoch=epoch if train else -1,
                    discovery=discovery,
                )
                if isinstance(self.model.graph_pooling, (GalerkinSheafPool, SheafSpectralPool)):
                    logits = self.model.graph_pooling(
                        encoded,
                        encoded.edge_index,
                        batch.node_type,
                        batch.batch,
                        batch.num_graphs,
                        face_groups,
                    )
                elif isinstance(self.model.graph_pooling, S2PoolH2):
                    logits = self.model.graph_pooling.run(
                        encoded=encoded,
                        face_groups=face_groups,
                        edge_index=batch.edge_index,
                        edge_type=batch.edge_type,
                        node_type=batch.node_type,
                        batch=batch.batch,
                        num_graphs=batch.num_graphs,
                        encode_from_stalk=self.model.encode_from_stalk,
                        build_faces=self._build_coarse_faces,
                    )
                else:
                    logits = self.model.graph_pooling(  # type: ignore[misc]
                        x_star=encoded.x_star,
                        h0=encoded.h0,
                        s_star=encoded.s_star,
                        delta0_x_star=encoded.delta0_x_star,
                        delta0_h0=encoded.delta0_h0,
                        edge_index=encoded.edge_index,
                        F_src=encoded.F_src,
                        F_tgt=encoded.F_tgt,
                        batch=batch.batch,
                        num_graphs=batch.num_graphs,
                    )
                labels = batch.y.view(-1)
                cls_loss = self.criterion(logits, labels)
                output_for_metrics = H2SNNOutput(
                    logits=logits,
                    delta0_x_star=encoded.delta0_x_star,
                    raw_face_curl=encoded.raw_face_curl,
                    delta1_delta0_h0=encoded.delta1_delta0_h0,
                    delta0_h0=encoded.delta0_h0,
                    x_star=encoded.x_star,
                    s_star=encoded.s_star,
                    instance_weight_means=getattr(
                        encoded,
                        "instance_weight_means",
                        None,
                    ),
                    instance_weight_stds=getattr(
                        encoded,
                        "instance_weight_stds",
                        None,
                    ),
                    instance_weight_nonempty_mask=getattr(
                        encoded,
                        "instance_weight_nonempty_mask",
                        None,
                    ),
                    operator_ramp=encoded.operator_ramp,
                )
                total_loss = cls_loss

            if train:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
                self.optimizer.step()

            all_logits.append(logits.detach())
            all_labels.append(labels.detach())
            loss_sum += float(total_loss.item())
            if discovery is not None:
                batch_discovery_metrics = build_ampss_metrics(
                    output_for_metrics,
                    discovery,
                    face_groups,
                    self.cfg,
                    schema_names=self._discovery_schema_names,
                )
                for key, value in batch_discovery_metrics.items():
                    discovery_sums[key] += value
                discovery_batches += 1
            num_batches += 1

        all_logits_t = torch.cat(all_logits, dim=0)
        all_labels_t = torch.cat(all_labels, dim=0)
        cls_metrics = compute_graph_classification_metrics(all_logits_t, all_labels_t)

        metrics = {
            "loss": loss_sum / max(1, num_batches),
            **cls_metrics,
        }
        if discovery_batches > 0:
            metrics.update(
                {key: value / discovery_batches for key, value in discovery_sums.items()}
            )
        return metrics

    def _build_coarse_faces(self, edge_index, edge_type, node_type, batch):
        """Re-enumerate typed 2-hop faces on a THS-Pool coarse graph (discovery-off).

        Returns (face_groups, closure_edge_index) — the callback used by S2PoolH2.run at each
        coarser resolution, mirroring the level-0 face build.
        """
        face_groups, _ = build_typed_path_face_groups(
            edge_index=edge_index,
            edge_type=edge_type,
            node_type=node_type,
            batch=batch,
            graph_id=None,
            max_paths_per_anchor=self.cfg.data.face_sample_k,
            exclude_identity_paths=self.cfg.data.exclude_identity_metapaths,
            seed=self.cfg.seed,
            min_path_length=2,
            max_path_length=2,
            selected_template_names=None,
            max_paths_per_schema=None,
            max_total_paths=None,
        )
        if face_groups:
            closure = torch.cat(
                [g.closure_edge_index for g in face_groups], dim=0
            ).t().contiguous()
        else:
            closure = edge_index.new_empty((2, 0))
        return face_groups, closure

    def _batch_discovery(
        self,
        face_groups,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        num_graphs: int = 0,
    ) -> tuple[tuple, torch.Tensor, DiscoveryMetadata | None]:
        if self._discovery_template is None or not self._discovery_schema_names:
            if face_groups:
                closure_edge_index = torch.cat(
                    [group.closure_edge_index for group in face_groups],
                    dim=0,
                ).t().contiguous()
            else:
                closure_edge_index = edge_index.new_empty((2, 0))
            return face_groups, closure_edge_index, None

        schema_id_by_name = {
            name: idx for idx, name in enumerate(self._discovery_schema_names)
        }
        kept_groups = []
        schema_ids = []
        closure_parts = []
        coverage_sets: list[set[int]] = [
            set() for _ in range(self._discovery_template.num_schemas)
        ]
        for group in face_groups:
            schema_id = schema_id_by_name.get(group.template_name)
            if schema_id is None:
                continue
            kept_groups.append(group)
            schema_ids.append(schema_id)
            closure_parts.append(group.closure_edge_index)
            if batch is not None:
                graph_units = batch[group.face_node_index[:, 0]].unique().tolist()
                coverage_sets[schema_id].update(int(value) for value in graph_units)

        if closure_parts:
            closure_edge_index = torch.cat(closure_parts, dim=0).t().contiguous()
        else:
            closure_edge_index = edge_index.new_empty((2, 0))

        schema_id_tensor = torch.tensor(
            schema_ids,
            dtype=torch.long,
            device=edge_index.device,
        )
        discovery = make_discovery_metadata(
            num_schemas=self._discovery_template.num_schemas,
            face_groups=kept_groups,
            schema_ids=schema_id_tensor,
            schema_incidence_mass=self._discovery_template.schema_incidence_mass,
            coverage_sets=coverage_sets,
            num_coverage_units=num_graphs,
        )
        return tuple(kept_groups), closure_edge_index, discovery

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
