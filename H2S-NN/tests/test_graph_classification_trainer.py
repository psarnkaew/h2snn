from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


class GraphClassificationTrainerTests(unittest.TestCase):
    def test_sheafcov_task_loss_reaches_amp_ss_instance_weights(self) -> None:
        torch = self._require_torch()
        from src.configs.config import TrainConfig
        from src.data.schema_structures import DiscoveryMetadata
        from src.data.structures import FaceGroup
        from src.models.h2snn_model import H2SNN, ModelDims

        face_group = FaceGroup(
            template_name="N0_R0_N1_R1_N0",
            node_type_sequence=("node_type_0", "node_type_1", "node_type_0"),
            face_node_index=torch.tensor([[0, 1, 2]]),
            face_edge_index=torch.tensor([[0, 1]]),
            relation_index=torch.tensor([[0, 1]]),
            face_edge_sign=torch.ones(1, 2),
            closure_edge_index=torch.tensor([[2, 0]]),
        )
        discovery = DiscoveryMetadata(
            num_schemas=1,
            schema_ids=torch.tensor([0]),
            schema_incidence_mass=torch.tensor([3.0]),
            coverage_schema_index=torch.tensor([0]),
            coverage_unit_index=torch.tensor([0]),
            num_coverage_units=1,
        )
        cfg = TrainConfig.from_dict(
            {
                "data": {
                    "dataset_name": "mutag",
                    "pooling_type": "sheafcov",
                },
                "model": {
                    "d": 4,
                    "d_r": 2,
                    "d_h": 8,
                    "K": 1,
                    "classifier_hidden_dim": 8,
                },
                "discovery": {
                    "enabled": True,
                    "mode": "auto",
                    "max_total_instances": 10,
                    "max_total_incidences": 30,
                    "require_target_start": False,
                    "require_target_end": False,
                },
                "device": "cpu",
            }
        )
        model = H2SNN(
            cfg,
            ModelDims(
                in_channels=3,
                num_node_types=2,
                num_relations=2,
                num_classes=2,
                discovery=discovery,
                num_nodes=3,
                face_lengths=(2,),
            ),
        )
        encoded = model.encode(
            x=torch.randn(3, 3),
            node_type=torch.tensor([0, 1, 0]),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            edge_type=torch.tensor([0, 1]),
            face_groups=(face_group,),
            closure_edge_index=torch.tensor([[2], [0]]),
            num_nodes=3,
            num_edges=2,
            epoch=-1,
            discovery=discovery,
        )
        logits = model.graph_pooling(  # type: ignore[misc]
            x_star=encoded.x_star,
            h0=encoded.h0,
            s_star=encoded.s_star,
            delta0_x_star=encoded.delta0_x_star,
            delta0_h0=encoded.delta0_h0,
            edge_index=encoded.edge_index,
            F_src=encoded.F_src,
            F_tgt=encoded.F_tgt,
            batch=torch.zeros(3, dtype=torch.long),
            num_graphs=1,
        )
        logits.square().sum().backward()
        assert model.instance_weight_network is not None
        weight_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.instance_weight_network.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(weight_grad, 0.0)

    def _require_torch(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"torch is required for this test: {exc}")
        return torch

    def _require_torch_geometric(self):
        try:
            from torch_geometric.data import Data
            from torch_geometric.loader import DataLoader
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"torch_geometric is required for this test: {exc}")
        return Data, DataLoader

    def test_early_stopping_tracks_validation_auroc(self) -> None:
        torch = self._require_torch()

        from src.configs.config import TrainConfig
        from src.data.graph_classification_dataset import GraphClassificationFold
        from src.models.h2snn_model import ModelDims
        from src.trainers.graph_classification_trainer import GraphClassificationTrainer
        from src.utils.checkpointer import Checkpointer
        from src.utils.logger import ExperimentLogger

        class FakeTrainer(GraphClassificationTrainer):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.seen_epochs: list[int] = []

            def _run_epoch(self, loader, epoch: int, train: bool) -> dict[str, float]:
                if train:
                    return {"loss": 1.0, "acc": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "auroc": 0.0}
                if loader == "val":
                    aurocs = [0.20, 0.50, 0.40, 0.39]
                    self.seen_epochs.append(epoch)
                    return {
                        "loss": 1.0,
                        "acc": 0.0,
                        "macro_f1": 0.0,
                        "micro_f1": 0.0,
                        "auroc": aurocs[epoch],
                    }
                return {
                    "loss": 1.0,
                    "acc": 0.0,
                    "macro_f1": 0.0,
                    "micro_f1": 0.0,
                    "auroc": 0.7 + 0.01 * epoch,
                }

        tmp_parent = Path(__file__).resolve().parent.parent
        with TemporaryDirectory(dir=tmp_parent) as tmpdir:
            cfg = TrainConfig.from_dict(
                {
                    "data": {
                        "dataset_name": "mutag",
                        "split_protocol": "stratified_holdout",
                        "run_index": 2,
                        "num_runs": 5,
                    },
                    "epochs": 4,
                    "device": "cpu",
                    "output_dir": tmpdir,
                    "exp_name": "trainer_test",
                    "checkpoint_metric": "auroc",
                    "checkpoint_mode": "max",
                    "early_stopping_patience": 2,
                }
            )
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            logger = ExperimentLogger(cfg.exp_dir)
            checkpointer = Checkpointer(cfg.exp_dir, metric=cfg.checkpoint_metric, mode=cfg.checkpoint_mode)
            trainer = FakeTrainer(
                model=model,  # type: ignore[arg-type]
                optimizer=optimizer,
                scheduler=None,
                cfg=cfg,
                logger=logger,
                checkpointer=checkpointer,
            )
            fold = GraphClassificationFold(
                dims=ModelDims(
                    in_channels=1,
                    num_node_types=1,
                    num_relations=1,
                    num_classes=2,
                    discovery=None,
                    num_nodes=None,
                ),
                train_loader="train",  # type: ignore[arg-type]
                val_loader="val",  # type: ignore[arg-type]
                test_loader="test",  # type: ignore[arg-type]
                fold_index=2,
                num_folds=5,
                run_index=2,
                num_runs=5,
            )

            trainer.fit(fold)
            logger.close()

            metrics_path = Path(tmpdir) / "trainer_test" / "best_test_metrics_run2.json"
            record = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertEqual(trainer.seen_epochs, [0, 1, 2, 3])
        self.assertEqual(record["epoch"], 1)
        self.assertAlmostEqual(record["val_auroc"], 0.50)
        self.assertAlmostEqual(record["test_auroc"], 0.71)

    def test_graph_batches_pass_nonempty_faces_to_encoder(self) -> None:
        torch = self._require_torch()
        Data, DataLoader = self._require_torch_geometric()

        from src.configs.config import TrainConfig
        from src.trainers.graph_classification_trainer import GraphClassificationTrainer
        from src.utils.checkpointer import Checkpointer
        from src.utils.logger import ExperimentLogger

        class FakeGraphModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))
                self.last_face_count = 0
                self.last_closure_count = 0
                self.last_pool_edge_count = 0

            def encode(
                self,
                x,
                node_type,
                edge_index,
                edge_type,
                face_groups,
                closure_edge_index,
                num_nodes,
                num_edges,
                epoch=-1,
                discovery=None,
            ):
                self.last_face_count = sum(group.num_faces for group in face_groups)
                self.last_closure_count = int(closure_edge_index.size(1))
                encoded_edge_index = torch.cat(
                    (edge_index, closure_edge_index),
                    dim=1,
                )
                hidden = x.new_zeros((num_nodes, 2))
                edge_hidden = x.new_zeros((encoded_edge_index.size(1), 2))
                face_hidden = x.new_zeros((self.last_face_count, 2))
                return SimpleNamespace(
                    x_star=hidden,
                    h0=hidden,
                    s_star=edge_hidden,
                    delta0_x_star=edge_hidden,
                    delta0_h0=edge_hidden,
                    F_src=torch.eye(2).expand(edge_hidden.size(0), 2, 2),
                    F_tgt=torch.eye(2).expand(edge_hidden.size(0), 2, 2),
                    raw_face_curl=face_hidden,
                    delta1_delta0_h0=face_hidden,
                    edge_index=encoded_edge_index,
                    instance_weight_means=None,
                    instance_weight_stds=None,
                    instance_weight_nonempty_mask=None,
                    operator_ramp=1.0,
                )

            def graph_pooling(
                self,
                x_star,
                h0,
                s_star,
                delta0_x_star,
                delta0_h0,
                edge_index,
                F_src,
                F_tgt,
                batch,
                num_graphs,
            ):
                self.last_pool_edge_count = int(edge_index.size(1))
                self.assert_pool_alignment = edge_index.size(1) == s_star.size(0)
                logits = x_star.new_zeros((num_graphs, 2))
                logits[:, 0] = 1.0 + self.weight
                return logits

        data = Data(
            x=torch.ones((3, 1)),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            y=torch.tensor([0], dtype=torch.long),
        )
        data.node_type = torch.tensor([0, 1, 0], dtype=torch.long)
        data.edge_type = torch.tensor([0, 1, 0], dtype=torch.long)
        data.graph_id = torch.tensor([19], dtype=torch.long)
        loader = DataLoader([data], batch_size=1, shuffle=False)

        tmp_parent = Path(__file__).resolve().parent.parent
        with TemporaryDirectory(dir=tmp_parent) as tmpdir:
            cfg = TrainConfig.from_dict(
                {
                    "data": {
                        "dataset_name": "mutag",
                        "face_sample_k": 10,
                        "exclude_identity_metapaths": True,
                    },
                    "device": "cpu",
                    "output_dir": tmpdir,
                    "exp_name": "graph_faces_trainer_test",
                }
            )
            model = FakeGraphModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            logger = ExperimentLogger(cfg.exp_dir)
            try:
                checkpointer = Checkpointer(
                    cfg.exp_dir,
                    metric=cfg.checkpoint_metric,
                    mode=cfg.checkpoint_mode,
                )
                trainer = GraphClassificationTrainer(
                    model=model,  # type: ignore[arg-type]
                    optimizer=optimizer,
                    scheduler=None,
                    cfg=cfg,
                    logger=logger,
                    checkpointer=checkpointer,
                )
                trainer._run_epoch(loader, epoch=0, train=False)
            finally:
                logger.close()

            self.assertEqual(model.last_face_count, 3)
            self.assertEqual(model.last_closure_count, 3)
            self.assertEqual(model.last_pool_edge_count, 6)
            self.assertTrue(model.assert_pool_alignment)

    def test_graph_batch_discovery_filters_selected_schema_ids(self) -> None:
        torch = self._require_torch()

        from src.data.schema_structures import DiscoveryMetadata
        from src.data.structures import FaceGroup
        from src.trainers.graph_classification_trainer import GraphClassificationTrainer

        selected = FaceGroup(
            template_name="N0_R0_N1_R1_N0",
            node_type_sequence=("node_type_0", "node_type_1", "node_type_0"),
            face_node_index=torch.tensor([[0, 1, 2]]),
            face_edge_index=torch.tensor([[0, 1]]),
            relation_index=torch.tensor([[0, 1]]),
            face_edge_sign=torch.ones(1, 2),
            closure_edge_index=torch.tensor([[2, 0]]),
        )
        unselected = FaceGroup(
            template_name="N0_R0_N1_R0_N1",
            node_type_sequence=("node_type_0", "node_type_1", "node_type_1"),
            face_node_index=torch.tensor([[0, 1, 1]]),
            face_edge_index=torch.tensor([[0, 0]]),
            relation_index=torch.tensor([[0, 0]]),
            face_edge_sign=torch.ones(1, 2),
            closure_edge_index=torch.tensor([[1, 0]]),
        )
        trainer = GraphClassificationTrainer.__new__(GraphClassificationTrainer)
        trainer._discovery_schema_names = ("N0_R0_N1_R1_N0",)
        trainer._discovery_template = DiscoveryMetadata(
            num_schemas=1,
            schema_ids=torch.empty(0, dtype=torch.long),
            schema_incidence_mass=torch.tensor([3.0]),
            coverage_schema_index=torch.empty(0, dtype=torch.long),
            coverage_unit_index=torch.empty(0, dtype=torch.long),
            num_coverage_units=0,
        )

        kept, closure_edge_index, discovery = trainer._batch_discovery(
            (selected, unselected),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            batch=torch.zeros(3, dtype=torch.long),
            num_graphs=1,
        )

        self.assertEqual([group.template_name for group in kept], ["N0_R0_N1_R1_N0"])
        self.assertTrue(torch.equal(closure_edge_index, torch.tensor([[2], [0]])))
        self.assertIsNotNone(discovery)
        self.assertTrue(torch.equal(discovery.schema_ids, torch.tensor([0])))
        self.assertTrue(
            torch.equal(discovery.schema_incidence_mass, torch.tensor([3.0]))
        )
        self.assertTrue(
            torch.equal(discovery.coverage_schema_index, torch.tensor([0]))
        )
        self.assertTrue(torch.equal(discovery.coverage_unit_index, torch.tensor([0])))

        empty_kept, empty_closure, empty_discovery = trainer._batch_discovery(
            tuple(),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            batch=torch.zeros(3, dtype=torch.long),
            num_graphs=1,
        )
        self.assertEqual(empty_kept, tuple())
        self.assertEqual(empty_closure.numel(), 0)
        self.assertIsNotNone(empty_discovery)
        self.assertTrue(
            torch.equal(
                empty_discovery.schema_incidence_mass,
                torch.tensor([3.0]),
            )
        )
        self.assertEqual(empty_discovery.coverage_schema_index.numel(), 0)
        self.assertEqual(empty_discovery.num_coverage_units, 1)

    def test_graph_discovery_template_uses_probe_selector(self) -> None:
        torch = self._require_torch()
        Data, _ = self._require_torch_geometric()

        from src.configs.config import TrainConfig
        from src.data.graph_classification_dataset import _build_graph_discovery_template

        cfg = TrainConfig.from_dict(
            {
                "data": {
                    "dataset_name": "mutag",
                    "split_protocol": "stratified_holdout",
                    "pooling_type": "sheafcov",
                },
                "discovery": {
                    "enabled": True,
                    "mode": "auto",
                    "selector_strategy": "probe",
                    "min_schema_length": 2,
                    "max_schema_length": 2,
                    "selected_schema_k": 1,
                    "max_total_instances": 10,
                    "max_total_incidences": 30,
                    "require_target_start": False,
                    "require_target_end": False,
                    "selector_holdout_ratio": 0.5,
                    "selector_probe_max_iter": 25,
                    "selector_stop_tolerance": 1e-12,
                },
            }
        )

        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        empty_edge_type = torch.empty(0, dtype=torch.long)
        positive_path = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        positive_edge_type = torch.tensor([0, 1], dtype=torch.long)
        node_type = torch.tensor([0, 1, 0], dtype=torch.long)
        train_list = [
            Data(edge_index=empty_edge_index, edge_type=empty_edge_type, node_type=node_type, y=torch.tensor([0]), num_nodes=3),
            Data(edge_index=empty_edge_index, edge_type=empty_edge_type, node_type=node_type, y=torch.tensor([0]), num_nodes=3),
            Data(edge_index=positive_path, edge_type=positive_edge_type, node_type=node_type, y=torch.tensor([1]), num_nodes=3),
            Data(edge_index=positive_path, edge_type=positive_edge_type, node_type=node_type, y=torch.tensor([1]), num_nodes=3),
        ]

        discovery, selected_names, selected_lengths, report = _build_graph_discovery_template(
            train_list=train_list,
            config=cfg,
        )

        self.assertIsNotNone(discovery)
        self.assertEqual(len(selected_names), 1)
        self.assertEqual(selected_lengths, (2,))
        # Two length-2 faces over four training graphs: (2 * (2 + 1)) / 4.
        self.assertTrue(
            torch.equal(discovery.schema_incidence_mass, torch.tensor([1.5]))
        )
        self.assertIsNotNone(report)
        self.assertEqual(report.selected_indices, (0,))  # type: ignore[union-attr]
        self.assertEqual(report.final_realized_incidence, 1.5)  # type: ignore[union-attr]
        self.assertTrue(report.materialized_face_hash)  # type: ignore[union-attr]
        self.assertEqual(len(report.candidate_statistics), 1)  # type: ignore[union-attr]
        self.assertEqual(
            report.coverage_diagnostics[0].definition,  # type: ignore[union-attr]
            "graph_presence_in_training_fold_postquota",
        )
        self.assertEqual(
            report.coverage_diagnostics[1].definition,  # type: ignore[union-attr]
            "graph_selected_materialized_support",
        )

    def test_discovery_graph_batches_use_discovery_sampling_cap(self) -> None:
        torch = self._require_torch()
        Data, DataLoader = self._require_torch_geometric()

        from src.configs.config import TrainConfig
        from src.data.schema_structures import DiscoveryMetadata
        from src.trainers.graph_classification_trainer import GraphClassificationTrainer
        from src.utils.checkpointer import Checkpointer
        from src.utils.logger import ExperimentLogger

        class FakeGraphModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))
                self.last_face_count = 0
                self.last_discovery_schema_ids = None

            def encode(
                self,
                x,
                node_type,
                edge_index,
                edge_type,
                face_groups,
                closure_edge_index,
                num_nodes,
                num_edges,
                epoch=-1,
                discovery=None,
            ):
                self.last_face_count = sum(group.num_faces for group in face_groups)
                self.last_discovery_schema_ids = (
                    None if discovery is None else discovery.schema_ids.detach().cpu()
                )
                hidden = x.new_zeros((num_nodes, 2))
                edge_hidden = x.new_zeros((num_edges, 2))
                face_hidden = x.new_zeros((self.last_face_count, 2))
                return SimpleNamespace(
                    x_star=hidden,
                    h0=hidden,
                    s_star=edge_hidden,
                    delta0_x_star=edge_hidden,
                    delta0_h0=edge_hidden,
                    F_src=torch.eye(2).expand(edge_hidden.size(0), 2, 2),
                    F_tgt=torch.eye(2).expand(edge_hidden.size(0), 2, 2),
                    raw_face_curl=face_hidden,
                    delta1_delta0_h0=face_hidden,
                    edge_index=edge_index,
                    instance_weight_means=None,
                    instance_weight_stds=None,
                    instance_weight_nonempty_mask=None,
                    operator_ramp=1.0,
                )

            def graph_pooling(
                self,
                x_star,
                h0,
                s_star,
                delta0_x_star,
                delta0_h0,
                edge_index,
                F_src,
                F_tgt,
                batch,
                num_graphs,
            ):
                logits = x_star.new_zeros((num_graphs, 2))
                logits[:, 0] = 1.0 + self.weight
                return logits

        data = Data(
            x=torch.ones((5, 1)),
            edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]], dtype=torch.long),
            y=torch.tensor([0], dtype=torch.long),
        )
        data.node_type = torch.tensor([0, 1, 1, 0, 0], dtype=torch.long)
        data.edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        data.graph_id = torch.tensor([23], dtype=torch.long)
        loader = DataLoader([data], batch_size=1, shuffle=False)

        tmp_parent = Path(__file__).resolve().parent.parent
        with TemporaryDirectory(dir=tmp_parent) as tmpdir:
            cfg = TrainConfig.from_dict(
                {
                    "data": {
                        "dataset_name": "mutag",
                        "face_sample_k": 10,
                        "pooling_type": "sheafcov",
                    },
                    "discovery": {
                        "enabled": True,
                        "mode": "auto",
                        "min_schema_length": 2,
                        "max_schema_length": 2,
                        "max_instances_per_anchor": 1,
                        "max_instances_per_schema": 10,
                        "max_total_instances": 10,
                        "max_total_incidences": 30,
                        "require_target_start": False,
                        "require_target_end": False,
                    },
                    "device": "cpu",
                    "output_dir": tmpdir,
                    "exp_name": "graph_discovery_cap_test",
                }
            )
            model = FakeGraphModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            logger = ExperimentLogger(cfg.exp_dir)
            try:
                trainer = GraphClassificationTrainer(
                    model=model,  # type: ignore[arg-type]
                    optimizer=optimizer,
                    scheduler=None,
                    cfg=cfg,
                    logger=logger,
                    checkpointer=Checkpointer(
                        cfg.exp_dir,
                        metric=cfg.checkpoint_metric,
                        mode=cfg.checkpoint_mode,
                    ),
                )
                trainer._discovery_schema_names = ("N0_R0_N1_R1_N0",)
                trainer._discovery_template = DiscoveryMetadata(
                    num_schemas=1,
                    schema_ids=torch.empty(0, dtype=torch.long),
                    schema_incidence_mass=torch.tensor([3.0]),
                    coverage_schema_index=torch.empty(0, dtype=torch.long),
                    coverage_unit_index=torch.empty(0, dtype=torch.long),
                    num_coverage_units=0,
                )
                trainer._run_epoch(loader, epoch=0, train=False)
            finally:
                logger.close()

        self.assertEqual(model.last_face_count, 1)
        self.assertTrue(torch.equal(model.last_discovery_schema_ids, torch.tensor([0])))


if __name__ == "__main__":
    unittest.main()
