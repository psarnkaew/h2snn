from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


class GraphClassificationTrainerTests(unittest.TestCase):

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

if __name__ == "__main__":
    unittest.main()
