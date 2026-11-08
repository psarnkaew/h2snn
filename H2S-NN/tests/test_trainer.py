from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class TrainerSmokeTests(unittest.TestCase):
    def test_trainer_writes_expected_artifacts(self) -> None:
        try:
            import torch
            from src.configs.config import TrainConfig
            from src.models.h2snn_model import build_model
            from src.trainers.trainer import build_trainer
            from src.utils.checkpointer import Checkpointer
            from src.utils.logger import ExperimentLogger
            from src.utils.misc import save_config
        except ImportError as exc:  # pragma: no cover - environment specific
            self.skipTest(f"full model dependencies are required for this test: {exc}")

        from tests.test_factorized_maps import _random_pap_data

        data = _random_pap_data()
        tmp_parent = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(dir=tmp_parent) as tmpdir:
            cfg = TrainConfig.from_dict(
                {
                    "device": "cpu",
                    "epochs": 1,
                    "exp_name": "trainer_smoke",
                    "output_dir": tmpdir,
                    "checkpoint_metric": "acc",
                    "checkpoint_mode": "max",
                    "model": {"d": 4, "d_r": 2, "K": 1, "classifier_hidden_dim": 8},
                    # Hand-built fixture without a fact layer: nothing to resample.
                    "data": {"resample_paths": False, "path_source": "schemas"},
                }
            )
            model = build_model(cfg, data)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            logger = ExperimentLogger(cfg.exp_dir)
            checkpointer = Checkpointer(cfg.exp_dir, metric="acc", mode="max")
            save_config(cfg, cfg.exp_dir / "config.yaml")
            trainer = build_trainer(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                cfg=cfg,
                logger=logger,
                checkpointer=checkpointer,
            )

            try:
                trainer.fit(data)

                self.assertTrue((cfg.exp_dir / "metrics.jsonl").exists())
                self.assertTrue((cfg.exp_dir / "run.log").exists())
                self.assertTrue((cfg.exp_dir / "config.yaml").exists())
                self.assertTrue((cfg.exp_dir / "checkpoints" / "last.pt").exists())
                self.assertTrue((cfg.exp_dir / "checkpoints" / "best.pt").exists())
            finally:
                logger.close()


if __name__ == "__main__":
    unittest.main()
