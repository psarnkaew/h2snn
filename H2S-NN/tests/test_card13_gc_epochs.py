"""Card 13: the four GC experiment configs train 500 epochs (HetSheaf's GC sweep files) with patience 50.

`TrainConfig.from_dict` on each `configs/experiments/<dataset>_h2snn_v3b.yaml` gives `epochs == 500` and
`early_stopping_patience == 50` (the patience comes from the `hetsheaf` protocol budget). Text only: no data.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from src.configs.config import TrainConfig

EXPERIMENTS = Path(__file__).resolve().parents[1] / "configs" / "experiments"
GC_DATASETS = ("mutag", "proteins", "enzymes", "nci1")


class Card13GcEpochsTests(unittest.TestCase):
    def test_gc_configs_train_500_epochs_with_patience_50(self) -> None:
        for dataset in GC_DATASETS:
            with self.subTest(dataset=dataset):
                path = EXPERIMENTS / f"{dataset}_h2snn_v3b.yaml"
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                cfg = TrainConfig.from_dict(raw)
                self.assertEqual(cfg.epochs, 500)
                self.assertEqual(cfg.early_stopping_patience, 50)


if __name__ == "__main__":
    unittest.main()
