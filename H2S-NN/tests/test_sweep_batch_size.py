"""Card 2 D6.7 (spec 1.17 Q4, author decision 4 of the card 2 review): H2S-NN's sweep tunes `data.batch_size` in
{32, 256, 1024} on the eight H2GB node-classification datasets and nowhere else. No data, no W&B, no training.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


class SweepBatchSizeTests(unittest.TestCase):
    def test_batch_size_axis_on_h2gb_nc_only(self) -> None:
        from scripts.run_h2snn_wandb_tuning import apply_sweep_overrides, build_sweep_config
        from src.configs.config import H2GB_NC_DATASETS, TrainConfig

        self.assertEqual(len(H2GB_NC_DATASETS), 8)
        for dataset in H2GB_NC_DATASETS:
            with self.subTest(dataset=dataset):
                sweep = build_sweep_config(task_type="node_classification", metric="macro_f1", epochs=1,
                                           dataset_name=dataset)
                self.assertEqual(sweep["parameters"]["batch_size"], {"values": [32, 256, 1024]})
                raw = yaml.safe_load((REPO / "configs" / "experiments" / f"{dataset}_h2snn_v3b.yaml").read_text())
                for b in (32, 256, 1024):  # a trial's value reaches the resolved config
                    payload = apply_sweep_overrides(raw, {"batch_size": b})
                    self.assertEqual(TrainConfig.from_dict(payload).data.batch_size, b)
        for task, dataset in (("node_classification", "acm"), ("node_classification", "dblp"),
                              ("node_classification", "imdb"), ("node_classification", "freebase"),
                              ("link_prediction", "acm_pa"), ("link_prediction", "freebase_bb"),
                              ("link_prediction", "lastfm"), ("graph_classification", "mutag"),
                              ("graph_classification", "nci1")):
            with self.subTest(task=task, dataset=dataset):
                sweep = build_sweep_config(task_type=task, metric="macro_f1", epochs=1, dataset_name=dataset)
                self.assertNotIn("batch_size", sweep["parameters"])
                raw = yaml.safe_load((REPO / "configs" / "experiments" / f"{dataset}_h2snn_v3b.yaml").read_text())
                with self.assertRaisesRegex(ValueError, "H2GB node-classification datasets only"):
                    apply_sweep_overrides(raw, {"batch_size": 256})  # refused off H2GB NC (MB2)


if __name__ == "__main__":
    unittest.main()
