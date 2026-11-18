"""Card 2 D1 (dry run): every H2S-NN cell of the cell list has a configuration that passes validation.

The cell list is the registry's datasets (`src/baselines/registry.DATASETS`) without the two large LP tasks, which
have no H2S-NN entry point (card 2 D0; design-session review). Each cell's `configs/experiments/<dataset>_h2snn_v3b.yaml`
resolves through `TrainConfig.from_dict` to the dataset's task, names r and l_max inside the tuned ranges (spec v1.18:
tuned, no calibrated r), and passes the tuner's base-config refusal (`reject_never_tuned_settings`). Text only: no data.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from src.baselines.registry import DATASETS, LARGE_LP
from src.configs.config import (DATASET_TASK_TYPES, H2GB_NC_DATASETS, MINIBATCH_BATCH_SIZES, TrainConfig,
                                reject_never_tuned_settings)

EXPERIMENTS = Path(__file__).resolve().parents[1] / "configs" / "experiments"
TUNED_R = (8, 16, 32, 64)        # spec §6 / card M10 (`SPACE_WISP_WALKS_PER_ANCHOR`)
TUNED_L_MAX = (2, 3, 4, 5)       # `SPACE_WISP_MAX_LENGTH`
# Written before card 2 with the calibrated r of the withdrawn E11 rule (acm 42, dblp 65, imdb unset = its table entry);
# kept unchanged (D1 adds missing configurations only); the sweep and the dry run override r.
PRE_CARD2 = frozenset({"acm", "dblp", "imdb", "acm_pa", "mutag"})


class ExperimentConfigTests(unittest.TestCase):
    def test_every_h2snn_cell_resolves_a_valid_configuration(self) -> None:
        cells = [(task, d) for task, names in DATASETS.items() for d in names if d not in LARGE_LP]
        self.assertEqual(len(cells), 22)
        for task, dataset in cells:
            with self.subTest(dataset=dataset):
                path = EXPERIMENTS / f"{dataset}_h2snn_v3b.yaml"
                self.assertTrue(path.is_file(), path)
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(raw["data"]["dataset_name"], dataset)
                cfg = TrainConfig.from_dict(raw)
                self.assertEqual(cfg.task_type, task)
                self.assertEqual(DATASET_TASK_TYPES[dataset], task)
                self.assertEqual(cfg.data.path_source, "wisp")
                self.assertGreaterEqual(cfg.data.wisp_walks_per_anchor, 1)
                if dataset not in PRE_CARD2:
                    self.assertIn(raw["data"]["wisp_walks_per_anchor"], TUNED_R)
                    self.assertIn(raw["data"]["wisp_max_length"], TUNED_L_MAX)
                if dataset in H2GB_NC_DATASETS:
                    self.assertEqual(cfg.data.training_mode, "minibatch")
                    self.assertIn(cfg.data.batch_size, MINIBATCH_BATCH_SIZES)
                reject_never_tuned_settings(raw)


if __name__ == "__main__":
    unittest.main()
