"""Legacy-removal card (`docs/H2S-NN_legacy_removal_card.md`).

LR-2: an empty model section (and an empty configuration) resolves to the anchored v3-B model: propagation
`anchored`, arm `full`, `ablation.coupling: symmetric` (B2), and the built model solves the anchored system.
Card mutation: the default back to `chebyshev` -> `test_empty_model_section_is_the_anchored_v3b_model` fails.
"""

from __future__ import annotations

import unittest

import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


class V3BDefaultTests(unittest.TestCase):
    def test_empty_model_section_is_the_anchored_v3b_model(self) -> None:
        from src.configs.config import ModelConfig, TrainConfig
        from src.models.h2snn_model import H2SNN, ModelDims

        model_config = ModelConfig.from_dict({})
        self.assertEqual((model_config.propagation, model_config.arm), ("anchored", "full"))
        self.assertEqual((ModelConfig().propagation, ModelConfig().arm), ("anchored", "full"))
        cfg = TrainConfig.from_dict({"device": "cpu", "data": {"dataset_name": "acm"}})
        self.assertEqual((cfg.model.propagation, cfg.model.arm, cfg.ablation.coupling),
                         ("anchored", "full", "symmetric"))
        torch.manual_seed(0)
        model = H2SNN(cfg, ModelDims(in_channels=4, num_node_types=2, num_relations=2,
                                     num_classes=2, face_lengths=(2,)))
        self.assertTrue(model.anchored)
        self.assertIsNotNone(model.theta_tau)


if __name__ == "__main__":
    unittest.main()
