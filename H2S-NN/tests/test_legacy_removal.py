"""Legacy-removal card (`docs/H2S-NN_legacy_removal_card.md`).

LR-2: an empty model section (and an empty configuration) resolves to the anchored v3-B model: propagation
`anchored`, arm `full`, `ablation.coupling: symmetric` (B2), and the built model solves the anchored system.
Card mutation: the default back to `chebyshev` -> `test_empty_model_section_is_the_anchored_v3b_model` fails.

LR-3 (fail-closed): every removed value and every removed key raises `ValueError` naming the setting
and "removed by the legacy-removal card (v1/v2)", in a configuration and in a tuning search space.
The lists are spelled out here (not read from `src/configs/config.py`), so dropping a name from the
registry fails these tests. Card mutations: remove the raise for one value / for one key -> the
corresponding test fails.
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



MESSAGE = "removed by the legacy-removal card \\(v1/v2\\)"
# (section, key) -> removed values (B1-B5; inventory docs/legacy_evidence/inventory.md)
REMOVED_VALUES = {
    ("model", "propagation"): ("chebyshev",),
    ("model", "path_weight"): ("sigmoid_gamma",),
    ("model", "map_family"): ("dense",),
    ("ablation", "coupling"): ("cooperative", "listen_only", "pair_attention"),
    ("data", "path_sampler"): ("uniform_instance",),
}
REMOVED_KEYS = {
    "model": ("K", "layer_dropout", "eta_init", "gamma_init", "coupling_gate_init", "encoder",
              "node_edge_memory_guard_fraction"),
    "ablation": ("no_path_cells", "free_path_maps", "endpoint_edges", "holonomy_edges",
                 "per_side_scale", "identity_maps"),
}
# a value each removed key used to accept (the type does not matter: the name raises)
_OLD_VALUE = {"K": 2, "layer_dropout": 0.0, "eta_init": 0.5, "gamma_init": 1.0,
              "coupling_gate_init": 0.6, "encoder": "filter_bank",
              "node_edge_memory_guard_fraction": 0.5}


class FailClosedTests(unittest.TestCase):
    def test_removed_values_raise(self) -> None:
        from src.configs.config import PROPAGATIONS, TrainConfig

        self.assertEqual(PROPAGATIONS, ("anchored",))
        for (section, key), values in REMOVED_VALUES.items():
            for value in values:
                with self.subTest(setting=f"{section}.{key}", value=value):
                    with self.assertRaisesRegex(ValueError, f"{section}\\.{key}: {value} .*{MESSAGE}"):
                        TrainConfig.from_dict({"data": {"dataset_name": "acm"},
                                               section: {key: value}} if section != "data" else
                                              {"data": {"dataset_name": "acm", key: value}})

    def test_removed_values_raise_on_direct_construction(self) -> None:
        from src.configs.config import AblationConfig, ModelConfig

        for (section, key), values in REMOVED_VALUES.items():
            cls = {"model": ModelConfig, "ablation": AblationConfig}.get(section)
            for value in values if cls is not None else ():
                with self.subTest(setting=f"{section}.{key}", value=value):
                    with self.assertRaisesRegex(ValueError, f"{section}\\.{key}: {value} .*{MESSAGE}"):
                        cls(**{key: value})

    def test_removed_keys_raise(self) -> None:
        from src.configs.config import TrainConfig

        for section, keys in REMOVED_KEYS.items():
            for key in keys:
                with self.subTest(setting=f"{section}.{key}"):
                    with self.assertRaisesRegex(ValueError, f"{section}\\.{key} is {MESSAGE}"):
                        TrainConfig.from_dict({section: {key: _OLD_VALUE.get(key, True)}})

    def test_removed_keys_raise_in_a_tuning_space(self) -> None:
        from src.configs.config import reject_never_tuned_settings

        for section, keys in REMOVED_KEYS.items():
            for key in keys:
                for name in (f"{section}.{key}", key):
                    with self.subTest(search=name):
                        with self.assertRaisesRegex(ValueError, f"{name!r}: {MESSAGE}"):
                            reject_never_tuned_settings(None, {name: {"values": [1, 2]}})
                with self.subTest(base=f"{section}.{key}"):  # named in the tuning base config
                    with self.assertRaisesRegex(ValueError, f"{section}\\.{key} is {MESSAGE}"):
                        reject_never_tuned_settings({section: {key: _OLD_VALUE.get(key, True)}})
        for (section, key), values in REMOVED_VALUES.items():
            for value in values:
                with self.subTest(base=f"{section}.{key}", value=value):
                    with self.assertRaisesRegex(ValueError, f"{section}\\.{key}: {value} .*{MESSAGE}"):
                        reject_never_tuned_settings({section: {key: value}})

    def test_kept_values_still_load(self) -> None:
        """Control: the kept values next to the removed ones load."""
        from src.configs.config import TrainConfig

        TrainConfig.from_dict({"data": {"dataset_name": "acm", "path_sampler": "walk",
                                        "path_source": "schemas"},
                               "model": {"propagation": "anchored", "path_weight": "exp",
                                         "map_family": "factorized"},
                               "ablation": {"coupling": "symmetric", "h0_only_readout": False}})


if __name__ == "__main__":
    unittest.main()
