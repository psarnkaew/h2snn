"""D12 G0b diagnostics (gap localization ladder): `data.feature_normalization`, a control, never tuned,
default off. (`ablation.identity_maps` and its tests were removed by the legacy-removal card, B1; the
name now raises.)"""
from __future__ import annotations

import unittest

import torch

from src.configs.config import TrainConfig, reject_never_tuned_settings


class NeverTunedTests(unittest.TestCase):
    def test_diagnostics_rejected_in_search_spaces(self) -> None:
        # (legacy-removal B1: ablation.identity_maps was deleted; a removed key raises in a search
        # space, tests/test_legacy_removal.py)
        for name in ("data.feature_normalization",):
            with self.subTest(name=name), self.assertRaises(ValueError):
                reject_never_tuned_settings(None, {name: {"values": [True, False]}})
        reject_never_tuned_settings(None, {"model.d": {"values": [16, 32]}})  # control: allowed


class FeatureNormalizationTests(unittest.TestCase):
    def test_config_values(self) -> None:
        self.assertEqual(TrainConfig.from_dict({"data": {"dataset_name": "acm"}}).data.feature_normalization,
                         "none")
        TrainConfig.from_dict({"data": {"dataset_name": "acm", "feature_normalization": "row"}})
        with self.assertRaises(ValueError):
            TrainConfig.from_dict({"data": {"dataset_name": "acm", "feature_normalization": "l2"}})

    def test_row_switch_applies_pyg_normalize_features_per_type(self) -> None:
        """build_dataset applies PyG's transform to the HeteroData before stacking; checked here
        on a small HeteroData: per type, rows sum to 1 after subtracting that type's minimum."""
        from torch_geometric.data import HeteroData
        from torch_geometric.transforms import NormalizeFeatures

        from src.data.hetero_flatten import stack_node_features

        data = HeteroData()
        data["paper"].x = torch.tensor([[0.0, 2.0, 2.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        data["author"].x = torch.tensor([[5.0], [3.0]])
        data["term"].num_nodes = 2
        stacked_raw = stack_node_features(data)[0]
        stacked = stack_node_features(NormalizeFeatures()(data))[0]
        expected_paper = torch.tensor([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(stacked[:3], expected_paper))
        self.assertTrue(torch.allclose(stacked[3:5, :1], torch.tensor([[1.0], [0.0]])))  # (5-3)/2, 0/clamp(1)
        self.assertTrue(torch.equal(stacked[5:], torch.zeros(2, 3)))  # featureless type untouched
        self.assertFalse(torch.equal(stacked, stacked_raw))  # control


if __name__ == "__main__":
    unittest.main()
