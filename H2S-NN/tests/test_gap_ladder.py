"""D12 G0b diagnostics (gap localization ladder): `ablation.identity_maps` and
`data.feature_normalization`. Both are controls, never tuned, default off."""
from __future__ import annotations

import unittest

import torch

from src.configs.config import AblationConfig, TrainConfig, reject_never_tuned_settings
from src.models.map_operators import EdgeMapPair, IdentityMapOperator
from src.models.vertex_path_laplacian import VertexPathSheafLaplacian
from tests.test_factorized_maps import _random_operator
from tests.test_v2_model import _float64_pap, _v2_model

SHEAF_FREE = {"identity_maps": True, "no_path_cells": True, "coupling": "symmetric"}


def _random_graph(seed: int, num_nodes: int = 9, num_edges: int = 20):
    """Random multigraph with a repeated edge and a self-loop (both count as in Delta_0)."""
    generator = torch.Generator().manual_seed(seed)
    edge_index = torch.randint(num_nodes, (2, num_edges), generator=generator)
    edge_index[:, 1] = edge_index[:, 0]  # duplicate edge
    edge_index[:, 2] = torch.tensor([3, 3])  # self-loop
    return edge_index, num_nodes


def _dense_normalized_graph_laplacian(edge_index: torch.Tensor, num_nodes: int, d: int) -> torch.Tensor:
    """`D^{-1/2} (D - A) D^{-1/2} (x) I_d`, A counting every stored edge in both directions,
    D_v = number of incidences of v (a self-loop counts twice, matching edge_degree)."""
    adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float64)
    degree = torch.zeros(num_nodes, dtype=torch.float64)
    for u, v in edge_index.t().tolist():
        adjacency[u, v] += 1.0
        adjacency[v, u] += 1.0
        degree[u] += 1.0
        degree[v] += 1.0
    inv = torch.where(degree > 0, degree.rsqrt(), torch.zeros_like(degree))
    laplacian = torch.diag(degree) - adjacency
    return torch.kron(inv.view(-1, 1) * laplacian * inv.view(1, -1), torch.eye(d, dtype=torch.float64))


def _materialize(laplacian: VertexPathSheafLaplacian, num_nodes: int, d: int) -> torch.Tensor:
    basis = torch.eye(num_nodes * d, dtype=torch.float64).view(-1, num_nodes, d)
    return torch.stack([laplacian.apply_normalized(b).reshape(-1) for b in basis], 1)


class IdentityMapsOperatorTests(unittest.TestCase):
    def test_identity_maps_give_the_normalized_graph_laplacian(self) -> None:
        d = 3
        for seed in range(5):
            edge_index, n = _random_graph(seed)
            identity = IdentityMapOperator.ones(edge_index.size(1), d, torch.zeros(1, dtype=torch.float64))
            laplacian = VertexPathSheafLaplacian(EdgeMapPair(identity, identity), edge_index, n,
                                                 (), None, checkpoint=False)
            error = (_materialize(laplacian, n, d) - _dense_normalized_graph_laplacian(edge_index, n, d)).abs().max()
            self.assertLessEqual(float(error), 1e-12)

    def test_control_learned_maps_are_not_the_graph_laplacian(self) -> None:
        d = 3
        edge_index, n = _random_graph(0)
        m = edge_index.size(1)
        maps = EdgeMapPair(_random_operator(m, 2, d, 1, seed=1, group_ids=torch.zeros(m, dtype=torch.long)),
                           _random_operator(m, 2, d, 1, seed=2, group_ids=torch.zeros(m, dtype=torch.long)))
        laplacian = VertexPathSheafLaplacian(maps, edge_index, n, (), None, checkpoint=False)
        error = (_materialize(laplacian, n, d) - _dense_normalized_graph_laplacian(edge_index, n, d)).abs().max()
        self.assertGreater(float(error), 1e-3)


class IdentityMapsModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = _float64_pap()

    def _logits(self, model) -> torch.Tensor:
        with torch.no_grad():
            return model(self.data).logits

    def test_state_dict_unchanged_and_default_off(self) -> None:
        self.assertFalse(AblationConfig().identity_maps)
        base = _v2_model(self.data, ablation={"no_path_cells": True, "coupling": "symmetric"})
        free = _v2_model(self.data, ablation=SHEAF_FREE)
        self.assertEqual(list(base.state_dict()), list(free.state_dict()))
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, free.state_dict()[key]), key)

    def test_sheaf_free_logits_ignore_the_map_module(self) -> None:
        for ablation, expect_equal in ((SHEAF_FREE, True), ({"no_path_cells": True, "coupling": "symmetric"}, False)):
            with self.subTest(identity_maps=expect_equal):
                model = _v2_model(self.data, ablation=ablation).double().eval()
                before = self._logits(model)
                with torch.no_grad():
                    for parameter in model.node_to_edge.parameters():
                        parameter.add_(torch.randn_like(parameter))
                change = float((self._logits(model) - before).abs().max())
                if expect_equal:
                    self.assertEqual(change, 0.0)
                else:
                    self.assertGreater(change, 1e-6)

    def test_sheaf_free_still_propagates(self) -> None:
        """Not vacuous: the sheaf-free model's logits depend on neighbours (graph diffusion runs)."""
        model = _v2_model(self.data, ablation=SHEAF_FREE).double().eval()
        before = self._logits(model)
        authors = self.data.node_type == 1
        x = self.data.x.clone()
        x[authors] = x[authors] + 1.0
        from tests.test_v2_model import _with

        moved = _with(self.data, x=x)
        with torch.no_grad():
            after = model(moved).logits
        self.assertGreater(float((after - before).abs().max()), 1e-6)

    def test_config_rejects_invalid_combinations(self) -> None:
        for bad in ({"identity_maps": True},
                    {"identity_maps": True, "no_path_cells": True, "h0_only_readout": True},
                    {"identity_maps": True, "no_path_cells": True, "coupling": "pair_attention"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                AblationConfig(**bad)
        AblationConfig(**SHEAF_FREE)  # valid


class NeverTunedTests(unittest.TestCase):
    def test_identity_maps_rejected_in_tuning_base_config(self) -> None:
        with self.assertRaises(ValueError):
            reject_never_tuned_settings({"ablation": {"identity_maps": True}})

    def test_diagnostics_rejected_in_search_spaces(self) -> None:
        for name in ("ablation.identity_maps", "data.feature_normalization"):
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
