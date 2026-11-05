"""Task card D4: v2 model with the vertex-2-cell sheaf Laplacian, and the D1a amendment.

Test ids follow the task card (T1-T14). D1's own tests (`tests/test_factorized_maps.py`)
are not edited: they pin the per-side scales of `architecture: v1`. The per-edge versions
of the D1 tests whose formulas change live here.
"""

from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.configs.config import ModelConfig, TrainConfig
from src.models.h2snn_model import H2SNN, ModelDims
from src.models.map_operators import (
    EdgeMapPair,
    FactorizedMapOperator,
    PathTransportOperator,
    normalize_reflectors,
)
from src.models.sheaf_maps import FactorizedNodeToEdgeSheaf
from tests.test_factorized_maps import (
    _holonomy_statistics,
    _holonomies,
    _random_pap_data,
)

F64_TOL = 1e-10
F32_TOL = 1e-5
EPS = 0.1  # map_scale_eps used throughout


# ----------------------------------------------------------------------------- fixtures
def _per_edge_module(data, d: int = 8, k: int = 2, seed: int = 0,
                     dtype: torch.dtype = torch.float64, map_scale: str = "per_edge"):
    torch.manual_seed(seed)
    num_relations = len(data.relation_names)
    module = FactorizedNodeToEdgeSheaf(
        d=d, d_r=4, num_relations=num_relations, hidden_dim=16, householder_k=k,
        scale_eps=EPS, orthogonal_map="fasthpp", map_scale=map_scale,
    ).to(dtype)
    h = torch.randn(data.num_nodes, d, dtype=dtype)
    relation_embeddings = torch.randn(num_relations + 1, 4, dtype=dtype)
    output = module(h, data.edge_index, data.edge_type, relation_embeddings)
    return module, h, relation_embeddings, output


def _chain_paths(num_paths: int, length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`num_paths` disjoint paths v_0 -> ... -> v_L, every edge stored along its path.

    Returns (face_node_index [P, L+1], face_edge_index [P, L], edge_index [2, P*L]).
    """
    nodes = torch.arange(num_paths * (length + 1)).view(num_paths, length + 1)
    edges = torch.arange(num_paths * length).view(num_paths, length)
    edge_index = torch.stack([nodes[:, :-1].reshape(-1), nodes[:, 1:].reshape(-1)])
    return nodes, edges, edge_index


def _spanning_scale(num_edges: int, generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    """Log-uniform scales in [eps, 50 + eps], with both ends present exactly."""
    log_lo, log_hi = math.log(EPS), math.log(50.0 + EPS)
    scale = torch.exp(log_lo + (log_hi - log_lo) * torch.rand(num_edges, generator=generator,
                                                                dtype=torch.float64))
    scale[0] = EPS
    scale[-1] = 50.0 + EPS
    return scale.to(dtype)


def _chain_operators(num_edges: int, k: int, d: int, seed: int, dtype: torch.dtype,
                     shared: bool) -> EdgeMapPair:
    """Random factorized maps on a chain; scales span [eps, 50]; shared or per side."""
    generator = torch.Generator().manual_seed(seed)
    group_ids = torch.arange(num_edges) % 2

    def side(scale: torch.Tensor) -> FactorizedMapOperator:
        rotations, _ = torch.linalg.qr(torch.randn(2, d, d, generator=generator,
                                                   dtype=torch.float64))
        return FactorizedMapOperator(
            scale=scale,
            reflectors=normalize_reflectors(
                torch.randn(num_edges, k, d, generator=generator, dtype=torch.float64), 1e-6
            ).to(dtype),
            rotations=rotations.to(dtype),
            group_ids=group_ids,
        )

    src_scale = _spanning_scale(num_edges, generator, dtype)
    tgt_scale = src_scale if shared else _spanning_scale(num_edges, generator, dtype).flip(0)
    return EdgeMapPair(src=side(src_scale), tgt=side(tgt_scale))


def _materialize_transports(operator: PathTransportOperator, num_nodes: int, d: int,
                            dtype: torch.dtype) -> torch.Tensor:
    """`[P, L+1, d, d]` matrices `T_{v_i}`, read off column by column through `apply`."""
    columns = []
    for column in range(d):
        x = torch.zeros(num_nodes, d, dtype=dtype)
        x[:, column] = 1.0
        columns.append(operator.apply(x))
    return torch.stack(columns, dim=-1)


def _dense_transports(maps: EdgeMapPair, face_edge_index: torch.Tensor) -> torch.Tensor:
    """Independent reference: `T_{v_i} = prod_{j<=i} inv(F_src) F_tgt`, general inverse."""
    F_src = maps.src.materialize()[face_edge_index]
    F_tgt = maps.tgt.materialize()[face_edge_index]
    num_paths, length, d, _ = F_src.shape
    eye = torch.eye(d, dtype=F_src.dtype).expand(num_paths, d, d)
    transports = [eye]
    for j in range(length):
        transports.append(transports[-1] @ torch.linalg.inv(F_src[:, j]) @ F_tgt[:, j])
    return torch.stack(transports, dim=1)


def _orthogonality_error(matrices: torch.Tensor) -> float:
    d = matrices.size(-1)
    eye = torch.eye(d, dtype=matrices.dtype)
    return float((matrices.transpose(-2, -1) @ matrices - eye).abs().max())


def _config(**model: object) -> TrainConfig:
    base = {"d": 8, "d_r": 4, "d_h": 8, "K": 2, "classifier_hidden_dim": 16}
    base.update(model)
    return TrainConfig.from_dict({"device": "cpu", "checkpoint_metric": "acc", "model": base})


# ----------------------------------------------------------------------------- D1a (T12)
class PerEdgeScaleTests(unittest.TestCase):
    """D1a: `F_src = c_e R_src H(u_src)`, `F_tgt = c_e R_tgt H(u_tgt)`, one `c_e` per edge."""

    def test_T12a_both_sides_share_exactly_one_scale_per_edge(self) -> None:
        data = _random_pap_data()
        _, _, _, output = _per_edge_module(data)
        self.assertIs(output.maps.src.scale, output.maps.tgt.scale)
        self.assertEqual(tuple(output.maps.src.scale.shape), (data.num_edges,))
        # Independently of the stored field: the singular values of both materialized maps
        # all equal the same c_e.
        c = output.maps.src.scale
        for side in (output.maps.src, output.maps.tgt):
            singular = torch.linalg.svdvals(side.materialize())
            self.assertLess(float((singular - c.unsqueeze(-1)).abs().max()), F64_TOL)

    def test_T12a_per_side_ablation_keeps_two_scales(self) -> None:
        data = _random_pap_data()
        _, _, _, output = _per_edge_module(data, map_scale="per_side")
        self.assertIsNot(output.maps.src.scale, output.maps.tgt.scale)
        self.assertGreater(float((output.maps.src.scale - output.maps.tgt.scale).abs().max()), 1e-3)

    def test_T12b_edge_transports_are_orthogonal(self) -> None:
        data = _random_pap_data()
        _, _, _, output = _per_edge_module(data)
        transport = (torch.linalg.inv(output.maps.src.materialize())
                     @ output.maps.tgt.materialize())
        self.assertLess(_orthogonality_error(transport), F64_TOL)

    def test_T12b_path_transports_are_orthogonal_at_every_length(self) -> None:
        d = 6
        cases = [(2, torch.float64, F64_TOL), (3, torch.float64, F64_TOL),
                 (4, torch.float64, F64_TOL), (4, torch.float32, F32_TOL)]
        for length, dtype, tol in cases:
            with self.subTest(length=length, dtype=dtype):
                face_nodes, face_edges, edge_index = _chain_paths(40, length)
                maps = _chain_operators(edge_index.size(1), 2, d, seed=length, dtype=dtype,
                                        shared=True)
                operator = PathTransportOperator(maps.src, maps.tgt, face_nodes, face_edges,
                                                 edge_index, shared_scale=True)
                transports = _materialize_transports(operator, face_nodes.numel(), d, dtype)
                self.assertLess(_orthogonality_error(transports), tol)
                x = torch.randn(face_nodes.numel(), d, dtype=dtype,
                                generator=torch.Generator().manual_seed(1))
                norms_in = x[face_nodes].norm(dim=-1)
                norms_out = operator.apply(x).norm(dim=-1)
                self.assertLess(float(((norms_out - norms_in) / norms_in).abs().max()), tol)
                self.assertGreater(float(maps.src.scale.max() / maps.src.scale.min()), 400.0)

    def test_T12b_negative_control_per_side_scales_break_orthogonality(self) -> None:
        d = 6
        for length in (2, 3, 4):
            with self.subTest(length=length):
                face_nodes, face_edges, edge_index = _chain_paths(40, length)
                maps = _chain_operators(edge_index.size(1), 2, d, seed=length,
                                        dtype=torch.float64, shared=False)
                operator = PathTransportOperator(maps.src, maps.tgt, face_nodes, face_edges,
                                                 edge_index)
                transports = _materialize_transports(operator, face_nodes.numel(), d,
                                                     torch.float64)
                self.assertGreater(_orthogonality_error(transports), 1e-3)
                with self.assertRaisesRegex(ValueError, "shared"):
                    PathTransportOperator(maps.src, maps.tgt, face_nodes, face_edges, edge_index,
                                          shared_scale=True)

    def test_T12b_shared_scale_path_uses_no_scale(self) -> None:
        """The fast path never touches `c`: transports are bit-identical for any shared scale,
        and equal both D1's scaled path and an independent general-inverse product."""
        face_nodes, face_edges, edge_index = _chain_paths(30, 3)
        maps = _chain_operators(edge_index.size(1), 2, 5, seed=0, dtype=torch.float64,
                                shared=True)
        x = torch.randn(face_nodes.numel(), 5, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(2))
        fast = PathTransportOperator(maps.src, maps.tgt, face_nodes, face_edges, edge_index,
                                     shared_scale=True)
        rescaled_scale = maps.src.scale * 7.3 + 1.0

        def with_scale(operator: FactorizedMapOperator) -> FactorizedMapOperator:
            return FactorizedMapOperator(rescaled_scale, operator.reflectors,
                                         operator.rotations, operator.group_ids)

        rescaled = PathTransportOperator(with_scale(maps.src), with_scale(maps.tgt), face_nodes,
                                         face_edges, edge_index, shared_scale=True)
        self.assertTrue(torch.equal(fast.apply(x), rescaled.apply(x)))
        w = torch.randn(face_nodes.size(0), 4, 5, dtype=torch.float64)
        self.assertTrue(torch.equal(fast.apply_T(w, face_nodes.numel()),
                                    rescaled.apply_T(w, face_nodes.numel())))
        generic = PathTransportOperator(maps.src, maps.tgt, face_nodes, face_edges, edge_index)
        self.assertLess(float((fast.apply(x) - generic.apply(x)).abs().max()), F64_TOL)
        reference = _dense_transports(maps, face_edges)
        dense = (reference @ x[face_nodes].unsqueeze(-1)).squeeze(-1)
        self.assertLess(float((fast.apply(x) - dense).abs().max()), F64_TOL)
        # Adjoint identity <T x, w> = <x, T^T w>.
        lhs = float((fast.apply(x) * w).sum())
        rhs = float((x * fast.apply_T(w, face_nodes.numel())).sum())
        self.assertLess(abs(lhs - rhs), F64_TOL * max(1.0, abs(lhs)))

    def test_T12c_scale_lower_bound_and_finite_gradients_at_extremes(self) -> None:
        data = _random_pap_data()
        for dtype in (torch.float64, torch.float32):
            for raw in (-50.0, 50.0):
                with self.subTest(dtype=dtype, raw=raw):
                    module, h, relation_embeddings, _ = _per_edge_module(data, dtype=dtype)
                    last = module.edge_scale_mlp[-1]
                    with torch.no_grad():
                        last.weight.zero_()
                        last.bias.fill_(raw)
                    h.requires_grad_(True)
                    maps = module(h, data.edge_index, data.edge_type, relation_embeddings).maps
                    c = maps.src.scale
                    self.assertGreaterEqual(float(c.min()), EPS)
                    expected = EPS if raw < 0 else 50.0 + EPS
                    self.assertLess(float((c - expected).abs().max()), 1e-6)
                    x = torch.randn(data.num_edges, 8, dtype=dtype)
                    total = (maps.src.apply(x).square().sum() + maps.tgt.apply_inverse(x).sum())
                    total.backward()
                    self.assertTrue(bool(torch.isfinite(total)))
                    self.assertTrue(bool(torch.isfinite(h.grad).all()))
                    for parameter in module.parameters():
                        if parameter.grad is not None:
                            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))


class PerEdgeVersionsOfD1Tests(unittest.TestCase):
    """D1 tests whose formulas change under per-edge scales; D1 thresholds unchanged."""

    def test_T10_per_edge_storage_matches_the_bound_exactly(self) -> None:
        """Unique float storage held by both operators: `2 M k d + M + 2 |R| d^2`."""

        def held_tensors(operator: FactorizedMapOperator) -> list[torch.Tensor]:
            found: list[torch.Tensor] = []
            for value in vars(operator).values():
                items = value if isinstance(value, (tuple, list)) else (value,)
                found.extend(item for item in items if isinstance(item, torch.Tensor))
            return found

        data = _random_pap_data()
        num_relations = len(data.relation_names)
        M = data.num_edges
        for d, k in ((8, 1), (8, 2), (32, 2), (32, 4)):
            with self.subTest(d=d, k=k):
                _, _, _, output = _per_edge_module(data, d=d, k=k)
                unique: dict[tuple[int, tuple[int, ...], tuple[int, ...]], torch.Tensor] = {}
                for side in (output.maps.src, output.maps.tgt):
                    side.apply(torch.zeros(M, d, dtype=torch.float64))  # builds the layout
                    for tensor in held_tensors(side):
                        if tensor.is_floating_point():
                            unique[(tensor.data_ptr(), tuple(tensor.shape),
                                    tuple(tensor.stride()))] = tensor
                floats = sum(t.numel() for t in unique.values())
                self.assertEqual(floats, 2 * M * k * d + M + 2 * num_relations * d * d)
                if (d, k) == (32, 2):
                    self.assertGreaterEqual(2 * M * d * d / floats, 10.0)

    def test_T4_per_edge_maps_are_side_specific_and_node_dependent(self) -> None:
        data = _random_pap_data()
        module, h, relation_embeddings, output = _per_edge_module(data)
        src = output.maps.src.materialize()
        tgt = output.maps.tgt.materialize()
        self.assertGreater(float((src - tgt).flatten(1).norm(dim=1).min()), 1e-3)
        node = 3
        perturbed = h.clone()
        perturbed[node] += torch.randn(module.d, generator=torch.Generator().manual_seed(5),
                                       dtype=h.dtype)
        moved = module(perturbed, data.edge_index, data.edge_type, relation_embeddings).maps
        src_change = (moved.src.materialize() - src).flatten(1).norm(dim=1)
        tgt_change = (moved.tgt.materialize() - tgt).flatten(1).norm(dim=1)
        incident = (data.edge_index[0] == node) | (data.edge_index[1] == node)
        self.assertGreater(float(src_change[data.edge_index[0] == node].min()), 1e-3)
        self.assertGreater(float(tgt_change[data.edge_index[1] == node].min()), 1e-3)
        self.assertEqual(float(src_change[~incident].max()), 0.0)
        self.assertEqual(float(tgt_change[~incident].max()), 0.0)

    def test_T5_per_edge_holonomy_is_nontrivial_and_varies(self) -> None:
        data = _random_pap_data()
        _, _, _, output = _per_edge_module(data)
        for group in data.face_groups:
            with self.subTest(template=group.template_name):
                holonomy = _holonomies(output.maps, group.face_edge_index)
                self.assertLess(_orthogonality_error(holonomy), F64_TOL)
                deviation, spread = _holonomy_statistics(holonomy)
                self.assertGreater(deviation, 0.1)
                self.assertGreater(spread, 0.05)
        tied = EdgeMapPair(src=output.maps.src, tgt=output.maps.src)
        for group in data.face_groups:
            deviation, spread = _holonomy_statistics(_holonomies(tied, group.face_edge_index))
            self.assertLess(deviation, 1e-10)
            self.assertLess(spread, 1e-10)


# ----------------------------------------------------------------------------- config, scope
class ArchitectureConfigTests(unittest.TestCase):
    def test_map_scale_defaults_follow_the_architecture(self) -> None:
        self.assertEqual(ModelConfig().architecture, "v1")
        self.assertEqual(ModelConfig().map_scale, "per_side")
        self.assertEqual(ModelConfig.from_dict({}).map_scale, "per_side")
        self.assertEqual(ModelConfig.from_dict({"architecture": "v2"}).map_scale, "per_edge")
        self.assertEqual(
            ModelConfig.from_dict({"architecture": "v2", "map_scale": "per_side"}).map_scale,
            "per_side",
        )

    def test_invalid_architecture_and_scale_values_are_rejected(self) -> None:
        for raw in ({"architecture": "v3"}, {"map_scale": "per_node"},
                    {"architecture": "v2", "map_scale": "per_node"},
                    {"map_scale": "per_edge"},  # v1 keeps D1's per-side scales
                    {"architecture": "v1", "map_family": "factorized", "map_scale": "per_edge"}):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    ModelConfig.from_dict(raw)

    def test_v2_defaults_are_valid_and_documented(self) -> None:
        config = ModelConfig.from_dict({"architecture": "v2", "map_family": "factorized"})
        self.assertEqual((config.gamma_init, config.eta_init), (1.0, 0.5))
        self.assertTrue(config.checkpointing)
        self.assertEqual((config.path_maps, config.path_mode, config.readout),
                         ("transport", "cells", "full"))
        self.assertFalse(config.uniform_path_gate or config.ablate_no_path_cells)
        ModelConfig.from_dict({"architecture": "v2", "map_family": "dense"})
        for switch in ({"path_maps": "free"}, {"path_mode": "endpoint_edges"},
                       {"path_mode": "holonomy_edges"}, {"uniform_path_gate": True},
                       {"ablate_no_path_cells": True}, {"map_scale": "per_side"},
                       {"readout": "h0_only"}, {"checkpointing": False}):
            with self.subTest(switch=switch):
                ModelConfig.from_dict({"architecture": "v2", **switch})

    def test_no_v2_key_is_reachable_from_v1(self) -> None:
        for raw in ({"gamma_init": 2.0}, {"eta_init": 0.3}, {"checkpointing": False},
                    {"path_maps": "free"}, {"path_mode": "endpoint_edges"},
                    {"path_mode": "holonomy_edges"}, {"uniform_path_gate": True},
                    {"ablate_no_path_cells": True}, {"readout": "h0_only"}):
            for family in ("dense", "factorized"):
                with self.subTest(raw=raw, family=family):
                    with self.assertRaisesRegex(ValueError, "v2"):
                        ModelConfig.from_dict({"map_family": family, **raw})

    def test_v2_rejects_v1_switches_bad_ranges_and_conflicts(self) -> None:
        cases = [
            {"ablate_no_faces": True}, {"ablate_no_delta1": True},
            {"ablate_identity_maps": True}, {"use_orthogonal": False},
            {"learnable_eta": False}, {"face_encoder": "lstm"},
            {"face_map_param": "triangular"},
            {"eta_init": 0.0}, {"eta_init": 1.0}, {"gamma_init": 0.0}, {"gamma_init": -1.0},
            {"path_maps": "hypergraph"}, {"path_mode": "edges"}, {"readout": "x_only"},
            {"checkpointing": "false"}, {"uniform_path_gate": 1},
            {"path_maps": "free", "path_mode": "endpoint_edges"},
            {"path_maps": "free", "map_scale": "per_side"},
            {"ablate_no_path_cells": True, "path_maps": "free"},
            {"ablate_no_path_cells": True, "path_mode": "holonomy_edges"},
            {"ablate_no_path_cells": True, "uniform_path_gate": True},
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    ModelConfig.from_dict({"architecture": "v2", **raw})

    def test_v1_model_class_refuses_v2_configs_for_every_task(self) -> None:
        dims = ModelDims(in_channels=6, num_node_types=2, num_relations=2, num_classes=3)
        for task in ("graph_classification", "link_prediction", "node_classification"):
            with self.subTest(task=task):
                config = _config(architecture="v2", map_family="factorized")
                config.data.task_type = task
                with self.assertRaisesRegex(NotImplementedError, "D4"):
                    H2SNN(config, dims)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
