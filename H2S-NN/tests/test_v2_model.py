"""H2S-NN model (plan §4): vertex-2-cell sheaf Laplacian, D1a per-edge scale, ablations.

Test ids follow task card D4 (T2-T14). D1's operator tests are in
`tests/test_factorized_maps.py`; the task heads and the golden outputs are in
`tests/test_task_heads.py`.
"""

from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.configs.config import AblationConfig, ModelConfig, OptimConfig, TrainConfig
from src.models.h2snn_model import H2SNN
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


# ----------------------------------------------------------------------------- operators (T2-T5, T7, T14)
def _operators(num_edges: int, d: int, k: int, generator: torch.Generator,
               src_scale: torch.Tensor, tgt_scale: torch.Tensor,
               num_groups: int = 3) -> EdgeMapPair:
    """Random factorized maps (float64) with the given per-side scales."""
    group_ids = torch.randint(num_groups, (num_edges,), generator=generator)

    def side(scale: torch.Tensor) -> FactorizedMapOperator:
        rotations, _ = torch.linalg.qr(torch.randn(num_groups, d, d, generator=generator,
                                                   dtype=torch.float64))
        return FactorizedMapOperator(
            scale=scale,
            reflectors=normalize_reflectors(
                torch.randn(num_edges, k, d, generator=generator, dtype=torch.float64), 1e-6),
            rotations=rotations,
            group_ids=group_ids,
        )

    return EdgeMapPair(src=side(src_scale), tgt=side(tgt_scale))


class _Complex:
    """Random small 2-complex: a directed multigraph with one self-loop, one isolated node,
    extra edges that close cycles, and simple paths (lengths 2-4) along stored edges, grouped
    by length. Scales: `per_edge` (shared c_e) or `per_side`; values drawn from `scale_values`
    (or uniform in [0.5, 2] when None)."""

    def __init__(self, seed: int, num_nodes: int = 10, d: int = 3, k: int = 2,
                 scale_mode: str = "per_edge", scale_values: tuple[float, ...] | None = None,
                 lengths: tuple[int, ...] = (2, 3, 4), paths_per_length: int = 2,
                 extra_edges: int = 4) -> None:
        g = torch.Generator().manual_seed(seed)
        self.generator = g
        self.num_nodes = num_nodes
        self.d = d
        active = num_nodes - 1  # node num_nodes-1 stays isolated
        src: list[int] = []
        tgt: list[int] = []
        self.groups: list[tuple[torch.Tensor, torch.Tensor]] = []
        for length in lengths:
            node_rows, edge_rows = [], []
            for _ in range(paths_per_length):
                nodes = torch.randperm(active, generator=g)[: length + 1].tolist()
                edge_ids = []
                for a, b in zip(nodes[:-1], nodes[1:]):
                    edge_ids.append(len(src))
                    src.append(a)
                    tgt.append(b)
                node_rows.append(nodes)
                edge_rows.append(edge_ids)
            self.groups.append((torch.tensor(node_rows), torch.tensor(edge_rows)))
        for _ in range(extra_edges):
            a, b = torch.randperm(active, generator=g)[:2].tolist()
            src.append(a)
            tgt.append(b)
        self.self_loop_node = int(torch.randint(active, (1,), generator=g))
        src.append(self.self_loop_node)
        tgt.append(self.self_loop_node)
        self.edge_index = torch.tensor([src, tgt])
        num_edges = len(src)

        def draw() -> torch.Tensor:
            if scale_values is None:
                return 0.5 + 1.5 * torch.rand(num_edges, generator=g, dtype=torch.float64)
            choice = torch.randint(len(scale_values), (num_edges,), generator=g)
            return torch.tensor(scale_values, dtype=torch.float64)[choice]

        c_src = draw()
        c_tgt = c_src if scale_mode == "per_edge" else draw()
        self.scale_mode = scale_mode
        self.maps = _operators(num_edges, d, k, g, c_src, c_tgt)
        self.weights = [torch.rand(nodes.size(0), generator=g, dtype=torch.float64)
                        for nodes, _ in self.groups]

    def transports(self, nodes: torch.Tensor, edges: torch.Tensor) -> PathTransportOperator:
        return PathTransportOperator(self.maps.src, self.maps.tgt, nodes, edges, self.edge_index,
                                     shared_scale=self.scale_mode == "per_edge")

def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1.0))


class DenseOperatorTests(unittest.TestCase):
    def test_dense_operator_closed_form_inverse_and_shapes(self) -> None:
        from src.models.map_operators import DenseMapOperator

        g = torch.Generator().manual_seed(0)
        c = 0.1 + 5 * torch.rand(12, generator=g, dtype=torch.float64)
        maps = _operators(12, 5, 2, g, c, c).src
        dense = DenseMapOperator(maps.materialize(), maps.scale)
        x = torch.randn(12, 5, generator=g, dtype=torch.float64)
        inverse = torch.linalg.inv(maps.materialize())
        self.assertLess(_rel_err(dense.apply_inverse(x), (inverse @ x.unsqueeze(-1)).squeeze(-1)),
                        F64_TOL)
        self.assertLess(_rel_err(dense.apply_inverse_T(x),
                                 (inverse.transpose(-2, -1) @ x.unsqueeze(-1)).squeeze(-1)), F64_TOL)
        for name in ("apply", "apply_T", "apply_inverse", "apply_orthogonal", "apply_orthogonal_T"):
            self.assertLess(_rel_err(getattr(dense, name)(x), getattr(maps, name)(x)), F64_TOL)
        x3 = torch.randn(12, 3, 5, generator=g, dtype=torch.float64)
        self.assertLess(_rel_err(dense.apply(x3), maps.apply(x3)), F64_TOL)
        rows = torch.tensor([3, 0, 3])
        self.assertTrue(torch.equal(dense.select(rows).maps, dense.maps[rows]))
        with self.assertRaisesRegex(ValueError, "scale"):
            DenseMapOperator(maps.materialize()).apply_inverse(x)

    def test_path_transport_accepts_dense_operators(self) -> None:
        from src.models.map_operators import DenseMapOperator

        for mode in ("per_edge", "per_side"):
            with self.subTest(mode=mode):
                complex_ = _Complex(seed=3, scale_mode=mode)
                maps = complex_.maps
                dense = EdgeMapPair(DenseMapOperator(maps.src.materialize(), maps.src.scale),
                                    DenseMapOperator(maps.tgt.materialize(), maps.tgt.scale))
                x = torch.randn(complex_.num_nodes, 3, dtype=torch.float64)
                for nodes, edges in complex_.groups:
                    shared = mode == "per_edge"
                    a = PathTransportOperator(maps.src, maps.tgt, nodes, edges,
                                              complex_.edge_index, shared_scale=shared)
                    b = PathTransportOperator(dense.src, dense.tgt, nodes, edges,
                                              complex_.edge_index, shared_scale=shared)
                    self.assertLess(_rel_err(b.apply(x), a.apply(x)), F64_TOL)
                    self.assertLess(_rel_err(b.squared_scales(), a.squared_scales()), F64_TOL)


class PathLaplacianTests(unittest.TestCase):
    def test_T7_cell_state_is_the_least_squares_minimizer(self) -> None:
        from src.models.vertex_path_laplacian import PathCellTerm

        for mode in ("per_edge", "per_side"):
            complex_ = _Complex(seed=8, scale_mode=mode)
            x = torch.randn(complex_.num_nodes, complex_.d, dtype=torch.float64,
                            generator=torch.Generator().manual_seed(4))
            for (nodes, edges), weights in zip(complex_.groups, complex_.weights):
                with self.subTest(mode=mode, length=edges.size(1)):
                    term = PathCellTerm(complex_.transports(nodes, edges), weights)
                    y, z = term.cell_values(x)
                    n, d = y.size(1), y.size(2)
                    stacked = torch.eye(d, dtype=torch.float64).repeat(n, 1)
                    for p in range(y.size(0)):
                        solution = torch.linalg.lstsq(stacked, y[p].reshape(-1, 1)).solution
                        self.assertLess(_rel_err(z[p], solution.view(-1)), F64_TOL)
                        self.assertLess(_rel_err(z[p], y[p].sum(0) / n), F64_TOL)


# ----------------------------------------------------------------------------- v2 model (T1, T6, T8-T10, T13)
def _v2_config(ablation: dict | None = None, **model: object) -> TrainConfig:
    base = {"map_family": "factorized", "d": 8, "d_r": 4,
            "classifier_hidden_dim": 16, "face_mlp_hidden_dim": 16, "edge_mlp_hidden_dim": 16}
    base.update(model)
    return TrainConfig.from_dict({"device": "cpu", "checkpoint_metric": "acc", "model": base,
                                  "ablation": ablation or {},
                                  "data": {"path_source": "schemas"}})  # schema fixtures (I2)


def _v2_model(data, seed: int = 0, ablation: dict | None = None, **model: object):
    from src.models.h2snn_model import build_model

    torch.manual_seed(seed)
    return build_model(_v2_config(ablation, **model), data)


# Legacy-removal: `_v2_model` builds the default model, the anchored v3-B model (the v2 model and its
# helpers were removed in LR-3).
class V2ModelTests(unittest.TestCase):
    def test_build_model_builds_the_node_classification_head(self) -> None:
        data = _random_pap_data()
        model = _v2_model(data)
        self.assertIsInstance(model, H2SNN)
        self.assertIsNotNone(model.readout)
        self.assertIsNone(model.link_readout)
        self.assertIsNone(model.graph_readout)
        output = model(data)
        self.assertEqual(tuple(output.logits.shape), (data.target_indices.numel(), 3))
        self.assertTrue(bool(torch.isfinite(output.logits).all()))

    def test_T8_no_dense_map_tensor_in_v2_forward_or_backward(self) -> None:
        """Flags `[M|F|F_g|F_g(L+1), d, d]` shapes and any tensor with >= min(M, F) d^2 / 4
        elements. (The dense-family control was removed with the dense map family, legacy-removal B4.)"""
        from tests.test_factorized_maps import _ShapeRecorder

        data = _random_pap_data(num_papers=100, num_authors=50, max_faces=270)
        d = 32
        M = data.num_edges
        F = sum(group.num_faces for group in data.face_groups)
        forbidden = {M, F}
        for group in data.face_groups:
            forbidden |= {group.num_faces, group.num_faces * (group.length + 1)}
        numel_limit = min(M, F) * d * d // 4
        self.assertGreater(F, M // 2, "fixture: F must be comparable to M")

        def hits(recorder: _ShapeRecorder, phase: str) -> list[tuple[int, ...]]:
            return [shape for p, shape in recorder.shapes if p == phase and (
                (len(shape) >= 3 and shape[-2:] == (d, d) and math.prod(shape[:-2]) in forbidden)
                or math.prod(shape) >= numel_limit)]

        def record(model) -> _ShapeRecorder:
            recorder = _ShapeRecorder()
            with recorder:
                output = model(data)
                loss = output.logits.square().sum()
                recorder.phase = "backward"
                loss.backward()
            return recorder

        model = _v2_model(data, d=d, face_mlp_hidden_dim=128, edge_mlp_hidden_dim=128)
        self.assertLess(max(p.numel() for p in model.parameters()), numel_limit)
        v2 = record(model)
        self.assertEqual(hits(v2, "forward"), [])
        self.assertEqual(hits(v2, "backward"), [])
        self.assertTrue(any(p == "backward" for p, _ in v2.shapes))
        # (legacy-removal B4: the dense-family control was removed with the dense map family)

    def test_T9_deleted_components_are_absent(self) -> None:
        import src.models.sheaf_maps as sheaf_maps
        from tests.test_factorized_maps import _ShapeRecorder

        data = _random_pap_data()
        model = _v2_model(data)
        for name in model.state_dict():
            for deleted in ("edge_initializer", "edge_to_face", "map_head", "edge_norm",
                            "node_norm", "diffusion.eta_1"):
                self.assertNotIn(deleted, name)
        for deleted in ("EdgeInitializer", "SequenceEdgeToFaceSheaf", "NodeToEdgeSheaf"):
            self.assertFalse(hasattr(sheaf_maps, deleted), deleted)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                self.assertTrue(name.startswith("node_encoder"), name)
        M = data.num_edges
        extended = M + sum(group.num_faces for group in data.face_groups)
        # no closure edges in Delta_0: the maps of the solved system cover exactly the M observed facts
        # (legacy-removal B3: read from the anchored state; was the v2 operator's src_idx)
        with torch.no_grad():
            state = model.encode(data.x, data.node_type, data.edge_index, data.edge_type,
                                 data.face_groups, data.num_nodes).operator
        self.assertEqual(state.maps.src.num_edges, M)

        def edge_cochains(model_) -> list[tuple[int, ...]]:
            recorder = _ShapeRecorder()
            with recorder:
                output = model_(data)
                recorder.phase = "backward"
                output.logits.square().sum().backward()
            return [s for _, s in recorder.shapes if len(s) == 2 and s[0] == extended]

        self.assertEqual(edge_cochains(model), [])

    def test_T10_gradients_reach_every_learned_component(self) -> None:
        data = _random_pap_data()
        model = _v2_model(data)
        model(data).logits.square().sum().backward()
        # the encoder's own parameters: theta_tau of the anchored solve (legacy-removal B3: was
        # theta_eta (first_order) or the filter bank V of the v2 model)
        groups = ["theta_tau",
                  "path_gate", "path_encoder",
                  "node_to_edge.reflector_mlp", "node_to_edge.edge_scale_mlp",
                  "node_to_edge.rotation_params", "node_encoder", "relation_embedding",
                  "readout"]
        named = dict(model.named_parameters())
        for prefix in groups:
            with self.subTest(prefix=prefix):
                grads = [p.grad for n, p in named.items() if n.startswith(prefix)]
                self.assertTrue(grads)
                self.assertTrue(all(g is not None for g in grads))
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)

    def test_gate_initialization_is_near_one_half(self) -> None:
        data = _random_pap_data()
        model = _v2_model(data)
        # b = log(path_weight_init) = log 0.5 in float32 (exp, Part H): w_f starts near 0.5.
        expected = float(torch.tensor(math.log(0.5), dtype=torch.float32))
        self.assertEqual(float(model.path_gate.bias.reshape(-1)[0]), expected)
        output = model(data)
        self.assertLess(abs(float(output.path_gate_mean) - 0.5), 0.1)
        self.assertLess(float(output.path_gate_max), 0.75)

    def test_checkpointing_switch_does_not_change_outputs_or_gradients(self) -> None:
        from tests.test_factorized_maps import _as_float64

        data = _as_float64(_random_pap_data())
        results = []
        for checkpointing in (True, False):
            model = _v2_model(data, checkpointing=checkpointing).double().eval()
            output = model(data)
            output.logits.square().sum().backward()
            results.append((output.logits, {n: p.grad for n, p in model.named_parameters()}))
        self.assertLess(_rel_err(results[1][0], results[0][0]), F64_TOL)
        for name, grad in results[0][1].items():
            self.assertLess(_rel_err(results[1][1][name], grad), F64_TOL, name)

    def test_trainer_logs_eta_gamma_uses_scalar_group_and_checkpoints_reload(self) -> None:
        import tempfile
        from pathlib import Path

        from src.builders import build_optimizer
        from src.models.h2snn_model import build_model
        from src.trainers.trainer import build_trainer
        from src.utils.checkpointer import Checkpointer
        from src.utils.logger import ExperimentLogger
        from src.utils.misc import save_config

        data = _random_pap_data()
        tmp_parent = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(dir=tmp_parent) as tmpdir:
            cfg = TrainConfig.from_dict({
                "device": "cpu", "epochs": 2, "exp_name": "v2_smoke", "output_dir": tmpdir,
                "checkpoint_metric": "acc", "checkpoint_mode": "max",
                # legacy-removal B3: the anchored v3-B model (was the v2 model with first_order and
                # sigmoid_gamma: eta, gamma and four gate tensors in the scalar group)
                "model": {"d": 8, "d_r": 4, "classifier_hidden_dim": 16},
                # Hand-built fixture without a fact layer: nothing to resample.
                "data": {"resample_paths": False, "path_source": "schemas"},
                "optim": {"lr": 1e-3, "weight_decay": 1e-4, "scalar_lr_scale": 10.0}})
            torch.manual_seed(0)
            model = build_model(cfg, data)
            logger = ExperimentLogger(cfg.exp_dir)
            checkpointer = Checkpointer(cfg.exp_dir, metric="acc", mode="max")
            save_config(cfg, cfg.exp_dir / "config.yaml")
            optimizer = build_optimizer(model, cfg)
            # theta_tau and the bias b of g (v3-B scalars; legacy-removal B3: was theta_eta,
            # theta_gamma and the four gate-head tensors) in their own group: lr x 10, no weight decay.
            scalar_group = [g for g in optimizer.param_groups
                            if any(p is model.theta_tau for p in g["params"])]
            self.assertEqual(len(scalar_group), 1)
            self.assertEqual(len(scalar_group[0]["params"]), 2)
            self.assertTrue(any(p is model.path_gate.bias for p in scalar_group[0]["params"]))
            self.assertAlmostEqual(scalar_group[0]["lr"], 1e-2)
            self.assertEqual(scalar_group[0]["weight_decay"], 0.0)
            other = [g for g in optimizer.param_groups if g is not scalar_group[0]]
            self.assertEqual(len(other), 1)
            self.assertAlmostEqual(other[0]["lr"], 1e-3)
            self.assertAlmostEqual(other[0]["weight_decay"], 1e-4)
            self.assertEqual(sum(len(g["params"]) for g in optimizer.param_groups),
                             len(list(model.parameters())))
            trainer = build_trainer(model=model, optimizer=optimizer,
                                    scheduler=None, cfg=cfg, logger=logger,
                                    checkpointer=checkpointer)
            try:
                trainer.fit(data)
                import json

                records = [json.loads(line) for line in
                           (cfg.exp_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(records), 2)
                for record in records:  # legacy-removal B3: tau and log w_f (were eta and gamma)
                    self.assertIn("tau", json.dumps(record))
                    self.assertIn("path_log_w_max", json.dumps(record))
                self.assertTrue((cfg.exp_dir / "checkpoints" / "best.pt").exists())
                import yaml

                with (cfg.exp_dir / "config.yaml").open(encoding="utf-8") as handle:
                    reloaded_cfg = TrainConfig.from_dict(yaml.safe_load(handle))
                self.assertEqual(reloaded_cfg.ablation, AblationConfig())
                self.assertEqual(reloaded_cfg.optim.scalar_lr_scale, 10.0)
                fresh = build_model(reloaded_cfg, data)
                Checkpointer(cfg.exp_dir, metric="acc", mode="max").load(fresh, tag="best")
            finally:
                logger.close()


# ----------------------------------------------------------------------------- ablations (T11)
def _with(data, **changes):
    from src.data.structures import PreparedH2SData

    fields = {name: getattr(data, name) for name in PreparedH2SData.__slots__}
    fields.update(changes)
    return PreparedH2SData(**fields)


def _float64_pap():
    from tests.test_factorized_maps import _as_float64

    return _as_float64(_random_pap_data())


class AblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = _float64_pap()
        self.authors = self.data.node_type == 1
        # Authors sit at odd positions of every PAP / PAPAP path: never an endpoint.
        for group in self.data.face_groups:
            self.assertFalse(bool(self.authors[group.face_node_index[:, [0, -1]]].any()))
            self.assertTrue(bool(self.authors[group.face_node_index[:, 1]].all()))
        generator = torch.Generator().manual_seed(9)
        self.author_noise = torch.randn(int(self.authors.sum()), 8, generator=generator,
                                        dtype=torch.float64)

    def _model(self, ablation: dict | None = None, **model):
        return _v2_model(self.data, ablation=ablation, **model).double().eval()

    def _logits(self, model, data=None) -> torch.Tensor:
        with torch.no_grad():
            return model(self.data if data is None else data).logits

    def test_T11_no_path_cells_makes_logits_independent_of_path_data(self) -> None:
        # Legacy-removal B1/B3: the v2 switch `ablation.no_path_cells` is deleted; the v3-B arm
        # `model.arm: no_path_cells` is its counterpart. The v2 checks "no theta_gamma, no path
        # encoder, no gate" (nothing built for the path term) become "no cell in the solved system".
        model = self._model(arm="no_path_cells")
        no_faces = _with(self.data, face_groups=())
        self.assertTrue(torch.equal(self._logits(model), self._logits(model, no_faces)))
        self._logits(model)
        self.assertEqual(model.last_solver_stats["num_cells"], 0.0)
        control = self._model()
        self.assertGreater(float((self._logits(control) - self._logits(control, no_faces))
                                 .abs().max()), 1e-6)

    def test_T11_uniform_gate_gives_w_identically_one(self) -> None:
        # Part H: `uniform_path_gate` is `path_weight_input: uniform`. Legacy-removal B3: under the
        # v3-B weight w = exp(g) (sigmoid_gamma is deleted) the uniform rung gives w = exp(b) for every
        # cell (no input reaches g; was w = 1 under sigmoid_gamma), and a learned g varies across cells.
        model = self._model({"path_weight_input": "uniform"})
        self.assertIsNone(model.path_gate)
        self.assertIsNone(model.path_encoder)
        with torch.no_grad():
            operator = model.encode(self.data.x, self.data.node_type, self.data.edge_index,
                                    self.data.edge_type, self.data.face_groups,
                                    self.data.num_nodes).operator
        constant = torch.exp(model.path_log_weight.detach())
        for weights in operator.path_weights:
            self.assertTrue(torch.equal(weights, constant.expand_as(weights)))
        learned = self._model()
        with torch.no_grad():
            weights = learned.encode(self.data.x, self.data.node_type, self.data.edge_index,
                                     self.data.edge_type, self.data.face_groups,
                                     self.data.num_nodes).operator.path_weights
        all_weights = torch.cat(weights)
        self.assertGreater(float(all_weights.max() - all_weights.min()), 1e-3)

    def test_T11_h0_only_readout_ignores_the_graph(self) -> None:
        moved_x = self.data.x.clone()
        moved_x[self.authors] += 1.0
        variants = [_with(self.data, x=moved_x),
                    _with(self.data, face_groups=())]
        model = self._model({"h0_only_readout": True})
        self.assertEqual(model.readout[0].in_features, 8)
        for variant in variants:
            self.assertTrue(torch.equal(self._logits(model), self._logits(model, variant)))
        control = self._model()
        for variant in variants:
            self.assertGreater(float((self._logits(control) - self._logits(control, variant))
                                     .abs().max()), 1e-6)

    def test_every_ablation_trains_one_step(self) -> None:
        # Legacy-removal (B1-B4, B3): the kept boolean switches, the C2 weight ladder and the v3-B arms
        # (the counterpart of the removed v2 path switches), on the anchored model; the v2 switches,
        # sigmoid_gamma, the gate modes and the dense family were removed with v2.
        from src.configs.config import V3B_ARMS

        switches = [({name: True}, {}) for name in AblationConfig.__dataclass_fields__
                    if name not in ("coupling", "path_weight_input")]
        switches += [({"path_weight_input": arm}, {}) for arm in ("endpoints", "schema", "uniform")]
        switches += [({"coupling": "symmetric"}, {})]
        switches += [({}, {"arm": arm}) for arm in V3B_ARMS]
        for ablation, model_overrides in switches:
            with self.subTest(ablation=ablation, model=model_overrides):
                model = _v2_model(_random_pap_data(), ablation=ablation, **model_overrides)
                output = model(_random_pap_data())
                output.logits.square().sum().backward()
                self.assertTrue(bool(torch.isfinite(output.logits).all()))
                self.assertTrue(all(bool(torch.isfinite(p.grad).all())
                                    for p in model.parameters() if p.grad is not None))


# ----------------------------------------------------------------------------- config, scope
class ConfigTests(unittest.TestCase):
    def test_defaults_are_the_full_model(self) -> None:
        config = TrainConfig.from_dict({})
        self.assertEqual(config.ablation, AblationConfig())
        self.assertFalse(any(getattr(config.ablation, name)
                             for name in AblationConfig.__dataclass_fields__
                             if name not in ("coupling", "path_weight_input")))  # booleans off
        self.assertEqual(config.ablation.path_weight_input, "full")  # Part H: the full g
        self.assertEqual(config.model.path_weight, "exp")  # Part H: w_f = exp(g), no gamma
        self.assertEqual(config.data.path_source, "wisp")  # spec v1.4 E7 (Part I, I2)
        # legacy-removal LR-2 (B2): the default coupling is `symmetric` (was D10's `cooperative`),
        # and the default model is the anchored v3-B model
        self.assertEqual(config.ablation.coupling, "symmetric")
        self.assertEqual((config.model.propagation, config.model.arm), ("anchored", "full"))
        # legacy-removal B3: the v2-only fields (`coupling_gate_init`, `gamma_init`, `eta_init`) and
        # properties (`map_scale`, `path_mode`) are gone; v3-B counterparts: the one diffusion
        # scalar tau starts at tau_init = 1, and the maps have one scale per edge (D1a, no switch).
        for name in ("coupling_gate_init", "gamma_init", "eta_init"):
            self.assertFalse(hasattr(config.model, name), name)
        for name in ("map_scale", "path_mode"):
            self.assertFalse(hasattr(config.ablation, name), name)
        self.assertEqual(config.model.tau_init, 1.0)
        self.assertEqual(config.model.map_family, "factorized")
        self.assertEqual(config.optim.scalar_lr_scale, 10.0)

    def test_unknown_keys_fail_loudly(self) -> None:
        """Old (v1 / AMPSS) settings and typos are rejected in every section."""
        cases = [
            {"discovery": {"enabled": False}},
            {"model": {"architecture": "v2"}},
            {"model": {"eta_0": 0.1}},
            {"model": {"map_scale": "per_side"}},
            {"model": {"ablate_no_path_cells": True}},
            {"data": {"pooling_type": "sheafcov"}},
            {"data": {"closure_scale": 1.0}},
            {"ablation": {"no_path_cell": True}},
            {"optim": {"scalar_lr": 10.0}},
            {"epoch": 3},
            {"data": {"dataset_name": "acm",
                      "metapath_templates": [{"name": "PAP", "node_types": ["paper"],
                                              "weight": 1}]}},
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "Unknown"):
                    TrainConfig.from_dict(raw)

    def test_invalid_values_and_conflicting_ablations_are_rejected(self) -> None:
        # legacy-removal B3: the eta_init / gamma_init ranges are v2-only (removed keys, they raise
        # through tests/test_legacy_removal.py); v3-B counterpart: tau_init must be positive.
        for raw in ({"tau_init": 0.0}, {"tau_init": -1.0},
                    {"checkpointing": "false"}, {"map_family": "lowrank"}):
            with self.subTest(model=raw):
                with self.assertRaises(ValueError):
                    ModelConfig.from_dict(raw)
        # legacy-removal B3: the conflicts among the removed v2 path switches are gone with them;
        # the non-boolean switch check moves to a kept switch, and the kept conflict
        # (path_weight_context without a node input) replaces the switch conflicts.
        for raw in ({"h0_only_readout": "true"},
                    {"path_weight_context": True, "path_weight_input": "uniform"},
                    {"path_weight_input": "nope"}):
            with self.subTest(ablation=raw):
                with self.assertRaises(ValueError):
                    AblationConfig.from_dict(raw)
        for value in (0.0, -1.0):
            with self.subTest(scalar_lr_scale=value):
                with self.assertRaises(ValueError):
                    OptimConfig.from_dict({"scalar_lr_scale": value})


class TwoLatentContextTests(unittest.TestCase):
    """T11-ctx (Part K, K1; spec v1.6 E10) under the default geometry_context: only g reads h0
    (the v2 path-specific maps and pair-attention scores that read [h0, c] were removed by the
    legacy-removal card). Geometry rows are injected directly, because perturbing h0 moves the
    neighbours' context c by design."""

    def setUp(self) -> None:
        self.data = _float64_pap()

    def _model(self, ablation: dict | None = None, **model):
        built = _v2_model(self.data, ablation=ablation, **model).double().eval()
        self.assertIsNotNone(built.relational_context, "default geometry_context is on")
        return built

    def _inputs(self, model):
        with torch.no_grad():
            h0 = model.encode_nodes(self.data.x, self.data.node_type)
            return h0, model.geometry_inputs(h0, self.data.edge_index, self.data.edge_type)

    def _maps(self, model, geometry):
        with torch.no_grad():
            return model.node_to_edge(geometry, self.data.edge_index, self.data.edge_type,
                                      model.relation_embedding.weight).maps

    @staticmethod
    def _moved(model, geometry, nodes, c_only: bool = True):
        moved = geometry.clone()
        start = model.d if c_only else 0
        generator = torch.Generator().manual_seed(11)
        for node in nodes:
            moved[node, start:] += torch.randn(moved.size(1) - start, generator=generator,
                                               dtype=moved.dtype)
        return moved

    def test_b_g_reads_h0_only_unless_the_appendix_switch(self) -> None:
        # legacy-removal B4/B5: the endpoint_edges / free_path_maps subcases were deleted with the
        # v2 switches.
        for appendix, expect_change in ((False, False), (True, True)):
            with self.subTest(path_weight_context=appendix):
                model = self._model({"path_weight_context": appendix})
                h0, geometry = self._inputs(model)
                maps = self._maps(model, geometry)
                cell_nodes = sorted({n for g in self.data.face_groups
                                     for n in g.face_node_index[0].tolist()})
                moved = self._moved(model, geometry, cell_nodes)  # c parts only
                # legacy-removal B3: g's w_f as the anchored model computes them
                groups = tuple(g for g in self.data.face_groups if g.num_faces > 0)
                with torch.no_grad():
                    before = model._cell_weights(h0, geometry, groups)[1]
                    after = model._cell_weights(h0, moved, groups)[1]
                changed = any(not torch.equal(a, b) for a, b in zip(before, after))
                self.assertEqual(changed, expect_change)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
