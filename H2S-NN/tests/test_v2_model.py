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

    def laplacian(self, gamma: float | None, free: bool = False, terms: bool = True):
        from src.models.vertex_path_laplacian import (
            FreeIncidenceOperator, PathCellTerm, VertexPathSheafLaplacian)

        path_terms = []
        if terms:
            for (nodes, edges), weights in zip(self.groups, self.weights):
                if free:
                    incidence = FreeIncidenceOperator(self.free_maps(nodes), nodes)
                else:
                    incidence = self.transports(nodes, edges)
                path_terms.append(PathCellTerm(incidence, weights))
        return VertexPathSheafLaplacian(
            self.maps, self.edge_index, self.num_nodes, tuple(path_terms),
            None if gamma is None else torch.tensor(gamma, dtype=torch.float64),
            checkpoint=False,
        )

    def free_maps(self, nodes: torch.Tensor) -> FactorizedMapOperator:
        g = torch.Generator().manual_seed(int(nodes.sum()) + 7)
        size = nodes.numel()
        return _operators(size, self.d, 2, g, torch.ones(size, dtype=torch.float64),
                          torch.ones(size, dtype=torch.float64), num_groups=1).src


def _matrix(apply, num_nodes: int, d: int) -> torch.Tensor:
    """Dense `[N d, N d]` matrix of a linear map on `[N, d]` node cochains."""
    columns = []
    for index in range(num_nodes * d):
        x = torch.zeros(num_nodes * d, dtype=torch.float64)
        x[index] = 1.0
        columns.append(apply(x.view(num_nodes, d)).reshape(-1))
    return torch.stack(columns, dim=1)


def _dense_delta0(maps: EdgeMapPair, edge_index: torch.Tensor, num_nodes: int,
                  keep: torch.Tensor | None = None) -> torch.Tensor:
    """Independent dense `Delta_0 = B^T B` from materialized maps (rows `F_tgt x_t - F_src x_s`)."""
    F_src, F_tgt = maps.src.materialize(), maps.tgt.materialize()
    d = F_src.size(-1)
    num_edges = F_src.size(0)
    B = torch.zeros(num_edges * d, num_nodes * d, dtype=torch.float64)
    for e in range(num_edges):
        if keep is not None and not bool(keep[e]):
            continue
        s, t = int(edge_index[0, e]), int(edge_index[1, e])
        B[e * d:(e + 1) * d, s * d:(s + 1) * d] -= F_src[e]
        B[e * d:(e + 1) * d, t * d:(t + 1) * d] += F_tgt[e]
    return B.T @ B


def _dense_path_term(transports: torch.Tensor, nodes: torch.Tensor, weights: torch.Tensor,
                     num_nodes: int) -> torch.Tensor:
    """Kron-reduced `sum_f w_f B_f^T (I - 1 1^T/|f|) B_f` from materialized `T_v` `[P, n, d, d]`."""
    num_paths, n, d, _ = transports.shape
    projector = torch.kron(torch.eye(n, dtype=torch.float64)
                           - torch.full((n, n), 1.0 / n, dtype=torch.float64),
                           torch.eye(d, dtype=torch.float64))
    total = torch.zeros(num_nodes * d, num_nodes * d, dtype=torch.float64)
    for p in range(num_paths):
        B = torch.zeros(n * d, num_nodes * d, dtype=torch.float64)
        for i in range(n):
            v = int(nodes[p, i])
            B[i * d:(i + 1) * d, v * d:(v + 1) * d] = transports[p, i]
        total += weights[p] * B.T @ projector @ B
    return total


def _dense_path_part(complex_: _Complex) -> torch.Tensor:
    total = 0.0
    for (nodes, edges), weights in zip(complex_.groups, complex_.weights):
        total = total + _dense_path_term(_dense_transports(complex_.maps, edges), nodes, weights,
                                         complex_.num_nodes)
    return total


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
    def test_T2_path_operator_equals_dense_kron_reduced_reference(self) -> None:
        """Mixed path lengths 2, 3, 4 in one complex; per-edge and per-side scales."""
        for mode in ("per_edge", "per_side"):
            for seed in range(3):
                with self.subTest(mode=mode, seed=seed):
                    complex_ = _Complex(seed=seed, scale_mode=mode)
                    operator = complex_.laplacian(gamma=1.0)
                    got = _matrix(operator.apply_path, complex_.num_nodes, complex_.d)
                    want = _dense_path_part(complex_)
                    self.assertLess(_rel_err(got, want), F64_TOL)
                    full = _matrix(operator.apply, complex_.num_nodes, complex_.d)
                    want_full = _dense_delta0(complex_.maps, complex_.edge_index,
                                              complex_.num_nodes) + want
                    self.assertLess(_rel_err(full, want_full), F64_TOL)

    def test_T3_normalized_operator_is_self_adjoint_and_psd(self) -> None:
        for mode, values in (("per_edge", (EPS, 50.0 + EPS)), ("per_side", (EPS, 50.0 + EPS)),
                             ("per_edge", None)):
            with self.subTest(mode=mode, values=values):
                complex_ = _Complex(seed=11, scale_mode=mode, scale_values=values)
                operator = complex_.laplacian(gamma=2.5)
                g = torch.Generator().manual_seed(1)
                x = torch.randn(complex_.num_nodes, complex_.d, generator=g, dtype=torch.float64)
                y = torch.randn(complex_.num_nodes, complex_.d, generator=g, dtype=torch.float64)
                lhs = float((x * operator.apply_normalized(y)).sum())
                rhs = float((operator.apply_normalized(x) * y).sum())
                self.assertLess(abs(lhs - rhs), F64_TOL * max(1.0, abs(lhs)))
                matrix = _matrix(operator.apply_normalized, complex_.num_nodes, complex_.d)
                self.assertLess(float((matrix - matrix.T).abs().max()), F64_TOL)
                eigenvalues = torch.linalg.eigvalsh(0.5 * (matrix + matrix.T))
                self.assertGreaterEqual(float(eigenvalues.min()), -1e-12)
                quadratic = float((x * operator.apply_normalized(x)).sum())
                self.assertGreaterEqual(quadratic, -1e-12 * float(x.square().sum()))

    def test_D_is_the_per_incidence_degree_of_the_operators_own_terms(self) -> None:
        """`D_v I` equals the dense diagonal block of `L` at every node without a self-loop;
        at the self-loop node it equals the block without that edge plus `c_src^2 + c_tgt^2`."""
        for mode in ("per_edge", "per_side"):
            with self.subTest(mode=mode):
                complex_ = _Complex(seed=5, scale_mode=mode, scale_values=(EPS, 50.0 + EPS))
                gamma = 0.7
                operator = complex_.laplacian(gamma=gamma)
                degree = operator.degree()
                d = complex_.d
                loop_edge = complex_.edge_index.size(1) - 1
                keep = torch.ones(complex_.edge_index.size(1), dtype=torch.bool)
                keep[loop_edge] = False
                dense = (_dense_delta0(complex_.maps, complex_.edge_index, complex_.num_nodes, keep)
                         + gamma * _dense_path_part(complex_))
                eye = torch.eye(d, dtype=torch.float64)
                for v in range(complex_.num_nodes):
                    block = dense[v * d:(v + 1) * d, v * d:(v + 1) * d]
                    if v == complex_.self_loop_node:
                        block = block + (complex_.maps.src.scale[loop_edge] ** 2
                                         + complex_.maps.tgt.scale[loop_edge] ** 2) * eye
                    self.assertLess(_rel_err(degree[v] * eye, block), F64_TOL)
                self.assertEqual(float(degree[complex_.num_nodes - 1]), 0.0)

    def test_T5_normalized_spectrum_is_bounded_by_two(self) -> None:
        """60 fixtures: c in {eps, 50} (40 per-edge, 20 per-side), random w_f, gamma."""
        max_normalized = 0.0
        max_unnormalized = 0.0
        for seed in range(60):
            mode = "per_edge" if seed < 40 else "per_side"
            complex_ = _Complex(seed=100 + seed, scale_mode=mode,
                                scale_values=(EPS, 50.0 + EPS))
            g = torch.Generator().manual_seed(seed)
            gamma = float(10 ** (4 * torch.rand(1, generator=g) - 2))
            operator = complex_.laplacian(gamma=gamma)
            normalized = _matrix(operator.apply_normalized, complex_.num_nodes, complex_.d)
            unnormalized = _matrix(operator.apply, complex_.num_nodes, complex_.d)
            lam = float(torch.linalg.eigvalsh(0.5 * (normalized + normalized.T)).max())
            self.assertLessEqual(lam, 2.0 + 1e-9, f"seed {seed}")
            isolated = complex_.num_nodes - 1
            rows = normalized[isolated * complex_.d:(isolated + 1) * complex_.d]
            self.assertEqual(float(rows.abs().max()), 0.0)
            max_normalized = max(max_normalized, lam)
            max_unnormalized = max(max_unnormalized,
                                   float(torch.linalg.eigvalsh(unnormalized).max()))
        self.assertGreater(max_unnormalized, 2.0)  # negative control: normalization matters
        self.assertGreater(max_normalized, 1.0)    # the bound is not met vacuously

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


def _tree_complex(seed: int, num_nodes: int = 12, d: int = 3,
                  scale_values: tuple[float, ...] | None = (EPS, 50.0 + EPS)):
    """Random rooted tree, edges stored parent -> child, and root-ward chains of length 2
    and 3 along tree edges. Returns (complex-like object, explicit global sections [d, N, d])."""
    g = torch.Generator().manual_seed(seed)
    parent = [-1] + [int(torch.randint(v, (1,), generator=g)) for v in range(1, num_nodes)]
    edge_of = {v: v - 1 for v in range(1, num_nodes)}  # edge id of (parent[v] -> v)
    edge_index = torch.tensor([[parent[v] for v in range(1, num_nodes)],
                               list(range(1, num_nodes))])
    chains: dict[int, list[tuple[list[int], list[int]]]] = {2: [], 3: []}
    for v in range(num_nodes):
        for length in (2, 3):
            nodes = [v]
            while len(nodes) < length + 1 and parent[nodes[0]] >= 0:
                nodes.insert(0, parent[nodes[0]])
            if len(nodes) == length + 1:
                chains[length].append((nodes, [edge_of[u] for u in nodes[1:]]))
    complex_ = _Complex.__new__(_Complex)
    complex_.num_nodes = num_nodes
    complex_.d = d
    complex_.edge_index = edge_index
    complex_.scale_mode = "per_edge"
    num_edges = num_nodes - 1
    if scale_values is None:
        c = 0.5 + 1.5 * torch.rand(num_edges, generator=g, dtype=torch.float64)
    else:
        c = torch.tensor(scale_values, dtype=torch.float64)[
            torch.randint(len(scale_values), (num_edges,), generator=g)]
    complex_.maps = _operators(num_edges, d, 2, g, c, c)
    complex_.groups = [(torch.tensor([n for n, _ in chains[L]]),
                        torch.tensor([e for _, e in chains[L]])) for L in (2, 3) if chains[L]]
    complex_.weights = [torch.rand(nodes.size(0), generator=g, dtype=torch.float64)
                        for nodes, _ in complex_.groups]
    # Global sections: x_root free; delta_e x = F_tgt x_child - F_src x_parent = 0.
    F_src, F_tgt = complex_.maps.src.materialize(), complex_.maps.tgt.materialize()
    sections = torch.zeros(d, num_nodes, d, dtype=torch.float64)
    sections[:, 0] = torch.eye(d, dtype=torch.float64)
    for v in range(1, num_nodes):  # parents have smaller ids
        e = edge_of[v]
        sections[:, v] = (torch.linalg.solve(F_tgt[e], F_src[e] @ sections[:, parent[v]].T)).T
    return complex_, sections


def _kernel_dimension(matrix: torch.Tensor, threshold: float = 1e-9) -> int:
    return int((torch.linalg.eigvalsh(0.5 * (matrix + matrix.T)).abs() < threshold).sum())


class SectionPreservationTests(unittest.TestCase):
    """T4. Eigenvalue threshold 1e-9 (stated before running), on normalized operators."""

    def test_T4_sections_are_preserved_and_the_kernel_is_unchanged(self) -> None:
        for seed in range(3):
            with self.subTest(seed=seed):
                complex_, sections = _tree_complex(seed)
                self.assertGreater(sum(n.size(0) for n, _ in complex_.groups), 3)
                edge_only = complex_.laplacian(gamma=None, terms=False)
                full = complex_.laplacian(gamma=3.0)
                for section in sections:
                    scale = float(section.abs().max())
                    self.assertLess(float(edge_only.apply(section).abs().max()) / scale, F64_TOL)
                    self.assertLess(float(full.apply_path(section).abs().max()) / scale, F64_TOL)
                N, d = complex_.num_nodes, complex_.d
                kernel_edges = _kernel_dimension(_matrix(edge_only.apply_normalized, N, d))
                kernel_full = _kernel_dimension(_matrix(full.apply_normalized, N, d))
                self.assertEqual(kernel_edges, d)  # connected tree: H^0 has dimension d
                self.assertEqual(kernel_full, kernel_edges)

    def test_T4_negative_control_free_maps_shrink_the_kernel(self) -> None:
        for seed in range(3):
            with self.subTest(seed=seed):
                complex_, sections = _tree_complex(seed)
                free = complex_.laplacian(gamma=3.0, free=True)
                N, d = complex_.num_nodes, complex_.d
                kernel_free = _kernel_dimension(_matrix(free.apply_normalized, N, d))
                self.assertLess(kernel_free, d)
                residual = max(float(free.apply_path(s).abs().max() / s.abs().max())
                               for s in sections)
                self.assertGreater(residual, 1e-3)


class ScaleGaugeTests(unittest.TestCase):
    def test_T14_node_rescaled_per_side_scales_give_the_same_normalized_delta0(self) -> None:
        from src.models.vertex_path_laplacian import VertexPathSheafLaplacian

        complex_ = _Complex(seed=21, extra_edges=8)  # extra edges close cycles
        g = torch.Generator().manual_seed(0)
        c = complex_.maps.src.scale
        node_gauge = 0.5 + 1.5 * torch.rand(complex_.num_nodes, generator=g, dtype=torch.float64)
        src_idx, tgt_idx = complex_.edge_index

        def normalized_delta0(src_scale: torch.Tensor, tgt_scale: torch.Tensor) -> torch.Tensor:
            maps = EdgeMapPair(
                FactorizedMapOperator(src_scale, complex_.maps.src.reflectors,
                                      complex_.maps.src.rotations, complex_.maps.src.group_ids),
                FactorizedMapOperator(tgt_scale, complex_.maps.tgt.reflectors,
                                      complex_.maps.tgt.rotations, complex_.maps.tgt.group_ids),
            )
            operator = VertexPathSheafLaplacian(maps, complex_.edge_index, complex_.num_nodes,
                                                (), None, checkpoint=False)
            return _matrix(operator.apply_normalized, complex_.num_nodes, complex_.d)

        per_edge = normalized_delta0(c, c)
        gauged = normalized_delta0(node_gauge[src_idx] * c, node_gauge[tgt_idx] * c)
        generic = normalized_delta0(
            0.5 + 1.5 * torch.rand(c.shape, generator=g, dtype=torch.float64),
            0.5 + 1.5 * torch.rand(c.shape, generator=g, dtype=torch.float64))
        self.assertLess(float((gauged - per_edge).abs().max()), F64_TOL)
        self.assertGreater(float((generic - per_edge).abs().max()), 1e-3)


# ----------------------------------------------------------------------------- v2 model (T1, T6, T8-T10, T13)
def _v2_config(ablation: dict | None = None, **model: object) -> TrainConfig:
    base = {"map_family": "factorized", "d": 8, "d_r": 4, "K": 2,
            "classifier_hidden_dim": 16, "face_mlp_hidden_dim": 16, "edge_mlp_hidden_dim": 16}
    base.update(model)
    return TrainConfig.from_dict({"device": "cpu", "checkpoint_metric": "acc", "model": base,
                                  "ablation": ablation or {}})


def _v2_model(data, seed: int = 0, ablation: dict | None = None, **model: object):
    from src.models.h2snn_model import build_model

    torch.manual_seed(seed)
    return build_model(_v2_config(ablation, **model), data)


def _a1_p_a2():
    """Path A1 -> P -> A2 (nodes 0, 1, 2), one 2-cell on it, plus a pendant node 3 on A2."""
    from src.data.structures import FaceGroup

    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_type = torch.tensor([0, 1, 0])
    nodes = torch.tensor([[0, 1, 2]])
    edges = torch.tensor([[0, 1]])
    group = FaceGroup(template_name="APA", node_type_sequence=("a", "p", "a"),
                      face_node_index=nodes, face_edge_index=edges,
                      relation_index=edge_type[edges], face_edge_sign=torch.ones(1, 2))
    return edge_index, edge_type, (group,)


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

    def test_T6_path_term_reaches_the_far_endpoint_and_the_intermediate_node(self) -> None:
        """One step with fixed maps: d x1(A1)/d x0(A2) != 0 iff gamma > 0 (nothing else links
        A1 to A2 in one step), and d x1(A1)/d x0(P) changes with gamma."""
        from src.models.vertex_path_laplacian import VertexPathSheafLaplacian

        edge_index, edge_type, groups = _a1_p_a2()
        dims = ModelDims(in_channels=5, num_node_types=2, num_relations=2, num_classes=2,
                         face_lengths=(2,))
        torch.manual_seed(0)
        model = H2SNN(_v2_config(), dims).double()
        h0 = torch.randn(4, 8, dtype=torch.float64)
        operator = model.build_operator(h0, edge_index, edge_type, groups, num_nodes=4)
        self.assertEqual(len(operator.laplacian.path_terms), 1)

        def block(gamma: float, row: int, column: int) -> torch.Tensor:
            laplacian = VertexPathSheafLaplacian(
                operator.maps, edge_index, 4, operator.laplacian.path_terms,
                torch.tensor(gamma, dtype=torch.float64), checkpoint=False)

            def step(x0: torch.Tensor) -> torch.Tensor:
                return x0 - operator.eta * laplacian.apply_normalized(x0)

            jacobian = torch.autograd.functional.jacobian(step, h0.detach().clone())
            return jacobian[row, :, column, :]

        far_zero = block(0.0, 0, 2)
        self.assertEqual(float(far_zero.abs().max()), 0.0)
        for gamma in (0.3, 2.0):
            self.assertGreater(float(block(gamma, 0, 2).abs().max()), 1e-8)
        middle_change = (block(0.3, 0, 1) - block(2.0, 0, 1)).abs().max()
        self.assertGreater(float(middle_change), 1e-8)
        # The clause above also holds through D alone (gamma enters D_{A1}, D_P), so it does
        # not prove intermediate reach by itself: the path term's own blocks must link P and A1
        # in both directions.
        path_only = torch.autograd.functional.jacobian(operator.laplacian.apply_path, h0.clone())
        self.assertGreater(float(path_only[0, :, 1, :].abs().max()), 1e-8)  # A1 <- P
        self.assertGreater(float(path_only[1, :, 0, :].abs().max()), 1e-8)  # P <- A1
        # The model's own gamma also reaches A2 (and the ablation does not).
        jacobian = torch.autograd.functional.jacobian(
            lambda x0: x0 - operator.eta * operator.laplacian.apply_normalized(x0), h0.clone())
        self.assertGreater(float(jacobian[0, :, 2, :].abs().max()), 1e-8)
        torch.manual_seed(0)
        ablated = H2SNN(_v2_config({"no_path_cells": True}), dims).double()
        ablated_operator = ablated.build_operator(h0, edge_index, edge_type, groups, 4)
        jacobian = torch.autograd.functional.jacobian(
            lambda x0: x0 - ablated_operator.eta
            * ablated_operator.laplacian.apply_normalized(x0), h0.clone())
        self.assertEqual(float(jacobian[0, :, 2, :].abs().max()), 0.0)

    def test_T8_no_dense_map_tensor_in_v2_forward_or_backward(self) -> None:
        """Flags `[M|F|F_g|F_g(L+1), d, d]` shapes and any tensor with >= min(M, F) d^2 / 4
        elements. Control: the dense map family ([M, d, d] maps) is flagged in both passes."""
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
        dense = record(_v2_model(data, d=d, face_mlp_hidden_dim=128, edge_mlp_hidden_dim=128,
                                 map_family="dense"))
        self.assertTrue(hits(dense, "forward"))
        self.assertTrue(hits(dense, "backward"))

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
        h0 = model.encode_nodes(data.x, data.node_type)
        operator = model.build_operator(h0, data.edge_index, data.edge_type, data.face_groups,
                                        data.num_nodes)
        self.assertEqual(operator.laplacian.src_idx.numel(), M)  # no closure edges in Delta_0

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
        groups = ["theta_eta", "theta_gamma", "path_gate", "path_encoder",
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
        self.assertEqual(float(model.path_gate.bias.abs().max()), 0.0)
        output = model(data)
        self.assertLess(abs(float(output.path_gate_mean) - 0.5), 0.1)
        self.assertLess(float(output.path_gate_max), 0.75)

    def test_T13_dense_and_factorized_families_match_on_materialized_maps(self) -> None:
        import copy

        from src.models.map_operators import DenseMapOperator
        from src.models.sheaf_maps import FactorizedNodeToEdgeSheafOutput
        from tests.test_factorized_maps import _as_float64

        class Materialized(torch.nn.Module):
            def __init__(self, factorized: FactorizedNodeToEdgeSheaf) -> None:
                super().__init__()
                self.factorized = factorized

            def forward(self, h, edge_index, edge_type, relation_embeddings):
                maps = self.factorized(h, edge_index, edge_type, relation_embeddings).maps
                return FactorizedNodeToEdgeSheafOutput(maps=EdgeMapPair(
                    DenseMapOperator(maps.src.materialize(), maps.src.scale),
                    DenseMapOperator(maps.tgt.materialize(), maps.tgt.scale)))

        data = _as_float64(_random_pap_data())
        for map_scale in ("per_edge", "per_side"):
            with self.subTest(map_scale=map_scale):
                factorized = _v2_model(
                    data, ablation={"per_side_scale": map_scale == "per_side"}
                ).double().eval()
                dense = copy.deepcopy(factorized)
                dense.node_to_edge = Materialized(dense.node_to_edge)
                outputs = []
                for model in (factorized, dense):
                    output = model(data)
                    output.logits.square().sum().backward()
                    outputs.append(output)
                self.assertLess(_rel_err(outputs[1].logits, outputs[0].logits), F64_TOL)
                grads_a = {n: p.grad for n, p in factorized.named_parameters()}
                grads_b = {n.replace("node_to_edge.factorized.", "node_to_edge."): p.grad
                           for n, p in dense.named_parameters()}
                self.assertEqual(set(grads_a), set(grads_b))
                for name, grad in grads_a.items():
                    self.assertLess(_rel_err(grads_b[name], grad), F64_TOL, name)
        dense_family = _v2_model(_random_pap_data(), map_family="dense")
        output = dense_family(_random_pap_data())
        output.logits.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(output.logits).all()))
        self.assertGreater(float(dense_family.node_to_edge.src_mlps[0][0].weight.grad.abs().sum()), 0)
        self.assertGreater(
            float(dense_family.node_to_edge.edge_scale_mlp[0].weight.grad.abs().sum()), 0)

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
                "model": {"d": 8, "d_r": 4, "K": 2, "classifier_hidden_dim": 16},
                # Hand-built fixture without a fact layer: nothing to resample.
                "data": {"resample_paths": False},
                "optim": {"lr": 1e-3, "weight_decay": 1e-4, "scalar_lr_scale": 10.0}})
            torch.manual_seed(0)
            model = build_model(cfg, data)
            logger = ExperimentLogger(cfg.exp_dir)
            checkpointer = Checkpointer(cfg.exp_dir, metric="acc", mode="max")
            save_config(cfg, cfg.exp_dir / "config.yaml")
            optimizer = build_optimizer(model, cfg)
            # theta_eta, theta_gamma and the four gate-head tensors (plan §12.4) in their own
            # group: lr x 10, no weight decay.
            scalar_group = [g for g in optimizer.param_groups
                            if any(p is model.theta_eta for p in g["params"])]
            self.assertEqual(len(scalar_group), 1)
            self.assertEqual(len(scalar_group[0]["params"]), 6)
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
                for record in records:
                    self.assertIn("eta", json.dumps(record))
                    self.assertIn("gamma", json.dumps(record))
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
        model = self._model({"no_path_cells": True})
        self.assertIsNone(model.theta_gamma)
        self.assertIsNone(model.path_encoder)
        self.assertIsNone(model.path_gate)
        no_faces = _with(self.data, face_groups=())
        self.assertTrue(torch.equal(self._logits(model), self._logits(model, no_faces)))
        control = self._model()
        self.assertGreater(float((self._logits(control) - self._logits(control, no_faces))
                                 .abs().max()), 1e-6)

    def test_T11_uniform_gate_gives_w_identically_one(self) -> None:
        model = self._model({"uniform_path_gate": True})
        self.assertIsNone(model.path_gate)
        self.assertIsNone(model.path_encoder)
        h0 = model.encode_nodes(self.data.x, self.data.node_type)
        operator = model.build_operator(h0, self.data.edge_index, self.data.edge_type,
                                        self.data.face_groups, self.data.num_nodes)
        for weights in operator.path_weights:
            self.assertTrue(torch.equal(weights, torch.ones_like(weights)))
        learned = self._model()
        h0 = learned.encode_nodes(self.data.x, self.data.node_type)
        weights = learned.build_operator(h0, self.data.edge_index, self.data.edge_type,
                                         self.data.face_groups, self.data.num_nodes).path_weights
        self.assertGreater(float((torch.cat(weights) - 1).abs().min()), 0.1)

    def test_T11_endpoint_edges_ignore_intermediate_nodes_in_the_path_term(self) -> None:
        # Geometry from h0 alone: under the D12 geometry_context the endpoints' context c is a
        # per-node neighbourhood mean that includes intermediates (not path-specific), so moving
        # an intermediate's h0 moves c by design; the path-specific claim is tested here.
        for mode, expect_equal in (("endpoint_edges", True), ("cells", False)):
            with self.subTest(path_mode=mode):
                model = self._model({"endpoint_edges": mode == "endpoint_edges"},
                                    geometry_context=False)
                h0 = model.encode_nodes(self.data.x, self.data.node_type)
                maps = model.node_to_edge(h0, self.data.edge_index, self.data.edge_type,
                                          model.relation_embedding.weight).maps
                h0_moved = h0.clone()
                h0_moved[self.authors] += self.author_noise
                x = torch.randn(self.data.num_nodes, 8, dtype=torch.float64,
                                generator=torch.Generator().manual_seed(3))
                x_moved = x.clone()
                x_moved[self.authors] += self.author_noise
                N = self.data.num_nodes
                with torch.no_grad():
                    terms, _ = model.path_terms(h0, maps, self.data.edge_index,
                                                self.data.face_groups)
                    terms_moved, _ = model.path_terms(h0_moved, maps, self.data.edge_index,
                                                      self.data.face_groups)
                    base = sum(term.apply(x, N) for term in terms)
                    via_h0 = sum(term.apply(x, N) for term in terms_moved)
                    via_x = sum(term.apply(x_moved, N) for term in terms)
                    diag = sum(term.diagonal(N) for term in terms)
                    diag_moved = sum(term.diagonal(N) for term in terms_moved)
                self.assertGreater(float(base.abs().max()), 1e-3)  # the term is not empty
                if expect_equal:
                    self.assertTrue(torch.equal(base, via_h0))
                    self.assertTrue(torch.equal(base, via_x))
                    self.assertTrue(torch.equal(diag, diag_moved))
                else:  # control: the 2-cell reads the intermediate nodes
                    self.assertGreater(float((base - via_h0).abs().max()), 1e-6)
                    self.assertGreater(float((base - via_x).abs().max()), 1e-6)

    def test_T11_holonomy_edges_equal_the_dense_closure_edge(self) -> None:
        from src.models.vertex_path_laplacian import HolonomyEdgeTerm

        for map_scale in ("per_edge", "per_side"):
            with self.subTest(map_scale=map_scale):
                model = self._model({"holonomy_edges": True,
                                     "per_side_scale": map_scale == "per_side"})
                h0 = model.encode_nodes(self.data.x, self.data.node_type)
                with torch.no_grad():
                    operator = model.build_operator(h0, self.data.edge_index, self.data.edge_type,
                                                    self.data.face_groups, self.data.num_nodes)
                N, d = self.data.num_nodes, 8
                for term, group, weights in zip(operator.laplacian.path_terms,
                                                self.data.face_groups, operator.path_weights):
                    self.assertIsInstance(term, HolonomyEdgeTerm)
                    holonomy = _dense_transports(operator.maps, group.face_edge_index)[:, -1]
                    want = torch.zeros(N * d, N * d, dtype=torch.float64)
                    for p in range(group.num_faces):
                        B = torch.zeros(d, N * d, dtype=torch.float64)
                        v0, vL = int(group.face_node_index[p, 0]), int(group.face_node_index[p, -1])
                        B[:, v0 * d:(v0 + 1) * d] += torch.eye(d, dtype=torch.float64)
                        B[:, vL * d:(vL + 1) * d] -= holonomy[p]
                        want += weights[p] * B.T @ B
                    with torch.no_grad():
                        got = _matrix(lambda x: term.apply(x, N), N, d)
                        diagonal = term.diagonal(N)
                    self.assertLess(_rel_err(got, want), F64_TOL)
                    blocks = torch.stack([want[v * d:(v + 1) * d, v * d:(v + 1) * d].diagonal()
                                          for v in range(N)])
                    self.assertLess(_rel_err(diagonal.unsqueeze(-1).expand_as(blocks), blocks),
                                    F64_TOL)

    def test_T11_free_maps_are_orthogonal_and_per_incidence(self) -> None:
        from src.models.vertex_path_laplacian import FreeIncidenceOperator

        model = self._model({"free_path_maps": True})
        h0 = model.encode_nodes(self.data.x, self.data.node_type)
        with torch.no_grad():
            operator = model.build_operator(h0, self.data.edge_index, self.data.edge_type,
                                            self.data.face_groups, self.data.num_nodes)
        for term in operator.laplacian.path_terms:
            incidence = term.incidence
            self.assertIsInstance(incidence, FreeIncidenceOperator)
            maps = incidence.maps.materialize()  # scale 1: the orthogonal factor itself
            self.assertLess(_orthogonality_error(maps), F64_TOL)
            nodes = incidence.face_node_index.reshape(-1)
            repeated = int(torch.mode(nodes).values)
            rows = (nodes == repeated).nonzero().view(-1)
            self.assertGreater(rows.numel(), 1)
            self.assertGreater(float((maps[rows[0]] - maps[rows[1]]).abs().max()), 1e-3)
        model(self.data).logits.square().sum().backward()
        self.assertGreater(float(model.free_maps.reflector_mlp[0].weight.grad.abs().sum()), 0.0)

    def test_T11_per_side_scales_reach_transports_and_degree(self) -> None:
        model = self._model({"per_side_scale": True})
        h0 = model.encode_nodes(self.data.x, self.data.node_type)
        with torch.no_grad():
            operator = model.build_operator(h0, self.data.edge_index, self.data.edge_type,
                                            self.data.face_groups, self.data.num_nodes)
        self.assertIsNot(operator.maps.src.scale, operator.maps.tgt.scale)
        for term in operator.laplacian.path_terms:
            self.assertFalse(term.incidence.shared_scale)
            squares = term.incidence.squared_scales()
            self.assertGreater(float((squares - 1).abs().max()), 1e-3)
            self.assertTrue(torch.equal(squares[:, 0], torch.ones_like(squares[:, 0])))
        per_edge = self._model()
        with torch.no_grad():
            h0 = per_edge.encode_nodes(self.data.x, self.data.node_type)
            terms = per_edge.build_operator(h0, self.data.edge_index, self.data.edge_type,
                                            self.data.face_groups,
                                            self.data.num_nodes).laplacian.path_terms
        for term in terms:
            self.assertTrue(torch.equal(term.incidence.squared_scales(),
                                        torch.ones_like(term.incidence.squared_scales())))

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
        # boolean switches (D10's `coupling` is a mode, covered below)
        # identity_maps (D12 G0b diagnostic) is defined only together with no_path_cells.
        requires = {"identity_maps": {"no_path_cells": True}}
        switches = [({name: True, **requires.get(name, {})}, {})
                    for name in AblationConfig.__dataclass_fields__ if name != "coupling"]
        switches += [({"coupling": mode}, {}) for mode in ("cooperative", "listen_only",
                                                           "symmetric")]
        switches.append(({}, {"map_family": "dense"}))
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
                             if name != "coupling"))  # every boolean switch is off
        self.assertEqual(config.ablation.coupling, "cooperative")  # D10 default
        self.assertEqual(config.model.coupling_gate_init, 0.6)
        self.assertEqual(config.ablation.map_scale, "per_edge")
        self.assertEqual(config.ablation.path_mode, "cells")
        self.assertEqual(config.model.map_family, "factorized")
        self.assertEqual((config.model.gamma_init, config.model.eta_init), (1.0, 0.5))
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
        for raw in ({"eta_init": 0.0}, {"eta_init": 1.0}, {"gamma_init": 0.0},
                    {"checkpointing": "false"}, {"map_family": "lowrank"}):
            with self.subTest(model=raw):
                with self.assertRaises(ValueError):
                    ModelConfig.from_dict(raw)
        for raw in ({"no_path_cells": "true"}, {"endpoint_edges": True, "holonomy_edges": True},
                    {"free_path_maps": True, "endpoint_edges": True},
                    {"free_path_maps": True, "per_side_scale": True},
                    {"no_path_cells": True, "uniform_path_gate": True},
                    {"no_path_cells": True, "free_path_maps": True}):
            with self.subTest(ablation=raw):
                with self.assertRaises(ValueError):
                    AblationConfig.from_dict(raw)
        for value in (0.0, -1.0):
            with self.subTest(scalar_lr_scale=value):
                with self.assertRaises(ValueError):
                    OptimConfig.from_dict({"scalar_lr_scale": value})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
