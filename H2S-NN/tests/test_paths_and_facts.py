"""Fact layer, path-cell builder and O(L) path term (task card D7-D9, plan §9).

Test ids follow the task card (T1-T16; T3, T7 and T15 were dropped by the user).
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.data.face_geometry import validate_face_orientation
from src.models.map_operators import DenseMapOperator, EdgeMapPair, PathTransportOperator
from src.models.vertex_path_laplacian import HolonomyEdgeTerm, PathCellTerm
from tests.test_v2_model import _chain_operators

F64_TOL = 1e-10


# ----------------------------------------------------------------------------- fixtures
class SignedPaths:
    """Overlapping simple paths of mixed length on `num_nodes` nodes, one new fact per hop.

    With `signed`, each hop's sign is drawn uniformly from {+1, -1} and the fact is stored
    `v_{j-1} -> v_j` (+1) or `v_j -> v_{j-1}` (-1); otherwise every hop is +1.
    """

    def __init__(self, num_nodes: int, lengths: tuple[int, ...], paths_per_length: int,
                 generator: torch.Generator, signed: bool) -> None:
        self.num_nodes = num_nodes
        self.nodes, self.edges, self.signs = [], [], []
        pairs = []
        next_edge = 0
        for length in lengths:
            nodes = torch.stack([torch.randperm(num_nodes, generator=generator)[: length + 1]
                                 for _ in range(paths_per_length)])
            edges = torch.arange(next_edge, next_edge + paths_per_length * length)
            edges = edges.view(-1, length)
            next_edge += paths_per_length * length
            if signed:
                signs = torch.randint(0, 2, (paths_per_length, length), generator=generator)
                signs = (signs * 2 - 1).float()
            else:
                signs = torch.ones(paths_per_length, length)
            forward = signs > 0
            src = torch.where(forward, nodes[:, :-1], nodes[:, 1:])
            tgt = torch.where(forward, nodes[:, 1:], nodes[:, :-1])
            pairs.append(torch.stack([src.reshape(-1), tgt.reshape(-1)]))
            self.nodes.append(nodes)
            self.edges.append(edges)
            self.signs.append(signs)
        self.edge_index = torch.cat(pairs, dim=1)

    def transports(self, maps: EdgeMapPair, shared: bool) -> list[PathTransportOperator]:
        return [PathTransportOperator(maps.src, maps.tgt, nodes, edges, self.edge_index,
                                      shared_scale=shared, face_edge_sign=signs)
                for nodes, edges, signs in zip(self.nodes, self.edges, self.signs)]


def _as_dense_family(maps: EdgeMapPair) -> EdgeMapPair:
    """The same conformal maps behind the dense operator interface (dense map family)."""
    return EdgeMapPair(
        src=DenseMapOperator(maps.src.materialize(), maps.src.scale),
        tgt=DenseMapOperator(maps.tgt.materialize(), maps.tgt.scale),
    )


def _signed_dense_transports(maps: EdgeMapPair, edges: torch.Tensor,
                             signs: torch.Tensor) -> torch.Tensor:
    """Independent reference `[P, L+1, d, d]`: `T_{v_i} = prod_{j<=i} Q_j` with
    `Q_j = inv(F_src) F_tgt` (+1) or `inv(F_tgt) F_src` (-1), general inverse."""
    F_src = maps.src.materialize()[edges]
    F_tgt = maps.tgt.materialize()[edges]
    num_paths, length, d, _ = F_src.shape
    forward = (signs > 0).view(num_paths, length, 1, 1)
    near = torch.where(forward, F_src, F_tgt)
    far = torch.where(forward, F_tgt, F_src)
    transports = [torch.eye(d, dtype=F_src.dtype).expand(num_paths, d, d)]
    for j in range(length):
        transports.append(transports[-1] @ torch.linalg.inv(near[:, j]) @ far[:, j])
    return torch.stack(transports, dim=1)


def _materialize(transport: PathTransportOperator, num_nodes: int, d: int) -> torch.Tensor:
    """`[P, L+1, d, d]` matrices `T_{v_i}` read off column by column through `apply`."""
    columns = []
    for column in range(d):
        x = torch.zeros(num_nodes, d, dtype=torch.float64)
        x[:, column] = 1.0
        columns.append(transport.apply(x))
    return torch.stack(columns, dim=-1)


def _dense_path_laplacian(maps: EdgeMapPair, paths: SignedPaths, weights: list[torch.Tensor],
                          d: int) -> torch.Tensor:
    """Reference `sum_f w_f B_f^T (I - 1 1^T / n) B_f` as an explicit `[N d, N d]` matrix."""
    num_nodes = paths.num_nodes
    output = torch.zeros(num_nodes * d, num_nodes * d, dtype=torch.float64)
    for nodes, edges, signs, w in zip(paths.nodes, paths.edges, paths.signs, weights):
        transports = _signed_dense_transports(maps, edges, signs)
        n = nodes.size(1)
        centering = torch.eye(n, dtype=torch.float64) - 1.0 / n
        projector = torch.kron(centering, torch.eye(d, dtype=torch.float64))
        for p in range(nodes.size(0)):
            B = torch.zeros(n * d, num_nodes * d, dtype=torch.float64)
            for i in range(n):
                v = int(nodes[p, i])
                B[i * d:(i + 1) * d, v * d:(v + 1) * d] = transports[p, i]
            output += w[p] * B.T @ projector @ B
    return output


def _count_steps(transport: PathTransportOperator) -> list[int]:
    """Wrap `_step`/`_step_T` to count `Q_j` applications per path: a call on `[P, d]`
    counts 1, a call on a `[P, m, d]` slab of positions counts `m`."""
    counter = [0]
    step, step_T = transport._step, transport._step_T

    def counted(fn):
        def inner(j, y):
            counter[0] += y.numel() // (y.size(0) * y.size(-1))
            return fn(j, y)
        return inner

    transport._step, transport._step_T = counted(step), counted(step_T)
    return counter


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1.0))


MAP_SETTINGS = [(shared, dense) for shared in (True, False) for dense in (False, True)]


def _setting(seed: int, shared: bool, dense: bool, signed: bool, d: int = 4):
    generator = torch.Generator().manual_seed(seed)
    paths = SignedPaths(9, (2, 3, 4), 4, generator, signed)
    maps = _chain_operators(paths.edge_index.size(1), k=2, d=d, seed=seed,
                            dtype=torch.float64, shared=shared)
    operators = _as_dense_family(maps) if dense else maps
    return generator, paths, maps, operators


# ----------------------------------------------------------------------------- D9 (T1)
class OLPathTermTests(unittest.TestCase):
    """T1: the O(L) `PathCellTerm.apply` equals the dense reference."""

    def _check(self, shared: bool, dense_family: bool, seed: int, signed: bool = False) -> float:
        d = 4
        generator, paths, maps, operators = _setting(seed, shared, dense_family, signed, d)
        weights = [torch.rand(nodes.size(0), generator=generator, dtype=torch.float64) * 3.0
                   for nodes in paths.nodes]
        x = torch.randn(paths.num_nodes, d, generator=generator, dtype=torch.float64)
        output = torch.zeros_like(x)
        for transport, w in zip(paths.transports(operators, shared), weights):
            output += PathCellTerm(transport, w).apply(x, paths.num_nodes)
        reference = _dense_path_laplacian(maps, paths, weights, d)
        expected = (reference @ x.reshape(-1)).view(paths.num_nodes, d)
        return _rel(output, expected)

    def test_T1_matches_dense_reference_mixed_lengths_scales_families_and_signs(self) -> None:
        for shared, dense_family in MAP_SETTINGS:
            for signed in (False, True):
                for seed in range(3):
                    with self.subTest(shared=shared, dense_family=dense_family, signed=signed,
                                      seed=seed):
                        self.assertLess(self._check(shared, dense_family, seed, signed), F64_TOL)

    def test_T1_uses_2L_transport_steps_per_path(self) -> None:
        for length in (2, 3, 4):
            for shared in (True, False):
                with self.subTest(length=length, shared=shared):
                    generator = torch.Generator().manual_seed(length)
                    paths = SignedPaths(8, (length,), 3, generator, signed=True)
                    maps = _chain_operators(paths.edge_index.size(1), k=2, d=4, seed=length,
                                            dtype=torch.float64, shared=shared)
                    transport = paths.transports(maps, shared)[0]
                    counter = _count_steps(transport)
                    PathCellTerm(transport, torch.ones(3, dtype=torch.float64)).apply(
                        torch.randn(8, 4, dtype=torch.float64), 8)
                    self.assertEqual(counter[0], 2 * length)


# ----------------------------------------------------------------------------- D7 (T2)
class SignedTransportTests(unittest.TestCase):
    """T2: signed transport, functoriality on every boundary edge, exact adjoints."""

    def test_T2_signed_transport_matches_dense_product(self) -> None:
        for shared, dense in MAP_SETTINGS:
            for seed in range(3):
                with self.subTest(shared=shared, dense=dense, seed=seed):
                    _, paths, maps, operators = _setting(seed, shared, dense, signed=True)
                    self.assertTrue(any(bool((s < 0).any()) for s in paths.signs))
                    for transport, edges, signs in zip(paths.transports(operators, shared),
                                                       paths.edges, paths.signs):
                        got = _materialize(transport, paths.num_nodes, 4)
                        want = _signed_dense_transports(maps, edges, signs)
                        self.assertLess(_rel(got, want), F64_TOL)

    def test_T2_functoriality_on_every_boundary_edge_with_mixed_signs(self) -> None:
        """`T_{v_{j-1}} F_{v_{j-1}<|e_j}^-1 = T_{v_j} F_{v_j<|e_j}^-1` for every hop, where
        the side of each endpoint is read from the stored edge, not from the sign."""
        for shared, dense in MAP_SETTINGS:
            with self.subTest(shared=shared, dense=dense):
                _, paths, maps, operators = _setting(5, shared, dense, signed=True)
                F_src, F_tgt = maps.src.materialize(), maps.tgt.materialize()
                for transport, nodes, edges in zip(paths.transports(operators, shared),
                                                   paths.nodes, paths.edges):
                    T = _materialize(transport, paths.num_nodes, 4)
                    for j in range(1, nodes.size(1)):
                        e = edges[:, j - 1]
                        stored_src = paths.edge_index[0, e]
                        prev_is_src = (stored_src == nodes[:, j - 1]).view(-1, 1, 1)
                        F_prev = torch.where(prev_is_src, F_src[e], F_tgt[e])
                        F_next = torch.where(prev_is_src, F_tgt[e], F_src[e])
                        left = T[:, j - 1] @ torch.linalg.inv(F_prev)
                        right = T[:, j] @ torch.linalg.inv(F_next)
                        self.assertLess(_rel(left, right), F64_TOL)

    def test_T2_every_method_is_consistent_and_apply_T_is_the_exact_adjoint(self) -> None:
        for shared, dense in MAP_SETTINGS:
            with self.subTest(shared=shared, dense=dense):
                generator, paths, maps, operators = _setting(7, shared, dense, signed=True)
                for transport, nodes, edges, signs in zip(paths.transports(operators, shared),
                                                          paths.nodes, paths.edges, paths.signs):
                    n = paths.num_nodes
                    T = _signed_dense_transports(maps, edges, signs)
                    x = torch.randn(n, 4, generator=generator, dtype=torch.float64)
                    w = torch.randn(nodes.size(0), nodes.size(1), 4, generator=generator,
                                    dtype=torch.float64)
                    y = transport.apply(x)
                    # adjoint: <apply x, w> = <x, apply_T w>
                    self.assertLess(abs(float((y * w).sum() - (x * transport.apply_T(w, n)).sum()))
                                    / max(1.0, float(y.abs().max())), F64_TOL)
                    # Horner sum and prefix recursion against the dense transports
                    self.assertLess(_rel(transport.transport_sum(x), y.sum(dim=1)), F64_TOL)
                    z = w[:, 0]
                    want_prefix = (T.transpose(-2, -1) @ z.view(-1, 1, 4, 1)).squeeze(-1)
                    self.assertLess(_rel(transport.adjoint_prefix(z), want_prefix), F64_TOL)
                    # holonomy and its adjoint
                    self.assertLess(_rel(transport.apply_holonomy(z),
                                         (T[:, -1] @ z.unsqueeze(-1)).squeeze(-1)), F64_TOL)
                    self.assertLess(_rel(transport.apply_holonomy_T(z),
                                         (T[:, -1].transpose(-2, -1) @ z.unsqueeze(-1))
                                         .squeeze(-1)), F64_TOL)
                    # squared scales: T^T T = s^2 I
                    gram = T.transpose(-2, -1) @ T
                    s2 = transport.squared_scales()
                    eye = torch.eye(4, dtype=torch.float64)
                    self.assertLess(_rel(gram, s2.view(*s2.shape, 1, 1) * eye), F64_TOL)

    def test_T2_holonomy_term_uses_the_signed_step(self) -> None:
        _, paths, maps, operators = _setting(3, False, False, signed=True)
        transport = paths.transports(operators, False)[1]
        T = _signed_dense_transports(maps, paths.edges[1], paths.signs[1])
        x = torch.randn(paths.num_nodes, 4, dtype=torch.float64)
        weights = torch.rand(transport.face_node_index.size(0), dtype=torch.float64)
        got = HolonomyEdgeTerm(transport, weights).apply(x, paths.num_nodes)
        nodes = paths.nodes[1]
        delta = x[nodes[:, 0]] - (T[:, -1] @ x[nodes[:, -1]].unsqueeze(-1)).squeeze(-1)
        want = torch.zeros_like(x)
        want.index_add_(0, nodes[:, 0], weights.unsqueeze(-1) * delta)
        want.index_add_(0, nodes[:, -1], -(T[:, -1].transpose(-2, -1)
                                           @ (weights.unsqueeze(-1) * delta).unsqueeze(-1))
                        .squeeze(-1))
        self.assertLess(_rel(got, want), F64_TOL)

    def test_T2_orientation_check_accepts_signed_hops_and_rejects_the_rest(self) -> None:
        nodes = torch.tensor([[0, 1, 2]])
        edges = torch.tensor([[0, 1]])
        edge_index = torch.tensor([[0, 2], [1, 1]])  # 0 -> 1 forward, 2 -> 1 stored reversed
        validate_face_orientation(nodes, edges, edge_index, "t",
                                  face_edge_sign=torch.tensor([[1.0, -1.0]]))
        with self.assertRaises(ValueError):  # unsigned: hop 2 is not stored along the path
            validate_face_orientation(nodes, edges, edge_index, "t")
        with self.assertRaises(ValueError):  # wrong sign on hop 1
            validate_face_orientation(nodes, edges, edge_index, "t",
                                      face_edge_sign=torch.tensor([[-1.0, -1.0]]))
        with self.assertRaises(ValueError):  # sign outside {+1, -1}
            validate_face_orientation(nodes, edges, edge_index, "t",
                                      face_edge_sign=torch.tensor([[1.0, 0.0]]))
        with self.assertRaises(ValueError):  # non-simple path, signs consistent
            validate_face_orientation(torch.tensor([[0, 1, 0]]), torch.tensor([[0, 0]]),
                                      torch.tensor([[0], [1]]), "t",
                                      face_edge_sign=torch.tensor([[1.0, -1.0]]))


if __name__ == "__main__":
    unittest.main()
