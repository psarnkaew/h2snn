"""Fact layer, path-cell builder and O(L) path term (task card D7-D9, plan §9).

Test ids follow the task card (T1-T16).
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.models.map_operators import DenseMapOperator, EdgeMapPair, PathTransportOperator
from src.models.vertex_path_laplacian import PathCellTerm
from tests.test_v2_model import _chain_operators, _dense_transports

F64_TOL = 1e-10


# ----------------------------------------------------------------------------- fixtures
def _random_paths(num_nodes: int, lengths: tuple[int, ...], paths_per_length: int,
                  generator: torch.Generator) -> tuple[list[torch.Tensor], list[torch.Tensor],
                                                       torch.Tensor]:
    """Overlapping simple paths of mixed length on `num_nodes` nodes, one new edge per hop
    stored along its path. Returns per-length (face_node_index, face_edge_index) and the
    `[2, E]` edge list."""
    node_groups, edge_groups, pairs = [], [], []
    next_edge = 0
    for length in lengths:
        nodes = torch.stack([torch.randperm(num_nodes, generator=generator)[: length + 1]
                             for _ in range(paths_per_length)])
        edges = torch.arange(next_edge, next_edge + paths_per_length * length).view(-1, length)
        next_edge += paths_per_length * length
        pairs.append(torch.stack([nodes[:, :-1].reshape(-1), nodes[:, 1:].reshape(-1)]))
        node_groups.append(nodes)
        edge_groups.append(edges)
    return node_groups, edge_groups, torch.cat(pairs, dim=1)


def _as_dense_family(maps: EdgeMapPair) -> EdgeMapPair:
    """The same conformal maps behind the dense operator interface (dense map family)."""
    return EdgeMapPair(
        src=DenseMapOperator(maps.src.materialize(), maps.src.scale),
        tgt=DenseMapOperator(maps.tgt.materialize(), maps.tgt.scale),
    )


def _dense_path_laplacian(maps: EdgeMapPair, node_groups: list[torch.Tensor],
                          edge_groups: list[torch.Tensor], weights: list[torch.Tensor],
                          num_nodes: int, d: int) -> torch.Tensor:
    """Reference `sum_f w_f B_f^T (I - 1 1^T / n) B_f` as an explicit `[N d, N d]` matrix,
    `B_f` built from the general-inverse transports `_dense_transports`."""
    dtype = maps.src.materialize().dtype
    output = torch.zeros(num_nodes * d, num_nodes * d, dtype=dtype)
    for nodes, edges, w in zip(node_groups, edge_groups, weights):
        transports = _dense_transports(maps, edges)  # [P, n, d, d]
        n = nodes.size(1)
        centering = torch.eye(n, dtype=dtype) - torch.full((n, n), 1.0 / n, dtype=dtype)
        for p in range(nodes.size(0)):
            B = torch.zeros(n * d, num_nodes * d, dtype=dtype)
            for i in range(n):
                v = int(nodes[p, i])
                B[i * d:(i + 1) * d, v * d:(v + 1) * d] = transports[p, i]
            P = torch.kron(centering, torch.eye(d, dtype=dtype))
            output += w[p] * B.T @ P @ B
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


# ----------------------------------------------------------------------------- D9 (T1)
class OLPathTermTests(unittest.TestCase):
    """T1: the O(L) `PathCellTerm.apply` equals the dense reference."""

    def _check(self, shared: bool, dense_family: bool, seed: int) -> float:
        num_nodes, d, k = 9, 4, 2
        generator = torch.Generator().manual_seed(seed)
        node_groups, edge_groups, edge_index = _random_paths(num_nodes, (2, 3, 4), 4, generator)
        maps = _chain_operators(edge_index.size(1), k=k, d=d, seed=seed, dtype=torch.float64,
                                shared=shared)
        operators = _as_dense_family(maps) if dense_family else maps
        weights = [torch.rand(nodes.size(0), generator=generator, dtype=torch.float64) * 3.0
                   for nodes in node_groups]
        x = torch.randn(num_nodes, d, generator=generator, dtype=torch.float64)
        output = torch.zeros_like(x)
        for nodes, edges, w in zip(node_groups, edge_groups, weights):
            transport = PathTransportOperator(operators.src, operators.tgt, nodes, edges,
                                              edge_index, shared_scale=shared)
            output += PathCellTerm(transport, w).apply(x, num_nodes)
        reference = _dense_path_laplacian(maps, node_groups, edge_groups, weights, num_nodes, d)
        expected = (reference @ x.reshape(-1)).view(num_nodes, d)
        return float((output - expected).abs().max() / expected.abs().max().clamp_min(1.0))

    def test_T1_matches_dense_reference_mixed_lengths_both_scale_modes(self) -> None:
        for shared in (True, False):
            for dense_family in (False, True):
                for seed in range(3):
                    with self.subTest(shared=shared, dense_family=dense_family, seed=seed):
                        self.assertLess(self._check(shared, dense_family, seed), F64_TOL)

    def test_T1_uses_2L_transport_steps_per_path(self) -> None:
        for length in (2, 3, 4):
            for shared in (True, False):
                with self.subTest(length=length, shared=shared):
                    generator = torch.Generator().manual_seed(length)
                    nodes, edges, edge_index = _random_paths(8, (length,), 3, generator)
                    maps = _chain_operators(edge_index.size(1), k=2, d=4, seed=length,
                                            dtype=torch.float64, shared=shared)
                    transport = PathTransportOperator(maps.src, maps.tgt, nodes[0], edges[0],
                                                      edge_index, shared_scale=shared)
                    counter = _count_steps(transport)
                    PathCellTerm(transport, torch.ones(3, dtype=torch.float64)).apply(
                        torch.randn(8, 4, dtype=torch.float64), 8)
                    self.assertEqual(counter[0], 2 * length)


if __name__ == "__main__":
    unittest.main()
