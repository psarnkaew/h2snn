"""Fact layer, path-cell builder and O(L) path term (task card D7-D9, plan §9).

Test ids follow the task card (T1-T16; T3, T7 and T15 were dropped by the user).
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.data.face_geometry import validate_face_orientation
from src.data.fact_layer import apply_fact_layer, build_fact_layer
from src.models.map_operators import DenseMapOperator, EdgeMapPair, PathTransportOperator
from src.models.sheaf_maps import DenseConformalNodeToEdgeSheaf, FactorizedNodeToEdgeSheaf
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
    return _dense_cells(maps, paths.nodes, paths.edges, paths.signs, weights, paths.num_nodes, d)


def _dense_cells(maps: EdgeMapPair, node_groups, edge_groups, sign_groups, weights,
                 num_nodes: int, d: int) -> torch.Tensor:
    output = torch.zeros(num_nodes * d, num_nodes * d, dtype=torch.float64)
    for nodes, edges, signs, w in zip(node_groups, edge_groups, sign_groups, weights):
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


# ----------------------------------------------------------------------------- D7 (T4)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _layer(edges: list[tuple[int, int, int]], num_relations: int, num_nodes: int,
           mode: str = "tied"):
    edge_index = torch.tensor([[u for u, _, _ in edges], [v for _, _, v in edges]])
    edge_type = torch.tensor([r for _, r, _ in edges])
    names = tuple(f"r{r}" for r in range(num_relations))
    layer = build_fact_layer(edge_index, edge_type, names, num_nodes, mode)
    tied_index, tied_type = apply_fact_layer(layer, edge_index, edge_type)
    return layer, tied_index, tied_type


class FactLayerDefinitionTests(unittest.TestCase):
    """T4: the one definition on fixtures, on real data, and the tuning check."""

    def test_T4_exact_mirror_pair_gives_one_cell_per_fact(self) -> None:
        layer, index, types = _layer([(0, 0, 1), (2, 0, 3), (1, 1, 0), (3, 1, 2)], 2, 4)
        self.assertEqual(layer.rho, (1, 0))
        self.assertEqual(layer.kept, (0,))
        self.assertEqual(layer.self_inverse, ())
        self.assertEqual(index.tolist(), [[0, 2], [1, 3]])  # orientation of the lower id
        self.assertEqual(types.tolist(), [0, 0])
        self.assertEqual((layer.facts_before, layer.facts_after), (4, 2))
        self.assertEqual(layer.resolve("r0"), (0, 1))
        self.assertEqual(layer.resolve("r1"), (0, -1))

    def test_T4_self_inverse_relation_gives_one_cell_per_unordered_pair(self) -> None:
        layer, index, types = _layer([(1, 0, 0), (0, 0, 1), (2, 0, 2), (3, 0, 1), (1, 0, 3)],
                                     1, 4)
        self.assertEqual(layer.rho, (0,))
        self.assertEqual(layer.self_inverse, (0,))
        self.assertEqual(sorted(map(tuple, index.t().tolist())), [(0, 1), (1, 3), (2, 2)])
        self.assertEqual(layer.facts_after, 3)  # self-loop once

    def test_T4_reciprocal_facts_of_a_directed_relation_stay_two_cells(self) -> None:
        cite = [(0, 0, 1), (1, 0, 0), (0, 0, 2)]
        ref = [(v, 1, u) for u, _, v in cite]
        layer, index, _ = _layer(cite + ref, 2, 3)
        self.assertEqual(layer.rho, (1, 0))  # cite is not its own reversal: rho(cite) = ref
        self.assertEqual(sorted(map(tuple, index.t().tolist())), [(0, 1), (0, 2), (1, 0)])

    def test_T4_a_missing_reverse_edge_leaves_the_pair_untied(self) -> None:
        layer, index, types = _layer([(0, 0, 1), (2, 0, 3), (1, 1, 0)], 2, 4)
        self.assertEqual(layer.rho, (None, None))
        self.assertEqual(layer.unmatched_edges, (2, 1))
        self.assertEqual(layer.kept, (0, 1))
        self.assertEqual(index.size(1), 3)
        self.assertEqual(types.tolist(), [0, 0, 1])

    def test_T4_untied_keeps_every_stored_edge_and_relation(self) -> None:
        edges = [(0, 0, 1), (2, 0, 3), (1, 1, 0), (3, 1, 2), (1, 2, 1)]
        layer, index, types = _layer(edges, 3, 4, mode="untied")
        self.assertEqual(layer.rho, (1, 0, 2))  # rho is still computed and reported
        self.assertEqual(layer.kept, (0, 1, 2))
        self.assertEqual(layer.self_inverse, ())
        self.assertEqual(index.tolist(), [[0, 2, 1, 3, 1], [1, 3, 0, 2, 1]])
        self.assertEqual(types.tolist(), [0, 0, 1, 1, 2])

    def test_T4_two_exact_partners_are_an_error(self) -> None:
        with self.assertRaises(ValueError):
            _layer([(0, 0, 1), (1, 1, 0), (1, 2, 0)], 3, 2)

    @unittest.skipUnless((REPO_ROOT / "data" / "acm").exists(), "HGB data not present")
    def test_T4_real_hgb_counts(self) -> None:
        from src.configs.config import TrainConfig
        from src.data.dataset import _load_hetero_data
        from src.data.hetero_flatten import flatten_edges, stack_node_features

        expected = {"acm": (547872, 273936, 4), "dblp": (239566, 119783, 3),
                    "imdb": (86642, 43321, 3)}
        for name, (before, after, pairs) in expected.items():
            if not (REPO_ROOT / "data" / name).exists():
                continue
            with self.subTest(dataset=name):
                config = TrainConfig.from_dict(
                    {"data": {"dataset_name": name, "root": str(REPO_ROOT / "data")}})
                data = _load_hetero_data(config)
                x, _, _, slices, _ = stack_node_features(data)
                edge_index, edge_type, names, _ = flatten_edges(data, slices)
                layer = build_fact_layer(edge_index, edge_type, names, x.size(0), "tied")
                self.assertEqual((layer.facts_before, layer.facts_after), (before, after))
                self.assertEqual(sum(layer.unmatched_edges), 0)
                self.assertEqual(len(layer.kept), pairs)
                self.assertEqual(layer.self_inverse, ())
                if name == "acm":
                    self.assertEqual(layer.rho[names.index("paper__cite__paper")],
                                     names.index("paper__ref__paper"))

    @unittest.skipUnless((REPO_ROOT / "data" / "tud").exists(), "TU data not present")
    def test_T4_real_tu_relations_are_self_inverse(self) -> None:
        from torch_geometric.datasets import TUDataset

        from src.data.graph_classification_dataset import _load_tu_type_tensors, _tie_tu_facts

        expected = {"MUTAG": (7442, 3721), "PROTEINS": (162088, 81044),
                    "ENZYMES": (74564, 37282), "NCI1": (265506, 132753)}
        for tud, (before, after) in expected.items():
            if not (REPO_ROOT / "data" / "tud" / tud).exists():
                continue
            with self.subTest(dataset=tud):
                dataset = TUDataset(root=str(REPO_ROOT / "data" / "tud"), name=tud)
                data_list = [dataset[i].clone() for i in range(len(dataset))]
                _, edge_types, _, num_relations = _load_tu_type_tensors(
                    REPO_ROOT / "data", tud, data_list)
                layer, tied, _ = _tie_tu_facts(data_list, edge_types, num_relations, "tied")
                self.assertEqual((layer.facts_before, layer.facts_after), (before, after))
                self.assertEqual(sum(d.edge_index.size(1) for d in tied), after)
                self.assertEqual(layer.self_inverse, tuple(range(num_relations)))

    def test_T4_untied_is_rejected_by_the_tuning_check(self) -> None:
        from scripts.run_h2snn_wandb_tuning import (apply_sweep_overrides, build_sweep_config,
                                                    load_raw_config)
        from src.configs.config import reject_never_tuned_settings

        with self.assertRaises(ValueError):
            reject_never_tuned_settings({"data": {"fact_layer": "untied"}})
        with self.assertRaises(ValueError):
            reject_never_tuned_settings(None, {"fact_layer": {"values": ["tied", "untied"]}})
        reject_never_tuned_settings({"data": {"fact_layer": "tied"}}, {"lr": {}})
        with self.assertRaises(ValueError):  # a trial applied on an untied base config
            apply_sweep_overrides({"data": {"fact_layer": "untied"}}, {"lr": 1e-3})
        for task in ("node_classification", "link_prediction", "graph_classification"):
            sweep = build_sweep_config(task_type=task, metric="val_acc", epochs=50)
            self.assertFalse(any("fact_layer" in key for key in sweep["parameters"]))
        with TemporaryDirectory(dir=REPO_ROOT) as tmp:
            path = Path(tmp) / "base.yaml"
            path.write_text("data:\n  dataset_name: acm\n  fact_layer: untied\n",
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                load_raw_config(path)
        for notebook in sorted((REPO_ROOT / "notebooks").glob("*tuning*.ipynb")):
            with self.subTest(notebook=notebook.name):
                self.assertNotIn("fact_layer", notebook.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- D7 (T5, T16)
def _map_module(family: str, map_scale: str, shared: tuple[int, ...], num_relations: int,
                d: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    if family == "factorized":
        module = FactorizedNodeToEdgeSheaf(
            d=d, d_r=2, num_relations=num_relations, hidden_dim=8, householder_k=2,
            scale_eps=0.1, orthogonal_map="fasthpp", map_scale=map_scale, checkpoint=False,
            shared_role_relations=shared)
    else:
        module = DenseConformalNodeToEdgeSheaf(
            d=d, d_r=2, num_relations=num_relations, hidden_dim=8, scale_eps=0.1,
            orthogonal_map="fasthpp", map_scale=map_scale, memory_guard_fraction=0.9,
            shared_role_relations=shared)
    return module.double()


def _normalized_delta0(maps: EdgeMapPair, edge_index: torch.Tensor, num_nodes: int,
                       d: int) -> torch.Tensor:
    """Dense `D^-1/2 Delta_0 D^-1/2`, `Delta_0 = sum_e B_e^T B_e`, `D_v = sum c^2`."""
    F_src, F_tgt = _materialized(maps.src), _materialized(maps.tgt)
    L = torch.zeros(num_nodes * d, num_nodes * d, dtype=torch.float64)
    degree = torch.zeros(num_nodes, dtype=torch.float64)
    for e in range(edge_index.size(1)):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        B = torch.zeros(d, num_nodes * d, dtype=torch.float64)
        B[:, u * d:(u + 1) * d] -= F_src[e]
        B[:, v * d:(v + 1) * d] += F_tgt[e]
        L += B.T @ B
        degree[u] += maps.src.scale[e] ** 2
        degree[v] += maps.tgt.scale[e] ** 2
    scale = degree.clamp_min(1e-300).rsqrt().repeat_interleave(d)
    return scale.view(-1, 1) * L * scale.view(1, -1)


def _materialized(operator) -> torch.Tensor:
    return operator.materialize() if hasattr(operator, "materialize") else operator.maps


def _flip(edge_index: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    flipped = edge_index.clone()
    flipped[:, rows] = edge_index.flip(0)[:, rows]
    return flipped


def _two_relation_graph(seed: int):
    """Relation 0 (to be shared) and relation 1 (directed) on 7 nodes."""
    generator = torch.Generator().manual_seed(seed)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 6), (6, 1), (2, 5), (3, 3)]
    edge_index = torch.tensor(pairs).t()
    edge_type = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0])
    h = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    rel = torch.randn(2, 2, generator=generator, dtype=torch.float64)
    return edge_index, edge_type, h, rel


FAMILIES = [(family, scale) for family in ("factorized", "dense")
            for scale in ("per_edge", "per_side")]


class SharedRoleTests(unittest.TestCase):
    """T5 (self-inverse relations) and T16 (direction_blind_maps): orientation invariance."""

    def _change(self, family: str, scale: str, shared: tuple[int, ...],
                flip_rows: torch.Tensor) -> float:
        edge_index, edge_type, h, rel = _two_relation_graph(1)
        module = _map_module(family, scale, shared, num_relations=2)
        with torch.no_grad():
            before = _normalized_delta0(module(h, edge_index, edge_type, rel).maps,
                                        edge_index, 7, 4)
            flipped = _flip(edge_index, flip_rows)
            after = _normalized_delta0(module(h, flipped, edge_type, rel).maps, flipped, 7, 4)
        return float((before - after).abs().max())

    def test_T5_flipping_cells_of_a_self_inverse_relation_leaves_delta0_unchanged(self) -> None:
        relation0 = torch.tensor([0, 2, 4, 6, 8])  # includes the self-loop (3, 3)
        for family, scale in FAMILIES:
            for subset in (relation0, relation0[:2], relation0[2:3]):
                with self.subTest(family=family, scale=scale, subset=subset.tolist()):
                    self.assertLess(self._change(family, scale, (0,), subset), F64_TOL)

    def test_T5_negative_control_separate_roles_change_delta0(self) -> None:
        for family, scale in FAMILIES:
            with self.subTest(family=family, scale=scale):
                self.assertGreater(self._change(family, scale, (), torch.tensor([0, 2])), 1e-3)

    def test_T16_direction_blind_maps_make_delta0_invariant_to_flipping_any_cell(self) -> None:
        for family, scale in FAMILIES:
            for subset in (torch.tensor([1, 3]), torch.tensor([0, 1, 5, 7]), torch.arange(9)):
                with self.subTest(family=family, scale=scale, subset=subset.tolist()):
                    self.assertLess(self._change(family, scale, (0, 1), subset), F64_TOL)
            with self.subTest(family=family, scale=scale, control="relation 1 directed"):
                self.assertGreater(self._change(family, scale, (0,), torch.tensor([1, 3])),
                                   1e-3)

    def test_T16_model_shares_roles_per_fact_layer_and_ablation(self) -> None:
        from src.configs.config import TrainConfig
        from src.models.h2snn_model import H2SNN, ModelDims

        dims = ModelDims(in_channels=5, num_node_types=2, num_relations=3, num_classes=2,
                         face_lengths=(2,), self_inverse_relations=(2,))
        for blind, want in ((False, (2,)), (True, (0, 1, 2))):
            config = TrainConfig.from_dict({"device": "cpu",
                                            "ablation": {"direction_blind_maps": blind}})
            model = H2SNN(config, dims)
            self.assertEqual(model.shared_role_relations, want)
            self.assertEqual(model.node_to_edge.shared_role_relations, want)
        with self.assertRaises(ValueError):
            TrainConfig.from_dict({"ablation": {"direction_blind_maps": "yes"}})


# ----------------------------------------------------------------------------- D7 (T6)
def _moderate_scales(maps: EdgeMapPair, generator: torch.Generator, shared: bool) -> EdgeMapPair:
    """Same frames and reflectors, scales log-uniform in [0.5, 2]. With `_chain_operators`'
    per-side scales spanning [0.1, 50] the normalized Delta_0 of the tree has genuine
    non-kernel eigenvalues down to ~1e-9 (docs/D7_evidence/u3_checks_output.txt), which no
    fixed kernel threshold can separate; the property itself is scale-free."""
    from src.models.map_operators import FactorizedMapOperator

    def draw():
        return torch.exp((torch.rand(maps.src.num_edges, generator=generator,
                                     dtype=torch.float64) * 2 - 1) * 0.6931471805599453)

    src_scale = draw()
    tgt_scale = src_scale if shared else draw()

    def side(op, scale):
        return FactorizedMapOperator(scale=scale, reflectors=op.reflectors,
                                     rotations=op.rotations, group_ids=op.group_ids)

    return EdgeMapPair(src=side(maps.src, src_scale), tgt=side(maps.tgt, tgt_scale))


class SectionPreservationTests(unittest.TestCase):
    """T6: on a tree with mixed-direction path hops, sections are in ker L_P and
    `dim ker(Delta_0 + gamma L_P) = dim ker Delta_0` (eigenvalue threshold 1e-9)."""

    KERNEL_TOL = 1e-9

    def _tree(self, seed: int):
        generator = torch.Generator().manual_seed(seed)
        parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 5}
        pairs = []
        for child, par in parent.items():  # random stored orientation per tree edge
            pairs.append((par, child) if torch.rand(1, generator=generator) < 0.5
                         else (child, par))
        edge_index = torch.tensor(pairs).t()
        lookup = {frozenset(p): e for e, p in enumerate(pairs)}
        node_paths = {2: [[3, 1, 0], [1, 0, 2], [5, 2, 6], [7, 5, 2]],
                      3: [[3, 1, 0, 2], [4, 1, 0, 2], [7, 5, 2, 0]]}
        groups = []
        for length, sequences in node_paths.items():
            nodes = torch.tensor(sequences)
            edges = torch.tensor([[lookup[frozenset((s[j], s[j + 1]))] for j in range(length)]
                                  for s in sequences])
            signs = torch.tensor([[1.0 if pairs[lookup[frozenset((s[j], s[j + 1]))]][0] == s[j]
                                   else -1.0 for j in range(length)] for s in sequences])
            groups.append((nodes, edges, signs))
        return edge_index, groups

    def test_T6_sections_are_preserved_with_signed_hops(self) -> None:
        d, num_nodes, gamma = 3, 8, 0.7
        for shared in (True, False):
            for seed in range(3):
                with self.subTest(shared=shared, seed=seed):
                    edge_index, groups = self._tree(seed)
                    self.assertTrue(any(bool((s < 0).any()) for _, _, s in groups))
                    generator = torch.Generator().manual_seed(100 + seed)
                    maps = _moderate_scales(
                        _chain_operators(edge_index.size(1), k=2, d=d, seed=seed,
                                         dtype=torch.float64, shared=shared),
                        generator, shared)
                    F_src, F_tgt = maps.src.materialize(), maps.tgt.materialize()
                    weights = [torch.rand(n.size(0), generator=generator, dtype=torch.float64)
                               + 0.5 for n, _, _ in groups]
                    L_P = _dense_cells(maps, [g[0] for g in groups], [g[1] for g in groups],
                                       [g[2] for g in groups], weights, num_nodes, d)
                    # a section: x_0 free, then F_tgt x_tgt = F_src x_src on every edge
                    x = torch.zeros(num_nodes, d, dtype=torch.float64)
                    x[0] = torch.randn(d, dtype=torch.float64)
                    known = {0}
                    while len(known) < num_nodes:
                        for e in range(edge_index.size(1)):
                            u, v = int(edge_index[0, e]), int(edge_index[1, e])
                            if u in known and v not in known:
                                x[v] = torch.linalg.solve(F_tgt[e], F_src[e] @ x[u])
                                known.add(v)
                            elif v in known and u not in known:
                                x[u] = torch.linalg.solve(F_src[e], F_tgt[e] @ x[v])
                                known.add(u)
                    # Relative to the row magnitude |L_P| |x|: per-side scales span [0.1, 50],
                    # so entries of L_P reach ~1e10 (docs/D7_evidence/u3_t6_diag_output.txt).
                    residual = (L_P @ x.reshape(-1)).abs().max()
                    magnitude = (L_P.abs() @ x.reshape(-1).abs()).max()
                    self.assertLess(float(residual / magnitude), F64_TOL)
                    # kernel dimensions (normalized operators, D > 0 on every node)
                    delta0 = torch.zeros_like(L_P)
                    degree = torch.zeros(num_nodes, dtype=torch.float64)
                    for e in range(edge_index.size(1)):
                        u, v = int(edge_index[0, e]), int(edge_index[1, e])
                        B = torch.zeros(d, num_nodes * d, dtype=torch.float64)
                        B[:, u * d:(u + 1) * d] -= F_src[e]
                        B[:, v * d:(v + 1) * d] += F_tgt[e]
                        delta0 += B.T @ B
                        degree[u] += maps.src.scale[e] ** 2
                        degree[v] += maps.tgt.scale[e] ** 2
                    for (nodes, edges, signs), w in zip(groups, weights):
                        n = nodes.size(1)
                        squares = (_signed_dense_transports(maps, edges, signs)
                                   .square().sum(dim=(-2, -1)) / d)  # s_i^2
                        per_incidence = (1 - 1 / n) * w.unsqueeze(1) * squares
                        degree.index_add_(0, nodes.reshape(-1), gamma * per_incidence.reshape(-1))
                    scale = degree.rsqrt().repeat_interleave(d)
                    norm = scale.view(-1, 1) * scale.view(1, -1)
                    for name, matrix in (("delta0", delta0), ("full", delta0 + gamma * L_P)):
                        eigenvalues = torch.linalg.eigvalsh(norm * matrix)
                        kernel = int((eigenvalues <= self.KERNEL_TOL).sum())
                        # connected tree: exactly one d-dimensional space of global sections
                        self.assertEqual(kernel, d, name)
                        # the threshold separates: the next eigenvalue is far above it
                        self.assertGreater(float(eigenvalues[kernel]), 1e3 * self.KERNEL_TOL, name)


# ----------------------------------------------------------------------------- D7 (T10)
def _acm_like(mode: str):
    """paper(0-3), author(4-5): cite/ref mirror pair, paper->author / author->paper pair."""
    cite = [(0, 1), (1, 2), (2, 0), (3, 1)]
    writes = [(0, 4), (1, 4), (2, 5), (3, 5)]
    edges = ([(u, 0, v) for u, v in cite] + [(v, 1, u) for u, v in cite]
             + [(u, 2, v) for u, v in writes] + [(v, 3, u) for u, v in writes])
    names = ("paper__cite__paper", "paper__ref__paper", "paper__to__author", "author__to__paper")
    tuples = (("paper", "cite", "paper"), ("paper", "ref", "paper"),
              ("paper", "to", "author"), ("author", "to", "paper"))
    edge_index = torch.tensor([[u for u, _, _ in edges], [v for _, _, v in edges]])
    edge_type = torch.tensor([r for _, r, _ in edges])
    layer = build_fact_layer(edge_index, edge_type, names, 6, mode)
    return layer, tuple(tuples[r] for r in layer.kept)


class SchemaTests(unittest.TestCase):
    """T10: schemas with direction."""

    def test_T10_citation_chain_and_co_citation_are_distinct(self) -> None:
        from src.configs.config import MetaPathConfig
        from src.data.schemas import resolve_schemas

        layer, tuples = _acm_like("tied")
        chain = MetaPathConfig("chain", ("paper",) * 3,
                               hops=(("paper__cite__paper", "+"), ("paper__cite__paper", "+")))
        cocite = MetaPathConfig("cocite", ("paper",) * 3,
                                hops=(("paper__cite__paper", "+"), ("paper__cite__paper", "-")))
        schemas = resolve_schemas((chain, cocite), layer, tuples)
        self.assertEqual([s.hops for s in schemas], [((0, 1), (0, 1)), ((0, 1), (0, -1))])
        self.assertNotEqual(schemas[0].key, schemas[1].key)

    def test_T10_ambiguous_transition_without_hops_raises(self) -> None:
        from src.configs.config import MetaPathConfig
        from src.data.schemas import resolve_schema

        for mode in ("tied", "untied"):
            layer, tuples = _acm_like(mode)
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "'paper' -> 'paper'.*2 .*candidates"):
                    resolve_schema(MetaPathConfig("PPP", ("paper",) * 3), layer, tuples)
        layer, tuples = _acm_like("tied")  # PAP is unique: the tied paper-author fact both ways
        pap = resolve_schema(MetaPathConfig("PAP", ("paper", "author", "paper")), layer, tuples)
        self.assertEqual(pap.hops, ((1, 1), (1, -1)))

    def test_T10_a_schema_and_its_reverse_share_one_group(self) -> None:
        from src.configs.config import MetaPathConfig
        from src.data.schemas import resolve_schemas, schema_key

        layer, tuples = _acm_like("tied")
        forward = MetaPathConfig("chain", ("paper",) * 3,
                                 hops=(("paper__cite__paper", "+"), ("paper__cite__paper", "+")))
        backward = MetaPathConfig("chain_rev", ("paper",) * 3,  # ref = cite reversed
                                  hops=(("paper__ref__paper", "+"), ("paper__ref__paper", "+")))
        apa = MetaPathConfig("APA", ("author", "paper", "author"))
        pap = MetaPathConfig("PAP", ("paper", "author", "paper"))
        schemas = resolve_schemas((forward, backward, pap, apa), layer, tuples)
        self.assertEqual([s.name for s in schemas], ["chain", "PAP", "APA"])
        mixed = ("paper", "author"), ((1, -1),)
        self.assertEqual(schema_key(*mixed), schema_key(("author", "paper"), ((1, 1),)))

    def test_T10_untied_rejects_minus_hops(self) -> None:
        from src.configs.config import MetaPathConfig, TrainConfig
        from src.data.schemas import resolve_schema

        layer, tuples = _acm_like("untied")
        with self.assertRaises(ValueError):
            resolve_schema(MetaPathConfig("x", ("paper",) * 2,
                                          hops=(("paper__cite__paper", "-"),)), layer, tuples)
        with self.assertRaises(ValueError):
            TrainConfig.from_dict({"data": {
                "dataset_name": "acm", "fact_layer": "untied", "path_sampler": "dfs",
                "metapath_templates": [
                {"name": "x", "node_types": ["paper", "paper"],
                 "hops": [["paper__cite__paper", "-"]]}]}})
        with self.assertRaises(ValueError):  # malformed hop
            MetaPathConfig("x", ("paper", "paper"), hops=(("paper__cite__paper", "?"),))

    def test_T10_self_inverse_hops_have_no_direction(self) -> None:
        from src.configs.config import MetaPathConfig
        from src.data.schemas import resolve_schema

        layer, _, _ = _layer([(0, 0, 1), (1, 0, 0), (1, 0, 2), (2, 0, 1)], 1, 3)
        schema = resolve_schema(MetaPathConfig("NNN", ("node",) * 3), layer,
                                (("node", "bond", "node"),))
        self.assertEqual(schema.hops, ((0, 0), (0, 0)))
        self.assertEqual(schema.key, (("node",) * 3, ((0, 0), (0, 0))))

    @unittest.skipUnless((REPO_ROOT / "data" / "acm").exists(), "HGB data not present")
    def test_T10_every_default_schema_resolves_on_real_data(self) -> None:
        from src.configs.config import TrainConfig
        from src.data.dataset import _load_hetero_data
        from src.data.hetero_flatten import flatten_edges, stack_node_features
        from src.data.schemas import resolve_schemas

        for name in ("acm", "dblp", "imdb"):
            if not (REPO_ROOT / "data" / name).exists():
                continue
            config = TrainConfig.from_dict(
                {"data": {"dataset_name": name, "root": str(REPO_ROOT / "data")}})
            data = _load_hetero_data(config)
            x, _, _, slices, _ = stack_node_features(data)
            edge_index, edge_type, names, tuples = flatten_edges(data, slices)
            for mode in ("tied", "untied"):
                with self.subTest(dataset=name, mode=mode):
                    layer = build_fact_layer(edge_index, edge_type, names, x.size(0), mode)
                    kept = tuple(tuples[r] for r in layer.kept)
                    schemas = resolve_schemas(config.data.metapath_templates, layer, kept)
                    self.assertEqual(len(schemas), len(config.data.metapath_templates))
                    if mode == "untied":
                        self.assertTrue(all(d == 1 for s in schemas for _, d in s.hops))


# ----------------------------------------------------------------------------- D8 fixtures
def _small_hetero(seed: int = 0):
    """Papers 0-5, authors 6-8; tied facts: cite (0) incl. a reciprocal pair, writes (1)
    stored paper -> author. Returns edge_index, edge_type, node_type, names, tuples."""
    cite = [(0, 1), (1, 0), (1, 2), (2, 3), (3, 0), (4, 1), (5, 2), (4, 5)]
    writes = [(0, 6), (1, 6), (2, 6), (2, 7), (3, 7), (4, 8), (5, 8), (5, 7)]
    edges = [(u, 0, v) for u, v in cite] + [(u, 1, v) for u, v in writes]
    edge_index = torch.tensor([[u for u, _, _ in edges], [v for _, _, v in edges]])
    edge_type = torch.tensor([r for _, r, _ in edges])
    node_type = torch.tensor([0] * 6 + [1] * 3)
    return edge_index, edge_type, node_type, ("paper", "author"), (
        ("paper", "cite", "paper"), ("paper", "to", "author"))


SCHEMAS_T8 = (
    ("PAP", ("paper", "author", "paper"), ((1, 1), (1, -1))),       # palindromic
    ("cocite", ("paper", "paper", "paper"), ((0, 1), (0, -1))),     # mixed directions
    ("PPA", ("paper", "paper", "author"), ((0, 1), (1, 1))),        # not palindromic
)


def _resolved(schemas, r: int):
    from src.data.schemas import ResolvedSchema

    return tuple(ResolvedSchema(name, types, hops, r) for name, types, hops in schemas)


def _draw(seed: int, r: int, merge: str = "sum", sampler: str = "walk", schemas=SCHEMAS_T8):
    from src.data.path_sampler import build_schema_path_groups

    edge_index, edge_type, node_type, type_names, _ = _small_hetero()
    return build_schema_path_groups(
        edge_index, edge_type, node_type, type_names, _resolved(schemas, r), (), True,
        anchors_of=lambda t: (node_type == type_names.index(t)).nonzero(as_tuple=True)[0],
        default_walks=r, seed=seed, sampler=sampler, merge=merge)


def _apply_groups(groups, maps, edge_index, x) -> torch.Tensor:
    output = torch.zeros_like(x)
    for group in groups:
        transport = PathTransportOperator(maps.src, maps.tgt, group.face_node_index,
                                          group.face_edge_index, edge_index, shared_scale=True,
                                          face_edge_sign=group.face_edge_sign)
        output += PathCellTerm(transport, group.path_weight.double()).apply(x, x.size(0))
    return output


def _exact_target(maps, x) -> torch.Tensor:
    """`sum_pi sum_a sum_walk p(walk) 1{simple} L_f x`, walks enumerated exhaustively."""
    edge_index, edge_type, node_type, type_names, _ = _small_hetero()
    from src.data.path_sampler import build_half_edges, hop_csr

    half = build_half_edges(edge_index, edge_type, node_type.numel(), (), True)
    output = torch.zeros_like(x)
    for _, types, hops in SCHEMAS_T8:
        csrs = [hop_csr(half, r, d) for r, d in hops]
        anchors = (node_type == type_names.index(types[0])).nonzero(as_tuple=True)[0].tolist()
        stack = [([a], [], [], 1.0) for a in anchors]
        while stack:
            nodes, facts, signs, p = stack.pop()
            j = len(facts)
            if j == len(hops):
                if len(set(nodes)) == len(nodes):
                    transport = PathTransportOperator(
                        maps.src, maps.tgt, torch.tensor([nodes]), torch.tensor([facts]),
                        edge_index, shared_scale=True, face_edge_sign=torch.tensor([signs]))
                    output += PathCellTerm(transport, torch.tensor([p], dtype=torch.float64)
                                           ).apply(x, x.size(0))
                continue
            csr = csrs[j]
            start, end = int(csr.ptr[nodes[-1]]), int(csr.ptr[nodes[-1] + 1])
            for k in range(start, end):
                stack.append((nodes + [int(csr.other[k])], facts + [int(csr.fact[k])],
                              signs + [float(csr.sign[k])], p / (end - start)))
    return output


class EstimatorTests(unittest.TestCase):
    """T8: the merged-weight estimator is unbiased for the stated target."""

    DRAWS, R = 4000, 3

    def _statistics(self, merge: str) -> tuple[float, float]:
        edge_index, *_ = _small_hetero()
        maps = _chain_operators(edge_index.size(1), k=2, d=3, seed=11, dtype=torch.float64,
                                shared=True)
        x = torch.randn(9, 3, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        target = _exact_target(maps, x).reshape(-1)
        samples = torch.stack([_apply_groups(_draw(seed, self.R, merge), maps, edge_index, x)
                               .reshape(-1) for seed in range(self.DRAWS)])
        mean = samples.mean(dim=0)
        stderr = samples.std(dim=0) / self.DRAWS ** 0.5
        z = (mean - target).abs() / stderr.clamp_min(1e-300)
        z = torch.where((mean - target).abs() < 1e-12, torch.zeros_like(z), z)
        return float((mean - target).norm() / target.norm()), float(z.max())

    def test_T8_mean_of_draws_matches_the_exact_target(self) -> None:
        relative, z = self._statistics("sum")
        self.assertLessEqual(relative, 0.02)
        self.assertLessEqual(z, 5.0)

    def test_T8_negative_control_dropping_duplicates_is_biased(self) -> None:
        _, z = self._statistics("drop_duplicates")
        self.assertGreater(z, 10.0)


class IdentityTests(unittest.TestCase):
    """T9: cell identity by facts, weights summed."""

    def _merge(self, rows, weights):
        from src.data.path_sampler import Walks, merge_cells

        edge_index, _, node_type, _, _ = _small_hetero()
        walks = Walks(
            nodes=torch.tensor([r[0] for r in rows]), facts=torch.tensor([r[1] for r in rows]),
            signs=torch.tensor([r[2] for r in rows]),
            relations=torch.tensor([r[3] for r in rows]),
            directions=torch.tensor([r[4] for r in rows]),
            weight=torch.tensor(weights, dtype=torch.float64))
        return merge_cells(walks, node_type, edge_index, lambda s: (str(s), ("t",) * 3))

    def test_T9_a_path_and_its_reversal_are_one_cell_with_summed_weight(self) -> None:
        # 2 -cite(fact 2: 1->2, sign -1)- 1 -cite(fact 0: 0->1, sign -1)- 0, and its reversal
        forward = ([2, 1, 0], [2, 0], [-1, -1], [0, 0], [-1, -1])
        backward = ([0, 1, 2], [0, 2], [1, 1], [0, 0], [1, 1])
        groups = self._merge([forward, backward, forward], [0.5, 0.25, 0.5])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].num_faces, 1)
        self.assertAlmostEqual(float(groups[0].path_weight[0]), 1.25, places=6)
        # canonical reading: the smaller signature; directions are coded d + 1, so the
        # (cite -, cite -) reading [2, 1, 0] precedes (cite +, cite +)
        self.assertEqual(groups[0].face_node_index.tolist(), [[2, 1, 0]])

    def test_T9_paths_through_different_reciprocal_facts_are_different_cells(self) -> None:
        # 3 -cite(fact 4: 3->0)- 0 then to 1 via fact 0 (0->1, +) or fact 1 (1->0, -)
        via_forward = ([3, 0, 1], [4, 0], [1, 1], [0, 0], [1, 1])
        via_backward = ([3, 0, 1], [4, 1], [1, -1], [0, 0], [1, -1])
        groups = self._merge([via_forward, via_backward], [1.0, 1.0])
        cells = sum(g.num_faces for g in groups)
        self.assertEqual(cells, 2)
        for group in groups:
            keys = group.face_edge_index.tolist()
            self.assertEqual(len(keys), len(set(map(tuple, keys))))

    def test_T9_no_two_cells_share_a_key_in_real_draws(self) -> None:
        for seed in range(20):
            for group in _draw(seed, 5):
                keys = list(map(tuple, group.face_edge_index.tolist()))
                self.assertEqual(len(keys), len(set(keys)))
                reversed_keys = {k[::-1] for k in keys}
                palindromic = group.node_type_sequence == group.node_type_sequence[::-1]
                if not palindromic:
                    continue
                # a path and its reversal never both appear
                self.assertFalse(any(k in reversed_keys and k != k[::-1] for k in keys))


class StabilityTests(unittest.TestCase):
    """T13: lambda_max(D^-1/2 L D^-1/2) <= 2 + 1e-9 with m_f and signs, 60 fixtures."""

    def test_T13_lambda_max_at_most_two(self) -> None:
        from src.models.vertex_path_laplacian import VertexPathSheafLaplacian

        worst = 0.0
        for fixture in range(60):
            shared = fixture % 2 == 0
            generator = torch.Generator().manual_seed(1000 + fixture)
            paths = SignedPaths(7, (2, 3), 3, generator, signed=True)
            maps = _chain_operators(paths.edge_index.size(1), k=2, d=3, seed=fixture,
                                    dtype=torch.float64, shared=shared)
            terms = []
            for transport in paths.transports(maps, shared):
                m = 3.0 * (1.0 - torch.rand(transport.face_node_index.size(0),
                                            generator=generator, dtype=torch.float64))
                w = torch.rand(m.numel(), generator=generator, dtype=torch.float64)
                terms.append(PathCellTerm(transport, w * m))
            gamma = torch.rand(1, generator=generator, dtype=torch.float64)[0] * 3
            laplacian = VertexPathSheafLaplacian(maps, paths.edge_index, 7, tuple(terms), gamma,
                                                 checkpoint=False)
            basis = torch.eye(7 * 3, dtype=torch.float64).view(7 * 3, 7, 3)
            matrix = torch.stack([laplacian.apply_normalized(b).reshape(-1) for b in basis], 1)
            worst = max(worst, float(torch.linalg.eigvalsh(0.5 * (matrix + matrix.T))[-1]))
        self.assertLessEqual(worst, 2.0 + 1e-9)


class UniformInstanceTests(unittest.TestCase):
    """T16 (part): uniform_instance draws instances uniformly; it refuses L > 2."""

    def _hub(self):
        # anchor paper 0 with authors 1, 2, 3; author 1 has 30 more papers, authors 2, 3 have 3
        writes = [(0, 1), (0, 2), (0, 3)]
        paper = 4
        for author, count in ((1, 30), (2, 3), (3, 3)):
            for _ in range(count):
                writes.append((paper, author))
                paper += 1
        edge_index = torch.tensor(writes).t()
        node_type = torch.tensor([0] + [1, 1, 1] + [0] * (paper - 4))
        return edge_index, torch.zeros(len(writes), dtype=torch.long), node_type

    def _counts(self, sampler: str, draws: int) -> torch.Tensor:
        from src.data.path_sampler import build_half_edges, hop_csr, sample_walks

        edge_index, edge_type, node_type = self._hub()
        half = build_half_edges(edge_index, edge_type, node_type.numel(), (), True)
        csrs = [hop_csr(half, 0, 1), hop_csr(half, 0, -1)]
        walks = sample_walks(csrs, torch.tensor([0]), draws, torch.Generator().manual_seed(3),
                             uniform_instance=(sampler == "uniform_instance"))
        return torch.bincount(walks.nodes[:, 2], minlength=node_type.numel())[4:]

    def test_T16_uniform_instance_is_uniform_over_instances(self) -> None:
        draws = 36 * 1000
        counts = self._counts("uniform_instance", draws).double()
        accepted = counts.sum()
        expected = accepted / 36
        sigma = (expected * (1 - 1 / 36)) ** 0.5
        self.assertLess(float(((counts - expected).abs() / sigma).max()), 5.0)
        walk = self._counts("walk", draws).double()  # control: 1/90 vs 1/9 per instance
        self.assertGreater(float(((walk - walk.sum() / 36).abs()
                                  / (walk.sum() / 36) ** 0.5).max()), 10.0)

    def test_T16_uniform_instance_refuses_long_schemas(self) -> None:
        with self.assertRaisesRegex(ValueError, "L <= 2"):
            _draw(0, 2, sampler="uniform_instance",
                  schemas=(("PPPA", ("paper",) * 3 + ("author",), ((0, 1), (0, 1), (1, 1))),))


class SamplerConfigAndFreeWalkTests(unittest.TestCase):
    def test_config_rules(self) -> None:
        from src.configs.config import TrainConfig

        with self.assertRaisesRegex(ValueError, "dfs"):
            TrainConfig.from_dict({"data": {"dataset_name": "acm", "fact_layer": "tied",
                                            "path_sampler": "dfs"}})
        with self.assertRaises(ValueError):
            TrainConfig.from_dict({"data": {"dataset_name": "acm", "path_sampler": "bfs"}})
        for sampler in ("walk", "uniform_instance"):  # the schema sampler's variants
            TrainConfig.from_dict({"data": {"dataset_name": "acm", "fact_layer": "tied",
                                            "path_sampler": sampler, "path_source": "schemas"}})

    def test_batch_without_the_highest_self_inverse_relation(self) -> None:
        from src.data.path_sampler import build_free_path_groups

        # relations 0..3 self-inverse, but this batch only holds relation 0
        groups = build_free_path_groups(torch.tensor([[0, 1], [1, 2]]),
                                        torch.zeros(2, dtype=torch.long),
                                        torch.zeros(3, dtype=torch.long), None, (0, 1, 2, 3),
                                        tied=True, walks_per_anchor=4, seed=0)
        self.assertEqual(sum(g.num_faces for g in groups), 1)

    def test_schema_free_walks_on_a_self_inverse_relation(self) -> None:
        from src.data.fact_layer import FactLayer
        from src.data.path_sampler import build_free_path_groups

        # undirected 6-cycle plus a chord, stored once per pair (tied, rho(r) = r)
        pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)]
        edge_index = torch.tensor(pairs).t()
        groups = build_free_path_groups(edge_index, torch.zeros(7, dtype=torch.long),
                                        torch.zeros(6, dtype=torch.long), ("bond",), (0,),
                                        tied=True, walks_per_anchor=20, seed=0)
        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group.relation_index.unique().tolist(), [0])
        self.assertTrue(bool((group.face_edge_sign == 1).any() & (group.face_edge_sign == -1).any()))
        keys = {tuple(k) for k in group.face_edge_index.tolist()}
        self.assertEqual(len(keys), group.num_faces)
        self.assertFalse(any(k[::-1] in keys for k in keys))  # one cell per path up to reversal
        # 2-paths up to reversal = sum_v C(deg v, 2); with 20 walks per anchor all are drawn
        degree = torch.bincount(edge_index.reshape(-1), minlength=6)
        self.assertEqual(group.num_faces, int((degree * (degree - 1) // 2).sum()))
        self.assertTrue(bool((group.path_weight > 0).all()))


# ----------------------------------------------------------------------------- D8 (T12)
def _groups_equal(a, b) -> bool:
    return len(a) == len(b) and all(
        x.template_name == y.template_name
        and all(torch.equal(getattr(x, f), getattr(y, f))
                for f in ("face_node_index", "face_edge_index", "face_edge_sign", "path_weight"))
        for x, y in zip(a, b))


def _prepared_small(config):
    from src.data.dataset import build_node_classification_paths
    from src.data.structures import PreparedH2SData

    edge_index, edge_type, node_type, type_names, tuples = _small_hetero()
    layer = build_fact_layer(edge_index, edge_type, ("paper__cite__paper", "paper__to__author"),
                             9, "tied")
    slices = {"paper": (0, 6), "author": (6, 9)}
    groups = build_node_classification_paths(config, layer, slices, node_type, type_names,
                                             edge_index, edge_type, tuples, seed=config.seed)
    mask = torch.ones(6, dtype=torch.bool)
    return PreparedH2SData(
        x=torch.randn(9, 3), y=torch.zeros(9, dtype=torch.long), node_type=node_type,
        node_type_names=type_names, node_slices=slices, feature_dims={"paper": 3, "author": 3},
        edge_index=edge_index, edge_type=edge_type, relation_names=layer.kept_names,
        relation_tuples=tuples, train_mask=torch.ones(9, dtype=torch.bool),
        val_mask=torch.ones(9, dtype=torch.bool), test_mask=torch.ones(9, dtype=torch.bool),
        target_node_type="paper", target_indices=torch.arange(6),
        target_y=torch.zeros(6, dtype=torch.long), target_train_mask=mask,
        target_val_mask=mask, target_test_mask=mask, num_classes=2, is_multilabel=False,
        face_groups=groups, metrics=torch.zeros(0), fact_layer=layer)


def _small_config(resample: bool):
    from src.configs.config import TrainConfig

    return TrainConfig.from_dict({"device": "cpu", "seed": 7, "data": {
        "dataset_name": "acm", "fact_layer": "tied", "path_sampler": "walk",
        "resample_paths": resample, "face_sample_k": 3,
        "metapath_templates": [{"name": "PAP", "node_types": ["paper", "author", "paper"]},
                               {"name": "cocite", "node_types": ["paper"] * 3,
                                "hops": [["paper__cite__paper", "+"],
                                         ["paper__cite__paper", "-"]]}]}})


class ProtocolTests(unittest.TestCase):
    """T12: evaluation draw fixed; training draws differ between epochs; seeds reproduce."""

    def test_T12_node_classification(self) -> None:
        from src.trainers.trainer import Trainer

        for resample in (True, False):
            config = _small_config(resample)
            data = _prepared_small(config)
            trainer = Trainer.__new__(Trainer)
            trainer.cfg, trainer.device = config, torch.device("cpu")
            draws = [trainer.training_draw(data, data, epoch).face_groups for epoch in range(4)]
            again = [trainer.training_draw(data, data, epoch).face_groups for epoch in range(4)]
            with self.subTest(resample=resample):
                for first, second in zip(draws, again):  # same seed reproduces
                    self.assertTrue(_groups_equal(first, second))
                if resample:
                    self.assertTrue(_groups_equal(draws[0], data.face_groups))  # seed + 0
                    self.assertFalse(any(_groups_equal(draws[i], draws[j])
                                         for i in range(4) for j in range(i + 1, 4)))
                else:
                    self.assertTrue(all(d is data.face_groups for d in draws))
                # the evaluation draw is the one built with the data, rebuilt identically
                self.assertTrue(_groups_equal(_prepared_small(config).face_groups,
                                              data.face_groups))

    def test_T12_graph_classification(self) -> None:
        from torch_geometric.data import Batch, Data

        from src.trainers.graph_classification_trainer import GraphClassificationTrainer

        config = _small_config(True)
        trainer = GraphClassificationTrainer.__new__(GraphClassificationTrainer)
        trainer.cfg = config
        trainer.model = type("M", (), {"self_inverse_relations": (0,)})()
        self.assertEqual({trainer.draw_seed(e, False) for e in range(5)}, {config.seed})
        self.assertEqual([trainer.draw_seed(e, True) for e in range(3)],
                         [config.seed, config.seed + 1, config.seed + 2])
        graph = Data(edge_index=torch.tensor([[0, 1, 2, 3, 0, 1], [1, 2, 3, 4, 2, 4]]),
                     edge_type=torch.zeros(6, dtype=torch.long),
                     node_type=torch.zeros(5, dtype=torch.long), num_nodes=5)
        batch = Batch.from_data_list([graph, graph.clone()])
        same = [trainer.batch_path_groups(batch, seed=config.seed) for _ in range(2)]
        self.assertTrue(_groups_equal(*same))
        other = trainer.batch_path_groups(batch, seed=config.seed + 1)
        self.assertFalse(_groups_equal(same[0], other))
        no_resample = _small_config(False)
        trainer.cfg = no_resample
        self.assertEqual({trainer.draw_seed(e, True) for e in range(5)}, {no_resample.seed})

    def test_T12_config_rules(self) -> None:
        from src.configs.config import TrainConfig

        with self.assertRaises(ValueError):
            TrainConfig.from_dict({"data": {"dataset_name": "acm", "fact_layer": "untied",
                                            "path_sampler": "dfs", "resample_paths": True}})
        with self.assertRaises(ValueError):
            TrainConfig.from_dict({"data": {"dataset_name": "acm", "fact_layer": "tied",
                                            "path_sampler": "walk", "resample_paths": "yes"}})


# ----------------------------------------------------------------------------- D7 (T11)
def _write_tied_lastfm(root: Path) -> None:
    """Local HGB LastFM with a mirror pair and a self-inverse relation: rel 0 user->artist
    (target), rel 3 artist->user = exact reverse of rel 0 in link.dat, rel 1 user-user
    stored in both orientations, rel 2 artist->tag. link.dat.test: rel-0 pairs not in
    link.dat."""
    folder = root / "LastFM"
    folder.mkdir(parents=True)
    generator = torch.Generator().manual_seed(0)
    users, artists, tags = range(0, 12), range(12, 24), range(24, 30)
    with (folder / "node.dat").open("w", encoding="utf-8") as handle:
        for node_type, ids in enumerate((users, artists, tags)):
            for raw in ids:
                features = ",".join(f"{float(v):.4f}" for v in torch.rand(3, generator=generator))
                handle.write(f"{raw}\tn{raw}\t{node_type}\t{features}\n")
    train_pairs, test_pairs = set(), set()
    for user in users:
        chosen = torch.randperm(12, generator=generator)[:6].tolist()
        for rank, artist in enumerate(chosen):
            (test_pairs if rank == 5 else train_pairs).add((user, 12 + artist))
    friends = {(u, (u + 1) % 12) for u in users} | {(u, (u + 5) % 12) for u in users}
    with (folder / "link.dat").open("w", encoding="utf-8") as handle:
        for user, artist in sorted(train_pairs):
            handle.write(f"{user}\t{artist}\t0\t1.0\n")
            handle.write(f"{artist}\t{user}\t3\t1.0\n")
        for a, b in sorted(friends | {(b, a) for a, b in friends}):
            handle.write(f"{a}\t{b}\t1\t1.0\n")
        for artist in artists:
            handle.write(f"{artist}\t{24 + artist % 6}\t2\t1.0\n")
    with (folder / "link.dat.test").open("w", encoding="utf-8") as handle:
        for user, artist in sorted(test_pairs):
            handle.write(f"{user}\t{artist}\t0\t1.0\n")


def _lp_config(root: Path, mode: str, sampler: str, resample: bool = False):
    from src.configs.config import TrainConfig

    return TrainConfig.from_dict({"device": "cpu", "seed": 3, "data": {
        "dataset_name": "lastfm", "root": str(root), "fact_layer": mode,
        "path_sampler": sampler, "resample_paths": resample, "face_sample_k": 4}})


class LinkPredictionFactTests(unittest.TestCase):
    """T11: no held-out fact, in either orientation, in the message graph or on any path."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write_tied_lastfm(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build(self, mode: str, sampler: str, resample: bool = False):
        from src.data.link_prediction_dataset import build_link_prediction_dataset

        return build_link_prediction_dataset(_lp_config(self.root, mode, sampler, resample))

    def test_T11_split_is_by_fact(self) -> None:
        from src.data.link_prediction_dataset import _load_local_hgb_link_data

        split = _load_local_hgb_link_data(_lp_config(self.root, "untied", "dfs"))
        rel = {r[1]: r for r in split.valid_edge_index_by_relation}
        self.assertTrue(torch.equal(split.valid_edge_index_by_relation[rel["rel_3"]],
                                    split.valid_edge_index_by_relation[rel["rel_0"]].flip(0)))
        self.assertTrue(torch.equal(split.train_edge_index_by_relation[rel["rel_3"]],
                                    split.train_edge_index_by_relation[rel["rel_0"]].flip(0)))
        friends_valid = _pairs_of(split.valid_edge_index_by_relation[rel["rel_1"]])
        friends_train = _pairs_of(split.train_edge_index_by_relation[rel["rel_1"]])
        self.assertGreater(len(friends_valid), 0)
        self.assertEqual(friends_valid, {(b, a) for a, b in friends_valid})  # both sides
        self.assertFalse(friends_valid & friends_train)

    def test_T11_no_held_out_fact_in_message_graph_or_paths(self) -> None:
        from src.data.fact_layer import held_out_fact_leaks

        for mode, sampler in (("tied", "walk"), ("untied", "dfs"), ("untied", "walk")):
            with self.subTest(mode=mode, sampler=sampler):
                data = self._build(mode, sampler)
                layer = data.fact_layer
                names = layer.relation_names
                inverse = names.index("artist__rel_3__user")
                self.assertEqual(layer.rho[names.index("user__rel_0__artist")], inverse)
                held = torch.cat([data.val_pos_edge_index, data.test_pos_edge_index], dim=1)
                target = layer.relation_names.index("user__rel_0__artist")
                relation = torch.full((held.size(1),), target, dtype=torch.long)
                self.assertGreater(held.size(1), 0)
                self.assertGreater(sum(g.num_faces for g in data.face_groups), 0)
                self.assertEqual(held_out_fact_leaks(layer, data.edge_index, data.edge_type,
                                                     held, relation), 0)
                # independent, representation-free check: neither orientation of a
                # held-out pair is any message edge or any path edge
                pairs = _pairs_of(held) | {(b, a) for a, b in _pairs_of(held)}
                used = _pairs_of(data.edge_index)
                for group in data.face_groups:
                    used |= _pairs_of(data.edge_index[:, group.face_edge_index.reshape(-1)])
                self.assertFalse(pairs & used)
                # negative control: inject the reverse orientation of one held-out fact
                u, v = int(held[0, 0]), int(held[1, 0])
                if mode == "tied":  # the fact is stored in the orientation of its kept id
                    if target in layer.kept:
                        edge, etype = [[u], [v]], layer.kept.index(target)
                    else:
                        edge, etype = [[v], [u]], layer.kept.index(inverse)
                else:  # untied: the reverse orientation (v, rho(target), u)
                    edge, etype = [[v], [u]], inverse
                leaked_index = torch.cat([data.edge_index, torch.tensor(edge)], dim=1)
                leaked_type = torch.cat([data.edge_type, torch.tensor([etype])])
                self.assertEqual(held_out_fact_leaks(layer, leaked_index, leaked_type,
                                                     held, relation), 1)

    def test_T11_tied_layer_on_link_prediction(self) -> None:
        data = self._build("tied", "walk")
        summary = data.fact_layer.summary()
        self.assertEqual(summary["rho"]["user__rel_0__artist"], "artist__rel_3__user")
        self.assertEqual(summary["self_inverse_relations"], ["user__rel_1__user"])
        # one of the mirror pair is kept: the lower flattened id (names sort artist < user)
        self.assertIn("artist__rel_3__user", data.relation_names)
        self.assertNotIn("user__rel_0__artist", data.relation_names)
        friends = data.edge_index[:, data.edge_type == data.relation_names.index(
            "user__rel_1__user")]
        self.assertTrue(bool((friends[0] <= friends[1]).all()))  # one cell per pair

    def test_T12_link_prediction_resampling(self) -> None:
        from src.trainers.link_prediction_trainer import LinkPredictionTrainer

        data = self._build("tied", "walk", resample=True)
        trainer = LinkPredictionTrainer.__new__(LinkPredictionTrainer)
        trainer.cfg = _lp_config(self.root, "tied", "walk", True)
        trainer.device = torch.device("cpu")
        draws = [trainer.training_draw(data, data, e).face_groups for e in range(3)]
        again = [trainer.training_draw(data, data, e).face_groups for e in range(3)]
        for first, second in zip(draws, again):
            self.assertTrue(_groups_equal(first, second))
        self.assertTrue(_groups_equal(draws[0], data.face_groups))  # seed + 0 = eval draw
        self.assertFalse(_groups_equal(draws[1], draws[2]))
        trainer.cfg = _lp_config(self.root, "tied", "walk", False)
        self.assertIs(trainer.training_draw(data, data, 5), data)

    def test_T11_config(self) -> None:
        with self.assertRaises(ValueError):
            _lp_config(self.root, "tied", "uniform_instance")


class BaselineBundleIdentityTests(unittest.TestCase):
    """D7 option A (plan §9.5): baseline bundles keep the pre-D7 data whatever the H2S-NN defaults."""

    @staticmethod
    def _explicit_old_config(name: str):
        from src.configs.config import TrainConfig

        return TrainConfig.from_dict({"data": {
            "dataset_name": name, "run_index": 0, "fact_layer": "untied",
            "path_sampler": "dfs", "resample_paths": False, "path_source": "schemas"},
            "seed": 42,
            "device": "cpu"})

    def test_defaults_are_the_model(self) -> None:
        from src.configs.config import DataConfig, TrainConfig

        for data in (DataConfig(), TrainConfig.from_dict({"data": {"dataset_name": "acm"}}).data):
            self.assertEqual((data.fact_layer, data.path_sampler, data.resample_paths),
                             ("tied", "walk", True))

    def test_node_bundle_equals_explicit_untied_dfs(self) -> None:
        from src.baselines.bundles import build_bundle
        from src.data.dataset import build_dataset

        bundle = build_bundle("node_classification", "acm", overrides={"device": "cpu"}).data
        old = build_dataset(self._explicit_old_config("acm"))
        for field in ("x", "y", "edge_index", "edge_type", "target_y", "train_mask"):
            self.assertTrue(torch.equal(getattr(bundle, field), getattr(old, field)), field)
        self.assertEqual(bundle.relation_names, old.relation_names)
        self.assertEqual(len(bundle.face_groups), len(old.face_groups))
        for a, b in zip(bundle.face_groups, old.face_groups):
            for field in ("face_node_index", "face_edge_index", "relation_index", "face_edge_sign"):
                self.assertTrue(torch.equal(getattr(a, field), getattr(b, field)), field)
        self.assertEqual(bundle.fact_layer.mode, "untied")

    def test_graph_bundle_equals_explicit_untied(self) -> None:
        from src.baselines.bundles import build_bundle
        from src.data.graph_classification_dataset import build_graph_classification_dataset

        fold = build_bundle("graph_classification", "mutag", overrides={"device": "cpu"}).fold
        old = build_graph_classification_dataset(self._explicit_old_config("mutag"))
        for split in ("train", "val", "test"):
            a, b = getattr(fold, f"{split}_loader").dataset, getattr(old, f"{split}_loader").dataset
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assertTrue(torch.equal(x.edge_index, y.edge_index))
                self.assertTrue(torch.equal(x.edge_type, y.edge_type))

    def test_pins_cannot_be_overridden(self) -> None:
        from src.baselines.bundles import _base_config

        for key, value in (("fact_layer", "tied"), ("path_sampler", "walk"), ("resample_paths", True)):
            with self.assertRaises(ValueError):
                _base_config("acm", run_index=0, seed=42, overrides={"data": {key: value}})


def _pairs_of(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in edge_index.t().tolist()}


if __name__ == "__main__":
    unittest.main()
