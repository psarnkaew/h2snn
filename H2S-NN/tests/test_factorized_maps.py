"""Task card D1: factorized, matrix-free, conformal-orthogonal restriction maps.

    F_{v<|e} = c_{v<|e} * R_{psi(e),side} * H_k(u_{v<|e})

Test ids follow the task card (T1-T11). Every property test that could pass vacuously
has a negative control showing that the same check fails when the property is broken.
"""

from __future__ import annotations

import math
import unittest

try:
    import torch
    from torch import nn
    from torch.utils._python_dispatch import TorchDispatchMode
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.configs.config import ModelConfig, TrainConfig
from src.data.structures import FaceGroup, PreparedH2SData
from src.models.coboundary_ops import CoboundaryOps
from src.models.h2snn_model import ModelDims, build_model
from src.models.map_operators import (
    DenseMapOperator,
    EdgeMapPair,
    FactorizedMapOperator,
    PathTransportOperator,
    StackedMapOperator,
    normalize_reflectors,
)
from src.models.pooling_utils import (
    pool_edge_stalks_to_nodes,
    pool_edge_stalks_to_nodes_matrix_free,
)
from src.models.sheaf_maps import (
    FaceMapCache,
    FactorizedNodeToEdgeSheaf,
    NodeToEdgeSheaf,
    NodeToEdgeSheafOutput,
    _gather_factorized_path_maps,
    compose_face_transport,
)

F64_TOL = 1e-10
F32_TOL = 1e-5


# ----------------------------------------------------------------------------- fixtures
def _random_pap_data(seed: int = 0, num_papers: int = 24, num_authors: int = 16,
                     max_faces: int = 40) -> PreparedH2SData:
    """Paper/author graph (both edge directions stored) with PAP (L=2) and PAPAP (L=4)
    path faces, every path edge traversed along its stored direction, paths simple."""
    generator = torch.Generator().manual_seed(seed)
    num_nodes = num_papers + num_authors
    src: list[int] = []
    dst: list[int] = []
    rel: list[int] = []
    for paper in range(num_papers):
        for author in torch.randperm(num_authors, generator=generator)[:3].tolist():
            src += [paper, num_papers + author]
            dst += [num_papers + author, paper]
            rel += [0, 1]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(rel, dtype=torch.long)
    out_edges: dict[int, list[tuple[int, int]]] = {node: [] for node in range(num_nodes)}
    for edge_id, (u, v) in enumerate(zip(src, dst)):
        out_edges[u].append((edge_id, v))

    groups: list[FaceGroup] = []
    closures: list[torch.Tensor] = []
    for name, length in (("PAP", 2), ("PAPAP", 4)):
        seen: set[tuple[int, ...]] = set()
        node_seqs: list[list[int]] = []
        edge_seqs: list[list[int]] = []
        attempts = 0
        while len(node_seqs) < max_faces and attempts < 50 * max_faces:
            attempts += 1
            nodes = [int(torch.randint(num_papers, (1,), generator=generator))]
            edges: list[int] = []
            for _ in range(length):
                options = out_edges[nodes[-1]]
                edge_id, nxt = options[int(torch.randint(len(options), (1,), generator=generator))]
                if nxt in nodes:
                    break
                nodes.append(nxt)
                edges.append(edge_id)
            if len(edges) == length and tuple(nodes) not in seen:
                seen.add(tuple(nodes))
                node_seqs.append(nodes)
                edge_seqs.append(edges)
        face_node_index = torch.tensor(node_seqs, dtype=torch.long)
        face_edge_index = torch.tensor(edge_seqs, dtype=torch.long)
        closure = torch.stack([face_node_index[:, -1], face_node_index[:, 0]], dim=1)
        types = tuple("paper" if i % 2 == 0 else "author" for i in range(length + 1))
        groups.append(
            FaceGroup(
                template_name=name,
                node_type_sequence=types,
                face_node_index=face_node_index,
                face_edge_index=face_edge_index,
                relation_index=edge_type[face_edge_index],
                face_edge_sign=torch.ones(face_edge_index.shape),
                closure_edge_index=closure,
            )
        )
        closures.append(closure.t())

    x = torch.randn(num_nodes, 6, generator=generator)
    target_y = torch.randint(3, (num_papers,), generator=generator)
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    y[:num_papers] = target_y
    split = torch.arange(num_papers) % 3
    pad = torch.zeros(num_authors, dtype=torch.bool)
    return PreparedH2SData(
        x=x,
        y=y,
        node_type=torch.cat([torch.zeros(num_papers), torch.ones(num_authors)]).long(),
        node_type_names=("paper", "author"),
        node_slices={"paper": (0, num_papers), "author": (num_papers, num_nodes)},
        feature_dims={"paper": 6, "author": 6},
        edge_index=edge_index,
        edge_type=edge_type,
        relation_names=("paper__to__author", "author__to__paper"),
        relation_tuples=(("paper", "to", "author"), ("author", "to", "paper")),
        train_mask=torch.cat([split == 0, pad]),
        val_mask=torch.cat([split == 1, pad]),
        test_mask=torch.cat([split == 2, pad]),
        target_node_type="paper",
        target_indices=torch.arange(num_papers),
        target_y=target_y,
        target_train_mask=split == 0,
        target_val_mask=split == 1,
        target_test_mask=split == 2,
        num_classes=3,
        is_multilabel=False,
        face_groups=tuple(groups),
        closure_edge_index=torch.cat(closures, dim=1),
        metrics=torch.zeros(8),
    )


def _as_float64(data: PreparedH2SData) -> PreparedH2SData:
    fields = {name: getattr(data, name) for name in PreparedH2SData.__slots__}
    fields["x"] = data.x.double()
    return PreparedH2SData(**fields)


def _random_operator(num_edges: int, k: int, d: int, num_groups: int, seed: int,
                     dtype: torch.dtype = torch.float64,
                     group_ids: torch.Tensor | None = None) -> FactorizedMapOperator:
    generator = torch.Generator().manual_seed(seed)
    rotations, _ = torch.linalg.qr(torch.randn(num_groups, d, d, generator=generator, dtype=dtype))
    if group_ids is None:
        group_ids = torch.randint(num_groups, (num_edges,), generator=generator)
    return FactorizedMapOperator.from_raw(
        raw_reflectors=torch.randn(num_edges, k, d, generator=generator, dtype=dtype),
        raw_scale=torch.randn(num_edges, generator=generator, dtype=dtype),
        rotations=rotations,
        group_ids=group_ids,
        scale_eps=0.1,
        reflector_eps=1e-6,
    )


def _module_maps(data: PreparedH2SData, d: int = 8, k: int = 2, seed: int = 0,
                 dtype: torch.dtype = torch.float64):
    """A float64 FactorizedNodeToEdgeSheaf, node states h, and its maps on `data`."""
    torch.manual_seed(seed)
    num_relations = len(data.relation_names)
    module = FactorizedNodeToEdgeSheaf(
        d=d, d_r=4, num_relations=num_relations, hidden_dim=16, householder_k=k,
        scale_eps=0.1, orthogonal_map="fasthpp",
    ).to(dtype)
    h = torch.randn(data.num_nodes, d, dtype=dtype)
    relation_embeddings = torch.randn(num_relations + 1, 4, dtype=dtype)
    output = module(h, data.edge_index, data.edge_type, relation_embeddings)
    return module, h, relation_embeddings, output


def _config(**model: object) -> TrainConfig:
    base = {"d": 8, "d_r": 4, "d_h": 8, "K": 2, "classifier_hidden_dim": 16}
    base.update(model)
    return TrainConfig.from_dict({"device": "cpu", "checkpoint_metric": "acc", "model": base})


def _mv(matrices: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    return (matrices @ vectors.unsqueeze(-1)).squeeze(-1)


def _max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


# ----------------------------------------------------------------------------- T1-T3
class OperatorAlgebraTests(unittest.TestCase):
    def test_T1_rotation_and_householder_factors_are_orthogonal(self) -> None:
        data = _random_pap_data()
        for dtype, tol in ((torch.float64, F64_TOL), (torch.float32, F32_TOL)):
            with self.subTest(dtype=dtype):
                module, _, _, output = _module_maps(data, dtype=dtype)
                eye = torch.eye(module.d, dtype=dtype)
                rotations = module.rotations().reshape(-1, module.d, module.d)
                self.assertLess(_max_err(rotations @ rotations.transpose(-2, -1), eye), tol)
                for side in (output.maps.src, output.maps.tgt):
                    product = side.reflection_product()
                    self.assertLess(_max_err(product @ product.transpose(-2, -1), eye), tol)
                    # Hence F F^T = c^2 I: conformal-orthogonal (relative to c^2).
                    maps = side.materialize()
                    gram = maps @ maps.transpose(-2, -1) / side.scale.view(-1, 1, 1) ** 2
                    self.assertLess(_max_err(gram, eye), tol)

    def test_T1_negative_control_detects_a_non_orthogonal_factor(self) -> None:
        operator = _random_operator(6, 2, 5, 2, seed=1)
        broken = FactorizedMapOperator(
            operator.scale, operator.reflectors * 0.9, operator.rotations, operator.group_ids
        )
        product = broken.reflection_product()
        self.assertGreater(_max_err(product @ product.transpose(-2, -1), torch.eye(5, dtype=torch.float64)), 1e-2)

    def test_T2_apply_and_apply_T_match_dense_reference(self) -> None:
        for k in (1, 2, 4):
            with self.subTest(k=k):
                operator = _random_operator(11, k, 6, 3, seed=k)
                dense = operator.materialize()
                x = torch.randn(11, 6, dtype=torch.float64)
                batch = torch.randn(11, 3, 6, dtype=torch.float64)
                self.assertLess(_max_err(operator.apply(x), _mv(dense, x)), F64_TOL)
                self.assertLess(
                    _max_err(operator.apply_T(x), _mv(dense.transpose(-2, -1), x)), F64_TOL
                )
                self.assertLess(
                    _max_err(operator.apply(batch), _mv(dense.unsqueeze(1), batch)), F64_TOL
                )
                # The reference is built independently (explicit outer products), so a
                # wrong reflection order in `apply` would show up here.
                reversed_order = FactorizedMapOperator(
                    operator.scale, operator.reflectors.flip(1), operator.rotations,
                    operator.group_ids,
                )
                if k > 1:
                    self.assertGreater(
                        _max_err(reversed_order.apply(x), _mv(dense, x)), 1e-3
                    )

    def test_T3_closed_form_inverse_is_the_true_inverse(self) -> None:
        operator = _random_operator(13, 2, 6, 3, seed=7)
        x = torch.randn(13, 6, dtype=torch.float64)
        self.assertLess(_max_err(operator.apply_inverse(operator.apply(x)), x), F64_TOL)
        self.assertLess(_max_err(operator.apply(operator.apply_inverse(x)), x), F64_TOL)
        self.assertLess(_max_err(operator.apply_inverse_T(operator.apply_T(x)), x), F64_TOL)
        # Against an independent general inverse, not just self-consistency.
        reference = torch.linalg.inv(operator.materialize())
        self.assertLess(_max_err(operator.materialize_inverse(), reference), F64_TOL)
        self.assertLess(_max_err(operator.apply_inverse(x), _mv(reference, x)), F64_TOL)
        self.assertLess(
            _max_err(operator.apply_inverse_T(x), _mv(reference.transpose(-2, -1), x)), F64_TOL
        )
        maps, inverses = operator.materialize_with_inverse()
        self.assertLess(_max_err(maps, operator.materialize()), F64_TOL)
        self.assertLess(_max_err(inverses, reference), F64_TOL)

    def test_stacked_and_dense_operators_match_their_matrices(self) -> None:
        factorized = _random_operator(5, 2, 4, 2, seed=3)
        closure = torch.randn(3, 4, 4, dtype=torch.float64)
        stacked = StackedMapOperator((factorized, DenseMapOperator(closure)))
        dense = torch.cat([factorized.materialize(), closure], dim=0)
        x = torch.randn(8, 4, dtype=torch.float64)
        self.assertEqual(stacked.num_edges, 8)
        self.assertLess(_max_err(stacked.apply(x), _mv(dense, x)), F64_TOL)
        self.assertLess(_max_err(stacked.apply_T(x), _mv(dense.transpose(-2, -1), x)), F64_TOL)


# ----------------------------------------------------------------------------- T4, T5
class SideAndNodeDependenceTests(unittest.TestCase):
    def test_T4_maps_are_side_specific_and_node_dependent(self) -> None:
        data = _random_pap_data()
        module, h, relation_embeddings, output = _module_maps(data)
        src = output.maps.src.materialize()
        tgt = output.maps.tgt.materialize()
        side_gap = (src - tgt).flatten(1).norm(dim=1)
        self.assertGreater(float(side_gap.min()), 1e-3, "F_src == F_tgt on some edge")

        node = 3
        perturbed = h.clone()
        perturbed[node] += torch.randn(module.d, generator=torch.Generator().manual_seed(5),
                                       dtype=h.dtype)
        moved = module(perturbed, data.edge_index, data.edge_type, relation_embeddings).maps
        src_change = (moved.src.materialize() - src).flatten(1).norm(dim=1)
        tgt_change = (moved.tgt.materialize() - tgt).flatten(1).norm(dim=1)
        as_src = data.edge_index[0] == node
        as_tgt = data.edge_index[1] == node
        self.assertTrue(bool(as_src.any()) and bool(as_tgt.any()))
        self.assertGreater(float(src_change[as_src].min()), 1e-3)
        self.assertGreater(float(tgt_change[as_tgt].min()), 1e-3)
        # Locality (fix spec F5): maps of edges not touching the node are untouched.
        untouched = ~(as_src | as_tgt)
        self.assertEqual(float(src_change[untouched].max()), 0.0)
        self.assertEqual(float(tgt_change[untouched].max()), 0.0)


def _holonomies(maps: EdgeMapPair, face_edge_index: torch.Tensor) -> torch.Tensor:
    """Path holonomy per instance via the existing v1 transport construction."""
    F_src_path, F_tgt_path, F_src_inv = _gather_factorized_path_maps(maps, face_edge_index)
    eye = torch.eye(F_src_path.size(-1), dtype=F_src_path.dtype)
    A_f = eye.expand(face_edge_index.size(0), -1, -1)
    return compose_face_transport(A_f, F_src_path, F_tgt_path, F_src_inv).closure_F_src


def _holonomy_statistics(holonomy: torch.Tensor) -> tuple[float, float]:
    """(min_f ||Hol_f - I||_F / sqrt(d), mean_f ||Hol_f - mean Hol||_F / sqrt(d))."""
    d = holonomy.size(-1)
    eye = torch.eye(d, dtype=holonomy.dtype)
    deviation = (holonomy - eye).flatten(1).norm(dim=1) / math.sqrt(d)
    spread = (holonomy - holonomy.mean(dim=0)).flatten(1).norm(dim=1) / math.sqrt(d)
    return float(deviation.min()), float(spread.mean())


class HolonomyTests(unittest.TestCase):
    """T5. Pre-registered thresholds: every instance deviates from I by > 0.1 (per-entry
    RMS), and the mean spread around the group mean exceeds 0.05. The negative control
    (src map tied to tgt map) must fall below 1e-10 on both, i.e. fail the positive check."""

    DEVIATION_MIN = 0.1
    SPREAD_MIN = 0.05

    def test_T5_holonomy_is_nontrivial_and_varies_within_a_relation_sequence(self) -> None:
        data = _random_pap_data()
        _, _, _, output = _module_maps(data)
        for group in data.face_groups:
            with self.subTest(template=group.template_name):
                # Same relation sequence across the whole group (one schema per group).
                self.assertEqual(int(torch.unique(group.relation_index, dim=0).size(0)), 1)
                self.assertGreaterEqual(group.num_faces, 10)
                deviation, spread = _holonomy_statistics(
                    _holonomies(output.maps, group.face_edge_index)
                )
                self.assertGreater(deviation, self.DEVIATION_MIN)
                self.assertGreater(spread, self.SPREAD_MIN)

    def test_T5_negative_control_tied_maps_collapse_holonomy_to_identity(self) -> None:
        data = _random_pap_data()
        _, _, _, output = _module_maps(data)
        tied = EdgeMapPair(src=output.maps.src, tgt=output.maps.src)
        for group in data.face_groups:
            with self.subTest(template=group.template_name):
                deviation, spread = _holonomy_statistics(
                    _holonomies(tied, group.face_edge_index)
                )
                self.assertLess(deviation, 1e-10)
                self.assertLess(spread, 1e-10)
                self.assertFalse(deviation > self.DEVIATION_MIN or spread > self.SPREAD_MIN)


# ----------------------------------------------------------------------------- T6-T8
class _Path:
    """One simple path v0 -> ... -> vL on nodes 0..L, edges 0..L-1, factorized maps."""

    def __init__(self, length: int, d: int, k: int, seed: int) -> None:
        self.length = length
        self.d = d
        group_ids = torch.arange(length) % 2
        self.src = _random_operator(length, k, d, 2, seed, group_ids=group_ids)
        self.tgt = _random_operator(length, k, d, 2, seed + 100, group_ids=group_ids)
        self.face_node_index = torch.arange(length + 1).view(1, -1)
        self.face_edge_index = torch.arange(length).view(1, -1)
        self.h0 = torch.randn(length + 1, d, generator=torch.Generator().manual_seed(seed + 3),
                              dtype=torch.float64)

    def transport(self, A_f: torch.Tensor):
        F_src, F_tgt, F_src_inv = _gather_factorized_path_maps(
            EdgeMapPair(self.src, self.tgt), self.face_edge_index
        )
        return compose_face_transport(A_f, F_src, F_tgt, F_src_inv)

    def dense_transports(self) -> list[torch.Tensor]:
        """T_{v_i} = prod_{j<=i} (F_src^j)^-1 F_tgt^j via general inverses (A_f = I)."""
        src, tgt = self.src.materialize(), self.tgt.materialize()
        current = torch.eye(self.d, dtype=torch.float64)
        out = [current]
        for j in range(self.length):
            current = current @ torch.linalg.inv(src[j]) @ tgt[j]
            out.append(current)
        return out


def _delta1_delta0_residual(path: _Path, closure_src: torch.Tensor,
                            closure_tgt: torch.Tensor, face_maps: torch.Tensor) -> float:
    maps = EdgeMapPair(
        src=StackedMapOperator((path.src, DenseMapOperator(closure_src))),
        tgt=StackedMapOperator((path.tgt, DenseMapOperator(closure_tgt))),
    )
    nodes = list(range(path.length + 1))
    edge_index = torch.tensor([nodes[:-1] + [nodes[-1]], nodes[1:] + [nodes[0]]])
    delta0 = CoboundaryOps.delta0_matrix_free(path.h0, maps, edge_index[0], edge_index[1])
    cache = FaceMapCache(
        template_name="path",
        face_edge_index=torch.arange(path.length + 1).view(1, -1),
        relation_index=torch.zeros(1, path.length, dtype=torch.long),
        face_edge_sign=torch.ones(1, path.length + 1, dtype=torch.float64),
        maps=face_maps,
        group_id=0,
        instance_start=0,
        instance_end=1,
    )
    delta1 = CoboundaryOps.delta1(delta0, (cache,), lambda c: c.maps).concatenated
    return float(delta1.norm() / delta0.norm())


class ChainConditionTests(unittest.TestCase):
    def test_T6_delta1_delta0_vanishes_with_factorized_maps(self) -> None:
        for length in (2, 3, 4):
            for k in (1, 2, 4):
                with self.subTest(length=length, k=k):
                    path = _Path(length, d=6, k=k, seed=10 * length + k)
                    A_f = _random_operator(1, 2, 6, 1, seed=99).materialize()
                    transport = path.transport(A_f)
                    residual = _delta1_delta0_residual(
                        path, transport.closure_F_src, transport.closure_F_tgt,
                        transport.face_maps,
                    )
                    self.assertLess(residual, F64_TOL)

    def test_T6_negative_control_wrong_closure_map_breaks_the_chain(self) -> None:
        path = _Path(4, d=6, k=2, seed=3)
        transport = path.transport(torch.eye(6, dtype=torch.float64).unsqueeze(0))
        identity = torch.eye(6, dtype=torch.float64).unsqueeze(0)
        residual = _delta1_delta0_residual(path, identity, identity, transport.face_maps)
        self.assertGreater(residual, 1e-2)

    def test_T6_full_model_chain_condition_float64(self) -> None:
        data = _as_float64(_random_pap_data())
        for k in (1, 2, 4):
            with self.subTest(k=k):
                torch.manual_seed(k)
                model = build_model(_config(map_family="factorized", householder_k=k), data)
                model = model.double().eval()
                with torch.no_grad():
                    output = model(data)
                residual = float(output.delta1_delta0_h0.norm() / output.delta0_h0.norm())
                self.assertLess(residual, F64_TOL)

    def test_T7_functoriality_via_either_boundary_edge_including_closure(self) -> None:
        for length in (2, 4):
            with self.subTest(length=length):
                path = _Path(length, d=5, k=2, seed=length)
                A_f = _random_operator(1, 3, 5, 1, seed=41).materialize()
                transport = path.transport(A_f)
                face = transport.face_maps[0]  # [L+1, d, d]: path edges, then closure
                src, tgt = path.src.materialize(), path.tgt.materialize()
                closure_src, closure_tgt = transport.closure_F_src[0], transport.closure_F_tgt[0]
                L = length
                via = {0: (face[0] @ src[0], face[L] @ closure_tgt),
                       L: (face[L - 1] @ tgt[L - 1], face[L] @ closure_src)}
                for i in range(1, L):
                    via[i] = (face[i - 1] @ tgt[i - 1], face[i] @ src[i])
                dense = path.dense_transports()
                for i, (first, second) in via.items():
                    self.assertLess(_max_err(first, second), F64_TOL, f"node {i}")
                    self.assertLess(_max_err(first, A_f[0] @ dense[i]), F64_TOL, f"node {i}")

    def test_T7_negative_control_free_face_map_breaks_functoriality(self) -> None:
        path = _Path(3, d=5, k=2, seed=8)
        transport = path.transport(torch.eye(5, dtype=torch.float64).unsqueeze(0))
        face = transport.face_maps[0].clone()
        face[1] = face[1] + 0.1 * torch.randn(5, 5, dtype=torch.float64)
        src, tgt = path.src.materialize(), path.tgt.materialize()
        self.assertGreater(_max_err(face[0] @ tgt[0], face[1] @ src[1]), 1e-3)

    def test_T8_matrix_free_path_transport_equals_dense_product(self) -> None:
        data = _random_pap_data()
        _, _, _, output = _module_maps(data)
        src, tgt = output.maps.src.materialize(), output.maps.tgt.materialize()
        x = torch.randn(data.num_nodes, 8, dtype=torch.float64)
        for group in data.face_groups:
            with self.subTest(template=group.template_name):
                operator = PathTransportOperator(
                    output.maps.src, output.maps.tgt, group.face_node_index,
                    group.face_edge_index, data.edge_index,
                )
                values = operator.apply(x)
                self.assertEqual(tuple(values.shape), (group.num_faces, group.length + 1, 8))
                reference = torch.empty_like(values)
                for f in range(group.num_faces):
                    current = torch.eye(8, dtype=torch.float64)
                    reference[f, 0] = x[group.face_node_index[f, 0]]
                    for j in range(group.length):
                        e = group.face_edge_index[f, j]
                        current = current @ torch.linalg.inv(src[e]) @ tgt[e]
                        reference[f, j + 1] = current @ x[group.face_node_index[f, j + 1]]
                self.assertLess(_max_err(values, reference), F64_TOL)
                w = torch.randn_like(values)
                lhs = float((values * w).sum())
                rhs = float((x * operator.apply_T(w, data.num_nodes)).sum())
                self.assertLess(abs(lhs - rhs), F64_TOL * max(1.0, abs(lhs)))

    def test_T8_path_transport_rejects_edges_against_the_path(self) -> None:
        data = _random_pap_data()
        _, _, _, output = _module_maps(data)
        group = data.face_groups[0]
        reversed_edges = data.edge_index.flip(0)  # every edge now points against the path
        with self.assertRaisesRegex(ValueError, "stored src->tgt direction"):
            PathTransportOperator(output.maps.src, output.maps.tgt, group.face_node_index,
                                  group.face_edge_index, reversed_edges)


# ----------------------------------------------------------------------------- T9, T10
class NumericalGuardTests(unittest.TestCase):
    def test_T9_finite_gradients_with_near_zero_reflectors(self) -> None:
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                generator = torch.Generator().manual_seed(0)
                raw = torch.randn(6, 2, 4, generator=generator, dtype=dtype)
                raw[0, 0] = 0.0
                raw[1, 1] = 1e-20
                raw[2, 0] = 1e-7
                raw.requires_grad_(True)
                raw_scale = torch.full((6,), -50.0, dtype=dtype, requires_grad=True)
                rotations, _ = torch.linalg.qr(torch.randn(2, 4, 4, generator=generator, dtype=dtype))
                operator = FactorizedMapOperator.from_raw(
                    raw, raw_scale, rotations, torch.tensor([0, 1, 0, 1, 0, 1]),
                    scale_eps=0.1, reflector_eps=1e-6,
                )
                self.assertGreaterEqual(float(operator.scale.min()), 0.1)
                x = torch.randn(6, 4, generator=generator, dtype=dtype)
                total = (operator.apply(x).square().sum() + operator.apply_inverse(x).square().sum()
                         + operator.apply_T(x).sum())
                total.backward()
                self.assertTrue(bool(torch.isfinite(total)))
                self.assertTrue(bool(torch.isfinite(raw.grad).all()))
                self.assertTrue(bool(torch.isfinite(raw_scale.grad).all()))

    def test_T9_zero_reflector_mlp_keeps_model_finite(self) -> None:
        data = _random_pap_data()
        torch.manual_seed(0)
        model = build_model(_config(map_family="factorized"), data)
        last = model.node_to_edge.reflector_mlp[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()  # u = 0 exactly for every incidence
        output = model(data)
        (output.logits.square().sum() + output.s_star.square().sum()).backward()
        self.assertTrue(bool(torch.isfinite(output.logits).all()))
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)

    def test_normalized_reflectors_are_unit_above_eps(self) -> None:
        raw = torch.tensor([[3.0, 4.0], [1e-3, 0.0], [0.0, 0.0]], dtype=torch.float64)
        unit = normalize_reflectors(raw, eps=1e-6)
        self.assertLess(_max_err(unit.norm(dim=-1)[:2], torch.ones(2, dtype=torch.float64)), F64_TOL)
        self.assertEqual(float(unit[2].abs().max()), 0.0)

    def test_T10_storage_matches_the_bound_exactly(self) -> None:
        """Counts the tensors the operators actually hold (found generically, after an
        apply has built the lazy grouping), not the operator's own report."""

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
                _, _, _, output = _module_maps(data, d=d, k=k)
                floats = 0
                integers = 0
                for side in (output.maps.src, output.maps.tgt):
                    side.apply(torch.zeros(M, d, dtype=torch.float64))  # builds the layout
                    tensors = held_tensors(side)
                    floats += sum(t.numel() for t in tensors if t.is_floating_point())
                    integers += sum(t.numel() for t in tensors if not t.is_floating_point())
                    self.assertEqual(
                        side.stored_elements(),
                        sum(t.numel() for t in tensors if t.is_floating_point()),
                    )
                self.assertEqual(floats, 2 * M * (k * d + 1) + 2 * num_relations * d * d)
                # Integer index storage per side: relation ids (a reference to edge_type)
                # plus the relation-sort permutation and its inverse.
                self.assertLessEqual(integers, 2 * 3 * M)
                if (d, k) == (32, 2):
                    self.assertGreaterEqual(2 * M * d * d / floats, 10.0)


# ----------------------------------------------------------------------------- integration
class _MaterializedFactorizedMaps(nn.Module):
    """Feeds a factorized module's maps, materialized, into the v1 dense pipeline."""

    def __init__(self, factorized: FactorizedNodeToEdgeSheaf) -> None:
        super().__init__()
        self.factorized = factorized

    def forward(self, h, edge_index, edge_type, relation_embeddings) -> NodeToEdgeSheafOutput:
        maps = self.factorized(h, edge_index, edge_type, relation_embeddings).maps
        F_src, F_tgt = maps.src.materialize(), maps.tgt.materialize()
        return NodeToEdgeSheafOutput(
            F_src=F_src,
            F_tgt=F_tgt,
            delta0_h0=CoboundaryOps.delta0(h, F_src, F_tgt, edge_index[0], edge_index[1]),
            F_src_scale=maps.src.scale,
        )


class _ShapeRecorder(TorchDispatchMode):
    """Records the shape of every tensor any aten op produces (forward and backward)."""

    def __init__(self) -> None:
        super().__init__()
        self.phase = "forward"
        self.shapes: list[tuple[str, tuple[int, ...]]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for item in result if isinstance(result, (tuple, list)) else (result,):
            if isinstance(item, torch.Tensor):
                self.shapes.append((self.phase, tuple(item.shape)))
        return result


class ModelIntegrationTests(unittest.TestCase):
    def test_factorized_model_equals_v1_pipeline_fed_the_same_maps(self) -> None:
        """Matrix-free delta0/L0/L1/pooling == v1 dense code on the materialized maps."""
        data = _as_float64(_random_pap_data())
        torch.manual_seed(0)
        factorized = build_model(_config(map_family="factorized", K=3), data).double().eval()
        dense = build_model(_config(K=3), data).double().eval()
        shared = {key: value for key, value in factorized.state_dict().items()
                  if not key.startswith("node_to_edge.")}
        missing, unexpected = dense.load_state_dict(shared, strict=False)
        self.assertTrue(all(key.startswith("node_to_edge.") for key in missing))
        self.assertEqual(unexpected, [])
        dense.node_to_edge = _MaterializedFactorizedMaps(factorized.node_to_edge)
        with torch.no_grad():
            got = factorized(data)
            want = dense(data)
        for field in ("logits", "x_star", "s_star", "delta0_h0", "delta0_x_star", "raw_face_curl"):
            self.assertLess(_max_err(getattr(got, field), getattr(want, field)), F64_TOL, field)

        # Backward too: every parameter gradient (checkpointed matrix-free path vs v1 code).
        weights = torch.randn_like(got.logits)
        gradients = []
        for model in (factorized, dense):
            factorized.zero_grad(set_to_none=True)
            dense.zero_grad(set_to_none=True)
            output = model(data)
            ((output.logits * weights).sum() + output.s_star.square().mean()).backward()
            named = model.named_parameters()
            gradients.append({name.replace("node_to_edge.factorized.", "node_to_edge."): p.grad
                              for name, p in named if p.grad is not None})
        self.assertEqual(set(gradients[0]), set(gradients[1]))
        self.assertTrue(any(name.startswith("node_to_edge.") for name in gradients[0]))
        for name, grad in gradients[0].items():
            self.assertLess(_max_err(grad, gradients[1][name]), F64_TOL, name)

    def test_matrix_free_coboundary_and_pooling_match_dense(self) -> None:
        operator_src = _random_operator(9, 2, 4, 2, seed=1)
        operator_tgt = _random_operator(9, 2, 4, 2, seed=2, group_ids=operator_src.group_ids)
        maps = EdgeMapPair(operator_src, operator_tgt)
        F_src, F_tgt = operator_src.materialize(), operator_tgt.materialize()
        edge_index = torch.randint(6, (2, 9), generator=torch.Generator().manual_seed(4))
        x = torch.randn(6, 4, dtype=torch.float64)
        s = torch.randn(9, 4, dtype=torch.float64)
        src_idx, tgt_idx = edge_index
        self.assertLess(_max_err(CoboundaryOps.delta0_matrix_free(x, maps, src_idx, tgt_idx),
                                 CoboundaryOps.delta0(x, F_src, F_tgt, src_idx, tgt_idx)), F64_TOL)
        self.assertLess(_max_err(CoboundaryOps.delta0_T_matrix_free(s, maps, src_idx, tgt_idx, 6),
                                 CoboundaryOps.delta0_T(s, F_src, F_tgt, src_idx, tgt_idx, 6)), F64_TOL)
        self.assertLess(_max_err(CoboundaryOps.L0_matrix_free(x, maps, src_idx, tgt_idx, 6),
                                 CoboundaryOps.L0(x, F_src, F_tgt, src_idx, tgt_idx, 6)), F64_TOL)
        lhs = float((CoboundaryOps.delta0_matrix_free(x, maps, src_idx, tgt_idx) * s).sum())
        rhs = float((x * CoboundaryOps.delta0_T_matrix_free(s, maps, src_idx, tgt_idx, 6)).sum())
        self.assertLess(abs(lhs - rhs), F64_TOL)
        self.assertLess(_max_err(pool_edge_stalks_to_nodes_matrix_free(s, edge_index, maps, 6),
                                 pool_edge_stalks_to_nodes(s, edge_index, F_src, F_tgt, 6)), F64_TOL)

    def test_matrix_free_gradients_match_dense_reference(self) -> None:
        """Backward, not just forward: grads w.r.t. every factor and the input agree with
        the v1 dense operators applied to the materialized maps (float64)."""
        generator = torch.Generator().manual_seed(11)
        E, k, d, N = 12, 2, 4, 7
        group_ids = torch.randint(3, (E,), generator=generator)
        edge_index = torch.randint(N, (2, E), generator=generator)
        leaves = {
            "u_src": torch.randn(E, k, d, generator=generator, dtype=torch.float64),
            "u_tgt": torch.randn(E, k, d, generator=generator, dtype=torch.float64),
            "c_src": torch.randn(E, generator=generator, dtype=torch.float64),
            "c_tgt": torch.randn(E, generator=generator, dtype=torch.float64),
            "rot": torch.linalg.qr(torch.randn(3, d, d, generator=generator,
                                               dtype=torch.float64))[0],
            "x": torch.randn(N, d, generator=generator, dtype=torch.float64),
            "s": torch.randn(E, d, generator=generator, dtype=torch.float64),
        }
        weights = torch.randn(3, E, d, generator=generator, dtype=torch.float64)

        def grads(matrix_free: bool) -> dict[str, torch.Tensor]:
            leaf = {name: value.clone().requires_grad_(True) for name, value in leaves.items()}
            src = FactorizedMapOperator.from_raw(leaf["u_src"], leaf["c_src"], leaf["rot"],
                                                 group_ids, 0.1, 1e-6)
            tgt = FactorizedMapOperator.from_raw(leaf["u_tgt"], leaf["c_tgt"], leaf["rot"],
                                                 group_ids, 0.1, 1e-6)
            maps = EdgeMapPair(src, tgt)
            F_src, F_tgt = src.materialize(), tgt.materialize()
            src_idx, tgt_idx = edge_index
            if matrix_free:
                l0 = CoboundaryOps.L0_matrix_free(leaf["x"], maps, src_idx, tgt_idx, N)
                l1 = CoboundaryOps.L1_matrix_free(leaf["s"], maps, src_idx, tgt_idx, N,
                                                  (), lambda c: c.maps, E)
                pooled = pool_edge_stalks_to_nodes_matrix_free(leaf["s"], edge_index, maps, N)
                d0 = CoboundaryOps.delta0_matrix_free(leaf["x"], maps, src_idx, tgt_idx)
            else:
                l0 = CoboundaryOps.L0(leaf["x"], F_src, F_tgt, src_idx, tgt_idx, N)
                l1 = CoboundaryOps.L1(leaf["s"], F_src, F_tgt, src_idx, tgt_idx, N,
                                      (), lambda c: c.maps, E)
                pooled = pool_edge_stalks_to_nodes(leaf["s"], edge_index, F_src, F_tgt, N)
                d0 = CoboundaryOps.delta0(leaf["x"], F_src, F_tgt, src_idx, tgt_idx)
            loss = ((l0.square().sum() + pooled.square().sum())
                    + (weights[0] * l1).sum() + (weights[1] * d0).sum())
            loss.backward()
            return {name: value.grad for name, value in leaf.items()}

        got, want = grads(True), grads(False)
        for name in leaves:
            self.assertIsNotNone(got[name], name)
            self.assertLess(_max_err(got[name], want[name]), F64_TOL, name)

    def _record(self, model, data) -> _ShapeRecorder:
        recorder = _ShapeRecorder()
        with recorder:
            output = model(data)
            loss = output.logits.square().sum() + output.s_star.square().sum()
            recorder.phase = "backward"
            loss.backward()
        return recorder

    def test_no_dense_map_tensor_over_observed_edges_in_forward_or_backward(self) -> None:
        """Every tensor any op produces in forward or backward (incl. checkpoint replay).

        Two rules: (i) shape `[..., d, d]` whose leading dims multiply to the observed or
        extended edge count; (ii) any tensor holding >= M*d*d/4 elements, whatever its
        shape (catches `[2M, d, d]`, `[M, d*d]`, `[M, k, d, d]`). The fixture keeps the v1
        face tensors (which D1 may still materialize per chunk) far below rule (ii)."""
        # M = 1200 > d*d, so no parameter (e.g. the v1 d*d-output A_f head) reaches rule (ii).
        data = _random_pap_data(num_papers=200, num_authors=100, max_faces=10)
        d = 32
        M = data.num_edges
        extended = M + sum(group.num_faces for group in data.face_groups)
        forbidden = {M, extended}
        numel_limit = M * d * d // 4
        face_sizes = set()
        for group in data.face_groups:
            face_sizes |= {group.num_faces, group.num_faces * group.length,
                           group.num_faces * (group.length + 1)}
        face_sizes.add(sum(group.num_faces for group in data.face_groups))
        self.assertFalse(face_sizes & forbidden, "fixture collision")
        self.assertLess(max(face_sizes) * d * d, numel_limit, "fixture: face tensors too big")

        def dense_hits(recorder: _ShapeRecorder, phase: str) -> list[tuple[int, ...]]:
            return [shape for p, shape in recorder.shapes if p == phase and (
                (len(shape) >= 3 and shape[-2:] == (d, d) and math.prod(shape[:-2]) in forbidden)
                or math.prod(shape) >= numel_limit)]

        torch.manual_seed(0)
        model = build_model(_config(map_family="factorized", d=32), data)
        largest_parameter = max(p.numel() for p in model.parameters())
        self.assertLess(largest_parameter, numel_limit, "fixture: a parameter trips rule (ii)")
        factorized = self._record(model, data)
        self.assertEqual(dense_hits(factorized, "forward"), [])
        self.assertEqual(dense_hits(factorized, "backward"), [])
        self.assertTrue(any(p == "backward" for p, _ in factorized.shapes))
        # Negative control: the same detector flags v1 in both passes.
        torch.manual_seed(0)
        dense = self._record(build_model(_config(d=32), data), data)
        self.assertTrue(dense_hits(dense, "forward"))
        self.assertTrue(dense_hits(dense, "backward"))

    def test_factorized_runs_with_v1_switches(self) -> None:
        """The v1 model switches that stay reachable under D1 still run, finitely."""
        from tests.test_models import _make_prepared_data

        discovery = {"enabled": True, "mode": "auto", "max_total_instances": 10,
                     "max_total_incidences": 30}
        cases = [
            ("no_faces", {"ablate_no_faces": True}, None, _random_pap_data),
            ("no_delta1", {"ablate_no_delta1": True}, None, _random_pap_data),
            ("lstm_triangular", {"face_encoder": "lstm", "face_map_param": "triangular"},
             None, _random_pap_data),
            ("auto_weights", {"d": 4, "K": 1}, discovery,
             lambda: _make_prepared_data("acm", with_discovery=True)),
        ]
        for label, model_overrides, discovery_config, make in cases:
            with self.subTest(case=label):
                torch.manual_seed(0)
                data = make()
                raw = {"device": "cpu", "checkpoint_metric": "acc", "model": {
                    "d": 8, "d_r": 4, "d_h": 8, "K": 2, "classifier_hidden_dim": 16,
                    "map_family": "factorized", **model_overrides}}
                if discovery_config is not None:
                    raw["discovery"] = discovery_config
                model = build_model(TrainConfig.from_dict(raw), data)
                output = model(data)
                output.logits.square().sum().backward()
                self.assertTrue(bool(torch.isfinite(output.logits).all()))
                self.assertIsNotNone(model.node_to_edge.rotation_params.grad)

    def test_gradients_reach_rotations_reflector_and_scale_mlps(self) -> None:
        data = _random_pap_data()
        torch.manual_seed(0)
        model = build_model(_config(map_family="factorized"), data)
        output = model(data)
        output.logits.square().sum().backward()
        groups = {"rotation_params": [], "reflector_mlp": [], "scale_mlp": []}
        for name, parameter in model.node_to_edge.named_parameters():
            for prefix in groups:
                if name.startswith(prefix):
                    groups[prefix].append(parameter.grad)
        for prefix, grads in groups.items():
            self.assertTrue(grads, prefix)
            self.assertTrue(all(g is not None for g in grads), prefix)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0, prefix)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_float32_matches_cpu_float64_reference(self) -> None:
        data = _random_pap_data()
        module, h, relation_embeddings, output = _module_maps(data, dtype=torch.float64)
        cuda_module = FactorizedNodeToEdgeSheaf(
            d=8, d_r=4, num_relations=2, hidden_dim=16, householder_k=2,
            scale_eps=0.1, orthogonal_map="fasthpp",
        ).cuda()
        cuda_module.load_state_dict({k: v.float() for k, v in module.state_dict().items()})
        cuda_maps = cuda_module(h.float().cuda(), data.edge_index.cuda(), data.edge_type.cuda(),
                                relation_embeddings.float().cuda()).maps
        x = torch.randn(data.num_edges, 8, dtype=torch.float64)
        for got, want in ((cuda_maps.src, output.maps.src), (cuda_maps.tgt, output.maps.tgt)):
            self.assertEqual(got.scale.device.type, "cuda")
            self.assertEqual(got.scale.dtype, torch.float32)
            result = got.apply(x.float().cuda()).double().cpu()
            reference = want.apply(x)
            self.assertLess(_max_err(result, reference) / float(reference.abs().max()), F32_TOL)
        model = build_model(_config(map_family="factorized"), data).cuda()
        logits = model(data.to("cuda")).logits
        self.assertTrue(bool(torch.isfinite(logits).all()))


# ----------------------------------------------------------------------------- config, scope, T11
class ConfigAndScopeTests(unittest.TestCase):
    def test_defaults_keep_v1(self) -> None:
        config = ModelConfig()
        self.assertEqual(config.map_family, "dense")
        self.assertEqual(config.householder_k, 2)

    def test_invalid_values_are_rejected(self) -> None:
        cases = [
            {"map_family": "lowrank"},
            {"householder_k": 0},
            {"map_family": "factorized", "d": 4, "householder_k": 5},
            {"map_scale_eps": 0.0},
            {"map_family": "factorized", "use_orthogonal": False},
            {"map_family": "factorized", "ablate_identity_maps": True},
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    ModelConfig.from_dict(raw)
        with self.assertRaises(ValueError):
            ModelConfig(householder_k=True)  # type: ignore[arg-type]
        for not_an_int in (2.9, True, "2"):
            with self.subTest(householder_k=not_an_int):
                with self.assertRaises(ValueError):
                    ModelConfig.from_dict({"householder_k": not_an_int})
        # Dense configs never trip the factorized-only bound (v1 unchanged for any d).
        self.assertEqual(ModelConfig.from_dict({"d": 1}).householder_k, 2)

    def test_out_of_scope_tasks_raise_not_implemented(self) -> None:
        dims = ModelDims(in_channels=6, num_node_types=2, num_relations=2, num_classes=3)
        from src.models.h2snn_model import H2SNN

        for task in ("graph_classification", "link_prediction"):
            with self.subTest(task=task):
                config = _config(map_family="factorized")
                config.data.task_type = task
                with self.assertRaisesRegex(NotImplementedError, "D1"):
                    H2SNN(config, dims)

    def test_injected_maps_raise_not_implemented_under_factorized(self) -> None:
        data = _random_pap_data()
        model = build_model(_config(map_family="factorized"), data)
        eye = torch.eye(8).expand(data.num_edges, -1, -1)
        with self.assertRaisesRegex(NotImplementedError, "D1"):
            model.encode_from_stalk(
                h0=torch.randn(data.num_nodes, 8), edge_index=data.edge_index,
                edge_type=data.edge_type, face_groups=data.face_groups,
                closure_edge_index=data.closure_edge_index, num_nodes=data.num_nodes,
                num_edges=data.num_edges, injected_maps=(eye, eye),
            )

    # Outputs of the PRE-change code (commit 0172044), recorded on CPU in float32 with the
    # seeds used below: eval logits, train-mode logits sum, sum of squared gradients.
    T11_GOLDEN = {
        "acm": (False, (0.08945903182029724, -0.05324907600879669,
                        0.18663734197616577, -0.11267569661140442),
                -0.006220608949661255, 18.25995569351168),
        "dblp": (False, (-0.25717800855636597, 0.012932587414979935,
                         -0.20732030272483826, -0.04156850278377533),
                 -0.4914790764451027, 7.6137750301077585),
        "imdb": (False, (0.08945903182029724, -0.05324907600879669,
                         0.18663734197616577, -0.11267569661140442),
                 -0.006220608949661255, 18.25995569351168),
        "acm_auto": (True, (0.1350582093000412, -0.10817259550094604,
                            0.18875104188919067, -0.1167091429233551),
                     0.001686006784439087, 21.306526011802973),
    }

    def test_T11_dense_matches_pre_change_golden_outputs(self) -> None:
        """v1 regression against values recorded from the pre-change code (card: <= 1e-6)."""
        from tests.test_models import _make_prepared_data

        small = {"d": 4, "d_r": 2, "d_h": 8, "K": 2, "classifier_hidden_dim": 8}
        discovery = {"enabled": True, "mode": "auto", "max_total_instances": 10,
                     "max_total_incidences": 30}
        for case, (auto, logits, train_sum, grad_sq) in self.T11_GOLDEN.items():
            with self.subTest(case=case):
                raw = {"device": "cpu", "checkpoint_metric": "acc",
                       "model": dict(small, K=1) if auto else small}
                if auto:
                    raw["discovery"] = discovery
                torch.manual_seed(1234)
                data = _make_prepared_data(case.removesuffix("_auto"), with_discovery=auto)
                config = TrainConfig.from_dict(raw)
                torch.manual_seed(42)
                model = build_model(config, data)
                model.eval()
                with torch.no_grad():
                    eval_logits = model(data).logits
                model.train()
                torch.manual_seed(7)
                output = model(data)
                (output.logits.square().sum() + output.s_star.square().mean()
                 + output.x_star.square().mean()).backward()
                got_grad_sq = sum(float(p.grad.double().square().sum())
                                  for p in model.parameters() if p.grad is not None)
                self.assertLess(_max_err(eval_logits.double().flatten(),
                                         torch.tensor(logits, dtype=torch.float64)), 1e-6)
                self.assertLess(abs(float(output.logits.double().sum()) - train_sum), 1e-6)
                self.assertLess(abs(got_grad_sq - grad_sq) / grad_sq, 1e-6)

    def test_T11_explicit_dense_is_identical_to_the_default(self) -> None:
        """Bit-for-bit equality with the pre-change code is checked by a one-off script
        (see the D1 report); here: `map_family='dense'` builds exactly the v1 model."""
        data = _random_pap_data()
        outputs = []
        for model_config in ({}, {"map_family": "dense", "householder_k": 3}):
            torch.manual_seed(0)
            model = build_model(_config(**model_config), data).eval()
            self.assertIsInstance(model.node_to_edge, NodeToEdgeSheaf)
            with torch.no_grad():
                outputs.append((list(model.state_dict()), model(data)))
        (keys_a, out_a), (keys_b, out_b) = outputs
        self.assertEqual(keys_a, keys_b)
        self.assertTrue(torch.equal(out_a.logits, out_b.logits))
        self.assertTrue(torch.equal(out_a.s_star, out_b.s_star))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
