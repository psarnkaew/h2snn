"""Task card D1: factorized, matrix-free, conformal-orthogonal restriction maps.

    F_{v<|e} = c_{v<|e} * R_{psi(e),side} * H_k(u_{v<|e})

Test ids follow the D1 task card (T1-T5, T8-T10; T6, T7 and T11 tested the deleted v1
face code and v1 regression). Every property test that could pass vacuously has a negative
control showing that the same check fails when the property is broken. Per-edge (D1a)
versions and the model-level tests are in tests/test_v2_model.py.
"""

from __future__ import annotations

import math
import unittest

try:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.configs.config import ModelConfig, TrainConfig
from src.data.structures import FaceGroup, PreparedH2SData
from src.models.h2snn_model import build_model
from src.models.map_operators import (
    EdgeMapPair,
    FactorizedMapOperator,
    PathTransportOperator,
    normalize_reflectors,
)
from src.models.sheaf_maps import FactorizedNodeToEdgeSheaf

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
        types = tuple("paper" if i % 2 == 0 else "author" for i in range(length + 1))
        groups.append(
            FaceGroup(
                template_name=name,
                node_type_sequence=types,
                face_node_index=face_node_index,
                face_edge_index=face_edge_index,
                relation_index=edge_type[face_edge_index],
                face_edge_sign=torch.ones(face_edge_index.shape),
            )
        )

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
    base = {"d": 8, "d_r": 4, "K": 2, "classifier_hidden_dim": 16}
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
    """Path holonomy `prod_j inv(F^{e_j}_src) F^{e_j}_tgt` per instance, from materialized
    maps and a general inverse (independent of the matrix-free operators)."""
    F_src = maps.src.materialize()[face_edge_index]
    F_tgt = maps.tgt.materialize()[face_edge_index]
    d = F_src.size(-1)
    holonomy = torch.eye(d, dtype=F_src.dtype).expand(face_edge_index.size(0), d, d)
    for j in range(face_edge_index.size(1)):
        holonomy = holonomy @ torch.linalg.inv(F_src[:, j]) @ F_tgt[:, j]
    return holonomy


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
        (output.logits.square().sum() + output.x_star.square().sum()).backward()
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


# ----------------------------------------------------------------------------- config
class ConfigAndScopeTests(unittest.TestCase):
    def test_invalid_values_are_rejected(self) -> None:
        cases = [
            {"map_family": "lowrank"},
            {"householder_k": 0},
            {"map_family": "factorized", "d": 4, "householder_k": 5},
            {"map_scale_eps": 0.0},
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
        # Dense maps never trip the factorized-only bound k <= d.
        self.assertEqual(ModelConfig.from_dict({"d": 1, "map_family": "dense"}).householder_k, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
