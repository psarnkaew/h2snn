"""Model change TD-1 (`docs/H2S-NN_model_change_td1_card.md` M3; spec 1.25 Z1-Z4): the TD-1 maps,
the removed keys, the type-only read-out, the head inputs and the bounded cell weights.

float64 unless stated; synthetic graphs only (the real data root is unreachable). Test ids follow the
card (T9 is folded into T1, T13 is dropped by the card's scope cut).

T1  every transport from random parameters is in SO(d) (||Q^T Q - I||_inf <= 1e-12, det >= 1 - 1e-10;
    fact transports and path transports with reversed hops); stored_elements = E (m^2 + 1) +
    R (d^2 + d m) exactly
T2  fresh model: Q_e = R_r (rho(r) != r) and Q_e = I (rho(r) = r), <= 1e-14
T3  closed form I + U (exp C - I) U^T (production builder) vs torch.linalg.matrix_exp(U C U^T) on the
    dense d x d matrix, random C, <= 1e-12                                     [mutation 2 target]
T4  flipping the stored orientation of rho(r) = r facts leaves L_hat Y unchanged, <= 1e-12
                                                                                [mutation 1 target]
T5  adjoint <F x, y> = <x, F^T y> and inverse F^-1 F x = x, both sides, both relation kinds, <= 1e-12
T6  Thm 4 on the production operator (circle construction of docs/thm4_td1_proof.md injected into
    the production `td1_edge_maps`): energy at the section and ||L Y|| <= 1e-10; anchored solve
    (tol 1e-12) x* = h0 <= 1e-8; argmax accuracy 1.0; negative control (one fact rotated by 0.3 rad):
    energy > 1e-3
T7  torch.autograd.gradcheck: edge MLP -> m x m exp -> operator apply; W_r -> QR -> U_r
T8  removed keys raise with the spec-1.25 message; nothing in src/ imports relational_context
T10 evaluation.type_only_maps: every distinct-role Q_e = R_r (<= 1e-14); refused in a training
    forward and in a tuning config
T11 one training step on an ACM-like fixture: loss finite; frames and both MLP layers get nonzero
    gradients; W_r gets an exactly zero gradient at C_e = 0 and a nonzero one at the second step
T12 free_cell_maps, fact_maps_identity, direction_blind_maps: one step finite; direction-blind
    transports have no frame (Q = exp(A)), <= 1e-12
T14 head input widths (x*: d; ego_head: [x*, h0] = the 1.24 head input; instance_features:
    [x*, inst]); ego_head refused in a main-table tuning base config
T15 bounded weights: |g| up to 1e3 gives a log-weight range <= 2L and finite gradients; at
    initialisation w_f = exp(L tanh(g/L)) on every rung (uniform: exp(L tanh(b/L))), NC, LP and GC
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import re
import unittest
from pathlib import Path

import torch

from src.models.map_operators import (
    EdgeMapPair,
    PathTransportOperator,
    TD1MapOperator,
    orthonormal_subspaces,
    skew_from_coefficients,
)
from src.models.sheaf_energy import (
    EliminatedOperator,
    SheafEnergyOperator,
    build_cell_group,
)
from src.models.sheaf_maps import TD1Factors, TD1NodeToEdgeSheaf, td1_edge_maps
from tests.partMB_fixtures import restore_real_data, unreachable_real_data

F64 = torch.float64
ORTHO_TOL = 1e-12
DET_TOL = 1e-10
INIT_TOL = 1e-14
CLOSED_FORM_TOL = 1e-12
FLIP_TOL = 1e-12
ADJOINT_TOL = 1e-12
THM4_TOL = 1e-10
THM4_X_TOL = 1e-8
NOFRAME_TOL = 1e-12
WEIGHT_TOL = 1e-12
SRC = Path(__file__).resolve().parents[1] / "src"
SMALL = {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
         "classifier_hidden_dim": 16, "propagation": "anchored"}


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


# ----------------------------------------------------------------------------- fixtures
def _graph(seed: int = 0, num_nodes: int = 9, num_relations: int = 3, num_facts: int = 24):
    """Random facts of `num_relations` relations (one self-loop), node states and embeddings."""
    if num_facts > num_nodes * (num_nodes - 1):
        raise ValueError("more facts than ordered node pairs")
    g = torch.Generator().manual_seed(seed)
    pairs = []
    while len(pairs) < num_facts:
        u, v = torch.randint(num_nodes, (2,), generator=g).tolist()
        if u != v and (u, v) not in pairs:
            pairs.append((u, v))
    pairs.append((2, 2))
    edge_index = torch.tensor(pairs).t()
    edge_type = torch.arange(edge_index.size(1)) % num_relations
    h = torch.randn(num_nodes, 8, generator=g, dtype=F64)
    rel = torch.randn(num_relations, 4, generator=g, dtype=F64)
    return edge_index, edge_type, h, rel


def _builder(shared: tuple[int, ...], num_relations: int = 3, d: int = 8, rank: int = 4,
             angle: float = 0.5, seed: int = 0, checkpoint: bool = True) -> TD1NodeToEdgeSheaf:
    """The production builder in float64; `angle` > 0 draws the skew rows (zero at init)."""
    torch.manual_seed(seed)
    module = TD1NodeToEdgeSheaf(d=d, d_r=4, num_relations=num_relations, hidden_dim=16,
                                map_rank=rank, scale_eps=0.1, orthogonal_map="fasthpp",
                                checkpoint=checkpoint, shared_role_relations=shared).double()
    if angle:
        with torch.no_grad():
            module.edge_mlp[-1].weight[:module.num_skew].normal_(0.0, angle)
            module.edge_mlp[-1].bias[:module.num_skew].normal_(0.0, angle)
    return module


def _transports(maps: EdgeMapPair) -> torch.Tensor:
    """`Q_e = F_src^T F_tgt / c_e^2`, `[E, d, d]`."""
    c2 = maps.src.scale.square().view(-1, 1, 1)
    return maps.src.materialize().transpose(-2, -1) @ maps.tgt.materialize() / c2


def _ortho_error(q: torch.Tensor) -> float:
    eye = torch.eye(q.size(-1), dtype=q.dtype)
    return float((q.transpose(-2, -1) @ q - eye).abs().max())


def _mlp_skew(module: TD1NodeToEdgeSheaf, h, rel, edge_index, edge_type) -> torch.Tensor:
    """`C_e` computed independently of the builder: raw MLP outputs, antisymmetrised by hand on the
    shared-role rows (C_e(u, v) = (k(u, v) - k(v, u)) / 2)."""
    with torch.no_grad():
        u, v = edge_index
        forward = module.edge_mlp(torch.cat([h[u], h[v], rel[edge_type]], -1))[:, :module.num_skew]
        backward = module.edge_mlp(torch.cat([h[v], h[u], rel[edge_type]], -1))[:, :module.num_skew]
    shared = torch.tensor([int(t) in module.shared_role_relations for t in edge_type.tolist()])
    coefficients = torch.where(shared.view(-1, 1), 0.5 * (forward - backward), forward)
    return skew_from_coefficients(coefficients, module.map_rank)


def _config(model: dict | None = None, ablation: dict | None = None, data: dict | None = None,
            dataset: str = "acm", evaluation: dict | None = None):
    from src.configs.config import TrainConfig

    return TrainConfig.from_dict({
        "device": "cpu", "seed": 3,
        "data": {"dataset_name": dataset, "wisp_max_length": 3, "wisp_walks_per_anchor": 5,
                 **(data or {})},
        "model": {**SMALL, **(model or {})},
        "ablation": ablation or {},
        "evaluation": evaluation or {}})


def _acm(cfg):
    from src.data.dataset import build_dataset
    from tests.partO_fixtures import acm_like_hetero, patched_acm_loader

    with patched_acm_loader(acm_like_hetero(seed=9)):
        return build_dataset(cfg)


def _model(cfg, data, seed: int = 0):
    from src.models.h2snn_model import build_model

    torch.manual_seed(seed)
    return build_model(cfg, data)


def _randomize_skew(model, scale: float = 0.3, seed: int = 4) -> None:
    g = torch.Generator().manual_seed(seed)
    last = model.node_to_edge.edge_mlp[-1]
    k = model.node_to_edge.num_skew
    with torch.no_grad():
        last.weight[:k] = scale * torch.randn(last.weight[:k].shape, generator=g)
        last.bias[:k] = scale * torch.randn(k, generator=g)


# ----------------------------------------------------------------------------- T1, T2
class TransportTests(unittest.TestCase):
    def test_T1_every_transport_is_in_SO_d_and_storage_is_exact(self) -> None:
        edge_index, edge_type, h, rel = _graph()
        worst_ortho, worst_det = 0.0, 0.0
        for shared in ((), (2,), (0, 1, 2)):
            with self.subTest(shared=shared):
                module = _builder(shared, angle=1.0)
                maps = module(h, edge_index, edge_type, rel).maps
                q = _transports(maps)
                worst_ortho = max(worst_ortho, _ortho_error(q))
                worst_det = max(worst_det, float((1.0 - torch.linalg.det(q)).abs().max()))
                # path transports (T9 folded in): chains with reversed hops through the production
                # PathTransportOperator, read off column by column
                nodes = torch.tensor([[0, 1, 2], [3, 4, 5]])
                facts = torch.tensor([[0, 1], [2, 3]])
                chain = torch.tensor([[0, 1], [2, 1], [3, 4], [4, 5]]).t()  # 2nd, 3rd... mixed
                signs = torch.tensor([[1.0, -1.0], [1.0, 1.0]])
                chain_type = torch.tensor([0, 1, 2, 0])
                chain_maps = module(h, chain, chain_type, rel).maps
                transport = PathTransportOperator(chain_maps.src, chain_maps.tgt, nodes, facts,
                                                  chain, shared_scale=True, face_edge_sign=signs)
                basis = torch.eye(module.d, dtype=F64)
                t_mat = torch.stack([transport.apply(basis[k].expand(h.size(0), -1))
                                     for k in range(module.d)], dim=-1)          # [P, L+1, d, d]
                worst_ortho = max(worst_ortho, _ortho_error(t_mat))
                worst_det = max(worst_det, float((1.0 - torch.linalg.det(t_mat)).abs().max()))
        self.assertLessEqual(worst_ortho, ORTHO_TOL)
        self.assertLessEqual(worst_det, DET_TOL)
        # storage: every relation with rho(r) != r (the formula of the card)
        for d, rank in ((8, 4), (16, 4), (8, 2)):
            with self.subTest(d=d, rank=rank):
                module = _builder((), d=d, rank=rank)
                hd = torch.randn(h.size(0), d, dtype=F64)
                maps = module(hd, edge_index, edge_type, rel).maps
                E, R = edge_index.size(1), 3
                self.assertEqual(maps.tgt.stored_elements(), E * (rank ** 2 + 1) + R * (d * d + d * rank))
                self.assertEqual(maps.src.stored_elements(), E)  # c_e I on the source role
        self.worst = (worst_ortho, worst_det)

    def test_T2_fresh_model_transports_are_the_frames(self) -> None:
        edge_index, edge_type, h, rel = _graph()
        module = _builder((2,), angle=0.0)
        maps = module(h, edge_index, edge_type, rel).maps
        q = _transports(maps)
        frames = module.frames()
        eye = torch.eye(module.d, dtype=F64)
        worst = 0.0
        for e in range(edge_index.size(1)):
            r = int(edge_type[e])
            want = eye if r == 2 else frames[module.framed_relations.index(r)]
            worst = max(worst, float((q[e] - want).abs().max()))
        self.assertLessEqual(worst, INIT_TOL)
        self.worst = worst


# ----------------------------------------------------------------------------- T3
class ClosedFormTests(unittest.TestCase):
    def test_T3_closed_form_equals_the_dense_matrix_exponential(self) -> None:
        edge_index, edge_type, h, rel = _graph(seed=2)
        worst = 0.0
        for shared in ((), (1,)):
            module = _builder(shared, angle=1.0)
            maps = module(h, edge_index, edge_type, rel).maps
            u = module.subspaces().detach()
            skew = _mlp_skew(module, h, rel, edge_index, edge_type)
            q = _transports(maps)
            frames = module.frames().detach()
            for e in range(edge_index.size(1)):
                r = int(edge_type[e])
                dense = torch.linalg.matrix_exp(u[r] @ skew[e] @ u[r].T)          # d x d
                if r in module.shared_role_relations:
                    want = dense                     # exp(A/2) exp(A/2)
                else:
                    want = frames[module.framed_relations.index(r)] @ dense
                worst = max(worst, float((q[e] - want).abs().max()))
            # the matrix-free apply equals the materialized closed form
            x = torch.randn(edge_index.size(1), module.d, dtype=F64)
            for side in (maps.src, maps.tgt):
                dense_apply = (side.materialize() @ x.unsqueeze(-1)).squeeze(-1)
                worst = max(worst, float((side.apply(x) - dense_apply).abs().max()))
        self.assertLessEqual(worst, CLOSED_FORM_TOL)
        self.worst = worst


# ----------------------------------------------------------------------------- T4
class OrientationTests(unittest.TestCase):
    def test_T4_flipping_shared_role_facts_leaves_the_operator_unchanged(self) -> None:
        from tests.test_sheaf_energy import Fixture

        fx = Fixture(d=8)  # relation 1 is the shared-role relation below; cells with signed hops
        g = torch.Generator().manual_seed(17)
        h = torch.randn(fx.num_nodes, 8, generator=g, dtype=F64)
        rel = torch.randn(2, 4, generator=g, dtype=F64)
        module = _builder((1,), num_relations=2, angle=1.0)
        flip = (fx.edge_type == 1).nonzero(as_tuple=True)[0]
        self.assertGreater(flip.numel(), 2)
        flipped_index = fx.edge_index.clone()
        flipped_index[:, flip] = fx.edge_index.flip(0)[:, flip]

        def operator(edge_index, flipped: bool) -> SheafEnergyOperator:
            maps = module(h, edge_index, fx.edge_type, rel).maps
            cells = []
            for group, beta in zip(fx.groups, fx.beta_by_group):
                if flipped:  # a hop on a flipped fact is now traversed against its storage
                    on = torch.isin(group.face_edge_index, flip)
                    group = dataclasses.replace(group, face_edge_sign=torch.where(
                        on, -group.face_edge_sign, group.face_edge_sign))
                cells.append(build_cell_group(maps, group, edge_index, beta))
            return SheafEnergyOperator(maps, edge_index, fx.num_nodes, cells)

        before, after = operator(fx.edge_index, False), operator(flipped_index, True)
        y = torch.randn(before.layout.rows, 8, generator=g, dtype=F64)
        with torch.no_grad():
            change = float((before.apply_normalized(y) - after.apply_normalized(y)).abs().max())
        self.assertLessEqual(change, FLIP_TOL)
        # control: the same flip on a relation with distinct roles changes the operator
        directed = _builder((), num_relations=2, angle=1.0)
        maps_a = directed(h, fx.edge_index, fx.edge_type, rel).maps
        maps_b = directed(h, flipped_index, fx.edge_type, rel).maps
        a = SheafEnergyOperator(maps_a, fx.edge_index, fx.num_nodes, [])
        b = SheafEnergyOperator(maps_b, flipped_index, fx.num_nodes, [])
        y0 = torch.randn(a.layout.rows, 8, generator=g, dtype=F64)
        with torch.no_grad():
            self.assertGreater(float((a.apply_normalized(y0) - b.apply_normalized(y0)).abs().max()),
                               1e-3)
        self.change = change


# ----------------------------------------------------------------------------- T5
class AdjointInverseTests(unittest.TestCase):
    def test_T5_adjoint_and_inverse_on_both_sides_and_relation_kinds(self) -> None:
        edge_index, edge_type, h, rel = _graph(seed=5)
        module = _builder((2,), angle=1.0)
        maps = module(h, edge_index, edge_type, rel).maps
        g = torch.Generator().manual_seed(6)
        worst = 0.0
        for side in (maps.src, maps.tgt):
            x = torch.randn(edge_index.size(1), module.d, generator=g, dtype=F64)
            y = torch.randn(edge_index.size(1), module.d, generator=g, dtype=F64)
            for kind in (edge_type == 2, edge_type != 2):
                lhs = (side.apply(x) * y)[kind].sum()
                rhs = (x * side.apply_T(y))[kind].sum()
                worst = max(worst, float((lhs - rhs).abs()) / max(1.0, float(lhs.abs())))
                worst = max(worst, float((side.apply_inverse(side.apply(x)) - x)[kind].abs().max()))
                worst = max(worst, float((side.apply_inverse_T(side.apply_T(x)) - x)[kind].abs().max()))
                worst = max(worst, float((side.apply_orthogonal_T(side.apply_orthogonal(x))
                                          - x)[kind].abs().max()))
        self.assertLessEqual(worst, ADJOINT_TOL)
        self.worst = worst


# ----------------------------------------------------------------------------- T6
def thm4_production(fixture, num_classes: int = 5, rank: int = 4, seed: int = 61,
                    perturb: float = 0.0):
    """The circle construction of `docs/thm4_td1_proof.md` injected into the production operator
    (`td1_edge_maps` -> `SheafEnergyOperator`): plane P = span(p1, p2) = U_r e1, U_r e2 for every
    relation, R_r = I, C_e = (theta_{y(u)} - theta_{y(v)}) J; relation 1 has rho(r) = r (half angles).
    `perturb` rotates one heterophilic fact of relation 0 by that many radians more."""
    g = torch.Generator().manual_seed(seed)
    n, d, E = fixture.num_nodes, fixture.d, fixture.edge_index.size(1)
    labels = torch.randint(num_classes, (n,), generator=g)
    theta = 2.0 * math.pi * torch.arange(num_classes, dtype=F64) / num_classes
    basis, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=F64))
    p1, p2 = basis[:, 0], basis[:, 1]
    classes = torch.cos(theta).unsqueeze(-1) * p1 + torch.sin(theta).unsqueeze(-1) * p2
    J = torch.zeros(rank, rank, dtype=F64)
    J[1, 0], J[0, 1] = 1.0, -1.0                     # J e1 = e2, J e2 = -e1
    u, v = fixture.edge_index
    angle = theta[labels[u]] - theta[labels[v]]
    heterophilic = ((labels[u] != labels[v]) & (fixture.edge_type == 0)).nonzero(as_tuple=True)[0]
    if perturb:
        angle = angle.clone()
        angle[heterophilic[0]] += perturb
    half = torch.where(fixture.edge_type == 1, 0.5, 1.0).to(F64)
    rotation = torch.linalg.matrix_exp((half * angle).view(-1, 1, 1) * J) - torch.eye(rank, dtype=F64)
    subspace = torch.stack([basis[:, :rank], basis[:, :rank]])       # U_r e1 = p1, U_r e2 = p2
    factors = TD1Factors(scale=fixture.maps.src.scale, rotation=rotation, subspaces=subspace,
                         frames=torch.eye(d, dtype=F64).unsqueeze(0))
    maps = td1_edge_maps(factors, fixture.edge_type, 2, (1,))
    cells = [build_cell_group(maps, group, fixture.edge_index, beta)
             for group, beta in zip(fixture.groups, fixture.beta_by_group)]
    op = SheafEnergyOperator(maps, fixture.edge_index, n, cells)
    return op, labels, classes, heterophilic.numel()


class Thm4ProductionTests(unittest.TestCase):
    def test_T6_class_separating_section_on_the_production_operator(self) -> None:
        from src.models.anchored_solve import AnchoredSystem, solve_anchored
        from tests.test_sheaf_energy import Fixture

        fixture = Fixture(d=8)
        op, labels, classes, heterophilic = thm4_production(fixture)
        self.assertGreater(heterophilic, 2)
        x = classes[labels]                                               # x_v = a_{y(v)}
        y_sec = op.initial_guess(x)                                       # (x, F x, T x)
        with torch.no_grad():
            ly = op.apply(y_sec)
            energy = 0.5 * float((y_sec * ly).sum())
            residual = float(ly.norm())
        self.assertLessEqual(abs(energy), THM4_TOL)
        self.assertLessEqual(residual, THM4_TOL)
        # anchored solve with the normalised operator: h0 = D_v^{1/2} a_{y(v)} is a fixed point
        h0 = op.degree()[0].sqrt().unsqueeze(-1) * x
        numbers = {"energy": energy, "residual": residual}
        for name, operator in (("full", op), ("eliminated", EliminatedOperator(op))):
            system = AnchoredSystem(operator, torch.tensor(3.0, dtype=F64))
            y, iterations, converged, _ = solve_anchored(system, h0, None, 1e-12, 1000)
            x_star = y[:fixture.num_nodes]
            error = float((x_star - h0).abs().max())
            accuracy = float(((x_star @ classes.T).argmax(-1) == labels).double().mean())
            numbers[name] = (error, accuracy, int(iterations.max()))
            self.assertTrue(bool(converged.all()))
            self.assertLessEqual(error, THM4_X_TOL, numbers)
            self.assertEqual(accuracy, 1.0)
        # negative control: one heterophilic fact rotated by 0.3 rad more
        bad, labels_b, classes_b, _ = thm4_production(fixture, perturb=0.3)
        x_b = classes_b[labels_b]
        y_b = bad.initial_guess(x_b)
        with torch.no_grad():
            energy_b = 0.5 * float((y_b * bad.apply(y_b)).sum())
        numbers["negative_control_energy"] = energy_b
        self.assertGreater(energy_b, 1e-3, numbers)
        self.numbers = numbers


# ----------------------------------------------------------------------------- T7
class GradcheckTests(unittest.TestCase):
    def test_T7_gradcheck_through_the_mlp_exponential_and_qr(self) -> None:
        edge_index, edge_type, _, _ = _graph(seed=7, num_nodes=5, num_facts=8)
        edge_index, edge_type = edge_index[:, :8], edge_type[:8]
        module = _builder((2,), d=6, rank=4, angle=0.5, checkpoint=False)
        g = torch.Generator().manual_seed(8)
        h = torch.randn(5, 6, generator=g, dtype=F64, requires_grad=True)
        rel = torch.randn(3, 4, generator=g, dtype=F64, requires_grad=True)
        x = torch.randn(8, 6, generator=g, dtype=F64)

        def through_mlp(h_in, rel_in):
            maps = module(h_in, edge_index, edge_type, rel_in).maps
            return maps.tgt.apply(x), maps.src.apply_T(x)

        self.assertTrue(torch.autograd.gradcheck(through_mlp, (h, rel)))
        last = module.edge_mlp[-1].weight.detach().clone().requires_grad_(True)

        def through_weights(weight):
            maps = torch.func.functional_call(module, {"edge_mlp.2.weight": weight},
                                              (h.detach(), edge_index, edge_type, rel.detach())).maps
            return maps.tgt.apply(x)

        self.assertTrue(torch.autograd.gradcheck(through_weights, (last,)))
        raw = torch.randn(3, 6, 4, generator=g, dtype=F64, requires_grad=True)
        with torch.no_grad():
            factors = module(h, edge_index, edge_type, rel).factors

        def through_qr(raw_in):
            u = orthonormal_subspaces(raw_in)
            maps = td1_edge_maps(factors._replace(subspaces=u), edge_type, 3, (2,))
            return maps.tgt.apply(x), maps.src.apply(x)

        self.assertTrue(torch.autograd.gradcheck(through_qr, (raw,)))


# ----------------------------------------------------------------------------- T8
class RemovedKeyTests(unittest.TestCase):
    def test_T8_removed_keys_raise_and_no_context_import(self) -> None:
        from src.configs.config import TrainConfig, reject_never_tuned_settings

        cases = (("model", "geometry_context", False, "Z1"),
                 ("model", "geometry_context", True, "Z1"),
                 ("ablation", "path_weight_context", True, "Z1"),
                 ("ablation", "path_weight_context", False, "Z1"),
                 ("model", "householder_k", 2, "Z2"))
        for section, key, value, item in cases:
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, f"spec 1.25 {item}"):
                    TrainConfig.from_dict({section: {key: value}})
                with self.assertRaisesRegex(ValueError, f"spec 1.25 {item}"):
                    reject_never_tuned_settings({section: {key: value}})
                for name in (key, f"{section}.{key}"):
                    with self.assertRaisesRegex(ValueError, f"spec 1.25 {item}"):
                        reject_never_tuned_settings({}, [name])
        self.assertIsNone(importlib.util.find_spec("src.models.relational_context"))
        importers = [str(path.relative_to(SRC)) for path in SRC.rglob("*.py")
                     if re.search(r"^\s*(from|import)\s+\S*relational_context",
                                  path.read_text(encoding="utf-8"), re.MULTILINE)]
        self.assertEqual(importers, [])
        # map_rank: fixed at 4 in tuning, never a search key; 2 <= m <= d accepted for tests
        with self.assertRaises(ValueError):
            reject_never_tuned_settings({"model": {"map_rank": 2}})
        reject_never_tuned_settings({"model": {"map_rank": 4}})
        for name in ("map_rank", "model.map_rank"):
            with self.assertRaises(ValueError):
                reject_never_tuned_settings({}, [name])
        for bad in (1, 9, 2.5, True):
            with self.subTest(map_rank=bad), self.assertRaises(ValueError):
                _config(model={"map_rank": bad})
        self.assertEqual(_config().model.map_rank, 4)


# ----------------------------------------------------------------------------- T10
class TypeOnlyTests(unittest.TestCase):
    def test_T10_type_only_read_out(self) -> None:
        from src.configs.config import reject_never_tuned_settings

        cfg = _config(evaluation={"type_only_maps": True})
        data = _double(_acm(cfg))
        model = _model(cfg, data).double()
        _randomize_skew(model)
        model.eval()
        with torch.no_grad():
            state = model.encode(data.x, data.node_type, data.edge_index, data.edge_type,
                                 data.face_groups, data.num_nodes).operator
            frames = model.node_to_edge.frames()
        q = _transports(state.maps)
        builder = model.node_to_edge
        worst = 0.0
        for e in range(data.edge_index.size(1)):
            r = int(data.edge_type[e])
            if r in builder.shared_role_relations:
                want = torch.eye(model.d, dtype=F64)
            else:
                want = frames[builder.framed_relations.index(r)]
            worst = max(worst, float((q[e] - want).abs().max()))
        self.assertTrue(builder.framed_relations)
        self.assertEqual(q.dtype, F64)
        self.assertLessEqual(worst, INIT_TOL)
        # control: the same model without the switch has instance rotations
        control = _model(_config(), data).double()
        control.load_state_dict(model.state_dict())
        control.eval()
        with torch.no_grad():
            q_full = _transports(control.encode(data.x, data.node_type, data.edge_index,
                                                data.edge_type, data.face_groups,
                                                data.num_nodes).operator.maps)
        self.assertGreater(float((q_full - q).abs().max()), 1e-3)
        # refused in a training forward and in tuning
        model.train()
        with self.assertRaisesRegex(ValueError, "type_only_maps"):
            model(data, epoch=0)
        with self.assertRaisesRegex(ValueError, "type_only_maps"):
            reject_never_tuned_settings({"evaluation": {"type_only_maps": True}})
        for name in ("type_only_maps", "evaluation.type_only_maps"):
            with self.assertRaises(ValueError):
                reject_never_tuned_settings({}, [name])
        reject_never_tuned_settings({"evaluation": {"type_only_maps": False}})
        self.worst = worst


# ----------------------------------------------------------------------------- T11, T12
def _train_step(model, data, optimizer) -> torch.Tensor:
    model.train()
    optimizer.zero_grad()
    logits = model(data, epoch=0).logits
    mask = data.target_train_mask
    loss = torch.nn.functional.cross_entropy(logits[mask], data.target_y[mask])
    loss.backward()
    return loss


class TrainingStepTests(unittest.TestCase):
    def test_T11_one_training_step_reaches_the_td1_parameters(self) -> None:
        from src.builders import build_optimizer

        cfg = _config()
        data = _acm(cfg)
        model = _model(cfg, data)
        optimizer = build_optimizer(model, cfg)
        named = dict(model.named_parameters())
        loss = _train_step(model, data, optimizer)
        self.assertTrue(bool(torch.isfinite(loss)))
        norms = {n: float(p.grad.norm()) for n, p in named.items() if p.grad is not None}
        for name in ("node_to_edge.frame_params", "node_to_edge.edge_mlp.0.weight",
                     "node_to_edge.edge_mlp.2.weight"):
            self.assertGreater(norms.get(name, 0.0), 0.0, name)
        skew_rows = named["node_to_edge.edge_mlp.2.weight"].grad[:model.node_to_edge.num_skew]
        self.assertGreater(float(skew_rows.norm()), 0.0, "C_e receives a gradient at C_e = 0")
        # at C_e = 0 the instance part has dA/dU = 0: W_r's gradient is exactly zero at step 1
        self.assertEqual(norms.get("node_to_edge.subspace_params", 0.0), 0.0)
        optimizer.step()
        loss2 = _train_step(model, data, optimizer)
        self.assertTrue(bool(torch.isfinite(loss2)))
        self.assertGreater(float(named["node_to_edge.subspace_params"].grad.norm()), 0.0,
                           "W_r receives a gradient once C_e != 0")
        self.norms = norms

    def test_T12_arms_train_one_step_and_direction_blind_has_no_frame(self) -> None:
        from src.builders import build_optimizer

        for name, kwargs in (("free_cell_maps", {"model": {"arm": "free_cell_maps"}}),
                             ("fact_maps_identity", {"ablation": {"fact_maps_identity": True}}),
                             ("direction_blind_maps", {"ablation": {"direction_blind_maps": True}})):
            with self.subTest(arm=name):
                cfg = _config(**kwargs)
                data = _acm(cfg)
                model = _model(cfg, data)
                optimizer = build_optimizer(model, cfg)
                loss = _train_step(model, data, optimizer)
                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertTrue(all(bool(torch.isfinite(p.grad).all())
                                    for p in model.parameters() if p.grad is not None))
                optimizer.step()
                self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.parameters()))
        # direction-blind: no frame, Q_e = exp(U_r C_e U_r^T) with C_e antisymmetric (float64)
        edge_index, edge_type, h, rel = _graph(seed=9)
        module = _builder((0, 1, 2), angle=1.0)
        self.assertEqual(module.framed_relations, ())
        self.assertEqual(module.frame_params.numel(), 0)
        maps = module(h, edge_index, edge_type, rel).maps
        q = _transports(maps)
        u = module.subspaces().detach()
        skew = _mlp_skew(module, h, rel, edge_index, edge_type)
        dense = torch.stack([torch.linalg.matrix_exp(u[int(r)] @ skew[e] @ u[int(r)].T)
                             for e, r in enumerate(edge_type.tolist())])
        worst = float((q - dense).abs().max())
        self.assertLessEqual(worst, NOFRAME_TOL)
        self.worst = worst


# ----------------------------------------------------------------------------- T14
class HeadInputTests(unittest.TestCase):
    def test_T14_head_reads_x_star_and_ego_head_is_the_1_24_input(self) -> None:
        from src.configs.config import reject_never_tuned_settings
        from src.models.h2snn_model import ModelDims, build_graph_classification_model

        cfg = _config()
        data = _acm(cfg)
        model = _model(cfg, data).eval()
        d = model.d
        self.assertEqual(model.readout[0].in_features, d)
        with torch.no_grad():
            encoded = model.encode(data.x, data.node_type, data.edge_index, data.edge_type,
                                   data.face_groups, data.num_nodes)
        self.assertTrue(torch.equal(encoded.representation, encoded.x_star))
        ego = _model(_config(ablation={"ego_head": True}), data).eval()
        self.assertEqual(ego.readout[0].in_features, 2 * d)
        with torch.no_grad():
            enc = ego.encode(data.x, data.node_type, data.edge_index, data.edge_type,
                             data.face_groups, data.num_nodes)
        self.assertTrue(torch.equal(enc.representation, torch.cat([enc.x_star, enc.h0], -1)))
        inst = _model(_config(model={"arm": "instance_features"}), data).eval()
        self.assertEqual(inst.readout[0].in_features, 2 * d)
        with torch.no_grad():
            enc_i = inst.encode(data.x, data.node_type, data.edge_index, data.edge_type,
                                data.face_groups, data.num_nodes)
        self.assertTrue(torch.equal(enc_i.representation[:, :d], enc_i.x_star))
        self.assertGreater(float(enc_i.representation[:, d:].abs().max()), 0.0)
        # LP and GC heads
        dims = ModelDims(in_channels=5, num_node_types=2, num_relations=4, num_classes=2,
                         face_lengths=(2, 3), path_actions=True)
        for ablation, width in (({}, d), ({"ego_head": True}, 2 * d)):
            gc = build_graph_classification_model(
                _config(dataset="mutag", ablation=ablation), dims)
            self.assertEqual(gc.graph_readout.decoder.in_features, width)
            from src.models.h2snn_model import H2SNN

            lp = H2SNN(_config(dataset="lastfm", ablation=ablation), dims)
            first = next(m for m in lp.link_readout.modules() if isinstance(m, torch.nn.Linear))
            self.assertEqual(first.in_features, width)
        # refused in a main-table tuning base config, never a search key
        with self.assertRaises(ValueError):
            reject_never_tuned_settings({"ablation": {"ego_head": True}})
        for name in ("ego_head", "ablation.ego_head"):
            with self.assertRaises(ValueError):
                reject_never_tuned_settings({}, [name])
        reject_never_tuned_settings({"ablation": {"ego_head": False}})
        with self.assertRaises(ValueError):
            _config(ablation={"ego_head": True, "h0_only_readout": True})


# ----------------------------------------------------------------------------- T15
class BoundedWeightTests(unittest.TestCase):
    def test_T15_bounded_cell_weights(self) -> None:
        from src.models.h2snn_model import H2SNN, ModelDims, build_graph_classification_model

        cfg = _config()
        data = _acm(cfg)
        bound = cfg.model.path_log_weight_bound
        self.assertEqual(bound, 10.0)
        groups = tuple(g for g in data.face_groups if g.num_faces > 0)
        # at initialisation: w_f = exp(L tanh(g / L)) on every rung, g from the gate itself
        for rung in ("full", "endpoints", "schema", "uniform"):
            with self.subTest(rung=rung):
                model = _model(_config(ablation={"path_weight_input": rung}), data).double().eval()
                h0 = model.encode_nodes(data.x.double(), data.node_type)
                with torch.no_grad():
                    _, weights, latents = model._cell_weights(h0, h0, groups)
                for w, latent, group in zip(weights, latents, groups):
                    if rung == "uniform":
                        b = model.path_log_weight
                        want = torch.exp(bound * torch.tanh(b / bound)).expand(group.num_faces)
                    else:
                        g_f = model.path_gate(latent).squeeze(-1)
                        want = torch.exp(bound * torch.tanh(g_f / bound))
                    self.assertLessEqual(float((w - want).abs().max()), WEIGHT_TOL)
        # LP and GC use the same weights
        dims = ModelDims(in_channels=5, num_node_types=3, num_relations=4, num_classes=2,
                         face_lengths=(2, 3), path_actions=True)
        for model in (H2SNN(_config(dataset="lastfm"), dims).double(),
                      build_graph_classification_model(_config(dataset="mutag"), dims).double()):
            with torch.no_grad():
                model.path_gate.bias.fill_(250.0)  # g = 250 without the bound
                latent = torch.randn(7, model.path_gate.in_features, dtype=F64)
                group = dataclasses.replace(groups[0],
                                            face_node_index=groups[0].face_node_index[:7],
                                            face_edge_index=groups[0].face_edge_index[:7])
                w = model._path_weights(latent, group, latent)
            self.assertLessEqual(float(w.log().max()), bound + 1e-12)
        # |g| up to 1e3: range <= 2L, finite gradients, no range stop in the forward
        model = _model(cfg, data).double()
        with torch.no_grad():
            model.path_gate.weight.mul_(5e3)
        model.train()
        logits = model(_double(data), epoch=0).logits
        stats = model.last_path_stats
        self.assertLessEqual(stats["path_log_w_range"], 2 * bound)
        h0 = model.encode_nodes(data.x.double(), data.node_type)
        raw = torch.cat([model.path_gate(lat).squeeze(-1)
                         for lat in model._cell_weights(h0, h0, groups)[2]])
        self.assertGreater(float(raw.abs().max()), 100.0, "fixture: g reaches large values")
        logits.square().sum().backward()
        self.assertTrue(all(bool(torch.isfinite(p.grad).all())
                            for p in model.parameters() if p.grad is not None))
        self.range = stats["path_log_w_range"]


def _double(data):
    from tests.test_factorized_maps import _as_float64

    return _as_float64(data)


if __name__ == "__main__":
    unittest.main()
