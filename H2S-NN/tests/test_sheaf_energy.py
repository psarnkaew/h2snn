"""v3-B card U1 (`docs/H2S-NN_v3B_card.md` §2): the three-level sheaf energy operator.

T1 functoriality `R_{e_i,f} F_{v<|e_i} = T_{v,f}` at both ends of every incidence on a real WISP
   draw (IMDB, r = 2, 50 anchors, fp32): max norm <= 1e-5.
T2 matrix-free `L` and `L_hat` vs an independent dense construction (N = 12, E = 20, F = 6, d = 4,
   fp64): <= 1e-12.
T3 symmetry `<Y1, L_hat Y2> = <L_hat Y1, Y2>` <= 1e-12 and PSD (min eig >= -1e-10).
"""

from __future__ import annotations

import math
from pathlib import Path
import unittest

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.data.structures import FaceGroup
from src.models.map_operators import EdgeMapPair, FactorizedMapOperator, normalize_reflectors
from src.models.sheaf_energy import (
    IncidenceComplex,
    SheafEnergyOperator,
    build_cell_group,
    materialize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
T1_TOL = 1e-5
T2_TOL = 1e-12
T3_SYM_TOL = 1e-12
T3_PSD_TOL = -1e-10


# ----------------------------------------------------------------------------- fixture
def _orthogonal_maps(num_edges: int, edge_type: torch.Tensor, d: int, k: int,
                     generator: torch.Generator, dtype: torch.dtype) -> EdgeMapPair:
    """Random factorized maps, one shared scale per fact in [0.5, 2] (log-uniform)."""
    num_relations = int(edge_type.max()) + 1
    log_scale = math.log(0.5) + math.log(4.0) * torch.rand(num_edges, generator=generator,
                                                            dtype=torch.float64)
    scale = log_scale.exp().to(dtype)

    def side() -> FactorizedMapOperator:
        rotations, _ = torch.linalg.qr(torch.randn(num_relations, d, d, generator=generator,
                                                   dtype=torch.float64))
        reflectors = normalize_reflectors(
            torch.randn(num_edges, k, d, generator=generator, dtype=torch.float64), 1e-6)
        return FactorizedMapOperator(scale=scale, reflectors=reflectors.to(dtype),
                                     rotations=rotations.to(dtype), group_ids=edge_type)

    return EdgeMapPair(src=side(), tgt=side())


class Fixture:
    """Random graph with cycles, one self-loop fact, and simple paths (mixed hop signs) as
    cells; facts on 0, 1 and >= 2 cells. `cells` is a list of (nodes, facts, signs)."""

    def __init__(self, seed: int = 3, num_nodes: int = 12, num_facts: int = 20, num_cells: int = 6,
                 d: int = 4, k: int = 2, dtype: torch.dtype = torch.float64) -> None:
        g = torch.Generator().manual_seed(seed)
        self.num_nodes, self.d, self.dtype = num_nodes, d, dtype
        pairs: list[tuple[int, int]] = []
        while len(pairs) < num_facts - 1:
            u, v = torch.randint(num_nodes, (2,), generator=g).tolist()
            if u != v and (u, v) not in pairs and (v, u) not in pairs:
                pairs.append((u, v))
        pairs.append((3, 3))  # one self-loop fact (both roles on the same node)
        self.edge_index = torch.tensor(pairs, dtype=torch.long).T
        self.edge_type = torch.randint(2, (num_facts,), generator=g)
        incident: dict[int, list[tuple[int, int, int]]] = {v: [] for v in range(num_nodes)}
        for e, (u, v) in enumerate(pairs[:-1]):
            incident[u].append((e, v, +1))
            incident[v].append((e, u, -1))
        cells: list[tuple[list[int], list[int], list[int]]] = []
        lengths = [2, 2, 3, 3, 4, 4]
        attempts = 0
        while len(cells) < num_cells and attempts < 10_000:
            attempts += 1
            length = lengths[len(cells)]
            nodes = [int(torch.randint(num_nodes, (1,), generator=g))]
            facts, signs = [], []
            for _ in range(length):
                options = [o for o in incident[nodes[-1]] if o[1] not in nodes]
                if not options:
                    break
                e, nxt, sign = options[int(torch.randint(len(options), (1,), generator=g))]
                nodes.append(nxt)
                facts.append(e)
                signs.append(sign)
            if len(facts) == length and (nodes, facts) not in [(c[0], c[1]) for c in cells]:
                cells.append((nodes, facts, signs))
        self.cells = cells
        self.groups: list[FaceGroup] = []
        self.beta_by_group: list[torch.Tensor] = []
        for length in sorted({len(c[1]) for c in cells}):
            rows = [c for c in cells if len(c[1]) == length]
            self.groups.append(FaceGroup(
                template_name=f"L{length}", node_type_sequence=(),
                face_node_index=torch.tensor([c[0] for c in rows]),
                face_edge_index=torch.tensor([c[1] for c in rows]),
                relation_index=self.edge_type[torch.tensor([c[1] for c in rows])],
                face_edge_sign=torch.tensor([c[2] for c in rows], dtype=torch.float32),
                path_weight=None,
            ))
            self.beta_by_group.append(
                torch.exp(0.5 * torch.randn(len(rows), generator=g, dtype=torch.float64)).to(dtype))
        self.maps = _orthogonal_maps(num_facts, self.edge_type, d, k, g, dtype)
        self.generator = g

    # --- the operator under test
    def operator(self, node_cell: bool = True, fact_cell: bool = True) -> SheafEnergyOperator:
        cell_groups = [build_cell_group(self.maps, group, self.edge_index, beta)
                       for group, beta in zip(self.groups, self.beta_by_group)]
        return SheafEnergyOperator(self.maps, self.edge_index, self.num_nodes, cell_groups,
                                   node_cell=node_cell, fact_cell=fact_cell)

    def complex(self) -> IncidenceComplex:
        return IncidenceComplex.from_face_groups(self.edge_index, self.num_nodes, self.groups)

    # --- independent dense reference (explicit matrices; no UP/DOWN code)
    def dense_reference(self, node_cell: bool = True, fact_cell: bool = True
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(L, D)` on `[x; s; z]` from the energy's definition, block by block."""
        d, n, e_count = self.d, self.num_nodes, self.edge_index.size(1)
        f_count = len(self.cells)
        rows = n + e_count + f_count
        L = torch.zeros(rows * d, rows * d, dtype=self.dtype)
        D = torch.zeros(rows, dtype=self.dtype)
        eye = torch.eye(d, dtype=self.dtype)
        scale = self.maps.src.scale
        F_src = self.maps.src.materialize() / scale.view(-1, 1, 1)
        F_tgt = self.maps.tgt.materialize() / scale.view(-1, 1, 1)
        a = scale.square()

        def block(i: int) -> slice:
            return slice(i * d, (i + 1) * d)

        def add(i: int, j: int, Fi: torch.Tensor, Fj: torch.Tensor, w: torch.Tensor) -> None:
            L[block(i), block(i)] += w * Fi.T @ Fi
            L[block(j), block(j)] += w * Fj.T @ Fj
            L[block(i), block(j)] -= w * Fi.T @ Fj
            L[block(j), block(i)] -= w * Fj.T @ Fi
            D[i] += w
            D[j] += w

        for e in range(e_count):
            u, v = int(self.edge_index[0, e]), int(self.edge_index[1, e])
            add(u, n + e, F_src[e], eye, a[e])
            add(v, n + e, F_tgt[e], eye, a[e])
        cell_id = 0
        betas = torch.cat(self.beta_by_group)
        ordered = [c for group in self.groups for c in self.cells
                   if len(c[1]) == group.length]
        for nodes, facts, signs in ordered:
            beta = betas[cell_id]
            T = eye.clone()
            if node_cell:
                add(nodes[0], n + e_count + cell_id, T, eye, beta)
            for i, (e, sign) in enumerate(zip(facts, signs)):
                near = F_src[e] if sign > 0 else F_tgt[e]
                far = F_tgt[e] if sign > 0 else F_src[e]
                R = T @ near.T
                T = R @ far
                if fact_cell:
                    add(n + e, n + e_count + cell_id, R, eye, beta)
                if node_cell:
                    add(nodes[i + 1], n + e_count + cell_id, T, eye, beta)
            cell_id += 1
        return L, D


def _normalize(L: torch.Tensor, D: torch.Tensor, d: int) -> torch.Tensor:
    inv = torch.where(D > 0, D.clamp_min(1e-300).rsqrt(), torch.zeros_like(D))
    inv = inv.repeat_interleave(d)
    return inv.unsqueeze(1) * L * inv.unsqueeze(0)


# ----------------------------------------------------------------------------- T2 / T3
class OperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Fixture()
        complex_ = self.fixture.complex()
        counts = complex_.cells_per_fact
        self.assertEqual(complex_.num_cells, 6)
        self.assertTrue(bool((counts == 0).any()) and bool((counts == 1).any())
                        and bool((counts >= 2).any()),
                        f"fixture needs facts on 0, 1 and >= 2 cells, got {counts.tolist()}")
        self.assertTrue(any(s < 0 for _, _, signs in self.fixture.cells for s in signs),
                        "fixture needs a reversed hop")

    def test_T2_matrix_free_equals_dense(self) -> None:
        for node_cell, fact_cell in ((True, True), (False, True), (True, False)):
            with self.subTest(node_cell=node_cell, fact_cell=fact_cell):
                op = self.fixture.operator(node_cell, fact_cell)
                L_ref, D_ref = self.fixture.dense_reference(node_cell, fact_cell)
                rows = op.layout.rows
                L_mf = materialize(op.apply, rows, self.fixture.d, torch.float64)
                self.assertLess(float((L_mf - L_ref).abs().max()), T2_TOL)
                D_mf = torch.cat(op.degree())
                self.assertLess(float((D_mf - D_ref).abs().max()), T2_TOL)
                Lhat_mf = materialize(op.apply_normalized, rows, self.fixture.d, torch.float64)
                Lhat_ref = _normalize(L_ref, D_ref, self.fixture.d)
                self.assertLess(float((Lhat_mf - Lhat_ref).abs().max()), T2_TOL)
        # the incidence lists agree with the groups
        complex_ = self.fixture.complex()
        self.assertEqual(complex_.fact_cell.size(0), sum(len(c[1]) for c in self.fixture.cells))
        self.assertEqual(complex_.node_cell.size(0), sum(len(c[0]) for c in self.fixture.cells))
        self.assertEqual(complex_.num_shared_facts, int((complex_.cells_per_fact >= 2).sum()))

    def test_T3_normalized_operator_is_symmetric_and_psd(self) -> None:
        op = self.fixture.operator()
        g = torch.Generator().manual_seed(11)
        rows, d = op.layout.rows, self.fixture.d
        y1 = torch.randn(rows, d, generator=g, dtype=torch.float64)
        y2 = torch.randn(rows, d, generator=g, dtype=torch.float64)
        y1, y2 = y1 / y1.norm(), y2 / y2.norm()
        lhs = float((y1 * op.apply_normalized(y2)).sum())
        rhs = float((op.apply_normalized(y1) * y2).sum())
        self.assertLess(abs(lhs - rhs), T3_SYM_TOL)
        Lhat = materialize(op.apply_normalized, rows, d, torch.float64)
        self.assertLess(float((Lhat - Lhat.T).abs().max()), T3_SYM_TOL)
        eigenvalues = torch.linalg.eigvalsh(Lhat)
        self.assertGreaterEqual(float(eigenvalues.min()), T3_PSD_TOL)
        # every diagonal block of L_hat is I (D > 0 everywhere on this fixture)
        blocks = Lhat.view(rows, d, rows, d).diagonal(dim1=0, dim2=2).permute(2, 0, 1)
        self.assertLess(float((blocks - torch.eye(d, dtype=torch.float64)).abs().max()), T2_TOL)


# ----------------------------------------------------------------------------- T1 (real draw)
@unittest.skipUnless((REPO_ROOT / "data" / "imdb").exists(), "HGB IMDB data not present")
class FunctorialityOnRealDrawTests(unittest.TestCase):
    """T1 on one real WISP draw: IMDB, r = 2, 50 anchors, fp32, d = 32."""

    def test_T1_fact_to_cell_maps_are_functorial_on_imdb(self) -> None:
        from src.configs.config import TrainConfig
        from src.data.dataset import build_dataset
        from src.data.wisp_sampler import build_wisp_path_groups
        from src.models.sheaf_maps import FactorizedNodeToEdgeSheaf

        config = TrainConfig.from_dict({
            "data": {"dataset_name": "imdb", "root": str(REPO_ROOT / "data"),
                     "wisp_walks_per_anchor": 2, "wisp_max_length": 4},
            "model": {"d": 32}, "device": "cpu",
        })
        data = build_dataset(config)
        start, end = data.node_slices["movie"]
        anchors = torch.arange(start, min(start + 50, end))
        groups = build_wisp_path_groups(
            edge_index=data.edge_index, edge_type=data.edge_type, node_type=data.node_type,
            self_inverse=data.fact_layer.self_inverse, anchors=anchors, walks_per_anchor=2,
            max_length=4, seed=config.seed)
        groups = tuple(g for g in groups if g.num_faces > 0)
        self.assertTrue(groups)
        torch.manual_seed(0)
        d = 32
        module = FactorizedNodeToEdgeSheaf(
            d=d, d_r=16, num_relations=len(data.relation_names), hidden_dim=64, householder_k=2,
            scale_eps=0.1, orthogonal_map="fasthpp", map_scale="per_edge",
            shared_role_relations=data.fact_layer.self_inverse)
        h = torch.randn(data.num_nodes, d)
        relation_embeddings = torch.randn(len(data.relation_names), 16)
        with torch.no_grad():
            maps = module(h, data.edge_index, data.edge_type, relation_embeddings).maps
            scale = maps.src.scale
            worst = 0.0
            incidences = 0
            for group in groups:
                cell = build_cell_group(maps, group, data.edge_index,
                                        torch.ones(group.num_faces))
                P, L = group.face_edge_index.shape
                basis = torch.eye(d)
                # T_{v_i,f} e_k and R_{e_i,f} e_k for every basis vector, read off matrix-free
                T = torch.stack([cell.transport.apply(basis[k].expand(data.num_nodes, d))
                                 for k in range(d)], dim=-1)                     # [P, L+1, d, d]
                R = torch.stack([cell.fact_maps.apply(basis[k].expand(data.num_edges, d))
                                 for k in range(d)], dim=-1)                     # [P, L, d, d]
                sel_src = maps.src.select(group.face_edge_index.reshape(-1))
                sel_tgt = maps.tgt.select(group.face_edge_index.reshape(-1))
                F_src = (sel_src.materialize() / scale[group.face_edge_index.reshape(-1)].view(-1, 1, 1)
                         ).view(P, L, d, d)
                F_tgt = (sel_tgt.materialize() / scale[group.face_edge_index.reshape(-1)].view(-1, 1, 1)
                         ).view(P, L, d, d)
                reversed_hop = (group.face_edge_sign < 0).view(P, L, 1, 1)
                F_near = torch.where(reversed_hop, F_tgt, F_src)
                F_far = torch.where(reversed_hop, F_src, F_tgt)
                near_residual = R @ F_near - T[:, :-1]
                far_residual = R @ F_far - T[:, 1:]
                worst = max(worst, float(near_residual.norm(dim=(-2, -1)).max()),
                            float(far_residual.norm(dim=(-2, -1)).max()))
                incidences += P * L
        self.assertGreater(incidences, 0)
        self.assertLess(worst, T1_TOL, f"functoriality residual {worst} on {incidences} incidences")


if __name__ == "__main__":
    unittest.main()
