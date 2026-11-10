"""v3-B theory checks (Part L, unit L8b; spec v1.11 §4), fp64, CPU, on the T2/T7 fixture
(`tests.test_sheaf_energy.Fixture`: 12 nodes, 20 facts incl. one self-loop, connected, cycle rank 8;
6 cells of lengths 2-4 with reversed hops).

Kernel (Thm 3)
  K1 flat sheaf (every fact map I, a_e > 0, beta_f > 0): dim ker L_hat = d (eigenvalues <= 1e-10;
     the next eigenvalue is reported); every kernel vector of L, and D^{-1/2} times every kernel
     vector of L_hat, has x constant, s_e = x and z_f = x.
  K2 arm free_cell_maps (random orthogonal R_{e,f} per incidence) on the same flat fact maps:
     dim ker L_hat = 0 (smallest eigenvalue reported).
  K3 random orthogonal fact maps: dim ker L_hat = dim ker Delta_0, computed independently, for
     (a) the fixture's factorized maps and (b) a gauge transform of the flat sheaf
     (F_{v<|e} = G_e O_v^T, dense random orthogonal G_e, O_v), whose node kernel is d.
Fixed point (Thm 6 with Thm 4's maps)
  F1 maps that are I or one reflection per incidence make x_v = a_{y(v)} a global section for a
     heterophilic labelling y; the anchored solve returns it: ||x* - h0|| / ||h0|| <= 1e-10 with the
     unnormalised L (h0 = a_y) and with L_hat (h0 = D_v^{1/2} a_y), by dense solve and by PCG.
Unroll vs implicit
  U1 gradient of the T10 loss by autograd through an unrolled PCG run (tol 1e-12) vs the
     AnchoredSolve implicit gradient (tol 1e-12): relative error ||g_unroll - g_implicit|| /
     ||g_implicit|| <= 1e-6 for every parameter group incl. theta_tau and h0.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.models.anchored_solve import AnchoredSystem, SolveStats, anchored_solve, pcg, solve_anchored
from src.models.map_operators import (
    DenseMapOperator,
    EdgeMapPair,
    FactorizedMapOperator,
    normalize_reflectors,
)
from src.models.sheaf_energy import (
    EliminatedOperator,
    SheafEnergyOperator,
    build_cell_group,
    materialize,
)
from tests.test_anchored_solve import Parameters, build_operator
from tests.test_sheaf_energy import Fixture

F64 = torch.float64
KERNEL_EIG_TOL = 1e-10      # an eigenvalue <= this counts as kernel
STRUCTURE_TOL = 1e-8        # kernel vectors equal 1 (x) c entrywise (eigenvector accuracy ~ eps / gap)
FIXED_POINT_TOL = 1e-10
UNROLL_TOL = 1e-6
SOLVE_TOL = 1e-12
PCG_FIXED_POINT_TOL = 1e-13
T_MAX = 100
TAU_FIXED_POINT = 3.0


# ----------------------------------------------------------------------------- builders
def flat_maps(fixture: Fixture) -> EdgeMapPair:
    """Every fact map is I (k = 2 equal reflectors, identity frames), scales from the fixture."""
    g = torch.Generator().manual_seed(21)
    E, d = fixture.edge_index.size(1), fixture.d
    num_relations = int(fixture.edge_type.max()) + 1
    v = normalize_reflectors(torch.randn(E, 1, d, generator=g, dtype=F64), 1e-6)
    reflectors = torch.cat([v, v], dim=1)  # H(v) H(v) = I
    frames = torch.eye(d, dtype=F64).expand(num_relations, d, d).clone()
    scale = fixture.maps.src.scale

    def side() -> FactorizedMapOperator:
        return FactorizedMapOperator(scale, reflectors.clone(), frames.clone(), fixture.edge_type)

    return EdgeMapPair(src=side(), tgt=side())


def dense_maps(src: torch.Tensor, tgt: torch.Tensor, scale: torch.Tensor) -> EdgeMapPair:
    """Explicit orthogonal maps `[E, d, d]` (times the shared scale) behind the operator API."""
    return EdgeMapPair(src=DenseMapOperator(scale.view(-1, 1, 1) * src, scale),
                       tgt=DenseMapOperator(scale.view(-1, 1, 1) * tgt, scale))


def free_incidence_maps(fixture: Fixture, seed: int = 31) -> list[FactorizedMapOperator]:
    """One random orthogonal map per fact-cell incidence of each face group (arm free_cell_maps)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for group in fixture.groups:
        rows = group.face_edge_index.numel()
        rotation, _ = torch.linalg.qr(torch.randn(1, fixture.d, fixture.d, generator=g, dtype=F64))
        reflectors = normalize_reflectors(torch.randn(rows, 2, fixture.d, generator=g, dtype=F64),
                                          1e-6)
        out.append(FactorizedMapOperator(torch.ones(rows, dtype=F64), reflectors, rotation,
                                         torch.zeros(rows, dtype=torch.long)))
    return out


def energy_operator(fixture: Fixture, maps: EdgeMapPair,
                    free_maps: list[FactorizedMapOperator] | None = None) -> SheafEnergyOperator:
    cells = [build_cell_group(maps, group, fixture.edge_index, beta,
                              free_maps=None if free_maps is None else free_maps[i])
             for i, (group, beta) in enumerate(zip(fixture.groups, fixture.beta_by_group))]
    return SheafEnergyOperator(maps, fixture.edge_index, fixture.num_nodes, cells)


def dense_operators(op: SheafEnergyOperator, d: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = op.layout.rows
    return (materialize(op.apply, rows, d, F64), materialize(op.apply_normalized, rows, d, F64))


def kernel(matrix: torch.Tensor, tol: float = KERNEL_EIG_TOL
           ) -> tuple[int, torch.Tensor, torch.Tensor]:
    """`(dim ker, eigenvalues ascending, kernel basis [n, dim])` of a symmetric matrix."""
    eigenvalues, vectors = torch.linalg.eigh(0.5 * (matrix + matrix.T))
    dim = int((eigenvalues <= tol).sum())
    return dim, eigenvalues, vectors[:, :dim]


def node_laplacian(fixture: Fixture, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Dense `Delta_0 = delta^T delta`, `(delta x)_e = F_src,e x_u - F_tgt,e x_v` (orthogonal
    factors; the weights a_e > 0 do not change the kernel)."""
    E, d, n = fixture.edge_index.size(1), fixture.d, fixture.num_nodes
    delta = torch.zeros(E * d, n * d, dtype=F64)
    for e in range(E):
        u, v = int(fixture.edge_index[0, e]), int(fixture.edge_index[1, e])
        delta[e * d:(e + 1) * d, u * d:(u + 1) * d] += src[e]
        delta[e * d:(e + 1) * d, v * d:(v + 1) * d] -= tgt[e]
    return delta.T @ delta


def constant_section_deviation(basis: torch.Tensor, rows: int, d: int) -> float:
    """Max over kernel vectors of `max |Y - 1 (x) Y_0|` after scaling each vector to unit max."""
    worst = 0.0
    for k in range(basis.size(1)):
        y = basis[:, k].view(rows, d)
        y = y / y.abs().max()
        worst = max(worst, float((y - y[0]).abs().max()))
    return worst


def flat_kernel_numbers(fixture: Fixture) -> dict[str, float]:
    """K1 numbers (shared with the mutation tests)."""
    op = energy_operator(fixture, flat_maps(fixture))
    L, Lhat = dense_operators(op, fixture.d)
    rows, d = op.layout.rows, fixture.d
    dim_hat, eig_hat, basis_hat = kernel(Lhat)
    dim_l, _, basis_l = kernel(L)
    inv_sqrt = torch.cat(op.inverse_sqrt_degree()).repeat_interleave(d)
    lifted = inv_sqrt.unsqueeze(1) * basis_hat  # D^{-1/2} ker L_hat = ker L
    return {
        "dim_ker_L_hat": dim_hat,
        "dim_ker_L": dim_l,
        "max_kernel_eigenvalue": float(eig_hat[dim_hat - 1]) if dim_hat else float("nan"),
        "next_eigenvalue": float(eig_hat[dim_hat]),
        "structure_L": constant_section_deviation(basis_l, rows, d),
        "structure_L_hat": constant_section_deviation(lifted, rows, d),
    }


# ----------------------------------------------------------------------------- K1-K3
class KernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Fixture()

    def test_K1_flat_sheaf_kernel_is_the_constant_sections(self) -> None:
        numbers = flat_kernel_numbers(self.fixture)
        d = self.fixture.d
        self.assertEqual(numbers["dim_ker_L_hat"], d, numbers)
        self.assertEqual(numbers["dim_ker_L"], d, numbers)
        self.assertGreater(numbers["next_eigenvalue"], 1e3 * KERNEL_EIG_TOL, numbers)
        self.assertLess(numbers["structure_L"], STRUCTURE_TOL, numbers)
        self.assertLess(numbers["structure_L_hat"], STRUCTURE_TOL, numbers)
        self.numbers = numbers

    def test_K2_free_cell_maps_collapse_the_kernel(self) -> None:
        op = energy_operator(self.fixture, flat_maps(self.fixture),
                             free_incidence_maps(self.fixture))
        _, Lhat = dense_operators(op, self.fixture.d)
        dim, eigenvalues, _ = kernel(Lhat)
        self.assertEqual(dim, 0, f"smallest eigenvalue {float(eigenvalues[0]):.3e}")
        self.assertGreater(float(eigenvalues[0]), KERNEL_EIG_TOL)
        self.smallest = float(eigenvalues[0])

    def test_K3_kernel_equals_the_node_sheaf_kernel(self) -> None:
        fixture = self.fixture
        d, n, E = fixture.d, fixture.num_nodes, fixture.edge_index.size(1)
        scale = fixture.maps.src.scale
        # (a) the fixture's random factorized maps
        src_a = fixture.maps.src.materialize() / scale.view(-1, 1, 1)
        tgt_a = fixture.maps.tgt.materialize() / scale.view(-1, 1, 1)
        # (b) gauge transform of the flat sheaf: F_{v<|e} = G_e O_v^T (dense, random orthogonal)
        g = torch.Generator().manual_seed(41)
        G, _ = torch.linalg.qr(torch.randn(E, d, d, generator=g, dtype=F64))
        O, _ = torch.linalg.qr(torch.randn(n, d, d, generator=g, dtype=F64))
        src_b = G @ O[fixture.edge_index[0]].transpose(-2, -1)
        tgt_b = G @ O[fixture.edge_index[1]].transpose(-2, -1)
        cases = {
            "factorized random": (fixture.maps, src_a, tgt_a, None),
            "gauge of flat": (dense_maps(src_b, tgt_b, scale), src_b, tgt_b, d),
        }
        self.numbers = {}
        for name, (maps, src, tgt, expected) in cases.items():
            with self.subTest(case=name):
                dim_node, eig_node, _ = kernel(node_laplacian(fixture, src, tgt))
                op = energy_operator(fixture, maps)
                _, Lhat = dense_operators(op, d)
                dim_full, eig_full, _ = kernel(Lhat)
                self.numbers[name] = (dim_full, dim_node, float(eig_full[dim_full]),
                                      float(eig_node[dim_node]))
                self.assertEqual(dim_full, dim_node, self.numbers[name])
                if expected is not None:
                    self.assertEqual(dim_node, expected)


# ----------------------------------------------------------------------------- F1
def thm4_section(fixture: Fixture, num_classes: int = 3, seed: int = 51
                 ) -> tuple[EdgeMapPair, torch.Tensor, int]:
    """Labelling y, distinct unit class vectors a_c, and maps I (source role) / I or one
    reflection H(w) (target role) with F_src a_{y(u)} = F_tgt a_{y(v)} on every fact.
    Returns (maps, x_v = a_{y(v)} [N, d], number of heterophilic facts)."""
    g = torch.Generator().manual_seed(seed)
    n, d, E = fixture.num_nodes, fixture.d, fixture.edge_index.size(1)
    labels = torch.randint(num_classes, (n,), generator=g)
    classes = torch.randn(num_classes, d, generator=g, dtype=F64)
    classes = classes / classes.norm(dim=-1, keepdim=True)
    eye = torch.eye(d, dtype=F64)
    src = eye.expand(E, d, d).clone()
    tgt = eye.expand(E, d, d).clone()
    heterophilic = 0
    for e in range(E):
        u, v = int(fixture.edge_index[0, e]), int(fixture.edge_index[1, e])
        p, q = int(labels[u]), int(labels[v])
        if p != q:
            w = classes[q] - classes[p]
            w = w / w.norm()
            tgt[e] = eye - 2.0 * torch.outer(w, w)  # H(w) a_q = a_p
            heterophilic += 1
    return dense_maps(src, tgt, fixture.maps.src.scale), classes[labels], heterophilic


class _Unnormalised:
    """`L` (not `L_hat`) behind the solver's operator interface (F1, unnormalised case)."""

    def __init__(self, op: SheafEnergyOperator) -> None:
        self.op = op
        self.num_nodes = op.num_nodes
        self.layout = op.layout

    def apply_normalized(self, y: torch.Tensor) -> torch.Tensor:
        return self.op.apply(y)

    def initial_guess(self, h0: torch.Tensor) -> torch.Tensor:
        return self.op.initial_guess(h0)


def _dense_anchored(matrix: torch.Tensor, h0: torch.Tensor, tau: float) -> torch.Tensor:
    n_d = h0.numel()
    M = tau * matrix
    M[:n_d, :n_d] += torch.eye(n_d, dtype=F64)
    b = torch.zeros(matrix.size(0), dtype=F64)
    b[:n_d] = h0.reshape(-1)
    return torch.linalg.solve(M, b)[:n_d].view_as(h0)


def fixed_point_errors(fixture: Fixture) -> dict[str, float]:
    maps, section, heterophilic = thm4_section(fixture)
    op = energy_operator(fixture, maps)
    L, Lhat = dense_operators(op, fixture.d)
    d_v = op.degree()[0]
    h_unnorm = section
    h_norm = d_v.sqrt().unsqueeze(-1) * section
    tau = torch.tensor(TAU_FIXED_POINT, dtype=F64)

    def rel(x: torch.Tensor, h: torch.Tensor) -> float:
        return float((x - h).norm() / h.norm())

    out = {"heterophilic_facts": float(heterophilic), "facts": float(fixture.edge_index.size(1))}
    out["dense_L"] = rel(_dense_anchored(L, h_unnorm, TAU_FIXED_POINT), h_unnorm)
    out["dense_L_hat"] = rel(_dense_anchored(Lhat, h_norm, TAU_FIXED_POINT), h_norm)
    for name, operator, h in (("pcg_L", _Unnormalised(op), h_unnorm),
                              ("pcg_L_hat", op, h_norm),
                              ("pcg_L_hat_eliminated", EliminatedOperator(op), h_norm)):
        system = AnchoredSystem(operator, tau)
        y0 = torch.zeros(operator.layout.rows, fixture.d, dtype=F64)
        y, iterations, converged, _ = solve_anchored(system, h, y0, PCG_FIXED_POINT_TOL, T_MAX)
        out[name] = rel(y[:fixture.num_nodes], h)
        out[name + "_iterations"] = float(iterations.max())
    # the section itself is in the kernel: ||L Y_sec|| (unnormalised)
    y_sec = op.initial_guess(h_unnorm)  # D-B7 guess = (h, F h, T h) exactly for a section
    out["residual_L_section"] = float(op.apply(y_sec).norm() / y_sec.norm())
    return out


class FixedPointTests(unittest.TestCase):
    def test_F1_global_section_is_the_fixed_point(self) -> None:
        numbers = fixed_point_errors(Fixture())
        self.assertGreater(numbers["heterophilic_facts"], 0)
        for key in ("dense_L", "dense_L_hat", "pcg_L", "pcg_L_hat", "pcg_L_hat_eliminated",
                    "residual_L_section"):
            self.assertLess(numbers[key], FIXED_POINT_TOL, f"{key}: {numbers}")
        self.numbers = numbers


# ----------------------------------------------------------------------------- U1
def _t10_loss(params: Parameters, y: torch.Tensor) -> torch.Tensor:
    x_star = y[:params.fixture.num_nodes]
    return (params.weight * x_star.square()).sum() + (params.weight * x_star).sum()


def unroll_vs_implicit(fixture: Fixture | None = None) -> dict[str, float]:
    """Relative error per parameter group between the unrolled and the implicit gradient."""
    fixture = fixture or Fixture()
    params = Parameters(fixture)
    groups = params.groups()

    def leaves() -> dict[str, list[torch.Tensor]]:
        return {k: [t.clone().requires_grad_(True) for t in v] for k, v in groups.items()}

    def operator_params(v: dict[str, list[torch.Tensor]]) -> tuple[torch.Tensor, ...]:
        return (v["fact_scales"][0], *v["reflectors"], *v["rotations"], *v["cell_weights"])

    implicit = leaves()
    stats = SolveStats()
    y = anchored_solve(lambda *p: build_operator(fixture, *p), implicit["h0"][0],
                       implicit["theta_tau"][0], operator_params(implicit), SOLVE_TOL, T_MAX, stats)
    flat_implicit = [t for v in implicit.values() for t in v]
    grads_implicit = torch.autograd.grad(_t10_loss(params, y), flat_implicit, allow_unused=True)

    unrolled = leaves()
    operator = build_operator(fixture, *operator_params(unrolled))
    system = AnchoredSystem(operator, torch.exp(unrolled["theta_tau"][0]))
    h0 = unrolled["h0"][0]
    y_u, iterations, converged, _ = pcg(system.apply, system.right_hand_side(h0),
                                        operator.initial_guess(h0), system.jacobi_inverse(),
                                        SOLVE_TOL, T_MAX)
    flat_unrolled = [t for v in unrolled.values() for t in v]
    grads_unrolled = torch.autograd.grad(_t10_loss(params, y_u), flat_unrolled, allow_unused=True)

    out: dict[str, float] = {"forward_iterations": float(stats.forward_iterations),
                             "backward_iterations": float(stats.backward_iterations),
                             "unrolled_iterations": float(iterations.max()),
                             "unrolled_converged": float(bool(converged.all()))}
    index = 0
    for name, tensors in groups.items():
        num = 0.0
        den = 0.0
        for _ in tensors:
            gi = grads_implicit[index]
            gu = grads_unrolled[index]
            gi = torch.zeros_like(flat_implicit[index]) if gi is None else gi
            gu = torch.zeros_like(flat_unrolled[index]) if gu is None else gu
            num += float((gu - gi).square().sum())
            den += float(gi.square().sum())
            index += 1
        out[name] = (num ** 0.5) / max(den ** 0.5, 1e-300)
        out[name + "_norm"] = den ** 0.5
    return out


GROUPS = ("fact_scales", "reflectors", "rotations", "cell_weights", "theta_tau", "h0")


class UnrollTests(unittest.TestCase):
    def test_U1_unrolled_gradient_equals_the_implicit_gradient(self) -> None:
        numbers = unroll_vs_implicit()
        self.assertEqual(numbers["unrolled_converged"], 1.0)
        self.assertLess(numbers["forward_iterations"], T_MAX)
        self.assertLess(numbers["backward_iterations"], T_MAX)
        for name in GROUPS:
            self.assertGreater(numbers[name + "_norm"], 0.0, name)
            self.assertLess(numbers[name], UNROLL_TOL, f"{name}: {numbers}")
        self.numbers = numbers


if __name__ == "__main__":
    unittest.main()
