"""L8b: print the numbers behind tests/test_v3b_theory.py (run from H2S-NN/ in the h2snn env)."""
import sys
sys.path.insert(0, ".")
import torch
from tests.test_sheaf_energy import Fixture
from tests import test_v3b_theory as t

fixture = Fixture()
print("K1 flat:", t.flat_kernel_numbers(fixture))
op = t.energy_operator(fixture, t.flat_maps(fixture), t.free_incidence_maps(fixture))
_, Lhat = t.dense_operators(op, fixture.d)
dim, eig, _ = t.kernel(Lhat)
print("K2 free_cell_maps: dim ker", dim, "smallest eigenvalues", [f"{float(v):.4e}" for v in eig[:3]])
case = t.KernelTests("test_K3_kernel_equals_the_node_sheaf_kernel"); case.setUp()
case.test_K3_kernel_equals_the_node_sheaf_kernel()
print("K3 (dim ker L_hat, dim ker Delta_0, next eig L_hat, next eig Delta_0):", case.numbers)
print("F1:", t.fixed_point_errors(fixture))
print("U1:", t.unroll_vs_implicit(fixture))
