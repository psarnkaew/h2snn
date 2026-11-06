"""U1 evidence: T1 errors, per-path step counts (old L(L+1) vs new 2L), and the s^2 mutation.
Run from H2S-NN/: python ../docs/D7_evidence/u1_checks.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "H2S-NN"))
import torch

from src.models import vertex_path_laplacian as vpl
from src.models.map_operators import PathTransportOperator
from tests.test_paths_and_facts import OLPathTermTests, _chain_operators, _count_steps, _random_paths

case = OLPathTermTests()
for shared in (True, False):
    for dense in (False, True):
        errs = [case._check(shared, dense, s) for s in range(3)]
        print(f"T1 shared={shared} dense_family={dense}: max rel err {max(errs):.3e}")

for L in (2, 3, 4):
    g = torch.Generator().manual_seed(L)
    nodes, edges, ei = _random_paths(8, (L,), 3, g)
    maps = _chain_operators(ei.size(1), k=2, d=4, seed=L, dtype=torch.float64, shared=True)
    t = PathTransportOperator(maps.src, maps.tgt, nodes[0], edges[0], ei, shared_scale=True)
    c = _count_steps(t)
    x = torch.randn(8, 4, dtype=torch.float64)
    y = t.apply(x); t.apply_T(y, 8)
    old = c[0]; c[0] = 0
    vpl.PathCellTerm(t, torch.ones(3, dtype=torch.float64)).apply(x, 8)
    print(f"L={L}: old apply+apply_T steps/path {old}, new O(L) steps/path {c[0]}")

orig = vpl.PathCellTerm._apply_transport
def mutated(self, x, num_nodes):
    tr = self.incidence; nodes = tr.face_node_index
    z = tr.transport_sum(x) / self.cell_size
    residual = (x[nodes] - tr.adjoint_prefix(z)) * self.weights.view(-1, 1, 1)  # x_i, not s_i^2 x_i
    out = x.new_zeros((num_nodes, x.size(-1)))
    out.index_add_(0, nodes.reshape(-1), residual.reshape(-1, x.size(-1)))
    return out
vpl.PathCellTerm._apply_transport = mutated
for shared in (True, False):
    print(f"mutation x_i instead of s_i^2 x_i, shared={shared}: rel err {case._check(shared, False, 0):.3e}")
vpl.PathCellTerm._apply_transport = orig
