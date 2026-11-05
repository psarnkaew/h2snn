"""D4 mutation check: inject each required defect (monkeypatch, no file edit) and run the
whole `tests.test_v2_model` module against it. A mutation no test catches is vacuous.

    python <this>        (run from the H2S-NN repo root, h2snn env)
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from src.models import map_operators, sheaf_maps, vertex_path_laplacian  # noqa: E402

PTO = map_operators.PathTransportOperator
PCT = vertex_path_laplacian.PathCellTerm
VPL = vertex_path_laplacian.VertexPathSheafLaplacian


@contextmanager
def patched(owner, name, replacement):
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


# 1. Wrong transport order: T_i = Q_i ... Q_1 instead of Q_1 ... Q_i (adjoint kept consistent).
def reversed_apply(self, x):
    values = x[self.face_node_index]
    for j in range(1, self.length + 1):
        values = torch.cat([values[:, :j], self._step(j - 1, values[:, j:])], dim=1)
    return values


def reversed_apply_T(self, w, num_nodes):
    values = w
    for j in range(self.length, 0, -1):
        values = torch.cat([values[:, :j], self._step_T(j - 1, values[:, j:])], dim=1)
    output = w.new_zeros((num_nodes, w.size(-1)))
    output.index_add_(0, self.face_node_index.reshape(-1), values.reshape(-1, w.size(-1)))
    return output


# 2. Missing 1/|f| in z_f.
def sum_cell_values(self, x):
    y = self.incidence.apply(x)
    return y, y.sum(dim=1)


# 3. Normalization with a D that is not the operator's own (edge part only, path part omitted).
def edge_only_degree(self):
    return self.edge_degree()


# 4. w_f not applied in D.
def unweighted_diagonal(self, num_nodes):
    per_incidence = (1.0 - 1.0 / self.cell_size) * self.incidence.squared_scales()
    output = per_incidence.new_zeros(num_nodes)
    output.index_add_(0, self.incidence.face_node_index.reshape(-1), per_incidence.reshape(-1))
    return output


# 5. Path term applied to endpoints only (intermediate positions dropped).
def endpoints_only_apply(self, x, num_nodes):
    y = self.incidence.apply(x)
    keep = torch.zeros(y.size(1), dtype=y.dtype, device=y.device)
    keep[0] = keep[-1] = 1.0
    z = y[:, [0, -1]].mean(dim=1)
    residual = (y - z.unsqueeze(1)) * keep.view(1, -1, 1) * self.weights.view(-1, 1, 1)
    return self.incidence.apply_T(residual, num_nodes)


# 6. Different scales on the two sides under map_scale="per_edge".
_original_per_edge = sheaf_maps.FactorizedNodeToEdgeSheaf._forward_per_edge


def split_scale_per_edge(self, *args):
    out = _original_per_edge(self, *args)
    tgt = out.maps.tgt
    tgt_copy = map_operators.FactorizedMapOperator(tgt.scale * 1.001, tgt.reflectors,
                                                  tgt.rotations, tgt.group_ids)
    return sheaf_maps.FactorizedNodeToEdgeSheafOutput(
        maps=map_operators.EdgeMapPair(src=out.maps.src, tgt=tgt_copy))


MUTATIONS = [
    ("1 wrong transport order", [(PTO, "apply", reversed_apply), (PTO, "apply_T", reversed_apply_T)]),
    ("2 z_f without 1/|f|", [(PCT, "cell_values", sum_cell_values)]),
    ("3 D not operator-consistent (edge part only)", [(VPL, "degree", edge_only_degree)]),
    ("4 w_f not applied in D", [(PCT, "diagonal", unweighted_diagonal)]),
    ("5 path term on endpoints only", [(PCT, "apply", endpoints_only_apply)]),
    ("6 per_edge: different scales on the two sides",
     [(sheaf_maps.FactorizedNodeToEdgeSheaf, "_forward_per_edge", split_scale_per_edge)]),
]


def failing_tests() -> tuple[int, list[str]]:
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_v2_model")
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    names = sorted({test.id().split(".", 2)[-1].split(" ")[0]
                    for test, _ in result.failures + result.errors})
    return result.testsRun, names


def main() -> None:
    run, failed = failing_tests()
    print(f"unmutated: ran {run}, failing {len(failed)}: {failed}")
    vacuous = 0
    for label, patches in MUTATIONS:
        managers = [patched(owner, name, fn) for owner, name, fn in patches]
        for manager in managers:
            manager.__enter__()
        try:
            run, failed = failing_tests()
        finally:
            for manager in reversed(managers):
                manager.__exit__(None, None, None)
        if not failed:
            vacuous += 1
        print(f"[{label}] failing {len(failed)}:")
        for name in failed:
            print(f"    {name}")
    print(f"mutations={len(MUTATIONS)} vacuous={vacuous}")


if __name__ == "__main__":
    main()
