"""U1 (D1a) self-review: decisive numbers and two teeth checks. Run from H2S-NN/.

Prints orthogonality errors of path transports (per-edge vs per-side control), and runs two
targeted tests against monkeypatched defects (no file is edited):
  M-a  the shared-scale fast path silently falls back to the scaled maps;
  M-b  the per-edge module gives the target side its own copy of the scale (x 1.001).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from src.models import map_operators, sheaf_maps  # noqa: E402
from tests import test_v2_model as t  # noqa: E402


def orthogonality_numbers() -> None:
    for length in (2, 3, 4):
        for dtype in (torch.float64, torch.float32):
            for shared in (True, False):
                nodes, edges, edge_index = t._chain_paths(40, length)
                maps = t._chain_operators(edge_index.size(1), 2, 6, seed=length, dtype=dtype,
                                          shared=shared)
                op = map_operators.PathTransportOperator(maps.src, maps.tgt, nodes, edges,
                                                         edge_index, shared_scale=shared)
                err = t._orthogonality_error(t._materialize_transports(op, nodes.numel(), 6, dtype))
                print(f"L={length} {str(dtype):14s} shared={shared!s:5s} max|T^T T - I| = {err:.3e}")


def run(test_names: list[str]) -> tuple[int, int]:
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in test_names)
    result = unittest.TextTestRunner(stream=open("NUL", "w"), verbosity=0).run(suite)
    return result.testsRun, len(result.failures) + len(result.errors)


def main() -> None:
    orthogonality_numbers()
    fast = "tests.test_v2_model.PerEdgeScaleTests.test_T12b_shared_scale_path_uses_no_scale"
    share = ["tests.test_v2_model.PerEdgeScaleTests.test_T12a_both_sides_share_exactly_one_scale_per_edge",
             "tests.test_v2_model.PerEdgeVersionsOfD1Tests.test_T10_per_edge_storage_matches_the_bound_exactly"]
    print("unmutated:", run([fast] + share))

    original_step = map_operators.PathTransportOperator._step

    def scaled_step(self, j, y):
        return self._src[j].apply_inverse(self._tgt[j].apply(y))

    map_operators.PathTransportOperator._step = scaled_step
    print("M-a fast path uses scales  -> (run, failed):", run([fast]))
    map_operators.PathTransportOperator._step = original_step

    original_forward = sheaf_maps.FactorizedNodeToEdgeSheaf._forward_per_edge

    def split_scale(self, *args):
        out = original_forward(self, *args)
        tgt = out.maps.tgt
        copy = map_operators.FactorizedMapOperator(tgt.scale * 1.001, tgt.reflectors,
                                                   tgt.rotations, tgt.group_ids)
        return sheaf_maps.FactorizedNodeToEdgeSheafOutput(
            maps=map_operators.EdgeMapPair(src=out.maps.src, tgt=copy))

    sheaf_maps.FactorizedNodeToEdgeSheaf._forward_per_edge = split_scale
    print("M-b per-side copy of c_e    -> (run, failed):", run(share))
    sheaf_maps.FactorizedNodeToEdgeSheaf._forward_per_edge = original_forward


if __name__ == "__main__":
    main()
