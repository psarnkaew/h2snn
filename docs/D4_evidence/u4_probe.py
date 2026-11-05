"""U4 self-review numbers (run from H2S-NN/): T8 fixture margins, T6 magnitudes, gate init."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from src.models.h2snn_v2 import H2SNNv2  # noqa: E402
from src.models.h2snn_model import ModelDims  # noqa: E402
from src.models.vertex_path_laplacian import VertexPathSheafLaplacian  # noqa: E402
from tests import test_v2_model as t  # noqa: E402
from tests.test_factorized_maps import _ShapeRecorder, _random_pap_data  # noqa: E402


def main() -> None:
    data = _random_pap_data(num_papers=100, num_authors=50, max_faces=270)
    d = 32
    M = data.num_edges
    F = sum(g.num_faces for g in data.face_groups)
    limit = min(M, F) * d * d // 4
    model = t._v2_model(data, d=d, face_mlp_hidden_dim=128, edge_mlp_hidden_dim=128)
    recorder = _ShapeRecorder()
    with recorder:
        out = model(data)
        recorder.phase = "backward"
        out.logits.square().sum().backward()
    largest = max(recorder.shapes, key=lambda item: math.prod(item[1]))
    print(f"T8 fixture: M={M} F={F} groups={[g.num_faces for g in data.face_groups]} "
          f"numel_limit={limit}")
    print(f"T8 v2 largest tensor: {largest} = {math.prod(largest[1])} elements "
          f"({math.prod(largest[1]) / limit:.2f} of the limit); "
          f"largest parameter = {max(p.numel() for p in model.parameters())}")
    print(f"gate init on this fixture: mean={float(out.path_gate_mean):.4f} "
          f"max={float(out.path_gate_max):.4f}")

    edge_index, edge_type, groups = t._a1_p_a2()
    dims = ModelDims(in_channels=5, num_node_types=2, num_relations=2, num_classes=2,
                     face_lengths=(2,))
    torch.manual_seed(0)
    v2 = H2SNNv2(t._v2_config(), dims).double()
    h0 = torch.randn(4, 8, dtype=torch.float64)
    op = v2.build_operator(h0, edge_index, edge_type, groups, 4)
    for gamma in (0.0, 0.3, 2.0):
        lap = VertexPathSheafLaplacian(op.maps, edge_index, 4, op.laplacian.path_terms,
                                       torch.tensor(gamma, dtype=torch.float64), checkpoint=False)
        jac = torch.autograd.functional.jacobian(
            lambda x0: x0 - op.eta * lap.apply_normalized(x0), h0.clone())
        print(f"T6 gamma={gamma}: max|dx1(A1)/dx0(A2)|={float(jac[0, :, 2].abs().max()):.3e} "
              f"max|dx1(A1)/dx0(P)|={float(jac[0, :, 1].abs().max()):.4e}")


if __name__ == "__main__":
    main()
