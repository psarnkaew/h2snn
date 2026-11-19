"""Shared fixtures of the map tests (data graphs, holonomy helpers, the shape recorder) and the CUDA
integration check of the node->edge builder.

Spec 1.25 Z2 replaced the Householder family `F = c R_{psi,side} H_k(u)` (task card D1) by TD-1 maps
`F_src = c I`, `F_tgt = c R_r exp(U_r C_e U_r^T)`; the D1 operator tests (T1-T5, T9, T10 of card D1)
were removed with it, and the TD-1 tests are in `tests/test_td1_maps.py` (model change card M3).
"""

from __future__ import annotations

import math
import unittest

try:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.configs.config import TrainConfig
from src.data.structures import FaceGroup, PreparedH2SData
from src.models.h2snn_model import build_model
from src.models.map_operators import EdgeMapPair
from src.models.sheaf_maps import TD1NodeToEdgeSheaf

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


def _module_maps(data: PreparedH2SData, d: int = 8, rank: int = 4, seed: int = 0,
                 dtype: torch.dtype = torch.float64, angle: float = 0.3):
    """A TD-1 builder (spec 1.25 Z2), node states h, and its maps on `data`; the skew rows of the
    edge MLP's last layer are drawn N(0, angle^2) (zero at init) so the instance rotations are not I."""
    torch.manual_seed(seed)
    num_relations = len(data.relation_names)
    module = TD1NodeToEdgeSheaf(
        d=d, d_r=4, num_relations=num_relations, hidden_dim=16, map_rank=rank,
        scale_eps=0.1, orthogonal_map="fasthpp",
    ).to(dtype)
    if angle:
        with torch.no_grad():
            module.edge_mlp[-1].weight[:module.num_skew].normal_(0.0, angle)
    h = torch.randn(data.num_nodes, d, dtype=dtype)
    relation_embeddings = torch.randn(num_relations + 1, 4, dtype=dtype)
    output = module(h, data.edge_index, data.edge_type, relation_embeddings)
    return module, h, relation_embeddings, output


def _config(**model: object) -> TrainConfig:
    base = {"d": 8, "d_r": 4, "classifier_hidden_dim": 16}
    base.update(model)
    return TrainConfig.from_dict({"device": "cpu", "checkpoint_metric": "acc", "model": base,
                                  "data": {"path_source": "schemas"}})  # schema fixtures (I2)


def _mv(matrices: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    return (matrices @ vectors.unsqueeze(-1)).squeeze(-1)


def _max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


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
        cuda_module = TD1NodeToEdgeSheaf(
            d=8, d_r=4, num_relations=2, hidden_dim=16, map_rank=4,
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
