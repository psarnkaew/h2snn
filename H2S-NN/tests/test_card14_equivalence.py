"""Card 14 (a): every rewritten function gives the old function's output on one fixed input.

The references are the 66e9349 loops kept in `tests/reference/card14_reference.py`:
* `build_scope_batch` (vectorised over targets) vs the per-target loop, from one WISP draw merged
  per anchor on a random heterogeneous graph (self-loops, a self-inverse relation, two targets
  sharing cells' nodes);
* `sample_wisp_walks_chunked` (one call) vs the per-chunk loop;
* `build_graph_wisp_path_groups` (one call per batch) vs the per-graph loop.
Each runs on the CPU and, when CUDA is available, on the GPU (compared with the CPU reference).
"""

from __future__ import annotations

import unittest
from dataclasses import fields

import torch

from src.data.scopes import build_scope_batch
from src.data.structures import FaceGroup, PreparedH2SData
from src.data.wisp_sampler import (build_action_graph, build_wisp_anchor_cells,
                                   sample_wisp_walks_chunked)
from src.data.wisp_tasks import build_graph_wisp_path_groups
from tests.reference.card14_reference import (old_build_graph_wisp_path_groups,
                                              old_build_scope_batch, old_cells_per_target,
                                              old_sample_wisp_walks_chunked)

SELF_INVERSE = (2,)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def random_graph(num_nodes: int = 60, num_edges: int = 220, seed: int = 3) -> PreparedH2SData:
    g = torch.Generator().manual_seed(seed)
    node_type = torch.randint(0, 3, (num_nodes,), generator=g)
    node_type[:20] = 0
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst[:6] = src[:6]  # self-loops
    edge_type = torch.randint(0, 3, (num_edges,), generator=g)
    edge_type[:6] = 2
    targets = torch.arange(20)
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    y[:20] = torch.randint(0, 3, (20,), generator=g)
    train = torch.zeros(20, dtype=torch.bool)
    train[:12] = True
    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    node_mask[:12] = True
    return PreparedH2SData(
        x=torch.randn(num_nodes, 5, generator=g), y=y, node_type=node_type,
        node_type_names=("a", "b", "c"), node_slices={"a": (0, 20)},
        feature_dims={"a": 5, "b": 5, "c": 5}, edge_index=torch.stack([src, dst]),
        edge_type=edge_type, relation_names=("r0", "r1", "r2"),
        relation_tuples=(("a", "r0", "b"), ("a", "r1", "c"), ("b", "r2", "b")),
        train_mask=node_mask, val_mask=~node_mask & (y >= 0), test_mask=~node_mask & (y >= 0),
        target_node_type="a", target_indices=targets, target_y=y[:20],
        target_train_mask=train, target_val_mask=~train, target_test_mask=~train,
        num_classes=3, is_multilabel=False, face_groups=(), metrics=torch.zeros(0))


class _Same(unittest.TestCase):
    def assert_tensor(self, name: str, new: torch.Tensor | None, old: torch.Tensor | None) -> None:
        if old is None or new is None:
            self.assertIs(new, old, name)
            return
        self.assertEqual((new.dtype, tuple(new.shape)), (old.dtype, tuple(old.shape)), name)
        self.assertTrue(torch.equal(new.cpu(), old.cpu()), name)

    def assert_groups(self, new: tuple[FaceGroup, ...], old: tuple[FaceGroup, ...]) -> None:
        self.assertEqual(len(new), len(old))
        for k, (a, b) in enumerate(zip(new, old)):
            for f in fields(FaceGroup):
                va, vb = getattr(a, f.name), getattr(b, f.name)
                if isinstance(vb, torch.Tensor) or vb is None:
                    self.assert_tensor(f"group {k}.{f.name}", va, vb)
                else:
                    self.assertEqual(va, vb, f"group {k}.{f.name}")


class ScopeBatchEquivalence(_Same):
    def test_scope_batch_equals_the_per_target_loop(self) -> None:
        graph = random_graph()
        targets = torch.tensor([7, 2, 15, 0, 11, 19, 4])
        cells = build_wisp_anchor_cells(graph.edge_index, graph.edge_type, graph.node_type,
                                        SELF_INVERSE, torch.unique(targets), 6, 4, seed=5,
                                        chunk_size=3)
        old = old_build_scope_batch(graph, targets, old_cells_per_target(cells, targets))
        self.assertEqual(len(old.cell_scope), 3)  # lengths 2, 3, 4 all present
        for device in DEVICES:
            with self.subTest(device=device):
                g = graph.to(device)
                cells_d = build_wisp_anchor_cells(g.edge_index, g.edge_type, g.node_type,
                                                  SELF_INVERSE, torch.unique(targets.to(device)),
                                                  6, 4, seed=5, chunk_size=3)
                for name, new in (("anchor cells", build_scope_batch(g, targets, cells_d)),
                                  ("per-target lists", build_scope_batch(
                                      g, targets, old_cells_per_target(cells_d, targets)))):
                    with self.subTest(input=name):
                        self.assertEqual(new.num_scopes, old.num_scopes)
                        for f in ("targets", "target_rows", "global_node_id", "global_fact_id",
                                  "node_scope", "fact_scope"):
                            self.assert_tensor(f, getattr(new, f), getattr(old, f))
                        for a, b in zip(new.cell_scope, old.cell_scope, strict=True):
                            self.assert_tensor("cell_scope", a, b)
                        for f in fields(PreparedH2SData):
                            va, vb = getattr(new.data, f.name), getattr(old.data, f.name)
                            if f.name == "face_groups":
                                self.assert_groups(va, vb)
                            elif f.name == "scope_index":
                                for x, y in zip(va[:3], vb[:3]):
                                    if isinstance(y, tuple):
                                        for p, q in zip(x, y, strict=True):
                                            self.assert_tensor("scope_index.cell", p, q)
                                    else:
                                        self.assert_tensor("scope_index", x, y)
                                self.assertEqual(va.num_scopes, vb.num_scopes)
                            elif isinstance(vb, torch.Tensor):
                                self.assert_tensor(f"data.{f.name}", va, vb)


class DrawEquivalence(_Same):
    def test_chunked_draw_equals_the_chunk_loop(self) -> None:
        graph = random_graph()
        anchors = torch.tensor([3, 9, 1, 17, 0, 12, 5, 8, 19, 14, 2, 6, 10, 4, 18, 11, 7])
        action = build_action_graph(graph.edge_index, graph.edge_type, graph.num_nodes,
                                    SELF_INVERSE)
        old = old_sample_wisp_walks_chunked(action, graph.node_type, anchors, 5, 4, 21, 4)
        self.assertGreater(old.nodes.size(0), 0)
        for device in DEVICES:
            with self.subTest(device=device):
                action_d = build_action_graph(graph.edge_index.to(device),
                                              graph.edge_type.to(device), graph.num_nodes,
                                              SELF_INVERSE)
                new = sample_wisp_walks_chunked(action_d, graph.node_type.to(device),
                                                anchors.to(device), 5, 4, 21, 4)
                for f in fields(new):
                    self.assert_tensor(f.name, getattr(new, f.name), getattr(old, f.name))

    def test_graph_draw_equals_the_graph_loop(self) -> None:
        from torch_geometric.data import Batch, Data

        graphs = []
        for k, n in enumerate((7, 11, 5)):
            g = torch.Generator().manual_seed(k)
            ring = torch.arange(n)
            src = torch.cat([ring, (ring + 2) % n])
            dst = torch.cat([(ring + 1) % n, ring])
            graphs.append(Data(x=torch.ones(n, 1), edge_index=torch.stack([src, dst]),
                               edge_type=torch.randint(0, 2, (2 * n,), generator=g),
                               node_type=torch.randint(0, 3, (n,), generator=g),
                               y=torch.tensor([0]), graph_id=torch.tensor([40 + 3 * k])))
        batch = Batch.from_data_list(graphs)
        args = (batch.edge_index, batch.edge_type, batch.node_type, batch.batch, batch.graph_id,
                (0, 1), 4, 3, 9)
        old = old_build_graph_wisp_path_groups(*args)
        self.assertGreater(len(old), 0)
        for device in DEVICES:
            with self.subTest(device=device):
                new = build_graph_wisp_path_groups(*(a.to(device) if isinstance(a, torch.Tensor)
                                                     else a for a in args))
                self.assert_groups(new, old)


if __name__ == "__main__":
    unittest.main()
