"""Card 1d H5 (author decision 1a): HINormer on a graph with nodes of in-degree 0.

HINormer builds `adjM + adjM.T` with self-loops removed (run.py:95-96), so a node without a non-self edge has in-degree
0 (HGB Freebase: 45 book nodes). DGL's `GraphConv` refuses such a graph unless `allow_zero_in_degree` is set; the
adapter sets it on every `GraphConv` of `model.GCNLayers` at both places a HINormer model is built.
(i) A graph with an isolated target node: `fit_task` (HGB path) and `build_model` (H2GB path) give finite outputs.
    Before the fix both raise DGL's 0-in-degree error. Card mutation: drop the flag -> (i) fails.
(ii) A graph without isolated nodes: outputs with and without the flag are bit-identical (same seed).
Synthetic graphs only; the real data root is unreachable.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from src.baselines.protocol import NODE_CLASSIFICATION, BaselineParams
from tests.partMB_fixtures import restore_real_data, unreachable_real_data

BOOKS, FILMS, NUM_CLASSES = 8, 4, 3
SMALL = {"hidden_dim": 8, "num_layers": 1, "num_gnns": 2, "num_heads": 2, "dropout": 0.0, "len_seq": 4}


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def two_type_graph(isolated: bool) -> SimpleNamespace:
    """Flattened graph: books 0-7 (target, 4 attributes), films 8-11 (2 attributes). Book i -> film i % 4, stored one
    way, plus book-book pairs (0,1), (2,3); with `isolated`, book 7 has no edge at all."""
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(BOOKS + FILMS, 4)
    x[:BOOKS] = torch.randn(BOOKS, 4, generator=g)
    x[BOOKS:, :2] = torch.randn(FILMS, 2, generator=g)
    books = range(BOOKS - 1) if isolated else range(BOOKS)
    edges = [(b, BOOKS + b % FILMS) for b in books] + [(0, 1), (2, 3)]
    masks = {name: torch.zeros(BOOKS, dtype=torch.bool) for name in ("train", "val", "test")}
    for b in range(BOOKS):
        masks["train" if b < 4 else "val" if b < 6 else "test"][b] = True
    return SimpleNamespace(
        x=x, node_type=torch.tensor([0] * BOOKS + [1] * FILMS), node_type_names=("book", "film"),
        node_slices={"book": (0, BOOKS), "film": (BOOKS, BOOKS + FILMS)}, feature_dims={"book": 4, "film": 2},
        edge_index=torch.tensor(edges).t(), target_node_type="book",
        target_y=torch.arange(BOOKS) % NUM_CLASSES, target_indices=torch.arange(BOOKS),
        target_train_mask=masks["train"], target_val_mask=masks["val"], target_test_mask=masks["test"])


def graph_inputs(data: SimpleNamespace):
    """What `_fit_h2gb` builds before the model: the view, its graph (self-loops removed) and the forward inputs."""
    import dgl

    from src.baselines.adapters import hinormer
    from src.baselines.bundles import NodeBundle

    bundle = NodeBundle(dataset="freebase", data=data, num_classes=NUM_CLASSES, is_multilabel=False)
    view = bundle.homogeneous(feats_type=0)
    hinormer.restore_legacy_message_functions()
    source, destination = view.edge_index_without_self_loops
    graph = dgl.graph((source, destination), num_nodes=view.num_nodes)
    node_seq = hinormer.build_node_sequences_for(graph, data.target_indices, SMALL["len_seq"])
    inputs = ([f.to_dense() if f.is_sparse else f for f in view.features_list], node_seq,
              torch.eye(len(view.node_type_names)), view.node_type, True)
    return bundle, view, graph, inputs


class HINormerIsolatedNodeTests(unittest.TestCase):
    def test_isolated_node_gives_finite_outputs_at_both_build_sites(self) -> None:
        from src.baselines.adapters import hinormer

        data = two_type_graph(isolated=True)
        bundle, view, graph, inputs = graph_inputs(data)
        self.assertEqual(int((graph.in_degrees() == 0).sum()), 1)

        # HGB path: the model built inside fit_task (hinormer.py, `HINormer(...)` in `fit_task`)
        record = hinormer.HINormerAdapter().fit_task(
            bundle, BaselineParams({**SMALL, "feats_type": 0, "epochs": 1, "early_stopping": 1}),
            task=NODE_CLASSIFICATION, seed=0, device=torch.device("cpu"))
        self.assertTrue(math.isfinite(float(record["test_micro_f1"])))

        # H2GB path: `build_model`
        torch.manual_seed(0)
        model = hinormer.build_model(graph, NUM_CLASSES, view.in_dims, len(view.node_type_names),
                                     BaselineParams(SMALL))
        model.eval()
        with torch.no_grad():
            out = model(*inputs)
        self.assertEqual(tuple(out.shape), (BOOKS, NUM_CLASSES))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_flag_leaves_outputs_unchanged_without_isolated_nodes(self) -> None:
        from src.baselines.adapters import hinormer

        _, view, graph, inputs = graph_inputs(two_type_graph(isolated=False))
        self.assertEqual(int((graph.in_degrees() == 0).sum()), 0)
        models = []
        for flag in (True, False):
            torch.manual_seed(0)
            model = hinormer.build_model(graph, NUM_CLASSES, view.in_dims, len(view.node_type_names),
                                         BaselineParams(SMALL))
            for layer in model.GCNLayers:
                layer.set_allow_zero_in_degree(flag)
            self.assertTrue(all(layer._allow_zero_in_degree is flag for layer in model.GCNLayers))
            models.append(model.eval())
        for (name, a), (_, b) in zip(models[0].state_dict().items(), models[1].state_dict().items()):
            self.assertTrue(torch.equal(a, b), name)
        with torch.no_grad():
            with_flag, without_flag = models[0](*inputs), models[1](*inputs)
        self.assertTrue(torch.equal(with_flag, without_flag))


if __name__ == "__main__":
    unittest.main()
