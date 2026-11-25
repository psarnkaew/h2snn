"""Card 22: H2GB NC failures on v3b `2e3e160`. Synthetic graphs; the data, plan and split helpers are patched, the
adapters' training and evaluation code runs unchanged.

1. (A) Hetero²Net `_fit_h2gb`, GPU, `LOADER_WORKERS` 2 and 0: training epochs, the B1a subset evaluation and the
   validation/test passes run without a device error (the label channel `y_emb` and `y` reach the model's device).
2. (B) IEEE-CIS-G shape (target `transaction` with `x`, the other types featureless): `fit_homogeneous_on_hgt` with
   2 workers runs; on loaders built before the target's `x` is set, `hgb_homogeneous_batch` gives the new `x` for the
   target and None for the featureless types, and `batch_type_inputs` their one-hot rows.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


class _Params(dict):
    report = None
    source = "test"


def _plan(dataset: str):
    from src.baselines.minibatch import MinibatchPlan

    return MinibatchPlan(dataset, batch_size=4, epoch_targets=8, steps_per_epoch=2, epochs=2, eval_period=1)


def _spec(target: str):
    from src.data.h2gb_nc import H2GBNCSpec

    return H2GBNCSpec("toy", "Toy", "none", None, target, "multiclass", "test")


def _patches(graph, target: str, workers: int) -> list:
    import src.baselines.minibatch as mb
    import src.evaluation.validation_subset as vs
    from src.data.h2gb_nc import ALL_H2GB_NC_SPECS

    return [mock.patch.object(mb, "h2gb_graph", lambda bundle: graph),
            mock.patch.object(mb, "minibatch_plan", lambda dataset, batch_size: _plan(dataset)),
            mock.patch.object(mb, "LOADER_WORKERS", workers),
            mock.patch.object(vs, "validation_subset", lambda dataset, ids, seed: ids),
            mock.patch.dict(ALL_H2GB_NC_SPECS, {"toy": _spec(target)})]


def ieee_graph(seed: int = 0, with_target_x: bool = True):
    """`transaction` (target, 60, binary labels, masks), `card` (10) and `addr` (8) featureless; reverses stored."""
    from torch_geometric.data import HeteroData

    g = torch.Generator().manual_seed(seed)
    n = {"transaction": 60, "card": 10, "addr": 8}
    data = HeteroData()
    if with_target_x:
        data["transaction"].x = torch.randn(n["transaction"], 5, generator=g)
    else:
        data["transaction"].num_nodes = n["transaction"]
    data["card"].num_nodes, data["addr"].num_nodes = n["card"], n["addr"]
    data["transaction"].y = torch.arange(n["transaction"]) % 2
    order = torch.randperm(n["transaction"], generator=g)
    for name, part in (("train", order[:30]), ("val", order[30:45]), ("test", order[45:])):
        mask = torch.zeros(n["transaction"], dtype=torch.bool)
        mask[part] = True
        data["transaction"][f"{name}_mask"] = mask
    for other in ("card", "addr"):
        index = torch.stack([torch.arange(n["transaction"]), torch.randint(0, n[other], (n["transaction"],),
                                                                           generator=g)])
        data["transaction", f"to_{other}", other].edge_index = index
        data[other, f"rev_to_{other}", "transaction"].edge_index = index.flip(0)
    return data


class Hetero2NetH2GBDeviceTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_1_fit_h2gb_on_gpu_with_workers(self) -> None:
        from src.baselines.adapters import hetero2net
        from tests.test_baselines_h2gb_classb import typed_graph

        for workers in (2, 0):
            with self.subTest(workers=workers):
                graph = typed_graph(width=6)
                for node_type in graph.node_types:
                    graph[node_type].x = graph[node_type].x.float()
                patches = _patches(graph, "paper", workers) + [
                    mock.patch.object(hetero2net, "bundle_data_root", lambda bundle: None)]
                for p in patches:
                    p.start()
                try:
                    record = hetero2net.Hetero2NetAdapter()._fit_h2gb(
                        SimpleNamespace(dataset="toy", metadata={"seed": 0}),
                        _Params(hidden=8, batch_size=4, p=0.7), seed=0, device=torch.device("cuda"))
                finally:
                    for p in reversed(patches):
                        p.stop()
                self.assertIn("val_subset_macro_f1", record)
                self.assertTrue(any(key.startswith("test_") for key in record))


class FeaturelessTypesTests(unittest.TestCase):
    def test_2_ieee_shape_with_workers(self) -> None:
        import src.baselines.minibatch as mb

        graph = ieee_graph()
        patches = _patches(graph, "transaction", 2) + [
            mock.patch.object(mb, "h2gb_sample_sizes", lambda method, dataset, types: {t: [4, 4] for t in types})]
        for p in patches:
            p.start()
        try:
            def build_model(dims, num_classes):
                return torch.nn.ModuleList([torch.nn.Linear(d, num_classes) for d in dims])

            def forward(model, inputs, edge_index, num_nodes):
                return torch.cat([layer(x) for layer, x in zip(model, inputs)])

            torch.manual_seed(0)
            record = mb.fit_homogeneous_on_hgt(SimpleNamespace(dataset="toy", metadata={"seed": 0}),
                                               _Params(batch_size=4), fanout_method="gcn", seed=0,
                                               device=torch.device("cpu"), build_model=build_model, forward=forward)
        finally:
            for p in reversed(patches):
                p.stop()
        self.assertIn("val_subset_macro_f1", record)

        # The card's case: the target's `x` set after the 2-worker loaders were built.
        data = ieee_graph(with_target_x=False)
        node_types = tuple(data.node_types)
        counts = {t: int(data[t].num_nodes) for t in node_types}
        fanout = {t: [4, 4] for t in node_types}
        train = mb.hgt_loader(data, fanout, "transaction", "train", 4, shuffle=True, steps=2, num_workers=2)
        val = mb.eval_loader(data, fanout, "transaction", "val", 4, num_workers=2)
        late = torch.arange(60, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
        data["transaction"].x = late
        for label, loader in (("train", train), ("val", val)):
            batch = next(iter(loader))
            features, _, _ = mb.hgb_homogeneous_batch(batch, node_types)
            store = batch["transaction"]
            self.assertTrue(torch.equal(features[node_types.index("transaction")], late[store.n_id]), label)
            self.assertIsNone(features[node_types.index("card")], label)
            inputs = mb.batch_type_inputs(batch, node_types, counts)
            card = inputs[node_types.index("card")]
            self.assertEqual(card.size(1), counts["card"], label)
            self.assertTrue(torch.equal(card.argmax(1), batch["card"].n_id), label)


if __name__ == "__main__":
    unittest.main()
