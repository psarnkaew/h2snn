"""Part 2 E1 (slim mode): class-A adapters on H2GB's `hgt` batches, synthetic graph only.

* One training step on a sampled batch is finite and moves the parameters: GCN, GAT (HGB's models on HGB's homogeneous
  reduction of the batch), HAN (H2GB's `HeteroGNNModel` + `HANConv` from H2GB's HAN YAML), H2G-former (H2GB's
  `GTModel` from H2GB's H2Gformer YAML, label channel on).
* H2G-former's label channel reads training labels only: with the channel prepared by the adapter, permuting the
  validation and test labels changes no output of an evaluation batch.

Card mutation: the label channel reads all labels (`prepare_label_channel` keeps every label and marks every target
as training) -> `test_h2gformer_label_channel_reads_training_labels_only` fails.
The real data root is unreachable.
"""

from __future__ import annotations

import unittest

import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data
from tests.test_baselines_minibatch import full_fanout, synthetic_hetero


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


class _Params(dict):
    report = None
    source = "test"


def _first_batch(data, split: str = "train", batch_size: int = 8, depth: int = 2):
    from src.baselines.minibatch import hgt_loader

    count = int(data["a"][f"{split}_mask"].sum())
    loader = hgt_loader(data, full_fanout(data, depth), "a", split, batch_size, shuffle=False,
                        steps=-(-count // batch_size))
    return next(iter(loader))


def _one_step_moves(test, model, forward, labels):
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(forward(), labels)
    test.assertTrue(torch.isfinite(loss).item())
    loss.backward()
    optimizer.step()
    test.assertTrue(any(not torch.equal(b, p) for b, p in zip(before, model.parameters())), "no parameter moved")


class OneStepTests(unittest.TestCase):
    def _homogeneous(self, name: str) -> None:
        from src.baselines.minibatch import batch_type_inputs, h2gb_input_dims, hgb_homogeneous_batch
        from src.baselines.registry import get_adapter

        data = synthetic_hetero()
        adapter = get_adapter(name)
        node_types, counts, dims = h2gb_input_dims(data)
        torch.manual_seed(0)
        model = adapter.build_hgb_model(dims, 3, _Params(hidden_dim=8, num_layers=1, num_heads=2, dropout=0.0))
        batch = _first_batch(data)
        size = int(batch["a"].batch_size)

        def forward():
            inputs = batch_type_inputs(batch, node_types, counts)
            _, edge_index, offsets = hgb_homogeneous_batch(batch, node_types)
            out = adapter.hgb_forward(model, inputs, edge_index, sum(x.size(0) for x in inputs))
            return out[offsets["a"][0]:offsets["a"][0] + size]

        _one_step_moves(self, model, forward, batch["a"].y[:size])

    def test_gcn_step(self) -> None:
        self._homogeneous("gcn")

    def test_gat_step(self) -> None:
        self._homogeneous("gat")

    def test_han_step(self) -> None:
        from src.baselines.adapters._h2gb_network import build_hetero_gnn, configure_from_yaml, seed_predictions
        from src.baselines.adapters._pyg_compat import ensure_pyg_submodules
        from src.baselines.runtime import pushd_sys_path

        data = synthetic_hetero()
        batch = _first_batch(data)
        ensure_pyg_submodules()
        with pushd_sys_path("h2gb"):
            configure_from_yaml("han", "ogbn-mag", data, "a", torch.device("cpu"), {
                "gnn.dim_inner": 8, "gnn.layers_mp": 2, "gnn.attn_heads": 2, "gnn.dropout": 0.0,
                "train.sampler": "hgt"})
            torch.manual_seed(0)
            model = build_hetero_gnn(data, "synthetic", 3)
        size = int(batch["a"].batch_size)
        _one_step_moves(self, model, lambda: seed_predictions(model, batch.clone(), "a"), batch["a"].y[:size])


def _h2gformer(data):
    from src.baselines.adapters import _h2gformer_h2gb as h2gf
    from src.baselines.adapters._h2gformer_encoders import LABEL, RAW
    from src.baselines.adapters._h2gb_network import configure_from_yaml
    from src.baselines.adapters._pyg_compat import ensure_pyg_submodules
    from src.baselines.runtime import pushd_sys_path

    ensure_pyg_submodules()
    with pushd_sys_path("h2gb"):
        # IEEE-CIS's H2Gformer config: raw + label channels (H2GB_STACKS), no MetaPath2Vec
        configure_from_yaml("h2gformer", "ieee-cis-g", data, "a", torch.device("cpu"),
                            h2gf.overrides(_Params(hidden_dim=16, gt_layers=2, attn_heads=2, dropout=0.0), 3,
                                           (RAW, LABEL)))
        torch.manual_seed(0)
        return h2gf.build_gtmodel(data, "synthetic", 3)


class H2GformerTests(unittest.TestCase):
    def test_h2gformer_step(self) -> None:
        from src.baselines.adapters._h2gb_network import seed_predictions
        from src.baselines.adapters._h2gformer_h2gb import prepare_label_channel

        data = synthetic_hetero()
        labels = data["a"].y.clone()
        batch = _first_batch(data)
        prepare_label_channel(data, "a", labels)
        batch = _first_batch(data)
        model = _h2gformer(data)
        size = int(batch["a"].batch_size)

        def forward():
            b = batch.clone()
            b.split = "train"
            return seed_predictions(model, b, "a")

        _one_step_moves(self, model, forward, labels[batch["a"].n_id[:size]])

    def test_h2gformer_label_channel_reads_training_labels_only(self) -> None:
        from src.baselines.adapters._h2gb_network import seed_predictions
        from src.baselines.adapters._h2gformer_h2gb import prepare_label_channel
        from src.baselines.minibatch import eval_loader

        outputs = []
        for permute in (False, True):
            data = synthetic_hetero()
            labels = data["a"].y.clone()
            held = ~data["a"].train_mask
            if permute:  # change every validation/test label (to another class)
                labels[held] = (labels[held] + 1) % 3
            loaders = [eval_loader(data, full_fanout(data, 2), "a", split, 4) for split in ("val", "test")]
            prepare_label_channel(data, "a", labels)
            model = _h2gformer(data).eval()
            out = []
            with torch.no_grad():
                for loader in loaders:
                    for batch in loader:
                        batch.split = "val"
                        out.append(seed_predictions(model, batch, "a"))
            outputs.append(torch.cat(out))
        self.assertTrue(torch.equal(outputs[0], outputs[1]),
                        f"validation/test labels reach the output: max |diff| "
                        f"{(outputs[0] - outputs[1]).abs().max().item():.3e}")


def _mag_typed_graph(seed: int = 0):
    """ogbn-mag's types and the relations H2GB's ogbn-mag MetaPath2Vec schema walks (encoding_generator.py:263-274)."""
    from torch_geometric.data import HeteroData

    g = torch.Generator().manual_seed(seed)
    n = {"paper": 120, "author": 40, "institution": 6, "field_of_study": 10}
    data = HeteroData()
    for t, count in n.items():
        data[t].x = torch.randn(count, 4, generator=g)
    data["paper"].y = torch.arange(n["paper"]) % 3
    order = torch.randperm(n["paper"], generator=g)
    for name, part in (("train", order[:70]), ("val", order[70:95]), ("test", order[95:])):
        mask = torch.zeros(n["paper"], dtype=torch.bool)
        mask[part] = True
        data["paper"][f"{name}_mask"] = mask

    def rel(src, dst, count):
        return torch.unique(torch.stack([torch.randint(0, n[src], (count,), generator=g),
                                         torch.randint(0, n[dst], (count,), generator=g)]), dim=1)

    for (src, name, dst), count in ((("author", "writes", "paper"), 200), (("paper", "has_topic", "field_of_study"), 150),
                                    (("author", "affiliated_with", "institution"), 50), (("paper", "cites", "paper"), 150)):
        index = rel(src, dst, count)
        data[src, name, dst].edge_index = index
        data[dst, f"rev_{name}", src].edge_index = index.flip(0)
    return data


class H2GformerMagYearTests(unittest.TestCase):
    """Card 1c F6 (G-7): mag-year with ogbn-mag's MetaPath2Vec schema (H2GB's own preprocess_Metapath, run on a
    MAG-typed synthetic graph through the name shim), then one training step with the MetaPath2Vec channel."""

    def test_mag_year_metapath_schema_and_step(self) -> None:
        import tempfile

        from src.baselines.adapters import _h2gformer_h2gb as h2gf
        from src.baselines.adapters._h2gb_network import DatasetShim, configure_from_yaml, seed_predictions
        from src.baselines.adapters._h2gformer_encoders import LABEL, METAPATH, RAW, metapath_dim_pe
        from src.baselines.adapters._pyg_compat import ensure_pyg_submodules
        from src.baselines.runtime import pushd_sys_path

        self.assertEqual(h2gf.SCHEMA_NAME["mag-year"], "ogbn-mag")
        data = _mag_typed_graph()
        labels = data["paper"].y.clone()
        loader = None
        from src.baselines.minibatch import hgt_loader

        loader = hgt_loader(data, {t: [10 ** 6] * 2 for t in data.node_types}, "paper", "train", 8, shuffle=False,
                            steps=1)
        h2gf.prepare_label_channel(data, "paper", labels)
        params = _Params(hidden_dim=16, gt_layers=2, attn_heads=2, dropout=0.0)
        ensure_pyg_submodules()
        with tempfile.TemporaryDirectory() as tmp, pushd_sys_path("h2gb"):
            configure_from_yaml("h2gformer", "mag-year", data, "paper", torch.device("cpu"),
                                {**h2gf.overrides(params, 3, (RAW, LABEL, METAPATH)), "num_workers": 0})
            h2gf.attach_h2gb_metapath(DatasetShim(data, "mag-year"), data, __import__("pathlib").Path(tmp),
                                      h2gf.SCHEMA_NAME["mag-year"])
            for t in data.node_types:
                self.assertEqual(tuple(data[t]["pestat_Hetero_Metapath"].shape),
                                 (int(data[t].num_nodes), metapath_dim_pe(16)))
            torch.manual_seed(0)
            model = h2gf.build_gtmodel(data, "synthetic", 3)
        batch = next(iter(loader))
        size = int(batch["paper"].batch_size)

        def forward():
            b = batch.clone()
            b.split = "train"
            return seed_predictions(model, b, "paper")

        _one_step_moves(self, model, forward, labels[batch["paper"].n_id[:size]])


if __name__ == "__main__":
    unittest.main()
