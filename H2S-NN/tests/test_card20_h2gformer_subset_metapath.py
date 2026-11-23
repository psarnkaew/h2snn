"""Card 20: H2G-former's B1a validation subset loader sees the MetaPath2Vec channel (H2GB mini-batch path).

Card 17's cache fixture (ogbn-mag, a MAG-typed synthetic graph, H2GB's `preprocess_Metapath` replaced by a stub that
writes a known embedding) runs the real `_h2gformer_h2gb.fit` with a plan of one training step and one epoch, so the
B1a subset is evaluated once. Before card 20 that evaluation fails with `mat1 and mat2 shapes cannot be multiplied`
(the subset view has no `pestat_Hetero_Metapath`). A subset batch must carry `pestat_Hetero_Metapath` on every node
type a training batch carries it on.

`C20_LOSS_OUT=<file>`: the first training-step loss is written there (training identity against the base commit).
The real data root is unreachable.
"""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

PE = "pestat_Hetero_Metapath"


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


class SubsetLoaderMetapathTests(unittest.TestCase):
    def test_subset_evaluation_has_the_metapath_channel(self) -> None:
        import src.baselines.minibatch as minibatch
        from src.baselines.adapters import _h2gformer_h2gb as h2gf
        from src.baselines.adapters._h2gb_network import DatasetShim
        from src.baselines.adapters._h2gformer_encoders import H2GB_STACKS, METAPATH
        from src.baselines.runtime import pushd_sys_path
        from tests.test_baselines_h2gb_classa import _mag_typed_graph

        self.assertIn(METAPATH, H2GB_STACKS["ogbn-mag"])
        with pushd_sys_path("h2gb"):
            import H2GB.loader.encoding_generator as generator  # type: ignore[import-not-found]

        def stub(pe_dir, dataset):  # H2GB's preprocess_Metapath: a known embedding, `metapath_<dim>.pt` under pe_dir
            from H2GB.graphgym.config import cfg  # type: ignore[import-not-found]

            data = dataset[0]
            start, end, offset = {}, {}, 0
            for t in data.node_types:
                start[t], end[t] = offset, offset + int(data[t].num_nodes)
                offset = end[t]
            dim = int(cfg.posenc_Hetero_Metapath.dim_pe)
            emb = torch.nn.Embedding(offset, dim)
            with torch.no_grad():
                emb.weight.copy_(torch.arange(offset * dim, dtype=torch.float32).view(offset, dim) / (offset * dim))
            Path(pe_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"model": emb, "start": start, "end": end},
                       Path(pe_dir) / f"{generator.METAPATH_PT_NAME}_{dim}.pt")

        class Params(dict):
            report = None
            source = "test"

        plan = minibatch.MinibatchPlan(dataset="ogbn-mag", batch_size=8, epoch_targets=8, steps_per_epoch=1,
                                       epochs=1, eval_period=1, patience=None)
        losses: list[float] = []
        original_ce = torch.nn.functional.cross_entropy

        def recording_ce(*a, **k):
            loss = original_ce(*a, **k)
            losses.append(float(loss.detach()))
            return loss

        seen: dict = {}
        original_train = minibatch.train_minibatch_node_classification

        def train(**kwargs):
            seen.update(kwargs)
            return original_train(**kwargs)

        error = None
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(generator, "preprocess_Metapath", stub), \
                mock.patch("src.data.h2gb_nc.load_h2gb_dataset",
                           lambda *a, **k: DatasetShim(_mag_typed_graph(), "ogbn-mag")), \
                mock.patch("src.baselines.bundles.bundle_data_root", lambda bundle: Path(tmp)), \
                mock.patch.object(minibatch, "minibatch_plan", lambda *a, **k: plan), \
                mock.patch.object(minibatch, "h2gb_sample_sizes",
                                  lambda method, dataset, types_: {t: [4, 4] for t in types_}), \
                mock.patch.object(minibatch, "train_minibatch_node_classification", train), \
                mock.patch("torch.nn.functional.cross_entropy", recording_ce):
            params = Params(hidden_dim=16, gt_layers=2, attn_heads=2, dropout=0.0, metapath_cache=tmp)
            bundle = types.SimpleNamespace(dataset="ogbn-mag", metadata={"seed": None})
            try:
                h2gf.fit(bundle, params, seed=0, device=torch.device("cpu"))
            except RuntimeError as exc:
                error = exc
            finally:
                out = os.environ.get("C20_LOSS_OUT")
                if out:
                    Path(out).write_text(repr(losses[0]) if losses else "none", encoding="utf-8")
            if error is not None:
                raise error
            self.assertTrue(losses)
            train_batch = next(iter(seen["train_loader"]))
            subset_batch = next(iter(seen["val_subset_loader"]))
        train_types = {t for t in train_batch.node_types if PE in train_batch[t]}
        self.assertTrue(train_types)
        missing = sorted(t for t in train_types if PE not in subset_batch[t])
        self.assertEqual(missing, [], f"subset batch lacks {PE} on {missing}")


if __name__ == "__main__":
    unittest.main()
