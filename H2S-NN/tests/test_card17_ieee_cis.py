"""Card 17: the five H2GB IEEE-CIS-G cells (GCN, GAT, R-GCN, HAN, H2G-former) as published-configuration runs.

* The five cells are `published` (no longer `filled`), each file resolves: `published_params_path` exists, its params
  equal `yaml_published_values` of H2GB's yaml key by key (lines included), `model.loss_fun_weight == [1, 5]`.
* Known answer: the shared loop's `weighted_cross_entropy` with a weight list equals H2GB's multi-class form
  `F.nll_loss(F.log_softmax(logits, -1), y, weight=w)` and differs from the unweighted loss; `cross_entropy` is
  `F.cross_entropy` itself (bit-identical to before); no weight list is refused.
* MetaPath2Vec cache (step 3c): two H2G-former fits at seeds 0 and 1 on one MetaPath dataset (ogbn-mag, synthetic
  graph, H2GB's `preprocess_Metapath` replaced by a stub that counts calls) use one `pe_dir` `<cache>/<dataset>/dim<d>`
  (no seed) and train once, at seed 0.

Card mutation: drop the weight in `h2gb_criterion` -> `test_weighted_loss_known_answer` fails.
The real data root is unreachable.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
import yaml

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

CELLS = ("gcn", "gat", "rgcn", "han", "h2gformer")
DATASET = "ieee-cis-g"


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


class PublishedCellsTests(unittest.TestCase):
    def test_five_cells_published_and_resolve(self) -> None:
        from src.baselines.cell_status import cell_status
        from src.baselines.h2gb_published import H2GB_PUBLISHED_CELLS, yaml_key_lines, yaml_published_values
        from src.baselines.minibatch import h2gb_config_file
        from src.baselines.protocol import NODE_CLASSIFICATION
        from src.baselines.published import load_published_params, published_params_path

        for model in CELLS:
            with self.subTest(model=model):
                self.assertEqual(cell_status(model, NODE_CLASSIFICATION, DATASET), "published")
                self.assertIn((model, DATASET), H2GB_PUBLISHED_CELLS)
                path = published_params_path(model, NODE_CLASSIFICATION, DATASET)
                self.assertTrue(path.exists(), path)
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                expected = yaml_published_values(model, DATASET)
                self.assertEqual(payload["params"], expected)
                self.assertEqual(list(payload["params"]), list(expected))
                lines = yaml_key_lines(h2gb_config_file(model, DATASET))
                self.assertEqual(payload["lines"], {key: lines[key] for key in expected})
                self.assertIn(h2gb_config_file(model, DATASET).name, payload["source"])
                self.assertEqual(payload["params"]["model.loss_fun"], "weighted_cross_entropy")
                self.assertEqual(payload["params"]["model.loss_fun_weight"], [1, 5])
                self.assertEqual(load_published_params(model, NODE_CLASSIFICATION, DATASET).values, expected)


class WeightedLossTests(unittest.TestCase):
    def test_weighted_loss_known_answer(self) -> None:
        from src.baselines.h2gb_published import h2gb_criterion

        logits = torch.tensor([[2.0, -1.0], [0.5, 0.25], [-1.5, 1.0], [0.0, 3.0], [1.0, 1.0]])
        y = torch.tensor([0, 1, 1, 0, 0])
        weight = torch.tensor([1.0, 5.0])
        criterion = h2gb_criterion({"model.loss_fun": "weighted_cross_entropy", "model.loss_fun_weight": [1, 5]})
        h2gb = F.nll_loss(F.log_softmax(logits, -1), y, weight=weight)
        self.assertTrue(torch.allclose(criterion(logits, y), h2gb, atol=1e-7, rtol=0))
        self.assertFalse(torch.allclose(criterion(logits, y), F.cross_entropy(logits, y), atol=1e-4, rtol=0))
        # by hand: sum_i w_{y_i} * -log p_{y_i} / sum_i w_{y_i}
        nll = -F.log_softmax(logits, -1).gather(1, y[:, None]).squeeze(1)
        self.assertAlmostEqual(float(criterion(logits, y)), float((weight[y] * nll).sum() / weight[y].sum()), places=6)

    def test_cross_entropy_unchanged_and_no_weight_refused(self) -> None:
        from src.baselines.h2gb_published import h2gb_criterion

        self.assertIs(h2gb_criterion({"model.loss_fun": "cross_entropy"}), F.cross_entropy)
        for args in ({"model.loss_fun": "weighted_cross_entropy", "model.loss_fun_weight": None},
                     {"model.loss_fun": "weighted_cross_entropy"}, {"model.loss_fun": "focal"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                h2gb_criterion(args)


class MetapathCacheTests(unittest.TestCase):
    def test_second_seed_reuses_the_seed0_cache(self) -> None:
        import types

        from src.baselines.adapters import _h2gformer_h2gb as h2gf
        from src.baselines.adapters._h2gb_network import DatasetShim
        from src.baselines.adapters._h2gformer_encoders import METAPATH, H2GB_STACKS, metapath_dim_pe
        from src.baselines.runtime import pushd_sys_path
        from tests.test_baselines_h2gb_classa import _mag_typed_graph

        self.assertIn(METAPATH, H2GB_STACKS["ogbn-mag"])
        with pushd_sys_path("h2gb"):
            import H2GB.loader.encoding_generator as generator  # type: ignore[import-not-found]
        calls: list[tuple[str, int]] = []

        def stub(pe_dir, dataset):  # H2GB's preprocess_Metapath: writes `metapath_<dim>.pt` under pe_dir
            from H2GB.graphgym.config import cfg  # type: ignore[import-not-found]

            calls.append((str(pe_dir), torch.initial_seed()))
            data = dataset[0]
            start, end, offset = {}, {}, 0
            for t in data.node_types:
                start[t], end[t] = offset, offset + int(data[t].num_nodes)
                offset = end[t]
            dim = int(cfg.posenc_Hetero_Metapath.dim_pe)
            Path(pe_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"model": torch.nn.Embedding(offset, dim), "start": start, "end": end},
                       Path(pe_dir) / f"{generator.METAPATH_PT_NAME}_{dim}.pt")

        class Params(dict):
            report = None
            source = "test"

        plan = types.SimpleNamespace(batch_size=8, steps_per_epoch=1)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(generator, "preprocess_Metapath", stub), \
                mock.patch("src.data.h2gb_nc.load_h2gb_dataset",
                           lambda *a, **k: DatasetShim(_mag_typed_graph(), "ogbn-mag")), \
                mock.patch("src.baselines.bundles.bundle_data_root", lambda bundle: Path(tmp)), \
                mock.patch("src.baselines.minibatch.minibatch_plan", lambda *a, **k: plan), \
                mock.patch("src.baselines.minibatch.hgt_loader", lambda *a, **k: None), \
                mock.patch("src.baselines.minibatch.eval_loader", lambda *a, **k: None), \
                mock.patch("src.baselines.minibatch.val_subset_loader", lambda *a, **k: (None, torch.zeros(0))), \
                mock.patch("src.baselines.minibatch.train_minibatch_node_classification", lambda **k: {}), \
                mock.patch.object(h2gf, "share_metapath_with_subset_view", lambda *a, **k: None), \
                mock.patch.object(h2gf, "build_gtmodel", lambda *a, **k: torch.nn.Linear(1, 1)):
            params = Params(hidden_dim=16, metapath_cache=tmp)
            bundle = types.SimpleNamespace(dataset="ogbn-mag", metadata={"seed": None})
            for seed in (0, 1):
                h2gf.fit(bundle, params, seed=seed, device=torch.device("cpu"))
            expected = Path(tmp) / "ogbn-mag" / f"dim{metapath_dim_pe(16)}"
            self.assertEqual(calls, [(str(expected), 0)])
            self.assertTrue(expected.is_dir())
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "ogbn-mag").iterdir()), [expected.name])


if __name__ == "__main__":
    unittest.main()
