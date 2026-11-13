"""Part P, P3 (`docs/H2S-NN_partP_card.md`): Freebase NC for H2S-NN and every NC baseline.

Real HGB Freebase files (data read only, when present):
- the loader's counts equal the M1 gate's (`docs/partM_evidence/m1_data_counts_output.txt:36-39`):
  nodes per type, facts before/after tying, 7,954 labelled books and their class counts;
- the 24/6/70 split sizes and disjointness; unlabelled books in no mask;
- the relations without a stored reverse (D7's untied fallback) equal a direct check of link.dat;
- the fixed-literature-schema row builds from a Freebase `full` configuration and draws every
  listed meta-path (BB as two hops) from the loader's graph (spec v1.20).
Tiny Freebase-format release (`tests.partP_freebase_fixture`):
- `scripts/schema_readout.py` runs end to end on a synthetic 1-epoch H2S-NN Freebase run;
- each NC adapter's `fit_task` runs one epoch (environment-blocked adapters skipped with the reason);
  SeHGNN, PSHGCN, H2SGNN keep a specific registry reason.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.baselines.protocol import NODE_CLASSIFICATION, BaselineParams
from src.data.data_layout import REPO_DATA_ROOT, hgb_nc_raw_dir, nc_root
from tests.partP_freebase_fixture import write_freebase_release

REAL = (nc_root(REPO_DATA_ROOT) / "freebase").exists()
M1_NODES = {"book": 40402, "film": 19427, "music": 82351, "sports": 1025, "people": 17641,
            "location": 9368, "organization": 2731, "business": 7153}
M1_CLASSES = [429, 3886, 1748, 185, 962, 359, 385]


def _full(root: str | Path | None = None, **top):
    from src.configs.config import TrainConfig

    data = {"dataset_name": "freebase", "wisp_max_length": 2, "wisp_walks_per_anchor": 7}
    if root is not None:
        data["root"] = str(root)
    return TrainConfig.from_dict({
        "device": "cpu", "seed": 0, **top, "data": data,
        "model": {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                  "classifier_hidden_dim": 16, "propagation": "anchored", "arm": "full"},
        "ablation": {"coupling": "symmetric"}})


@unittest.skipUnless(REAL, "HGB Freebase data not present")
class RealLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.data.dataset import _load_hetero_data, _prepare_target_labels
        from src.data.fact_layer import build_fact_layer
        from src.data.hetero_flatten import flatten_edges, stack_node_features

        cls.cfg = _full()
        cls.hetero = _load_hetero_data(cls.cfg)
        x, _, cls.types, cls.slices, cls.dims = stack_node_features(cls.hetero)
        edge_index, edge_type, cls.names, _ = flatten_edges(cls.hetero, cls.slices)
        cls.layer = build_fact_layer(edge_index, edge_type, cls.names, x.size(0), "tied")
        cls.labels = _prepare_target_labels(cls.hetero, cls.slices, "book",
                                            cls.cfg.data.acm_val_ratio,
                                            cls.cfg.data.acm_split_seed)

    def test_counts_equal_the_m1_gate(self) -> None:
        self.assertEqual(self.cfg.data.target_node_type, "book")
        self.assertEqual({n: e - s for n, (s, e) in self.slices.items()}, M1_NODES)
        self.assertEqual(set(self.dims.values()), {0})  # no node features
        self.assertEqual((self.layer.facts_before, self.layer.facts_after), (1057688, 869706))
        target_y = self.labels[5]
        labelled = target_y >= 0
        self.assertEqual(int(labelled.sum()), 7954)
        self.assertEqual(torch.bincount(target_y[labelled]).tolist(), M1_CLASSES)

    def test_split_sizes_and_disjointness(self) -> None:
        _, train, val, test, target_indices, target_y, *_ = self.labels
        masks = [m[target_indices] for m in (train, val, test)]
        self.assertEqual([int(m.sum()) for m in masks], [1909, 477, 5568])  # 24 / 6 / 70
        self.assertFalse(bool((masks[0] & masks[1]).any() | (masks[0] & masks[2]).any()
                              | (masks[1] & masks[2]).any()))
        unlabelled = target_y < 0
        self.assertEqual(int(unlabelled.sum()), 32448)
        self.assertFalse(bool((unlabelled & (masks[0] | masks[1] | masks[2])).any()))

    def test_untied_fallback_relations_equal_a_direct_check(self) -> None:
        folder = hgb_nc_raw_dir(REPO_DATA_ROOT, "freebase")
        pairs: dict[int, set] = defaultdict(set)
        with (folder / "link.dat").open(encoding="utf-8") as handle:
            for line in handle:
                u, v, r, _ = line.rstrip("\n").split("\t")
                pairs[int(r)].add((int(u), int(v)))
        direct = sorted(r for r in pairs
                        if not any(pairs[q] == {(v, u) for u, v in pairs[r]} for q in pairs))
        fallback = sorted(r for r in range(len(self.names)) if self.layer.rho[r] is None)
        self.assertEqual(fallback, direct)
        self.assertEqual(len(fallback), 29)

    def test_literature_row_draws_every_listed_path(self) -> None:
        from src.analysis.schema_readout import literature_signatures
        from src.configs.config import fixed_literature_schema_config, is_literature_row
        from src.data.dataset import build_dataset

        row = fixed_literature_schema_config(_full())
        self.assertTrue(is_literature_row(row.data))
        data = build_dataset(row)  # data read only: the row's evaluation draw
        listed = literature_signatures("freebase", data.fact_layer, data.relation_tuples,
                                       data.node_type_names)
        self.assertEqual(len(listed), 7)
        bb = listed["BB"]
        self.assertEqual((len(bb) - 1) // 3, 2)  # two book-book hops
        drawn = {tuple(g.signature_table[i]) for g in data.face_groups
                 for i in g.signature_index.unique().tolist()}
        self.assertEqual(drawn, set(listed.values()))


class TinyReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        write_freebase_release(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_schema_readout_runs_on_a_freebase_run(self) -> None:
        import scripts.schema_readout as readout
        from src.builders import build_experiment_components
        from src.configs.config import LITERATURE_METAPATHS, TrainConfig
        from src.utils.misc import save_config

        runs = Path(self._tmp.name) / "runs"
        cfg = _full(self.root, epochs=1, exp_name="fb_seed0", output_dir=str(runs),
                    checkpoint_metric="macro_f1", checkpoint_mode="max")
        cfg.exp_dir.mkdir(parents=True)
        save_config(cfg, cfg.exp_dir / "config.yaml")
        data, _, _, _, logger, _, trainer = build_experiment_components(cfg)
        try:
            trainer.fit(data)
        finally:
            logger.close()
        out = runs / "readout"
        readout.main(["--runs", str(cfg.exp_dir), "--out_dir", str(out), "--permutations", "5"])
        record = json.loads((out / "freebase" / "fb_seed0.json").read_text("utf-8"))
        self.assertEqual([item["listed"] for item in record["literature"]],
                         [t.name for t in LITERATURE_METAPATHS["freebase"]])
        self.assertTrue((out / "freebase" / "schema_readout.md").exists())

    def test_every_nc_adapter_runs_one_epoch(self) -> None:
        import src.data.data_layout as layout
        from src.baselines.bundles import build_bundle
        from src.baselines.registry import (
            FREEBASE_VENDOR_REASON,
            get_adapter,
            models_for_task,
            unsupported_reason,
        )
        from tests.test_partP_derived_lp import environment_blocker

        blocked = {"chebnetii": "chebnetii", "nsd": "nsd", "h2sgnn": "h2sgnn"}
        # Part P-fix F4: adapters read the bundle's data root; the default root is an empty dir
        empty = self.root / "empty_default_root"
        empty.mkdir(exist_ok=True)
        with mock.patch.object(layout, "REPO_DATA_ROOT", empty):
            bundle = build_bundle(NODE_CLASSIFICATION, "freebase",
                                  overrides={"device": "cpu", "data": {"root": str(self.root)}})
            for spec in models_for_task(NODE_CLASSIFICATION):
                with self.subTest(adapter=spec.name):
                    reason = unsupported_reason(spec.name, NODE_CLASSIFICATION, "freebase")
                    if spec.name in ("sehgnn", "pshgcn", "h2sgnn"):
                        self.assertEqual(reason, FREEBASE_VENDOR_REASON)
                        self.assertFalse(get_adapter(spec.name).supports(
                            NODE_CLASSIFICATION, "freebase"))
                        continue
                    self.assertIsNone(reason)
                    adapter = get_adapter(spec.name)
                    self.assertTrue(adapter.supports(NODE_CLASSIFICATION, "freebase"))
                    blocker = environment_blocker(spec.name)
                    if spec.name in blocked and blocker is None:
                        from src.baselines.runtime.vendor_paths import VENDOR_ROOTS

                        root = VENDOR_ROOTS[blocked[spec.name]]
                        if not root.is_dir() or not any(root.rglob("*.py")):
                            blocker = f"vendor clone {root} is empty on this machine"
                    if blocker is not None:
                        self.skipTest(blocker)
                    record = adapter.fit_task(bundle, BaselineParams(
                        {"epochs": 1, "early_stopping": 1}), task=NODE_CLASSIFICATION, seed=0,
                        device=torch.device("cpu"))
                    for key in ("test_macro_f1", "test_micro_f1", "val_macro_f1"):
                        self.assertTrue(math.isfinite(float(record[key])), key)


if __name__ == "__main__":
    unittest.main()
