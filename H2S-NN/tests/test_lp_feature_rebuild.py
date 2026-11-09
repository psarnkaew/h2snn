"""J5.3 (task card Part J, spec v1.9 F12): rebuilt LP features equal the rule applied to the
node's training facts (bitwise, by an independent code path), dropped types become featureless,
every other row stays byte-identical, and the release is never written."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "lp"))

import lp_feature_leak as fl  # noqa: E402
import lp_feature_rules as fr  # noqa: E402
import rebuild_lp_features as rb  # noqa: E402
from scripts.lp import build_derived_lp as bd  # noqa: E402
from tests.test_lp_feature_rules import write_release  # noqa: E402

TARGET = 0  # paper -> author in the toy release


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


class FeatureRebuildTests(unittest.TestCase):
    def build(self, rule: str, keyword_type: int | None = None) -> tuple[fr.TaskSpec, dict]:
        release = self.root / f"release_{rule}"
        write_release(release, rule)
        task = self.root / "lp_derived" / rule
        bd.build_task(release, task, TARGET, seed=42, val_frac=0.09, test_frac=0.10)
        spec = fr.TaskSpec(rule, task, release, TARGET, keyword_type)
        self.release_md5 = {n: md5(release / n) for n in ("node.dat", "link.dat", "info.dat")}
        return spec, fr.identify(spec)

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def check_rebuilt_bitwise(self, spec: fr.TaskSpec, record: dict, type_id: int) -> None:
        info = record["types"][str(type_id)]
        self.assertEqual(info["action"], "rebuilt")
        source, agg = info["rule"].split("/")
        g, mapping, other, _, rel_types, _, _ = fl.chosen_rule(spec, type_id, source, agg)
        self.assertEqual(mapping.tolist(), info["mapping"])
        stored = fr.read_nodes(spec.folder)
        train = fl.training_pairs(spec.folder, spec.target)
        expected = fl.rule_on_training(train, stored[type_id].ids, stored[other].ids,
                                       rel_types[spec.target][0] == type_id, g, mapping, agg)
        self.assertTrue(np.array_equal(stored[type_id].x.toarray(), expected))
        # the loader's float32 tensor too (through the loader's own node.dat reader; J5 review #4)
        from src.data.link_prediction_dataset import _load_local_hgb_nodes

        names = {0: "paper", 1: "author", 2: "keyword"}
        loaded = _load_local_hgb_nodes(spec.folder, names).data[names[type_id]].x.numpy()
        self.assertTrue(np.array_equal(loaded, expected.astype(np.float32)))

    def test_or_rule_is_rebuilt_from_training_facts_only(self) -> None:
        spec, ident = self.build("or")
        record = rb.rebuild_task(spec, ident)
        self.check_rebuilt_bitwise(spec, record, 1)
        self.assertGreater(record["types"]["1"]["nodes_changed"], 0)  # held-out papers removed
        self.assertEqual(record["types"]["0"]["action"], "unchanged")

    def test_mean_rule_over_mapped_tokens_is_rebuilt(self) -> None:
        spec, ident = self.build("mean_tokens", keyword_type=2)
        record = rb.rebuild_task(spec, ident)
        self.check_rebuilt_bitwise(spec, record, 1)
        self.assertEqual(record["types"]["1"]["format"], "repr")

    def test_type_without_exact_rule_is_dropped(self) -> None:
        spec, ident = self.build("own")
        ident["types"]["1"]["affected"] = True  # as if affected without an exact rule
        record = rb.rebuild_task(spec, ident)
        self.assertEqual(record["types"]["1"]["action"], "dropped")
        self.assertIsNone(fr.read_nodes(spec.folder)[1].x)

    def test_other_rows_byte_identical_and_release_untouched(self) -> None:
        spec, ident = self.build("or")
        record = rb.rebuild_task(spec, ident)
        release = (spec.release / "node.dat").read_bytes().splitlines(keepends=True)
        task = (spec.folder / "node.dat").read_bytes().splitlines(keepends=True)
        self.assertEqual(len(release), len(task))
        rebuilt_type = b"\t1\t"
        for a, b in zip(release, task):
            if rebuilt_type not in a:
                self.assertEqual(a, b)
        self.assertEqual({n: md5(spec.release / n) for n in self.release_md5}, self.release_md5)
        meta = json.loads((spec.folder / "split_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["output_md5"]["node.dat"], md5(spec.folder / "node.dat"))
        self.assertEqual(record["rows_rewritten"], 24)

    def test_rebuild_is_idempotent(self) -> None:
        spec, ident = self.build("or")
        rb.rebuild_task(spec, ident)
        first = (spec.folder / "node.dat").read_bytes()
        rb.rebuild_task(spec, ident)
        self.assertEqual((spec.folder / "node.dat").read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
