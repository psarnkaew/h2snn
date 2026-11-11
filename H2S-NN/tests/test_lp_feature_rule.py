"""J6.1/J6.3 (spec v1.10 F17, F20): the feature rule removes whole blocks and single columns and
nothing else; the loader refuses a derived task whose node.dat does not match the rule's record
(missing record, the split builder re-run, a hand-edited value, a removed type with features)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.lp import build_derived_lp as bd
from scripts.lp.apply_feature_rule import apply_rule
from tests.test_lp_derived import derived_config, write_release
from src.data.data_layout import lp_derived_root

TARGET = ("paper", "rel_2", "author")


class FeatureRuleAndGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_release(self.root / "release")
        self.task_dir = lp_derived_root(self.root) / "acm_pa"
        self.build()
        self.record = apply_rule("acm_pa", self.task_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def build(self) -> None:
        bd.build_task(self.root / "release", self.task_dir, 2, seed=42, val_frac=0.09,
                      test_frac=0.10)

    def load(self):
        from src.data.link_prediction_dataset import build_link_prediction_dataset

        return build_link_prediction_dataset(derived_config(self.root, "acm_pa", TARGET))

    def test_rule_removes_the_block_and_keeps_every_other_row(self) -> None:
        release = (self.root / "release" / "node.dat").read_bytes().splitlines()
        task = (self.task_dir / "node.dat").read_bytes().splitlines()
        for a, b in zip(release, task):
            if a.split(b"\t")[2] == b"1":
                self.assertEqual(b.split(b"\t")[3], b"")  # author block removed
            else:
                self.assertEqual(a, b)
        self.assertEqual(self.record["widths"]["1"], 0)

    def test_rule_removes_single_columns(self) -> None:
        record = apply_rule("acm_pa", self.task_dir,
                            {"remove_types": [], "remove_columns": {0: [1]}})
        rows = [r.split(b"\t") for r in (self.task_dir / "node.dat").read_bytes().splitlines()]
        papers = [r for r in rows if r[2] == b"0"]
        self.assertTrue(all(r[3] == b"0.1" for r in papers))  # "0.1,0.<i>" -> column 1 gone
        self.assertEqual(record["widths"]["0"], 1)

    def test_rule_applied_task_loads(self) -> None:
        data = self.load()
        self.assertGreater(data.train_pos_edge_index.size(1), 0)

    def test_missing_record_is_refused(self) -> None:
        (self.task_dir / "feature_rule.json").unlink()
        with self.assertRaisesRegex(ValueError, "feature_rule.json"):
            self.load()

    def test_rerun_of_the_split_builder_is_refused(self) -> None:
        self.build()  # restores the release node.dat (author features back)
        with self.assertRaisesRegex(ValueError, "does not match feature_rule.json"):
            self.load()

    def test_hand_edited_value_is_refused(self) -> None:
        path = self.task_dir / "node.dat"
        path.write_bytes(path.read_bytes().replace(b"0.1,0.1\n", b"0.1,0.2\n", 1))
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.load()

    def test_removed_type_with_features_is_refused_even_with_a_matching_md5(self) -> None:
        import hashlib

        path = self.task_dir / "node.dat"
        rows = path.read_bytes().split(b"\n")
        # every author row gets features (one consistent width), so only the record check sees it
        rows = [r + b"0.3,0.4" if r.split(b"\t")[2:3] == [b"1"] else r for r in rows]
        self.assertGreater(sum(r.endswith(b"0.3,0.4") for r in rows), 1)
        path.write_bytes(b"\n".join(rows))
        rec_path = self.task_dir / "feature_rule.json"
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        rec["node_dat_md5"] = hashlib.md5(path.read_bytes()).hexdigest()  # forged record
        rec_path.write_text(json.dumps(rec), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "widths|different widths"):
            self.load()


if __name__ == "__main__":
    unittest.main()
