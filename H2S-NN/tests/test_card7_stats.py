"""Card 7 T2, updated by card 9 W9 (pre-registration v3 §1, §3): `scripts/ablation/stats.py` on synthetic numbers.

v2 tests removed with the code they tested (v3 D1): Wilcoxon p, Holm per claim, the bootstrap CI.
Kept and moved to v3:
- the claim rules: >= 3 of 4 datasets; C15 needs both IMDB and Freebase; a dataset is "better" only with a positive
  mean difference and >= 4 of 5 strict wins (the crafted edge cases of the rule are card 9 T1, `test_card9_v3.py`);
- a missing dataset leaves a claim undecided (`None`);
- the 2x2 mean differences and the `complementary` flag; S0 and the literature row (roles swapped) reported;
- JSON and Markdown are written; no p-value key anywhere in the output.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ablation"))

import stats  # noqa: E402

DATASETS = ("acm", "dblp", "imdb", "freebase")
BASE = 0.80
UP = [0.010, 0.011, 0.012, 0.013, 0.014]            # full - row > 0 on every seed: better
MIXED = [0.010, -0.011, 0.012, -0.013, 0.014]       # 3 wins: not better
DOWN = [-v for v in UP]


def rows_for(dataset: str, row: str, values: list[float]) -> list[dict]:
    return [{"dataset": dataset, "row": row, "configuration": "a", "seed": s + 1, stats.METRIC: v}
            for s, v in enumerate(values)]


def table(pattern: dict[tuple[str, str], list[float]]) -> list[dict]:
    """`full` = BASE + 0.05 on every seed; each (row, dataset) given as the differences full - row."""
    full = [BASE + 0.05] * 5
    out = []
    for dataset in DATASETS:
        out += rows_for(dataset, "full", full)
    for (row, dataset), diffs in pattern.items():
        out += rows_for(dataset, row, [f - d for f, d in zip(full, diffs)])
    return out


class ClaimRuleTests(unittest.TestCase):
    def test_three_of_four_and_c15_rules(self) -> None:
        pattern = {}
        for d in DATASETS:
            pattern[("no_path_cells", d)] = MIXED if d == "freebase" else UP     # 3 of 4 -> holds
            pattern[("endpoints", d)] = UP if d in ("acm", "dblp") else MIXED      # 2 of 4 -> does not hold
            pattern[("uniform", d)] = UP                                           # C2-w: 4 of 4
            pattern[("instance_features", d)] = DOWN                               # 0 wins, negative mean
            pattern[("S2", d)] = UP if d in ("imdb", "freebase") else MIXED        # C15: both heterophilic
        result = {c["claim"]: c for c in stats.analyse(table(pattern))["claims"]}
        self.assertEqual(set(result), {"C2-cells", "C2", "C2-w", "C14", "C15"})
        self.assertTrue(result["C2-cells"]["holds"])
        self.assertEqual(result["C2-cells"]["better"], {"acm": True, "dblp": True, "imdb": True, "freebase": False})
        self.assertFalse(result["C2"]["holds"])
        self.assertTrue(result["C2"]["if_not"].startswith("kill rule"))
        self.assertTrue(result["C2-w"]["holds"])
        self.assertEqual(result["C2-w"]["row"], "uniform")
        self.assertFalse(result["C14"]["holds"])
        self.assertTrue(result["C15"]["holds"])
        pattern[("S2", "freebase")] = MIXED                                         # IMDB alone is not enough
        pattern[("S2", "acm")] = UP
        pattern[("S2", "dblp")] = UP
        result = {c["claim"]: c for c in stats.analyse(table(pattern))["claims"]}
        self.assertFalse(result["C15"]["holds"])

    def test_missing_dataset_leaves_the_claim_undecided(self) -> None:
        pattern = {("no_path_cells", d): UP for d in ("acm", "dblp", "imdb")}
        result = {c["claim"]: c for c in stats.analyse(table(pattern))["claims"]}
        self.assertIsNone(result["C2-cells"]["holds"])
        self.assertIsNone(result["C2-cells"]["per_dataset"]["freebase"])
        self.assertTrue(result["C2-cells"]["per_dataset"]["acm"]["better"])
        pattern = {("S2", d): UP for d in ("imdb",)}                               # C15 needs Freebase too
        result = {c["claim"]: c for c in stats.analyse(table(pattern))["claims"]}
        self.assertIsNone(result["C15"]["holds"])


class ReportedTests(unittest.TestCase):
    def test_two_by_two_differences_and_complementary(self) -> None:
        # full = 0.85; no_path_cells = 0.83; S2 = 0.80; S0 = 0.79 on every seed and dataset
        rows = []
        for d in DATASETS:
            for row, value in (("full", 0.85), ("no_path_cells", 0.83), ("S2", 0.80), ("S0", 0.79)):
                rows += rows_for(d, row, [value] * 5)
        result = stats.analyse(rows)["two_by_two"]
        expected = {"full - no_path_cells": 0.02, "S2 - S0": 0.01, "full - S2": 0.05, "no_path_cells - S0": 0.04}
        for name, value in expected.items():
            for d in DATASETS:
                self.assertAlmostEqual(result["differences"][name]["per_dataset"][d]["mean_difference"], value,
                                       places=12)
        self.assertEqual(result["complementary"], {d: True for d in DATASETS})       # 0.02 > 0.01
        rows = [r for r in rows if not (r["row"] == "S0" and r["dataset"] == "acm")]
        self.assertIsNone(stats.analyse(rows)["two_by_two"]["complementary"]["acm"])

    def test_s0_and_literature_are_reported(self) -> None:
        pattern = {("S0", d): UP for d in DATASETS}
        pattern.update({("literature", d): DOWN for d in DATASETS})                 # literature above full
        result = stats.analyse(table(pattern))
        s0 = result["reported_only"][0]
        self.assertEqual(s0["row"], "S0")
        self.assertTrue(all(r["better"] for r in s0["per_dataset"].values()))
        literature = result["literature"]
        self.assertEqual(literature["difference"], "literature - full")
        for r in literature["per_dataset"].values():
            self.assertTrue(r["literature_better"])
            self.assertNotIn("better", r)
            self.assertAlmostEqual(r["mean_difference"], 0.012, places=12)

    def test_markdown_and_json_are_written_without_p_values(self) -> None:
        pattern = {(row, d): UP for row in ("no_path_cells", "S2", "uniform") for d in DATASETS}
        result = stats.analyse(table(pattern))
        text = json.dumps(result)
        for key in ("p_raw", "p_holm", "bootstrap", "wilcoxon", "median_difference", "claims_b"):
            self.assertNotIn(key, text)
        md = stats.markdown(result)
        for word in ("C2-cells", "C2-w", "2x2", "Literature row", "Complementary"):
            self.assertIn(word, md)


if __name__ == "__main__":
    unittest.main()
