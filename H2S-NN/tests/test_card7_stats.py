"""Card 7 T2 (`docs/H2S-NN_card7_card.md` V2): `scripts/ablation/stats.py` on synthetic numbers.

- Wilcoxon p equals scipy's on known inputs (10 positive distinct differences: exact two-sided p = 2/1024);
- Holm on a hand-checked example; Holm is per claim (m = 4 datasets), never over all claims (mutation 2's target);
- the claim rules: >= 3 of 4 datasets; C15 needs both IMDB and Freebase; a dataset passes only with adjusted
  p < 0.05 and a positive median difference;
- the 2x2 differences and the bootstrap CI (fixed seed, contains the mean difference).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ablation"))

import stats  # noqa: E402

DATASETS = ("acm", "dblp", "imdb", "freebase")
BASE = 0.80
UP = [0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019]   # full - row > 0, distinct
MIXED = [0.010, -0.011, 0.012, -0.013, 0.014, -0.015, 0.016, -0.017, 0.018, -0.019]


def rows_for(dataset: str, row: str, configuration: str, values: list[float]) -> list[dict]:
    return [{"dataset": dataset, "row": row, "configuration": configuration, "seed": s + 1,
             stats.METRIC: v} for s, v in enumerate(values)]


def table(pattern: dict[tuple[str, str], list[float]]) -> list[dict]:
    """`full` (a) = BASE + 0.05 on every seed; each (row, dataset) given as the differences full - row."""
    full = [BASE + 0.05] * 10
    out = []
    for dataset in DATASETS:
        out += rows_for(dataset, "full", "a", full)
    for (row, dataset), diffs in pattern.items():
        for configuration in ("a", "b") if row in ("no_path_cells", "endpoints", "instance_features", "S2") \
                else ("a",):
            out += rows_for(dataset, row, configuration, [f - d for f, d in zip(full, diffs)])
    return out


class WilcoxonAndHolmTests(unittest.TestCase):
    def test_wilcoxon_matches_scipy_and_the_exact_value(self) -> None:
        from scipy.stats import wilcoxon

        p = stats.wilcoxon_p(np.array(UP))
        self.assertAlmostEqual(p, 2 / 1024, places=12)
        self.assertAlmostEqual(stats.wilcoxon_p(np.array(MIXED)),
                               float(wilcoxon(np.array(MIXED), alternative="two-sided").pvalue), places=12)
        self.assertIsNone(stats.wilcoxon_p(np.zeros(10)))

    def test_holm_hand_checked_example(self) -> None:
        # sorted: freebase .005*4 = .02; acm .01*3 = .03; imdb .03*2 = .06; dblp .04*1 = .04 -> max(.06, .04) = .06
        adjusted = stats.holm({"acm": 0.01, "dblp": 0.04, "imdb": 0.03, "freebase": 0.005})
        expected = {"freebase": 0.02, "acm": 0.03, "imdb": 0.06, "dblp": 0.06}
        for key, value in expected.items():
            self.assertAlmostEqual(adjusted[key], value, places=12, msg=key)
        self.assertEqual(stats.holm({"a": 0.5, "b": None}), {"a": 0.5, "b": None})

    def test_holm_is_per_claim_over_four_datasets(self) -> None:
        pattern = {(row, d): UP for row in ("no_path_cells", "endpoints", "instance_features", "S2", "S0",
                                            "literature") for d in DATASETS}
        result = stats.analyse(table(pattern))
        for claim in result["claims_b"] + result["claims_a"]:
            for dataset, r in claim["per_dataset"].items():
                self.assertEqual(r["m"], 4)
                self.assertAlmostEqual(r["p_raw"], 2 / 1024, places=12)
                self.assertAlmostEqual(r["p_holm"], 4 * 2 / 1024, places=12, msg=(claim["claim"], dataset))


class ClaimRuleTests(unittest.TestCase):
    def test_three_of_four_and_c15_rules(self) -> None:
        pattern = {}
        for d in DATASETS:
            pattern[("no_path_cells", d)] = MIXED if d == "freebase" else UP     # 3 of 4 pass -> holds
            pattern[("endpoints", d)] = UP if d in ("acm", "dblp") else MIXED      # 2 of 4 -> does not hold
            pattern[("instance_features", d)] = [-v for v in UP]                   # significant, wrong sign
            pattern[("S2", d)] = UP if d in ("imdb", "freebase") else MIXED        # C15: both heterophilic
        result = {c["claim"]: c for c in stats.analyse(table(pattern))["claims_b"]}
        self.assertTrue(result["C2-cells"]["holds"])
        self.assertEqual(result["C2-cells"]["passes"],
                         {"acm": True, "dblp": True, "imdb": True, "freebase": False})
        self.assertFalse(result["C2"]["holds"])
        self.assertEqual(result["C2"]["if_not"], "kill rule")
        self.assertFalse(result["C14"]["holds"])                       # median < 0 fails though p is small
        self.assertTrue(all(r["p_holm"] < 0.05 for r in result["C14"]["per_dataset"].values()))
        self.assertTrue(result["C15"]["holds"])
        pattern[("S2", "freebase")] = MIXED                                 # IMDB alone is not enough
        pattern[("S2", "acm")] = UP
        pattern[("S2", "dblp")] = UP
        result = {c["claim"]: c for c in stats.analyse(table(pattern))["claims_b"]}
        self.assertFalse(result["C15"]["holds"])

    def test_missing_dataset_leaves_the_claim_undecided(self) -> None:
        pattern = {("no_path_cells", d): UP for d in ("acm", "dblp", "imdb")}
        result = {c["claim"]: c for c in stats.analyse(table(pattern))["claims_b"]}
        self.assertIsNone(result["C2-cells"]["holds"])
        self.assertEqual(result["C2-cells"]["per_dataset"]["acm"]["m"], 3)


class TwoByTwoAndCITests(unittest.TestCase):
    def test_two_by_two_differences(self) -> None:
        # full = 0.85; no_path_cells = 0.83; S2 = 0.80; S0 = 0.79 on every seed and dataset
        rows = []
        for d in DATASETS:
            for row, value in (("full", 0.85), ("no_path_cells", 0.83), ("S2", 0.80), ("S0", 0.79)):
                rows += rows_for(d, row, "a", [value] * 10)
        result = stats.analyse(rows)["two_by_two_a"]
        expected = {"full - no_path_cells": 0.02, "S2 - S0": 0.01, "full - S2": 0.05, "no_path_cells - S0": 0.04}
        for name, value in expected.items():
            for d in DATASETS:
                self.assertAlmostEqual(result[name]["per_dataset"][d]["mean_difference"], value, places=12)
                self.assertAlmostEqual(result[name]["per_dataset"][d]["median_difference"], value, places=12)

    def test_bootstrap_ci_is_fixed_and_contains_the_mean(self) -> None:
        diffs = np.array(MIXED)
        low, high = stats.bootstrap_ci(diffs)
        self.assertEqual((low, high), stats.bootstrap_ci(diffs))
        self.assertLessEqual(low, diffs.mean())
        self.assertGreaterEqual(high, diffs.mean())

    def test_markdown_and_json_are_written(self) -> None:
        import json

        pattern = {(row, d): UP for row in ("no_path_cells", "S2") for d in DATASETS}
        result = stats.analyse(table(pattern))
        json.dumps(result)
        text = stats.markdown(result)
        self.assertIn("C2-cells", text)
        self.assertIn("2x2", text)


if __name__ == "__main__":
    unittest.main()
