"""Card 7 T5, updated by card 9 W9 (pre-registration v3 §4.1): the Q3 schema analysis on crafted read-out records.

- perfectly stable rho rankings over seeds 1-5 -> mean tau-b = 1, stable (v2's smallest permutation p removed: no
  null in v3); alternating reversed rankings -> mean tau-b -0.2, not stable (v2: "shuffled -> large p");
  fewer than 4 schemas -> n/a; the c ranking's tau beside; v2's Holm-over-datasets test is now "the stable rule and
  the figures" (no Holm in v3);
- the top schemas: length-2, mean coverage >= 1 %, the ten largest by coverage;
- the literature count (seeds of 5 with rho > 1; v2 counted seeds 1-10);
- a schema maps to its target-type meta-path (forward and reversed hops) for the homophily link.
The v3-only checks (seed 0 ignored, no permutation key, `favour` only at 5 of 5) are card 9 T7.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "scripts" / "ablation"))

import q3_schema as q3  # noqa: E402

RHOS = [2.0, 1.5, 1.2, 1.0, 0.8, 0.5]


def alternating(dataset: str) -> list[dict]:
    """Seeds 1-5 with the rho ranking, its reverse, the ranking, its reverse, the ranking: 4 pairs tau +1, 6 pairs
    tau -1 -> mean -0.2."""
    return [record(dataset, s, RHOS if s % 2 else RHOS[::-1]) for s in range(1, 6)]


def schema(i: int, c: float, rho: float | None, span: int = 2) -> dict:
    return {"name": f"s{i}", "relations": f"paper -[r{i} +]-> x -[r{i} -]-> paper", "signature": [0, i, 2, 1, i, 0, 0],
            "span": span, "c": c, "I": c * (rho or 0.0), "rho": rho, "mean_g": 0.0, "num_cells": 1}


def record(dataset: str, seed: int, rhos: list[float], coverages: list[float] | None = None,
           literature: list[float | None] | None = None) -> dict:
    coverages = coverages or [0.05] * len(rhos)
    out = {"dataset": dataset, "seed": seed, "run": f"{dataset}_{seed}", "path_source": "wisp",
           "schemas": [schema(i, c, r) for i, (c, r) in enumerate(zip(coverages, rhos))]}
    out["literature"] = None if literature is None else [
        {"listed": f"L{i}", "relations": f"lit{i}", "rho": r} for i, r in enumerate(literature)]
    return out


class StabilityTests(unittest.TestCase):
    def test_perfectly_stable_rankings_give_tau_one(self) -> None:
        records = [record("acm", s, [r * (1 + 0.01 * s) for r in RHOS]) for s in range(1, 6)]
        schemas = q3.top_schemas(records)
        result = q3.stability(records, schemas)
        self.assertEqual(result["seeds"], [1, 2, 3, 4, 5])
        self.assertEqual(result["pairs"], 10)
        self.assertAlmostEqual(result["mean_tau_b_rho"], 1.0, places=12)
        self.assertTrue(result["stable"])
        self.assertAlmostEqual(result["mean_tau_b_c"], 0.0, places=12)    # constant coverage: tau-b ties -> 0

    def test_alternating_rankings_are_not_stable(self) -> None:
        records = alternating("acm")
        result = q3.stability(records, q3.top_schemas(records))
        self.assertAlmostEqual(result["mean_tau_b_rho"], -0.2, places=12)
        self.assertFalse(result["stable"])

    def test_fewer_than_four_schemas_is_na(self) -> None:
        records = [record("acm", s, [2.0, 1.0, 0.5]) for s in range(1, 6)]
        self.assertEqual(q3.stability(records, q3.top_schemas(records))["status"], "n/a")

    def test_the_stable_rule_and_the_figures(self) -> None:
        by_dataset = {"acm": [record("acm", s, RHOS) for s in range(1, 6)], "dblp": alternating("dblp")}
        with tempfile.TemporaryDirectory() as tmp:
            result = q3.analyse(by_dataset, Path(tmp), None, make_figures=True)
            acm, dblp = result["datasets"]["acm"]["stability"], result["datasets"]["dblp"]["stability"]
            self.assertTrue(acm["stable"])
            self.assertFalse(dblp["stable"])
            for path in result["datasets"]["acm"]["figures"]:
                self.assertTrue(Path(path).exists() and Path(path).stat().st_size > 0)
            self.assertTrue((Path(tmp) / "q3_schema.md").exists())


class SelectionAndLiteratureTests(unittest.TestCase):
    def test_top_schemas_by_mean_coverage_threshold_and_cap(self) -> None:
        coverages = [0.3, 0.2, 0.009, 0.15] + [0.02] * 10
        records = [record("acm", s, [1.0] * len(coverages), coverages) for s in range(1, 6)]
        records[0]["schemas"].append(schema(99, 0.9, 1.0, span=3))       # length 3: never selected
        top = q3.top_schemas(records)
        self.assertEqual(len(top), 10)
        names = [records[0]["schemas"][[tuple(s["signature"]) for s in records[0]["schemas"]].index(k)]["name"]
                 for k in top]
        self.assertEqual(names[:3], ["s0", "s1", "s3"])
        self.assertNotIn("s2", names)                                     # mean coverage 0.9 % < 1 %
        self.assertNotIn("s99", names)

    def test_literature_count(self) -> None:
        records = [record("acm", s, [1.0] * 4, literature=[1.5 if s <= 3 else 0.9, None, 1.0])
                   for s in range(1, 6)]
        counts = {x["listed"]: x for x in q3.literature_counts(records)}
        self.assertEqual(counts["L0"]["seeds"], 5)
        self.assertEqual(counts["L0"]["seeds_rho_above_1"], 3)
        self.assertEqual(counts["L1"]["seeds_rho_above_1"], 0)            # absent from the draw
        self.assertEqual(counts["L2"]["seeds_rho_above_1"], 0)            # rho = 1 is not > 1


class SchemaMappingTests(unittest.TestCase):
    def test_schema_maps_to_the_target_meta_path(self) -> None:
        import torch
        from torch_geometric.data import HeteroData

        data = HeteroData()
        data["paper"].num_nodes, data["author"].num_nodes = 3, 2
        writes = torch.tensor([[0, 1, 2], [0, 1, 1]])
        data["paper", "to", "author"].edge_index = writes
        data["author", "to", "paper"].edge_index = writes.flip(0)
        relations = "paper -[paper__to__author +]-> author -[paper__to__author -]-> paper"
        self.assertEqual(q3.schema_metapath(data, "paper", relations),
                         (("paper", "to", "author"), ("author", "to", "paper")))
        self.assertIsNone(q3.schema_metapath(data, "author", relations))      # ends are not the target type
        self.assertEqual(q3.hop_edge_types("a -[x__y__z ~]-> a -[x__y__z +]-> b"), [("x__y__z", 0), ("x__y__z", 1)])


if __name__ == "__main__":
    unittest.main()
