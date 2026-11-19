"""Card 7 T5 (`docs/H2S-NN_card7_card.md` V4): the Q3 schema analysis on crafted read-out records.

- perfectly stable rho rankings over seeds 1-10 -> mean tau-b = 1 and the smallest permutation p; independently
  shuffled rankings -> a large p; fewer than 4 schemas -> n/a; Holm over the datasets; the c ranking's tau beside;
- the top schemas: length-2, mean coverage >= 1 %, the ten largest by coverage;
- the literature count (seeds with rho > 1);
- a schema maps to its target-type meta-path (forward and reversed hops) for the homophily link.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "scripts" / "ablation"))

import q3_schema as q3  # noqa: E402

DRAWS = 200


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
    def test_perfectly_stable_rankings_give_tau_one_and_the_smallest_p(self) -> None:
        rhos = [2.0, 1.5, 1.2, 1.0, 0.8, 0.5]
        records = [record("acm", s, [r * (1 + 0.01 * s) for r in rhos]) for s in range(0, 11)]
        schemas = q3.top_schemas(records)
        result = q3.stability(records, schemas, draws=DRAWS)
        self.assertEqual(result["seeds"], list(range(1, 11)))           # seed 0 is not a stability seed
        self.assertAlmostEqual(result["mean_tau_b_rho"], 1.0, places=12)
        self.assertAlmostEqual(result["p_permutation"], 1 / (DRAWS + 1), places=12)
        self.assertAlmostEqual(result["mean_tau_b_c"], 0.0, places=12)    # constant coverage: tau-b ties -> 0

    def test_shuffled_rankings_give_a_large_p(self) -> None:
        rng = np.random.default_rng(3)
        records = [record("acm", s, list(rng.permutation(6) + 0.5)) for s in range(1, 11)]
        result = q3.stability(records, q3.top_schemas(records), draws=DRAWS)
        self.assertLess(abs(result["mean_tau_b_rho"]), 0.3)
        self.assertGreater(result["p_permutation"], 0.05)

    def test_fewer_than_four_schemas_is_na(self) -> None:
        records = [record("acm", s, [2.0, 1.0, 0.5]) for s in range(1, 11)]
        self.assertEqual(q3.stability(records, q3.top_schemas(records), draws=DRAWS)["status"], "n/a")

    def test_holm_over_datasets_and_the_stable_rule(self) -> None:
        rhos = [2.0, 1.5, 1.2, 1.0, 0.8, 0.5]
        rng = np.random.default_rng(5)
        by_dataset = {"acm": [record("acm", s, rhos) for s in range(1, 11)],
                      "dblp": [record("dblp", s, list(rng.permutation(6) + 0.5)) for s in range(1, 11)]}
        with tempfile.TemporaryDirectory() as tmp:
            result = q3.analyse(by_dataset, Path(tmp), None, draws=DRAWS, make_figures=True)
            acm, dblp = result["datasets"]["acm"]["stability"], result["datasets"]["dblp"]["stability"]
            self.assertAlmostEqual(acm["p_holm"], min(1.0, 2 * acm["p_permutation"]), places=12)
            self.assertTrue(acm["stable"])
            self.assertFalse(dblp["stable"])
            for path in result["datasets"]["acm"]["figures"]:
                self.assertTrue(Path(path).exists() and Path(path).stat().st_size > 0)
            self.assertTrue((Path(tmp) / "q3_schema.md").exists())


class SelectionAndLiteratureTests(unittest.TestCase):
    def test_top_schemas_by_mean_coverage_threshold_and_cap(self) -> None:
        coverages = [0.3, 0.2, 0.009, 0.15] + [0.02] * 10
        records = [record("acm", s, [1.0] * len(coverages), coverages) for s in range(1, 4)]
        records[0]["schemas"].append(schema(99, 0.9, 1.0, span=3))       # length 3: never selected
        top = q3.top_schemas(records)
        self.assertEqual(len(top), 10)
        names = [records[0]["schemas"][[tuple(s["signature"]) for s in records[0]["schemas"]].index(k)]["name"]
                 for k in top]
        self.assertEqual(names[:3], ["s0", "s1", "s3"])
        self.assertNotIn("s2", names)                                     # mean coverage 0.9 % < 1 %
        self.assertNotIn("s99", names)

    def test_literature_count(self) -> None:
        records = [record("acm", s, [1.0] * 4, literature=[1.5 if s <= 7 else 0.9, None, 1.0])
                   for s in range(0, 11)]
        counts = {x["listed"]: x for x in q3.literature_counts(records)}
        self.assertEqual(counts["L0"]["seeds"], 10)
        self.assertEqual(counts["L0"]["seeds_rho_above_1"], 7)
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
