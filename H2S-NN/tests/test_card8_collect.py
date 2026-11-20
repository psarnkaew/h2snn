"""Card 8 C7 (review addition): `scripts/tsubame/collect.py` puts every published-run record
(`<out>/experiments/baselines/<token>/published_seed<s>.json`) into `results.csv` with kind `published_seed` and the
provenance of the published line run that wrote it (v3b SHA, host, job, task, GPU, wall time).

Card mutation: the published records skipped (`yield from published_rows(...)` removed) -> this test fails.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path


class PublishedRowsTests(unittest.TestCase):
    def test_published_records_reach_results_csv_with_provenance(self) -> None:
        from scripts.tsubame import collect
        from scripts.tsubame.make_cell_list import published_line

        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            token = "gcn_node_classification_mag-year"
            cell_dir = out / "experiments" / "baselines" / token
            cell_dir.mkdir(parents=True)
            for seed in (0, 1):
                (cell_dir / f"published_seed{seed}.json").write_text(json.dumps(
                    {"task": "node_classification", "dataset": "mag-year", "model": "gcn", "monitor": "macro_f1",
                     "val_macro_f1": 0.5 + seed, "test_macro_f1": 0.4, "published_seed": seed}))
            line = published_line("node_classification", "gcn", "mag-year", out)
            prov = out / "logs" / "prov"
            prov.mkdir(parents=True)
            key = {"job_id": "J1", "task_id": "2", "slot": 0, "start": "2026-11-20T00:00:00"}
            events = [{**key, "event": "start", "sha": "abc", "host": "h1", "gpu_name": "H100", "line": line,
                       "before": {"token": token, "units": []}},
                      {**key, "event": "end", "wall_s": 12.5, "after": {"final_seed_files": [0, 1]}}]
            (prov / "p.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
            summary = collect.collect(out)
            with (out / "collected" / "results.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue((out / "collected" / "cells" / f"{token}.tar").exists())
        self.assertEqual(summary["published_seeds"], 2)
        published = [r for r in rows if r["kind"] == "published_seed"]
        self.assertEqual(len(published), 2)
        self.assertEqual(sorted(r["trial_or_seed"] for r in published), ["0", "1"])
        for r in published:
            self.assertEqual((r["task"], r["dataset"], r["method"]), ("node_classification", "mag-year", "gcn"))
            self.assertEqual((r["v3b_sha"], r["host"], r["job_id"], r["task_id"], r["gpu_name"], r["wall_time_s"]),
                             ("abc", "h1", "J1", "2", "H100", "12.5"))
            self.assertEqual(r["monitor"], "val_macro_f1")
        self.assertEqual(summary["rows_without_provenance"], 0)


if __name__ == "__main__":
    unittest.main()
