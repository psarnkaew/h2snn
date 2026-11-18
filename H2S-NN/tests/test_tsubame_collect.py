"""TSUBAME card T4: status and collection on a synthetic output tree (nothing trained, no metric computed).

The tree has two cells: a finished sweep of 3 records written over two line runs (the first killed by the job
limit, so it has no end event) with 2 final seeds, and a sweep stopped by the 24 h guard with 1 record.
Checks: the row count equals the number of records; every value is copied unchanged; the provenance columns are
filled (each record carries the job of the run that wrote it); one tar per cell; SHA256SUMS match; status.py reads
the same files. `lines.py`'s paths equal the launcher's (`state_path_for`) on every phase-1 line.
Mutation (`docs/tsubame_evidence/t4_mutation.py`): a record silently skipped -> this test fails.
"""

from __future__ import annotations

import csv
import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.tsubame import collect, status
from scripts.tsubame.lines import parse_line, runnable_lines

NC = "node_classification"


def line(out: Path, model: str, dataset: str, final: bool = False) -> str:
    from scripts.tsubame.make_cell_list import final_seeds_line, tuning_line

    return (final_seeds_line if final else tuning_line)(NC, model, dataset, out)


def trial(model: str, dataset: str, i: int) -> dict:
    return {"model": model, "dataset": dataset, "task": NC, "trial_name": f"{model}-{i}",
            "val_macro_f1": 0.1 + i / 7, "test_macro_f1": 0.2 + i / 9, "test_micro_f1": 0.3 + i / 11,
            "sweep_params": {"lr": 10 ** -(i + 2), "dropout": 0.5}, "wandb_run_id": f"w{model}{i}"}


def snap(text: str, records: int, seeds=()) -> dict:
    parsed = parse_line(text)
    return {"token": parsed.token, "kind": parsed.kind, "final_seed_files": list(seeds),
            "units": [{"state": str(p), "exists": records >= 0, "records": max(records, 0)}
                      for p in parsed.state_paths()]}


def build_tree(out: Path) -> dict:
    a, b = line(out, "gcn", "acm"), line(out, "han", "acm")
    fa = line(out, "gcn", "acm", final=True)
    (out / "cells_tuning.txt").write_text(f"# header\n{a}\n# unsupported x\n{b}\n", encoding="utf-8")
    pa, pb = parse_line(a), parse_line(b)
    recs_a = [trial("gcn", "acm", i) for i in range(3)]
    recs_b = [trial("han", "acm", 0)]
    for parsed, recs, extra in ((pa, recs_a, {}),
                                (pb, recs_b, {"line_status": "exceeds_24h",
                                              "open_attempt": {"attempt_index": 1, "starts": 2}})):
        path = parsed.state_paths()[0]
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"monitor": "macro_f1", "records": recs, "attempted_trials": len(recs),
                                    "target_trials": 100 if extra else 3, "failed_trials": 0, **extra}))
        (path.parent / "tuning_state_run0_worker_status_0.json").write_text("{}")  # not a state
    seeds = {s: {**trial("gcn", "acm", 0), "final_seed": s, "monitor": "macro_f1", "test_macro_f1": 0.5 + s / 13}
             for s in (0, 1)}
    pa.cell_dir.mkdir(parents=True)
    for s, rec in seeds.items():
        (pa.cell_dir / f"final_seed{s}.json").write_text(json.dumps(rec))
    prov = out / "logs" / "prov"
    prov.mkdir(parents=True)
    (out / "logs" / "lines").mkdir(parents=True)

    def start(job, text, before, t, gpu="NVIDIA H100"):
        return {"event": "start", "line_no": 1, "line": text, "sha": "f" * 40, "host": f"r{job}n1", "job_id": job,
                "task_id": "1", "slot": 0, "cuda_visible_devices": "0", "gpu_name": gpu, "start": t,
                "log": str(out / "logs" / "lines" / f"cells_tuning.line001.{job}.1.0.log"), "before": before}

    events = [
        start("100", a, snap(a, -1), "2026-11-20T00:00:00+09:00"),  # writes record 0, then killed: no end event
        start("200", a, snap(a, 1), "2026-11-21T00:00:00+09:00"),   # writes records 1, 2
        {"event": "end", "line_no": 1, "job_id": "200", "task_id": "1", "slot": 0,
         "start": "2026-11-21T00:00:00+09:00", "end": "x", "wall_s": 3600, "exit": 0, "after": snap(a, 3)},
        start("300", b, snap(b, 0), "2026-11-20T00:00:01+09:00"),
        {"event": "end", "line_no": 2, "job_id": "300", "task_id": "1", "slot": 0,
         "start": "2026-11-20T00:00:01+09:00", "end": "x", "wall_s": 86400, "exit": 1, "after": snap(b, 1)},
        start("400", fa, snap(fa, 3), "2026-11-22T00:00:00+09:00"),
        {"event": "end", "line_no": 1, "job_id": "400", "task_id": "1", "slot": 0,
         "start": "2026-11-22T00:00:00+09:00", "end": "x", "wall_s": 120, "exit": 0, "after": snap(fa, 3, (0, 1))},
    ]
    (prov / "all.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    for e in events:
        if e["event"] == "start":
            Path(e["log"]).write_text("log\n")
    return {"trials": {r["trial_name"]: r for r in recs_a + recs_b}, "seeds": seeds,
            "job_of_trial": {"gcn-0": "100", "gcn-1": "200", "gcn-2": "200", "han-0": "300"}}


class CollectTests(unittest.TestCase):
    def test_rows_equal_records_values_unchanged_provenance_filled(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            truth = build_tree(out)
            summary = collect.collect(out)
            with (out / "collected" / "results.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            n_records = len(truth["trials"]) + len(truth["seeds"])
            self.assertEqual(len(rows), n_records)
            self.assertEqual(summary["rows"], n_records)
            self.assertEqual(summary["rows_without_provenance"], 0)
            for r in rows:
                with self.subTest(row=r["trial_or_seed"], kind=r["kind"]):
                    src = (truth["trials"][r["trial_or_seed"]] if r["kind"] == "trial"
                           else truth["seeds"][int(r["trial_or_seed"])])
                    self.assertEqual((r["task"], r["dataset"], r["method"]), (src["task"], src["dataset"], src["model"]))
                    self.assertEqual(float(r["val_monitor"]), src["val_macro_f1"])  # exact (repr round-trip)
                    self.assertEqual(json.loads(r["test_metrics"]),
                                     {k: v for k, v in src.items() if k.startswith("test_")})
                    self.assertEqual(json.loads(r["sweep_params"]), src["sweep_params"])
                    for col in ("v3b_sha", "host", "job_id", "task_id", "gpu_name"):
                        self.assertTrue(r[col], col)
                    if r["kind"] == "trial":
                        self.assertEqual(r["job_id"], truth["job_of_trial"][r["trial_or_seed"]])
                    else:
                        self.assertEqual((r["job_id"], r["wall_time_s"]), ("400", "120"))
            walls = {r["trial_or_seed"]: r["wall_time_s"] for r in rows if r["kind"] == "trial"}
            self.assertEqual(walls, {"gcn-0": "", "gcn-1": "3600", "gcn-2": "3600", "han-0": "86400"})
            # one tar per cell, sums match
            tars = sorted(p.name for p in (out / "collected" / "cells").glob("*.tar"))
            self.assertEqual(tars, ["gcn_node_classification_acm.tar", "han_node_classification_acm.tar"])
            with tarfile.open(out / "collected" / "cells" / tars[0]) as tar:
                names = tar.getnames()
            self.assertIn("state/gcn_node_classification_acm/tuning_state_run0.json", names)
            self.assertIn("experiments/baselines/gcn_node_classification_acm/final_seed1.json", names)
            self.assertIn("logs/lines/cells_tuning.line001.200.1.0.log", names)
            for entry in (out / "collected" / "SHA256SUMS").read_text().splitlines():
                digest, name = entry.split("  ")
                self.assertEqual(hashlib.sha256((out / "collected" / name).read_bytes()).hexdigest(), digest)

    def test_status_reads_the_same_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            build_tree(out)
            rows = status.collect_status(out, out / "cells_tuning.txt")
        self.assertEqual([(r["token"], r["finished"], r["records"], r["status"], r["killed_starts"], r["final_seeds"])
                          for r in rows],
                         [("gcn_node_classification_acm", 3, 3, "done", 0, 2),
                          ("han_node_classification_acm", 1, 1, "exceeds_24h", 2, 0)])
        self.assertNotEqual(rows[0]["last_log"], "-")

    def test_line_paths_are_the_launchers(self) -> None:
        from scripts.tsubame.make_cell_list import tuning_list
        from src.baselines.campaign import Cell, build_units, state_path_for

        with tempfile.TemporaryDirectory() as d:
            for text in runnable_lines("\n".join(tuning_list("baselines-1", Path(d)))):
                parsed = parse_line(text)
                cell = Cell(model=parsed.model, task=parsed.task, dataset=parsed.dataset)
                units = build_units(parsed.task, base_seed=0, num_runs=parsed.num_runs)
                self.assertEqual(parsed.state_paths(), [state_path_for(parsed.state_dir, cell, u) for u in units])
                self.assertEqual(parsed.token, cell.token)


if __name__ == "__main__":
    unittest.main()
