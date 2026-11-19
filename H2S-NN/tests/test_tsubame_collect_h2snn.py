"""TSUBAME card T7.2: status and collection of `h2snn` lines on a synthetic H2S-NN output tree (nothing trained).

The tree follows `scripts/run_h2snn_wandb_tuning.py` (paths checked against it in `test_tsubame_h2snn_lines.py`):
`<out>/state/<token>/tuning_state_run0.json` (records as its train_trial / enrich_run_record write them: `dataset`,
`task_type`, `exp_name`, no `model`), `<out>/experiments/h2snn/<token>/final_seed<s>.json`, and the per-trial run
directories `<out>/experiments/h2snn/<token>_tune_run0_<trial>/` its train_trial writes (not cells).
Checks: rows = records (trial / final_seed), method `h2snn`, task and dataset of the cell, values copied unchanged,
provenance filled; one tar per cell (no trial run directory archived); status reads the same files.
Mutation (`docs/tsubame_evidence/t7_2_mutation.py`): an `h2snn` record skipped -> this test fails.
"""

from __future__ import annotations

import csv
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.tsubame import collect, status
from scripts.tsubame.lines import parse_line
from scripts.tsubame.make_cell_list import h2snn_final_seeds_line, h2snn_tuning_line

CELLS = (("node_classification", "acm", "macro_f1"), ("link_prediction", "ogbn-mag_pp", "auroc"))


def record(task: str, dataset: str, monitor: str, i: int, token: str) -> dict:
    return {"run_index": 0, "num_runs": 1, "seed": 0, "dataset": dataset, "task_type": task,
            "monitor": f"val_{monitor}", "exp_name": f"{token}_tune_run0_t{i}", "trial_name": f"{token}-t{i}",
            f"val_{monitor}": 0.3 + i / 17, f"test_{monitor}": 0.25 + i / 19, "best_epoch": 3 + i,
            "sweep_params": {"d": 32, "lr": 10 ** -(i + 3)}, "wandb_run_id": f"w{i}{dataset}"}


def snap(text: str, records: int, seeds=()) -> dict:
    parsed = parse_line(text)
    return {"token": parsed.token, "kind": parsed.kind, "final_seed_files": list(seeds),
            "units": [{"state": str(p), "exists": records >= 0, "records": max(records, 0)}
                      for p in parsed.state_paths()]}


def build_tree(out: Path) -> dict:
    lines = {dataset: h2snn_tuning_line(task, dataset, out) for task, dataset, _ in CELLS}
    (out / "cells_tuning_h2snn.txt").write_text("# h2snn\n" + "\n".join(lines.values()) + "\n")
    truth: dict = {"trials": {}, "seeds": {}, "job_of": {}}
    events = []
    for n_records, (task, dataset, monitor) in zip((3, 1), CELLS):
        line = parse_line(lines[dataset])
        recs = [record(task, dataset, monitor, i, line.token) for i in range(n_records)]
        path = line.state_paths()[0]
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"monitor": monitor, "base_exp_name": line.token, "records": recs,
                                    "attempted_trials": n_records, "target_trials": n_records}))
        (path.parent / "tuning_state_run0_worker_context.json").write_text("{}")
        for r in recs:
            truth["trials"][r["trial_name"]] = (task, dataset, r)
            trial_dir = line.output_dir / r["exp_name"]           # the trial's own run directory
            trial_dir.mkdir(parents=True)
            (trial_dir / "best_test_metrics.json").write_text("{}")
        job = "700" if dataset == "acm" else "800"
        t = f"2026-11-2{1 if dataset == 'acm' else 2}T00:00:00+09:00"
        events += [
            {"event": "start", "line": lines[dataset], "sha": "e" * 40, "host": f"r{job}", "job_id": job,
             "task_id": "1", "slot": 0, "gpu_name": "NVIDIA H100", "start": t,
             "log": str(out / "logs" / "lines" / f"cells_tuning_h2snn.line00{1 if dataset == 'acm' else 2}.{job}.1.0.log"),
             "before": snap(lines[dataset], -1)},
            {"event": "end", "job_id": job, "task_id": "1", "slot": 0, "start": t, "wall_s": 7200, "exit": 0,
             "after": snap(lines[dataset], n_records)},
        ]
        for r in recs:
            truth["job_of"][r["trial_name"]] = job
    # final seeds of the acm cell
    task, dataset, monitor = CELLS[0]
    final = h2snn_final_seeds_line(task, dataset, out)
    fline = parse_line(final)
    fline.cell_dir.mkdir(parents=True)   # summary_dir = output_dir / exp_name (the launcher's run_final_seeds)
    for s in (0, 1):
        rec = {**record(task, dataset, monitor, 0, fline.token), "final_seed": s, "trial_name": f"final_seed{s}"}
        (fline.cell_dir / f"final_seed{s}.json").write_text(json.dumps(rec))
        truth["seeds"][s] = (task, dataset, rec)
    (fline.cell_dir / "final_seeds_summary.json").write_text("{}")   # not a seed record
    events += [
        {"event": "start", "line": final, "sha": "e" * 40, "host": "r900", "job_id": "900", "task_id": "2",
         "slot": 1, "gpu_name": "NVIDIA H100", "start": "2026-11-23T00:00:00+09:00",
         "log": str(out / "logs" / "lines" / "cells_final_seeds_h2snn.line001.900.2.1.log"),
         "before": snap(final, 3)},
        {"event": "end", "job_id": "900", "task_id": "2", "slot": 1, "start": "2026-11-23T00:00:00+09:00",
         "wall_s": 600, "exit": 0, "after": snap(final, 3, (0, 1))},
    ]
    (out / "logs" / "prov").mkdir(parents=True)
    (out / "logs" / "lines").mkdir(parents=True)
    (out / "logs" / "prov" / "h.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    for e in events:
        if e["event"] == "start":
            Path(e["log"]).write_text("log\n")
    return truth


class H2SNNCollectTests(unittest.TestCase):
    def test_h2snn_rows_equal_records_values_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            truth = build_tree(out)
            summary = collect.collect(out)
            with (out / "collected" / "results.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            n = len(truth["trials"]) + len(truth["seeds"])
            self.assertEqual((len(rows), summary["rows"], summary["rows_without_provenance"]), (n, n, 0))
            for r in rows:
                with self.subTest(kind=r["kind"], id=r["trial_or_seed"]):
                    task, dataset, src = (truth["trials"][r["trial_or_seed"]] if r["kind"] == "trial"
                                          else truth["seeds"][int(r["trial_or_seed"])])
                    self.assertEqual((r["task"], r["dataset"], r["method"]), (task, dataset, "h2snn"))
                    self.assertEqual(dataset, src["dataset"])
                    monitor = src["monitor"]
                    self.assertEqual((r["monitor"], float(r["val_monitor"])), (monitor, src[monitor]))
                    self.assertEqual(json.loads(r["test_metrics"]),
                                     {k: v for k, v in src.items() if k.startswith("test_")})
                    self.assertEqual(json.loads(r["sweep_params"]), src["sweep_params"])
                    for col in ("v3b_sha", "host", "job_id", "task_id", "gpu_name"):
                        self.assertTrue(r[col], col)
                    expected_job = truth["job_of"][r["trial_or_seed"]] if r["kind"] == "trial" else "900"
                    self.assertEqual(r["job_id"], expected_job)
            tars = sorted(p.name for p in (out / "collected" / "cells").glob("*.tar"))
            self.assertEqual(tars, ["h2snn_link_prediction_ogbn-mag_pp.tar", "h2snn_node_classification_acm.tar"])
            with tarfile.open(out / "collected" / "cells" / "h2snn_node_classification_acm.tar") as tar:
                names = tar.getnames()
            self.assertIn("state/h2snn_node_classification_acm/tuning_state_run0.json", names)
            self.assertIn("experiments/h2snn/h2snn_node_classification_acm/final_seed1.json", names)
            self.assertFalse([x for x in names if "_tune_run0_" in x], "a trial run directory was archived")

    def test_status_reads_h2snn_lines(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            build_tree(out)
            rows = status.collect_status(out, out / "cells_tuning_h2snn.txt")
        self.assertEqual([(r["token"], r["finished"], r["records"], r["status"], r["final_seeds"]) for r in rows],
                         [("h2snn_node_classification_acm", 3, 3, "done", 2),
                          ("h2snn_link_prediction_ogbn-mag_pp", 1, 1, "done", 0)])
        self.assertTrue(all(r["last_log"] != "-" for r in rows))


if __name__ == "__main__":
    unittest.main()
