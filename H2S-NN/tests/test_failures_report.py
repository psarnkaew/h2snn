"""Card 10 A2.3: `scripts/tsubame/failures.py` on a synthetic output tree (the layout of sweep_slots.sh /
lab_slots.sbatch and of `src/utils/wandb_campaign.py` `record_failed_attempt`).

Two cells (a baseline and an H2S-NN line, plus the baseline's final-seeds line sharing its state); four failed trials
over two line runs (jobs 801 and 802): an OOM, a stop condition (tau), a T_max (PCGConvergenceError) and an `other`.
Every failure is one row with its condition, its job / host / GPU (the run that made that attempt) and the error's
last lines from that run's log; the state files are not changed. Mutation (`docs/card10_evidence/mutation_a23.py`):
a failure is skipped -> fails.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.tsubame import failures as F  # noqa: E402  (the mutation replaces F)
from scripts.tsubame.lines import parse_line, snapshot  # noqa: E402
from scripts.tsubame.make_cell_list import final_seeds_line, h2snn_tuning_line, tuning_line  # noqa: E402

NC = "node_classification"
OOM = "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
STOP = "StopConditionError: stop condition tau: 1.2 > 1 at epoch 3"
PCG = "PCGConvergenceError: T_max reached in scope path"
OTHER = "ValueError: index 7 is out of bounds"


def traceback(error: str) -> list[str]:
    return ["Traceback (most recent call last):", '  File "worker.py", line 1, in <module>', "    run()", error]


class FailuresTests(unittest.TestCase):
    def build(self, root: Path) -> tuple[Path, Path, dict[str, Path]]:
        out = root / "out"
        lines = [tuning_line(NC, "gcn", "acm", out), h2snn_tuning_line(NC, "dblp", out),
                 final_seeds_line(NC, "gcn", "acm", out)]
        list_path = out / "cells_tuning.txt"
        (out / "logs" / "lines").mkdir(parents=True)
        (out / "logs" / "prov").mkdir(parents=True)
        list_path.write_text("# list\n" + "\n".join(lines) + "\n")
        states = {}
        for line_no, text in ((1, lines[0]), (2, lines[1])):
            line = parse_line(text)
            state = line.state_paths()[0]
            state.parent.mkdir(parents=True, exist_ok=True)
            states[line.token] = state
            # run 1 (job 801) makes attempts 0-1, run 2 (job 802) attempts 2-3
            failures = ([{"attempt_index": 0, "return_code": 1, "error": OOM},
                         {"attempt_index": 3, "return_code": 1, "error": OTHER}] if line_no == 1 else
                        [{"attempt_index": 1, "return_code": 1, "error": STOP,
                          "failure": {"condition": "tau", "value": 1.2, "limit": 1.0, "epoch": 3}},
                         {"attempt_index": 2, "return_code": 1, "error": PCG,
                          "failure": {"condition": "T_max", "value": None, "limit": None, "epoch": 0}}])
            for job, attempted, host, gpu in ((801, 0, "r1n1", "NVIDIA H100"), (802, 2, "g3n01", "NVIDIA RTX 3090")):
                state.write_text(json.dumps({"attempted_trials": attempted, "records": [],
                                             "failures": [f for f in failures if f["attempt_index"] < attempted]}))
                log = out / "logs" / "lines" / f"cells_tuning.line{line_no:03d}.{job}.{line_no}.0.log"
                body = ["head=x", "noise"]
                for f in failures:
                    if attempted <= f["attempt_index"] < attempted + 2:
                        body += ["trial start", *traceback(f["error"]), "trial end"]
                log.write_text("\n".join(body) + "\n")
                event = {"event": "start", "line_no": line_no, "line": text, "sha": "s", "host": host,
                         "job_id": str(job), "task_id": str(line_no), "slot": 0, "gpu_name": gpu,
                         "start": f"2026-11-2{job - 800}T00:00:0{line_no}", "log": str(log), "before": snapshot(line)}
                with (out / "logs" / "prov" / f"{job}.{line_no}.0.jsonl").open("a") as handle:
                    handle.write(json.dumps(event) + "\n")
            state.write_text(json.dumps({"attempted_trials": 4, "records": [], "failures": failures}))
        return out, list_path, states

    def test_every_failure_reported_with_condition_run_and_tail(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out, list_path, states = self.build(Path(d))
            before = {k: p.read_text() for k, p in states.items()}
            with redirect_stdout(io.StringIO()) as printed:
                F.main(["--out", str(out), "--list", str(list_path)])
            with (out / "failures.tsv").open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            got = sorted((r["cell"], r["attempt"], r["condition"], r["named_condition"], r["job_id"], r["host"],
                          r["gpu_name"]) for r in rows)
            self.assertEqual(got, sorted([
                ("gcn_node_classification_acm", "0", "OOM", "", "801", "r1n1", "NVIDIA H100"),
                ("gcn_node_classification_acm", "3", "other", "", "802", "g3n01", "NVIDIA RTX 3090"),
                ("h2snn_node_classification_dblp", "1", "StopConditionError", "tau", "801", "r1n1", "NVIDIA H100"),
                ("h2snn_node_classification_dblp", "2", "PCGConvergenceError", "T_max", "802", "g3n01",
                 "NVIDIA RTX 3090")]))
            errors = {"0": OOM, "3": OTHER, "1": STOP, "2": PCG}
            for r in rows:
                tail = r["error_tail"].split(" | ")
                self.assertEqual(tail[-1], errors[r["attempt"]])           # ends at this trial's error line
                self.assertEqual(tail[0], "trial start")
                self.assertIn(f".{r['job_id']}.", r["log"])
            self.assertIn("failed trials 4", printed.getvalue())
            self.assertIn("OOM: 1", printed.getvalue())
            self.assertEqual({k: p.read_text() for k, p in states.items()}, before)   # report only


if __name__ == "__main__":
    unittest.main()
