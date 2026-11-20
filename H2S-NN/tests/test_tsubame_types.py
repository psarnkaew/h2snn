"""Card 10 A2.1: every whole-H100 TSUBAME resource type in `submit_chain.sh` + `sweep_slots.sh` (fake Grid Engine of
`tests/test_tsubame_slots.py`). For gpu_1, node_q, node_h, node_f (GPUs per task 1, 1, 2, 4; handbook 5.1.1) and lists
of 1, 3, 5 runnable lines, the chain's array is run task by task as qsub was told: every line runs exactly once and
each slot's CUDA_VISIBLE_DEVICES is its own GPU of that resource (index < GPUs per task, one line per (task, GPU) at
a time). MIG types (node_o, gpu_h) are refused with nothing submitted.
Mutation (`docs/card10_evidence/mutation_a21.py`): S of node_h set to 4 -> slots on GPUs the task does not have -> fails.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests import test_tsubame_slots as t5

GPUS = {"gpu_1": 1, "node_q": 1, "node_h": 2, "node_f": 4}


class TypeTests(unittest.TestCase):
    def chain(self, site: t5.FakeSite, list_path: Path, out: Path, kind: str):
        import subprocess
        return subprocess.run(["bash", "scripts/tsubame/submit_chain.sh", "--list", str(list_path), "--out", str(out),
                               "--type", kind, "--legs", "1"], cwd=site.h2, env=site.env(), capture_output=True,
                              text=True, timeout=120)

    def test_every_whole_h100_type_runs_each_line_once_on_its_own_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            site = t5.FakeSite(Path(d))
            for kind, gpus in GPUS.items():
                for n in (1, 3, 5):
                    with self.subTest(type=kind, lines=n):
                        site.calls.unlink(missing_ok=True)
                        list_path, lines = site.write_list(n, f"_{kind}")
                        out = site.root / f"run_{kind}_{n}"
                        proc = self.chain(site, list_path, out, kind)
                        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                        argv = json.loads(site.qsub_log.read_text().splitlines()[-1])
                        self.assertEqual(argv[argv.index("-l") + 1], f"{kind}=1")
                        first, last = (int(x) for x in argv[argv.index("-t") + 1].split("-"))
                        s, g = int(argv[-2]), int(argv[-1])
                        for task in range(first, last + 1):
                            run = site.run_slots(list_path, out, s, g, task, 9500 + task)
                            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                        calls = site.recorded()
                        self.assertEqual(sorted(" ".join(c["argv"]) for c in calls),
                                         sorted(f"{x} --resume" for x in lines))       # each line exactly once
                        for c in calls:
                            self.assertLess(int(c["cuda"]), gpus)                     # a GPU the task has
                        # G = K * S >= lines: every line has its own slot, i.e. its own (task, GPU)
                        self.assertEqual(len({(c["task"], c["cuda"]) for c in calls}), len(calls))
                        site.live.write_text("")

    def test_mig_types_refused(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            site = t5.FakeSite(Path(d))
            list_path, _ = site.write_list(3)
            for kind in ("node_o", "gpu_h"):
                with self.subTest(type=kind):
                    proc = self.chain(site, list_path, site.root / "o", kind)
                    self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                    self.assertIn("MIG", proc.stderr)
            self.assertFalse(site.qsub_log.exists() and site.qsub_log.read_text().strip())


if __name__ == "__main__":
    unittest.main()
