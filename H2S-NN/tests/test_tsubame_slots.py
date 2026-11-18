"""TSUBAME card T5: the job logic of `scripts/tsubame/sweep_slots.sh` and `submit_chain.sh`, run locally with a fake
Grid Engine (`SGE_TASK_ID`, `JOB_ID`, `qsub`, `qstat`), a fake `nvidia-smi` and a fake launcher that records its
arguments (the `python` on PATH; `scripts/tsubame/lines.py` still runs on the real interpreter).

The scripts run from `H2S-NN/` of a temporary git checkout whose directory is named by its HEAD, as on TSUBAME.
(1) slots: for node_f (S = 4) and gpu_1 (S = 1) and lists of 1, 4, 5 and 13 runnable lines (with comment and blank
    lines), every line runs exactly once over all tasks and slots, as `<line> --resume`, each with its slot's
    CUDA_VISIBLE_DEVICES; a dirty checkout or a directory not named by HEAD is refused.
    Mutation (`docs/tsubame_evidence/t5_mutation.py`): slot index without the task offset -> a line runs twice.
(2) chain: legs are held on the previous leg, `-g` is on the command line (never in the output or submitted.tsv),
    a second chain for the same list is refused while a job of the first is queued or running.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: the scripts under test (the mutation replaces SLOTS with a mutated copy)
SLOTS = REPO / "scripts" / "tsubame" / "sweep_slots.sh"
CHAIN = REPO / "scripts" / "tsubame" / "submit_chain.sh"
LINES = REPO / "scripts" / "tsubame" / "lines.py"
GROUP = "t4-test-group-xyz"


def executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class FakeSite:
    """A checkout named by its HEAD, a fake HOME (conda, ~/.t4_group) and fake commands on PATH."""

    def __init__(self, root: Path) -> None:
        self.root = root
        work = root / "repo"
        h2 = work / "H2S-NN" / "scripts" / "tsubame"
        h2.mkdir(parents=True)
        shutil.copy(SLOTS, h2 / "sweep_slots.sh")
        shutil.copy(CHAIN, h2 / "submit_chain.sh")
        shutil.copy(LINES, h2 / "lines.py")
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
        subprocess.run(["git", "init", "-q", str(work)], check=True)
        subprocess.run([*git, "-C", str(work), "add", "-A"], check=True)
        subprocess.run([*git, "-C", str(work), "commit", "-q", "-m", "fake"], check=True)
        self.sha = subprocess.run(["git", "-C", str(work), "rev-parse", "HEAD"], check=True, capture_output=True,
                                  text=True).stdout.strip()
        self.checkout = root / self.sha
        work.rename(self.checkout)
        self.h2 = self.checkout / "H2S-NN"
        self.home = root / "home"
        (self.home / "miniforge3" / "etc" / "profile.d").mkdir(parents=True)
        (self.home / "miniforge3" / "etc" / "profile.d" / "conda.sh").write_text("conda() { return 0; }\n")
        (self.home / ".t4_group").write_text(GROUP + "\n")
        self.bin = root / "bin"
        self.bin.mkdir()
        self.calls = root / "calls.jsonl"
        self.qsub_log = root / "qsub.jsonl"
        self.live = root / "live_jobs"
        self.live.write_text("")
        real_python = sys.executable
        executable(self.bin / "python", f"""#!/bin/bash
# only the launcher is faked; the slot script's own helpers (lines.py, python -c) run on the real interpreter
if [ "$1" != "scripts/run_baseline_tuning.py" ]; then exec "{real_python}" "$@"; fi
exec "{real_python}" -c 'import json, os, sys
print(json.dumps({{"argv": sys.argv[1:], "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "task": os.environ.get("SGE_TASK_ID"), "job": os.environ.get("JOB_ID")}}),
      file=open(os.environ["T4_CALLS"], "a"), flush=True)' "$@"
""")
        executable(self.bin / "nvidia-smi", """#!/bin/bash
if [ "$1" = "-i" ]; then echo "Fake H100 slot $2"; else for i in 0 1 2 3; do echo "Fake H100, 95830 MiB, 999.99"; done; fi
""")
        executable(self.bin / "qsub", f"""#!/bin/bash
n=$(( $(wc -l < "{self.qsub_log}" 2>/dev/null || echo 0) + 1 ))
id=$(( 5000 + n ))
"{real_python}" -c 'import json, sys; print(json.dumps(sys.argv[1:]), file=open(sys.argv[1], "a"))' "{self.qsub_log}" "$@"
echo "$id" >> "{self.live}"
t=""; prev=""; for a in "$@"; do [ "$prev" = "-t" ] && t="$a"; prev="$a"; done
echo "$id.${{t}}:1"
""")
        executable(self.bin / "qstat", f"""#!/bin/bash
if [ -s "{self.live}" ]; then
  echo "job-ID  prior  name  user  state  submit/start  queue  jclass  slots  ja-task-ID"
  echo "-------------------------------------------------------------------------------"
  while read -r id; do echo "$id 0.5 t4-x user qw 11/18/2026 1"; done < "{self.live}"
fi
""")

    def env(self, **extra: str) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SGE_", "SLURM_"))}
        env.pop("JOB_ID", None)
        env.update({"HOME": str(self.home), "PATH": f"{self.bin}:{env.get('PATH', '/usr/bin:/bin')}",
                    "T4_CALLS": str(self.calls)})
        env.update(extra)
        return env

    def write_list(self, n: int, tag: str = "") -> tuple[Path, list[str]]:
        out = self.root / f"out{n}{tag}"
        lines = [f"scripts/run_baseline_tuning.py --task node_classification --models m{i:02d} --datasets acm "
                 f"--num_runs 1 --base_seed 0 --n_trials 100 --output_dir {out}/experiments/baselines "
                 f"--state_dir {out}/state --wandb_project p" for i in range(n)]
        text = ["# header comment", ""]
        for i, line in enumerate(lines):
            text.append(line)
            if i % 3 == 1:
                text += ["# unsupported x y z: reason", "   "]
        path = self.root / f"cells{n}{tag}.txt"
        path.write_text("\n".join(text) + "\n")
        return path, lines

    def run_slots(self, list_path: Path, out: Path, s: int, g: int, task: int, job: int) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "scripts/tsubame/sweep_slots.sh", str(list_path), str(out), str(s), str(g)],
                              cwd=self.h2, env=self.env(SGE_TASK_ID=str(task), JOB_ID=str(job)),
                              capture_output=True, text=True, timeout=300)

    def recorded(self) -> list[dict]:
        if not self.calls.exists():
            return []
        return [json.loads(x) for x in self.calls.read_text().splitlines() if x.strip()]


class SlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.site = FakeSite(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_leg(self, n: int, s: int, tasks: int | None = None) -> tuple[list[str], list[dict]]:
        site = self.site
        site.calls.unlink(missing_ok=True)
        tag = f"_s{s}_k{tasks or 0}"  # every leg its own list and output directory
        list_path, lines = site.write_list(n, tag)
        k = tasks or math.ceil(n / s)
        g = k * s
        out = site.root / f"out{n}{tag}"
        for t in range(1, k + 1):
            proc = site.run_slots(list_path, out, s, g, t, 9000 + t)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn(f"head={site.sha}", proc.stdout)
        return lines, site.recorded()

    def check_every_line_once(self, n: int, s: int, tasks: int | None = None) -> None:
        lines, calls = self.run_leg(n, s, tasks)
        ran = [" ".join(c["argv"]) for c in calls]
        self.assertEqual(sorted(ran), sorted(f"{line} --resume" for line in lines))  # each line exactly once
        k = tasks or math.ceil(n / s)
        g = k * s
        for c in calls:
            i = lines.index(" ".join(c["argv"][:-1]))           # 0-based runnable index
            slot = i % g                                        # the global slot that owns line i
            self.assertEqual(int(c["task"]), slot // s + 1)
            self.assertEqual(c["cuda"], str(slot % s))          # the slot's own GPU
        out = self.site.root / f"out{n}_s{s}_k{tasks or 0}"
        logs = sorted((out / "logs" / "lines").glob("*.log"))
        self.assertEqual(len(logs), n)                          # one log file per line run
        prov = [json.loads(x) for p in (out / "logs" / "prov").glob("*.jsonl") for x in p.read_text().splitlines()]
        self.assertEqual(sorted(e["event"] for e in prov), ["end"] * n + ["start"] * n)
        self.assertTrue(all(e["sha"] == self.site.sha for e in prov if e["event"] == "start"))

    def test_every_line_runs_exactly_once(self) -> None:
        for s, name in ((4, "node_f"), (1, "gpu_1")):
            for n in (1, 4, 5, 13):
                with self.subTest(type=name, lines=n):
                    self.check_every_line_once(n, s)

    def test_fewer_slots_than_lines_stride_by_g(self) -> None:
        self.check_every_line_once(13, 4, tasks=2)  # G = 8: slot g runs lines g and g + 8

    def test_dirty_checkout_or_wrong_directory_is_refused(self) -> None:
        site = self.site
        list_path, _ = site.write_list(4)
        script = site.h2 / "scripts" / "tsubame" / "lines.py"
        script.write_text(script.read_text() + "\n# local edit\n")
        proc = site.run_slots(list_path, site.root / "out4", 4, 4, 1, 1)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("uncommitted changes", proc.stdout)
        subprocess.run(["git", "-C", str(site.checkout), "checkout", "-q", "--", "."], check=True)
        moved = site.root / "not-a-sha"
        site.checkout.rename(moved)
        proc = subprocess.run(["bash", "scripts/tsubame/sweep_slots.sh", str(list_path), str(site.root / "o"), "4", "4"],
                              cwd=moved / "H2S-NN", env=site.env(SGE_TASK_ID="1", JOB_ID="1"),
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("is not the SHA in the directory name", proc.stdout)
        self.assertEqual(site.recorded(), [])


class ChainTests(unittest.TestCase):
    def test_legs_held_group_on_command_line_second_chain_refused(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            site = FakeSite(Path(d))
            list_path, _ = site.write_list(13)
            out = site.root / "runout"

            def chain(*extra: str) -> subprocess.CompletedProcess:
                return subprocess.run(["bash", "scripts/tsubame/submit_chain.sh", "--list", str(list_path),
                                       "--out", str(out), *extra], cwd=site.h2, env=site.env(),
                                      capture_output=True, text=True, timeout=120)

            proc = chain("--type", "node_f", "--legs", "3")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            calls = [json.loads(x) for x in site.qsub_log.read_text().splitlines()]
            self.assertEqual(len(calls), 3)
            ids = ["5001", "5002", "5003"]
            for leg, argv in enumerate(calls):
                self.assertEqual(argv[argv.index("-g") + 1], GROUP)
                self.assertEqual(argv[argv.index("-l") + 1], "node_f=1")
                self.assertEqual(argv[argv.index("-t") + 1], "1-4")              # ceil(13 / 4) tasks
                self.assertEqual(argv[-5:], ["scripts/tsubame/sweep_slots.sh", str(list_path.resolve()),
                                             str(out.resolve()), "4", "16"])
                if leg == 0:
                    self.assertNotIn("-hold_jid", argv)
                else:
                    self.assertEqual(argv[argv.index("-hold_jid") + 1], ids[leg - 1])  # held on the previous leg
            self.assertTrue(all(i in proc.stdout for i in ids))
            tsv = (out / "submitted.tsv").read_text()
            self.assertEqual(len(tsv.splitlines()), 4)
            self.assertNotIn(GROUP, tsv + proc.stdout + proc.stderr)
            # a second chain for the same list while the first is queued: refused, nothing submitted
            again = chain("--type", "gpu_1", "--legs", "2")
            self.assertEqual(again.returncode, 3, again.stdout + again.stderr)
            self.assertIn("already has a chain", again.stderr)
            self.assertEqual(len(site.qsub_log.read_text().splitlines()), 3)
            # once its jobs are gone, a new chain (here gpu_1: 13 tasks, S = 1) is accepted
            site.live.write_text("")
            third = chain("--type", "gpu_1", "--legs", "1")
            self.assertEqual(third.returncode, 0, third.stdout + third.stderr)
            last = json.loads(site.qsub_log.read_text().splitlines()[-1])
            self.assertEqual((last[last.index("-t") + 1], last[-2:]), ("1-13", ["1", "13"]))


if __name__ == "__main__":
    unittest.main()
