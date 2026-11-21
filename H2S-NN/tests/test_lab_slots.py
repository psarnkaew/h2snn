"""Card 10 L1: the job logic of `scripts/slurm/lab_slots.sbatch` and `scripts/slurm/submit_lab.sh`, run locally with
a fake SLURM (`SLURM_ARRAY_JOB_ID`, `SLURM_ARRAY_TASK_ID`, `sbatch`, `squeue`), a fake `nvidia-smi` and a fake
launcher (the `python` on PATH: a `run_baseline_tuning.py` line appends one trial record to the sweep state that
`scripts/tsubame/lines.py` names for it; every other python call runs on the real interpreter).

The scripts run from `H2S-NN/` of a temporary git checkout whose directory is named by its HEAD, as on the lab.
(1) slots: for lists of 1, 5 and 13 runnable lines (with comment and blank lines), every line runs exactly once over
    the array, as `<line> --resume`; tasks past the last line run nothing; line logs and provenance have the
    TSUBAME names and format, and `scripts/tsubame/collect.py` joins every record to its job, task, host, GPU and
    SHA. A dirty checkout, a directory not named by HEAD and a GPU of another type are refused.
(2) submit: --gpu-type/--mem/--time are required (--time <= 72 h); legs are chained with afterany; cells are recorded
    with the GPU type; a cell recorded with another GPU type is refused (condition 1; nothing is submitted); a second
    chain for a list with a pending job is refused.
    Mutation (`docs/card10_evidence/mutation_l1.py`): the GPU-type record is not checked -> test (2) fails.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.tsubame import collect  # noqa: E402

#: the scripts under test (the mutation replaces SUBMIT with a mutated copy)
SLOTS = REPO / "scripts" / "slurm" / "lab_slots.sbatch"
SUBMIT = REPO / "scripts" / "slurm" / "submit_lab.sh"
LINES = REPO / "scripts" / "tsubame" / "lines.py"


def executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class FakeLab:
    """A checkout named by its HEAD, a fake HOME (conda) and fake SLURM / nvidia-smi / launcher on PATH."""

    def __init__(self, root: Path, gpu_name: str = "NVIDIA GeForce RTX 3090") -> None:
        self.root = root
        work = root / "repo"
        (work / "H2S-NN" / "scripts" / "slurm").mkdir(parents=True)
        (work / "H2S-NN" / "scripts" / "tsubame").mkdir(parents=True)
        shutil.copy(SLOTS, work / "H2S-NN" / "scripts" / "slurm" / "lab_slots.sbatch")
        shutil.copy(SUBMIT, work / "H2S-NN" / "scripts" / "slurm" / "submit_lab.sh")
        shutil.copy(LINES, work / "H2S-NN" / "scripts" / "tsubame" / "lines.py")
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
        self.bin = root / "bin"
        self.bin.mkdir()
        self.calls = root / "calls.jsonl"
        self.sbatch_log = root / "sbatch.jsonl"
        self.live = root / "live_jobs"
        self.live.write_text("")
        real = sys.executable
        executable(self.bin / "python", f"""#!/bin/bash
# only the launcher is faked; lines.py, python -c and the submit script's checks run on the real interpreter
if [ "$1" != "scripts/run_baseline_tuning.py" ]; then exec "{real}" "$@"; fi
exec "{real}" -c 'import json, os, shlex, sys
sys.path.insert(0, os.getcwd())
from scripts.tsubame.lines import parse_line
argv = sys.argv[1:]
print(json.dumps({{"argv": argv, "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "task": os.environ.get("SLURM_ARRAY_TASK_ID"), "job": os.environ.get("SLURM_ARRAY_JOB_ID")}}),
      file=open(os.environ["LAB_CALLS"], "a"), flush=True)
line = parse_line(shlex.join([a for a in argv if a != "--resume"]))
path = line.state_paths()[0]
path.parent.mkdir(parents=True, exist_ok=True)
state = json.loads(path.read_text()) if path.exists() else {{"monitor": "val_macro_f1", "records": []}}
state["records"].append({{"model": line.model, "task": line.task, "dataset": line.dataset,
                         "trial_name": "t%d" % len(state["records"]), "val_macro_f1": 0.5}})
path.write_text(json.dumps(state))' "$@"
""")
        self.set_gpu(gpu_name)
        executable(self.bin / "sbatch", f"""#!/bin/bash
n=$(( $(wc -l < "{self.sbatch_log}" 2>/dev/null || echo 0) + 1 ))
id=$(( 7000 + n ))
"{real}" -c 'import json, sys; print(json.dumps(sys.argv[2:]), file=open(sys.argv[1], "a"))' "{self.sbatch_log}" "$@"
echo "$id" >> "{self.live}"
echo "$id"
""")
        executable(self.bin / "squeue", f"""#!/bin/bash
[ "$1" = "-h" ] || echo "JOBID"
cat "{self.live}"
""")

    def set_gpu(self, name: str) -> None:
        executable(self.bin / "nvidia-smi", f"""#!/bin/bash
if [ "$1" = "-i" ]; then echo "{name}"; else echo "{name}, 24576 MiB, 999.99"; fi
""")

    def env(self, **extra: str) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SGE_", "SLURM_"))}
        env.pop("JOB_ID", None)
        env.update({"HOME": str(self.home), "PATH": f"{self.bin}:{env.get('PATH', '/usr/bin:/bin')}",
                    "LAB_CALLS": str(self.calls), "CUDA_VISIBLE_DEVICES": "0"})
        env.update(extra)
        return env

    def write_list(self, n: int, tag: str = "", models: str = "m") -> tuple[Path, Path, list[str]]:
        out = self.root / f"out{n}{tag}"
        lines = [f"scripts/run_baseline_tuning.py --task node_classification --models {models}{i:02d} --datasets acm "
                 f"--num_runs 1 --base_seed 0 --n_trials 100 --output_dir {out}/experiments/baselines "
                 f"--state_dir {out}/state --wandb_project p" for i in range(n)]
        text = ["# header comment", ""]
        for i, line in enumerate(lines):
            text.append(line)
            if i % 3 == 1:
                text += ["# unsupported x y z: reason", "   "]
        path = self.root / f"cells{n}{tag}.txt"
        path.write_text("\n".join(text) + "\n")
        return path, out, lines

    def run_task(self, list_path: Path, out: Path, task: int, job: int, gpu_type: str = "3090",
                 cwd: Path | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "scripts/slurm/lab_slots.sbatch", str(list_path), str(out), gpu_type],
                              cwd=cwd or self.h2, env=self.env(SLURM_ARRAY_TASK_ID=str(task),
                                                               SLURM_ARRAY_JOB_ID=str(job)),
                              capture_output=True, text=True, timeout=300)

    def submit(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "scripts/slurm/submit_lab.sh", *args], cwd=self.h2, env=self.env(),
                              capture_output=True, text=True, timeout=120)

    def recorded(self) -> list[dict]:
        if not self.calls.exists():
            return []
        return [json.loads(x) for x in self.calls.read_text().splitlines() if x.strip()]

    def sbatch_calls(self) -> list[list[str]]:
        if not self.sbatch_log.exists():
            return []
        return [json.loads(x) for x in self.sbatch_log.read_text().splitlines() if x.strip()]


class SlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.lab = FakeLab(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def check_every_line_once(self, n: int) -> None:
        lab = self.lab
        lab.calls.unlink(missing_ok=True)
        list_path, out, lines = lab.write_list(n)
        job = 9100 + n
        for t in range(1, n + 3):                                   # two tasks past the last line
            proc = lab.run_task(list_path, out, t, job)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn(f"head={lab.sha}", proc.stdout)
        calls = lab.recorded()
        self.assertEqual(sorted(" ".join(c["argv"]) for c in calls), sorted(f"{x} --resume" for x in lines))
        for c in calls:                                             # task t ran runnable line t
            self.assertEqual(lines.index(" ".join(c["argv"][:-1])) + 1, int(c["task"]))
            self.assertEqual(c["cuda"], "0")
        # TSUBAME names: <list>.line<NNN>.<job>.<task>.0.log and <job>.<task>.0.jsonl
        logs = sorted(p.name for p in (out / "logs" / "lines").glob("*.log"))
        self.assertEqual(logs, [f"{list_path.stem}.line{t:03d}.{job}.{t}.0.log" for t in range(1, n + 1)])
        provs = sorted(p.name for p in (out / "logs" / "prov").glob("*.jsonl"))
        self.assertEqual(provs, sorted(f"{job}.{t}.0.jsonl" for t in range(1, n + 1)))
        starts, ends = collect.load_events(out)
        self.assertEqual(len(starts), n)
        self.assertEqual(len(ends), n)
        for e in starts:
            self.assertEqual((e["sha"], e["job_id"], e["slot"], e["site"], e["gpu_type"]),
                             (lab.sha, str(job), 0, "lab", "3090"))
            self.assertEqual(e["gpu_name"], "NVIDIA GeForce RTX 3090")
            self.assertEqual(e["task_id"], str(e["line_no"]))
            self.assertIsInstance(e["before"], dict)
        # collect.py joins every record to its line run
        rows = list(collect.iter_rows(out))
        self.assertEqual(len(rows), n)
        for r in rows:
            self.assertEqual((r["v3b_sha"], r["job_id"], r["gpu_name"]), (lab.sha, str(job), "NVIDIA GeForce RTX 3090"))
            self.assertNotEqual(r["wall_time_s"], "")
            self.assertEqual(lines.index(next(x for x in lines if f"--models {r['method']} " in x)) + 1,
                             int(r["task_id"]))

    def test_every_line_runs_exactly_once(self) -> None:
        for n in (1, 5, 13):
            with self.subTest(lines=n):
                self.check_every_line_once(n)

    def test_dirty_wrong_directory_or_wrong_gpu_is_refused(self) -> None:
        lab = self.lab
        list_path, out, _ = lab.write_list(4)
        proc = lab.run_task(list_path, out, 1, 1, gpu_type="titan_rtx")          # GPU is a 3090
        self.assertEqual(proc.returncode, 2)
        self.assertIn("is not of type titan_rtx", proc.stdout)
        script = lab.h2 / "scripts" / "tsubame" / "lines.py"
        script.write_text(script.read_text() + "\n# local edit\n")
        proc = lab.run_task(list_path, out, 1, 1)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("uncommitted changes", proc.stdout)
        subprocess.run(["git", "-C", str(lab.checkout), "checkout", "-q", "--", "."], check=True)
        moved = lab.root / "not-a-sha"
        lab.checkout.rename(moved)
        proc = lab.run_task(list_path, out, 1, 1, cwd=moved / "H2S-NN")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("is not the SHA in the directory name", proc.stdout)
        self.assertEqual(lab.recorded(), [])


class SubmitTests(unittest.TestCase):
    def test_required_arguments_and_time_limit(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lab = FakeLab(Path(d))
            list_path, out, _ = lab.write_list(3)
            base = ["--list", str(list_path), "--out", str(out)]
            for args in (["--mem", "8G", "--time", "1:00:00"],                         # no --gpu-type
                         ["--gpu-type", "3090", "--time", "1:00:00"],                  # no --mem
                         ["--gpu-type", "3090", "--mem", "8G"],                        # no --time
                         ["--gpu-type", "v100", "--mem", "8G", "--time", "1:00:00"],
                         ["--gpu-type", "3090", "--mem", "8G", "--time", "72:00:01"],
                         ["--gpu-type", "3090", "--mem", "8G", "--time", "3-00:00:01"]):
                with self.subTest(args=args):
                    proc = lab.submit(*base, *args)
                    self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
            self.assertEqual(lab.sbatch_calls(), [])
            self.assertEqual(lab.submit(*base, "--gpu-type", "3090", "--mem", "8G", "--time", "3-00:00:00")
                             .returncode, 0)

    def test_chain_gpu_type_record_and_refusals(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            lab = FakeLab(Path(d))
            list_path, out, lines = lab.write_list(5)
            proc = lab.submit("--list", str(list_path), "--out", str(out), "--gpu-type", "3090",
                              "--mem", "48G", "--time", "72:00:00", "--legs", "3")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            calls = lab.sbatch_calls()
            self.assertEqual(len(calls), 3)
            for leg, argv in enumerate(calls):
                self.assertIn("--gres=gpu:3090:1", argv)
                self.assertIn("--mem=48G", argv)
                self.assertIn("--time=72:00:00", argv)
                self.assertIn("--array=1-5", argv)
                self.assertEqual(argv[-4:], ["scripts/slurm/lab_slots.sbatch", str(list_path.resolve()),
                                             str(out.resolve()), "3090"])
                deps = [a for a in argv if a.startswith("--dependency=")]
                self.assertEqual(deps, [] if leg == 0 else [f"--dependency=afterany:{7000 + leg}"])
            with (out / "lab_gpu_types.tsv").open(newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([(r["cell"], r["gpu_type"]) for r in rows],
                             [(f"m{i:02d}_node_classification_acm", "3090") for i in range(5)])
            self.assertEqual(len((out / "submitted.tsv").read_text().splitlines()), 4)
            # a second chain for the same list while the first is pending: refused, nothing submitted
            again = lab.submit("--list", str(list_path), "--out", str(out), "--gpu-type", "3090",
                               "--mem", "48G", "--time", "1:00:00")
            self.assertEqual(again.returncode, 3, again.stdout + again.stderr)
            self.assertIn("already has a chain", again.stderr)
            lab.live.write_text("")
            # condition 1: a list holding a recorded cell (its final seeds line) on another GPU type: refused
            seeds = lab.root / "cells_seeds.txt"
            seeds.write_text(lines[2].replace("--n_trials 100", "--final_seeds 0 1 2 3 4") + "\n")
            other = lab.submit("--list", str(seeds), "--out", str(out), "--gpu-type", "titan_rtx",
                               "--mem", "48G", "--time", "1:00:00")
            self.assertEqual(other.returncode, 3, other.stdout + other.stderr)
            self.assertIn("m02_node_classification_acm is recorded with GPU type 3090", other.stderr)
            self.assertEqual(len(lab.sbatch_calls()), 3)
            # the same cell on its own GPU type is accepted and not recorded twice
            same = lab.submit("--list", str(seeds), "--out", str(out), "--gpu-type", "3090",
                              "--mem", "48G", "--time", "1:00:00")
            self.assertEqual(same.returncode, 0, same.stdout + same.stderr)
            self.assertEqual(len((out / "lab_gpu_types.tsv").read_text().splitlines()), 6)


if __name__ == "__main__":
    unittest.main()
