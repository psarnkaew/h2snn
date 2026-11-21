"""Card 10 A3.3: the slot scripts pass `--resume` only to the launchers that define it.

Both `scripts/tsubame/sweep_slots.sh` (fake Grid Engine, `tests/test_tsubame_slots.FakeSite`) and
`scripts/slurm/lab_slots.sbatch` (fake SLURM, `tests/test_lab_slots.FakeLab`) run one list with a baseline tuning
line, an H2S-NN tuning line, a card 7 `run_row.py` line and a card 8 published line, through a fake `python` that
records the argv of every launcher call (`scripts/tsubame/lines.py` and `python -c` still run on the real
interpreter). The tuning and `run_row.py` lines get `--resume` as their last word; the published line runs exactly as
the list states it (`scripts/run_baseline_published.py` has no `--resume`: argparse would refuse it, so every
published line failed at start on 2d822a4).
Mutation (`docs/card10_evidence/mutation_a33.py`): `--resume` appended unconditionally -> fails on both scripts.
"""

from __future__ import annotations

import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests import test_lab_slots, test_tsubame_slots  # noqa: E402


def lines_for(out: Path) -> dict[str, str]:
    return {
        "baseline": (f"scripts/run_baseline_tuning.py --task node_classification --models gcn --datasets acm "
                     f"--num_runs 1 --base_seed 0 --n_trials 100 --output_dir {out}/experiments/baselines "
                     f"--state_dir {out}/state --wandb_project p"),
        "h2snn": (f"scripts/run_h2snn_wandb_tuning.py --config configs/acm_h2snn_v3b.yaml "
                  f"--exp_name h2snn_node_classification_acm --output_dir {out}/experiments/h2snn "
                  f"--state_dir {out}/state/h2snn_acm --n_trials 100 --wandb_project p"),
        "run_row": f"scripts/tsubame/run_row.py --done {out}/done/row1.json -- scripts/perturbation/run_checked.py --x 1",
        "published": (f"scripts/run_baseline_published.py --task link_prediction --models gcn --datasets lastfm "
                      f"--final_seeds 0 1 2 3 4 --output_dir {out}/experiments/baselines --state_dir {out}/state "
                      f"--wandb_project p"),
    }


#: the launcher calls are recorded and not run; every other python call runs on the real interpreter
FAKE_PYTHON = """#!/bin/bash
case "$1" in
    scripts/run_*.py|scripts/tsubame/run_row.py)
        exec "{real}" -c 'import json, os, sys
print(json.dumps({{"argv": sys.argv[1:]}}), file=open(os.environ["{env}"], "a"), flush=True)' "$@";;
esac
exec "{real}" "$@"
"""


class ResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def check(self, calls: list[dict], lines: dict[str, str]) -> None:
        ran = {" ".join(c["argv"][:1]): c["argv"] for c in calls}
        self.assertEqual(len(calls), len(lines), calls)                    # every line ran once, none skipped
        for kind, line in lines.items():
            with self.subTest(line=kind):
                argv = ran[shlex.split(line)[0]]
                want = shlex.split(line) + ([] if kind == "published" else ["--resume"])
                self.assertEqual(argv, want)

    def write_list(self, lines: dict[str, str]) -> Path:
        path = self.root / "cells_mixed.txt"
        path.write_text("# mixed list\n" + "\n".join(lines.values()) + "\n", encoding="utf-8")
        return path

    def test_tsubame_slots(self) -> None:
        site = test_tsubame_slots.FakeSite(self.root / "t4")
        test_tsubame_slots.executable(site.bin / "python", FAKE_PYTHON.format(real=sys.executable, env="T4_CALLS"))
        out = site.root / "out"
        lines = lines_for(out)
        proc = site.run_slots(self.write_list(lines), out, 4, 4, 1, 9301)   # node_f: one slot per line
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.check(site.recorded(), lines)
        logs = "".join(p.read_text() for p in (out / "logs" / "lines").glob("*.log"))
        self.assertIn(f"python {lines['published']}\n", logs)               # the logged command has no --resume

    def test_lab_slots(self) -> None:
        lab = test_lab_slots.FakeLab(self.root / "lab")
        test_lab_slots.executable(lab.bin / "python", FAKE_PYTHON.format(real=sys.executable, env="LAB_CALLS"))
        out = lab.root / "out"
        lines = lines_for(out)
        list_path = self.write_list(lines)
        for task in range(1, len(lines) + 1):
            proc = lab.run_task(list_path, out, task, 9302)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.check(lab.recorded(), lines)
        logs = "".join(p.read_text() for p in (out / "logs" / "lines").glob("*.log"))
        self.assertIn(f"python {lines['published']}\n", logs)


if __name__ == "__main__":
    unittest.main()
