"""Cluster-setup2 CS7-4: `docs/partMB_evidence/mb0_reference.py compare` exits 1 and names the tensor.

A fresh reference is saved on this machine (so the unperturbed compare is exact here), then a copy
with one tensor perturbed must exit 1 and print that tensor's name; the unperturbed copy exits 0.
The committed references are only read by the script, never written by this test.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT.parent / "docs" / "partMB_evidence" / "mb0_reference.py"
PERTURBED = "eval_logits_1"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, capture_output=True, text=True)


class Mb0CompareExitCodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ref = Path(cls.tmp.name) / "ref.pt"
        saved = run("save", str(cls.ref))
        if saved.returncode != 0:
            raise RuntimeError(f"save failed: {saved.stderr[-2000:]}")
        reference = torch.load(cls.ref, weights_only=False)
        reference[PERTURBED] = reference[PERTURBED].clone()
        reference[PERTURBED].view(-1)[0] += 1.0
        cls.perturbed = Path(cls.tmp.name) / "perturbed.pt"
        torch.save(reference, cls.perturbed)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_unperturbed_reference_exits_0(self) -> None:
        done = run("compare", str(self.ref))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-2000:])
        self.assertIn("differing: none", done.stdout)

    def test_one_perturbed_tensor_exits_1_with_its_name(self) -> None:
        done = run("compare", str(self.perturbed))
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr[-2000:])
        diff_lines = [ln.strip() for ln in done.stdout.splitlines() if ln.strip().startswith("DIFF")]
        self.assertEqual(diff_lines, [f"DIFF .{PERTURBED}"])


if __name__ == "__main__":
    unittest.main()
