"""Part M3b B1: scripts/setup/apply_vendor_patches.sh on a throwaway git repository.

Cases: a clean clone at the pinned commit gets the patch (exit 0, file changed); a second run reports
"already applied" and changes nothing (exit 0, same bytes); a clone whose file conflicts with the patch is
refused (exit 1, unchanged); a clone at another commit is refused (exit 1). Card mutation: the script
reapplies an applied patch -> `test_second_run_is_a_no_op` fails. Also checks that the recorded H2GB
patch touches exactly the two files the card names.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "setup" / "apply_vendor_patches.sh"
ORIGINAL = "line one\nline two\nline three\n"
PATCHED = "line one\nline 2 (patched)\nline three\n"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipIf(shutil.which("git") is None or shutil.which("bash") is None, "needs git and bash")
class ApplyVendorPatchesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.vendor = self.tmp / "vendor"
        self.clone = self.vendor / "Toy"
        self.clone.mkdir(parents=True)
        _git(self.clone, "init", "-q")
        _git(self.clone, "config", "user.email", "test@example.invalid")
        _git(self.clone, "config", "user.name", "test")
        (self.clone / "a.py").write_text(ORIGINAL)
        (self.clone / "untouched.py").write_text("keep me\n")
        _git(self.clone, "add", "a.py", "untouched.py")
        _git(self.clone, "commit", "-q", "-m", "pinned")
        self.commit = _git(self.clone, "rev-parse", "HEAD")
        # the patch = git diff of the edited clone, then the clone is reset to the pinned file
        (self.clone / "a.py").write_text(PATCHED)
        self.patches = self.tmp / "patches"
        self.patches.mkdir()
        (self.patches / "Toy.patch").write_text(_git(self.clone, "diff", "--no-color") + "\n")
        (self.clone / "a.py").write_text(ORIGINAL)
        self.manifest = self.tmp / "VENDORS.tsv"
        self.manifest.write_text("name\turl\tcommit\tlicence\tused_by\tkind\n"
                                 f"Toy\thttps://example.invalid/toy\t{self.commit}\tMIT\tnone\tclone\n")

    def _run(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SCRIPT), "--vendor-dir", str(self.vendor), "--patch-dir", str(self.patches),
             "--manifest", str(self.manifest)], capture_output=True, text=True)

    def test_clean_clone_gets_the_patch(self) -> None:
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("APPLIED", result.stdout)
        self.assertEqual((self.clone / "a.py").read_text(), PATCHED)
        self.assertEqual((self.clone / "untouched.py").read_text(), "keep me\n")

    def test_second_run_is_a_no_op(self) -> None:
        self.assertEqual(self._run().returncode, 0)
        before = {p.name: _sha(p) for p in self.clone.glob("*.py")}
        status_before = _git(self.clone, "status", "--porcelain")
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ALREADY APPLIED", result.stdout)
        self.assertEqual({p.name: _sha(p) for p in self.clone.glob("*.py")}, before)
        self.assertEqual(_git(self.clone, "status", "--porcelain"), status_before)

    def test_conflicting_clone_is_refused_unchanged(self) -> None:
        conflicting = "line one\nsomething else\nline three\n"
        (self.clone / "a.py").write_text(conflicting)
        result = self._run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("CONFLICT", result.stderr)
        self.assertEqual((self.clone / "a.py").read_text(), conflicting)

    def test_clone_at_another_commit_is_refused(self) -> None:
        (self.clone / "untouched.py").write_text("moved on\n")
        _git(self.clone, "commit", "-q", "-am", "another commit")
        result = self._run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("REFUSED", result.stderr)
        self.assertEqual((self.clone / "a.py").read_text(), ORIGINAL)


class RecordedPatchTests(unittest.TestCase):
    def test_h2gb_patch_touches_exactly_the_two_dataset_files(self) -> None:
        text = (REPO / "vendor" / "patches" / "H2GB.patch").read_text()
        files = sorted(line.split(" b/", 1)[1] for line in text.splitlines()
                       if line.startswith("diff --git "))
        self.assertEqual(files, ["H2GB/datasets/ieee_cis_dataset.py", "H2GB/datasets/pokec_dataset.py"])


if __name__ == "__main__":
    unittest.main()
