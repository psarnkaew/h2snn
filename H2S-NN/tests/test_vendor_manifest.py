"""Cluster-setup CS3: vendor/VENDORS.tsv is complete and pins every vendor to a commit.

Reads the manifest only; nothing is cloned. A row of kind `unresolved` has no URL in the sources
the card names and fails `test_every_row_has_a_commit` until the author supplies one.
Cluster-setup2 CS7-2: kind `not-needed` (no adapter, test or document uses the vendor; author
decision) has no URL and no commit; every `kind=clone` directory is git-ignored
(`VendorIgnoreTests`, driven by the manifest through `git check-ignore`).
"""

import csv
import re
import subprocess
import unittest
from pathlib import Path

from src.baselines.runtime.vendor_paths import VENDOR_DIR, VENDOR_ROOTS

MANIFEST = Path(__file__).resolve().parents[1] / "vendor" / "VENDORS.tsv"
COLUMNS = ["name", "url", "commit", "licence", "used_by", "kind"]
KINDS = {"clone", "in-repo", "no-clone", "unresolved", "not-needed"}
REPO = MANIFEST.parents[1]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"manifest columns {reader.fieldnames} != {COLUMNS}")
        rows = list(reader)
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"row with a wrong number of fields: {row}")
    return rows


class VendorManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = read_manifest(MANIFEST)

    def test_manifest_parses_with_unique_names_and_known_kinds(self) -> None:
        names = [row["name"] for row in self.rows]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual({row["kind"] for row in self.rows} - KINDS, set())

    def test_every_vendor_root_has_a_row(self) -> None:
        names = {row["name"] for row in self.rows}
        top_dirs = {path.relative_to(VENDOR_DIR).parts[0] for path in VENDOR_ROOTS.values()}
        self.assertEqual(top_dirs - names, set())

    def test_every_row_has_a_commit(self) -> None:
        bad = [row["name"] for row in self.rows
               if row["kind"] != "not-needed" and not re.fullmatch(r"[0-9a-f]{7,40}", row["commit"])]
        self.assertEqual(bad, [], "rows without a 7-40 hex commit (URL not found in the sources)")

    def test_clone_rows_pin_a_full_hash_and_a_url(self) -> None:
        for row in self.rows:
            if row["kind"] == "clone":
                self.assertRegex(row["commit"], r"^[0-9a-f]{40}$", row["name"])
                self.assertRegex(row["url"], r"^https://github\.com/[\w.-]+/[\w.-]+$", row["name"])

    def test_not_needed_rows_carry_no_url_or_commit(self) -> None:
        for row in self.rows:
            if row["kind"] == "not-needed":
                self.assertEqual((row["url"], row["commit"]), ("-", "-"), row["name"])


def ignore_problems(rows: list[dict[str, str]], repo: Path = REPO) -> list[str]:
    """Every kind=clone directory is ignored; in-repo vendors and VENDORS.tsv are not."""

    def ignored(rel: str) -> bool:
        # --no-index: decide from the ignore rules alone, also for paths that are tracked
        done = subprocess.run(["git", "check-ignore", "-q", "--no-index", rel], cwd=repo)
        if done.returncode not in (0, 1):
            raise RuntimeError(f"git check-ignore failed ({done.returncode}) for {rel}")
        return done.returncode == 0

    problems = []
    for row in rows:
        probe = f"vendor/{row['name']}/ignore_probe.txt"
        if row["kind"] == "clone" and not ignored(probe):
            problems.append(f"clone not ignored: {row['name']}")
        if row["kind"] == "in-repo" and ignored(probe):
            problems.append(f"in-repo vendor ignored: {row['name']}")
    if ignored("vendor/VENDORS.tsv"):
        problems.append("vendor/VENDORS.tsv ignored")
    return problems


class VendorIgnoreTests(unittest.TestCase):
    def test_every_clone_directory_is_ignored(self) -> None:
        self.assertEqual(ignore_problems(read_manifest(MANIFEST)), [])


if __name__ == "__main__":
    unittest.main()
