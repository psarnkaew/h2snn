"""Cluster-setup CS3: vendor/VENDORS.tsv is complete and pins every vendor to a commit.

Reads the manifest only; nothing is cloned. A row of kind `unresolved` has no URL in the sources
the card names and fails `test_every_row_has_a_commit` until the author supplies one.
"""

import csv
import re
import unittest
from pathlib import Path

from src.baselines.runtime.vendor_paths import VENDOR_DIR, VENDOR_ROOTS

MANIFEST = Path(__file__).resolve().parents[1] / "vendor" / "VENDORS.tsv"
COLUMNS = ["name", "url", "commit", "licence", "used_by", "kind"]
KINDS = {"clone", "in-repo", "no-clone", "unresolved"}


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
        bad = [row["name"] for row in self.rows if not re.fullmatch(r"[0-9a-f]{7,40}", row["commit"])]
        self.assertEqual(bad, [], "rows without a 7-40 hex commit (URL not found in the sources)")

    def test_clone_rows_pin_a_full_hash_and_a_url(self) -> None:
        for row in self.rows:
            if row["kind"] == "clone":
                self.assertRegex(row["commit"], r"^[0-9a-f]{40}$", row["name"])
                self.assertRegex(row["url"], r"^https://github\.com/[\w.-]+/[\w.-]+$", row["name"])


if __name__ == "__main__":
    unittest.main()
