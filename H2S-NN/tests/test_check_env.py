"""Cluster-setup CS2: the lock comparator on fake installed mappings (nothing is installed)."""

import unittest
from pathlib import Path

from scripts.setup.check_env import DEFAULT_LOCK, compare, parse_lock

LOCK = """
# comment
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.6.0
-f https://data.pyg.org/whl/torch-2.6.0+cu124.html
torch_geometric==2.8.0   # trailing comment
numpy==2.4.6
dgl==2.5.0+cu124; sys_platform == "linux"
"""


class CompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.locked = parse_lock(LOCK)

    def test_parse_normalises_names_and_drops_options(self) -> None:
        self.assertEqual(
            self.locked,
            {"torch": "2.6.0", "torch-geometric": "2.8.0", "numpy": "2.4.6", "dgl": "2.5.0+cu124"},
        )

    def test_equal(self) -> None:
        installed = {"torch": "2.6.0", "torch_geometric": "2.8.0", "numpy": "2.4.6", "dgl": "2.5.0+cu124"}
        self.assertEqual(compare(self.locked, installed), [])

    def test_missing(self) -> None:
        installed = {"torch": "2.6.0", "torch-geometric": "2.8.0", "numpy": "2.4.6"}
        findings = compare(self.locked, installed)
        self.assertEqual([(f.name, f.kind) for f in findings], [("dgl", "missing")])
        self.assertTrue(findings[0].is_difference)

    def test_different(self) -> None:
        installed = {"torch": "2.6.0", "Torch-Geometric": "2.8.0", "numpy": "2.3.0", "dgl": "2.5.0+cu124"}
        findings = compare(self.locked, installed)
        self.assertEqual([(f.name, f.kind, f.installed) for f in findings], [("numpy", "different", "2.3.0")])

    def test_extra_local_tag_is_a_note_not_a_difference(self) -> None:
        installed = {"torch": "2.6.0+cu124", "torch-geometric": "2.8.0", "numpy": "2.4.6", "dgl": "2.5.0+cu124"}
        findings = compare(self.locked, installed)
        self.assertEqual([(f.name, f.kind) for f in findings], [("torch", "local-tag")])
        self.assertFalse(findings[0].is_difference)

    def test_locked_local_tag_must_match_exactly(self) -> None:
        installed = {"torch": "2.6.0", "torch-geometric": "2.8.0", "numpy": "2.4.6", "dgl": "2.5.0"}
        findings = compare(self.locked, installed)
        self.assertEqual([(f.name, f.kind) for f in findings], [("dgl", "different")])

    def test_wrong_version_with_local_tag_is_different(self) -> None:
        installed = {"torch": "2.5.1+cu124", "torch-geometric": "2.8.0", "numpy": "2.4.6", "dgl": "2.5.0+cu124"}
        self.assertEqual([f.kind for f in compare(self.locked, installed)], ["different"])

    def test_non_exact_pin_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_lock("numpy>=2.0\n")

    def test_repository_lock_parses(self) -> None:
        # cluster-setup3 CS8-3 (stated test change): the default lock is linux-lock-b.txt (env h2snn-b);
        # the 27 pins of linux-lock.txt plus hyper-connections==0.4.2, torch 2.4.1, dgl 2.4.0+cu124.
        self.assertEqual(Path(DEFAULT_LOCK).name, "linux-lock-b.txt")
        pins = parse_lock(Path(DEFAULT_LOCK).read_text(encoding="utf-8"))
        self.assertEqual(pins["torch"], "2.4.1")
        self.assertEqual(pins["dgl"], "2.4.0+cu124")
        self.assertEqual(pins["hyper-connections"], "0.4.2")
        # card 3 Part A, S-A0-2 (author, 2026-11-16; stated change outside Part A's files): strenum for HetSheaf (H-1).
        self.assertEqual(pins["strenum"], "0.4.15")
        self.assertEqual(len(pins), 29)


if __name__ == "__main__":
    unittest.main()
