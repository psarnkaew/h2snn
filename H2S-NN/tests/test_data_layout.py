"""Part M, M1: one place for the data roots (data/NC, data/LP, data/GC); no old-layout path."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from src.data.data_layout import (REPO_DATA_ROOT, gc_root, hgb_nc_raw_dir, lp_derived_root,
                                  lp_root, nc_root, tu_raw_dir)

REPO = Path(__file__).resolve().parents[1]


class DataLayoutTests(unittest.TestCase):
    def test_roots(self) -> None:
        root = Path("x")
        self.assertEqual(nc_root(root), Path("x/NC"))
        self.assertEqual(lp_root(root), Path("x/LP"))
        self.assertEqual(lp_derived_root(root), Path("x/LP/lp_derived"))
        self.assertEqual(gc_root(root), Path("x/GC"))
        self.assertEqual(hgb_nc_raw_dir(root, "acm"), Path("x/NC/acm/raw/ACM"))
        self.assertEqual(hgb_nc_raw_dir(root, "freebase"), Path("x/NC/freebase/raw/Freebase"))
        self.assertEqual(tu_raw_dir(root, "MUTAG"), Path("x/GC/MUTAG/raw"))
        self.assertEqual(REPO_DATA_ROOT, REPO / "data")

    def test_no_old_layout_paths_in_code(self) -> None:
        """No loader, adapter or script builds a path of the pre-M1 layout (`data/<key>/<key>/raw`,
        `root / "tud"`, `data / "lp_derived"`, a dataset folder directly under the data root)."""
        patterns = [
            re.compile(r'/ "tud"'),
            re.compile(r'"data" / "lp_derived"'),
            re.compile(r'"data" / key / key'),
            re.compile(r'"data" / (key|name|dataset|str\(name\))\b'),
            re.compile(r'root / "(acm|dblp|imdb|freebase|amazonbook)"'),
            re.compile(r'(acm|dblp|imdb|freebase)/(acm|dblp|imdb|freebase)/raw'),
        ]
        offenders = []
        for folder in ("src", "scripts"):
            for path in (REPO / folder).rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                for pattern in patterns:
                    if pattern.search(text):
                        offenders.append(f"{path.relative_to(REPO)}: {pattern.pattern}")
        self.assertEqual(offenders, [])

    def test_loader_specs_live_under_lp(self) -> None:
        from src.data.link_prediction_dataset import _LOCAL_HGB_LINK_DATASETS, local_hgb_link_dir

        for name in ("acm_pa", "dblp_ap", "imdb_ma", "freebase_bb", "pubmed", "lastfm"):
            folder = local_hgb_link_dir(Path("x"), name)
            self.assertEqual(folder.parts[:2], ("x", "LP"), name)
        self.assertEqual(local_hgb_link_dir(Path("x"), "acm_pa"), Path("x/LP/lp_derived/acm_pa"))
        self.assertIn("pubmed", _LOCAL_HGB_LINK_DATASETS)


if __name__ == "__main__":
    unittest.main()
