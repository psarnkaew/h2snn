"""Part P-fix, F4 (`docs/H2S-NN_partPfix_card.md`): every adapter reads the bundle's data root.

- no module under `src/baselines` reads `REPO_DATA_ROOT` in code (AST: names and imports); the
  bundle helper `bundle_data_root` is the one way to a root;
- `build_bundle` records the resolved `data.root`; `hgb_raw_dir(dataset, root)` resolves under it;
  SeHGNN's vendor arguments point under the given root;
- with the bundle's `data.root` at a tiny Freebase-format fixture and `REPO_DATA_ROOT` patched to an
  empty directory, H2G-former, Hetero^2Net and HAN (the fixed data paths whose vendors are present)
  run one epoch (finite keys).
Card mutation: one site reverted to `REPO_DATA_ROOT` -> its test fails.
"""

from __future__ import annotations

import ast
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.baselines.protocol import NODE_CLASSIFICATION, BaselineParams
from tests.partP_freebase_fixture import write_freebase_release

BASELINES = Path(__file__).resolve().parent.parent / "src" / "baselines"


def _reads_repo_data_root(path: Path) -> list[int]:
    """Lines where `path` names or imports REPO_DATA_ROOT in code (docstrings excluded)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "REPO_DATA_ROOT":
            lines.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == "REPO_DATA_ROOT":
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and any(a.name == "REPO_DATA_ROOT"
                                                      for a in node.names):
            lines.append(node.lineno)
    return lines


class SourceTests(unittest.TestCase):
    def test_no_baseline_module_reads_repo_data_root(self) -> None:
        offenders = {str(p.relative_to(BASELINES)): _reads_repo_data_root(p)
                     for p in sorted(BASELINES.rglob("*.py"))}
        self.assertEqual({k: v for k, v in offenders.items() if v}, {})


class RootTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        write_freebase_release(cls.root)
        cls.empty = cls.root / "empty_default_root"
        cls.empty.mkdir()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _bundle(self):
        from src.baselines.bundles import build_bundle

        return build_bundle(NODE_CLASSIFICATION, "freebase",
                            overrides={"device": "cpu", "data": {"root": str(self.root)}})

    def test_bundle_carries_its_root_and_raw_dirs_resolve_under_it(self) -> None:
        import src.data.data_layout as layout
        from src.baselines.adapters import sehgnn
        from src.baselines.bundles import bundle_data_root, hgb_raw_dir

        with mock.patch.object(layout, "REPO_DATA_ROOT", self.empty):
            bundle = self._bundle()
        self.assertEqual(bundle_data_root(bundle), self.root.resolve())
        self.assertEqual(hgb_raw_dir("freebase", bundle_data_root(bundle)),
                         self.root.resolve() / "NC" / "freebase" / "raw" / "Freebase")
        acm = self.root / "NC" / "acm" / "raw" / "ACM"
        acm.mkdir(parents=True)
        args = sehgnn._vendor_args("acm", BaselineParams({}), self.root)
        self.assertEqual(Path(args.root), acm.parent)
        with self.assertRaises(FileNotFoundError):  # the empty default root has no ACM
            hgb_raw_dir("acm", self.empty)

    def test_fixed_adapters_run_from_the_bundle_root(self) -> None:
        import src.data.data_layout as layout
        from src.baselines.registry import get_adapter

        with mock.patch.object(layout, "REPO_DATA_ROOT", self.empty):
            bundle = self._bundle()
            for name in ("h2gformer", "hetero2net", "han"):
                with self.subTest(adapter=name):
                    record = get_adapter(name).fit_task(
                        bundle, BaselineParams({"epochs": 1, "early_stopping": 1}),
                        task=NODE_CLASSIFICATION, seed=0, device=torch.device("cpu"))
                    for key in ("test_macro_f1", "val_macro_f1"):
                        self.assertTrue(math.isfinite(float(record[key])), key)


if __name__ == "__main__":
    unittest.main()
