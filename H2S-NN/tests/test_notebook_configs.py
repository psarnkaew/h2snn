"""Part M3b B9: every `notebooks/h2snn_*_colab.ipynb` references only config files that exist (the legacy removal
deleted the `*_h2snn.yaml` files the notebooks loaded); a path built from a variable (f-string) is not allowed, since
it cannot be checked."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = re.compile(r"configs/[A-Za-z0-9_./{}-]*\.ya?ml")


class NotebookConfigTests(unittest.TestCase):
    def test_h2snn_notebooks_reference_existing_configs_only(self) -> None:
        notebooks = sorted((REPO / "notebooks").glob("h2snn_*_colab.ipynb"))
        self.assertEqual(len(notebooks), 8)
        for path in notebooks:
            source = "\n".join("".join(cell.get("source", []))
                               for cell in json.loads(path.read_text(encoding="utf-8"))["cells"])
            for reference in CONFIG_PATH.findall(source):
                if reference.startswith("configs/generated/"):
                    continue
                with self.subTest(notebook=path.name, config=reference):
                    self.assertNotIn("{", reference)
                    self.assertTrue((REPO / reference).is_file(), reference)


if __name__ == "__main__":
    unittest.main()
