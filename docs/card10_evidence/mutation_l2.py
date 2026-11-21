"""Card 10 L2 mutation: grid_points.largest returns the minimum of a size axis (the smallest point instead of the
largest) -> GridPointTests (baseline cells and H2S-NN configs) fail.

    python ../docs/card10_evidence/mutation_l2.py          (from H2S-NN/; exit 0 = the mutation was caught)
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))

from tests import test_fit_check  # noqa: E402

source = Path(test_fit_check.G.__file__).read_text()
original = "    return max(values)\n"
assert source.count(original) == 1, "mutation anchor not found"
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "grid_points_mutated.py"
    path.write_text(source.replace(original, "    return min(values)\n"))
    spec = importlib.util.spec_from_file_location("grid_points_mutated", path)
    mutated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutated)
    caught = True
    with mock.patch.object(test_fit_check, "G", mutated):
        for name in ("GridPointTests.test_baseline_max_point_every_cell", "GridPointTests.test_h2snn_max_point_every_config"):
            result = unittest.TextTestRunner(verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromName(name, test_fit_check))
            caught &= not result.wasSuccessful()
print("MUTATION L2 (size axes at their minimum):", "caught by both launcher tests" if caught else "NOT CAUGHT")
sys.exit(0 if caught else 1)
