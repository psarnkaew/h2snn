"""Card 10 A2.2 mutation: fit_check.py headroom threshold 100 % instead of 75 % -> a 90 % peak is `fits` ->
FitCheckTests.test_headroom_and_phases fails.

    python ../docs/card10_evidence/mutation_a22.py          (from H2S-NN/; exit 0 = the mutation was caught)
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))

from tests import test_fit_check  # noqa: E402

source = Path(test_fit_check.F.__file__).read_text()
original = "HEADROOM = 0.75\n"
assert source.count(original) == 1, "mutation anchor not found"
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "fit_check_mutated.py"
    path.write_text(source.replace(original, "HEADROOM = 1.0\n"))
    spec = importlib.util.spec_from_file_location("fit_check_mutated", path)
    mutated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutated)
    with mock.patch.object(test_fit_check, "F", mutated):
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromName("FitCheckTests.test_headroom_and_phases", test_fit_check))
caught = not result.wasSuccessful()
print("MUTATION A2.2 (threshold 100 %):", "caught" if caught else "NOT CAUGHT")
sys.exit(0 if caught else 1)
