"""Card 10 A3.2 mutation: fit_check.py build leaves published lines out of the probe list (as before A3.2), so they
get no fit_check.tsv row -> FitCheckTests.test_published_lines_get_a_row_and_follow_it fails.

    python ../docs/card10_evidence/mutation_a32.py          (from H2S-NN/; exit 0 = the mutation was caught)
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
original = "        token, probe = probe_line(text, fit_out)\n"
assert source.count(original) == 1, "mutation anchor not found"
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "fit_check_mutated.py"
    path.write_text(source.replace(original, original + "        if \"run_baseline_published\" in probe:\n"
                                                         "            continue\n"))
    spec = importlib.util.spec_from_file_location("fit_check_mutated", path)
    mutated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutated)
    with mock.patch.object(test_fit_check, "F", mutated):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(
            "FitCheckTests.test_published_lines_get_a_row_and_follow_it", test_fit_check))
caught = not result.wasSuccessful()
print("MUTATION A3.2 (published lines without a probe):", "caught" if caught else "NOT CAUGHT")
sys.exit(0 if caught else 1)
