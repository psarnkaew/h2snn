"""Card 10 A2.3 mutation: failures.py skips a failure (the first entry of every state) ->
FailuresTests.test_every_failure_reported_with_condition_run_and_tail fails.

    python ../docs/card10_evidence/mutation_a23.py          (from H2S-NN/; exit 0 = the mutation was caught)
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))

from tests import test_failures_report  # noqa: E402

source = Path(test_failures_report.F.__file__).read_text()
original = 'for entry in state["failures"]:'
assert source.count(original) == 1, "mutation anchor not found"
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "failures_mutated.py"
    path.write_text(source.replace(original, 'for entry in state["failures"][1:]:'))
    spec = importlib.util.spec_from_file_location("failures_mutated", path)
    mutated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutated)
    with mock.patch.object(test_failures_report, "F", mutated):
        suite = unittest.defaultTestLoader.loadTestsFromName(
            "FailuresTests.test_every_failure_reported_with_condition_run_and_tail", test_failures_report)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
caught = not result.wasSuccessful()
print("MUTATION A2.3 (a failure skipped):", "caught" if caught else "NOT CAUGHT")
sys.exit(0 if caught else 1)
