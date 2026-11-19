"""TSUBAME card T4 mutation: collect.py silently skips a record (the first of every state) -> the test fails.

    python ../docs/tsubame_evidence/t4_mutation.py          (from H2S-NN/; exit 0 = the mutation was caught)
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))

from scripts.tsubame import collect  # noqa: E402
from tests import test_tsubame_collect  # noqa: E402

original = collect.trial_records
with mock.patch.object(collect, "trial_records", lambda state: original(state)[1:]):
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "CollectTests.test_rows_equal_records_values_unchanged_provenance_filled", test_tsubame_collect)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
caught = not result.wasSuccessful()
print("MUTATION T4 (a record skipped):", "caught (the test fails)" if caught else "NOT CAUGHT")
sys.exit(0 if caught else 1)
