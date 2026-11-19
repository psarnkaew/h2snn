"""TSUBAME card T7.2 mutation: collect.py silently skips an `h2snn` record (the first record of every H2S-NN state)
-> test_h2snn_rows_equal_records_values_unchanged fails.

    python ../docs/tsubame_evidence/t7_2_mutation.py          (from H2S-NN/; exit 0 = the mutation was caught)
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))

from scripts.tsubame import collect  # noqa: E402
from tests import test_tsubame_collect_h2snn  # noqa: E402

original = collect.trial_records


def skip_h2snn(state):
    records = original(state)
    return records[1:] if str(state.get("base_exp_name", "")).startswith("h2snn_") else records


with mock.patch.object(collect, "trial_records", skip_h2snn):
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "H2SNNCollectTests.test_h2snn_rows_equal_records_values_unchanged", test_tsubame_collect_h2snn)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
caught = not result.wasSuccessful()
print("MUTATION T7.2 (an h2snn record skipped):", "caught (the test fails)" if caught else "NOT CAUGHT")
sys.exit(0 if caught else 1)
