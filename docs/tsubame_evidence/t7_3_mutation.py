"""TSUBAME card T7.3 mutation: the final-seeds gate removed (every sweep counts as finished) -> a final-seeds line
for an unfinished cell appears -> test_final_seeds_only_for_finished_sweeps fails.

    python ../docs/tsubame_evidence/t7_3_mutation.py          (from H2S-NN/; exit 0 = the mutation was caught)
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))

from scripts.tsubame import make_cell_list  # noqa: E402
from tests import test_tsubame_final_seeds_gate  # noqa: E402

with mock.patch.object(make_cell_list, "sweep_finished", lambda line: (True, 0, None)):
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "FinalSeedsGateTests.test_final_seeds_only_for_finished_sweeps", test_tsubame_final_seeds_gate)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
caught = not result.wasSuccessful()
print("MUTATION T7.3 (final-seeds gate removed):", "caught (the test fails)" if caught else "NOT CAUGHT")
sys.exit(0 if caught else 1)
