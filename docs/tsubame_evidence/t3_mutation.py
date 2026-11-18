"""TSUBAME card T3 mutation: the runnable filter dropped (every phase-1 registry row becomes a line) -> the test fails.

    python ../docs/tsubame_evidence/t3_mutation.py          (from H2S-NN/; exit 0 = the mutation was caught)
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))

from scripts.tsubame import make_cell_list  # noqa: E402
from tests import test_tsubame_cell_list  # noqa: E402

with mock.patch.object(make_cell_list, "is_listed", lambda row: True):
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "CellListTests.test_phase1_lines_are_the_registry_runnable_cells", test_tsubame_cell_list)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
caught = not result.wasSuccessful()
print("MUTATION T3 (runnable filter dropped):", "caught (the test fails)" if caught else "NOT CAUGHT")
sys.exit(0 if caught else 1)
