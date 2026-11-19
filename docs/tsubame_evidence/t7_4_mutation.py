"""TSUBAME card T7.4 mutation: `baselines-1` no longer excludes the `baselines-early` cells -> the HetSheaf / CSNN
ACM and DBLP lines are in two lists -> test_phases_pairwise_disjoint_and_cover_their_cells fails.

    python ../docs/tsubame_evidence/t7_4_mutation.py          (from H2S-NN/; exit 0 = the mutation was caught)
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))

from scripts.tsubame import make_cell_list  # noqa: E402
from tests import test_tsubame_phases  # noqa: E402

with mock.patch.object(make_cell_list, "excluded", lambda phase, row: None):
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "PhaseTests.test_phases_pairwise_disjoint_and_cover_their_cells", test_tsubame_phases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
caught = not result.wasSuccessful()
print("MUTATION T7.4 (no exclusion):", "caught (the lists overlap)" if caught else "NOT CAUGHT")
sys.exit(0 if caught else 1)
