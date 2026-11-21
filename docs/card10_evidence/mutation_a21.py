"""Card 10 A2.1 mutation: submit_chain.sh gives node_h S = 4 (it has 2 GPUs) -> slots on GPUs 2 and 3 the task does
not have -> TypeTests.test_every_whole_h100_type_runs_each_line_once_on_its_own_gpu fails.

    python ../docs/card10_evidence/mutation_a21.py          (from H2S-NN/; exit 0 = the mutation was caught)
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))

from tests import test_tsubame_slots, test_tsubame_types  # noqa: E402

source = test_tsubame_slots.CHAIN.read_text()
original = "node_h) S=2;;"
assert source.count(original) == 1, "mutation anchor not found"
with tempfile.TemporaryDirectory() as d:
    mutated = Path(d) / "submit_chain.sh"
    mutated.write_text(source.replace(original, "node_h) S=4;;"))
    with mock.patch.object(test_tsubame_slots, "CHAIN", mutated):
        suite = unittest.defaultTestLoader.loadTestsFromName(
            "TypeTests.test_every_whole_h100_type_runs_each_line_once_on_its_own_gpu", test_tsubame_types)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
caught = not result.wasSuccessful()
print("MUTATION A2.1 (node_h S = 4):", "caught" if caught else "NOT CAUGHT")
sys.exit(0 if caught else 1)
