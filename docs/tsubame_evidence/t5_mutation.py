"""TSUBAME card T5 mutation: sweep_slots.sh computes the global slot without the task offset (g = k) -> lines of
slot k run in every task, so a line runs twice -> test (1) fails.

    python ../docs/tsubame_evidence/t5_mutation.py          (from H2S-NN/; exit 0 = the mutation was caught)
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))

from tests import test_tsubame_slots  # noqa: E402

source = test_tsubame_slots.SLOTS.read_text()
original = 'g=$(( (TASK - 1) * S + $1 ))'
assert source.count(original) == 1, "mutation anchor not found"
with tempfile.TemporaryDirectory() as d:
    mutated = Path(d) / "sweep_slots.sh"
    mutated.write_text(source.replace(original, 'g=$(( $1 ))'))
    with mock.patch.object(test_tsubame_slots, "SLOTS", mutated):
        suite = unittest.defaultTestLoader.loadTestsFromName("SlotTests.test_every_line_runs_exactly_once",
                                                             test_tsubame_slots)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
caught = not result.wasSuccessful()
print("MUTATION T5 (slot index without the task offset):", "caught (a line runs twice)" if caught else "NOT CAUGHT")
sys.exit(0 if caught else 1)
