"""Card 10 A3.3 mutation: both slot scripts append --resume unconditionally (resume_flag gives --resume for every
launcher) -> ResumeTests.test_tsubame_slots and test_lab_slots fail.

    python ../docs/card10_evidence/mutation_a33.py          (from H2S-NN/; exit 0 = the mutation was caught)
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))

from tests import test_lab_slots, test_slot_resume, test_tsubame_slots  # noqa: E402

original = '*) echo "";;'
results = []
with tempfile.TemporaryDirectory() as d:
    for module, name in ((test_tsubame_slots, "test_tsubame_slots"), (test_lab_slots, "test_lab_slots")):
        source = Path(module.SLOTS).read_text()
        assert source.count(original) == 1, f"mutation anchor not found in {module.SLOTS}"
        path = Path(d) / Path(module.SLOTS).name
        path.write_text(source.replace(original, '*) echo "--resume";;'))
        with mock.patch.object(module, "SLOTS", path):
            suite = unittest.defaultTestLoader.loadTestsFromName(f"ResumeTests.{name}", test_slot_resume)
            results.append(unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
caught = not any(results)
print("MUTATION A3.3 (--resume appended unconditionally):", "caught" if caught else "NOT CAUGHT", results)
sys.exit(0 if caught else 1)
