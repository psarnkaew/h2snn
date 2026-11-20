"""Card 10 L1 mutation: submit_lab.sh does not check the GPU-type record (a cell recorded with one GPU type is
accepted with another) -> SubmitTests.test_chain_gpu_type_record_and_refusals fails.

    python ../docs/card10_evidence/mutation_l1.py          (from H2S-NN/; exit 0 = the mutation was caught)
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))

from tests import test_lab_slots  # noqa: E402

source = test_lab_slots.SUBMIT.read_text()
original = "if cell in recorded and recorded[cell] != gpu_type:"
assert source.count(original) == 1, "mutation anchor not found"
with tempfile.TemporaryDirectory() as d:
    mutated = Path(d) / "submit_lab.sh"
    mutated.write_text(source.replace(original, "if False:"))
    with mock.patch.object(test_lab_slots, "SUBMIT", mutated):
        suite = unittest.defaultTestLoader.loadTestsFromName(
            "SubmitTests.test_chain_gpu_type_record_and_refusals", test_lab_slots)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
caught = not result.wasSuccessful()
print("MUTATION L1 (GPU-type record not checked):", "caught" if caught else "NOT CAUGHT")
sys.exit(0 if caught else 1)
