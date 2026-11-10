"""L8c: run tests/test_v3b_mutations.py and print each mutation's failing message (run from
H2S-NN/ in the h2snn env)."""
import sys
import unittest
sys.path.insert(0, ".")
from tests import test_v3b_mutations as m

result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.loadTestsFromModule(m))
print("\nunpatched runs:", {k.rsplit(".", 1)[-1]: v[0] for k, v in m._UNPATCHED.items()})
for mutation, rows in sorted(m.MUTATION_RESULTS.items()):
    for test, message in rows.items():
        print(f"{mutation} | {test} | {message}")
sys.exit(0 if result.wasSuccessful() else 1)
