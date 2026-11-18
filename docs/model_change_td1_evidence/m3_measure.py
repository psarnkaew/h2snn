"""Model change TD-1, M3: run the card's tests (tests/test_td1_maps.py) and print the measured errors each
test stores (`worst`, `numbers`, `change`, `norms`, `range`), for the report's test table.

    python ../docs/model_change_td1_evidence/m3_measure.py      (from H2S-NN/, environment E)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import tests.test_td1_maps as m  # noqa: E402

m.setUpModule()
try:
    loader = unittest.TestLoader()
    for case in loader.loadTestsFromModule(m):
        for test in case:
            result = unittest.TestResult()
            test.run(result)
            status = "OK" if result.wasSuccessful() else f"FAIL {result.failures or result.errors}"
            stored = {k: getattr(test, k) for k in ("worst", "numbers", "change", "range")
                      if hasattr(test, k)}
            print(f"{test.id().split('.')[-1]}: {status} {stored}", flush=True)
finally:
    m.tearDownModule()
