"""Card 3 B3.3 mutation: PSHGCN's H2GB path without `run_seed` (in-process patch, no source edit). The shared
`train_index_minibatch` is wrapped to drop the `run_seed` keyword the adapter passes, as if `pshgcn.py` did not pass
it. Expected: the PSHGCN B1a test fails (PDNS subset refused / wrong), the H2SGNN one still passes; unpatched, both pass."""
import sys
import unittest
from unittest import mock

sys.path.insert(0, ".")
import src.baselines.minibatch as mb  # noqa: E402

BASE = "tests.test_freebase_baselines.B1aSubsetTests."
REAL = mb.train_index_minibatch


def without_run_seed(**kw):
    kw.pop("run_seed", None)
    return REAL(**kw)


def run(name):
    result = unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromName(BASE + name))
    return result.wasSuccessful()


ok = True
for label, patched in (("mutant (pshgcn without run_seed)", True), ("real", False)):
    ctx = mock.patch.object(mb, "train_index_minibatch", without_run_seed) if patched else mock.patch.dict({}, {})
    with ctx:
        p = run("test_pshgcn_h2gb_path_evaluates_the_run_seeds_subset")
        h = run("test_h2sgnn_h2gb_path_evaluates_the_run_seeds_subset")
    want_p = not patched
    print(f"{label}: pshgcn test {'pass' if p else 'FAIL'} (expected {'pass' if want_p else 'FAIL'}); "
          f"h2sgnn test {'pass' if h else 'FAIL'} (expected pass)")
    ok &= (p == want_p) and h
print("mutation check", "OK" if ok else "UNEXPECTED")
sys.exit(0 if ok else 1)
