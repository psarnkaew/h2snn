"""U1 three-mutation check for D7 (task card Part D, reduced to three by the author):
 M1  -1 hop uses Q instead of Q^T          (hop signs ignored when the transport factors are chosen)
 M2  duplicates dropped instead of summed  (merge_cells forced to merge="drop_duplicates")
 M3  reversed orientation of a held-out fact not removed (rho ignored by the link split)
Each mutation is applied at run time (no file is edited) and the targeted tests must FAIL."""
import io, sys, unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path.cwd()))
import src.data.path_sampler as ps
import src.models.map_operators as mo
import src.data.link_prediction_dataset as lp
from tests import test_paths_and_facts as T

_real_merge = ps.merge_cells
def _drop(*a, **k):
    k["merge"] = "drop_duplicates"
    if len(a) > 5:
        a = a[:5]
    return _real_merge(*a, **k)

def run(classes):
    suite = unittest.TestSuite()
    for c in classes:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(c))
    res = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    return res

CASES = {
 "M1 -1 hop uses Q instead of Q^T": ([T.SignedTransportTests, T.SectionPreservationTests, T.OLPathTermTests],
        [mock.patch.object(mo, "_mix_rows", lambda a, b, mask: a)]),
 "M2 duplicates dropped instead of summed": ([T.EstimatorTests, T.IdentityTests],
        [mock.patch.object(ps, "merge_cells", _drop)]),
 "M3 reversed orientation of a held-out fact not removed": ([T.LinkPredictionFactTests],
        [mock.patch.object(lp, "inverse_relations_typed", lambda *a, **k: {})]),
}
res = run([c for cl, _ in CASES.values() for c in cl])
print(f"control (unmutated): ran={res.testsRun} failures={len(res.failures)} errors={len(res.errors)}")
for name, (classes, patches) in CASES.items():
    for p in patches: p.start()
    try:
        res = run(classes)
    finally:
        mock.patch.stopall()
    bad = [str(t[0]).split()[0] for t in res.failures + res.errors]
    print(f"{name}: ran={res.testsRun} failing={len(bad)} -> {'CAUGHT' if bad else 'NOT CAUGHT'} {sorted(set(bad))[:4]}")
