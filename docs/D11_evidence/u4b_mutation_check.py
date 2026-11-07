"""U4b three-mutation check (task card Part F). Each mutation is applied at run time; the targeted
tests must FAIL, the unmutated control must pass.
 M1  self-loop omitted from the softmax   (self entries left unnormalized, outside the softmax)
 M2  Q used instead of Q^T on a -1 hop    (hop signs ignored when the transport factors are chosen)
 M3  log m_f omitted                      (merged cell weight does not enter the score)
"""
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))
import src.models.map_operators as mo
import src.models.pair_attention as pa
from tests import test_pair_attention as T

_real_softmax = pa.segment_softmax


def _self_outside_softmax(scores, receivers, num_nodes):
    out = torch.empty_like(scores)
    out[num_nodes:] = _real_softmax(scores[num_nodes:], receivers[num_nodes:], num_nodes)
    out[:num_nodes] = torch.exp(scores[:num_nodes])
    return out


def _no_log_m(self, h0, nodes, latent, log_merged, checkpoint):
    return _real_cell_scores(self, h0, nodes, latent, torch.zeros_like(log_merged), checkpoint)


import torch  # noqa: E402

_real_cell_scores = pa.PairAttention.cell_scores


def run(classes_or_cases):
    suite = unittest.TestSuite()
    for item in classes_or_cases:
        if isinstance(item, str):
            suite.addTest(unittest.defaultTestLoader.loadTestsFromName(item, T))
        else:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(item))
    return unittest.TextTestRunner(stream=io.StringIO()).run(suite)


CASES = {
    "M1 self-loop omitted from the softmax": (
        ["AttentionWeightTests", "SectionTests"],
        [mock.patch.object(pa, "segment_softmax", _self_outside_softmax)]),
    "M2 Q instead of Q^T on a -1 hop": (
        ["SectionTests", "DenseReferenceTests"],
        [mock.patch.object(mo, "_mix_rows", lambda a, b, mask: a)]),
    "M3 log m_f omitted": (
        ["MergedWeightTests"],
        [mock.patch.object(pa.PairAttention, "cell_scores", _no_log_m)]),
}
control = run([getattr(T, n) for names, _ in CASES.values() for n in names])
print(f"control (unmutated): ran={control.testsRun} failures={len(control.failures)} errors={len(control.errors)}")
for name, (names, patches) in CASES.items():
    for patch in patches:
        patch.start()
    try:
        res = run([getattr(T, n) for n in names])
    finally:
        mock.patch.stopall()
    bad = sorted({str(t[0]).split()[0] for t in res.failures + res.errors})
    print(f"{name}: ran={res.testsRun} failing={len(bad)} -> {'CAUGHT' if bad else 'NOT CAUGHT'} {bad[:4]}")
