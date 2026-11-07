"""G2 three-mutation check (task card Part G). Run from H2S-NN/ with the h2snn environment.
Each mutation is applied at run time; its targeted test must FAIL; the unmutated control must pass.
 M1  normalization of V_k removed            -> T-G2 must fail
 M2  recurrence on A instead of A - I         -> T-G4 must fail
 M3  gates applied as B Delta_hat G           -> T-G3 must fail
"""
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))
import src.models.filter_bank as fb  # noqa: E402
import src.models.h2snn_model as hm  # noqa: E402
from src.models.pair_attention import AttentionOperator  # noqa: E402
from tests import test_filter_bank as T  # noqa: E402

_real_apply = hm.apply_operator


def _swapped_gates(operator, x):
    if isinstance(operator, AttentionOperator):
        return _real_apply(operator, x)
    y = x if operator.listen is None else operator.listen.unsqueeze(-1) * x      # G first
    y = operator.laplacian.apply_normalized(y)
    return y if operator.broadcast is None else operator.broadcast.unsqueeze(-1) * y  # then B


TESTS = {
    "T-G2": "FilterBankOperatorTests.test_TG2_gain_in_metric_M_is_at_most_one_with_control",
    "T-G3": "FilterBankOperatorTests.test_TG3_cooperation_through_the_filter_exact_zero_blocks",
    "T-G4": "FilterBankModelTests.test_TG4_d11_initialization_equals_first_order",
}
CASES = {
    "M1 normalization of V_k removed": ("T-G2", mock.patch.object(fb.ChebyshevFilterBank, "weights",
                                                                 lambda self: self.V)),
    "M2 recurrence on A instead of A - I": ("T-G4", mock.patch.object(
        fb, "chebyshev_shift", lambda apply_a, x: apply_a(x))),
    "M3 gates applied as B Delta_hat G": ("T-G3", mock.patch.object(hm, "apply_operator", _swapped_gates)),
}


def run(name: str) -> unittest.TestResult:
    suite = unittest.defaultTestLoader.loadTestsFromName(TESTS[name], T)
    return unittest.TextTestRunner(stream=io.StringIO()).run(suite)


for name in TESTS:
    result = run(name)
    print(f"control (unmutated) {name}: ran={result.testsRun} failures={len(result.failures)} "
          f"errors={len(result.errors)}")
for label, (target, patch) in CASES.items():
    with patch:
        result = run(target)
    failed = len(result.failures) + len(result.errors)
    first = (result.failures + result.errors)[0][1].strip().splitlines()[-1] if failed else ""
    print(f"{label}: {target} ran={result.testsRun} failing={failed} -> "
          f"{'CAUGHT' if failed else 'NOT CAUGHT'} | {first[:160]}")
