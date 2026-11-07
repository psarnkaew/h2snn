"""U2: leave the gate heads out of the scalar group -> T-F1 (movement + group contents) must fail."""
import io, sys, unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path.cwd()))
from src.models.h2snn_model import H2SNN
from tests.test_cooperative import LearningFixTests as T

def run(names):
    return unittest.TextTestRunner(stream=io.StringIO()).run(unittest.TestSuite(T(n) for n in names))

names = ["test_TF1_the_scalar_group_moves_eta_and_both_gate_heads", "test_TF1_group_contents_and_defaults"]
ok = run(names)
print(f"control: failures={len(ok.failures)} errors={len(ok.errors)}")
orig = H2SNN.scalar_parameters
with mock.patch.object(H2SNN, "scalar_parameters", lambda self: [p for p in (self.theta_eta, self.theta_gamma) if p is not None]):
    res = run(names)
print(f"mutation (gate heads not in the fast group): failures={len(res.failures)} errors={len(res.errors)} -> {'CAUGHT' if res.failures or res.errors else 'NOT CAUGHT'}")
