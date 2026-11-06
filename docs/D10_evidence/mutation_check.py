"""D10 mutation check: each mutation of the production code must make its targeted tests fail
(and the unmutated code must pass them). Run from H2S-NN/: python ../docs/D10_evidence/mutation_check.py"""
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch

from src.models import h2snn_model as hm
from tests import test_cooperative as T

PRODUCTION_DIFFUSE = hm.H2SNN.diffuse
PRODUCTION_GATES = hm.H2SNN.coupling_gates


def swapped_sides(self, x0, operator):  # B Delta G instead of G Delta B
    x = x0
    for _ in range(self.num_steps):
        y = x if operator.listen is None else operator.listen.unsqueeze(-1) * x
        y = operator.laplacian.apply_normalized(y)
        if operator.broadcast is not None:
            y = operator.broadcast.unsqueeze(-1) * y
        x = x - operator.eta * y
        x = self.layer_dropout(x)
    return x


def no_listen_gate(self, x0, operator):  # B applied, G missing
    x = x0
    for _ in range(self.num_steps):
        y = x if operator.broadcast is None else operator.broadcast.unsqueeze(-1) * x
        x = x - operator.eta * operator.laplacian.apply_normalized(y)
        x = self.layer_dropout(x)
    return x


def no_sigmoid(self, h0):  # unbounded gates
    listen = None if self.listen_gate is None else self.listen_gate(h0).squeeze(-1)
    broadcast = None if self.broadcast_gate is None else self.broadcast_gate(h0).squeeze(-1)
    return listen, broadcast


TARGETS = {
    "swapped sides (B Delta G)": ("diffuse", swapped_sides, [
        "CooperativeBehaviourTests.test_T4_a_node_with_zero_listen_gate_is_unchanged_exactly",
        "CooperativeBehaviourTests.test_T4_a_node_with_zero_broadcast_gate_influences_nobody",
        "GatedOperatorTests.test_T2_gated_step_equals_the_dense_reference"]),
    "gates without the sigmoid": ("coupling_gates", no_sigmoid, [
        "SpectrumTests.test_T3_real_spectrum_and_bound_with_gates"]),
    "B applied, G missing": ("diffuse", no_listen_gate, [
        "CooperativeBehaviourTests.test_T4_a_node_with_zero_listen_gate_is_unchanged_exactly",
        "GatedOperatorTests.test_T2_gated_step_equals_the_dense_reference"]),
}


def run(ids):
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(f"tests.test_cooperative.{i}")
                               for i in ids)
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    failed = {t.id().split(".")[-1] for t, _ in result.failures + result.errors}
    return failed


all_caught = True
for name, (attr, replacement, ids) in TARGETS.items():
    baseline = run(ids)
    setattr(hm.H2SNN, attr, replacement)
    try:
        mutated = run(ids)
    finally:
        hm.H2SNN.diffuse, hm.H2SNN.coupling_gates = PRODUCTION_DIFFUSE, PRODUCTION_GATES
    caught = bool(mutated) and not baseline
    all_caught &= caught
    print(f"{name:28s} unmutated failures: {sorted(baseline) or 'none'}; mutated failures: "
          f"{sorted(mutated) or 'none'}  -> {'CAUGHT' if caught else 'NOT CAUGHT'}")
print("ALL THREE CAUGHT" if all_caught else "SOME NOT CAUGHT")
sys.exit(0 if all_caught else 1)
