"""Part M, M5 mutation check: re-insert BatchNorm1d between pooling and the Linear of the graph
readout (the pre-M5 head) and run tests/test_gc_protocol.py; a test must fail. Restore; all pass.
Run from H2S-NN/."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "H2S-NN"))

from torch import nn  # noqa: E402

import src.models.graph_readout as gr  # noqa: E402

original_init, original_forward = gr.GraphReadout.__init__, gr.GraphReadout.forward


def init_with_norm(self, input_dim, num_classes, pooling):
    original_init(self, input_dim, num_classes, pooling)
    self.norm = nn.BatchNorm1d(input_dim)


def forward_with_norm(self, node_representation, batch, num_graphs):
    if self.attention is not None:
        pooled = self.attention(node_representation, batch, num_graphs)
    else:
        pooled = gr.pool_nodes(node_representation, batch, num_graphs, self.pooling)
    return self.decoder(self.norm(pooled))


def run(label: str) -> unittest.TestResult:
    sys.modules.pop("tests.test_gc_protocol", None)
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_gc_protocol.PoolingTests")
    result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w")).run(suite)
    failed = [t.id().split(".")[-1] for t, _ in result.failures + result.errors]
    print(f"{label}: ran {result.testsRun}, failed {len(failed)}: {failed}")
    return result


gr.GraphReadout.__init__, gr.GraphReadout.forward = init_with_norm, forward_with_norm
mutated = run("MUTATION BatchNorm re-inserted")
gr.GraphReadout.__init__, gr.GraphReadout.forward = original_init, original_forward
clean = run("restored")
ok = bool(mutated.failures or mutated.errors) and clean.wasSuccessful()
print("mutation caught" if ok else "MUTATION NOT CAUGHT")
sys.exit(0 if ok else 1)
