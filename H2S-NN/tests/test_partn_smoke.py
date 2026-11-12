"""Part N-fix, F2: the GC collapse check of `scripts/partn_smoke.py` (author decision 1 of the
Part N review). (c1) the logits and (c2) the pooled graph embeddings must vary across the
validation graphs (mean over coordinates of the std over graphs > 1e-6); the predicted-class count
is recorded, not a gate; a capture shorter than the validation set is a script error."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

from src.models.graph_readout import GraphReadout

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "partn_smoke.py"


def _smoke_module():
    spec = importlib.util.spec_from_file_location("_partn_smoke_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


SMOKE = _smoke_module()
NUM_GRAPHS, NODES_PER_GRAPH, DIM, CLASSES = 6, 3, 4, 2


def _validation_pass(readout: GraphReadout, node_rows: torch.Tensor,
                     expected: int = NUM_GRAPHS) -> dict:
    """One validation pass of a fake GC model through the real readout, one graph per batch
    (batch size 1, the GC protocol), captured by the smoke's hooks."""
    capture = SMOKE.GCValidationCapture(readout.decoder)
    capture.start()
    with torch.no_grad():
        for graph in range(NUM_GRAPHS):
            rows = node_rows[graph * NODES_PER_GRAPH:(graph + 1) * NODES_PER_GRAPH]
            readout(rows, torch.zeros(NODES_PER_GRAPH, dtype=torch.long), 1)
    try:
        return capture.finish(expected)
    finally:
        capture.remove()


def _readout(seed: int = 0) -> GraphReadout:
    torch.manual_seed(seed)
    return GraphReadout(DIM, CLASSES, pooling="mean").eval()


class GCCollapseCheckTests(unittest.TestCase):
    def test_identical_pooled_embeddings_fail_c1_and_c2(self) -> None:
        """Every graph has the same nodes, so the same pooled embedding and, through one Linear
        layer, the same logits."""
        one_graph = torch.randn(NODES_PER_GRAPH, DIM)
        result = _validation_pass(_readout(), one_graph.repeat(NUM_GRAPHS, 1))
        self.assertFalse(result["c1_ok"])
        self.assertFalse(result["c2_ok"])
        self.assertEqual(result["graphs_captured"], NUM_GRAPHS)

    def test_constant_head_fails_c1_only(self) -> None:
        readout = _readout()
        with torch.no_grad():
            readout.decoder.weight.zero_()
        result = _validation_pass(readout, torch.randn(NUM_GRAPHS * NODES_PER_GRAPH, DIM))
        self.assertFalse(result["c1_ok"])
        self.assertTrue(result["c2_ok"])

    def test_one_predicted_class_with_varied_outputs_passes(self) -> None:
        """Majority-class prediction early in training is not collapse: not a gate."""
        readout = _readout()
        with torch.no_grad():
            readout.decoder.bias.copy_(torch.tensor([10.0, -10.0]))
        result = _validation_pass(readout, torch.randn(NUM_GRAPHS * NODES_PER_GRAPH, DIM))
        self.assertEqual(result["classes_predicted"], 1)
        self.assertTrue(result["c1_ok"])
        self.assertTrue(result["c2_ok"])

    def test_capture_shorter_than_the_validation_set_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validation_pass(_readout(), torch.randn(NUM_GRAPHS * NODES_PER_GRAPH, DIM),
                             expected=NUM_GRAPHS + 1)

    def test_capture_records_only_while_active(self) -> None:
        """Training and test passes (capture inactive) add no rows."""
        readout = _readout()
        capture = SMOKE.GCValidationCapture(readout.decoder)
        rows = torch.randn(NODES_PER_GRAPH, DIM)
        batch = torch.zeros(NODES_PER_GRAPH, dtype=torch.long)
        readout(rows, batch, 1)                     # not active
        capture.start()
        readout(rows, batch, 1)
        capture.active = False
        readout(rows, batch, 1)                     # not active
        self.assertEqual(len(capture.logits), 1)
        self.assertEqual(len(capture.pooled), 1)
        capture.remove()

    def test_the_gate_of_a_gc_run_is_c1_and_c2(self) -> None:
        """`checks` gates GC on (c1) and (c2) only; the class count is recorded."""
        rec = SMOKE.Recorder()
        rec.train_losses = [(0, 0.7), (1, 0.6)]
        rec.val_losses = [(0, 0.7), (1, 0.8)]
        rec.class_counts = [{"epoch": 0, "classes_predicted": 1}]
        rec.x_star_std = [0.4]
        base = {"stop_condition": None, "parameters_moved": 3,
                "epochs": [{"train_loss": 0.7}, {"train_loss": 0.6}]}
        ok = {"graphs_captured": 6, "graphs_expected": 6, "c1_ok": True, "c2_ok": True}
        rec.gc_checks = [{"epoch": 0, **ok}]
        self.assertTrue(SMOKE.checks("gc", rec, base)["c_no_collapse"])
        rec.gc_checks = [{"epoch": 0, **ok, "c2_ok": False}]
        self.assertFalse(SMOKE.checks("gc", rec, base)["c_no_collapse"])


if __name__ == "__main__":
    unittest.main()
