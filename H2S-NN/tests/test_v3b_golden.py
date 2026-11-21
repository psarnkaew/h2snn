"""Legacy-removal card, LR-1: the v3-B golden (`docs/legacy_evidence/v3b_golden_cluster.pt`, captured on the
code before any removal, g2n01, E, 1 thread) is reproduced bit for bit: every configuration's evaluation
logits / scores, training loss, parameter gradients and anchored-solver statistics (NC arms and ladder
rungs, LP decoders, GC poolings, mini-batch NC and LP; `docs/legacy_evidence/v3b_golden.py`).

It replaces the v2 golden tests (`test_task_heads` golden, `test_cooperative` T1), which captured the v2
model. Like the CS8 bit-exact tests it first asserts the pinned kernel environment E and compares with
1 thread; a failure names every differing tensor.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data
from tests.pinned_kernels import assert_kernel_path_pinned, one_thread

DOCS = Path(__file__).resolve().parents[2] / "docs" / "legacy_evidence"
GOLDEN_SCRIPT = DOCS / "v3b_golden.py"
GOLDEN_REFERENCE = DOCS / "v3b_golden_cluster.pt"


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


class V3BGoldenTests(unittest.TestCase):
    def test_every_v3b_configuration_is_bit_identical_to_the_capture(self) -> None:
        assert_kernel_path_pinned(self)
        spec = importlib.util.spec_from_file_location("v3b_golden", GOLDEN_SCRIPT)
        golden = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(golden)  # type: ignore[union-attr]
        stored = torch.load(GOLDEN_REFERENCE, weights_only=False)
        with one_thread():
            records, skipped = golden.references()
        self.assertEqual(sorted(skipped), sorted(stored["skipped"]),
                         "the configurations that cannot be built changed")
        ref, now = {}, {}
        golden._flatten("", stored["records"], ref)
        golden._flatten("", records, now)
        self.assertGreater(len(ref), 1000)
        differing = [k for k in ref if k not in now or ref[k].dtype != now[k].dtype
                     or ref[k].shape != now[k].shape or not torch.equal(ref[k], now[k])]
        # Card 12 amendment 1 (author, 2026-11-22): the one stats key added after the capture
        # (spec 1.30, the PCG zero right-hand-side rule's log field) is not in the capture; only
        # that key is ignored. Every captured tensor above keeps its bit-identity check.
        new = sorted(k for k in set(now) - set(ref)
                     if k.rsplit(".", 1)[-1] != "pcg_zero_rhs_scopes")
        self.assertEqual((differing, new), ([], []),
                         f"{len(differing)} differing tensors: {differing[:40]}")


if __name__ == "__main__":
    unittest.main()
