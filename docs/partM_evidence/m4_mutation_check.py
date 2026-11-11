"""Part M, M4 mutation check: restrict the link-prediction anchors to one endpoint type (the head
type) and run tests/test_wisp_tasks.py; at least one test must fail. Then restore and rerun (all
pass). Run from H2S-NN/."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "H2S-NN"))

import torch  # noqa: E402

import src.data.wisp_tasks as wt  # noqa: E402

original = wt.link_prediction_anchors


def heads_only(node_slices, target_relation):
    return torch.arange(*node_slices[target_relation[0]])


def run(label: str) -> unittest.TestResult:
    sys.modules.pop("tests.test_wisp_tasks", None)  # re-import: it binds the current function
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_wisp_tasks")
    result = unittest.TextTestRunner(verbosity=0, stream=open("nul" if sys.platform == "win32"
                                                                 else "/dev/null", "w")).run(suite)
    failed = [t.id().split(".")[-1] for t, _ in result.failures + result.errors]
    print(f"{label}: ran {result.testsRun}, failed {len(failed)}: {failed}")
    return result


wt.link_prediction_anchors = heads_only
mutated = run("MUTATION anchors = head type only")
wt.link_prediction_anchors = original
clean = run("restored")
ok = (len(mutated.failures) + len(mutated.errors) > 0) and clean.wasSuccessful()
print("mutation caught" if ok else "MUTATION NOT CAUGHT")
sys.exit(0 if ok else 1)
