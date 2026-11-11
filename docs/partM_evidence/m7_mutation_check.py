"""Part M, M7 mutation check: re-enable a random fallback (a head without 2-hop candidates gets
uniform random tails, as the pre-M7 loader did) and run the loader tests of
tests/test_hgb_lp_negatives.py; a test must fail. Restore; all pass. Run from H2S-NN/."""
from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "H2S-NN"))

import torch  # noqa: E402

import src.data.link_prediction_dataset as lpd  # noqa: E402
from src.data.hgb_lp_negatives import HGBTestNegatives  # noqa: E402

original = lpd.hgb_test_negatives


def with_fallback(test_pos, adjacency, num_nodes, dst_range, seed=1):
    result = original(test_pos, adjacency, num_nodes, dst_range, seed)
    rng = random.Random(seed + 7)
    extra = []
    for head in result.heads_without_candidates.tolist():
        k = int((test_pos[0] == head).sum())
        extra += [(head, rng.randrange(*dst_range)) for _ in range(k)]
    if not extra:
        return result
    edges = torch.cat([result.edge_index, torch.tensor(extra).t()], dim=1)
    return HGBTestNegatives(edges, torch.empty(0, dtype=torch.long), 0)


def run(label: str) -> unittest.TestResult:
    sys.modules.pop("tests.test_hgb_lp_negatives", None)
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.test_hgb_lp_negatives.LoaderNegativeTests")
    result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w")).run(suite)
    failed = [t.id().split(".")[-1] for t, _ in result.failures + result.errors]
    print(f"{label}: ran {result.testsRun}, failed {len(failed)}: {failed}")
    return result


lpd.hgb_test_negatives = with_fallback
mutated = run("MUTATION random fallback re-enabled")
lpd.hgb_test_negatives = original
clean = run("restored")
ok = bool(mutated.failures or mutated.errors) and clean.wasSuccessful()
print("mutation caught" if ok else "MUTATION NOT CAUGHT")
sys.exit(0 if ok else 1)
