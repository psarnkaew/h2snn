"""Part N, N1 mutation checks (run from H2S-NN/):
(a) validation positives added in both orientations to HGB's adjacency -> a test must fail;
(b) the random set without the redraw (a known neighbour may be kept) -> a test must fail.
Each mutation is undone and the tests pass again."""
from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "H2S-NN"))

import torch  # noqa: E402

import src.data.hgb_lp_negatives as neg  # noqa: E402

ORIGINAL_ADJ, ORIGINAL_RANDOM = neg.hgb_adjacency, neg.hgb_random_test_negatives


def adjacency_both_ways(both_ways, one_way, num_nodes):
    return ORIGINAL_ADJ(torch.cat([both_ways.reshape(2, -1), one_way.reshape(2, -1)], dim=1),
                        torch.empty((2, 0), dtype=torch.long), num_nodes)


def random_without_redraw(test_pos, both_ways, one_way, num_nodes, dst_range, seed=1):
    import numpy as np

    pairs = np.unique(test_pos.numpy().T.reshape(-1, 2), axis=0)
    rng = random.Random(seed)
    heads = [h for h, _ in pairs.tolist()]
    tails = [rng.randrange(*dst_range) for _ in heads]
    return torch.tensor([heads, tails], dtype=torch.long).reshape(2, -1)


def run(label: str, name: str) -> unittest.TestResult:
    sys.modules.pop("tests.test_hgb_lp_negatives", None)
    suite = unittest.defaultTestLoader.loadTestsFromName(name)
    result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w")).run(suite)
    failed = [t.id().split(".")[-1] for t, _ in result.failures + result.errors]
    print(f"{label}: ran {result.testsRun}, failed {len(failed)}: {failed}")
    return result


ok = True
neg.hgb_adjacency = adjacency_both_ways
m = run("MUTATION (a) validation in both orientations",
        "tests.test_hgb_lp_negatives.TwoHopCandidateTests")
neg.hgb_adjacency = ORIGINAL_ADJ
c = run("restored (a)", "tests.test_hgb_lp_negatives.TwoHopCandidateTests")
ok &= bool(m.failures or m.errors) and c.wasSuccessful()

neg.hgb_random_test_negatives = random_without_redraw
m = run("MUTATION (b) random set without the redraw", "tests.test_hgb_lp_negatives.RandomSetTests")
neg.hgb_random_test_negatives = ORIGINAL_RANDOM
c = run("restored (b)", "tests.test_hgb_lp_negatives.RandomSetTests")
ok &= bool(m.failures or m.errors) and c.wasSuccessful()
print("both mutations caught" if ok else "MUTATION NOT CAUGHT")
sys.exit(0 if ok else 1)
