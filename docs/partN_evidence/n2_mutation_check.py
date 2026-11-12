"""Part N, N2 mutation check (run from H2S-NN/): re-add the exclusion of known positives to the
training negatives (redraw while the pair is a training positive, at most 100 tries, then drop the
draw, as the pre-N2 sampler effectively did) -> a test must fail; restored -> all pass."""
from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "H2S-NN"))

import torch  # noqa: E402

import src.data.hgb_lp_negatives as neg  # noqa: E402

ORIGINAL = neg.hgb_train_negatives


def excluding(pos_edge_index, dst_range, seed):
    rng = random.Random(seed)
    known = set(map(tuple, pos_edge_index.t().tolist()))
    heads, tails = [], []
    for head in pos_edge_index[0].tolist():
        for _ in range(100):
            tail = rng.randrange(*dst_range)
            if (head, tail) not in known:
                heads.append(head)
                tails.append(tail)
                break
    return torch.tensor([heads, tails], dtype=torch.long).reshape(2, -1)


def run(label: str) -> unittest.TestResult:
    sys.modules.pop("tests.test_hgb_lp_negatives", None)
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.test_hgb_lp_negatives.TrainNegativeTests")
    result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w")).run(suite)
    failed = [t.id().split(".")[-1] for t, _ in result.failures + result.errors]
    print(f"{label}: ran {result.testsRun}, failed {len(failed)}: {failed}")
    return result


neg.hgb_train_negatives = excluding
mutated = run("MUTATION exclusion of known positives re-added")
neg.hgb_train_negatives = ORIGINAL
clean = run("restored")
ok = bool(mutated.failures or mutated.errors) and clean.wasSuccessful()
print("mutation caught" if ok else "MUTATION NOT CAUGHT")
sys.exit(0 if ok else 1)
