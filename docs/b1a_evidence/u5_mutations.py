"""Card 1f U5 mutations, in-process (no source file edited). Run from H2S-NN/ with PYTHONPATH=. Exit 0 iff both are caught.
- PDNS: every seed's subset is drawn from the seed-0 split -> test_pdns_subset_is_drawn_per_run_seed FAILS.
- Hetero2Net Mutation 1: the loop keeps the last epoch's weights (`load_state_dict` a no-op)
  -> test_hetero2net_loop_has_no_test_pass_and_restores_best_weights FAILS."""
import sys
import unittest
from unittest import mock

import torch

import src.evaluation.validation_subset as vs
from tests import test_b1a_validation_subset as module

original = vs.validation_subset


def seed0_split(dataset, val_ids, run_seed=None):
    if dataset == "pdns":
        return original(dataset, module.pdns_split(0)["a"].val_mask.nonzero().view(-1), 0)
    return original(dataset, val_ids, run_seed)


MUTATIONS = {
    "PDNS seed-0 split": ("B1aReviewTests.test_pdns_subset_is_drawn_per_run_seed",
                          mock.patch.object(vs, "validation_subset", seed0_split)),
    "Hetero2Net Mutation 1": ("B1aReviewTests.test_hetero2net_loop_has_no_test_pass_and_restores_best_weights",
                              mock.patch.object(torch.nn.Linear, "load_state_dict", lambda self, *a, **k: None)),
}
module.setUpModule()
all_caught = True
try:
    for name, (test, patch) in MUTATIONS.items():
        suite = unittest.defaultTestLoader.loadTestsFromName(test, module)
        with patch:
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        caught = bool(result.failures) and not result.errors
        all_caught &= caught
        print(f"MUTATION {name}: {'CAUGHT' if caught else 'NOT CAUGHT'} "
              f"(failures={len(result.failures)}, errors={len(result.errors)})", flush=True)
finally:
    module.tearDownModule()
sys.exit(0 if all_caught else 1)
