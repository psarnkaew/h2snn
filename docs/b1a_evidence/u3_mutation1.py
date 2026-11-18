"""Card 1f U3 Mutation 1, in-process (no source file edited): the loops keep the last epoch's weights instead of
restoring the best (`load_state_dict` on the toy model is a no-op). Expected: test_best_weights_are_restored FAILS.
Run from H2S-NN/."""
import sys
import unittest
from unittest import mock

import torch

from tests import test_b1a_validation_subset as module

suite = unittest.defaultTestLoader.loadTestsFromName("B1aTests.test_best_weights_are_restored", module)
module.setUpModule()
try:
    with mock.patch.object(torch.nn.Linear, "load_state_dict", lambda self, *args, **kwargs: None):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
finally:
    module.tearDownModule()
caught = len(result.failures) + len(result.errors)
print(f"MUTATION 1: {'CAUGHT' if caught and not result.errors else 'NOT CAUGHT'} "
      f"(failures={len(result.failures)}, errors={len(result.errors)})")
sys.exit(0 if caught and not result.errors else 1)
