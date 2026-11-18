"""TSUBAME card T2 mutation: the guard never fires (MAX_UNFINISHED_ATTEMPTS = infinity) -> test (1) must fail.

    python ../docs/tsubame_evidence/t2_mutation.py          (from H2S-NN/; exit 0 = the mutation was caught)
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))

from src.utils import wandb_campaign  # noqa: E402
from tests import test_tsubame_guard  # noqa: E402

with mock.patch.object(wandb_campaign, "MAX_UNFINISHED_ATTEMPTS", float("inf")):
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "GuardTests.test_always_dying_trial_is_started_twice_then_exceeds_24h", test_tsubame_guard)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
caught = not result.wasSuccessful()
print("MUTATION T2 (limit = inf):", "caught (test (1) fails)" if caught else "NOT CAUGHT")
sys.exit(0 if caught else 1)
