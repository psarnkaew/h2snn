"""Card 10 L3 mutation: site_split.py writes a lab line also to the TSUBAME list (a line in both) ->
SiteSplitTests.test_split_disjoint_union_and_rules fails.

    python ../docs/card10_evidence/mutation_l3.py          (from H2S-NN/; exit 0 = the mutation was caught)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from scripts.slurm import site_split  # noqa: E402
from tests import test_site_split  # noqa: E402

original = site_split.split


def mutated(lines, fit, lab_out, t4_out):
    lab, t4, errors = original(lines, fit, lab_out, t4_out)
    return lab, [x for x in lines if x in set(t4) | set(lab)], errors   # every lab line also in t4


site_split.split = mutated
suite = unittest.defaultTestLoader.loadTestsFromName("SiteSplitTests.test_split_disjoint_union_and_rules",
                                                     test_site_split)
result = unittest.TextTestRunner(verbosity=2).run(suite)
caught = not result.wasSuccessful()
print("MUTATION L3 (a line in both lists):", "caught" if caught else "NOT CAUGHT")
sys.exit(0 if caught else 1)
