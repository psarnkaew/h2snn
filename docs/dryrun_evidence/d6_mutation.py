"""Card 2 D6.8 mutation check: the final-seeds test must fail when the selection takes the last trial instead of the
best by the monitor. The mutated launcher lives in memory only (no file is changed).
    python ../docs/dryrun_evidence/d6_mutation.py          (from H2S-NN/)
"""
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
source = Path("scripts/run_baseline_tuning.py").read_text()
original = "    return max(records, key=lambda record: metric_score(record, monitor))\n"
assert source.count(original) == 1, "selection line not found"
mutated = source.replace(original, "    return records[-1]  # MUTATION: the last trial\n")
import scripts  # noqa: E402

spec = importlib.util.spec_from_loader("scripts.run_baseline_tuning", loader=None)
module = importlib.util.module_from_spec(spec)
module.__file__ = str(Path("scripts/run_baseline_tuning.py").resolve())
exec(compile(mutated, module.__file__, "exec"), module.__dict__)
sys.modules["scripts.run_baseline_tuning"] = module
scripts.run_baseline_tuning = module
result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromName(
    "tests.test_baseline_final_seeds.BaselineFinalSeedsTests.test_best_trial_by_monitor_and_its_params_unchanged_per_seed"))
caught = not result.wasSuccessful()
print(f"MUTATION select-last-trial: {'caught (test fails)' if caught else 'NOT caught'}")
sys.exit(0 if caught else 1)
