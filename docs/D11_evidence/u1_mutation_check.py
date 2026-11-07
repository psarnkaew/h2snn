"""U1 three-mutation check: remove each baseline data pin in turn; the bundle-identity tests must fail."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import src.baselines.bundles as bundles
from tests.test_paths_and_facts import BaselineBundleIdentityTests as T

names = ["test_node_bundle_equals_explicit_untied_dfs", "test_pins_cannot_be_overridden"]
original = dict(bundles.BASELINE_DATA_PINS)
for key in list(original):
    bundles.BASELINE_DATA_PINS.clear()
    bundles.BASELINE_DATA_PINS.update({k: v for k, v in original.items() if k != key})
    res = unittest.TextTestRunner(stream=open("nul" if sys.platform == "win32" else "/dev/null", "w")).run(
        unittest.TestSuite(T(n) for n in names[:1]))
    print(f"mutation: pin {key!r} removed -> failures={len(res.failures)} errors={len(res.errors)} "
          f"-> {'CAUGHT' if (res.failures or res.errors) else 'NOT CAUGHT'}")
bundles.BASELINE_DATA_PINS.clear(); bundles.BASELINE_DATA_PINS.update(original)
res = unittest.TextTestRunner(stream=open("nul", "w")).run(unittest.TestSuite(T(n) for n in names))
print(f"control (unmutated): failures={len(res.failures)} errors={len(res.errors)}")
