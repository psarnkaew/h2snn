"""I6 mutations: each must make the overlap bit-identity test fail.
Run from H2S-NN/: python ../docs/WISP_evidence/i6_mutation_check.py"""
import subprocess
import sys
from pathlib import Path

TEST = "tests.test_wisp.WispOverlapSamplingTests.test_overlap_is_bit_identical_for_both_proposals"
MUTATIONS = {
    "prefetch draws the wrong epoch's seed": (
        "pending = executor.submit(self._draw_groups, data, epoch + 1, None)",
        "pending = executor.submit(self._draw_groups, data, epoch + 2, None)"),
    "adaptive prefetch drops the proposal (q = p)": (
        "pending = executor.submit(self._draw_groups, data, epoch + 1,\n                                          self._draw_proposal(data))",
        "pending = executor.submit(self._draw_groups, data, epoch + 1,\n                                          None)"),
}
path = Path("src/trainers/trainer.py")
original = path.read_text(encoding="utf-8")
caught = 0
for name, (old, new) in MUTATIONS.items():
    assert original.count(old) == 1, name
    path.write_text(original.replace(old, new), encoding="utf-8")
    try:
        out = subprocess.run([sys.executable, "-m", "unittest", TEST], capture_output=True, text=True)
    finally:
        path.write_text(original, encoding="utf-8")
    caught += out.returncode != 0
    print(f"{name}: {'CAUGHT' if out.returncode else 'NOT CAUGHT'} ({out.stderr.strip().splitlines()[-1]})")
clean = subprocess.run([sys.executable, "-m", "unittest", TEST], capture_output=True, text=True)
print(f"caught {caught}/{len(MUTATIONS)}; clean after restore: {clean.stderr.strip().splitlines()[-1]}")
