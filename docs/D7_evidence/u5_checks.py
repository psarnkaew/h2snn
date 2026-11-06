"""U5 evidence: T8 statistics (sum vs dedup control), T16 uniform_instance counts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
from tests import test_paths_and_facts as T

est = T.EstimatorTests()
for merge in ("sum", "drop_duplicates"):
    rel, z = est._statistics(merge)
    print(f"T8 merge={merge:16s} draws={est.DRAWS} r={est.R}: rel err {rel:.4f} (pass <= 0.02), max|z| {z:.2f} (sum: <= 5, dedup control: > 10)")
u = T.UniformInstanceTests()
for sampler in ("uniform_instance", "walk"):
    c = u._counts(sampler, 36000).double()
    e = c.sum() / 36
    print(f"T16 {sampler:16s}: accepted {int(c.sum())}, per-instance counts min {int(c.min())} max {int(c.max())}, expected-if-uniform {float(e):.1f}, max|dev|/sqrt(E) {float(((c-e).abs()/e.sqrt()).max()):.2f}")
