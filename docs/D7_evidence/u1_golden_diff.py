"""U1 golden rule: max abs difference of every recorded entry (init, eval/train logits,
gradients) of the current code vs docs/heads_evidence/v2_golden.pt, per case.
Run from H2S-NN/: python ../docs/D7_evidence/u1_golden_diff.py <reference.pt>"""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "H2S-NN"))
spec = importlib.util.spec_from_file_location("v2_golden", ROOT / "docs/heads_evidence/v2_golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

reference = torch.load(Path(sys.argv[1]), weights_only=False)
overall = 0.0
for case, _, overrides, ablations in golden.CASES:
    entry = reference[case]
    record = golden.run_case(golden._deserialize(entry["data"]), overrides, ablations)
    expected = entry["record"]
    assert set(record) == set(expected), case
    worst, worst_key, unequal = 0.0, "", 0
    for key, value in record.items():
        want = expected[key]
        if value is None or want is None:
            assert value is None and want is None, (case, key)
            continue
        assert value.dtype == want.dtype and value.shape == want.shape, (case, key)
        if not torch.equal(value, want):
            unequal += 1
            diff = float((value - want).abs().max())
            if diff > worst:
                worst, worst_key = diff, key
    overall = max(overall, worst)
    status = "bit-identical" if unequal == 0 else f"{unequal} entries differ, max|diff|={worst:.3e} ({worst_key})"
    print(f"{case:28s} {status}")
print(f"OVERALL max|diff| = {overall:.3e}  (rule: <= 1e-5 re-capture, > 1e-5 stop)")
