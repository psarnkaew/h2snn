"""U1: a re-capture must change only the recorded outputs, never the fixture data.
python ../docs/D7_evidence/u1_golden_fixture_check.py <old.pt> <new.pt>"""
import sys
import torch

def walk(a, b, path=""):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        return sum(walk(a[k], b[k], f"{path}/{k}") for k in a)
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        return sum(walk(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b)))
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), path
        return 1
    assert a == b, path
    return 1

old = torch.load(sys.argv[1], weights_only=False)
new = torch.load(sys.argv[2], weights_only=False)
assert set(old) == set(new)
n = sum(walk(old[c]["data"], new[c]["data"], c) for c in old)
print(f"fixture data identical across {len(old)} cases ({n} leaves)")
