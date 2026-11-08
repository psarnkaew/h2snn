"""I1 mutations (pre-registered): each must make a rewritten T-H7 test fail.
Run from H2S-NN/: python ../docs/WISP_evidence/i1_mutation_check.py"""
import subprocess
import sys
from pathlib import Path

path = Path("src/models/h2snn_model.py")
original = path.read_text(encoding="utf-8")
mutations = {
    "far-end off-by-one in the endpoints encoder (review MINOR-6)":
        ("endpoints = face_node_index[:, [0, length]]", "endpoints = face_node_index[:, [0, length - 1]]"),
    "g reads [h0, c] by default (the v1.3 input)":
        ("self.path_weight_context = ablation.path_weight_context", "self.path_weight_context = True"),
}
for name, (old, new) in mutations.items():
    assert original.count(old) == 1, name
    path.write_text(original.replace(old, new), encoding="utf-8")
    try:
        out = subprocess.run([sys.executable, "-m", "unittest", "tests.test_wisp", "-k", "TH7"],
                             capture_output=True, text=True)
        failed = [l for l in out.stderr.splitlines() if l.startswith(("FAIL:", "ERROR:"))]
        print(f"{name}: {'CAUGHT' if out.returncode else 'NOT CAUGHT'} ({out.stderr.strip().splitlines()[-1]})")
        for line in failed[:6]:
            print("   ", line[:150])
    finally:
        path.write_text(original, encoding="utf-8")
