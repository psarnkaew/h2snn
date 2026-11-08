"""H4 negative controls: break one read-out property at a time; a T-H11 test must fail."""
import subprocess, sys
from pathlib import Path
path = Path("src/analysis/wisp_readouts.py")
original = path.read_text(encoding="utf-8")
mutations = {
    "pair weight w/|f| instead of w/(2|f|)": [("return weights * total / (2.0 * size)", "return weights * total / size")],
    "pair-form degree ignores keep (counts all pairs)": [("degree = self.keep.sum(dim=2, keepdim=True)", "degree = torch.full_like(self.keep.sum(dim=2, keepdim=True), self.size - 1.0)")],
    "I_sigma not normalised": [("    grand = sum(totals.values())\n    return {k: v / grand for k, v in totals.items()}\n\n\ndef i_v_sigma", "    return dict(totals)\n\n\ndef i_v_sigma")],
    "schema mask drops nothing": [("            keep[c, i, j] = 0.0\n            keep[c, j, i] = 0.0\n", "            pass\n")],
    "schema mask drops one order only": [("            keep[c, j, i] = 0.0\n", "")],
}
for name, edits in mutations.items():
    text = original
    for old, new in edits:
        assert text.count(old) == 1, name
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    try:
        out = subprocess.run([sys.executable, "-m", "unittest", "tests.test_wisp", "-k", "TH11"],
                             capture_output=True, text=True)
        tail = out.stderr.strip().splitlines()[-1]
        print(f"{name}: {'CAUGHT' if out.returncode else 'NOT CAUGHT'} ({tail})")
    finally:
        path.write_text(original, encoding="utf-8")
