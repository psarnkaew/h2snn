"""Card 12 unit P mutations: each must make `tests.test_anchored_solve` fail.

For each mutation the script copies `src/` and `tests/` of this worktree into a temporary directory
(the repository is never modified), applies one text replacement to the copy of
`src/models/anchored_solve.py` (the old text must occur exactly once), runs
`python -m unittest tests.test_anchored_solve` there and records which tests failed. The unmutated
copy is run first as the control (must pass). Exit 0 iff the control passes and every mutation fails.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # H2S-NN/
TARGET = "src/models/anchored_solve.py"
MUTATIONS = (
    ("remove_the_rule",
     "    zero_rhs = zero_rhs_mask(b_norm)\n",
     "    zero_rhs = torch.zeros_like(b_norm, dtype=torch.bool)\n"),
    ("rho_zero",
     "ZERO_RHS_RATIO = 1e-10\n",
     "ZERO_RHS_RATIO = 0.0\n"),
    ("triggered_rows_y0",
     "        y = torch.where(per_row(zero_rhs), torch.zeros_like(y), y)\n",
     "        y = torch.where(per_row(zero_rhs), y0, y)\n"),
)


def run(name: str, old: str | None, new: str | None) -> tuple[int, list[str]]:
    with tempfile.TemporaryDirectory(prefix=f"c12mut_{name}_") as tmp:
        tmp_path = Path(tmp)
        for sub in ("src", "tests"):
            shutil.copytree(ROOT / sub, tmp_path / sub, symlinks=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
        for link in ("vendor", "data", "configs"):
            if (ROOT / link).exists():
                os.symlink(ROOT / link, tmp_path / link)
        if old is not None:
            path = tmp_path / TARGET
            text = path.read_text()
            assert text.count(old) == 1, f"{name}: mutation text found {text.count(old)} times"
            path.write_text(text.replace(old, new))
        proc = subprocess.run([sys.executable, "-m", "unittest", "-v", "tests.test_anchored_solve"],
                              cwd=tmp_path, capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        failed = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\S+)", out, flags=re.M)))
        summary = [line for line in out.splitlines() if line.startswith(("Ran ", "OK", "FAILED"))]
        print(f"== {name}: exit {proc.returncode}; {' | '.join(summary)}")
        for test in failed:
            print(f"   failed: {test}")
        return proc.returncode, failed


def main() -> int:
    ok = True
    code, _ = run("control_unmutated", None, None)
    if code != 0:
        print("control FAILED: the unmutated copy must pass")
        ok = False
    for name, old, new in MUTATIONS:
        code, failed = run(name, old, new)
        caught = code != 0 and bool(failed)
        print(f"   mutation {name}: {'CAUGHT' if caught else 'NOT CAUGHT'}")
        ok = ok and caught
    print("mutations: all caught" if ok else "mutations: NOT all caught")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
