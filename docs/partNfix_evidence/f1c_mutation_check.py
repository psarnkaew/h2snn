"""Part N-fix F1c: F1's two mutation checks against the HGB oracle tests.

M1: the random set with the test positives in another order (reversed).
M2: the 2-hop adjacency with validation in both orientations.
Each mutation is applied to `src/data/hgb_lp_negatives.py` in place, the oracle class
`tests.test_hgb_oracle.HGBOracleTests` runs in a fresh interpreter, and the original text is
restored (also on error). Run from `H2S-NN/`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TARGET = Path("src/data/hgb_lp_negatives.py")
MUTATIONS = {
    "M1 random set, positives in another order": (
        "                      axis=0)                                       # row-major sorted, unique\n",
        "                      axis=0)[::-1]                                 # MUTATION: reversed\n",
    ),
    "M2 2-hop adjacency, validation both ways": (
        "    rows = np.concatenate([r1, c1, r2])\n    cols = np.concatenate([c1, r1, c2])\n",
        "    rows = np.concatenate([r1, c1, r2, c2])  # MUTATION: validation both ways\n"
        "    cols = np.concatenate([c1, r1, c2, r2])\n",
    ),
}


def run_oracle() -> tuple[int, str]:
    done = subprocess.run([sys.executable, "-m", "unittest", "tests.test_hgb_oracle.HGBOracleTests"],
                          capture_output=True, text=True)
    lines = [line for line in done.stderr.splitlines()
             if line.startswith(("FAIL:", "ERROR:", "Ran ", "OK", "FAILED"))]
    return done.returncode, "\n".join(lines)


def main() -> None:
    original = TARGET.read_text(encoding="utf-8")
    try:
        for name, (old, new) in MUTATIONS.items():
            assert original.count(old) == 1, name
            TARGET.write_text(original.replace(old, new), encoding="utf-8")
            code, summary = run_oracle()
            print(f"== {name}: {'CAUGHT' if code != 0 else 'NOT CAUGHT'}\n{summary}\n")
            TARGET.write_text(original, encoding="utf-8")
    finally:
        TARGET.write_text(original, encoding="utf-8")
    code, summary = run_oracle()
    print(f"== restored: {'clean' if code == 0 else 'FAILING'}\n{summary}")


if __name__ == "__main__":
    main()
