"""J5.5 diagnosis of acm_pa's new random fill: is every randomly filled half-list the one whose
kept endpoint has no training fact of the target relation (a cold-start node: no L3 path, its PPR
mass stays on itself and its other relations, and, for an author, a zero rebuilt feature)?
A half-list is "random" when none of its 250 entries scores above zero; HeaRT then fills it
uniformly. Read only. Run from H2S-NN/: python ../docs/lp_evidence/j55_acm_cold_start.py"""
import json
from pathlib import Path

import numpy as np

D = Path("data/lp_derived/acm_pa")
meta = json.loads((D / "heart" / "meta.json").read_text(encoding="utf-8"))
train = set()
for line in (D / "link.dat").read_text(encoding="utf-8").splitlines():
    s, o, r = line.split("\t")[:3]
    if r == "2":
        train.add((int(s), int(o)))
for line in (D / "link.dat.valid").read_text(encoding="utf-8").splitlines():
    s, o, r = line.split("\t")[:3]
    train.discard((int(s), int(o)))
papers_with_fact = {p for p, _ in train}
authors_with_fact = {a for _, a in train}
for split in ("valid", "test"):
    pos = np.load(D / "heart" / f"heart_{split}_pos.npy")
    cold_author = sum(int(a) not in authors_with_fact for _, a in pos)
    cold_paper = sum(int(p) not in papers_with_fact for p, _ in pos)
    filled = meta[split]["lists_with_random_fill"]
    print(f"acm_pa {split}: positives {len(pos)}; kept author without a training fact "
          f"{cold_author}; kept paper without a training fact {cold_paper}; half-lists with "
          f"random fill {filled} (entries {meta[split]['random_fill_entries']} = "
          f"{meta[split]['random_fill_entries'] / 250:.0f} x 250)")
