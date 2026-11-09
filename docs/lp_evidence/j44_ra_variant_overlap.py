"""J4.4: how much does the same-type pair heuristic matter? HeaRT's code computes CN(a,v)/deg(v)
under the name RA (j44_heart_ra_semantics.txt). This compares the primary files (`heart/`, HeaRT
as coded) with the variant built with textbook RA (`heart_variant_ra/`, same seed, same PPR and
features): per list, the share of its 500 negatives that both variants contain.
Run from H2S-NN/ after:
  python scripts/lp/build_heart_negatives.py --tasks pubmed freebase_bb --cn-metric ra \
      --out-subdir heart_variant_ra
  python ../docs/lp_evidence/j44_ra_variant_overlap.py"""
from pathlib import Path

import numpy as np

FOLDERS = {"pubmed": Path("data/PubMed"), "freebase_bb": Path("data/lp_derived/freebase_bb")}
for task, folder in FOLDERS.items():
    for split in ("valid", "test"):
        a = np.load(folder / "heart" / f"heart_{split}_samples.npy")
        b = np.load(folder / "heart_variant_ra" / f"heart_{split}_samples.npy")
        assert a.shape == b.shape
        shared = []
        for la, lb in zip(a, b):
            sa = {tuple(p) for p in la.tolist()}
            sb = {tuple(p) for p in lb.tolist()}
            shared.append(len(sa & sb) / len(sa))
        shared = np.array(shared)
        print(f"{task} {split}: {len(shared)} lists; share of negatives in both variants: "
              f"mean {shared.mean():.3f}, median {np.median(shared):.3f}, "
              f"min {shared.min():.3f}, max {shared.max():.3f}")
