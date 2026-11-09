"""Reviewer probe (read-only): on the committed-pipeline acm_pa test files, how well does the
feature-only rule "paper support is a subset of author support" separate positives from the
HeaRT negatives? (Author features = OR of all own papers, held-out ones included.)"""
from pathlib import Path
import numpy as np

D = Path(r"C:\Users\Acer\Desktop\h2snn\H2S-NN\data\lp_derived\acm_pa")
feat = {}
with open(D / "node.dat", encoding="utf-8") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) == 4 and p[3]:
            feat[int(p[0])] = np.array([float(v) for v in p[3].split(",")]) > 0


def contained(u, v):
    return bool((feat[u] & ~feat[v]).sum() == 0)


for split in ("valid", "test"):
    pos = np.load(D / "heart" / f"heart_{split}_pos.npy")
    neg = np.load(D / "heart" / f"heart_{split}_samples.npy")
    pos_rate = np.mean([contained(u, v) for u, v in pos])
    sub = neg[:, ::10, :].reshape(-1, 2)  # every 10th negative
    neg_rate = np.mean([contained(u, v) for u, v in sub])
    # per-positive rank of the positive under the rule (contained=1, else 0), optimistic MRR-like
    hits = 0
    for i, (u, v) in enumerate(pos):
        if contained(u, v):
            n_tied = sum(contained(a, b) for a, b in neg[i])
            hits += n_tied == 0
    print(f"acm_pa {split}: positives with paper-support within author-support {pos_rate:.4f}; "
          f"HeaRT negatives (1/10 sample) {neg_rate:.4f}; positives ranked strictly first by the "
          f"rule alone: {hits}/{len(pos)}")
