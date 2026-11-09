"""Reviewer probe (read-only): do IMDB movie features (kept, 'own') encode the movie's actors
through actor_{1,2,3}_facebook_likes (info.dat attributes 3, 4, 12; z-scored columns 3, 4, 12)?

Scorer s(m, a) = number of distinct values among m's columns {3,4,12} that also occur in those
columns of a's TRAINING movies (link.dat rel 2 minus link.dat.valid), excluding values carried
by more than `freq_cap` movies (missing/zero fill). Ranked on the stored HeaRT lists with HeaRT's
rank. Run from H2S-NN/.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

task = Path("data/lp_derived/imdb_ma")
COLS = (3, 4, 12)
TARGET = 2

movie_vals = {}
with open(task / "node.dat", encoding="utf-8") as h:
    for line in h:
        p = line.rstrip("\r\n").split("\t")
        if p[2] == "0":
            v = p[3].split(",")
            movie_vals[int(p[0])] = tuple(v[c] for c in COLS)   # exact text, no float rounding


def rows(path):
    out = set()
    with open(path, encoding="utf-8") as h:
        for line in h:
            s, o, r = line.rstrip("\r\n").split("\t")[:3]
            if int(r) == TARGET:
                out.add((int(s), int(o)))
    return out


train = rows(task / "link.dat") - rows(task / "link.dat.valid")
freq = Counter(v for vals in movie_vals.values() for v in set(vals))
print("movies", len(movie_vals), "train facts", len(train))
print("most common values:", freq.most_common(5))


def rank(pos, neg):
    ge = (neg >= pos[:, None]).sum(1)
    gt = (neg > pos[:, None]).sum(1)
    return 0.5 * (ge + gt) + 1.0


for freq_cap in (10**9, 50, 5):
    actor_vals = defaultdict(set)
    for m, a in train:
        for v in movie_vals[m]:
            if freq[v] <= freq_cap:
                actor_vals[a].add(v)

    def score(m, a):
        return len({v for v in movie_vals[m] if freq[v] <= freq_cap} & actor_vals.get(a, set()))

    for split in ("valid", "test"):
        pos = np.load(task / "heart" / f"heart_{split}_pos.npy")
        neg = np.load(task / "heart" / f"heart_{split}_samples.npy")
        ps = np.array([score(int(m), int(a)) for m, a in pos], dtype=float)
        ns = np.array([[score(int(m), int(a)) for m, a in lst] for lst in neg], dtype=float)
        r = rank(ps, ns)
        warm = np.array([int(a) in actor_vals for _, a in pos])
        print(f"cap {freq_cap} {split}: lists {len(pos)} MRR {np.mean(1/r):.4f} "
              f"Hits@10 {np.mean(r <= 10):.4f}; positives with score>0 {np.mean(ps > 0):.3f}; "
              f"warm actors {warm.mean():.3f}; MRR on warm {np.mean(1/r[warm]):.4f}; "
              f"pos shape {pos.shape} neg shape {neg.shape}")
