"""J6.5 (spec v1.10 F19): per task and split, the positives whose endpoint has no training fact
of the target relation (link.dat target rows minus link.dat.valid; test facts are not in
link.dat), by side, and the HeaRT random-fill share (heart/meta.json). Same-type tasks count a
node's training facts in either orientation. Read only.
NOT RUN: J6.5 was not done, and is not required now (author decision 2026-11-10: build the LP
edges only, no further experiments or tests). Kept so it can be run later.
Run from H2S-NN/: python ../docs/lp_evidence/j65_cold_start.py"""
import json
from pathlib import Path

import numpy as np

TASKS = {"acm_pa": ("lp_derived/acm_pa", 2, False), "dblp_ap": ("lp_derived/dblp_ap", 0, False),
         "imdb_ma": ("lp_derived/imdb_ma", 2, False),
         "freebase_bb": ("lp_derived/freebase_bb", 0, True), "pubmed": ("PubMed", 2, True),
         "lastfm": ("LastFM", 0, False)}


def target_rows(path: Path, target: int) -> set:
    out = set()
    with open(path, encoding="utf-8") as h:
        for line in h:
            s, o, r = line.split("\t")[:3]
            if int(r) == target:
                out.add((int(s), int(o)))
    return out


rows = []
for task, (folder, target, same_type) in TASKS.items():
    d = Path("data") / folder
    train = target_rows(d / "link.dat", target) - target_rows(d / "link.dat.valid", target)
    if same_type:
        warm_src = warm_dst = {u for p in train for u in p}
    else:
        warm_src, warm_dst = {s for s, _ in train}, {o for _, o in train}
    meta = json.loads((d / "heart" / "meta.json").read_text(encoding="utf-8"))
    for split in ("valid", "test"):
        pos = np.load(d / "heart" / f"heart_{split}_pos.npy")
        cs = np.array([int(u) not in warm_src for u, _ in pos])
        cd = np.array([int(v) not in warm_dst for _, v in pos])
        cold = cs | cd
        m = meta[split]
        share = m["random_fill_entries"] / (m["positives"] * meta["num_samples"])
        rows.append((task, split, len(pos), int((~cold).sum()), int(cold.sum()),
                     int((cs & ~cd).sum()), int((cd & ~cs).sum()), int((cs & cd).sum()),
                     100.0 * share))
print("| Task | Split | Positives | Warm | Cold (either side) | Cold source only | "
      "Cold destination only | Cold both | Random-fill share |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]:,} | {r[3]:,} | {r[4]:,} ({100 * r[4] / r[2]:.1f} %) | "
          f"{r[5]:,} | {r[6]:,} | {r[7]:,} | {r[8]:.1f} % |")
