"""Are endpoint features a function of the target relation? (DBLP authors vs their papers, IMDB
actors vs their movies.) Read only. Compares nodes with identical neighbour sets to random pairs."""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
d, rel, node_t, other_t = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
feat, typ = {}, {}
with open(d / "node.dat", encoding="utf-8") as f:
    for line in f:
        p = line.rstrip("\n").split("\t"); typ[int(p[0])] = int(p[2])
        if len(p) == 4 and p[3]: feat[int(p[0])] = tuple(float(v) for v in p[3].split(","))
nbr = defaultdict(set)
with open(d / "link.dat", encoding="utf-8") as f:
    for line in f:
        s, o, r, _ = line.rstrip("\n").split("\t"); s, o, r = int(s), int(o), int(r)
        if r == rel:
            a, b = (s, o) if typ[s] == node_t else (o, s); nbr[a].add(b)
groups = defaultdict(list)
for a, ns in nbr.items(): groups[frozenset(ns)].append(a)
same_set_pairs = [(g[0], x) for g in groups.values() if len(g) > 1 for x in g[1:]]
eq_same = sum(feat[a] == feat[b] for a, b in same_set_pairs)
nodes = list(nbr); rng = np.random.default_rng(0)
rp = [(nodes[i], nodes[j]) for i, j in rng.integers(0, len(nodes), (5000, 2)) if i != j]
eq_rand = sum(feat[a] == feat[b] for a, b in rp)
print(f"pairs with identical neighbour sets: {len(same_set_pairs)}, identical features {eq_same}; "
      f"random pairs {len(rp)}, identical features {eq_rand}")
