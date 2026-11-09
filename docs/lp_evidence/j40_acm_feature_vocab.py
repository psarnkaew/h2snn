"""Do ACM paper and author features share one vocabulary? Test: is each author's vector an
aggregate (OR / sum / mean) of the vectors of the papers they wrote (paper-author, rel 2)?"""
import sys
from pathlib import Path
import numpy as np
d = Path(sys.argv[1])
feat, typ = {}, {}
with open(d / "node.dat", encoding="utf-8") as f:
    for line in f:
        p = line.rstrip("\n").split("\t"); typ[int(p[0])] = int(p[2])
        if len(p) == 4 and p[3]:
            feat[int(p[0])] = np.array([float(v) for v in p[3].split(",")])
papers = sorted(i for i, t in typ.items() if t == 0); authors = sorted(i for i, t in typ.items() if t == 1)
P = np.stack([feat[i] for i in papers]); A = np.stack([feat[i] for i in authors])
print("paper feat values", np.unique(P)[:5], "nnz/row", (P != 0).sum(1).mean())
print("author feat values", np.unique(A)[:8], "nnz/row", (A != 0).sum(1).mean())
pidx = {i: k for k, i in enumerate(papers)}; aidx = {i: k for k, i in enumerate(authors)}
wrote = {}
with open(d / "link.dat", encoding="utf-8") as f:
    for line in f:
        s, o, r, _ = line.rstrip("\n").split("\t")
        if int(r) == 2: wrote.setdefault(aidx[int(o)], []).append(pidx[int(s)])
agg = {"or": lambda X: (X.sum(0) > 0).astype(float), "sum": lambda X: X.sum(0), "mean": lambda X: X.mean(0)}
for name, fn in agg.items():
    exact = sum(np.allclose(A[a], fn(P[ps])) for a, ps in wrote.items())
    print(f"author == {name} of own papers: {exact}/{len(wrote)}")
# support containment: author nonzeros inside the union of own papers' nonzeros
inside = sum(set(np.nonzero(A[a])[0]) <= set(np.nonzero(P[ps].sum(0))[0]) for a, ps in wrote.items())
print("author support within own papers' support:", inside, "/", len(wrote))
rnd = np.random.default_rng(0); perm = rnd.permutation(len(authors)); keys = list(wrote)
ins_r = sum(set(np.nonzero(A[keys[perm[k] % len(keys)]])[0]) <= set(np.nonzero(P[wrote[a]].sum(0))[0]) for k, a in enumerate(keys))
print("control (random author vs other's papers):", ins_r, "/", len(keys))
