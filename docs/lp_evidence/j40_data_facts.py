"""Data facts for the J4 pre-registration (read only)."""
from collections import Counter
from pathlib import Path
import sys
root = Path(sys.argv[1])
def rows(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t"); out.append((int(p[0]), int(p[1]), int(p[2])))
    return out
def types(path):
    t = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t"); t[int(p[0])] = int(p[2])
    return t
for name in ["PubMed", "LastFM"]:
    d = root / name
    t = types(d / "node.dat"); print(name, "node types", Counter(t.values()))
    link = rows(d / "link.dat"); test = rows(d / "link.dat.test")
    by = {}
    for s, o, r in link: by.setdefault(r, set()).add((s, o))
    for r in sorted(by):
        e = by[r]; ex = next(iter(e)); rev = {(o, s) for s, o in e}
        loops = sum(s == o for s, o in e)
        print(f"  rel {r} types {t[ex[0]]}->{t[ex[1]]} edges {len(e)} loops {loops} "
              f"reverse-stored-in-same-rel {len(e & rev) - loops} self_inverse {e == rev}")
    tb = Counter(r for _, _, r in test); print("  test relations", dict(tb))
    for r in tb:
        te = {(s, o) for s, o, rr in test if rr == r}
        tr = by.get(r, set())
        print(f"  rel {r} test {len(te)}; test whose reverse is in link.dat rel {r}: "
              f"{sum((o, s) in tr for s, o in te)}; test loops {sum(s == o for s, o in te)}; "
              f"test pair also in link.dat {len(te & tr)}; test with both orientations in test "
              f"{sum((o, s) in te for s, o in te if s != o)}")
fb = root / "freebase/freebase/raw/Freebase"
link = rows(fb / "link.dat"); e = {(s, o) for s, o, r in link if r == 0}
loops = {(s, o) for s, o in e if s == o}; nl = e - loops
recip = {(s, o) for s, o in nl if (o, s) in nl}
pairs = {(min(s, o), max(s, o)) for s, o in nl}
print("Freebase rel0 edges", len(e), "loops", len(loops), "non-loop", len(nl), "with reverse stored",
      len(recip), f"({len(recip)/len(nl):.3f})", "unordered pairs", len(pairs))
