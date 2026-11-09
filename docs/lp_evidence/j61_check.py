"""J6.1 gate: in each derived LP task, the removed blocks/columns are absent and everything else
is bitwise equal to the HGB release (rows of untouched types byte-identical; kept movie values
text-equal to the release's same column); feature_rule.json matches node.dat; the
node-classification releases are unchanged (md5 against j53_nc_release_md5_before.txt).
Own parser; read only. Run from H2S-NN/: python ../docs/lp_evidence/j61_check.py"""
import hashlib
import json
from pathlib import Path

EXPECTED = {"acm_pa": ({1}, {}), "dblp_ap": ({0}, {}), "imdb_ma": ({2}, {0: {3, 4, 7, 12}}),
            "freebase_bb": (set(), {})}
ok = True
for task, (types, cols) in EXPECTED.items():
    d = Path("data/lp_derived") / task
    rec = json.loads((d / "feature_rule.json").read_text(encoding="utf-8"))
    release = Path(rec["release"])
    rel = (release / "node.dat").read_bytes().splitlines(keepends=True)
    new = (d / "node.dat").read_bytes().splitlines(keepends=True)
    same_rows = len(rel) == len(new)
    removed_absent = untouched_identical = kept_values_equal = True
    n_removed = n_cols = 0
    for a, b in zip(rel, new):
        pa, pb = a.rstrip(b"\r\n").split(b"\t"), b.rstrip(b"\r\n").split(b"\t")
        t = int(pa[2])
        if pa[:3] != pb[:3] or a[len(a.rstrip(b"\r\n")):] != b[len(b.rstrip(b"\r\n")):]:
            untouched_identical = False
        if t in types:
            n_removed += 1
            removed_absent &= len(pb) == 4 and pb[3] == b""
        elif t in cols:
            va, vb = pa[3].split(b","), pb[3].split(b",")
            kept = [v for j, v in enumerate(va) if j not in cols[t]]
            kept_values_equal &= vb == kept
            removed_absent &= len(vb) == len(va) - len(cols[t])
            n_cols += 1
        else:
            untouched_identical &= a == b
    md5_ok = hashlib.md5((d / "node.dat").read_bytes()).hexdigest() == rec["node_dat_md5"]
    meta = json.loads((d / "split_meta.json").read_text(encoding="utf-8"))
    meta_ok = meta["output_md5"]["node.dat"] == rec["node_dat_md5"]
    stale = (d / "feature_rebuild.json").exists()
    passed = (same_rows and removed_absent and untouched_identical and kept_values_equal
              and md5_ok and meta_ok and not stale)
    ok &= passed
    print(f"{task}: rows {len(new)}; removed-type rows featureless {n_removed} "
          f"({'yes' if removed_absent else 'NO'}); column-reduced rows {n_cols} "
          f"(kept values text-equal {kept_values_equal}); other rows byte-identical "
          f"{untouched_identical}; record md5 {md5_ok}; split_meta md5 {meta_ok}; "
          f"stale feature_rebuild.json {stale}: {'PASS' if passed else 'FAIL'}")
before = {}
for line in Path("../docs/lp_evidence/j53_nc_release_md5_before.txt").read_text().splitlines():
    digest, path = line.split(maxsplit=1)
    before[path.lstrip("*")] = digest
changed = [p for p, dg in before.items() if hashlib.md5(Path(p).read_bytes()).hexdigest() != dg]
ok &= not changed
print(f"node-classification releases: {len(before)} files, changed {len(changed)}")
print("ALL PASS" if ok else "SOME FAIL")
