"""J6.2: merge the per-task audit outputs (j62_audit_<task>.json) into j62_audit.json and print
the Ctrl-G relations and every family's worst case (cap, split) against Ctrl-G. Read only.
Run from docs/lp_evidence/: python j62_summary.py"""
import json
from pathlib import Path

TASKS = ["acm_pa", "dblp_ap", "imdb_ma", "freebase_bb", "pubmed", "lastfm"]
merged = []
for t in TASKS:
    path = Path(f"j62_audit_{t}.json")
    if not path.exists():
        print(f"{t}: MISSING")
        continue
    merged.extend(json.loads(path.read_text(encoding="utf-8")))
Path("j62_audit.json").write_text(json.dumps(merged, indent=1), encoding="utf-8")
for r in merged:
    print(f"== {r['task']}: families {r['family_count']}" + (f" ({r['note']})" if "note" in r else "")
          + (f"; task {'PASS' if r['pass'] else 'FAIL'}" if r["families"] else ""))
    for x, c in r["ctrl_g"].items():
        for rel, m in c["per_relation"].items():
            print(f"  Ctrl-G candidate type {x} {rel}: valid MRR {m['valid']['mrr']:.4f} "
                  f"H@10 {m['valid']['hits@10']:.4f}; test MRR {m['test']['mrr']:.4f} "
                  f"H@10 {m['test']['hits@10']:.4f}")
    for f in r["families"]:
        cells = [(cap, s, v) for cap, cr in f["caps"].items() for s, v in cr.items()]
        worst_m = max(cells, key=lambda z: z[2]["diff_mrr"])
        worst_h = max(cells, key=lambda z: z[2]["diff_hits@10"])
        print(f"  {'PASS' if f['pass'] else 'FAIL'} {f['family']}: max diff MRR "
              f"{worst_m[2]['diff_mrr']:+.4f} (cap {worst_m[0]}, {worst_m[1]}); max diff Hits@10 "
              f"{worst_h[2]['diff_hits@10']:+.4f} (cap {worst_h[0]}, {worst_h[1]})")
