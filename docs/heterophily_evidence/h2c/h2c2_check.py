"""Card 6 H2c-2 real-data check: the timed-out whole-graph run (17388_7, job 17403) captured calcHomophily's own
prints (the harness redirects stdout), so the only per-meta-path values in its log are tqdm's final descriptions,
`Running edge homophily:<round(h_edge, 3)>` at 100 % for each meta-path, in calcHomophily's order, of fill seed 0
(the 17th bar, seed 1's first path, stopped at 63 %). Each H2c-2 part's seed-0 h_edge, rounded to 3 decimals, must
equal the printed value of its index. h_insensitive and h_adj were never printed. Exit 1 on a mismatch.
    python h2c2_check.py <timed-out log> <parts dir> <out json>"""
import json
import re
import sys
from pathlib import Path

log, parts_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
text = log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
done = re.findall(r"Running edge homophily:([0-9.]+): 100%\|[^|]*\| (\d+)/(\d+) \[([0-9:]+)<", text)
bars, last = [], None  # a finished bar is redrawn several times: keep one per (value, elapsed)
for value, a, b, elapsed in done:
    key = (value, elapsed)
    if key != last:
        bars.append({"printed_h_edge": value, "elapsed": elapsed})
    last = key
rows, mismatch = [], False
for f in sorted(parts_dir.glob("nc__h-pokec__h2__mp*.json")):
    q = json.loads(f.read_text())
    i = q["part"]["metapath_index"]
    run = q["per_problem"]["label"]["runs"][0]
    assert run["seed"] == 0
    r = run["per_metapath"][0] if run["per_metapath"] else None
    row = {"metapath_index": i, "metapath": q["part"]["metapath"], "job": q["job"], "wall_s": q["wall_s"],
           "peak_rss_gb": q["peak_rss_gb"], "h_edge": r and r["h_edge"], "h_insensitive": r and r["h_insensitive"],
           "h_adj": r and r["h_adj"]}
    if i < len(bars):
        row.update(bars[i], equal_to_printed=(r is not None and f"{round(r['h_edge'], 3)}" == bars[i]["printed_h_edge"]))
        mismatch |= not row["equal_to_printed"]
    rows.append(row)
    print(json.dumps(row))
result = {"bars_at_100": len(bars), "parts": len(rows), "compared": sum("equal_to_printed" in r for r in rows),
          "mismatch": mismatch, "rows": rows}
out.write_text(json.dumps(result, indent=1))
print("bars", len(bars), "compared", result["compared"], "mismatch", mismatch)
sys.exit(1 if mismatch else 0)
