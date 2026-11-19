"""The two card 5 T4 `baselines_1_cell_list` captures, 894fd74 vs v3b 1646efd: their full line diff, and a check
that every changed line is in the 4 cells ops card T7.4 moved to `baselines-early` (HetSheaf, CSNN on ACM/DBLP NC)."""
import difflib
import json
from pathlib import Path

ev = Path(__file__).resolve().parents[1] / "card5_evidence"
old = json.loads((ev / "base_capture_894fd74.json").read_text())["baselines_1_cell_list"].splitlines()
new = json.loads((ev / "base_capture_1646efd.json").read_text())["baselines_1_cell_list"].splitlines()
diff = [d for d in difflib.unified_diff(old, new, "894fd74", "1646efd", lineterm="", n=0)]
print("\n".join(diff))
removed = [d[1:] for d in diff if d.startswith("-") and not d.startswith("---")]
added = [d[1:] for d in diff if d.startswith("+") and not d.startswith("+++")]
print(f"lines 894fd74 {len(old)}, 1646efd {len(new)}; removed {len(removed)}, added {len(added)}")
cells = {(t, d, m) for t in ("node_classification",) for d in ("acm", "dblp") for m in ("hetsheaf", "csnn")}
ok_added = all(a.startswith("# in phase baselines-early: ") and tuple(a.split(": ", 1)[1].split()) in cells
               for a in added) and len(added) == 4
ok_removed = len(removed) == 4 and all(any(f"--models {m} " in r and f"--datasets {d} " in r and "--task node_classification" in r
                                           for (_, d, m) in cells) for r in removed)
moved = {(t, d, m) for a in added for (t, d, m) in [tuple(a.split(": ", 1)[1].split())]}
print(f"added = the 4 '# in phase baselines-early' comments of {sorted(moved)}: {ok_added}")
print(f"removed = the 4 tuning lines of the same cells (now in baselines-early): {ok_removed}")
print(f"every other line identical: {[l for l in old if l not in removed] == [l for l in new if l not in added]}")
