"""Card 2 D2: the per-cell records (d2_dryrun.py) -> cells.csv and cells.md. Fields of the card's Rule only; the
records hold no metric value. Run anywhere (text only):  python d2_table.py <records dir> <out dir> [cell list ...]
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

records_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
expected = []
for listing in sys.argv[3:]:
    expected += [line.strip() for line in Path(listing).read_text().splitlines() if line.strip()]

FIELDS = ["cell", "model", "task", "dataset", "attempt", "status", "note", "error_first_line", "error_at",
          "gpu", "gpu_total_gib", "host", "job", "gpu_peak_alloc_gib", "gpu_peak_reserved_gib", "host_peak_gib",
          "t_build_s", "t_train_epoch_s", "t_val_pass_s", "t_wall_s", "pcg_train_fwd_max", "pcg_train_bwd_max",
          "pcg_val_fwd_max", "pcg_solves_train", "pcg_not_converged", "stop_condition", "oom_tried_to_allocate",
          "step_down", "point", "head"]


def r2(x):
    return "" if x is None or x == "" else f"{float(x):.2f}"


def row_of(r: dict) -> dict:
    t = r.get("times_s", {})
    pcg = r.get("pcg", {})
    is_h2snn = r["model"] == "h2snn"
    status, note = r["status"], ""
    at = r.get("raised_at", "") or ""
    if status == "error" and not is_h2snn:
        # every baseline cell runs through fit_cell (src/baselines); the deepest src/baselines frame names the code
        frames = re.findall(r'File "[^"]*/H2S-NN/(src/baselines/[^"]+)", line (\d+), in (\w+)', r.get("traceback_tail", ""))
        where = f"{frames[-1][0]}:{frames[-1][1]} ({frames[-1][2]})" if frames else (r.get("repo_frame", "") or "")
        note = f"blocked: baseline code (for session 1) at {where}"
    tried = re.search(r"Tried to allocate ([0-9.]+ [KMGT]iB)", r.get("error", "") or "")
    tried = tried.group(1) if tried else ""
    if status == "OOM":
        note = f"needs > {r2(r.get('gpu_peak_alloc_gib'))} GiB allocated + {tried} requested"
    if is_h2snn:
        train_s, val_s = t.get("train_epoch"), t.get("val_pass")
    else:  # baselines: until the first .eval() after the bundle (model build + training epoch), then to the report
        train_s, val_s = r.get("t_until_first_eval_s"), r.get("t_first_eval_to_report_s")
    first = (r.get("error", "") or "").splitlines()[0][:200] if r.get("error") else ""
    stop = r.get("stop_condition")
    return {
        "cell": r["cell"], "model": r["model"], "task": r["task"], "dataset": r["dataset"],
        "attempt": r.get("attempt", ""), "status": status, "note": note, "error_first_line": first,
        "error_at": at, "gpu": r.get("gpu", ""), "gpu_total_gib": r.get("gpu_total_gib", ""),
        "host": r.get("host", ""), "job": f"{r.get('array_job') or r.get('job')}" + (
            f"_{r['array_task']}" if r.get("array_task") else ""),
        "gpu_peak_alloc_gib": r2(r.get("gpu_peak_alloc_gib")), "gpu_peak_reserved_gib": r2(r.get("gpu_peak_reserved_gib")),
        "host_peak_gib": r2(r.get("host_peak_gib")), "t_build_s": r2(t.get("build")), "t_train_epoch_s": r2(train_s),
        "t_val_pass_s": r2(val_s), "t_wall_s": r2(r.get("wall_s")),
        "pcg_train_fwd_max": pcg.get("train_epoch/fwd", {}).get("max_iters", ""),
        "pcg_train_bwd_max": pcg.get("train_epoch/bwd", {}).get("max_iters", ""),
        "pcg_val_fwd_max": pcg.get("val_pass/fwd", {}).get("max_iters", ""),
        "pcg_solves_train": pcg.get("train_epoch/fwd", {}).get("solves", ""),
        "pcg_not_converged": sum(v.get("not_converged", 0) for v in pcg.values()) if pcg else "",
        "stop_condition": json.dumps(stop) if stop else "",
        "oom_tried_to_allocate": tried,
        "step_down": json.dumps(r.get("step_down") or {}, sort_keys=True) if r.get("step_down") else "",
        "point": json.dumps(r.get("point", {}), sort_keys=True), "head": r.get("head", "")[:12],
    }


rows = [row_of(json.loads(p.read_text())) for p in sorted(records_dir.glob("*.json"))]
order = {c: i for i, c in enumerate(expected)}
ATTEMPTS = ["24GB", "24GB-40G", "24GB-stepdown-hidden", "24GB-stepdown", "A6000", "A6000-120G"]
rows.sort(key=lambda x: (order.get(x["cell"], 10**6), ATTEMPTS.index(x["attempt"]) if x["attempt"] in ATTEMPTS else 99))
out_dir.mkdir(parents=True, exist_ok=True)
with (out_dir / "cells.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

# Status at the largest memory point on a 24 GB GPU: the 24GB attempt, or its host-memory retry (24GB-40G) when the
# first was killed by the host-memory request. Step-down rows (one memory axis one grid step lower) are reported apart.
final: dict[str, dict] = {}
for x in rows:
    if x["attempt"] in ("24GB", "24GB-40G"):
        final[x["cell"]] = x
missing = [c for c in expected if c not in final]
counts = Counter((x["status"] if not x["note"].startswith("blocked") else "blocked: baseline code")
                 for x in final.values())
stepdown = [x for x in rows if x["attempt"] == "24GB-stepdown"]
PENDING = [  # queued when this table was built (card 2 change of plan, 2026-11-16): reported as pending
    ("h2snn|node_classification|freebase", "A6000", "16889"), ("h2snn|link_prediction|dblp_ap", "A6000", "16889"),
    ("h2snn|link_prediction|freebase_bb", "A6000", "16889"), ("h2snn|link_prediction|pubmed", "A6000", "16889"),
    ("h2snn|link_prediction|lastfm", "A6000", "16889"), ("h2snn|node_classification|h-pokec", "A6000-120G (g3an01)", "16875"),
    ("han|node_classification|freebase", "24GB-40G (3090)", "16882"),
    ("h2gformer|node_classification|freebase", "24GB-40G (3090)", "16882"),
    ("hetero2net|node_classification|freebase", "24GB-40G (3090)", "16882")]
done = {(x["cell"], x["attempt"].split(" ")[0]) for x in rows}
pending = [p for p in PENDING if (p[0], p[1].split(" ")[0]) not in done]
L = ["# Card 2 D2 — dry run, one row per cell and attempt", "",
     "Status per cell at the largest point on a 24 GB GPU: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
     + (f"; not run {len(missing)}" if missing else "") + f"; cells {len(expected) or len(final)}.", "",
     "Times in seconds; memory in GiB (GPU: `torch.cuda.max_memory_allocated` / reserved; host: peak RSS of the "
     "process). H2S-NN: train = one epoch (draw + steps), val = one full validation pass; PCG = max iterations of any "
     "scope in any solve (fwd / bwd in training, fwd in validation). Baselines: train = from the end of the bundle "
     "build to the first `.eval()` (model construction + the epoch), val = from there to the epoch report (validation; "
     "and test where the loop evaluates it there). No metric value is recorded.", "",
     "| cell | attempt | status | note / error (first file:line) | GPU (node) | GPU peak | host peak | build | train | val | PCG tr f/b, val | stop |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for x in rows:
    err = x["note"] or ""
    if x["status"] not in ("OK",) and x["error_at"]:
        err = (err + "; " if err else "") + f"`{x['error_at']}` {x['error_first_line'][:120]}"
    pcg = (f"{x['pcg_train_fwd_max']}/{x['pcg_train_bwd_max']}, {x['pcg_val_fwd_max']}"
           if x["pcg_train_fwd_max"] != "" else "")
    L.append(f"| {x['cell']} | {x['attempt']} | {x['status']} | {err} | {x['gpu'].replace('NVIDIA ', '')} ({x['host']}) | "
             f"{x['gpu_peak_alloc_gib']} | {x['host_peak_gib']} | {x['t_build_s']} | {x['t_train_epoch_s']} | "
             f"{x['t_val_pass_s']} | {pcg} | {x['stop_condition']} |")
L += ["", "## OOM at 24 GB: one memory axis one grid step lower (author, 2026-11-16)", "",
      "Largest point of the cell's search space with ONE axis one grid step lower (the axis that dominates memory in the "
      "OOM trace); everything else unchanged. A measurement for the author's decision, not a restriction of the space.", "",
      "| cell | axis stepped down | result at 24 GB | GPU peak | GPU (node) |", "|---|---|---|---|---|"]
for x in stepdown:
    axis = ", ".join(f"{k} = {v}" for k, v in json.loads(x["step_down"] or "{}").items())
    if x["status"] == "OK":
        res = f"fits at 24 GB with {axis.replace(' = ', ' <= ') if 'True' not in axis and 'False' not in axis else axis}"
    elif x["status"] == "OOM":
        res = f"still OOM ({x['note']})"
    else:
        res = f"{x['status']}: `{x['error_at']}` {x['error_first_line'][:100]}"
    L.append(f"| {x['cell']} | {axis} | {res} | {x['gpu_peak_alloc_gib']} | {x['gpu'].replace('NVIDIA ', '')} ({x['host']}) |")
if pending:
    L += ["", "## Queued, not run when this table was built (pending)", "", "| cell | attempt | job |", "|---|---|---|"]
    L += [f"| {c} | {a} | {j} |" for c, a, j in pending]
if missing:
    L += ["", "## Not run", ""] + [f"- {c}" for c in missing]
(out_dir / "cells.md").write_text("\n".join(L) + "\n")
print(dict(counts), "missing", len(missing))
