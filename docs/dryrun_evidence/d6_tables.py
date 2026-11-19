"""Card 2 D6: the D6 records -> d6_ranges.md (D6.3), d6_central.md / d6_central.csv (D6.4), d6_runs.md (D6.5, D6.6).
Text only; no metric is in the records.   python d6_tables.py <D6 records dir> <evidence dir>
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

rec_dir, ev = Path(sys.argv[1]), Path(sys.argv[2])
records = [json.loads(p.read_text()) for p in sorted(rec_dir.glob("*.json"))]
spaces = json.loads((ev / "d2_spaces.json").read_text())
H2SNN_GRID = {"model.d": [16, 32, 64], "data.wisp_max_length": [2, 3, 4, 5], "data.wisp_walks_per_anchor": [8, 16, 32, 64]}
H2GB = ("ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns")


def r2(x):
    return "" if x is None or x == "" else f"{float(x):.2f}"


def oom_need(r):
    tried = re.search(r"Tried to allocate ([0-9.]+ [KMGT]iB)", r.get("error", "") or "")
    return f"> {r2(r.get('gpu_peak_alloc_gib'))} GiB + {tried.group(1) if tried else '?'}"


def grid(cell, axis):
    if cell.startswith("h2snn|"):
        return H2SNN_GRID.get(axis)
    return (spaces.get(cell, {}).get(axis) or {}).get("values")


def short(axis):
    return {"data.wisp_walks_per_anchor": "r", "data.wisp_max_length": "l_max", "model.d": "d"}.get(axis, axis)


def status_of(r):
    if r["status"] == "error" and r["model"] != "h2snn" and "src/baselines" in (r.get("traceback_full") or r.get("traceback_tail", "")):
        return "blocked: baseline code"
    return r["status"]


# ---------------------------------------------------------------- D6.3 ranges
d2 = {(r["cell"], r["attempt"]): r for r in csv.DictReader(open(ev / "cells.csv"))}
ladders: dict[str, list[dict]] = {}
for r in records:
    m = re.fullmatch(r"D6\.3-s(\d+)", r.get("attempt", ""))
    if m:
        ladders.setdefault(r["cell"], []).append({**r, "_step": int(m.group(1))})
rows = []
for (cell, attempt), x in d2.items():
    if attempt != "24GB-stepdown":
        continue
    sd = json.loads(x["step_down"] or "{}")
    steps = sorted(ladders.get(cell, []), key=lambda r: r["_step"])
    fit = next((s for s in steps if s["status"] == "OK"), None)
    if x["status"] == "OK":
        point, src, peak, res = sd, "D2 step-down (1 step)", x["gpu_peak_alloc_gib"], "fits"
    elif fit is not None:
        point, src, peak, res = fit["set"], f"D6.3 ladder step {fit['_step']} (+ D2's step)", r2(fit["gpu_peak_alloc_gib"]), "fits"
    elif steps:
        last = steps[-1]
        point, src, peak = last["set"], f"D6.3 ladder, {len(steps)} steps", r2(last.get("gpu_peak_alloc_gib"))
        res = (f"still OOM at the ladder's end ({oom_need(last)})" if last["status"] == "OOM"
               else f"{status_of(last)} at step {last['_step']}: `{last.get('raised_at', '')}` {(last.get('error') or '')[:90]}")
    else:
        point, src, peak, res = sd, "D2 step-down", x["gpu_peak_alloc_gib"], f"{x['status']} (no D6.3 ladder: {x['note'][:80]})"
    axes = "; ".join(f"{short(a)} {grid(cell, a)} → ≤ {v}" if not isinstance(v, bool) else f"{a} {grid(cell, a)} → {v}"
                     for a, v in point.items())
    rows.append((cell, axes, res, peak, src))
L = ["# Card 2 D6.3 — OOM cells: the largest grid point that fits a 24 GB GPU (decision 1)", "",
     "From D2's largest point, the axis of D2's step-down one grid step at a time; at its minimum the next memory axis "
     "(previous axes kept at their minimum); every other axis at its largest value. TITAN RTX / RTX 3090 (24 GB), no "
     "A6000. 'axis [full range] → fitting range'. HAN freebase_bb (segfault at heads 4) is session 1's (not laddered).", "",
     "| cell | axis: full range → fitting range | result | GPU peak GiB | from |", "|---|---|---|---|---|"]
L += [f"| {c} | {a} | {res} | {p} | {s} |" for c, a, res, p, s in rows]
L += ["", "Not laddered (no grid axis changes the failing allocation, or not a memory failure):", "",
      "- **H2S-NN Freebase NC**: the card's first step, mini-batch at B 1024, is refused by the config "
      "(`src/configs/config.py:1088-1094`: training mode fixed per dataset, mini-batch outside H2GB only as the acm/dblp "
      "analysis flag); spec 1.17's \"Freebase mini-batch if full batch does not fit\" is not implemented. **Stop item** "
      "(a new setting). The full-batch ladder above (r ≤ 8) is information only.",
      "- **H²GFormer, Hetero²Net Freebase NC**: D2 retry at 40G host (16882_2, 16882_3, RTX 3090): host fits (peak 35.3 "
      "GiB), CUDA OOM moving the graph to the GPU (one 25.27 GiB tensor, `torch_geometric/data/data.py:364` from "
      "`h2gformer.py:568` / `hetero2net.py:264`, before any model axis applies): no 24 GB grid point fits.",
      "- **HAN Freebase NC**: segmentation fault again at 40G host (16882_1, vendor HGB NC): session 1's (card 1e).",
      "- **HAN freebase_bb LP**: segmentation fault at heads 4 (D2 step-down): session 1's (card 1e)."]
(ev / "d6_ranges.md").write_text("\n".join(L) + "\n")

# ---------------------------------------------------------------- D6.4 central timings
central = [r for r in records if r.get("attempt") == "D6.4-central"]
FIELDS = ["cell", "status", "gpu", "host", "job", "gpu_peak_alloc_gib", "host_peak_gib", "t_build_s", "t_train_epoch_s",
          "t_val_pass_s", "val_source", "val_targets", "ms_per_val_target", "t_test_pass_s", "error_at", "point"]
out = []
for r in sorted(central, key=lambda r: (r["model"] != "h2snn", r["task"], r["dataset"], r["model"])):
    t = r.get("times_s", {})
    if r["model"] == "h2snn":
        train, val, test, src = t.get("train_epoch"), t.get("val_pass"), t.get("test_pass"), "val_pass phase"
    elif "val_pass" in t:  # H2GB mini-batch loop, minibatch.evaluate_split timed
        train, val, test, src = r.get("t_until_first_eval_s"), t.get("val_pass"), t.get("test_pass"), "evaluate_split"
    elif r.get("t_eval_to_metric1_s") is not None:
        train, val, test, src = (r.get("t_until_first_eval_s"), r.get("t_eval_to_metric1_s"),
                                 r.get("t_metric1_to_metric2_s"), "first eval → 1st metric call")
    else:
        train, val, test, src = r.get("t_until_first_eval_s"), r.get("t_first_eval_to_report_s"), None, \
            "first eval → epoch report (val + test where the loop evaluates both)"
    n = r.get("val_targets")
    out.append({"cell": r["cell"], "status": status_of(r), "gpu": (r.get("gpu") or "").replace("NVIDIA ", ""),
                "host": r.get("host", ""), "job": f"{r.get('array_job')}_{r.get('array_task')}",
                "gpu_peak_alloc_gib": r2(r.get("gpu_peak_alloc_gib")), "host_peak_gib": r2(r.get("host_peak_gib")),
                "t_build_s": r2(t.get("build")), "t_train_epoch_s": r2(train), "t_val_pass_s": r2(val),
                "val_source": src if val is not None else "", "val_targets": n if n is not None else "",
                "ms_per_val_target": r2(1000 * val / n) if (val is not None and n) else "", "t_test_pass_s": r2(test),
                "error_at": (r.get("raised_at") or "") if r["status"] != "OK" else "",
                "point": json.dumps(r.get("point", {}), sort_keys=True)})
with (ev / "d6_central.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(out)
L = ["# Card 2 D6.4 — central-point timing (decision B2a): one epoch + one full validation pass", "",
     "Central point: each numeric grid axis at the middle of its grid (lower middle for an even count), log-uniform at "
     "the geometric middle, uniform at the middle, categoricals at the adapter's default (H2S-NN: the YAML's); H2S-NN "
     "on H2GB NC at B 256. Times in s; `val targets` = targets of one full validation pass; ms/target = validation "
     "time per target (to scale to the B1a subset). Baselines' train = model construction + the epoch. No metric.", "",
     "| cell | status | GPU (node) | GPU peak | host peak | train epoch | val pass | val targets | ms/target | val timed by |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for x in out:
    L.append(f"| {x['cell']} | {x['status']}{(' `' + x['error_at'] + '`') if x['error_at'] else ''} | {x['gpu']} ({x['host']}) | "
             f"{x['gpu_peak_alloc_gib']} | {x['host_peak_gib']} | {x['t_train_epoch_s']} | {x['t_val_pass_s']} | "
             f"{x['val_targets']} | {x['ms_per_val_target']} | {x['val_source']} |")
L += ["", "Not run (design session 2026-11-17: H-Pokec runs only what B2a needs, H2S-NN's central point): array tasks "
      "16924_3 to 16924_11 cancelled while pending (60G each, one at a time, blocking g2tn01's second GPU): the central "
      "lines of GCN, GAT, ChebNetII, NSD, CSNN, HAN, HINormer, H²GFormer, Hetero²Net on H-Pokec NC (each would stop in "
      "`bundles._base_config` (H3b) like the other H2GB NC baseline cells above)."]
(ev / "d6_central.md").write_text("\n".join(L) + "\n")

# ---------------------------------------------------------------- D6.5, D6.6
#: runs killed before writing a record (from sacct / the job log)
NO_RECORD = ["- **h2snn|node_classification|h-pokec** attempt `D6.6-60G` (largest point: d 64, l_max 5, r 64, B 1024): "
             "TIMEOUT at the 8 h limit on NVIDIA GeForce RTX 3090 (g3n01, job 16924_1 = 16966, 60G host): sacct MaxRSS "
             "39.79 GiB (41722788K), no host OOM, no CUDA OOM; the training epoch did not finish in 8 h (no record "
             "written). The node's RAM (64 GB, `sinfo`) allowed 60G, so g3an01 was not used."]
L = ["# Card 2 D6.5 / D6.6 runs", ""] + NO_RECORD
for r in records:
    if r.get("attempt", "").startswith(("D6.5", "D6.6")):
        t = r.get("times_s", {})
        L.append(f"- **{r['cell']}** attempt `{r['attempt']}`: {r['status']} on {r.get('gpu', '')} ({r.get('host', '')}, job "
                 f"{r.get('array_job')}_{r.get('array_task')}); GPU peak {r2(r.get('gpu_peak_alloc_gib'))} GiB, host peak "
                 f"{r2(r.get('host_peak_gib'))} GiB; times {json.dumps({k: round(v, 1) for k, v in t.items()})}"
                 + (f"; error `{r.get('raised_at', '')}` {(r.get('error') or '')[:160].splitlines()[0]}" if r["status"] != "OK" else ""))
(ev / "d6_runs.md").write_text("\n".join(L) + "\n")
print(len(rows), "ranges;", len(out), "central;", "statuses", sorted({x["status"] for x in out}))
