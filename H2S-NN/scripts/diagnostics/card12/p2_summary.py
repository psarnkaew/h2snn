"""Card 12 P2 summary: PDNS `balmy-sweep-7` with the zero right-hand-side rule vs card 11 B2a.

Inputs: the new run's card 11 probe (`--probe`, one row per PCG call), its zero-RHS probe (`--zero_rhs`, one row
per call: triggered scopes and their ||b_s||), its `metrics.jsonl`, and card 11 B2a's probe and metrics
(`docs/card11_evidence/B/b2_pdns_tmax100_{probe.tsv.gz,metrics.jsonl}` at 37fc77a). Writes to `--out`:

- `p2_per_epoch.tsv`: per epoch / mode / direction: solves, iterations max / mean / p95, max relative residual,
  unconverged scopes, triggered scopes and the ||b_s|| of each;
- `p2_triggers.tsv`: every call with a triggered scope (epoch, step, direction, scopes, ||b_s||, max ||b_s||);
- `p2_vs_card11.tsv`: per solve, matched in call order within (epoch, mode, direction), tau, iterations max /
  mean and max relative residual of both runs, up to the end of card 11's log; the first solve that differs;
- `p2_metrics_vs_card11.tsv`: train loss and validation Macro-F1 per finished epoch.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as h:
        return list(csv.DictReader(h, delimiter="\t"))


def metrics(path: Path) -> dict[int, dict]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[int(r["epoch"])] = r
    return out


def write(path: Path, data: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in data for k in r))
    with path.open("w", encoding="utf-8") as h:
        h.write("\t".join(cols) + "\n")
        for r in data:
            h.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"== {path.name}\n" + path.read_text(), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--zero_rhs", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--ref_probe", type=Path, required=True)
    ap.add_argument("--ref_metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    new, ref, zr = rows(args.probe), rows(args.ref_probe), rows(args.zero_rhs)
    if len(new) != len(zr):
        print(f"WARNING: probe rows {len(new)} vs zero-RHS rows {len(zr)}", flush=True)

    # zero-RHS rows are written before the inner call, card 11 rows after it: same call order
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(new):
        groups[(r["epoch"], r["mode"], r["direction"])].append(i)
    per_epoch = []
    for key in sorted(groups, key=lambda k: (int(k[0]), k[1], k[2])):
        idx = groups[key]
        it = np.array([float(new[i]["iters_max"]) for i in idx])
        trig = [(zr[i]["step"], zr[i]["triggered_b_norms"]) for i in idx
                if i < len(zr) and int(zr[i]["triggered"]) > 0]
        per_epoch.append({
            "epoch": key[0], "mode": key[1], "direction": key[2], "solves": len(idx),
            "iters_max": int(it.max()), "iters_mean": f"{it.mean():.2f}", "iters_p95": f"{np.percentile(it, 95):.1f}",
            "rel_res_max": f"{max(float(new[i]['rel_res_max']) for i in idx):.3e}",
            "unconverged_scopes": sum(int(new[i]["unconverged_scopes"]) for i in idx),
            "steps_last": new[idx[-1]]["step"],
            "triggered_scopes": sum(int(zr[i]["triggered"]) for i in idx if i < len(zr)),
            "triggered (step: ||b_s||)": "; ".join(f"{s}: {b}" for s, b in trig),
        })
    write(args.out / "p2_per_epoch.tsv", per_epoch)
    write(args.out / "p2_triggers.tsv", [
        {"call": i, "epoch": r["epoch"], "mode": r["mode"], "step": r["step"], "direction": r["direction"],
         "num_scopes": r["num_scopes"], "scopes": r["triggered_scopes"], "b_norms": r["triggered_b_norms"],
         "b_norm_max": r["b_norm_max"], "b_norm_median": r["b_norm_median"]}
        for i, r in enumerate(zr) if int(r["triggered"]) > 0])

    ref_groups: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(ref):
        ref_groups[(r["epoch"], r["mode"], r["direction"])].append(i)
    cmp, first_diff, same = [], None, 0
    for key, ridx in ref_groups.items():
        nidx = groups.get(key, [])
        for k, ri in enumerate(ridx):
            if k >= len(nidx):
                break
            a, b = ref[ri], new[nidx[k]]
            fields = ("tau", "iters_max", "iters_mean", "rel_res_max", "unconverged_scopes")
            equal = all(a[f] == b[f] for f in fields)
            entry = {"epoch": a["epoch"], "mode": a["mode"], "direction": a["direction"], "step": a["step"],
                     **{f"card11_{f}": a[f] for f in fields}, **{f"card12_{f}": b[f] for f in fields},
                     "equal": equal, "card12_triggered": zr[nidx[k]]["triggered"] if nidx[k] < len(zr) else ""}
            if equal:
                same += 1
            elif first_diff is None:
                first_diff = entry
            if not equal or (a["mode"] == "train" and a["epoch"] == "1" and int(a["step"]) >= 165):
                cmp.append(entry)
    total = sum(min(len(v), len(groups.get(k, []))) for k, v in ref_groups.items())
    print(f"solves compared {total}, equal {same}; first difference: {first_diff}", flush=True)
    write(args.out / "p2_vs_card11.tsv", cmp)

    nm, rm = metrics(args.metrics), metrics(args.ref_metrics)
    write(args.out / "p2_metrics_vs_card11.tsv", [
        {"epoch": e,
         "card11_train_loss": rm.get(e, {}).get("train", {}).get("loss", ""),
         "card12_train_loss": nm[e]["train"]["loss"],
         "card11_val_macro_f1": rm.get(e, {}).get("val", {}).get("macro_f1", ""),
         "card12_val_macro_f1": nm[e]["val"].get("macro_f1", ""),
         "train_loss_equal": rm.get(e, {}).get("train", {}).get("loss") == nm[e]["train"]["loss"],
         "val_macro_f1_equal": rm.get(e, {}).get("val", {}).get("macro_f1") == nm[e]["val"].get("macro_f1")}
        for e in sorted(nm)])


if __name__ == "__main__":
    main()
