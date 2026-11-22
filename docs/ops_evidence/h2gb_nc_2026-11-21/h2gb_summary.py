"""Read-only summary of the H2S-NN H2GB NC tuning trials on TSUBAME (ops, 2026-11-21).

Usage: python3 h2gb_summary.py <OUT> <dest_dir>
Writes <dest_dir>/summary.tsv (one row per trial) and <dest_dir>/<dataset>_furthest.tsv (per-epoch rows of the trial
that reached the most epochs). No path of the host is written.
"""
import datetime as dt
import glob
import json
import os
import re
import statistics
import sys

OUT, DEST = sys.argv[1], sys.argv[2]
os.makedirs(DEST, exist_ok=True)
EXP = os.path.join(OUT, "experiments", "h2snn")
DATASETS = ["ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns"]
TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) \| INFO \| Epoch (\d+) \|")

# GPU per dataset from provenance (line text names the dataset config)
gpu = {}
for p in glob.glob(os.path.join(OUT, "logs", "prov", "*.jsonl")):
    for raw in open(p):
        try:
            r = json.loads(raw)
        except ValueError:
            continue
        if r.get("event") != "start":
            continue
        m = re.search(r"configs/experiments/(\S+?)_h2snn_v3b\.yaml", r.get("line", ""))
        if m:
            gpu.setdefault(m.group(1), set()).add(r.get("gpu_name", "?"))


def cfg_value(text, key):
    m = re.search(rf"^\s*{key}:\s*(\S+)", text, re.M)
    return m.group(1) if m else ""


rows = []
for ds in DATASETS:
    best_dir, best_n = None, -1
    for d in sorted(glob.glob(os.path.join(EXP, f"h2snn_node_classification_{ds}_tune_run0_*"))):
        name = d.rsplit("_tune_run0_", 1)[1]
        cfg = open(os.path.join(d, "config.yaml")).read() if os.path.exists(os.path.join(d, "config.yaml")) else ""
        recs = []
        mp = os.path.join(d, "metrics.jsonl")
        if os.path.exists(mp):
            for raw in open(mp):
                try:
                    recs.append(json.loads(raw))
                except ValueError:
                    pass
        log = open(os.path.join(d, "run.log"), errors="replace").read() if os.path.exists(os.path.join(d, "run.log")) else ""
        times = [(int(m.group(2)), dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f"))
                 for m in (TS.match(line) for line in log.splitlines()) if m]
        per_epoch = [(t2 - t1).total_seconds() / (e2 - e1) for (e1, t1), (e2, t2) in zip(times, times[1:]) if e2 > e1]
        spe = statistics.median(per_epoch[1:] or per_epoch) if per_epoch else float("nan")
        stop = "running/killed"
        if "STOP CONDITION" in log:
            stop = re.search(r"STOP CONDITION (\w+)", log).group(1)
        elif "Traceback" in log:
            stop = "error"
        elif recs and recs[-1]["epoch"] >= 499:
            stop = "finished"
        best = max(recs, key=lambda r: r["val"].get("macro_f1", -1)) if recs else None
        last = recs[-1]["epoch"] if recs else -1
        rows.append([ds, name, ",".join(sorted(gpu.get(ds, {"?"}))), cfg_value(cfg, "batch_size"),
                     cfg_value(cfg, "eval_period"), f"{spe:.1f}", str(last), str(len(recs)),
                     str(best["epoch"]) if best else "", f"{best['val'].get('macro_f1', float('nan')):.4f}" if best else "",
                     f"{best['val'].get('acc', float('nan')):.4f}" if best else "",
                     f"{recs[-1]['val'].get('macro_f1', float('nan')):.4f}" if recs else "", stop])
        if last > best_n:
            best_n, best_dir = last, (d, name, recs)
    if best_dir:
        d, name, recs = best_dir
        with open(os.path.join(DEST, f"{ds}_furthest.tsv"), "w") as f:
            f.write(f"# {ds} trial {name}: per evaluated epoch; train metrics other than loss are not computed\n")
            f.write("epoch\ttrain_loss\tval_loss\tval_acc\tval_macro_f1\tval_micro_f1\n")
            for r in recs:
                v = r["val"]
                f.write(f"{r['epoch']}\t{r['train'].get('loss', float('nan')):.4f}\t{v.get('loss', float('nan')):.4f}\t"
                        f"{v.get('acc', float('nan')):.4f}\t{v.get('macro_f1', float('nan')):.4f}\t"
                        f"{v.get('micro_f1', float('nan')):.4f}\n")

with open(os.path.join(DEST, "summary.tsv"), "w") as f:
    f.write("dataset\ttrial\tgpu\tbatch\teval_period\ts_per_epoch\tlast_epoch\tn_evals\tbest_val_epoch\t"
            "best_val_macro_f1\tval_acc_at_best\tlast_val_macro_f1\tend\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
print(open(os.path.join(DEST, "summary.tsv")).read())
