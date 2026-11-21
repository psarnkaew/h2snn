"""s/epoch of the H2S-NN GC trials (TSUBAME, read-only). Usage: python3 gc_speed.py <OUT>"""
import datetime as dt, glob, json, os, re, statistics, sys
OUT = sys.argv[1]
TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) \| INFO \| Epoch (\d+) \|")
for ds in ["mutag", "proteins", "enzymes", "nci1"]:
    for d in sorted(glob.glob(os.path.join(OUT, "experiments", "h2snn", f"*graph_classification_{ds}_tune_run0_*"))):
        log = os.path.join(d, "run.log")
        if not os.path.exists(log):
            continue
        text = open(log, errors="replace").read()
        times = [(int(m.group(2)), dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f"))
                 for m in (TS.match(l) for l in text.splitlines()) if m]
        per = [(t2 - t1).total_seconds() / (e2 - e1) for (e1, t1), (e2, t2) in zip(times, times[1:]) if e2 > e1]
        first = (times[0][1] - dt.datetime.strptime(text[:23], "%Y-%m-%d %H:%M:%S,%f")).total_seconds() if times and re.match(r"\d{4}-", text) else float("nan")
        stop = re.search(r"STOP CONDITION (\w+)", text)
        es = "early-stop" if "Early stopping" in text else ""
        print(f"{ds}\t{os.path.basename(d).rsplit('_run0_',1)[1]}\tepochs_logged={len(times)}\tlast={times[-1][0] if times else -1}\t"
              f"s/epoch median={statistics.median(per[1:] or per) if per else float('nan'):.1f}\t"
              f"min={min(per) if per else float('nan'):.1f}\tmax={max(per) if per else float('nan'):.1f}\t"
              f"start->epoch0={first:.0f}s\t{stop.group(1) if stop else es or 'killed/running'}")
