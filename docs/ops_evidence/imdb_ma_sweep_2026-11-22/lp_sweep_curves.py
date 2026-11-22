"""Per-epoch train/val AUROC of the best and median trials of one H2S-NN LP sweep (ops evidence). Read-only.

Usage: python lp_sweep_curves.py <experiments/h2snn dir> <task token, e.g. imdb_ma> <out dir>
Trial = one run dir h2snn_link_prediction_<task>_tune_*; its selection score = val_auroc of best_test_metrics.json
(the sweep objective). Writes trials.tsv (one row per trial) and curves.tsv (per epoch, best and median trial).
"""
import json
import statistics
import sys
from pathlib import Path

root, task, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)


def lr_of(d):
    for line in (d / "config.yaml").read_text().splitlines():
        s = line.strip()
        if s.startswith("lr:") or s.startswith("learning_rate:"):
            return s.split(":", 1)[1].strip()
    return ""


trials = []
for d in sorted(root.glob(f"h2snn_link_prediction_{task}_tune_*")):
    try:
        best = json.loads((d / "best_test_metrics.json").read_text())
        rows = [json.loads(x) for x in (d / "metrics.jsonl").read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        continue
    if best.get("val_auroc") is None:
        continue
    trials.append({"trial": d.name.split("_tune_")[1], "val_auroc": best["val_auroc"], "best_epoch": best.get("best_epoch"),
                   "test_random_auroc": best.get("test_random_auroc"), "epochs_run": len(rows), "lr": lr_of(d), "rows": rows})

trials.sort(key=lambda t: t["val_auroc"], reverse=True)
with open(out / "trials.tsv", "w") as f:
    f.write("rank\ttrial\tval_auroc_best\tbest_epoch\ttest_random_auroc\tepochs_run\tlr\n")
    for i, t in enumerate(trials, 1):
        f.write(f"{i}\t{t['trial']}\t{t['val_auroc']:.4f}\t{t['best_epoch']}\t{t['test_random_auroc']:.4f}\t{t['epochs_run']}\t{t['lr']}\n")

med = trials[(len(trials) - 1) // 2]
pick = {"best": trials[0], "median": med}
with open(out / "curves.tsv", "w") as f:
    f.write("which\ttrial\tepoch\ttrain_loss\ttrain_auroc\tval_loss\tval_auroc\n")
    for which, t in pick.items():
        for r in t["rows"]:
            tr, va = r.get("train", {}), r.get("val", {})
            g = lambda d, k: "" if d.get(k) is None else f"{d[k]:.4f}"  # noqa: E731
            f.write(f"{which}\t{t['trial']}\t{r.get('epoch')}\t{g(tr, 'loss')}\t{g(tr, 'auroc')}\t{g(va, 'loss')}\t{g(va, 'auroc')}\n")

be = [t["best_epoch"] for t in trials if isinstance(t["best_epoch"], int)]
print(f"trials {len(trials)}; best val {trials[0]['val_auroc']:.4f} ({trials[0]['trial']}, epoch {trials[0]['best_epoch']}, "
      f"lr {trials[0]['lr']}); median val {med['val_auroc']:.4f} ({med['trial']}, epoch {med['best_epoch']}, lr {med['lr']}); "
      f"best_epoch 0 in {sum(1 for b in be if b == 0)} of {len(be)}; best_epoch median {statistics.median(be)}")
for which, t in pick.items():
    rows = t["rows"]
    tr = [r.get("train", {}).get("auroc") for r in rows]
    va = [r.get("val", {}).get("auroc") for r in rows]
    print(which, "train auroc first/last", tr[0], tr[-1], "| val auroc first/max/last", va[0], max(v for v in va if v is not None), va[-1])
