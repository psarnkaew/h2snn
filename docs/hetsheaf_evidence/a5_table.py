"""Card 3 A5: a5_central.csv and the ladder attempt table from the per-cell JSON lines (no metric is read or recorded)."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent


def rows(folder, prefix):
    out = []
    for path in sorted((HERE / folder).glob(f"{prefix}_*.jsonl"), key=lambda p: int(p.stem.split("_")[1])):
        out += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return out


def gpu(r):
    return r.get("gpu", "").replace("NVIDIA ", "").replace("GeForce ", "")


COLUMNS = ["cell", "status", "sheaf", "d", "layers", "hidden_channels", "batch_size", "pooling", "lr", "weight_decay",
           "dropout", "input_dropout", "train_epoch_s", "val_s", "test_s", "val_targets", "val_ms_per_target",
           "load_s", "peak_gpu_gib", "peak_host_gib", "gpu", "host", "job"]
with open(HERE / "a5_central.csv", "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=COLUMNS)
    writer.writeheader()
    for r in rows("a5_central", "central"):
        p = r["point"]
        writer.writerow({"cell": r["cell"], "status": r["status"], **{k: p.get(k, "") for k in COLUMNS[2:12]},
                         "train_epoch_s": r.get("train_epoch_s"), "val_s": r.get("val_s"), "test_s": r.get("test_s"),
                         "val_targets": r.get("val_targets"),
                         "val_ms_per_target": round(1000 * r["val_s"] / r["val_targets"], 3)
                         if r.get("val_s") and r.get("val_targets") else "",
                         "load_s": r.get("load_s"), "peak_gpu_gib": r.get("peak_gpu_gib"),
                         "peak_host_gib": r.get("peak_host_gib"), "gpu": gpu(r), "host": r.get("host"),
                         "job": r.get("job")})

print("| Cell | hidden_channels | d | layers | Status | Peak GPU GiB | Site (OOM: tried to allocate) | GPU, node, job |")
print("|---|---|---|---|---|---|---|---|")
for r in rows("a5_ladder", "ladder"):
    p, msg = r["point"], r.get("error", "")
    ask = msg.split("Tried to allocate ")[1].split(". GPU")[0] if "Tried to allocate" in msg else ""
    site = r.get("where", "").split("hetsheaf/")[-1]
    detail = f"`{site}` ({ask})" if r["status"] == "OOM" else (f"`{site}` {msg[:60]}" if site else "")
    print(f"| {r['cell'].split(':')[1]} | {p['hidden_channels']} | {p['d']} | {p['layers']} | {r['status']} | "
          f"{r.get('peak_gpu_gib', '')} | {detail} | {gpu(r)}, {r.get('host')}, {r.get('job')} |")
