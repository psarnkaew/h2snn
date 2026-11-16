"""Card 3 A4: the report's dry-run table from the per-cell JSON lines (no metric is read; none is recorded)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def rows(folder):
    out = []
    for path in sorted((HERE / folder).glob("cell_*.jsonl"), key=lambda p: int(p.stem.split("_")[1])):
        out += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return out


def fmt(r, where):
    status = r["status"]
    point = r.get("step_down", "largest")
    detail = ""
    if status == "OOM":
        msg = r.get("error", "")
        ask = msg.split("Tried to allocate ")[1].split(". GPU")[0] if "Tried to allocate" in msg else "?"
        detail = f"tried to allocate {ask} at `{r.get('where', '').split('hetsheaf/')[-1]}`"
    t = r.get("epoch_and_eval_s")
    return (f"| {r['cell'].replace('_classification', '').replace('_prediction', '')} | {where} | {point} | {status} | "
            f"{r.get('peak_gpu_gib', '')} | {r.get('peak_host_gib', '')} | {'' if t is None else t} | "
            f"{r.get('gpu', '').replace('NVIDIA ', '').replace('GeForce ', '')}, {r.get('host', '')}, job {r.get('job', '')} | {detail} |")


lines = ["| Cell | Run | Point | Status | Peak GPU GiB | Peak host GiB | Epoch + eval s | GPU, node, job | OOM at |",
         "|---|---|---|---|---|---|---|---|---|"]
lines += [fmt(r, "24 GB") for r in rows("a4_cells")]
lines += [fmt(r, "A6000") for r in rows("a4_cells_a6000")]
lines += [fmt(r, "11 GB (2080 Ti, A4 GC)") for r in
          (json.loads(l) for l in (HERE / "a4_cells_small.jsonl").read_text().splitlines()) if r["status"] == "OK"]
print("\n".join(lines))
