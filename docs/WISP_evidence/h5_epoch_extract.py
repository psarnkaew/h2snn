"""Extract from the H5 per-epoch logs (experiments/wisp_h5/*/metrics.jsonl, not committed):
the maximum logged log-weight over all epochs of every run, and the S1/S1' training curves.
Run from H2S-NN/: python ../docs/WISP_evidence/h5_epoch_extract.py > ../docs/WISP_evidence/h5_epoch_extract.txt"""
import json
from pathlib import Path

runs = sorted(Path("experiments/wisp_h5").iterdir())
overall = float("-inf")
print("# max over all epochs of train path_log_w_max, per run")
for run in runs:
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    values = [r["train"].get("path_log_w_max") for r in rows if r.get("train")]
    values = [v for v in values if v is not None]
    peak = max(values) if values else None
    overall = max(overall, peak) if peak is not None else overall
    print(f"{run.name:40s} epochs {len(rows):3d}  max log w {peak}")
print(f"# overall max: {overall}")
print("\n# train / val accuracy and loss at epochs 0, 10, 20, 40, 99 (functional checks only)")
for run in runs:
    if not run.name.startswith(("h5_s1p_full", "h5_s1p_oracle", "h5_s1_oracle_weight_s42", "h5_s1_full_s42")):
        continue
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    cells = [f"ep{e}: train {rows[e]['train']['acc']:.3f}/{rows[e]['train']['loss']:.3f} "
             f"val {rows[e]['val']['acc']:.3f}/{rows[e]['val']['loss']:.3f}" for e in (0, 10, 20, 40, 99)]
    print(run.name, " | ".join(cells))
