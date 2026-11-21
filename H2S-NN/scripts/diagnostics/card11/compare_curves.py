"""Card 11 A0/A3: per-epoch train / validation curves of a lab run against its TSUBAME reference.

Reads both `metrics.jsonl` files (the trainer's own log) and writes one TSV row per epoch:
train loss, train AUROC, val loss, val AUROC (LP) or val Macro-F1 (NC/GC), tau and PCG iterations,
for TSUBAME and lab side by side. Usage: `compare_curves.py --ref DIR --lab DIR --out TSV`
(`--ref` may be omitted for a run with no TSUBAME reference).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = (("train", "loss"), ("train", "auroc"), ("val", "loss"), ("val", "auroc"), ("val", "macro_f1"),
        ("train", "tau"), ("train", "pcg_fwd_iters"), ("train", "pcg_bwd_iters"))


def load(d: Path | None) -> dict[int, dict]:
    if d is None or not (d / "metrics.jsonl").exists():
        return {}
    out = {}
    for line in (d / "metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        out[int(r["epoch"])] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--lab", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ref, lab = load(a.ref), load(a.lab)
    keys = [k for k in KEYS if any(k[1] in r.get(k[0], {}) for r in list(lab.values()) + list(ref.values()))]
    sources = [("tsubame", ref), ("lab", lab)] if ref else [("lab", lab)]
    cols = ["epoch"] + [f"{name}_{sp}_{m}" for (sp, m) in keys for name, _ in sources]
    with a.out.open("w") as h:
        h.write("\t".join(cols) + "\n")
        for e in sorted(set(ref) | set(lab)):
            row = [str(e)]
            for sp, m in keys:
                for _, src in sources:
                    v = src.get(e, {}).get(sp, {}).get(m)
                    row.append("" if v is None else f"{float(v):.4f}")
            h.write("\t".join(row) + "\n")
    print(a.out.read_text())


if __name__ == "__main__":
    main()
