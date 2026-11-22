"""Card 11 B2/B3/C: summarise a solver-probe TSV (`solver_probe.py`) of one run.

Writes, next to `--out_prefix`:
- `<prefix>_per_epoch.tsv`: per epoch and mode (train / eval) and direction: solves, iterations
  (max, mean, p95), tau (min, max), relative residual max, largest scope rows (max), worst-scope
  max degree (max), Jacobi max, path_log_w range (max), mean seconds per call;
- `<prefix>_top_solves.tsv`: the solves with the most iterations (top `--top`), and the last
  `--tail` solves of the run (the failing one is the last if the run hit T_max);
- `<prefix>_drivers.tsv`: over training solves, Spearman of the iteration count with tau, the
  largest scope (rows), the worst scope's rows, its largest degree, its Jacobi max, the batch's
  largest degree and the path_log_w range (the B read-out: conditioning, scope / hub, or a jump).
Prints all three.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, action="append", required=True)
    ap.add_argument("--out_prefix", type=Path, action="append", required=True)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--tail", type=int, default=12)
    args = ap.parse_args()
    for probe, prefix in zip(args.probe, args.out_prefix):
        print(f"##### {probe}", flush=True)
        summarize(probe, prefix, args.top, args.tail)


def summarize(probe: Path, out_prefix: Path, top_n: int, tail_n: int) -> None:
    df = pd.read_csv(probe, sep="\t")
    if df.empty:
        print("empty probe file")
        return
    num = ["tau", "iters_max", "iters_mean", "rel_res_max", "largest_scope_rows", "worst_scope_rows",
           "worst_scope_max_degree", "worst_scope_jacobi_max", "jacobi_max", "max_degree",
           "path_log_w_range", "call_s", "num_scopes", "rows"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    g = df.groupby(["epoch", "mode", "direction"], sort=True)
    per = g.agg(solves=("iters_max", "size"), iters_max=("iters_max", "max"),
                iters_mean=("iters_max", "mean"), iters_p95=("iters_max", lambda s: s.quantile(0.95)),
                tau_min=("tau", "min"), tau_max=("tau", "max"), rel_res_max=("rel_res_max", "max"),
                largest_scope_rows=("largest_scope_rows", "max"),
                worst_scope_max_degree=("worst_scope_max_degree", "max"),
                jacobi_max=("jacobi_max", "max"), path_log_w_range=("path_log_w_range", "max"),
                s_per_call=("call_s", "mean"), wall_s_end=("wall_s", "max")).reset_index()
    per = per.round(4)
    per.to_csv(f"{out_prefix}_per_epoch.tsv", sep="\t", index=False)
    top = df.sort_values(["iters_max", "rel_res_max"], ascending=False).head(top_n).copy()
    top.insert(0, "which", "top")
    tail = df.tail(tail_n).copy()
    tail.insert(0, "which", "last")
    pd.concat([top, tail]).to_csv(f"{out_prefix}_top_solves.tsv", sep="\t", index=False)
    tr = df[df["mode"] == "train"]
    rows = []
    for direction in ("fwd", "bwd"):
        sub = tr[tr["direction"] == direction]
        for c in ("tau", "largest_scope_rows", "worst_scope_rows", "worst_scope_max_degree",
                  "worst_scope_jacobi_max", "max_degree", "path_log_w_range", "rows"):
            x, y = sub[c].to_numpy(float), sub["iters_max"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            r = spearmanr(x[ok], y[ok]).statistic if ok.sum() > 2 and np.ptp(x[ok]) > 0 and np.ptp(y[ok]) > 0 else float("nan")
            rows.append({"direction": direction, "variable": c, "n": int(ok.sum()),
                         "spearman_with_iters": round(float(r), 4)})
    pd.DataFrame(rows).to_csv(f"{out_prefix}_drivers.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 400):
        print(per.to_string(index=False))
        print(pd.concat([top, tail]).to_string(index=False))
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
