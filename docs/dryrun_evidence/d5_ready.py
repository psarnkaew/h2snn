"""Card 2 D5: baseline cells ready to tune now, and the launcher list for scripts/slurm/run_sweep_agents.sbatch.

Ready = status OK at the largest point on a 24 GB GPU (cells.csv, attempt 24GB) AND not changed by card 1d (Hetero2Net
HGB NC: validation route D2; PSHGCN on MAG; the H3b cells) or card 3 (HetSheaf; Freebase NC of SeHGNN, PSHGCN, H2SGNN).
Tuning as spec §6: plain Bayes, 100 trials, seed 0 (one unit, base seed 0). Run anywhere (text only):
    python d5_ready.py <evidence dir>
"""
import csv
import sys
from pathlib import Path

ev = Path(sys.argv[1])
rows = [r for r in csv.DictReader(open(ev / "cells.csv")) if r["attempt"] == "24GB"]
H3B = {("han", "node_classification", "imdb"), ("hinormer", "node_classification", "freebase"),
       ("han", "link_prediction", "imdb_ma"), ("h2sgnn", "link_prediction", "imdb_ma"),
       ("h2sgnn", "link_prediction", "freebase_bb"), ("sehgnn", "node_classification", "ieee-cis-g")} | {
       ("chebnetii", "node_classification", d) for d in ("ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem")}
HGB_NC = {"acm", "dblp", "imdb", "freebase"}


def excluded(m: str, t: str, d: str) -> str:
    if (m, t, d) in H3B:
        return "card 1d H3b"
    if m == "hetero2net" and t == "node_classification" and d in HGB_NC:
        return "card 1d D2 (Hetero2Net HGB NC validation route)"
    if m == "pshgcn" and d in ("ogbn-mag", "mag-year"):
        return "card 1d H1 (PSHGCN MAG)"
    if m == "hetsheaf":
        return "card 3 part A"
    if t == "node_classification" and d == "freebase" and m in ("sehgnn", "pshgcn", "h2sgnn"):
        return "card 3 part B"
    return ""


ready, held = [], []
for r in rows:
    if r["model"] == "h2snn":
        continue
    why = excluded(r["model"], r["task"], r["dataset"])
    if r["status"] == "OK":
        (held if why else ready).append((r, why))
ready.sort(key=lambda x: (x[0]["task"], x[0]["model"], x[0]["dataset"]))
lines = [f"scripts/run_baseline_tuning.py --task {r['task']} --models {r['model']} --datasets {r['dataset']} "
         f"--num_runs 1 --base_seed 0 --n_trials 100 --output_dir experiments/baselines --wandb_project h2snn_baseline"
         for r, _ in ready]
(ev / "d5_ready_baselines.txt").write_text("\n".join(lines) + "\n")
L = [f"Baseline cells ready to tune now: {len(ready)} (OK at 24 GB, not changed by card 1d or card 3); "
     f"held back although OK: {len(held)}.", "", "| # | task | model | dataset | GPU peak GiB | train epoch s |",
     "|---|---|---|---|---|---|"]
L += [f"| {i} | {r['task']} | {r['model']} | {r['dataset']} | {r['gpu_peak_alloc_gib']} | {r['t_train_epoch_s']} |"
      for i, (r, _) in enumerate(ready, 1)]
L += ["", "OK but held back (changed by card 1d / card 3):", ""]
L += [f"- {r['model']} {r['task']} {r['dataset']}: {why}" for r, why in held]
(ev / "d5_ready_baselines.md").write_text("\n".join(L) + "\n")
print(len(ready), len(held))
