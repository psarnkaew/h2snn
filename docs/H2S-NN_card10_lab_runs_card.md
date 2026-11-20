# Card 10 — Lab-cluster runs (SLURM) for light cells, with the same outputs and provenance as TSUBAME

Written 2026-11-20 by the design session (ops). Decision: `docs/experiment_settings_decisions.md`, row of 2026-11-20 "Training on the lab cluster unlocked (5 conditions); A6000 allowed". Report: `docs/H2S-NN_card10_report.md`. Evidence: `docs/card10_evidence/`.

**Renamed 2026-11-20:** this card was first committed as "card 9" (`653054f`); the model session's card 9 (ablation scripts to pre-registration v3) took that number and branch, so this is **card 10** (branch `card10`, base `v3b` `e5bdf10`). Ablation needs no tuning list any more (spec 1.29), so condition (3) concerns only the ablation seed runs and timing.

## Why
TSUBAME points are limited (balance 1,580 on 2026-11-20). Light cells move to the lab cluster (free), heavy cells stay on TSUBAME. The lab must produce the **same** sweep state, records, line logs and provenance as `scripts/tsubame/sweep_slots.sh`, so that `status.py`, `lines.py`, `make_cell_list.py --final-seeds` and `collect.py` work on lab outputs unchanged.

**Conditions (author, binding):** (1) one cell = one GPU type for all its trials and final seeds, recorded in provenance; (2) a cell goes to the lab only if the largest point of its search grid fits the lab GPU (checked by one trial at that point; OOM → the cell goes to TSUBAME); (3) ablation and timing stay on TSUBAME; (4) the perturbation study runs on one GPU type; (5) the paper states the GPU type per (method, dataset) and that timings are H100.

## Preconditions
- Work in a new worktree `/mnt/data/peeradon/h2snn-card10` (`git worktree add`), branch `card10` from `origin/v3b` `e5bdf10432e32c8fe31e8f3faff814951461d4d2`, then `git merge` the pinned design-docs SHA. Card 8 runs in parallel in `/mnt/data/peeradon/h2snn` on `card8`: **do not edit** `scripts/tsubame/make_cell_list.py`, `src/baselines/**`, `configs/baselines/**` (card 8 owns them).
- Read: this card; `docs/tsubame_guide.md` §3, §6, §9; `scripts/tsubame/{sweep_slots.sh,submit_chain.sh,lines.py,status.py,collect.py}`; `scripts/slurm/run_sweep_agents.sbatch`; `docs/lab-gpu-cluster.html` (SLURM rules); `sinfo -o "%n %G %m"`.

## Units (one commit each; push `card10` after each)

### L1 — Lab slot runner (`scripts/slurm/lab_slots.sbatch`, `scripts/slurm/submit_lab.sh`)
- `lab_slots.sbatch`: SLURM array job; task `t` runs line `t` of a cell list (blank and `#` lines skipped in the numbering exactly as `lines.py`), as `python <line> --resume`, with **the same** line-log path, provenance JSONL (`logs/prov/<job>.<task>.0.jsonl`: head, host, job id, task id, GPU name, start/end, exit) and start-of-job prints as `sweep_slots.sh`, so `collect.py` reads them. Refuses a dirty checkout or a HEAD that differs from the SHA in the checkout's directory name (as `sweep_slots.sh`). `--mem`, `--time` (≤ 72 h), `--gres=gpu:<type>:1` are required arguments of `submit_lab.sh`; no default GPU type.
- `submit_lab.sh --list FILE --out DIR --gpu-type {3090,titan_rtx,a6000,2080ti} --mem M --time T [--legs L]`: submits the array, and legs 2..L with `--dependency=afterany` on the previous leg (72 h limit). It records `(line, gpu_type)` in `<out>/lab_gpu_types.tsv` and **refuses** a line already recorded with another GPU type (condition 1) and a list that already has unfinished jobs (`squeue`), as `submit_chain.sh`.
- Checkout for lab runs: a read-only clone per approved `v3b` SHA at `/mnt/data/peeradon/h2snn-runs/<sha>/H2S-NN` (vendors by `clone_vendors.sh`, `data` → the lab data root), never the agents' working trees.
- Test (`tests/test_lab_slots.py`, fake `sbatch`/`squeue`/`nvidia-smi`, fake launcher): every line runs exactly once over the array, comment lines skipped, provenance written in the TSUBAME format and read back by `collect.py`'s reader; a second GPU type for a recorded line is refused. **Mutation:** the GPU-type record is not checked → the test fails.

### L2 — Largest-point fit check (`scripts/slurm/fit_check.py`)
- For each line of a list, one trial at the **largest point of every size axis** of that cell's search space (widths, layers, heads, `d`, `ℓ_max`, `r`, batch size; all other keys at their central value), 1 epoch, `--dry_run`, on the GPU type the line will use. Output `<out>/fit_check.tsv`: line, gpu type, peak GPU memory, status (`fits` / `oom` / `error`). Read the search spaces from the launchers' own functions (H2S-NN `build_sweep_config`, baselines `search_space`), never by guessing; if a launcher cannot run one fixed point without a new flag, stop and report.
- `fits` lines form the lab list; `oom` lines go to TSUBAME. Nothing else is decided by this check (not a result).
- Test: on a fake launcher, the chosen point is the maximum of every size axis (mutation: minimum → fails).

### L3 — Lab list split (`scripts/slurm/site_split.py`)
- Reads a TSUBAME phase list (from `make_cell_list.py`, unchanged) and a `fit_check.tsv`, writes `<list>.lab.txt` and `<list>.t4.txt` (disjoint; union = input), never moving a line that already has state under the other site's output directory. Test + mutation (a line in both → fails).

### L4 — Targeted tests (one CPU job, no `--gres`, 16G) and report
`git diff --stat e5bdf10 -- H2S-NN/src` must be empty (no `src/` change). Report: unit table, how to run (commands for the H2S-NN GC lines and for a baseline list), deviations.

## Rules
- Correctness only: no real-data run except L2's fit check, which is not a result and is run only after the author approves this card. No metric reported.
- Compute via `sbatch` with `--mem` and `--time`; the A6000 (`g3an01`) is allowed (author 2026-11-20).
- Git: one commit per unit; stage by explicit path; never `git add -A`, stash, reset or rebase; push only `card10`; never touch `v3b`.

## Stop conditions
- A file owned by card 8 needs a change.
- The provenance or output layout cannot match the TSUBAME layout without changing `collect.py`/`lines.py` (report the minimal change instead).
- A launcher cannot run a fixed largest point (L2).
- Any unexpected test or mutation outcome.

## Report
`docs/H2S-NN_card10_report.md`; last line `branch=card10 sha=<head as pushed> card=docs/H2S-NN_card10_lab_runs_card.md base=e5bdf10432e32c8fe31e8f3faff814951461d4d2`.

## Estimate
About half a day to one day; longer if a launcher needs a flag to pin the largest point.
