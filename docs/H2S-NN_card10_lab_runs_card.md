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

## Amendment 1 (2026-11-20, author: option A)
- L1 accepted (`cd0de1a`, job 18033), with its three additions: GPU-type record keyed by **cell** (a cell's tuning and final-seeds lines share one GPU type), GPU-name check against `--gpu-type`, `--cpus` default 4 and optional `--max-parallel`.
- **L2 = option A:** the launchers define and run the point. Order: do **L3 now**; then, after card 8 is on `v3b` (the ops session gives the SHA), `git merge` that `v3b` into `card10` and do L2: extend card 8's `--grid_point` of `scripts/run_baseline_tuning.py` with `max` (largest value of every size axis, central value of the others, as the L2 bullet), and add the same `--grid_point {min,max}` to `scripts/run_h2snn_wandb_tuning.py` (scripts only; no `src/` change; same guard: only with `--dry_run --n_trials 1`); each prints peak GPU memory. `fit_check.py` only builds those lines, submits them with `submit_lab.sh` and reads the results into `fit_check.tsv`. Option B (calling trial functions directly) is not used.
- Tests: `--grid_point max` picks the maximum of every size axis on both launchers (mutation: minimum → fails); the guard refuses it without `--dry_run --n_trials 1`.

## Amendment 2 (2026-11-20, author)
Do these after L3 (now) — they do not wait for card 8 — then stop again for L2 as Amendment 1 says.

**A2.1 — Every whole-H100 resource type on TSUBAME.** `scripts/tsubame/sweep_slots.sh` and `submit_chain.sh` accept `--type {gpu_1, node_q, node_h, node_f}` with slots per task S = 1, 1, 2, 4 (handbook §5.1.1: `node_q` 48 cores/192 GB/1 GPU, `node_h` 96/384/2, `node_f` 192/768/4, `gpu_1` 8/96/1). `node_o` and `gpu_h` (half an H100, MIG) are **refused** with a message (spec range rule: one full H100 94 GB per trial). The slot formula, the one-chain-per-list rule and `submitted.tsv` stay as they are; a list may be resubmitted later with another whole-H100 type (same GPU type, so the cell rule holds). Test (fake `qsub`, as T5): for each type and list lengths 1, 3, 5, every line runs exactly once and each slot sees its own `CUDA_VISIBLE_DEVICES`; MIG types refused. **Mutation:** S of `node_h` set to 4 → a line is skipped or doubled → fails. (These two files are not card 8's.)

**A2.2 — Fit-check headroom (L2).** A cell `fits` only if the peak GPU memory of the largest point, over one training epoch **and** the evaluation of the whole validation and test splits (the dry run's own evaluation; the test asserts all three phases are measured), is ≤ **75 %** of the GPU's memory (24 GB → 18 GB; A6000 48 GB → 36 GB). `fit_check.tsv` gains `peak_fraction`. Mutation: threshold 100 % → a crafted 90 % peak passes → fails.

**A2.3 — Failures are reported, never moved automatically (author).** New `scripts/tsubame/failures.py --out DIR --list FILE` (works on lab and TSUBAME outputs): one row per failed trial with line, cell, attempt, condition (`StopConditionError`, `PCGConvergenceError`, OOM, `other`), the error's last lines from the worker log, job id, host and GPU; writes `failures.tsv` and prints a summary. It reads the state's `failures[]` and the line logs only. A lab line with an OOM keeps running as it is; the ops session reports it to the author, who decides whether the cell moves. Test on a synthetic output tree (an OOM, a stop condition, an `other`); mutation: a failure skipped → fails.
