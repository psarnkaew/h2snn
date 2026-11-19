# TSUBAME card — run infrastructure for the real runs (phase 1: baselines on HGB NC and small LP)

Written 2026-11-18 by the design session (ops). Author request of 2026-11-18: "write the TSUBAME card; read the handbook; a detailed guide of only what we use; baselines first". Binding: this card and `docs/tsubame_guide.md` (same design-docs commit). Report: `docs/H2S-NN_tsubame_report.md`. Evidence: `docs/tsubame_evidence/`.

## Why
Tuning and final seeds run only on TSUBAME (decisions log 2026-11-17, "D6b scope; TSUBAME for tuning" and "TSUBAME quota"). Our launchers already do the hard part. Bayes + Hyperband W&B sweeps with fingerprinted, resumable state (`src/utils/wandb_campaign.py`), one process per trial, `--resume`, and `--final_seeds` (card 2 D3, D6.8). What is missing is the TSUBAME side:
1. the scheduler is **Grid Engine**, not SLURM;
2. a job may run **at most 24 h**;
3. at most **30 jobs run at once on weekdays** (100 on weekends), and that limit counts jobs, not GPUs;
4. results must come back to the lab cluster with their provenance.

Nothing in this card changes a model, a baseline adapter, a protocol or a search space.

## Preconditions
- `git fetch`; `origin/v3b` = `80e045a834820fef925d7241c1c4912a2f5eea92`; `origin/design-docs` contains this card's commit (SHA in the prompt). If either differs, stop.
- New branch `tsubame` from `80e045a`, then `git merge` the pinned design-docs SHA (merge, never rebase).
- Read `docs/tsubame_guide.md` (all), `scripts/slurm/run_sweep_agents.sbatch`, `scripts/run_baseline_tuning.py`, `src/utils/wandb_campaign.py` (state, resume, crashed-trial handling), `src/baselines/campaign.py`, and card 2 report D3, D5 and D6.8.

## Units (one commit each; push the branch after each)

### T0 — Inventory (no code)
`docs/tsubame_evidence/t0_inventory.md`. For one baseline tuning line, list every path the launcher writes (state file, trial records, `tuning_trials_run0.json`, final-seed records, W&B dirs, logs) and every environment variable or home-directory file it reads (W&B, conda, data root). Also state how a crashed trial is detected and re-run on `--resume` (file:line). This inventory fixes what T2 and T4 read.

### T1 — Grid Engine job scripts (`scripts/tsubame/`, new)
1. `sweep_slots.sh`: an array job. Each task is one node and runs `S` slots, `S` = GPUs of the resource type (4 for `node_f`, 1 for `gpu_1`). Slot `k` of task `t` is global slot `g = (t−1)·S + k`. With `G` = total slots, slot `g` runs lines `g, g+G, g+2G, …` of the cell list, one after another, each as `python <line> --resume`, with `CUDA_VISIBLE_DEVICES=k` and its own log file. Blank lines and `#` lines are skipped in the numbering exactly as `run_sweep_agents.sbatch` does. Header: `#$ -cwd`, `#$ -j y`, `#$ -l h_rt=24:00:00` (type and `-t` set by `submit_chain.sh`). At start it prints `git rev-parse HEAD`, `hostname`, `$JOB_ID`, `$SGE_TASK_ID`, and `nvidia-smi --query-gpu=name,memory.total,driver_version`. It runs `module purge` and then activates conda itself (no `-V`). It refuses to run if the checkout has uncommitted changes or `HEAD` is not the SHA in its directory name.
2. `submit_chain.sh --list FILE --type {node_f,gpu_1} --legs L [--first-task a --last-task b]`: submits leg 1 as an array of `ceil(lines / S)` tasks, then legs 2..L, each with `-hold_jid` on the previous leg. The group comes from `~/.t4_group` and goes on the `qsub` command line only. It prints the job ids and writes them to `<out>/submitted.tsv`. It refuses to start a chain for a list that already has a chain whose jobs are still queued or running (one chain per list, recorded in `submitted.tsv`).
3. `install_env.sh` (a `cpu_8` job): runs the existing `scripts/setup/install_env_b.sh h2snn-b <DGL file>`, then `build_sparse_tools.sh`, then prints `pip freeze`. It changes nothing in those scripts.
4. `env_check.sh` (a `gpu_1` trial run, ≤ 3 min): prints the driver and CUDA versions and the GPU name. It runs one CUDA operation each in torch, DGL and PyG (sparse matmul), then `tests/test_check_env.py` and `tests/test_factorized_maps.py -k cuda_float32` (the CUDA-only test that CPU suites skip), then an HTTPS reachability check of `api.wandb.ai`, `pypi.org` and `github.com` (status codes only, no credentials). Every check prints OK or FAIL, and the script exits non-zero on any FAIL.

### T2 — 24 h guard (`src/utils/wandb_campaign.py`; the only `src/` change)
Problem: a trial killed by the job limit stays "crashed" and is re-run on resume. A trial that takes longer than 24 h would therefore be re-run forever. Rule: the state records one attempt counter per trial (incremented when the trial starts, cleared when it finishes). On `--resume`, a trial with **2** unfinished attempts is not started again. The line ends with status `exceeds_24h`, written to the state and printed. Nothing else changes: selection, fingerprints and the order of trials are untouched, and the existing state files still load (old files have no counter, which means 0).
Tests (`tests/test_tsubame_guard.py`): (1) a worker that always dies is attempted exactly twice over three resumes, and the line status is `exceeds_24h`; (2) a worker that dies once and then finishes leaves the sweep complete, with the attempt cleared; (3) an old state file without counters resumes as before. **Mutation:** the guard never fires (limit = ∞) → test (1) fails. Existing `wandb_campaign` and launcher tests stay green.

### T3 — Cell lists (`scripts/tsubame/make_cell_list.py`)
- `--phase baselines-1`: one tuning line per **runnable** registry cell (`iter_cells()`, the same source as `campaign_summary`) with task NC on `acm`, `dblp`, `imdb` or LP on `acm_pa`, `dblp_ap`, `imdb_ma`, `pubmed`, `lastfm`, every baseline method. Line form as card 2 D5 (`--num_runs 1 --base_seed 0 --n_trials 100`), plus `--output_dir <out>/experiments/baselines --state_dir <out>/state --wandb_project h2snn_baseline_t4`. Sorted by (task, dataset, method). A cell with an unsupported reason is listed in a comment line with its reason, never run.
- `--final-seeds --out-dir <out>`: one `--final_seeds 0 1 2 3 4` line for each line whose sweep has 100 finished trials (read through T4's status reader). Any other line is listed as a comment with its count.
- Held out of phase 1 (later phases, other cards): Freebase NC and `freebase_bb` (S1 card 1e/1f and S3 Part B not on `v3b`), H2GB NC (B1a, card 1f), the large LP tasks, GC, H2S-NN.
Test (`tests/test_tsubame_cell_list.py`): the set of phase-1 lines equals the registry's runnable cells restricted to those 8 datasets; no duplicate; every line passes `run_baseline_tuning.py … --validate_only` (in-process call, nothing trained). **Mutation:** dropping the runnable filter → the test fails.

### T4 — Status and collection (`scripts/tsubame/status.py`, `collect.py`)
- `status.py --out DIR`: per line, prints finished trials / 100, crashed attempts, `exceeds_24h`, final seeds done / 5, and the last log time. It reads only the state and record files from T0. It never calls `qstat`.
- `collect.py --out DIR`: writes `collected/results.csv` with one row per trial and per final seed: task, dataset, method, kind (`trial`/`final_seed`), trial id or seed, sweep parameters (JSON), validation monitor, test metrics, v3b SHA, host, job id, GPU name, wall time. It also writes one `tar` per cell and `SHA256SUMS`. It reads existing records only and computes no new metric.
Test (`tests/test_tsubame_collect.py`, a synthetic output tree): the row count equals the number of records, values are copied unchanged, and the provenance columns are filled. **Mutation:** a record silently skipped → the test fails.

### T5 — Local proof of the job logic (lab cluster, CPU job)
1. `tests/test_tsubame_slots.py`: run `sweep_slots.sh` with fake `SGE_TASK_ID`/`JOB_ID`, a fake `nvidia-smi`, and a fake launcher that records its arguments. Check, for `node_f` and `gpu_1` and for list lengths 1, 4, 5 and 13, that every line runs exactly once over all tasks and slots, comment lines are skipped, each slot sees its own `CUDA_VISIBLE_DEVICES`, and the script refuses a dirty checkout. **Mutation:** slot index computed without the task offset → a line runs twice → the test fails.
2. `submit_chain.sh` with a fake `qsub` that echoes: legs are held on the previous leg, `-g` is on the command line, and a second chain for the same list is refused.
3. The targeted tests of T2–T5 and the existing `wandb_campaign`, launcher and `test_baselines*` tests run as one CPU job (`sbatch`, no `--gres`, 16G). No full suite: `src/` changes only in `wandb_campaign.py`, and golden/mb0 do not depend on it.

### T6 — On TSUBAME (the **author** runs it from `docs/tsubame_guide.md` §2–§4; the agent writes the exact commands into the report)
After the author approves the branch SHA and `v3b` moves:
1. Checkout of the approved SHA, vendors, data (phase 1), Miniforge, `install_env.sh`.
2. `env_check.sh` as a trial run → paste the output into `docs/tsubame_evidence/t6_env_check.txt`.
3. **One launcher check per loop class**, as a trial run where possible: `run_baseline_tuning.py … --dry_run --n_trials 1` on one HGB NC cell (full batch) and one LP cell. This checks the launcher, W&B and the paths; it is not a result (W&B tag `dry-run`, project `h2snn_baseline_t4_check`, never the tuning project).
4. If all are OK: generate the phase-1 list and submit the chain (guide §6).
A FAIL in step 2 or 3 stops everything. The design session then writes a fix card. Nothing is patched on TSUBAME.

## Rules
- No performance experiment, no metric reported, no comparison between methods. T6 step 3 is a launcher check with a budget of 1 epoch.
- Never edit a vendor, an adapter, a model component, a search space or a budget. `src/` changes only for T2.
- Never put a group name, a key or a username of TSUBAME in git. Scripts read `~/.t4_group`, and W&B uses `~/.netrc`.
- Lab cluster jobs: `sbatch`/`srun` with `--mem` and `--time`, no A6000 / `g3an01`, no GPU request (everything here is CPU).
- Git: one commit per unit; stage by explicit path; never `git add -A`, stash, reset or rebase; push only `tsubame`; never touch `v3b`.

## Stop conditions
- T0 finds that crashed trials are not re-run on `--resume` as the sbatch header says (the guard would then rest on a wrong premise).
- T2 needs a change outside `wandb_campaign.py`, or changes the order or selection of trials.
- T3's line set differs from the registry's runnable cells for the 8 datasets.
- Any test or mutation outcome other than expected.
Stop and report; do not fix past a stop.

## Report
`docs/H2S-NN_tsubame_report.md`: a table of units (commit, job ids, result); deviations from this card (each with its reason); the exact T6 commands for the author; open questions. Last line:
`branch=tsubame sha=<full head as pushed> card=docs/H2S-NN_tsubame_card.md base=80e045a834820fef925d7241c1c4912a2f5eea92`

## Estimate
T0–T5: about one working day for the agent, plus queue time for the CPU job. It takes longer if T0 finds the resume logic differs from the sbatch header (a stop). T6: about 2–3 h of the author's time on TSUBAME plus queue. A `node_f` wait could add hours, and if compute nodes have no internet the environment has to be installed offline, which adds about half a day.
