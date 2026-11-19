# TSUBAME card — report (run infrastructure, phase 1: baselines)

Card: `docs/H2S-NN_tsubame_card.md` and `docs/tsubame_guide.md` at design-docs `709e3eb`.
Base (prompt amendment): `v3b` = `9b7ae0b66de790a96ad1b6ad6d4b83caa7a4009c` (card 1e F3 + card 1f U6). Wherever the card says `80e045a`, this branch used `9b7ae0b`.
Branch: `tsubame`, worktree `/mnt/data/peeradon/h2snn-tsubame`.

## Preconditions
- `git fetch`:
  - `origin/v3b` = `9b7ae0b66de790a96ad1b6ad6d4b83caa7a4009c`;
  - `origin/design-docs` contains `709e3eb`.
- Branch `tsubame` created from `9b7ae0b`. `git merge 709e3eb` said "Already up to date": `709e3eb` is already an ancestor of `9b7ae0b`, so there is no merge commit.
- `origin/v3b` was checked again before T5 (job 17672) and before this report: still `9b7ae0b`. No overlap merge was needed.

## Units

| Unit | Commit | Job(s) | Result |
|---|---|---|---|
| T0 inventory | `06bae7e` | — | `docs/tsubame_evidence/t0_inventory.md`. Crashed trials are re-run on `--resume` exactly as the `run_sweep_agents.sbatch` header says (`wandb_campaign.py:631-637`, `:688`). **No stop.** |
| T1 job scripts | `3bb6a41` (+ two fixes in `2d5c348`) | tested in T5 | `scripts/tsubame/{sweep_slots.sh, submit_chain.sh, install_env.sh, env_check.sh, lines.py}` |
| T2 24 h guard | `9b2a564` | 17667 | 3/3 tests OK. Mutation (limit = ∞) caught: attempt 0 was started 3 times. |
| T3 cell lists | `c04b835` | 17669 | 2/2 tests OK. 89 lines = the registry's 89 runnable phase-1 cells, all pass `--validate_only`, 15 unsupported cells are comments. Mutation (runnable filter dropped) caught: 8 extra LP cells. |
| T4 status / collect | `2d5c348` | 17670 | 3/3 tests OK. Mutation (a record skipped) caught: 4 ≠ 6 rows. |
| T5 local proof | `f86f1fa` | 17683 (setup: 17674) | `tests/test_tsubame_slots.py`: slots for node_f/gpu_1 × 1/4/5/13 lines + a G < lines case, dirty and wrong-dir refusal, chain (held legs, `-g` on the command line only, second chain refused). One CPU job: T2–T5 tests + existing `wandb_campaign`, launcher and all `test_baselines*` tests: **186 OK, 0 skipped, 0 failures**. Mutations T2, T3, T4 and T5 all caught (T5: line `m00` runs twice). |

### T5 runs before the recorded one (development, all CPU jobs)
- **17672: 108 errors/failures.**
  - The baseline tests could not import their vendors: the new worktree had no `vendor/` clones and no `data`.
  - The new slot tests failed because my fake `python` also recorded the slot script's own `python -c` helper calls. That was a test bug; the script was right.
  - Fix:
    - `data` → a symlink to `/mnt/data/peeradon/h2snn/H2S-NN/data`, read only, as in the dry-run worktree;
    - `scripts/setup/clone_vendors.sh` + `apply_vendor_patches.sh`, the T6 commands;
    - `build_sparse_tools.sh` (job 17674);
    - the fake records only `scripts/run_baseline_tuning.py`.
  - Nothing is staged from `data/` or `vendor/`.
- **17675: 182 OK, 4 slot subtests failed.** The `gpu_1` subtests reused the `node_f` subtests' output directory and counted their provenance events. Another test bug, fixed with one output directory per subtest.
- **17677:** the same 4 subtests failed once more, in the old test's exact pattern.
  - I believe the compute node read a stale NFS copy of the just-edited test file. Files written by jobs also reached the login node late in this session.
  - Two further runs on two nodes (17681 on g2tn01, 17682 on g2n01) and the recorded run 17683 passed.
- **No line of a T1–T4 script changed after these failures.** Only `tests/test_tsubame_slots.py` changed.

Every job is a CPU `sbatch` job: no `--gres`, 16G, through `docs/tsubame_evidence/t_unit.sbatch`. Its output is in `docs/tsubame_evidence/t{2,3,4}_tests.txt` and `t5_targeted.txt` (verbose log `t5_targeted_verbose.log`). No GPU job ran, nothing was trained, no metric was produced, and no full suite ran.

### What each unit does
- **T1 `sweep_slots.sh`**:
  - Header `#$ -cwd`, `#$ -j y`, `#$ -l h_rt=24:00:00`.
  - Refuses a dirty checkout (tracked files) and a `HEAD` that differs from the repository directory's name.
  - Prints `HEAD`, host, `$JOB_ID`, `$SGE_TASK_ID` and `nvidia-smi --query-gpu=name,memory.total,driver_version`.
  - Runs `module purge` (after sourcing `/etc/profile.d/modules.sh` if present), then activates conda itself.
  - Slot `k` (0-based) of task `t` is global slot `g = (t−1)·S + k` and runs runnable lines `g+1, g+1+G, …`, each as `python <line> --resume` with `CUDA_VISIBLE_DEVICES=k` and its own log, `<out>/logs/lines/<list>.line<NNN>.<job>.<task>.<k>.log`.
  - Every line run writes a start event (before the launcher) and an end event to `<out>/logs/prov/<job>.<task>.<k>.jsonl`. Each event holds the SHA, host, job, task, slot, GPU name, times, exit code, and the line's state before and after (via `lines.py snapshot`).
  - Always exits 0, so Grid Engine never sees 99 or 100. Per-line exit codes are in the events and logs.
- **T1 `submit_chain.sh`**:
  - Leg 1 is an array of `K = ceil(lines/S)` tasks with `G = K·S`. Legs 2..L are each `-hold_jid` on the previous leg.
  - `-g $(~/.t4_group)` goes only on the `qsub` command line.
  - Writes job ids to stdout and `<out>/submitted.tsv`. `<out>` defaults to the list's directory.
  - Refuses a second chain for the same list while any of its jobs is in `qstat` (called once). `--first-task/--last-task` restrict `-t` and keep `K` and `G`.
- **T1 `install_env.sh` / `env_check.sh`**: as the card describes. Details are under Deviations.
- **T2**, in `src/utils/wandb_campaign.py` only (+35 lines):
  - Before each worker starts, the state gets `open_attempt = {attempt_index, starts}`. It is removed when the worker returns, whatever its outcome.
  - When an attempt has `starts ≥ MAX_UNFINISHED_ATTEMPTS = 2`, the line writes `line_status = "exceeds_24h"` to the state, prints it, and raises `TrialExceedsJobLimitError` without starting the attempt.
  - Unchanged: fingerprints, `TUNING_STATE_VERSION`, the order of attempts, `sweep_objective`, `select_best_trial` (card 1f U6), and records.
  - States without the key count 0 (test 3).
- **T3 `make_cell_list.py`**:
  - `--phase baselines-1` builds the list from `iter_cells()`, restricted to the 8 datasets and sorted by (task, dataset, method).
  - `--final-seeds --out-dir <out>` reads each tuning line through `lines.snapshot` (the reader `status.py` uses). A line whose sweep has 100 finished attempts and no `exceeds_24h` gets one `--final_seeds 0 1 2 3 4` line. Every other line becomes a comment with its count.
- **T4**:
  - `status.py`: finished/100, records, failed, killed starts, status, seeds/5 and last log time per line. It never calls `qstat`.
  - `collect.py`: `collected/results.csv` (one row per trial record and per final-seed record, values copied), `collected/cells/<token>.tar` and `SHA256SUMS`.

## Deviations from the card (each with its reason)
1. **Base SHA** `9b7ae0b`, not `80e045a`: per the prompt amendment.
2. **Line numbering.** The card says blank and `#` lines are "skipped in the numbering exactly as `run_sweep_agents.sbatch` does". That sbatch file's *header* says they are skipped, but its *code* (`sed -n "${SLURM_ARRAY_TASK_ID}p"`) numbers every file line and treats a comment line as a no-op task. I followed the card's words: only runnable lines are numbered, so the 15 comment lines of the phase-1 list take no slot. Either way every line runs exactly once.
3. **Provenance lives outside the records.** Trial records carry no host, job, GPU or timing (T0 §1), and T2 forbids touching records. So the slot script writes start and end events, and `collect.py` attributes record `i` of a state to the line run that started with ≤ `i` records. **`wall_time_s` is the wall time of that line run**, not of the trial, and it is empty when the 24 h limit killed the run (no end event).
4. **Final-seed lines in chained legs.** `--final_seeds` has no resume in the launcher, so a finished final-seeds line would retrain all 5 seeds in every later leg. `sweep_slots.sh` therefore skips a final-seeds line whose 5 seed files all exist (`lines.py complete`). A final-seeds run killed part-way retrains all its seeds in the next leg.
5. **"100 finished trials"** = `attempted_trials` (successful + pruned + failed attempts whose bookkeeping completed), not successful records: Hyperband-pruned trials have no record by design.
6. **The guard raises.** When the guard fires, the launcher exits non-zero for that line (`TrialExceedsJobLimitError`), and every later leg re-prints `exceeds_24h` at once. A phase-1 line holds one cell, so no other cell is held back. The guard is in the shared engine, so it applies to the H2S-NN tuning launcher too (open question 2).
7. **Extra file `scripts/tsubame/lines.py`** (stdlib only). It holds the list parsing and the line → state/output paths shared by the slot script, status, collect and make_cell_list. `test_line_paths_are_the_launchers` checks its paths against `campaign.state_path_for` on all 89 lines.
8. **`install_env.sh`** creates the env (`conda create -n h2snn-b python=3.12`) when it is missing, because `install_env_b.sh` installs into an existing env (its header). `install_env_b.sh` and `build_sparse_tools.sh` are called unchanged.
9. **`env_check.sh`** runs the CUDA-only test with `python -m unittest -k cuda_float32 tests.test_factorized_maps`, because the repository's tests are unittest. The check requires `Ran 1 test`, no skip, and exit 0.

## T6 — commands for the author (on TSUBAME; guide §2–§4)
After the author approves this branch and `v3b` moves to it, set `SHA` to the approved `v3b` SHA. Never edit a checkout. A FAIL in step 2 or 3 stops everything (card T6).

**0. One-time setup** (guide §2).
- Account, SSH key and group membership.
- Write the group name into `~/.t4_group` (one line):
  ```
  echo <group> > ~/.t4_group
  ```

**1. Checkout, vendors, data, Miniforge, env.** On the TSUBAME login node:
```
WORK=${HOME/home/work}; SHA=<approved v3b sha>; OUT=$WORK/h2snn-out/$SHA
git clone git@github.com:psarnkaew/h2snn.git $WORK/h2snn-runs/$SHA
cd $WORK/h2snn-runs/$SHA && git checkout --detach $SHA
cd H2S-NN && bash scripts/setup/clone_vendors.sh && bash scripts/setup/apply_vendor_patches.sh
mkdir -p $WORK/h2snn-data/NC $WORK/h2snn-data/LP $OUT/logs ~/wheels && ln -s $WORK/h2snn-data data
```
From the lab-cluster login node (`<t4>` = your TSUBAME login, `<WORK>` = the path printed by `echo $WORK` there):
```
rsync -av /mnt/data/peeradon/h2snn/H2S-NN/data/NC/{acm,dblp,imdb} <t4>:<WORK>/h2snn-data/NC/
rsync -av /mnt/data/peeradon/h2snn/H2S-NN/data/LP/ <t4>:<WORK>/h2snn-data/LP/
rsync -av /mnt/home/peeradon/wheels/dgl-2.4.0.th24.cu124-py312_0.tar.bz2 <t4>:~/wheels/
```
Back on TSUBAME. Download the Miniforge installer on the login node, then:
```
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
cd $WORK/h2snn-runs/$SHA/H2S-NN
qsub -g $(cat ~/.t4_group) -l cpu_8=1 -l h_rt=2:00:00 -o $OUT/logs scripts/tsubame/install_env.sh
```
The log in `$OUT/logs/install_env.sh.o<job>` must end with `exit=0`. If compute nodes have no internet, the install fails at pip; stop and report (guide §4).

Then, on the login node:
```
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b && wandb login
```
This writes the key to `~/.netrc`, never to the repo.

**2. Environment check** (trial run, no `-g`):
```
cd $WORK/h2snn-runs/$SHA/H2S-NN
qsub -l gpu_1=1 -l h_rt=0:03:00 -o $OUT/logs scripts/tsubame/env_check.sh
```
- Every line must say `OK`, and the log must end with `fails=0`.
- Paste the whole log into `docs/tsubame_evidence/t6_env_check.txt` (in a design-docs or branch commit as the design session says).

**3. One launcher check per loop class.** One HGB NC cell (full batch) and one LP cell, `--dry_run` (1 epoch), 1 trial, project `h2snn_baseline_t4_check`. It runs through `sweep_slots.sh` itself, so the slot script is checked on TSUBAME too, as a trial run of 2 array tasks:
```
CHK=$WORK/h2snn-out/$SHA-check; mkdir -p $CHK/logs
cat > $CHK/cells_check.txt <<EOF
scripts/run_baseline_tuning.py --task node_classification --models gcn --datasets acm --num_runs 1 --base_seed 0 --n_trials 1 --dry_run --output_dir $CHK/experiments/baselines --state_dir $CHK/state --wandb_project h2snn_baseline_t4_check
scripts/run_baseline_tuning.py --task link_prediction --models gcn --datasets acm_pa --num_runs 1 --base_seed 0 --n_trials 1 --dry_run --output_dir $CHK/experiments/baselines --state_dir $CHK/state --wandb_project h2snn_baseline_t4_check
EOF
qsub -l gpu_1=1 -l h_rt=0:03:00 -t 1-2 -o $CHK/logs scripts/tsubame/sweep_slots.sh $CHK/cells_check.txt $CHK 1 2
```
Then `python scripts/tsubame/status.py --out $CHK --list $CHK/cells_check.txt` must show `1/1` and `done` for both lines. Each line log (`$CHK/logs/lines/cells_check.line00{1,2}.*.log`) must end with `exit=0`. The W&B runs carry the tag `dry-run`. This is a launcher check, not a result.

If a trial run's 3 minutes are too short for data loading, run the same `qsub` with `-g $(cat ~/.t4_group)` and `-l h_rt=0:20:00`. That is still a check: 1 epoch, check project.

**4. If steps 2 and 3 are OK: phase-1 list and chain** (guide §6):
```
cd $WORK/h2snn-runs/$SHA/H2S-NN && . ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
python scripts/tsubame/make_cell_list.py --phase baselines-1 --out $OUT/cells_tuning.txt     # prints "89 runnable line(s), 15 comment line(s)"
bash scripts/tsubame/submit_chain.sh --list $OUT/cells_tuning.txt --type node_f --legs 6     # 23 node_f tasks per leg
python scripts/tsubame/status.py --out $OUT                                                   # watch (no qstat)
```
When lines show `done`:
```
python scripts/tsubame/make_cell_list.py --final-seeds --out-dir $OUT
bash scripts/tsubame/submit_chain.sh --list $OUT/cells_final_seeds.txt --type node_f --legs 1
```
Collect and copy back:
```
python scripts/tsubame/collect.py --out $OUT
rsync -av <t4>:<WORK>/h2snn-out/$SHA/collected/ /mnt/data/peeradon/h2snn-results/tsubame/$SHA/
```
- `make_cell_list.py` imports the registry only. If it takes over a minute on the login node, run it in a `cpu_8` job instead.
- If `node_f` waits for hours: `qdel` the unstarted leg, then resubmit those lines as `gpu_1`. Lines of `node_f` task `t` are `gpu_1` tasks `4t−3 … 4t`:
  ```
  submit_chain.sh --type gpu_1 --first-task 4t-3 --last-task 4t
  ```
  The one-chain rule refuses this while any job of the first chain for that list is still queued (open question 1).

## Open questions (T0–T5; answered 2026-11-19 in the decisions log: one chain per list stays, the guard stays for H2S-NN, no per-trial wall time now, the extra record is accepted)
1. **Switching unstarted tasks to `gpu_1`.** `submit_chain.sh` refuses a second chain for a list while any job of the first is queued or running. Switching some tasks to `gpu_1` therefore needs the remaining `node_f` legs `qdel`ed first. Should a partial `gpu_1` chain on disjoint tasks be allowed alongside a running one? I did not allow it, because the card says one chain per list.
2. **The guard also applies to H2S-NN sweeps.** `run_unit_sweep` is shared, so `run_h2snn_wandb_tuning.py` gets it too. That is harmless for phase 1. For phase 2 (H2GB NC, H2S-NN; guide §6.6), a trial longer than 24 h stops at `exceeds_24h` after two legs. Epoch-level resume or a reservation remains the design session's decision.
3. **Trial-level wall time** needs a timing key in the records, which is a launcher change, outside T2's scope. Until then `wall_time_s` is per line run (deviation 3).
4. **T0 edge case.** If a job is killed between the worker's record save and the parent's bookkeeping, the record is kept and the attempt still runs again. That line then ends with one more successful record than attempts it counts. It was recorded, not changed.

## T7 — H2S-NN lines and queue order (2026-11-19)

Card §T7 and `docs/tsubame_guide.md` at design-docs `58cfc1b`.

**Preconditions and base**
- The first attempt stopped at the precondition check, because card 5 was not yet on `v3b` (`v3b` was `59ab582`). Nothing was merged then.
- Once `origin/v3b` was `365a75763f45dee4b8853cc85ae06f0084295c47`, I ran:
  - `git merge origin/v3b`: a fast-forward of `tsubame` from `0b0219b`;
  - `git merge 58cfc1b`: merge commit `50f714d`.
- `origin/v3b` was still `365a757` when T7.6 was submitted and when this report was written. No overlap merge was needed.

**Base for this section:** `365a75763f45dee4b8853cc85ae06f0084295c47`.

### Units

| Item | Commit | Job | Result |
|---|---|---|---|
| T7.1 `lines.py` reads H2S-NN lines | `392cfcf` | 17849 | 7/7 OK, on all 48 `h2snn` lines (24 tuning + 24 final seeds) |
| T7.2 status / collect for `h2snn` | `7ab032b` | 17852 | 5/5 OK. Mutation (an `h2snn` record skipped) caught: 4 rows instead of 6 |
| T7.3 final-seeds gate for `h2snn` | `fc0b0fd` | 17849 | 4/4 OK. Mutation (gate removed) caught: an unfinished cell got a final-seeds line |
| T7.4 phase `baselines-early` + disjointness | `fc0b0fd` (code), `6988d49` (tests) | 17849 | 4/4 OK. Mutation (no exclusion) caught: the lists overlap |
| T7.5 host memory per `node_f` task | `f45f04b` | 17851 | 1/1 OK. Largest known sum 80.5 GB < 600 GB, so no reorder |
| T7.6 targeted CPU job (36G, no `--gres`) | (report commit) | 17853 | g2n01, 17 min 35 s: 198 tests OK (0 skipped, 0 failures; T2–T7 tests, card 5 `CellListTests`, W&B/launcher tests, every `test_baselines*`); mutations t2, t3, t4, t5, t7_2, t7_3, t7_4 all caught (`docs/tsubame_evidence/t7_6_targeted.txt`) |

All jobs are CPU `sbatch` jobs. Evidence is in `docs/tsubame_evidence/t7_*`. `make_cell_list.py` carries both T7.3 and T7.4, so its diff is in one commit (`fc0b0fd`), and T7.4's tests follow in `6988d49`.

**What each item does**
- **T7.1:** `parse_line` accepts `scripts/run_h2snn_wandb_tuning.py` lines. Paths follow the launcher's `main()` and `run_final_seeds`:
  - state: `<--state_dir or output_dir/exp_name>/tuning_state_run<u>.json`, for `u` in `run_start .. run_start + (run_count or num_runs or 1)`;
  - final seeds: `<output_dir>/<exp_name>/final_seed<s>.json`.

  The test calls the launcher in-process:
  - `--validate_only` prints the state path it would use, with and without `--resume`; that path equals `state_paths()` on all 24 tuning lines;
  - `run_final_seeds` (with `train_trial`, `write_run_record` and `write_summary_files` replaced by recorders) writes exactly `cell_dir/final_seed0..4.json` on all 24 final-seeds lines;
  - `complete()` is true only when all 5 files exist.

  Lines must name `--exp_name` and `--output_dir`, because the launcher's fallback is the YAML config, which the stdlib-only `lines.py` does not read. Every `make_cell_list` H2S-NN line names both. Baseline behaviour is unchanged (`test_tsubame_collect` passes).
- **T7.2:** `collect.py`:
  - reads final seeds under `experiments/*/<token>/` (baselines and `h2snn`);
  - sets `method = h2snn`, with task and dataset taken from the token `h2snn_<task>_<dataset>`, because H2S-NN records have `dataset` and `task_type` but no `model`;
  - treats as cells only state directories and directories holding final seeds. H2S-NN's per-trial run directories (`<token>_tune_run0_<trial>`, written by its `train_trial`) are neither rows nor archived.

  `status.py` needed no change, since it reads through `lines.py`.
- **T7.3:**
  - `--phase h2snn` no longer writes `cells_final_seeds_h2snn.txt`.
  - `h2snn_final_seeds_list()` stays as a function, because card 5's `test_T9_h2snn_phase` reads it and that test is outside this card's scope.
  - `--final-seeds` uses one gate, `sweep_finished`, for baselines and H2S-NN alike: one unit, `attempted_trials ≥ 100`, no `exceeds_24h`.
- **T7.4:**
  - `baselines-early` = HetSheaf and CSNN on NC `acm` and `dblp` (4 lines, all runnable).
  - **No LP cell is early.** Card 7 perturbs NC graphs only (`data/NC/<DATASET>_pert/`, ACM and DBLP; card 7 §manifest) and needs the tuned configurations, not final seeds.
  - `baselines-1` lists those 4 cells as comments (`# in phase baselines-early …`), which leaves it 85 lines.
  - The T3 test's expected set now excludes the 4 early cells; that is the only change to an existing test.
  - The three lists are pairwise disjoint in cells, state files and lines. Together they equal the 89 runnable phase-1 baseline cells plus the 24 H2S-NN cells.
- **T7.5:** `scripts/tsubame/host_memory.py` and `docs/tsubame_evidence/t7_5_host_memory.md`. With 4 slots per task, in card 5's order:

  | `node_f` task | lines | known host-peak sum |
  |---|---|---|
  | 1 | acm, dblp, imdb, freebase NC | 20.9 GB (freebase is a lower bound: CUDA OOM before its peak) |
  | 2 | ogbn-mag, mag-year, oag-cs, oag-eng | 39.3 GB |
  | 3 | oag-chem, ieee-cis-g, h-pokec, pdns | **80.5 GB** (h-pokec 39.8 GiB at the largest point, card 2 D6.6) |
  | 4 | acm_pa, dblp_ap, imdb_ma, freebase_bb | 24.5 GB (lower bounds) |
  | 5 | lastfm, pubmed, ogbn-mag_pp, h-pokec_uu | 6.2 GB **+ 2 unknown** (no real-data run of the large LP tasks, D4b) |
  | 6 | 4 GC | 6.6 GB |

  - No known sum is near 600 GB, so the list keeps card 5's order.
  - **Caveat:** apart from H-Pokec, these peaks come from earlier model versions (card 2 D2 / D6 central point), not from the TD-1 model of `365a757` or the largest search-space points.

### Deviations (T7)
1. **T7.4 LP cells:** not early (card 7 is NC only), as the card's "only if card 7 needs them" allows.
2. **The T3 test changed:** `baselines-1` now excludes the early cells, which T7.4 requires.
3. **`h2snn_final_seeds_list()` is kept as a function** (card 5's test reads it). It is never written to a file.
4. **`t_unit.sbatch` gained `--exclude=g3an01`.** My first two T7 jobs (17845, 17846: CPU only, 8 s each, no GPU) were placed on **g3an01**, the A6000 node, because the script did not exclude it. Their output is not used: 17849 reran T7.1 on g3n01. Every later job excluded g3an01.
5. **Development failures before the recorded runs**, all in my new tests' fixtures; no script changed after them:
   - 17849: T7.2's fixture wrote a final-seed file before creating its directory;
   - 17851: T7.2's synthetic logs both had line number `001`.

   The recorded T7.2 run is 17852.
6. **Collected tars leave out H2S-NN's per-trial run directories** (`config.yaml`, `best_test_metrics.json`, checkpoints). Their records are in the sweep state.

### T6 commands for the author (supersede the T6 section above; three chains)
Steps 0–2 are as above, with two changes from guide §3 (`58cfc1b`):
- copy **all** of `data/`:
  ```
  rsync -avL /mnt/data/peeradon/h2snn/H2S-NN/data/ <t4>:<WORK>/h2snn-data/
  ```
  Then compare `du -sh $WORK/h2snn-data` with the lab's `du -shL data` and state both in the T6 evidence;
- `vendor/` is always a real clone (`clone_vendors.sh`), never a symlink.

`SHA` = the approved `v3b` SHA (≥ `365a757`), `OUT=$WORK/h2snn-out/$SHA`, run from `$WORK/h2snn-runs/$SHA/H2S-NN`.

**3. Launcher checks (trial runs).** As above, plus one H2S-NN line in `$CHK/cells_check.txt`:
```
scripts/run_h2snn_wandb_tuning.py --config configs/experiments/acm_h2snn_v3b.yaml --exp_name h2snn_check_acm --num_runs 1 --base_seed 0 --n_trials 1 --dry_run --output_dir $CHK/experiments/h2snn --state_dir $CHK/state/h2snn_check_acm --wandb_project h2snn_h2snn_t4_check
qsub -l gpu_1=1 -l h_rt=0:03:00 -t 1-3 -o $CHK/logs scripts/tsubame/sweep_slots.sh $CHK/cells_check.txt $CHK 1 3
python scripts/tsubame/status.py --out $CHK --list $CHK/cells_check.txt          # 3 lines: 1/1, done
```
If 3 minutes are too short, use `-g $(cat ~/.t4_group) -l h_rt=0:20:00`; it is still a check.

**4. Three lists, three chains, in the author's order.**
```
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
python scripts/tsubame/make_cell_list.py --phase h2snn           --out $OUT/cells_tuning_h2snn.txt   # 24 lines
python scripts/tsubame/make_cell_list.py --phase baselines-early --out $OUT/cells_tuning_early.txt   # 4 lines
python scripts/tsubame/make_cell_list.py --phase baselines-1     --out $OUT/cells_tuning.txt         # 85 lines
bash scripts/tsubame/submit_chain.sh --list $OUT/cells_tuning_h2snn.txt --type node_f --legs 6   # 6 tasks
bash scripts/tsubame/submit_chain.sh --list $OUT/cells_tuning_early.txt --type node_f --legs 6   # 1 task
bash scripts/tsubame/submit_chain.sh --list $OUT/cells_tuning.txt       --type node_f --legs 6   # 22 tasks
```
- Jobs running at once: 6 + 1 + 22 = **29 ≤ 30** on weekdays. Legs 2–6 are held, so only leg 1 of each chain runs at a time.
- **Submit in this order.** The H2S-NN chain is submitted first, so its jobs are the oldest in the queue. Exact start order depends on Grid Engine's scheduling, which this branch does not control.
- If a list's lines are not all `done` when its chain ends, submit the same command again. The one-chain rule allows it once the old jobs have left `qstat`.

**Watch** (no `qstat`):
```
python scripts/tsubame/status.py --out $OUT --list $OUT/cells_tuning_h2snn.txt
python scripts/tsubame/status.py --out $OUT --list $OUT/cells_tuning_early.txt
python scripts/tsubame/status.py --out $OUT --list $OUT/cells_tuning.txt
```

**Final seeds** (only lines whose sweep has 100 finished trials; rerun as more lines finish):
```
python scripts/tsubame/make_cell_list.py --final-seeds --out-dir $OUT --tuning-list $OUT/cells_tuning_h2snn.txt --out $OUT/cells_final_seeds_h2snn.txt
bash scripts/tsubame/submit_chain.sh --list $OUT/cells_final_seeds_h2snn.txt --type node_f --legs 1
python scripts/tsubame/make_cell_list.py --final-seeds --out-dir $OUT --tuning-list $OUT/cells_tuning.txt --out $OUT/cells_final_seeds.txt
bash scripts/tsubame/submit_chain.sh --list $OUT/cells_final_seeds.txt --type node_f --legs 1
```
- The early list needs no final seeds for card 7, which reads tuned configurations. If the main table wants them, run the same pair with `cells_tuning_early.txt`.
- A rewritten final-seeds list is a new chain for the same list file. Submit it after the previous final-seeds chain has left `qstat`. Lines already complete are skipped (`lines.py complete`).

**Collect** (covers all lists in `$OUT`):
```
python scripts/tsubame/collect.py --out $OUT
rsync -av <t4>:<WORK>/h2snn-out/$SHA/collected/ /mnt/data/peeradon/h2snn-results/tsubame/$SHA/
```

### Open questions (T7)
1. **H2GB H2S-NN trials longer than 24 h** (card: known risk, decision pending). Such a line ends `exceeds_24h` after two legs. `status.py` shows it, and its first trials measure the risk.
2. **Host peaks were not measured at the TD-1 model** or at the largest points, except H-Pokec. The two large-LP tasks have no record at all. The first `node_f` leg of the `h2snn` chain shows the real sums: per-slot peaks are not in the provenance, but an OOM kill would show in the line log.
3. **Disk.** Every H2S-NN trial keeps its own run directory with a checkpoint. 24 cells × 100 trials share the 125 GiB home+work quota. Nothing was measured here. Check `t4-user-info disk home` after the first leg.

## Last line
A commit cannot contain its own SHA. The head as pushed (this report commit) is in the review message:
`branch=tsubame sha=<this commit> card=docs/H2S-NN_tsubame_card.md base=365a75763f45dee4b8853cc85ae06f0084295c47`
(T0–T5 were reported at base `9b7ae0b66de790a96ad1b6ad6d4b83caa7a4009c`.)
