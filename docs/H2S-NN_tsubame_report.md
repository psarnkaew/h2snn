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

## Open questions
1. **Switching unstarted tasks to `gpu_1`.** `submit_chain.sh` refuses a second chain for a list while any job of the first is queued or running. Switching some tasks to `gpu_1` therefore needs the remaining `node_f` legs `qdel`ed first. Should a partial `gpu_1` chain on disjoint tasks be allowed alongside a running one? I did not allow it, because the card says one chain per list.
2. **The guard also applies to H2S-NN sweeps.** `run_unit_sweep` is shared, so `run_h2snn_wandb_tuning.py` gets it too. That is harmless for phase 1. For phase 2 (H2GB NC, H2S-NN; guide §6.6), a trial longer than 24 h stops at `exceeds_24h` after two legs. Epoch-level resume or a reservation remains the design session's decision.
3. **Trial-level wall time** needs a timing key in the records, which is a launcher change, outside T2's scope. Until then `wall_time_s` is per line run (deviation 3).
4. **T0 edge case.** If a job is killed between the worker's record save and the parent's bookkeeping, the record is kept and the attempt still runs again. That line then ends with one more successful record than attempts it counts. It was recorded, not changed.

## Last line
A commit cannot contain its own SHA. The head as pushed (this report commit) is in the review message:
`branch=tsubame sha=<this commit> card=docs/H2S-NN_tsubame_card.md base=9b7ae0b66de790a96ad1b6ad6d4b83caa7a4009c`
