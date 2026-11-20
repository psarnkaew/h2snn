# Card 10 report: lab-cluster runs (SLURM): L1, L3, A2.1, A2.3 done; L2 (with A2.2) and L4 wait for card 8 on `v3b`

**Verdict.** L1 is done: job 18033 ran 4 tests, all OK, and the mutation was caught. L1 was accepted with its three additions (Amendment 1). L3 is done: job 18035 ran 3 tests, all OK, and the mutation was caught; its choices were accepted (Amendment 2). Amendment 2 adds two units, both done: **A2.1** (job 18040: 6 tests OK, mutation caught) and **A2.3** (job 18041: 1 test OK, mutation caught). **L2 (with A2.2, the 75 % headroom) and L4 wait**: once card 8 is on `v3b` and the ops session gives the SHA, I merge it and do them. No `src/` change was made: `git diff --stat e5bdf10 -- H2S-NN/src` is empty. No real-data run was made.

## Units

| Unit | State | Files | Evidence |
|---|---|---|---|
| L1 lab slot runner | done (`cd0de1a`) | `scripts/slurm/lab_slots.sbatch`, `scripts/slurm/submit_lab.sh`, `tests/test_lab_slots.py` | `card10_evidence/l1_18033.log`: 4 tests OK; mutation `mutation_l1.py` (the GPU-type record is not checked) caught |
| L2 fit check | **waits for card 8 on `v3b`** (option A) | none yet | see below |
| L3 site split | done (`9ead523`) | `scripts/slurm/site_split.py`, `tests/test_site_split.py` | `card10_evidence/l3_18035.log`: 3 tests OK; mutation `mutation_l3.py` (a lab line also written to the TSUBAME list) caught |
| A2.1 whole-H100 TSUBAME types | done (`8a8d524`) | `scripts/tsubame/submit_chain.sh` (types), `scripts/tsubame/sweep_slots.sh` (header only), `tests/test_tsubame_types.py` | `card10_evidence/a21_18040.log`: 6 tests OK (new test + unchanged T5 suite); mutation `mutation_a21.py` (`node_h` S = 4) caught |
| A2.2 75 % headroom | with L2 | none yet | none |
| A2.3 failure report | done (`ca119cd`) | `scripts/tsubame/failures.py`, `tests/test_failures_report.py` | `card10_evidence/a23_18041.log`: 1 test OK; mutation `mutation_a23.py` (a failure skipped) caught |
| L4 targeted tests and report | after L2 | this file | none |

## L1: what it does
- **`lab_slots.sbatch`**: array task `t` runs runnable line `t`, numbered as `lines.runnable_lines`, with `python <line> --resume`. A `--final_seeds` line whose seed files are all written is skipped. The output names are those of `sweep_slots.sh` with `S = 1`:
  - line logs: `logs/lines/<list>.line<NNN>.<job>.<task>.0.log`
  - provenance: `logs/prov/<job>.<task>.0.jsonl`, where `<job>` is `SLURM_ARRAY_JOB_ID`
  - start events also carry `"site":"lab","gpu_type":…` (`collect.py` ignores extra keys)
  - the test runs `collect.iter_rows` on the outputs and checks that every record gets the job, task, host, GPU name, SHA and wall time.
- **`lab_slots.sbatch` refuses**: a dirty checkout; a HEAD that differs from the directory SHA; a GPU whose `nvidia-smi` name does not match `--gpu-type`.
- **`submit_lab.sh`**:
  - `--gpu-type`, `--mem` and `--time` are required; `--time` must be ≤ 72 h.
  - Legs 2..L are submitted with `--dependency=afterany:<previous>`.
  - `submitted.tsv` has the same columns as `submit_chain.sh`.
  - A list that still has a pending or running job (one `squeue` call) is refused.
- **Condition 1 (one cell = one GPU type)**: each line's cell is recorded in `<out>/lab_gpu_types.tsv` with its GPU type. A list containing a cell already recorded under another GPU type is refused, and nothing is submitted.

**Deviations (please confirm):**
1. The record is keyed by **cell** (the `lines.py` token, i.e. the state and output directory), not by line text. This way a cell's tuning line and its final-seeds line share one GPU type, which is what condition 1 requires; keyed by line text, they would not. A line that `lines.py` cannot parse is refused.
2. The GPU-name check in the job is added (not in the card).
3. `--cpus` defaults to 4, as in `run_sweep_agents.sbatch`. `--max-parallel P` is optional and adds `%P` to the array.
4. As in `sweep_slots.sh`, the job exits 0 whatever the line's exit code; the exit code is in the end event and the line log.

## L2 stop: a launcher cannot run a fixed largest point
- **H2S-NN** (`scripts/run_h2snn_wandb_tuning.py`): the size axes are `d`, `wisp_max_length` (ℓ_max), `wisp_walks_per_anchor` (r), and `batch_size` on H2GB NC (`build_sweep_config`, l.321). Only `d` can be pinned (`--d_values`, l.133). The trial parameters always come from the W&B Bayes agent; the `--worker_context` worker also takes them from `wandb.agent` (`src/utils/wandb_campaign.py` `run_campaign_worker`).
- **Baselines** (`scripts/run_baseline_tuning.py`, l.64–99): at `e5bdf10` there is no flag that pins a point. Card 8's branch adds `--grid_point min` (probe lines, `--dry_run --n_trials 1`), which is not in the base and is min only. `run_baseline_tuning.py` is also being edited by card 8 (uncommitted in its tree).

**Decision (author, Amendment 1):** option A. The launchers define and run the largest point; `fit_check.py` only builds those lines, submits them with `submit_lab.sh` and writes `fit_check.tsv`.

## L3: what it does
`site_split.py --list L --fit-check F --gpu-type T --lab-out D1 --t4-out D2` writes `<list stem>.lab.txt` and `<list stem>.t4.txt` next to the list. The list itself is not changed.
- **Lists:** the two lists are disjoint, together they are exactly the runnable lines of the input, and each keeps the input order. The test checks this with lines from the real `make_cell_list.py` builders: baselines and H2S-NN, tuning and final seeds.
- **Join:** `fit_check.tsv` is joined by **cell** (the `lines.py` token), so a list written with either site's `--out-dir` splits the same way (the test checks both). Only rows of `--gpu-type` count, because `submit_lab.sh` takes one GPU type per lab list.
- **Lab or TSUBAME:** a line goes to the lab only if its cell's status is `fits`. `oom`, `error`, and a cell with no row on that GPU type go to TSUBAME.
- **State:** a cell with state at one site stays at that site. State means `<out>/state/<token>/tuning_state_*.json` or `<out>/experiments/*/<token>/final_seed*.json`, the layout `collect.py` reads.
- **Refused, nothing written:** lab state for a cell that is not `fits`; state at both sites; one cell with two statuses on the same GPU type.
- **Both site directories must exist.** On the lab, pass a local mirror of TSUBAME's `state/` and `experiments/`, e.g. `rsync -a --include='*/' --include='tuning_state_*.json' --include='final_seed*.json' --exclude='*' <t4>:<out>/ <mirror>/`. This way missing state is never taken for no state.
- **`fit_check.tsv` columns (the format L2 will write):** `line_no, cell, gpu_type, peak_mem_mib, status, line`, where status is `fits`, `oom` or `error`.

## A2.1: whole-H100 types on TSUBAME
`submit_chain.sh --type` accepts every resource type that gives each trial a whole H100. The slots per task, S, are the GPUs per task (handbook §5.1.1):

| Type | S |
|---|---|
| `gpu_1` | 1 |
| `node_q` | 1 |
| `node_h` | 2 |
| `node_f` | 4 |

- `node_o` and `gpu_h` (MIG, half an H100) are refused with a message, and nothing is submitted.
- Unchanged: the slot formula, the one-chain-per-list rule and `submitted.tsv`. `sweep_slots.sh` already takes S as an argument; only its header comment changed.
- **Test:** for each type and lists of 1, 3 and 5 lines, the test runs the array task by task as `qsub` was told. Every line runs exactly once, and each slot's `CUDA_VISIBLE_DEVICES` is a GPU the task has (index < GPUs per task), one line per (task, GPU).

## A2.3: failure report
`python scripts/tsubame/failures.py --out <out> --list <list>` works on both lab and TSUBAME outputs. It reads only the states' `failures[]` and the line logs, joined by provenance: attempt `a` belongs to the last run that started with at most `a` attempted trials.
- **Output:** `<out>/failures.tsv`, one row per failed trial, with columns `line_no, cell, unit, attempt, condition, named_condition, error, error_tail, job_id, task_id, host, gpu_name, log, state`. It also prints a summary per condition and per cell.
- **Condition**, checked in this order:
  1. `PCGConvergenceError`: the error type, or the named condition `T_max`.
  2. `StopConditionError`: a named condition such as `tau`, or the error type.
  3. `OOM`: "out of memory" in the error or its log lines.
  4. `other`.
- **`error_tail`:** the 5 lines of that run's log ending at this trial's error line (the traceback's last line).
- **Never moves or reruns anything.** A state shared by a cell's tuning line and its final-seeds line is read once. Lines that `lines.py` cannot parse are listed as skipped.
- **Test:** a synthetic tree with two cells (baseline and H2S-NN) and two runs (jobs 801 and 802), holding an OOM, a stop condition (`tau`), a `T_max` and an `other`. Each failure gets its own row with the right condition, job, host and GPU, and a tail that ends at its error. The state files are unchanged.

## How to run L1 (after an approved `v3b` SHA; not run in this card)
```
SHA=<approved v3b sha>; RUN=/mnt/data/peeradon/h2snn-runs/$SHA; OUT=/mnt/data/peeradon/h2snn-out-lab/$SHA
git clone git@github.com:psarnkaew/h2snn.git $RUN && cd $RUN && git checkout --detach $SHA
cd H2S-NN && bash scripts/setup/clone_vendors.sh && bash scripts/setup/apply_vendor_patches.sh
ln -s /mnt/data/peeradon/h2snn/H2S-NN/data data            # lab data root; vendor/ stays a real clone
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
python scripts/tsubame/make_cell_list.py --phase h2snn --out $OUT/cells_tuning_h2snn.txt   # then site_split.py (L3) -> .lab.txt / .t4.txt
bash scripts/slurm/submit_lab.sh --list $OUT/cells_tuning_h2snn.lab.txt --out $OUT --gpu-type 3090 --mem 48G --time 72:00:00 --legs 3
bash scripts/slurm/submit_lab.sh --list $OUT/cells_tuning.lab.txt       --out $OUT --gpu-type titan_rtx --mem 48G --time 72:00:00 --legs 2
python scripts/tsubame/status.py --out $OUT --list $OUT/cells_tuning_h2snn.lab.txt
python scripts/tsubame/collect.py --out $OUT
python scripts/tsubame/failures.py --out $OUT --list $OUT/cells_tuning_h2snn.lab.txt   # failures go to the author; nothing moves automatically
```
The `--mem` values above are placeholders: lab nodes have 64 GB RAM, and the A6000 node has 224 GB. The GPU type per list is the one its fit check used.
