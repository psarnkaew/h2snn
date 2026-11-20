# Card 10 report: lab-cluster runs (SLURM): stopped at L2

**Verdict.** L1 is done: job 18033 ran 4 tests, all OK, and the mutation was caught. **L2 hit a stop condition:** at base `e5bdf10`, neither launcher can run one fixed grid point without a new flag. L3 and L4 were not started, because the prompt says to stop at any stop condition. No `src/` change was made: `git diff --stat e5bdf10 -- H2S-NN/src` is empty. No real-data run was made.

## Units

| Unit | State | Files | Evidence |
|---|---|---|---|
| L1 lab slot runner | done (`cd0de1a`) | `scripts/slurm/lab_slots.sbatch`, `scripts/slurm/submit_lab.sh`, `tests/test_lab_slots.py` | `card10_evidence/l1_18033.log`: 4 tests OK; mutation `mutation_l1.py` (the GPU-type record is not checked) caught |
| L2 fit check | **stopped** | none | see below |
| L3 site split | not started | none | none |
| L4 targeted tests and report | report only | this file | none |

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

**Options**

| Option | What | Recommendation |
|---|---|---|
| A | Add `--grid_point max` to both launchers: largest value of every size axis, central value elsewhere, only with `--dry_run --n_trials 1`, and print the peak GPU memory (`torch.cuda.max_memory_allocated`). For baselines, extend card 8's `--grid_point` after card 8 merges into `v3b`. Then L2 runs `fit_check.py` on the lines with that flag. | **Recommended**: the launcher defines and runs the point, so nothing is guessed, and it reuses card 8's mechanism. Cost: one unit, after card 8 merges. |
| B | `fit_check.py` calls the launchers' `train_trial` in-process with the largest point. | Not recommended: it duplicates `main()`'s base-config setup (dry-run budget, device, batch bundle), which can drift from the real run path. |

**L3** (`site_split.py`) does not depend on how L2 runs, only on the `fit_check.tsv` columns. It can be done on your go, together with L2 or before it. Planned behaviour, open to your correction:
- the join is by cell token, so a list generated with either site's `--out-dir` splits the same way;
- `error` lines and lines missing from `fit_check.tsv` go to TSUBAME;
- a line with state under one site's `state/<token>` or `experiments/*/<token>` stays at that site;
- a line pinned to the lab that does not `fits` is an error.

## How to run L1 (after an approved `v3b` SHA; not run in this card)
```
SHA=<approved v3b sha>; RUN=/mnt/data/peeradon/h2snn-runs/$SHA; OUT=/mnt/data/peeradon/h2snn-out-lab/$SHA
git clone git@github.com:psarnkaew/h2snn.git $RUN && cd $RUN && git checkout --detach $SHA
cd H2S-NN && bash scripts/setup/clone_vendors.sh && bash scripts/setup/apply_vendor_patches.sh
ln -s /mnt/data/peeradon/h2snn/H2S-NN/data data            # lab data root; vendor/ stays a real clone
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
python scripts/tsubame/make_cell_list.py --phase h2snn --out $OUT/cells_tuning_h2snn.txt   # then L3's .lab.txt
bash scripts/slurm/submit_lab.sh --list $OUT/cells_tuning_h2snn.lab.txt --out $OUT --gpu-type 3090 --mem 48G --time 72:00:00 --legs 3
bash scripts/slurm/submit_lab.sh --list $OUT/cells_tuning.lab.txt       --out $OUT --gpu-type titan_rtx --mem 48G --time 72:00:00 --legs 2
python scripts/tsubame/status.py --out $OUT --list $OUT/cells_tuning_h2snn.lab.txt
python scripts/tsubame/collect.py --out $OUT
```
The `--mem` values above are placeholders: lab nodes have 64 GB RAM, and the A6000 node has 224 GB. The GPU type per list is the one its fit check used.

branch=card10 sha=5d72f21b25cd889fe537d546a933e87f1e847a96 card=docs/H2S-NN_card10_lab_runs_card.md base=e5bdf10432e32c8fe31e8f3faff814951461d4d2
