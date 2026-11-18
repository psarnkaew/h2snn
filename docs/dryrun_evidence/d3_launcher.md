# Card 2 D3 — tuning launcher end to end, resume, seeds, read-out, evaluation draw

Job 16741 (g2n01, RTX 2080 Ti 11 GB), steps 98–104; W&B project `h2snn-dryrun`. Scripts: `d3_job.sh` (launchers,
seeds, read-out, draw check), `d3_kill_resume.sh` (SIGKILL an agent mid-trial, then resume), `d3_eval_draw_check.py`.
Logs with the trainers' console lines stay in `/mnt/data/peeradon/dryrun_scratch/d3` (outside the repository). No metric
value is recorded here.

## Changes (H2S-NN `scripts/` only; no `src/` change, so the v3b_golden / mb0 compares are unaffected)

| file | change |
|---|---|
| `scripts/run_h2snn_wandb_tuning.py` | `--dry_run`: epochs = 1 (refused with any other `--epochs`), W&B tag `dry-run`; `dry_run` and `epochs` refused as sweep parameters (`DRY_RUN_KEYS`). **Fix:** `_epoch_callback` returns when there is no W&B run (`--final_seeds` raised in `wandb.log`) |
| `scripts/run_h2snn_runs.py` | `--dry_run`: `cfg.epochs = 1`, tag `dry-run` |
| `scripts/run_baseline_tuning.py` | `--dry_run` as above. **Fix:** `--task` no longer required by argparse (the sweep worker is spawned with `--worker_context` only and exited before any trial); still required without a worker context. **Fix:** `summarize_unit(records, monitor, path)` (best trial by the monitor) replaces the lambda that passed the monitor as `summarize_cell`'s task ("Unsupported task_type 'macro_f1'") |
| `scripts/slurm/run_sweep_agents.sbatch` | new: job array, one launcher line per task, `--resume`, 72 h, 60G, one GPU |
| `tests/test_dry_run_flag.py` | new: 3 flag tests + one synthetic test per fix (3); each fix test fails with its bug restored (job 16741 steps 99, 101) — `d3_tests.txt` |

## Results (exit codes and file inventory only)

| check | result |
|---|---|
| H2S-NN tuning trial, ACM / acm_pa / MUTAG (`--n_trials 1 --dry_run --resume`) | OK; trial dir has `config.yaml` and `checkpoints/best.pt`; `best_tuned_run0.json` written |
| resume of the same H2S-NN command | finds the fingerprinted state, same sweep, nothing left to run (exit 0) |
| baseline tuning GCN on ACM; resume with `--n_trials 2` | after the fixes: OK; resume adds trials to the same sweep (`d3_gcn_resume.txt`) |
| agent SIGKILLed mid-trial, then resumed (`d3_kill_resume.sh`, sweep `jwxnvcqo`) | 5/5 trials finished, 0 failed |
| `run_h2snn_runs.py` on the trial config; `--final_seeds 0` | OK (after the fix) |
| `schema_readout.py` on the ACM trial | OK |
| evaluation draw (spec §6): train → reload in a fresh process | `reload_equal` true; trial's own run dir rebuilds the same cells (`trial_reload_cells_equal` true); negative control (seed + 1) detected; 4 groups, 19,606 cells |

## Commands for the real campaign (Monday)

Cells ready = the OK rows of `cells.md` (D2). One launcher line per cell in a list file, then one agent per GPU:

```
cd /mnt/data/peeradon/h2snn/H2S-NN
# H2S-NN, one line per <ds>_h2snn_v3b.yaml that ran OK in D2
scripts/run_h2snn_wandb_tuning.py --config configs/experiments/<ds>_h2snn_v3b.yaml --output_dir experiments/tuning --wandb_project h2snn
# baselines, one line per (task, model, dataset) that ran OK in D2
scripts/run_baseline_tuning.py --task <task> --models <model> --datasets <ds> --num_runs 1 --output_dir experiments/baselines --wandb_project h2snn_baseline

sbatch --array=1-<lines>%<gpus> scripts/slurm/run_sweep_agents.sbatch <list file>     # first run and every resume
# then, per cell: scripts/run_h2snn_wandb_tuning.py --config ... --final_seeds 0 1 2 3 4 (or run_h2snn_runs.py on best_tuned)
```

Open for the author: the H2S-NN sweep has no `batch_size` axis (D1 finding), so H2GB cells tune at the YAML's B only;
the OOM cells of D2 need a placement decision (A6000 / smaller point) before they enter a list.
