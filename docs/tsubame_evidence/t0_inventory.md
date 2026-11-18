# T0 — inventory of one baseline tuning line (no code)

Base: `9b7ae0b` (v3b, card 1e F3 + card 1f U6) merged with design-docs `709e3eb` (already an ancestor: the merge was "Already up to date").
Paths are relative to `H2S-NN/`. Every line runs with `H2S-NN/` as its working directory, as `run_sweep_agents.sbatch` does.

The example line (phase 1, card 2 D5 form; `<out>` = `$WORK/h2snn-out/<sha>`):

```
scripts/run_baseline_tuning.py --task node_classification --models gcn --datasets acm --num_runs 1 --base_seed 0 \
  --n_trials 100 --output_dir <out>/experiments/baselines --state_dir <out>/state --wandb_project h2snn_baseline_t4 --resume
```

`<token>` = `Cell.token` = `<model>_<task>_<dataset>` (`src/baselines/campaign.py:56`), here `gcn_node_classification_acm`.
`--num_runs 1 --base_seed 0` gives one unit, `run0`, seed 0 (`build_units`, `campaign.py:60`). `--resume` is accepted on a first run: with no state, `resolve_campaign_state` returns `"new"` and the sweep is created (`wandb_campaign.py:302-303`, `600-629`).

## 1. Paths the launcher writes

| Path | Writer (file:line) | When | Content |
|---|---|---|---|
| `<out>/state/<token>/tuning_state_run0.json` | `save_tuning_state` (`wandb_campaign.py:200`), path from `state_path_for` (`campaign.py:340`) | created with the sweep (`:629`); rewritten before the trial loop (`:633`), by the worker after each successful trial (`:448`), and by the parent after each attempt (`:721`) | **sweep state**: `state_version` 5, fingerprints, `sweep_id`, `target_trials`, `attempted_trials`, `successful_trials`, `failed_trials`, `pruned_trials`, `records` (one dict per successful trial: metrics, `sweep_params`, `trial_name`, `wandb_run_id`, `wandb_sweep_id`, …), `failures`, `pruned`, `failed_by_condition` |
| `<out>/state/<token>/tuning_state_run0.json.tmp` | `save_tuning_state` (`:202-205`) | transient (atomic replace) | — |
| `<out>/state/<token>/tuning_state_run0_continuationN.json` | `next_continuation_state_path` (`:261`) | only when a state with other fingerprints already exists (never in a fresh `<out>`) | as the state |
| `<out>/state/<token>/tuning_state_run0_worker_context.json` | parent (`:641`, `:668`) | before every attempt (overwritten) | the worker's context (base config, attempt index, sweep id, W&B project/tags) |
| `<out>/state/<token>/tuning_state_run0_worker_status_<k>.json` | worker (`write_worker_status`, `:486-523`) | end of attempt `k` (a re-run attempt `k` overwrites it) | `ok`, `pruned`, `worker_pid`, `error`, `failure` |
| `<out>/experiments/baselines/<token>/tuning_trials_run0.json` | `run_unit_sweep` (`:730-733`) | after the trial loop (every launcher run, also when nothing was left to run) | the list of trial records (= `state["records"]`) |
| `<out>/experiments/baselines/<token>/tuning_failures_run0.json` | `:749-750` | after the trial loop | failed-trial summary |
| `<out>/experiments/baselines/<token>/best_tuned_run0.json` | `summarize_unit` → `write_run_record` (`run_baseline_tuning.py:185-188`) | after the trial loop | the selected trial (`select_best_trial` = max `sweep_objective`) |
| `<out>/experiments/baselines/<token>/tuning_summary.json` | `summarize_cell` (`run_baseline_tuning.py:337-339`, `campaign.py:294`) | after the unit | cell summary + unit record |
| `<out>/experiments/baselines/<token>/final_seed<s>.json` | `run_final_seeds` (`run_baseline_tuning.py:230`) | `--final_seeds` lines only | one record per seed (`final_seed`, `sweep_params`, `selected_trial`, `selected_from`, `monitor`, metrics) |
| `<out>/experiments/baselines/<token>/final_seeds_summary.json` | `run_baseline_tuning.py:232` | `--final_seeds` lines only | summary over seeds |
| `./wandb/run-*/`, `./wandb/sweep-*/` (relative to the working directory `H2S-NN/`) | the `wandb` library (`wandb.init`, `wandb.agent`) | every trial | W&B local run files; `WANDB_DIR` is not set by our code |
| stdout / stderr | launcher, worker (`subprocess.run` inherits them) | always | goes to the job log (T1: one log file per slot) |

No adapter writes a checkpoint or another file during `fit_cell`: a search of `src/baselines/` for `checkpoint`, `torch.save`, `tempfile` and file writes finds only `summarize_cell` (`campaign.py:335`). The H2S-NN checkpointer is not on this path.

Phase 1 needs no file outside `<out>` and `./wandb/`: the cell list and the job logs are written by the T1 scripts (`<out>/logs/`, `<out>/submitted.tsv`).

## 2. What the launcher reads

| Item | Where it is read | Note for TSUBAME |
|---|---|---|
| conda | the job script: `. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b` (`run_sweep_agents.sbatch:28`) | `$HOME/miniforge3` (guide §2.3); `-V` is not used (guide §5) |
| data root | `DataConfig.root` default `Path("data")` (`src/configs/config.py:794`, `:922`), resolved against the working directory (`src/baselines/bundles.py:403`) | `H2S-NN/data` → the symlink to `$WORK/h2snn-data` (guide §3); the working directory must be `H2S-NN/` |
| vendor code | `VENDOR_DIR = <H2S-NN>/vendor` (`src/baselines/runtime/vendor_paths.py:23-25`) | the checkout's own `vendor/` (clone + patches, guide §3) |
| W&B credentials | the `wandb` library: `~/.netrc` (written by `wandb login`) or `WANDB_API_KEY`; settings in `~/.config/wandb/` | `~/.netrc` on the shared home (guide §2.5); reachability from compute nodes is UNVERIFIED (T6) |
| W&B project / entity / mode / tags | CLI (`--wandb_project`, `--wandb_entity` default `""` → the key's default entity, `--wandb_mode` default `online`) | `h2snn_baseline_t4` (T3) |
| device | `--device` (unset) → CUDA if available (`campaign.py:124-132`) | the slot's `CUDA_VISIBLE_DEVICES` |
| environment variables | none: `src/` and the launcher read no environment variable (no `os.environ` / `getenv` in `src/baselines/`, `src/utils/wandb_campaign.py`, `scripts/run_baseline_tuning.py`) | `CUDA_VISIBLE_DEVICES` (set by the slot), plus whatever `wandb`, torch and DGL read themselves |
| DGL | `~/.dgl/config.json` (DGL's own backend config, written by DGL on first import) | shared home |

## 3. Crashed trials and `--resume` (the premise of T2)

- The parent counts an attempt only **after** its worker process returns. `attempted_trials` is read once before the loop (`wandb_campaign.py:631`) and the loop starts at that index (`:634-637`). After the worker returns, the parent writes `attempted_trials = attempt_index + 1` (`:688`) and saves the state (`:721`).
- **The worker dies, the parent survives** (an exception, a CUDA fault, a non-zero exit): there is no new record, so the attempt is a failed trial. It is appended to `failures` with the worker's status (`:692-717`), it counts as attempted, and it is **not** re-run. A Hyperband prune is told apart by the status file's `pruned` (`:703`).
- **The job is killed** (SLURM time limit or `scancel`; on TSUBAME the 24 h limit or `qdel`): the parent dies with the worker, so `:688` is never reached. On the next `--resume` the same `attempt_index` is started again (`:631-637`). This is a **new W&B run** of the same sweep: the controller hands out the next suggestion, and the killed run stays "crashed" in W&B. This is what `run_sweep_agents.sbatch:14-16` says ("its attempt is re-run on resume"), so T0's stop condition does not apply.
- Detection: a crashed attempt is visible only as `attempted_trials` not having advanced past it. There is no marker of it in the state today. A job killed in every leg therefore re-runs the same attempt forever, which is the gap T2 closes.
- **Edge case** (recorded, not changed): if the job is killed between the worker's save (`:448`) and the parent's save (`:721`), the record is kept and the attempt is still re-run. That line then ends with one more successful record than attempts it counts.

## 4. What T2 and T4 read from this

- T2 keeps its counter in the sweep state (`tuning_state_run0.json`). No new file.
- T4 `status.py` reads only:
  - `<out>/state/*/tuning_state_*.json` (excluding `_worker_*`), for finished attempts, records and the guard;
  - `<out>/experiments/baselines/*/final_seed*.json`;
  - the modification time of `<out>/logs/*` for the last log time.
- T4 `collect.py` reads the state's `records` (trials) and `final_seed<s>.json` (final seeds). It also reads the provenance that the slot script writes per line run into `<out>/logs/` (T1: v3b SHA, host, job id, task id, GPU name, start/end time).
