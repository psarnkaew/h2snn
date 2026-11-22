# Card 16 report: part 2, epoch budget of tuned cells (S12, 2026-11-22)

Card: `docs/H2S-NN_card16_card.md`, section "Part 2". Part 1 (binary F1) was withdrawn and is not implemented.

Branch `card16` starts at `merge-c12-c14` bab7b54. It merges the latest design-docs (2189fe4) in 93f1f01. The code is in 15e06cf.

## Files changed

**Budget table (`src/configs/config.py`)**
- `PROTOCOL_BUDGETS`:
  - HGB NC 300/30 → **500/40**
  - LP 40/40 → **500/40**
  - H2GB NC 500/None → **500/40**
  - GC (hetsheaf) is unchanged.
- The table's comment says the same.
- No other entry in `config.py` changed (card 15 also edits this file).

**`src/baselines/budget.py`**
- It already read the table. Only the comment changed.

**H2GB NC loops** (`src/baselines/minibatch.py`, `src/baselines/adapters/hetero2net.py`)
- `MinibatchPlan` has a new field `patience`. `minibatch_plan` takes it from the table (40).
- The new function `stops_early(epoch, best_epoch, plan)` is checked after each evaluation. It stops the run when `epoch − best_epoch ≥ patience`, so patience counts epochs:
  - at `eval_period` 4, patience 40 is 10 evaluations without improvement;
  - at OAG's `eval_period` 2, it is 20.
- It is used in the three H2GB loops: `train_minibatch_node_classification`, `train_index_minibatch` and `train_h2gb_epochs`.
- One irregular case: H2GB's evaluation rule also evaluates epoch 0. If epoch 0 is the best evaluation, the run stops after 11 evaluations, still 40 epochs after the best one.

**Published-configuration cells** (`src/baselines/h2gb_published.py`)
- `published_plan` sets `patience=None`, which keeps H2GB's own rule (no early stopping).
- Published HGB and LP cells already keep their source's `epochs`/`early_stopping` through `resolve_budget`. That path is unchanged.

**Stopped epoch on the record**
- Baselines: `campaign.fit_cell` wraps the run's report in `StoppedEpoch` (`src/baselines/progress.py`). The wrapper forwards every call exactly as before and keeps the last epoch a loop reported. The record gets `stopped_epoch`.
  - Every shared loop reports at each of its evaluations and stops only after one, so the last reported epoch is the epoch the run stopped at.
  - An adapter with its own loop that never calls `report_epoch` gets no `stopped_epoch`.
- H2S-NN: `scripts/run_h2snn_wandb_tuning.py` keeps the last epoch passed to the epoch callback. Tuning trials and final seeds get `stopped_epoch`.

**H2S-NN configs**
- No H2S-NN config sets `early_stopping_patience`.
- The 7 LP configs that pinned `epochs: 40` now have `epochs: 500`, with patience 40 from the table:
  - `dblp_ap`, `freebase_bb`, `h-pokec_uu`, `imdb_ma`, `lastfm`, `ogbn-mag_pp`, `pubmed`.
- `acm_pa_h2snn_v3b.yaml` is left as is. The task is dropped and card 15 replaces it with acm_pp.
- The H2GB NC configs set neither value, so they follow the table.
- The H2S-NN mini-batch trainer already counts patience in epochs (`bad_epochs += eval_period`), so it is unchanged.

**Tests**
- `tests/test_shared_protocol.py`, `tests/test_budget_table.py`, `tests/test_dry_run_flag.py` and `tests/test_partMBfix_flag.py` now pin the new values.
- New: `tests/test_card16_epoch_budget.py`.

Not changed: `record_budget` in `fit_cell` resolves the budget without the dataset. H2GB NC and HGB NC now have the same budget, so the stamped `budget_epochs`/`budget_patience` are correct.

## Tests

Only the budget tests and the new patience test were run. The full suite runs once on the final combined branch.

| Job | Commit | Node, GPU | Peak RSS | Result |
|---|---|---|---|---|
| 18290 | 93f1f01 + uncommitted edits (the same as 15e06cf) | g3n01, 3090, `--mem` 8G | 0.50 GiB | 24 tests OK |
| 18291 | 15e06cf | g3n01, 3090, `--mem` 1G (peak + 30 %) | 0.49 GiB | **24 tests OK** |

Targets: `tests.test_shared_protocol`, `tests.test_budget_table`, `tests.test_card16_epoch_budget`, `tests.test_dry_run_flag`, `tests.test_partMBfix_flag`. W&B tests ran with `WANDB_MODE` unset.

What the new test checks:
- `minibatch_plan("pdns", 256).patience == 40`.
- `eval_period` 4, patience 40, best evaluation at epoch 3: the run stops after 10 evaluations without improvement (epochs 7–43). `stopped_epoch` is 43.
- `eval_period` 2: 20 evaluations.
- Patience None (a published H2GB cell): the run goes to epoch 499.

branch=card16 sha=15e06cf2bace2ab395b7838bdfe93c6af68cb108 card=docs/H2S-NN_card16_card.md base=bab7b54d1e52b9145a80ee23ce462831016740ed
