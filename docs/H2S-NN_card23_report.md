# Card 23 report: H2S-NN regularisation pack

Card: `docs/H2S-NN_card23_card.md` (design-docs `7cf5b1d`, merged into `card23`). Base: `origin/v3b` = `b0f13fb`. Code commit: `fe9ec1d`.

## Diff by component (`git diff origin/v3b...card23 -- H2S-NN/src H2S-NN/scripts`: 11 files, all on the card's list)

- **C1 input dropout** (`model.input_dropout`, default 0.0, must be in [0, 1)).
  - `node_encoder.py`: `F.dropout` on each featured type's `x[mask][:, :type_dim]` before its `Linear`. At p = 0 the call is skipped.
  - `h2snn_model.py`: one plumbing line. `config.py`: the field, its validation and parsing.
- **C2 / C3** (`run_h2snn_wandb_tuning.py` only). New constants `SPACE_PACK_*`:
  - encoder dropout: uniform 0–0.99;
  - lr: log-uniform 1e-4 to 3e-2;
  - d on NC: {16, 32, 64, 128};
  - input dropout: uniform 0–0.9, under the `FEATURELESS_DATASETS` condition;
  - `ema_decay`: {0, 0.9, 0.99, 0.999}, on full-batch NC only.
  - `validate_trial_params` and `apply_sweep_overrides` know `input_dropout` (goes to model) and `ema_decay` (goes to optim). Both are checked to be in [0, 1).
  - `search_grids.py` and every baseline space are unchanged. `grid_points.py` lists `ema_decay` in `OTHER_AXES`.
- **C4 eval draws** (`data.eval_draws`, default 1).
  - `Trainer.build_eval_draws` builds draws k = 1..K−1 once in `fit`, with seed `cfg.seed + 1_000_003·k` (`config.EVAL_DRAW_SEED_STRIDE`) and the target measure p. Draw 0 is the existing evaluation draw.
  - `averaged_eval_logits` averages softmax over the K forwards (sigmoid for multi-label) and returns log p̄ (logit p̄ for multi-label). `_compute_metrics` then gives metrics and loss on the mean: CE(log p̄) = −log p̄_y.
  - One set of K forwards per epoch serves both validation and test.
- **C5 EMA** (`optim.ema_decay`, default 0).
  - θ_ema is set to θ just before the first `optimizer.step()`. After each step, θ_ema ← βθ_ema + (1−β)θ, over all parameters.
  - `evaluation_weights()` swaps θ_ema in for evaluation and for `save_if_best`, so `best.pt` holds θ_ema. `last.pt` and training stay on θ.
- **C6 smoothing** (`checkpoint_smoothing`, default 1).
  - The score is the trailing mean of the checkpoint metric over epochs max(0, e−w+1)..e.
  - `Checkpointer.save_if_best` gets an optional `score=` argument (None keeps today's behaviour).
  - Patience counts epochs without a gain in that score. With w > 1, the record also gets `val_smoothed_score`.
- **Guards**: `config.reject_full_batch_nc_options`, called first in the `__init__` of `MiniBatchNodeTrainer`, `LinkPredictionTrainer`, `LargeLinkTrainer` and `GraphClassificationTrainer`.

## Design choices where the card left room
1. **What switches the new space on.** The sweep uses the card-23 space when the base config sets any pack key to a non-off value (`config.regularisation_pack_keys_set`). The re-tuned cells' YAML carries `data.eval_draws: 8` and `checkpoint_smoothing: 5`. Without these keys, or with them at their off values, the space and its fingerprint are those of b0f13fb.
2. **Fingerprints.** New dataclass fields would otherwise change every `dataclass_to_dict` fingerprint. The tuning script's `canonical_fingerprint` (also imported by the GC repeated-CV script) now drops pack keys that sit at their off values (`config.without_default_pack_keys`) before hashing. `wandb_campaign.py` and `misc.py` are unchanged.
3. **C6 metric.** The smoothed metric is the configured `checkpoint_metric` (macro_f1 for the NC cells). The tuning objective is unchanged: the record's raw validation Macro-F1 at the selected epoch, and W&B `best/macro_f1`.
4. **C4 warm starts.** Each evaluation draw keeps its own PCG warm start: the trainer swaps the model's `_warm_start` per draw. This is speed only; the solve itself is unchanged.

## Tests (lab, CPU, g2n01, env h2snn-b, E pinned, WANDB_MODE unset)
- **Job 19004** (`card23_evidence/job19004_tests.log`): 72 tests OK. They cover `test_card23` (a–f), `test_v3b_golden` (unchanged, no new exclusion), `test_mb0_reference_compare`, `test_data`, `test_trainer`, `test_full_graph_early_stopping`, `test_search_spaces`, `test_sweep_batch_size` and `test_wandb_and_runs`.
- **mb0 cluster compare**: 126 of 136 tensors differ. This predates card 23 and is not caused by it.
  - Job 19005 (`job19005_mb0_base.log`): b0f13fb itself gives the same "136 tensors; differing: 126" against `mb0_reference_cluster.pt`. That reference (captured at e0a7ca1, 139 tensors) is stale.
  - In the same job, card23 against a fresh b0f13fb capture: **136 tensors, differing: none**.

## Fingerprint check (job 19003, `card23_evidence/fingerprints.py`, `fp_b0f13fb.jsonl` / `fp_card23.jsonl`)
- 17 shipped configs give identical base-config and sweep fingerprints on b0f13fb and card23. They cover NC, LP, GC and H2GB, ACM included: base `888ca7ac…`, sweep `c84be876…`.
- ACM with the pack gets new fingerprints (base `5800bce8…`, sweep `623bcf79…`) and the space given above.

branch=card23 sha=fe9ec1d97b8e32a9abc5eece1c3b1d90c263fda3 card=docs/H2S-NN_card23_card.md base=b0f13fb43bcc77bdf15f73a2caa026140179e441
