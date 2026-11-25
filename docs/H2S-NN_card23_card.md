# Card 23 — H2S-NN regularisation pack for the overfitting HGB NC cells (ACM first)

Written 2026-11-26 by the model design session. The author asked for it on 2026-11-26: "make ACM beat the other papers first; it overfits, find ways".
- Evidence: `docs/H2S-NN_gap_analysis_report.md` §1.3, §6, §10, §12 and `docs/handoff_model_gap_2026-11-25.md` (design-docs `af0ed76`).
- Report: `docs/H2S-NN_card23_report.md`.

These are deviations adopted after results were seen. Spec 1.33 records them, and the appendix reports them.
- Every value is chosen by the usual Bayesian search on **validation**.
- Every new option is **off by default**, and off it is bit-identical to `b0f13fb`. Existing runs and sweeps stay valid.

## Why (ACM NC, gap report §1.3, §10)
- The one-layer type encoder memorises the 726 training labels by about epoch 10. The selected checkpoint then sits at τ ≈ 1, before the diffusion has trained.
- Validation is coarse: 181 nodes, so one node is worth 0.55 points. It moves 2–3 points per epoch around the best epoch, so selection picks a validation spike.
- Stronger encoder dropout moves validation up (0.95 gives the highest, 94.97) and the best epoch to 50–67.
- On DBLP and IMDB the lr upper edge, 1e-2, binds. On IMDB, d = 128 shows a signal.

## Specification

Components C1–C3 act in every H2S-NN trainer through the model. Components C4–C6 apply to the **full-batch NC trainer** (`src/trainers/trainer.py`) only. Any other trainer raises if one of their keys is set to a non-default value.

**C1. Input-feature dropout** (`model.input_dropout`, default 0.0).
- In `TypeSpecificNodeEncoder.forward` (`src/models/node_encoder.py`), apply `F.dropout(x_t, p, training=self.training)` to each featured type's raw features `x[mask][:, :type_dim]`, before its `Linear`.
- Featureless types and the all-featureless identity branch are untouched.
- With p = 0 the call is skipped, so no torch RNG is consumed.
- Plumbing: `ModelConfig` (`src/configs/config.py`), `h2snn_model.py:430`.

**C2. Encoder dropout range up to 0.99.**
- H2S-NN's `encoder_dropout` axis becomes uniform 0–0.99, from 0–0.9.
- Use a new constant in `scripts/run_h2snn_wandb_tuning.py`. Do not change the shared `search_grids.DROPOUT` or any baseline space.
- Raise the validation bound in `validate_trial_params` to < 1.0, as now.
- `classifier_dropout` is unchanged.

**C3. lr and width edges.**
- H2S-NN lr axis: log-uniform 1e-4 to **3e-2** (was up to 1e-2).
- `d` values: add **128** for NC.
- Both are in the H2S-NN tuning script only.

**C4. Averaged evaluation draws** (`data.eval_draws`, default 1).
- With K > 1, validation and test use K evaluation draws.
  - Draw k = 0 is today's evaluation draw (seed `cfg.seed`).
  - Draw k ≥ 1 has seed `cfg.seed + 1_000_003·k`, a fixed constant, so the seeds never meet the training seeds `cfg.seed + epoch`.
- The softmax probabilities of the K forwards are averaged. Metrics and the validation loss are computed on the average.
- Build the K draws once, before training, like the existing evaluation draw.
- Re-tuned cells use **K = 8**. It is a fixed setting, not tuned.
- K = 1 is bit-identical.

**C5. EMA of the weights** (`optim.ema_decay`, default 0 = off).
- After each optimizer step: `θ_ema ← β·θ_ema + (1 − β)·θ`. All parameters are included, τ and the map MLPs too.
- Evaluation, checkpoint selection and the saved best checkpoint use θ_ema. Training continues on θ.
- θ_ema is initialised to θ at step 0.
- Tuning axis: β ∈ {0, 0.9, 0.99, 0.999}.

**C6. Smoothed checkpoint selection** (`checkpoint_smoothing`, default 1).
- With w > 1, epoch e is scored by the mean validation Macro-F1 of epochs max(0, e − w + 1)…e.
- The best epoch is the argmax of that score, and its weights are saved, the EMA weights when C5 is on. Patience counts epochs without a gain in the smoothed score.
- Test is read at the selected epoch only, as now.
- Re-tuned cells use **w = 5**. It is a fixed setting, not tuned.
- w = 1 is bit-identical.

**Tuning space of the re-tuned cells** = the old space with C2 and C3 applied, plus:
- `input_dropout` uniform 0–0.9;
- `ema_decay` ∈ {0, 0.9, 0.99, 0.999};
- fixed `eval_draws: 8` and `checkpoint_smoothing: 5`.

Add them under the same condition as `encoder_dropout` (`FEATURELESS_DATASETS` excluded) for C1. For C4–C6 the condition is full-batch NC.

**Fingerprints.** A config without the new keys, or with their defaults, gives the same sweep fingerprint and run identity as on `b0f13fb` (`src/utils/wandb_campaign.py:582-655`). Existing H2S-NN sweeps then resume unchanged. The new space gives a new fingerprint.

## Tests (only these; one CPU sbatch job)
1. Defaults:
   - `tests/test_v3b_golden.py` passes unchanged, with no new exclusion;
   - the mb0 compare tests pass.
2. One new file, `tests/test_card23.py`:
   - (a) C1: p = 0.5 in training zeroes about half of a featured type's raw entries. It leaves featureless embeddings untouched, is a no-op in eval mode, and with p = 0 leaves `torch.get_rng_state()` unchanged.
   - (b) C4: K = 1 gives the same logits as today. On a small fixture with K = 3, the probabilities equal the mean of the three single-draw forwards.
   - (c) C5: β = 0 gives the same parameters and metrics as today, step by step. With β = 0.9 after 3 steps, θ_ema equals the closed-form recursion, and the evaluation reads θ_ema.
   - (d) C6: on a synthetic validation sequence, w = 1 picks today's epoch and w = 3 picks the argmax of the trailing mean.
   - (e) Tuning space:
     - ACM's space has `input_dropout` [0, 0.9], `encoder_dropout` [0, 0.99], lr up to 3e-2, d including 128, and the `ema_decay` values;
     - Freebase's and LastFM's spaces have no `input_dropout`;
     - an ACM config without the new keys has the `b0f13fb` fingerprint.
   - (f) A minibatch or LP config with `eval_draws: 2` raises.
- No GPU smoke and no full suite. Ops runs the full suite at review.

## Preconditions
- Worktree `/mnt/data/peeradon/h2snn-card23`, branch `card23` from `origin/v3b` `b0f13fb43bcc77bdf15f73a2caa026140179e441`; `git merge` the design-docs SHA given in the prompt.
- Copy (do not symlink) the git-ignored `vendor/`; `data` → `/mnt/data/peeradon/h2snn/H2S-NN/data`.

## Rules
- `git diff origin/v3b...card23 -- H2S-NN/src H2S-NN/scripts` may touch only:
  - `src/configs/config.py`, `src/models/node_encoder.py`, `src/models/h2snn_model.py`;
  - `src/trainers/trainer.py` and the guard lines in the other trainers;
  - the checkpointer if C5/C6 need it;
  - `scripts/run_h2snn_wandb_tuning.py`, and `scripts/slurm/grid_points.py` if it lists the keys.
- No change to the method (solve, maps, WISP draw distribution), losses, baselines or their spaces.
- `sbatch` only, one CPU job (`--mem 16G --time 1:00:00`). No tuning and no performance runs.
- Stage by explicit path. Make one commit for code + tests and one for the report, then push `card23`. Never merge into `v3b`; never stash, reset or rebase.
- Stop and report if a default changes the golden test or an existing fingerprint.

## After the merge (ops; author decision of 2026-11-26)
1. **ACM NC first.**
   - A new 100-trial sweep on the space above: Bayesian search, tuning seed 0, selection by validation Macro-F1.
   - Then the best configuration on seeds 0–2. Lab, about 4–6 GPU-h.
2. **Round 2 (card 24): fact dropout and label smoothing.** Whether it runs is decided by ACM **validation**, not test.
3. **DBLP and IMDB.** The same space applies, whatever ACM shows.
   - Freebase has no node features. It gets C3–C6 only if the author asks.

## Report (`docs/H2S-NN_card23_report.md`, short)
- The diff.
- The tests with the job id.
- The fingerprint check.
- Last line: `branch=card23 sha=<head> card=docs/H2S-NN_card23_card.md base=b0f13fb43bcc77bdf15f73a2caa026140179e441`.

## Estimate
About 1 day including the queue (C4–C6 are the larger part).
