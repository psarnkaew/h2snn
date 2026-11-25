# Card 23 — H2S-NN input-feature dropout (new tuning axis)

Written 2026-11-26 by the model design session. The author asked for it on 2026-11-26 ("add input dropout and see whether ACM improves"). Evidence: `docs/H2S-NN_gap_analysis_report.md` §1.3, §10, §12 and `docs/handoff_model_gap_2026-11-25.md` M1 (design-docs `af0ed76`). Report: `docs/H2S-NN_card23_report.md`.

This is a **deviation** adopted after test results were seen. Spec 1.33 records it, and the paper's appendix reports it.

## Why
- On ACM NC the one-layer type encoder memorises the 726 training labels by about epoch 10. The selected checkpoint then sits at τ ≈ 1, before the diffusion has trained (gap report §1.3).
- Five baselines tune input-feature dropout over `search_grids.INPUT_DROP` (uniform 0–0.9): PSHGCN (published 0.85 on ACM), SeHGNN, H2SGNN, HetSheaf and NSD. H2S-NN has no such axis.
- Adding it over the same range brings H2S-NN's search space in line with those baselines.
- The value is chosen by the usual Bayesian search on validation, never by test.

## Specification
1. **Model.**
   - Add `ModelConfig.input_dropout: float = 0.0` (`src/configs/config.py`; parsed like `encoder_dropout`, range [0, 1)).
   - Pass it to `TypeSpecificNodeEncoder` (`src/models/h2snn_model.py:430`).
   - In `TypeSpecificNodeEncoder.forward` (`src/models/node_encoder.py`), apply `nn.functional.dropout(x_t, p=input_dropout, training=self.training)` to each featured type's raw features `x[mask][:, :type_dim]`, before its `Linear`.
     - Featureless types (learned embeddings) and the all-featureless identity branch are not touched.
     - When `input_dropout == 0` the dropout call is **skipped**, not called with p = 0. The code path, and the torch RNG it consumes, are then identical to `b0f13fb`.
   - The WISP draws use their own counter-based stream (spec 1.30 AD2), so a nonzero `input_dropout` does not change them.
2. **Tuning axis.**
   - In `scripts/run_h2snn_wandb_tuning.py`, add `input_dropout`, uniform 0–0.9 (the same values as `search_grids.INPUT_DROP`).
   - Add it under the same condition as `encoder_dropout` (datasets not in `FEATURELESS_DATASETS`).
   - Validate it in `validate_trial_params` and apply it in the payload like `encoder_dropout`.
   - Also add it to `scripts/slurm/grid_points.py` if that file lists the dropout keys.
3. **Fingerprints.** A config without `input_dropout`, or with `input_dropout: 0.0`, must give the same sweep fingerprint and run identity as on `b0f13fb` (`src/utils/wandb_campaign.py:582-655`). Existing H2S-NN sweeps then resume unchanged. A sweep whose space contains the new axis gets a new fingerprint, which is intended.

## Tests (only these; CPU, one sbatch job)
1. `input_dropout = 0`: the existing golden test (`tests/test_v3b_golden.py`) passes unchanged, with no new exclusion.
2. One new test in `tests/test_node_encoder.py` (or a new file `tests/test_card23_input_dropout.py`):
   - (a) p = 0.5 in training mode zeroes about half of the raw feature entries before the `Linear`, and leaves featureless types' embeddings untouched;
   - (b) in eval mode the output is unchanged by p;
   - (c) with p = 0, `torch.get_rng_state()` after a forward equals the state before it (no RNG consumed).
3. Tuning space:
   - ACM's sweep space contains `input_dropout` with min 0 and max 0.9;
   - Freebase's and LastFM's sweep spaces do not contain it;
   - the fingerprint of an ACM config without the key equals the `b0f13fb` fingerprint.
- No GPU smoke and no full suite. Ops runs the full suite at review.

## Preconditions
- Worktree `/mnt/data/peeradon/h2snn-card23`, branch `card23` from `origin/v3b` `b0f13fb43bcc77bdf15f73a2caa026140179e441` (cards 21 and 22 merged); `git merge` the design-docs SHA given in the prompt.
- Copy (do not symlink) the git-ignored `vendor/`; `data` → `/mnt/data/peeradon/h2snn/H2S-NN/data`.

## Rules
- `git diff origin/v3b...card23 -- H2S-NN/src H2S-NN/scripts` touches only:
  - `src/configs/config.py`, `src/models/node_encoder.py`, `src/models/h2snn_model.py`;
  - `scripts/run_h2snn_wandb_tuning.py`, and `scripts/slurm/grid_points.py` if needed.
- No other model, loss, schedule or search-space change.
- `sbatch` only, one CPU job (`--mem 16G --time 1:00:00`). No tuning and no performance runs: the sweeps below are ops' after the merge.
- Stage by explicit path. Make one commit for code + tests and one for the report, then push `card23`. Never merge into `v3b`; never stash, reset or rebase.
- Stop and report if the golden test changes with `input_dropout = 0`, or if the fingerprint of existing configs changes.

## After the merge (ops; author decision of 2026-11-26)
- **Rule fixed before any result.** The new space (old space + `input_dropout`) replaces the old one on the HGB NC cells ACM, DBLP and IMDB, whatever ACM shows.
- Runs on each cell:
  - a new 100-trial sweep with the usual Bayesian search, tuning seed 0, selection by validation Macro-F1;
  - then the best configuration on seeds 0–2.
- Order: ACM first (≈ 3–4 GPU-h on the lab), then DBLP and IMDB.
- Freebase has no node features (`FEATURELESS_DATASETS`), so it has no such axis and keeps its runs.
- The cell's number is the new one even if it is lower. The old sweep stays in the record as the pre-deviation run.

## Report (`docs/H2S-NN_card23_report.md`, short)
- The diff.
- The test results with the job id.
- The fingerprint check.
- Last line: `branch=card23 sha=<head> card=docs/H2S-NN_card23_card.md base=b0f13fb43bcc77bdf15f73a2caa026140179e441`.

## Estimate
About 1–2 h including the queue.
