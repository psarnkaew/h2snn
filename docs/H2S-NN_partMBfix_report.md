# Part MB-fix report — training order inside a pass, the ACM/DBLP analysis flag, functoriality on scopes

**Card:** `docs/H2S-NN_partMBfix_card.md` (MF1–MF4). **Decisions:** `docs/experiment_settings_decisions.md`, rows "Part MB review" of 2026-11-13 (D-MB1, D-MB2, analysis flag, D-MB3/D-MB4, tests). **Branch:** `part-mb`, on the docs commit `e62a39ec3feced70d33d5133638740c85d29a491` (`git rev-parse HEAD` checked before starting). **Evidence:** `docs/partMBfix_evidence/`.
**Rule kept:**
- correctness only: synthetic graphs, mutations each run against its target test only, the full suite once;
- no real data, no metric reported;
- seed formulas unchanged (D-MB1);
- MB0 compare after every unit.

| unit | commit | result |
|---|---|---|
| MF1 | `0a96a72` | a batch never crosses a pass boundary (D-MB2) |
| MF2 | `3eca2fb` | flag without B: B = all training targets, 1 epoch = one pass = one step, full-batch budget |
| MF3 | `89afe46` | T1 functoriality on a batch of 4 scopes, below `T1_TOL` |
| MF4 | this commit | full suite, report |

---

## MF1 — Training order inside a pass (D-MB2)

**Change (VERIFIED-CODE, `src/trainers/minibatch_trainer.py`).**
- `TrainingOrder.next(B)` returns `min(B, targets left in the current pass)`.
- It reshuffles (new seeded permutation, `pass_seed` unchanged) only when the pass is exhausted. The last batch of a pass holds the remaining targets, with no repeat.
- The step loop is unchanged: exactly `steps_per_epoch = epoch_targets / B` calls per epoch, whatever their sizes. The draw seed stays `training_draw_seed(base, epoch, batch)`.
- The loss is the mean over the batch's own targets (unchanged code; the batch is now shorter).

**Tests (VERIFIED-RUN).** New `tests/test_partMBfix_order.py` (2). With `tests.test_partMB_trainer` → **Ran 10, OK** (`mf1_tests_output.txt`):
- (i) 18 targets, B = 4: batch sizes 4,4,4,4,2,4,4,4,4,2; each pass is a permutation; no batch repeats a target.
- (ii) and (iii), synthetic ACM-shaped graph, flag with B = 8, `epoch_targets = 16` (14 training targets):
  - 2 steps per epoch over 3 epochs, so `steps_taken = 6`;
  - a spy on the loss input sees exactly the batch sizes of an independent `TrainingOrder` with the same seed, including the short batches.
- (iv) determinism per seed and a different order per pass (existing test, kept).

**Changed assertion (named, decision D-MB2).** `tests/test_partMB_trainer.py::EpochTests.test_training_order_passes`:
- **Old:** 18 calls of `next(5)` = 90 targets = 5 passes. That held only while batches crossed pass boundaries.
- **New:** 20 calls = 5 passes. A pass of 18 is now 4 batches (5, 5, 5, 3).
- The other assertions of the test are unchanged.

**Mutation (VERIFIED-RUN, `mf1_mutation.txt`).** The old crossing behaviour (fill up to B from the next pass) → (i) **FAILED**: `[4, 4, 4, 4, 4, 4, 4, 4, 4, 4] != [4, 4, 4, 4, 2, 4, 4, 4, 4, 2]`.

**MB0 compare (`mf1_mb0_compare.txt`):** `compared 139 tensors; differing: none`.

## MF2 — The analysis flag's batch and epoch

**Where it is resolved, and why.**
- In the config (`DataConfig.resolved`, `src/configs/config.py`), the flag on ACM/DBLP with `batch_size` and `epoch_targets` unset keeps both at 0, meaning "whole pass".
- The trainer (`MiniBatchNodeTrainer.batch_plan`) turns that into B = epoch = the training-target count of the graph it trains on. The count is known only after the dataset is built (labels and the validation carve), not at config resolution.
- Explicit `batch_size`/`epoch_targets` keep their meaning: epoch default 8,192 with an explicit B.
- An `epoch_targets` without B on the flag raises (ambiguous).
- H2GB datasets: unchanged (B required).

**Budget (verified, not assumed).** For ACM, `TrainConfig.from_dict` gives the same values with and without the flag:
- `epochs` 300, `early_stopping_patience` 30 (the HGB budget, `protocol_budget`);
- `checkpoint_metric` `macro_f1`, `checkpoint_mode` `max`.

On the flag, evaluation is every epoch (`eval_period` 1), and patience counts epochs (D-MB4), so the rule is the full-batch trainer's.

**Tests (VERIFIED-RUN).** New `tests/test_partMBfix_flag.py` (4). With `test_partMBfix_order`, `test_partMB_trainer` and `test_search_spaces` → **Ran 25, OK** (`mf2_tests_output.txt`):
- The flag resolves to (B, epoch, eval period) = (0, 0, 1), and the four budget fields equal the full-batch config's.
- `batch_plan` = (14, 14). Each of 2 epochs is one step, whose batch is every training target (count and ids).
- `fit` over 3 epochs evaluates validation and test every epoch (6 calls).
- The tuner refuses the flag; ogbn-mag without B raises (`batch_size`).
- Explicit B: the MB4 tests pass unchanged.

**Changed assertions:** none.

**Mutation (VERIFIED-RUN, `mf2_mutation.txt`).** The flag falls back to 8,192 epoch targets → the one-step test **FAILED**: `(14, 8192) != (14, 14)`.

**MB0 compare (`mf2_mb0_compare.txt`):** `differing: none`.

## MF3 — Functoriality of the forced cell maps on a batch of scopes

**Change.** Test only: new `tests/test_partMBfix_functoriality.py`. It ports the T1 computation of `tests/test_sheaf_energy.py:302-358`:
- **Batch:** 4 scopes (papers 0, 1, 3, 4) of the synthetic hand graph of `tests/partMB_fixtures.py`, from a mini-batch draw (r = 6, `ℓ_max` 3, chunks of 2) merged per anchor.
- **Maps:** factorized maps (`FactorizedNodeToEdgeSheaf`, d = 32, fasthpp, per-edge scale) on the batch graph, `build_cell_group` per group.
- **Check:** the matrix-free `T` and `R` give `max(‖R F_near − T_near‖, ‖R F_far − T_far‖) < T1_TOL`, with `T1_TOL` imported from `test_sheaf_energy` (1e-5), fp32 as there.
- **Preconditions asserted:** at least one reversed hop, a featureless author copied into two scopes, and `validate_face_orientation` on every batch group.
- On failure the test names the worst incidence (scope, cell, hop).

**Result on the unmutated code (VERIFIED-RUN):** **passes**, so the card's stop condition did not occur. `tests.test_partMBfix_functoriality tests.test_partMB_scopes tests.test_sheaf_energy` → **Ran 10, OK** (`mf3_tests_output.txt`; `test_sheaf_energy`'s IMDB T1 is skipped without IMDB data, as before).

**Mutation (VERIFIED-RUN, `mf3_mutation.txt`).** One batch cell's `face_edge_sign` flipped in the scope builder → the test **FAILED** (error). It is caught first by `validate_face_orientation`, which the card asks to keep in the test: `ValueError: … not stored in the direction its face_edge_sign states`. The T1 residual is therefore not reached under this mutation (stated).

**MB0 compare (`mf3_mb0_compare.txt`):** `differing: none`.

## MF4 — Full suite and report

**Full suite, run once (VERIFIED-RUN).** `python -m unittest discover -s tests` from `H2S-NN/` on `89afe46` → **Ran 674 tests in 2,118 s; FAILED (failures=2, errors=2, skipped=7)** (`mf4_full_suite_output.txt`; INFO lines of synthetic fixture trainings removed).
- **Failures.** Exactly the four known `test_baselines` failures: `VendorIsolationTests.test_all_declared_vendor_clones_are_present`, `VendorIsolationTests.test_colliding_vendor_module_names_do_not_leak`, `ReproducibilityTests.test_vendor_model_init_is_reproducible_under_seed_everything`, `BaselineNotebookTests.test_notebooks_are_committed_clean`. The 7 skips are the known environment skips.
- **Test count.** 667 (Part MB) + 2 (MF1) + 4 (MF2) + 1 (MF3) = **674**.
- **MB0 compare after the suite:** `compared 139 tensors; differing: none`.
- **No stop condition met.** Full batch unchanged; no method or seed-formula change; MF3 passes on the unmodified code.

## Changed existing assertions (each because of D-MB2 or the flag decision)

| test | old | new | decision row |
|---|---|---|---|
| `test_partMB_trainer.py::EpochTests.test_training_order_passes` | 18 calls of `next(5)` = 5 passes of 18 | 20 calls of `next(5)` = 5 passes (a pass = 5, 5, 5, 3) | Part MB review (D-MB2) |

No other existing assertion changed.

## Decisions for the author

- **D-MBF1, unseen targets within an epoch.** With a short last batch, an epoch that contains a pass boundary trains on fewer than `epoch_targets` targets, as D-MB2 states (H2GB's behaviour). Nothing else to decide.
- **D-MBF2, flag with `epoch_targets` but no B.** I made this raise, because it is ambiguous. Alternative: treat it as B = all targets with several passes per epoch.

## Out-of-scope findings (not fixed)

1. Large-LP epoch loop, CSR scope builder, and the one-relation-per-pair check on the large-LP data: M3 card (decisions D-MB3, Part MB review tests/M3).
2. `test_sheaf_energy`'s IMDB T1 test still needs IMDB data and is skipped here. MF3 now covers T1 on synthetic scopes without data.
