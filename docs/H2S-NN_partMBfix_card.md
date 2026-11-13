# Task card — Part MB-fix: training order inside a pass, the ACM/DBLP analysis flag, functoriality on scopes

**Design authority:** spec **v1.21**; `docs/minibatch_design_v3B.md`; decisions MB0–MB6, N1–N6 and the **Part MB review rows of 2026-11-13** (D-MB1..D-MB4, analysis flag, tests) in `docs/experiment_settings_decisions.md`; wording drafted in `docs/spec_v1.22_draft_edits.md` W-1 (not applied: the decisions bind; never edit the spec). `docs/H2S-NN_partMB_card.md` and `docs/H2S-NN_partMB_report.md` give the context. `H2S-NN/AGENTS.md`/`CLAUDE.md` bind for environment, coding, testing and evidence tags.
**Branch:** continue on `part-mb` from the docs commit given in the prompt (on top of `0e56719`). **Report:** `docs/H2S-NN_partMBfix_report.md`; evidence `docs/partMBfix_evidence/`.

## Rule
Correctness only: unit tests on synthetic graphs, mutation checks, the full suite once. No run on real data, no metric reported as a result, no comparison. Full batch stays bit-identical: `docs/partMB_evidence/mb0_reference.py compare` must report `differing: none` after every unit.

## Decisions (author, 2026-11-13)
| ID | Decision |
|---|---|
| D-MB1 | Seed formulas stay exactly as implemented (`chunk_seed`, `pass_seed`, `training_draw_seed`). **No change.** |
| D-MB2 | **H2GB's loader:** a training batch never crosses a pass boundary. The last batch of a pass holds the remaining targets (fewer than B, no target repeated) and counts as one optimizer step; the loss is the mean over its own targets. The next step starts the next pass (new seeded permutation). Steps per epoch stay `epoch_targets / B`, so an epoch that contains a boundary sees fewer than `epoch_targets` targets (as H2GB: `LoaderWrapper`, `vendor/H2GB/H2GB/sampler/custom_sampler.py:37-50`, counts iterations and re-creates the loader on exhaustion; PyG loaders keep the short last batch). |
| Flag | **ACM/DBLP analysis flag (N4):** when `training_mode: minibatch` is set on ACM or DBLP and `batch_size`/`epoch_targets` are not set (0), B = the number of training targets and 1 epoch = one pass = one optimizer step; `eval_period` 1 (already); epochs, patience and checkpoint rule are those of the full-batch run of the same dataset (same config fields; verify, do not assume). Explicit `batch_size`/`epoch_targets` stay accepted in configs (tests set them); the tuner keeps refusing the flag and `epoch_targets`. On the H2GB datasets nothing changes (B required, 8,192 targets). |
| D-MB3, D-MB4 | No change in this card (large-LP loop → M3; patience counted in epochs). |

## Units

### MF1 — Training order inside a pass (D-MB2)
`TrainingOrder.next(B)` returns `min(B, remaining targets of the current pass)` targets; at the end of a pass the next call reshuffles first. The trainer's step loop takes exactly `steps_per_epoch` calls per epoch whatever their sizes; the training draw seed stays `training_draw_seed(base, epoch, batch)`.
**Tests:** (i) 18 targets, B = 4: batch sizes over two passes are 4,4,4,4,2,4,4,4,4,2; each pass is a permutation; no batch holds a target twice; (ii) steps per epoch unchanged (`epoch_targets/B`) and the step counter counts the short batch as one step; (iii) the loss of a short batch is the mean over its own targets (spy on the loss input size); (iv) order deterministic per seed, differs per pass (existing test kept). **Mutation:** the old crossing behaviour (fill up to B from the next pass) → (i) fails.

### MF2 — The analysis flag's batch and epoch
Resolve B and the epoch for the flag as in the decision (where the training-target count is known: the trainer or dataset build; state where and why). A full-batch config field set on ACM resolves to the same `epochs`, `early_stopping_patience` and checkpoint rule with and without the flag.
**Tests (synthetic ACM-shaped graph):** with the flag and no B: one step per epoch, the batch is every training target (count and ids), evaluation every epoch; `epochs`/patience equal the full-batch run's; explicit B still works (MB4 tests unchanged); the tuner still refuses the flag; H2GB datasets without B still raise. **Mutation:** the flag falls back to 8,192 epoch targets → the one-step test fails.

### MF3 — Functoriality of the forced cell maps on a batch of scopes
Port the T1 computation of `tests/test_sheaf_energy.py:302-358` (`FunctorialityOnRealDrawTests`, skipped without IMDB data) to a batch of ≥ 3 scopes on the synthetic graph of `tests/partMB_fixtures.py` (mini-batch draw, per-anchor merge): factorized maps on the batch graph, `build_cell_group` per group, `‖R F_near − T_near‖` and `‖R F_far − T_far‖` below `T1_TOL` (same constant, fp32 as there, or float64 with a stated tolerance no looser), with at least one reversed hop and one featureless node in two scopes. Keep `validate_face_orientation`. **Mutation:** one batch cell's `face_edge_sign` flipped → the test fails.

### MF4 — Full suite and report
Full suite once. Expected: only the four known `test_baselines` failures and the 7 environment skips; MB0 compare `differing: none`. Report per unit (done, VERIFIED-RUN, deviations), any existing assertion changed **because of D-MB2 or the flag** (listed with old and new value and the decision row), "Decisions for the author", out-of-scope findings.

## Rules and stop conditions
- One commit per unit, explicit paths, tests with the change. Never stage line-ending-only diffs, `Claude outputs/`, `data/`, `H2S-NN/paper/`, vendor clones. Forbidden: loosening tolerances, deleting or skipping tests, silencing exceptions, special-casing inputs, editing the spec or the cards. An existing assertion may change only where D-MB2 or the flag decision changes the expected value; list each one in the report.
- Tests never reach real data (a fixture makes the real data root unreachable at import time).
- **Stop:** any full-batch output changes → stop the unit, report the first differing tensor; a change would alter the method (energy, maps, `g`, solver, WISP's `p`) or the seed formulas → stop the unit; MF3 fails on the unmutated code → stop and report the worst incidence (scope, cell, hop), do not relax `T1_TOL`.
- **Not stop conditions:** any metric value; PCG iteration counts.
