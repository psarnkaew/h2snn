# Task card — Part N: LP negatives and GC metric as their sources, stop conditions in tuning, correctness smoke

**Design authority:** spec **v1.16** (changelog K1–K6; draft `docs/spec_v1.16_draft_edits.md`) and `docs/experiment_settings_decisions.md` (rows of 2026-11-12). `H2S-NN/AGENTS.md`/`CLAUDE.md` bind for environment (conda env `h2snn`), coding, testing (`unittest`) and evidence tags.
**Branch:** `part-n`, created from `part-m` at `738b2db`. **Report:** `docs/H2S-NN_partN_report.md`.

## Goal and the one rule of this card
Implement the decisions taken after Part M and show that the code is **correct** and that training **runs**: losses finite, no collapse. **No performance experiments, no empirical results, no metric is reported as a result, nothing is compared between methods** (author, 2026-11-12: time is short). WISP is not touched.

## Scope rules
- One commit per unit; tests with the change; mutation checks where marked; VERIFIED-RUN output in the report.
- Do not change: WISP (sampler, anchors, `ℓ_max`, `r`), the energy, maps, solver, readouts; the open items (`feats_type`, `concat_mlp`, GC epochs 200/500, mini-batch design).
- Forbidden: loosening tolerances, deleting or skipping tests (except N6, which removes tests of datasets removed by spec 1.5), silencing exceptions, special-casing inputs, editing the spec to fit the code. Out-of-scope problems are listed, not fixed.

## Units

### N0 — Preconditions
Read spec v1.16 (§6 in full, changelog K1–K6), the decisions log rows of 2026-11-12, Part M report "Decisions for the author". Create `part-n` from `738b2db`. Commit by explicit path: `docs/H2S-NN_methodology_spec.md`, `docs/spec_v1.15_draft_edits.md`, `docs/spec_v1.16_draft_edits.md`, `docs/spec_backups/H2S-NN_methodology_spec_v1.15.md`, `docs/experiment_settings_decisions.md`, `docs/H2S-NN_partN_card.md`, `docs/handoff_2026-11-12.md` ("docs: spec v1.16, Part N card"). **Do not run the full suite here** (author, time): the pre-state is Part M's M12 run (557 tests, 9 known failures, `partM_evidence/m12_suite_failures.txt`). During N1–N6 run only the affected test modules; the full suite runs once, in N8.

### N1 — LP test sets as HGB's runner (K1, K2)
Read HGB's `LP/benchmark/methods/baseline/run_new.py` and `LP/benchmark/scripts/data_loader.py` (record how they were read and the commit/date; `vendor/HGB` is empty).
- 2-hop set (`hgb_test_negatives`): validation positives enter the adjacency **in their stored orientation only**; train and test facts as HGB's code builds them (state exactly, with the line read). Everything else unchanged from M7.
- Random set: port `get_test_neigh_w_random` (seed 1; one tail of the tail type per test positive; redraw while the tail is a known neighbour of the head from train, validation or test; known-neighbour sets exactly as HGB builds them).
- Write both negative files per task (`hgb_negatives/{test_neg_2hop_rel<r>.npy, test_neg_random_rel<r>.npy, meta.json}`); the loader and the baseline bundles expose both; evaluation computes ROC-AUC and MRR **per set**. A set with no negatives (dblp_ap, imdb_ma 2-hop) yields metrics `None` with the reason "no length-2 head→tail type path"; it never yields MRR = 1 or NaN.
- Report per task: negatives written per set; positives without 2-hop negatives (count, share).
**Tests:** hand graph where both sets are known by hand; a validation fact stored one way changes the 2-hop candidates exactly as HGB's loop (independent re-implementation of HGB's loop); random set: seed-1 determinism against an independent re-implementation, no negative is a known neighbour, count = test positives; empty 2-hop set → metrics `None` with the reason. **Mutation:** validation added in both orientations → a test fails; random set without the redraw → a test fails.

### N2 — LP training negatives as HGB (K3)
Port `get_train_neg` (one random tail of the tail type per training positive, known positives not excluded). Draw them when HGB's runner draws them (per epoch or once; quote the line), seeded `run seed + epoch` if per epoch, else `run seed`. Every method's LP training uses them (H2S-NN trainer and the shared baseline loop).
**Tests:** count and type range; determinism per seed; known positives may appear (not excluded); both H2S-NN and the baseline loop call the same function. **Mutation:** exclusion of known positives re-added → a test fails.

### N3 — GC metrics as HetSheaf (K4)
Macro-F1 and Micro-F1 for GC = torchmetrics `F1Score(task="multiclass", average="macro"|"micro", num_classes=C)` as `hetsheaf/graph_classification/graph_classifier.py` builds them (record the torchmetrics version in `h2snn`; if absent, install the version HetSheaf pins and record it). Used by H2S-NN's GC trainer and the baselines' GC loop. NC metrics unchanged (M8).
**Tests:** hand example with every class present (equals sklearn macro); hand example with a class absent from truth and prediction (document torchmetrics' value in the test, whatever it is); both paths give identical values.

### N4 — Stop conditions and failed trials (K5)
- Non-finite stop: raise a named `StopConditionError` subclass when any `g(ξ_f)` is not finite (checked before `exp`), logged like M6's conditions.
- Tuner (H2S-NN script and the baseline campaign): a trial that raises any `StopConditionError` or `PCGConvergenceError`, or whose training loss becomes non-finite (any method), is recorded as failed: no score is sent to the Bayesian search, the record names the condition, epoch and value, and the campaign summary counts failed trials per method and dataset.
**Tests:** forced non-finite `g` raises; a fake trial raising each condition is recorded failed with no score; a NaN loss in a baseline loop is recorded failed; the summary count.

### N5 — REMOVED (author, 2026-11-12)
The per-graph evaluation warm start is not needed for correctness (the difference is within the solver tolerance and evaluation order is fixed). Listed as later work; do not implement.

### N6 — Removed datasets: tests and adapter lists
Remove `PolynomialRelationKeyTests` cases and `SUPPORTED_LP_DATASETS` entries that need Amazon or YouTube (removed in spec 1.5; data not on disk). Nothing else.

### N7 — Correctness smoke: finite losses, no collapse (no performance)
Script `scripts/partn_smoke.py`; v3-B anchored, `arm: full`, d = 32, `ℓ_max` = 4, `r` = 8, default protocol otherwise. **Runs (reduced, author 2026-11-12): NC acm (2 epochs); LP the 6 local tasks (2 epochs each); GC MUTAG only (2 epochs, full protocol, batch 1); GCN baseline on acm, acm_pa and MUTAG (2 epochs).** No other GC dataset is run (their loaders are covered by Part M tests).
**Checks per run (record all values):** (a) every training and validation loss finite; (b) no stop condition fired (if one fires: record values, no setting is changed); (c) **no collapse**: NC/GC — validation predictions use ≥ 2 classes (IMDB: per genre, predictions not identical for every movie), and the mean over coordinates of the std over nodes of `x*` > 1e-6; LP — std of validation scores > 1e-6 on every test set; (d) parameters move; (e) last-epoch training loss vs first (reported, not a gate).
**Not reported:** accuracy, F1, ROC-AUC, MRR or any comparison. Metrics are computed by the code but left out of the report except as the inputs of checks (c).

### N8 — Full suite and report
Full suite; report per unit (done, VERIFIED-RUN, deviations), the N7 check table (finite / stop / collapse / params moved / loss first→last), heads and positives without 2-hop negatives per task, failed-trial behaviour, "Decisions for the author", out-of-scope findings.

## Stop conditions
- (a) or (c) of N7 fails: stop N7, report the run's values and the configuration; do not change settings.
- A change would alter the method or needs an open decision: stop the unit, report.
- HGB's code cannot be read: stop N1/N2, report.
**Not stop conditions:** any metric value; a stop condition firing in N7 (report it).
