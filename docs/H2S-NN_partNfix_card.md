# Task card — Part N-fix: HGB ports checked against HGB's own code, GC collapse check redefined, torchmetrics pinned, smoke completed

**Design authority:** spec **v1.16** (K1–K6) and `docs/experiment_settings_decisions.md` (rows "Part N review", 2026-11-12). `H2S-NN/AGENTS.md`/`CLAUDE.md` bind for environment (conda env `h2snn`), coding, testing (`unittest`) and evidence tags.
**Input:** `docs/H2S-NN_partN_report.md` (Part N, branch `part-n`, last commit `c46c1a4`).
**Branch:** `part-nfix`, created from `part-n` at `c46c1a4`. **Report:** `docs/H2S-NN_partNfix_report.md`.

## Goal and the one rule of this card
Close the three gaps the Part N review found: (1) the HGB ports (N1, N2) were checked only against re-implementations written from a summary of HGB's code, never against HGB's code itself; (2) the GC collapse check of N7 (c) measured "has not learned yet" instead of collapse (designer's error in the Part N card); (3) torchmetrics differs from HetSheaf's lock. Then finish the N7 smoke. **Same rule as Part N: no performance experiment, no empirical result; no accuracy, F1, ROC-AUC, MRR or comparison between methods is reported.** WISP is not touched.

## Scope rules
- One commit per unit; tests with the change; VERIFIED-RUN output in the report; evidence in `docs/partNfix_evidence/`.
- Do not change: WISP, the energy, maps, solver, readouts, the NC collapse check, any tolerance; the open items (`feats_type`, `concat_mlp`, GC epochs 200/500, mini-batch design, derived-LP baseline wiring).
- Forbidden: loosening tolerances, deleting or skipping tests, silencing exceptions, special-casing inputs, editing the spec to fit the code. Out-of-scope problems are listed, not fixed.
- Do not stage the adapter files with line-ending-only differences, anything under `Claude outputs/`, or `Slides template.pptx`. Stage every file by explicit path.

## Units

### F0 — Preconditions
Read this card, the Part N report ("Decisions for the author", N1, N2, N7), the decisions log rows "Part N review". Create `part-nfix` from `c46c1a4`. Commit by explicit path `docs/H2S-NN_partNfix_card.md`, `docs/experiment_settings_decisions.md`, `docs/handoff_2026-11-12.md`, `docs/H2S-NN_methodology_spec.md` (v1.17, applied by the designer), `docs/spec_v1.17_draft_edits.md`, `docs/spec_backups/H2S-NN_methodology_spec_v1.16.md`, `docs/minibatch_design_v3B.md` (the source of v1.17), `docs/spec_v1.18_draft_edits.md` (draft, not applied) ("docs: spec v1.17, v1.18 draft, Part N-fix card, Part N review decisions"). The binding spec for this card stays v1.16 + this card: v1.17 changes only mini-batch rules, which no unit here touches. No full suite here.

### F1 — HGB's code as the oracle (N1, N2)
1. Clone `https://github.com/THUDM/HGB` into `H2S-NN/vendor/HGB` (empty now). Record the commit in `H2S-NN/vendor/HGB/CLONED_COMMIT.txt` (same convention as `vendor/HINormer`). HGB declares no licence: **do not commit HGB's files**. Add to `H2S-NN/.gitignore`: `vendor/HGB/*` and `!vendor/HGB/CLONED_COMMIT.txt`; commit only those two files' changes.
2. Read `LP/benchmark/scripts/data_loader.py` (`get_test_neigh`, `get_test_neigh_w_random`, `get_valid_neg`, `get_train_neg`, `evaluate`, the loader's construction of `links`, `links_test`, `valid_pos`) and `LP/benchmark/methods/baseline/run_new.py` (where each function is called: per epoch, per batch step, once). Quote file:line for every point the Part N report states at summary level (N1 "How HGB was read", N2 "HGB") and mark each point **confirmed** or **different** (with the actual line).
3. Oracle tests (new `tests/test_hgb_oracle.py`; skipped with a clear message only if `vendor/HGB` is absent, never otherwise): write tiny datasets in HGB's file format (`node.dat`, `link.dat`, `link.dat.test`, `info.dat` as HGB's loader expects; at least: one with a validation fact stored one way, one with a head without 2-hop candidates, one with two node types and two relations), load them with **HGB's own `data_loader` class**, and compare with our functions on the same split:
   - 2-hop set: HGB `get_test_neigh()` vs `hgb_test_negatives` — identical (head, tail) lists in identical order.
   - Random set: HGB `get_test_neigh_w_random()` vs `hgb_random_test_negatives` — identical lists and order.
   - Training negatives: HGB `get_train_neg(edge_types=[r])` vs `hgb_train_negatives`, with Python's global `random` seeded to the same value before each call (HGB draws from the global generator) — identical.
   - Validation negatives: HGB `get_valid_neg(edge_types=[r])` vs ours, same seeding — identical.
   - The validation split itself: HGB's `valid_pos` vs ours on the same files (if HGB's split draws randomly, seed both the same way and state how).
   If a difference appears: **stop F1**, report the case, HGB's line and ours; do not change our code to match until the author has read it (a difference may change K1–K3 in the spec).
4. Keep the existing Part N tests (they test our code on our hand graphs); the oracle tests are added, not substituted.
5. **Decision 4 (validation negatives).** If `run_new.py` re-draws `get_valid_neg` per batch step or per epoch, the code keeps **one draw per run** (author decision 2026-11-12: deterministic model selection; same for every method) and the report states the exact HGB line for the spec text. If HGB draws once, ours already matches. No code change either way.
**Mutation (VERIFIED-RUN):** our random set with the positives in a different order → an oracle test fails; our 2-hop adjacency with validation in both orientations → an oracle test fails.

### F2 — GC collapse check redefined (decision 1)
In `scripts/partn_smoke.py`, for GC only (NC and LP checks unchanged):
- Check (c) for GC becomes: over the validation graphs, (c1) the mean over logit coordinates of the std over graphs of the logits > 1e-6, and (c2) the mean over coordinates of the std over graphs of the pooled graph embedding (the readout's input to its Linear layer) > 1e-6. Both must hold. The number of predicted classes is still recorded per epoch but is **not** a gate (reason: on MUTAG, 125:63 graphs, about 19 validation graphs, predicting the majority class after 2 epochs is normal early training, not collapse).
- Count check (closes the UNVERIFIED point of Part N N7): the number of validation graphs whose logits were captured equals `len(fold.val_loader.dataset)`; a mismatch is a script error (fix the capture, report it), not a model result.
- The x* std over the nodes of the validation graphs stays recorded (as Part N).
**Tests:** a fake GC model with identical pooled embeddings for every graph (hence identical logits, since the readout is one Linear layer) fails (c1) and (c2); one with varied embeddings but a constant-output head (Linear weights zero) fails (c1) only; a fake with varied logits and embeddings but one predicted class passes; a capture shorter than the dataset raises.

### F3 — torchmetrics as HetSheaf's lock (decision 6)
Install torchmetrics **1.8.2** in `h2snn` (HetSheaf's `uv.lock`); record any dependency change pip reports (if 1.8.2 cannot install with the env's torch, stop F3 and report). Pin `torchmetrics==1.8.2` in `requirements/core.txt` (and the H2GB group if it lists torchmetrics). Re-run `tests/test_gc_metrics.py`: if the absent-class value differs from 1.9.0's (macro 11/15, micro 3/4), update the recorded value in the test to 1.8.2's and state both values in the report (the test documents the version in use).

### F4 — Smoke completed (no performance)
With the F2 script, 2 epochs each, same configuration as Part N N7: **H2S-NN GC MUTAG; GCN NC acm; GCN LP pubmed** (replaces GCN acm_pa, which no baseline supports yet; decision 3); **GCN GC MUTAG**. Record the same table as Part N N7 (a)–(e) plus (c1), (c2), the count check and the predicted-class count. **freebase_bb is not run here** (decision 2: lab server); give the exact command for the server in the report.
**Stop conditions of F4:** (a) a non-finite loss, or (c) for NC/LP as in Part N, (c1)/(c2) for GC: stop F4, report values and configuration, change nothing. **Not stop conditions:** any metric, a single predicted class, a stop condition of the model firing (report it).

### F5 — Full suite and report
Full suite once. Report per unit (done, VERIFIED-RUN, deviations), the F1 confirmed/different table with HGB file:line, the oracle-test results, the F4 table, torchmetrics before/after, "Decisions for the author", out-of-scope findings. Expected known failures: the vendor-clone and notebook failures of Part N N8, minus any that F1's clone resolves (state which).

## Stop conditions (whole card)
- F1: any difference between HGB's code and ours (stop F1; F2–F3 may continue; F4 waits).
- HGB cannot be cloned or its loader cannot run on the tiny files: stop F1, report what fails.
- A change would alter the method or needs an open decision: stop the unit, report.
