# Part N report — LP negatives and GC metric as their sources, stop conditions in tuning, correctness smoke

Card: `docs/H2S-NN_partN_card.md`. Design authority: spec v1.16 (K1–K6) and the decisions log rows of 2026-11-12 ("Part N trimmed"). N5 removed by the author. **The one rule of this card:** no performance experiment and no empirical result; no accuracy, F1, ROC-AUC, MRR or comparison between methods appears in this report. Evidence: `docs/partN_evidence/`. Commands run from `H2S-NN/` with `C:\Users\Acer\anaconda3\envs\h2snn\python.exe`.

## N0 — Preconditions

- Read: spec v1.16 §6 (LP, GC, tuning, stop conditions), changelog K1–K6; decisions log rows of 2026-11-12; Part M report "Decisions for the author".
- Branch **`part-n` created from `part-m` at `738b2db`**. Docs commit **`eb1d213`** "docs: spec v1.16, Part N card", by explicit path: `docs/H2S-NN_methodology_spec.md`, `docs/spec_v1.16_draft_edits.md`, `docs/spec_backups/H2S-NN_methodology_spec_v1.15.md`, `docs/experiment_settings_decisions.md`, `docs/H2S-NN_partN_card.md`, `docs/handoff_2026-11-12.md` (`docs/spec_v1.15_draft_edits.md` is in the card's list but was unchanged since Part M's M0 commit, so nothing was staged for it). Not staged: `docs/minibatch_design_v3B.md` (another session), `docs/spec_v1.17_draft_edits.md` (not in the card's list).
- No full suite at N0 (author): the pre-state is Part M's M12 run (557 tests, 9 known failures, `partM_evidence/m12_suite_failures.txt`).

## N1 — LP test sets as HGB's runner (K1, K2)

**How HGB was read.** `LP/benchmark/methods/baseline/run_new.py` and `LP/benchmark/scripts/data_loader.py`, THUDM/HGB `master`, last commit returned by the GitHub API for both paths: `2ccb1d85d1a54dd604cd5a3ae70e596e18b54eda` (2023-03-18, "update data loader"; the path filter may not isolate each file). Read 2026-11-12 through a fetch agent. `data_loader.py`'s `get_test_neigh`, `get_test_neigh_w_random`, `get_valid_neg`, `get_train_neg` and `evaluate` were returned as code text in Part M (M7); in Part N the agent returned a **structural summary instead of verbatim code** (the repository declares no licence), with the points below. `vendor/HGB` is empty. VERIFIED-SOURCE (summary level):
- `run_new.py`: for each test relation, trains, loads the best checkpoint, then evaluates on `dl.get_test_neigh()` → `res_2hop` and on `dl.get_test_neigh_w_random()` → `res_random`, both printed; keys `roc_auc`, `MRR`.
- `get_test_neigh`: `pos_links` = Σ_r (`links['data'][r]` + its transpose) + Σ_r (`links_test['data'][r]` + transpose) + Σ_r `coo_matrix(valid_pos[r])` **without transpose** (validation in its stored orientation only).
- `get_test_neigh_w_random`: `random.seed(1)`; known neighbours of each head = nonzero columns of `pos_links` (same three parts); for each test positive (sparse `nonzero()` order) a tail `randrange` of the tail type, redrawn while it is a known neighbour.

**Done.**
- `src/data/hgb_lp_negatives.py`: `hgb_adjacency(both_ways, one_way, n)` (training and test facts both ways, validation positives as stored); `hgb_test_negatives(test_pos, both_ways, one_way, n, dst_range)` (2-hop set; everything else as M7); new `hgb_random_test_negatives` (random set: positives in row-major sorted unique order, `random.Random(1)`, redraw while known; raises if a head knows every tail, where HGB's loop would never end); `NO_TWO_HOP_PATH_REASON = "no length-2 head→tail type path"`.
- Loader (`link_prediction_dataset.py`): `both_ways` = the stored message graph (every relation, held-out facts removed) + the test facts of every relation; `one_way` = the validation facts of every relation as stored. **Note:** for tasks whose target relation has a stored reverse relation (acm_pa, dblp_ap, imdb_ma), our split also lists the reverse relation's held-out copies as validation facts of that relation, so those validation facts enter in both orientations through the two relations — the same as HGB's code applied to our files; for LastFM and freebase_bb (no reverse relation) the orientation is one-way. Both sets are computed (or read from the task's files); `PreparedLinkPredData` gains `test_neg_random_edge_index`, `test_neg_random_edge_relation`, `test_2hop_empty_reason` (`test_neg_edge_index` is the 2-hop set). Baseline bundles expose the same object.
- Evaluation per set: `src/utils/misc.lp_test_set_metrics` computes ROC-AUC, AUPR, MRR per set (`2hop_*`, `random_*`, plus `<set>_negatives`); a set without negatives gives `None` and `<set>_reason`, never MRR = 1 or NaN. Used by the H2S-NN LP trainer (`_eval_test_sets`, one encode) and the baselines' LP loop. Run summaries (`run_reports.summarize_records`, the baseline campaign's `summarize_cell`) skip `None`, write `mean: None` and the reason; LP headline metrics are `test_2hop_auroc, test_2hop_mrr, test_random_auroc, test_random_mrr`; the W&B flattener skips `None`/text.
- Files: `scripts/lp/write_hgb_negatives.py` writes `hgb_negatives/{test_neg_2hop_rel<r>.npy, test_neg_random_rel<r>.npy, meta.json}` (Part M's single-set file is removed); the loader refuses a Part M meta file.

**Negatives written per task (VERIFIED-RUN, `n1_write_hgb_negatives_output.txt`):**

| Task | test positives | 2-hop negatives | random negatives | heads without 2-hop candidates | positives without 2-hop negatives (share) | 2-hop reason |
|---|---|---|---|---|---|---|
| acm_pa | 995 | 763 | 995 | 196 / 839 | 232 (23.3 %) | — |
| dblp_ap | 1,964 | 0 | 1,964 | 1,256 / 1,256 | 1,964 (100 %) | no length-2 head→tail type path |
| imdb_ma | 1,478 | 0 | 1,478 | 1,340 / 1,340 | 1,478 (100 %) | no length-2 head→tail type path |
| freebase_bb | 18,723 | 18,646 | 18,723 | 75 / 9,645 | 77 (0.41 %) | — |
| PubMed | 8,528 | 8,525 | 8,528 | 3 / 3,048 | 3 (0.04 %) | — |
| LastFM | 18,567 | 18,567 | 18,567 | 0 / 1,878 | 0 | — |

freebase_bb moved from 72 to 77 positives without 2-hop negatives against M7 (validation now one-way); the other tasks are unchanged.

**Tests (VERIFIED-RUN).** `tests/test_hgb_lp_negatives.py` (16): the hand graph's 2-hop and random sets; a validation fact stored 1→6 vs 6→1 changes head 0's candidates exactly as an independent dense re-implementation of HGB's loop; the 2-hop set on a random graph with a one-way validation set equals that re-implementation (seed 1); no 2-hop candidate is a known pair or a self-loop; random set: equal to an independent re-implementation of `get_test_neigh_w_random`, deterministic, count = test positives, no negative is a known neighbour; an empty 2-hop set gives `None` with the reason; loader on a tiny LastFM release: both sets equal the re-implementations on the loader's split, a friendless user gets no 2-hop negative but a random one, validation negatives = `get_valid_neg`, both files written, read back identically and refused for other positives; batching unchanged. Updated for the new keys: `test_data` (ported negatives test: new signature), `test_wandb_and_runs` (summary with an empty 2-hop set → `None` + reason), `test_baselines` (two LP-key tests). Affected modules: **Ran 289 tests; failures=2, errors=7, all nine in the M12 known set, none new** (`n1_tests_output.txt`).
- **Mutations (VERIFIED-RUN, `n1_mutation_check.py` → `n1_mutation_output.txt`):** (a) validation in both orientations → `test_validation_orientation_changes_the_candidates_as_hgb` fails; (b) random set without the redraw → both random-set tests fail; restored → 0. *Both caught.*
- **Listed, not changed:** HGB's runner re-draws validation negatives (`get_valid_neg`) at every batch step; here they are drawn once per run (deterministic model selection). Listed under decisions.

## N2 — LP training negatives as HGB (K3)

**HGB (VERIFIED-SOURCE, summary level, same reading as N1):** `get_train_neg` pairs each training positive's head with one tail `random.randrange` over the tail type's id range, without rejecting known edges; `run_new.py` calls `dl.get_train_neg(edge_types=[test_edge_type])` **inside its epoch loop** (re-drawn every epoch). `run_new.py` sets no seed itself (Python's global `random` is only re-seeded by `get_test_neigh*`), so HGB's training draws are unseeded; here they are seeded `run seed + epoch` as the card says.

**Done.**
- `src/data/hgb_lp_negatives.py`: `hgb_train_negatives` (= `get_train_neg`) and `hgb_epoch_train_negatives(data, epoch)`: per target relation, `neg_sampling_ratio` tails per positive in the positives' order, seed `data.run_seed + epoch` (+ a relation offset for multi-relation targets); under `neg_sampling_mode: fixed` the epoch-0 draw is reused. The default of `neg_sampling_mode` is now `resample_each_epoch` (HGB's runner); the LP presets already said so.
- `PreparedLinkPredData.run_seed` (the loader's `config.seed`), so the baselines' loop, which receives only the data object, uses the same seed as H2S-NN.
- Both callers use it: `LinkPredictionTrainer._train_negatives(data, epoch)` and `src/baselines/training.sample_train_negatives(data, device, epoch)` (called from `train_link_prediction` every epoch). The old samplers (PyG `negative_sampling` excluding known positives; the candidate pool) are removed from both.

**Tests (VERIFIED-RUN).** `tests/test_hgb_lp_negatives.TrainNegativeTests` (4): count, order and tail-type range (incl. ratio 2), values equal `Random(run seed + epoch)`; same epoch → same draw, next epoch → a different draw, `fixed` → the same draw every epoch; with a one-node tail type every negative is that node, a known positive (not excluded); the H2S-NN trainer and the baselines' loop both call `hgb_epoch_train_negatives` and get the same negatives. Affected modules: **Ran 287 tests; the nine M12 known failures, none new** (`n2_tests_output.txt`).
- **Mutation (VERIFIED-RUN, `n2_mutation_check.py`):** exclusion of known positives re-added → 2 of 4 tests fail; restored → 0. *Caught.*

## N3 — GC metrics as HetSheaf (K4)

**Done.** `src/utils/misc.hetsheaf_f1`: Macro- and Micro-F1 = torchmetrics `F1Score(task="multiclass", average="macro" | "micro", num_classes=C)` on the logits, as HetSheaf's `GraphClassifier` builds them (`hetsheaf/graph_classification/graph_classifier.py:56-67`, commit `6964e1e`; VERIFIED-SOURCE, local clone). `compute_graph_classification_metrics` uses it; that function serves both H2S-NN's GC trainer and the baselines' GC loop. NC metrics unchanged (M8). **torchmetrics version: 1.9.0** in `h2snn` (already installed; HetSheaf's `uv.lock` pins 1.8.2 — not installed, the installed version is used and recorded). `torchmetrics` added to `requirements/core.txt` (it is now imported by core code; it was only in the H2GB group).

**Tests (VERIFIED-RUN).** New `tests/test_gc_metrics.py` (3): every class present → equals sklearn's macro/micro; class 2 absent from truth and prediction (C = 3) → torchmetrics 1.9.0 gives macro 11/15 (the absent class is not averaged) and micro 3/4, recorded in the test; H2S-NN's GC trainer and the baselines' GC loop use the same function and give identical values. With `test_nc_metrics` (one docstring updated), `test_gc_protocol`, `test_graph_classification_trainer`, `test_task_heads`, `test_shared_protocol`, `test_data`: **Ran 54 tests, OK** (`n3_tests_output.txt`). No mutation is marked for N3.
