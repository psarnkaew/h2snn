# H2S-NN — Part N-fix report

Card: `docs/H2S-NN_partNfix_card.md`. Binding: spec v1.16 (K1–K6) + the card; spec on disk is v1.17 (mini-batch rules only, untouched here); `docs/spec_v1.18_draft_edits.md` is a draft and was not implemented. **Same rule as Part N:** no performance experiment and no empirical result; no accuracy, F1, ROC-AUC, MRR or comparison between methods appears here. Evidence: `docs/partNfix_evidence/`. Commands run from `H2S-NN/` with `C:\Users\Acer\anaconda3\envs\h2snn\python.exe`.

## F0 — Preconditions

- Read: the card; Part N report (N1, N2, N7, "Decisions for the author"); decisions log rows "Part N review" (2026-11-12, decisions 1–7) and "HGB source".
- `part-n` verified at `c46c1a41c7ee8a47afe6e16996abece678e4a3d9`; branch **`part-nfix`** created from it. No `.git/index.lock` was present.
- Commit **`a5570f4`** "docs: spec v1.17, v1.18 draft, Part N-fix card, Part N review decisions", by explicit path: `docs/H2S-NN_partNfix_card.md`, `docs/experiment_settings_decisions.md`, `docs/handoff_2026-11-12.md`, `docs/H2S-NN_methodology_spec.md` (v1.17), `docs/spec_v1.17_draft_edits.md`, `docs/spec_backups/H2S-NN_methodology_spec_v1.16.md`, `docs/minibatch_design_v3B.md`, `docs/spec_v1.18_draft_edits.md`. Not staged: `docs/H2S-NN_partO_card.md` (untracked, not in the card's F0 list). No suite run.

## F1 — HGB's code as the oracle (N1, N2) — **STOPPED: the validation split differs**

**Clone.** `git clone --depth 1 https://github.com/THUDM/HGB` → `H2S-NN/vendor/HGB`, commit **`ca6fd5bb0c1ca32e63b132c8bfe8f11a4a6629fe`** (2023-07-28, "Update README.md"), recorded in `vendor/HGB/CLONED_COMMIT.txt`. The clone's own `.git` was removed after recording the commit (otherwise git treats the folder as an embedded repository and will not stage `CLONED_COMMIT.txt`; `vendor/HINormer` has no `.git` either). `.gitignore`: `vendor/HGB/*`, `!vendor/HGB/CLONED_COMMIT.txt`; `git status -uall vendor/HGB` lists only `CLONED_COMMIT.txt`. Part N cited `2ccb1d8` (2023-03-18) from a path-filtered API query; the clone's HEAD is later (a README change). The lines below are from the clone.

### Part N's summary-level points against HGB's code (VERIFIED-SOURCE, `vendor/HGB` at `ca6fd5b`)

`DL` = `LP/benchmark/scripts/data_loader.py`, `RUN` = `LP/benchmark/methods/baseline/run_new.py`.

| # | Part N statement (N1 "How HGB was read", N2 "HGB") | HGB line | verdict |
|---|---|---|---|
| 1 | `run_new.py`: per test relation, train, load the best checkpoint, evaluate on `get_test_neigh()` → `res_2hop` and on `get_test_neigh_w_random()` → `res_random`, both printed; keys `roc_auc`, `MRR` | RUN:98 (loop over `links_test` relations), RUN:178 (load checkpoint), RUN:182 + 206–209 (2-hop), RUN:211 + 223–226 (random), RUN:231–232 (print); DL:269 (keys) | **confirmed**. Not in Part N: RUN:188–194 replace the 2-hop list by `<dataset>_ini_<r>_label.txt` when that file exists in the data folder (none of our six tasks has one; checked `find data -iname "*_label.txt"` → nothing) |
| 2 | `get_test_neigh`: `pos_links` = Σ training facts + transpose, Σ test facts + transpose, Σ `valid_pos` **without** transpose | DL:351–353 (`links` + `.T`), DL:354–355 (`links_test` + `.T`), DL:356–359 (`coo_matrix(valid_pos[r])`, no `.T`) | **confirmed** |
| 3 | `random.seed(1)` in both test-set functions | DL:347, DL:418 | **confirmed** |
| 4 | 2-hop candidate: `(A·A)[h,t] > 0`, `A[h,t] == 0`, `t != h`, `t` of the tail type | DL:361–363 (`A·A`, binarised), DL:364–367 (minus `pos_links`, minus identity), DL:369–372 (keep > 0), DL:383–384 (head and tail type ranges) | **confirmed** (note: DL:366 casts `pos_links` to `int`; with a weight < 1 a known pair would survive. Every weight in our six tasks' `link.dat`/`link.dat.test` is an integer ≥ 1 (checked with awk; PubMed min 1), so no effect here) |
| 5 | heads ascending; `random.choices(candidates, k = #test positives of h)`; a head without candidates gets no negative | DL:403 (`sorted`), DL:410–411 | **confirmed** |
| 6 | `get_test_neigh_w_random`: known neighbours = nonzero columns of `pos_links` (same three parts); per test positive in `nonzero()` order one `randrange` tail of the tail type, redrawn while known | DL:423–431 (same three parts), DL:433–437, DL:443–449 | **confirmed** |
| 7 | `get_train_neg`: each training positive's head with one `randrange` tail of the tail type, known edges not rejected | DL:315–327 | **confirmed** |
| 8 | `run_new.py` calls `get_train_neg(edge_types=[r])` inside the epoch loop | RUN:112–113 (once per epoch, before the batch loop) | **confirmed** |
| 9 | `run_new.py` sets no seed; HGB's training draws are unseeded | no `random.seed`/`np.random.seed` in `run_new.py` or `utils/`; the only seeds are DL:347, 418, 470 (`random.seed(1)` in the three test-set functions) | **confirmed** (consequence: after the first test relation's evaluation, the global generator continues from seed 1 for the next relation's training draws) |
| 10 | `get_valid_neg` re-drawn at every batch step (decision 4) | RUN:121 (batch loop), RUN:150–152 (`dl.get_valid_neg(edge_types=[test_edge_type])` inside it, before every validation pass) | **confirmed**: per batch step. Ours stays one draw per run (author decision 4); this is the line for the spec text |
| 11 | validation split: `train_ratio = 0.9`, first fact of every head kept in training | DL:39 (split at construction), DL:44–73; DL:56–59 (first fact of a head → train), DL:62–67 (`random.random() < train_ratio` → train, else validation), one global stream over the relations in `link.dat` order (DL:48–50), no minimum | **different** from ours (below) |
| 12 | `evaluate` | DL:240–269: ROC-AUC over all pairs; MRR = mean over heads of `1/(1 + rank of the best-ranked positive)`, heads without a positive skipped | read; not in Part N's summary and not in the card's oracle list (not compared) |
| 13 | construction of `links`, `links_test` | DL:516–540: one sparse matrix per relation, duplicates summed (`tocsr`, DL:295), relation meta from the first row of the relation; DL:40: `get_train_neg()` and `get_valid_neg()` are also called once at construction (they consume the global generator after the split) | read |

### Oracle tests (VERIFIED-RUN)

New `tests/test_hgb_oracle.py` (skips only if `vendor/HGB` is absent). It writes three tiny HGB-format datasets (`node.dat`, `link.dat`, `link.dat.test`, `info.dat`) and loads them with **HGB's own `data_loader`**, imported from its file path under the private name `_hgb_oracle_lp_data_loader`:
- `one_way_validation`: user 0–5, item 6–15, one relation user→item (validation facts stored one way);
- `head_without_candidates`: user 0–4, item 5–12, rel 0 user→item, rel 1 item→item among items 5–8 only; user 4 has no 2-hop item;
- `two_types_two_relations`: user 0–7, item 8–19, rel 0 user→item, rel 1 user→user stored one way.

Each is loaded with the global generator seeded to 0, 1, 2, 3 and 7 (15 cases). Our functions get HGB's split (`links` after the split + `links_test` both ways, `valid_pos` as stored). HGB's training/validation negatives are called with the global generator seeded to the case seed, ours with `random.Random(seed)`.

`python -m unittest -v tests.test_hgb_oracle` → **Ran 6 tests; FAILED (failures=15)** (`f1_oracle_output.txt`):

| test | HGB function | result |
|---|---|---|
| `test_datasets_cover_the_cards_cases` | HGB's loading | ok (user 4 has a positive and no 2-hop negative; 2 types × 2 relations; validation facts exist) |
| `test_two_hop_set_equals_get_test_neigh` | `get_test_neigh()` vs `hgb_test_negatives` | **ok: identical lists, identical order, 15/15** |
| `test_random_set_equals_get_test_neigh_w_random` | `get_test_neigh_w_random()` vs `hgb_random_test_negatives` | **ok: identical, 15/15** |
| `test_training_negatives_equal_get_train_neg` | `get_train_neg(edge_types=[r])` vs `hgb_train_negatives`, every relation | **ok: identical** |
| `test_validation_negatives_equal_get_valid_neg` | `get_valid_neg(edge_types=[r])` vs `hgb_valid_negatives`, every relation | **ok: identical** |
| `test_validation_split_equals_get_train_valid_pos` | `get_train_valid_pos` vs `_split_edge_index_hgb_by_head` (first relation, same seed) | **FAIL in 15/15** (e.g. `one_way_validation`, seed 0: HGB validation `[(2,8), (3,12), (3,13), (3,15), (5,13), (5,15)]`, ours `[(5,15)]`) |

### The difference (stop condition of F1)

- **HGB** (DL:55–67): over the relation's facts in CSR `nonzero()` order, the first fact of each head → training; every other fact draws `random.random()` from the global generator: `< 0.9` → training, else validation. One stream for all relations (DL:48–50). Validation may be empty (no minimum).
- **Ours** (`H2S-NN/src/data/link_prediction_dataset.py:325-360`, `_split_edge_index_hgb_by_head`): the same order and the same first-fact rule, but `rng.random() < val_ratio` (0.10) → validation (`:349`), a separate `random.Random(split_seed + relation_id * 9973)` per relation (`:432-435`), and a fallback that moves the last candidate into validation when none was drawn (`:354-358`).
- **Cause pinned (VERIFIED-RUN, `f1_split_diagnostic.py` → `f1_split_diagnostic_output.txt`):** replaying HGB's rule (`u ≥ 0.9` → validation, no fallback) with `random.Random(seed)` over our sorted facts reproduces HGB's validation set exactly in all 15 cases. So the fact order is the same; the draws differ (same probability, 0.10, opposite end of the unit interval), plus the fallback and the per-relation seeding.
- **Where it matters:** our loader uses the carve only when the release has no `link.dat.valid` (`:410-413`, spec v1.8 F9). All six local tasks ship one (LastFM 7,301 rows, PubMed 1,942, acm_pa 895, dblp_ap 1,768, freebase_bb 16,894, imdb_ma 1,330), so **none of them goes through the carve today**. HGB's loader never reads `link.dat.valid`: it always carves `link.dat` with the unseeded global generator. The docstring at `:414` says a file written from the carve reproduces it, so the LastFM and PubMed files were presumably written by our rule (UNVERIFIED: provenance of those files not checked in this card).
- **Not changed.** As the card says, our code was not changed. The failing oracle test stays in the suite as the record of the difference (skipping it or marking it an expected failure would be forbidden). The mutation checks of F1 (random-set order, two-way validation) were **not run** because F1 stopped. F2 and F3 continue; per the card, F4 waits on this.

**Commit:** F1 (partial): `H2S-NN/.gitignore`, `H2S-NN/vendor/HGB/CLONED_COMMIT.txt`, `H2S-NN/tests/test_hgb_oracle.py`, the evidence and this report.

## F2 — GC collapse check redefined (decision 1)

**Done** (`H2S-NN/scripts/partn_smoke.py`; NC and LP checks unchanged).
- `GCValidationCapture`: a forward pre-hook and a forward hook on the readout's **last Linear layer** (H2S-NN `GraphReadout.decoder`, `src/models/graph_readout.py`; GCN `SharedGraphHead.classifier`, `src/baselines/heads.py:105`) capture, only during the validation pass, its input (the pooled graph embedding) and its output (the logits) for every validation graph. One hook point for both models, the one the card names ("the readout's input to its Linear layer"). For GCN the input is `dropout(pooled)`, which is the pooled embedding in eval mode.
- `gc_validation_checks`: (c1) mean over logit coordinates of the std over graphs > 1e-6; (c2) the same for the pooled embedding; **both gate** (c) for GC. The predicted-class count is recorded per epoch, **not a gate**; the x* std over the nodes of the validation graphs stays recorded (Part N).
- Count check (closes Part N N7's UNVERIFIED point): captured rows must equal `len(fold.val_loader.dataset)` (H2S-NN) / `len(val_loader.dataset)` (GCN); a mismatch raises `ValueError` ("GC capture error"), which the script reports as a script error (exit 1), never as a model result.
- The H2S-NN GC capture no longer wraps `forward_batch`; it hooks the readout. The validation pass is still identified by `loader is fold.val_loader` inside `_run_epoch`.

**Tests (VERIFIED-RUN).** New `tests/test_partn_smoke.py` (6), with the real `GraphReadout` (mean pooling, one graph per batch): identical pooled embeddings for every graph → (c1) and (c2) fail; varied embeddings with a zero-weight Linear → (c1) fails, (c2) holds; varied outputs with one predicted class (bias +10/−10) → passes, class count 1; capture shorter than the validation set → `ValueError`; the capture records only while active; `checks` gates GC on (c1) and (c2) only. `python -m unittest -v tests.test_partn_smoke` → **Ran 6 tests, OK** (`f2_tests_output.txt`). No other module imports the script.

## F3 — torchmetrics as HetSheaf's lock (decision 6)

**Done.**
- Before (VERIFIED-RUN, `pip show`): torchmetrics **1.9.0**, torch 2.6.0+cu124, lightning-utilities 0.15.3.
- `python -m pip install "torchmetrics==1.8.2"` → exit 0; pip uninstalled 1.9.0 and installed **1.8.2**, changing no other package; after: torchmetrics 1.8.2, torch 2.6.0+cu124, lightning-utilities 0.15.3 (`f3_pip_install_output.txt`).
- Pinned `torchmetrics==1.8.2` in `H2S-NN/requirements/core.txt` (comment updated) and `H2S-NN/requirements/baselines-h2gb.txt` (the H2GB group also listed it).
- **Absent-class value:** `tests/test_gc_metrics.py`'s case (y = [0, 0, 1, 1], p = [0, 1, 1, 1], C = 3; class 2 in neither) gives **macro 0.733333 = 11/15 and micro 0.75 = 3/4 on 1.8.2, the same values as 1.9.0** (`f3_absent_class_value.txt`). The recorded value is unchanged; only the test's docstring now names 1.8.2 as the version in use.

**Tests (VERIFIED-RUN).** `tests.test_gc_metrics` → **Ran 3, OK** (`f3_gc_metrics_output.txt`). Modules that import the GC metric or read the requirement files: `test_gc_metrics`, `test_nc_metrics`, `test_data`, `test_gc_protocol`, `test_graph_classification_trainer`, `test_baselines` → **Ran 126; failures=2, errors=2**, all four in Part N N8's known set (vendor clones ×3, notebook) (`f3_tests_output.txt`); `test_wandb_and_runs` → **Ran 12, OK** (`f3_tests_wandb_output.txt`).
