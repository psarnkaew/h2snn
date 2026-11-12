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

## F4 — Smoke completed: **NOT RUN (waits on F1, card stop rule)** — superseded: F4 ran in the continuation below

The card says: "F1: any difference between HGB's code and ours (stop F1; F2–F3 may continue; F4 waits)." F1 found the validation-split difference, so none of the four runs (H2S-NN GC MUTAG; GCN NC acm; GCN LP pubmed; GCN GC MUTAG) was started. The F2 script is ready. When the author releases F4, the commands are (from `H2S-NN/`, env `h2snn`):

```
python scripts/partn_smoke.py --run h2snn:gc:mutag --device cuda --out ../docs/partNfix_evidence/f4/h2snn_gc_mutag.json
python scripts/partn_smoke.py --run gcn:nc:acm     --device cuda --out ../docs/partNfix_evidence/f4/gcn_nc_acm.json
python scripts/partn_smoke.py --run gcn:lp:pubmed  --device cuda --out ../docs/partNfix_evidence/f4/gcn_lp_pubmed.json
python scripts/partn_smoke.py --run gcn:gc:mutag   --device cuda --out ../docs/partNfix_evidence/f4/gcn_gc_mutag.json
```

**freebase_bb, for the lab server (decision 2; not run here).** From `H2S-NN/`, with the server's conda:

```
conda run -n h2snn python scripts/partn_smoke.py --run h2snn:lp:freebase_bb --device cuda --out ../docs/partNfix_evidence/f4/h2snn_lp_freebase_bb.json
```

Same configuration as Part N N7: template `configs/experiments/acm_pa_h2snn_v3b.yaml` with `dataset_name: freebase_bb`, `target_relation: [book, rel_0, book]`, `wisp_max_length: 4`, `wisp_walks_per_anchor: 8`, 2 epochs. Exit codes: 0 = checks recorded, 2 = non-finite loss or collapse, 1 = error.

## F5 — Full suite and report: **NOT RUN (follows F4)** — see the continuation below (still not run)

The full suite runs once, in F5, after F4. With F4 waiting, it was not run; the affected-module results per unit are above. Two things the F5 run will show:
- `tests.test_hgb_oracle.HGBOracleTests.test_validation_split_equals_get_train_valid_pos` fails (15 subtests). This is the F1 difference, kept as its record until the author decides.
- Known failures of Part N N8: `test_all_declared_vendor_clones_are_present` still fails, but its missing list dropped from 10 entries to 5. The clone resolves `hgb_nc`, `hgb_nc_gnn`, `hgb_nc_han`, `hgb_lp` and `hgb_lp_gnn`; still missing are `sehgnn`, `sehgnn_data`, `pshgcn`, `pshgcn_lp` and `chebnetii` (VERIFIED-RUN, `f3_tests_output.txt` vs `docs/partN_evidence/n8_full_suite_output.txt`). `test_colliding_vendor_module_names_do_not_leak`, `test_vendor_model_init_is_reproducible_under_seed_everything` (PSHGCN not cloned) and `test_notebooks_are_committed_clean` still fail.

## Per unit

| unit | commit | done | VERIFIED-RUN | deviations |
|---|---|---|---|---|
| F0 | `a5570f4` | branch `part-nfix` from `c46c1a4`; 8 docs by explicit path | — (no suite, per card) | none |
| F1 | `cf19946` | HGB cloned (`ca6fd5b`, not committed); 13 points read with file:line; `tests/test_hgb_oracle.py` | 4 functions identical to HGB in 15/15 cases; **split differs in 15/15** | **stopped**: mutation checks not run; our code not changed; the clone's `.git` removed so `CLONED_COMMIT.txt` can be staged |
| F2 | `5229957` | GC check = (c1) logits and (c2) pooled embeddings vary; class count recorded; count check | `test_partn_smoke`: 6 OK | none |
| F3 | `1c00bca` | torchmetrics 1.9.0 → 1.8.2, pinned in the core and H2GB groups | absent-class value unchanged; affected modules: only the 4 known failures | none |
| docs | `f3a8f7c` | designer's continuation docs (card, decisions log, v1.18 draft) by explicit path | — | none |
| F4 | `a75e4ec` | H2S-NN GC MUTAG run (passes the redefined check); GCN NC acm run | 2 runs, table in the continuation | **stopped at GCN NC acm** (NC check (c): one validation class); GCN LP pubmed and GCN GC MUTAG not run; three script fixes (GCN LP capture added, `BaselineParams.source`, exit code of a run error) |
| F1c | see the F1c section | split test replaced by the two tests of decision 2; F1's two mutation checks | oracle module: 7 tests OK; M1 caught; **M2 not caught by the oracle tests** (caught by the Part N hand-graph test) | **stopped** (continuation stop condition: an F1c mutation not caught); test data not changed; first version of the data test had a wrong expectation for the reverse relation (fixed, below) |
| F5 | — | not run (F4 and F1c stopped) | — | — |

## Decisions for the author

Settled by the author on 2026-11-12 (continuation): the split (option (a)); the test replacement; F4 released; decision 4's HGB line (`run_new.py:150-152`, inside the batch loop at `:121`) is in draft v1.18 Z-1. Open after the continuation:

1. **GCN NC acm, one predicted class (the F4 stop).** The NC collapse check is unchanged from Part N and gates on "≥ 2 classes predicted on the validation nodes". GCN NC acm predicts one class after each of its 2 full-batch steps; its x* spread (0.0126, 0.0155), losses, stop conditions and parameter movement pass. Should the NC check, like GC's (decision 1), record the class count without gating on it (e.g. gate on the spread of the logits over validation nodes instead), or is this a stop? GCN LP pubmed and GCN GC MUTAG wait on this, and F5 waits on both stops.
2. **M2 is not caught by the oracle tests (the F1c stop).** On the three tiny oracle datasets × 5 seeds, adding validation in both orientations changes no test head's 2-hop candidates and, in one case, one head's known-neighbour set without changing its random draws (`f1c_m2_diagnostic_output.txt`). So these datasets don't exercise validation orientation. The Part N hand-graph test `test_validation_orientation_changes_the_candidates_as_hgb` catches M2, but it is not HGB's code. Proposal: add a fourth oracle dataset in which a one-way validation fact opens a 2-hop path to a tail-type node from a test head (e.g. user→user validation u1→u2 with u1's items not reachable from u2 otherwise). Then M2 must make `get_test_neigh` and ours differ, and the mutation check is re-run. Needs your approval: it changes the card's oracle test data after a stop.

## Out-of-scope findings (not fixed)

- `run_new.py:188-194`: HGB's runner replaces the 2-hop test list by `<dataset>_ini_<r>_label.txt` (its hidden-label file) when that file exists in the data folder. None of our six tasks has one, so there is no effect here; worth a sentence in the spec's K1 text.
- `data_loader.py:366` casts `pos_links` to `int` when removing known pairs, so a link weight < 1 would leave a known pair among the 2-hop candidates. All weights in our six tasks are integers ≥ 1, so there is no effect here.
- After `get_test_neigh` (`random.seed(1)`), HGB's global generator continues from that state into the next test relation's training draws (multi-relation targets only).
- HGB's `evaluate` (DL:240-269) was read but not oracle-tested (it is not in the card's list); our `compute_link_prediction_metrics` was not compared with it.
- The provenance of `data/LP/LastFM/link.dat.valid` and `data/LP/PubMed/link.dat.valid` (presumably written by our carve) was not checked.
- Unchanged from Part N: the stand-alone import error of `tests/test_full_graph_early_stopping.py`; `LINK_PREDICTION_PRESETS` still holds `youtube`, `amazon` and `amazonbook`; the adapter files with line-ending-only differences (not staged); `docs/H2S-NN_partO_card.md` is untracked (not mine, not staged).

## Continuation after the F1 stop (author, 2026-11-12)

Author decisions: split option (a) (fixed files and split code unchanged; the differences go into the spec, draft v1.18 Z-4); the failing split test is replaced by the two tests of F1c; F4 runs now. HEAD verified at `649dbce`. Designer docs committed by explicit path: **`f3a8f7c`** "docs: Part N-fix continuation, split decision, v1.18 draft Z-1/Z-4" (`docs/H2S-NN_partNfix_card.md`, `docs/experiment_settings_decisions.md`, `docs/spec_v1.18_draft_edits.md`).

## F4 — Smoke completed — **STOPPED at GCN NC acm (check (c), NC: one predicted class)**

**Script changes before and during F4 (`H2S-NN/scripts/partn_smoke.py`; script fixes, not settings):**
1. **GCN link prediction had no capture.** The GCN branch handled NC and GC only (its `else` assumed GC), so `gcn:lp:pubmed` would have recorded no score spreads and exited as a false collapse. Part N never reached that run. Added: `train_link_prediction` is wrapped like the NC/GC loops; after each epoch's report, one evaluation `embed()` of that epoch's weights gives the std of the scores of the validation pairs and of each non-empty test set (positives + that set's negatives), and the validation BCE loss. No metric is computed.
2. **`source="smoke"` is not accepted by `BaselineParams`** (`src/baselines/protocol.py:61-63`: `published` | `tuned`). The first GCN NC acm attempt raised before training (`ValueError: BaselineParams.source must be 'published' or 'tuned', got 'smoke'`; kept as `f4/gcn_nc_acm_first_attempt_script_error.json`). Now `published` when GCN has published parameters for the cell, else `tuned`; the values are the published set or the adapter's defaults, with `epochs: 2` in both cases (`resolve_budget` reads it).
3. **Exit code of a run error.** That attempt exited 2 because "losses finite" was false on an empty list. Now a run error exits 1 unless a non-finite loss was actually recorded; exit 2 is reserved for a non-finite loss or a failed (c)/(c1)/(c2) check.
`tests.test_partn_smoke` → Ran 6, OK after each change.

**Runs (2 epochs, card configuration; VERIFIED-RUN, `docs/partNfix_evidence/f4/<run>.json`; trainers' console logs kept out of the evidence).** Device: NVIDIA GeForce RTX 4060 Laptop GPU.

| run | (a) losses finite (train steps) | (b) stop condition | (c) | (c1) logit std, epoch 0 / 1 | (c2) pooled std, epoch 0 / 1 | count (captured / expected) | classes predicted (val) | x* std mean | (d) params moved | (e) train loss epoch 0 → 1 | val loss epoch 0, 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H2S-NN GC MUTAG | yes (300) | none | **no collapse** ((c1) and (c2) hold) | 0.1241 / 0.1842 | 0.1143 / 0.1165 | 18 / 18, 18 / 18 | 1, 1 (recorded, not a gate) | 0.4401, 0.4055 | 50 / 50 | 0.5892 → 0.5460 | 0.7076, 0.8809 |
| GCN NC acm | yes (2) | none | **fails: 1 class predicted on the validation nodes in both epochs** (NC check unchanged from Part N: ≥ 2 classes and x* std > 1e-6) | — | — | — | 1, 1 | 0.0126, 0.0155 (> 1e-6) | 13 / 13 | 1.0970 → 1.0948 | 1.0938, 1.0919 |
| GCN LP pubmed | not run (F4 stopped) | | | | | | | | | | |
| GCN GC MUTAG | not run (F4 stopped) | | | | | | | | | | |

**Stop (card F4: "(c) for NC/LP as in Part N").** GCN NC acm predicts one class on the validation nodes after each of its 2 epochs; its x* spread passes (0.0126, 0.0155 > 1e-6), all 13 parameter tensors moved, both training steps and both validation losses are finite, no stop condition fired. F4 was stopped there; no setting was changed and nothing was re-run. Configuration: `gcn`, node classification, `acm`, published parameters for this cell: none, so the adapter's defaults (`hidden_dim` 64, `num_layers` 2, `dropout` 0.5, AdamW lr 5e-4, wd 1e-4, `feats_type` = the HGB default for ACM), `epochs` 2, seed 42, `run_index` 0, full-batch training (one step per epoch). As in Part N, x* for GCN is its last layer's output over the target nodes (the vendor GCN has no separate embedding). I read the card's NC check as still gating on the class count (decision 1 redefines GC only) and the general "a single predicted class is not a stop" as the GC case; the author's call is listed under decisions. No interpretation of the cause is offered (2 full-batch steps).

The H2S-NN GC MUTAG run reproduces Part N's training and validation losses to 4 decimals (same seed, same configuration) and now passes (c) as redefined; its captured-graph count equals the validation set (18), which closes Part N's UNVERIFIED point.

## F1c — close F1 — **STOPPED: mutation M2 not caught by the oracle tests**

**Test replacement (author decision 2; our split code and data unchanged).** In `H2S-NN/tests/test_hgb_oracle.py`, `test_validation_split_equals_get_train_valid_pos` is replaced by:
- (i) `HGBOracleTests.test_hgb_rule_replayed_over_our_facts_equals_get_train_valid_pos`: HGB's rule replayed over our sorted facts reproduces HGB's `valid_pos` and `train_pos` exactly, for every relation of the three tiny datasets × 5 seeds. The rule: one `random.Random(seed)` stream over the relations in `link.dat` order, first fact of a head to training, else `u ≥ 0.9` to validation, no minimum. The test also checks that HGB's relation order is the `link.dat` order. The replay moved here from `f1_split_diagnostic.py`. It documents draft v1.18 Z-4 (fact order and rule agree; only the draw differs).
- (ii) `FixedValidationFileTests.test_loader_reads_the_fixed_validation_file_exactly` (no HGB needed, so outside the skip-if-absent class), on all six local tasks:
  - `link.dat.valid` exists and lists the **target relation only** (F13; checked: acm_pa 895 rows of rel 2, dblp_ap 1,768 of rel 0, imdb_ma 1,330 of rel 2);
  - the loader's validation facts of the target equal the file's rows exactly (no duplicates);
  - on acm_pa, dblp_ap and imdb_ma, the one stored reverse relation's validation facts equal those rows flipped (`_file_validation_facts`, `link_prediction_dataset.py:378-383`; decision 5);
  - for these relations, training and validation are disjoint and their union is the `link.dat` facts;
  - **every other relation's facts are all in the message graph**.
- No other test was removed. The unused import of `_split_edge_index_hgb_by_head` was dropped.
- **Deviation (my error, fixed):** the first version of (ii) expected the reverse relation's rows in the file. It failed on the three derived tasks (`AssertionError: … ('author', 'rel_3', 'paper')` for acm_pa). The files list the target only, and the loader derives the reverse copies, so the test now states that.

`python -m unittest -v tests.test_hgb_oracle` → **Ran 7 tests, OK** in 43 s (`f1c_oracle_output.txt`).

**F1's mutation checks (VERIFIED-RUN, `f1c_mutation_check.py` → `f1c_mutation_output.txt`; each applied to `src/data/hgb_lp_negatives.py` in place, oracle class run, original restored):**

| mutation | result |
|---|---|
| M1: random set with the test positives in another order (reversed) | **caught**: `test_random_set_equals_get_test_neigh_w_random` fails in 15/15 cases |
| M2: 2-hop adjacency with validation in both orientations | **NOT caught**: Ran 6, OK |
| restored | clean (Ran 6, OK); `git status` shows `hgb_lp_negatives.py` unchanged |

**Why M2 is not caught (VERIFIED-RUN, `f1c_m2_diagnostic.py` → `f1c_m2_diagnostic_output.txt`).** For every dataset × seed I compared the one-way adjacency with the both-ways one. No test head's 2-hop candidates change in any of the 15 cases. One head's known-neighbour set changes in one case (`two_types_two_relations`, seed 2), and its random draws never hit the affected tail. The reasons:
- in `one_way_validation` (user→item only), a user's 2-hop reach contains no item at all;
- in the other two datasets, the reversed validation facts open no new head→tail-type path.

My F1 datasets meet the card's letter ("a validation fact stored one way") but not its purpose (orientation must matter).

**M2 against the Part N hand-graph test (VERIFIED-RUN, `f1c_m2_part_n_test_output.txt`):** caught. `tests.test_hgb_lp_negatives.TwoHopCandidateTests.test_validation_orientation_changes_the_candidates_as_hgb` fails (Ran 20, failures=1); restored. That test compares with a re-implementation, not with HGB's code.

**Stop.** "An F1c mutation not caught (stop, report)." The oracle datasets were not changed. The proposal is under "Decisions for the author" (2).

**Commit:** F1c: `H2S-NN/tests/test_hgb_oracle.py`, the F1c evidence and this report.

## F5 — Full suite: **NOT RUN**

F5 waits: F4 stopped (GCN NC acm) and F1c stopped (M2). The full suite runs once, after both are resolved. Expected from the affected-module runs:
- `test_hgb_oracle`: 7 OK (the old split test is gone).
- `test_partn_smoke`: 6 OK; `test_gc_metrics`: 3 OK.
- `test_baselines`: the 4 known failures. `test_all_declared_vendor_clones_are_present` now lists 5 missing clones instead of Part N's 10: SeHGNN ×2, PSHGCN ×2, ChebNetII.
