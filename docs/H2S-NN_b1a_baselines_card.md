# Card 1f — B1a on the baseline side: fixed validation subset in the shared H2GB NC loops

Written 2026-11-17 by the design session. Author decision **B1a** (decisions log row "Card 2 review", 2026-11-16). For session 1. **Amended 2026-11-18 (author, "รวม 1f"):** runs on `baselines-h2gb-6` right after card 1e F3, after merging `v3b` (SHA in the prompt); one suite (U4) and one review cover card 1e F3 and card 1f. Report `docs/H2S-NN_b1a_baselines_report.md`; evidence `docs/b1a_evidence/`.

## Why
Card 2 found that whole-split validation at every evaluation epoch dominates H2GB NC trial cost (mag-year 128 GPU-h per trial for H2S-NN at the largest point). B1a: on the 8 H2GB NC datasets, validation **during training** (checkpoint selection; tuning objective) uses a fixed subset of the validation split of H2GB's own size, drawn once with a fixed seed, **the same nodes for every method** (baselines and H2S-NN); the selected checkpoint is evaluated on the **whole** validation and test split (MB3 kept for reported numbers).

| Dataset | Subset size (`min(split, size)`) |
|---|---|
| ogbn-mag | 16,384 |
| mag-year | 8,192 |
| oag-cs, oag-eng, oag-chem | 4,096 |
| ieee-cis-g | 8,192 |
| pdns | 8,192 |
| h-pokec | 16,384 |

## Current code (v3b `1867681`, read by the design session)
- `src/baselines/minibatch.py:214-265` `train_minibatch_node_classification` (class A/C: HGTLoader batches) and `:304-357` `train_index_minibatch` (class B: row batches) evaluate the **whole** validation **and test** split at every evaluation epoch (`is_eval_epoch`, `:76-78`) and keep the test metrics of the best validation epoch.
- Callers: `_h2gformer_h2gb.py:157`, `hgb_han.py:453`, `hetsheaf.py:785` (A/C); `_pshgcn_mag.py:215`, `chebnetii.py:284`, `hinormer.py:305`, `sehgnn.py:389` (B). Every other H2GB NC adapter must be found and listed (U0).

## Units (one commit each)
### U0 — Inventory (no code)
List every H2GB NC training path of every registry method (file:line), and which shared loop it uses. A path with its own loop (not one of the two above) is a **stop item** (report it; no change).

### U1 — The subset (new file, shared with H2S-NN)
New module `src/evaluation/validation_subset.py` (add-only; outside `src/baselines/` by design, so card 5 imports the same function for H2S-NN):
- `validation_subset(dataset: str, val_ids: Tensor) -> Tensor`: sorted ids; size from the table (`min(len(val_ids), size)`); drawn with `torch.Generator().manual_seed(SEED)` (`SEED` a module constant, 0) by `randperm` over `val_ids` sorted ascending, first `size`, then sorted. Depends only on the dataset's validation ids, never on the method, the trial or the training seed.
- `subset_fingerprint(ids) -> str` (sha256 of the int64 bytes).
- Write `docs/b1a_evidence/subset_fingerprints.json`: per dataset, split size, subset size, fingerprint (from the real `data/NC/` splits; CPU job). Card 5 checks H2S-NN against this file.
- PDNS: state which validation set the loop uses (temporal protocol) and draw from that.

### U2 — The loops
In both shared loops:
1. At an evaluation epoch, evaluate the **subset only** (class A/C: an HGTLoader over a mask of the subset ids, H2GB's loader called unchanged; class B: the subset rows). **No test evaluation during training.**
2. Selection: best subset Macro-F1 (ties: earliest epoch, as now); keep a CPU copy of the best epoch's `state_dict`.
3. After the last epoch: load the best weights; evaluate the **whole** validation and test splits exactly as `evaluate_split` / `evaluate` do now (every target once).
4. Record: whole-split val/test metrics (as now) **plus** `val_subset_macro_f1` at the best epoch and `val_subset_size`, `val_subset_fingerprint`. The tuning objective (what the sweep reads) is `val_subset_macro_f1`.
5. Datasets outside the table are refused by `validation_subset` (no silent whole-split fallback).

### U3 — Tests and mutations (`tests/test_b1a_validation_subset.py`) (cut 2026-11-18, author: keep (1) determinism + same nodes for every method, as one test; (2) no test pass during training; (3) best weights restored + Mutation 1; existing tests. The other rows below are dropped)
| Test | Expectation |
|---|---|
| Subset determinism | two calls equal; size per table; subset ⊂ val ids; independent of training seed and method |
| Same nodes for every method | one A/C adapter and one B adapter on a synthetic H2GB-shaped graph see identical subset ids (captured) |
| No test pass during training | instrumented loader / `forward_rows`: test targets touched only after the last epoch, once |
| Best weights restored | toy model whose weights differ by epoch: the record's test metric equals the metric of the best-subset epoch's weights recomputed independently |
| Whole-split final evaluation | final val/test pass covers every target once (existing `evaluate_split` refusal still active) |
| **Mutation 1** | use last-epoch weights instead of best → "best weights restored" fails |
| **Mutation 2** | draw the subset with the training seed → determinism / same-nodes test fails |
| Existing tests | `test_baselines*`, golden and mb0 `differing: none` (H2S-NN untouched) |

### U4 — Suite (amended 2026-11-18, author: the real-data step is dropped; the first tuning trial on TSUBAME is the real-data check)
~~One H2GB NC dry-run step per loop class~~ (dropped). Full suite once **as a CPU job** (no GPU request; CUDA-only tests skip; allowed because H2S-NN `src/` is untouched: golden/mb0 cannot change) (author 2026-11-18). Report short; last line `branch=baselines-h2gb-6 sha=<full head> card=docs/H2S-NN_baselines_freebase_sparse_card.md+docs/H2S-NN_b1a_baselines_card.md base=<v3b sha>`.

## Files
`src/baselines/minibatch.py`, `src/evaluation/validation_subset.py` (new; `src/evaluation/__init__.py` only if the package does not exist), adapters only if a caller must pass the dataset name (stated per file), `tests/test_b1a_validation_subset.py`, `docs/b1a_evidence/**`, the report. Anything else: stop and ask. Never `src/trainers/`, `src/config.py` (card 5).

## Rules and stop conditions
- No metric is computed for comparison; no performance run.
- Resource rule (`working_preferences.md`): `--mem` from recorded peak + 30 %, ≥ 16G; no A6000 / g3an01; one cell per job on a 24 GB GPU not shared for timing.
- Stop: a path with its own loop (U0); a loader that cannot take a subset mask without editing H2GB; PDNS validation not a single id set.
- Git: one commit per unit; merge (never rebase) `v3b`; stage by explicit path; never stash/reset others' work; push only the branch; wait for review.

### U5 — Review follow-up (2026-11-18, design session; decisions log row "Card 1e F3 + 1f review")
On `baselines-h2gb-6` from `08b5a16`. No GPU job; nothing trained on real data.
1. **Merge** `origin/v3b` (`714b08d`; merge, never rebase).
2. **Suite fixes** (files added to this card's list): `tests/test_baselines.py:52` total runnable cells 254 → 253; regenerate the notebooks with `scripts/generate_baseline_notebooks.py` and commit only the generator's output.
3. **PDNS (stop item, decided (a)):** the subset is drawn from **each run seed's** validation split: `validation_subset(dataset, split_ids, run_seed)` for PDNS takes the validation ids of that seed's split and draws the B1a size with an RNG seeded by (fixed subset seed, run seed); the other 7 datasets keep the single fixed subset. Same run seed → same nodes for every method. Record the per-seed fingerprint. Tests: same seed and split → identical subset across two calls and across both shared loops; different run seeds → subsets inside their own validation split; mutation: draw from the seed-0 split for every seed → the inside-split test fails.
4. **Hetero²Net (stop item, decided (a)):** apply the U2 rule inside its own loop (subset during training, no test pass during training, best weights restored, one whole-split val/test pass at the end); one test as U3 (no test pass during training) with Mutation 1.
5. **Sweep objective (card text holds):** the end-of-trial value the sweep reads is `val_subset_macro_f1` at the best epoch (`wandb_campaign.py:434`, added to this card's list); whole-split val/test stay recorded. One test: the sweep's objective key equals the subset metric.
6. **Amended (author, 2026-11-18): no full suite here.** Run only the new/changed tests and the two tests that failed (`test_campaign_matrix_matches_planned_cell_counts`, `test_committed_notebooks_match_the_generator`) as one short CPU job (no `--gres`); the full suite runs once at the freeze. Report section "U5"; last line `branch=baselines-h2gb-6 sha=<sha> card=docs/H2S-NN_baselines_freebase_sparse_card.md+docs/H2S-NN_b1a_baselines_card.md base=714b08d`.

### U6 — Review follow-up of U5 (2026-11-18, ops session; author chose option A; decisions log row "Card 1f U5 review")
On `baselines-h2gb-6` from `6878cd3`. No GPU job; nothing trained on real data.
1. **Merge** `origin/v3b` (`80e045a`, card 2 D4b; merge, never rebase).
2. **Trial selection = sweep objective (option A).** `scripts/run_baseline_tuning.py: select_best_trial` (`:194-200`, added to this card's list) ranks trials by `sweep_objective(record, monitor)` (`src/utils/wandb_campaign.py`), not by `metric_score`; ties keep the first trial. So the seed-0 sweep, the trial picked for `--final_seeds` and the sweep's end-of-trial value use one criterion: `val_subset_macro_f1` on the 8 H2GB NC datasets, the whole-split monitor elsewhere (HGB NC, LP, GC; unchanged).
3. **No silent fallback.** `sweep_objective` raises when the record's dataset is in `validation_subset.SUBSET_SIZES` and the monitor is macro-F1 but `val_subset_macro_f1` is missing (message names card 1f U6). Datasets outside the table keep `metric_score`.
4. **Tests** (add to `tests/test_baseline_final_seeds.py` or `test_b1a_validation_subset.py`): (i) two H2GB NC trial records whose subset and whole-split rankings disagree → `select_best_trial` returns the subset winner, and an HGB record pair → the whole-split winner; (ii) an H2GB NC record without the subset key → `sweep_objective` raises. **Mutation:** `select_best_trial` back to `metric_score` → (i) fails.
5. Targeted tests (the U5 set + the new ones + `test_baseline_final_seeds`, `test_wandb_and_runs`, `test_failed_trials`) as one CPU job (no `--gres`, 16G). Report section "U6"; last line `branch=baselines-h2gb-6 sha=<sha> card=docs/H2S-NN_baselines_freebase_sparse_card.md+docs/H2S-NN_b1a_baselines_card.md base=80e045a`.
