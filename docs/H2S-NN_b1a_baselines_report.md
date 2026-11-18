# Report — card 1f (B1a on the baseline side) and card 1e F3: combined review

Branch `baselines-h2gb-6`. Merged v3b `a8828a8` (`28226ce`) and design-docs `6da1199` (`e19d54e`). Session S1. No GPU job; no model was run on real data; no metric was computed. The evidence is in `docs/b1a_evidence/`. The card 1e report is `docs/H2S-NN_baselines_freebase_sparse_report.md`.

## Result: suite FAILED (stop condition). 741 of 743 OK, 2 failures, both caused by card 1e F3.2

U4 ran as job 17649, CPU only, `--gres=none --mem=32G`, on g3n01 at `5e9b9eb`, and took 1,856 s. Log: `u4_suite.txt`. Result: `Ran 743 tests ... FAILED (failures=2, skipped=1)`. The one skip is the CUDA-only test `test_cuda_float32_matches_cpu_float64_reference`.

| Failing test | Cause | Fix (not applied: the card says stop) |
|---|---|---|
| `test_baselines.RegistryTests.test_campaign_matrix_matches_planned_cell_counts` | `253 != 254`. F3.2 made HAN `freebase_bb` N/A. I updated the per-task LP count (76 → 75) but missed the total runnable count. | `test_baselines.py:52`: change 254 to 253, with "card 1e F3.2: − HAN freebase_bb". |
| `test_baselines.LaneNotebookTests.test_committed_notebooks_match_the_generator` (lane `link_prediction_dgl`) | The committed notebook still lists `freebase_bb` for HAN; the generator no longer does. | Rerun `scripts/generate_baseline_notebooks.py` and commit `link_prediction_dgl.ipynb`. That file is **outside both cards' file lists**, so it needs approval. |

None of the failures is in card 1f's code: the B1a tests, the loop tests and every adapter test pass. Proposed next step: apply both fixes, then run the suite once more as a CPU job.

## Card 1e F3 (details in the 1e report)

| Unit | Commit | |
|---|---|---|
| F3.1 | `96d53b8` | `src/baselines/runtime/blocked_spmm.py` (row blocks of ≤ 2^26 − 1 nnz, `narrow` + same `torch_sparse.matmul` + `cat`), bound in `_h2sgnn_lp.py` through `bound_attribute`; vendor unedited. Tests: forward `torch.equal`, value/x gradients atol 1e-6 on uneven blocks; binding restored (also on exception). |
| F3.2 | `a56a801` | HAN NC `freebase` `num_heads` {4} (DERIVED); HAN LP `freebase_bb` N/A. Origin of both suite failures. |
| report | `72f9eb8` | `docs/H2S-NN_baselines_freebase_sparse_report.md` |

## Card 1f

| Unit | Commit | Content |
|---|---|---|
| U0 | `912e229` | `u0_inventory.md`: every wired H2GB NC path, with file:line. Class A/C uses `train_minibatch_node_classification`: GCN, GAT, NSD and CSNN through `fit_homogeneous_on_hgt`, plus HAN, H2G-former and HetSheaf. Class B uses `train_index_minibatch`: HiNormer, SeHGNN, ChebNetII, PSHGCN-MAG. **Stop item: Hetero2Net** has its own loop (`hetero2net.py:519-571`). It is unchanged and still validates on the whole split, test included, at each eval epoch. |
| U1 | `142b077` | `src/evaluation/validation_subset.py` (new package): `validation_subset` (SEED 0, `randperm` over the sorted val ids, first `min(split, size)`, sorted) and `subset_fingerprint` (sha256 of the int64 ids). `subset_fingerprints.json` comes from the real splits (CPU job 17644, peak 7.7 GiB). **PDNS is refused**, a stop item: see below. |
| U2 | `6a566dd` | Both loops. At an eval epoch they score the subset only, with no test pass. They keep the best subset macro-F1 epoch (ties: the earliest) as a CPU `state_dict`. After the last epoch they restore it and run one whole-split val/test pass (`evaluate_split` coverage check unchanged). The record adds `val_subset_macro_f1`, `val_subset_size` and `val_subset_fingerprint`, and `report_epoch` gets the subset score. A/C: `val_subset_loader` runs H2GB's loader unchanged on a shallow copy of the graph whose target `val_mask` holds only the subset. It is passed in by `fit_homogeneous_on_hgt`, `hgb_han.py`, `_h2gformer_h2gb.py` (built before the label channel rewrites the masks) and `hetsheaf.py`. B: the subset comes from `plan.dataset`, so no caller changed. |
| U3 | `5e9b9eb` | `tests/test_b1a_validation_subset.py` has 3 tests. (1) Determinism and size per table, a subset of the val ids, independent of the training seed; the A/C and B loops evaluate the identical subset ids. (2) Test targets are touched only in the final pass, once each. (3) Best weights restored: weights are scripted per epoch, the best epoch is 1 and the last is 4, and the test metric equals an independent recomputation. **Mutation 1** (`load_state_dict` made a no-op in-process, no file edited) is **caught** in both loops. Job 17645: 12 tests OK. |
| U4 | — | suite FAILED, see the top of this report |

### Subset sizes (real splits, `subset_fingerprints.json`)

| Dataset | val split | subset |
|---|---|---|
| ogbn-mag | 64,879 | 16,384 |
| mag-year | 184,097 | 8,192 |
| oag-cs | 19,551 | 4,096 |
| oag-eng | 28,524 | 4,096 |
| oag-chem | 59,664 | 4,096 |
| ieee-cis-g | 14,423 | 8,192 |
| h-pokec | 408,160 | 16,384 |
| pdns | — | refused |

## Decisions for the design session

1. **PDNS (stop condition: "PDNS validation not a single id set").** PDNS's 70/20/10 split is drawn from the run seed (`src/data/h2gb_nc.py:74-81`; H2GB `pdns_dataset.py:302-312`), so the validation ids change with the seed. `validation_subset("pdns", …)` raises an error instead of falling back silently. **As a result, every PDNS cell on the shared loops now fails at start** until this is decided. Options:
   - (a) Draw the subset from each seed's validation split with the fixed SEED. It is then the same for every method at a given seed, but it depends on the run seed.
   - (b) Fix PDNS's split to one seed for all runs. This changes the split protocol.
   - (c) Use the whole validation split for PDNS, as a stated exception.
2. **Hetero2Net (U0 stop item).**
   - (a) Apply the U2 rule inside `hetero2net.py::_fit_h2gb`: about 1–2 h plus one test.
   - (b) Keep whole-split validation and state the difference.
3. **The sweep's final objective.** The per-epoch W&B series now carries the subset score, through `report_epoch`. The end-of-trial value `best/macro_f1` (`src/utils/wandb_campaign.py:434`, `metric_score(record, monitor)`) still reads `val_macro_f1`, which is now the whole-split validation score at the selected checkpoint. Making the sweep read `val_subset_macro_f1`, as the card states, needs a change in `campaign.py` or `wandb_campaign.py`, which is outside the card's files.
4. **The two suite fixes above**, including the notebook file outside the file list.

## Notes

- **Resource rule:** CPU jobs at 16G (17548/17552/17644/17645 for tests) and 32G (fingerprints, suite). No GPU requested, no A6000, no g3an01.
- **Jobs:** 17547 (F3 tests) was cancelled before it started, since U4 covers F3. 17548 and 17552 failed at setup: `h2snn-b` has no pytest, and `PYTHONPATH` was not set. They were resubmitted as 17644/17645 with `unittest` and `PYTHONPATH`; the logs were overwritten by the reruns.
- `src/trainers/`, `src/configs/config.py` and vendor code were not touched. H2S-NN `src/` is untouched apart from the new `src/evaluation/`.

(The 08b5a16 review line, superseded by the U5 section below: branch=baselines-h2gb-6 sha=5e9b9eb7dc0af47acc3ea0f40ca7df1c3141c254, base a8828a8.)

## U5 — review follow-up (2026-11-18; decisions log row "Card 1e F3 + 1f review")

Merged design-docs `8acd764` (`card 1f U5` from `9961a0d`) and v3b `714b08d` (merge, no rebase). No GPU job, no real data, no full suite (author: the full suite runs once at the freeze).

| Item | Commit | Content |
|---|---|---|
| U5.2 suite fixes | `ef84452` | `test_baselines.py:52` total runnable cells 254 → 253; notebooks regenerated by `scripts/generate_baseline_notebooks.py` (CPU job 17659). The only output that changed is `link_prediction_dgl.ipynb`. |
| U5.3 PDNS = (a) | `278386a` | `validation_subset(dataset, val_ids, run_seed)`. PDNS draws from that run seed's validation split, with the RNG seeded by `subset_generator_seed(run_seed)` (sha256 of `SEED` and the run seed). A missing run seed raises an error. The other 7 datasets keep the single fixed subset and ignore `run_seed`. Both loops take `run_seed`, and callers pass `bundle.metadata["seed"]`, the seed `h2gb_graph` uses to draw the split (A/C: `fit_homogeneous_on_hgt`, HAN, H2G-former, HetSheaf; B: SeHGNN, ChebNetII, HiNormer). The per-seed fingerprint is the run record's `val_subset_fingerprint`; `subset_fingerprints.json` states the rule for PDNS. |
| U5.4 Hetero2Net = (a) | `31abed2` | Its loop moved into `hetero2net.train_h2gb_epochs`. Eval epochs score the subset only, through a NeighborLoader over the subset ids on `train_data`, which is their validation graph. There is no test pass during training. The best weights are restored, then one whole-split val/test pass runs. |
| U5.5 sweep objective | `dcbb6ef` | `wandb_campaign.sweep_objective` logs `val_subset_macro_f1` at the best epoch as the end-of-trial sweep metric (`:434`) when the record carries it and the monitor is macro-F1. Otherwise it falls back to `metric_score`, so H2S-NN is unchanged. Whole-split val/test stay in the record and in the `best/*` summary. |
| U5.6 tests | `6edaeb3` | One CPU job, no `--gres` (17660, `u5_tests.txt`). It ran the new/changed tests (`test_b1a_validation_subset`, `test_baselines_minibatch`), the two U4 failures, and the existing tests of the changed `wandb_campaign.py` (`test_wandb_and_runs`, `test_failed_trials`): **Ran 33 tests, OK**. Mutations run in-process with no file edited (`u5_mutations.py`): **PDNS drawn from the seed-0 split for every seed → CAUGHT** (outside its own split, seeds 1–2, both loops); **Hetero2Net Mutation 1 → CAUGHT**. |

New tests (`tests/test_b1a_validation_subset.py`, `B1aReviewTests`):
- **PDNS:** with the same seed and split, two calls agree, and so do the A/C and B loops. For seeds 0–2, each subset lies inside its own validation split, and the subsets differ across seeds.
- **Hetero2Net loop:** the test and whole-validation passes each run once, after the last epoch. Best epoch 1 is restored while the last epoch is 4, and the test metric equals an independent recomputation.
- **Sweep objective:** `sweep_objective` returns the subset metric. The worker logs it, and the worker no longer calls `metric_score`.

### Open (outside the card's files; not changed)
- `scripts/run_baseline_tuning.py:198` picks the best trial after a sweep with `metric_score`, which reads `val_macro_f1` (now the whole-split validation at the selected checkpoint), not `val_subset_macro_f1`. If the selection should match the sweep objective, that line needs the same rule: a one-line change plus one test.
- Hetero2Net, PDNS and H2SGNN Freebase have not run on real data under the new rules. The first TSUBAME trial is the real-data check.

branch=baselines-h2gb-6 sha=6edaeb390b5e1488c54902851bf1324dfae08bf5 card=docs/H2S-NN_baselines_freebase_sparse_card.md+docs/H2S-NN_b1a_baselines_card.md base=714b08d
