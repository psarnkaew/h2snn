# Card 8 report: published configurations, R-GCN and MAGNN adapters, TSUBAME lists

**Verdict: done.** C0–C7 are complete after amendments 1–6 and the review's C7 addition (`published_seed` rows in
`collect.py`). The final CPU job (18038, 36G, no GPU) ran 302 targeted tests on the merged head a58d19f: all OK, none
skipped. All 7 card mutations are caught. H2S-NN is untouched:
`git diff fc0ebaa -- H2S-NN/src/models H2S-NN/src/trainers H2S-NN/configs/experiments` is empty. No real-data run.

Branch `card8` = `v3b` fc0ebaa plus merges of design-docs b5adecf, 790ff68, 377f16e, 3489b8e and 5f6a1e6 (merge commit
15794d6, amendment 6), and of `v3b` e5bdf10 (merge commit b9f8b23, card 9). Both merges were clean. `make_cell_list.py`
keeps both sides: card 9's `ablation` changes and card 8's `published` / `baselines-2` / `probe` phases.

## 1. Units

| unit | commit | job | result |
|---|---|---|---|
| C0 inventory (+ amendments 1, 3, 4, 5) | b18f3ec, 1000b0c, 5683753, 0bc171a | — | 73 filled cells (NC 49, GC 24); table in `docs/card8_evidence/c0_inventory.md` |
| C1 published configurations | c6bfd39, 9b4c55d | 17963, 17987 | 23 + 13 tests OK; mutation "scheduler dropped from the H2GB mapping" caught |
| C2b LP step size, DistMult | d68f9c5 | 17989 | 16 OK; mutations "one step per epoch" and "diagonal DistMult" caught |
| C2 R-GCN | ba4cdea | 17997 | 8 OK |
| C3 MAGNN | 49517de | 18002 | 19 OK; mutation "instances on the full graph" caught |
| C4 registry, search spaces | 836ad74 | 18013 | 119 OK |
| C5 TSUBAME lists | 31eed94 | 18014 | 13 OK; mutation "no filled exclusion" caught |
| C6 targeted job | a44218d | 18015 | 242 OK, 0 skipped; all 5 mutations caught |
| C7 W&B logging + `published_seed` rows | d36caaa, a58d19f | 18025, 18034, 18037 | mutations "no `test/*` in the HGB NC payload" and "published records skipped by collect" caught; 43 + 22 OK |
| final targeted job (merged head) | a7025b7 | 18038 | 302 OK, 0 skipped; all 7 mutations caught |

## 2. Cell statuses and lists

| task | filled | published | tune | probe | N/A |
|---|---|---|---|---|---|
| NC (15 × 12) | 49 | 33 | 89 | 8 | 1 |
| LP (12 × 8) | 0 | 5 | 88 | 2 | 1 |
| GC | 24 | — | — | — | baseline cells not run (option (a)) |

| phase | lines | `node_f` tasks |
|---|---|---|
| `h2snn` | unchanged (byte-identical to fc0ebaa, tested) | — |
| `baselines-early` | 4 (byte-identical) | 1 |
| `baselines-1` (rebuilt: tune cells only) | 61 | 16 |
| `baselines-2` (new) | 112 | 28 |
| `published` (new: `run_baseline_published.py … --final_seeds 0 1 2 3 4`) | 38 | 10 |
| `probe` (new: 1 trial, `--dry_run --grid_point min`) | 10 | 3 |

The tuning lines (4 + 61 + 112 = 177) equal the 89 + 88 tune cells.

## 3. Adapters changed by C2b

There is no adapter-specific LP epoch loop. Every small-LP adapter trains through the shared `training.train_link_prediction` with `**budget_kwargs(LINK_PREDICTION, params)`, so the batch of 8,192 positives per step applies to all of them:
- `hgb_gnn.py` (GCN, GAT);
- `_homogeneous_lp.py` (NSD, CSNN);
- `hgb_han.py`, `pshgcn.py`, `h2sgnn.py`, `hetero2net.py`, `h2gformer.py`, `hetsheaf.py`;
- `rgcn.py`, `magnn.py` (new).

A static test checks this. Large LP (`fit_large_lp`) was already per 8,192 and is unchanged. The shared `DistMultDecoder` is now a full `[R, d, d]` matrix per relation (HGB's init), and every LP baseline uses it.

## 4. Decisions and deviations

- **Published H2GB cells (30):** each runs H2GB's own model and the yaml's optimizer, cosine schedule, clipping, accumulation, epochs, eval period and fan-out.
  - Data, evaluation and logging keys stay ours (`h2gb_published.EXCLUDED_KEYS`; protocol option 1).
  - On the binary datasets the network has `C` outputs with cross-entropy (the existing stated difference).
- **GCN × PubMed/LastFM LP (amendment 4):** run in our loop with validation once per epoch. HGB's patience of 3 validations is applied as 3 epochs.
- **MAGNN:**
  - ACM/DBLP/IMDB NC are filled and not wired: ACM's preprocessing adds self-loops and IMDB uses full-batch `MAGNN_nc`. The registry carries the reason.
  - Off HGB, MAGNN's instances are enumerated by our code in MAGNN's format; this is tested equal to HGB's `get_adjlist_pkl`.
  - The LP sampler re-implements `parse_adjlist_LastFM` for generic endpoint types, because HGB's version hard-codes the LastFM types and `.to('cuda')`.
  - When a whole batch has no instance of a meta-path, each target gets one self-instance (MAGNN LP's own rule). Without this, MAGNN's code crashes on an empty reshape (`base_MAGNN.py:150`).
  - Large LP (probe cells) uses MAGNN's NC layer on each sampled subgraph, without the training mask.
- **R-GCN:**
  - HGB's `EntityClassify` is rebuilt on HGB's own `BaseRGCN` with its three builders verbatim, because `entity_classify.py` calls `pynvml.nvmlInit()` at import.
  - The LP width grid is {40, 60, 80, 120}, because `"bdd"` needs width divisible by `n_bases`; HGB's default 64 is left out.
  - Tuned R-GCN cells on H2GB use H2GB's own R-GCN with the tuned keys, as HAN's class-A path (there are no such cells today).
- **GC option (a):** every baseline GC cell is unsupported with that reason, and the code is kept. The GC lane notebooks are no longer generated; the old files are kept because the repo rules say notebooks are not deleted.
- **Existing tests updated, with reasons in comments:**
  - registry counts (runnable 271);
  - the contract and published-runnable tests skip GC option (a);
  - `SearchSpaceCoverageTests` checks only files of tuned cells;
  - the large-LP method list gains R-GCN and MAGNN;
  - `test_fanouts_equal_h2gb_yaml` expects 40 yamls instead of 32;
  - the two TSUBAME list tests expect tune cells only.
- **Bugs found and fixed in C4:**
  - The R-GCN/MAGNN vendor blocks left HGB's `utils` and `model` modules in memory; this broke HAN in one process. The blocks now also push their HGB root, and a regression test covers it.
  - The PSHGCN LastFM yaml from C2b did not parse.
- **Memory:** jobs touching `test_partP_freebase` need 32G (peak 28 GB, the known Freebase-test peak). Interim jobs used 24–32G; C6 used 36G.

## 5. C7 (amendment 6 and the review addition)

- **Per epoch, every baseline loop** (all logged only; selection does not read them, tested):
  - the shared NC, LP and GC loops: `train/loss`, `val/loss`, `val/*`, `test/*`; on LP the test keys are the random set's `auroc`, `mrr`, `aupr`;
  - H2GB NC (the shared mini-batch and index loops, Hetero²Net's H2GB loop): `train/loss` and `val_subset/*` at evaluation epochs; the whole val and test sets once, at the end, in the record;
  - large LP: `test/*` when validation improves;
  - adapter-owned loops: Hetero²Net HGB NC, SeHGNN, and MAGNN (whose test is evaluated only when validation improves).
- **A W&B run for every run.** Final seeds of both launchers and published seeds each open a run: group `<token>_final` / `<token>_published`, tag `final_seed` / `published`, name `<token>_seed<s>` (`src/baselines/wandb_runs.py`). The H2S-NN path is opened in its launcher, with no `src/` change.
- **Summary at the end of every run, trials included:** selected epoch, `val_*`, `test_*`, wall time, GPU name, git HEAD, and the resolved config. Trial summaries are written in each launcher's `train_trial`, so `src/utils/wandb_campaign.py` is unchanged. W&B errors are logged and never fail a run.
- **Published runner:** writes `published_seed<s>.json` per seed and takes `--wandb_project` (published and baseline final-seed lines name `h2snn_baseline_t4`).
- **`collect.py`:** one `published_seed` row per published-seed record, with the same provenance columns, and the published cells are archived.
- **Deviations and fixes:**
  - The H2S-NN final-seed lines are unchanged (card 7's T9 pins them byte for byte), so their W&B project is the launcher default `h2snn` unless the line names one.
  - `report_epoch` passes `extra` / `prefix` only to callbacks that accept them; job 18036 found 12 B1a test errors from older callbacks.
  - Card 7 T9 and card 5 T4 no longer pin `baselines-1` to 1646efd, which card 8 C5 rebuilt. Card 5 T4 now checks that every kept line of the older methods is a line of the capture. The timing list has 31 lines (R-GCN and MAGNN join the main table). These are other sessions' tests: please review.
  - Job 18025 aborted on a second DGL import caused by my test helper (`mock.patch.dict(sys.modules)` dropped DGL); fixed.
  - Job 18026 aborted at the same place after the fix. That abort did not recur: the identical command passed on the same node in 18034, and none of five bisect jobs reproduced it. It is recorded as unexplained.

## 6. Review fixes (ops review of e3120eb) and full suite

- **F1:** `wandb_runs.open_run` is a no-op when no project is given or W&B is disabled (`mode` "disabled" or `WANDB_MODE=disabled`). The H2S-NN final-seeds path reads `wandb_project` / `wandb_entity` / `wandb_mode` with `getattr`, so the three tests that build `Namespace(final_seeds=..., monitor=None)` run unchanged. New test: `test_open_run_is_a_no_op_without_a_project_or_when_disabled`.
- **F2:** `rgcn.py` and `magnn.py` declare `READS_BATCH_PLAN = True`. Each path was checked:
  - R-GCN H2GB NC tuned: `minibatch_plan`;
  - R-GCN H2GB NC published: `published_plan`, which checks the yaml against `minibatch_plan`;
  - MAGNN H2GB NC (probe): `minibatch_plan` and `train_index_minibatch`;
  - both large-LP paths: `fit_large_lp` (`LP_POSITIVES_PER_STEP`, the protocol budget).
  `test_budget_table` is unchanged.
- **Full suite** (`python -m unittest discover -s tests -t .`, job 18064, CPU, 36G, g3n01): **954 tests, OK (skipped=1)**, 0 failures, 0 errors. The PSHGCN `sparse_tools` tests pass in this checkout. The job started on e3120eb with the F1/F2 working-tree changes, which are exactly commit 49252d2, committed while the job ran. SLURM's FAILED state comes from my wrapper's `${PIPESTATUS}` line in `sh` ("Bad substitution", after the suite printed OK).
- **Freeze check:** `git diff fc0ebaa -- H2S-NN/src/models H2S-NN/src/trainers H2S-NN/configs/experiments` is empty.

## 7. Open (not blocking)

- The probe lines decide MAGNN N/A on the H100. Every first TSUBAME line is the real-data check of its cell.

branch=card8 sha=49252d2e615948f514fc5155bba79ee16aaea862 card=docs/H2S-NN_card8_published_rgcn_magnn_card.md base=fc0ebaa605280f2d987c4848cc3109c7d3b66680
