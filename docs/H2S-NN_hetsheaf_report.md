# HetSheaf adapter (card 3 Part A) — report

Card `docs/H2S-NN_hetsheaf_freebase_card.md` Part A (A0–A4); branch `hetsheaf` from `v3b` `2ef9cac14ae03136efb07ad0c71233697ae138b1`,
merges `9309297` (pinned design-docs, clean), `f37e247` (card update: notebook regeneration, clean), **v3b `7df909d`** and **v3b `ae7aef0`**
(design session's A4 plan; resolution below). Env `h2snn-b`; jobs in allocation 16741 (g2n01, `srun --overlap`) until it
ended, then sbatch; scripts and logs in `docs/hetsheaf_evidence/`.
No metric was written or compared.

**Verdict.** HetSheaf runs on every registry cell: HGB NC 4, local LP 6, GC 4, H2GB NC 8 (class C) and both large LP
tasks (ogbn-mag P–P with the per-hop link sampler, D2; H-Pokec U–U with H2GB's `hgt` sampler and per-type lists, G-8),
i.e. 24 runnable cells. The model code runs unchanged on torch 2.4.1. Getting it imported needed two author decisions on stop
items (S-A0-1, S-A0-2), and wiring the registry row needed author-approved edits outside Part A's file list (listed below).
Full suite before the merge: 706 tests OK, golden and mb0 `differing: none`; after the last v3b merge (`ae7aef0`): 738 tests OK with the pinned kernel environment (golden and mb0 bit-exact tests included).

**A4 in one line.** At the largest point of the grid on a 24 GB GPU: 11 cells fit (imdb, imdb_ma, lastfm, the 8 H2GB NC);
2 fit at `hidden_channels` ≤ 32 (dblp, dblp_ap); 7 are still OOM at `hidden_channels` 32 (acm, freebase, acm_pa,
freebase_bb, pubmed, ogbn-mag P–P, H-Pokec U–U), their A6000 retries **pending** (queued, not waited for); the 4 GC cells
fit on the 11 GB 2080 Ti.

## Units
| Unit | Commit | Jobs (16741 step) | Result |
|---|---|---|---|
| A0 preconditions, H-1 | `513d30c`, `3cfd421` | env check, install, probe; 118, 121, 139 | readings agree with H-2/H-3; stops S-A0-1, S-A0-2 (author decided) |
| A1 NC + LP (HGB, local) | `05b4197` | 150, 151 (mutation), 154/156 | 4 NC + 6 LP cells; mutation caught |
| A2 GC | `62ba92f` | 158 | 4 GC cells |
| A3 H2GB NC + large LP | `2c283f1` | 163–167 | 8 H2GB NC + ogbn-mag P–P |
| merge v3b `7df909d` | `56f0bbe` | sbatch 16893, 16894, 16896, 16900 (CPU) | conflicts + two semantic breaks resolved (below); 107 targeted tests OK |
| A4 H-Pokec U–U large LP (author 2026-11-16) | `3125113` | sbatch 16915 (CPU) | registry cell wired; 107 tests OK; coverage 252/252 |
| merge v3b `ae7aef0` (card 1d H5) | `4a6d94a` | sbatch 17092 (CPU), suites 17093, 17094 | counts conflict only (union 254); notebooks unchanged on regeneration; 738 tests OK |
| A4 dry run, suite, report | this commit | 174; sbatch 16876 (harness error), 16878 (cancelled), 16903, 16916, 16958, 16959, 16979, 16980 (24 GB); A6000 16963, 16981, 17091 (pending) | below |

## A0 (`hetsheaf_evidence/a0_preconditions.md`)
- Every reading of Part P P0 (a)–(e) holds at `6964e1e`, with file:line in the A0 file. No file:line contradicts H-2 or H-3.
- `strenum==0.4.15` was installed `--no-deps`. The `pip freeze` diff is that one line; torch, PyG and DGL are unchanged, and golden/mb0 report `differing: none`.
- The model code needs nothing newer than torch 2.4.1: all three families with heterogeneous and attention learners give finite forward and backward passes.
- **S-A0-1 (author: bypass + alias).** H-1 assumed `strenum` was the only missing package. In fact, importing the package's `__init__` chain needs `lightning` and `omegaconf` (`hetsheaf/__init__.py:4-10`, `models/__init__.py:4`), and SheafPool's module imports `lightning`.
  - The adapter registers empty package modules over the vendor folders, so those `__init__` files never run.
  - For the one import of `sheaf_graph_classifier`, `lightning.pytorch` is mapped to the installed `pytorch_lightning` 2.6.5; their lock pins `lightning` 2.6.0 and `pytorch-lightning` 2.6.1. The alias replaces the imports at `sheaf_graph_classifier.py:10` and `graph_classifier.py:7, 11` (base class `L.LightningModule`, `:24`).
  - The alias is removed right after that import. Both mechanisms are created only inside the HetSheaf adapter, never at registry or package import, as the design session's condition 1 asks. A test checks that importing the registry and another adapter leaves no `lightning`, `omegaconf` or `hetsheaf` module behind, and that no `lightning` module remains after SheafPool's import.
  - Nothing else was installed (condition 2).
- **S-A0-2 (author).** `strenum==0.4.15` is in `requirements/linux-lock-b.txt`, and `tests/test_check_env.py` checks it with the pin count raised from 28 to 29 (a stated change outside Part A's files). **Note:** card 1c adds `sparse_tools` to the same lock; whichever branch merges second sets the count to 30.

## What runs (scheme, file:line in the adapter docstring)
| Cells | Data path | Model | Head / loop |
|---|---|---|---|
| HGB NC (ACM, DBLP, IMDB, Freebase) | PyG `HGBDataset` as their data module (`hgb.py:61`, never downloading), shared masks on the target, their chain `Constant(node_types=None)`, `ToUndirected`, `RemoveSelfLoops`, `NormalizeFeatures`, `RemoveDuplicatedEdges` (`hgb.py:51-60`, **`SubsetTrainSplit` removed**), `coalesce`, `to_homogeneous` | their transductive encoder (family × learner) | their `Linear(hidden_dim, C)`; CE, IMDB `multilabel_soft_margin_loss`; shared full-batch loop |
| derived LP (acm_pa, dblp_ap, imdb_ma, freebase_bb) | the training message graph through the same HGB NC chain | same | shared link head (dot / DistMult) |
| LastFM LP | their `link_pred.py:39-46` chain (`Constant`, `ToUndirected`, `NormalizeFeatures`, `RemoveSelfLoops`) | same | shared link head |
| PubMed LP | their HGB-LP construction (`hgb.py:331-357`: `rev_` per relation, `ones(n, 128)` without features, no transform) | same | shared link head |
| GC (4 TU) | shared GC bundle (HetSheaf's protocol: constant features, seed-42 split, batch 1) | their inductive encoder (`edge_weights` false, `output_dim` 64 as their GC script) | their readout (categorical: mean / max / add / attention / `late_token` = SheafPool) + their `Linear`; shared GC loop |
| H2GB NC (8, class C) | their chain once on the whole H2GB graph; H2GB's `hgt` sampler with the dataset's GCN YAML list (S1) via `minibatch.hgt_loader` / `eval_loader` / `train_minibatch_node_classification` (called, not edited); each batch rebuilt with `x` / `edge_index` only and passed through `to_homogeneous` (type ids per subgraph) | their transductive encoder rebound per batch as their inductive classes rebind | their NC head; shared H2GB loop |
| ogbn-mag P–P (D2) | their chain applied in place to the task's message graph; `h2gb_sampler` of the class-C reference method (per-hop `link` sampler on MAG) | same, rebound | shared link head; the shared loop `large_lp_training.fit_large_lp` via `build_large_lp` (card 1c F7 convention) |
| H-Pokec U–U (G-8) | their chain applied in place to the task's message graph; `h2gb_sampler` falls back to H2GB's `hgt` sampler with its per-type lists (as NSD and CSNN), seeded by the step's unique endpoints | same, rebound | shared link head; the shared loop via `build_large_lp` |

**Search space (H-3).**
- One categorical `sheaf` over their (family, learner) pairs: diag × 8 learners, bundle and general × 16 each (the 8 plus `attention` and the seven `attention_*`), as in their sweeps for our datasets. That is 40 values with no invalid combination, and the prior is uniform per pair.
- Their sweep axes: `d` {2..5}, `layers` {2..8}, `hidden_channels` {8, 16, 32, 64}, `input_dropout` U[0, 0.9].
- Shared axes: lr, weight decay, dropout.
- Task-specific: LP `decoder_type`; GC `pooling`; H2GB `batch_size`.
- `initial_dropout` and `output_dim` are not tuned. The first has no effect; the second only sizes `lin2`, which is never called. It is set as their scripts set it: NC the class count, LP 64, GC 64.

## Tests (Slim) and mutation (`tests/test_baselines_hetsheaf.py`, 10 tests)
| Check | Result |
|---|---|
| Preprocessing equality: our chain vs their `HGBBaseDataModule.prepare_data` (`hgb.py:50-91`), with `SubsetTrainSplit` replaced by the identity and `HGBDataset` returning the synthetic release; `x`, `edge_index`, `node_type`, `edge_type` exact | OK |
| Shared masks unchanged after the chain (label-route protection) | OK; **mutation** (`SubsetTrainSplit` in our chain) caught: `val_mask` differs (step 151) |
| One NC step and one LP step (all three LP branches) finite and moving | OK |
| One GC step finite and moving, `late_token` and `mean` | OK |
| One sampled H2GB NC step; one large-LP step | OK |
| Rebinding equivalence (new mechanism): an encoder rebound to a batch equals the same weights built on that batch's graph, exactly, for diag, bundle and general | OK |
| Held-out-fact test with HetSheaf's loader configuration (chain incl. `ToUndirected` on the message graph; step and endpoint samplers; depth 1–3) | OK |
| Import route (condition 1) | OK |
| Full suite at `2c283f1`, g2n01, 2080 Ti visible (`a4_full_suite_g2n01.txt`) | **Ran 706 tests, OK**; golden 1365 tensors and mb0 139 tensors `differing: none` |
| After the v3b merge: test_baselines, test_baselines_hetsheaf, test_baselines_large_lp (incl. v3b's wired-method step test, now with HetSheaf), test_check_env (job 16900, `a4_merge_tests.txt`) | **107 OK**; coverage 251/251 runnable cells have an adapter |
| Full suite at `4a6d94a` (merge of v3b `ae7aef0`), g2n01, 2080 Ti visible, pinned kernel environment E (job 17094, `a4_full_suite_post_merge.txt`) | **Ran 738 tests, OK** (bit-exact golden and mb0 tests included) |
| First attempt at the same commit (job 17093, `a4_full_suite_post_merge_unpinned.txt`) | 735 OK, 3 failed: the three bit-exact tests refused to run because my launcher did not export E (`tests/pinned_kernels.py`, new in v3b; `scripts/slurm/run_tests.sbatch` exports it); launcher fixed, suite rerun |

## A4 — one epoch + one validation pass (card 2's rule; no metric)
Configuration: `d` 5, `layers` 8, `hidden_channels` 64, `general/attention_type_ensemble`; batch 1024 on H2GB NC; LP DistMult; GC `late_token`. One process per cell (`a4_dryrun.py`).
- Time is wall time of the training epoch plus the validation (and test) pass, as the shared loops run them.
- Peak GPU memory is `max_memory_allocated`; host memory is `ru_maxrss`.
- The first attempt (allocation 16741, shared 2080 Ti with ~3 GB held by other processes; sbatch 16876 with a harness
  error in the bundle config) is not used. Measurements: one cell per job on a 24 GB GPU not shared (TITAN RTX / 3090;
  g2n01 and g3an01 excluded), array 16903 (cells 0–10), 16958 / 16959 (cells 11–18), 16916 (H-Pokec U–U); an OOM cell
  runs once more in the same job with `hidden_channels` 64 → 32 (the one step-down); cells still OOM retried once on the
  A6000 (16963).
- **Resource rule (design session 2026-11-16):** `--mem` from need: recorded host peak (this table's earlier rows or
  the D2/D6 `cells.csv`) + 30 %, at least 16G; 60G only where the record needs it (H-Pokec U–U). Cells 11–18, queued at
  60G, were cancelled before starting and resubmitted (PDNS NC and freebase_bb 19G from 14.32 / 14.11 GiB; local LP and
  ogbn-mag P–P 16G); cell 10 (H-Pokec NC) was already running at 60G and left to finish. g2n01 stays excluded for these
  cells: none has a recorded GPU peak below 9 GiB.

| Cell | Run | Point | Status | Peak GPU GiB | Peak host GiB | Epoch + eval s | GPU, node, job | OOM at |
|---|---|---|---|---|---|---|---|---|
| node:acm | 24 GB | largest | OOM | 16.51 | 1.58 |  | RTX 3090, g3n01, job 16942 | tried to allocate 7.12 GiB at `models/sheaf_gnn/transductive/disc_models.py:560` |
| node:acm | 24 GB | hidden_channels=32 | OOM | 20.04 | 1.57 |  | RTX 3090, g3n01, job 16942 | tried to allocate 3.56 GiB at `models/sheaf_gnn/transductive/disc_models.py:560` |
| node:dblp | 24 GB | largest | OOM | 22.1 | 2.44 |  | RTX 3090, g3n01, job 16943 | tried to allocate 3.25 GiB at `models/sheaf_gnn/transductive/disc_models.py:560` |
| node:dblp | 24 GB | hidden_channels=32 | OK | 21.93 | 2.57 | 2.4 | RTX 3090, g3n01, job 16943 |  |
| node:imdb | 24 GB | largest | OK | 17.07 | 2.23 | 2.2 | RTX 3090, g3n01, job 16944 |  |
| node:freebase | 24 GB | largest | OOM | 21.41 | 1.39 |  | RTX 3090, g3n01, job 16945 | tried to allocate 10.45 GiB at `models/sheaf_gnn/transductive/disc_models.py:560` |
| node:freebase | 24 GB | hidden_channels=32 | OOM | 19.36 | 1.39 |  | RTX 3090, g3n01, job 16945 | tried to allocate 5.22 GiB at `models/sheaf_gnn/transductive/disc_models.py:560` |
| node:ogbn-mag | 24 GB | largest | OK | 4.01 | 7.97 | 132.2 | RTX 3090, g3n01, job 16946 |  |
| node:mag-year | 24 GB | largest | OK | 4.02 | 7.97 | 358.7 | RTX 3090, g3n01, job 16948 |  |
| node:oag-cs | 24 GB | largest | OK | 1.07 | 15.0 | 45.5 | RTX 3090, g3n01, job 16950 |  |
| node:oag-eng | 24 GB | largest | OK | 0.94 | 12.09 | 38.7 | RTX 3090, g3n01, job 16951 |  |
| node:oag-chem | 24 GB | largest | OK | 1.42 | 24.81 | 52.0 | RTX 3090, g3n01, job 16952 |  |
| node:ieee-cis-g | 24 GB | largest | OK | 0.77 | 11.36 | 34.4 | RTX 3090, g3n01, job 16953 |  |
| node:h-pokec | 24 GB | largest | OK | 13.21 | 10.37 | 796.8 | RTX 3090, g3n01, job 16956 |  |
| node:pdns | 24 GB | largest | OK | 1.34 | 7.51 | 69.9 | TITAN RTX, g2tn01, job 16970 |  |
| link:acm_pa | 24 GB | largest | OOM | 16.52 | 1.58 |  | TITAN RTX, g2tn01, job 16973 | tried to allocate 7.08 GiB at `models/sheaf_gnn/transductive/disc_models.py:560` |
| link:acm_pa | 24 GB | hidden_channels=32 | OOM | 20.02 | 1.57 |  | TITAN RTX, g2tn01, job 16973 | tried to allocate 3.54 GiB at `models/sheaf_gnn/transductive/disc_models.py:560` |
| link:dblp_ap | 24 GB | largest | OOM | 21.97 | 3.56 |  | TITAN RTX, g2tn01, job 16974 | tried to allocate 3.16 GiB at `models/sheaf_gnn/transductive/disc_models.py:560` |
| link:dblp_ap | 24 GB | hidden_channels=32 | OK | 21.82 | 3.68 | 2.6 | TITAN RTX, g2tn01, job 16974 |  |
| link:imdb_ma | 24 GB | largest | OK | 16.52 | 2.18 | 2.3 | TITAN RTX, g2tn01, job 16975 |  |
| link:freebase_bb | 24 GB | largest | OOM | 21.19 | 3.27 |  | TITAN RTX, g2tn01, job 16958 | tried to allocate 10.14 GiB at `models/sheaf_gnn/transductive/disc_models.py:560` |
| link:freebase_bb | 24 GB | hidden_channels=32 | OOM | 19.28 | 3.27 |  | TITAN RTX, g2tn01, job 16958 | tried to allocate 5.07 GiB at `models/sheaf_gnn/transductive/disc_models.py:560` |
| link:pubmed | 24 GB | largest | OOM | 23.27 | 1.6 |  | TITAN RTX, g2tn01, job 16976 | tried to allocate 3.40 GiB at `models/sheaf_gnn/transductive/disc_models.py:560` |
| link:pubmed | 24 GB | hidden_channels=32 | OOM | 23.23 | 1.6 |  | TITAN RTX, g2tn01, job 16976 | tried to allocate 56.00 MiB at `src/baselines/training.py:239` |
| link:lastfm | 24 GB | largest | OK | 19.03 | 1.44 | 2.4 | TITAN RTX, g2tn01, job 16977 |  |
| link:ogbn-mag_pp | 24 GB | largest | OOM | 18.95 | 8.1 |  | TITAN RTX, g2tn01, job 16979 | tried to allocate 67.52 GiB at `models/sheaf_gnn/sheaf_models.py:1236` |
| link:ogbn-mag_pp | 24 GB | hidden_channels=32 | OOM | 18.95 | 8.09 |  | TITAN RTX, g2tn01, job 16979 | tried to allocate 33.76 GiB at `models/sheaf_gnn/sheaf_models.py:1236` |
| link:h-pokec_uu | 24 GB | largest | OOM | 22.97 | 7.83 |  | TITAN RTX, g2tn01, job 16980 | tried to allocate 3.36 GiB at `models/sheaf_gnn/transductive/disc_models.py:560` |
| link:h-pokec_uu | 24 GB | hidden_channels=32 | OOM | 23.01 | 7.83 |  | TITAN RTX, g2tn01, job 16980 | tried to allocate 1.71 GiB at `src/baselines/large_lp_training.py:181` |
| graph:mutag | 11 GB (2080 Ti, A4 GC) | largest | OK | 0.03 | 1.54 | 7.9 | RTX 2080 Ti, g2n01, job 16741 |  |
| graph:proteins | 11 GB (2080 Ti, A4 GC) | largest | OK | 0.26 | 1.52 | 18.6 | RTX 2080 Ti, g2n01, job 16741 |  |
| graph:enzymes | 11 GB (2080 Ti, A4 GC) | largest | OK | 0.06 | 1.52 | 12.4 | RTX 2080 Ti, g2n01, job 16741 |  |
| graph:nci1 | 11 GB (2080 Ti, A4 GC) | largest | OK | 0.05 | 1.55 | 105.4 | RTX 2080 Ti, g2n01, job 16741 |  |

**Per cell (design session A4 item 4):**
- fits at 24 GB at the largest point: imdb, imdb_ma, lastfm, ogbn-mag, mag-year, oag-cs, oag-eng, oag-chem, ieee-cis-g,
  h-pokec, pdns (NC); GC mutag, proteins, enzymes, nci1 (11 GB);
- fits at 24 GB with `hidden_channels` ≤ 32: dblp (NC), dblp_ap (LP);
- still OOM at `hidden_channels` 32 (needs > 24 GiB: allocated + requested exceeds the 23.6 GiB card): acm (20.0 + 3.6),
  freebase (19.4 + 5.2), acm_pa (20.0 + 3.5), freebase_bb (19.3 + 5.1), pubmed (23.2 + 0.05), H-Pokec U–U
  (21.6 + 1.7); ogbn-mag P–P needs > 52 GiB at `hidden_channels` 32 (one 33.8 GiB request at `sheaf_models.py:1236`,
  the sheaf learner's per-edge input on the step subgraph; 67.5 GiB at 64), so it cannot fit on the A6000 (48 GB)
  at either point;
- A6000 retries (one per still-OOM cell, largest point, step-down on OOM): 16963 (acm, freebase), 16981 (acm_pa,
  freebase_bb, pubmed), 17091 (ogbn-mag P–P, H-Pokec U–U): **pending** at the time of this report.

Harness errors (not cells' results): sbatch 16876 and the first runs of cells 18 and 23 failed in `build_bundle`'s
config check before the adapter (`config.py:1023` batch size, `:924` task type, `:1159` target relation for the large LP
tasks; card 1d H3b); the script passes those config values as overrides (docstring of `a4_dryrun.py`); the discarded
records are in `a4_cells_rerun_harness/` and `a4_cells_big.jsonl`.

## Deviations and stated points
- **Their code vs the shared protocol (H-2):**
  - AdamW on the shared axes, no scheduler. This replaces their NC `CosineAnnealingLR` (`node_classifier.py:178-192`), and their LP/GC optimisers that ignore the swept lr/wd (`sheaf_link_predictor.py:105-117`; `graph_classifier.py:155-161`).
  - `SubsetTrainSplit` and their `RandomLinkSplit` / GC split are never applied.
  - LP uses the shared head, not their `Linear(2h, 1)`.
- **Per-dataset preprocessing branch** (table above). Their code has no LP task on ACM/DBLP/IMDB/Freebase or on H2GB graphs; those take their HGB NC chain.
- **H2GB:** the chain runs once on the whole graph. Applying it per subgraph would change features, because `NormalizeFeatures` shifts by each type's global minimum. Featureless types take `Constant`'s column of ones, as their chain does, not the one-hot rows the other class-C adapters use.
- **Rebinding, not their inductive classes, on H2GB batches.** Those classes cannot be built with `edge_weights=True`: `EdgeWeightLearner` indexes the graph it is built on (`sheaf_models.py:199-205`), and their GC configs set the flag false. So the NC model (`edge_weights` true) is rebound per batch the way their inductive `forward` does it (`inductive/disc_models.py:24-30`). The equivalence test checks it.
- **Large LP:** before the merge, a typed copy of the homogeneous-only loop (author decision). v3b (card 1c F7) rewrote that loop as `fit_large_lp(build=...)` with an adapter-supplied model and `Sampler`, and removed `step_subgraph` / `endpoint_loader`; the copy is gone and HetSheaf supplies `build_large_lp(d, params, device, sampler=None)` like Hetero2Net. The loop samples `d.graph`, so their chain is applied to that object in place (`_replace_in_place`). `large_lp_training.py` is called, not edited.
- **Merge resolution (v3b `ae7aef0`):** counts = union (254; NC 138, LP 76, GC 40); the generator left every notebook unchanged; pin count 29. `test_all_declared_vendor_clones_are_present` needed card 1d's `sparse_tools` clone: copied from the main worktree (commit `0596f00`, git-ignored, as every vendor clone here).
- **Merge resolution (v3b `7df909d`):** `tests/test_baselines.py` counts = union (251; NC 136, LP 75, GC 40); `link_prediction_sheaf.ipynb` regenerated with the unchanged generator; registry `_LP_POKEC` (removed in v3b) replaced by a HetSheaf-specific reason; `tests/test_check_env.py` stays at 29 pins. No `src/` file outside `src/baselines/` changed between `2c283f1` and the merge, so golden/mb0 were not rerun.
- **Binary GC:** MUTAG is configured `multiclass` in their configs (`configs/dataset/mutag.yaml:14`); shared CE over the class logits.
- **Joint (family, learner) categorical:** the prior is uniform per pair, not per family.

## Changes outside Part A's file list (each approved by the author in this session)
| File | Change |
|---|---|
| `tests/test_check_env.py` | pin count 28 → 29, `pins["strenum"]` check (S-A0-2) |
| `tests/test_baselines.py` | runnable counts 193 → 216 (NC 103 → 115, LP 54 → 61, GC 36 → 40); the `ADAPTER_PENDING["hetsheaf"]` assertion becomes "registered row"; after the v3b merge 251 (NC 136, LP 75, GC 40); with H-Pokec U–U 252 (LP 76) |
| `tests/test_baselines_large_lp.py` | (merge-forced; approved by the author 2026-11-16) `"hetsheaf"` added to the wired-method name list of v3b's `test_each_wired_method_one_lp_step_is_finite_and_moves`; the test then runs HetSheaf's `build_large_lp` with the other adapters |
| `src/baselines/runtime/dependencies.py` | `"hetsheaf": ("sheaf",)`; `strenum` added to the sheaf group's modules |
| `src/baselines/runtime/vendor_paths.py` | `VENDOR_ROOTS["hetsheaf"]`; `hetsheaf` in `NAMESPACED_VENDORS` (its modules are never evicted by another vendor's block) |
| `requirements/baselines-sheaf.txt` | `strenum==0.4.15` |
| `scripts/generate_baseline_notebooks.py` | two add-only entries (`MODEL_VENDORS`, `MODEL_NOTES`); notebooks regenerated, `--check` OK |

## Decisions / items for the design session
| # | Item | Recommendation |
|---|---|---|
| 1 | Per-dataset preprocessing branch for LP (PubMed HGB-LP construction, LastFM `link_pred.py` chain, the HGB NC chain elsewhere) | keep, as "follow the source's code exactly" |
| 2 | `pytorch_lightning` is needed for SheafPool's import but is listed only in the `h2gb` requirement group, not in the `sheaf` group that `hetsheaf` maps to | add it to the sheaf group (a dependency-map edit outside this card) |
| 3 | H-Pokec U–U: wired in A4 (author). The earlier "MAG P–P only" was the author's answer to a question in this session (choice "Mirror loop, MAG P–P only"), not a line in a file; no file:line exists, and the decision is superseded. The held-out-fact test of `test_baselines_hetsheaf.py` is not parametrised by task (synthetic `user` task, link sampler), so no test was added; v3b's `test_no_held_out_fact_in_any_sampler_configuration` covers the `hgt` sampler | done |
| 4 | Cells still OOM at 24 GB (table above) | recorded only; A6000 retries pending if listed |

## Out of scope (not done)
- `requirements.txt`'s comment line still says `baselines-sheaf.txt  CSNN, NSD` (the comment omits HetSheaf).


## A5 — OOM ladder and central-point timing (2026-11-17; base `1867681`; docs only, no suite)
A5.1: A6000 jobs 16963, 16981, 17091 cancelled; no record of them is used. Evidence `hetsheaf_evidence/a5_*`
(`a5_dryrun.py`, `a5_job.sh`, `a5_table.py`, `a5_ladder/`, `a5_central/`). One cell per job; all jobs print HEAD
(`1867681`), hostname, job id, sha256 of the adapter and scripts. `--mem` = A4 host peak + 30 %, >= 16G (oag-cs 20G,
oag-chem 33G, H-Pokec 60G); g3an01 excluded; g2n01 excluded except for the GC jobs (which ran on g3n01). No metric.

**A5.2 ladder** (`hidden_channels` 32 -> 16 -> 8, then `d`, then `layers`; A4's 64 and 32 OOM records reused; full
attempt table `a5_ladder_table.md`, ranges `a5_ranges.md`):

| Cell | Largest fitting point (other axes at A4's) | Peak GPU GiB |
|---|---|---|
| NC acm, NC freebase, LP acm_pa, LP freebase_bb | `hidden_channels` 8 (16: OOM) | 12.8 / 19.6 / 12.9 / 19.6 |
| LP pubmed | `hidden_channels` 16 | 12.2 |
| NC dblp, LP dblp_ap | `hidden_channels` 32 (A4) | 21.9 / 21.8 |
| LP ogbn-mag P-P | **none** (stop item) | |
| LP H-Pokec U-U | **not established** (stop item) | |

**Stop items (not fixed):**
1. ogbn-mag P-P is OOM at the bottom of the ladder (`hidden_channels` 8, `d` 2, `layers` 2): `sheaf_models.py:1238`
   (`h_cat`, the sheaf learner's per-edge input), 15.81 GiB allocated + 9.65 GiB requested on 23.68 GiB; identical at
   every `layers` 8..2; sites 1236-1238 at the higher points. DERIVED: about 5.7e7-8.1e7 edges per sampled step
   subgraph, so the 3.4 GiB estimate does not hold.
2. H-Pokec U-U at `hidden_channels` 16, `d` 5, `layers` 8: not OOM; `AssertionError` at
   `transductive/disc_models.py:571` (their finite check on the forward output, "numerical instabilities of SVD")
   after 4,070 s, peak GPU 12.13 GiB. The ladder steps only on OOM, so it stopped; one run, not rerun.

**A5.3 central point** (card 2's `central_point` rule over `search_space`: `d` 3, `layers` 5, `hidden_channels` 16,
`lr` 1e-3 (geometric middle), `weight_decay` 5e-4 (middle of 9), dropout / input_dropout 0.45, H2GB `batch_size` 256,
`sheaf` `diag/type_concat`, GC `pooling` `late_token`). Categoricals: no published HetSheaf value exists for these
cells, so the adapter's value when the parameter is absent is used (`hetsheaf.py:401`, `:620`; prompt "at the
adapter's default"); card 2's fallback (first grid value: `diag/local_concat`, `mean`) was not used. Timing:
train epoch = last `train()` -> first `eval()`; validation = first `evaluate_split` (H2GB NC) or first `eval()` ->
first metric call (GC); CUDA synchronised. Columns of `a5_central.csv` follow card 2's D6 record fields
(`val_targets`, phase times); card 2's D6.4 central table is not published yet, so its exact columns could not be
matched.

| Cell | Train epoch s | Validation s | Val targets | GPU, job |
|---|---|---|---|---|
| ogbn-mag | 26.2 | 206.9 | 64,879 | 3090, 17125 |
| mag-year | 27.1 | 595.4 | 184,097 | 3090, 17126 |
| oag-cs | 5.9 | 13.2 | 19,551 | 3090, 17127 |
| oag-eng | 5.1 | 16.5 | 28,524 | 3090, 17128 |
| oag-chem | 5.3 | 35.8 | 59,664 | 3090, 17129 |
| ieee-cis-g | 3.2 | 4.8 | 14,423 | 3090, 17130 |
| h-pokec | 28.7 | 1,277.5 | 408,160 | TITAN RTX, 17131 |
| pdns | 4.1 | 6.9 | 17,450 | 3090, 17132 |
| GC mutag | 3.0 | 0.09 | 18 | 3090, 17133 |
| GC proteins | 9.6 | 0.57 | 111 | 3090, 17134 |
| GC enzymes | 5.2 | 0.30 | 60 | 3090, 17135 |
| GC nci1 | 34.8 | 2.15 | 411 | 3090, 17136 |

Only `docs/` changed in A5 (no file outside `docs/`; no suite). Decisions table item 4 above (A6000 retries) is
superseded by A5.1.

branch=hetsheaf sha=<see final message> card=docs/H2S-NN_hetsheaf_freebase_card.md base=1867681dc197ab536ad5cf1fbede4a9df9abffab
