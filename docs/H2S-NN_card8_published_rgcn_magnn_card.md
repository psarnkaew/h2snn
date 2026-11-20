# Card 8 — Published configurations (5 seeds), R-GCN and MAGNN adapters, rebuilt TSUBAME lists

Written 2026-11-20 by the design session (ops). Decisions: `docs/experiment_settings_decisions.md`, rows of 2026-11-20 "Published tuned configurations; LP and GC model sets" and "Card 8: R-GCN and MAGNN" (author). Report: `docs/H2S-NN_card8_report.md`. Evidence: `docs/card8_evidence/`.

## Why
1. **Published configurations are the tuned hyperparameters** of their cells (author). A cell of the paper tables that is still empty and has a published configuration (HGB, H2GB, or the baseline's own repository) runs 5 seeds (0–4) with that configuration, exactly as the file states. A cell that already holds a published number is not run. An empty cell without a published configuration is tuned as before (100 Bayesian trials on seed 0, then seeds 0–4).
2. **R-GCN and MAGNN** are in the paper's NC table (and R-GCN in the GC table) but have no adapter. They are added for NC and LP (GC keeps R-GCN's published HetSheaf number; MAGNN is not in GC: the TU graphs define no meta-paths, and HetSheaf's GC HAN uses PyG `HANConv` on edge types, i.e. length-1 paths; author 2026-11-20).
3. **GC option (a):** the GC baseline rows are HetSheaf's published numbers. No baseline GC cell is run; every baseline GC cell leaves the runnable set.
4. **LP** uses every method with an LP adapter: the 10 existing ones plus R-GCN and MAGNN.

Nothing in this card changes H2S-NN (`src/models`, `src/trainers`, H2S-NN configs, the H2S-NN launcher): the freeze `365a757` stays. `src/baselines/` changes are covered by the author's decisions above.

## Preconditions
- Work in `/mnt/data/peeradon/h2snn`. `git fetch`; `origin/v3b` = `fc0ebaa605280f2d987c4848cc3109c7d3b66680`; `origin/design-docs` contains this card (SHA in the prompt). If either differs, stop.
- New branch `card8` from `fc0ebaa`, then `git merge` the pinned design-docs SHA (merge, never rebase).
- Read: this card; spec §6 (Tuning, LP, GC, Baselines bullets); `src/baselines/{registry,campaign,published,budget,search_grids,bundles,training,minibatch,large_lp_training}.py`; `scripts/run_baseline_published.py`; `scripts/tsubame/{make_cell_list,lines}.py`; `docs/published_number_register.md`; `docs/predefined_metapaths.md`; vendor: HGB `NC/benchmark/methods/{RGCN,MAGNN}`, `LP/benchmark/methods/{RGCN,MAGNN,GNN,baseline}`, H2GB `configs/<dataset>/*-{GCN,GAT,HAN,H2Gformer,RGCN,HINormer}.yaml` and the H2GB modules they name.

## Units (one commit each; push `card8` after each)

### C0 — Inventory (no code): `docs/card8_evidence/c0_inventory.md`
1. **Filled cells.** From `docs/published_number_register.md`, the (task, dataset, method) cells that hold a published number for the table metric (NC Macro-F1, LP ROC-AUC, GC Macro-F1). Write them as `configs/baselines/paper_filled_cells.tsv` (task, dataset, method, register line). These cells are never run.
2. **Published configurations of empty cells.** For every empty (task, dataset, method) cell of the method set below, state whether a published configuration exists, with the file:line that gives each value. Rule: a cell is "published" only if a file (H2GB yaml, HGB README/run script command, the method's own README/config) gives its values for **that** dataset; argparse defaults alone do not make a configuration, except where the source's README runs the script for that dataset without flags. Expected (check each; differences are not a stop, state them):
   - NC, H2GB yaml: GCN, GAT, HAN, R-GCN × {mag-year, oag-cs, oag-eng, oag-chem, h-pokec, pdns}; H2G-former × {ogbn-mag, mag-year, oag-cs, oag-eng, oag-chem, h-pokec, pdns}; HINormer × {mag-year, oag-eng, pdns} (and any other H2GB dataset whose cell is empty and has a yaml).
   - NC, existing `configs/baselines/`: PSHGCN IMDB, H2SGNN IMDB.
   - NC, own repositories: SeHGNN ogbn-mag (`SeHGNN/ogbn/Readme.md`), PSHGCN ogbn-mag (`PSHGCN/ogbn-mag`): published only if the command uses inputs our adapter supports (no extra embeddings, spec 1.23); else tuned.
   - NC, HGB: MAGNN Freebase (`MAGNN/run_Freebase.py`).
   - LP: GAT PubMed and LastFM (HGB LP code), PSHGCN LastFM (existing file), R-GCN and MAGNN on PubMed/LastFM only where empty.
3. **Final cell table:** every cell of the methods below on every task, with status `filled` / `published` / `tune` / `probe` / `N/A (reason)`. Method set: NC 15 (GCN, GAT, ChebNetII, NSD, CSNN, R-GCN, HAN, MAGNN, SeHGNN, HINormer, PSHGCN, H2G-former, Hetero²Net, H2SGNN, HetSheaf); LP 12 (GCN, GAT, NSD, CSNN, R-GCN, HAN, MAGNN, PSHGCN, H2G-former, Hetero²Net, H2SGNN, HetSheaf); GC: none run.
Stop if the register cannot be mapped to cells without guessing (list the rows).

### C1 — Published configurations as files
- One file per `published` cell: `configs/baselines/<method>/<task>/<dataset>.yaml` with `source:` (file:line for every value) and `params:`, as the existing 16 files.
- **"Exactly as the file states"** (author): every hyperparameter and training-schedule key of the source runs as written: width, layers, heads, dropout, lr, weight decay, optimizer and its scheduler (H2GB: cosine with warm-up), gradient clipping, batch size and accumulation, epochs, evaluation period, sampler fan-outs. Values outside our search grids are allowed for published runs. **Kept from our protocol** (they are data and evaluation, not hyperparameters): files, splits, negatives, feature preprocessing of each method's own code, the metric functions, B1a subset selection on the 8 H2GB NC datasets and the single whole-split test after restoring the best weights, LP selection on validation ROC-AUC. A source key that our runner cannot execute exactly is a **stop** (list it with file:line); do not approximate.
- The published runner (`run_baseline_published.py` → `fit_cell(source="published")`) accepts a scheduler / clipping / accumulation only for `source="published"`; the tuner keeps refusing them. Default base seed of the published runner becomes 0 (spec seeds 0–4); the perturbation manifest that passes `--base_seed 1` is unchanged.
- Test: for one H2GB yaml cell (GCN, ogbn-mag) and one HGB command cell (MAGNN Freebase), the resolved run arguments equal the source values key by key (parse the source file in the test). Mutation: drop the scheduler from the H2GB mapping → the test fails.

### C2 — R-GCN adapter (`src/baselines/adapters/rgcn.py`)
- **NC, HGB datasets:** HGB's `NC/benchmark/methods/RGCN` model, run as its code runs it (filled cells are not run, but the adapter supports them for the perturbation and checks). **NC, H2GB datasets:** H2GB's own R-GCN (the module its `*-RGCN.yaml` names) on H2GB's `hgt` sampler, as GCN/GAT/HAN class A (decisions 2026-11-16, class A).
- **LP** (8 tasks): HGB's `LP/benchmark/methods/RGCN` encoder, our `SharedLinkHead` (dot / DistMult), the shared LP loop and files; large LP through `fit_large_lp`. Search space: the source's argparse keys (hidden, layers, bases, …); values contain the source's published values where they lie inside the shared lr / weight-decay / dropout axes (shared axes are not widened, as SeHGNN Freebase, decisions 2026-11-18).
- GC: not run (published).
- Tests: one "finite loss and parameters move" per loop class (HGB NC, H2GB NC, LP small, LP large); split/negative equality comes from the shared bundle (assert the adapter receives the bundle's masks/files).

### C3 — MAGNN adapter (`src/baselines/adapters/magnn.py`)
- **NC:** HGB's `NC/benchmark/methods/MAGNN` (`run_ACM/DBLP/IMDB/Freebase.py`): its meta-path lists and instance preprocessing, our masks. H2GB NC: the same model with meta-paths from the rule below; these cells are `probe` (C5).
- **LP:** HGB's `LP/benchmark/methods/MAGNN` model (`run_LastFM.py` form) with our files and negatives and its own decoder replaced by `SharedLinkHead` (as every LP baseline). Meta-paths per endpoint type `t`: the paths of HGB's published table that start at `t` (`docs/predefined_metapaths.md`; LastFM: MAGNN's own lists); for an endpoint type with no listed path, the symmetric length-2 schemas `t–X–t` over every neighbour type `X` of `t` in the message graph (held-out facts removed). Large LP: `probe`.
- GC: not run (no meta-paths in the TU graphs; author 2026-11-20).
- Instance enumeration must exclude held-out LP facts (assert as `assert_no_held_out_facts` does) and must read labels only of training nodes (NC: none read at all; state it).
- Tests: "finite and parameters move" per task class; LP held-out route test (an instance through a held-out test fact must not exist) + **mutation** (enumerate on the full graph → the test fails).

### C4 — Registry and search spaces
- Register `rgcn` (NC 12, LP 8) and `magnn` (NC 12, LP 8). Remove every baseline GC cell from the runnable set with reason "GC option (a): HetSheaf published number" (keep the code). Cells N/A by C0/C5 carry their reason.
- `SearchSpaceCoverageTests` covers the new `configs/baselines/` files (published values reachable or the cell is `published`, not tuned).

### C5 — TSUBAME lists (`scripts/tsubame/make_cell_list.py`)
- New phase **`published`**: one `run_baseline_published.py … --final_seeds 0 1 2 3 4` line per `published` cell (output and state dirs as the tuning lines).
- **`baselines-1`** rebuilt: tuning lines for `tune` cells on the datasets of the old phase (NC acm, dblp, imdb; LP acm_pa, dblp_ap, imdb_ma, pubmed, lastfm), excluding `filled`, `published` and `baselines-early` cells.
- New phase **`baselines-2`**: tuning lines for every other `tune` cell (Freebase NC, freebase_bb, H2GB NC, large LP).
- New phase **`probe`**: one 1-trial `--dry_run` line per `probe` cell at the smallest grid point, to decide N/A on the H100 (range rule 1). Not a result.
- `h2snn` and `baselines-early` stay byte-identical. Test: phases pairwise disjoint; union = every non-`filled`, non-N/A cell exactly once; mutation: drop the `filled` exclusion → overlap with the filled set → fails.
- Report the line counts per phase and the `node_f` tasks per chain.

### C6 — Targeted tests (one CPU job, no `--gres`, 36G)
New tests of C1–C5, the existing `test_baselines*`, `test_tsubame_*` and `wandb_campaign` tests. No full suite (H2S-NN `src/` untouched; golden/mb0 cannot change); state `git diff --stat fc0ebaa -- H2S-NN/src/models H2S-NN/src/trainers H2S-NN/configs/experiments` = empty.

## Rules
- Correctness only: no real-data run on the lab cluster (no smoke, no timing, no metric). The first TSUBAME line of each new cell is its real-data check.
- Vendor code is never edited; adapters adapt at run time; no `vendor/` symlink.
- Never touch H2S-NN model, trainer, launcher or experiment configs.
- Lab jobs: `sbatch` with `--mem` and `--time`, no A6000 / `g3an01`.
- Git: one commit per unit; stage by explicit path; never `git add -A`, stash, reset or rebase; push only `card8`; never touch `v3b`.

## Stop conditions
- A published source key cannot be executed exactly (C1).
- The register cannot be mapped to cells (C0).
- MAGNN's or R-GCN's vendor code needs editing to run on our data.
- Any change needed outside `src/baselines/`, `scripts/`, `configs/baselines/`, `tests/`, `docs/card8_evidence/` and the report.
- Any unexpected test or mutation outcome.

## Report
`docs/H2S-NN_card8_report.md`: unit table (commit, job id, result); the C0 cell table; line counts per phase; deviations with reasons; open questions. Last line:
`branch=card8 sha=<full head as pushed> card=docs/H2S-NN_card8_published_rgcn_magnn_card.md base=fc0ebaa605280f2d987c4848cc3109c7d3b66680`

## Estimate
2–3 agent days; MAGNN (instance preprocessing on our derived LP tasks) is the longest part. Longer if HGB's MAGNN loaders are tied to their own file formats.

## Amendment 1 (2026-11-20, author, after the C1 stop; decisions log row "Card 8 amendment 1")
Continue from `card8` head `1236f4c`; first `git merge` the design-docs SHA given in the prompt.

**A1 — the 16 ‡ cells and MAGNN Freebase.**
- GCN, GAT × {mag-year, oag-cs, oag-eng, oag-chem, pdns} (10) and GCN × h-pokec (1): **published, run with H2GB's own `HomoGNNModel`** (`layer_type` GCN / GAT) exactly as each yaml states, through the same H2GB code path as R-GCN (C2). The tuned GCN/GAT cells keep HGB's models (decision S1). The report lists the two implementations per row.
- GAT × h-pokec: **tune** (`Pokec-GAT.yaml:61` says `layer_type: GCN`).
- HINormer × {mag-year, oag-eng, pdns}: **tune** (H2GB's `HINormerModel` is not the model we run, G-1).
- PSHGCN × ogbn-mag: **tune** (its batch 10,000 and pass-based epoch would need a second H2GB loop).
- MAGNN × freebase: **published** from HGB `run_Freebase.py` (confirmed).
- Test of C1 for the H2GB path uses GCN on one of these datasets (instead of GCN ogbn-mag, which is filled).

**A2 — evaluation protocol stays ours (author: option 1).** Published runs use the source's model and every hyperparameter / schedule key; data, splits, negatives, metrics, B1a selection and the single whole-split test stay the shared protocol (as C1 states). No change.

**C2b (new unit, before C2) — baseline LP step size as HGB (finding #1 of the ops audit).** Today the small-LP baseline loop takes one optimizer step per epoch on all training positives (`src/baselines/training.py:221-240`), while HGB's runner and H2S-NN take shuffled batches of 8,192 positives (`link_prediction_trainer.py:238-263`; HGB `LP/benchmark/methods/baseline/run_new.py`, batch loop). Make every baseline's small-LP training take `ceil(P / 8,192)` steps per epoch: positives in a seeded shuffled order, batch 8,192 (`LP_POSITIVES_PER_STEP`), the epoch's training negatives split in the same ratio (as H2S-NN), one encode per step. Applies to the shared loop and to every adapter with its own LP loop (list them with file:line; PSHGCN, H2SGNN, HAN, Hetero²Net, H2G-former, HetSheaf, NSD/CSNN, R-GCN, MAGNN). Large LP (`fit_large_lp`) already steps per 8,192 positives: unchanged. Test: steps per epoch = `ceil(P/8192)` on a synthetic task with P = 20,000 for the shared loop and for one own-loop adapter; **mutation:** one step per epoch → fails. Existing LP tests stay green. Record in the report which adapters changed.

**C2b addition (Amendment 2, 2026-11-20, author) — DistMult as HGB.** The baselines' `DistMultDecoder` (`src/baselines/heads.py:33-55`) is diagonal (`[R, d]`); HGB's is a full `d × d` matrix per relation (`HGB/LP/benchmark/methods/baseline/GNN.py:10-20`, xavier_normal), and H2S-NN's is full as well. Make the shared baseline DistMult a full `d × d` matrix per relation, initialised as HGB's (xavier_normal, gain as HGB's code), scored `uᵀ W_r v`; correct the docstring (`heads.py:36-39`). `dot` unchanged. Test: on a crafted pair u = e₁, v = e₂ a full W can give a non-zero score and `score(u,v) ≠ score(v,u)` for a non-symmetric W; parameter shape `[R, d, d]`; **mutation:** diagonal weights → fails. H2S-NN's decoder (with its Linear+GELU projection) is not touched; the projection is listed in the appendix as a difference.
