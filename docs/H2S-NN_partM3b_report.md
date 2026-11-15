# Part M3b report — the H2GB datasets, the large LP tasks and the stale items, on the lab cluster

Card `docs/H2S-NN_partM3b_card.md`; branch `part-m3b` from `v3b` `d6f3b883815c08d7408ff83689327f507fd500da` with
`design-docs` `a463268e51537139f0b5cbe7092b9b3608650d9f` merged `--no-ff` (merge commit `23faa32`). Everything ran under
SLURM in `h2snn-b` (the author's allocation 16741 on g2n01 via `srun --overlap`, or `sbatch`); the login node did git, editing
and submission only. Evidence lives in `docs/partM3b_evidence/` (every job script prints commit, host and job id). No training run
on real data, no metric reported as a result, no comparison between methods. Nothing under `data/` was deleted, overwritten or
committed; `data/` received the B6 task files (new folder `LP/large/`) and the B9 provenance entries (appended), both as the
card asks.

## Summary

| Unit | Commit | State | Decisive evidence |
|---|---|---|---|
| B0 preconditions | `6406823` | done | `b0_preconditions.md`, `b0_vendor_state.txt` |
| B1 vendor patch, helper, sbatch | `fc5d9aa` | done | `b1_gate.txt` (patched files byte-identical to the author's), `b1_tests.txt` |
| B2 H2GB NC loaders (7 of 8), counts, registry | `17b5451` | done | `h2gb_nc_counts.md`: all 7 equal Table 1 (nodes, edges, types); `b2_gate.txt`, `b2_mutation.txt` |
| B3 OAG identity | `0decdf8` | done | `oag_identity.md` |
| B4 venue-leak audit | `a06b001` | done | `leak_audit.md`: oag-cs / oag-eng / oag-chem pass |
| B5 PDNS | `9ccdf9a` | **stopped** (protocol ≠ card) | `pdns_protocol.md`, decision below |
| B6 large LP tasks + epoch loop | `7ab6a0f` | done | `b6_build_*.json`, `b6_mutation.txt`, `b6_gate.txt`, `b6_gate_inplace.txt` |
| B7 meta-path items | `ce3a3e1` | done (two decisions) | `b7_metapaths.md`, `b7_gate.txt` |
| B8 one budget table, adapters | `af8153e` | done (all adapters pending) | `b8_budget_and_cells.md`, `b8_gate.txt`, `b8_gate_inplace.txt` |
| B9 stale items | `1849364` | done (item 4 kept, stated) | `b9_gate.txt`, `b9_gate_inplace.txt`, `h2gb_provenance_section.md` |
| B10 full suite + this report | this commit | see B10 | `b10_suite_*.txt` |

Gates after every unit that touched `src/`: `v3b_golden.py compare` (1365 tensors, 24 configurations) and `mb0_reference.py
compare` (139 tensors) reported **differing: none** with E exported and 1 thread, every time (B2, B4, B6, B7, B8, B9; and
B10). Cluster references were never written.

## B0 — preconditions (VERIFIED-SOURCE / VERIFIED-CODE)
`origin/v3b` and `origin/design-docs` equalled the pinned SHAs after `git fetch`. `b0_preconditions.md` records how H2GB builds
each of the eight datasets (file:line), what training reads (`dataset[0]`) and the loader contract; `b0_vendor_state.txt`
records the vendor clone (`d3b49db`, the author's three modified files with sha256) and the sha256 of the protected files. One
observation of the first draft (PDNS "undefined `data`") was wrong — that block is a string literal — and was corrected before
the commit.

## B1 — vendor patch, data helper, sbatch
- `H2S-NN/vendor/patches/H2GB.patch` (the author's two edits: `ieee_cis_dataset.py`, `pokec_dataset.py` only) and
  `scripts/setup/apply_vendor_patches.sh` (pinned commit from `VENDORS.tsv`; reverse-check → `ALREADY APPLIED`, check → apply,
  else `CONFLICT`, exit 1); `clone_vendors.sh` calls it after cloning (one line).
- `scripts/data/h2gb_use_dataset.py` = the author's `examples/use_dataset.py` (usage lines and `H2GB_ROOT`/`DATA_DIR` only);
  `scripts/slurm/run_use_dataset.sbatch` on `h2snn-b`.
- VERIFIED-RUN (`b1_gate.txt`, login01, git only): a fresh clone + patch gives files **byte-identical** to the author's
  (sha256 `4b4cb7d1…`, `76570b56…`); a second application says `ALREADY APPLIED`; the author's clone and the protected files are
  unchanged before/after. Tests `tests/test_vendor_patches.py` (5) OK (`b1_tests.txt`).
- Finding: a **second `clone_vendors.sh` run refuses the patched clone** (`REFUSED H2GB: dirty clone`, exit 1) — the existing
  dirty-clone guard sees the patch. See Decisions.

## B2 — the H2GB NC loaders (seven; PDNS in B5)
`src/data/h2gb_nc.py` loads each dataset **through H2GB's own `preformat_*`** (H2GB's transforms, splits and target type) with
the H2GB class swapped for an offline subclass whose `_download`/`_process` only require the processed file (OAG's `raw/` is
empty and PyG would otherwise download). `build_dataset` reads it; registry and presets updated; the folders of
`data_layout.H2GB_NC_DIRS` follow B3.

Counts (VERIFIED-RUN, `h2gb_nc_counts.md`; jobs 16756–16765 on g3an01, h-pokec 16741 step 32 on g2n01; 60G requested, peaks
3.5–8.7 GiB, loads 11–42 s):

| Dataset | nodes (types) | Table 1 | edges (types) loaded | Table 1 | target feat (Table 1) | classes (Table 1) |
|---|---|---|---|---|---|---|
| ogbn-mag | 1,939,743 (4) | equal | 42,182,144 (7) | equal | 128 (128) | 349 (349) |
| mag-year | 1,939,743 (4) | equal | 42,182,144 (7) | equal | 128 (128) | 5 (5) |
| oag-cs | 1,112,691 (4) | equal | 27,537,448 (22) | equal | 768 (768) | 3,514 (3,514) |
| oag-eng | 929,315 (4) | equal | 12,346,854 (22) | equal | 768 (768) | **3,957 (3,956)** |
| oag-chem | 1,918,881 (4) | equal | 38,098,014 (22) | equal | 768 (768) | **2,986 (2,985)** |
| ieee-cis-g | 153,880 (12) | equal | 2,873,472 (22) | equal | **4,785 (4,823)** | 2 (2) |
| h-pokec | 1,731,977 (16) | equal | 51,774,836 (31) | equal | **65 (66)** | 2 (2) |

The bold differences equal H2GB's processed files and code (not a stop, card "Not stop conditions"); classes are counted as
`max y + 1`. mag and mag-year share graph and features; only labels and masks differ (`b2_counts_compare-mag.json`).
Mutation (`b2_mutation.txt`): letting the offline class download makes `test_oag_with_empty_raw_never_downloads` fail.
Gate `b2_gate.txt`: 133 tests OK, golden/mb0 none.

## B3 — OAG identity (VERIFIED-CODE + VERIFIED-RUN)
H2GB's `oag_dataset.py:55-57` maps the name `engineering` to the hub file `OAG_Chemistry.pt` and `chemistry` to
`OAG_Engineering.pt`. Counting both folders (jobs 16759/16760, kept in `b3_counts_by_folder/`) settles it against Table 1:
**folder `OAG/engineering` holds the chemistry field (1,918,881 nodes); folder `OAG/chemistry` holds the engineering field
(929,315 nodes).** `oag-eng` and `oag-chem` load the verified folder (`oag_identity.md`).

## B4 — venue-leak audit
`src/data/leak_audit.py` (inventory of node types / relations / attributes for label-source names, and a label-fingerprint probe:
majority share of a relation's neighbours against a permuted-label control; a relation carries the label at share ≥ 0.95, fixed
in code before the audit). Job 16771 (g3an01, 37 s, 10.3 GiB). **oag-cs, oag-eng, oag-chem pass**: no venue type, no paper–venue
relation in `dataset[0]` or in the flattened relations; highest share 0.60. The other datasets are informational: no edge type
carries the label. The probe is my reading of F18 for a node label (stated in `leak_audit.md`).

## B5 — PDNS (stopped)
VERIFIED-CODE/RUN (`pdns_protocol.md`, job 16773, g3an01, 82 s, 6.5 GiB): H2GB's code is **transductive**, not the temporal
protocol of the card: one graph over windows 0–59, a random 70/20/10 split of its labelled domains (unseeded `randperm`, fixed by
`seed_everything`), and the per-window test graph it builds (`test_list=[6]`) is never read. Counts equal Table 1 (1,173,558
nodes, 76,797,104 edges); labelled 87,252; split 61,077 / 17,450 / 8,725; window 60 has 547 newly labelled domains. No loader or
registry change; PDNS stays data-pending. Options under Decisions.

## B6 — the two large LP tasks and the large-LP epoch loop
Code: `src/data/large_lp.py` (tasks, unordered pairs, split 81/9/10 by unordered pair seed 42 with the rule of
`build_derived_lp.split_facts`, `held_out_fact_mask` over both orientations and every relation), `remove_held_out_facts` uses it;
`src/data/hgb_lp_negatives.py` gains a chunked 2-hop candidate computation (never a dense product; equal to the unchunked one, test)
and an exact list-free equivalent of `random.choices` (test); `LinkEpochLoop` in `src/trainers/minibatch_lp.py` (8,192 positives
per step, 1:1 negatives from `hgb_train_negatives` seeded per epoch, `TrainingOrder` over positives, AUROC on all validation
links each epoch, patience counted in epochs, `logged_stop_conditions`); `scripts/lp/build_large_lp.py` (never overwrites).

Builds (VERIFIED-RUN; jobs 16779 / 16780 on g3an01, requested 40G / 6 h — the script header says 128G, overridden on the
command line because 128G is not schedulable; the code was the B6 working tree on `9ccdf9a`, logged):

| | ogbn-mag P–P | H-Pokec U–U |
|---|---|---|
| target relation | `paper\|cites\|paper` | `user\|has_friend\|user` |
| facts loaded / unordered pairs | 10,792,672 / 5,396,336 | 44,603,928 / 22,301,964 |
| pairs with > 1 fact | 5,396,336 — every pair, as its two orientations (H2GB stores `cites` symmetrised) | 22,301,964 — every pair, as its two orientations after `ToUndirected` (stored: 8,320,600 of 22,301,964 in both) |
| repeated identical facts / self loops / pair in > 1 relation | 0 / 0 / 0 | 0 / 0 / 0 |
| split train / valid / test (pairs) | 4,371,032 / 485,670 / 539,634 | 18,064,591 / 2,007,177 / 2,230,196 |
| facts removed from the message graph | 2,050,608 (other relations 0) | 8,474,746 (other relations 0) |
| **held-out facts left, either orientation** | **0** | **0** |
| test negatives 2-hop / random; valid | 539,634 / 539,634; 485,670 | 2,230,163 / 2,230,196; 2,007,177 |
| heads without 2-hop candidates | 0 of 256,755 | **33 of 648,970 (5.1e-5)**; 33 test positives without a 2-hop negative |
| 2-hop time; peak RSS | 14,367 s; 15.76 GiB (sacct 15.5 GiB) | 1,558 s; 9.16 GiB (sacct 9.8 GiB) |
| files (sha256 in `meta.json`) | `split.npz` `5009402e…`, `negatives.npz` `38fde8f5…` | `split.npz` `9ca8efae…`, `negatives.npz` `6e715184…` |

**Pair duplicates:** every pair carries two facts, its two orientations; `remove_held_out_facts` removes both (0 left), so the
removal covers them — not a stop. **F17 / F18** (`large_lp_feature_rule.md`): no feature of `paper` or `user` is computed from
citations / friendships (attribute inventory; the fingerprint probe is not run for LP tasks, per J6.2) — no stop. The `x` of
`paper` as OGB's word2vec text average is recalled, not opened (UNVERIFIED, stated there).

Tests (`tests/test_large_lp.py`, synthetic): (i) shared endpoint computed once and equal to its own scope, float64 atol 1e-10,
with the negative control; (ii) no held-out fact in any scope of a step, both orientations; (iii) epoch / patience counting (3
cases); (iv) split by unordered pair (2); (v) `concat_mlp` refused; chunked 2-hop equality; `_choices` equality. Mutations
(VERIFIED-RUN `b6_mutation.txt`, 16741 step 44): scopes including the other endpoints' cells → (i) fails (difference 0.119);
removal in one orientation → (ii) fails ("2 held-out fact(s) present"); both restored byte-identical, suite OK.
Registry: `LARGE_LP_BUILT`; the two LP cells lose `DATA_PENDING_REASON` (files in place) and say adapter pending.

Gate: run on a `git archive` export of `7ab6a0f` (so later units' uncommitted edits are excluded; `m3b_gate_job.sh` with
`M3B_TREE`): golden and mb0 **differing: none**; of 163 tests, 12 fail **only** because the export's vendor clones are symlinks
(`vendor_paths.py:88` resolves module files to the real path, so H2GB's modules are not evicted and register twice); those
modules pass in place (`b6_gate_inplace.txt`, 103 OK). The B6 gate files went into the B7 commit (they could only exist after
the B6 commit); likewise each later unit's gate record is in the next commit.

## B7 — meta-path items (`b7_metapaths.md`)
- **Hetero²Net on MAG** (VERIFIED-CODE/RUN): their MAG branch uses PyG `OGB_MAG(preprocess='metapath2vec')` — paper features are
  OGB's, **bit-identical** to the B2 loader's (`b7_mag_features.txt`, 16741 step 43); author / institution / field features are
  PyG's pretrained MetaPath2Vec embeddings from `mag_metapath2vec_emb.zip` (**not on disk**), whereas H2GB averages paper
  features. Decision below.
- **SeHGNN / PSHGCN on H-Pokec** (their own enumeration on the schema, no propagation; 16741 step 42): SeHGNN 48 / 319 / **1,055**
  feature keys at 2 / 3 / 4 hops (≈ 1,050 expected); stored propagated features 1,213.6 / 10,877.7 / 29,685.4 GiB. PSHGCN 32 /
  303 / 1,039 / 5,840 / 21,681 terms per pass for K = 1–5; activations 6.6–105.7 GiB at K = 1. Only PSHGCN K = 1 with hidden ≤ 64
  fits a GPU (A6000). Node memory from `sinfo` and GPU memory from CLAUDE.md: `docs/lab-gpu-cluster.html` could not be read (the
  session's permission policy refused the read). Test `tests/test_metapath_counts.py` (3, hand-counted schema). Gate
  `b7_gate.txt` (export of `ce3a3e1`): none / none, 3 OK.

## B8 — one budget table; adapters
`src/configs/config.py`: `BATCH_BUDGETS` beside `PROTOCOL_BUDGETS`, `budget_class`, `batch_budget`, `budget_table()`, and
`LP_POSITIVES_PER_STEP = 8192`, which H2S-NN reads (the `data.link_batch_size` default and the large-LP loop; values unchanged).
`src/baselines/budget.py`: `resolve_budget`/`budget_kwargs`/`record_budget` take the dataset, and `batch_plan(task, dataset)`.
**Bug fixed:** `resolve_budget(task, params)` ignored the dataset, so a tuned baseline on an H2GB dataset would have got the HGB
budget (300 epochs, patience 30) instead of 500 epochs without early stopping (VERIFIED-RUN by test).

| class | epochs | patience | mode | batch sizes | epoch | positives / step |
|---|---|---|---|---|---|---|
| hgb_nc | 300 | 30 | full batch | — | one full-graph step | — |
| h2gb_nc | 500 | none | mini-batch | 32, 256, 1024 | 8,192 training targets | — |
| local_lp | 40 | 40 | full batch | — | one pass over training positives | 8,192 |
| large_lp | 40 | 40 | mini-batch | — | one pass over training positives | 8,192 |
| gc | 200 | 50 | per graph | — | one pass over training graphs | — |

**No adapter can consume the B2/B6 bundles without changes beyond its adapter** (`b8_budget_and_cells.md`): every `supports()`
lists HGB / local datasets only; NC training is full-batch (`train_full_batch_node_classification`, `training.py:50`), LP
training embeds the whole graph each step (`training.py:168-183`) and does not read `split.npz`/`negatives.npz`. HAN, SeHGNN,
PSHGCN, H2SGNN read HGB raw files; SeHGNN/PSHGCN also have the B7 memory finding; Hetero²Net waits on the B7 decision. Each
registry cell on the 7 H2GB NC datasets and the 2 large LP tasks now carries its adapter's own "adapter pending: …" reason; no
adapter declares `READS_BATCH_PLAN`; none was wired by guessing; no model code changed. Tests `tests/test_budget_table.py`; M9
`test_shared_protocol` unchanged and OK. Gate `b8_gate.txt` (export of `af8153e`, 122 OK) and `b8_gate_inplace.txt`
(`test_baselines`, `test_h2gb_nc`, 92 OK); none / none.

## B9 — stale items
1. CLAUDE.md / AGENTS.md: v2 golden names, `*_h2snn.yaml` and D3's η replaced by `test_v3b_golden` / `v3b_golden.py` / mb0 and the
   `*_v3b.yaml` files — these lines only.
2. The eight `h2snn_*_colab.ipynb`: each points at an existing `configs/experiments/*_v3b.yaml` or, where none exists, starts from
   the dataset's preset (`{'data': {'dataset_name': …}}`; for the tuning notebooks written at run time as
   `preset_base_config.yaml` next to the run's outputs, since their scripts take a file). Edited by
   `docs/partM3b_evidence/b9_notebooks_edit.py` (each replacement must match once; outputs and execution counts cleared; JSON layout
   kept). New test `tests/test_notebook_configs.py`.
3. `_dgl_compat.py` docstring: DGL 2.4.0+cu124 conda package via `install_env_b.sh`.
4. **`--decoder_type` of `run_graph_classification_repeated_cv_wandb_tuning.py:59` is kept**: it is read —
   the script calls `_apply_base_cli(raw, args)` (`:341`), which reads `args.decoder_type` (`run_h2snn_wandb_tuning.py:292-293`);
   removing the argument raises `AttributeError`. Card condition "after checking it is read nowhere" not met.
5. `test_wandb_and_runs.py`: the two `include_decoder_type=False` leftovers removed.
6. `path_sampler._pick` without `counts` and its branch; golden/mb0 none after the change (`b9_gate.txt`).
7. `data/PROVENANCE.md`: H2GB section appended (sources, H2GB classes, commit `d3b49db`, processed files and raw files with size
   and sha256, the B1 patch, the OAG identity, the date) plus the B6 task files (job 16741 step 53; text kept as
   `h2gb_provenance_section.md`); existing lines unchanged; not committed (`data/` is git-ignored).
Gate `b9_gate.txt` (export of `1849364`: notebook configs, WISP, W&B tests, 67 OK) and `b9_gate_inplace.txt` (`test_baselines`
incl. notebook-clean, 85 OK); none / none.

## B10 — full suite
VERIFIED-RUN at `184936452009e998fcc9db61a5d9c31dbffb1b05` (B9 head; the working tree had no tracked change), one run per CPU type,
`scripts/slurm/run_tests.sbatch` (E exported):

| Node (CPU) | Job | Result | Bit-exact compares (`m3b_gate_job.sh B10`) |
|---|---|---|---|
| g2n01 (i9-10900K, AVX2; 2080 Ti) | 16741 (allocation, `srun --overlap`), 788 s | **Ran 671 tests, OK** (`b10_suite_g2n01_16741.txt`) | golden 1365 / mb0 139 tensors, **differing: none** (step 57, `b10_gate_g2n01_16741.txt`) |
| g3an01 (i9-10900X, AVX512) | 16786 (`sbatch --nodelist=g3an01 --gres=none`), 846 s | **Ran 671 tests, OK (skipped=1)** — the skip is `test_cuda_float32_matches_cpu_float64_reference` ("CUDA not available"): g3an01's only GPU was held by another user's job, so this run is CPU only, as for the legacy removal (`b10_suite_g3an01_16786.txt`) | job 16787, **differing: none** (`b10_gate_g3an01_16787.txt`) |

671 = the 635 tests of the legacy removal + 36 new (`test_vendor_patches` 5, `test_h2gb_nc`, `test_leak_audit` 4,
`test_large_lp`, `test_metapath_counts` 3, `test_budget_table`, `test_notebook_configs` 1); the notebook-clean check passes.

## Decisions for the author
1. **PDNS protocol (B5).** (a) H2GB's code exactly — transductive, windows 0–59, H2GB's random 70/20/10 split from the run seed
   (recommended: "follow the source's code exactly", and it is the protocol behind H2GB's numbers; the spec's "temporal inductive"
   line would be corrected); (b) the temporal protocol of the card with window 60 as test (547 test domains), a stated deviation;
   (c) H2GB's unused per-window test graph. PDNS stays data-pending until you choose.
2. **Hetero²Net features on ogbn-mag / mag-year (B7).** (a) their MetaPath2Vec embeddings for author / institution / field
   (download `mag_metapath2vec_emb.zip` once, by you) — recommended, feature preprocessing is part of the method (spec §6);
   (b) H2GB's averaged features for every method (no download; a stated deviation from their MAG branch).
3. **SeHGNN and PSHGCN on H-Pokec (B7).** Under their own code only PSHGCN K = 1 (hidden ≤ 64) fits a node; SeHGNN fits nowhere.
   Search ranges were left unchanged; which cells to run, restrict or footnote is yours.
4. **Adapters (B8).** All twelve baselines are "adapter pending" on the H2GB NC and large LP cells (reasons in the registry);
   wiring each to the bundle, `batch_plan` and the large-LP files is a further card.
5. **`clone_vendors.sh` re-run (B1).** After the patch the clone is "dirty" and a second run exits 1 (`REFUSED`), although the
   patch then reports `ALREADY APPLIED`. Accepting a clone whose only changes are the patch needs a change to the dirty-clone
   guard (outside B1's one line); the current behaviour is safe but noisy.
6. **Large-LP heads without 2-hop candidates (B6).** 33 H-Pokec test heads (5.1e-5) have no 2-hop candidate, so 33 test positives
   have no 2-hop negative (HGB's `get_test_neigh` behaviour, kept). Reported only.
7. **`--decoder_type` in the repeated-CV script (B9 item 4)** is kept because it is read; removing it needs `_apply_base_cli` to
   stop reading it for graph classification — your call.

## Deviations from the card
- **Memory requests:** 128G (card's B6 starting point / script header) is not schedulable on this cluster (only g3an01 has more
  than 64 GB, and long 128G jobs waited behind reservations); B2/B4/B5 used 60G and B6 40G / 6 h. Peaks 3.5–16 GiB; no job
  exceeded its request.
- **`docs/lab-gpu-cluster.html` was not read** (refused by the session's permission policy); node memory from `sinfo`, GPU memory
  from CLAUDE.md.
- **Gate records one commit late** (B6–B9): each unit was gated on an export of its own commit, so its record is in the next commit.
- **B6 build code** ran as the B6 working tree on top of `9ccdf9a` (logged in the build logs) and was committed as `7ab6a0f`
  after the builds.

## Out-of-scope findings
- `src/baselines/runtime/vendor_paths.py:88` resolves module paths (`Path.resolve()`) but compares them with an unresolved vendor
  root: with a symlinked vendor clone the isolation does not evict modules (H2GB: `Module with 'GraphormerBias' already defined`).
  Harmless in the repository layout; it breaks a symlinked checkout.
- When `data/NC` is absent, a test in the suite downloads HGB ACM / DBLP / IMDB through PyG (seen once: in the first B6 export,
  37 MB into `/mnt/data/peeradon/m3b_gate/7ab6a0f…/H2S-NN/data`, not the project's `data/`; which test triggers it is UNVERIFIED —
  check: run `test_baselines` with an empty `data/`).
- Scratch exports used by the gates remain under `/mnt/data/peeradon/m3b_gate/` (outside the repository; ≈ the repo's tracked
  size each, plus the 37 MB download). Removing them was blocked by the session's safety check; delete when convenient.

branch=part-m3b sha=<see final message> card=docs/H2S-NN_partM3b_card.md base=d6f3b883815c08d7408ff83689327f507fd500da
