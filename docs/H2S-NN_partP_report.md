# Part P report — baselines runnable on every local cell; Part O follow-up

**Card:** `docs/H2S-NN_partP_card.md` (P0b, P0, P1, P1b, P2, P3, P4). **Design authority:** spec v1.20. **Decisions:** `docs/experiment_settings_decisions.md` (rows "Part O review", "Part O follow-up"). **Branch:** `part-p` from `part-o` at `b883f3810d43487365f7c4c1e3b663d19f09312a` (`git rev-parse HEAD` checked before branching). **Evidence:** `docs/partP_evidence/`.
**Rule kept throughout:** correctness only. No smoke run on real data, no performance experiment, no metric reported as a result, no comparison between methods.

---

## P0b — Part O follow-up: the fixed-literature-schema row (spec v1.20 U1; D-O3, D-O4, D-O6)

**Scope.** H2S-NN's comparison row only (`is_literature_row`). No baseline, no WISP main-model output and no legacy schema row (`r` unset) changes.

**Changes (VERIFIED-CODE).**
- `src/data/wisp_sampler.py`: WISP's per-hop step is factored into two helpers, and WISP itself now calls them.
  - `open_actions` (unvisited fact count per action, plus the visited half-edges);
  - `unvisited_fact_step` (WISP's choice of a uniform fact of the chosen action to an unvisited node).
  - **Bit-identity (VERIFIED-RUN, `p0b_wisp_bit_identity.txt`, script `p0b_wisp_reference.py`).** WISP draws on the `test_wisp` fixture, for `ℓ_max` 2/3/4 with and without an adaptive proposal (6 draws, 66 tensors), were saved before the change and compared after it: `differing: none`.
- `src/data/literature_paths.py`:
  - **Hop rule (D-O3).** `sample_listed_walks` uses WISP's step with the action forced to each listed (relation, direction). A dead end (no unvisited fact of that action) discards the walk. D8's `sample_walks` is no longer used by the row.
  - **Mass (D-O4).** `sample_literature_walks` gives each walk of listed path π the mass `m = 1/(k·r_π)`. Cells are still merged by summing (`merge_length_group`).
- `src/configs/config.py`, **run name (D-O6).** `fixed_literature_schema_config` sets `exp_name = <exp_name>__literature_row` (`LITERATURE_ROW_SUFFIX`) and leaves `output_dir` unchanged. The diff must be exactly `{data.path_source, exp_name}`.

**Tests (VERIFIED-RUN).** New `tests/test_partP_literature_row.py`, 7 tests:
- **(a) Hop rule.** Hand graph: anchor P1, author A7 with papers P1, P2, P3.
  - Each P1–A7–P_j instance occurs with unconditional frequency 1/2 (10⁵ walks, < 4σ), and P1 is never chosen.
  - On the same one-action graph, `sample_wisp_walks` gives `log p = log ½` exactly for every walk, the same frequencies, and the same set of walks.
  - A dead end discards the walk: only author A8 (who wrote only P1) → 0 walks; A7 or A8 → acceptance 1/2 (< 4σ).
- **(b) Mass.** With k = 2 and r = 20,000, and P(not discarded) = ½ from P1, each path's mass sum is within 4σ of `(1/k)·½`. Without dead ends, the total per anchor is exactly 1.0 in float64 for (k, r) ∈ {(2,8), (4,16), (1,64), (2,64)}, and within 1e-15 for (3, 16), whose shares are not dyadic.
- **(c) Row vs `full`.** On K(6,4) neither has dead ends, and both totals per anchor are exactly 1. On the dead-end graph both are ≤ 1, and the row's total at P1 is < 1.
- **(d) Run directory.** The row's `exp_dir` ≠ the `full` run's, `output_dir` is equal, and building a row from a row raises.
- **(e) Part O tests.** Updated only where they asserted the old rule, each named here:
  - `tests/test_partO_literature_row.py::BuilderTests.test_builder_changes_only_the_path_source`: the expected diff is now `{data.path_source, exp_name}`, and the test asserts the suffix.
  - `…::DrawTests.test_sampler_asks_for_the_split`: it recorded calls to D8's `sample_walks`; it now records `sample_listed_walks` (same assertions: walks per path `(3,3,3,3,2,2)`, lengths, every target).
  - `…::RealListTests.test_lists_resolve_and_have_instances_on_the_real_graphs`: the instance check now draws with the row's own sampler instead of D8's walk (data read only).
  - All Part O tests (`test_partO_literature_row`, `test_partO_readout`, `test_partO_reload`) → **Ran 21, OK**.

**Module run (VERIFIED-RUN, `p0b_tests_output.txt`).** `python -m unittest tests.test_wisp tests.test_task_heads tests.test_partP_literature_row tests.test_partO_literature_row tests.test_partO_readout tests.test_partO_reload` → **Ran 81 tests in 300 s, OK**. This includes all of `test_wisp` and the bit-for-bit golden NC outputs of `test_task_heads`.

**Mutations (card) (VERIFIED-RUN).**
- `m = 1/r_π` → **FAILED (failures=6)**: both subtests of (b)'s expectation, and the exact total of 1 (`p0b_mutation_mass.txt`).
- D8's rejection walk (visited nodes allowed at each hop, non-simple walks rejected at the end) → **FAILED (failures=3)**: (a) frequencies off by about 104σ, and the dead-end acceptance (`p0b_mutation_d8.txt`).
- `literature_paths.py` was restored after each mutation.

---

## P0 — Preconditions and readings (no code)

**Project side (VERIFIED-CODE).**
- `src/baselines/protocol.py:1-20,100-114`: NC runs the authors' model with their head; LP and GC use the authors' encoder plus the shared head (`src/baselines/heads.py`). `fit_task` returns the keys of `report_metrics_for_task` plus the validation monitor.
- `registry.py:32-52`:
  - The derived LP tasks carry the pending reason "adapter support pending … SUPPORTED_LP_DATASETS".
  - Freebase NC carries "loader pending".
  - `ADAPTER_PENDING["hetsheaf"]` is at l.103-105.
- `bundles.py`:
  - `BASELINE_DATA_PINS` (l.56-63): untied, DFS, no resampling, `path_source: schemas`.
  - `HGB_RAW_DIRNAME` (l.65) and `HGB_DEFAULT_FEATS_TYPE` (l.43-47) cover ACM, DBLP and IMDB only.
  - `build_bundle` (l.364-417) accepts `overrides`, including `data.root`, which the tests use to point at synthetic releases.
- One LP adapter end to end: `hgb_gnn.py` → `_homogeneous_lp.fit_homogeneous_link_prediction`.
  - It uses HGB's homogeneous view of the training-only message graph and the shared link head (dot / DistMult).
  - Training is AdamW with lr/wd from the parameters, through `training.train_link_prediction` with `budget_kwargs` (`_homogeneous_lp.py:59-99`).
- The derived tasks have no config preset. Their targets are recorded in `scripts/lp/write_hgb_negatives.py:44-47` and match `split_meta.json`'s `target_relation`: acm_pa `rel_2`, dblp_ap `rel_0`, imdb_ma `rel_2`, freebase_bb `rel_0`.

**HetSheaf** (vendor clone `vendor/Heterogeneous-Sheaf-Neural-Networks` at `6964e1e`; paper `references/HETSHEAF.pdf`). Readings by a reader agent, with the decisive points re-checked by me (VERIFIED-RUN marked). Paths are relative to the clone.

- **(a) Variants.**
  - The main Tables 2–4 report **one** row, "HetSheaf (ours)" (paper p. 9), next to "Sheaf-NSD" (the type-agnostic `local_concat` learner).
  - Selection: Bayesian sweeps of 150 runs per restriction-map family, with the learner as a categorical, then the best validation configuration retrained on seeds 42–46 (paper App. C.1, C.1.1; `README.md:190-192,204`; `scripts/run_best_seeds.py:30-51`). The families are diag, bundle O(d), general (`configs/model/{diag,bundle,general}_sheaf.yaml`), attention (SheafAN, `transductive/disc_models.py:1743`) and poly (`*_poly_sheaf.yaml`).
  - Learners (`models/sheaf_gnn/config.py:99-120`, `sheaf_models.py`): NSD=`local_concat`, TE=`type_concat`, ensemble=`type_ensemble`, EE, NE, types, NT, ET, plus attention variants for non-diagonal maps.
  - Which family gives each main-row number is **UNVERIFIED**: the main values match no cell of Tables 11/14.
  - The GC main row is the SheafPool row of Table 1, so `pooling=late_token`.
  - The per-learner × map grids, PolyNSD and SheafAN are appendix tables (11–16).
- **(b) Sweep spaces** (`sweeps/{nc,lp,gc}/<dataset>/*.yaml`; dump in `p0_hetsheaf_sweeps.txt`):
  - Shared: `d` int 2–5, `layers` 2–8, `hidden_channels` {8,16,32,64}, `input_dropout`/`dropout`/`initial_dropout` U[0, 0.9], `output_dim` {32,64,128}, `learning_rate` {1e-5…5e-2}, `weight_decay` {1e-5…5e-3}, 500 epochs, bayes with `run_cap 150`.
  - Family additions: attention adds `heads` 1–8; poly adds `poly_layers_K` {2,3,4,5,8,12,16} and ChebyshevType1.
  - `initial_dropout` and `output_dim` have no effect (`config.py:68`; `lin2` is commented out in every forward).
  - LP and GC scripts never pass lr/wd to the model. LP uses AdamW defaults (`sheaf_link_predictor.py:105-117`); GC hard-codes 1e-3/5e-4 (`graph_classifier.py:155-161`).
  - NC builds AdamW plus `CosineAnnealingLR` (`node_classifier.py:178-192`). No gradient clipping anywhere.
- **(c) Heads.**
  - NC: `Linear(hidden_dim, C)` on the encoder output; cross-entropy, or multilabel soft margin for IMDB (`node_classifier.py:39,66-71`).
  - LP: `Linear(2·hidden, 1)` on `[h_u‖h_v]` with BCE (`sheaf_link_predictor.py:27,35-53`).
  - GC: pooling ∈ {mean, max, add, attention, `late_token`=SheafPool} (`graph_classifier.py:43-54`, `sheaf_graph_classifier.py:33-142`).
  - The encoder reads `data.x`, `data.node_type` and `data.edge_type` (long, contiguous ids, one-hot inside the learners, `sheaf_models.py:293-305`). Transductive models use the `edge_index` fixed at construction (`sheaf_base.py:18`).
  - The paper's per-type input projection and its NC L2 normalisation / LP JKNet are not in the code. There is one shared `lin1` on zero-padded features from `to_homogeneous`.
- **(d) Transforms** (`datasets/hgb.py:51-60`, VERIFIED-RUN by reading), in order:
  1. `T.Constant(node_types=None)`
  2. `T.ToUndirected()`
  3. **`SubsetTrainSplit(val_ratio=0.1)`** (not in the card's list; an unseeded 10% validation carve from `train_mask`, `datasets/utils.py:31-88`)
  4. `RemoveSelfLoops()`
  5. `T.NormalizeFeatures()`
  6. `T.RemoveDuplicatedEdges()`
  - Then `to_homogeneous()` (`hgb.py:73-82`) sets int64 `node_type`/`edge_type`, with `rev_*` edge types from `ToUndirected`.
  - LP (`link_pred.py:39-46,83-94`): Constant, ToUndirected, NormalizeFeatures, RemoveSelfLoops, then its own `RandomLinkSplit`.
  - GC (`graph_classification.py:11-111,230-252`): `ExtractTypes` (types from one-hot columns) and its own seed-42 split.
  - Points that conflict with shared splits, to be stated in P1: `SubsetTrainSplit` and HetSheaf's LP/GC splits must not be applied (shared splits rule).
- **(e) Dependencies (VERIFIED-RUN, `p0_hetsheaf_imports.txt`).**
  - `import hetsheaf` → `ModuleNotFoundError: No module named 'strenum'`; `lightning` and `hydra` are missing too.
  - `import torch_householder` → `CalledProcessError: where cl` (needs MSVC).
  - HetSheaf pins `requires-python >=3.13,<3.14`, `torch==2.8.0`, `lightning==2.6.0` and `torch-householder==1.0.1` (`pyproject.toml:16-29`). `h2snn` is Python 3.12.13 with torch 2.6.0+cu124 and PyG 2.8.0.

## P1 — HetSheaf adapter: **STOPPED (card stop condition)**

HetSheaf's code cannot run in `h2snn`, for two reasons:
- **Version conflict:** it requires Python 3.13 and torch 2.8 (the env has 3.12 / 2.6).
- **Missing pieces:** three packages (`strenum`, `lightning`, `hydra-core`), and the compiled `torch_householder`, which needs a C++ compiler absent on this machine.

The card says: "P0 finds that HetSheaf's code cannot run in `h2snn` (dependency conflict): stop P1, report the conflict; P2–P3 continue." So no adapter was written and `ADAPTER_PENDING["hetsheaf"]` is unchanged. I did not install packages or stub `torch_householder`. A stub would change their restriction maps (`orth=householder` is the default), and installing `lightning==2.6.0` against torch 2.6 is untested. The options are in **Decisions for the author (D-P4)**.

## P1b — H2G-former's encoder as H2GB's (spec v1.19 T2, D-P3)

**Changes.**
- New `src/baselines/adapters/_h2gformer_encoders.py`, with five parts:
  - **`encoder_stack(task, dataset, metapath_pe)`:**
    - H2GB datasets take their config's `node_encoder_name` exactly (no key): raw+label+MetaPath2Vec on ogbn-mag, mag-year, oag-chem, PDNS, H-Pokec; raw+label on IEEE-CIS-G, oag-cs, oag-eng.
    - HGB NC is raw+label, LP raw, GC raw, plus MetaPath2Vec when `metapath_pe` is on.
    - The key exists only where a walk schema exists: HGB NC (ACM, DBLP, IMDB, Freebase), the four derived LP tasks and LastFM. PubMed and GC have none.
  - **`walk_schema(dataset)`:** the listed meta-paths concatenated in table order.
    - NC uses `LITERATURE_METAPATHS`, the same HGB table as the row, as PyG edge types.
    - A derived LP task uses its source release's table at the head type, renamed to `rel_<HGB link id>` (`HGB_RELATIONS`, link-id order; Freebase's 36 relations from `info.dat`).
    - LastFM uses UU, UAU, UATAU.
    - A hop against a stored relation uses PyG's `rev_<relation>` (as `T.ToUndirected` names it). The MetaPath2Vec walk graph adds those copies (`with_reverse_edges`).
  - **`metapath2vec_embeddings`:** H2GB's own `MetaPath2Vec` class and training loop settings (ogbn-mag branch, `encoding_generator.py:256-270,520`: walk 64, context 7, 30 walks per node, 8 negatives, batch 128, SparseAdam lr 0.01, 5 epochs; `dim_pe` 128 and 1 layer from `configs/ogbn-mag/ogbn-mag-H2Gformer.yaml:21-24`). The result is stored as `pestat_Hetero_Metapath`, as `master_loader.py:266-281` stores it.
  - **`training_label_input`:** gives H2GB's `Hetero_Label` encoder a `y` whose non-training entries are a placeholder, and an all-False `val_mask`.
  - **`configure_encoders`:** sets graphgym's `node_encoder_name` and `posenc_*`.
- `h2gformer.py`: NC builds the model through `build_node_classification_model` with the resolved stack.
  - Evaluation runs with `batch.split = 'val'`. H2GB's `'test'` branch would add validation labels (`hetero_label_encoder.py:42-44`).
  - LP attaches MetaPath2Vec on the training message graph only (`build_edge_index_dict`).
  - The record names the stack. The search space gains `metapath_pe ∈ {True, False}` where the key exists.
- **Deviations, each stated (decisions D-P5, D-P6, D-P7):**
  - (i) IMDB has no label channel: H2GB's encoder one-hot encodes one label per node (`hetero_label_encoder.py:49-59`) and IMDB is multi-label.
  - (ii) With MetaPath2Vec on, H2GB's composed encoder gives the raw channel `hidden − dim_pe` dimensions (`composed_encoders.py:54-58`). `hidden_dim ≤ 128` now raises (a failed trial). Our grid {64, 128, 256} therefore leaves only 256 with the channel on; H2GB uses 256–512 with it.
  - (iii) Node types the table never visits (ACM term, IMDB keyword, several Freebase types) get a zero positional encoding. H2GB's own schemas visit every type of their graphs, so H2GB never meets this case.
  - H2GB datasets stay "data pending". Their stacks are wired and tested against their config files only.

**Tests (VERIFIED-RUN).** `python -m unittest -v tests.test_partP_h2gformer` → **Ran 10 tests in 327 s, OK** (`p1b_tests_output.txt`):
- **Stack.** The T2 table for every NC, LP and GC cell. `metapath_pe` on PubMed and on H2GB datasets raises. H2GB's stacks equal the `node_encoder_name` read from its eight config files.
- **Schemas.** Each is cyclic, and its consecutive hops meet. Every hop is an edge type of the graph, with `rev_` copies: by name for all nine datasets, and on the real ACM/DBLP/IMDB/Freebase graphs by PyG loading (data read only). The schemas equal hand-written concatenations for ACM, IMDB, LastFM and acm_pa, and the lengths for DBLP (10) and Freebase (17).
- **Settings.** MetaPath2Vec settings equal values parsed from H2GB's `encoding_generator.py` ogbn-mag branch and `ogbn-mag-H2Gformer.yaml`.
- **Route.** On a tiny ACM-shaped graph, poisoning every non-training label leaves the model output identical (`torch.equal`), also with H2GB's `'test'` split. Flipping training labels changes it.
- **MetaPath channel.** Embeddings are produced (`[N, 128]`, finite; `term` zero). The output changes when they are replaced, so they are consumed. `hidden_dim = 128` with the channel on raises.

**Mutations (VERIFIED-RUN).**
- All labels to the label channel (unmasked `y`, all-True masks) → the route test **FAILED** (`p1b_mutation_labels.txt`).
- The last listed meta-path dropped from every schema → **FAILED** on ACM, IMDB, LastFM and acm_pa (`p1b_mutation_schema.txt`).
- Restored after each mutation.

## P2 — Derived LP tasks for every LP baseline

**Changes (VERIFIED-CODE).**
- `src/configs/config.py`: the four derived tasks gain `DATASET_TASK_TYPES` entries and `LINK_PREDICTION_PRESETS` targets, so `build_bundle` builds them by name. The targets are those of `scripts/lp/write_hgb_negatives.py:44-47` and `split_meta.json`. No data, split or negative file changes.
- Every LP adapter's `SUPPORTED_LP_DATASETS` adds the four tasks: GCN/GAT (`hgb_gnn.py`), NSD, CSNN, HAN (`hgb_han.py`), PSHGCN, H2G-former, Hetero²Net, H2SGNN.
- `registry.py`: the derived-LP pending reason is removed, because no derived cell needs one. HetSheaf stays under `ADAPTER_PENDING` (P1 stopped).
- `_polynomial_lp.py` (PSHGCN, H2SGNN and HAN's LP metapath operators):
  - **The problem.** Releases derived from HGB NC data store each relation's reverse as its own relation: paper→author `rel_2` / author→paper `rel_3`; ACM's cite/ref. The type-pair key check raised `NotImplementedError` on every derived task.
  - **The rule.** A relation whose message-graph edge set is exactly the transpose of an earlier relation (types swapped) adds no operator; the kept relation's transposed key, or its same-type symmetrisation, already is that operator. Genuinely parallel relations still raise.
  - **Unchanged elsewhere.** The key sets of LastFM and PubMed cannot change, because only cases that raised before are affected.

**Per-adapter points (card).**
- **(i) Featureless types after F17** (ACM author, DBLP author, IMDB actor: blocks removed). Each adapter's own rule applies, unchanged: the homogeneous view's `feats_type` identity fallback, `TypedNodeInput`'s learned lookup, and HAN/HGB's own handling. The tiny releases contain such types (author after `apply_rule`, plus an attribute-less type).
- **(ii) Reverse relations of the target.** The loader removes held-out facts in both orientations. `build_derived_lp.build_task` drops the reverse rows, and the loader's guards (`_assert_reverse_partner_is_tied`, `held_out_fact_leaks`) refuse a task that keeps them. The new test checks every adapter-side graph.
- **(iii) PSHGCN/H2SGNN keys.** Handled by the stored-reverse rule above.
- **(iv) The 2-hop set.** The 2-hop "–" cells (dblp_ap, imdb_ma: no 2-hop path from the head type to the tail type) already yield `None` with `test_2hop_reason` through `lp_test_set_metrics` (Part N); tested end to end here.

**Tests (VERIFIED-RUN).** New `tests/test_partP_derived_lp.py`, built on the project's own pipeline: `tests.test_lp_derived.write_release` or a DBLP-shaped writer, then `build_task`, `apply_rule`, and `build_bundle` by name with `data.root` set to a temp dir. `python -m unittest -v tests.test_partP_derived_lp` → **Ran 5 tests in 154 s, OK (skipped=4)** (`p2_tests_output.txt`):
- **One-epoch fits.** GCN, GAT, HAN, H2G-former and Hetero²Net each return finite `test_random_*` and `test_2hop_*` AUROC/AUPR.
- **Skipped for environment reasons, each with its reason in the output.** These four are **not verified locally**:
  - NSD, PSHGCN and H2SGNN: their vendor clones are empty on this machine.
  - CSNN: `torch_householder` needs MSVC `cl`.
  - The skip is an explicit precondition (`environment_blocker`), not caught errors.
- **No held-out pair** (validation or test, either orientation) is in the bundle's message graph, HGB's homogeneous view, the typed edge dicts, or the polynomial relation operators.
- **Keys.** The polynomial keys on the tiny ACM-shaped task are {AA, AB, BA, AC, CA}, with self-loops only on AA. Each operator equals its hand edge set: AB = paper→author facts, BA its transpose, AA = cite ∪ citeᵀ plus the full-matrix diagonal of their `set_diag()`. Parallel relations still raise.
- **Empty 2-hop set.** A DBLP-shaped author→paper task has an empty HGB 2-hop set. GCN's record gives `None` for both 2-hop metrics with the bundle's reason, and finite random-set metrics.
- **Existing test updated (named).** `tests/test_baselines.py::RegistryTests.test_campaign_matrix_matches_planned_cell_counts` pinned the matrix at 90 runnable cells (LP 18). The derived tasks add 4 × 9 LP cells, giving **126 (LP 54)**. Only those two numbers and the comment changed.
- Related existing tests: `test_baselines` Registry/AdapterContract/SearchSpaceCoverage/PolynomialRelationKey, `test_lp_derived`, and `test_wisp` WispDefault/WispConfigValidation → Ran 54. One failure before the count update: the count test above. After it, `RegistryTests` → OK (`p2_related_tests_output.txt`).

**Mutation (VERIFIED-RUN, `p2_mutation_reverse.txt`).** `build_task` keeps the reverse rows of test facts → the bundle refuses to build (`ValueError: … a held-out fact could leak into the message graph as its reverse row`), and the P2 tests **FAILED (errors=1)**. The loader guard catches it first; the adapter-side check behind it was shown above on the clean task. Restored.

## P3 — Freebase NC for H2S-NN and every NC baseline

**H2S-NN (VERIFIED-CODE).**
- `src/configs/config.py`: `DATASET_TASK_TYPES["freebase"]` and the NC preset (target `book`, no old preset list).
- `src/data/dataset.py`: `_load_hetero_data` reads `HGBDataset(name="Freebase")`.
- Nothing else is new:
  - HGB's labels go through the existing `_prepare_target_labels` (24/6/70 as for ACM/DBLP/IMDB).
  - Unlabelled books (`y = -1`) are in no mask, so they enter neither loss nor metrics.
  - Every type is featureless, and the encoder's existing per-node path handles that.
  - The fact layer is the existing D7 rule: relations without a stored reverse stay untied.
- Full batch (spec v1.17 Q3); not run here.

**Baselines (VERIFIED-CODE).**
- `bundles.py`: `HGB_RAW_DIRNAME["freebase"]`, and `HGB_DEFAULT_FEATS_TYPE["freebase"] = 0`. On Freebase that is the identity fallback for every type. It also makes the table-driven adapters (GCN, GAT, ChebNetII, NSD, CSNN, HiNormer) accept Freebase.
- HAN reads HGB's own `load_freebase` (`HAN/utils.py:175,337-341`) and, as HGB's script does, builds `HAN_freebase` there (`HAN/main.py:45-52`; an FC layer on the sparse identity input). With plain `HAN`, its dropout fails on the sparse input.
- Hetero²Net uses its own `'FreeBase'` branch (`hetero2net/dataset.py:48-53`).
- H2G-former reads Freebase through PyG.
- **SeHGNN, PSHGCN, H2SGNN keep a specific registry reason** (`FREEBASE_VENDOR_REASON`). They read HGB's files through their own loaders, and their clones are empty on this machine. I could not read whether their code has a Freebase branch, so wiring them would be guesswork (UNVERIFIED).
- `registry.py`: the "loader pending" reason is removed.
- `runtime/vendor_paths.py` (bug found by the tests): `pushd_sys_path` evicted only other vendors' modules. So once this project's own `scripts` package had been imported, HGB HAN's `from scripts.data_loader import …` failed (`ModuleNotFoundError: No module named 'scripts.data_loader'`). Now any loaded top-level module whose name is also a module or package at a pushed vendor root is displaced for the block and restored afterwards.

**Real Freebase (VERIFIED-RUN, data read only).** The loader reproduces the M1 gate (`docs/partM_evidence/m1_data_counts_output.txt:36-39`):

| quantity | value |
|---|---|
| nodes per type | book 40,402, film 19,427, music 82,351, sports 1,025, people 17,641, location 9,368, organization 2,731, business 7,153 |
| facts before / after tying | 1,057,688 / 869,706 |
| labelled books | 7,954 |
| class counts | [429, 3886, 1748, 185, 962, 359, 385] |
| split train / val / test (24/6/70) | 1,909 / 477 / 5,568, disjoint |
| unlabelled books (in no mask) | 32,448 |

- **Untied fallback, per relation.** 29 relations have no stored reverse and stay untied facts (D7), equal to a direct check of `link.dat`: `book__and__book`, `book__to__film`, `book__on__sports`, `book__on__location`, `book__about__organization`, `music__in__book`, `music__in__film`, `music__for__sports`, `music__on__location`, `sports__in__film`, `sports__on__location`, `people__to__{book,film,music,sports}`, `people__on__location`, `people__in__{organization,business}`, `location__in__film`, `organization__in__film`, `organization__to__{music,sports}`, `organization__on__location`, `organization__for__business`, `business__about__{book,film,music,sports}`, `business__on__location`. The other 7 (`*__and__*` of film, music, sports, people, location, organization, business) are self-inverse.
- **Row and read-out on Freebase (spec v1.20; Part O out-of-scope finding 1).** `fixed_literature_schema_config` builds the row from a Freebase `full` configuration. Its evaluation draw on the loader's graph contains all 7 listed paths and no other full-path signature, with BB as two book–book hops.

**Tests (VERIFIED-RUN).** New `tests/test_partP_freebase.py`, plus `tests/partP_freebase_fixture.py` (a tiny release in HGB Freebase's raw format: TYPE/LINK `info.dat` sections, 36 relations, unlabelled books). `python -m unittest -v tests.test_partP_freebase` → **Ran 6 tests in 170 s, OK (skipped=3)** (`p3_tests_output.txt`):
- the four real-data tests above;
- `scripts/schema_readout.py` end to end on a synthetic 1-epoch H2S-NN Freebase run: JSON with the 7 listed paths, plus the table (no value read);
- one-epoch `fit_task` on the tiny release (finite `test_macro_f1`, `test_micro_f1`, `val_macro_f1`) for GCN, GAT, HAN, HiNormer, H2G-former and Hetero²Net;
- ChebNetII, NSD (empty clones) and CSNN (no MSVC) skipped with the reason: **not verified locally**;
- SeHGNN, PSHGCN, H2SGNN carry `FREEBASE_VENDOR_REASON`, and their `supports()` agrees.
- **Existing test updated (named).** `test_baselines.py::RegistryTests.test_campaign_matrix_matches_planned_cell_counts`: runnable 126 → **135**, NC 36 → **45**. Only the counts and the comment changed.

---

## P4 — Registry, coverage matrix, full suite

**Coverage matrix (VERIFIED-RUN, `docs/partP_evidence/coverage.json`, from `registry.iter_cells()` plus the HetSheaf `ADAPTER_PENDING` rows).**
- Registry: 288 cells (the 24 HetSheaf rows come on top), **135 runnable**: NC 45, LP 54, GC 36.
- These per-task counts are the input of the tuning cost.

| model | NC acm/dblp/imdb | NC freebase | NC H2GB (8) | LP derived (4) | LP pubmed/lastfm | LP large (2) | GC TU (4) |
|---|---|---|---|---|---|---|---|
| gcn | runnable 3/3 | runnable 1/1 | 0/8: data pending | runnable 4/4 | runnable 2/2 | 0/2: data pending | runnable 4/4 |
| gat | runnable 3/3 | runnable 1/1 | 0/8: data pending | runnable 4/4 | runnable 2/2 | 0/2: data pending | runnable 4/4 |
| chebnetii | runnable 3/3 | runnable 1/1 | 0/8: data pending | 0/4: no implementation | 0/2: no implementation | 0/2: no implementation | 0/4: no implementation |
| nsd | runnable 3/3 | runnable 1/1 | 0/8: data pending | runnable 4/4 | runnable 2/2 | 0/2: data pending | runnable 4/4 |
| csnn | runnable 3/3 | runnable 1/1 | 0/8: data pending | runnable 4/4 | runnable 2/2 | 0/2: data pending | runnable 4/4 |
| han | runnable 3/3 | runnable 1/1 | 0/8: data pending | runnable 4/4 | runnable 2/2 | 0/2: data pending | runnable 4/4 |
| sehgnn | runnable 3/3 | 0/1: not wired (vendor clone) | 0/8: data pending | 0/4: no implementation | 0/2: no implementation | 0/2: no implementation | 0/4: no implementation |
| hinormer | runnable 3/3 | runnable 1/1 | 0/8: data pending | 0/4: no implementation | 0/2: no implementation | 0/2: no implementation | 0/4: no implementation |
| pshgcn | runnable 3/3 | 0/1: not wired (vendor clone) | 0/8: data pending | runnable 4/4 | runnable 2/2 | 0/2: data pending | runnable 4/4 |
| h2gformer | runnable 3/3 | runnable 1/1 | 0/8: data pending | runnable 4/4 | runnable 2/2 | 0/2: data pending | runnable 4/4 |
| hetero2net | runnable 3/3 | runnable 1/1 | 0/8: data pending | runnable 4/4 | runnable 2/2 | 0/2: data pending | runnable 4/4 |
| h2sgnn | runnable 3/3 | 0/1: not wired (vendor clone) | 0/8: data pending | runnable 4/4 | runnable 2/2 | 0/2: data pending | runnable 4/4 |
| hetsheaf | 0/3: adapter pending (P1 stopped) | 0/1: adapter pending (P1 stopped) | 0/8: adapter pending (P1 stopped) | 0/4: adapter pending (P1 stopped) | 0/2: adapter pending (P1 stopped) | 0/2: adapter pending (P1 stopped) | 0/4: adapter pending (P1 stopped) |

"Runnable" is the registry's statement: the cell is wired and its adapter's `supports()` agrees. **Locally verified by a one-epoch fit:** GCN, GAT, HAN, H2G-former, Hetero²Net (derived LP and Freebase NC) and HiNormer (Freebase NC). **Not verified on this machine:** NSD, CSNN, ChebNetII, PSHGCN, H2SGNN (empty clones / no MSVC). Their wiring goes through the same shared code as the verified ones, but no run of theirs was possible here.

**Full suite, run once (VERIFIED-RUN).** `python -m unittest discover -s tests` from `H2S-NN/` on `aa9f3de` → **Ran 632 tests in 1,070 s; FAILED (failures=8, errors=2, skipped=7)** (`p4_full_suite_output.txt`; INFO log lines of synthetic fixture trainings removed).
- Test count: 604 (Part O O4) + 7 (P0b) + 10 (P1b) + 5 (P2) + 6 (P3) = **632**.
- Skips: the 7 environment-blocked subtests of P2 and P3.

| failing test | status |
|---|---|
| `test_baselines.VendorIsolationTests.test_all_declared_vendor_clones_are_present` | known (F5/O4 set; empty clones) |
| `test_baselines.VendorIsolationTests.test_colliding_vendor_module_names_do_not_leak` | known |
| `test_baselines.ReproducibilityTests.test_vendor_model_init_is_reproducible_under_seed_everything` | known |
| `test_baselines.BaselineNotebookTests.test_notebooks_are_committed_clean` | known |
| `test_baselines.LaneNotebookTests.test_committed_notebooks_match_the_generator`, lanes `link_prediction_{core,dgl,h2gb,hetero2net,sheaf}` (5 subtests) | **new, from P2**. The lane notebooks are generated from the registry, and the derived tasks are now runnable. **Fixed in P4:** `python scripts/generate_baseline_notebooks.py` rewrote exactly these 5 notebooks; `--check` → "13 notebooks match the generator"; `tests.test_baselines.LaneNotebookTests` → **Ran 10, OK** (`p4_notebooks_after_regeneration.txt`). |
| `test_shared_protocol.OptimizerTests.test_every_adapter_builds_adamw` | **new, from P1b, left failing: decision D-P9.** `_h2gformer_encoders.py` builds `torch.optim.SparseAdam` for H2GB's MetaPath2Vec pretraining (H2GB's setting, spec T2). The test's rule is "AdamW everywhere" (spec J4). |


## Decisions for the author

| ID | Question | Default used / recommendation | Alternative |
|---|---|---|---|
| D-P1 | HetSheaf rows | Card default: one row per variant the main tables report. P0(a) found **one** main-table row ("HetSheaf (ours)", chosen over restriction-map families by per-family sweeps), so this means one HetSheaf row, with the map family and learner tuned as categoricals as in their sweeps. Not implemented (P1 stopped). | One row per map family (diag / bundle / general / attention / poly) |
| D-P2 | HetSheaf GC readout | Card default: HetSheaf's own readouts, incl. `late_token` (SheafPool; their GC main row uses it). Not implemented (P1 stopped). | The shared GC head |
| D-P3 | H2G-former without an H2GB config | Decided (spec T2); implemented in P1b | — |
| **D-P4** | **HetSheaf cannot run in `h2snn`** (Python 3.13 / torch 2.8 pins; `strenum`, `lightning`, `hydra-core` missing; `torch_householder` needs a C++ compiler) | Recommended: a separate requirement group or env for HetSheaf on the Linux cluster (compiler present). Before P1 resumes there, check whether their model code runs on torch 2.6 (only an import-level check was done here). | (b) Install `strenum`, `lightning==2.6.0`, `hydra-core` into `h2snn` and build `torch_householder` with MSVC (Windows). (c) Replace `torch_householder` with their own native `householder_orgqr` (`utils/linalg.py:5`): a change to their code, not recommended. |
| D-P5 | H2G-former's label channel on IMDB | No label channel on IMDB: H2GB's encoder is single-label (`hetero_label_encoder.py:49-59`) | A multi-label variant (our construction, not H2GB's) |
| D-P6 | H2G-former width with MetaPath2Vec on | `hidden_dim ≤ dim_pe = 128` raises (a failed trial). With the grid {64, 128, 256}, only 256 runs with the channel on. Recommended: keep `dim_pe = 128` (H2GB's) and add 512 to `hidden_dim`, as H2GB's configs use 256–512 with it. | Scale `dim_pe` with the width (not H2GB's setting) |
| D-P7 | Node types the HGB table never visits (MetaPath2Vec) | Zero positional encoding | Untrained random embeddings |
| D-P8 | Freebase NC for SeHGNN, PSHGCN, H2SGNN | Specific reason kept; wire their own Freebase preprocessing once the clones are available (cluster) | Exclude them from the Freebase column |
| **D-P9** | **Spec J4 (AdamW everywhere) vs spec T2 (H2GB's MetaPath2Vec settings: `SparseAdam`, needed for its sparse embeddings; AdamW rejects sparse gradients)**. `test_every_adapter_builds_adamw` fails on `_h2gformer_encoders.py`, and I left it failing rather than change the test's rule. | Recommended: J4 governs the training of the baseline model; MetaPath2Vec is H2GB's preprocessing of input encodings, so keep `SparseAdam` there and narrow the test to model optimizers, with this exemption named. | (b) AdamW with dense MetaPath2Vec embeddings (not H2GB's setting; memory). |

## Out-of-scope findings (not fixed)

1. **Vendor clones empty on this machine:** ChebNetII, Cooperative-Sheaf-NN, H2SGNN, PSHGCN, SeHGNN, neural-sheaf-diffusion (NSD), dgl, fasth, hgbi, tudataset, HSB-GDL, LatGRL (`find vendor/<x> -name "*.py"` → 0). NSD, PSHGCN, H2SGNN and ChebNetII cannot be exercised locally. Part N-fix F5 listed fewer missing clones, so the machine's vendor state has changed since (UNVERIFIED why).
2. **HetSheaf's code deviates from its paper.** The per-type input projection, the NC L2 normalisation and the LP JKNet readout are described but absent. The LP and GC scripts never pass the swept lr/wd to the model (LP uses AdamW defaults, GC hard-codes 1e-3/5e-4), and `initial_dropout`/`output_dim` are swept but have no effect. **Recommendation, to settle when P1 resumes:** run their code as written, state these points, and apply the shared lr/wd axes (shared protocol).
3. **`SubsetTrainSplit` is an extra HetSheaf transform** (`hgb.py:55`, unseeded 10% validation carve) that the card's list omits. It must not be applied, because splits are shared.
4. **Spec v1.20 U2 (Kendall τ over length-2 only)** is not in the card. `scripts/schema_readout.py` still writes τ over all spans and over length 2. Following U2 needs only the all-spans block dropped from the table; not done (out of the card's scope).
5. **H2G-former's NC data path** (`_build_hetero_data`) and Hetero²Net's loader read `REPO_DATA_ROOT`, not the bundle's `data.root`. The tests point the data root at the fixture. A campaign run with a non-default root would read the default one. Recommendation: thread the root through `NodeBundle.metadata`.
6. **CSNN on Windows** needs MSVC for `torch_householder` (as HetSheaf does), so CSNN cannot be run locally at all.
