# Part P-fix report — carry-over fixes from the Part P review

**Card:** `docs/H2S-NN_partPfix_card.md` (F0–F5). **Decisions:** `docs/experiment_settings_decisions.md`, rows "Part P review" (D-P6, D-P9) and "Part P follow-up"; spec v1.20 U2 / D-O7 for F3. **Branch:** `part-pfix` from `part-p` at `a08d97a046fffd135de109791fc1f47f0df562ee` (`git rev-parse HEAD` checked before branching). **Evidence:** `docs/partPfix_evidence/`.
**Rule kept throughout:** correctness only (unit tests, mutation checks, one-epoch fits on tiny synthetic fixtures, the full suite once). No smoke run on real data, no performance experiment, no metric reported, no comparison between methods.

| unit | commit | result |
|---|---|---|
| F0 | — (no code) | readings below |
| F1 | `19bf359` | AdamW test with one named exception (D-P9); 3 mutations caught |
| F2 | `bd71f9e` | H2G-former `hidden_dim` ∈ {64,128,256,512}, `dim_pe = min(128, hidden_dim // 2)`; mutation caught |
| F3 | `cdd54ca` | read-out τ over length-2 sub-path schemas only; mutation caught |
| F4 | `aed53a6` | every adapter reads the bundle's data root; no `REPO_DATA_ROOT` in `src/baselines`; mutation caught |
| F5 | this commit | full suite, report |

---

## F0 — Preconditions (VERIFIED-SOURCE / VERIFIED-CODE)

- **Decision rows read:**
  - D-P6: `hidden_dim` ∈ {64,128,256,512}, `dim_pe = min(128, hidden_dim/2)`, no invalid combination.
  - D-P9: keep `SparseAdam` for MetaPath2Vec pretraining; the test names this one exception.
  - Part P follow-up: the card.
  - D-O7 / U2: τ over length-2 only.
- **Part P report:** D-P6, D-P9 and out-of-scope findings 4 (read-out still printed two τ blocks) and 5 (H2G-former's and Hetero²Net's data paths read `REPO_DATA_ROOT`).
- **Code read:**
  - `h2gformer.py`: grid `hidden_dim` {64,128,256} at l.206, width at l.135.
  - `_h2gformer_encoders.py`: `configure_encoders` raised for `hidden_dim ≤ 128` with MetaPath2Vec; `dim_pe` was fixed at 128.
  - `tests/test_shared_protocol.py::OptimizerTests`: any non-AdamW optimizer failed.
  - `scripts/schema_readout.py:117-121`: two τ blocks, all spans and length 2.
  - `src/analysis/schema_readout.kendall_matrix`: `span` argument (the card writes `length=2`; the code's name is `span`).
- **`REPO_DATA_ROOT` sites in `src/baselines`** (grep before F4):
  - `adapters/h2gformer.py:520,523` (`_build_hetero_data`);
  - `adapters/hetero2net.py:57,84,91` (`_load_vendor_graph`);
  - `bundles.py:76,85` (`hgb_raw_dir`). Through it, a third path: HAN (`hgb_han._load_vendor_graph`), SeHGNN (`sehgnn._vendor_args`), PSHGCN and H2SGNN (`_hgb_vendor.load_vendor_dataset`).

## F1 — AdamW test and the MetaPath2Vec exception (D-P9)

- **Change:** `tests/test_shared_protocol.py::OptimizerTests.test_every_adapter_builds_adamw`.
  - It now reads every `optim.X(` call with `ast`, together with its enclosing top-level function, and checks that the regex view and the AST view agree.
  - It removes exactly one named exception, `PRETRAINING_EXCEPTION = ("_h2gformer_encoders.py", "SparseAdam", "metapath2vec_embeddings")`, with the reason and references (D-P9, spec v1.19 T2, J4) in one place. It also requires that this call occurs exactly once.
  - Every remaining optimizer must be AdamW.
  - No adapter code changed.
- **Tests (VERIFIED-RUN, `f1_tests_output.txt`):** `python -m unittest -v tests.test_shared_protocol` → **Ran 8, OK**.
- **Mutations (VERIFIED-RUN, `f1_mutations.txt`), each failing the test:**
  - (a) a model optimizer (`_homogeneous_lp.py`) → `Adam`: `{'_homogeneous_lp.py': ['Adam']} != {}`.
  - (b) the same → `SparseAdam`: `{'_homogeneous_lp.py': ['SparseAdam']} != {}`.
  - (c) a second, unnamed `SparseAdam` site in `_h2gformer_encoders.py`: `{'_h2gformer_encoders.py': ['SparseAdam']} != {}`.
  - Restored after each.

## F2 — H2G-former width and `dim_pe` (D-P6)

- **Change:**
  - `_h2gformer_encoders.metapath_dim_pe(hidden) = min(128, hidden // 2)`.
  - `metapath2vec_embeddings(…, dim_pe=…)` trains at that width.
  - `configure_encoders` sets `posenc_Hetero_Metapath.dim_pe` from `cfg.gt.dim_hidden` and no longer raises. H2GB's composed encoder then keeps `hidden − dim_pe > 0` (`composed_encoders.py:54-58`).
  - `h2gformer.py`: the grid is `hidden_dim` ∈ {64, 128, 256, 512}, and `attach_metapath_embeddings` takes the width.
  - With `metapath_pe` off nothing changes. H2GB datasets keep their config stacks.
- **H2GB configs (VERIFIED-RUN, read from `vendor/H2GB/configs/*/*H2Gformer*.yaml`).** The MetaPath2Vec pairs are PDNS (64, 32), Pokec and oag-chem (256, 128), ogbn-mag and mag-year (512, 128). All are reachable under the rule.
- **New tests (`tests/test_partPfix_h2gformer_width.py`):**
  - grid, and `dim_pe` per width (64→32, 128→64, 256→128, 512→128), raw width > 0;
  - the H2GB config pairs read from the files are reachable;
  - every (`hidden_dim`, `metapath_pe`) pair of the space (8) builds the model on the tiny ACM-shaped graph and runs one forward: finite output, positional width equal to the hand value, raw width positive.
- **Existing tests updated (named), asserting the old rule:**
  - `tests/test_partP_h2gformer.py::MetapathChannelTests.test_narrow_hidden_width_is_refused_with_metapath` ("`hidden_dim = 128` raises") is replaced by `test_hidden_width_128_builds_with_metapath` (it builds; `dim_pe = 64`).
  - `…::test_embeddings_are_produced_and_consumed` now passes the width (`hidden_dim = 160` → `dim_pe = 80`) and expects 80 instead of the fixed 128.
- **Tests (VERIFIED-RUN, `f2_tests_output.txt`):** `tests.test_partPfix_h2gformer_width tests.test_partP_h2gformer` → **Ran 13 in 319 s, OK**.
- **Mutation (VERIFIED-RUN, `f2_mutation_fixed_dim_pe.txt`):** `dim_pe` fixed at 128 → **FAILED**.
  - At width 64: `RuntimeError: … negative dimension -64`.
  - At width 128: failure, positional width 128 ≠ 64.
  - **Deviation in my own first test version:** it compared the width with `metapath_dim_pe` itself, and at 128 a zero-width raw channel still builds. So only width 64 failed until I changed the check to the hand values and to raw width > 0. The evidence file is from the corrected test. Restored.

## F3 — Read-out τ over length-2 only (spec v1.20 U2, D-O7)

- **Change:** `scripts/schema_readout.py`.
  - `length2_tau(records, permutations)` = `kendall_matrix(records, span=2, …)` is the one τ the script computes.
  - The table has one block, "Kendall tau … (length-2 sub-path schemas)". The all-spans block is removed.
  - A per-dataset `schema_readout.json` holds `kendall_tau_length2`. The per-run JSONs never held τ, which is computed across runs. The library keeps `span=None` as an option; the script does not use it.
- **New test (`tests/test_partPfix_readout_tau.py`):**
  - on two synthetic 1-epoch runs, exactly one τ heading (length 2) and exactly one τ key in the JSON;
  - its rows equal `kendall_matrix(run records, span=2)`;
  - a guard that the fixture separates the two definitions.
  - No value is reported.
- **Tests (VERIFIED-RUN, `f3_tests_output.txt`):** `tests.test_partPfix_readout_tau tests.test_partO_readout` → **Ran 9, OK**. Part O's script test still finds a "Kendall tau" heading.
- **Mutation (VERIFIED-RUN, `f3_mutation_all_spans.txt`):** the script computes τ over all spans → **FAILED**. Restored.

## F4 — Data root from the bundle

- **Change:**
  - `bundles.py`: `build_bundle` records `data_root` (`Path(config.data.root).resolve()`, the root its own loader read) in every bundle's metadata (NC, LP, GC). New `bundle_data_root(bundle)`. `hgb_raw_dir(dataset, root)` takes the root.
  - **Sites fixed:**

| site | was | now |
|---|---|---|
| `adapters/h2gformer.py` `_build_hetero_data` | `nc_root(REPO_DATA_ROOT)` | `nc_root(bundle_data_root(bundle))` |
| `adapters/hetero2net.py` `_load_vendor_graph` (2 uses) | `nc_root(REPO_DATA_ROOT)` | `_load_vendor_graph(dataset, data_root)`, called with `bundle_data_root(bundle)` |
| `bundles.py` `hgb_raw_dir` | `hgb_nc_raw_dir(REPO_DATA_ROOT, key)` | `hgb_nc_raw_dir(root, key)` |
| `adapters/hgb_han.py` `_load_vendor_graph` (HAN) | `hgb_raw_dir(dataset)` | `hgb_raw_dir(dataset, root)`, root from the bundle |
| `adapters/sehgnn.py` `_vendor_args` (SeHGNN) | `hgb_raw_dir(dataset)` | `_vendor_args(dataset, params, root)` |
| `adapters/_hgb_vendor.py` `load_vendor_dataset` (PSHGCN, H2SGNN) | `hgb_raw_dir(dataset)` | `load_vendor_dataset(vendor, dataset, args, root)`; `pshgcn.py`, `h2sgnn.py` pass `bundle_data_root(bundle)` |

  - Every other adapter (GCN, GAT, ChebNetII, NSD, CSNN, HiNormer, and every LP/GC path) reads the bundle's in-memory data only, with no file path (grep and AST: no `REPO_DATA_ROOT`). No vendor code changed, so the stop condition was not met.
- **New tests (`tests/test_partPfix_data_root.py`):**
  - an AST check that no module under `src/baselines` names or imports `REPO_DATA_ROOT` in code;
  - the bundle carries its resolved root, `hgb_raw_dir` resolves under it, and SeHGNN's vendor arguments point under the given root;
  - the empty default root has no raw directory;
  - with the bundle's root at the tiny Freebase-format fixture and `REPO_DATA_ROOT` patched to an empty directory, H2G-former, Hetero²Net and HAN each run one epoch (finite `test_macro_f1`, `val_macro_f1`).
  - SeHGNN, PSHGCN and H2SGNN cannot be run here: empty clones, and no Freebase support (D-P8). Their paths are covered by the AST check and by SeHGNN's argument test only.
- **Existing test updated (named):** `tests/test_partP_freebase.py::TinyReleaseTests.test_every_nc_adapter_runs_one_epoch` patched `hetero2net.REPO_DATA_ROOT`, which no longer exists, to reach its fixture. It now patches the default root to an empty directory. The fixture is reached through the bundle alone.
- **Tests (VERIFIED-RUN, `f4_tests_output.txt`):** `tests.test_partPfix_data_root tests.test_partP_freebase.TinyReleaseTests tests.test_baselines.AdapterContractTests tests.test_baselines.RegistryTests` → **Ran 11, OK (skipped=3)**. The skips are environment ones (ChebNetII, NSD, CSNN).
- **Mutation (VERIFIED-RUN, `f4_mutation_hetero2net.txt`):** Hetero²Net's loader reverted to `REPO_DATA_ROOT` → **FAILED**.
  - The AST test reports `{'adapters\\hetero2net.py': [59, 93]}`.
  - The fit errors: the module-level import bound the real repository root, so the loader started reading the real Freebase files and failed allocating 27 GB before any training. No run and no metric resulted.
  - Restored.

## F5 — Full suite and report

**Full suite, run once (VERIFIED-RUN).** `python -m unittest discover -s tests` from `H2S-NN/` on `aed53a6` → **Ran 639 tests in 1,191 s; FAILED (failures=2, errors=2, skipped=7)** (`f5_full_suite_output.txt`; INFO log lines of synthetic fixture trainings removed).
- **Failures.** Exactly the four known `test_baselines` failures (Part N-fix F5, Part O O4, Part P P4): `VendorIsolationTests.test_all_declared_vendor_clones_are_present`, `VendorIsolationTests.test_colliding_vendor_module_names_do_not_leak`, `ReproducibilityTests.test_vendor_model_init_is_reproducible_under_seed_everything`, and `BaselineNotebookTests.test_notebooks_are_committed_clean`.
- **The D-P9 test.** `test_shared_protocol.OptimizerTests.test_every_adapter_builds_adamw` **passes**. It was Part P's one new failure.
- **Skips.** The 7 are the environment skips of Part P (empty vendor clones, no MSVC for CSNN).
- **Test count.** 632 (Part P P4) + 3 (F2) + 1 (F3) + 3 (F4) = **639**.

## Decisions for the author

- **D-PF1, the τ JSON.** F3 adds one dataset-level `schema_readout.json` holding the length-2 τ (with its permutation p). The card asks for τ "in the table and the JSON", and the per-run JSONs never held τ, which is computed across runs. Confirm, or name another place.
- No other open decision. D-P6, D-P9 and U2 were implemented as decided.

## Out-of-scope findings (not fixed)

1. SeHGNN, PSHGCN and H2SGNN read the bundle's root now (F4). Their fits could not be exercised here, because their clones are empty (cluster-setup card).
2. Outside `src/baselines`, H2S-NN's own loaders read `config.data.root` (relative to the working directory). `REPO_DATA_ROOT` still appears in tests that check real data, and in `src/data/data_layout.py` as the default. Not a baseline path; unchanged.
3. The card writes `kendall_matrix(..., length=2)`; the library's argument is `span`. Unchanged.
