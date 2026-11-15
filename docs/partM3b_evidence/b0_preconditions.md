# B0 — preconditions (Part M3b; no code)

Branch `part-m3b` from `v3b` `d6f3b883815c08d7408ff83689327f507fd500da`, merge `--no-ff` of `design-docs`
`a463268e51537139f0b5cbe7092b9b3608650d9f` (no conflict; merge commit `23faa32`). `git fetch` first: `origin/v3b` and
`origin/design-docs` equal the pinned SHAs. VERIFIED-CODE by reading on login01, 2026-11-15. Paths of H2GB are relative to
`H2S-NN/vendor/H2GB` (commit `d3b49db63ad66f8a3282dbe699d50ce10ed3062a`; state in `b0_vendor_state.txt`).

## Read
Card, `docs/working_preferences.md`, `docs/minibatch_design_v3B.md` (whole), `docs/H2S-NN_partM_card.md` M3, `docs/predefined_metapaths.md`,
decisions-log rows 2026-11-12 (Datasets (NC)), 2026-11-13 (Part MB review D-MB1–4, tests/M3 items; Meta-paths on H2GB M-path 1–4;
Plan), 2026-11-15 (CS7 review, Legacy-removal review). H2GB: `H2GB/datasets/{mag,oag,ieee_cis,pokec,pdns}_dataset.py`,
`H2GB/loader/master_loader.py:146-405, 475-776`, `H2GB/loader/split_generator.py:12-92`, `H2GB/sampler/custom_sampler.py` (where the
data object is read), the eight `configs/*/<name>-MLP*.yaml` dataset blocks, `examples/use_dataset.py` (author's copy). H2S-NN:
`src/data/dataset.py`, `src/data/data_layout.py`, `src/data/hetero_flatten.py:18-84`, `src/configs/config.py` (mini-batch,
budgets, dataset resolution), `src/baselines/registry.py`. The remaining files of the card's B0 list (`link_prediction_dataset.py`,
`hgb_lp_negatives.py`, `scopes.py`, `minibatch_trainer.py`, `budget.py`, the H2G-former and Hetero²Net adapters,
`calcHomophily.py`) are read in the unit that changes or measures them (B5–B8) and cited there.

## How H2GB builds and loads each dataset

H2GB's entry point is `create_dataset` → `load_dataset_master(format, name, dataset_dir)` (`master_loader.py:146`), with
`dataset_dir = <cfg.dataset.dir>`; `PyG-ogbn-mag`/`PyG-mag-year` append `mag` (`:165-166`), every other `PyG-<id>` appends `<id>`
(`:168`), `IEEE-CIS` appends `IEEE-CIS` (`:228`). Splits: `split_mode` is `standard` (`H2GB/config/split_config.py:13`), and for
`task: hetero_node` `setup_standard_split` does nothing (`split_generator.py:52-90`): the split is the masks on the data object.
Training and evaluation read **`dataset[0]`**, i.e. the stored graph with the dataset's `transform` applied
(`custom_sampler.py:92, 330`: `HGTLoader(dataset[0], ...)`).

| Dataset | Config (`format`, `name`, entity) | Class, file:line | Root → processed file | Transform at load (`master_loader.py`) | Target / label | Split source |
|---|---|---|---|---|---|---|
| ogbn-mag | `PyG-ogbn-mag`, `ogbn-mag`, paper | `MAGDataset` (`mag_dataset.py:22`) via `preformat_MAG` (`:620-632`) | `NC/mag` → `mag/ogbn-mag/processed/data.pt` (`mag_dataset.py:77-88`) | none | paper, venue, 349 classes (`:128`), single-label | OGB official split (`:130-142`) |
| mag-year | `PyG-mag-year`, `mag-year`, paper | `MAGDataset(name='mag-year')` | `mag/mag-year/processed/data.pt` | none | paper, year in 5 quantile classes (`:163-165`) | random 50/25/25 at processing time, no seed (`:166-172`); baked into the processed file |
| oag-cs | `PyG-OAG`, `cs`, paper | `OAGDataset` (`oag_dataset.py:17`) via `preformat_OAG` (`:491-513`) | `NC/OAG` → `OAG/cs/processed/data.pt` (`oag_dataset.py:75-85`) | `ToUndirected(merge=True)` (`:500`) | paper, venue of `PV_Journal` (`oag_dataset.py:138-150`) | year ≤ 2016 / 2017 / ≥ 2018 (`:151-153`) |
| oag-eng | `PyG-OAG`, `engineering` | as oag-cs | `OAG/engineering/processed/data.pt` | as oag-cs | as oag-cs | as oag-cs |
| oag-chem | `PyG-OAG`, `chemistry` | as oag-cs | `OAG/chemistry/processed/data.pt` | as oag-cs | as oag-cs | as oag-cs |
| IEEE-CIS-G | `IEEE-CIS`, `IEEE-CIS`, transaction | `IeeeCisDataset` (`ieee_cis_dataset.py:103`) via `preformat_IEEE_CIS` (`:754-776`) | `NC/IEEE-CIS` → `IEEE-CIS/processed/data.pt` | `ToUndirected, AddSelfLoops, NormalizeFeatures` (`:764`) | transaction, `isFraud`, binary (`ieee_cis_dataset.py:227`) | set in `preformat_IEEE_CIS` on node order: first 80 % train, next 10 % val, rest test (`:766-775`) |
| H-Pokec | `PyG-Pokec`, `Pokec`, user | `PokecDataset` (`pokec_dataset.py:34`) via `preformat_Pokec` (`:575-602`) | `NC/Pokec` → `Pokec/processed/data.pt` | `ToUndirected()` (`:585`) | user, gender (`pokec_dataset.py:280`; −1 = unknown), binary | random 50/25/25 at processing time, no seed (`:291-301`); baked in |
| PDNS | `PyG-PDNS`, `PDNS`, domain_node | `PDNSDataset` (`pdns_dataset.py:14`) via `preformat_PDNS` (`:604-617`, `start=0, end=60`, `domains2.csv`) | `NC/PDNS`; **no processed file**: the class reads `raw/` on every load (`_load_train_test`) | `ToUndirected()` (`:613`) | domain_node, malicious/benign | temporal; B5 |

Download behaviour (PyG 2.8.0, `torch_geometric/data/dataset.py:112-116, 226-231, 238-266`): `Dataset.__init__` calls `_download`
when the class overrides `download` and **any raw file is missing** (it also `makedirs(raw_dir)`), and `_process` when it overrides
`process` and the processed file is missing. `OAGDataset` overrides `download` and its raw folder `NC/OAG/raw` is empty, so
instantiating it as is would call `download_dataset_from_hub` (`oag_dataset.py:87-88`) although the processed file exists.
`MAGDataset` has no `download`; IEEE-CIS and Pokec have their raw files. Consequence for B2: the loader instantiates each H2GB class
through a subclass whose `_download`/`_process` only check that the processed file exists (and raise otherwise); the class's own
`__init__` then loads the processed file exactly as H2GB does (stated deviation: no download/processing hooks).

Observations to carry: (i) PDNS `_load_train_test` builds the training graph with `_load_data(..., start, end, num_val)` and one test
graph per entry of `test_list` (`pdns_dataset.py:173-187`); the block after it that names an undefined `data` is inside a string
literal (`:188-...`) and never runs (B5 reads this in full); (ii) the PDNS raw folder holds
all 69 files of `raw_file_names` (`pdns_dataset.py:66-91`) plus `gt/`, more than the card's list; (iii) mag-year and H-Pokec splits
come from an unseeded `train_test_split` at processing time, so they are whatever the processed files hold.

## H2S-NN's loader contract
`build_dataset` (`src/data/dataset.py:225`) calls `_load_hetero_data(config)` (`:27-53`, a `HeteroData`), then
`stack_node_features` (`src/data/hetero_flatten.py:18-54`: every node type stacked, features padded to the largest dimension, a type
without `x` gets `ensure_feature_matrix`), `flatten_edges` (`:57-84`: every `data.edge_types` entry becomes one relation), the fact
layer, and `_prepare_target_labels` (`dataset.py:63-133`): `data[target].y` (1-D long or 2-D multilabel), `train_mask`
required, `val_mask` (a deterministic split of train if absent), `test_mask`. The target type comes from
`config.data.target_node_type`. Registry: `H2GB_NC`, `LARGE_LP`, `DATASET_PENDING` (`src/baselines/registry.py:26-43`); config:
`H2GB_NC_DATASETS`, `LARGE_LP_DATASETS`, `default_training_mode` (`src/configs/config.py:85-118, 584-586`).
