# LR-0 inventory: legacy (v1/v2) items, references, class

Branch `legacy-removal` at the merge `e2d297a` (base `v3b` `a6d6816` + `design-docs` `32efd0e`). VERIFIED-CODE by grep and reading on login01
(2026-11-15). Paths relative to `H2S-NN/` unless they start with `docs/`. Classes: **delete**, **keep (reason)**, **unclear** (kept, listed for the
author), and **blocked** (the class depends on a decision in "Blocking findings").

## Blocking findings (stop: the design session decides before LR-1)

### B1 — `identity_maps` (kept by A2) exists only in the v2 propagation (card stop condition: a kept item that needs code of a delete item)

- `TrainConfig.__post_init__` refuses `ablation.identity_maps` (and `ablation.no_path_cells`, `free_path_maps`, `endpoint_edges`, `holonomy_edges`,
  `per_side_scale`) under `model.propagation: anchored` (`src/configs/config.py:1551-1567`): "… belong to the legacy propagation; under
  model.propagation: anchored use model.arm". `AblationConfig` requires `identity_maps` to come with `ablation.no_path_cells: true`
  (`config.py:1444-1449`), itself refused under anchored.
- The identity maps are applied only in `H2SNN.build_operator` (`src/models/h2snn_model.py:999-1001`, `IdentityMapOperator.ones`,
  `src/models/map_operators.py:282`). `build_operator` builds the v2 operator `VertexPathSheafLaplacian` (`h2snn_model.py:1023-1030`) with the
  listen/broadcast gates and eta, and `encode` then runs `propagate` (`h2snn_model.py:1222-1229`): the Chebyshev filter bank
  (`src/models/filter_bank.py`) or the first-order update (`diffuse`). The anchored path (`_encode_anchored`, l.1264) never reads `identity_maps`.
- So a kept `identity_maps` needs, beyond a pure move: `build_operator`, `VertexPathSheafLaplacian`, `propagate`/`diffuse`, the filter bank or the
  first-order update (with `K`, `eta_init`, `layer_dropout`), i.e. most of the v2 propagation that A1 deletes.
- The card's LR-1 list names "`identity_maps` with `no_path_cells`" as an NC **anchored** configuration; on the unchanged code it cannot be built
  (left out of LR-1 by the card's own rule, but the removal question in LR-3 remains).
- Options for the author (design session):
  (a) keep the v2 propagation path only as far as `identity_maps` needs it (`chebyshev` + one encoder, `VertexPathSheafLaplacian`,
      `build_operator` without gates/attention); most of A1's removal then does not happen for the propagation;
  (b) give `identity_maps` an anchored form (F = I with scale 1 in the three-level energy, under `arm: no_path_cells`): a new v3-B configuration
      (no existing v3-B output changes, but it is an addition to the method's arms; spec wording and a card);
  (c) drop `identity_maps` (reverse A2), or postpone the removal of the v2 propagation until the sheaf-vs-no-sheaf control (B2 of the open
      items) is decided.

### B2 — `ablation.coupling: symmetric` is the v3-B value (the card lists every `coupling` value as legacy)

- Under anchored, `coupling` must be `symmetric` (`config.py:1554-1558`); `reject_never_tuned_settings` allows exactly that pair (`config.py:127-131`).
- It is set by every v3-B configuration: the five `configs/experiments/*_v3b.yaml` (`ablation: coupling: symmetric`), the gate script
  `docs/partMB_evidence/mb0_reference.py` (`_config`, `"ablation": {"coupling": "symmetric"}`), `tests/test_v3b_model.py:32` and the Part MB/O fixtures.
- Proposal (needs confirmation): `coupling` keeps `symmetric` only (or becomes optional, absent = symmetric) and `cooperative`, `listen_only`,
  `pair_attention` raise "removed by the legacy-removal card (v1/v2)". Raising on the key itself would break the gate script and every v3-B YAML.

### B3 — scale of the test adaptation (not a stop; for planning)

- 25 test files build a `TrainConfig`/model without `propagation: anchored`, i.e. they run the v2 model through today's default (grep: files with
  `TrainConfig.from_dict|build_model(|H2SNN(` and no `anchored`): `test_cooperative, test_data, test_failed_trials, test_gap_ladder,
  test_factorized_maps, test_full_graph_early_stopping, test_graph_classification_trainer, test_hgb_oracle, test_gc_protocol, test_hgb_lp_negatives,
  test_imdb_protocol, test_lp_valid_files, test_lp_derived, test_partM3afix_concat_mlp, test_pair_attention, test_paths_and_facts,
  test_shared_protocol, test_trainer, test_task_heads, test_stop_conditions, test_sheaf_energy, test_v2_model, test_wandb_and_runs, test_wisp,
  test_wisp_tasks`. Several test kept code (trainers, protocols, data, stop conditions) and reach it only through the v2 default; after LR-2/LR-3
  they need an anchored configuration ("adapt"), which changes the model they train (assertions on finiteness/shapes/protocol stay; any numeric
  assertion tied to the v2 model would not). The card's grep found 18 files at `8c4f1c5`; the same grep at `e2d297a` finds 11 (below). The
  per-test-function classification (delete / adapt / keep) is **deferred until B1 is decided**: under B1 (a) most v2-model tests are keep, under
  (b)/(c) delete.

## Values

| Item | Where (file:line) | What | References | Class |
|---|---|---|---|---|
| `propagation: chebyshev` | `config.py:94` (`PROPAGATIONS`), `:1200` default, `:1349` from_dict default, `:127` guard default | v1.2–1.10 propagation (spec v1.11 R13 replaced it) | `h2snn_model.py:19-26, 502-505, 1217-1229`; `tests/test_v3b_model.py:150-155` (asserts the legacy default) | **blocked (B1)**; else delete (value raises; `PROPAGATIONS` keeps `anchored`) |
| `encoder: filter_bank`, `first_order` | `config.py:32` (`ENCODERS`), `:1191` default, `:1346` | D12 filter bank / D11 first-order update | `h2snn_model.py:55, 569-573, 702-704, 748-752, 1039-1061, 1134, 1162, 1177-1192`; `src/models/filter_bank.py`; trainers' diagnostics (below); `docs/heads_evidence/v2_golden.py` (first_order) | **blocked (B1)**; under anchored never read |
| `coupling: cooperative`, `listen_only`, `pair_attention` | `config.py:25` (`COUPLING_MODES`), `:1394` default | D10 gates, D11 pair attention | `h2snn_model.py:654-660, 680-688, 945-989`; `src/models/pair_attention.py`; `scripts/analyze_kernel.py` | delete (B2: `symmetric` kept) |
| `coupling: symmetric` | as above | the v3-B value (required under anchored) | five `*_v3b.yaml`, `mb0_reference.py`, fixtures | **keep (B2)** |
| `path_weight: sigmoid_gamma` | `config.py:86`, `:1194` | D11/D12 weight; anchored needs `exp` (`config.py:1248-1250`) | `h2snn_model.py:613-617, 638-643`; tests (7 files) | delete (`exp` only) |
| `path_mode: endpoint_edges` (`ablation.endpoint_edges`) | `config.py:1371`, `path_mode` property | v2 arm; refused under anchored | `h2snn_model.py:296-328 (EndpointEncoder), 369-407 (EndpointEdgeMaps), 620-628`; `vertex_path_laplacian.py:138 (EdgePairTerm)` | delete; **keep `EndpointEncoder`** (the `endpoints` rung of `path_weight_input` uses it, `h2snn_model.py:633-637`) |
| `path_sampler: uniform_instance` | `config.py:641`, `:811-814`; `src/data/path_sampler.py:30, 46, 157-172, 307-317` | v2 sampler ablation (L ≤ 2) | tests (2 files) | delete (`walk` and `dfs` kept) |
| `path_sampler: dfs` | `path_sampler.py:46`, `dataset.py:190`, `link_prediction_dataset.py:833`, `graph_classification_trainer.py:238` | pre-D7 DFS | `src/baselines/bundles.py:63` | **keep (baseline data pin)** |

## Keys

| Key | Where | Reads (v3-B?) | Class |
|---|---|---|---|
| `model.K` | `config.py:1162`, `:1330` | v2 steps / filter-bank order (`h2snn_model.py:495, 704`); anchored no | **blocked (B1)**; else delete |
| `model.layer_dropout` | `config.py:1163`, `:1331` | `propagate`/`diffuse` (v2) | **blocked (B1)**; else delete |
| `model.eta_init` | `config.py:1176`, `:1341` | `theta_eta`, filter-bank init (v2) | **blocked (B1)**; else delete |
| `model.gamma_init` | `config.py:1175`, `:1340` | `theta_gamma` under `sigmoid_gamma` | delete |
| `model.coupling_gate_init` | `config.py:1179`, `:1342` | gate heads (v2) | delete |
| `model.encoder` | `config.py:1190` | v2 only | **blocked (B1)** |
| `ablation.per_side_scale` | `config.py:1385` | D1 per-incidence scales; refused under anchored | delete |
| `ablation.free_path_maps` | `config.py:1369` | v2 arm (`FreeIncidenceMaps` in path terms); refused under anchored | delete the switch; **keep `FreeIncidenceMaps`** (arm `free_cell_maps`, `h2snn_model.py:735-745`) |
| `ablation.holonomy_edges` | `config.py:1373` | v2 arm; refused under anchored | delete the switch; **keep `HolonomyEdgeTerm`** (arm `closure_only`, `sheaf_energy.py:23-25`) |
| `ablation.endpoint_edges` | `config.py:1371` | v2 arm | delete (see values) |
| `ablation.no_path_cells` | `config.py:1367` | v2 switch; under anchored the arm `no_path_cells` replaces it | **unclear / blocked (B1)**: the card keeps "`no_path_cells`"; the v3-B arm stays either way; the ablation switch is needed only by `identity_maps` |
| `ablation.identity_maps` | `config.py:1400` | v2 only (B1) | **keep by A2 → blocked (B1)** |
| `ablation.h0_only_readout` | `config.py:1387` | works under anchored (`encode` returns h0 before any propagation, `h2snn_model.py:1214-1215`; no v2 module is built under h0_only) | **keep (G0b)** |
| `data.feature_normalization`, `data.compute_relation_diagnostics` | `config.py` DataConfig | data side, independent of the propagation | **keep (G0b)** |
| `uniform_path_gate` | only in `NEVER_TUNED_LEGACY_KEYS` (`config.py:47`) and a comment (`:1377`) | already removed (Part H renamed it `path_weight_input: uniform`); not a field | delete the name from the list only if a fail-closed list replaces it (it is already an unknown key) |
| `data.num_label_prop_steps` | `config.py:617`, `dataset.py:291`, `preprocessing.py:318-333` | IMDB label-propagation preprocessing; check needed whether a v3-B IMDB config reads it | **unclear** (kept) |
| `data.face_sample_k` | `config.py:620`, `dataset.py:215`, `link_prediction_dataset.py:840`, `graph_classification_trainer.py:245`, `config.py:1034` | DFS quota (baselines) and the legacy schema row; set (unused under WISP) in `configs/experiments/acm_h2snn_v3b.yaml` | **keep (baseline pin / schemas row)** |
| `data.exclude_identity_metapaths` | `config.py:621`, `dataset.py:197`, `link_prediction_dataset.py:841`, `graph_classification_trainer.py:246` | DFS (baselines) | **keep (baseline pin)** |
| `data.metapath_templates` | `config.py:616`, `dataset.py:172-200`, `config.py:1024-1054` | schema presets / the schemas source | **unclear** (kept): the literature row refuses it (`config.py:887-890`), the r = 0 legacy schema row uses presets |
| `data.overlap_sampling` | `config.py:660`, `trainer.py:111-115` | WISP redraw worker thread in the NC trainer (no result change) | **keep (v3-B training)** |
| `data.resample_paths` | `config.py:644`, trainers | WISP per-epoch redraw; `path_source: wisp` needs it true (`config.py:866-868`); baselines pin false | **keep** |
| `data.fact_layer` (untied) | `config.py`, `src/data/fact_layer.py` | baselines pin `untied` | **keep (baseline pin)** |

## Modules and functions

| Item | References | Class |
|---|---|---|
| `src/models/filter_bank.py` (`ChebyshevFilterBank`, `chebyshev_shift`, `first_order_chebyshev_coefficients`) | `h2snn_model.py:55, 702-704`; trainers' diagnostics; `tests/test_filter_bank.py`, `test_v2_model`, `test_cooperative`, `test_task_heads`, `test_v3b_model:70,155`; `docs/heads_evidence/v2_golden.py`; `docs/D12_evidence/*.py` (history, not imported by tests) | **blocked (B1)** |
| `src/models/pair_attention.py` (`PairAttention`, `AttentionOperator`, `build_attention_operator`, `segment_softmax`) | `h2snn_model.py:680-688, 945-979`; `tests/test_pair_attention.py`, `test_gap_ladder`, `test_wisp`, `test_v2_model`; `scripts/analyze_kernel.py` | delete |
| `vertex_path_laplacian.VertexPathSheafLaplacian` (l.209) | `h2snn_model.py:1023`; `src/analysis/wisp_readouts.py` (`delta_loss`); tests (5 files) | **blocked (B1)** |
| `vertex_path_laplacian.EdgePairTerm` (l.138) | `h2snn_model.py` (endpoint_edges path terms) | delete |
| `vertex_path_laplacian.FreeIncidenceOperator` (l.49) | `h2snn_model.py` (free_path_maps), `tests/test_v2_model.py` | delete (check: not used by the arm `free_cell_maps`, which uses `sheaf_energy.py:187`) |
| `vertex_path_laplacian.PathCellTerm` (l.86), `HolonomyEdgeTerm` (l.166), `PathTerm` (l.194) | `sheaf_energy.py`, `map_operators.py`, `wisp_readouts.py` | **keep (v3-B reaches them)**; a pure move out of the v2 module is allowed |
| `src/models/coboundary_ops.py` | `vertex_path_laplacian.py` only (+ `docs/D1_evidence/*.py`, history) | delete with `VertexPathSheafLaplacian` if that goes (blocked, B1) |
| `src/analysis/wisp_readouts.delta_loss` (l.230) | `tests/test_wisp.py`; uses `VertexPathSheafLaplacian`, `PathCellTerm` | delete (spec §6: not reported) — independent of B1 |
| `H2SNN.build_operator`, `diffuse`, `propagate`, `coupling_gates`, `coupling_stats`, `lambda_max`, `energy_trace`, `eta_lambda_max`, `operator_diagnostics`, `eta`, `gamma`, `_build_attention_operator`, `diagnostics_available` | `h2snn_model.py:748-1197` | **blocked (B1)** (pair-attention and gate parts: delete) |
| filter-bank / operator diagnostics in trainers | `src/trainers/trainer.py:221-226`, `src/trainers/link_prediction_trainer.py:275-280` (`diagnostics_available` is false under anchored) | **blocked (B1)** |
| `scripts/analyze_kernel.py` (kernel of the v2 operator `Delta_hat`) | `tests/test_cooperative.py` | delete (v2 operator; spec v1.11 replaced it) unless B1 (a) |
| `docs/heads_evidence/v2_golden.py` (+ `v2_golden.pt`, `v2_golden_cluster.pt`) | `tests/test_task_heads.py:27`, `tests/test_cooperative.py:34` (the two v2 golden tests) | tests: delete (card), replaced by the LR-1 v3-B golden; files stay in `docs/` as history |
| `docs/partMB_evidence/mb0_reference.py` | gate; `tests/test_mb0_reference_compare.py` | **keep** (v3-B anchored; sets `coupling: symmetric`, see B2) |

## Configs (`configs/experiments/`)

| File | Class |
|---|---|
| `acm_h2snn_v3b.yaml`, `acm_pa_h2snn_v3b.yaml`, `dblp_h2snn_v3b.yaml`, `imdb_h2snn_v3b.yaml`, `mutag_h2snn_v3b.yaml` | keep (v3-B) |
| `acm_h2snn.yaml`, `dblp_h2snn.yaml`, `imdb_h2snn.yaml`, `lastfm_h2snn.yaml`, `pubmed_h2snn.yaml`, `enzymes_h2snn.yaml`, `mutag_h2snn.yaml`, `nci1_h2snn.yaml`, `proteins_h2snn.yaml`, `amazon_h2snn.yaml`, `amazonbook_h2snn.yaml`, `youtube_h2snn.yaml` (12, no `propagation: anchored`) | delete in LR-4 (check first that no kept test or script loads them; the notebooks reference `configs/experiments/<dataset>_h2snn.yaml`: a decision for LR-4) |

## Test files naming legacy settings (card grep, at `e2d297a`: 11 files)

| File | Test functions | Lines with a legacy name |
|---|---|---|
| `tests/test_cooperative.py` | 24 | 20 |
| `tests/test_filter_bank.py` | 18 | 23 |
| `tests/test_gap_ladder.py` | 10 | 1 |
| `tests/test_pair_attention.py` | 12 | 20 |
| `tests/test_paths_and_facts.py` | 50 | 8 |
| `tests/test_search_spaces.py` | 11 | 3 |
| `tests/test_task_heads.py` | 5 | 7 |
| `tests/test_v2_model.py` | 45 | 45 |
| `tests/test_v3b_model.py` | 2 | 5 (asserts the legacy default and `filter_bank is None` under anchored) |
| `tests/test_wandb_and_runs.py` | 13 | 5 |
| `tests/test_wisp.py` | 48 | 16 |

Per-test-function classification: deferred until B1 is decided (see B3).
