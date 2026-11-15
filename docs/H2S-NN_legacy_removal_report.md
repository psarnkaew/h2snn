# H2S-NN legacy-removal report (card `docs/H2S-NN_legacy_removal_card.md`)

Branch `legacy-removal`, base `v3b` `a6d681644f7923ee72ae94faf7fcece5ed1d598a` (Part M3a-fix head) + `design-docs` merges
`32efd0e` (LR-0), `a419754` (LR-0b, B1–B3), `216316a` (LR-2a, B4/B5). Evidence: `docs/legacy_evidence/` (every job script and its
output). Everything ran in `h2snn-b` with the pinned kernel environment E; bit-exact checks with 1 thread.

## Result

| Gate | g2n01 (i9-10900K, AVX2) | g3an01 (i9-10900X, AVX512) |
|---|---|---|
| `run_tests.sbatch` (whole suite) at `c683838` | `Ran 635 tests — OK` (16741 step 27, GPU 2080 Ti; `lr5_suite_g2n01.txt`) | `Ran 635 tests — OK (skipped=1)` (16749, CPU only; `lr5_suite_g3an01_16749.txt`) |
| `v3b_golden.py compare` at `c683838` | `compared 1365 tensors in 24 configurations; differing: none` (16741 step 28) | same, `differing: none` (16748) |
| `mb0_reference.py compare` at `c683838` | `compared 139 tensors; differing: none` (16741 step 28) | same, `differing: none` (16748) |

g3an01 ran `run_tests.sbatch` without a GPU (`sbatch --nodelist=g3an01 --gres=none --time=1:00:00`): the GPU job 16747 was held
until ≈ 2026-11-17 behind another user's array job reserving the a6000, so it was cancelled after 16749 passed. The node's AVX512 CPU
(the reason for the second node) is exercised under E; the one skip is
`test_factorized_maps.ModelIntegrationTests.test_cuda_float32_matches_cpu_float64_reference` ("CUDA not available"), which passed on g2n01.

Test count: 711 (M3a-fix) + 2 (LR-1/LR-2: `test_v3b_golden`, `test_legacy_removal` default test) − 82 deleted − 2 duplicate ids
(below) + 6 added (LR-3) = **635** (loader listings `test_ids_lr2.txt` 713, `test_ids_lr3.txt` 635).

## Units and commits

| Unit | Commit | Content |
|---|---|---|
| LR-0 | `e2d297a` merge, `ba967ea` | inventory; **stop** (B1 below) |
| LR-0b | `7a2b643` merge, `7495e58` | inventory after B1–B3; per-test classes `lr0b_classes.md` (80 delete, 136 adapt, rest keep) |
| LR-1 | `9e9087e` | v3-B golden on the unchanged code (`v3b_golden.py`, 24 configurations, 1,365 tensors), `tests/test_v3b_golden.py` |
| LR-2 | (stop, `lr2_stop.md`) → `afda3e3` merge, `888d12e` | defaults `propagation: anchored`, `coupling: symmetric`; `tests/test_legacy_removal.py`; B3 adaptations |
| LR-3 | `4e70e6a` | deletion; fail-closed registry; test deletions/adaptations; readout without `hidden_dim`/`dropout` |
| LR-4 | `c683838` | 12 legacy YAMLs; CLAUDE.md/AGENTS.md; baseline notebook outputs; grep record |
| LR-5 | this commit | suite on both nodes, compares, this report |

## Stops and decisions

**LR-0 stop (B1).** `identity_maps` (kept by A2) and `ablation.no_path_cells` exist only in the v2 propagation: `TrainConfig`
refused them under `propagation: anchored`, and the identity maps were applied only in `H2SNN.build_operator`, which built the v2
operator `VertexPathSheafLaplacian` and ran the Chebyshev filter bank / first-order steps. Keeping them needed most of the v2
propagation (card stop: a kept item that needs code of a delete item). Also raised: `coupling: symmetric` is the v3-B value (B2), and
the scale of the test adaptation (B3). Record: `docs/legacy_evidence/inventory.md`, "Findings of LR-0".
- **B1 = (c):** `identity_maps` and `ablation.no_path_cells` deleted with the v2 propagation; the v3-B arm `model.arm: no_path_cells` kept.
- **B2:** `ablation.coupling: symmetric` stays and is the default; `cooperative`, `listen_only`, `pair_attention` raise.
- **B3:** tests of kept code move to an anchored configuration with their assertions; an assertion on a v2-only quantity is replaced
  by its v3-B counterpart (listed per test below); any other assertion change is a stop.

**LR-2 stop (B4/B5).** With the LR-2 defaults the probe suite (`lr2_probe_suite.txt`, 16741, `9e9087e` + LR-2 working tree) gave
713 tests, 13 failures, 90 errors, all as LR-0b classified, except that `map_family: dense` (listed as kept by the card) runs only in
the v2 model: the anchored solve passes the factorized `reflectors`/`rotations` (`h2snn_model.py:1339` at the time) and
`DenseMapOperator` has none (`AttributeError`). T13, T8's dense control and the dense subcase of `every_ablation` would have had to be
deleted (card stop: a test of kept code that would have to be deleted). Record: `docs/legacy_evidence/lr2_stop.md`.
- **B4 = (a):** `map_family: dense` deleted with v2 (the value raises): the dense model branch (`DenseConformalNodeToEdgeSheaf`),
  `test_v2_model` T13, T8's dense control, the dense subcase of `every_ablation`. `DenseMapOperator` / `DenseOperatorTests` are kept
  because kept `src/` code uses `DenseMapOperator` (`sheaf_energy.py:41,154`; `PathTransportOperator` accepts it).
- **B5:** further card items that cannot be built are deleted with v2 and listed (not a stop), except the author's explicit kept
  items (G0b diagnostics, the `no_path_cells` arm, `coupling: symmetric`, the baseline data pins and their code). Items deleted under
  B5: the memory guard (`src/utils/memory_guard.py`, `model.node_edge_memory_guard_fraction`, 2 tests: it guarded only the dense family)
  and the subcases of kept-code tests that name a removed setting (listed under "Adapted tests").

## Inventory summary (`docs/legacy_evidence/inventory.md`)

- **Deleted.** Values: `propagation: chebyshev`; `encoder` (`filter_bank`, `first_order`, the key); `coupling: cooperative |
  listen_only | pair_attention`; `path_weight: sigmoid_gamma`; `map_family: dense` (B4); `path_sampler: uniform_instance`. Keys:
  `K`, `layer_dropout`, `eta_init`, `gamma_init`, `coupling_gate_init`, `node_edge_memory_guard_fraction`; ablation `no_path_cells`,
  `free_path_maps`, `endpoint_edges`, `holonomy_edges`, `per_side_scale`, `identity_maps`. Modules: `src/models/filter_bank.py`,
  `pair_attention.py`, `coboundary_ops.py`, `src/utils/memory_guard.py`, `scripts/analyze_kernel.py`; in kept files the v2 branches
  of `H2SNN`, `VertexPathSheafLaplacian`/`EdgePairTerm`/`FreeIncidenceOperator`, `IdentityMapOperator`,
  `DenseConformalNodeToEdgeSheaf`, `delta_loss`, the `uniform_instance` sampler, the trainers' filter-bank diagnostics. Configs: the 12
  `configs/experiments/*_h2snn.yaml`. Tests: below.
- **Kept, with reason.** v3-B code (every `V3B_ARMS` value, the C2 rungs, `geometry_context: false`, `path_weight_context`,
  `direction_blind_maps`, `map_family: factorized`, wisp and schemas, the three heads, full batch and mini-batch, the spec §6
  read-outs `PairCellTerm`/`schema_keep`/`cell_energy_pairs`); `PathTerm`, `PathCellTerm`, `HolonomyEdgeTerm`, `IncidenceOperator`
  (bodies unchanged); `DenseMapOperator` (B4); the `per_side` option of `FactorizedNodeToEdgeSheaf` (a kept class; its module tests
  stay); the author's kept items (`feature_normalization`, `h0_only_readout`, `compute_relation_diagnostics`, `model.arm:
  no_path_cells`, `coupling: symmetric`, the baseline pins `src/baselines/bundles.py:62-68` with the `dfs` sampler and the untied fact
  layer); `_pick`'s now-unused `counts` parameter in `src/data/path_sampler.py` (listed, signature not edited).
- **Unclear (kept, listed):** `num_label_prop_steps`, `metapath_templates` (LR-0b).
- **Fail-closed** (`src/configs/config.py:22-53`): `REMOVED_KEYS`/`REMOVED_VALUES`; a configuration that names one raises
  `ValueError` "`<section>.<key>[: <value>]` is removed by the legacy-removal card (v1/v2); a configuration that names it raises"
  (every `from_dict`, `ModelConfig`/`AblationConfig` construction, `reject_never_tuned_settings` for base configs and search-space
  names). `PROPAGATIONS = ("anchored",)`.

## Lines removed per file (`git diff --numstat -M a6d6816 HEAD -- H2S-NN`, before this report; added / removed)

| File | + | − |
|---|---|---|
| `src/models/h2snn_model.py` | 82 | 655 |
| `src/configs/config.py` | 103 | 235 |
| `src/models/vertex_path_laplacian.py` | 7 | 199 |
| `src/models/pair_attention.py` (deleted) | 0 | 198 |
| `scripts/analyze_kernel.py` (deleted) | 0 | 140 |
| `src/models/sheaf_maps.py` | 2 | 138 |
| `src/models/filter_bank.py` (deleted) | 0 | 129 |
| `src/models/coboundary_ops.py` (deleted) | 0 | 79 |
| `src/utils/memory_guard.py` (deleted) | 0 | 63 |
| `src/analysis/wisp_readouts.py` | 3 | 39 |
| `src/models/map_operators.py` | 0 | 34 |
| `src/data/path_sampler.py` | 6 | 15 |
| `src/trainers/trainer.py`, `link_prediction_trainer.py` | 2 + 2 | 13 + 13 |
| `src/models/link_readout.py` | 2 | 6 |
| `src/builders.py` | 2 | 2 |
| `configs/experiments/*_h2snn.yaml` (12 deleted) | 0 | 352 |
| `tests/test_cooperative.py` (deleted) | 0 | 837 |
| `tests/test_v2_model.py` | 92 | 641 |
| `tests/test_filter_bank.py` → `tests/test_relational_context.py` (rename, kept classes) | 174 | 453 |
| `tests/test_pair_attention.py` (deleted) | 0 | 386 |
| `tests/test_wisp.py` | 35 | 149 |
| `tests/test_gap_ladder.py` | 7 | 118 |
| `tests/test_paths_and_facts.py` | 13 | 86 |
| `tests/test_cuda_hardening.py` | 0 | 49 |
| `tests/test_task_heads.py` | 13 | 35 |
| `tests/test_v3b_model.py` | 16 | 17 |
| `tests/test_legacy_removal.py` (new) | 133 | 0 |
| `tests/test_v3b_golden.py` (new) | 58 | 0 |
| `tests/test_search_spaces.py`, `test_factorized_maps.py`, `test_partO_literature_row.py`, `test_partM3afix_concat_mlp.py`, `test_trainer.py`, `test_hgb_lp_negatives.py`, `test_shared_protocol.py` | 7, 2, 2, 2, 1, 1, 1 | 6, 3, 1, 2, 1, 1, 1 |
| `CLAUDE.md`, `AGENTS.md` | 15, 3 | 17, 3 |
| `notebooks/baselines_graph_classification.ipynb` (one JSON line) | 1 | 1 |
| **Total (51 files)** | **787** | **5117** |

## Deleted tests (82; reason)

Ids from the loader listings (`test_ids_lr2.txt` vs `test_ids_lr3.txt`, 16741 step 23). Two further ids disappeared without a test being
deleted: `test_paths_and_facts.SectionPreservationTests.test_T6_sections_are_preserved_with_signed_hops` was also collected through
its imports in the deleted `test_cooperative.py` and `test_pair_attention.py`; it is kept and runs once.

| Tests | Reason |
|---|---|
| `test_cooperative.py`: all 24 (`SymmetricIsThePreviousModelTests` 4, `GatedOperatorTests` 2, `SpectrumTests` 2, `CooperativeBehaviourTests` 3, `FixedPointTests` 1, `GateGradientTests` 1, `LearningFixTests` 3, `MetricGradientFlowTests` 4, `DiagnosticsTests` 3, `KernelScriptTests` 1) | gates, the v2 operator, the v2 golden, `analyze_kernel` (all deleted) |
| `test_pair_attention.py`: all 12 (`AttentionWeightTests` 3, `DenseReferenceTests` 2, `MergedWeightTests` 3, `NonExpansionTests` 2, `Prop4ControlTests` 1, `SectionTests` 1) | pair attention (B2) |
| `test_filter_bank.py`: `FilterBankOperatorTests` (6: TG1, TG2, TG3, TG5, TG6, gain diagnostic), `FilterBankModelTests` (6: TG4, TG8, eta-only-under-first-order, norm-sum diagnostic, operator diagnostics, `test_switches_are_never_tuned` — its kept `geometry_context` part moved, below) | the Chebyshev filter bank / first-order encoder |
| `test_gap_ladder.py`: `IdentityMapsOperatorTests` (2), `IdentityMapsModelTests` (4), `NeverTunedTests.test_identity_maps_rejected_in_tuning_base_config` | `identity_maps` (B1) |
| `test_v2_model.py`: `PathLaplacianTests` T2, T3, D, T5 | `VertexPathSheafLaplacian` |
| `test_v2_model.py`: `ScaleGaugeTests.test_T14_node_rescaled_per_side_scales_give_the_same_normalized_delta0` | `per_side_scale` switch + v2 normalized Δ₀ |
| `test_v2_model.py`: `SectionPreservationTests` T4 and its negative control | v2 operator kernel; free path maps |
| `test_v2_model.py`: `V2ModelTests.test_T6_path_term_reaches_the_far_endpoint_and_the_intermediate_node`, `test_T13_dense_and_factorized_families_match_on_materialized_maps` | v2 path term; dense family (B4) |
| `test_v2_model.py`: `AblationTests` T11 endpoint_edges, free_maps, holonomy_edges, per_side_scales | deleted v2 switches (B1/B5) |
| `test_v2_model.py`: `TwoLatentContextTests` a, c, d, `test_context_off_keeps_one_encoder_per_arm` | endpoint/free arms, pair attention (correction of LR-0b: context_off was "adapt"; both its subcases are deleted arms) |
| `test_paths_and_facts.py`: `StabilityTests.test_T13_lambda_max_at_most_two`; `UniformInstanceTests` (2) | v2 operator; `uniform_instance` |
| `test_wisp.py`: `WispWeightOperatorTests` (2), `WispLadderTests.test_TH6_exp_operator_has_no_gamma_multiplier`, `WispReadoutTests.test_TH11_delta_loss_of_an_absent_schema_is_zero` | v2 operator; `delta_loss` |
| `test_task_heads.py`: `NodeClassificationGoldenTests.test_outputs_and_gradients_are_bit_identical_to_the_pre_removal_capture` | v2 golden (replaced by `test_v3b_golden`, LR-1) |
| `test_cuda_hardening.py`: `test_memory_guard_accepts_safe_and_rejects_unsafe_workloads`, `test_model_memory_guard_fraction_is_validated` | memory guard of the dense family (B4/B5) |

**Added (6):** `test_legacy_removal.FailClosedTests` — `test_removed_values_raise`, `test_removed_values_raise_on_direct_construction`,
`test_removed_keys_raise`, `test_removed_keys_raise_in_a_tuning_space`, `test_kept_values_still_load`;
`test_relational_context.ContextNeverTunedTests.test_switches_are_never_tuned` (the kept `geometry_context` part of the deleted
`FilterBankModelTests.test_switches_are_never_tuned`). LR-1/LR-2 added `test_v3b_golden` and
`test_legacy_removal.V3BDefaultTests` (counted in 713).

## Adapted tests (what changed)

LR-2 (`888d12e`, B3: the default became anchored):
- `test_task_heads` LP/GC heads: logged `eta`/`filter_norm_sum` → `tau` and `theta_tau`; `K` removed from `SMALL_MODEL`.
- `test_v2_model`: T10 (`theta_eta`/`theta_gamma` → `theta_tau`); T11 no_path_cells (→ `arm: no_path_cells`, `last_solver_stats["num_cells"] == 0`);
  T11 uniform (w = 1 under sigmoid_gamma → `exp(path_log_weight)` constant, learned weights vary by > 1e-3); trainer logs (`tau`,
  `path_log_w_max`, scalar group of 2: `theta_tau`, `path_gate.bias`); gate-init `expected` = float32 `log(0.5)`.
- `test_filter_bank` (now `test_relational_context`) context tests: the geometry row; `reflector_mlp[0].in_features`.
- `test_v3b_model` T14 default test; `test_wisp` TH6, TH9 (arm subcases), round trip.
- `test_partO_literature_row::test_builder_refuses_other_inputs`, `test_search_spaces::test_the_anchored_model_can_be_tuned`: named
  `chebyshev` (both subcases deleted in LR-3, below).

LR-3 (`4e70e6a`; "subcase deleted" = a subcase that names a removed setting, B4/B5; the setting raises and `FailClosedTests` covers it):
- `test_v2_model`: `test_every_ablation_trains_one_step` (kept switches, the C2 rungs, `coupling: symmetric` and every `V3B_ARMS` arm on
  the anchored model; v2 switches, sigmoid_gamma, gate modes, dense subcases deleted); `test_defaults_are_the_full_model`
  (`coupling_gate_init == 0.6`, `(gamma_init, eta_init) == (1.0, 0.5)`, `map_scale == "per_edge"`, `path_mode == "cells"` → the fields and
  properties do not exist, and `tau_init == 1.0`); `test_invalid_values_and_conflicting_ablations_are_rejected` (eta/gamma ranges →
  `tau_init` 0 and −1; v2 switch conflicts → non-boolean `h0_only_readout`, `path_weight_context` with `uniform`;
  `path_weight_input: nope` kept); `TwoLatentContextTests.test_b` (endpoint/free subcases deleted); T8 (dense control deleted, B4);
  T9 (`state.maps.src.num_edges == M` from `encode().operator`); `K` dropped from `_config`.
- `test_wisp`: `_ladder_model` without `legacy`; `WispLadderTests._weights` (w_f from the anchored `_cell_weights`, was the v2
  `build_operator`; assertions unchanged); TH6 (`theta_gamma is None`, `gamma() is None` → no `theta_gamma` attribute, no parameter named
  gamma; sigmoid/gamma_init subcases deleted); round trip, direct construction, never-tuned, path-weight-context, ladder appendix pair
  (sigmoid / gamma_init / no_path_cells / endpoint / free / pair_attention subcases deleted; the appendix rejection checked on
  `schema`/`uniform`); paths-invariance test (first_order and v2-switch subcases deleted); `wisp_config_rejections` (uniform_instance
  subcase deleted); overlap test (`K` dropped).
- `test_paths_and_facts`: `_map_module`/`FAMILIES` factorized only (B4); `test_config_rules`, `test_T11_config` (uniform_instance subcases deleted).
- `test_factorized_maps`: dense config subcase deleted; `K` dropped. `test_trainer`, `test_hgb_lp_negatives`, `test_shared_protocol`: `K`
  dropped from the model section.
- `test_v3b_model`: T14 every arm (`filter_bank`/`theta_eta`/`listen_gate is None`, `output.eta/gamma is None` → attributes absent);
  T14 config (chebyshev / cooperative / holonomy_edges / sigmoid_gamma subcases deleted).
- `test_search_spaces::test_K_layer_dropout_and_legacy_keys_are_refused`: regex for `K`/`layer_dropout` in a search space
  "not a component of the method" → "removed by the legacy-removal card" (card LR-3 prescribes this message for a removed key in a
  tuning space; the refusal itself is unchanged). `test_the_anchored_model_can_be_tuned`: legacy-propagation subcase deleted.
- `test_partO_literature_row::test_builder_refuses_other_inputs`: "legacy propagation" case deleted (raises at load).
- `test_gap_ladder::test_diagnostics_rejected_in_search_spaces`: `ablation.identity_maps` name deleted (B1).
- `test_relational_context::RelationalContextTests`, `FactoredFirstLayerTests`, `ContextModelTests`: unchanged (file renamed).
- `test_task_heads`: v2 golden imports/constants removed. `test_partM3afix_concat_mlp`: readout built without `hidden_dim`/`dropout`.

Corrections of the LR-0b classification found on the way (all applied above): `test_context_off…` delete; T8 adapt; T13 delete;
`PerEdgeScaleTests` per_side tests keep (module option); `test_TH11_partial_pair_form…` keep (`PairCellTerm` kept); literature-row and
search-spaces tests adapt; memory guard deleted (B4/B5); `DenseMapOperator` kept.

## Golden coverage (`v3b_golden.py`, captured at `7495e58` before any deletion)

| Configuration | Captured |
|---|---|
| NC full batch, `model.arm` ∈ full, covering_only, hub_A, closure_only, no_path_cells, instance_features, free_cell_maps (7) | yes |
| NC `ablation.path_weight_input` endpoints, schema, uniform (3) | yes |
| NC `geometry_context: false`; `path_weight_context: true`; `direction_blind_maps`; `h0_only_readout`; `path_source: schemas` (r = 12) | yes (5) |
| NC `map_family: dense` | **no** (could not be built under anchored on the unchanged code: `AttributeError`; deleted by B4, now raises the removed-value message) |
| NC IMDB-style multilabel head | yes |
| LP decoder dot, distmult | yes (2) |
| GC pooling mean, max, add, attention | yes (4) |
| Mini-batch: NC batch of scopes; LP batch with scopes | yes (2) |
| **Total** | 24 configurations, 1,365 tensors (eval logits/scores, train loss, every gradient, solver statistics) |

Compares: LR-1 g2n01 (16741 step 14) and g3an01 (16744) none; after LR-2 (step 19), LR-3 (steps 21, 22), LR-4 (step 26), LR-5 (step 28,
16748): none. mb0 compare after LR-2, LR-3, LR-4, LR-5: none.

## Mutation table

| Unit | Mutation (restored identical in the same job) | Target test | Result |
|---|---|---|---|
| LR-1 | `sheaf_energy.py` covering_only degree term `(g.length + 1 if self.node_cell else 0)` → `else 1` | `test_v3b_golden` | fails: 44 differing tensors, all `nc_arm_covering_only` (`lr1_mutation.txt`, step 16) |
| LR-2 | `ModelConfig` default back to `chebyshev` | `test_legacy_removal.V3BDefaultTests` | fails: `('chebyshev', 'full') != ('anchored', 'full')` (`lr2_gate.txt`, step 19) |
| LR-3 | raise for one value removed (`sigmoid_gamma` dropped from `REMOVED_VALUES`) | `FailClosedTests` | 3 failures: `test_removed_values_raise`, `..._on_direct_construction`, `..._in_a_tuning_space` (`lr3_mutation.txt`, step 25) |
| LR-3 | raise for one key removed (`encoder` dropped from `REMOVED_KEYS`) | `FailClosedTests` | 4 failures: `test_removed_keys_raise`, `test_removed_keys_raise_in_a_tuning_space` ×3 (step 25) |

(Step 24 ran the same mutations; its second mutation's edit pattern also matched `NEVER_TUNED_LEGACY_KEYS`, so the edit was refused
by its own `assert` and nothing was mutated; the pattern was fixed and step 25 is the record.)

## Grep record (`lr4_grep.sh` → `lr4_grep.txt`, LR-4)

Pattern: every removed value, key, module and class name (`chebyshev|filter_bank|…|"K"|"encoder"|ablation\.no_path_cells|…`) over
`src/ tests/ scripts/ configs/` (`*.py *.yaml *.sh *.sbatch`). Hits: `src/configs/config.py` 22, `tests/test_legacy_removal.py` 13,
`tests/test_v2_model.py` 16, `tests/test_wisp.py` 10, `tests/test_wandb_and_runs.py` 6, `tests/test_search_spaces.py` 5,
`src/baselines/adapters/*` 12, 15 other files 1–4 each; `scripts/` 0, `configs/` 0. Classified:
- the fail-closed registry and messages (`config.py:22-53`) and their tests (`test_legacy_removal.py`);
- comments/docstrings recording a removal (`config.py`, `path_sampler.py:30`, `sheaf_maps.py:4`, `vertex_path_laplacian.py:6`, the
  `legacy-removal` notes in adapted tests);
- assertions that a removed name is absent or refused (`test_wandb_and_runs.py:148-149, 217-218, 240-241`, `test_v2_model.py:799`,
  `test_v3b_model.py:71, 154`, `test_wisp.py:332`, `test_search_spaces.py:92`);
- unrelated: `K` of the baselines PSHGCN, H2SGNN, ChebNetII; `"dense"` as a PyG submodule (`_pyg_compat.py:44`); `per_side` scales of
  the kept factorized-map option (`test_v2_model.py:175`);
- two docstrings in kept class bodies, left verbatim: `vertex_path_laplacian.py:84` (`HolonomyEdgeTerm`), `h2snn_model.py:256`
  (`EndpointEncoder`), "ablation `path_mode: …`". `NEVER_TUNED_LEGACY_KEYS` (`config.py:77-84`) still lists removed names; the
  removed-key check above it raises first.

## LR-4 details

- Legacy YAMLs: no kept test or script loads one (grep of `tests/ src/ scripts/`); each set `model.K` and so already raised.
- Notebook: `execution_count` → null, `outputs` → [] (cells 2, 4, 5, 6, 7); parsed comparison against HEAD OK (27 cells, same sources,
  metadata, top-level keys; compact JSON layout kept) — `lr4_notebook_check.py`/`.txt`. `BaselineNotebookTests`: OK (step 26).
- CLAUDE.md / AGENTS.md: D1 row without `| dense`; CLAUDE.md config sections (removed switches, `θ_η`/`θ_γ`, memory guard → kept
  switches, `θ_τ`, bias of g), architecture block (K first-order steps with η, γ → the anchored solve), trainer logs (`tau` and solver
  statistics); AGENTS.md key-files line; both gain "v1/v2 settings are removed (legacy-removal card); a configuration that names one raises."

## Out of scope, listed for the author

- CLAUDE.md/AGENTS.md still name the deleted v2 golden tests (pinned-kernel bullet, "Testing"), the deleted `*_h2snn.yaml` files
  (dataset list; the W&B-validate and Windows train commands) and D3's `η = σ(θ)` (not removed settings named as options; the card
  authorises only that edit).
- The eight `notebooks/h2snn_*_colab.ipynb` name `configs/experiments/<dataset>_h2snn.yaml` (deleted; they set `K` and would raise).
- `_pick`'s unused `counts` parameter (`src/data/path_sampler.py`); the two kept docstrings above.

## Jobs (HEAD, node, job id)

| Purpose | Job / step | Node | HEAD |
|---|---|---|---|
| LR-1 capture | 16741 / 13 | g2n01 | `7495e58` |
| LR-1 compare | 16741 / 14; 16744 | g2n01; g3an01 | `7495e58` |
| LR-1 mutation | 16741 / 16 | g2n01 | `7495e58` |
| LR-2 probe suite (stop) | 16741 | g2n01 | `9e9087e` + LR-2 working tree |
| LR-2 modules (189 OK) | 16741 | g2n01 | `afda3e3` + LR-2 working tree |
| LR-2 gate + mutation | 16741 / 19 | g2n01 | `afda3e3` + LR-2 working tree |
| LR-3 gate (targeted, 215 tests, 1 failure fixed: `test_search_spaces` regex) | 16741 / 21 | g2n01 | `888d12e` + LR-3 working tree |
| LR-3 compares + whole suite (635, 1 failure: notebook) | 16741 / 22 | g2n01 | `888d12e` + LR-3 working tree |
| LR-3 test-id listing | 16741 / 23 | g2n01 | `888d12e` (+ working tree) |
| LR-3 mutations | 16741 / 24, 25 | g2n01 | `888d12e` + LR-3 working tree |
| LR-4 gate (96 OK) | 16741 / 26 | g2n01 | `4e70e6a` + LR-4 working tree |
| LR-5 `run_tests.sbatch` | 16741 / 27 | g2n01 | `c683838` |
| LR-5 compares | 16741 / 28 | g2n01 | `c683838` |
| LR-5 `run_tests.sbatch` (CPU only) | 16749 | g3an01 | `c683838` |
| LR-5 `run_tests.sbatch` with GPU (held by the queue, cancelled) | 16747 | g3an01 | — |
| LR-5 compares | 16748 | g3an01 | `c683838` |

branch=legacy-removal sha=<see final message> card=docs/H2S-NN_legacy_removal_card.md base=a6d681644f7923ee72ae94faf7fcece5ed1d598a
