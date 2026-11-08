# Part I inventory (I0, 2026-11-09): what I1–I6 touch, with file:line at `94fc397`
Paths are relative to `H2S-NN/` unless stated.

## I1: `g` reads `h₀` only; `ablation.path_weight_context`
- **Path encoders (the first layer of `g`).**
  - `src/models/h2snn_model.py:216-261` `PathEncoder`, `:264-295` `EndpointEncoder`, `:298-315` `SchemaEncoder`.
  - They are built at `:559-579`, with input width `geometry_dim` (`:497-502`; d + 2R·d under `geometry_context`).
  - The `endpoints` width rule is at `:574` (`endpoint_hidden_width`, `:318-327`, `node_dim = geometry_dim`).
- **The latent's consumers.** `_path_latent` (`:722-733`) is called with:
  - the geometry, in `path_terms` (`:765-803`; `build_operator` passes `geometry`, `:876`);
  - the geometry, in `_build_attention_operator` (`:826`).
- **Uses of the latent:**
  - `_path_weights` (`:754-763`): `g`, the ladder.
  - `endpoint_maps(latent)` (`:788`): the `endpoint_edges` arm's maps.
  - `free_maps(h0=geometry, nodes, latent)` (`:790`): the `free_path_maps` arm; its node input stays the geometry.
  - The pair-attention latents (`:826`): the C4b arm.
- **The node inputs that keep `[h₀, c]`:**
  - `node_to_edge` (maps, `:503-531`, `input_dim=geometry_dim`);
  - `listen_gate` / `broadcast_gate` (`:612-615`, `geometry_dim`);
  - `coupling_gates(geometry)` (`:892`);
  - `free_maps`' node input.
- **The context:** `src/models/relational_context.py:28-44`, the per-(relation, side) mean of neighbours' `h₀`.
- **Config:**
  - `src/configs/config.py:814-930` `AblationConfig` (fields, `__post_init__` `:849`, `from_dict` `:918`);
  - `reject_never_tuned_settings` `:49-121`; `NEVER_TUNED_WISP_KEYS` `:36-38`.
- **Tests:**
  - `tests/test_wisp.py:426-441` (T-H7 intermediates), `:443-466` (endpoint perturbation and the schema-equality check), `:468-483` (context off/on), `:485` (capacity rule), `:338-353` `_ladder_model`;
  - T-H5 `:273-336`; T-H6 in `WispLadderTests`.
- **Goldens:** `docs/heads_evidence/v2_golden.py:82-83` pins `geometry_context: false`, so `g` already reads `h₀`. Unaffected (to be verified bit for bit).

## I2: the default `path_source: wisp` for node classification
- `src/configs/config.py`:
  - `DataConfig.path_source` default `:325`; `from_dict` `:379`; `resolved()` `:407-535` (task inference `:442`, WISP checks `:447-476`);
  - `_resolved_node_classification` `:537-574`; `NODE_CLASSIFICATION_PRESETS` `:169-200` (ACM, DBLP, IMDB are the only node-classification datasets, `DATASET_TASK_TYPES`);
  - `wisp_walks_per_anchor` default 42 `:327`.
- `src/models/h2snn_model.py:1130-1137` `build_model`: under `wisp` the model gets `face_lengths = 2..ℓ_max` and `path_actions`. Schema-cell fixtures built through a default node-classification config would mismatch.
- `src/data/dataset.py:143-155`: the `path_source` dispatch.
- **Call sites in tests that build node-classification models or data from a config:**
  - `test_cooperative.py` (7), `test_factorized_maps.py` (2), `test_pair_attention.py` (1), `test_paths_and_facts.py` (1), `test_trainer.py` (1), `test_v2_model.py` (3);
  - `test_v2_model.py:1273` asserts the old default;
  - `docs/heads_evidence/v2_golden.py:59-90` `build_config` (to pin `path_source: schemas`);
  - `scripts/profile_model.py:40-41`.
- **Link prediction and graph classification:** `wisp` is rejected for them (`config.py:465-470`), so the default must resolve by task.

## I3: S1′ oracle with scale; learning mechanism (evidence script only)
- `docs/WISP_evidence/h5_synthetic.py` (`graph_s1p`, `config`, `relevance_of_cells`, `main`).
- **The optimizer groups:** `src/builders.py:44-67` (`build_parameter_groups` from `model.scalar_parameters()`, `h2snn_model.py:688-706`).
- **The readout input** `[x_K, h₀]`: `h2snn_model.py:1076`, `:1092`. No readout-without-`h₀` arm exists in `src/`.
- **The epoch hook:** `src/trainers/trainer.py:82-136` (`fit(..., epoch_callback)`).

## I4: S2 without the length confound (evidence script only)
- `docs/WISP_evidence/h5_synthetic.py` `graph_s2` and the S2 read-out block (`main`).
- The read-outs are in `src/analysis/wisp_readouts.py` (`mean_log_weight`, `i_pi`, `full_schema_keys`). The rank must be strict; ties count against σ\*.

## I5: test gaps from the H7 review
- **MINOR-5:**
  - `src/trainers/trainer.py:47-80` `training_draw` (`_previous_draw`, `proposal_from_mass`);
  - `:96-99` (the training draw vs `prepared_data` for evaluation);
  - `src/data/dataset.py:226-236` (the evaluation draw, no proposal).
- **MINOR-7:**
  - `config.py:712-771` `ModelConfig.__post_init__`; `:773-782` the `from_dict`-only rejections (`gamma_init` under `exp`; `path_weight_init` under `sigmoid_gamma`);
  - `DataConfig` has no `__post_init__` (its checks live in `resolved()`).
- **MINOR-8:** `config.py:85-94` (the base-config rejections), `:107-112` (search space; the message omits `wisp_proposal`).
- **MINOR-10:** `tests/test_wisp.py:86-107` (`dense_path_laplacian`, `exact_path_laplacian`), `:148-158` (T-H2, q = p), `:590-600` (biased `q`).
- **The transport operator:** `src/models/map_operators.py` `PathTransportOperator`; the fixture maps are `tests/test_v2_model.py:74-94` `_chain_operators`.

## I6: overlap sampling
- `src/trainers/trainer.py:47-80` (`training_draw`; the seed is `cfg.seed + epoch`; the adaptive proposal reads `model.last_path_weights` from the previous training forward) and `:82-152` (`fit`).
- **Sampler RNG:** each draw seeds its own `torch.Generator` (`src/data/wisp_sampler.py:335`; `src/data/path_sampler.py:310`, `:355`). No global RNG is used.
- Link prediction and graph classification have their own trainers (`link_prediction_trainer.py:55`, `graph_classification_trainer.py:221`). They are unchanged (the card scopes I6 to the sampler of node classification; LP and GC are unchanged in I2).
