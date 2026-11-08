# Part K inventory (K0, 2026-11-09): what K1–K3 touch, with file:line at `bf7f333`
Paths are relative to `H2S-NN/` unless stated.

## K1: two latents (E10)
- **`src/models/h2snn_model.py`:**
  - `:555-556` `path_input_dim` (`g`'s per-node input width; `geometry_dim` only under `path_weight_context`).
  - `:565-574` `endpoint_edges`: one `EndpointEncoder(path_input_dim)` feeds both `g` and `EndpointEdgeMaps` (`:570`).
  - `:575-578` `full` / `free_path_maps`: one `PathEncoder(path_input_dim)`.
  - `:587-594` `FreeIncidenceMaps(node_dim=geometry_dim, latent_dim=hidden)`.
- **The latent's consumers:**
  - `path_terms` `:794` (latent from `path_input`); `:802` `EdgePairTerm(self.endpoint_maps(latent))`; `:804` `self.free_maps(h0=geometry, nodes, latent)`.
  - `_build_attention_operator` `:840-841` (pair-attention cell latents from `path_inputs`, i.e. `h₀`). `src/models/pair_attention.py:158-180` `build_attention_operator`: `cell_scores(h0, nodes, latent, …)`. Its self and edge scores read raw `h₀`, unchanged.
- **Module creation order:** the filter bank is created last (`:657`), and the gate heads are created "last" before it. A new map encoder is created after the filter bank, so that every other parameter initialises as before.
- **Goldens:** `docs/heads_evidence/v2_golden.py:82-91` pins `geometry_context: false`, `schemas`, `first_order` and `symmetric`. Its cases include `pap_free_path_maps` and `pap_endpoint_edges` (`:38-56`). There, geometry = `h₀` = `g`'s input, so no second encoder may be created.
- **Tests:**
  - `tests/test_v2_model.py:1068-1081` `AblationTests` (fixture `_float64_pap`: PAP/PAPAP cells, authors never at an end; `_v2_model` builds with the default `geometry_context: true`);
  - `:1114-1150` T11 for `endpoint_edges` (context off);
  - `tests/test_wisp.py` T-H7 (the ladder at the default context);
  - `tests/test_pair_attention.py`.

## K2 / K2b (evidence scripts only)
- `docs/WISP_evidence/h5_synthetic.py::graph_s1p` (`:87-121`), `config` (`:161-171`), `relevance_of_cells` (`:174-184`).
- `docs/WISP_evidence/i3_s1p_oracle.py`: the I3 oracle harness, mechanism log and component assembly. K2 copies its pattern; β is not a parameter.
- `docs/WISP_evidence/i4_s2_unconfounded.py::graph_s2u`, `readout`, `strict_rank`.
- **Optimizer and schedule:**
  - `src/builders.py:44-54` (two groups: the scalar group has lr = `optim.lr × scalar_lr_scale` and weight_decay 0);
  - `:69-77` (`CosineAnnealingLR(T_max=cfg.epochs)`);
  - `src/trainers/trainer.py:208-213` (`zero_grad` on the optimizer's parameters only; `clip_grad_norm_(model.parameters(), 1.0)`; one step per epoch, full batch);
  - `:166-172` (the scheduler steps once per epoch).
- **Settings:** `configs/experiments/acm_h2snn.yaml` (adamw, lr 0.001, weight_decay 1e-4, cosine, grad_clip 1.0; model d 32, d_r 16, K 4); `src/configs/config.py:1017` `scalar_lr_scale` 10.
- **The operator:** `apply_operator` (`h2snn_model.py`, `A x = G Δ̂ B x`); the filter bank `src/models/filter_bank.py` (`x = Σ_k T_k(A − I)(h₀ W_k)`, k = 0..K, `:83,91`).

## K3: guards (E11, review m1, NIT)
- **`src/configs/config.py`:**
  - `:36-38` `NEVER_TUNED_WISP_KEYS`; `:49-125` `reject_never_tuned_settings` (search space `:108-125`);
  - `:174-210` `NODE_CLASSIFICATION_PRESETS` (`wisp_walks_per_anchor` 42/65/12 at ℓ_max = 4: `:185`, `:193`, `:205`);
  - `:376-400` `DataConfig.__post_init__`; `:461-535` `resolved()` (the r preset, `:509-511`);
  - `:922-990` `AblationConfig.__post_init__` (the `path_weight_context` combinations, `:954`).
- **Consumers of an unresolved `path_source` (`""`):** `src/data/dataset.py:143`; `src/models/h2snn_model.py:1148`.
- **Tests that assert the old `r` behaviour (authorised change, E11):** `tests/test_wisp.py:249` (the Part H rejections test: r passes the search-space guard) and `:1202` (the I5 never-tuned test).
- **Calibration:** `docs/WISP_evidence/h6_calibrate_r.py` (the H1/H6 incidence rule; budgets from `h0_data_facts.json`).

## K4
- `docs/H2S-NN_WISP_followup_report.md` (erratum appended).
- `H2S-NN/AGENTS.md:72` and `H2S-NN/CLAUDE.md:734`: the D2 rows (the author's uncommitted files; edit only the default statement).
