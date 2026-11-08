# H0 inventory: consumers of the interfaces Part H changes

Written 2026-11-08 on branch `wisp`, before any H1 code. Lines are VERIFIED-CODE at `19a9d36` (grep
plus reading). `src` paths are relative to `H2S-NN/`. Each block gives the H unit that touches it.

## 1. `FaceGroup` (`src/data/structures.py:12-40`; `face_lengths_of` :43-50)
**Change (H1).** Cells are grouped by length, not signature, and `relation_index` becomes genuinely
per cell. New fields, with defaults so that every existing constructor stays valid: a per-cell
signature id, and a signature table for the read-outs.

| Consumer | file:line | What it relies on |
|---|---|---|
| D8 schema sampler | `src/data/path_sampler.py:241-283` (`merge_cells`; constant-relation check :273) | one group per signature; `relation_index` constant |
| schema builder | `src/data/path_sampler.py:292-333` | groups per length, each merged per signature |
| free (LP/GC) builder | `src/data/path_sampler.py:336-` | grouped by realised signature |
| relation check | `src/data/face_geometry.py:79-100` (`validate_relation_sequence_constant`) | raises if relations vary inside a group |
| DFS builders | `src/data/graph_faces.py:186-210`, `src/data/preprocessing.py:385-475` | constant relation rows (untied/dfs baselines) |
| NC dataset | `src/data/dataset.py:126-167` (`build_node_classification_paths`), call :209 | the D8 sampler only |
| LP dataset | `src/data/link_prediction_dataset.py:716-760` | `build_free_path_groups`; leak check reads `template_name` :755-760 |
| LP structures | `src/data/link_pred_structures.py:69` | tuple of groups |
| GC trainer | `src/trainers/graph_classification_trainer.py:207-224` | `build_free_path_groups` per batch |
| GC dataset | `src/data/graph_classification_dataset.py:421` | `face_lengths=(2,)` hard-coded |
| NC trainer resampling | `src/trainers/trainer.py:46-62` (`training_draw`) | rebuilds the groups with seed `cfg.seed + epoch` |
| LP trainer resampling | `src/trainers/link_prediction_trainer.py:55-60` | same, on the message graph |
| model | `src/models/h2snn_model.py:30, 646-705, 713-740, 756, 874, 904, 923, 957` | `face_node_index`, `relation_index[0]` (:653), `path_weight`, `num_faces` |
| preprocessing diagnostics | `src/data/preprocessing.py:155, 317` | iterate the groups |
| tests | `tests/test_factorized_maps.py:23,58,83-88,309` (asserts one relation row per group); `tests/test_v2_model.py:747-755`; `tests/test_data.py:250`; `tests/test_paths_and_facts.py:1035,1051,1325` | construction and constant relation rows |
| goldens | `docs/heads_evidence/v2_golden.py:29, 98, 111-113` | `FaceGroup.__slots__` round trip; it filters to known slots, so new defaulted fields are safe |

## 2. `build_schema_path_groups` (`src/data/path_sampler.py:292-333`)
**Change (H1).** Unchanged. It stays the `path_source: schemas` comparison row.
Consumers: `src/data/dataset.py:10,155`; `tests/test_paths_and_facts.py:801-816, 901`.

## 3. `PathEncoder` (`src/models/h2snn_model.py:203-248`) and `EndpointEncoder` (:251-283)
**Change (H2).** Per-cell relation embeddings, so the encoder gathers `relation_embeddings[relation_index]`
per cell instead of one schema token. Ladder inputs (`schema`, `endpoints`) and the capacity rule
for `endpoints` are added.

| Consumer | file:line |
|---|---|
| schema token from row 0 | `h2snn_model.py:239` (PathEncoder), `:274` (EndpointEncoder) |
| construction | `h2snn_model.py:498, 507-520` |
| latent call with `group.relation_index[0]` | `h2snn_model.py:646-658` (`_path_latent`) |
| pair-attention latents | `h2snn_model.py:729` |
| free maps (latent input) | `h2snn_model.py:334-384, 521-529` |

## 4. `_path_weights` (`src/models/h2snn_model.py:660-665`)
**Change (H2).** `exp(g)` under `path_weight: exp`, and the ladder.

| Consumer | file:line |
|---|---|
| the path term | `h2snn_model.py:683-686` (× merged `m_f`) |
| coupling statistics | `h2snn_model.py:996-1007` (`path_gate_mean/max`) |
| evidence scripts (not code paths) | `docs/D12_evidence/g3c_binding.py`, `g3c_v2_oracle.py` (monkeypatch) |

## 5. `theta_gamma` / `gamma()`
**Change (H2).** Under `exp` there is no `θ_γ` and `gamma_init` is rejected; `b` (the bias of `g`)
joins the scalar optimizer group.

| Consumer | file:line |
|---|---|
| parameter | `h2snn_model.py:497, 503-505` |
| `gamma()` | `h2snn_model.py:598-601` |
| logging | `h2snn_model.py:612-614` (`diffusion_scalars`) |
| scalar optimizer group | `h2snn_model.py:619-631`; `src/builders.py:46-55` |
| operator | `h2snn_model.py:774-789`; `DiffusionOperator.gamma` :171; `H2SNNOutput.gamma` :197, :992, :1005 |
| Laplacian | `src/models/vertex_path_laplacian.py:226-237` (required when path terms exist), `:250-256` (`D`), `:282-283` (`L x`), `:303` (energy split): `gamma` multiplies the path part |
| pair attention | `src/models/pair_attention.py:107-115` (duck-typed `gamma = None`) |
| config | `src/configs/config.py:617, 676-679, 707` (`gamma_init`) |
| tests | `tests/test_cooperative.py:585`; `test_filter_bank.py:380, 400`; `test_pair_attention.py:375`; `test_task_heads.py:158`; `test_v2_model.py:898, 1006, 1082, 1264, 1290`; `test_wandb_and_runs.py:145` |
| evidence | `docs/D12_evidence/g3b_smoke.py` (reads `model.gamma()`) |

## 6. `uniform_path_gate`
**Change (H2).** Removed; its behaviour is `path_weight_input: uniform` (strict loading, no shim).

| Consumer | file:line |
|---|---|
| config field and validation | `src/configs/config.py:730, 757-764, 783-788` |
| model | `h2snn_model.py:517, 530, 662` |
| tests | `tests/test_pair_attention.py:354`; `tests/test_v2_model.py:1092, 1298` |
| goldens | `docs/heads_evidence/v2_golden.py:40` (case `uniform_path_gate` → `path_weight_input: uniform` under `sigmoid_gamma`, which must stay bit-identical) |
| evidence | `docs/D12_evidence/g3c_binding.py` (arm name; historical) |

## 7. `face_lengths`
**Change (H1/H2).** WISP lengths are 2..ℓ_max. `face_lengths_of` reads the draw, so a length
missing from one draw but present in a later training draw would have no projection. H2 builds
projections for every length in 2..ℓ_max under `wisp`.

| Consumer | file:line |
|---|---|
| `ModelDims` | `h2snn_model.py:113, 134, 155` |
| encoders | `h2snn_model.py:207-216, 235-236, 255-259, 510, 519` |
| max positions | `h2snn_model.py:525, 583` |
| GC dims | `src/data/graph_classification_dataset.py:421` |
| tests | `tests/test_cooperative.py:51, 225`; `test_paths_and_facts.py:537`; `test_task_heads.py:204`; `test_v2_model.py:778` |

## 8. `path_weight` (`FaceGroup.path_weight` = merged `m_f`)
**Change (H1).** `m_f` = Σ (1/r)·p/(½p+½q); with `q = p` the ratio is exactly 1, so `m_f` = Σ 1/r,
as in D8.

| Consumer | file:line |
|---|---|
| definition | `src/data/structures.py:21, 39` |
| producer | `src/data/path_sampler.py:281` |
| model | `h2snn_model.py:685-686, 729-732` |
| tests | `tests/test_cooperative.py:143`; `test_paths_and_facts.py:816, 901, 1043, 1051`; `test_v2_model.py:1098-1103, 1156` |

## 9. Config key collisions (`src/configs/config.py`)
VERIFIED-CODE, by grep: none of `path_source`, `wisp_*`, `path_weight` (as a config key),
`path_weight_init` or `path_weight_input` exists. The name `path_weight` is already a
`FaceGroup` field (`m_f`); `model.path_weight` lives in a different namespace, so there is no
collision, but the report will spell out which is which.
`reject_never_tuned_settings` is at :36-90 (key tuples :21-31). `gamma_init` is used in no sweep
config (the test only asserts it is absent from sweeps).

## 10. Data facts (VERIFIED-RUN: `python ../docs/WISP_evidence/h0_data_facts.py`; `h0_data_facts.json`)

| | ACM | DBLP | IMDB |
|---|---|---|---|
| anchors (target type) | paper 3025 | author 4057 | movie 4932 |
| facts | 273,936 | 119,783 | 43,321 |
| self-loop facts | 8 | 0 | 0 |
| duplicate facts (same direction) | 0 | 0 | 0 |
| actions at the target type | 5 | 3 | 3 |
| schemas run, eval draw (seed 42): cells | PAP 12,630; PSP 121,744 | APA 6,016; APTPA 168,948 | MDM 8,165; MAM 39,519 |
| schemas run: incidences (the `r` budget) | 403,122 | 862,788 | 143,052 |

- **Reciprocal cites.** ACM has 25 reciprocal cite pairs (P→P′ and P′→P). They are distinct
  (relation, direction) actions, so "uniform unvisited neighbour via an action" equals "uniform
  over that action's facts to unvisited nodes" on all three datasets.
- **Self-inverse relations.** There are none in HGB; the fixtures provide one.

## 11. Not touched
- Fact layer, transports (`PathTransportOperator`), `PathCellTerm` (D9 O(L) term), filter bank,
  gates, relational context.
- Link prediction and graph classification samplers (WISP for them is out of scope). They must
  still run under `path_weight: exp` (H6 smoke).
