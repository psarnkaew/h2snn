# Card 4 K0 — preconditions (no code)

Paths relative to `H2S-NN/` unless they start with `docs/`. All VERIFIED-CODE at the worktree head
`8bd6647` (= `v3b` 2ef9cac + merge of `design-docs` 9309297).

## Branch and merge
- `git fetch origin`: `origin/v3b` = `2ef9cac14ae03136efb07ad0c71233697ae138b1` (as pinned);
  `origin/design-docs` contains `93092977c5045c5c9b851d412c81306312d52b23` (`merge-base --is-ancestor`: yes).
- `git worktree add -b ablation-prep /mnt/data/peeradon/h2snn-ablprep 2ef9cac…`, then
  `git merge --no-ff 9309297…` → `8bd6647`, no conflict.
- Vendor clones copied (not linked) from `/mnt/data/peeradon/h2snn/H2S-NN/vendor`; each `git rev-parse HEAD`
  equals its `vendor/VENDORS.tsv` commit (ChebNetII ded6c18, Cooperative-Sheaf-NN a1a12f7,
  deep-neural-sheaf-diffusion 3f706c2, dgl 3d16000, fasth b3fddf1, H2GB d3b49db, H2SGNN 93c33fa,
  Hetero2Net 4913394, Heterogeneous-Sheaf-Neural-Networks 6964e1e, HGB ca6fd5b, HSB-GDL 4126e64,
  neural-sheaf-diffusion 11e21b5, PSHGCN f87406c, SeHGNN e92bd37, tudataset 26e6ef6). `H2S-NN/data` is a
  symlink to `/mnt/data/peeradon/h2snn/H2S-NN/data` (untracked, never staged).

## Map operator under `propagation: anchored`
- Fact maps: `FactorizedNodeToEdgeSheaf` (`src/models/sheaf_maps.py:108`), built at
  `src/models/h2snn_model.py:436`, called once per forward at `h2snn_model.py:722`; per-edge scale
  `c_e = softplus(edge_scale_mlp([g_u, g_v, rel])) + eps` (`sheaf_maps.py:251-271`), reflectors from
  `reflector_mlp` (`sheaf_maps.py:235-249`), frames `R_{ψ,side}` from `rotation_params` (`sheaf_maps.py:170, 182-193`).
  Input `g = [h0, c]` (`h2snn_model.py:598-603`; `c` parameter-free, `src/models/relational_context.py`).
- Operator class: `FactorizedMapOperator` (`src/models/map_operators.py:95`); rebuilt from the parameter
  tensors inside the solve's `build` closure (`h2snn_model.py:751-783`, params at `:784-785`).
- Transports: `PathTransportOperator` (`map_operators.py:379`; step `Q_j` at `:465-475`, shared scale ⇒
  orthogonal factors only); built per cell group by `build_cell_group` (`src/models/sheaf_energy.py:230-241`),
  fact→cell maps `FactCellTransport` (`sheaf_energy.py:132`).
- Fact weight `a_e = c_e²` (`sheaf_energy.py:300`): the scale is the fact weight, the maps act through their
  orthogonal factor only (`sheaf_energy.py:370-396`).

## Jacobi / diagonal preconditioner input
- `AnchoredSystem.jacobi_inverse` (`src/models/anchored_solve.py:215-229`) reads
  `EliminatedOperator.diagonal_scalars` (`sheaf_energy.py:535-561`), which uses `op.a` (= `c_e²`), `op.beta`
  and `SheafEnergyOperator.degree` (`sheaf_energy.py:339-362`). No map enters the diagonal except through the
  omitted self-loop cross term `2 w_u w_v F_uᵀ F_v` (docstring `sheaf_energy.py:536-539`).

## Arms of pre-registration §2 (status in v3-B code)
| Arm | Key | Where | Exists |
|---|---|---|---|
| `full` | `model.arm: full` | `src/configs/config.py:133, 1276` | yes |
| `endpoints` | `ablation.path_weight_input: endpoints` | `config.py:1404`; `h2snn_model.py:468-472` (`endpoint_hidden_width`, `:308-318`) | yes |
| `schema` | `ablation.path_weight_input: schema` | `h2snn_model.py:473-474` | yes |
| `uniform` | `ablation.path_weight_input: uniform` | `h2snn_model.py:475-484` (`path_log_weight`, `:482`) | yes |
| `no_path_cells` (β_f = 0) | `model.arm: no_path_cells` | `config.py:133`; `h2snn_model.py:728` (cells not in the solve) | yes |
| `closure_only` | `model.arm: closure_only` | `h2snn_model.py:771-775` (`HolonomyEdgeTerm`) | yes |
| `instance_features` | `model.arm: instance_features` | `h2snn_model.py:490-491, 692-708, 823-824` | yes |
| `hub_A` | `model.arm: hub_A` | `h2snn_model.py:778-782` (`fact_cell=False`, no kept fact states) | yes |
| `covering_only` | `model.arm: covering_only` | `h2snn_model.py:778` (`node_cell=False`) | yes |
| `direction_blind_maps` | `ablation.direction_blind_maps: true` | `config.py:1413`; `h2snn_model.py:424-427` | yes |
| `free_cell_maps` | `model.arm: free_cell_maps` | `h2snn_model.py:534-543, 742-748, 762-769` | yes |
| geometry context off | `model.geometry_context: false` | `config.py:1262`; `h2snn_model.py:431-435` | yes |
| `h0`-only | `ablation.h0_only_readout: true` | `config.py:1410`; `h2snn_model.py:397, 671-672` | yes |
| fixed-literature-schema row | `data.path_source: schemas` | `config.py:742, 1571` (`is_literature_row`) | yes |
| appendix pair | `ablation.path_weight_context: true` | `config.py:1408`; `h2snn_model.py:458-460` | yes |
| S2, S0 (maps = I) | — | — | new (K1) |

Every §2 arm marked "exists" exists in v3-B code: no stop item. Pairing by seed: WISP draws are seeded by
`torch.Generator().manual_seed(chunk_seed(seed, k))` / `manual_seed(int(seed))`
(`src/data/wisp_sampler.py:394, 467`), the training draw of epoch `e` by `cfg.seed + e`
(`src/trainers/trainer.py:68-84`).

## Oracle hooks of Part K and the variance gate
- `docs/WISP_evidence/h5_synthetic.py:87-122` `graph_s1p(n, authors=8, per_author=8, bits=6)`;
  `:174-184` `relevance_of_cells` (relevant iff a cell ends at the relevant cpaper); `:31-33` split (seed 1, 50/25/25).
- `docs/WISP_evidence/i3_s1p_oracle.py:51-63` the v2 oracle (`H2SNN._path_weights` replaced), `:76-112`
  `mechanism` (relevant share of `D`, v2 operator); `docs/WISP_evidence/k2_capacity.py:52-64`
  `install_oracle` (fixed-β oracle: `w_f = exp(b + β o_f)`, `b` learnable in the scalar group, β a buffer
  with `requires_grad` for `∂L/∂β`), `:67-88` logistic probe, `:91-133` dispersion.
- `docs/WISP_evidence/i4_s2_unconfounded.py:35-58` `graph_s2u(n)`, `:61-125` read-out (σ\* strict rank).
- `docs/WISP_evidence/h3_variance_gate.py` (93 lines): q from `proposal_from_mass`
  (`src/data/wisp_sampler.py:269-302`), 20 draws (seeds 1000–1019), 16 Rademacher probes (generator seed 7),
  `V = mean_probes Σ_rows M2/20` of `L_P z`, `L_P = Σ_f w_f m_f L_f` (v2 `apply_path`, i.e. `PathCellTerm`
  per group: `src/models/vertex_path_laplacian.py:31-80`, still in v3-B code).
  These scripts call the removed v2 API (`build_operator`, `apply_operator`, `op.laplacian`); K2–K5 port them
  under `scripts/known_answer/` without changing estimators, graphs or criteria.

## Readings recorded before any code (decisions of this card's agent; DERIVED)
1. **S2 and `a_e`.** In v3-B the fact weight is the map's scale, `a_e = c_e²`, produced by
   `edge_scale_mlp` inside the same module as the reflectors and frames. The card asks for `a_e` unchanged
   **and** the geometry network not called; both hold only if "geometry network" means the part that produces
   the orthogonal factor `R H` (frames `rotation_params` and `reflector_mlp`). S2 therefore keeps `c_e`
   (hence `a_e`, read from `[h0, c]`) and replaces `R H` by `I`; the frame parameters and `reflector_mlp` are
   absent from S2's parameter set.
2. **Key name.** `ablation.identity_maps` is a removed v2 key that raises (`config.py:23-27`); the new key is
   `ablation.maps ∈ {learned, identity}`, default `learned`.
3. **Tuning rule.** The existing arm keys are refused in a tuning base config (`config.py:183-192`); `maps:
   identity` follows the same rule, and `maps` is refused in a search space.
