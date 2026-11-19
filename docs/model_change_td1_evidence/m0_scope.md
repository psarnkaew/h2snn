# Model change TD-1 — M0 preconditions and scope (S6, 2026-11-18)

- Worktree `/mnt/data/peeradon/h2snn-model`, branch `model-td1` from `v3b` `9b7ae0b66de790a96ad1b6ad6d4b83caa7a4009c`; `git merge --no-ff 0a9ee3c710a2ee4f06eb3e38436efa173e42b218` (design-docs) → merge commit `ca389e57e4d9394206f482579351f9e3ecffb53b`.
- Suite count at base `9b7ae0b`: **762** test cases (`unittest` discovery over `tests/`, no import failure; job 17684, g2n01, `m0_collect_17684.txt`). `pytest` is not installed in `h2snn-b`, so the count is unittest's (the suite runner is `python -m unittest discover -s tests`). Consistent with the card: D4b 752 OK + 1 skipped = 753 at `45c38ea`, plus 9 added by card 1f.
- Bit-exact references at base (SHA-256):
  - golden `docs/legacy_evidence/v3b_golden_cluster.pt` `6198d18d29ca0df9488e15d350030d0deebd9529a73cb93d053834c6968b1687`
  - mb0 `docs/partMB_evidence/mb0_reference_cluster.pt` `83f5f57d4feee19624f5bb9e38e935032813546c7a159c4f523d36d183c91700`

## Files that reference the removed symbols (grep over `H2S-NN/{src,scripts,tests,configs}`, `CLAUDE.md`, also notebooks/YAML/JSON/sh: none there)

Symbols: `geometry_context`, `path_weight_context`, `RelationalContext`, `relational_context`, `householder_k`, `reflector`, `normalize_reflectors`, `FactorizedNodeToEdgeSheaf`, `FactorizedMapOperator`, `FreeIncidenceMaps`.

| File | Plan (unit) |
|---|---|
| `src/configs/config.py` | Remove `model.geometry_context`, `ablation.path_weight_context` (M1) and `model.householder_k` (M2) from dataclasses, `NEVER_TUNED_*`, guards; a config naming them raises `ValueError` naming spec 1.25 Z1 / Z2. Add `model.map_rank` (M2), `ablation.ego_head` (M2b), `model.path_log_weight_bound` (M2c), `evaluation.type_only_maps` (M2) |
| `src/models/relational_context.py` | Delete (M1) |
| `src/models/h2snn_model.py` | Geometry input `h0`; remove `factored_inputs` / `gathered_linear` branches of `PathEncoder`, `EndpointEncoder`, map MLPs (M1); TD-1 builder and operator in the solve, `FreeIncidenceMaps` → TD-1 family, N2c switch (M2); head reads `x*` (M2b); bounded `log w_f` (M2c) |
| `src/models/sheaf_maps.py` | Remove `gathered_linear` / `mlp_on_blocks` factored path (M1); `FactorizedNodeToEdgeSheaf` → TD-1 builder (M2) |
| `src/models/map_operators.py` | Remove `FactorizedMapOperator`, `normalize_reflectors`; add `TD1MapOperator`; `_mix_rows` for it (M2) |
| `src/models/sheaf_energy.py` | Type hints / docstrings `FactorizedMapOperator` → `TD1MapOperator` (M2); no change in the energy |
| `src/models/fasthpp.py`, `src/models/orthogonal.py` | "reflector" = FastHPP's own Householder vectors that build the frame `R_r` (kept); **no change** |
| `scripts/ablation/arms.py` | Drop `geometry_context_off` (M1) and the appendix pair `appendix_full` / `appendix_endpoints` (they set the removed `path_weight_context`; spec Z-8/Z-9) (M1); add `ego_head` (M2b; pre-registration Z-32) |
| `scripts/ablation/param_table.py` | Map-network parameter names (`rotation_params`, `reflector_mlp`) → TD-1 names (M2) |
| `scripts/known_answer/s3_audit.py` | (ii) probe input `[x_u, x_v, rel]` (no `c`) (M1); (i) tag balance keeps its definition (neighbour tag means) through a local copy of the parameter-free mean, since `RelationalContext` is deleted (M1) |
| `scripts/v3b_cost_smoke.py` | Drop the `relational_context` gradient group (M1); add `stored_elements` and forward/backward timing, run against either code tree (M6) |
| `CLAUDE.md` (line 888 config summary; line 795 D1 row) | Remove `path_weight_context`; note TD-1 replaces `householder_k` (M1/M2) |
| `tests/test_relational_context.py` | Delete (M1) |
| `tests/test_factorized_maps.py` | Delete: Householder-operator tests, replaced by the card's T1–T15 in a new `tests/test_td1_maps.py` (M2/M3) |
| `tests/test_v2_model.py` | Remove T11-ctx (`TwoLatentContextTests`) and appendix-pair cases (M1); port operator fixtures and parameter names to TD-1 (M2) |
| `tests/test_wisp.py` | Remove `geometry_context` / `path_weight_context` parameters and appendix-pair subcases (M1) |
| `tests/test_anchored_solve.py`, `tests/test_sheaf_energy.py`, `tests/test_v3b_theory.py`, `tests/test_paths_and_facts.py` | Operator fixtures built from `FactorizedMapOperator` / `FactorizedNodeToEdgeSheaf` → TD-1 operator / builder (M2); the properties tested are unchanged |
| `tests/test_partMBfix_functoriality.py` | Builder → TD-1 builder (M2) |
| `tests/test_card4_identity_maps.py` | Parameter names `reflector_mlp` / `rotation_params` → TD-1 names (M2) |
| `tests/test_cuda_hardening.py` | FastHPP internals only (frames); **no change** |
| `docs/legacy_evidence/v3b_golden.py` (golden script; card M4) | Cases: drop `nc_geometry_context_false`, `nc_path_weight_context`; add `nc_ego_head` (M4) |

Not changed although the grep hits: `AGENTS.md:133` (D1 decisions row, `householder_k`; the card names only `CLAUDE.md`) and `docs/legacy_evidence/lr0b_classify.py` (history of the legacy-removal card).

Outside the grep but in the card: head widths (M2b, `h2snn_model.py`; `link_readout.py` / `graph_readout.py` take the width from the model), the training-entry refusal of `evaluation.type_only_maps` (M2), golden/mb0 re-capture (M4).
