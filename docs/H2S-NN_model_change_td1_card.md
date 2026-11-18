# Task card — Model change 1.25: remove `c`, TD-1 maps (rank 4, one frame per relation), head reads `x*` only, bounded cell weights, N2c read-out; regenerate golden/mb0; rerun known-answer and cost

**Owner (design):** model session. **Executor:** S4 (or a fresh coding session), new worktree.
**Branch:** new branch `model-td1` from `v3b` `80e045a` (D4b included), in a new worktree `/mnt/data/peeradon/h2snn-model` (`git worktree add`). Then `git merge --no-ff` of `design-docs` at the SHA given in the prompt (brings spec 1.25 and this card). Merge, never rebase.
**Binding (in this order):**
1. Spec **v1.25** (`docs/H2S-NN_methodology_spec.md`; items Z-1–Z-38 in `docs/spec_v1.25_draft_edits.md`).
2. `docs/thm4_td1_proof.md` and `docs/thm4_td1_evidence/check_thm4_td1.py`.
3. `docs/experiment_settings_decisions.md`, rows of 2026-11-18: "Relational context `c` removed", "Restriction maps: SO(d) kept; instance part → TD-1", "TD-1 rank m; pre-implementation diagnostic", "Readout: head reads `x*` only", "free_cell_maps", "Model change: `R_r` init, H2GB timing, bounded `g`".
4. The pre-registration (§4.3 N2c).
5. `docs/working_preferences.md`.

**Mode:** method change plus correctness. The only real runs are two known-answer reruns (synthetic) and the cost smoke on ACM. No metric on real data, no tuning.

**Scope cut for time (author, 2026-11-18):** no factored first-layer path; fewer tests (T9 folded into T1, T13 dropped); two mutation checks; golden/mb0 on one node; known-answer S1′-B `full` runs and S3 only; cost smoke ACM only; short report.

## 0. What changes, in one place

```
geometry input of every map/scale MLP:  h0 only            (was [h0, c]; RelationalContext removed)
per fact e = (u -> v), relation r, rho(r) != r:
    F_{u<e} = c_e * I
    F_{v<e} = c_e * R_r * exp(U_r C_e U_r^T)
per fact with rho(r) = r (and every fact under ablation.direction_blind_maps):
    F_{u<e} = c_e * exp(-U_r C_e U_r^T / 2),   F_{v<e} = c_e * exp(+U_r C_e U_r^T / 2)
    C_e antisymmetric in the endpoints: C_e(u,v) = -C_e(v,u)
R_r   in SO(d): FastHPP frame, ONE per relation (was two: per (relation, side)); none for rho(r) = r
U_r   in R^{d x m}, m = 4: U_r = thin-QR(W_r).Q with sign(diag R) > 0; W_r a learned parameter
C_e   in so(4): 6 coefficients, linear output of one per-edge MLP over [h_u, h_v, rel_emb(r)]
c_e   = softplus(s) + map_scale_eps, s from the SAME MLP (one extra output)
exp(U C U^T) = I + U (exp(C) - I) U^T      (exp(C): 4x4, torch.linalg.matrix_exp, batched)
Q_e   = F_{u<e}^T F_{v<e} / c_e^2  in SO(d);   a_e = c_e^2  (unchanged)
```

- **MLP.** `[h_u, h_v, rel_emb]` (dim `2d + d_r`) → `edge_mlp_hidden_dim` → GELU → `6 + 1`.
- **Initialisation.**
  - Skew part of the last layer: weights and bias are zero, so `C_e = 0` and `Q_e = R_r` at init.
  - Scale output: initialised as the current `edge_scale_mlp` last layer.
  - `R_r`: the current FastHPP initialisation (`N(0, 1/d)` on its parameters).
  - `W_r`: `N(0, 1)`.
- **`ρ(r) = r`.** Evaluate the MLP in both input orders. Skew = `(k(u,v) − k(v,u))/2`; scale input = `(s(u,v) + s(v,u))/2`, as today's symmetrised scale.
- **What stays unchanged.**
  - Forced face maps (Thm 1), the energy, degrees, Jacobi, elimination, solver, `g`, heads and WISP.
  - The `PathTransportOperator` interface: transports come from the map operators as before.
- **Arm `fact_maps_identity` (S2).** `R_r = I` and `C_e ≡ 0`, scale `c_e` kept. This is today's identity operator path; keep it.
- **Arm `free_cell_maps` (C17).** `FreeIncidenceMaps` today uses "the same factorized family as the node→edge maps" (`R_position H_k(u)`). It moves to the same TD-1 family: one frame `R_position`, one `U_position`, and `C_{v,f}` from its existing MLP over `[h_v, path_latent_f, onehot(position)]` (6 outputs, zero-initialised). It stays not forced by functoriality. *Design-session choice, to keep C17 comparing forced vs free within one family; stop and report if it conflicts with anything in the spec.*

## Units

### M0 — Worktree, merge, preconditions
- Create the branch and worktree as above. Merge `design-docs` at the given SHA.
- Record `git rev-parse HEAD`, the suite count at base (no run needed: take D4b's 752 OK + 1 skipped at `45c38ea`), and the golden/mb0 reference SHA-256 at base.
- List every file that references `geometry_context`, `path_weight_context`, `RelationalContext`, `relational_context`, `householder_k`, `reflector`, `normalize_reflectors`, `FactorizedNodeToEdgeSheaf`, `FactorizedMapOperator`, `FreeIncidenceMaps` (src, scripts, tests, configs, CLAUDE.md), with a one-line plan per file. This list is the M1/M2 scope; anything outside it needs a stop.

### M1 — Remove `c` (spec Z-1–Z-9, Z-20)
- **Config.** Remove `model.geometry_context` and `ablation.path_weight_context` from the dataclasses and every guard (`src/configs/config.py`, the `NEVER_TUNED_*` lists, arm tables). A config that still sets either key raises `ValueError` naming "spec 1.25 Z1".
- **Model.**
  - Delete `src/models/relational_context.py`.
  - In `h2snn_model.py`, the geometry input is `h0`, and `path_input_dim = d`. Remove the `factored_inputs` constructor argument and its `gathered_linear` branch in `PathEncoder`, the map MLP and the scale MLP (today it is switched by `model.geometry_context`, not a config key of its own); every MLP keeps its existing unfactored path (concatenate, then `Linear`), which is what runs today with `geometry_context` off.
- **Scripts.**
  - Drop the `geometry_context_off` arm in `scripts/ablation/arms.py`.
  - `scripts/known_answer/s3_audit.py`: the fact-input probe becomes `[x_u, x_v, rel]` (no `c`). Say so in the report, since it changes audit (ii).
  - `scripts/v3b_cost_smoke.py` and `CLAUDE.md:888` (docs).
- **Tests.**
  - Delete `tests/test_relational_context.py`.
  - Update `tests/test_v2_model.py` (T11-ctx, appendix-pair cases) and `tests/test_wisp.py`.
  - Removed keys and the grep check (nothing in `src/` imports `relational_context`) are covered by T8.

### M2 — TD-1 maps (spec Z-10–Z-18)
- **Builder.** Replace `FactorizedNodeToEdgeSheaf` (`src/models/sheaf_maps.py`) with a TD-1 builder that holds:
  - one FastHPP frame per relation that has `ρ(r) ≠ r`;
  - `W_r` per relation;
  - the one per-edge MLP;
  - the antisymmetrisation for `ρ(r) = r`.
  It returns two operators per edge set (src side, tgt side), as today.
- **Operator.** New `TD1MapOperator` in `src/models/map_operators.py`, implementing everything `FactorizedMapOperator` provides and its callers use: `apply`, `apply_T`, `apply_inverse`, `apply_inverse_T`, `apply_orthogonal`, `apply_orthogonal_T`, `select`, `stored_elements`, `num_edges`, `d`.
  - **Stored per fact:** the side's `exp(·)` matrix as `E − I` (4×4) and the scale.
  - **Stored per relation:** `U_r` and the frame.
  - **`apply(x)`:** `x + U_g ((E_e − I)(U_gᵀ x))`, then the frame GEMM per relation group (skipped for `ρ(r) = r` and on the src side), then the scale. Transpose and inverse follow.
  - `stored_elements` = `E·(m²+1) + R·(d² + d·m)`.
- **Config.**
  - Add `model.map_rank: int = 4`. Allowed `2 ≤ m ≤ d` for tests. A tuning base config must have 4, and the key is refused as a search key (`NEVER_TUNED`).
  - Remove `model.householder_k`: it is refused with a message naming spec 1.25 Z2.
  - `orthogonal_map: fasthpp` stays (frames).
- **Arms.**
  - `direction_blind_maps`: every relation is treated as `ρ(r) = r` (no frames, antisymmetric `C`).
  - `fact_maps_identity`: unchanged identity path.
  - `free_cell_maps`: as in §0.
- **N2c read-out (pre-registration §4.3).** Add `evaluation.type_only_maps: bool = False`. When true, `C_e ≡ 0` (skip the exp part) in evaluation forwards only. It is refused in any training or tuning config, so it can be set only by evaluation and analysis scripts.

### M2b — Head reads `x*` only (spec Z-21–Z-34; author decision 2026-11-18 "Readout: head reads `x*` only")
- NC head: `x*` → Linear(d, 128) → GELU → Dropout → Linear (was Linear(2d, 128) on `[x*, h0]`). LP: per endpoint Linear(d, d) → GELU → decoder. GC: pooling over `x*` of every node, then one Linear.
- `instance_features` keeps its concatenation: head input `[x*, instance feature]`.
- New arm switch `ablation.ego_head: bool = False`: the head reads `[x*, h0]` (the 1.24 readout, same widths as 1.24). Refused in a main-table tuning base config, like the other arm keys; an arm of configuration (a) (not separately tuned). `h0_only_readout` is unchanged.

### M2c — Bounded cell weights (spec Z-35–Z-38; author decision 2026-11-18, option c1)
`log w_f = L·tanh(g(ξ_f)/L)` with `L = 10` (config `model.path_log_weight_bound: float = 10.0`, never tuned, refused as a search key), applied wherever `w_f = exp(g)` is formed today (every ladder rung: `full`, `endpoints`, `schema`, `uniform`; GC and LP paths). The log-weight range stop stays; it cannot fire.

### M3 — Tests (float64 unless stated; each with the tolerance given)
| # | Test | Tolerance |
|---|---|---|
| T1 | Every transport from random parameters: `‖QᵀQ − I‖∞`, `det Q`; `stored_elements` equals `E·(m²+1) + R·(d² + d·m)` | ≤ 1e-12; ≥ 1 − 1e-10; exact |
| T2 | Fresh model: `Q_e = R_r` (distinct roles), `Q_e = I` (`ρ(r) = r`) | ≤ 1e-14 |
| T3 | Closed form vs `torch.linalg.matrix_exp(U C Uᵀ)` on the dense d×d matrix, random `C` | ≤ 1e-12 |
| T4 | Flipping the stored orientation of `ρ(r) = r` facts leaves the energy operator unchanged (apply `L̂` to random `Y`) | ≤ 1e-12 |
| T5 | Adjoint `⟨Fx, y⟩ = ⟨x, Fᵀy⟩`; inverse `F⁻¹Fx = x`; both sides, both relation kinds | ≤ 1e-12 |
| T6 | **Thm 4 on the production operator**: inject the circle construction of `docs/thm4_td1_proof.md` through a test-only hook (`U_r ⊇ P`, `R_r = I`, `C_e` per fact) on a fixture with faces. Check: energy at the section, `‖LY‖`, anchored solve with tight tolerance (1e-12, `T_max` large) gives `x* = h0`, argmax accuracy 1.0. Negative control: one fact rotated by 0.3 rad gives energy > 1e-3 | ≤ 1e-10; `x*` ≤ 1e-8 |
| T7 | `torch.autograd.gradcheck` through the per-edge MLP → 4×4 exp → operator apply, and through `W_r` → QR → `U_r` (tiny fixture) | gradcheck default |
| T8 | Removed keys raise (`geometry_context`, `path_weight_context`, `householder_k`) with the spec-1.25 message; nothing in `src/` imports `relational_context` | — |
| T10 | `evaluation.type_only_maps`: every distinct-role `Q_e = R_r`; refused in a training config and a tuning config | ≤ 1e-14 |
| T11 | One training step on ACM-like fixture: loss finite; `W_r`, frames and both MLP layers receive nonzero gradients | — |
| T12 | `free_cell_maps`, `fact_maps_identity`, `direction_blind_maps`: one step finite; direction-blind transports have no frame (`Q = exp(A)`) | ≤ 1e-12 |
| T15 | Bounded weights: for random large `g` (|g| up to 1e3) the log-weight range ≤ 2L and gradients are finite; at initialisation `w_f` equals `exp(L·tanh(b/L))` for every rung | exact / ≤ 1e-12 |
| T14 | Head input widths: NC/LP/GC read `x*` (width d); `ego_head` reads `[x*, h0]` (2d) and equals the 1.24 head input on a fixture; `instance_features` reads `[x*, inst]`; `ego_head` refused in a main-table tuning base config | exact |

**Mutation checks** (each must make at least one test fail; report which):
1. Drop the antisymmetrisation (T4).
2. Use `exp(C)` instead of `exp(C) − I` in the closed form (T3).

### M4 — Golden and mb0 (re-captured by design)
- **References.** `docs/legacy_evidence/v3b_golden_cluster.pt` (script `docs/legacy_evidence/v3b_golden.py`, job template `docs/legacy_evidence/lr1_job.sh`) and the mb0 reference (`docs/partMB_evidence/mb0_reference.py`, `docs/partMB_evidence/mb0_reference_cluster.pt`).
- **Re-capture** both on one node with environment E and 1 thread, exactly as LR-1 / CS8, then compare once on the same node (second run). No second CPU type.
- **Report.** Old and new SHA-256, the node(s), the job ids, and `differing: none` for the compare.
- The golden configurations are the same list as before minus the removed arms (`geometry_context_off`, appendix pair) plus `ego_head`. List any configuration that was dropped or added.

### M5 — Full suite
Full suite on one node (sbatch, one GPU not shared, environment E). Expected: everything passes, the new golden and mb0 compare `differing: none`. Report the count vs 752 OK + 1 skipped at the base.

### M6 — Reruns (one round, no tuning)
- **Known-answer.** Rerun card 4 K3 S1′-B (the `full` runs only; `covering_only` is not rerun) and K4 S3 (after the new audit (ii)) with exactly card 4's frozen settings (`scripts/known_answer/common.py`), seeds and run counts. The only change is the model. K5 S2′ is **not** rerun: the report states that its card 4 result was obtained on the 1.24 model.
  - Report each pre-registered quantity next to card 4's value.
  - The criteria are unchanged and are read exactly as in `docs/known_answer_preregistration_draft.md`.
  - A different outcome is a finding, **not a stop**.
- **Cost smoke.** `scripts/v3b_cost_smoke.py` on ACM only at the card 2 central configuration (`d` 32, `ℓ_max` 3, `r` 16). Run both `v3b` `80e045a` and the new head in the same job on the same GPU.
  - Report time per epoch (forward, backward), PCG iterations (forward, adjoint), peak memory, and `stored_elements` of the map operators.
  - Design-session prediction (FLOP model): operator apply ≈ 0.5×, map build ≈ 0.2×, stored geometry per fact 257 → 17 numbers at d 64. Report measured/predicted. The prediction is **not** a gate.
- **H2GB central-point timing (B2a's `N`). Not run in this card.** The author's 2026-11-18 decision dropped real-data dry runs on the lab cluster. It is measured by the first TSUBAME trial unless the author says otherwise.

### M7 — Report
- **File:** `docs/H2S-NN_model_change_td1_report.md`.
- **Content (short):** files changed per unit; test table with measured errors; mutation results; golden/mb0 hashes; suite count; known-answer side-by-side table; cost table; deviations; decisions for the author. No narrative beyond one line per item.
- **Last line:** `branch=model-td1 sha=<full head sha> card=docs/H2S-NN_model_change_td1_card.md base=80e045a834820fef925d7241c1c4912a2f5eea92`.

## Files
- `src/configs/config.py`
- `src/models/{h2snn_model.py, sheaf_maps.py, map_operators.py}` (and `relational_context.py` deleted); `src/models/fasthpp.py` and `orthogonal.py` only if a frame helper needs it
- evaluation/analysis entry for N2c
- `scripts/ablation/arms.py`, `scripts/known_answer/s3_audit.py`, `scripts/v3b_cost_smoke.py`
- `CLAUDE.md`
- tests
- golden/mb0 references and their capture logs
- evidence folder `docs/model_change_td1_evidence/`
- the report

Nothing else. In particular: not `src/baselines/`, not the spec, cards or pre-registrations, and not WISP, solver, `g` or heads.

## Rules and stop conditions
- **Fixed things.** Do not change ranges, budgets, tolerances, defaults other than those listed, `g`, WISP, the solver or the heads. `m` is 4 everywhere outside unit tests.
- **Stop and report if:**
  - T1, T3, T5 or T6 fail beyond tolerance after one honest fix attempt;
  - a `c` or reflector dependence turns up in a path not in the M0 list;
  - the known-answer scripts need any setting change beyond the model;
  - the golden re-capture is not reproducible on the second run on the same node;
  - the cost smoke shows the new model slower or heavier than `v3b` by more than 1.2× on ACM (unexpected under the FLOP model; report, do not optimise);
  - anything in the spec contradicts this card.
- **Not stop conditions:** known-answer outcomes that differ from card 4, and measured cost ratios that differ from the prediction (but stay within 1.2×).
- **Jobs.** sbatch only; one GPU not shared (no `--overlap` into another session's allocation); `--mem` = recorded peak + 30 %; never the A6000 / g3an01. Job scripts print `git rev-parse HEAD`, hostname, `$SLURM_JOB_ID`.
- **Git.** One commit per unit (M0–M7). Push `model-td1` after each unit; never push another branch. Never stash, reset, rebase, amend or force-push. Do not merge into `v3b`: the design session reviews and the author fast-forwards.
- **Estimate.** M0–M5 ≈ 1.5–2 working days for the agent, plus suite and capture jobs. M6 known-answer ≈ half of card 4's GPU time (queue-dependent). Cost smoke < 0.5 GPU-h. The queue is the main risk; M6 can start as soon as M5 passes.
