# H2S-NN — Part L report: v3-B three-level sheaf energy, anchored solve, arms

**Card:** `docs/H2S-NN_v3B_card.md` (2026-11-11). **Spec:** `docs/H2S-NN_methodology_spec.md` v1.11. **Written:** 2026-11-11, by the Claude Code session that did the work; no independent review (§5).
**Tags:** VERIFIED-RUN (command and numbers in this report), VERIFIED-CODE (file:line on branch `v3b`), DERIVED, UNVERIFIED, DEVIATION (anything not as the card specifies, with reason).
**Rules kept:** no training beyond the 2-epoch IMDB smoke, no tuning, no accuracy number anywhere in this report (the smoke's loss values are not accuracy numbers; its metric values were not stored). The tests listed in card §2 are the only tests written.

## 1. Branch, base, commits
- Branch `v3b` from `c41bf28` (`git checkout -b v3b c41bf28`; `git rev-parse HEAD` at start: `c41bf28c2b5fe7663c45e227fec11419d564975d`). The other sessions' uncommitted files (`AGENTS.md`, `CLAUDE.md`, `docs/H2S-NN_v2_plan.md`, `docs/possible_reviewer_qa.md`, `docs/session_handoff.md`, `docs/working_preferences.md`, `paper/*`, `references/*.pdf`, `Claude outputs/`, `vendor/*`, …) were never staged, edited, stashed or reset; every commit staged its files by explicit path. Not pushed, merged or rebased.
- Baseline before U1 (VERIFIED-RUN `conda run -n h2snn python -m unittest discover -s tests`, output `docs/v3B_evidence/baseline_suite_output.txt`): **Ran 478 tests, FAILED (failures=11, errors=29)**; all 40 in `test_baselines` (23) and `test_heterophily_metrics` (17). The 17 heterophily errors are `OSError: [WinError 4551] An Application Control policy has blocked this file` while importing the vendor clones (the environment-policy blocker named in `CLAUDE.md`), not code failures; Part J recorded these tests as passing on 2026-11-10 with the same clones, so the policy state changed since. Neither module touches the code of this card.

| Unit | Commit | Title (abridged) |
|---|---|---|
| U1 | `c424646` | L1: three-level sheaf energy operator; T1, T2, T3 |
| U2 | `054c6cf` | L2: exact elimination of fact states; T7; IMDB `|E_shared|/|E_P|` count |
| U3 | `43118d8` | L3: anchored solve (per-scope PCG, Jacobi, warm start, schedule, implicit backward); T9, T10 |
| U4 | `d0e879e` | L4: model path `propagation: anchored`, arms, θ_τ group, validated keys, IMDB v3b config; T14 |
| U5 | `1f474c7` | L5: 2-epoch IMDB cost smoke script and JSON; suite outputs |
| U6 | see `git log` (this report, the addendum, and the T7 test refinement of §2.2) | L6 |

## 2. Per unit

### 2.1 U1 — operator (`src/models/sheaf_energy.py`)
**Built (VERIFIED-CODE):**
- `IncidenceComplex.from_face_groups` (`sheaf_energy.py:57-117`): node–fact `[2,E]`, fact–cell `[I_ef, 5] = (f, i, e, v_{i-1}, v_i)`, node–cell `[I_nc, 2]`, `cells_per_fact`, cell lengths, global cell ids across face groups; `num_path_facts` (`|E_P|`), `num_shared_facts` (`|E_shared|`).
- `FactCellTransport` (`:132-184`): `R_{e_i,f} = T_{v_{i-1},f} F_{v_{i-1}◁e_i}ᵀ` from the signed steps of `PathTransportOperator` (`_src[j]` is the near-node map after row mixing, so a −1 hop uses the target role); `up` = the card's UP recursion (`:157`), `down` = DOWN (`:165`), `apply` per incidence for tests (`:175`). `FreeFactCellMaps` (`:186`) for the arm `free_cell_maps`.
- `SheafEnergyOperator` (`:264-461`): `apply` = `L Y`, `degree` = `(D_v, D_e, D_f)` exactly as card §1 (self-loop facts count both roles, `:342-345`), `apply_normalized` = `D^{-1/2} L D^{-1/2} Y` with zero rows where `D = 0`, `initial_guess` (D-B7), flags `node_cell` / `fact_cell` and extra `node_terms` for the arms, `cells_per_fact`.
- `PathTransportOperator` was **not** refactored (card §5 risk): `FactCellTransport` reads its private `_src`, `_step`, `_step_T` (`:154-183`). A coupling to private attributes; listed in §4.

**Tests (`tests/test_sheaf_energy.py`; VERIFIED-RUN `conda run -n h2snn python -m unittest tests.test_sheaf_energy -v` → 4 tests OK; the numbers below from `scratchpad/u1_numbers.py` / `t1_numbers.py`, same fixtures and code):**

| Test | What | Tolerance | Result | PASS |
|---|---|---|---|---|
| T1 | `R_{e_i,f} F_{v◁e_i} = T_{v,f}` at both ends of every fact–cell incidence on one real WISP draw: IMDB, `r = 2`, 50 anchors (`seed 42`), `ℓ_max = 4`, `d = 32`, fp32, maps from a randomly initialised `FactorizedNodeToEdgeSheaf` on random `h` (functoriality is algebraic; any maps) | max ‖·‖ ≤ 1e-5 | 86 cells (L2/L3/L4: 37/28/21), 242 incidences, 107 reversed hops; max residual **4.23e-6** | PASS |
| T2 | matrix-free `L`, `D`, `L̂` vs an independent dense construction (explicit `T`, `R` matrices; N = 12, E = 20 incl. one self-loop fact, F = 6 cells of lengths 2,2,3,3,4,4 with mixed hop signs, d = 4, fp64), for `full`, `covering_only`, `hub_A` term sets | ≤ 1e-12 | `|L_mf − L_ref|` 5.6e-15 / 5.3e-15 / 5.6e-15; `|D|` ≤ 1.8e-15; `|L̂|` ≤ 1.3e-15 | PASS |
| T3 | `<Y₁, L̂Y₂> − <L̂Y₁, Y₂>`; dense `L̂` symmetry; PSD; diagonal blocks of `L̂` = I | ≤ 1e-12; min eig ≥ −1e-10 | 0.0; 2.5e-16; min eig **0.0472**, max eig 1.911; blocks = I to 1.3e-15 | PASS |

**DEVIATIONs U1:** none from the card's specification. Remark: T1's maps are not a trained model's maps (the card does not say whose); recorded in §4.

### 2.2 U2 — elimination (`EliminatedOperator`, `sheaf_energy.py:463-577`)
**Built (VERIFIED-CODE):** `EliminatedOperator(operator, kept=None)` applies the Schur complement of `M = PᵀP + τL̂` over the eliminated fact states exactly and matrix-free: because the fact block of `L̂` is `I`, the optimal eliminated state is `s̃*_e = gather_e / D_e` with `gather_e = a_e(F_u x̃_u + F_v x̃_v) + Σ_f β_f R_{e,f}ᵀ z̃_f`; the reduced operator is the full one with those states filled in (`:505-533`; DERIVED in the docstring and checked by T7). Default `kept` = facts on ≥ 2 cells (`E_shared`), as Prop. 7′; any set is exact (T7 also checks `kept = ∅`). `diagonal_scalars` (`:535`) gives the scalar diagonal of the reduced `L̂` (1 − Σ Schur corrections) for the Jacobi preconditioner; `fact_states` (`:567`) recovers `s*`; `statistics()`.
**Skip rule:** on the IMDB evaluation draw (`r = 12`, `ℓ_max = 4`, seed 42) `|E| = 43,321`, `|E_P| = 33,356`, `|E_shared| = 25,427`, **`|E_shared|/|E_P| = 0.762 < 0.9`** → U2 done (VERIFIED-RUN `scratchpad/t1_numbers.py`; the same numbers appear in the smoke JSON). 37,614 cells, 111,916 fact–cell and 149,530 node–cell incidences.

| Test | What | Tolerance | Result | PASS |
|---|---|---|---|---|
| T7 | dense fp64 solves of `(PᵀP + τL̂)Y = Pᵀh` on the fixture (facts on 0 / 1 / ≥ 2 cells: `cells_per_fact` has 7 zeros, 8 ones, 5 twos), `τ ∈ {1, 7.5}`, term sets `full` / `covering_only` / `hub_A`, `kept = E_shared` (5 states) and `kept = ∅` (0 states): `x*`, `z*`, and the recovered `s*` vs the full-state solve | ≤ 1e-10 | all 24 comparisons ≤ **1.3e-15**; reduced `L̂` symmetric to 2.5e-16; `diagonal_scalars` = diagonal of the materialised reduced `L̂` to ≤ 1e-10 on every row but the eliminated self-loop's node (its cross term is not scalar; §4) | PASS |

**DEVIATION U2-a (`tests/test_sheaf_energy.py:275-287`):** the diagonal-scalar comparison in T7 was first guarded by "skip if any eliminated fact is a self-loop", which on this fixture (self-loop fact on 0 cells, always eliminated) skipped the comparison in every sub-case; found while writing this report, the guard was replaced by excluding only that node's row and T7 rerun (OK). The edit is to a U2 test file after the L2 commit; it is committed with L6.

### 2.3 U3 — solver (`src/models/anchored_solve.py`)
**Built (VERIFIED-CODE):** `pcg` (`anchored_solve.py:83-137`): batched per-scope PCG exactly as card §1 (`segment_dot` `:73`; per-scope `α`, `β`, stopping `‖r‖_s ≤ tol‖b‖_s`; converged scopes masked with `α = β = 0`; one scope = plain inner products). `AnchoredSystem` (`:140`): `M Y`, `Pᵀh₀`, `J⁻¹`. `solve_anchored` (`:175`): D-B7 initial guess or warm start, raises `PCGConvergenceError` at `T_max` (the stop condition is enforced, not just logged). `solver_tolerance` (`:64`): 1e-3 for epochs 0–9, then 1e-4 (0-indexed trainer epochs = "epochs 1–10"); `epoch = −1` (evaluation) → 1e-4. `AnchoredSolve` (`:193-265`): forward under no-grad; backward solves `Mλ = g` by the same PCG from `λ₀ = 0`, then `∂ℒ/∂θ = −∇_θ[τ(θ) λᵀ L̂(θ) Y*]` by autograd of the scalar with `λ, Y*` detached (`∂/∂θ_τ = −τλᵀL̂Y*` included), `∂ℒ/∂h₀ = Pλ`; `SolveStats` records forward/backward iterations and residuals.

| Test | What | Tolerance | Result | PASS |
|---|---|---|---|---|
| T9 | PCG from the D-B7 guess vs dense `torch.linalg.solve` of the materialised `M`, fp64, tol 1e-10, on the reduced (5 fact states, 23 rows) and full-state (20 states, 38 rows) operators, `τ ∈ {1, 9}`; warm start from the solution | ≤ 1e-8; iterations < 100 | `|Y_pcg − Y_dense|` 2.3e-10 / 3.1e-11 / 4.1e-10 / 5.4e-11; iterations 23 / 42 / 27 / 55 (reduced τ=1, reduced τ=9, full τ=1, full τ=9); warm start 0 iterations; cond(M) 7.8 / 20.6 / 9.9 / 29.5 | PASS |
| T10 | implicit gradient vs central finite differences, fp64, ε = 1e-6, dense-solve references; loss `Σ W⊙x*² + Σ W⊙x*`; parameter groups: fact scales (softplus), reflectors (src+tgt), rotations (matrix_exp of a skew matrix, src+tgt), cell weights (exp), θ_τ, h₀ | relative error ≤ 1e-6 per group | worst over 3 random directions per group: scales **6.2e-8**, reflectors 9.0e-9, rotations 3.2e-9, cell weights 3.5e-8, θ_τ 9.1e-10, h₀ 3.4e-10; forward and backward PCG < 100 iterations | PASS |

VERIFIED-RUN `conda run -n h2snn python -m unittest tests.test_anchored_solve -v` → 2 tests OK (144 s; the FD references materialise `M` 36 times); numbers from `scratchpad/u3_numbers.py`.

**DEVIATION U3-a (`anchored_solve.py:159-172`, `sheaf_energy.py:535-561`):** the Jacobi preconditioner uses the exact scalar diagonal of the operator being solved: on the full-state system this is the card's `J = diag(1+τ, τ, τ)`; on the reduced system the node and cell scalars are `1 − Σ_{e∈E_elim} w²` (the Schur corrections), which the card's formula does not include. Reason: the card calls `J` "exact Jacobi after normalisation", which for the reduced system is this; both are SPD preconditioners and T9 passes with it. For an eliminated self-loop fact the scalar omits a non-scalar cross term (`2 w_u w_v F_uᵀF_v`); preconditioning only, never correctness.
**DEVIATION U3-b (`tests/test_anchored_solve.py:150-184`):** T10 compares directional derivatives along 3 random unit directions per parameter group (FD by dense solves), not a coordinate-wise FD over all ~460 coordinates (which would need ~85,000 operator applications); the relative-error criterion is the card's.
**DEVIATION U3-c:** the backward raises `PCGConvergenceError` at `T_max` too (`anchored_solve.py:239-242`); the card names only the forward.

### 2.4 U4 — model + arms (`src/models/h2snn_model.py`, `src/configs/config.py`)
**Config (VERIFIED-CODE `config.py`):** `PROPAGATIONS` (`:52`), `V3B_ARMS` (`:55`), `NEVER_TUNED_V3B_KEYS` (`:59`); `ModelConfig.propagation` (default `chebyshev`, `:836`), `arm` (`:841`), `tau_init`, `solver_tolerance`, `solver_warmup_tolerance`, `solver_warmup_epochs`, `solver_max_iterations` (`:842-850`) with validation (`:853-872`); `TrainConfig.__post_init__` requires `ablation.coupling: symmetric` and refuses the legacy path switches under `anchored` (`:1172-1190`); `reject_never_tuned_settings` refuses `model.arm ≠ full` in a base config and any of the seven keys in a search space (`:138-160`). Defaults leave every legacy config unchanged.
**Model (VERIFIED-CODE `h2snn_model.py`):** `anchored` flag (`:499-503`); no `θ_η`, no filter bank, no gate heads under it (`:552-555`, `:707-709`, validation); `θ_τ = log(tau_init)` created last (`:730`), in `scalar_parameters()` (`:788`) → the 10× lr group; `tau()` (`:755`); `diffusion_scalars` logs `tau` and the solver statistics (`:761-781`); `diagnostics_available` False under anchored (`:1056-1059`; the trainer's spectral diagnostics assume η or the filter bank). `encode(..., epoch=-1)` (`:1185-1201`) → `_encode_anchored` (`:1246-1342`): geometry `[h₀, c]` once, maps, `β_f = w_f m_f` from g (`_cell_weights`, `:1214`), the `build(*params)` closure that rebuilds the (eliminated) operator from the map factors and `β` (so the implicit backward differentiates through them, `:1278-1310`), tolerance from the schedule, warm start in eval mode when the system has the same row count (`:1319-1325`), `anchored_solve`, `x* = Y*[:N]`, head input `[x*, h₀]`. Arms: `covering_only` → `node_cell=False`; `hub_A` → `fact_cell=False`, `kept = ∅`; `closure_only` → no cells, `HolonomyEdgeTerm(transport, β_f)` as a node term (weights enter `D_v` at both endpoints); `no_path_cells` → no cells; `instance_features` → no cells in the solve and `_instance_features` (`:1229`) concatenated to the head (`representation_dim = 3d`, `:663`); `free_cell_maps` → `FreeIncidenceMaps` over the hop positions (`:731-742`), one free orthogonal `R_{e_i,f}` per fact–cell incidence, `T` still forced; `uniform / schema / endpoints` = the existing `ablation.path_weight_input` ladder (unchanged, usable with `anchored`).
**Config file:** `configs/experiments/imdb_h2snn_v3b.yaml` (IMDB, anchored, full, d = 32, r resolved to 12, card solver values).

| Test | What | Result | PASS |
|---|---|---|---|
| T14 (`tests/test_v3b_model.py`) | every arm of `V3B_ARMS` on `_random_pap_data()` (24 papers, 16 authors, 144 facts, 40 PAP + 40 PAPAP cells; `|E_P| = 124`, `|E_shared| = 80`), d = 8, fp32: forward + backward, finite loss, gradient norm per group; the solver converged (< `T_max`) forward and backward; warm start reused in eval; config validation and never-tuned refusals; the legacy default builds no `θ_τ` and keeps the filter bank | all 7 arms: finite loss, nonzero gradient in `node_encoder`, `relation_embedding`, `fact_maps`, `θ_τ`, `readout`; g nonzero in every arm except `no_path_cells` (exactly 0, as it should: g is never read); `free_cell_maps` parameters exist and receive gradient only in that arm; fact states: 80 in `full` / `covering_only` / `free_cell_maps`, **0 in `hub_A`** (and in the cell-free arms); cells 80 vs 0 as expected; PCG fwd/bwd iterations on this fixture 4–10 / 4–8 at tol 1e-3 | PASS |

VERIFIED-RUN `conda run -n h2snn python -m unittest tests.test_v3b_model -v` → 2 tests OK (2.9 s). Legacy modules after the edits (VERIFIED-RUN, output `docs/v3B_evidence/u4_legacy_modules_output.txt`): `test_v2_model test_task_heads test_wisp test_filter_bank test_cooperative test_pair_attention test_trainer test_gap_ladder test_factorized_maps test_full_graph_early_stopping test_wandb_and_runs` → **Ran 193 tests, OK** (the golden bit-identity test of `test_task_heads` included).

**DEVIATION U4-a (`src/builders.py` untouched; `h2snn_model.py:788`):** θ_τ is in the existing scalar optimizer group (10× base lr, no weight decay) together with the path-gate bias `b`, not in a group of its own: the card allows changes only to `h2snn_model.py` and configs, and the groups are built in `builders.py`. Same learning rate as the card asks (10×). Option in §4.
**DEVIATION U4-b (`h2snn_model.py:1185`):** `encode()` gained an `epoch: int = -1` keyword (the tolerance schedule needs the epoch; `forward` passes it; the link and graph trainers call `encode` without it and get the final tolerance).
**DEVIATION U4-c (`h2snn_model.py:1229-1244`):** `instance_features` feeds `Σ_f β_f mean_{u∈f} T_{u,f} h₀_u` to **both endpoints** of every cell, each in its own frame (`T_{v₀} = I`; `T_{v_ℓ}ᵀ·mean` for the far end), because the stored cell does not record which endpoint was the WISP anchor (canonical orientation may reverse the walk, and a cell drawn from both ends is merged). Only target nodes are read by the head. Options in §4.
**DEVIATION U4-d:** `model.checkpointing` is not applied inside `SheafEnergyOperator` (the legacy Laplacian checkpointed each term): the autograd graph of the one operator application in the backward is kept in memory. Cost visible in §3.

### 2.5 U5 — 2-epoch IMDB smoke (`scripts/v3b_cost_smoke.py`, JSON `docs/v3B_evidence/v3b_cost_smoke_imdb.json`)
VERIFIED-RUN (from `H2S-NN/`): `conda run -n h2snn python scripts/v3b_cost_smoke.py --config configs/experiments/imdb_h2snn_v3b.yaml --device cuda --epochs 2 --out ../docs/v3B_evidence/v3b_cost_smoke_imdb.json`. RTX 4060 Laptop (8 GB), torch 2.6.0+cu124, fp32, d = 32, r = 12, ℓ_max = 4, arm `full`, seed 42, AdamW lr 1e-3 (scalar group 1e-2: 2 parameters, θ_τ and b), grad clip 1.0, no OOM (r not halved). 792,248 parameters. Each epoch: a fresh training draw (as the trainer, seed 42 + epoch), one full-batch forward + backward + optimizer step, then one evaluation forward on the fixed draw (not in the card's step; added to exercise the warm start, reported separately).

| Gate (card U5) | Epoch 0 | Epoch 1 | PASS |
|---|---|---|---|
| loss finite | 0.71235 | 0.67731 | PASS |
| loss of epoch 2 differs from epoch 1 | — | 0.67731 ≠ 0.71235 | PASS |
| every parameter group has a nonzero gradient (incl. θ_τ) | node_encoder 0.451, relation_embedding 4.1e-4, fact_maps 0.069, g encoder 2.3e-4, g gate 2.2e-4, readout 0.377, **θ_τ 1.7e-3** | 0.335, 4.4e-4, 0.052, 3.3e-4, 3.4e-4, 0.308, **2.2e-3** | PASS |
| PCG converged (< T_max = 100) in every solve | train fwd 8 / bwd 8; eval fwd 10 | train fwd 8 / bwd 7; eval fwd 8 (warm start) | PASS |
| stop conditions: τ ≤ 50, max log-weight ≤ 30 | τ 1.0101, max log w −0.206 | τ 1.0202, max log w −0.319 | not hit |

Relative residuals at stop: train 4.9e-4 / 4.9e-4 (tol 1e-3), 4.4e-4 / 9.8e-4; eval 6.4e-5, 8.8e-5 (tol 1e-4).

### 2.6 U6 — docs
`docs/minibatch_v3B_addendum.md` (M2 → tolerance and `T_max`; per-scope PCG; what the mini-batch card must add) and this report. No existing doc edited.

## 3. U5 numbers (IMDB) next to v2 H6
v2 reference: `docs/H2S-NN_WISP_report.md` H6, IMDB `full`, r = 12: 37,614 cells, 149,530 node–cell incidences, draw 0.40 s, train 0.63 s, epoch 1.33 s, peak 889 MB (K = 4 Chebyshev, gates, same d = 32, same GPU class). No v1 comparison (v1 was measured on ACM). **These are report-only numbers; the cost gate (memory ≤ 3× v2, PCG ≤ 40, time ≤ 4× v2) is deferred by the card to ACM/DBLP runs and is not claimed here.**

| Quantity | v3-B epoch 0 | v3-B epoch 1 | v2 H6 |
|---|---|---|---|
| peak `max_memory_allocated` (MB) | 1,310 | 1,324 | 889 |
| draw s (CPU, WISP) | 0.33 | 0.41 | 0.40 |
| train step s (fwd + bwd + step) | 4.53 (includes CUDA / cuDNN warm-up of the first step) | **0.74** | 0.63 |
| epoch s (draw + train) | 4.87 | **1.15** | 1.33 |
| evaluation forward s | 0.49 | 0.30 | — |
| PCG iterations fwd / bwd (train, tol 1e-3) | 8 / 8 | 8 / 7 | — (K = 4 applications) |
| PCG iterations eval (tol 1e-4) | 10 | 8 (warm start) | — |
| `|E|`, `|E_P|`, `|E_shared|` (train draw) | 43,321 / 33,356 / 25,427 | 43,321 / 33,215 / 25,397 | — |
| eliminated share (facts without a state) | 41.3 % | 41.4 % | — |
| fact states kept, cells | 25,427 / 37,614 | 25,397 / 37,666 | 0 / 37,614 |
| state rows `N + |E_shared| + F` | 84,461 | 84,483 | 21,420 |
| τ after the step | 1.0101 | 1.0202 | — |
| max / mean log w_f | −0.21 / −0.83 | −0.32 / −1.10 | ≤ 1.07 (H6) |

Reading (DERIVED, not a gate): memory 1.49× v2, epoch time 0.86× v2 at the warm-up tolerance 1e-3 with 8 iterations (≈ 16 operator applications forward + 15 backward vs v2's 4 + 4). Memory is dominated by the autograd graph of the one operator application in the backward (DEVIATION U4-d) and the CUDA allocator; the PCG vectors are 84k × 32 × 4 B ≈ 11 MB each. Iteration counts at tol 1e-4 and on ACM/DBLP are unmeasured; the toy-fixture condition numbers (§2.3) are not transferable.

## 4. Judgment calls — each a decision for the author
1. **Jacobi on the reduced system** (DEVIATION U3-a). Options: (a) keep the exact reduced scalars (implemented); (b) the card's literal `diag(1+τ, τ, τ)` (one line in `AnchoredSystem.jacobi_inverse`); (c) block Jacobi for eliminated self-loop facts (d×d blocks; only matters if self-loop facts are eliminated and many).
2. **θ_τ's optimizer group** (DEVIATION U4-a): (a) keep it in the scalar group with `b` (10× lr, implemented); (b) a third group of its own in `builders.py` (one-line change outside the card's file list; same lr unless the author wants another).
3. **`instance_features` anchor** (DEVIATION U4-c): (a) both endpoints in their own frames (implemented); (b) store the WISP anchor per cell in `FaceGroup` (sampler change; a cell drawn from two anchors needs two entries or a split of `m_f`); (c) `v₀` of the canonical orientation only (frame-consistent but anchor-blind).
4. **Elimination set**: the card's `E_shared` is implemented as the default. The Schur complement is exact for any set (T7 also passes with every fact eliminated; `hub_A` uses that), and eliminating all facts gives a smaller state (IMDB: 59k instead of 84k rows) at the same per-application cost. Options: (a) keep `E_shared` (spec §3 table: "state `(N + |E_shared| + F)·d`"); (b) eliminate every fact in `full` too and recover `s*` on demand (`fact_states`); (c) make it a solver setting (never tuned). Note that (b) changes nothing in `x*, z*` (exact) but changes what L14 / C11 call "fact states".
5. **Evaluation tolerance**: evaluation solves use the final tolerance 1e-4 from epoch 0 (`epoch = −1`), training solves the schedule. Options: (a) keep; (b) evaluation follows the current training epoch's tolerance (pass the epoch to `_eval_epoch`).
6. **Initial guess** `s₀ = ½(F_u h_u + F_v h_v)`, `z₀` = plain mean (card D-B7, implemented): under normalisation the optimum given `x = h` is `D_e^{-1/2}·gather` instead; a free change in `initial_guess` that may save a few iterations. Options: (a) keep the card's; (b) the normalised optimum.
7. **Checkpointing inside the operator** (DEVIATION U4-d): (a) none (implemented; 1.3 GB on IMDB); (b) wrap the backward's `λᵀL̂Y*` recomputation per term in `checkpoint_if_grad` as the legacy Laplacian does (memory down, one more operator application in time).
8. **`PathTransportOperator` private attributes** used by `FactCellTransport`: (a) leave; (b) a later housekeeping card exposes `near_maps(j)`, `step`, `step_T` publicly.
9. **Full elimination in the Jacobi/scope statistics of `hub_A`**: `hub_A` reports `num_shared_facts` of the draw but `num_fact_states = 0`; fine for the log, but C11's "capacity" reading should use the latter.
10. **Backward raising at `T_max`** (DEVIATION U3-c): (a) raise (implemented; the stop condition then stops training as the spec intends); (b) record only.
11. **T1's maps**: randomly initialised, not a trained model's; functoriality is algebraic. A re-run on a trained checkpoint is one line in the test if the author wants it.

## 5. Not done, and why
- **Cost gate vs v2 / v1 and ACM/DBLP cost runs**: deferred by the card (U5 is IMDB only; the gate runs before C13 is written).
- **Multi-scope PCG test** (batch = one scope at a time, bit-identical): not in the card's test list; the code path exists (`scope_rows`, `num_scopes`) and is UNVERIFIED (addendum §2).
- **Kernel test (dim ker L̂ = dim ker Δ₀; collapse under `free_cell_maps`)**, **mutation tests**, **unroll-vs-implicit check** of `v3B_final_model.html` §2.1: not in the card; not written.
- **Full suite after the last commit**: run after L5 (§5a); L6 adds only docs and the T7 refinement, which was rerun alone.
- **Independent review**: none; the author asked for the report and a stop.
- **Removal of the v2 code paths** (filter bank, gates, `VertexPathSheafLaplacian`): out of scope (card §0); they stay behind `propagation: chebyshev` and their tests pass (§2.4).
- The smoke's accuracy-type metrics were computed by the trainer's metric function but not stored or read.

### 5a. Full suite after L5 (VERIFIED-RUN, `docs/v3B_evidence/final_suite_output.txt`)
`conda run -n h2snn python -m unittest discover -s tests` → **Ran 486 tests (478 + the 8 new), FAILED (failures=11, errors=29)**: the same 23 in `test_baselines` and 17 in `test_heterophily_metrics` as the baseline (§1), no new failure; all 8 new tests pass. The U2 test refinement (DEVIATION U2-a) is included in this run.

## 6. Time spent per unit (agent wall-clock, approximate)
Reading the card, spec, decisions and code: ~35 min. U1 ~40 min · U2 ~10 min (written with U1, committed separately) · U3 ~30 min (T10's FD references dominate the test's 144 s) · U4 ~50 min (incl. one indentation defect of my own found by the test run and fixed) · U5 ~10 min · U6 ~35 min. Total ≈ 3.5 h against the card's 7–9 h estimate. Nothing in the card's §5 risk list (refactoring `PathTransportOperator`, OOM) materialised.
