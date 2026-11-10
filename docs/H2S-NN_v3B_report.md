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

## 7. L7: ACM / DBLP cost smoke and PCG at tolerance 1e-4 (2026-11-11; author decision (b): ACM + DBLP)
**Scope:** the spec §6 cost gate (C13) on ACM, DBLP and IMDB. This follows the handoff prompt (`docs/v3B_handoff_2026-11-11b.md` §5), with DBLP added on the author's order. No accuracy numbers, no tuning, no other runs. Machine and environment as in §2.5: RTX 4060 Laptop 8 GB, torch 2.6.0+cu124, fp32, seed 42, arm `full`, d = 32, ℓ_max = 4, AdamW lr 1e-3, scalar group 1e-2, grad clip 1.0. The spec on disk (v1.11) differs from the `v3b` HEAD; it is another session's uncommitted edit, was read but not edited or staged here.

### 7.1 What was added (VERIFIED-CODE)
- **Configs** `H2S-NN/configs/experiments/acm_h2snn_v3b.yaml` (r = 42) and `dblp_h2snn_v3b.yaml` (r = 65). `r` is the calibrated value at ℓ_max = 4 (`WISP_WALKS_PER_ANCHOR`, `src/configs/config.py:255-259`), written explicitly. Every `model`, `ablation` and `optim` key equals `imdb_h2snn_v3b.yaml` (checked with diff). ACM keeps `acm_val_ratio: 0.2` and `acm_split_seed: 42` from `acm_h2snn.yaml`.
- **Script** `scripts/v3b_cost_smoke.py`: new opt-in flag `--final_tolerance_step`. It is off by default, so the §2.5 command is unchanged. After the epochs it runs `final_tolerance_probe`: one training forward + backward on a fresh draw (seed 42 + `epochs`), with the model's epoch set to `solver_warmup_epochs` (= 10, the first epoch at tolerance 1e-4). There is no optimizer step. Peak memory is reset before the step and measured for that step alone.

### 7.2 Commands (VERIFIED-RUN; run from `H2S-NN/`; all exit 0; no OOM, so `r` was never halved)
```
conda run -n h2snn python scripts/v3b_cost_smoke.py --config configs/experiments/acm_h2snn_v3b.yaml  --device cuda --epochs 2 --final_tolerance_step --out ../docs/v3B_evidence/v3b_cost_smoke_acm.json
conda run -n h2snn python scripts/v3b_cost_smoke.py --config configs/experiments/dblp_h2snn_v3b.yaml --device cuda --epochs 2 --final_tolerance_step --out ../docs/v3B_evidence/v3b_cost_smoke_dblp.json
conda run -n h2snn python scripts/v3b_cost_smoke.py --config configs/experiments/imdb_h2snn_v3b.yaml --device cuda --epochs 2 --final_tolerance_step --out ../docs/v3B_evidence/v3b_cost_smoke_imdb_l7.json
```
Each JSON has its console output beside it (`*_log.txt`). The L5 JSON `v3b_cost_smoke_imdb.json` is untouched. The ratios come from `docs/v3B_evidence/l7_ratios.py`, run from the repository root.

### 7.3 Numbers (VERIFIED-RUN; epoch 1 is the steady state, epoch 0 includes CUDA warm-up)
- **v2 reference:** `docs/H2S-NN_WISP_report.md` H6. Its evaluation draws are the same as ours: 103,103 / 206,698 / 37,614 cells.
- **v1 reference:** ACM 9,260 MB / 3.0 s (spec C13).

| Quantity | ACM | DBLP | IMDB (this session) |
|---|---|---|---|
| N, facts `|E|`, parameters | 10,942 / 273,936 / 488,678 | 26,128 / 119,783 / 359,895 | 21,420 / 43,321 / 792,248 |
| r, cells (eval draw) | 42, 103,103 | 65, 206,698 | 12, 37,614 |
| `|E_P|` / `|E_shared|` / ratio | 100,902 / 31,459 / 0.312 | 106,414 / 85,121 / 0.800 | 33,356 / 25,427 / 0.762 |
| eliminated share (facts without a state) | 88.5 % | 28.9 % | 41.3 % |
| state rows `N + |E_shared| + F` | 145,504 | 317,947 | 84,461 |
| loss epoch 0 → 1 (finite, different) | 1.1192 → 1.0572 | 1.4056 → 1.3704 | 0.7124 → 0.6773 (= L5) |
| nonzero gradient in every group incl. θ_τ (epochs 0, 1) | yes, yes | yes, yes | yes, yes |
| τ after epoch 1; max log w_f | 1.0103; 0.089 | 1.0194; 0.242 | 1.0202; −0.319 |
| PCG train fwd/bwd at tol 1e-3 (epochs 0, 1) | 7/6, 7/6 | 8/7, 8/7 | 8/8, 8/7 |
| PCG eval fwd at tol 1e-4 (cold, warm) | 9, 7 | 11, 8 | 10, 8 |
| **PCG train fwd/bwd at tol 1e-4** (§7.1 step) | **9 / 8** | **11 / 10** | **10 / 10** |
| peak MB, epoch 1 (× v2) | 2,801 (1.69×) | 4,942 (1.83×) | 1,324 (1.49×) |
| peak MB, tol-1e-4 step (× v2) | 2,833 (1.71×) | 5,009 (1.85×) | 1,340 (1.51×) |
| epoch s = draw + train, epoch 1 (× v2) | 4.32 (1.53×) | 6.20 (1.60×) | 3.12 (2.35×) |
| draw s (× v2) | 1.76 (1.87×) | 3.76 (1.88×) | 0.88 (2.19×) |
| train step s (× v2) | 2.56 (1.83×) | 2.43 (1.79×) | 2.25 (3.57×) |
| draw + fwd/bwd at tol 1e-4, s (× v2 epoch) | 5.05 (1.79×) | 7.34 (1.90×) | 3.51 (2.64×) |
| vs v1 (ACM) | memory 0.30×; time 1.44× (caveat in 7.4) | — | — |

### 7.4 Timing is not reliable in this session (VERIFIED-RUN; cause UNVERIFIED)
- **Finding:** the machine ran slower in this session than during H6 and L5.
  - The IMDB rerun reproduces L5 exactly in every deterministic quantity: losses, gradient norms, PCG iterations and residuals, `|E_shared|`, and peak memory (1,324.0 MB).
  - Its wall-clock is about 2–3× L5's. Epoch 1: 3.12 s vs 1.15 s; train 2.25 s vs 0.74 s; draw 0.88 s vs 0.41 s.
  - The WISP draw runs on the CPU only and does not depend on the model. It is also about 1.9× slower than in H6 on ACM and DBLP.
- **Checked after the runs:** the laptop was on AC power (battery 100 %), the GPU was idle (P8, 0 %), and other desktop processes were running (iCloud, Steam, …).
- **Cause:** not established; thermal or power state, or CPU contention, are possible. Settling it needs v2 and v3-B timed back to back in one session.
- **Reading (DERIVED):**
  - Memory and PCG iterations are exact.
  - The time ratios compare this session against H6 numbers taken in a faster machine state, so they are likely overstated. On ACM and DBLP the train-step ratios (1.8×) are about the same as the 1.9× slowdown of the model-independent draw.
  - The ratios still meet the gate, so the pass does not depend on the slowdown; only the size of the ratio does.
  - The v1 time comparison (1.44×) carries the same caveat, and v1's definition of "time" was not checked.

### 7.5 Cost gate (spec §6: memory ≤ 3× v2, PCG ≤ 40 at tol 1e-4, time ≤ 4× v2)
| | ACM | DBLP | IMDB |
|---|---|---|---|
| memory (tol-1e-4 step) | 1.71× ✓ | 1.85× ✓ | 1.51× ✓ |
| PCG at tol 1e-4 (max of fwd, bwd, eval) | 9 ✓ | 11 ✓ | 10 ✓ |
| time (worse of the two epoch definitions in 7.3) | 1.79× ✓ | 1.90× ✓ | 2.64× ✓ (L5: 0.86×) |
| **gate** | **met** | **met** | **met** |

**Stop conditions:** none hit. PCG stayed far below T_max = 100, τ ≤ 1.02, max log-weight ≤ 0.25, and there was no OOM at r = 42 / 65 / 12.

**What C13 can say now:** memory and PCG iterations can be written as results. Time can be written as "within the gate" with the 7.4 caveat, or as a measured ratio after the paired timing of 7.6 (decision 1(b)).

**Observations (DERIVED, not gates):**
- **PCG iterations barely change across datasets:** 9–11 at tol 1e-4 and 6–8 at tol 1e-3, while the state grows from 84k to 318k rows. This is consistent with Prop. 10's size-free bound for the node system. It is only two points per tolerance, not the C5 slope test.
- **The eliminated share depends strongly on the dataset.** On ACM only 31 % of path facts keep a state, because at r = 42 most of them lie on a single cell; on DBLP 80 % do. DBLP's 85k fact states and 207k cells explain its higher memory ratio (1.85×).
  - Under author decision 4(a) the kept set stays `E_shared`.
  - Eliminating every fact would cut DBLP's state to 233k rows (DERIVED: 317,947 − 85,121). The memory saving was ~16 MB on IMDB (§4 item 4) and was not measured on DBLP.
- **Memory:** about 1.5–1.85× v2, and 0.30× v1 on ACM. At this `d`, this answers the reviewers' scalability concern on ACM (v1 used 9.3 GB).

### 7.6 DEVIATIONs and decisions for the author
- **DEVIATION L7-a:** the tol-1e-4 measurement is one forward + backward on a fresh draw (seed + 2) without an optimizer step, as the handoff prompt asked ("one extra forward + backward"); it is not a full epoch at tol 1e-4. Because the flag lives in the smoke script, the IMDB measurement also reran the 2 epochs. This gave IMDB a second, independent timing, which is what exposed 7.4.
- **DEVIATION L7-b:** this report's per-epoch peak memory includes the evaluation forward, as in L5; H6's peak covers the training step only. The evaluation runs without gradients and does not set the peak. The gate uses the tol-1e-4 step's peak, which covers the training step alone.
- **Decision 1 (timing):**
  - (a) Accept the gate on these numbers, with the 7.4 caveat in the report.
  - (b) **Recommended:** before a time ratio goes into the paper, run one paired timing (the v2 H6 configuration and v3-B back to back in one session, 2 epochs each on ACM and DBLP; about 15 min of runs).
  - (c) Defer timing to the server.
- **Decision 2:** merge of `v3b` (handoff §3 step 2). It was waiting for the ACM numbers, which are now available.

## 8. L8: r never changed, theory checks, mutation tests, part-j fast-forward (2026-11-11)
**Scope:** correctness only (author decision (3)). No training, no performance or timing runs, no accuracy numbers. Branch `v3b` from `948596a`. Commits: L8a `9fcd477`, L8b `164f7da`, L8c `38bf468`, and this report (`L8: report section 8`, the commit `part-j` points to after L8d). The spec on disk (v1.11, uncommitted, another session's) was read but not staged or edited.

### 8.0 Author decisions recorded (2026-11-11)
1. **§7.6 decision 1:** no paired v2/v3-B timing; v2 is no longer used. The C13 results from L7 are **memory** (1.51× / 1.71× / 1.85× v2 on IMDB / ACM / DBLP; 0.30× v1 on ACM) and **PCG iterations** (10 / 9 / 11 at tol 1e-4). Time against v1 and the baselines is measured later on the server. The time ratios in §7.3 are not C13 results.
2. **§7.6 decision 2:** merge approved as a fast-forward of `part-j` (8.4).
3. Only correctness tests in this unit.

### 8.1 L8a: `scripts/v3b_cost_smoke.py` never changes r (spec E11, E14)
**What changed (VERIFIED-CODE):**
- Removed: the OOM fallback that halved `r`, and the `--walks_per_anchor` option.
- `load_config` refuses a resolved `r` that is not the dataset's calibrated value when a table exists (`WISP_WALKS_PER_ANCHOR`).
- On `torch.cuda.OutOfMemoryError` the script writes `{config, dataset, walks_per_anchor, stopped, error}` to the JSON and exits with status 1.

**Checks (VERIFIED-RUN, `docs/v3B_evidence/l8a_checks.py`, from `H2S-NN/`; no GPU, no run):**

| Check | Result | PASS |
|---|---|---|
| the three v3b configs load | r = 12 / 42 / 65 | PASS |
| ACM config with r = 21 | rejected (`resolved r = 21 is not the calibrated value 42`) | PASS |
| `--walks_per_anchor 21` | argparse exit 2 (option gone) | PASS |
| simulated OOM (`run` raises `OutOfMemoryError`) | exit status 1, JSON written, `run` called once with r = 42 | PASS |
| `python -m unittest tests.test_v3b_model -v` | 2 tests OK (5.5 s) | PASS |

### 8.2 L8b: `tests/test_v3b_theory.py` (fp64, CPU, T2/T7 fixture)
**Fixture:** 12 nodes, 20 facts incl. one self-loop, connected, cycle rank 8; 6 cells of lengths 2, 2, 3, 3, 4, 4 with reversed hops; d = 4.
**Commands (VERIFIED-RUN):** `python -m unittest tests.test_v3b_theory -v` (5 tests OK, 25 s); numbers from `docs/v3B_evidence/l8b_numbers.py`.

| Test | What | Tolerance | Result | PASS |
|---|---|---|---|---|
| K1 (Thm 3) | flat sheaf: every fact map I (k = 2 equal reflectors, identity frames), the fixture's a_e and β_f | kernel = eigenvalues ≤ 1e-10; structure ≤ 1e-8 | dim ker L̂ = dim ker L = **4 = d**; largest kernel eigenvalue 4.2e-16, **next eigenvalue 0.0839**. Kernel vectors are constant sections (x_v = s_e = z_f = c): max deviation 4.0e-14 for ker L, 3.3e-14 for D^{-1/2} ker L̂ | PASS |
| K2 (Thm 3, control) | arm `free_cell_maps`: random orthogonal R_{e,f} per incidence on the same flat fact maps | dim ker = 0 | dim ker L̂ = **0**; smallest eigenvalue **0.0562** | PASS |
| K3 (Thm 3) | dim ker L̂ vs dim ker Δ₀, Δ₀ built independently as δᵀδ from the dense maps | equal | (a) fixture's random factorized maps: **0 = 0** (next eigenvalues 0.0472 / 0.312); (b) gauge of the flat sheaf, F_{v◁e} = G_e O_vᵀ with dense random orthogonal G_e, O_v: **4 = 4** (next 0.0839 / 0.569) | PASS |
| F1 (Thm 6 with Thm 4's maps) | 3 classes, random distinct unit vectors a_c; source role I, target role I or one reflection H(w) with H(w)a_q = a_p; **14 of 20 facts heterophilic**; τ = 3; all 6 cells | ‖x* − h₀‖/‖h₀‖ ≤ 1e-10 | unnormalised L, h₀ = a_y: dense 5.3e-15, PCG 4.4e-15 (48 iterations). L̂, h₀ = D_v^{1/2} a_y: dense 8.2e-16, PCG 7.5e-15 (31 iterations), PCG on the eliminated operator 1.2e-14 (21 iterations). ‖L Y_sec‖/‖Y_sec‖ = 8.6e-16 | PASS |
| U1 (unroll vs implicit) | T10's loss and parameterisation; autograd through `pcg` (tol 1e-12) vs `AnchoredSolve` (tol 1e-12) | ‖g_u − g_i‖/‖g_i‖ ≤ 1e-6 per group | 30 iterations each (fwd, bwd, unrolled). Fact scales 1.6e-11, reflectors 3.7e-11, rotations 4.6e-11, cell weights 1.9e-11, **θ_τ 8.4e-12**, **h₀ 5.9e-12** | PASS |

**DEVIATION L8b-a:** F1 uses `DenseMapOperator`, not `FactorizedMapOperator`, for its maps. With k = 2 the factorized family is a rotation times two reflections (det +1), so it cannot express a single reflection on some incidences next to I on others. The operator, transports and solver are the production code; only the map container differs. K3(b) uses dense maps for the same reason (arbitrary orthogonal G_e O_vᵀ).
**DEVIATION L8b-b:** the card gives no tolerance for "kernel vectors have x constant, s = x, z = x". I used 1e-8 (eigenvector accuracy is about eps/gap with gap 0.084); the measured deviation is 4e-14. Under L̂ the kernel is D^{1/2} ker L, so the constant structure is checked on D^{-1/2} ker L̂ and on ker L directly.

### 8.3 L8c: `tests/test_v3b_mutations.py`
**Method:** each mutation is monkeypatched inside the test only. The named test runs in a nested `unittest` run: it must pass unpatched and fail under the mutation, and the failure message is recorded.
**Commands (VERIFIED-RUN):** `docs/v3B_evidence/l8c_mutation_numbers.py` → 6 tests OK (196 s); output in `l8c_mutation_output.txt`. Unpatched: T1, K1, T2, U1 and T7 pass.

| Mutation | Named test (tolerance) | Failing number under the mutation | Caught |
|---|---|---|---|
| m1 forward (stored source-role) map on a reversed hop in `FactCellTransport`, no adjoint | T1, IMDB draw (1e-5) | functoriality residual **8.11** on 242 incidences | yes |
| m2 R_{e,f} → G_f R_{e,f}, G_f = exp(0.01·skew) per cell (UP, DOWN, apply consistent, so L stays symmetric) | T1 (1e-5); K1 (dim ker = d) | T1 **0.473**; flat kernel **dim 4 → 0** (smallest eigenvalue 3.8e-6) | yes |
| m3 node–cell term dropped inside `full` (`node_cell` forced False after init) | T2 (1e-12) | max \|L_mf − L_ref\| **7.98** | yes |
| m4 D_f = 2ℓ_f β_f (missing +1) | T2 (1e-12) | **2.00** (L and D vs the dense reference) | yes |
| m5 implicit backward without the τ factor | U1 (1e-6) | relative error **0.350** (= τ − 1, τ = e^0.3) in fact scales, reflectors, rotations and cell weights; **θ_τ gradient exactly 0**; h₀ unchanged (5.9e-12, as it should: ∂/∂h₀ = Pλ has no τ) | yes |
| m6 eliminated state weighted by D_e^{-1/2} instead of D_e^{-1} (wrong Schur weight) | T7 (1e-10) | **8.88** | yes |

**DEVIATION L8c-a (m4):** the request names "T2/T3 diagonal-block check". T2 fails; **T3 passes under m4** (recorded in the test, not asserted).
- **Reason (DERIVED, VERIFIED-CODE `sheaf_energy.py:434`):** `apply` takes the diagonal of L from the same `degree()`. A wrong D_f therefore enters L and its normalisation alike, and the diagonal blocks of L̂ stay I.
- **Consequence:** T3's block check tests the normalisation, not the degree formula. Only T2's independent dense reference checks D. A degree error that is consistent between `degree()` and `apply` is caught by T2 alone.
- Under m4 the operator is also no longer the Hessian of Eq. E, yet T3's PSD check passed on this fixture.

### 8.4 Full suite and L8d
**Full suite (VERIFIED-RUN):** `python -m unittest discover -s tests` after L8c, output in `docs/v3B_evidence/l8_final_suite_output.txt`.
- **Result:** Ran **497 tests** (486 + 5 theory + 6 mutation), FAILED (failures=11, errors=9), 2,217 s.
- **All 20 failures are in `test_baselines`,** and they are a subset of the 40 known environment failures (§5a): diffed against `final_suite_output.txt`, there is no failure that was not there before.
- **Changed since L5:** the 17 `test_heterophily_metrics` errors and 3 of the `test_baselines` errors now pass. The WinError 4551 imports of some vendor clones succeed today, so the Application Control state has changed again. Nothing in this unit touches those modules.
- **Run time:** 2,217 s vs 756 s at L5. It includes the 196 s mutation module and, presumably, the slow machine state of §7.4. No timing conclusions are drawn.

**L8d:** `git branch -f part-j <this report's commit>`: no checkout, no merge commit, working tree untouched. `part-j` (`c41bf28`) is an ancestor of `v3b` (checked with `git merge-base --is-ancestor`), so this is a fast-forward. The verification (`git log --oneline -1 part-j`) is given in the reply to the author; it cannot be written into the commit it names.

### 8.5 Decisions for the author
1. **T3 and the degree formula (8.3, m4).** Options:
   - (a) leave as is: T2 covers it;
   - (b) add an independent check of `degree()` against incidence counts (D_v, D_e, D_f from the fixture's lists) to T3 or to a new test;
   - (c) make `apply` compute its diagonal from the blocks it applies rather than from `degree()`. This would change the method's code, so it is not recommended without a reason beyond testing.

   **Recommendation (a):** T2 already compares D with an independent construction.
2. **Application Control state** changed between L5 and L8 (8.4). The heterophily-metric tests pass today, but the environment is not stable. Before computing heterophily metrics, rerun those tests on the day.

### 8.6 Time per sub-unit (agent wall-clock, approximate)
| Sub-unit | Time |
|---|---|
| Reading card, code, tests | ~4 min |
| L8a | ~5 min |
| L8b | ~5 min (tests 25 s) |
| L8c | ~10 min (two mutation runs of ~200 s; the first showed m4/T3, 8.3) |
| Full suite | 37 min |
| Report | ~10 min |
| **Total** | ≈ 1.2 h |
