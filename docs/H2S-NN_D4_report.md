# H2S-NN — Task card D4 report: v2 model with the vertex–2-cell sheaf Laplacian (incl. D1a)

| | |
|---|---|
| Date | 2026-11-06 |
| Task card | D4 (`H2S-NN_agent_prompt.md`, Part C), including the D1a amendment; plan `H2S-NN_v2_plan.md` §1.7, §4 |
| Branch | `D1matbounding_D4coupling` (local commits, **not pushed**) |
| Commits | `0eb7683` U1 D1a · `9eb2409` U2 config · `1ebef89` U3 operators · `8b6bc29` U4 model · `a137418` U5 ablations · `d85e593` U6 mutation check · `a1f2c62` U7 measurements · `7b12c2b` U8 docs (base `b0e0fab`) |
| Environment | conda `h2snn`, Python 3.12, torch 2.6.0+cu124, Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU (8.00 GiB) |
| Evidence | `docs/D4_evidence/` (scripts, raw outputs) |

Status tags as in the prompt: `VERIFIED-CODE` (file:line, paths relative to `H2S-NN/`, line numbers at `7b12c2b`), `VERIFIED-RUN`, `VERIFIED-SOURCE`, `DERIVED`, `UNVERIFIED`, `PROPOSAL`.

Author decisions applied (from the inventory review): A–H as recommended; F with the wording in §3 item 1; no independent sub-agent review (I); mutation check limited to the six required defects.

---

## 0. Is D4 complete?

**The implementation, the tests T1–T14, the six-defect mutation check, the ACM measurements and the smoke run are done. Two things are not as the card or plan wants them:**

1. **T6 as written in the card does not prove intermediate-node reach.** Its clause "∂x₁(A₁)/∂x₀(P) changes when γ changes" also holds through the degree normalization alone, so it passed under the endpoints-only mutation. I strengthened T6 (§3 item 2); the mutation is now caught by T6.
2. **The plan's Tier-0 memory criterion (≥ 10× lower peak than v1, §3.5) is not shown.** At d = 8, v2 factorized uses **5.5×** less peak memory than v1 dense and 2.7× less than v1 factorized. At d = 32, v1 does not fit this GPU at all, so only a lower bound (≥ 3.0×) exists. Not a D4 stop condition, but it is the plan's gate.

| Card requirement | Status |
|---|---|
| Unit 1, D1a per-edge scale with `map_scale` switch; D1 tests unchanged; per-edge versions of changed tests | Done |
| Config keys `architecture`, `gamma_init`, `eta_init`, `checkpointing`, ablation switches, validated | Done |
| v2 accepts `map_family ∈ {dense, factorized}`; `DenseMapOperator` closed-form inverse `Fᵀ/c²` | Done |
| Path-cell operator `L_P` + diagonal on `PathTransportOperator`, one per face group | Done |
| v2 module selected in `build_model`; trainer runs it unchanged | Done (no trainer change) |
| Deletions in v2 (edge cochain, `L₁`, `EdgeInitializer`, `δ₁`, `A_f`, face caches, closure edges, edge LayerNorm, edge pooling, ramp); no normalization on `x_K` | Done (T9) |
| Ablation hooks (7 switches) implemented and tested, not run | Done (T11) |
| LP / GC under v2 raise `NotImplementedError` naming D4 | Done |
| Tests T1–T14 | Done (T6 strengthened) |
| Mutation check (6 required defects) | Done, 0 vacuous |
| Measurements: ACM v2, factorized k ∈ {1, 2} at d = 32, dense at largest d, checkpointing on/off, params; smoke run | Done |
| Stop conditions | None triggered |
| Independent review | Skipped by author decision |

---

## 1. Summary

- `model.architecture: v2` builds `H2SNNv2`: `x ← x − η D^{-1/2}(Δ₀ + γL_P)D^{-1/2}x` over observed edges plus path 2-cells, readout `MLP([x_K, h₀])`. D1a (`map_scale: per_edge`, one `c_e` per edge) makes every transport exactly orthogonal. v1 is bit-for-bit unchanged.
- **Tests:** 44 new tests pass; D1's 32 pass unchanged. All 6 required mutations are caught (0 vacuous) after strengthening T6. Full suite: 284 tests, the same 24 pre-existing `test_baselines` failures as before, none new.
- **Memory (ACM, one forward + backward):** v2 factorized k = 2 at d = 32 fits at **2.519 GiB** (v1 does not fit at d = 32). At d = 8: 1.111 GiB vs v1 dense 6.082 GiB (5.5×), below the plan's 10× target.
- **Smoke run (20 epochs, ACM, d = 32):** stable; η·λ_max 0.914 → 0.960, loss finite. η and γ barely moved.
- **Not done:** accuracy experiments (need approval).

---

## 2. Files changed

| File | Why |
|---|---|
| `src/configs/config.py` | `architecture`, `map_scale` (None → `per_edge` for v2, `per_side` for v1), `gamma_init`, `eta_init`, `checkpointing`, `path_maps`, `path_mode`, `uniform_path_gate`, `ablate_no_path_cells`, `readout`; `_validate_v2_keys` (`:789`): v1 rejects every non-default v2 key, v2 rejects v1-only switches and conflicting ablations |
| `src/models/map_operators.py` | `checkpoint_if_grad(enabled=)`; `FactorizedMapOperator.apply_orthogonal(_T)` (`:243`); `DenseMapOperator` with scale, closed-form inverse, `select`, `[E,n,d]` (`:283`); `PathTransportOperator(shared_scale=)` scale-free path (`:439`, `:449`), `squared_scales` (`:461`), `apply_holonomy(_T)` (`:478`) |
| `src/models/sheaf_maps.py` | `FactorizedNodeToEdgeSheaf(map_scale=, checkpoint=)`: per-edge `edge_scale_mlp([h_u, h_v, rel_emb])` (`:390`, `:444`, `:496`), one `[M]` tensor shared by both operators; new `DenseConformalNodeToEdgeSheaf` (`:536`, decision C) |
| `src/models/coboundary_ops.py` | `L0_matrix_free(checkpoint=True)` |
| `src/models/vertex_path_laplacian.py` (new) | `PathCellTerm` (`:81`), `FreeIncidenceOperator` (`:44`), `EdgePairTerm` (`:116`), `HolonomyEdgeTerm` (`:144`), `VertexPathSheafLaplacian` (`:187`: `edge_degree`, `degree`, `apply`, `apply_path`, `apply_normalized`) |
| `src/models/h2snn_v2.py` (new) | `H2SNNv2` (`:245`), `PathEncoder`, `EndpointEncoder`, `EndpointEdgeMaps`, `FreeIncidenceMaps`, `H2SNNv2Output`, `V2Operator`; `build_operator` (`:427`) is used by `forward` and by the tests |
| `src/models/h2snn_model.py` | `build_model` dispatches on `architecture` (`:885`); `H2SNN` refuses non-v1 configs naming D4 (`:172`); passes `map_scale` |
| `tests/test_v2_model.py` (new) | 44 tests: T1–T14, controls, config, scope, trainer |
| `CLAUDE.md`, `AGENTS.md` | State columns (D1a, D4), v2 code paths, planning documents now at `../docs/` |
| `docs/D4_evidence/` | inventory check, probes, mutation check, measurement and smoke scripts with outputs |

`tests/test_factorized_maps.py` is **unchanged** (`git diff b0e0fab HEAD -- H2S-NN/tests/test_factorized_maps.py` is empty).

---

## 3. Deviations from the spec

1. **`D` is the per-incidence sum, not exactly the block diagonal of `L`** (decision F). ACM has **16 self-loops** (8 `paper__cite__paper`, 8 `paper__ref__paper`; VERIFIED-RUN, `acm_graph_facts.txt`). A self-loop's diagonal block is `(F_tgt − F_src)ᵀ(F_tgt − F_src)`, not a scalar. `D_v` counts both incidences, `c_src² + c_tgt²` (`vertex_path_laplacian.py:219`). This is Prop. 3's `D_e`, so `L ≼ 2D` still holds (DERIVED: `(a − b)² ≤ 2a² + 2b²`). The test `test_D_is_the_per_incidence_degree_of_the_operators_own_terms` (`:558`) asserts equality with the dense block diagonal at every other node and with "block without the loop + c_src² + c_tgt²" at the loop node.
2. **T6 strengthened.** The card clause "∂x₁(A₁)/∂x₀(P) changes when γ changes" is satisfied through `D_{A₁}^{-1/2} D_P^{-1/2}` alone, because γ enters `D`. The mutation check showed it (§4.4). T6 now also requires the path term's own Jacobian blocks `A₁ ← P` and `P ← A₁` to be > 1e-8 (`tests/test_v2_model.py:776`). Nothing was loosened.
3. **T8 fixture constraint.** The card's size rule (`≥ min(M, F)·d²/4` elements) would flag ordinary parameters when F ≪ M (e.g. the 22,528-element path-encoder weight). The fixture uses F = 540 ≈ 0.9 M (M = 600), d = 32; the largest legitimate tensor is 0.56 of the limit (§4.3).
4. **T5 scales.** `c_e ∈ {ε, 50.1}` (ε = 0.1; 50.1 = `softplus(50) + ε`, the T12 upper end). 40 fixtures per-edge, 20 per-side.
5. **Choices the card leaves open** (decisions A–E, G, as approved):
   - AMPSS auto under v2 raises `NotImplementedError` naming D2 (`h2snn_v2.py:256`).
   - Gate `g`: zero bias, default weight init (`:345`); measured `w_f` at init: mean 0.507, max 0.64 (§4.3).
   - v2 dense family: `DenseConformalNodeToEdgeSheaf`, FastHPP `Q` per edge/side from v1-style per-relation MLPs, scale by the factorized law. v1's `NodeToEdgeSheaf` is untouched.
   - `free`: orthogonal `R_position H_k(u)` from `[h_v, latent_f, onehot(position)]` (scale 1); requires `per_edge`. `endpoint_edges`: maps and gate from an endpoint-only latent `[h_{v₀}, h_{v_L}, schema]`. All path modes weight their term by `γ·w_f`.
   - v1 rejects every non-default v2 key, including `checkpointing: false` and `map_scale: per_edge`.
   - Defaults `eta_init = 0.5`, `gamma_init = 1.0`.
6. **v1 numeric keys silently ignored under v2:** `eta_0`, `eta_1`, `d_h`, `edge_init_hidden_dim`, `use_delta0_init`, `face_chunk_size`, `data.closure_scale`. Only switches that select deleted machinery are rejected.
7. **Relation embedding** has `|R|` rows in v2 (v1 had one extra closure row).
8. **The type encoder keeps its per-type LayerNorm on `h₀`** ("TypeEncoder unchanged from v1"); no normalization is applied to `x_K`.
9. **Path latent memory.** The path encoder is checkpointed; the `[P, hidden]` latent itself is kept for the gate's backward (≈ 86 MB on ACM, DERIVED 168,524 × 128 × 4 B).

---

## 4. Verification evidence

Commands from `H2S-NN/` with `C:\Users\Acer\anaconda3\envs\h2snn\python.exe` (`AGENTS.md:34`). Two overlapping `conda run` calls failed at activation once, so the env Python is called directly.

### 4.1 Suites (VERIFIED-RUN)

| Run | Command | Decisive output |
|---|---|---|
| Baseline (U0, `b0e0fab`) | `-m unittest discover -s tests -v` | `Ran 240 tests` / `FAILED (failures=11, errors=13)` |
| Final (`7b12c2b`) | same | `Ran 284 tests` / `FAILED (failures=11, errors=13)` |
| Failing names, baseline vs final | `Compare-Object baseline_fail_names.txt final_fail_names.txt` | no difference; all 24 in `test_baselines` |
| D1 + D4 tests (U6 state) | `-m unittest tests.test_v2_model tests.test_factorized_maps` | `Ran 76 tests` / `OK` |
| v1 bit-for-bit, after U0, U1, U3, U4, U5 | `..\docs\D1_evidence\t11_regression.py compare ..\docs\D1_evidence\t11_reference.pt` | `compared 912 entries across 9 cases; mismatches=0` (each time) |

### 4.2 Required tests (`tests/test_v2_model.py:line`)

| ID | Test | What it asserts | Result |
|---|---|---|---|
| T1 | t11 compare; D1 golden `test_T11_dense_matches_pre_change_golden_outputs` | bit-for-bit vs pre-D1 snapshot; ≤ 1e-6 vs pre-change values | 0/912; pass |
| T2 | `:524` | `L_P` and full `L` equal an independent dense Kron-reduced `Σ w_f Bᵀ(I − 11ᵀ/\|f\|)B` built from `torch.linalg.inv` transports; L ∈ {2, 3, 4} mixed; per-edge and per-side | pass; worst error 2.56e-15 |
| T3 | `:539` | `\|⟨x,Ly⟩ − ⟨Lx,y⟩\| ≤ 1e-10`, `‖M − Mᵀ‖ ≤ 1e-10`, min eig ≥ −1e-12, `⟨x,Lx⟩ ≥ −1e-12‖x‖²`, normalized operator, c ∈ {ε, 50} | pass |
| T4 | `:675`, control `:692` | Tree + root-ward chains (L = 2, 3), c ∈ {ε, 50}: explicit sections give `Δ₀K = 0` and `L_P K = 0` (≤ 1e-10); `dim ker Δ₀ = d = dim ker(Δ₀ + γL_P)` at threshold 1e-9. Control, free maps: kernel < d | pass; kernel eigenvalues ≤ 1.34e-15, smallest outside ≥ 8.3e-7; free kernel = 0 |
| T5 | `:582` | 60 fixtures: `λ_max(D^{-1/2}LD^{-1/2}) ≤ 2 + 1e-9`; isolated node has zero rows; control: unnormalized `λ_max > 2` | pass; max normalized 1.99999014, unnormalized up to 2.4e16 (60/60 > 2) |
| D | `:558` | `D_v I` equals the dense diagonal block (self-loop rule in §3 item 1), per-edge and per-side | pass |
| T6 | `:776` | One step, maps fixed: `(A₁, A₂)` block exactly 0 at γ = 0 and > 1e-8 at γ ∈ {0.3, 2}; `(A₁, P)` block changes with γ; path-term blocks `A₁ ← P`, `P ← A₁` > 1e-8; `ablate_no_path_cells` gives exactly 0 | pass; `(A₁,A₂)`: 0, 1.95e-2, 7.91e-2; `(A₁,P)`: 0.325, 0.310, 0.270 |
| T7 | `:606` | `z_f` = least-squares solution = mean (≤ 1e-10), per-edge and per-side | pass |
| T8 | `:826` | dispatch-mode detector over forward + backward: no `[M\|F\|F_g\|F_g(L+1), d, d]` and nothing ≥ `min(M,F)d²/4`; v1 dense **and** v1 factorized flagged in both passes | pass |
| T9 | `:869` | no `edge_initializer`, `edge_to_face`, `map_head`, `edge_norm`, `node_norm` keys; no `EdgeInitializer` / `SequenceEdgeToFaceSheaf`; LayerNorm only in `node_encoder`; Δ₀ has exactly M edges; no `[M+F, d]` tensor (v1 control has one) | pass |
| T10 | `:903` | non-zero gradients for θ_η, θ_γ, gate, path encoder, reflector MLP, edge-scale MLP, `R_ψ`, encoder, relation embeddings, readout | pass |
| T11 | `:1075`–`:1222` | one test per switch, each with a control (no path cells, uniform gate, endpoint edges, holonomy edges vs dense closure edge, free maps, per-side, h₀-only); every switch trains one finite step | pass |
| T12 | `:136`–`:229` | (a) `src.scale is tgt.scale`; singular values of both maps = `c_e` (≤ 1e-10). (b) edge transports and `T_v` orthogonal for L ∈ {2, 3, 4}, float32 at L = 4 with c ∈ [ε, 50]; scale-free path bit-identical under any shared scale, equals D1 path and `linalg.inv` product. (c) `c(−50) = ε`, `c(+50) = 50.1`, gradients finite. Control: per-side breaks (b) | pass; orthogonality 2.4e-15–3.0e-15 (f64), ≤ 9.5e-7 (f32); per-side control 1.4e8–8.8e9 |
| T13 | `:927` | factorized v2 vs the same model with materialized maps through `DenseMapOperator`: logits and every gradient ≤ 1e-10 (per-edge and per-side); dense family runs, gradients reach its MLPs | pass |
| T14 | `:706` | on a graph with cycles: per-side `s_v·c_e` gives the same normalized Δ₀ (≤ 1e-10); generic per-side scales differ (> 1e-3) | pass |

Also: `test_checkpointing_switch_does_not_change_outputs_or_gradients` (`:971`, ≤ 1e-10); `test_trainer_runs_v2_unchanged_and_checkpoints_reload` (`:998`: `Trainer.fit` two epochs, reload from saved `config.yaml` + `best.pt`); `test_v2_rejects_ampss_auto_discovery` (`:985`); config tests (`:1239`–`:1303`).

**Per-edge versions of D1 tests, old vs new assertion (not weaker):**

| D1 test | D1 assertion (unchanged, still run) | Per-edge version |
|---|---|---|
| T10 | `floats == 2·M·(k·d+1) + 2·\|R\|·d²` exactly; ratio ≥ 10 at d = 32, k = 2 | `floats == 2·M·k·d + M + 2·\|R\|·d²` exactly (unique storage); ratio ≥ 10 at d = 32, k = 2 (`:258`) |
| T4 | side gap > 1e-3; perturbed maps change > 1e-3; non-incident maps exactly unchanged | same thresholds (`:286`) |
| T5 | deviation > 0.1, spread > 0.05; tied control < 1e-10 | same, plus holonomy orthogonal ≤ 1e-10 (`:305`) |

### 4.3 Decisive numbers (VERIFIED-RUN)

- `u1_probe_output.txt`: transport orthogonality per-edge 2.4e-15 – 3.0e-15 (f64), 6.0e-7 – 9.5e-7 (f32); per-side 1.45e8 – 8.8e9. Teeth checks: scaled-maps-in-fast-path caught (1/1), per-side copy of `c_e` caught (2/2).
- `u3_probe_output.txt`: T2, T4 and T5 numbers as in §4.2.
- `u4_probe_output.txt`: T8 fixture M = 600, F = 540 (270 + 270), limit 138,240 elements; largest v2 tensor `[600, 128]` (0.56 of the limit); largest parameter 22,528. Gate init mean 0.5071, max 0.6415. T6 numbers as in §4.2.
- ACM face groups (inventory item 2, `acm_face_check_output.txt`): PAP 29,210 and PSP 139,314 faces, L = 2. Orientation OK, simple, relation sequence constant (`[2, 3]`, `[4, 5]`), closure = (v_L, v₀). 0 nodes without observed edges.

### 4.4 Mutation check (VERIFIED-RUN, `mutation_check_d4_output.txt`)

`mutation_check_d4.py` monkeypatches one defect at a time (no file edited) and runs all 44 tests of `tests.test_v2_model`. Unmutated: 0 failing. **`mutations=6 vacuous=0`.**

| Mutation | Tests that fail |
|---|---|
| 1 Wrong transport order (`T_i = Q_i…Q_1`, adjoint consistent) | T2, T4 (sections), T12b (scale-free path vs dense) |
| 2 `z_f` without `1/\|f\|` | T2, T3, T4, T7 |
| 3 `D` not operator-consistent (edge part only, path part omitted) | D test, T5, T3 (incidental, see below) |
| 4 `w_f` not applied in `D` | D test only |
| 5 Path term on endpoints only | T2, T6, T11 endpoint test |
| 6 `per_edge` with different scales on the two sides (target ×1.001) | T12a, T10′, T5 per-edge, T12b; plus every v2 forward, because `PathTransportOperator(shared_scale=True)` raises (20 tests) |

Notes:
- **Mutation 5 initially escaped T6.** It was caught only by T2 and T11. The diagnosis and fix are in §3 item 2; the rerun shows T6 failing.
- **Mutation 4 is caught only by the D test.** Omitting `w_f < 1` makes `D` larger, so `λ_max ≤ 2` still holds; T5 cannot detect it by construction (DERIVED).
- **The T3 failure under mutation 3 is incidental.** With an inconsistent `D`, normalized entries reach ~1e21 under per-side scales, and T3's absolute 1e-10 symmetry check fails on roundoff.

---

## 5. Measurements

**Setup.**
- Script: `docs/D4_evidence/measure_d4.py <family> <k> <d> <checkpointing>`, one process per configuration.
- ACM from `configs/experiments/acm_h2snn.yaml` with `architecture: v2` and the overrides shown (K = 4, d_r = 16, hidden 128).
- One warm-up, then 5 timed forward + cross-entropy backward passes. Peak is `max_memory_allocated` after `reset_peak_memory_stats`.
- Capped with `set_per_process_memory_fraction(0.95)` = 7.60 GiB, as in D1.
- RTX 4060 Laptop 8 GiB. ACM: N = 10,942, M = 547,872, 168,524 path cells.

**Pre-registered (before running):** factorized k = 2 at d = 32 peaks at 2.5–4.0 GiB (wrong if > 5 GiB or out of memory); checkpointing off at d = 32 runs out of memory; dense fits at d = 8 (≈ 3.5–4 GiB) and runs out of memory at d = 16.

### 5.1 Peak memory, time, parameters (VERIFIED-RUN, `measure_d4.jsonl`)

| v2 configuration | Peak | Time (ms) | Params (node→edge) |
|---|---|---|---|
| factorized k = 2, d = 32, ckpt on | **2.519 GiB** | 1774.3 ± 2.8 | 306,727 (37,313) |
| factorized k = 1, d = 32, ckpt on | 1.827 GiB | 1238.1 ± 2.8 | 302,599 (33,185) |
| factorized k = 2, d = 32, ckpt off | out of memory (7.44 GiB at failure) | — | |
| factorized k = 1, d = 32, ckpt off | out of memory (7.44 GiB) | — | |
| factorized k = 2, d = 16, ckpt on / off | 1.427 / 7.248 GiB | 1017.9 ± 3.4 / 2613.9 ± 69.2 | 156,183 (18,977) |
| factorized k = 2, d = 8, ckpt on / off | 1.111 / 4.731 GiB | 652.4 ± 10.3 / 558.5 ± 7.7 | 82,447 (11,345) |
| dense, d = 8, ckpt on / off | 3.895 / 5.517 GiB | 2439.3 ± 4.1 / 2341.8 ± 5.4 | 200,831 (129,729) |
| dense, d = 16 | out of memory (6.65 GiB at failure) | — | 491,639 (354,433) |

All pre-registered expectations held. The k = 2, d = 32 peak (2.519) sits at the low end of the predicted band.

### 5.2 Against the D1 report's v1 numbers (same d, full model)

| d | v1 dense | v1 factorized k = 2 | v2 factorized k = 2 | v2 vs v1 dense | v2 vs v1 factorized |
|---|---|---|---|---|---|
| 32 | out of memory (5.89 GiB at failure) | out of memory (7.54 GiB) | 2.519 GiB | v1 does not fit; ≥ 3.0× (lower bound, 7.60 / 2.519) | ≥ 3.0× (lower bound) |
| 16 | out of memory (7.36 GiB) | 6.550 GiB, 5953 ms | 1.427 GiB, 1018 ms | — | 4.6× memory, 5.8× time |
| 8 | 6.082 GiB, 3177 ms | 3.004 GiB, 1729 ms | 1.111 GiB, 652 ms | **5.5×** memory, 4.9× time | 2.7× memory, 2.7× time |

Parameters at d = 32: v1 dense 1,615,909, v1 factorized k = 2 461,782 (D1 report), v2 factorized k = 2 **306,727**. Ratios DERIVED from the tables.

**Reading.**
- v2 removes the v1 face/edge machinery that dominated D1's peak. d = 32 now fits in 2.5 GiB.
- The plan's Tier-0 target "≥ 10× lower peak memory" (§3.5) is **not shown**: 5.5× at d = 8 against v1 dense. At d = 32 the v1 baseline cannot be measured on 8 GiB.
- Checkpointing costs ≈ 17% time at d = 8 (652 vs 559 ms) and saves 3.6 GiB. The off-run at d = 16 was slower than the on-run; its cause is UNVERIFIED (it ran within 0.35 GiB of the cap).

### 5.3 Smoke stability run (VERIFIED-RUN, `smoke_d4.jsonl`)

`smoke_d4.py`: ACM, seed 42, 20 epochs, factorized k = 2, d = 32, AdamW lr 1e-3, cosine schedule, grad clip 1.0 (the trainer's step). λ_max is from 100 power iterations on `D^{-1/2}LD^{-1/2}` in eval mode. **No accuracy computed or claimed.**

| Epoch | Train loss | η | γ | w_f mean / max | max c | λ_max | η·λ_max |
|---|---|---|---|---|---|---|---|
| 0 | 1.1000 | 0.5002 | 0.9994 | 0.566 / 0.678 | 1.04 | 1.8264 | 0.9137 |
| 5 | 0.8477 | 0.4997 | 0.9977 | 0.453 / 0.688 | 1.32 | 1.8637 | 0.9313 |
| 10 | 0.5945 | 0.4989 | 0.9959 | 0.315 / 0.715 | 1.67 | 1.9070 | 0.9515 |
| 15 | 0.4735 | 0.4986 | 0.9950 | 0.260 / 0.731 | 1.86 | 1.9231 | 0.9588 |
| 19 | 0.4484 | 0.4986 | 0.9949 | 0.247 / 0.740 | 1.90 | 1.9250 | 0.9597 |

- No divergence: loss finite and η·λ_max < 2 at every epoch.
- `max c` grows (1.04 → 1.91) without an upper bound, as specified.
- **η and γ barely move** in 20 epochs at lr 1e-3. The mean gate falls from 0.57 to 0.25.

---

## 6. Self-review checklist (D4)

| Item | Status | Evidence |
|---|---|---|
| Path term reaches **intermediate** nodes (T6); endpoints-only mutation caught | PASS after a fix | Initially FAIL: T6 as specified passed under mutation 5. T6 strengthened (§3 item 2); mutation 5 now fails T6, T2, T11 |
| `D` built from exactly the operator's own terms (same `w_f`, γ, c); T5 with negative control | PASS | D test (`:558`) at every node; T5 60/60 ≤ 2 with unnormalized control 60/60 > 2; mutations 3 and 4 caught |
| No `[M,d,d]`, `[F,d,d]`, `[F,L+1,d,d]` in forward **or** backward (T8, dense control) | PASS | T8 with v1 dense and v1 factorized flagged; largest v2 tensor 0.56 of the size limit |
| Nothing from the deleted list exists in v2 (T9) | PASS | T9: state_dict keys, module types, Δ₀ edge count = M, no `[M+F, d]` (v1 control has it) |
| Sections preserved with transport, destroyed with free maps (T4) | PASS | kernel d = d with transport; 0 with free maps; spectral gap ≥ 8.3e-7 vs kernel ≤ 1.3e-15 |
| v1 bit-for-bit unchanged (T1) | PASS | t11 compare 0/912 after every code unit; D1 golden test passes |
| D1a: both sides share `c_e`; transports orthogonal (T12 with per-side control); D1 tests not weaker; no upper bound on `c` | PASS | T12a `is` identity + SVD; T12b with per-side control 1e8–1e9; D1 file unchanged (§2) with assertions in §4.2; `c(+50) = 50.1` asserted (`:229`) |
| Every ablation switch tested (T11), none reachable from v1 | PASS | T11 tests (`:1075`–`:1222`); `test_no_v2_key_is_reachable_from_v1` (`:1273`) |
| Every new config key validated; no hard-coded hyperparameters | PASS, with notes | `_validate_v2_keys` (`config.py:789`). Kept guards: `REFLECTOR_NORM_EPS = 1e-6` (D1); `D_v = 0 → 0` via a safe `rsqrt` (no epsilon). Hard-coded initializations: rotation std `1/√d` for the free and endpoint maps (D1 precedent); gate bias 0 |
| Each test would fail if its property were broken (mutation check, 0 vacuous) | PASS | §4.4: 6/6 caught; negative controls inside T4, T5, T8, T9, T11, T12, T14 |
| (Not in the card) dtype/device | partial | float64 unit tests; CUDA float32 exercised only by the ACM measurements and smoke run, not by a v2 unit test |

---

## 7. Open items

**UNVERIFIED (each with the check that would resolve it):**

1. **Accuracy.** Tier-0 v2 vs v1, the §4.6 ablations, dense vs factorized k ∈ {1, 2}, per-edge vs per-side. Needs approval; 10 seeds per plan.
2. **Tier-0 ≥ 10× peak memory.** Check: measure v1 dense at d = 32 on a GPU with ≥ 16 GiB, or accept the d = 8 comparison (5.5×).
3. **Whether η and γ learn at all at lr 1e-3.** They moved < 0.01 in 20 epochs. Check: a longer run logging θ_η, θ_γ and their gradients.
4. **Cause of the slow checkpointing-off run at d = 16** (2.6 s vs 1.0 s on). Check: rerun at a lower memory fraction or with allocator statistics.
5. **CUDA nondeterminism of `index_add_` under checkpoint replay** (carried from D1).

**Out-of-scope problems noticed (not fixed):**

- **`src/analysis/diffusion_trajectory.py`** calls `model.encode(...)`, which v2 does not have.
- **W&B resume.** The new config keys change the base fingerprint again (`scripts/run_h2snn_wandb_tuning.py`); existing sweep states will not resume.
- **Memory guard.** `src/utils/memory_guard.py` still has no model for the factorized family; the v2 dense family reuses the v1 guard.
- **Plan §1.2:51** still says "sequential/Horner" for path transports; §4.2 was corrected.
- **24 pre-existing failures** in `tests/test_baselines.py`.
- **Untracked local files:**
  - the data cache `docs/D4_evidence/acm_prepared.pt` (≈ 80 MB);
  - two helper lists, `baseline_fail_ids.txt` and `d1_final_fail_ids.txt`;
  - `docs/D1_evidence/__pycache__/`.
- **Uncommitted author edits** in `docs/H2S-NN_agent_prompt.md` and `docs/H2S-NN_v2_plan.md` (not mine; left unstaged).

---

## 8. Decisions needed from the author

1. **Tier-0 memory gate.** Accept "v2 fits d = 32 in 2.5 GiB where v1 does not, 5.5× at d = 8" in place of "≥ 10×", or measure v1 at d = 32 on a larger GPU?
2. **Accuracy experiments.** Approve Tier 0 (DBLP, ACM, IMDB; 10 seeds) with the §4.6 ablations. Dense fits only at d = 8 on this GPU, so the dense vs factorized row must run at d = 8 here or at d = 32 elsewhere.
3. **η/γ learning rate.** Keep the shared lr, or add a parameter group for the scalars? (Experiment-phase change; not implemented.)
4. **v1 keys ignored under v2** (§3 item 6). Reject them, or leave them ignored?
5. **Card text.** Update T6 in the card to include the path-term Jacobian clause, so future runs don't rely on the weaker form.
6. **Git.** Push `D1matbounding_D4coupling` (9 commits including this report)?

---

## Appendix A — Pre-step inventory (VERIFIED-CODE at `b0e0fab`)

| Consumer | Reads from the model | v2 decision |
|---|---|---|
| `scripts/train_h2snn.py:55-57`, `run_h2snn_runs.py:146`, `run_h2snn_wandb_tuning.py:529` → `src/builders.py:155-179` | `build_model` | dispatch on `architecture` |
| `trainers/trainer.py:133,148,159` | `model(data, epoch)`, `output.logits` | provided |
| `trainers/trainer.py:181-189` → `ampss_metrics.py:21-24` | instance-weight stats, `operator_ramp` (only with AMPSS auto) | v2 rejects AMPSS auto (decision A) |
| `analysis/model_loading.py:23,35,45` | config.yaml → `build_model` → `load_state_dict` | provided (tested: `:998`) |
| `scripts/profile_model.py:41`, `analysis/profiling.py:13-19,30,42` | `named_children`, `output.logits` | provided |
| `analysis/training_separability.py:266,279,300` | `output.x_star`, `output.logits` | provided (`x_star = x_K`) |
| `analysis/diffusion_trajectory.py:16-28` | `model.encode`, `x_trajectory` | not provided (out of scope) |
| `H2SNNOutput` `s_star`, `delta0_*`, `raw_face_curl`, `delta1_delta0_h0` | tests and t11 script only (v1) | not provided |
| LP / GC trainers | `H2SNN.encode` | `H2SNN` raises naming D4 for non-v1 configs |

ACM face check (item 2): §4.3.

## Appendix B — Work units: pre-registration and outcome

| Unit | Pre-registered (before running) | Wrong if | Outcome |
|---|---|---|---|
| U0 | Suite = 24 baseline failures; t11 0 mismatches; D1 32 pass | any difference | 240 tests, 11 F + 13 E (same names as D1 final); 0/912; 32 OK |
| U1 D1a | T10′ exact; T12 a/b/c ~1e-15 f64, ≤ 1e-5 f32; fast path scale-invariant; per-side control fails | any failure; D1 file edited | 45 OK; numbers §4.3; D1 file unchanged; t11 0/912 |
| U2 Config | validation table | a listed case accepted, or a default rejected | 48 OK |
| U3 Operators | T2/T3/T7/T14 ≤ 1e-10; T4 threshold 1e-9; T5 ≤ 2 + 1e-9 with control; D check | T3/T4/T5 fail after 3 attempts → stop | 58 OK on first run; t11 0/912 |
| U4 Model | T6, T8 (v1 flagged), T9, T10, T13 ≤ 1e-10; trainer unchanged | any failure or trainer edit | 68 OK; trainer untouched; t11 0/912 |
| U5 Ablations | one T11 row per switch with control | any switch fails its row | 76 OK; one vacuity risk found and closed (endpoint term non-zero assert) |
| U6 Mutations | 6 caught; expected catchers listed | any vacuous | 6/6 caught; **T6 missed mutation 5** → T6 strengthened → caught |
| U7 Measurements | factorized k = 2, d = 32: 2.5–4.0 GiB; ckpt off d = 32 out of memory; dense d = 8 fits, d = 16 out of memory | > 5 GiB or out of memory at k = 2, d = 32 | 2.519 GiB; all expectations held; smoke stable |
| U8 Close | full suite same 24 failures | any new failure | 284 tests, same 24 |

## Appendix C — Reproducing the evidence

From `H2S-NN/`, with `P=C:\Users\Acer\anaconda3\envs\h2snn\python.exe` and `E=..\docs\D4_evidence`:

```
%P% -m unittest tests.test_v2_model tests.test_factorized_maps
%P% -m unittest discover -s tests -v
%P% ..\docs\D1_evidence\t11_regression.py compare ..\docs\D1_evidence\t11_reference.pt
%P% %E%\acm_face_check.py
%P% %E%\u1_probe.py & %P% %E%\u3_probe.py & %P% %E%\u4_probe.py
%P% %E%\mutation_check_d4.py
%P% %E%\measure_d4.py factorized 2 32 1
%P% %E%\smoke_d4.py
```

`acm_face_check.py` and `measure_d4.py` cache the prepared ACM data as `acm_prepared.pt` next to themselves (not committed).
