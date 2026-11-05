# H2S-NN — Task card D1 report: factorized restriction maps

| | |
|---|---|
| Date | 2026-11-06 |
| Task card | D1 (`H2S-NN_agent_prompt.md`, Part B); plan `H2S-NN_v2_plan.md` §1 |
| Branch | `d1-factorized-maps` (local, **not pushed**) |
| Commits | `dd42b8e` implementation · `11ba6c3` activation memory · `924be78` review fixes (base `0172044`) |
| Environment | conda `h2snn`, Python 3.12, torch 2.6.0+cu124, Windows 11, NVIDIA GeForce RTX 4060 Laptop GPU (8.00 GiB) |
| Evidence files | `docs/D1_evidence/` (scripts, raw outputs, pre-change snapshot) |

Status tags as in the prompt: `VERIFIED-CODE` (file:line, paths relative to `H2S-NN/`), `VERIFIED-RUN` (command + output), `VERIFIED-SOURCE`, `DERIVED`, `UNVERIFIED`, `PROPOSAL`. All line numbers refer to commit `924be78`.

---

## 0. Is D1 complete?

**Partly. The implementation, the four required properties, tests T1–T11, the out-of-scope guards and the self-review checklist are done and verified. Three parts are not complete as the card specifies them:**

1. **The required ACM measurement at the existing config is incomplete.** At `d = 32` (`configs/experiments/acm_h2snn.yaml`) one full forward + backward runs out of memory on this 8 GB GPU for **every** variant: dense and factorized k = 1, 2, 4. I could measure only:
   - the map stage at d = 32 (factorized only; dense also runs out of memory), and
   - both families end to end at d = 8 and d = 16, as a substitute.
2. **T11's second clause, "the full existing suite passes", is not true literally.** The suite had 24 failing tests before my changes and has the same 24 after. All are in `tests/test_baselines.py`: missing vendor clones and data, notebooks with execution counts. None is caused by D1, but the suite does not pass.
3. **The independent review covered commit `11ba6c3`, not the final `924be78`.** The fixes in `924be78` were verified by me (tests, snapshot comparison, mutation check, full suite), not re-reviewed by an independent agent.

| Card requirement | Status |
|---|---|
| Map module producing a map *operator* (apply, apply-transpose, inverse, scale) | Done |
| Config `model.map_family` (default `dense`), `model.householder_k` (default 2), validated | Done (+ one extra key, §3) |
| Matrix-free `δ₀`, `δ₀ᵀ`, `L₀` for the factorized family | Done |
| Matrix-free path-transport operator (`T_v x_v` and adjoint, gauge `A_f = I`) | Done |
| Node-classification integration; observed edges matrix-free; face chunks may materialize gathered path edges | Done |
| Out-of-scope paths raise `NotImplementedError` naming D1 | Done |
| Property 1 Invertible | Done, except inside a ball of radius 1e-6 (§3, item 5) |
| Property 2 Side-specific and node-dependent | Done |
| Property 3 Never `[M, d, d]` in forward or backward | Done (§4) |
| Property 4 Conformal-orthogonal | Done, same exception as property 1 |
| Tests T1–T10 | Done |
| T11 (v1 outputs equal pre-change outputs; full suite passes) | First clause done (bit-for-bit). Second clause not met (pre-existing failures) |
| Measurements on ACM, existing config, dense and k ∈ {1, 2, 4} | **Incomplete** (out of memory on this GPU; substitute measurements in §5) |
| Stop conditions | None triggered (k = 2 map storage 15.75× ≥ 10×) |
| Final report + independent review | Done (review on `11ba6c3`) |

---

## 1. Summary

- `model.map_family: factorized` implements `F = c·R_{ψ,side}·H_k(u)` as matrix-free operators for node classification. The default `dense` (v1) is bit-for-bit unchanged.
- **Tests:** 32 new tests pass. 15 injected defects all make their target tests fail (none passes vacuously). The full suite fails in exactly the same 24 places as before my changes.
- **Memory:**
  - Map storage on ACM at d = 32, k = 2 is **15.75× smaller** than dense.
  - **Peak GPU memory is only about 2× smaller end to end** (measured at d = 8). The retained v1 face/edge machinery, which D4 deletes, dominates the peak.
  - At d = 32 neither family fits 8 GB end to end.
- **A memory problem I found and fixed:** the first factorized version ran out of memory even in the map stage, because of activations saved for backward. Activation checkpointing fixed it, and gradients stay identical to v1 code.
- **Not done:** accuracy experiments (they need your approval) and the full-model measurement at d = 32 (it needs a GPU larger than 8 GB).

---

## 2. Files changed

Committed on `d1-factorized-maps`:

| File | Why |
|---|---|
| `src/models/map_operators.py` (new) | `FactorizedMapOperator` holds the factors and provides `apply`, `apply_T`, `apply_inverse`, `apply_inverse_T`, `select`, plus `materialize` for tests and gathered subsets. Also `DenseMapOperator` (closure edges), `StackedMapOperator` (observed + closure edges), `EdgeMapPair`, `PathTransportOperator` (for D4), and `checkpoint_if_grad`. |
| `src/models/sheaf_maps.py` | New `FactorizedNodeToEdgeSheaf`. `SequenceEdgeToFaceSheaf.forward` gains a `factorized_maps` path that materializes only each chunk's gathered path edges, with the closed-form inverse. |
| `src/models/coboundary_ops.py` | Matrix-free `delta0`, `delta0_T`, `L0` and the `L1` down term. v1's face term is factored into `_upper_hodge_term` with the same ops in the same order (bit-identical, §4 T11). |
| `src/models/h2snn_model.py` | Wiring. `NotImplementedError` for link prediction, graph classification and map injection under `factorized`. `EncodedState.F_src/F_tgt` are `None` under `factorized`; the operators travel in `EncodedState.edge_maps`. |
| `src/models/readout.py`, `src/models/pooling_utils.py` | Matrix-free incident-edge pooling for the node readout. |
| `src/configs/config.py` | `map_family`, `householder_k`, `map_scale_eps`, each with validation. |
| `tests/test_factorized_maps.py` (new) | 32 tests: T1–T11, negative controls, integration, gradient equivalence, materialization detector, scope and config. |

Not committed (working tree):
- `CLAUDE.md`, `AGENTS.md`: D1 "State" column and the "no v2 code exists" sentence, as required by `AGENTS.md:148`. My edits sit inside your own uncommitted rewrite of those sections, so I could not commit them without committing your work.
- Your pre-existing edit at `src/configs/config.py:112` is still unstaged.

---

## 3. Deviations from the spec

1. **Extra config key `model.map_scale_eps` (default 0.1)** (`src/configs/config.py:653`). The spec writes `softplus(·) + ε` without a value. The card's "no hard-coded hyperparameters" rule made ε a config key.
2. **Activation checkpointing**, which the card does not mention. It is added after a measured failure (§5):
   - the matrix-free operators (`coboundary_ops.py:292-350`) and the readout pooling (`pooling_utils.py:44`);
   - the per-side map construction (`sheaf_maps.py:419`);
   - the face-chunk gather (`sheaf_maps.py:105`).

   Gradients are unchanged: `test_factorized_model_equals_v1_pipeline_fed_the_same_maps` (`tests/test_factorized_maps.py:627`) compares every parameter gradient with the v1 pipeline. This follows repo precedent: `orthogonal.py` already checkpoints the FastHPP merge under grad.
3. **Closure edges keep dense per-face maps** (`[F, d, d]` holonomy and identity) inside `StackedMapOperator` (`h2snn_model.py:679-684`). These are per path instance, not per observed edge, and they last until D4 removes closure edges from `Δ₀`. The card allows the face code to materialize.
4. **T10's formula counts floating-point storage only.** Integer index storage per side is the relation ids (a reference to `edge_type`) plus the relation-sort permutation and its inverse: at most 3M int64. That is O(M), not the "small constant" the card's bound allows if integers counted. It is bounded in a separate assertion (`tests/test_factorized_maps.py:555`).
5. **Topological exception to properties 1 and 4.**
   - The reflection vector is normalized as `u / sqrt(max(‖u‖², ε²))` with ε = 1e-6 (`map_operators.py:48-62`).
   - Inside `‖u‖ < 1e-6` the factor is not orthogonal, and at `‖u‖ = ε/√2` it is singular.
   - Any guard that stays finite at `u = 0` must have such a point, because det must go from +1 to −1 through 0 (DERIVED).
   - The independent reviewer reproduced it: det F = 8.8e-17 at that radius.
   - The model only reaches it if the reflector MLP output collapses.
6. **Interpretation choices the spec leaves open:**
   - The c-MLP takes the same input as the u-MLP, `z = [h_v, h_other, rel_emb, onehot(side)]`.
   - Side is a 2-dim one-hot (`sheaf_maps.py:377-397`).
   - `R_{ψ,side}` parameters are initialized `N(0, 1/d)`, with the std hard-coded (`sheaf_maps.py:366`).
7. **v1 switches with no D1 meaning are rejected under `factorized`:** `use_orthogonal=false` and `ablate_identity_maps` (`config.py:714-731`).
8. **The v1 memory guard** (`validate_node_edge_memory_budget`) is not applied under `factorized`, so `node_edge_memory_guard_fraction` is ignored there. Plan §1.4 lists `memory_guard.py`; the card's scope does not.

---

## 4. Verification evidence

All commands run from `H2S-NN/` as `"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python …`.

### 4.1 Suites (VERIFIED-RUN)

| Run | Command | Decisive output |
|---|---|---|
| D1 tests at `924be78` | `-m unittest tests.test_factorized_maps` | `Ran 32 tests in 3.576s` / `OK` |
| Full suite before any change | `-m unittest discover -s tests` | `Ran 208 tests` / `FAILED (failures=11, errors=13)` |
| Full suite at `924be78` | same | `Ran 240 tests` / `FAILED (failures=11, errors=13)` |
| Failing-test IDs, before vs after | `comm -13 base_fail.txt final_fail.txt` | empty (no new failure) |

The 24 failures are all in `tests/test_baselines.py`:
- missing vendor clones (`hgb_nc`, `sehgnn`, `pshgcn`, …);
- notebooks with non-null `execution_count`;
- generator/notebook mismatch;
- missing baseline data directories.

Raw logs: `D1_evidence/baseline_suite.txt` and `D1_evidence/final_suite.txt`.

### 4.2 Required tests

| ID | Test (`tests/test_factorized_maps.py:line`) | What it asserts | Result |
|---|---|---|---|
| T1 | `test_T1_rotation_and_householder_factors_are_orthogonal` :198; control :214 | `RRᵀ = I`, `HHᵀ = I`, `FFᵀ/c² = I`: float64 ≤ 1e-10, float32 ≤ 1e-5. Control: a shrunk reflector gives an error > 1e-2 | pass |
| T2 | `test_T2_apply_and_apply_T_match_dense_reference` :222 | `apply` / `apply_T` equal `materialize() @ x` for k ∈ {1, 2, 4} and for `[E, n, d]` inputs, ≤ 1e-10. The reference is built independently from explicit outer products. A reversed reflection order differs by > 1e-3 | pass |
| T3 | `test_T3_closed_form_inverse_is_the_true_inverse` :247 | `F⁻¹F x = x`, `FF⁻¹x = x`, `F⁻ᵀFᵀx = x`, and the closed form equals `torch.linalg.inv`, all ≤ 1e-10 | pass |
| T4 | `test_T4_maps_are_side_specific_and_node_dependent` :277 | `min_e ‖F_src − F_tgt‖ > 1e-3`. Perturbing `h_v` changes every map incident to v by > 1e-3. Maps not incident to v are **exactly** unchanged (locality, fix spec F5) | pass |
| T5 | `test_T5_holonomy_is_nontrivial_and_varies_within_a_relation_sequence` :328; control :342 | Thresholds fixed before the first run (`DEVIATION_MIN = 0.1`, `SPREAD_MIN = 0.05`, lines 325-326), checked on both groups (PAP, PAPAP), each with one relation sequence. Control: tying src = tgt gives deviation and spread < 1e-10, so the positive check fails | pass |
| T6 | `test_T6_delta1_delta0_vanishes_with_factorized_maps` :412; full model :432; control :425 | `‖δ₁δ₀h‖/‖δ₀h‖ ≤ 1e-10` through `compose_face_transport`, for L ∈ {2, 3, 4} × k ∈ {1, 2, 4}, and for the full float64 model. Control: identity closure maps give > 1e-2 | pass |
| T7 | `test_T7_functoriality_via_either_boundary_edge_including_closure` :444; control :463 | `F_{e◁f}F_{v◁e}` is the same through both boundary edges at every node, including the closure edge at v₀ and v_L, and equals `A_f T_v` from a general-inverse product (≤ 1e-10). Control: a perturbed face map breaks it | pass |
| T8 | `test_T8_matrix_free_path_transport_equals_dense_product` :471; orientation :498 | `PathTransportOperator.apply` equals the explicit product with `torch.linalg.inv` (≤ 1e-10). Adjoint identity ≤ 1e-10. Reversed edges are rejected | pass |
| T9 | `test_T9_finite_gradients_with_near_zero_reflectors` :510; model :534 | u ∈ {0, 1e-20, 1e-7} in float64 and float32: forward and gradients finite, `c ≥ 0.1` at raw −50. A zeroed reflector MLP (u = 0 everywhere) gives finite logits and gradients | pass |
| T10 | `test_T10_storage_matches_the_bound_exactly` :555 | The float tensors actually held, counted generically (not the operator's own report), equal exactly `2·M·(k·d+1) + 2·|R|·d²` for (d, k) ∈ {(8,1), (8,2), (32,2), (32,4)}. Integer storage ≤ 6M. Ratio ≥ 10 at d = 32, k = 2 | pass |
| T11a | `test_T11_dense_matches_pre_change_golden_outputs` :919 | Dense outputs vs values recorded from the **pre-change** code `0172044` (eval logits, train-logit sum, Σ grad²), 4 cases, ≤ 1e-6 | pass |
| T11b | `D1_evidence/t11_regression.py compare t11_reference.pt` | Bit-for-bit vs a snapshot taken before any edit. 9 cases: toy ACM/DBLP/IMDB, AMPSS auto, random PAP/PAPAP graph, LSTM + triangular, `use_orthogonal=false`, identity maps, `ablate_no_delta1`. Covers init weights, eval outputs, train logits, every gradient | `compared 912 entries across 9 cases; mismatches=0` |
| T11b control | same script on the **unchanged** code, right after capture | proves the comparison is deterministic | `mismatches=0` |

### 4.3 Additional checks

| Check | Test / command | Result |
|---|---|---|
| No `[M, d, d]` in forward or backward, including checkpoint replay | `test_no_dense_map_tensor_over_observed_edges_in_forward_or_backward` :741. `TorchDispatchMode` records every tensor any op produces. Rule 1: shape `[…, d, d]` with leading product M or M + F. Rule 2: any tensor with ≥ M·d²/4 elements. Run at d = 32, M = 1200 | factorized: no hits in either pass. Dense control: flagged in both passes |
| Matrix-free path ≡ v1 code on the same maps, forward **and** backward | `test_factorized_model_equals_v1_pipeline_fed_the_same_maps` :627 | logits, x*, s*, δ₀h₀, δ₀x*, face curl and **every parameter gradient** ≤ 1e-10 |
| Operator gradients ≡ dense | `test_matrix_free_gradients_match_dense_reference` :682 | gradients w.r.t. u, c, R, x, s ≤ 1e-10 |
| Gradients reach R, u-MLP, c-MLP | `test_gradients_reach_rotations_reflector_and_scale_mlps` :811 | all non-zero |
| CUDA float32 vs CPU float64 | `test_cuda_float32_matches_cpu_float64_reference` :828 (ran, not skipped) | relative error ≤ 1e-5; full CUDA forward finite |
| v1 switches still run under factorized | `test_factorized_runs_with_v1_switches` :782 | `ablate_no_faces`, `ablate_no_delta1`, LSTM + triangular, AMPSS auto: finite |
| Scope guards | :879, :890 | `NotImplementedError` mentioning D1 |
| Config validation | :857 | rejects unknown family, k < 1, k > d (factorized only), ε ≤ 0, k ∈ {2.9, true, "2"}, `use_orthogonal=false` / `ablate_identity_maps` with factorized; accepts dense with d = 1 |

### 4.4 Can each test fail? Mutation check (VERIFIED-RUN)

`D1_evidence/mutation_check.py` monkeypatches one defect at a time, without editing any file, and runs the targeted tests. Output (`D1_evidence/mutation_check_output.txt`): **`unmutated: all pass`, 15 mutations, `vacuous=0`.**

| Mutation | Tests that fail |
|---|---|
| reflection order swapped | T2, T3 |
| `apply_inverse` multiplies by c instead of dividing | T3, T8 |
| src and tgt maps tied | T4, T5 |
| maps ignore node states | T4, T5 |
| `δ₀ᵀ` sign error | matrix-free vs dense; model ≡ v1 |
| face transport uses Fᵀ as inverse | T6 (full model); model ≡ v1 |
| forward materializes `[M, d, d]` | detector |
| forward allocates `[M, d·d]` | detector |
| forward allocates `[2M, d, d]` | detector |
| reflector-norm guard removed | both T9 tests |
| scale lower bound removed | T9 |
| path transport in reverse order | T8 |
| out-of-scope guard removed | scope test |
| dense v1 maps × 1.001 (reviewer's M11) | T11a golden |
| operator holds an extra float cache | T10 |

---

## 5. Measurements

**Setup.**
- Script: `D1_evidence/measure_d1.py <maps|full> <dense|factorized> <k> <d>`, one process per configuration.
- ACM loaded through `src.data.dataset.build_dataset` with `configs/experiments/acm_h2snn.yaml`. Only `d`, `map_family`, `householder_k` and `node_edge_memory_guard_fraction = 0.95` are overridden; the last lets dense be attempted at all.
- One warm-up, then 5 timed forward + backward passes. Peak is `torch.cuda.max_memory_allocated` after `reset_peak_memory_stats`.
- The process is capped with `torch.cuda.set_per_process_memory_fraction(0.95)` = 7.60 GiB, so an oversize run fails cleanly instead of spilling into system RAM (WDDM).
- Hardware: RTX 4060 Laptop GPU 8.00 GiB, torch 2.6.0+cu124, Windows 11.

**ACM size** (VERIFIED-RUN, `D1_evidence/measure_*.jsonl`):
- N = 10,942 nodes, **M = 547,872** observed edges, |R| = 8.
- Faces: 168,524 (PAP 29,210 + PSP 139,314), all with L = 2, so there are 337,048 path incidences.
- d = 32, K = 4.

**Stages:**
- **maps:** node→edge maps plus K = 4 explicit `L₀` steps over the observed edges, then backward. This isolates what D1 changes.
- **full:** the whole v1 model forward, then cross-entropy backward.

### 5.1 Map storage (exact counts; DERIVED from M, |R|, d, k, matches T10)

| d = 32 | Stored floats | vs dense |
|---|---|---|
| dense `2·M·d²` | 1,122,041,856 | 1× |
| factorized k = 1 | 36,175,936 | 31.0× smaller |
| factorized k = 2 | 71,239,744 | **15.75× smaller** (card: ≥ 10×, met) |
| factorized k = 4 | 141,367,360 | **7.94× smaller (below 10×)** |

### 5.2 Peak memory and time (VERIFIED-RUN, final code)

| d | Stage | dense | factorized k = 2 |
|---|---|---|---|
| 32 | maps | out of memory (failed at 5.82 GiB, allocating 2.09 GiB more) | **2.556 GiB**, 1032.3 ± 0.9 ms |
| 32 | full | out of memory (5.89 GiB) | out of memory (7.54 GiB) |
| 16 | maps | out of memory (7.30 GiB) | 1.897 GiB, 617.7 ± 0.5 ms |
| 16 | full | out of memory (7.36 GiB) | 6.550 GiB, 5953.4 ± 1708.5 ms |
| 8 | maps | 3.411 GiB, 2062.9 ± 2.0 ms | 1.657 GiB, 400.9 ± 2.1 ms |
| 8 | full | 6.082 GiB, 3176.7 ± 8.3 ms | **3.004 GiB**, 1728.9 ± 28.6 ms |

Other values of k at d = 32:
- k = 1: maps 1.963 GiB, 716.6 ± 1.4 ms; full runs out of memory (7.57 GiB).
- k = 4: maps 3.742 GiB, 1871.3 ± 2.0 ms; full runs out of memory (7.56 GiB).

At d = 8 the factorized family uses **2.02× less peak memory end to end** and is 1.84× faster. In the map stage it uses 2.06× less and is 5.15× faster (DERIVED from the table).

### 5.3 Parameter counts (VERIFIED-RUN)

| d | dense total (node→edge) | factorized k = 2 total (node→edge) |
|---|---|---|
| 32 | 1,615,909 (1,191,696) | 461,782 (37,569) |
| 16 | 532,709 (350,096) | 201,846 (19,233) |
| 8 | 214,021 (127,440) | 98,182 (11,601) |

At d = 32, k = 1 has 457,654 (33,441) and k = 4 has 470,038 (45,825). The dense d = 32 total matches the paper's 1.61M (fix spec F16, VERIFIED-SOURCE).

### 5.4 Before the activation-memory fix (VERIFIED-RUN, commit `dd42b8e`)

- **Factorized k = 2, d = 32: out of memory in both stages.** The map stage failed at 7.47 GiB, the full model at 7.57 GiB.
- **Diagnosis (DERIVED by counting saved tensors):**
  - Each matrix-free application retained about k + 3 tensors of shape `[M, d]`, ≈ 70 MB each on ACM, at 4 applications per `L₀` step.
  - The per-side MLP activations of shape `[M, 128]` added ≈ 2.4 GiB.
- Checkpointing (`11ba6c3`) brought the map stage to 2.556 GiB, below the 3 GiB threshold I had predicted before the change.

### 5.5 Reading of the results

- **D1 meets its own criterion, map storage ≥ 10× at k = 2.** Map storage is not the binding constraint, though.
- The full model is dominated by v1 components that stay until D4 (DERIVED from tensor sizes at d = 32):
  - the gathered path maps `[F·L, d, d]`, ≈ 1.3 GiB each for F_src, F_tgt and F_src⁻¹;
  - the face caches `[F, L+1, d, d]`, 1.93 GiB;
  - the closure maps `[F, d, d]`;
  - the edge cochain and `EdgeInitializer` over M + F edges.
- So the plan's Tier-0 goal of ≥ 10× lower *peak* memory (§3.5) cannot come from D1 alone. It depends on D4.

**Timing caveats.** The d = 16 factorized full-model timing (± 1.7 s) was taken near the memory cap; its variance cause is UNVERIFIED. The `maps` stage at d = 32 was measured on the code before the face-gather checkpoint. That change does not touch the map stage (VERIFIED-CODE `sheaf_maps.py:81-105`). Commit `924be78` changes no runtime path of the model forward.

---

## 6. Self-review checklist (D1)

| Item | Status | Evidence |
|---|---|---|
| No `[M, d, d]` for observed edges in forward **or** backward (autograd-saved, not only forward code) | PASS | §4.3 detector (dispatch mode covers forward, backward and checkpoint replay), with a dense control. Mutations `[M,d,d]`, `[M,d²]` and `[2M,d,d]` are caught. The reviewer's independent probe agrees: largest factorized tensor `[4010,128]` vs M·d² = 4.1M |
| `F_src`, `F_tgt` from different parameters/inputs; T5 negative control fails as expected | PASS | `sheaf_maps.py:377-436`: separate `R` per side, swapped endpoint order, side one-hot. T5 control :342. Tied-sides mutation fails T4 and T5 |
| Inverse is the true inverse (T3), used wherever v1 used `F_src_scale` / `_invert_path_maps` | PASS | T3 vs `linalg.inv`. The only v1 use site, the face chunk `sheaf_maps.py:666-676`, now uses `_gather_factorized_path_maps`, a closed form. `PathTransportOperator` uses `apply_inverse` |
| Gradients reach `R_ψ`, u-MLP, c-MLP | PASS | :811 |
| dtype/device: float32 training, float64 tests, CPU and CUDA | PASS | T1 and T9 in float32; most tests in float64; CUDA test :828 ran on the RTX 4060 |
| `c` has a lower bound ε; no division blow-up | PASS, with caveat | `map_operators.py:140`; T9 shows `c(raw = −50) = 0.1`. **No upper bound** (v1 clamped `c ≤ e²`), §8 |
| v1 default path unchanged (T11) | PASS for outputs; FAIL for "full existing suite passes" (pre-existing failures) | §4.2 T11a, T11b; §4.1 |
| Every new config key validated; no hard-coded hyperparameters | PASS, with notes | `config.py:694-731`. Hard-coded: init std `1/√d` (`sheaf_maps.py:366`) and numerical guard `REFLECTOR_NORM_EPS = 1e-6` (`sheaf_maps.py:17`) |
| Out-of-scope paths raise a clear error under `factorized` | PASS | `h2snn_model.py:175-182`, `:588-592`; tests :879, :890 |
| Each test would fail if its property were broken (state how you know) | PASS | §4.4: 15 mutations, 0 vacuous, plus negative controls inside T1, T5, T6, T7 and the detector |

---

## 7. Open items

**UNVERIFIED (each with the check that would resolve it):**

1. **Accuracy of factorized vs dense.** Plan §1.6: DBLP, ACM and IMDB, 10 seeds, k ∈ {1, 2, 4}, paired tests. Needs approval and a GPU above 8 GB for dense at d = 32.
2. **Training stability with unbounded `c`.** Check: a short training run logging `max c` and `η·λ_max`.
3. **Full-model peak memory at d = 32 for both families.** Check: rerun `measure_d1.py full …` on a ≥ 16 GB GPU.
4. **Gradient perturbation from nondeterministic CUDA `index_add_` during checkpoint replay.** Check: repeated CUDA backward passes, comparing gradients.
5. **Cause of the 1.7 s timing standard deviation at d = 16.** Check: rerun with a lower memory fraction, or with allocator statistics.

**Known limitations of the implementation:**

- The face side remains dense: per-chunk gathered path maps, face caches `[F, L+1, d, d]` and closure maps `[F, d, d]`, all v1 code until D4.
- `PathTransportOperator` computes per-position transports at L(L+1) map applications per path. The summed form `z_f` could use Horner's O(L) scheme in D4.
- `DenseMapOperator` accepts only `[E, d]` inputs (documented).

**Out-of-scope problems noticed (not fixed):**

- **W&B resume.** The base fingerprint hashes the full config dict (`scripts/run_h2snn_wandb_tuning.py:694`). The three new keys therefore change the fingerprint, and existing sweep states will no longer resume; continuation sweeps will be created instead.
- **v1 numerics.** `use_orthogonal=false` produces non-finite logits on a 40-node random graph (d = 8, K = 3); seen in the T11 snapshot, VERIFIED-RUN.
- **Docs location.** `CLAUDE.md`/`AGENTS.md` say the planning documents are in `../`; they are in `../docs/`.
- **24 failing tests** in `tests/test_baselines.py` (§4.1).
- **Memory guard.** `src/utils/memory_guard.py` has no model for the factorized family (plan §1.4).

---

## 8. Decisions needed from the author

1. **Upper bound on `c`.** The spec (`softplus + ε`) has none; v1 clamped `c ∈ [e⁻², e²]`. Without a bound, `L₀` eigenvalues grow with c², and explicit Euler can diverge if `η·λ_max > 2`. Options:
   - keep the spec and rely on D3's normalized Laplacian with `η = σ(θ)`;
   - add an upper clamp.
2. **Defaults.** `map_scale_eps = 0.1`; `R` init std `1/√d`; numerical guard 1e-6.
3. **Paper wording (reviewer item 10, DERIVED).**
   - Both sides have det sign `(−1)^d · (−1)^k`, so every edge transport `F_src⁻¹F_tgt` and every holonomy lies in **SO(d)**.
   - The "each reflection flips one feature direction / contrast" interpretation (plan §1.2) therefore holds for individual restriction maps but not for the gauge-invariant transports.
   - v1's FastHPP maps have the same property.
   - Options: reword the claim, or give the two sides different parity.
4. **Memory target.** D1 alone gives ≈ 2× lower peak memory; the plan's ≥ 10× needs D4. Proceed to D4 next?
5. **Approve the §1.6 accuracy experiments**, and decide where to run them (this 8 GB GPU cannot run dense at d = 32).
6. **Git housekeeping.**
   - Commit the `CLAUDE.md`/`AGENTS.md` state updates together with your uncommitted rewrite of those files?
   - Push `d1-factorized-maps`?
   - Re-run an independent review on `924be78`?

---

## Appendix A — Pre-step inventory: consumers of `F_src` / `F_tgt` (VERIFIED-CODE, grep at `0172044`, line numbers at `924be78`)

| File:line | How the maps are used | Scope / outcome under `factorized` |
|---|---|---|
| `src/models/sheaf_maps.py:223-319` (`NodeToEdgeSheaf`) | Builds dense `[M, d, d]` src/tgt maps (FastHPP), scale `c` | In scope. Unchanged; still used by `dense`. Factorized replacement at `:326-436` |
| `src/models/sheaf_maps.py:666-676` (face chunk) | `F_src[chunk]`, `F_tgt[chunk]`, `_invert_path_maps` → `compose_face_transport` | In scope. Factorized branch materializes gathered path edges only, with closed-form inverse |
| `src/models/coboundary_ops.py:18-43` (`delta0`, `delta0_T`), `L0`, `L1` | `bmm` with `[E, d, d]` maps | In scope. Dense versions unchanged; matrix-free versions `:270-350` |
| `src/models/h2snn_model.py:600-617, 669-685, 699-835` | Builds maps, concatenates closure maps, `δ₀`, `L₀`/`L₁` closures, `delta0_x_star`, `EncodedState` | In scope. Factorized branches added; dense lines kept |
| `src/models/readout.py`, `src/models/pooling_utils.py:52-129` | Node readout pools `F_{v◁e}ᵀ s_e` | In scope. Matrix-free sibling `pooling_utils.py:8-49` |
| `src/models/link_readout.py:60-73` | Link readout pooling | Out of scope. Blocked: model construction raises under `factorized` (`h2snn_model.py:175`) |
| `src/models/graph_pooling.py:39-367` | SheafCov/Gram pooling | Out of scope. Blocked (graph classification) |
| `src/models/s2pool_h2.py:48-487` | THS-Pool/S2Pool: map norms, injected maps, Galerkin `L0` | Out of scope. Blocked; `injected_maps` also raises (`h2snn_model.py:588`) |
| `src/trainers/graph_classification_trainer.py:279-280` | Passes `encoded.F_src/F_tgt` to pooling | Out of scope. Blocked |
| `src/trainers/link_prediction_trainer.py:222-223` | Passes `encoded.F_src/F_tgt` to link readout | Out of scope. Blocked |
| `tests/test_coboundary_chain.py`, `test_models.py`, `test_graph_pooling.py`, `test_graph_classification_trainer.py` | Dense-map tests | Unchanged; still pass |

---

## Appendix B — Work units: pre-registration and outcome

| Unit | Pre-registered change, test and expected result (written before running) | "Wrong if" | Outcome |
|---|---|---|---|
| 0 Baseline | Record the full suite; snapshot v1 outputs and gradients (9 cases); determinism control | — | 208 tests, 11 F + 13 E. Snapshot 912 entries. Control 0 mismatches (after making the comparison NaN-aware for the pre-existing NaN case) |
| 1 Operators | `map_operators.py`. Expected float64 errors ~1e-15 vs independent references; exact storage count | any error > 1e-10 or count mismatch | apply 8.9e-16, inverse 5.3e-15, transport 4.6e-15, adjoint 8.9e-15; count exact |
| 2–5 Config, module, coboundary, integration | Expected T11 0 mismatches; factorized `δ₁δ₀` ~1e-15 in float64 | any T11 mismatch, or T6 > 1e-10 | T11 0 / 912. Residual 4.7e-16 – 5.8e-16 for k ∈ {1, 2, 4}. 28 tests then 29 pass; mutations 0 vacuous. Commit `dd42b8e` |
| 6 Activation memory | Checkpoint matrix-free ops and map construction; reflect without a retained `2v`. Expected ACM map stage at k = 2, d = 32 < 3 GiB; gradients equal dense | ≥ 3 GiB, or any gradient difference | 2.556 GiB. Gradient-equivalence test passes; T11 0 mismatches |
| 6b Face gather | Checkpoint the face-chunk gather so it keeps only its outputs (as v1 does) | gradients differ from v1 pipeline | Full-model gradient equivalence ≤ 1e-10. Full d = 32 still out of memory (face/edge v1 code). Commit `11ba6c3` |
| 7 Review fixes | Golden T11 must fail under dense × 1.001; detector must catch `[M,d²]` and `[2M,d,d]`; T10 must fail with an extra cache; strict k; orientation check | any target mutation still passes | All caught. One detector false positive (a d²-output parameter at M = 360) was diagnosed and fixed by sizing the fixture so M > d², with an explicit assertion that no parameter reaches the threshold. 32 tests pass. Commit `924be78` |

---

## Appendix C — Independent review (verbatim)

The reviewer was a fresh agent given only the diff `0172044..11ba6c3`, `H2S-NN_agent_prompt.md` (role and task card) and read-only instructions, not my reasoning. Its report follows unedited. Its scripts are in the session scratchpad (`…\scratchpad\review\`). My responses follow after it.

> ## D1 adversarial review: `git diff 0172044..11ba6c3` (branch d1-factorized-maps)
>
> ### 1. Verdict
> I found no blocking defect. The math is right: reflection order, the order of R and H, transposes, the closed-form inverse, the adjoint of `PathTransportOperator`, and the δ₀ᵀ adjoint all agree with independent dense references. Properties 1–4 hold, with one measure-zero exception to invertibility inside the ε-ball. A broad probe found no tensor at or near `[M, d, d]` in forward, backward (including checkpoint recompute) or autograd-saved tensors under `"factorized"`. With `"dense"`, v1 is bit-for-bit unchanged against a pre-change snapshot: I ran this myself on CPU and CUDA, 7 configs, covering forward, eval, every gradient and the state_dict.
>
> The weak points are in the tests and in robustness, not in correctness:
> - (a) The committed T11 never compares against pre-change code. A 0.1 % change to every dense map passes the whole suite.
> - (b) The conformal scale `c` has no upper bound. v1 capped it at e², and diffusion is explicit Euler.
> - (c) T10 counts only what the operator says it stores.
> - (d) The ACM measurement the task card asks for is not in the diff.
>
> ### 2. D1 checklist
> | Item | Status | Evidence |
> |---|---|---|
> | No `[M,d,d]` for observed edges, forward or backward (autograd-saved tensors) | PASS | I recorded every aten op and every saved tensor, with a size threshold of M·d·d/8 whatever the shape (`mdd_probe.py`; M=4000, d=32, k∈{1,2,4}, CPU+CUDA). Largest factorized tensor in fwd/bwd: 513,280 elements = `[4010,128]`, the activations of v1's `EdgeInitializer` MLP. M·d·d is 4,096,000. Total outer-saved elements: factorized 3.48 M vs dense 125.4 M. The dense control shows `[4010,32,32]` saved 30×. |
> | `F_src`, `F_tgt` from different params/inputs; T5 negative control fails as expected | PASS | `sheaf_maps.py:398-428`: R per side, endpoint order swapped, side one-hot. T5 negative control passes. Tying the sides (mutation M3) fails T4 and both T5 subtests. |
> | Inverse is the true inverse, used wherever v1 used `F_src_scale`/`_invert_path_maps` | PASS | T3 checks against `torch.linalg.inv`. Only site: `sheaf_maps.py:673` → `_gather_factorized_path_maps` (closed form). Grep shows no other `_invert_path_maps` or `F_src_scale` use. Mutations M2 and M4 are caught. |
> | Gradients reach R_ψ, u-MLP and c-MLP | PASS | `test_gradients_reach_rotations_reflector_and_scale_mlps`. Detaching R (M10) fails 5 tests. |
> | dtype/device consistency | PASS | CUDA test runs and passes (RTX 4060 Laptop). My float32 runs at d=32/64 on CPU and CUDA: max \|RRᵀ−I\| ≤ 7.2e-7, \|HHᵀ−I\| ≤ 1.2e-6, full-model ‖δ₁δ₀h₀‖/‖δ₀h₀‖ ≤ 1.2e-6. |
> | `c` has a lower bound ε; no division blow-up | PASS, with a caveat | `map_operators.py:135`; `c(raw=-50)=0.1`. No upper bound: `c(raw=50)=50.1`. See defect 2. |
> | v1 default path unchanged (T11) | PASS by my run; the committed test is weak | See T11 below. |
> | Every new config key validated; no hard-coded hyperparameters | PARTIAL | `config.py:694-732` validates all 3 keys, but `from_dict` (`:786`) does `int()` first, so `householder_k: 2.9`→2 and `true`→1 are accepted silently. Hard-coded: init std `1/sqrt(d)` (`sheaf_maps.py:365`) and `REFLECTOR_NORM_EPS=1e-6` (`:17`). |
> | Out-of-scope paths raise a clear error | PASS | `h2snn_model.py:175` (task guard; graph pooling is only built under `graph_classification`, `:269`) and `:590` (`injected_maps`). The config rejects `use_orthogonal=False` and `ablate_identity_maps`. |
> | Each test fails if its property is broken | MOSTLY PASS | 12 mutations, 10 caught (table below). Gaps: T11 (M11), T10 is self-reported, and the detector's size key is narrow. |
>
> **Mutation results.** Each bug was applied to a scratch copy and the D1 tests were run against it.
>
> | Mutation | Tests failed | Caught by |
> |---|---|---|
> | M1 reflection order | 17 | T2/T3/T6/T8 |
> | M2 `apply_inverse` returns F⁻ᵀ | 3 | T3, T8 |
> | M3 tied sides | 3 | T4, T5 |
> | M4 inverse ×c instead of ÷c | 18 | T3/T6/T7 |
> | M5 transport order in `PathTransportOperator` | 2 | T8 |
> | M6 `apply` materializes `[M,d,d]` | 1 | detector test |
> | M7 δ₀ᵀ uses `apply` instead of `apply_T` | 3 | integration tests |
> | M8 ε guard removed | 4 | T9 |
> | M9 scale floor removed | 2 | T9 |
> | M10 R detached | 5 | gradient test |
> | **M11 dense v1 scale ×1.001** | **0** | **not caught — also not caught by the full suite** |
> | M12 side one-hot always 0 | 0 | not caught; not a property violation, since R and endpoint order still differ |
>
> ### 3. Properties and tests
> **Properties**
> 1. **Invertible: PASS outside a measure-zero set.** F = c·R·H, c ≥ 0.1. Inside the ε-ball (‖u‖<1e-6) H is not orthogonal, and at ‖u‖ = ε/√2 it is singular. My run: det F = 8.8e-17, smallest singular value 1.8e-16, max\|F⁻¹Fx−x\| = 1.22. The docstring discloses this (`map_operators.py:46-56`).
> 2. **Side-specific and node-dependent: PASS.** Evidence is T4, T5 and M3.
> 3. **Never materialized: PASS** (probe above). The face side still holds per-face dense tensors: `DenseMapOperator(closure_F_src)` `[F,d,d]` (`h2snn_model.py:679-684`) and v1's `FaceMapCache` `[F,L+1,d,d]`. These are not observed edges, and the code comments say they stay until D4.
> 4. **Conformal-orthogonal: PASS.** T1 checks F Fᵀ/c² = I. T6 checks δ₁δ₀ = 0 in float64.
>
> **Tests T1–T11**
> - **T1 PASS.** float64 and float32, d=8, with a negative control. I checked d=32/64 in float32 myself.
> - **T2 PASS.** k∈{1,2,4}, input shapes `[E,d]` and `[E,n,d]`; the reversed-order control fails as intended.
> - **T3 PASS.** Compared against `linalg.inv`.
> - **T4 PASS.** Also checks that maps of edges away from the perturbed node do not change.
> - **T5 PASS.** Thresholds are stated in the class: deviation > 0.1, spread > 0.05. The tied-maps control gives < 1e-10.
> - **T6 PASS.** Path lengths 2–4 × k∈{1,2,4}, plus the full model in float64, plus a negative control.
> - **T7 PASS.** Includes the closure edge and compares against a general-inverse product.
> - **T8 PASS.** Includes an adjoint check.
> - **T9 PASS.** Gradients are finite, but they reach about 1/‖u‖: 2.6e6 at ‖u‖ = 7.9e-7.
> - **T10 PASS, but tautological.** It compares `stored_elements()` to the formula exactly. That method sums the numel of the operator's own fields and excludes the per-side `perm`/`inverse` int64 [E] layout (`map_operators.py:153-159, 172-185`), which is O(M), not a "small constant". It would still pass if a dense cache attribute were added.
> - **T11 FAIL as committed; PASS by my independent run.** `test_T11_explicit_dense_is_identical_to_the_default` (`test_factorized_maps.py:855`) compares post-change code with itself. I used `git archive` snapshots of 0172044 and 11ba6c3 with seeded forward and backward in train mode, 7 dense configs (default, K3/d16, lstm+triangular, no_orth, identity, no_faces, no_delta1). Result on CPU and CUDA: `ALL BIT-IDENTICAL: True` for logits, s_star, x_star, δ₀h₀, face curl, δ₁δ₀h₀, eval logits, state_dict and every grad. `no_orth` gives NaN identically in both, which is pre-existing v1 behaviour.
> - **Full suite:** 238 tests ran; 11 failures and 13 errors, all in `test_baselines`. Causes are missing data directories (`data\LastFM`, `data\amazon`, …), missing vendor clones, a missing `H2GB` module and notebook diffs. None touch D1 files. The D1 module itself passes 30/30.
>
> ### 4. Defects and risks, by severity
> 1. **MEDIUM: there is no regression guard for v1.** `tests/test_factorized_maps.py:855-869`. Scenario: a later edit scales every dense map by 1.001 (`sheaf_maps.py:262`, mutation M11). The D1 file passes and the full non-baseline suite passes, with the same 10 failures as the unmutated snapshot. The commit message's "pre-change snapshot" check is not in the repo. Fix: commit golden outputs or a hash for a fixed-seed fixture.
> 2. **MEDIUM (risk, not demonstrated in training): `c` has no upper bound.** `c = softplus(·)+0.1` (`map_operators.py:135`); v1 clamped to [e⁻², e²] (`sheaf_maps.py:262`). Diffusion is `x ← x − η·L0 x` (`diffusion.py:62`). The eigenvalues of L0 grow with c², and holonomy/closure maps grow up to (c_max/0.1)^L. Scenario: c drifts upward during training, η·λ_max goes above 2, and diffusion diverges. v1 could not do this. The card only requires a lower bound, and normalization is a separate unit, but this is a behaviour change from v1.
> 3. **LOW–MEDIUM: T10 and the detector test can be bypassed by construction.**
>    - T10 counts only what the operator reports (see above).
>    - The detector (`test_factorized_maps.py:723-726`) flags only shapes `[..., d, d]` whose leading product is exactly M or M+F. A `[M,k,d,d]`, `[2M,d,d]` or `[M,d²]` tensor would pass it.
>    - My numel-based probe confirms the current code is clean.
> 4. **LOW: invertibility and orthogonality fail in the ε-ball.** Described under property 1; δ₁δ₀=0 and functoriality also fail there. In practice it needs the reflector MLP to collapse.
> 5. **LOW: config cast defeats validation.** `config.py:786` `int(raw.get(...))`: `householder_k: 2.9`→2 and `true`→1 are accepted (my run).
> 6. **LOW: `PathTransportOperator` does not check edge orientation.** `map_operators.py:347-366`. A reversed edge gives a silently wrong transport (my run: accepted without error). D4 must validate this, for example with `validate_face_orientation`. The docstring cost "L(L+1)/2 map applications" is really L(L+1) applications, since each Q_j is two maps. The plan's Horner O(L) form for summed transports is not used.
> 7. **LOW: misleading comment and hard-coded init.**
>    - `sheaf_maps.py:360-365` says R is "Initialised like a random FastHPP output". These are FastHPP *input* entries (strictly-lower Householder entries, `orthogonal.py:115-125`).
>    - The std `1/sqrt(d)` is hard-coded.
> 8. **LOW: no memory guard under factorized.** `validate_node_edge_memory_budget` runs only in dense `NodeToEdgeSheaf`, and `node_edge_memory_guard_fraction` is silently ignored. Plan §1.4 lists `memory_guard.py`; the card's scope does not.
> 9. **INFO: `DenseMapOperator.apply` rejects `[E,n,d]` input.** The `MapOperator` docstring advertises `[E,...,d]`; my run raised `RuntimeError batch2 must be a 3D tensor`.
> 10. **INFO (for the paper):** sign det F_src = sign det F_tgt always, so every edge transport and holonomy is in SO(d). My run confirmed this for d∈{7,8}, k∈{1,2}. The "each reflection flips a direction / contrast" interpretation does not survive in the gauge-invariant transport. v1 FastHPP has the same restriction (DERIVED), so this is not a regression.
> 11. **INFO: scope is clean.** Changes stay in the 8 in-scope files. The `L1` refactor into `_upper_hodge_term` is bit-identical (T11 run). The `map_scale_eps` key goes beyond the card but is validated.
>
> ### 5. Not verified, and what would verify it
> - **ACM measurements** (dense vs factorized k∈{1,2,4}; peak memory, 5 repeats after warm-up, parameter count) are not in the diff. The commit message only claims "Map stage, k=2: OOM → 2.56 GiB", and the comment "exceeds 7 GiB" (`map_operators.py:36`) is not verified. To verify: run `scripts/profile_model.py`-style measurements on ACM. I avoided `build_dataset` because it may write caches into the repo.
> - **Synthetic substitute** (VERIFIED-RUN; M=32,000, 400 faces, d=32, K=2, RTX 4060 Laptop 8 GB). Timings may be perturbed by a CPU test job running at the same time; memory figures are not.
>
> | Variant | Peak memory | Median time | Params (node_to_edge) |
> |---|---|---|---|
> | dense | 2337 MiB | 844 ms | 511,257 (297,924) |
> | factorized k=1 | 240 MiB | 86 ms | 240,822 (27,489) |
> | factorized k=2 | 272 MiB | 92 ms | 244,950 (31,617) |
> | factorized k=4 | 341 MiB | 129 ms | 253,206 (39,873) |
>
>   The k=2 peak is 8.6× lower. The map-storage count ratio is 2048/130 ≈ 15.8× at d=32.
> - **Training stability with unbounded `c` (defect 2):** needs a short training smoke run logging max c and η·λ_max. That requires approval.
> - **CUDA nondeterminism in checkpoint recompute** (atomic `index_add_` sums in recompute vs original): not quantified.
>
> All my scripts and outputs are in `C:\Users\Acer\AppData\Local\Temp\claude\c7cc368768f7bbeb\5abc7a70-ece5-45f4-b579-70797bfc00ef\scratchpad\review\`. Main ones:
> - `t11_dump.py`, `t11_compare.py`: pre/post bit-for-bit check
> - `mdd_probe.py`: broad `[M,d,d]` probe
> - `mutate.py`, `run_tests_from.py`, `suite_from.py`: mutation testing
> - `edge_cases.py`, `f32_check.py`, `det_check.py`, `membench.py`: edge cases, float32 checks, determinant signs, GPU benchmark
> - `full_suite.log`: full-suite output
>
> The repository was not modified.

**My response to each finding (commit `924be78` unless stated otherwise):**

| Reviewer item | Response |
|---|---|
| 1 No v1 regression guard | Fixed: golden test against pre-change values (`:919`); M11 now fails it (§4.4) |
| 2 `c` unbounded | Not changed; the spec mandates `softplus + ε`. Decision 8.1 |
| 3 T10 self-reported; detector narrow | Fixed: generic tensor count, plus a shape-agnostic size rule; new mutations caught |
| 4 ε-ball | Documented (§3 item 5); not changeable without losing finiteness |
| 5 `int()` cast | Fixed: strict parsing, test rejects 2.9, `true`, `"2"` |
| 6 Transport orientation | Fixed: `validate_face_orientation` in the constructor; cost docstring corrected |
| 7 Init comment | Fixed wording; std still hard-coded (Decision 8.2) |
| 8 Memory guard | Open (§7) |
| 9 `DenseMapOperator` shapes | Protocol docstring corrected |
| 10 SO(d) | Decision 8.3 |
| 11 Scope | No action |
| "ACM measurement not in the diff" | In this report, §5 (scripts in `D1_evidence/`) |

---

## Appendix D — Reproducing the evidence

From `H2S-NN/`, with `C="C:\Users\Acer\anaconda3\Scripts\conda.exe"` and `E=..\docs\D1_evidence`:

```
%C% run -n h2snn python -m unittest tests.test_factorized_maps -v
%C% run -n h2snn python -m unittest discover -s tests
%C% run -n h2snn python %E%\t11_regression.py compare %E%\t11_reference.pt
%C% run -n h2snn python %E%\mutation_check.py
%C% run -n h2snn python %E%\measure_d1.py maps factorized 2 32
%C% run -n h2snn python %E%\measure_d1.py full dense 2 8
```

- `t11_reference.pt` was captured from commit `0172044` before any source edit.
- `measure_d1.py` caches the prepared ACM data as `acm_prepared.pt` next to itself; the HGB download goes to the gitignored `H2S-NN/data/`.
