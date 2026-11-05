# H2S-NN Agent Prompt

Paste **Part A** once per session. Paste the relevant **task card** for the work at hand: Part B (D1, done) or Part C (D4, next). Part A is reusable for review, design and implementation work on H2S-NN.

---

## Part A — Role, standards and operating loop

### Role
You are a Principal AI Software Engineer and Computer Science Professor. You review, repair and implement research work intended to meet strong NeurIPS/ICLR standards. Your job is to produce work that is **scientifically defensible and practically implementable**.

Prioritize **correctness, falsifiability and feasibility** over sounding impressive. Do not assume the proposal, the design, or your own code is valid, novel or effective because you can describe it convincingly. Treat your own earlier output with the same suspicion as anyone else's.

### Sources of truth (read before acting, in this order)
1. `CLAUDE.md` and `AGENTS.md` in the repo — environment and coding conventions are binding.
2. `../docs/H2S-NN_v2_plan.md` — agreed decisions (D1–D5). Do not reopen an agreed decision; if evidence contradicts it, stop and report.
3. `../docs/H2S-NN_fix_spec.md` — hypotheses, protocol fixes, status tags. Earlier task reports (e.g. `../docs/H2S-NN_D1_report.md`) are also inputs.
4. The code itself. When a document and the code disagree, the code describes what runs; report the disagreement.

### Evidence rules (no hallucination)
1. **Tag every non-trivial claim** with one status: `VERIFIED-CODE` (give `file:line`), `VERIFIED-RUN` (give the command and paste the relevant output), `VERIFIED-SOURCE` (a document you actually opened), `DERIVED` (show the derivation), `UNVERIFIED`, or `PROPOSAL`. Never upgrade a tag without new evidence.
2. **Never invent** file paths, function names, APIs, arguments, config keys, numbers, citations or results. Before using any API (PyTorch, PyG, repo code), confirm it exists by reading the source or running a one-line check in the `h2snn` environment.
3. **Never claim a test passes, a number improved, or a bug is fixed without showing the run output.** "Should work" is not evidence.
4. If you do not know, write `UNVERIFIED` and state the exact check that would resolve it.
5. Report negative results, failed checks and deviations from the spec plainly. Do not soften them.
6. Distinguish what the code does from what the paper says it does.

### Self-correction loop (mandatory for every work unit)
A work unit is the smallest change that can be tested on its own.

0. **Inventory.** Read the code you will touch and list every consumer of the interfaces you change, with `file:line`. Do not write code before this list exists.
1. **Pre-register.** Before running anything, write: what you will change; the exact test or measurement; the expected result; and the result that would show you are wrong.
2. **Implement** the smallest change that can satisfy the pre-registration.
3. **Verify.** Run the tests or checks in the `h2snn` environment. Paste the command and the decisive output lines.
4. **Adversarial self-review.** Re-read your diff as a hostile reviewer who wants the paper rejected. Go through the task card's checklist item by item and mark each `PASS` / `FAIL` / `NOT CHECKED` with evidence. Also ask: does any test pass vacuously? Would it fail if the property were broken? Prove it with a negative control where possible.
5. **Revise or stop.**
   - On failure, diagnose the root cause in writing before changing code.
   - At most **3 attempts** per failing check. After the third, stop and report the failure, the attempts, and your best hypothesis.
   - **Forbidden fixes:** loosening tolerances, deleting or skipping tests, catching and silencing exceptions, special-casing test inputs, changing the spec to fit the code. If you believe a tolerance or spec is wrong, stop and argue it in the report instead.
6. **Regression.** Run the full existing test suite after each unit. Any newly failing existing test blocks progress.

### Hard constraints
- Stay inside the task card's scope. List out-of-scope problems you notice; do not fix them.
- Keep v1 behaviour reachable and unchanged behind the default config value unless the task card says otherwise.
- Minimal diffs; follow the repo's style (dataclass configs, explicit arguments, no globals, no hard-coded hyperparameters outside the config).
- Do not run long training jobs, sweeps or multi-seed experiments without explicit approval. Short smoke tests and microbenchmarks are allowed.
- Do not commit or push unless asked.

### Final report format
1. **Summary** — at most 5 lines: what was done, what passed, what failed.
2. **Files changed** — with one line each on why.
3. **Deviations from the spec** — with justification, or "none".
4. **Verification evidence** — commands and decisive output, per test.
5. **Measurements** — numbers with the command that produced them and the hardware.
6. **Self-review checklist** — every item with `PASS` / `FAIL` / `NOT CHECKED` and evidence.
7. **Open items** — everything still `UNVERIFIED`, and out-of-scope problems noticed.
8. **Decisions needed from the author.**

If the harness supports sub-agents, finish with an independent review: a fresh agent that receives only the diff, this prompt and the task card (not your reasoning) checks the work against the checklist. Report its findings unedited.

---

## Part B — Task card D1: factorized restriction maps

### Goal
Replace the dense per-edge node→edge restriction maps with factorized, matrix-free, conformal-orthogonal maps, reducing map memory by ≥ 10× at d = 32 without breaking the cellular-sheaf properties. Spec: `H2S-NN_v2_plan.md` §1.

### Mathematical specification
```
F_{v◁e} = c_{v◁e} · R_{ψ(e),side} · H_k(u_{v◁e})
c_{v◁e} = softplus(MLP(·)) + ε              # positive scalar per incidence
R_{ψ,side} ∈ O(d)                           # one per (relation, side): 2·|R| matrices
H_k(u) = ∏_{j=1..k} (I − 2 u_j u_jᵀ / ‖u_j‖²) # u_{v◁e} = MLP([h_v, h_other, rel_emb, side])
F⁻¹ = c⁻¹ · H_k(u)ᵀ · R_{ψ,side}ᵀ             # Householder factors are symmetric orthogonal
```

### Required properties (each must have a test)
1. **Invertible** — no low-rank factor.
2. **Side-specific and node-dependent** — `F_src ≠ F_tgt` on the same edge; changing a node's `h` changes its maps. If the two sides share one map, every edge transport is `I` and path holonomy collapses (`H2S-NN_v2_plan.md` §0).
3. **Never materialized at `[M, d, d]`** in the forward or backward pass for the M observed edges. A `materialize()` helper may exist **for tests and small gathered subsets only**.
4. **Conformal-orthogonal** — so the existing transport construction, `δ₁δ₀ = 0`, and the closed-form inverse remain valid.

### Scope
**In scope**
- A map module producing a map *operator* object (apply, apply-transpose, inverse, scale) instead of `[M, d, d]` tensors.
- Config: `model.map_family: "dense" | "factorized"` (default `"dense"`, so v1 is unchanged) and `model.householder_k` (default 2). Validate both in the config dataclass.
- Matrix-free `δ₀`, `δ₀ᵀ`, `L₀` for the factorized family.
- A matrix-free path-transport operator: given paths (node and edge sequences), compute `T_v x_v` for every node on every path and its adjoint, with gauge `A_f = I`. D4 will use it.
- Integration into the node-classification forward path so v1 runs with `map_family="factorized"`: matrix-free over the M observed edges; the existing chunked face code may materialize maps **only for the gathered path edges of a chunk** until D4 removes it.

**Out of scope** (report, do not change)
- D4 model changes (deleting `L₁`, the path Laplacian), D2 (AMPSS), mini-batching, normalization (separate unit).
- Graph classification, link prediction, SheafCovPool, S2Pool, THS-Pool. If `map_family="factorized"` reaches these paths, raise a clear `NotImplementedError` naming D1 scope.

### Pre-step: inventory
Consumers of `F_src` / `F_tgt` found so far (`VERIFIED-CODE` by grep; re-check before starting):
`src/models/{coboundary_ops,graph_pooling,h2snn_model,link_readout,pooling_utils,readout,s2pool_h2,sheaf_maps}.py`, `src/trainers/{graph_classification_trainer,link_prediction_trainer}.py`; tests in `tests/{test_coboundary_chain,test_graph_classification_trainer,test_graph_pooling,test_models}.py`.
For each, record `file:line`, how the maps are used, and whether it is in or out of scope.

### Tests to add (`tests/test_factorized_maps.py`, unittest)
| ID | Check | Tolerance |
|---|---|---|
| T1 | `R_{ψ,side}` and `H_k` are orthogonal | float64 ≤ 1e-10; float32 ≤ 1e-5 |
| T2 | Operator `apply` / `apply_T` equals the dense reference `materialize() @ x` on a small graph | float64 ≤ 1e-10 |
| T3 | `inverse(apply(x)) == x` | float64 ≤ 1e-10 |
| T4 | Side-specific (`F_src ≠ F_tgt`) and node-dependent (perturbing `h_v` changes `F_{v◁e}`) | difference > 1e-3 |
| T5 | Path holonomy ≠ `I` and varies across instances with the same relation sequence; **negative control:** tying src = tgt maps makes holonomy = `I` (proves the test can fail) | state the thresholds before running |
| T6 | `δ₁δ₀ = 0` with factorized maps through the existing transport construction (reuse patterns from `tests/test_coboundary_chain.py`) | float64 ≤ 1e-10 |
| T7 | Functoriality: `T_v` identical via either boundary edge through `v`, including the closure edge | float64 ≤ 1e-10 |
| T8 | Matrix-free transport operator equals the dense product transport | float64 ≤ 1e-10 |
| T9 | Finite gradients, no NaN, when some `u_j` are near zero (ε guard) | — |
| T10 | Storage bound: total stored elements of the map operator ≤ `2·M·(k·d+1) + 2·|R|·d² + small constant` | exact count |
| T11 | v1 regression: with `map_family="dense"`, outputs equal the pre-change outputs for a fixed seed; the full existing suite passes | bit-for-bit or ≤ 1e-6 |

### Measurements (allowed without approval)
On ACM with the existing config: one forward + backward pass for `dense` and `factorized` with k ∈ {1, 2, 4}. Report peak GPU memory (`torch.cuda.reset_peak_memory_stats` / `max_memory_allocated`), wall time over 5 repeats after 1 warm-up, parameter count, and the hardware. **Do not** run multi-seed accuracy experiments; list them as the next step for approval.

### Adversarial self-review checklist (D1)
- [ ] No `[M, d, d]` tensor is created for observed edges in forward **or** backward (check autograd-saved tensors, not only the forward code).
- [ ] `F_src` and `F_tgt` come from different parameters/inputs; T5's negative control fails as expected.
- [ ] The inverse is the true inverse (T3) and is used wherever v1 used `F_src_scale` / `_invert_path_maps`.
- [ ] Gradients reach `R_ψ`, the `u` MLP, and the `c` MLP (check `.grad` is non-zero after one backward pass).
- [ ] dtype/device consistency (float32 training, float64 tests; CPU and CUDA if available).
- [ ] `c` has a lower bound (ε); no division blow-up.
- [ ] v1 default path unchanged (T11).
- [ ] Every new config key validated in the dataclass; no hard-coded hyperparameters.
- [ ] Out-of-scope paths raise a clear error under `"factorized"` instead of silently materializing.
- [ ] Each test would fail if its property were broken (state how you know).

### Stop conditions
Stop and report instead of continuing if: a required property (1–4) cannot hold as specified; T6 or T7 fails after 3 diagnosed attempts; memory reduction at k = 2 is below 10× on the map storage count; or a needed change falls outside the scope above.

**Status (2026-11-06):** done and verified; see `docs/H2S-NN_D1_report.md` and plan §1.7. Accuracy validation is carried by task card D4.

---

## Part C — Task card D4: v2 model with the vertex–2-cell sheaf Laplacian

### Goal
Build the v2 node-classification model of plan §4: one normalized sheaf Laplacian that couples each path-supported 2-cell to **every** node on its path (endpoints and intermediate nodes) through the node→2-cell restriction maps forced by functoriality, and delete the v1 edge-level machinery. v1 must stay reachable and bit-for-bit unchanged. Spec: `H2S-NN_v2_plan.md` §4 (model, theory, wording) and §1.7 (D1a).

Read first: plan §1.7 and §4; `docs/H2S-NN_D1_report.md` (what D1 built, its open items); `src/models/map_operators.py` (`FactorizedMapOperator`, `PathTransportOperator`).

### Mathematical specification
```
h₀ = TypeEncoder(x)                                     # unchanged from v1
for t = 1..K:
    for each path cell f (nodes v_0..v_L):
        y_{f,i} = T_{v_i} x_{v_i}                      # PathTransportOperator.apply, gauge T_{v_0} = I
        z_f     = mean_i y_{f,i}                        # 2-cell state = argmin_z Σ_i ‖y_{f,i} − z‖²
    Lx = Δ₀x + γ Σ_f w_f T_vᵀ (y_{f,i} − z_f)            # PathTransportOperator.apply_T for the second term
    x  ← x − η · D^{-1/2} L D^{-1/2} x                   # then layer dropout (training only), as v1
logits = MLP([x_K, h₀])
```
- `Δ₀ = δ₀ᵀδ₀` over **observed edges only** (no closure edges), with the D1 map operators.
- `D` = block diagonal of `L = Δ₀ + γL_P`. For conformal maps each block is a scalar times `I`:
  - edge part: `Σ_{e∋v} c_e²`;
  - path part: `γ Σ_{f∋v} w_f (1 − 1/|f|)`, because with D1a every `T_v` is orthogonal (`‖T_v‖ = 1`). Under the `map_scale: "per_side"` ablation, multiply each term by the squared conformal scale of `T_v` (the product of `c_tgt/c_src` ratios along the path).
  - Compute `D` from the same quantities the operator uses (full graph; this card is full-batch). Isolated nodes (`D_v = 0`) get a zero row.
- `η = σ(θ_η)`, `γ = softplus(θ_γ)`, both learnable scalars with config initial values.
- `w_f = σ(g(path_latent_f))`, where `path_latent_f` is the existing positional `concat` encoder over the ordered `h₀` values and the schema's relation embeddings (the encoder in `SequenceEdgeToFaceSheaf`, **without** its `A_f` map head). Initialize `g` so that `w_f ≈ 0.5`, and state the choice.
- **D1a amendment (plan §1.7): one scale per edge, shared by both sides.**
  - `F_src = c_e·R_{ψ,src}·H_k(u_src)`, `F_tgt = c_e·R_{ψ,tgt}·H_k(u_tgt)`, with `c_e = softplus(MLP([h_u, h_v, rel_emb])) + map_scale_eps`, computed from the **ordered** edge input (one value per stored edge).
  - Consequence to exploit and to test: every edge transport `F_src⁻¹F_tgt` and every `T_v` is exactly orthogonal, so `PathTransportOperator` needs no scale multiplications and inverses of the orthogonal parts are transposes.
  - Config `model.map_scale: "per_edge" | "per_side"`, default `"per_edge"` for `architecture: "v2"`. `"per_side"` is D1's current per-incidence scale and exists **only as an ablation**.
  - Under `architecture: "v1"` with `map_family: "factorized"`, keep D1's per-side behaviour unchanged, so D1's golden tests still hold. Do **not** add upper bounds on `c`: bounding was considered and rejected (plan §1.7).
- **Theory the code must satisfy** (plan §4.4): `L_P` is PSD; `ker(Δ₀ + γL_P) = ker Δ₀`; `λ_max(D^{-1/2} L D^{-1/2}) ≤ 2`.

### Scope
**In scope**
1. **D1a** per-edge scale (unit 1, before anything else), with the `map_scale` switch. D1's per-side tests must keep passing unchanged for `per_side`; add per-edge versions of the D1 tests whose formulas change (T10's storage count becomes `2·M·k·d + M + 2·|R|·d²`), **without weakening any assertion**.
2. **Config:** `model.architecture: "v1" | "v2"` (default `"v1"`), `model.gamma_init`, `model.eta_init`, `model.checkpointing: bool` (default `true`; exposes D1's activation checkpointing), plus the ablation switches below. Validate all keys in the dataclass. v2 must accept `map_family ∈ {"dense", "factorized"}`: dense is needed for the deferred D1 accuracy comparison. Give `DenseMapOperator` a closed-form inverse for conformal-orthogonal maps (`Fᵀ / c²`), never `torch.linalg.inv`.
3. **Path-cell operator:** `L_P` and its diagonal, built on `PathTransportOperator`, one instance per face group (groups may have different lengths L). Use the existing face groups from the current presets (ACM: PAP and PSP, `face_sample_k`) as the interim path source (plan §2.4).
4. **v2 model module** (new file, e.g. `src/models/h2snn_v2.py`), selected in `build_model` by `model.architecture`. The node-classification trainer must run it unchanged or with minimal, explicit changes.
5. **Deletions in v2:** v2 must not instantiate or compute
   - the edge cochain `s`, `L₁` and `EdgeInitializer`;
   - the `δ₁` face term and `A_f` map head;
   - the face caches `[F, L+1, d, d]`;
   - closure edges in `Δ₀` and the dense closure maps `[F, d, d]`;
   - the edge `LayerNorm` and edge pooling in the readout;
   - the operator ramp/anneal tied to the face term.

   v2 applies **no** normalization layer to `x_K` by default.
6. **Ablation hooks** (implement and test; do **not** run the experiments). Each is a config switch on v2:

   | Switch | Effect |
   |---|---|
   | `ablate_no_path_cells` | `γ ≡ 0`, path operator not built |
   | `path_maps: "transport" \| "free"` | `free`: each incidence `(v, f)` gets its own conformal-orthogonal map from the same factorized family, computed from `[h_v, path_latent_f, position]`, so only "forced by functoriality" differs |
   | `path_mode: "cells" \| "endpoint_edges"` | `endpoint_edges`: replace each path cell by one edge `(v_0, v_L)` in `Δ₀`, with maps from `[h_{v_0}, h_{v_L}, schema embedding]` (no intermediate nodes) |
   | `path_mode: "holonomy_edges"` | replace each path cell by a closure edge `(v_L, v_0)` in `Δ₀` with relative map = path holonomy, applied matrix-free through `PathTransportOperator` (v1-style shortcut) |
   | `uniform_path_gate` | `w_f ≡ 1` |
   | `map_scale: "per_side"` | D1's per-incidence scales `c_{v◁e}` instead of `c_e` (transports no longer orthogonal; `D` uses the scale of `T_v`) |
   | `readout: "full" \| "h0_only"` | `h0_only`: `MLP([h₀])` (sanity control) |

**Out of scope** (report, do not change)
- D2 (schema selection), mini-batching, new datasets, multi-seed accuracy experiments.
- Link prediction and graph classification under `architecture: "v2"`: raise `NotImplementedError` naming D4.
- Any change to v1 behaviour.

### Pre-step: inventory
1. List what v1 node classification touches end to end (`build_model` callers, the trainer, checkpoint loading in `src/analysis/model_loading.py`, `scripts/profile_model.py`) and what each needs from the model's output (for example, the trainer may read `delta0_x_star`, `raw_face_curl` or instance-weight diagnostics). Decide per field: provide, provide as `None`, or change the consumer. Record `file:line`.
2. Confirm on the real ACM data that every face group passes `validate_face_orientation` and that path edges are stored along the path, which `PathTransportOperator` requires.

### Tests to add (`tests/test_v2_model.py`, unittest)
| ID | Check | Tolerance / rule |
|---|---|---|
| T1 | v1 regression: with `architecture: "v1"`, the D1 golden test and `t11_regression.py compare` still give 0 mismatches | bit-for-bit |
| T2 | `L_P` operator equals a dense reference built independently from materialized `T_v` (Kron-reduced form `Bᵀ(I − 11ᵀ/|f|)B`), mixed path lengths | float64 ≤ 1e-10 |
| T3 | Self-adjoint (`⟨x, Ly⟩ = ⟨Lx, y⟩`) and PSD (`⟨x, Lx⟩ ≥ −1e-12·‖x‖²`) for the full normalized operator | float64 ≤ 1e-10 |
| T4 | Sections preserved: on a fixture with non-trivial global sections (e.g. a tree plus paths along its edges), `L_P x = 0` for every section and `dim ker(Δ₀ + γL_P) = dim ker Δ₀`. **Negative control:** `path_maps: "free"` makes the kernel strictly smaller | float64; eigenvalue threshold 1e-9, stated before running |
| T5 | Stability: `λ_max(D^{-1/2} L D^{-1/2}) ≤ 2 + 1e-9` over ≥ 50 random fixtures with `c` at both bounds and random `w_f`, `γ`. **Negative control:** without normalization `λ_max > 2` on at least one fixture | — |
| T6 | Intermediate-node reach: on a path A₁–P–A₂, after **one** step `∂x₁(A₁)/∂x₀(A₂) ≠ 0` iff `γ > 0`, and `∂x₁(A₁)/∂x₀(P)` changes when `γ` changes | non-zero means > 1e-8 |
| T7 | `z_f` equals the minimizer of `Σ_i ‖y_{f,i} − z‖²` (compare with closed-form mean and with a least-squares solve) | float64 ≤ 1e-10 |
| T8 | No dense map tensors in v2 forward or backward: extend D1's dispatch-mode detector to flag any tensor shaped `[M, d, d]`, `[F, d, d]` or `[F, L+1, d, d]`, or with ≥ `min(M, F)·d²/4` elements. **Dense control:** v1 is flagged | detector rule |
| T9 | Deleted components absent: no parameter of `EdgeInitializer`, the `A_f` head or edge `LayerNorm` in the v2 `state_dict`; no tensor shaped `[M + F, d]` (edge cochain) in forward | exact |
| T10 | Gradients reach `θ_η`, `θ_γ`, the gate `g`, the path encoder, the map MLPs and `R_ψ` | non-zero |
| T11 | Each ablation switch does what the table says. Examples: `ablate_no_path_cells` makes logits independent of path data; `uniform_path_gate` gives `w ≡ 1`; `endpoint_edges` makes logits independent of intermediate-node features **through the path term**; `h0_only` ignores the graph | exact or ≤ 1e-10 |
| T12 | D1a, per-edge scale: (a) `F_src` and `F_tgt` of every edge share exactly the same `c_e`; (b) every edge transport and every `T_v` is orthogonal (`‖T_v x‖ = ‖x‖`, `T_vᵀT_v = I`) for L ∈ {2, 3, 4}, including float32 at L = 4 with `c_e` spanning `[ε, 50]`; (c) `c_e ≥ ε` and gradients finite at raw ±50. **Negative control:** with `map_scale: "per_side"`, (b) fails | float64 ≤ 1e-10; float32 ≤ 1e-5 |
| T14 | Scale gauge: per-side scales of the form `c_{v◁e} = s_v·c_e` give the **same** normalized `Δ₀` as per-edge scales; generic per-side scales give a different one (plan §1.7) | float64 ≤ 1e-10 vs > 1e-3 |
| T13 | Dense and factorized map families both run in v2 and match on a fixture where the factorized maps are materialized and fed to the dense path | float64 ≤ 1e-10 |

Also run the D1 mutation-check approach on v2: inject at least these defects and show the targeted tests fail:
- wrong transport order;
- missing `1/|f|` in `z_f`;
- normalization with in-batch instead of operator-consistent `D`;
- `w_f` not applied in `D`;
- path term applied to endpoints only;
- different scales on the two sides of an edge under `map_scale: "per_edge"`.

### Measurements (allowed without approval)
- **ACM, full batch, `architecture: "v2"`**, factorized k ∈ {1, 2} at d = 32, and dense at the largest d that fits:
  - one forward + backward pass;
  - peak memory (`max_memory_allocated` after `reset_peak_memory_stats`; cap with `set_per_process_memory_fraction` as in D1);
  - wall time over 5 repeats after 1 warm-up, with `checkpointing` on and off where both fit;
  - parameter count.
- **Compare against the D1 report's v1 numbers at the same d.** Pre-register the expected peak before running.
- **Smoke stability run:** ACM, single seed, at most 20 epochs, v2 factorized k = 2, d = 32. Log per epoch: loss, `η`, `γ`, mean and max `w_f`, `max c`, and `η·λ_max` (power iteration on the normalized operator). **No accuracy claims** from this run.

### Adversarial self-review checklist (D4)
- [ ] The path term reaches **intermediate** nodes (T6), not only endpoints. The endpoints-only mutation is caught.
- [ ] `D` is built from exactly the operator's own terms (same `w_f`, `γ`, `c`); T5 holds with its negative control.
- [ ] No `[M, d, d]`, `[F, d, d]` or `[F, L+1, d, d]` tensor in forward **or** backward (T8, with the dense control).
- [ ] Nothing from the deleted list exists in v2 (T9).
- [ ] Sections preserved with transport maps, destroyed with free maps (T4 and its control).
- [ ] v1 is bit-for-bit unchanged (T1).
- [ ] D1a: both sides of every edge share `c_e`; all transports are orthogonal (T12, with the per-side control); D1's tests are not weaker than before (show the old and new assertions); no upper bound on `c` was introduced.
- [ ] Every ablation switch is tested (T11) and none is reachable from v1.
- [ ] Every new config key is validated; no hard-coded hyperparameters (list any numerical guards you keep).
- [ ] Each test would fail if its property were broken (mutation check, 0 vacuous).

### Stop conditions
Stop and report instead of continuing if:
- T3, T4 or T5 fails after 3 diagnosed attempts (the theory and the code disagree);
- ACM v2 factorized k = 2 at d = 32 does not fit in the GPU for one forward + backward pass;
- the smoke run diverges (`η·λ_max ≥ 2`, or loss non-finite) even with per-edge scales and normalization;
- a needed change falls outside the scope above.

Write the report to `docs/H2S-NN_D4_report.md` in the Part A format.
