# H2S-NN Agent Prompt

Paste **Part A** once per session. Paste the relevant **task card** for the work at hand: Part B (D1, done), Part C (D4, done), Part D (D7–D9, done), Part E (D10, done), Part F (D11, done), Part G (D12, done), Part H (D2 → WISP, the replacement of AMPSS; done), Part I (WISP follow-up after spec v1.4; done, `bf7f333`), Part K (weight-channel capacity, map latents and guards; spec v1.6; done, `6ac401f`) or **Part J (+ J4, spec v1.8; evaluation scope of spec v1.5: heterophily measures, link-prediction datasets, cumulative advisor report; next)**. Part A is reusable for review, design and implementation work on H2S-NN.

**Naming (binding):** the meta-path mechanism of the current model is **WISP** (Walk-sampled, Instance-weighted Sheaf Paths; spec v1.3 §3.2). "AMPSS" means only the deleted v1 mechanism (tag `v1-submitted`). `../docs/H2S-NN_D2_task_card.md` (old "Part E: D2") is superseded by Part H; do not implement from it.

---

## Part A — Role, standards and operating loop

### Role
You are a Principal AI Software Engineer and Computer Science Professor. You review, repair and implement research work intended to meet strong NeurIPS/ICLR standards. Your job is to produce work that is **scientifically defensible and practically implementable**.

Prioritize **correctness, falsifiability and feasibility** over sounding impressive. Do not assume the proposal, the design, or your own code is valid, novel or effective because you can describe it convincingly. Treat your own earlier output with the same suspicion as anyone else's.

### Sources of truth (read before acting, in this order)
1. `CLAUDE.md` and `AGENTS.md` in the repo — environment and coding conventions are binding.
2. `../docs/H2S-NN_methodology_spec.md` (v1.6) — **the authority for design**: requirements R1–R12, components, theory, claims and protocol. Then `../docs/H2S-NN_v2_plan.md` — the decision log: agreed decisions (D1–D12), the design rationale of the final model (§12.0: assumptions A1–A5, properties P1–P5) and the reality-before-theory audit (§11). Do not reopen an agreed decision; if evidence contradicts it — including evidence from the data or from published models — stop and report.
3. `../docs/H2S-NN_fix_spec.md` — hypotheses, protocol fixes, status tags. Earlier task reports (e.g. `../docs/H2S-NN_D1_report.md`) are also inputs.
4. The code itself. When a document and the code disagree, the code describes what runs; report the disagreement.

### Evidence rules (no hallucination)
1. **Tag every non-trivial claim** with one status: `VERIFIED-CODE` (give `file:line`), `VERIFIED-RUN` (give the command and paste the relevant output), `VERIFIED-SOURCE` (a document you actually opened), `DERIVED` (show the derivation), `UNVERIFIED`, or `PROPOSAL`. Never upgrade a tag without new evidence.
2. **Never invent** file paths, function names, APIs, arguments, config keys, numbers, citations or results. Before using any API (PyTorch, PyG, repo code), confirm it exists by reading the source or running a one-line check in the `h2snn` environment.
3. **Never claim a test passes, a number improved, or a bug is fixed without showing the run output.** "Should work" is not evidence.
4. If you do not know, write `UNVERIFIED` and state the exact check that would resolve it.
5. Report negative results, failed checks and deviations from the spec plainly. Do not soften them.
6. Distinguish what the code does from what the paper says it does.

### Reality before theory (binding; added 2026-11-07)
Our own results (Lemma A′, Prop. 1–3, the stability bound, section preservation) are tools, not goals. They have pulled designs toward what is provable instead of what the data needs: the operator was kept symmetric only because the propositions require it, although influence in heterogeneous graphs is plausibly unequal (plan §10, §11). Prevent this:
1. **No theory veto.** Never reject, restrict or default away from a design only because it would break one of our propositions. Write every such constraint as `THEORY-CONSTRAINT: <what it forbids> — <what it may cost on real data>` and treat it as a hypothesis to test, not a fact.
2. **Reality check first.** Before recommending a design choice, answer two questions: what do the data and the domain suggest without our theory (direction of influence, hubs, degree skew, heterophily patterns, relation semantics)? What do the strongest published models for this setting do? Cite only sources you actually opened (`VERIFIED-SOURCE`); otherwise mark the claim `UNVERIFIED`.
3. **Steelman the alternative.** For every decision, state the strongest case for the option that breaks our theory. Then look for a variant that keeps that benefit and the guarantees (as D10's gates do). If none exists, say so plainly.
4. **Experiments decide; theory never removes an ablation.** When theory and reality disagree, add the ablation that settles it. Intuition, including the author's and yours, is also a hypothesis: do not adopt the intuitive option without the same test.
5. **Report it.** Every report has a section "Assumptions imposed for theory and reality checks": each `THEORY-CONSTRAINT` the work touched or added, and the reality checks done.

### Self-correction loop (mandatory for every work unit)
A work unit is the smallest change that can be tested on its own.

0. **Inventory.** Read the code you will touch and list every consumer of the interfaces you change, with `file:line`. Do not write code before this list exists.
1. **Pre-register.** Before running anything, write: what you will change; the exact test or measurement; the expected result; and the result that would show you are wrong. For a known-answer task, an oracle or a gate, also write (a) **reachability**: the parameter change the expected result needs and the change the run's optimizer and schedule can make (one scalar under Adam moves about Σ_t lr_t); an unreachable expectation is redesigned before running; and (b) an **information-route audit**: every route by which the label can reach a target within the receptive field, per arm, including memorisation routes (spec §0 rule 6).
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
- Keep the previous behaviour reachable and unchanged behind a config value unless the task card says otherwise (v1 itself was removed by D6 and lives on the tag `v1-submitted`).
- Minimal diffs; follow the repo's style (dataclass configs, explicit arguments, no globals, no hard-coded hyperparameters outside the config).
- Do not run long training jobs, sweeps or multi-seed experiments without explicit approval. Short smoke tests and microbenchmarks are allowed.
- Do not commit or push unless asked or the task card says so.

### Final report format
1. **Summary** — at most 5 lines: what was done, what passed, what failed.
2. **Files changed** — with one line each on why.
3. **Deviations from the spec** — with justification, or "none".
4. **Verification evidence** — commands and decisive output, per test.
5. **Measurements** — numbers with the command that produced them and the hardware.
6. **Self-review checklist** — every item with `PASS` / `FAIL` / `NOT CHECKED` and evidence.
7. **Assumptions imposed for theory and reality checks** — see "Reality before theory".
8. **Open items** — everything still `UNVERIFIED`, and out-of-scope problems noticed.
9. **Decisions needed from the author.**

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
| T5 | Stability: `λ_max(D^{-1/2} L D^{-1/2}) ≤ 2 + 1e-9` over ≥ 50 random fixtures with `c_e ∈ {ε, 50}` (and per-side scales under the ablation) and random `w_f`, `γ`. **Negative control:** without normalization `λ_max > 2` on at least one fixture | — |
| T6 | Intermediate-node reach: on a path A₁–P–A₂, after **one** step `∂x₁(A₁)/∂x₀(A₂) ≠ 0` iff `γ > 0`, and `∂x₁(A₁)/∂x₀(P)` changes when `γ` changes. **Also required (strengthened in the D4 run):** the path term's **own** Jacobian blocks `A₁ ← P` and `P ← A₁` are non-zero, because γ also enters `D`, so the γ-dependence alone passes even when the path term ignores P | non-zero means > 1e-8 |
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

**Status (2026-11-06):** done; see `docs/H2S-NN_D4_report.md` and plan §4.7–4.8 (heads ported in `bafb80b`, `e7c3ab1`).

---

## Part D — Task card D7–D9: fact layer, path-cell builder, O(L) path term

### Goal
Make the 2-complex that H2S-NN diffuses over match the problem setting and the sheaf theory:
- one oriented 1-cell per directed fact (D7);
- path 2-cells that are identified by their facts, drawn from a stated distribution, merged by summing weights, and built without a Python loop over nodes (D8);
- the path term computed in 2L transport steps per path (D9).

Spec: `H2S-NN_v2_plan.md` §9 (D7–D9) and §4.6 (new ablation rows). Evidence for every problem: `Reviews/H2S-NN_methodology_check.md` (project doc; numbers are quoted in plan §9).

Read first: plan §9, §4.2, §4.4; `docs/H2S-NN_D4_report.md`; `src/models/map_operators.py` (`PathTransportOperator`), `src/models/vertex_path_laplacian.py` (`PathCellTerm`, `HolonomyEdgeTerm`), `src/data/preprocessing.py` (`enumerate_metapath_faces`), `src/data/graph_faces.py` (`build_typed_path_face_groups`), `src/data/face_geometry.py`, `src/data/hetero_flatten.py` (`flatten_edges`), `src/data/structures.py` (`FaceGroup.face_edge_sign`), `src/data/link_prediction_dataset.py` (reverse-relation mirroring near `:891–933`, `_assert_no_message_passing_leak`, `_assert_no_leakage`), `src/configs/config.py` (`MetaPathConfig`, presets, `face_sample_k`).

### Mathematical specification
**D9 — O(L) path term** (per path cell `f` with nodes `v_0..v_L`, `n = L + 1`, merged weight `m_f`, gate `w_f`):
```
S   = x_{v_0} + Q_1 (x_{v_1} + Q_2 (x_{v_2} + … + Q_L x_{v_L}))      # Horner, L steps: S = Σ_i T_i x_i
z_f = S / n
t_0 = z_f ;  t_i = Q_iᵀ t_{i−1}                                       # L steps: t_i = T_iᵀ z_f
(L_f x)_{v_i} = w_f m_f ( s_i² x_{v_i} − t_i )                        # s_i² from squared_scales(); 1 under D1a
diag: w_f m_f (1 − 1/n) s_i²                                          # unchanged
```
Valid because `T_iᵀT_i = s_i² I` for every map family in the code (DERIVED, plan §9.3). The free-map ablation already costs O(n); leave its algorithm unchanged.

**D7 — transport with direction.** For hop `j`, with fact `e_j` and sign `σ_j = face_edge_sign[·, j]`:
```
σ_j = +1 (e_j stored v_{j−1} → v_j):  Q_j = F_src⁻¹ F_tgt
σ_j = −1 (e_j stored v_j → v_{j−1}):  Q_j = F_tgt⁻¹ F_src      (= Q_fwdᵀ under D1a)
```
Both follow from functoriality `T_{v_{j−1}} F_{v_{j−1}◁e}⁻¹ = T_{v_j} F_{v_j◁e}⁻¹`. `apply`, `apply_T`, `apply_holonomy(_T)`, the Horner sum and the prefix recursion must all use the signed step.

**D7 — fact layer (one definition, plan §9.1):**
```
ρ(r) = r′  iff  E(r′) = { (v, u) : (u, v) ∈ E(r) }      exact equality; r′ = r allowed; otherwise ρ(r) = ∅
fact      = (u, r, v) ≡ (v, ρ(r), u);   1 fact = 1 oriented 1-cell, stored in the orientation of its lower relation id
roles     = source and target role per relation; they coincide when ρ(r) = r (one frame, no side code, c_e symmetrized over both input orders)
```
`ρ` is computed from the data before training, with no threshold and no config key. Do **not** implement case-by-case rules, overlap thresholds, or a config entry for `ρ`. Unmatched edges are simply their own facts; report their count per relation.

**D8 — estimator.** For each (anchor `a`, schema `π`): draw `r` walks; at hop `j` pick a uniform neighbour among the facts of relation `ψ_j` in direction `σ_j` (out-neighbours for +1, in-neighbours for −1); reject walks that revisit a node (they contribute 0). Each accepted walk gives a cell with weight `1/r`. Cells with the same identity are merged and their weights **summed** into `m_f`. Identity key: the tuple of fact ids in canonical orientation (the lexicographically smaller of the fact sequence and its reversal). Then `E[L̂_P] = L_P^target` (plan §9.2) and `m_f` enters both `L_P` and `D`.

### Scope
**In scope**
1. **D9** in `PathCellTerm` / `PathTransportOperator` (new methods for the Horner sum and the prefix recursion); `HolonomyEdgeTerm` keeps using `apply_holonomy(_T)`.
2. **Signed transport** (`face_edge_sign ∈ {+1, −1}`) in `PathTransportOperator`; `validate_face_orientation` accepts −1 hops whose fact is stored reversed, and still rejects anything else and non-simple paths.
3. **Fact layer** as a data-preparation step shared by all three tasks, before path building, implementing the single definition above. Config `data.fact_layer: "tied" | "untied"` (default `"tied"`; `"untied"` = current behaviour). `untied` is an **ablation control only**: it must not appear in any tuning search space, sweep config or tuning notebook (add a check that rejects it there). Store `ρ`, the kept relations and the unmatched-edge counts in the prepared data and log them.
4. **Shared roles** for relations with `ρ(r) = r` in `FactorizedNodeToEdgeSheaf` and `DenseConformalNodeToEdgeSheaf` (one frame, no side code, `c_e` symmetrized over both input orders).
5. **Schemas with direction:** `MetaPathConfig` gains optional `hops: [[relation_name, "+" | "-"], …]`. Without `hops`, each node-type transition must resolve to exactly one (relation, direction) under the active fact layer; otherwise raise an error naming the transition and the candidates. A schema and its reverse form one face group.
6. **Walk sampler + cell identity** (D8), vectorized (per-relation CSR/CSC index tensors, all anchors of a schema at once; no Python loop over nodes or anchors), replacing `enumerate_metapath_faces` and `build_typed_path_face_groups` for all three tasks. Config `data.path_sampler: "walk" | "dfs" | "uniform_instance"` (default `"walk"`; `"dfs"` = current behaviour; `"uniform_instance"` = ablation, implemented for schemas with L ≤ 2 only, error otherwise). `r` is the existing `face_sample_k` / `sample_per_anchor`. `FaceGroup` gains the merged weight `m_f`.
7. **Protocol:** `data.resample_paths: bool` (default `true`). Training draws use seed `base_seed + epoch`; validation and test always use the draw with `base_seed`. Wire this into the node-classification, link-prediction and graph-classification trainers with the smallest explicit change.
8. **Link prediction under tied facts:** a held-out positive removes its single tied fact; extend the leak checks so that no held-out fact, in either orientation, appears in the message graph or on any sampled path, at train and at evaluation.
9. **Ablation switch** `ablation.direction_blind_maps: bool` (same frame both sides and no side code for **every** relation; the analogue of DSNN's `q = 0`). Validate all new keys in the dataclasses; reject combinations that make no sense (e.g. `fact_layer: untied` with explicit `-` hops).

**Out of scope** (report, do not change)
- D2 schema selection (schema gates, pruning), mini-batching, new datasets, multi-seed accuracy runs.
- Asymmetric in/out coupling strengths (would leave the self-adjoint operator class; plan §9.1).
- Eq. 9 (neighbourhood context) — open question 5 in the plan.

### Unit order (commit each unit on branch `D1_add` after its tests pass; do not push)
| Unit | Content | Gate |
|---|---|---|
| U1 | D9 O(L) path term | T1; golden re-capture rule below |
| U2 | Signed transport + orientation check | T2, T3 |
| U3 | Fact layer (one definition) + shared roles for `ρ(r) = r` + `direction_blind_maps` | T4, T5, T6, T16 (part) |
| U4 | Schemas with direction | T10 |
| U5 | Walk sampler, cell identity, `m_f`, `uniform_instance` | T7, T8, T9, T13, T15, T16 (part) |
| U6 | Resampling protocol in the three trainers | T12 |
| U7 | Link-prediction leak checks under tied facts | T11 |
| U8 | Measurements, report, CLAUDE.md/AGENTS.md state | — |

**Golden test rule (explicitly authorized here, an exception to Part A's "no loosened tolerances").** U1 changes only the summation order of an exact identity, so `tests/test_task_heads.py::NodeClassificationGoldenTests` may no longer be bit-identical. Before U1, run it and record the result. After U1, measure the max abs difference of logits and gradients against `docs/heads_evidence/v2_golden.pt`. If ≤ 1e-5 (float32), re-capture the golden file with `docs/heads_evidence/v2_golden.py`, commit it separately with the measured difference in the commit message, and from then on require bit-identity again. If > 1e-5, stop and report. After U3–U6 the golden test must still pass bit-for-bit with `fact_layer: untied`, `path_sampler: dfs`, `resample_paths: false`.

### Pre-step: inventory
1. Every consumer of `FaceGroup`, `PathTransportOperator`, `enumerate_metapath_faces`, `build_typed_path_face_groups`, `validate_face_orientation`, `flatten_edges`, `edge_type`/`relation_names`/`relation_tuples` (heads, trainers, analysis, scripts, notebooks, tests), with `file:line`.
2. **Per dataset** (ACM, DBLP, IMDB; every link-prediction dataset; every TU graph-classification dataset): the computed `ρ`, the number of facts before and after tying, unmatched edges per relation, reciprocal facts inside relations with `ρ(r) ≠ r`, and self-loops. Expected for HGB (VERIFIED-RUN in the plan): `ρ` pairs all 10 relation pairs with 0 unmatched edges; `ρ(cite) = ref`, so ACM's 25 reciprocal citations stay two facts each; 8 self-loops in `cite`; facts after tying ACM 273,936, DBLP 119,783, IMDB 43,321. TU graphs: expected `ρ(r) = r` (UNVERIFIED). Report any relation that has a near-but-not-exact partner (it stays untied by definition; do not add a threshold).
3. Where each trainer builds or reuses face groups, so U6 can be minimal.

### Tests to add (`tests/test_paths_and_facts.py`, unittest)
| ID | Check | Tolerance / rule |
|---|---|---|
| T1 | O(L) `PathCellTerm.apply` equals a dense reference `Σ w m Bᵀ(I − 11ᵀ/n)B` built from materialized `T_v`; mixed L ∈ {2, 3, 4}; per-edge and per-side scales; signed hops | float64 ≤ 1e-10 |
| T2 | Signed transport equals the dense product with `F_tgt⁻¹F_src` on −1 hops; functoriality holds on every boundary edge with mixed signs; `apply_T` is the exact adjoint | float64 ≤ 1e-10 |
| T3 | Reversal invariance: building a cell from the reversed fact sequence (base vertex `v_L`) gives the same `L_f` under tied facts | float64 ≤ 1e-10 |
| T4 | One definition on fixtures: exact mirror pair → one cell per fact; self-inverse relation → one cell per unordered pair; reciprocal facts of a relation with `ρ(r) ≠ r` stay two cells; a pair missing one reverse edge is not tied and its edges stay facts. Real HGB data: the expected counts of inventory item 2. `untied` is rejected by the tuning-config check | exact |
| T5 | Self-inverse relation (`ρ(r) = r`): flipping the stored orientation of any subset of its cells leaves normalized `Δ₀` unchanged. **Negative control:** with separate source/target roles it changes | ≤ 1e-10 vs > 1e-3 |
| T6 | Sections preserved with signed hops: on a tree with mixed-direction path hops, `L_P x = 0` for every section and `dim ker(Δ₀ + γL_P) = dim ker Δ₀` | eigenvalue threshold 1e-9, stated before running |
| T7 | Sampler distribution: on the 3-author × 30-paper fixture and the 300-paper hub fixture, empirical intermediate frequencies over ≥ 2,000 draws match the walk probabilities (state the test and threshold before running). **Negative control:** `path_sampler: dfs` fails it | pre-registered |
| T8 | Unbiasedness: on a small graph, the mean of `L̂_P x` over ≥ 2,000 draws approaches `L_P^target x` computed by exact enumeration with walk probabilities (pre-register the relative-error threshold). **Negative control:** merging by dropping duplicates instead of summing weights is biased | pre-registered |
| T9 | Identity: no two cells share a key; a path and its reversal map to one key with summed weight; paths through different reciprocal facts get different keys | exact |
| T10 | Schemas: ACM paper→paper citation chain (`cite +, cite +`) and co-citation (`cite +, cite −`) are distinct groups; an ambiguous transition without `hops` raises; a schema and its reverse share one group | exact |
| T11 | Link prediction: no held-out fact, in either orientation, in the message graph or on any sampled path (train and eval). **Negative control:** injecting a held-out fact is caught | exact |
| T12 | Protocol: evaluation draw identical across calls and epochs; training draws differ between epochs; same seed reproduces | exact |
| T13 | Stability with `m_f` and signs: `λ_max(D^{-1/2} L D^{-1/2}) ≤ 2 + 1e-9` over ≥ 50 random fixtures | — |
| T14 | Regression: with `fact_layer: untied`, `path_sampler: dfs`, `resample_paths: false`, the golden test passes (after the U1 rule) | bit-for-bit |
| T15 | Vectorized sampler: no Python loop over nodes or anchors (show the code); time grows at most ~linearly in anchors (10× anchors → ≤ 15× time, CPU, pre-registered) | pre-registered |
| T16 | Ablations do what they say: `fact_layer: untied` reproduces current cells and edges; `direction_blind_maps` makes `Δ₀` invariant to flipping any cell; `uniform_instance` samples instances uniformly (L ≤ 2) and raises for L > 2 | exact or ≤ 1e-10 |

**Mutation check** (show the targeted tests fail for each):
- −1 hop uses `Q` instead of `Qᵀ`;
- duplicates dropped instead of weights summed;
- a self-inverse relation keeps separate roles (side code or two frames);
- identity keyed by node sequence instead of facts;
- evaluation draw reseeded per call;
- O(L) term uses `x_i` instead of `s_i² x_i` (caught under per-side scales);
- the reversed orientation of a held-out fact not removed;
- `m_f` applied in `L_P` but not in `D`.

### Measurements (allowed without approval)
For ACM, DBLP and IMDB, before (`untied` + `dfs`) and after (`tied` + `walk`), d = 32, factorized k = 2:
- facts and `Δ₀` edges; cells, incidences, duplicates merged;
- mean distinct intermediates per anchor and schema;
- sampling time per draw (CPU and GPU if used);
- one forward + backward: peak memory and wall time (5 repeats after 1 warm-up), plus parameter count.

Pre-register the expected `Δ₀` ratio (≈ 0.5) and the transport-step ratio of U1 (L = 2: 6 → 4).

**Smoke run:** ACM, single seed, ≤ 20 epochs, defaults after U6. Log loss, η, γ, mean/max `w_f`, mean `m_f`, `η·λ_max`. **No accuracy claims.**

### Adversarial self-review checklist (D7–D9)
- [ ] O(L) term equals the dense reference for every map family and both signs (T1); transport steps per path are 2L (show the count).
- [ ] Every −1 hop uses the adjoint step, in `apply`, `apply_T`, holonomy, Horner and prefix (T2, mutation).
- [ ] `ρ` comes only from exact edge-set equality on the data: no names, no thresholds, no config key; near-partners are reported, not tied (T4).
- [ ] `fact_layer: untied` cannot enter any tuning search space (T4).
- [ ] Self-inverse relations are orientation-invariant (T5 with its control).
- [ ] Theory still holds: sections preserved (T6), `λ_max ≤ 2` with `m_f` (T13), `m_f` in both `L_P` and `D`.
- [ ] The sampler has a stated distribution and the estimator is unbiased for the stated target (T7, T8 with controls).
- [ ] Identity is by facts; weights are summed, never dropped (T8, T9).
- [ ] No leakage of held-out facts in either orientation (T11 with its control).
- [ ] Evaluation is deterministic; training resamples (T12).
- [ ] Old behaviour reachable with `untied` + `dfs` + no resampling, bit-for-bit after the U1 re-capture (T14).
- [ ] Every new config key is validated; no hard-coded hyperparameters.
- [ ] Each test would fail if its property were broken (mutation check, 0 vacuous).

### Stop conditions
Stop and report instead of continuing if:
- the U1 golden difference exceeds 1e-5;
- T2, T6 or T13 fails after 3 diagnosed attempts;
- a default schema needs a relation that the exact definition leaves untied;
- the sampler cannot be vectorized without enumerating all instances;
- leakage freedom (T11) cannot be guaranteed;
- a needed change falls outside the scope above.

Write the report to `docs/H2S-NN_D7_report.md` in the Part A format.

---

## Part E — Task card D10: cooperative coupling with a real spectrum

### Goal
Let influence be asymmetric (a node can listen without broadcasting, or broadcast without listening) while the diffusion operator keeps a real spectrum and its stability guarantee. Spec: `H2S-NN_v2_plan.md` §10 (operator, theory, limits, decision rule) and §11 item 1. Start only after Part D is committed; work on the same branch.

Read first: plan §10, §11, §4.4; `docs/H2S-NN_D7_report.md`; `src/models/h2snn_v2.py` (diffusion loop, `η`, `γ`, readout inputs), `src/models/vertex_path_laplacian.py` (normalized operator, `D`), the power-iteration code used for `η·λ_max` in the D4/D7 smoke runs.

### Mathematical specification
```
Δ̂   = D^{-1/2} (Δ₀ + γ L_P) D^{-1/2}                 # unchanged
g_v = σ(a_gᵀ h₀_v + β_g),   b_v = σ(a_bᵀ h₀_v + β_b)   # listen / broadcast gates in (0, 1)
x  ← x − η · G Δ̂ B x                                 # G = diag(g) ⊗ I_d, B = diag(b) ⊗ I_d; then layer dropout as now
```
- Two linear heads on `h₀` (computed once, like the maps). Initialize the biases so every gate starts at `model.coupling_gate_init` (new validated key, default 0.95) and the weights small, so training starts near the symmetric model; state the init.
- Effective strength from v to u is `g_u b_v`. `g_u = 0` ⇒ u's state does not change; `b_v = 0` ⇒ no node's update depends on `x_v`.
- **Theory the code must satisfy:** `G Δ̂ B` is similar to `(GB)^{1/2} Δ̂ (GB)^{1/2}` (PSD), so its eigenvalues are real and lie in `[0, 2·max g b]`; fixed points satisfy `B x ∈ ker Δ̂`. For `η·λ_max` logging, compute `λ_max` by power iteration on the symmetric similar form `(GB)^{1/2} Δ̂ (GB)^{1/2}`.

### Scope
**In scope**
1. Gate heads and the gated update in the shared encoder, so all three tasks get it.
2. `ablation.coupling: "cooperative" | "listen_only" | "symmetric"` (default `cooperative`). `listen_only`: `B = I`. `symmetric`: no gate heads instantiated, all gates 1 — exactly the current model.
3. Logging per epoch: mean and min of `g` and `b`, and `η·λ_max`.
4. Validation of the new keys; the gate init is the only new model hyperparameter. Neither `coupling` value nor `coupling_gate_init` enters a tuning search space.

**Out of scope** (report, do not change)
- Per-edge directional strength (breaks the real spectrum; plan §10.4).
- Gates or maps recomputed from `x_t` at each step, per-step channel mixing or nonlinearity (plan §11 item 2; separate decision).
- D2, mini-batching, new datasets, accuracy experiments.

### Tests (`tests/test_cooperative.py`, unittest; keep them few)
| ID | Check | Tolerance / rule |
|---|---|---|
| T1 | `coupling: symmetric` reproduces the current golden outputs of every task that has a golden test | bit-for-bit |
| T2 | The gated operator equals a dense reference `G Δ̂ B` built independently, with mixed path lengths and signed hops | float64 ≤ 1e-10 |
| T3 | Over ≥ 50 random fixtures with gates in [0.01, 1]: max \|Im λ\| ≤ 1e-9 relative to max \|λ\|, and `λ_max ≤ 2·max(g b) + 1e-9`. **Negative control:** a CSNN-style composed operator `(Δ_in)ᵀ Δ_out` on the same fixtures has \|Im λ\| > 1e-3 on at least one | — |
| T4 | Cooperative behaviour, one step: `g_u = 0` ⇒ `x_u` unchanged exactly; `b_v = 0` ⇒ `∂x₁(w)/∂x₀(v) = 0` for every w ≠ v; with `g_u ≠ g_v` the Jacobian blocks u←v and (v←u)ᵀ differ. Control: all gates 0.5 gives non-zero blocks | exact zero vs > 1e-8 |
| T5 | Fixed points: for sections `s` of `Δ̂`, `x = B⁻¹ s` gives `G Δ̂ B x = 0`, and nothing outside that set does (eigenvalue cut 1e-9, stated before running) | float64 |
| T6 | Gradients reach both gate heads | non-zero |

**Mutation check (3):** gates on swapped sides (`B Δ̂ G`) → T4 fails; gates without the sigmoid (unbounded) → T3's bound fails; `B` applied but `G` missing → T4 fails.

### Measurements (allowed without approval)
- ACM, d = 32, factorized k = 2, defaults after Part D: one forward + backward for `cooperative` vs `symmetric` — peak memory and wall time (3 repeats after 1 warm-up), parameter count. Pre-register: overhead ≤ 5% time, ≤ 2% memory.
- Smoke run: ACM, single seed, ≤ 20 epochs, `cooperative`. Log loss, η, γ, mean/min `g` and `b`, mean `w_f`, `η·λ_max`. **No accuracy claims.**

### Adversarial self-review checklist (D10)
- [ ] `symmetric` is bit-identical to the current model (T1).
- [ ] Spectrum real and bounded with gates; CSNN-style control complex (T3).
- [ ] Listen and broadcast behaviours exact (T4); the swapped-sides mutation is caught.
- [ ] No new tuning knob besides `coupling_gate_init`; `coupling` is not in any search space.
- [ ] Report section "Assumptions imposed for theory and reality checks" (Part A) is filled in.
- [ ] Each test would fail if its property were broken (3 mutations, 0 vacuous).

### Stop conditions
Stop and report instead of continuing if: `symmetric` is not bit-identical (T1); T3 or T5 fails after 3 diagnosed attempts; the smoke run diverges (`η·λ_max ≥ 2` or loss non-finite); a needed change falls outside the scope.

Commit with the message "D10 complete" and push. Write the report to `docs/H2S-NN_D10_report.md` in the Part A format, at most two pages; long output goes to `docs/D10_evidence/`.

---

## Part F — Task card D11: final model, learning fix, assumption tests, core-claim screen

### Goal
Make D10's gates a learned part of the final model (plan §12), prove in code that the update is a metric gradient flow and that its gate form is exactly the reversible (energy-admitting) class (Prop. 4), finish D7, build the `pair_attention` variant that **tests assumption A4**, and run the core-claim screen with pre-registered triggers. Spec: `H2S-NN_v2_plan.md` §12.0 (design rationale: assumptions A1–A5, properties P1–P5), §12.2 (theory incl. Prop. 4), §12.3 (alternatives and the assumption each tests), §12.6 (screen, triggers) and §9.5 (D7 decisions). D10 is already implemented (`16f6ad3`, `docs/H2S-NN_D10_report.md`). Reviewer-facing context: `docs/possible_reviewer_qa.md` (read Part 1 and Q2.11–Q2.14; do not edit it).

### Units
| Unit | Content | Gate |
|---|---|---|
| U0 | Commit the uncommitted spec edits (plan, prompt, D2 card, `possible_reviewer_qa.md`) as "docs: specs for D7–D11". New branch `tier0-screen` off `D1_add` | — |
| U1 | **Finish D7:** option A (baselines pinned to `untied` + `dfs` + no resampling in `bundles.py::_base_config`, with a test that bundle data stays bit-identical); flip the H2S-NN defaults to `tied` / `walk` / `resample_paths: true`; three-mutation check; write `docs/H2S-NN_D7_report.md` (short). First run the baseline path-builder check: any baseline that consumes path instances from our builders gets the walk sampler while keeping the stored untied graph | D7 tests; bundle bit-identity |
| U2 | **Learning fix:** gate heads, `θ_η`, `θ_γ` in their own optimizer group (lr = 10 × base, weight decay 0; check whether a scalar group already exists and extend it); `model.coupling_gate_init` default 0.95 → 0.6; `coupling: cooperative` stays the default (gates always on) | T-F1 |
| U3 | **Theory tests** (`tests/test_cooperative.py`): T-F2, T-F3, T-F3b | T-F2, T-F3, T-F3b |
| U4 | **Diagnostics:** per step, normalized energy `⟨x, Δ̂x⟩/‖x‖²` (edge and path-cell parts); per epoch, η, γ, gate mean/min; offline script: `dim ker Δ̂` (cut 1e-6) and a linear probe of the labels on the projection onto `ker Δ̂` for a trained model | smoke run |
| U4b | **`coupling: pair_attention`** — the A4 test variant, specification below (`tests/test_pair_attention.py`) | T-F4 … T-F7; 3 mutations |
| U5 | **Core-claim screen** (approved): ACM, DBLP, IMDB × {full, `ablate_no_path_cells` (γ = 0), `endpoint_edges`, `h0_only`, `pair_attention`} × 3 seeds; one fixed config, no tuning; parameter count per row; + one reference baseline (SeHGNN if it runs; otherwise report the strongest that runs before starting). Report against **every** trigger of plan §12.6, including the A4 rows | triggers |

### `pair_attention` specification (U4b)
Purpose: change **one** thing relative to D11 — how influence strength is formed — so the comparison tests A4 and nothing else. Same `h₀`, same maps (D1/D1a), same tied facts and walk-sampled cells (D7–D8), same single block of K steps, same head, same η. Gate heads are not instantiated. **Do not** add per-step weight matrices, nonlinearities or attention recomputed from `x_t` (that would also change A5).
```
scores, computed once from h₀ (like the maps), GATv2-style:
  s(u←e)  = aᵀ LeakyReLU(W [h₀_u ‖ h₀_v ‖ rel_emb(ψ_e) ‖ dir_emb(u's role on e)])   # e = fact joining u and v
  s(u←f)  = aᵀ LeakyReLU(W [h₀_u ‖ path_latent_f ‖ pos_emb(u's position in f)]) + log m_f
  s(u←u)  = aᵀ LeakyReLU(W [h₀_u ‖ h₀_u ‖ self_emb])
α_u = softmax over {u} ∪ {facts at u} ∪ {cells containing u}                        # per receiving node; self-loop included, as SheafAN Eq. 3
(P x)_u = α_uu x_u + Σ_e α_ue Q_{u←v} x_v + Σ_f α_uf T_uᵀ z_f                       # Q_{u←v}: the same signed orthogonal transport Δ₀ uses (no c_e); z_f as in D9
x ← x − η (x − P x)
```
- Confirm the GATv2 scoring form against `references/GATv2.pdf` before implementing (VERIFIED-SOURCE in the report); single head; parameter count reported next to D11's.
- `path_latent_f` is the existing path encoder used for `w_f`; `w_f`, `c_e²` and the gates do not enter this variant (attention replaces them).
- The variant applies wherever the encoder runs, but U5 uses only node classification.

### Tests
| ID | Check | Tolerance |
|---|---|---|
| T-F1 | After 50 full-batch epochs on a toy fixture with a learnable signal, η and every gate head have moved by > 0.05 from their init (the old setting, 0.95 init without the group, is the control and moves < 0.05) | pre-registered |
| T-F2 | Metric gradient identity: with `E(x) = ½⟨Bx, Δ̂Bx⟩` and `M = B G⁻¹`, `M⁻¹∇E` (autograd) equals `G Δ̂ B x` | float64 ≤ 1e-10 |
| T-F3 | `E` nonincreasing over ≥ 50 random fixtures × 50 steps for η ≤ 1; **control:** η = 3 makes `E` increase on at least one fixture | exact sign |
| T-F3b | **Prop. 4 (factorization):** from the D11 operator (edges + path cells, signed hops, random gates), extract the effective pairwise weights `w(u←v)` (scalar factor of each off-diagonal block); solve for a per-node metric by least squares on `log m_u − log m_v = log w(v←u) − log w(u←v)`; the residual is ≤ 1e-10 and `M·A` is symmetric to ≤ 1e-10. **Control:** the same check on `pair_attention`'s operator gives residual > 1e-3 on at least one fixture (also proves the A4 test is not vacuous) | float64 |
| T-F4 | `pair_attention`: every row of α (self + facts + cells) is positive and sums to 1 | ≤ 1e-12 |
| T-F5 | Sections preserved: for a global section `x` (consistent on all facts and cells; tree fixture with mixed-direction path hops), `P x = x` | float64 ≤ 1e-10 |
| T-F6 | Non-expansive (default `map_scale: per_edge`, so every transport is orthogonal): `max_u ‖x_u‖` nonincreasing over ≥ 50 random fixtures × 50 steps for η ∈ (0, 1]. **Control:** replacing the softmax by unnormalized `exp(s)` increases it on at least one fixture | exact sign |
| T-F7 | Dense reference: build `P` as an explicit matrix from α and materialized transports (`Q`, `T_v`, `z_f` written out); the operator equals it on fixtures with mixed path lengths and signed hops. The state dict differs from D11's only by the attention parameters added and the gate heads removed | float64 ≤ 1e-10; exact |

**Mutation check (U4b, 3):** self-loop omitted from the softmax → T-F4 or T-F5 fails; `Q` used instead of `Qᵀ` on a −1 hop → T-F5 fails; `log m_f` omitted → a targeted test showing duplicate-merged cells get the wrong weight fails (write it).

### Rules
- Ablation and test rows (`symmetric`, `listen_only`, diagonal maps, `ablate_no_path_cells`, `endpoint_edges`, `h0_only`, `pair_attention`) are evidence for the paper, never tuned, never defaults. `pair_attention` must be rejected by the tuning-config check like the others.
- `pair_attention` tests assumption A4; it is not a model option. Its consequence is fixed in plan §12.6 before any result: if it beats full by > 1 std on ≥ 2 of 3 datasets, report it and stop before Tier 0 so the author confirms with 10 seeds.
- Do not implement multi-block structure (maps or attention recomputed from `x_t`, per-step mixing or nonlinearity); it returns only through the `h0_only` trigger in §12.6.
- Save tokens: long output to `docs/D11_evidence/`; full suite once per unit; skip the independent sub-agent review.

### Stop conditions
T-F2, T-F3, T-F3b, T-F5 or T-F6 fails after 3 diagnosed attempts; the bundle bit-identity test fails; no strong baseline can run (report before the screen); the screen hits the "path cells don't beat γ = 0 or endpoint_edges" trigger or the `pair_attention` trigger (stop and report, do not continue to Tier 0).

Commit per unit; push when the screen report is written. Report: `docs/H2S-NN_D11_report.md` (Part A format, at most two pages, including "Assumptions imposed for theory and reality checks", and one line per assumption A1–A5 saying which screen row tested it and the outcome).

---

## Part G — Task card D12: pipeline gate, relational context, filter-bank encoder

### Goal
Implement methodology spec v1.2 (§3, §4): one encoder (gated Chebyshev filter bank over the vertex–2-cell sheaf) and geometry computed once from typed neighbourhood context. **First** find why every variant, including the `h₀`-only MLP, scores 6–10 points below published simple models (spec §6 pipeline gate). Spec: `H2S-NN_methodology_spec.md` v1.2 (authority); plan D12 row. Work on a new branch `d12-encoder` off `tier0-screen-d11`.

### Specification
```
c_v = concat over (relation, role) of mean_{u ∈ N_{ψ,role}(v)} h₀_u      # zeros if no neighbour; facts only (D7), both roles
geometry inputs: [h₀_v, c_v] replace h₀_v in the map MLPs (reflector, scale), the gate heads, and the path encoder
A   = G Δ̂ B                                                                 # existing operator (D4, D9, D10); no change
T_0 = h,  T_1 = (A − I) h,  T_{k+1} = 2 (A − I) T_k − T_{k−1}               # applied to vectors, never materialized
x   = Σ_{k=0..K} T_k(A − I) (h₀ W_k)                                        # W_k ∈ R^{d×d} acts on stalk coordinates (state is [N, d])
W_k = V_k / max(1, Σ_j ‖V_j‖₂)                                              # exact spectral norm of d×d (torch.linalg.matrix_norm(ord=2))
init: V_k = θ_k I with θ = Chebyshev coefficients of (1 − η₀(1 + ξ))^K, η₀ = model.eta_init  → starts exactly at D11 (Prop. 7)
y   = head([x, h₀])
```
- Config: `model.encoder: "filter_bank" | "first_order"` (default `filter_bank`; `first_order` = current D11 `diffuse`, the nested C8 arm); `model.geometry_context: bool` (default `true`; `false` = C9 arm). `K` keeps its meaning as the number of applications of `A`. `η` exists only under `first_order`. `pair_attention` stays as the C4b arm, unchanged. None of these switches may enter a tuning search space (extend the existing rejection check).
- `W_k` and the scalar group: `V_k` in the normal group; gates and `γ` stay in the scalar group (D11 learning fix).
- Self-loop facts: Lemma N holds only for nodes without them; document the count per dataset and keep the operator as is.

### Units
| Unit | Content | Gate |
|---|---|---|
| G0 | **Pipeline-correctness gate** (spec §6; done 2026-11-08, `20d9756`: data checks PASS, published-number reproduction not achievable without HetSheaf's unpublished sweep winner): (a) compare our train/val/test node ids with the official HGB splits for ACM, DBLP, IMDB; (b) run one reference baseline (R-GCN or SimpleHGN, whichever runs) on our pipeline, single seed, its published configuration; pre-register the tolerance (e.g. within 2 published std). If it fails, diagnose the cause (splits, features, label mapping, metric, early stopping, preprocessing) with evidence before any fix. **Stop and report** after G0 whatever the outcome | report |
| G0b | **Gap localization ladder** (after G0, before G1; author-approved 2026-11-08). All on our pipeline, ACM and DBLP, metric and selection as in G0, logging train/val/test macro-F1 and best epoch. Rungs: **L0** R-GCN feature-free (G0 arm O), 5 seeds (noise level + internal reference); **L1** one features-using reference on our pipeline (HetSheaf's GCN code and committed config), 3 seeds; **L2** `h0_only` through `trainer.py`, raw vs row-normalised input features, 3 seeds each; **L3** sheaf-free H2S-NN through `trainer.py` (identity restriction maps, `γ = 0`, `coupling: symmetric`, D11 update; add an `ablation.identity_maps` diagnostic switch if none exists, never tuned), 3 seeds; **L4** full D11 through `trainer.py`, 3 seeds. Pre-registered reading (write it to `docs/D12_evidence/g0b_preregistration.md` before running): (i) L2 row-normalised > raw by > 2 points → feature-preprocessing defect: fix it for every model (pipeline fix, not a design change); (ii) L3 within 2 points of L0 but L4 more than 2 below L3 → the learned sheaf (maps, gates, 2-cells) hurts: design-level, stop; (iii) L3 more than 2 below L0 → propagation or training (trainer settings, reach of K steps, readout): diagnose with train-F1 (low = underfitting, high train / low val = overfitting); (iv) L1 well below its published band on our pipeline → feature path of the baselines also suspect. **Stop and report after G0b** | report |
| G1 | Relational context `c` and `geometry_context` switch | T-G7 |
| G2 | Filter-bank encoder and `encoder` switch | T-G1 … T-G6, T-G8 |
| G3 | Cost gate: ACM at the default `d`, `filter_bank` vs `first_order`, one forward + backward, peak memory and time (3 repeats after 1 warm-up), parameter count; pre-registered overhead ≤ 25% time, ≤ 10% memory | measurement |
| G4 | Report `docs/H2S-NN_D12_report.md` (Part A format, ≤ 2 pages) | — |

### Tests (`tests/test_filter_bank.py`, unittest, float64 fixtures with path cells and signed hops)
| ID | Check | Tolerance |
|---|---|---|
| T-G1 | Spectrum of `A` real and in `[0, 2·max g b]` on ≥ 50 random fixtures | ≤ 1e-9 |
| T-G2 | Prop. 5: gain of the filter in the metric `M = BG⁻¹` (nodes with nonzero gates) ≤ 1 + 1e-9 for random `V_k`. **Control:** without the normalization the gain exceeds 1 on at least one fixture | — |
| T-G3 | Prop. 6: with `g_u = 0`, every block of the filter `u ← j` (j ≠ u) is 0; with `b_v = 0`, every block `i ← v` (i ≠ v) is 0. **Control:** gates 0.5 give a non-zero block | exact 0 vs > 1e-8 |
| T-G4 | Prop. 7: with the D11 initialization, `filter_bank` output equals `first_order` output for η ∈ {0.25, 0.5, 0.9}, K ∈ {2, 5} | ≤ 1e-10 |
| T-G5 | Lemma N: diagonal blocks of `Δ̂` equal `I` for nodes without self-loop facts | ≤ 1e-12 |
| T-G6 | Matrix-free filter equals a dense reference built from materialized `A` | ≤ 1e-10 |
| T-G7 | Context: changing one neighbour's `h₀` changes that node's maps and gates under `geometry_context: true` and not under `false`; `c` uses facts only and respects roles | exact |
| T-G8 | Gradients reach every `V_k`, the gate heads, `γ`, the map MLPs | non-zero |

**Mutation check (3):** normalization of `V_k` removed → T-G2 fails; recurrence written with `A` instead of `A − I` → T-G4 fails; gates applied as `B Δ̂ G` → T-G3 fails.

### Rules
- No protocol runs, sweeps or multi-seed experiments (they need the frozen spec and approval). G0's single-seed baseline run and G3's measurements are allowed.
- Comparison arms (`first_order`, `geometry_context: false`, `pair_attention`, `symmetric`, `ablate_no_path_cells`, `endpoint_edges`, `h0_only`) are evidence, never tuned, never defaults.
- Save tokens: long output to `docs/D12_evidence/`; full suite once per unit.

### Stop conditions
G0 (always stop and report); T-G2, T-G3 or T-G4 fails after 3 diagnosed attempts; the cost gate is exceeded; a needed change falls outside this card.

Commit per unit on `d12-encoder`; do not push until the author has read the G0 report.

---

## Part H — Task card D2: WISP (Walk-sampled, Instance-weighted Sheaf Paths), the replacement of AMPSS

### Goal
Implement the meta-path mechanism of methodology spec v1.3 §3.2: schema-free candidate instances, a stated target measure with an unbiased walk-level defensive importance-sampling estimator, instance-conditioned cell weights `w_f = exp(g(ξ_f))` with the C2 weight ladder, and evaluation-only schema read-outs. Remove nothing that the comparison rows need. Spec v1.3 (§2 R2, R8, R12; §3.2; §4 Props. S, K, R, E; §5 C2, C2b) is the authority; plan D2 row and §2.5 give the reasoning. This card replaces `../docs/H2S-NN_D2_task_card.md`.

### Preconditions (check first; stop and report if any fails)
1. **Part G is closed.** Another session works on Part G in this folder. Do not start while `git status` shows uncommitted changes in `src/` or `tests/` that are not yours, or while the Part G report has open units. Ask the author if unsure. Never edit, stash, reset or commit another session's changes.
2. Branch `wisp` off the last Part G commit.
3. **No real-data accuracy runs in this card.** The spec §6 WISP gate requires the G0b gap (h1–h3) to be explained first. Synthetic known-answer tasks (H5) and smoke/cost runs (H6) are allowed.

### What changes, in one table
| Part | v1 AMPSS | Interim (D8, now) | WISP (this card) |
|---|---|---|---|
| Candidates | Stage I schema scoring, ≥ 18 settings | fixed literature schema list per dataset | none needed: walk-first, schema-free |
| Sampling | DFS with quota (no distribution) | `r` uniform-neighbour walks per (anchor, schema), rejection of non-simple walks | stated target `p` (uniform length, uniform action, uniform unvisited neighbour), walk-level mixture `½p + ½q`, weight ≤ 2 |
| Weight | Stage II instance gate under Stage I | `γ · sigmoid(g(ξ_f))` | `exp(g(ξ_f))`, γ removed (it is `g`'s bias) |
| Schemas | selected | given | read out (`I_π`, `I_σ`, `I_{v,σ}`, `ΔL_σ`, overlap ratio, Kendall τ) |
| Knobs | ≥ 18 | `r`, the list | `ℓ_max` (support), `r` (precision) |

### Specification
```
anchors        = all nodes of the target type (node classification)
walk ω         : draw ℓ ~ Uniform{2..ℓ_max}; for j = 1..ℓ:
                   A_j = {(relation, direction) with ≥ 1 unvisited neighbour of v_{j−1}}     # self-avoiding by exclusion
                   if A_j = ∅: discard ω                                                     # sub-probability target
                   a_j ~ Uniform(A_j)  [target]   or  ∝ q_act(type(v_{j−1}), a) on A_j  [proposal]
                   v_j ~ Uniform(unvisited neighbours of v_{j−1} via a_j)
p(ω)           = 1/(ℓ_max − 1) · Π_j 1/|A_j| · 1/|N_j|
q(ω)           = q_len(ℓ) · Π_j q_act(t_j, a_j)/Σ_{a ∈ A_j} q_act(t_j, a) · 1/|N_j|
coin per walk  : with prob ½ draw the whole walk from p, else from q        # walk level, NOT per step
m_f            = Σ_{walks ω → cell f} (1/r) · p(ω) / (½p(ω) + ½q(ω))       # D8 identity and summed merge
w_f            = exp(g(ξ_f)),  ξ_f = [geom(v_0), …, geom(v_ℓ), rel_emb(a_1), …, rel_emb(a_ℓ)]   # geom = [h₀, c] (D12)
g              = Linear_ℓ → GELU → Linear(hidden, 1) with bias b;  b in the scalar optimizer group;  b₀ = log(path_weight_init)
L_P            = Σ_f w_f m_f L_f   (D9 O(L) term);   Δ̂ = D^{-1/2}(Δ₀ + L_P)D^{-1/2}   (no γ)
q update       : at epoch start, q_len and q_act ∝ previous epoch's Σ w_f m_f per length and per (type, action) used; detached
evaluation     : one fixed draw from p only (q = p), base seed; identical cells for every comparison arm
```
**Ladder** (`ablation.path_weight_input`, restrictions of `g`, same cells, same transports):
| Value | Input of `g` | Must hold |
|---|---|---|
| `uniform` | none (`w_f = exp(b)`) | replaces `ablation.uniform_path_gate` |
| `schema` | relation sequence and length only | equal `w` for cells with the same relation sequence, whatever their nodes |
| `endpoints` | `geom(v_0)`, `geom(v_ℓ)`, relation sequence | invariant to intermediate nodes; hidden width chosen by a fixed rule so its parameter count is within 5 % of `full` |
| `full` | all nodes and the relation sequence | default |

**Config (names are PROPOSALS; check collisions in `src/configs/config.py` first; strict loading, no shims):**
- `data.path_source: "schemas" | "wisp"` — `schemas` = D8 interim sampler with the literature presets (the comparison row). Default stays `schemas` during this card; the default flip is an author decision after H6.
- `data.wisp_max_length` (int ≥ 2, default 4), `data.wisp_walks_per_anchor` (int ≥ 1; default chosen in H1 so that the incidence count matches the `schemas` run on ACM; report the rule), `data.wisp_proposal: "uniform" | "adaptive"` (default `uniform` until the H3 gate).
- `model.path_weight: "sigmoid_gamma" | "exp"` (`sigmoid_gamma` = current behaviour, bit-identical; under `exp` there is no `θ_γ` and `model.gamma_init` is rejected), `model.path_weight_init` (> 0, default 0.5, only under `exp`).
- `ablation.path_weight_input: "full" | "endpoints" | "schema" | "uniform"`; `ablation.uniform_path_gate` is removed (its behaviour is `uniform`).
- Every key above except `wisp_walks_per_anchor` goes into `reject_never_tuned_settings`. Reject invalid combinations loudly (e.g. `path_weight_input ≠ full` together with `endpoint_edges` or `free_path_maps`, `wisp` with link prediction or graph classification).
- Link prediction and graph classification keep their schema-free fixed-length sampler (`build_free_path_groups`); WISP for them is out of scope. `path_weight: exp` must still run there (smoke).

**Grouping.** Group cells by **length**, not by signature (signatures can number in the hundreds; one group each would loop in Python). `FaceGroup.relation_index` becomes genuinely per cell (today the encoder reads row 0, `h2snn_model.py` `_path_latent`); `PathEncoder` gathers relation embeddings per cell. Keep a per-cell signature id and a signature table for the read-outs.

### Units
| Unit | Content | Gate |
|---|---|---|
| H0 | Preconditions; inventory of every consumer of `FaceGroup`, `build_schema_path_groups`, `PathEncoder`, `_path_weights`, `theta_gamma`/`gamma()`, `uniform_path_gate`, `face_lengths`, `path_weight` (with `file:line`); update the D2 rows of `AGENTS.md` and `CLAUDE.md` to "WISP (spec v1.3 §3.2)". Pre-register H1–H6 expectations in `../docs/WISP_evidence/preregistration.md` | inventory written before code |
| H1 | Sampler with `q = p`: target measure, self-avoidance by exclusion with exact counts `|A_j|`, `|N_j|`, walk probabilities in float64, identity/merge (reuse D8), length grouping, per-cell relation sequences, `path_source` switch, determinism protocol | T-H1, T-H2 (q = p), T-H4, T-H9, T-H10 |
| H2 | `path_weight: exp`, removal of γ under `exp`, ladder switch with the capacity rule, per-cell relation embeddings in `PathEncoder` | T-H5 … T-H8; D12 goldens pinned to `sigmoid_gamma` + `schemas` |
| H3 | Adaptive defensive proposal (walk-level mixture, detached tables, epoch update) | T-H2 (biased q), T-H3; **variance gate:** on ACM with a fixed trained model, 50 training draws, `adaptive` must reduce `E‖(L̂_P − mean) z‖²` over random probe vectors `z` (matrix-free) by ≥ 20 % at equal `r`; otherwise the default stays `uniform` and the report says so |
| H4 | Read-outs, evaluation only (no import from training code into them): `I_π`, `I_σ` (pair decomposition, Prop. K), `I_{v,σ}`, overlap ratio, `ΔL_σ` (`T` evaluation draws), Kendall τ across checkpoints with a permutation null | T-H11 |
| H5 | Known-answer tasks (synthetic, 3 seeds each, pre-registered): **S1** label carried only by instances whose relevance is a non-factorising function of all three nodes of a PAP-like instance (e.g. parity of one bit on P, A, P′; a product of per-edge positive weights cannot express it by a rank argument — write it out), endpoint relation counts identical across classes. Expect `full` ≥ 0.9 and `endpoints` near chance; report `no_path_cells`, `uniform`, `schema` too. **S2** a rare planted length-4 schema among ≥ 6 relation types carries the label; expect `I_σ` to rank it first on every seed. This also answers D12 report decision 3 | pre-registered expectations; a miss is reported, not fixed by tuning |
| H6 | Smoke and cost on ACM, DBLP, IMDB (`path_source: wisp`, each ladder value, 5 epochs; LP and GC 2 epochs with `path_weight: exp`). Report cells, incidences, distinct signatures, length histogram, max log-weight, sampling ms per epoch, forward + backward time and peak memory vs `schemas` at matched incidences. Pre-registered gates: sampling ≤ 25 % of epoch time; peak memory ≤ 1.25× the `schemas` run | measurement |
| H7 | Report `../docs/H2S-NN_WISP_report.md` (Part A format, ≤ 2 pages); propose (do not make) the default flip | — |

### Tests (`tests/test_wisp.py`, unittest, float64 fixtures; small typed graphs with ≥ 3 node types, a self-inverse relation, a reciprocal pair, a dead end and a hub)
| ID | Check | Tolerance |
|---|---|---|
| T-H1 | Exact distribution: enumerate every simple walk from the anchors with its `p(ω)`; 200k draws; empirical frequencies match within 4 s.e. for every walk; no drawn walk repeats a node; discarded mass equals the enumerated dead-end mass | 4 s.e. |
| T-H2 | Unbiasedness: Monte-Carlo mean of the assembled `L̂_P` (dense, small graph) equals the exact `L_P^target` for `q = p` and for a strongly biased `q`. **Control:** drop the IS ratio under the biased `q` → fails | 4 s.e. per entry |
| T-H3 | Every IS ratio ≤ 2 (float64) for random and near-degenerate `q`. **Control:** a per-step mixture exceeds 2 on some walk | exact |
| T-H4 | Identity and merge: a walk and its reversal give one cell with summed weight; walks through different reciprocal facts give different cells (reuse D8 cases) | exact |
| T-H5 | Props. 2, 3, 5 with `exp` weights spanning `e^{−10}…e^{10}`: kernel preserved, `λ_max(Δ̂) ≤ 2`, filter gain ≤ 1 | 1e-9 |
| T-H6 | Under `exp`: no `θ_γ` parameter; `b` is in the scalar optimizer group; `sigmoid_gamma` reproduces the D12 goldens bit for bit | exact |
| T-H7 | Ladder invariances: perturb an intermediate node's `h₀` → `w` unchanged for `uniform`, `schema`, `endpoints`, changed for `full`; perturb an endpoint → changed for `endpoints`, `full` only; same relation sequence, different nodes → equal `w` under `schema`; parameter counts `endpoints` vs `full` within 5 % | exact / 5 % |
| T-H8 | Length-neutral initialisation: with constant `g`, the expected total cell weight per anchor is equal across lengths on a fixture without dead ends | 4 s.e. |
| T-H9 | Determinism: same seed → identical cells and `m_f`; training draws differ across epochs; the evaluation draw is identical across all ladder arms and both proposals | exact |
| T-H10 | Cost shape: sampling time grows linearly in `r` (3 values, R² ≥ 0.98) and has no Python loop over anchors, walks or signatures (inspect; state the evidence) | — |
| T-H11 | Read-outs: `Σ_π I_π = Σ_σ I_σ = 1`; the pair decomposition reproduces each cell's energy (Prop. K); `ΔL_σ` of a schema absent from the draw is 0 | 1e-10 |

**Mutation check (5):** remove the IS ratio → T-H2 fails; allow revisits → T-H1 fails; per-step mixture → T-H3 fails; feed an intermediate to the `endpoints` arm → T-H7 fails; drop the summed merge (keep one weight per distinct cell) → T-H2 fails.

### Rules
- Comparison arms (`path_source: schemas`, every `path_weight_input` other than `full`, `no_path_cells`, `endpoint_edges`, `holonomy_edges`, `free_path_maps`, `sigmoid_gamma`) are evidence, never tuned, never defaults.
- No clamping of `g` or `w_f`, no floors on `q`, no thresholds on schemas. If the max log-weight exceeds 30 in H6, stop and diagnose; do not clamp.
- Nothing from labels, validation or a trained model may enter the target `p`. `q` may use the model's previous-epoch weights (variance only); the evaluation draw uses `p` only.
- No schema list may filter candidates. A partial schema list as a prior inside `q` is out of scope.
- Save long output to `../docs/WISP_evidence/`; full suite once per unit.

### Adversarial self-review checklist (WISP)
- [ ] `p(ω)` is computed from the exact available-action and unvisited-neighbour counts, not approximations, and matches enumeration (T-H1).
- [ ] The mixture is per walk; the ratio uses `½p + ½q` of the realised walk (T-H3 control fails as expected).
- [ ] The evaluation draw ignores `q` and the trained model; arms share it (T-H9).
- [ ] γ is gone under `exp` and not silently re-introduced (e.g. via `gamma_init` or a constant multiplier).
- [ ] `endpoints` sees no intermediate through any path other than the geometry context `c` (state what `c` contains).
- [ ] Read-out code cannot change training (no shared mutable state; read-outs run on a frozen checkpoint).
- [ ] Every test would fail if its property were broken (state how you know).

### Stop conditions
A precondition fails; T-H1, T-H2 or T-H3 fails after 3 diagnosed attempts; the H3 or H6 gate is missed (report, keep the safer default); max log-weight > 30; an S1/S2 expectation is missed (report with diagnosis; no redesign inside this card); a needed change falls outside this card.

Commit per unit on `wisp`; do not push until the author has read the H7 report.

---

## Part I — Task card D2 follow-up: WISP after Part H (spec v1.4)

### Goal
Implement spec v1.4 (E1–E7) and close the gaps Part H left: test the **capacity** of the weight channel with a correct oracle, see **why** `full` does not learn selection (mechanism, not tuning), remove S2's length confound, add the test gaps the H7 review accepted, flip the default sampler, and make sampling overlap training. Authority: `H2S-NN_methodology_spec.md` v1.4 (§2 R12, §3.2, §5 C2/C2b, §6 WISP settings and gate). Evidence base: `docs/H2S-NN_WISP_report.md`, `docs/WISP_evidence/h7_review_response.md`.

### Preconditions
1. Part H is closed (`94fc397` on `wisp`). `src/` and `tests/` have no uncommitted changes that are not yours; never stash, reset or commit another session's changes. The author's uncommitted D2-row edits in `AGENTS.md`/`CLAUDE.md` stay uncommitted.
2. Push `d12-encoder` and `wisp` to `origin` as backups (author approved). Do **not** merge either into another branch.
3. Create branch `wisp-followup` off `wisp` (`94fc397`).
4. No real-data accuracy runs; synthetic tasks are pass/fail functional checks; smoke and cost runs are allowed. H2S-NN is untuned while baselines are tuned: no number here is a performance result.

### Units
| Unit | Content | Gate |
|---|---|---|
| I0 | Preconditions; inventory (`file:line`) of everything I1–I6 touch; pre-registration `../docs/WISP_evidence/followup_preregistration.md` with the expected result and the result that would show it wrong, per unit | written before code |
| I1 | **`g` reads `h₀` only (spec E1–E2).** `ξ_f` = each instance node's `h₀` + relation embeddings; maps and gates keep `[h₀, c]`. Appendix switch `ablation.path_weight_context: false | true` (`true` = the v1.3 `[h₀, c]` input; appendix pair only; never tuned, rejected by the tuning check). T-H7 under the **default** context: perturbing an intermediate's `h₀` **and its neighbours' `h₀`** leaves `endpoints` (and `schema`, `uniform`) weights unchanged, changes `full`; perturb positions ℓ−1 and ℓ separately (review MINOR-6) | T-H7 (rewritten), T-H5/T-H6 still pass |
| I2 | **Default flip (E7):** `data.path_source: wisp` for node classification; `schemas` stays as the comparison arm. Link prediction and graph classification are unchanged. Goldens stay pinned to their explicit settings | default-config test; goldens bit-identical |
| I3 | **S1′ with a learnable-scale oracle + learning mechanism.** Oracle: `w_f = exp(b + β·o_f)`, `o_f ∈ {0, 1}` the known relevance, `b` and `β` learnable scalars (scalar group), everything else trained. Arms: `full`, oracle (3 seeds each); `endpoints`, `uniform`, `no_path_cells`, and `full` with the readout **without the `h₀` half** (G0b hypothesis H1; 1 seed each). Log per epoch: mean log-weight of relevant vs irrelevant cells, the relevant cells' share of their target's degree, train/val accuracy. Pass (capacity): oracle ≥ 0.9. If the oracle misses, diagnose structurally (the relevant cells' share of `D`; is the path part drowned by the edge part?) and report; do not redesign here | pre-registered; misses recorded, not stop conditions |
| I4 | **S2 without the length confound (E4):** ≥ 3 schemas of the planted length (4) present in the draw, only the planted one carries the label; criterion: highest mean learned log-weight per cell among full-walk schemas (secondary: lift of `I_π`); 3 seeds | pre-registered |
| I5 | **Test gaps accepted in the H7 review** (`h7_review_response.md`): a trainer-level test that an adaptive training draw leaves the evaluation face groups unchanged (MINOR-5); config validation in `__post_init__` so direct construction cannot bypass it, `wisp_*` keys rejected under `schemas`, a test per new key, `wisp_proposal` named in the error message; a transported-`L̂_P` version of T-H2 (MINOR-10) if it fits in float64 on the fixture | tests pass; each fails under a targeted mutation |
| I6 | **Overlap sampling:** draw epoch e+1's cells on CPU in a background worker while the GPU trains epoch e. Semantics unchanged: the seed schedule is identical, so cells, `m_f` and the training logits for a fixed seed are bit-identical with and without overlap (test). Measure the exposed sampling share of epoch time on ACM, DBLP, IMDB (2 epochs each); pre-registered gate ≤ 25 % | bit-identity test; gate measured |
| I7 | Report `../docs/H2S-NN_WISP_followup_report.md` (Part A format, ≤ 2 pages): per unit verdict, the oracle result and its meaning for the weight channel, the learning-mechanism logs, deviations, items for the cluster | — |

### Settings (author decisions, as in Part H)
H5-style runs sized under about 1 minute each, ≤ 100 epochs; full test suite once, at I7 (affected tests after each other unit); one commit per unit on `wisp-followup`.

### Rules
- Comparison arms (`path_source: schemas`, ladder values other than `full`, `path_weight_context: true`, the oracle, the readout without `h₀`) are evidence, never tuned, never defaults.
- No clamping of `g`, `w_f`, `b` or `β`.
- A missed gate or synthetic expectation is recorded with numbers and a diagnosis, and the run continues to I7.

### Stop conditions
A correctness test still failing after 3 diagnosed attempts; max log-weight > 30; a crash you cannot fix; the I6 bit-identity test fails (semantics changed); a change outside this card.

Commit per unit on `wisp-followup`; do not push it until the author has read the I7 report.

**Status (2026-11-09):** done (`bf7f333`, `docs/H2S-NN_WISP_followup_report.md`). Gate (ii) not met: the learnable-scale oracle could not move far enough, so capacity is untested (Part K).

---

## Part J — Evaluation scope (spec v1.5): heterophily measures, link-prediction datasets, cumulative advisor report

### Goal
Implement the data side of spec v1.5 (F2–F4) without touching the model: (J0) compute the ℋ² index and MLH exactly as their source papers define and implement them; (J1) build the redesigned link-prediction datasets and their evaluation files; (J2) assemble the cumulative advisor report. Authority: `H2S-NN_methodology_spec.md` v1.5 §1 (Heterophily, Tasks) and §6 (Node classification, Link prediction). Author decisions: `session_handoff.md`, "Scope decisions (author, 2026-11-09)".

### Preconditions (stop and report if any fails)
1. **Branch:** Parts I and K are closed (`bf7f333` on `wisp-followup`; `6ac401f` on `wisp-capacity`). In the main working tree run `git switch -c part-j` from `wisp-capacity` (`6ac401f`; updated 2026-11-09 after Part K so the history stays linear). Uncommitted changes already in the tree carry over (the author's `AGENTS.md`/`CLAUDE.md` edits, `docs/H2S-NN_v2_plan.md`, `references/`, untracked files): never commit, stash, reset or revert them, except the four documents of unit J-docs. Stage files **by explicit path only**; never `git add -A`, `git add .` or `git commit -a`. `docs/` is inside the repository.
2. **Datasets (updated 2026-11-09 12:40 JST):** HGB Freebase (`data/freebase/freebase/raw/Freebase/`, PyG `HGBDataset` layout), HGB link-prediction `data/PubMed/` and `data/LastFM/` are on disk; provenance and md5 in `H2S-NN/data/PROVENANCE.md` (the heterogeneous HGB releases, not Planetoid PubMed). **PubMed node-classification labels are not available** (the HNE release is no longer downloadable): J1.5 reports this and computes nothing. Do not download anything else; if a file you need is missing, list the expected path, skip only the units that need it, and continue.
3. **No model or trainer code changes.** The only `src/` file you may edit is `src/data/link_prediction_dataset.py`, and only to register the derived datasets (J1.3).
4. CPU only; no training, no sweeps, no GPU runs.

### Units
| Unit | Content | Gate |
|---|---|---|
| J-docs | Commit the author-approved documents as the first commit on `part-j`, message "docs: spec v1.5, agent prompt Part J, session handoff": `docs/H2S-NN_methodology_spec.md`, `docs/H2S-NN_agent_prompt.md`, `docs/session_handoff.md`, `docs/working_preferences.md` (explicit paths; nothing else) | `git show --stat` lists exactly these four |
| J0.1 | Inventory + pre-registration `../docs/heterophily_evidence/preregistration.md`: published values to reproduce, tolerance, what would show the implementation wrong | written before code |
| J0.2 | **Reference code.** Install `H2GB` from PyPI (or clone its GitHub repo) and get the Hetero²Net code (`vendor/H2GB`, `vendor/Hetero2Net` are empty). Locate the functions behind H2GB Table 2 and Hetero²Net Table 1 (`file:line`). From the code, not from memory, record: (a) the MLH aggregation used in H2GB Table 2 (mean or max); (b) de-duplication, symmetrisation and self-loop handling of meta-path-induced graphs; (c) the meta-path set (all length-2 target→target, or listed subsets as in Hetero²Net); (d) multilabel handling (IMDB). If (d) is in neither codebase: **stop J0** and report the candidates (Jaccard, per-label average) without choosing | report |
| J0.3 | `scripts/heterophily/compute_heterophily.py` calls the reference functions wherever possible; own code only where they cannot be called. Tests `tests/test_heterophily_metrics.py` (unittest): hand-computed toy graph (two node types, two meta-paths, known `H_edge`, `H_adj`); ours equals the reference on the toy and on one random graph; mutation (drop the class-imbalance correction of `H_adj`) caught | tests |
| J0.4 | **Validation gate** before any new number: reproduce Hetero²Net's DBLP MLH (homophily 0.87, APA) within ±0.01. On a miss, diagnose (dataset version, meta-path set, de-duplication); never adjust a definition to fit | gate |
| J0.5 | Compute for ACM, DBLP, IMDB, Freebase (full graphs): per meta-path `H_edge`, `H_node`, `H_adj`; MLH (both aggregations, marking H2GB's); ℋ²; meta-path list, edges per `G_P`, class counts. Cite (do not recompute) H2GB Table 2 for ogbn-mag, mag-year, IEEE-CIS-G, H-Pokec. Output `../docs/heterophily_evidence/{results.json, results.md, run_log.txt}`; every value with its source (computed / H2GB Table 2 / Hetero²Net Table 1) and convention (homophily or heterophily) | outputs |
| J1.1 | **Inventory for link prediction** `../docs/lp_evidence/inventory.md`: for ACM, DBLP, IMDB, Freebase — every relation with endpoint types, fact count after tying (D7), and `ρ`; the HGB raw files you read. **Freebase:** list its relations only; the author chooses the target relation, so do not build the Freebase task in this card. Confirm the HGB link-prediction file format our loader reads (`node.dat`: id, name, type, features; `link.dat` / `link.dat.test`: src, dst, relation, 4th column) with `file:line` | report |
| J1.2 | **Derived tasks** ACM paper–author, DBLP author–paper, IMDB movie–actor, written in HGB link-prediction format to `H2S-NN/data/lp_derived/{acm_pa,dblp_ap,imdb_ma}/` (gitignored by `/data/`; confirm). One fixed split **by fact** of the target relation, 81/9/10, seed recorded: test facts go to `link.dat.test`; `link.dat` holds everything else. Node features and ids taken unchanged from the HGB node-classification release. Script `scripts/lp/build_derived_lp.py`, deterministic | J1 tests |
| J1.3 | Register the derived datasets in `_LOCAL_HGB_LINK_DATASETS` (minimal diff) and load each through `build_link_prediction_dataset` with the existing leak checks (no held-out fact, either orientation, in the message graph or on a sampled path). **Check** whether the loader's validation split depends on the training seed; the spec requires one split for all 10 training seeds: if it varies, report it (it affects PubMed and LastFM too) and propose the smallest fix; do not apply it | load smoke + leak checks |
| J1.4 | **Hard negatives.** Read HeaRT (Li et al., NeurIPS 2023 Datasets and Benchmarks: "Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking") and its code; quote the number of negatives per positive, K of Hits@K and the heuristics. Adaptation (spec v1.5): the corrupted endpoint is drawn only from the target relation's type on that side; heuristics computed on the training message graph only. Write fixed validation and test negative files per derived task (and for PubMed, LastFM if present) under `data/lp_derived/<task>/heart/`, plus their generation script. If HeaRT's procedure cannot be adapted without a further choice, stop J1.4 and list the choices | J1 tests |
| J1.5 | **PubMed labels.** Check whether HGB PubMed node-classification node ids/names map onto the link-prediction release (fraction matched, how matched). If all target-type (disease) nodes match, compute task-level `H_edge` of the disease–disease target relation with the J0 code; otherwise report the mismatch and compute nothing | report |
| J2 | **Cumulative advisor report** (author decision: one document, a divider page per date): `../docs/advisor_report_cumulative/H2S-NN_progress_report_cumulative.html` + A4 PDF (Playwright, `printBackground: true`) + `README.md` (rules of `../docs/advisor_report/README.md` apply). Chapters 1–2: the existing reports of `../docs/advisor_report/` and `../docs/advisor_report_2026_11_08/`, content unchanged, each preceded by a full-page divider with its date (from their READMEs). Chapter 3 (9 November 2026): heterophily of the evaluation datasets — what MLH and ℋ² measure (one paragraph, one small example), one table for all node-classification datasets with the source of each value, and a note on datasets without node labels ("not measurable", with the reason). Only values that passed J0.4. Do not edit or delete the existing report folders. No internal codes, branch names, file paths, or accuracy numbers | PDF renders; every number traced |
| J3 | Report `../docs/H2S-NN_partJ_report.md` (Part A format, at most two pages) | — |

### J1 tests (`tests/test_lp_derived.py`, unittest)
- Split: the three parts are disjoint by fact (either orientation); proportions 81/9/10 within ±0.5 %; same seed → byte-identical files; every node of a test fact exists in `node.dat`.
- Negatives: none is a positive of any split (either orientation); every corrupted endpoint has the required type; the heuristics read only training facts (mutation: compute them on the full graph → a targeted test fails).
- Loader: the existing leak checks pass on every derived task; mutation: put one test fact back into `link.dat` → caught.

### Rules
- Never change a definition, split ratio or negative scheme to fit an expectation; a miss is recorded with a diagnosis.
- Long output to `../docs/heterophily_evidence/` and `../docs/lp_evidence/`; affected tests after each unit, full suite once at J3.

### Stop conditions
A precondition fails (except missing datasets: skip the units that need them); J0.2 (d) unresolved (stop J0 only); J0.4 missed after 3 diagnosed attempts; the reference code cannot be installed or located; memory blow-up materialising a `G_P` (report its size; no subsampling without approval); a needed change outside this card.

Git: commit per unit on `part-j`, staging by explicit path (code, tests, and that unit's evidence files under `docs/`). Never push or merge.

---

## Part J (continued) — J4: decisions of 2026-11-09 (spec v1.8), link-prediction completion, review

### Goal
Apply the author's decisions on the Part J report (spec v1.8 F6–F11): heterophily conventions, the loader fixes, the Freebase link-prediction task, HeaRT negatives for every task, the IMDB protocol checks; then the independent review and the push of `part-j`. Authority: `H2S-NN_methodology_spec.md` v1.8 §1 (Heterophily) and §6 (IMDB protocol, Link prediction). Evidence base: `docs/H2S-NN_partJ_report.md`, `docs/heterophily_evidence/`, `docs/lp_evidence/` (`inventory.md`, `j14_heart.md`).

### Preconditions (stop and report if any fails)
1. Continue on `part-j` (last commit `070c379`) in the main working tree. The preconditions and git rules of Part J still hold (stage by explicit path only; never commit, stash or reset others' changes; `docs/H2S-NN_v2_plan.md`, AGENTS.md, CLAUDE.md, `vendor/` clones stay uncommitted).
2. No training, no GPU runs, no model or trainer code. `src/` files you may edit: `src/data/link_prediction_dataset.py` only (J4.2, J4.3). Baseline adapters are read, not changed (J4.5).
3. Datasets are on disk (`H2S-NN/data/PROVENANCE.md`); download nothing except HeaRT's and the reference papers' sources needed to quote definitions (read in the scratchpad, outside the repo).

### Units
| Unit | Content | Gate |
|---|---|---|
| J4-docs | Commit the updated documents first, message "docs: spec v1.8, agent prompt (J4), session handoff": `docs/H2S-NN_methodology_spec.md`, `docs/H2S-NN_agent_prompt.md`, `docs/session_handoff.md` (explicit paths; nothing else) | `git show --stat` lists exactly these three |
| J4.0 | Pre-registration `docs/heterophily_evidence/j4_preregistration.md` and `docs/lp_evidence/j4_preregistration.md`: per unit the expected result and the result that would show it wrong; the HeaRT parameters quoted from HeaRT's code with `file:line` (negatives per positive, K, PPR settings, how heuristics are combined) and the L3 formula quoted from its source | written before code |
| J4.1 | **Heterophily conventions** (spec v1.8 §1). (a) IMDB per genre: for each of the 5 genres, `H_adj` and `H_edge` with H2GB `calcHomophily` on the binary label `1(g ∈ Y_v)`; IMDB ℋ² and MLH = means over genres and over `M₂`; report per-genre values. (b) Unlabelled target nodes: the reference fill exactly as `calcHomophily` applies it (on a clone, as fixed in J0.5), with a fixed recorded seed and the spread over 5 fill seeds; the labeled-only value as a diagnostic row for Freebase and IMDB (confirm ACM/DBLP are fully labelled, so the fill does not touch them). (c) Freebase on the symmetrised graph as H2GB's loaders build it. (d) H2GB's full `M₂` for every dataset; Hetero²Net sets and exact match stay as the reproduction rows. (e) Appendix: Zhao et al.'s Jaccard label homophily for IMDB (arXiv 2304.10398, Definition 1; quote it). Update `results.json`/`results.md` with the convention of every value | tests: hand toy with multi-label per-genre values; mutation "skip the fill" (labeled-only as the primary value) caught; mutation "exact match instead of per genre" caught |
| J4.2 | **Loader fixes.** (a) When `link.dat.valid` exists, validation = that file; its facts are removed from the message graph in both orientations; otherwise the current carve. (b) Guard: raise when a reverse-typed partner relation exists but ρ(target) is `None` (the J1.3 reverse-only leak). (c) Write `link.dat.valid` for PubMed and LastFM equal to the loader's current carve (no numeric change) | tests: carve == file for PubMed/LastFM; derived tasks evaluate 81/9/10; reverse-only fixture raises; mutations of (a) and (b) caught |
| J4.3 | **Freebase link prediction** `freebase_bb`: relation 0 (BOOK-and-BOOK). Positives exclude self-loops (they stay in the message graph as stored facts). Split 81/9/10 by **unordered pair**, seed recorded: both orientations of a pair fall in one part and are removed together from the message graph and from every sampled path. HGB link-prediction format under `data/lp_derived/freebase_bb/`; register in `_LOCAL_HGB_LINK_DATASETS`; task-level `H_edge` book–book with the J4.1 convention (fill; labeled-only diagnostic) | tests: no pair (either orientation) in two parts; self-loops absent from every split; mutation "split by directed fact" caught by a reciprocal-leak test; loader leak checks pass |
| J4.4 | **HeaRT negatives** for acm_pa, dblp_ap, imdb_ma, freebase_bb, pubmed, lastfm (validation and test): C1 same-type targets (PubMed, Freebase) use RA exactly as HeaRT (training message graph); cross-type targets (ACM P–A, DBLP A–P, IMDB M–A, LastFM user–artist) use the degree-normalised length-3 path score of Kovács et al. 2019 (open the source and quote the formula; if it cannot be verified, stop J4.4) on the target relation's **training** facts; C2 feature cosine only where both endpoint types share one feature space (PubMed; ACM only if J4 verifies from the HGB release that paper and author features use one vocabulary, otherwise not); C3 negatives for the fixed validation file; C4 without replacement within each positive's list (if a pool is smaller than the required count, stop and report). The corrupted endpoint is drawn from the target relation's type on that side. Files under `data/lp_derived/<task>/heart/` (PubMed, LastFM: `data/<name>/heart/`) + the generation script | the J1 negative tests (none is a positive of any split in either orientation; type constraint; heuristics read only training facts, mutation caught; no duplicate in a list; same seed → identical files) |
| J4.5 | **IMDB protocol checks** (spec v1.8 §6): a test that the 359 unlabelled movies enter neither loss nor metrics and that the threshold rule (logit > 0) is used; an inventory (`file:line`) of every baseline adapter's multilabel prediction rule and metric. Any adapter using top-k or its own metric is **reported**, not changed | test; inventory |
| J4.6 | Update chapter 3 of the cumulative advisor report with the computed Freebase and IMDB values and one sentence per convention; chapters 1–2 unchanged; re-render the PDF | PDF renders; numbers traced |
| J4.7 | Report: append §10 "J4" to `docs/H2S-NN_partJ_report.md` (Part A format, ≤ 1 page for J4); full suite once. Then an **independent review** by a fresh sub-agent given only the diff `6ac401f..HEAD`, Part A, Part J and J4; file it unedited as `docs/lp_evidence/j_independent_review.md` with your responses in `j_review_response.md`. Then **push `part-j`** to origin (author approved, after the review); no merge | — |

### Rules
- No definition, split ratio or negative scheme changes to fit an expectation; a miss is recorded with a diagnosis.
- Every heterophily value carries its convention and source (computed / H2GB Table 2 / Hetero²Net Table II).
- Affected tests after each unit; the full suite once at J4.7.

### Stop conditions
A precondition fails; a HeaRT parameter or the L3 formula cannot be verified from its source (stop J4.4 only); the Freebase pair split cannot guarantee leak freedom; a candidate pool too small for C4; a needed change outside this card.

Git: one commit per unit on `part-j`, staging by explicit path. Push `part-j` only at the end of J4.7, after the review; never merge.

---

## Part K — Weight-channel capacity, map latents and guards (spec v1.6)

### Goal
Settle WISP gate (ii) (spec v1.6 §6): **if a cell's weight is right, can the model use one decisive instance?** Part I's learnable-scale oracle could not answer this: one scalar moves only about 0.5 under the schedule. Also implement E10 (maps read `[h₀, c]` in every arm) and E11 (`r` never tuned), close the guard gaps of the I7 review, and correct the Part I report. Authority: `H2S-NN_methodology_spec.md` v1.6 (§0 rule 6, R12, §6 WISP settings and gate). Evidence base: `docs/H2S-NN_WISP_followup_report.md`, `docs/WISP_evidence/i7_independent_review.md`, `i7_review_response.md`, `i3_run_output.txt`.

### Preconditions
1. Part I is closed at `bf7f333` on `wisp-followup`. Push `wisp-followup` to `origin` as a backup (author approved). Do not merge anything.
2. Part K runs in the main working tree **before Part J** (both use the main tree; Part J's first unit commits the docs on `part-j`, and switching from that branch to a branch off `bf7f333` would delete those then-tracked docs from the tree). Check that no other agent is working in the tree (current branch `wisp-followup`, no `part-j` branch in use); otherwise stop and ask the author. Create branch `wisp-capacity` off `bf7f333`. Do not edit `src/data/link_prediction_dataset.py` (Part J owns it).
3. `src/` and `tests/` have no uncommitted changes that are not yours. Never commit, stash or reset the author's uncommitted edits (AGENTS.md, CLAUDE.md, `docs/*.md` specs, plan, prompt). K4 edits two statements in AGENTS.md/CLAUDE.md and leaves them uncommitted.
4. No real-data accuracy runs. Synthetic runs are functional checks; H2S-NN is untuned, and no number here is a performance result.

### Units
| Unit | Content | Gate |
|---|---|---|
| K0 | Preconditions; inventory (`file:line`) of what K1–K3 touch; pre-registration `../docs/WISP_evidence/k_preregistration.md`: per unit the expected result and the result that would show it wrong; for K2 also the **reachability statement** and the **information-route audit** of spec §0 rule 6, written from the graph code (`h5_synthetic.py::graph_s1p`) and the actual optimizer settings (`builders.py`), not copied from this card | written before code |
| K1 | Two latents (E10) | T11-ctx, goldens, 2 mutations |
| K2 | Capacity dose-response on S1′ (gate (ii)) | pre-registered; a miss is recorded, not a stop condition |
| K2b | S2 (I4 graph) with 1,000 training targets (author decision 2026-11-09, option A) | pre-registered; a miss is recorded, not a stop condition |
| K3 | Guards (E11, review m1, NIT) | one test per rule; each fails under a targeted mutation |
| K4 | Docs: erratum to the Part I report; D2 rows in AGENTS.md/CLAUDE.md (uncommitted) | — |
| K5 | Report `../docs/H2S-NN_WISP_capacity_report.md` (Part A format, ≤ 2 pages; includes the K2b schema table per seed); full suite once | — |

### K1 — two latents (spec v1.6 E10)
- Under `path_mode: endpoint_edges` and under `free_path_maps`, the latent that feeds the **maps** (`endpoint_maps`, `free_maps`) is computed from the geometry input (`[h₀, c]` under `geometry_context`). `g`'s latent is computed from `path_inputs` (`h₀` by default). Use two encoder instances of the same class where an arm needs both. The `full` cell arms are unchanged (one latent, for `g`).
- Pair attention (`coupling: pair_attention`): its path latents read the geometry input.
- `path_weight_context: true` keeps its meaning (`g` reads `[h₀, c]`); the map latent always reads the geometry.
- Report the parameter count of each affected arm before and after.

Tests (default `geometry_context: true`; inject geometry rows directly, because perturbing `h₀` moves neighbours' `c` by design):
- **T11-ctx (a):** the `endpoint_edges` maps change when an endpoint's geometry row changes, including a change of its `c` part only, and do not change when an intermediate's geometry row changes.
- **(b):** `g`'s latent does not change when only `c` changes (default), and does change under `path_weight_context: true`.
- **(c):** the `free_path_maps` maps change with the `c` part of any instance node.
- **(d):** the pair-attention scores change with `c`.
- Goldens stay bit-identical (they pin `geometry_context: false`; say so).
- **Mutations (2):** map latent fed `h₀` → (a), (c) fail; `g` fed the geometry under the default → (b) and T-H7 fail.

### K2 — capacity dose-response on S1′ (gate (ii))
Evidence script `../docs/WISP_evidence/k2_capacity.py`; the oracle lives in the script, as in I3, not in `src/`. Graph: `h5_synthetic.graph_s1p` unchanged except `n`; same split rule (50/25/25); same config as I3 (`resample_paths: false`, r = 640, ≤ 100 epochs, selection on validation).
- **Oracle:** `w_f = exp(b + β·o_f)`, `o_f ∈ {0, 1}` the known relevance. β is a **constant** (a buffer, not a parameter); `b` is learnable in the scalar group, `b₀ = log(path_weight_init)`; `g` is not used for weights; everything else trains; nothing is clamped.
- **Grid:** n = 500: β ∈ {0, log 8, log 64, log 512} × 3 seeds; n = 2000 (1,000 training targets): β ∈ {0, log 64, log 512} × 3 seeds. If n = 2000 does not fit in 8 GB, use the largest n ≤ 2000 that fits, chosen and written down before the gated runs.
- **Per-epoch log:** train/val accuracy; `b`; the median relevant-cell share and path share of the target's `D`; `∂L_train/∂β` at the run's β (β as a leaf tensor with `requires_grad`, never given to the optimizer).
- **Operator probe** (at initialisation and at the selected checkpoint; the model is not trained by it): `u = (A − I) h₀` with the model's geometry. In S1′ a cpaper's class reaches its paper in one application of `A` only through a cell. Fit a logistic regression of `y` on `u` restricted to target papers, on the training targets, and report accuracy on the test targets. Fixed solver and L2 (e.g. LBFGS, L2 = 1e-3), written in the pre-registration and never tuned. Control: the same probe on `h₀`.
- **Expected** (DERIVED, idealised: the head reads the one-application cell channel, transports keep the class direction, the 63 irrelevant cpaper classes are independent; SNR ≈ e^β/√63): accuracy ≈ 0.55 / 0.84 / ≥ 0.99 / ≥ 0.99 at β = 0 / log 8 / log 64 / log 512, for the probe and for the model at n = 2000. At n = 500 the model may fall short through memorisation (reported, not gated).
- **Gate (ii) passes** iff at n = 2000 the model reaches ≥ 0.9 on 3/3 seeds at β = log 64 or at β = log 512, and β = 0 stays ≤ 0.65 on every seed. If β = 0 reaches ≥ 0.9, that is a leak: report it, give no gate verdict, the task is redesigned later.
- **If the gate misses, read the probe** (fixed now):
  - probe ≥ 0.9 at log 512 while the model is < 0.9 → the information arrives; the limit is learning in the filter or head;
  - probe < 0.9 → the information does not arrive in one application: report the relevant share of `D` and the dispersion of the transported carrier states across relevant cells (define the measure in the pre-registration), and stop at the diagnosis. No redesign in this card.
- **`∂L/∂β` sign:** report, per β, the fraction of epochs before the best epoch with `−∂L/∂β > 0`. Not a gate. Reading fixed now: consistently positive → a learnable scale would move the right way given enough reach; mixed or negative → learning β from the loss is unreliable at this size.
- **Reachability (rule 6):** β is constant; `b` can move only about Σ_t lr_t (it shifts the total path mass only). State the actual scalar learning rate and schedule from the code.
- **Context arm:** `full` at n = 2000, 1 seed: the relevant − irrelevant log-weight gap. Not gated.

### K2b — S2 with more training targets (author decision 2026-11-09)
Question: in I4, σ\* = `a b c d` was strictly first on 1 of 3 seeds; it and its sibling `a b c e` (same prefix, different last relation) were lifted together, and seed 42 lifted `f g h k` first. The length-3 prefix `a b c` (same M3) was **not** lifted, so a shared-node route does not explain the tie. Hypothesis tested here: with 200 training targets `g` cannot separate schemas that differ only in their last relation before memorisation stops learning; with 1,000 training targets it can.
- Script `../docs/WISP_evidence/k2b_s2_n2000.py`: `i4_s2_unconfounded.graph_s2u(n=2000)` (1,000 training targets), otherwise I4's config and read-out code unchanged. Arms: `full` × seeds 42, 43, 44; `uniform` × seed 42 (non-vacuity control).
- Set `r` **before training** so that σ\* is drawn for every target in the evaluation draw (coverage 1.0, checked on the draw; this is set-up, not result-dependent); report `r` and coverage.
- Per-epoch log: train/val accuracy, mean log-weight of σ\* minus `a b c e`, best epoch.
- Pre-registration (K0 file) includes reachability (I4 already reached log-weights ≈ 8, so the range exists; state the needed σ\* − sibling gap is only > 0) and the information-route audit of `graph_s2u` per arm (σ\* cells carry C\*'s class in one application; `a b c e` and `a b c` reach C\* only through the fact `d` from M3; `f g h k` and the other schemas have no route except through T), written from the code.
- **Expected:** primary criterion (σ\* strictly highest mean learned log-weight per cell among full-walk schemas) met on 3/3 `full` seeds; `uniform` not strictly first.
- **Wrong if** any `full` seed ranks another schema ≥ σ\*. Consequence fixed now: C2b's S2 is reported as "not shown at default settings" and moves to the cluster protocol with the same criterion; the criterion is not loosened (no "top set" reading).
- Time: I4 runs at n = 400 took 109–139 s; n = 2000 is estimated at 5–10 min per run (UNVERIFIED).

### K3 — guards (E11, review m1, NIT)
- Search space: reject `data.wisp_walks_per_anchor`. `tests/test_wisp.py` (around `:1202`) asserts the opposite; changing that assertion is authorised by spec v1.6 E11.
- Tuning base config (node classification): reject `wisp_max_length ≠ 4`, `wisp_proposal: adaptive`, and `wisp_walks_per_anchor` other than 0 or the dataset's calibrated value.
- Reject `ablation.path_weight_context: true` together with `endpoint_edges`, `free_path_maps` or `coupling: pair_attention` (the appendix pair is the ladder only).
- `r` for other `ℓ_max`: calibrate `r` per dataset (ACM, DBLP, IMDB) for `ℓ_max ∈ {2, 3, 5, 6}` with the H1/H6 incidence rule (`h6_calibrate_r.py`); store the table next to the presets; an unset `r` resolves from the table for its `ℓ_max`; an `ℓ_max` without a calibrated value raises. Report the table.
- NIT: a directly built `DataConfig()` must not leave `path_source=""` consumable as `schemas`; either consumers raise on the unresolved value or it is resolved explicitly (state which).

### K4 — docs
- Append a dated **erratum** (not a rewrite) to `docs/H2S-NN_WISP_followup_report.md`:
  1. Withdrawn: "the readout without `h₀` memorises too, so G0b's H1 is not the mechanism". `x` contains `h₀W₀` (`src/models/filter_bank.py:83,91`) and the diagonal of `A − I` is non-zero, so that arm did not remove the ego route; H1 stays deferred.
  2. "β ≈ 7–9 to dominate `D`" is a sufficient condition, not a requirement; the idealised one-application estimate is e^β ≈ 10 for 0.9 accuracy (tested in K2).
- AGENTS.md and CLAUDE.md: in the D2 rows, change the stated default to `path_source: wisp` for node classification (spec v1.4 E7). Edit only those statements; do not commit them.

### Settings
Synthetic runs ≤ 100 epochs; n = 500 runs take about 1 minute; n = 2000 runs are estimated at 3–4 minutes (UNVERIFIED). Affected tests after each unit; the full suite once, at K5. One commit per unit on `wisp-capacity`.

### Rules
- The oracle, the probe and every comparison arm are evidence, never tuned, never defaults. The probe's solver and L2 are fixed in the pre-registration.
- No clamping of `g`, `w_f` or `b`.
- A gate miss is recorded with numbers and the pre-registered reading; the run continues to K5.

### Stop conditions
A correctness test still failing after 3 diagnosed attempts; max log-weight > 30; a crash you cannot fix; goldens not bit-identical; a change outside this card (including anything on Part J's branch `part-j` or in `src/data/link_prediction_dataset.py`).

Commit per unit on `wisp-capacity`; do not push it until the author has read the K5 report.

**Status (2026-11-09):** done (`6ac401f`, `docs/H2S-NN_WISP_capacity_report.md`). Gate (ii) met; K2b not counted (leaking task). Spec v1.7 records the decisions.
