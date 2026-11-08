# H2S-NN v2 — Revision Plan (agreed decisions)

**Date:** 2026-11-06
**Builds on:** `Reviews/H2S-NN_fix_spec.md` (status tags, hypotheses H1–H12, and §6 protocol fixes remain in force unless this plan supersedes them).
**Status tags** (same as the fix spec): `VERIFIED-CODE`, `VERIFIED-PAPER`, `VERIFIED-REPO-DOC`, `DERIVED`, `UNVERIFIED`, `PROPOSAL`.

---

## 0. Decision register

| ID | Decision | Status |
|---|---|---|
| D1 | Replace dense per-edge restriction maps with factorized, matrix-free, conformal-orthogonal maps. | **Implemented and verified** (branch `d1-factorized-maps`, commits `dd42b8e`, `11ba6c3`, `924be78`); accuracy validation moved into D4 — see §1.7 |
| D1a | **One scale per edge, shared by both sides:** `F_{v◁e} = c_e · R_{ψ,side} · H_k(u_{v◁e})`, `c_e = softplus(MLP([h_u, h_v, rel_emb])) + ε`. Every edge transport and path holonomy is then exactly orthogonal. Replaces per-side scales; supersedes the earlier "bound `c` to `[e⁻², e²]`" proposal, which did not fix the problem (§1.7). | **Agreed (2026-11-06)** — implemented as D4 unit 1 |
| D1b | Paper wording: edge transports and holonomies lie in **SO(d)**; claim "transports can reverse feature directions in pairs", not "a reflection flips a direction". | **Agreed (2026-11-06)** |
| D2 | **AMPSS is replaced by WISP (Walk-sampled, Instance-weighted Sheaf Paths)**: schema-free walk-first candidates (self-avoiding, relation-first, uniform length in {2..ℓ_max}; one walk = one cell), a stated target measure with walk-level defensive importance sampling (weight ≤ 2), instance-conditioned weights `w_f = exp(g(ξ_f))` (γ removed), schemas read out (`I_π`, `I_σ`, `ΔL_σ`), never selected. C2 becomes the weight ladder `uniform ⊂ schema ⊂ endpoints ⊂ full` with a kill rule. Spec v1.3 §3.2 is the authority; §2.5 below records the reasoning. "AMPSS" now names only the deleted v1 mechanism. | **Agreed (2026-11-08)** — task card Part H; implementation gated (§6 step 2f) |
| D3 | Scalability is the first engineering priority, so that heterophilic heterogeneous benchmarks (mostly large) become usable. Scale **v2**, not v1. | **Agreed** |
| D4 | 2-cells reach nodes through the **vertex–2-cell sheaf Laplacian** (option b only); edge-cochain machinery is deleted. | **Implemented and verified** (branch `D1matbounding_D4coupling`, 9 commits; `docs/H2S-NN_D4_report.md`) — see §4.7 |
| D5 | Fix baseline provenance and IMDB heterophily (fix spec H5, H8) in parallel. Independent of D1–D4. | Recommended (parallel track) |
| D7 | **Fact layer (one definition):** `ρ(r) = r′` iff `E(r′)` is exactly the reversal of `E(r)` (parameter-free, deterministic, no config key); a fact is a stored edge up to inversion, `(u, r, v) ≡ (v, ρ(r), u)`; one fact = one oriented 1-cell; a relation's two roles coincide when `ρ(r) = r`. A backward hop applies the adjoint transport. `fact_layer: untied` is an ablation control only and is **never tuned**. See §9.1. | **Agreed (2026-11-06)** — task card Part D |
| D8 | **Path-cell builder:** (i) a 2-cell is identified by its sequence of facts (canonical orientation); identical cells are merged by **summing** their sampling weights; (ii) instances are drawn by a relation- and direction-constrained random walk (uniform neighbour per hop, non-simple walks rejected), `r` walks per (anchor, schema), weight `1/r` per walk; (iii) vectorized; (iv) resampled every training epoch, one fixed draw for evaluation. Replaces the DFS-with-quota enumerators and is the interim sampler of §2.4 and the instance interface of D2. See §9.2. | **Agreed (2026-11-06)** — task card Part D. **Partly superseded by D2/WISP (2026-11-08):** for node classification, (ii) per-schema walks are replaced by WISP's schema-free target measure; (i) identity and summed merge, (iii) vectorization and (iv) the resampling protocol are kept. The schema sampler stays as the fixed-literature-schema comparison row |
| D9 | **O(L) path term:** `(L_f x)_{v_i} = w_f (s_i² x_i − T_iᵀ z_f)`, with `Σ_i T_i x_i` by Horner and `T_iᵀ z_f` by a prefix recursion: 2L transport steps per path instead of L(L+1). Exact identity (DERIVED from `T_iᵀT_i = s_i²I`); supersedes the §4.2 "correction". See §9.3. | **Agreed (2026-11-06)** — task card Part D |
| D10 | **Cooperative coupling with a real spectrum:** `x ← x − η G Δ̂ B x` with per-node listen gates `G = diag(g_v)` and broadcast gates `B = diag(b_v)` in (0, 1], computed from `h₀`. Influence becomes asymmetric (v → u has strength `g_u b_v`), while the operator stays similar to a PSD one, so stability and the fixed-point theory survive. Default `cooperative`; the symmetric operator (all gates 1) is the ablation, and the experiment decides. See §10. | **Implemented** (`16f6ad3` "D10 complete", `docs/H2S-NN_D10_report.md`); gates did not move in the smoke run (≈ 0.95) — fixed by D11 |
| D11 | **Final model (agreed 2026-11-07):** one diffusion block, maps from `h₀` (one sheaf), D10 listen/broadcast gates **always on** and fixed so they can learn; the update is the gradient flow of the sheaf energy in a learned node metric. Design rationale (assumptions A1–A5, gap, properties, derivation) in §12.0; Prop. 4 shows the gate form is exactly the directional influence that keeps an objective (A4 ⇔ P1). Pair attention and multi-block structure are excluded by A4 and A5 and **tested** (`pair_attention` and `h0_only` screen rows, pre-registered consequences in §12.6). See §12. Supersedes the open asymmetry decision of §10.5 and §11 item 1. | **Agreed (2026-11-07)**; units U0–U4b done (branch `tier0-screen-d11`). **Partly superseded by D12 (2026-11-08):** the update rule (now a nested comparison arm), `η`, and maps-from-`h₀`-only are replaced; one block, one sheaf, gates, learning fix and Prop. 4 are kept |
| D12 | **Single encoder and geometry context (methodology spec v1.2, 2026-11-08):** the first-order update is replaced by a gated Chebyshev filter bank `x = Σ_k T_k(GΔ̂B − I)(I ⊗ W_k)h₀`, `Σ‖W_k‖₂ ≤ 1` (R9: frequency selectivity per stalk coordinate; gain ≤ 1 for all parameters, Prop. 5; CSNN's cooperative property exact, Prop. 6; D11 nested, Prop. 7). Maps, gates and path weights are computed once from `[h₀, c]`, `c` = typed neighbour means (R10; Eq. 9 restored). `η` removed. A4 closed as limitation L1 (pair-specific strength ratios are the exact price of a self-adjoint operator, Prop. 4); per-edge attention stays only as evidence arm C4b. Authority for design is `H2S-NN_methodology_spec.md` v1.2; this plan stays the decision log. | **Agreed (2026-11-08)** — task card Part G |
| — | **Reality-before-theory audit:** assumptions kept only because our propositions need them are listed and tested against data and outside evidence. See §11; binding rule in `H2S-NN_agent_prompt.md` Part A. | **Agreed (2026-11-07)** |
| D6 | v2 is the only model. v1 is deleted (kept on git tag `v1-submitted`). Link prediction and graph classification are **ported** to the v2 encoder, not rewritten: task layer (datasets, splits, negative sampling, trainers, metrics) kept; only the heads change, and only where forced. See §4.8. | **Agreed (2026-11-06)** |

### What this plan supersedes in the fix spec
- **P3 (structured maps) as written is replaced by D1.** P3's `F = R_ψ·diag(s_e)` with one map per edge gives `F_src = F_tgt`, so every edge transport is `I`, the path holonomy collapses to `I`, and closure edges become plain endpoint edges (DERIVED). D1 fixes this with side-specific, node-dependent factors.
- **P5 (uniform relation-first walk sampler) is implemented by D8** (the interim sampler; D2 builds on it).
- **P2 (unsigned node–edge coupling) is rejected by D4**, and **P6 (typed edge pooling in the readout) is obsolete**: edge states and edge pooling are deleted.
- **H3 (`w_cl = 0`) framing is revised.** Closure edges carry the path holonomy, which depends on intermediate nodes. The correct control is a closure edge with endpoint-learned maps (no holonomy), not removing the closure edge.

---

## 1. D1 — Factorized restriction maps

### 1.1 Problem (VERIFIED-CODE)
- `NodeToEdgeSheaf` materializes `F_src, F_tgt ∈ [M, d, d]` (`src/models/sheaf_maps.py`). Memory `2·M·d²`; at d = 32 that is 2,048 floats per edge.
- Face side is also dense: per-incidence face maps `[F, ℓ+1, d, d]` (`compose_face_transport`) and a `d²`-output head for `A_f` (`SequenceEdgeToFaceSheaf.map_head`).
- Closure maps (`closure_F_src` = path holonomy) are materialized `[F, d, d]`.

### 1.2 Specification (PROPOSAL)

```
F_{v◁e} = c_{v◁e} · R_{ψ(e),side} · H_k(u_{v◁e})
```
- `c_{v◁e} = softplus(MLP(·)) + ε` — positive scalar per incidence. **Amended by D1a:** one scalar `c_e` per edge, shared by its source and target maps (§1.7).
- `R_{ψ,side} ∈ O(d)` — one per (relation, side): `2·|R|` matrices in total, shared by all edges of that relation.
- `H_k(u) = ∏_{j=1..k} (I − 2 u_j u_jᵀ / ‖u_j‖²)` — k Householder reflections; `u_{v◁e} = MLP([h_v, h_other, rel_emb, side])`.
- Inverse in closed form: `F⁻¹ = c⁻¹ · H_kᵀ · R_{ψ,side}ᵀ` (each reflection is its own inverse; reverse order).
- The product is conformal-orthogonal (scalar × orthogonal), the same class as v1's FastHPP maps, so transport, `δ₁δ₀ = 0`, and the gauge argument carry over unchanged.

**Four required conditions:**
1. **Invertible** — no low-rank factorization (transport and holonomy need `F⁻¹`).
2. **Side-specific and node-dependent** — otherwise holonomy becomes `I` (see P3 note above).
3. **Never materialized** — apply to vectors: k rank-1 reflections → grouped GEMM by `(ψ, side)` → scale.
4. **Face side too** — path transports applied matrix-free (sequentially along each path; 2L transport steps per path with the O(L) form of D9, §9.3); `A_f` reduced to an orthogonal gauge (identity) plus a scalar weight, removing the `d²` head and the `[F, ℓ+1, d, d]` tensor.

**Interpretation for the paper:** `R_ψ` is the relation's change of frame; the Householder factors make the map node-dependent. **Correction (D1b, DERIVED and confirmed by the independent review):** both sides of an edge have the same determinant sign `(−1)^d·(−1)^k`, so every edge transport `F_src⁻¹F_tgt` and every path holonomy lies in SO(d) (v1's FastHPP maps have the same property). A single transport cannot flip one direction; it can reverse directions **in pairs** (a 180° rotation in a plane). Write the contrast claim that way.

### 1.3 Expected savings (DERIVED arithmetic, must be measured)
| Map form (d = 32) | Floats stored per edge | Ratio vs dense |
|---|---|---|
| Dense v1 (`2·d²`) | 2,048 | 1× |
| D1, k = 2 (`2·(k·d+1)`) | 130 | ≈ 16× less |
| D1, k = 1 | 66 | ≈ 31× less |

Plus `2·|R|·d²` shared floats (negligible). Peak memory will fall by less than these ratios because autograd stores activations; it must be measured with `torch.cuda.max_memory_allocated`.

### 1.4 Code locations
- `src/models/sheaf_maps.py`: new `FactorizedNodeToEdgeSheaf`; matrix-free `compose_face_transport`; remove dense `A_f` head (keep behind a flag for ablation).
- `src/models/coboundary_ops.py`: `delta0`, `delta0_T`, `delta1`, `delta1_T` take map *operators* (callables) instead of `[M, d, d]` tensors.
- `src/utils/memory_guard.py`: update the memory model.

### 1.5 Acceptance tests
- `tests/test_coboundary_chain.py` passes with the new maps (`δ₁δ₀ = 0` to float32 roundoff).
- Functoriality: `T_v` identical via either boundary edge.
- Invertibility: `F⁻¹F = I` to roundoff.
- **Holonomy diversity:** variance of the path holonomy across instances of the same schema is clearly non-zero (guards against condition 2 failing silently).
- Gauge test (fix spec T-D) after the normalization change in §3.

### 1.6 Experiments and decision rule
- Variants: dense v1 (upper bound, HGB only), D1 with k ∈ {1, 2, 4}.
- DBLP, ACM, IMDB; 10 seeds; paired tests vs dense.
- **Accept** the smallest k within 1 std of dense on all three, with ≥ 10× lower map memory.
- **Fallback** if every k drops > 1 std: larger k, or block-diagonal orthogonal factors; as a last resort, report dense on HGB and D1 on large graphs, stated explicitly.

### 1.7 Status (2026-11-06) — source: `docs/H2S-NN_D1_report.md`
**Complete and verified (VERIFIED-RUN in the report):**
- `model.map_family: factorized` (default `dense` = v1, bit-for-bit unchanged against a pre-change snapshot, 912 entries, 0 mismatches); keys `householder_k`, `map_scale_eps`.
- Properties 1–4 hold; no `[M, d, d]` tensor in forward or backward (dispatch-mode detector with dense control).
- 32 new tests pass; 15 injected defects all caught (0 vacuous). Full suite: same 24 pre-existing failures in `tests/test_baselines.py` before and after.
- Map storage on ACM (d = 32): k = 1 → 31.0×, **k = 2 → 15.75×**, k = 4 → 7.94× smaller than dense. **k = 4 fails the 10× rule; candidates are k ∈ {1, 2}.**
- Parameters at d = 32: 1.62M (dense) → 0.46M (k = 2).
- Independent review: no blocking defect; its findings were fixed in `924be78`.

**Not complete:**
- **Accuracy (§1.6) not run.** Moved into D4: v2 supports both map families, so dense vs factorized k ∈ {1, 2} is compared on the v2 model (at a `d` where dense fits the GPU). It is already a row of the §4.6 ablation table.
- **Peak memory only ≈ 2× lower end to end** (d = 8: 6.08 → 3.00 GiB). At d = 32 neither family fits 8 GB in v1. The peak is dominated by v1 face/edge machinery that D4 deletes (gathered path maps `[F·L, d, d]`, face caches `[F, L+1, d, d]`, closure maps `[F, d, d]`, edge cochain).

**Decisions taken:**
- **D1a — one scale per edge, shared by both sides.**
  - **Problem.** With per-side scales, each edge transport carries the ratio `c_tgt/c_src`, and path transports multiply these ratios. `z_f` is then dominated by the node with the largest `‖T_v‖`, so a path cell degenerates toward one node, and float32 can overflow or underflow on long paths. Stability of `Δ₀` is **not** the issue: Prop. 3 holds for any positive scales once the Laplacian is normalized.
  - **Rejected band-aid: bounding `c` to `[e⁻², e²]`.** Each ratio can still reach `e⁴`, so a path of length L still spans `e^{4L}` (≈ 3·10³ for L = 2, ≈ 9·10⁶ for L = 4). It adds two arbitrary knobs and saturating gradients without removing the cause.
  - **Fix.** `F_src = c_e·R_{ψ,src}·H(u_src)`, `F_tgt = c_e·R_{ψ,tgt}·H(u_tgt)`. The edge transport `F_src⁻¹F_tgt = (c_e/c_e) × rotation` is exactly orthogonal, so `‖T_v‖ = 1` at every position of every path, at any length. `c_e` becomes a learned edge weight in `Δ₀` (the edge term is scaled by `c_e²`).
  - **Direction is unaffected.** (Storage of edges is superseded by D7, §9.1: one cell per fact, not per PyG relation type; the argument below still holds per cell.) Edges are oriented 1-cells built per PyG relation type (`src/data/hetero_flatten.py:66`); `R_{ψ,src} ≠ R_{ψ,tgt}`; reflection vectors depend on side and endpoint order; `c_e` is computed from the ordered input. The sheaf Laplacian was symmetric in v1 too; orientation only fixes signs.
  - **What is lost (DERIVED, numerically checked):** about one number per edge (at k = 2, d = 32: 130 → 129 per-edge degrees of freedom).
    - Per-side scales that factor as `c_{v◁e} = s_v·c_e` are a node rescaling `y_v = s_v x_v`, and the degree normalization cancels it **exactly** (normalized operators differ by 4e-16 on a two-cycle test graph). In the path term this part only rescales the gate `w_f`.
    - The genuinely lost part is **scale holonomy**: the product of `c_tgt/c_src` around a cycle differing from 1 (generic per-side scales change the normalized operator by 0.58 on the same graph). On a tree nothing is lost. This is exactly the part that produced the magnitude blow-up; rotational holonomy, which is kept, preserves norms.
  - **Theory.** Transports lie in SO(d) (D1b) with edge weights `c_e²`: a **weighted connection Laplacian** (vector diffusion maps, Singer & Wu; *Sheaf Neural Networks with Connection Laplacians*, Barbero et al. 2022; NSD's orthogonal model class). Prop. 1–3 are unchanged.
  - **Computation.** Norm-preserving transports in float32 at any path length (matters for longer `ℓ_max` on H2GB); inverses are pure transposes; path transport skips all scale multiplications; the path part of `D` becomes `γ·Σ_{f∋v} w_f (1 − 1/|f|)`; M fewer stored floats.
  - **Open (empirical).** Whether the data benefits from magnitude-changing cycles: ablation "per-side vs per-edge scale" (§4.6). If per-side wins by more than 1 std, keep per-side scales, restrict `ℓ_max`, and state the conditioning caveat in the paper.
- **D1b — SO(d) wording** (§1.2).
- **Closure maps stay dense until D4 removes them.** Making them matrix-free now would save ≈ 0.7 GB per `[F, d, d]` tensor on ACM but cost L chained map applications per use, and D4 deletes closure edges from `Δ₀` anyway.
- **Activation checkpointing** (added in `11ba6c3`) costs roughly one extra forward pass of the checkpointed parts during training only (helper `checkpoint_if_grad`); evaluation is unaffected. D4 should expose it as a config switch so it can be turned off where memory allows.

**Housekeeping before D4:** the working tree has uncommitted deletions (`HINPool/`, `HINormer/`, …), untracked `vendor/` and `references/`, and whole-file diffs that look like line-ending changes (e.g. `src/configs/config.py`). Commit or clean them so the D4 diff is reviewable.

---

## 2. D2 — AMPSS redesign → WISP (decided 2026-11-08; §2.1–2.4 kept as history, §2.5 is the decision)

### 2.1 Why it must change (VERIFIED-CODE / VERIFIED-PAPER)
- **Knobs:** `SchemaDiscoveryConfig` alone has ≥ 18 settings (`max_schema_candidates`, `max_instances_per_schema`, `max_instances_per_anchor`, `max_total_instances`, `max_total_incidences`, `selector_strategy`, `selector_seed`, `selector_max_examples`, `selector_holdout_ratio`, `selector_num_attempts`, `selector_probe_l2`, `selector_probe_max_iter`, `selector_stop_tolerance`, `selector_tie_tolerance`, `selector_min_examples_per_label`, `instance_weight_mode`, `warmup_epochs`, `operator_anneal_epochs`), plus `num_label_prop_steps`. None are reported in the paper.
- **Heuristics that contradict the thesis:** Stage I scores schemas from path counts, endpoint diversity and propagated labels at the far endpoint, using type-blind label propagation (fix spec F9).
- **Does not scale:** Python loop over candidates × directions × anchors × 50 attempts; hard error past 128 candidates (fix spec F10).
- **Weak evidence:** full AMPSS vs predefined schemas on DBLP = 96.01 vs 95.75 Macro-F1, single numbers, no std (manuscript ablation table).

### 2.2 Requirements any new mechanism must meet
1. **Scalable:** no full enumeration of instances on the data graph; instances produced per batch.
2. **Path-aware scoring:** no endpoint-only proxies.
3. **≤ 3 knobs.**
4. **Leak-free protocol:** training labels only; state exactly which labels and splits are used.
5. **Reportable:** selected schemas per dataset; significance vs fixed schemas; comparison with LMSPS; whether selected schemas shift with heterophily (reviewer R4).
6. **Interface for the rest of v2:** outputs per-batch path instances `(schema_id, node sequence, relation sequence)` plus a per-instance weight usable by D4.

### 2.3 Candidate directions (not decided — to be discussed)
- Enumerate schemas on the schema graph (types only) up to `ℓ_max`; learnable per-schema weights trained end to end; optional progressive pruning.
- Keep Stage II's idea (instance-level weights within a schema) — reviewer R3 praised it.
- Must be positioned against LMSPS (scalable meta-path search) and DiffMG.

### 2.4 Interim (so D3 is not blocked)
Use fixed, literature-standard schema lists per dataset plus uniform per-batch instance sampling (`r` instances per target per schema). This is also the "fixed schemas" control that the new selector must beat.

---

### 2.5 Decision (2026-11-08): WISP
Authority: methodology spec v1.3 (§2 R2/R8/R12, §3.2, §4 Props. S/K/R/E, §5 C2/C2b/C-global). Analysis documents (claude.ai project "H2SNN-fix"): `D2_decisions.md`, `D2_from_objective.md`, `D2_long_term_design.md`, `D2_interpretability.md`, `LMSPS_critical_analysis.md`.

**Why selection is not a component.** Hypothesis H is about instances. By Prop. E, cells with uniform weights are (in expectation) a polynomial of the fact diffusion, so the only freedom cells add is a weight that depends on the whole instance (R12). Schema-level selection (hard or soft) cannot depend on intermediates; it is needed only for cost and for interpretation. Cost is handled by sampling; interpretation by read-outs. A predefined list cannot be a requirement, because a dataset may have none or an incomplete one; LMSPS's own Limitation G (best max hop not automatic) and its search on the validation split are the costs of making selection a stage.

**The three decisions.** (a) candidates: walk-first, no schema list (a3); (b) sampling: stated target `p` + defensive IS with an optional adaptive proposal (b4); (c) weighting: `w_f = exp(g(ξ_f))` (c4), which contains uniform (c1) and soft per-schema weights (c3) as restrictions, giving the C2 ladder. Rejected: hard schema selection / two phases (c2: the old Stage I in another form; discrete, seed-sensitive, two runs; kept only as the post-hoc C-global analysis), RL sampling policies (c5: biased unless corrected, high variance), sparsemax/entmax schema weights (zeroed schemas get no gradient and cannot recover).

**Correction recorded (2026-11-08).** An earlier proposal used a stop action and called `ℓ_max` "only a safety cap". That was wrong: with a stop probability of 1/(|actions| + 1) (about 1/6 at ACM papers) the cap binds often, and any target needs a support. 1.3 states `ℓ_max` as the support (pre-registered, curve reported) and draws the length uniformly, which makes the initialisation length-neutral without a data-dependent bias; the target's shape inside the support is largely absorbed by `g`, which sees the relation sequence.

**Kept from AMPSS:** Stage II's idea (instance-level weights; reviewer R3 praised it). **Removed:** Stage I, its ≥ 18 settings, the DFS-with-quota sampler, the 128-candidate limit. **Knobs:** `ℓ_max` (support), `r` (precision); the proposal switch affects variance only and is fixed by a gate.

**Not part of WISP (candidates, each with its own test):** participation `κ` (N1), channel metric `A_f`, a partial schema list as a prior inside `q` (never a filter). Out of scope (author, 2026-11-08): large-graph mini-batch results, branching meta-structures (L7), AHGNN/LMSPS as mandatory baselines, transfer of read-out schemas to SeHGNN, unseen relations (L8).

---

## 3. D3 — Scalability (first priority)

### 3.1 Principle
**Scale v2, not v1.** Do not build mini-batch support for components that D4 may remove (edge cochain `s`, edge initializer, dense face maps). Every v2 component is built with a switch so D4 can be decided by ablation.

### 3.2 Components
1. **Factorized maps** — D1.
2. **Normalized Laplacians + bounded step** (fix spec Fact E): `Δ = D^{-1/2} L D^{-1/2}` with degrees from the **full graph**; `η = σ(θ)`; log `η·λ_max` per run (fix spec H9).
3. **Equivariant normalization** replacing `nn.LayerNorm(d)` on stalks (fix spec Fact D).
4. **Per-batch path instances** (interim sampler §2.4, later D2): build closure edges and 2-cells inside the batch so each batch is a closed subcomplex (fix spec P4 lemma).
5. **Mini-batch training:** PyG `NeighborLoader` on `HeteroData`. State the approximation: exact only if the batch covers the receptive field (≈ `K·ℓ_max` hops).
6. **No edge states to scale.** D4 deletes the edge cochain, `L₁`, the edge initializer and the dense face maps, so mini-batching covers node states, factorized maps and per-batch path cells only.

### 3.3 Dataset ladder

| Tier | Datasets | Mode | Purpose |
|---|---|---|---|
| 0 | DBLP, ACM, IMDB (HGB) | Full batch | Regression vs v1; D1 acceptance |
| 1 | Freebase (HGB): MLH 0.550, adjusted homophily 0.199 on the B–B schema (VERIFIED-REPO-DOC: `docs/pyg_dataset_eda_homophily_report.html`); size ≈ 180K nodes / ≈ 1M edges (UNVERIFIED) | Full batch | First moderately heterophilic heterogeneous benchmark; featureless → identity embeddings |
| 1 | Controlled rewiring of ACM/DBLP (fix spec H6), incl. homophilic control | Full batch | Causal evidence that gains track heterophily |
| 1 | IEEE-CIS-G (H2GB), ≈ 2.9M edges (UNVERIFIED, from revision-plan arithmetic) | Full batch if it fits, else mini-batch | Smallest H2GB set |
| 2 | ogbn-mag, mag-year, H-Pokec (H2GB) | Mini-batch | Large-scale heterophilic evidence; H2GB loaders, splits, metrics |

### 3.4 What to measure (answers R1, R3, R4)
Per dataset: `ℓ_max`, `r`, paths / closure edges / incidences per batch, sampling ms/batch, s/epoch, time-to-accuracy, peak GPU memory, parameter count.
Sweeps: `r ∈ {2,4,8,16}`, `ℓ_max ∈ {2,3,4}`, ogbn-mag subgraph fraction 10/25/50/100% with log–log slopes vs. incidences.

### 3.5 Acceptance
- Tier 0: v2 within 1 std of v1; ≥ 10× lower peak memory.
- Mini-batch vs full batch on ACM and DBLP within 1 std (separately tuned).
- Tier 2: ogbn-mag trains on a single GPU (memory size to be fixed once compute is known).

---

## 4. D4 — Path coupling: vertex–2-cell sheaf Laplacian (agreed)

### 4.1 Decision
One mechanism only: each path-supported 2-cell `f` is coupled directly to **every** node on its boundary (endpoints and intermediate nodes) through the node→2-cell restriction maps of the same cellular sheaf. Options (a) nonlinear Hodge coupling, (c) unsigned incidence coupling, and any (a)+(b) combination are **rejected**: (a) moves 2-cell information to nodes only through a nonlinearity (by Lemma A′), needs edge states at scale, and has no controllable mechanism; (c) is neither a gradient flow nor a sheaf operator.

### 4.2 Model
```
h₀ = TypeEncoder(x)
repeat K steps:
    z_f = Σ_{v∈f} T_v x_v / |f|                                   # 2-cell state in F(f): its best consistent value
    x  ← x − η [ Δ₀ x + γ Σ_{f∋v} w_f T_vᵀ (T_v x_v − z_f) ]       # one normalized sheaf Laplacian
logits = MLP([x_K, h₀])
```
- `Δ₀`: sheaf Laplacian on **observed** edges with D1 factorized maps, degree-normalized with full-graph degrees.
- `T_v = F_{v◁f} = F_{e◁f} F_{v◁e}`: node→2-cell restriction, forced by functoriality to be ordered parallel transport along the path (gauge `A_f = I`). Applied matrix-free with `PathTransportOperator`. ~~Correction (2026-11-06): per-position values are needed, so Horner's O(L) form does not apply.~~ **Withdrawn (D9):** because `T_vᵀT_v = s_v²I`, `T_vᵀ(T_v x_v − z_f) = s_v² x_v − T_vᵀ z_f`, so per-position values are not needed and the cost is 2L transport steps per path (§9.3).
- `w_f`: per-path gate from the ordered path (positional encoder over `h₀` and the relation sequence) — continues Stage II. D2 schema weights will multiply `w_f`.
- `η = σ(θ)`; `γ ≥ 0` learnable; both terms normalized (node degree and path size).
- With D1a (one scale per edge) every `T_v` is orthogonal, so `‖T_v‖ = 1` and the path part of the normalization diagonal is simply `γ Σ_{f∋v} w_f (1 − 1/|f|)`.

### 4.3 Deleted from v1 (complexity reduction)
Edge cochain `s` and `L₁` (down and up terms) · `EdgeInitializer` · `δ₁` face term in the dynamics · `d²`-output `A_f` head · `[F, ℓ+1, d, d]` face maps · closure edges inside `Δ₀` · edge `LayerNorm` · edge pooling in the readout · operator ramp/anneal tied to the face term.

### 4.4 Theory (paper content)
| Result | Statement | Status |
|---|---|---|
| Lemma A′ (Hodge separation) | For any polynomial or heat flow of the Hodge–Dirac operator `D = δ + δᵀ` on `C⁰ ⊕ C¹ ⊕ C²` with `δ₁δ₀ = 0`, the `C² → C⁰` block is zero, and the `C⁰ → C⁰` block depends only on `δ₀`. Linear Hodge dynamics cannot carry 2-cell information to nodes. Motivates the design. | DERIVED; numerical check: block ≤ 1e-11 for `Dᵏ`, k = 1..8, and `e^{−D}`; node output change 1e-16 (linear) vs 1e-2 (with nonlinearity) |
| Prop. 1 (forced transport) | If `δ₁δ₀ = 0` on a 2-cell with invertible maps, the node→2-cell maps are ordered parallel transport, and the closure edge's relative map `(F^{cl}_tgt)⁻¹F^{cl}_src` equals the path holonomy. Intermediate nodes and relation order enter the maps; no new parameters. | DERIVED (coefficient matching per node) |
| Prop. 2 (pullback Laplacian) | The coupling term is the Hodge 0-Laplacian of the pullback sheaf `π*F` on the star expansion (vertices = nodes ∪ 2-cells; edges = incidences `v ≤ f`; maps `T_v` and `I`). It is PSD; `ker(Δ₀ + γL_P) = ker Δ₀ = H⁰(X; F)`, i.e. path cells never contradict the graph sheaf. With free per-incidence maps the kernel generically collapses. | DERIVED; numerical check: dim ker = 3 (Δ₀), 3 (with transport path cell), 0 (with free maps) |
| Prop. 3 (stability) | Let `D` be the block diagonal of `L = Δ₀ + γL_P` (scalar × I per node for conformal maps). Each observed-edge term satisfies `L_e ≼ 2D_e` and each path term satisfies `L_f ≼ (1 − 1/|f|)⁻¹ D_f ≼ 2D_f` for `|f| ≥ 2`. Summing, `L ≼ 2D`, so `λ_max(D^{-1/2} L D^{-1/2}) ≤ 2` and any `η = σ(θ) ∈ (0, 1)` gives `η·λ_max < 2`. Isolated nodes (`D_v = 0`) get a zero row. | DERIVED; numerical check: max normalized λ = 2.0000000000000027 over 200 random complexes with per-side scales in `[e⁻², e²]` and path cells (bound is tight; the derivation holds for any positive scales, so it covers D1a); unnormalized λ reached 1.4e6. D4 test T5 re-checks it in the real code |

### 4.5 Sheaf status and wording
- **The whole 2-complex carries a genuine cellular sheaf** (a functor on the face poset), in v1 and v2. On a graph, any choice of maps is a cellular sheaf; with 2-cells the real condition is **functoriality**: the composite `v → e → f` must agree for both boundary edges through `v`. The transport construction enforces it.
- **Conditions:** invertible maps (D1); simple paths (no repeated node); closure map = path holonomy. Restricting to a closed subcomplex per mini-batch still gives a sheaf.
- **States are cochains, not sections.** Diffusion pulls them toward global sections and reaches them only as t → ∞ (oversmoothing). Write "diffuses over a cellular sheaf on the 2-complex, including its node→2-cell maps"; do **not** write "the model makes the graph a sheaf".
- **Do not call the node operator the Hodge Laplacian of the 2-complex** (Lemma A′ forbids it). Call it the **vertex–2-cell sheaf Laplacian** of `F`.
- **Closure edge `e_cl`:** part of the definition (closes the boundary so `f` is a CW 2-cell; its map is forced to the holonomy), not of the computation. State this explicitly.
- **Hypergraph distinction** (pre-empt reviewers): the computation has the form of a hyperedge `{A₁, P, A₂}` with a hub, but (i) the maps are determined by the boundary edges, (ii) they depend on path order, (iii) they preserve all global sections; free hypergraph-style maps do not (Prop. 2). Position against Sheaf Hypergraph Networks (Duta et al. 2023) and MAGNN.
- **Figure:** keep the 2-cell figure; add arrows from `A₁`, `P`, `A₂` into `f` labelled `F_{v⊴f}`, show `z_f` in `F(f)`, draw `e_cl` as the dashed boundary.

### 4.6 Ablations (each answers one reviewer question; 10 seeds; paired tests vs full)
| Variant | Question |
|---|---|
| `γ = 0` (no path cells) | Do path cells matter? |
| Transport `T_v` vs free per-incidence maps | Why sheaf transport rather than a hypergraph? |
| Path cells vs endpoint meta-path edges (same budget) | Do intermediate nodes matter? |
| Path cells vs holonomy closure edges in `Δ₀` (v1-style shortcut) | Does the 2-cell hub add anything over a path-conditioned endpoint edge? |
| Learned `w_f` vs uniform | Does path weighting matter? |
| `h₀`-only readout MLP | Sanity: is diffusion used at all? |
| Dense vs factorized k ∈ {1, 2} | Map family (D1; closes §1.6). k = 4 is excluded by the 10× storage rule (§1.7) |
| Per-edge scale `c_e` vs per-side scales `c_{v◁e}` | Does losing scale holonomy cost accuracy? (D1a) |
| Tied facts (D7) vs untied mirror relations | Does storing each fact twice, with unrelated maps, help or hurt? (D7; control only, never tuned) |
| Direction-blind maps (same frame both sides, no side code; the analogue of DSNN's `q = 0`) | Does edge direction matter? |
| Walk sampler (D8) vs instance-uniform sampling (same `r`) | Do diverse intermediate nodes matter? (R2 W3, R4) |
| WISP weight ladder `uniform ⊂ schema ⊂ endpoints ⊂ full` (spec v1.3 C2) | Do the intermediates of an instance matter (P2), beyond its relation sequence? |
| WISP vs fixed literature schemas (same incidence budget) | What does schema-free sampling cost or gain? (reported, not a claim) |

**Kill criterion:** if `γ = 0` or endpoint meta-path edges are within 1 std of the full model on Tier 0 **and** Tier 1, the core claim fails; narrow the paper (fix spec §7 option B). No additional component is added to rescue it.

### 4.7 Status (2026-11-06) — source: `docs/H2S-NN_D4_report.md`
**Verified (VERIFIED-RUN in the report):**
- `model.architecture: v2`; v1 bit-for-bit unchanged (0/912 after every unit); D1a per-edge scale with orthogonal transports (≤ 3e-15 float64, ≤ 1e-6 float32 at L = 4); all seven ablation switches, none reachable from v1.
- Theory holds in the real code: `L_P` equals the dense Kron-reduced reference (2.6e-15); PSD; sections preserved (kernel d) with transport maps and destroyed (kernel 0) with free maps; normalized `λ_max ≤ 1.99999` over 60 fixtures vs unnormalized up to 2.4e16; one-step reach of the intermediate node.
- 44 new tests; 6/6 required mutations caught; full suite unchanged (same 24 pre-existing `test_baselines` failures).
- `D` is the per-incidence sum because ACM has 16 self-loops; `L ≼ 2D` still holds.

**Memory (ACM, RTX 4060 8 GB):** v2 factorized k = 2 at d = 32 peaks at **2.52 GiB** (v1 cannot run at d = 32). At d = 8: v2 1.11 GiB vs v1 dense 6.08 GiB (**5.5×**). Parameters at d = 32: 1.62M (v1 dense) → 0.31M (v2).
**Decision:** the plan's "≥ 10× peak memory" target is replaced by these measured numbers. The 10× figure was a self-set target, not a reviewer requirement; the paper reports the actual scaling curves (§3.4) instead.

**Smoke run (20 epochs, ACM, d = 32):** stable (η·λ_max ≤ 0.96, loss finite). η and γ barely moved (0.50, 0.99): with full-batch training there is one optimizer step per epoch, so AdamW at lr 1e-3 can move each scalar's raw parameter by at most ≈ 0.02 in 20 epochs (≈ 0.2 in 200). The mean gate `w_f` fell from 0.57 to 0.25 — watch it in Tier 0; gates collapsing toward 0 would point at the kill criterion.

### 4.8 Task heads after v1 removal (D6)
Principle: one encoder; a head changes only when it is forced to, and no component is added without an exact reason.

| Head | Change | Exact reason |
|---|---|---|
| Node classification | None (done in D4): `MLP([x_K, h₀])` | — |
| Link prediction | **Forced, minimal:** the existing decoders `dot`, `distmult`, `concat_mlp` now read the node representation `[x_K, h₀]`. The decoder stays a per-dataset choice, selected on validation only and reported | v1's link readout pooled edge states, which v2 deleted. Nothing new is added. Path information reaches the decoder through `x_K` (Lemma A′; D4 test T6) |
| Graph classification | **Forced:** SheafCovPool (and the other v1 pools) replaced by **sum pooling of `[x_K, h₀]` → BatchNorm → MLP**; the `pooling_type` setting disappears | SheafCovPool's edge half reads deleted edge states. P1: the model is not gauge-equivariant (h₀ and the maps live in one learned basis), so the readout needs only permutation invariance. P2: SheafCovPool reduces each node's 3 stalk vectors to their 3×3 Gram matrix (6 of 3d numbers; `descriptor_dim = 3*3`), losing any information that differs by a rotation. P3: sum + MLP is permutation-invariant and can be injective on multisets (Xu et al., ICLR 2019), unlike mean or max. Repo evidence: SheafCovPool docstring says hierarchical variants "tied or lost against a plain invariant readout"; ENZYMES gain within std |

**Considered and rejected (no exact necessity):**
- *Sheaf-consistency link decoder* (`β_r − ‖Q_u x_u − Q_v x_v‖²`): principled, but it would add a claim needing its own ablation and confound comparison with baselines that have no sheaf maps. Revisit only if a result gives a reason.
- *Per-relation / per-schema energy features* for graph classification: a hypothesis without demonstrated need.
- *Disjoint message/supervision split* for link prediction: training positives are both in the message graph and supervised (VERIFIED-CODE `link_prediction_dataset.py:922`, `link_prediction_trainer.py:177,281`). This is a protocol question shared by all message-passing models; keep the HGB protocol. **Open check:** confirm HGB's reference code and our baseline runs train the same way; if not, match them.

**Follow-up before Tier 0 (small, no new test campaign):** a separate optimizer parameter group for the scalars `θ_η`, `θ_γ` (higher lr, no weight decay); reject v1 keys that v2 ignores silently; keep `docs/D4_evidence/acm_prepared.pt` out of git.

---

## 5. Parallel track — D5 integrity fixes
- Baseline provenance (fix spec H8): GCN/GAT/HAN/SeHGNN cells in Table 1 match published HGB/SeHGNN values and appear in no run output in the repo (VERIFIED-CODE search); HINormer IMDB Macro/Micro look swapped. Rerun or label every cell.
- IMDB heterophily under fair multilabel definitions (fix spec H5).
- Cite and benchmark MAGNN (in `references.bib`, never cited in the manuscript text — VERIFIED-PAPER) and Simple-HGN.

---

## 6. Order of work and gates

| Step | Work | Gate |
|---|---|---|
| 1 | D1 implemented and verified (§1.7). D5 audit still open | Done for D1 |
| 2 | **Done (§4.7).** D4 task card (`H2S-NN_agent_prompt.md` Part C): D1a per-edge scale, v2 model with normalized vertex–2-cell Laplacian, deletions, ablation hooks, ACM memory measurement | D4 card tests and stop conditions |
| 2b | D7–D9 task card (`H2S-NN_agent_prompt.md` Part D) on branch `D1_add`: fact layer, path-cell builder, O(L) path term | Part D tests and stop conditions |
| 2c | **Done.** D10 task card (Part E): cooperative gates, `ablation.coupling` (`16f6ad3`) | — |
| 2d | Finish D7 (option A, tied+walk defaults, D7 report) → D11 task card (Part F): learning fix, gradient-flow and factorization tests, diagnostics, `pair_attention` test variant → core-claim screen | Part F tests; screen triggers (§12.6) |
| 2e | **Part G (D12):** pipeline-correctness gate first (G0 done 2026-11-08: data PASS; feature-free R-GCN beats D11 by 5–10 points → G0b gap ladder before G1); then relational context, filter-bank encoder, tests (Props. 5–7, Lemma N), cost gate. Exploratory D11 screen recorded, not a design input (spec §6) | Part G gates |
| 2f | **Part H (D2 → WISP).** Starts only after Part G is closed (no concurrent edits of `h2snn_model.py`, `path_sampler.py`, `config.py`). Units H0–H6 (sampler, weights + ladder, adaptive proposal, read-outs, known-answer tasks, smoke + cost); no real-data accuracy run before the G0b gap is explained (spec §6 WISP gate) | Part H tests, mutation checks and stop conditions |
| 3 | Tier 0 experiments on v2: §4.6 ablations incl. dense vs factorized k ∈ {1, 2} (closes §1.6) and the C2 weight ladder on WISP cells | §1.6, §3.5 Tier-0, §4.6 and spec §5 C2 kill rule |
| 4 | Tier 1 (Freebase, rewiring, IEEE-CIS-G) | Heterophily evidence; decide claim scope (fix spec §7 A/B) |
| 5 | Mini-batch; Tier 2 (ogbn-mag, mag-year, H-Pokec) | §3.5 Tier-2 criteria |

## 7. Open questions (author decisions)
1. Available compute (GPU model and memory, hours) — sets Tier-2 feasibility and seed counts.
2. Target venue (theory-leaning vs. data/scale-leaning).
3. Default `k` if several pass.
4. ~~Keep closure edges inside `Δ₀`?~~ Resolved by D4: closure edges are definitional only; the v1-style holonomy shortcut is kept as an ablation control (§4.6).
5. Eq. 9 of the submitted paper (relation-wise neighbourhood context for the restriction maps) is not in the v2 code; the maps read `h₀`. Restore it, or delete it and its motivating sentence from the paper.
6. Theory-driven assumptions to re-examine: see §11 (items 2–4 have no decision yet).

## 8. Agent prompt
Implementation work is driven by `H2S-NN_agent_prompt.md` (role, evidence rules, reality-before-theory rule, self-correction loop, task cards). Task cards D1 (Part B), D4 (Part C), D7–D9 (Part D), D10 (Part E) and D11 (Part F) are done; D12 (Part G) is in progress; D2/WISP (Part H) runs after Part G.

---

## 9. D7–D9 — Fact layer, path-cell builder, O(L) path term (agreed 2026-11-06)

Source: `Reviews/H2S-NN_methodology_check.md` (independent numerical check of `e7c3ab1`) and the discussion of 2026-11-06, including *Sheaves Reloaded: A Directional Awakening* (Fiorini et al., ICLR 2026, arXiv:2506.02842).

### 9.1 D7 — Fact layer
**Problem (VERIFIED-RUN on the local processed HGB files).** All 10 relation pairs in ACM, DBLP and IMDB are exact mirrors (`E(r′) = reverse(E(r))`), so each fact is stored twice, as two 1-cells with unrelated maps. On one link (d = 8, init) the consistent space drops from 8 to 0, `Δ₀` is twice as large, and the second copy carries no direction: one oriented cell is already direction-aware (reversing it changes the operator by 0.83; block (v,u) = block (u,v)ᵀ; transport non-symmetric, ‖τ − τᵀ‖ = 1.05). ACM `cite` additionally has 25 reciprocal pairs inside the relation (two distinct facts each) and 8 self-loops. Reverse relations exist for message passing, where messages flow src → tgt only; the sheaf Laplacian is symmetric, so the reason disappears. The submitted paper is silent on this (Appendix counts DBLP's 3 mirror pairs as "six relation types").

**Definition (one rule, no cases, no knobs).**
```
ρ(r) = r′  iff  E(r′) = { (v, u) : (u, v) ∈ E(r) }      exact equality; r′ = r allowed; otherwise ρ(r) = ∅
fact      = stored typed edge up to inversion:  (u, r, v) ≡ (v, ρ(r), u)
1 fact    = 1 oriented 1-cell, stored in the orientation of its lower relation id (ties: lower (u, v))
roles     = each relation has a source role and a target role; they coincide when ρ(r) = r
```
Consequences (not separate rules): mirror pairs collapse to one cell per fact; a relation equal to its own reversal (e.g. TU graphs; UNVERIFIED per dataset) gives one cell per unordered pair with one shared role, which permutation equivariance requires (otherwise the operator depends on the arbitrary stored orientation); reciprocal facts of a directed relation (ACM `cite`, whose inverse is `ref`) stay two cells; an edge without a partner is its own fact.

**Status.** `ρ` is part of the input graph, like `φ` and `ψ`: deterministic, computed before training, independent of labels and metrics, no threshold, no config key. It is **not a hyperparameter and not a heuristic.** The modelling assumption it encodes, "an exactly reversed relation was added for message passing and repeats the same facts", is stated in the paper and its effect is measured by the ablation `fact_layer: untied` (§4.6). **`fact_layer` is never tuned:** tied is the definition of the model (Prop. 2 is meaningful only there); untied is a controlled experiment reported separately. Known limits: exact equality is brittle to a missing reverse edge (the pair then stays untied, i.e. the safe fallback; the pipeline reports unmatched edges); a directed relation whose stored edges are all reciprocal is treated as self-inverse, which loses nothing observable.

**Transport with direction.** For hop `j` of a path, if fact `e_j` is stored `v_{j−1} → v_j` (sign +1), `Q_j = F_src⁻¹F_tgt`; if stored `v_j → v_{j−1}` (sign −1), `Q_j = F_tgt⁻¹F_src = Q_fwdᵀ` under D1a. Functoriality `T_{v_{j−1}}F_{v_{j−1}◁e}⁻¹ = T_{v_j}F_{v_j◁e}⁻¹` gives both cases (DERIVED). `FaceGroup.face_edge_sign` carries the sign.

**Schemas** are sequences of (relation, direction) over the tied relations. A transition between node types that resolves to more than one (relation, direction) must be spelled out in the config (e.g. ACM paper→paper: citation chain vs co-citation); otherwise error. A schema and its reverse describe the same cells.

**Link prediction.** A held-out positive removes its single fact; the leak check asserts that no held-out fact, in either orientation, is in the message graph or on any sampled path.

**Paper.** Problem formulation: "We call relation r′ the inverse of r, ρ(r) = r′, when E(r′) is exactly the reversal of E(r); such relations are added by benchmarks so that messages flow in both directions. A fact is a stored typed edge up to inversion, (u, r, v) ≡ (v, ρ(r), u), and each fact is one oriented 1-cell whose endpoints take the source and target roles of its relation; the two roles coincide when ρ(r) = r. A path that traverses a fact against its orientation applies the adjoint transport." Prop. 1 is restated with signed boundaries. Related work: position against DSNN (direction through restriction maps, Hermitian/symmetric operator; H2S-NN learns the direction-dependent transport per relation and node pair instead of a fixed phase `q`). Appendix: relation counts after merging. Limitation: the coupling strength is equal in both directions (`c_e²`); only the transport differs (same as DSNN).

### 9.2 D8 — Path-cell builder
**Problems (VERIFIED-RUN).** (a) DFS-with-quota has no sampling distribution: with 3 authors × 30 papers and `r = 20`, every one of 200 seeds used 1 of 3 intermediates (random walk: 3 of 3); with a 300-paper hub author, 48–50 of 50 paths went through the hub. (b) A path and its reversal become two cells (3,690 of 16,300 in a synthetic PAP test), so weights depend on sampling luck. (c) Python loop per anchor; the preprocessing bottleneck for Tier 1–2.

**Target operator.** For anchor `a` and schema `π`: `L_P^target = Σ_π Σ_a E_{walk ~ p(·|a,π)}[ 1{walk simple} · L_{f(walk)} ]`, `p` = product of `1/out-degree` (in the hop's relation and direction) along the walk.

**Estimator.** `r` walks per (anchor, schema), weight `1/r` each; rejected walks contribute 0. Cells with the same identity are merged and their weights **summed** (so `E[L̂_P] = L_P^target`; dropping duplicates would bias it). The merged weight `m_f` multiplies `w_f` in both `L_P` and `D`, so Prop. 3 holds for every draw.

**Identity.** Key = tuple of fact ids in canonical orientation (the lexicographically smaller of the sequence and its reversal), plus the schema group of the canonical orientation. Not the node sequence: paths through different reciprocal facts are different cells.

**Anchors.** As now: every node of the schema's start type (node classification); per graph (graph classification); message graph only (link prediction).

**Protocol.** Training resamples every epoch (seed = base seed + epoch); validation and test use one fixed draw (base seed). Knobs: `ℓ` (from the schemas), `r`. Reported per dataset: cells, incidences, sampling ms.

### 9.3 D9 — O(L) path term
`(L_f x)_{v_i} = w_f m_f (s_i² x_i − T_iᵀ z_f)`, `z_f = (1/n) Σ_i T_i x_i`. `Σ_i T_i x_i = x_0 + Q_1(x_1 + Q_2(x_2 + ⋯ + Q_L x_L))` (L steps); `T_0ᵀz = z`, `T_iᵀz = Q_iᵀ(T_{i−1}ᵀz)` (L steps). Identity valid whenever `T_iᵀT_i = s_i²I` (all map families in the code). Checked against the current `PathCellTerm`: float64 ≤ 2e-12, float32 ≤ 7e-5, the same order as the current code vs a dense reference (VERIFIED-RUN, methodology check §3). Transport steps per path: L = 2: 6 → 4; L = 4: 20 → 8.

### 9.4 Not changed by D7–D9
D1/D1a maps, D4 operator and normalization, Prop. 2–3, heads (D6), ablation switches of §4.6 (new rows added). Prior v1 results used the DFS sampler and untied relations; all Tier 0 numbers come after Part D.

### 9.5 Decisions taken during the D7 run (2026-11-07; source: agent Phase-1 and U1–U7 reports on branch `D1_add`)
- **Link prediction split by fact.** `ρ` is computed on `link.dat` before splitting; a held-out positive removes its fact in both orientations. The old per-relation split leaked: on a fixture with a mirror relation, the reverse orientation of 8 of 21 held-out facts stayed in the message graph (now 0). Every link-prediction adapter takes its split from the same bundle builder, so baselines get the same split. Earlier link-prediction numbers (all methods) may be inflated on datasets with mirror relations; LastFM, PubMed, Amazon, YouTube are not on disk yet (UNVERIFIED).
- **MUTAG loader bug fixed (separate commit `2322d28`).** Labels were assigned by position while PyG reorders edges: 1,771 of 7,442 bonds had the wrong type, in 188/188 graphs. Every earlier MUTAG result (H2S-NN and the typed baselines h2gformer, hetero2net, h2sgnn, hgb_han) used corrupted bond types; rerun all.
- **Schema-free walks** for link prediction and graph classification (no configured schemas): uniform over all facts at each hop; cells grouped by the realized (relation, direction) sequence. Its target distribution is stated in the D7 report.
- **Baseline input format (option A, to confirm in the D7 reply).** Baselines keep their standard stored graph (`fact_layer: untied`, `path_sampler: dfs`, no resampling, set in `bundles.py::_base_config`). All methods receive the same facts; message-passing baselines need both stored directions. Paper sentence: "All methods receive the same facts; baselines receive them in their standard stored form with reverse relations, and H2S-NN receives them as tied facts."
- **Limitation contrast (CSNN).** Cooperative Sheaf Neural Networks (Ribeiro et al., arXiv:2507.00647v3, VERIFIED-SOURCE in `references/`) treat each undirected edge as two directed edges with per-node send/receive maps and the composed operator `(Δ_in)ᵀΔ_out`, whose spectrum can be complex (their §5). D10 provides node-level directional strength while keeping a real spectrum.

---

## 10. D10 — Cooperative coupling with a real spectrum (agreed 2026-11-07)

### 10.1 Why (reality, not theory)
- The symmetric operator forces equal influence in both directions. In heterogeneous graphs influence is plausibly unequal: a hub (an ACM subject touches tens of thousands of papers through PSP) should be able to inform its papers without being dragged toward each of them; under heterophily a node should be able to stop listening.
- Outside evidence points the same way (UNVERIFIED for H2S-NN): the strongest HGB models are directional attention models (HAN, HGT, Simple-HGN, SeHGNN); selective message passing is reported to help on heterophilic graphs (Co-GNN, Finkelshtein et al. 2024; CSNN, its authors' claim).
- The earlier plan kept symmetry because Prop. 2–3 need it (§9.1 limitation). That was a theory-driven constraint; §11 now lists such constraints for testing.

### 10.2 Operator
```
Δ̂   = D^{-1/2} (Δ₀ + γ L_P) D^{-1/2}                 # unchanged (D4, D7–D9)
g_v = σ(a_gᵀ h₀_v + β_g),   b_v = σ(a_bᵀ h₀_v + β_b)   # listen / broadcast gates in (0, 1)
x  ← x − η · G Δ̂ B x,   G = diag(g) ⊗ I_d,  B = diag(b) ⊗ I_d     # then layer dropout, as now
```
- Effective strength from v to u: `g_u b_v`; from u to v: `g_v b_u`. `g_u → 0`: u stops listening; `b_v → 0`: nobody hears v (CSNN's cooperative behaviours).
- Gates start near 1 (initial value is a validated config key) so training starts close to the symmetric model.
- Encoder-level: applies to edges and path cells, in all three tasks.

### 10.3 Theory (DERIVED; numerically checked)
- With `S = (G⁻¹B)^{1/2}`: `S (G Δ̂ B) S⁻¹ = (GB)^{1/2} Δ̂ (GB)^{1/2}`, which is PSD. So the eigenvalues are real and lie in `[0, 2·max_v g_v b_v] ⊂ [0, 2)`; `η = σ(θ) ∈ (0, 1)` keeps the explicit step stable. In `y = S x` the update is the gradient flow of a sheaf Dirichlet energy.
- Fixed points: `G Δ̂ B x = 0 ⇔ B x ∈ ker Δ̂` (gates > 0) — the sections, seen through the broadcast gates. Prop. 2 is restated this way.
- Check (200 random sheaves, gates in [0.01, 1]): max |Im λ| = 7e-17; max λ / (2·max g b) = 0.90. The CSNN-style composed operator on the same graphs had |Im λ| up to 2.6e4.

### 10.4 Limits
- **Node-level, not per-edge:** a node listens to all its neighbours with one gate. Per-edge directional strength breaks the similarity to a symmetric operator (complex spectrum possible). CSNN's directionality is node-level too.
- Gates read `h₀` only and stay fixed through the K steps, like the maps (§11 item 2).
- Novelty is modest: diagonal similarity scaling is standard, and learned node-wise diffusion rates exist in prior work (cite after a literature check). Pitch: cooperative sheaf diffusion that keeps a real spectrum and its stability guarantee.

### 10.5 Ablation and decision rule
`ablation.coupling: cooperative | listen_only | symmetric` (`listen_only`: `B = I`; `symmetric`: all gates 1, i.e. the current model). On Tier 0 and Tier 1, 10 seeds, paired tests with Holm correction: if `cooperative` is not significantly better than `symmetric`, the default reverts to `symmetric` and the gates leave the paper. `listen_only` shows which half matters. **Superseded by D11 (§12):** the gates are part of the model (always on); `symmetric` remains as a required ablation row.

---

## 11. Reality-before-theory audit (2026-11-07)
Rule (binding for agents, `H2S-NN_agent_prompt.md` Part A): a constraint kept only because one of our propositions needs it is a hypothesis, not a fact; it is tested against the data and outside evidence, and theory never removes an ablation.

| # | Assumption imposed for theory | Why we have it | Reality check | Test / status |
|---|---|---|---|---|
| 1 | Equal coupling strength in both directions | Prop. 2–3 need a symmetric operator | Directional attention models are strong on HGB; selective message passing helps under heterophily | **Resolved by D11 (§12):** per-node asymmetry as a metric gradient flow |
| 2 | Linear K-step diffusion with maps (and gates) computed once from `h₀`: no nonlinearity, no channel mixing between steps; maps never see diffused states | Gradient-flow reading; batch independence (fix spec F5); memory | NSD (Bodnar et al. 2022) and CSNN recompute maps per layer from the current states and use weight matrices plus a nonlinearity per layer; our flow may be under-expressive | Open. Candidate ablation for Tier 0: "layered" variant (maps and gates from `x_t` each step, per-step channel mixing and nonlinearity) vs current. Cost: K× map computation; mini-batch receptive field grows |
| 3 | One scale per edge (no scale holonomy) | Orthogonal transports, conditioning (D1a) | Unknown whether magnitude-changing cycles matter in real data | Existing ablation `map_scale: per_side` (§4.6) |
| 4 | Equal weight of all path positions in `z_f` (mean) | `z_f` is the minimizer of the cell energy | Endpoints and intermediate nodes may matter differently; `w_f` and the maps only partly compensate | Open, low priority; decide after Tier 0 |

---

## 12. D11 — Final model (agreed 2026-11-07; design rationale added the same day)

### 12.0 Design rationale (paper order: setting → assumptions → gap → properties → method)
**Setting.** Nodes have a type and their own attributes; edges have a relation type; labels live on a target type; neighbouring nodes often disagree (heterophily); the meaning of a connection between two target nodes depends on the whole metapath between them.

**Assumptions about the data** (each one is tested; §12.3):
| # | Assumption | Statement |
|---|---|---|
| A1 | Structured disagreement | Connected nodes are related by a relation- and node-dependent transformation, not by equality. Heterophily is systematic difference, not noise. |
| A2 | Path semantics | A metapath instance carries information its endpoints lack: its intermediate nodes and the order of its relations. |
| A3 | Coherence | Path-level information must not contradict edge-level information: a state consistent on every edge is consistent on every path cell. |
| A4 | Factorized trust | Influence of v on u = receptiveness of u × symmetric compatibility of the pair × reliability of v. Direction comes from node roles (attributes and type, through `h₀`); the pair contributes only symmetric compatibility. |
| A5 | Geometry from attributes | Maps, gates and path weights can be read once from `h₀`; diffusion propagates over that fixed geometry. (Weakest assumption; stated as such in the paper.) |

**Gap.** HAN/HGT/Simple-HGN/SeHGNN aggregate under a similarity assumption (no A1); MAGNN encodes intermediate nodes without coherence (no A3); NSD and HetSheaf are edge-only (no A2) and symmetric (no A4); sheaf hypergraph networks use free maps that destroy sections (no A3, Prop. 2); SheafAN/Res-SheafAN and CSNN are directional but have no objective (no P1) and no higher-order cells. (HetSheaf, SheafAN, CSNN, DNSD: VERIFIED-SOURCE; the HGNN characterisations: UNVERIFIED until each paper is opened.)

**Properties pursued.** P1 the update minimises a stated objective (known fixed points); P2 stable for any step size; P3 coherent across edges and 2-cells; P4 directional influence; P5 cost linear in edges and incidences.

**Derivation.**
| Requirement | Component | Result |
|---|---|---|
| A1 | Cellular sheaf; maps from relation, side and both nodes (D1, D1a) | Transports in SO(d) (D1b) |
| A2 + A3 | Path 2-cells with node→2-cell maps forced by functoriality | Prop. 1 (forced transport), Prop. 2 (sections preserved) |
| 2-cell information must reach nodes | Vertex–2-cell sheaf Laplacian (D4) | Lemma A′ (Hodge flows cannot) |
| A4 + P1 | Listen/broadcast gates on the symmetric Laplacian (D10) | **Prop. 4: A4 ⇔ P1** (§12.2) |
| P2 | Degree normalisation, `η ∈ (0, 1)` | Prop. 3 |
| P5 | Matrix-free maps (D1), O(L) path term (D9), walk sampler (D8), tied facts (D7) | Measured (D1, D4, D10 reports) |

Reviewer-facing answers built on this section: `docs/possible_reviewer_qa.md`.

### 12.1 Model
```
h₀ = TypeEncoder(x)                                   # tied facts (D7), walk-sampled path cells (D8)
maps (D1/D1a), gates g_v, b_v, path weights w_f  ←  computed once from h₀           # one sheaf
repeat K steps:   x ← x − η · G Δ̂ B x                 # Δ̂ = normalized edge + path-cell sheaf Laplacian (D4, D9)
head([x_K, h₀])                                       # D6 heads
```
- **Asymmetry is part of the model:** strength(u ← v) ∝ g_u · c_e² · b_v. Each direction differs (node "personalities": how much a node listens, how much it broadcasts); edges differ through the symmetric weight `c_e²` and the direction-dependent rotation.
- **One block.** Maps and gates are computed once from `h₀`, so the model diffuses over one sheaf and every mini-batch is a restriction of it.

### 12.2 Theory (DERIVED; numerically checked)
- **Metric gradient flow.** With `E(x) = ½⟨Bx, Δ̂ Bx⟩` (sheaf energy of the broadcast-weighted states) and the diagonal node metric `M = B G⁻¹`, the update direction `G Δ̂ B x` equals `M⁻¹∇E` exactly. So each step is a gradient step of a sheaf energy in a learned metric: asymmetric influence comes from geometry on the nodes, not from leaving the sheaf framework. Check: `|M⁻¹∇E − GΔ̂Bx| ≤ 9e-16`; `E` never increased over 100 random graphs × 50 steps with η ≤ 1 (largest change −4e-5).
- **Stability:** `M^{-1/2}(BΔ̂B)M^{-1/2} = (GB)^{1/2} Δ̂ (GB)^{1/2}` has eigenvalues in `[0, 2·max g b]`, so `E` decreases for any η ≤ 1 (D10 check: real spectrum, λ_max = 0.93 of the bound).
- **Fixed points:** `B x ∈ ker Δ̂` (global sections seen through the broadcast gates); path cells never contradict them (Prop. 2).
- **What the model cannot express:** circular (non-reversible) asymmetry, e.g. A dominates B, B dominates C, C dominates A, and pair-specific direction beyond node roles. By Kolmogorov's reversibility criterion, any operator with that property loses the real spectrum and the energy theory.
- **Prop. 4 (factorization; A4 ⇔ P1).** Let `w(u←v) > 0` be the total influence of v on u through edges and 2-cells, with transports `Q_vu = Q_uvᵀ`. The update `x ← x − ηAx` is the gradient flow of a quadratic sheaf energy in a positive per-node metric **iff** `w` satisfies detailed balance (`π_u w(u←v) = π_v w(v←u)`, some `π > 0`) **iff** `w(u←v) = g_u · s_uv · b_v` with `s` symmetric. *Proof (DERIVED):* `A = M⁻¹L` with `L` symmetric forces `m_u w(u←v) = m_v w(v←u)`; conversely set `s_uv = π_u w(u←v)`, `g = 1/π`, `b = 1`. Consequence: D11's form is the most general directional influence that keeps an objective. Check (VERIFIED-RUN, scratch script 2026-11-07; becomes Part F T-F3b): 100 random sheaves (d = 3, rotations, edges + 3-node path cells): D11-form weights give detailed-balance residual ≤ 1.6e-14, `‖MA − (MA)ᵀ‖/‖MA‖ ≤ 3e-15`, real spectrum; free per-direction weights give residual ≥ 0.78 (no metric exists) and max |Im λ| = 0.32.

### 12.3 Alternatives and the assumption each one tests
An alternative is not "rejected"; it is excluded by a stated assumption, and an experiment tests that assumption. If the test refutes the assumption, the model changes as pre-registered in §12.6, and the paper reports the refutation.
| Alternative | Excluded by | Test (same sheaf, same 2-cells, one variable changed) |
|---|---|---|
| Pair attention (SheafAN/Res-SheafAN-type strength, free per direction) | A4 | `coupling: pair_attention` (Part F U4b) |
| Multi-block (maps/gates recomputed from `x_t`, per-step channel mixing and nonlinearity) | A5 | `h0_only` row; trigger in §12.6 |
| No path cells / endpoint edges / holonomy closure edges | A2 | `γ = 0`, `endpoint_edges`, `holonomy_edges` |
| Free per-incidence maps (hypergraph) | A3 | `path_maps: free` |
| Diagonal maps | A2 (relation order needs non-commuting maps) | diagonal-map row |
| Per-side scales | D1a (no scale holonomy) | `map_scale: per_side` |

Facts about pair attention used in the paper (VERIFIED-SOURCE):
- SheafAN's adjacency includes self-loops (Eq. 3), so attention **can** stop listening by attending to itself. Do not claim otherwise.
- HetSheaf Table 11: SheafAN (non-residual, their implementation) is far below the Laplacian-based sheaf families on IMDB (54–56 vs 61–63 Macro-F1), and not uniformly worst on ACM/DBLP (Bundle-Poly is lower on both). Res-SheafAN was never tested on heterogeneous graphs.
- What we may claim: the two differ in assumption A4 and in what can be proved (P1: energy, real spectrum, characterised fixed points, readable gates — Prop. 4), plus SheafAN has no higher-order cells. What we may **not** claim: an accuracy advantage before the test; that an energy improves accuracy (with few steps fixed points are not reached); stability as a difference (with orthogonal maps and row-stochastic weights attention is non-expansive too); that only D11 combines edges and 2-cells in one rule (one softmax over self, edges and cells does it too and also preserves sections); exact mini-batches (neighbour sampling truncates the receptive field in both; D11 only avoids per-batch softmax renormalisation).
- Outside evidence leans toward attention, not against it: attention-based models (e.g. Simple-HGN) are strong on HGB (UNVERIFIED here; open the HGB paper before citing). A4 is a hypothesis with a real chance of being refuted; that is why its consequence is pre-registered in §12.6.

### 12.4 Learning fix (required; without it "learned" is false)
Full-batch training takes one optimizer step per epoch, so parameters initialized in the flat region of a sigmoid barely move (D4 smoke: η, γ ≈ constant; D10 smoke: gates ≈ 0.95). Fix: gate heads, `θ_η`, `θ_γ` in their own optimizer group (lr = 10 × base, weight decay 0); gates initialized at 0.6 (`model.coupling_gate_init`).

### 12.5 Known negatives (for the paper and the reviewers)
- Node-wise learned diffusion rates exist in prior work: the novelty of the gates is modest; the claim is "listen/broadcast cooperation realized as a sheaf gradient flow with a stability guarantee" (literature check required before writing it).
- One linear block may be under-expressive (resembles O(d)-NSD): measured by the `h0_only` row.
- Orthogonal maps vs diagonal: HetSheaf and DNSD report diagonal ≥ orthogonal on real data; path cells need non-commuting maps. Diagonal maps are an ablation row; if diagonal wins, the relation-order claim falls.
- Required ablation rows (evidence, not model options): symmetric (all gates 1), diagonal maps, no path cells (γ = 0), endpoint edges, h₀ only.

### 12.6 Core-claim screen and pre-registered triggers
ACM, DBLP, IMDB × {full D11, γ = 0, endpoint_edges, h0_only, pair_attention} × 3 seeds, one fixed config, no tuning, parameter count reported per row; reference baseline SeHGNN (or the strongest one that runs) and HetSheaf's published Table 11 numbers. Log per run: η, γ, gate means/mins and their trajectories; normalized energy after each step (edge and path-cell parts); for one ACM model per variant, `dim ker Δ̂` and a linear probe of the labels on the projection onto `ker Δ̂`.
| Result | Then |
|---|---|
| Path cells don't beat γ = 0 or endpoint_edges | Stop; narrow the paper before anything else |
| full ≈ h0_only (within 1 std) | Expressivity work (multi-block), with a parameter-matched comparison |
| Path cells win and full > h0_only | Tier 0 with 10 seeds and the full ablation table |
| Gates still do not move after the learning fix | Report it; the gates are then fixed hyperparameters and the paper says so |
| `pair_attention` > full by > 1 std on ≥ 2 of 3 datasets | A4 is in doubt: confirm in Tier 0 (10 seeds, paired tests, Holm). If confirmed, A4 is refuted: the model becomes pair attention on the same sheaf, P1 and the energy reading are dropped, stability is stated as non-expansiveness, Prop. 4 explains what was given up, and the paper reports the refutation. Decided now, before any result |
| `pair_attention` ≤ full + 1 std | A4 retained; the row is reported as the test of A4 |
