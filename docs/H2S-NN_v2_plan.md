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
| D2 | Redesign AMPSS while keeping its intent (task-driven meta-path selection). Mechanism not yet chosen. | **Agreed — flagged, mechanism open** |
| D3 | Scalability is the first engineering priority, so that heterophilic heterogeneous benchmarks (mostly large) become usable. Scale **v2**, not v1. | **Agreed** |
| D4 | 2-cells reach nodes through the **vertex–2-cell sheaf Laplacian** (option b only); edge-cochain machinery is deleted. | **Agreed (2026-11-06)** — see §4 |
| D5 | Fix baseline provenance and IMDB heterophily (fix spec H5, H8) in parallel. Independent of D1–D4. | Recommended (parallel track) |

### What this plan supersedes in the fix spec
- **P3 (structured maps) as written is replaced by D1.** P3's `F = R_ψ·diag(s_e)` with one map per edge gives `F_src = F_tgt`, so every edge transport is `I`, the path holonomy collapses to `I`, and closure edges become plain endpoint edges (DERIVED). D1 fixes this with side-specific, node-dependent factors.
- **P5 (uniform relation-first walk sampler) is deferred to D2.**
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
4. **Face side too** — path transports applied matrix-free (sequential/Horner along each path); `A_f` reduced to an orthogonal gauge (identity) plus a scalar weight, removing the `d²` head and the `[F, ℓ+1, d, d]` tensor.

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
  - **Direction is unaffected.** Edges are oriented 1-cells built per PyG relation type (`src/data/hetero_flatten.py:66`); `R_{ψ,src} ≠ R_{ψ,tgt}`; reflection vectors depend on side and endpoint order; `c_e` is computed from the ordered input. The sheaf Laplacian was symmetric in v1 too; orientation only fixes signs.
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

## 2. D2 — AMPSS redesign (flagged; mechanism not yet decided)

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
- `T_v = F_{v◁f} = F_{e◁f} F_{v◁e}`: node→2-cell restriction, forced by functoriality to be ordered parallel transport along the path (gauge `A_f = I`). Applied matrix-free (Horner-style accumulation along the path, one pass per step).
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

**Kill criterion:** if `γ = 0` or endpoint meta-path edges are within 1 std of the full model on Tier 0 **and** Tier 1, the core claim fails; narrow the paper (fix spec §7 option B). No additional component is added to rescue it.

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
| 2 | **Next: D4 task card** (`H2S-NN_agent_prompt.md` Part C): D1a per-edge scale, v2 model with normalized vertex–2-cell Laplacian, deletions, ablation hooks, ACM memory measurement | D4 card tests and stop conditions |
| 3 | Tier 0 experiments on v2: §4.6 ablations incl. dense vs factorized k ∈ {1, 2} (closes §1.6); decide D2 mechanism | §1.6, §3.5 Tier-0 and §4.6 criteria; D2 decision written here |
| 4 | Tier 1 (Freebase, rewiring, IEEE-CIS-G) | Heterophily evidence; decide claim scope (fix spec §7 A/B) |
| 5 | Mini-batch; Tier 2 (ogbn-mag, mag-year, H-Pokec) | §3.5 Tier-2 criteria |

## 7. Open questions (author decisions)
1. Available compute (GPU model and memory, hours) — sets Tier-2 feasibility and seed counts.
2. Target venue (theory-leaning vs. data/scale-leaning).
3. Default `k` if several pass.
4. ~~Keep closure edges inside `Δ₀`?~~ Resolved by D4: closure edges are definitional only; the v1-style holonomy shortcut is kept as an ablation control (§4.6).

## 8. Agent prompt
Implementation work is driven by `H2S-NN_agent_prompt.md` (role, evidence rules, self-correction loop, task cards). Task card D1 (Part B) is done; task card D4 (Part C) is next.
