# H2S-NN — Revision Specification (handoff document)

**Purpose.** This document is the working spec for revising H2S-NN after the WSDM 2027 rejection (paper #850). It is written for an AI or human collaborator who will plan and implement the changes. It states what we know, what we only believe, what must be tested, and in what order.

**Reading rules for the next agent.**
1. Every claim carries a status tag. Do not upgrade a tag without new evidence.
   - `VERIFIED-CODE`: read directly in repo `psarnkaew/h2s-nn` at commit `3228f4c`. File paths are given.
   - `VERIFIED-PAPER`: read in the submitted manuscript or supplement.
   - `VERIFIED-SOURCE`: read in a cited external paper or official docs.
   - `DERIVED`: proved in this document; check the proof.
   - `UNVERIFIED`: belief from memory, arithmetic, or reading without running. It must be checked before use.
   - `PROPOSAL`: a design idea with a stated rationale but no evidence yet.
2. **Do not assume any proposal works.** Each proposal has a falsification test and a kill criterion, and the test decides.
3. **Prefer narrowing a claim to building a new component.** A smaller paper whose every claim survives is the goal.
4. Repo conventions in `CLAUDE.md` apply. In particular: never invent citations or results, keep the implementation boundary explicit, and run everything in the `h2snn` conda environment.

---

## 1. Verdict in one paragraph

The core object, path-supported 2-cells in a cellular sheaf with transport-built edge-to-face maps, is sound, and all four reviewers liked it. The paper failed on evidence:
- (a) the heterophily claim rests on one dataset;
- (b) nothing isolates what the 2-cells contribute;
- (c) cost is unmeasured;
- (d) the reported protocol does not match the code.

Reading the code adds three model-level problems the reviewers only partly saw:
- (e) under the current linear Hodge dynamics, the 2-cell (face) term can only damp one component of the edge signal, and it reaches nodes only through a mean-pooled readout;
- (f) dense per-edge restriction maps, not 2-cells, dominate memory;
- (g) AMPSS selects schemas with an endpoint-level proxy, which contradicts the paper's own thesis.

The revision must first **measure** (e), then decide between a modest fix and narrowing the claim.

---

## 2. Verified facts the plan depends on

| # | Fact | Status | Source |
|---|---|---|---|
| F1 | Node and edge diffusion are decoupled linear explicit-Euler steps: `x ← x − η₀L₀x`, `s ← s − η₁L₁s`, with optional dropout. | VERIFIED-CODE | `src/models/diffusion.py` |
| F2 | `L₀ = δ₀ᵀδ₀` and `L₁ = δ₀δ₀ᵀ + δ₁ᵀWδ₁`, both **unnormalized**. | VERIFIED-CODE | `src/models/coboundary_ops.py` |
| F3 | Closure edges are first-class 1-cells inside `δ₀`, so `L₀` includes them. | VERIFIED-CODE | `h2snn_model.py` (closure concat before `apply_l0`) |
| F4 | Restriction maps are dense `[M, d, d]`, two sets (src and tgt), built per edge from `[h_u, h_v, rel_emb]` through FastHPP scaled-orthogonal maps. | VERIFIED-CODE | `src/models/sheaf_maps.py` (`NodeToEdgeSheaf`), `src/utils/memory_guard.py` |
| F5 | Maps depend only on the type-encoder output `h` of the endpoints and the relation, so they are batch-independent. | VERIFIED-CODE | `NodeToEdgeSheaf.forward` |
| F6 | `s₀ = MLP([h_u, h_v, rel_emb, δ₀h₀])` (with `use_delta0_init=True` by default). | VERIFIED-CODE | `EdgeInitializer`, `config.py` |
| F7 | `x*` and `s*` pass through `nn.LayerNorm(d)` before the readout. | VERIFIED-CODE | `h2snn_model.py` lines ~701–702 |
| F8 | The node readout concatenates `[x*, h₀, mean_e F_{v⊴e}ᵀ s*_e]`, where the mean is unsigned over all incident edges, including path and closure edges. | VERIFIED-CODE | `readout.py`, `pooling_utils.py` |
| F9 | AMPSS Stage I builds features from path counts, endpoint diversity and the **mean propagated label at the far endpoint**, then selects greedily with a linear probe. Label propagation ignores types. | VERIFIED-CODE | `node_stage1.py` (`_node_schema_feature_block`), `stage1_selector.py`, `preprocessing.py` (`label_propagation`) |
| F10 | Stage I path sampling is a Python loop over candidates × directions × anchors × 50 attempts. It raises an error past the candidate guard (128). | VERIFIED-CODE | `schema_sampling.py`, `node_stage1.py`, ACM config |
| F11 | The ACM config uses `d=32`, schema length 2–4, `K_S=6`, ≤48K instances, ≤144K incidences, `K=4`, `η=0.05`. | VERIFIED-CODE | `configs/experiments/acm_h2snn_ampss.yaml` |
| F12 | Graph-classification configs use `split_protocol: stratified_holdout` (HINPool 80/10/10 × 5). The paper states 10-fold CV × 5 seeds. | VERIFIED-CODE, VERIFIED-PAPER | `configs/experiments/{mutag,enzymes}_h2snn.yaml`, `src/baselines/campaign.py`, manuscript §4.1 |
| F13 | The shared baseline budget is 300/200/300 epochs with patience 100. The docstring records earlier unequal budgets (H2SGNN 1000/100 vs GCN 300/30). | VERIFIED-CODE | `src/baselines/budget.py` |
| F14 | MLH counts a multilabel pair as "same label" only on exact vector equality. | VERIFIED-PAPER | text under Eq. (5) |
| F15 | MLH values: DBLP 0.1277, ACM 0.2639, IMDB 0.7634. | VERIFIED-PAPER | Table 1 |
| F16 | ACM cost: 3.011 s/epoch, 9,260 MB, 1.61M parameters. | VERIFIED-PAPER | Table 11 |
| F17 | All H2GB datasets are node classification and are trained mini-batch (`hetero_neighbor`, example fan-out [20,15,10], V100 32 GB, 5 seeds). | VERIFIED-SOURCE | Lin et al. KDD 2025; H2GB docs |
| F18 | HetSheaf reports the HGB protocol with a 150-run sweep and best validation configuration, and no seed variance. | VERIFIED-SOURCE | arXiv:2409.08036 |

---

## 3. Mathematical diagnosis (what the 2-cell term actually does)

### 3.1 Lemma A — signed node readouts are blind to the face term (DERIVED)

Assume `δ₁δ₀ = 0` (verified to float32 roundoff per `CLAUDE.md`), constant `η₁`, and no dropout (eval mode). Then for every step k,

```
δ₀ᵀ s_{k+1} = δ₀ᵀ s_k − η₁ δ₀ᵀδ₀δ₀ᵀ s_k − η₁ (δ₁δ₀)ᵀ W δ₁ s_k = (I − η₁ L₀) δ₀ᵀ s_k
⇒ δ₀ᵀ s_K = (I − η₁ L₀)^K δ₀ᵀ s₀ .
```

- **Consequence:** any node quantity built from the **signed** coboundary adjoint of the edge state is independent of `W` and of the 2-cells, for every K.
- **Coupling the systems does not help:** coupling x and s through `δ₀ᵀ` would not let face information reach nodes.
- **What does not hold:** the claim is only about signed coupling. The unsigned mean in the readout (F8) is not a coboundary adjoint and is not covered by this lemma.

### 3.2 Lemma B — under linear Hodge flow, the face term only damps the curl component (DERIVED)

- **Decomposition:** by the Hodge decomposition `C¹ = im δ₀ ⊕ ker L₁ ⊕ im δ₁ᵀ` (Lim 2020), with orthogonality taken in the inner product used to define the adjoints.
- **Invariant subspaces:** `L₁` maps each summand to itself. Write `s₀ = g + h + c` for the gradient, harmonic and curl parts. Then:
  - `g_K = (I − η₁δ₀δ₀ᵀ)^K g`
  - `h_K = h`
  - `c_K = (I − η₁δ₁ᵀWδ₁)^K c`
- **Consequence 1:** the 2-cells influence `s_K` **only through the curl component of the learned initialization `s₀`**. They act as a low-pass filter that makes edge states around each 2-cell more cycle-consistent under transport. They do not move information from intermediate nodes into new subspaces.
- **Consequence 2:** if `c ≈ 0`, meaning `s₀` is mostly a gradient (note F6 feeds `δ₀h₀` into `s₀`), the face term does almost nothing.
- **What this means for the paper:** "the face term propagates path information" is not the right description of v1. The accurate description is "the face term regularizes edge states toward transport-consistency around each 2-cell; its effect reaches target nodes once, through an unsigned mean over incident edges (F8)."
- **R2 is half right:** information does reach targets through path edges e₁ and e_ℓ, not only through closure edges. It reaches them once, diluted by the mean, and only via the curl part of `s₀`.

### 3.3 Fact C — closure edges are a HAN-like shortcut in node diffusion (DERIVED from F3)

- **Shortcut:** for target-to-target schemas (APA, PAP, MAM), a closure edge connects two target nodes, so `L₀` contains a transport-weighted meta-path adjacency.
- **Risk:** part of the gain may come from this endpoint shortcut. That would be exactly the mechanism the paper criticizes. This is testable (H3).

### 3.4 Fact D — LayerNorm breaks the gauge-invariance argument (DERIVED from F7) — UNVERIFIED impact

- **Why it breaks:** `nn.LayerNorm(d)` subtracts the mean over stalk coordinates and applies per-coordinate affine parameters. For orthogonal `R`, `LN(Rs) ≠ R·LN(s)` in general.
- **Which claims depend on it:** the gauge co-transform argued in the docstring of `pool_edge_stalks_to_nodes` and the gauge-invariance claim of SheafCovPool assume states co-transform with the stalk basis. With LN applied before pooling, that assumption fails.
- **Fix options:**
  - (a) remove LN;
  - (b) replace it with an O(d)-equivariant normalization, such as dividing by `‖s‖₂` per stalk or RMS without per-coordinate gain;
  - (c) withdraw the gauge-invariance claim.
- **Test:** apply random orthogonal gauges and check output invariance numerically (T-D below).

### 3.5 Fact E — explicit Euler stability is not guaranteed (DERIVED from F1, F2) — UNVERIFIED in practice

- **Stability condition:** for PSD `L`, `x ← x − ηLx` is stable iff `η·λ_max(L) < 2` (standard; LeVeque 2007).
- **Why it can fail:** for scaled-orthogonal maps with scale ≤ c, `λ_max(L₀) ≤ 2c²·deg_max`. The Laplacian is unnormalized and η is unbounded (learnable softplus), so the condition can fail on high-degree nodes, which H2GB has. Mini-batch sampling also changes degrees.
- **Fix:** degree-normalized sheaf Laplacian `D^{-1/2} L D^{-1/2}` (Hansen & Ghrist 2019; Bodnar et al. 2022 use a normalized form), using full-graph degrees, plus a bounded step `η ∈ (0,1)`.
- **Test:** power-iteration `λ_max` on ACM and on an ogbn-mag subgraph, logged against the learned η.

---

## 4. Falsifiable hypotheses (run these before building anything large)

Each row is cheap relative to the redesign. Run them on the **current v1 code** first; the results decide how much redesign is justified.

| ID | Hypothesis | Test | Prediction if true | Kill / decision rule |
|---|---|---|---|---|
| H1 | v1's face term carries information that matters for prediction. | Eval-time masking of the `δ₁ᵀWδ₁` term (variant B), and separately masking face contributions on interior path edges vs. closure edges. 10 seeds; DBLP, ACM, IMDB. | Accuracy drops beyond 1 std when the face term is removed. | If B ≈ F (within 1 std, paired test n.s.), state in the paper that v1's gain does not come from the face term, and either adopt P2 (§5) or narrow the claim. |
| H2 | `s₀` has a non-trivial curl component. | Hodge decomposition of `s₀` and `s*` on sampled subgraphs, via LSQR with matrix-free `δ`. Report energy fractions. | Curl fraction is clearly > 0 and is reduced by diffusion. | If the curl fraction is ≈ 0, the face term is inert by Lemma B. Drop `δ₀h₀` from `s₀`, or redesign the initialization, before anything else. |
| H3 | The gain is not explained by the closure-edge shortcut in `L₀`. | Weighted inner product on 1-cochains: `L₀^{W₀} = δ₀ᵀW₀δ₀` with closure weight `w_cl ∈ {0, 1}`. This leaves `δ₁δ₀=0` intact. Retrain. | With `w_cl=0`, performance stays within 1 std of `w_cl=1`. | If performance collapses at `w_cl=0`, the model is mainly an endpoint model. Narrow the claim (option B) or require P2 to recover it. |
| H4 | Path-internal information reaches the target representation. | Linear probe from the target's pooled edge summary and `x*` to the type or label of the **intermediate** node (e.g., actor genre in M–A–M). Compare with variants A and B. | Probe accuracy for F exceeds A and B significantly. | If F ≈ B ≈ A, the "preserves internal path structure" claim is unsupported. |
| H5 | IMDB is heterophilic under a fair multilabel definition. | Recompute MLH with per-label averaging and with Jaccard distance; also compute adjusted homophily (Platonov et al. 2023). | IMDB remains clearly heterophilic. | If not, no HGB dataset supports the heterophily claim. H2GB becomes mandatory, or adopt option B. |
| H6 | Gains grow with heterophily. | Stratify test nodes by local meta-path heterophily `h_v` into 5 bins, plot `F − A` with bootstrap CIs, and run a degree-preserving rewiring sweep `p ∈ {0,…,80}%` on ACM and DBLP. | Monotone increase; CIs of the first and last bins do not overlap. | If false, drop "heterophilic" from the title and abstract (option B). |
| H7 | Dense maps, not 2-cells, dominate memory. | `torch.cuda.max_memory_allocated` with maps only, 2-cells disabled, and full model, on ACM. | Maps account for ≥ 40% of peak (estimated 4.5 of 9.3 GB). | If false, re-target the cost work at whatever dominates. |
| H8 | Reported baselines were rerun under our protocol. | Diff every Table 1–3 cell against published values (HGB, SeHGNN, HINormer, etc.) and against our run logs. | Cells match our logs, not the publications. | UNVERIFIED suspicion: several Table 1 cells (e.g., GCN/GAT/HAN on DBLP and IMDB) may equal HGB's published numbers digit-for-digit. If so, label them or rerun all of them. This is a reporting-integrity issue and is P0. |
| H9 | The learned η satisfies the stability bound. | Log `η·λ_max` per run. | `< 2` everywhere. | If violated, adopt the normalized Laplacian (Fact E) before scaling up. |
| H10 | The current gauge-invariance claim holds in code. | Random orthogonal gauge test (T-D). | Output change ≈ float error. | If it fails (expected, Fact D), fix the normalization or withdraw the claim. |

---

## 5. Model changes, ordered by evidence required

Each item lists **rationale, specification, code location, acceptance tests, risks**, and what would make us abandon it.

### P1. Correctness fixes (do regardless of H1–H4)
1. **Normalized Laplacian and bounded step** (Fact E).
   - **Spec:** `Δ₀ = D₀^{-1/2} L₀ D₀^{-1/2}`, with `D₀` the block-diagonal of `L₀` computed on the **full graph**; similarly for `L₁` (down and up parts normalized by their own diagonals). Use `η = σ(θ)`.
   - **Code:** `coboundary_ops.py`, `diffusion.py`.
   - **Acceptance:** `δ₁δ₀=0` test still passes; `η·λ_max < 2` logged; HGB accuracy within 1 std of v1.
2. **Equivariant normalization** (Fact D).
   - **Spec:** replace `nn.LayerNorm(d)` on `x*` and `s*` with per-stalk `v/‖v‖₂·g`, where g is a scalar gain, or remove it.
   - **Code:** `h2snn_model.py`.
   - **Acceptance:** gauge test T-D passes; accuracy within 1 std.
3. **Honest protocol and reporting:** see §6.

### P2. Node–edge coupling (only if H1/H4 show v1's channel is weak) — PROPOSAL
- **Rationale:** Lemma A rules out signed coupling. The only route by which 2-cell effects reach nodes during diffusion is unsigned transport, which v1 already uses once, at the readout.
- **Spec:** `x ← x − η₀Δ₀^{W₀}x + γ₀ |B₀|ᵀ s`, `s ← s − η₁Δ₁s + γ₁ |B₀| x`, where `|B₀|ᵀ s (v) = mean_{e∋v} F_{v⊴e}ᵀ s_e` (the readout operator, F8). Use `γ ≥ 0` (learnable) and closure weight `w_cl` (H3).
- **Caveat (must be stated in the paper):** with `γ > 0` the dynamics are no longer a gradient flow of the sheaf Dirichlet energy, so stability must be checked empirically, or bounded by `η·‖L‖ + γ‖|B₀|‖ < 2` [UNVERIFIED bound, derive before claiming]. Novelty is modest; present it as a design consequence of Lemma A, not as a contribution in itself.
- **Acceptance:** H4 probe accuracy improves over v1; H1 masking now shows a larger drop; overall accuracy ≥ v1 − 1 std; `γ=0` reproduces v1 exactly (unit test).
- **Abandon if:** no improvement on H4 or accuracy after a reasonable tuning budget. Then narrow the claim instead.

### P3. Structured, matrix-free restriction maps (required for H2GB) — PROPOSAL
- **Rationale:** H7, and the arithmetic `2·M·d²` gives 346 GB for mag-year at d=32 (forward values only, UNVERIFIED in practice).
- **Spec:** `F_{v⊴e} = R_{ψ(e)} · diag(s_e)`, with `R_ψ ∈ O(d)` per relation type (Householder or FastHPP on a `|R|`-sized batch) and `s_e = softplus(MLP(h_u,h_v,ψ(e))) + ε`.
  - Apply as `R_ψ(s_e ⊙ x)` grouped by relation; never materialize `[M,d,d]`.
  - Inverse: `diag(1/s_e)R_ψᵀ`.
- **Must re-establish:**
  - (i) Transport/functoriality and `δ₁δ₀=0`. App. A needs only invertibility; re-run `tests/test_coboundary_chain.py`.
  - (ii) Order sensitivity: products of `R_ψ` across different relations do not commute, while pure diagonal maps do.
  - (iii) Gauge semantics: with non-scalar `diag(s_e)` the map is not conformal-orthogonal. Re-check any invariance claim (likely restrict it to the conformal case `s_e ∈ ℝ₊·1`).
- **Ablation:** dense (upper bound, HGB only), `R_ψ·diag`, `R_ψ·(scalar)`, k-Householder matrix-free (k ∈ {2,4}), diagonal-only.
- **Synthetic order task:** labels depend on relation order along paths with identical relation multisets. Diagonal-only should fail when positional encoding in `A_f` is ablated.
- **Acceptance:** memory reduction ≥ 10× on ACM; HGB accuracy within 1 std of dense; order task separates diagonal from `R_ψ·diag`.
- **Abandon if:** accuracy drops > 1 std on all HGB sets. Fallback: k-Householder.

### P4. Mini-batch training as restricted sheaves (required for H2GB) — PROPOSAL with DERIVED lemma
- **Lemma (DERIVED, standard):** for a closed subcomplex Y ⊆ X (each cell's faces are in Y), `F|_Y` is a cellular sheaf, its coboundaries are restrictions, and `δ₁δ₀=0` on Y.
  - **Batch-independence:** the global sheaf is well-defined because maps depend only on local inputs (F5). Normalization must use full-graph degrees, and path gates must depend only on `h₀` and the relation sequence.
- **Exactness:** batch results equal full-graph results at a target only if the batch contains its receptive field. That is about `K·ℓ_max` hops for the face term, plus one hop for degrees if computed in-batch. Fan-out [20,15,10] is an approximation and must be stated as such.
- **Spec:** PyG `NeighborLoader` on `HeteroData`; build 2-cells inside the batch, so the subcomplex is closed by construction.
- **Acceptance:** on ACM and DBLP, mini-batch is within 1 std of full-graph (separately tuned, per Bajaj et al. 2024); report time-to-accuracy.

### P5. Replace AMPSS with an in-batch typed path sampler — PROPOSAL
- **Rationale:** F9 shows the selector uses endpoint evidence, which contradicts the thesis. F10 shows it cannot scale. The ~15 knobs are unreported (R3 Q3, R4).
  - LMSPS (Li et al. 2024) is **not** a drop-in fix: it scores meta-paths from endpoint-aggregated features (SeHGNN-style), so it reproduces the same contradiction.
- **Spec:**
  - For each target in the batch, draw `r` self-avoiding typed walks of length `ℓ ~ U{2..ℓ_max}`. At each step, pick a relation uniformly among those available, then a neighbor uniformly. Simple paths are required by the App. A proof.
  - Build closure edges and 2-cells in-batch.
  - Soft gate `w_f = σ(g(z_f ‖ e_{rel-seq}))`, with `z_f` computed from `h₀` only (P4 batch-independence) and **no sparsity penalty** (avoids new knobs).
  - Optional: importance correction `1/(r·p(f))` (GraphSAINT-style; Zeng et al. 2020), reported with and without.
- **Knobs:** `ℓ_max`, `r` (two).
- **Inference:** fixed sampler seed, or average over T ∈ {1,4,8} samples (report T).
- **Paired ablations:** identical sampler seeds per epoch across variants.
- **Acceptance (pre-registered):**
  - (i) Within 1 std of AMPSS on DBLP and IMDB at equal incidence budget.
  - (ii) Runs on all four H2GB datasets.
  - (iii) Sensitivity over `ℓ_max ∈ {2,3,4}`, `r ∈ {2,4,8,16}` is reported.
  - (iv) Top-5 schema importance (mean gate per schema) has Jaccard ≥ 0.6 across seeds [threshold is a PROPOSAL; adjust before running, not after].
- **Risk:** uniform relation-first walks may under-sample useful long schemas (e.g., APCPA on DBLP); measure coverage. Fallback: sample schemas proportional to learned schema weights inside the same training run (single stage).

### P6. Readout
- **Spec:** replace the single unsigned mean over all incident edges by typed pooling, kept separate for observed, path-member and closure edges, then concatenated. This avoids diluting path signals at high-degree targets. Keep the `F_{v⊴e}ᵀ` transport before pooling.
- **Acceptance:** improves H4 probe and/or accuracy; otherwise keep the mean and report it.

### Deprioritized or rejected (do not do without new evidence)
- **Rebuilding on DNSD** (Bourgerie et al. 2026). Its benefit targets depth with a nonlinearity on `ΔX`. v1 is linear with K ≤ 6, so in the linear case adjacency vs. Laplacian is a reparameterization of the residual (`x − ηΔx = (1−η)x + ηÂx`). Its diagonal maps commute, which loses relation order, our core claim. Use it as a **baseline** only.
- **Hard-concrete / L0 gates on paths:** they add two knobs, and cost is already bounded by sampling.
- **Keeping AMPSS Stage I** with tweaks: the endpoint proxy is structural (F9).
- **SheafCovPool as a main contribution:** the gain is within std on one dataset; the gauge claim is currently broken (Fact D); graph tasks cannot test heterophily.

---

## 6. Experimental protocol fixes

1. **Graph classification (if kept):**
   - Either describe the actual HINPool holdout and defend it as HINPool does, or (preferred) rerun with `scripts/run_graph_classification_repeated_cv_wandb_tuning.py` switched to **StratifiedKFold**, with inner 10% model selection (Errica et al. 2020), identical folds for all models, and per-fold results.
   - Add chance and majority rows. Debug any baseline below chance (ENZYMES GCN Macro-F1 12.75, GAT Micro-F1 16.55 vs 16.7% chance).
   - Replace Table 2 entirely.
2. **Baseline provenance (H8):**
   - Every cell labeled rerun / published† / quoted‡, generated by script from artifacts (`src/baselines/published.py` already separates published from tuned).
   - Rerun anything produced before the `budget.py` fix (F13).
   - Investigate anomalies: HINormer IMDB Micro < Macro (64.65 vs 67.83); CSNN DBLP Micro < Macro; GAT LastFM ROC-AUC 51.14 ≈ chance; LatGRL IMDB 50.73.
3. **Baselines to add:**
   - Simple-HGN (HGB's reference model, absent from Table 1);
   - HetSheaf reimplemented on `neural-sheaf-diffusion/`, reproducing DBLP ≈ 94.97 under its protocol within ~1 point before use;
   - R-GCN / R-GAT and H2G-former, SeHGNN, LMSPS for H2GB;
   - DNSD on the flattened graph;
   - HINPool for graph-level work (code in repo `HINPool/`).
   - **Equal tuning budget:** e.g., 50 W&B Bayesian trials per model × dataset, with search spaces from `search_grids.py` published in the appendix.
4. **Statistics:**
   - 10 seeds for node and link tasks;
   - paired Wilcoxon vs. the best executable baseline with Holm correction;
   - Friedman + Nemenyi across datasets (Demšar 2006);
   - define exactly how std is computed (R1 Q3);
   - separate split and initialization variance where feasible (Bouthillier et al. 2021).
5. **Link prediction:** keep the HGB protocol, but add hard-negative evaluation and MRR/Hits@K (HeaRT; Li et al. 2023). Near-saturated AUPR (~98) cannot separate methods. Call Amazon and YouTube "multi-relational, single node type".
6. **Datasets:**
   - Add H2GB IEEE-CIS-G, mag-year, ogbn-mag, H-Pokec, using H2GB loaders, splits, metrics and mini-batch protocol (F17).
   - Report MLH (fixed per H5), adjusted homophily and class-insensitive homophily (Lim et al. 2021), on both the length-2 target–target views and the views actually sampled.
7. **Scalability (R1, R3, R4):**
   - Per dataset: `ℓ_max`, `r`, paths, closure edges and incidences per batch, sampling ms, s/epoch, time-to-accuracy, peak memory, parameters.
   - Sweeps over `r`, `ℓ_max`, and ogbn-mag subgraph fraction (10/25/50/100%), with log–log slopes vs. `M̃ + I_bnd`.
   - Move App. G complexity into the main text.
   - Also time the old AMPSS preprocessing on HGB; this answers R3 and justifies P5.
8. **Reproducibility:** release configs, split and fold indices, seeds, sampler seeds, and a one-command script per table.

---

## 7. Claims and writing

**Decision A/B, made after H5 and H6:**
- **A:** keep "heterogeneous heterophilic", only if H5 holds or H2GB shows strong heterophily effects, **and** H6 is positive.
- **B:** retitle around path-supported cellular sheaves for heterogeneous graphs, with heterophily as one tested setting. B is the default if H6 fails.

**Contributions (target):**
- **C1:** path-supported 2-cells with a correct account of how 2-cell information reaches predictions (Lemma A, Lemma B, P2 if justified).
- **C2:** a scalable instantiation (P3, P4, P5) with measured cost.
- **C3:** a controlled evaluation (H1–H6).
- **SheafCovPool and graph classification:** move to the appendix, or to a separate HINPool-style paper claiming heterogeneity only.

**Text fixes:**
- Rewrite the abstract to claim only measured results. No "strongest overall" without tests.
- The introduction funnel should open with the IMDB intermediate-node study as Fig. 1 (only if H4 or the motivation study holds).
- Add a notation table; define the CW 2-cell (simple cycle = path + closure, orientation, signs); give tensor shapes.
- Redraw Fig. 3 and Fig. 4 with main-text notation, and show the `F_{v⊴e}ᵀ` transport.
- Add a figure of the actual information channels (§3).
- Add a limitations section: mini-batch approximation, sampler variance, the scope of heterophily tested, and the closure-edge shortcut finding.
- Fix the template: "Conference acronym ’XX", "14 pages", DOI placeholder, ACM reference format.
- Related work, organized by mechanism:
  - sheaves: Hansen & Ghrist 2019; Hansen & Gebhart 2020; Bodnar et al. 2022; Barbero et al. 2022; Duta et al. 2023 (sheaf hypergraphs); HetSheaf; DNSD;
  - cellular/topological: Bodnar et al. 2021 (CWN); Hodge Laplacians (Lim 2020; Schaub et al. 2020);
  - heterophilic heterogeneous: H2GB, Hetero²Net, H2SGNN, LatGRL;
  - meta-path search: HAN, GTN, DiffMG, LMSPS, SeHGNN;
  - graph-level: HINPool;
  - scaling: GraphSAGE, Cluster-GCN, GraphSAINT, Bajaj et al.

---

## 8. Phased plan with gates

| Phase | Work | Output | Gate to proceed |
|---|---|---|---|
| 0 Audit (cheap) | H5, H7, H8, H9, H10; time AMPSS; check graph-cls baseline split sizes (k/20 on MUTAG, though 10% should give 19). | Audit report | — |
| 1 Diagnose v1 | H1, H2, H3, H4 on HGB, 10 seeds | Mechanism report | Decides P2 (needed?) and the claim scope |
| 2 Correctness | P1 (normalized Laplacian, equivariant norm) | v1.1 | HGB within 1 std of v1; tests pass |
| 3 Scale | P3, P4, P5 (+P6 if H4 weak) | v2 | Acceptance criteria in §5; full vs. mini-batch within 1 std on ACM/DBLP |
| 4 Main experiments | HGB + H2GB, all baselines rerun, ablations A–K, H6, scalability | Tables and figures from scripts | Decide A/B (§7) |
| 5 Write | §7 | Manuscript | Claim-to-evidence audit: every sentence in the abstract, intro and conclusion maps to a table, figure or lemma |

**Ablation set (equal parameters ±5%, 10 seeds, paired tests vs. F):**
- A: endpoint-collapsed (path → direct edge)
- B: no face term
- C: `η₁=0`
- D: uniform gate
- E: random paths at equal budget
- F: full
- G: `γ=0`
- H: `w_cl=0`
- I: map family
- J: readout type
- K: HetSheaf reimplementation
- L: one shared encoder for all node types (G5)
- M: no relation embedding in the map and edge-initializer inputs (G5)

---

## 9. Open questions for the authors (human decisions)

1. Is the target venue one that values theory (NeurIPS/ICLR) or scale and data (KDD/WSDM)? This shifts weight between Lemmas A/B and the H2GB experiments.
2. Graph-level work: appendix, or a separate paper?
3. What compute is available? H2GB at full protocol, with 10 seeds and a full baseline sweep, is the dominant cost.
4. If H1 shows v1's face term is inert, do we invest in P2, or narrow to an "edge-consistency regularizer" story?

---

## 10. Verification queue (must be closed before writing results)

1. Table 1–3 cells vs. published values and our logs (H8).
2. IMDB MLH under fair multilabel definitions (H5).
3. MUTAG test-set size in the baseline path; the k/20 pattern.
4. Whether paper results predate the `budget.py` fix.
5. `require_target_start/end` in every reported run.
6. Actual `d` per reported run (memory arithmetic assumes 32).
7. `η·λ_max` per run (H9); gauge test (H10).
8. Functoriality and `δ₁δ₀=0` under P3 maps.
9. H2GB and LMSPS numbers re-read from the original tables before citing.

---

## 11. References

**Read in this review (VERIFIED-SOURCE):**
- J. Lin et al., *When Heterophily Meets Heterogeneity: Challenges and a New Large-Scale Graph Benchmark* (H2GB), KDD 2025, arXiv:2407.10916.
- L. Braithwaite et al., *Heterogeneous Sheaf Neural Networks*, arXiv:2409.08036.
- C. Li et al., *Long-range Meta-path Search on Large-scale Heterogeneous Graphs* (LMSPS), NeurIPS 2024.
- M.-Y. Hong et al., *HINPool*, AAAI-26.
- R. Bourgerie, Š. Girdzijauskas, V. Fodor, *Deep Neural Sheaf Diffusion*, ICML 2026 GFM Workshop.
- S. Bajaj et al., *GNN Training Systems: A Performance Comparison of Full-Graph and Mini-Batch*, arXiv:2406.00552.
- Y. Ding et al., *DiffMG*, KDD 2021.

**Standard references (verify bibliographic details before adding to the .bib, per `CLAUDE.md`):**
- Bodnar et al., *Neural Sheaf Diffusion*, NeurIPS 2022.
- Hansen & Ghrist, *Toward a Spectral Theory of Cellular Sheaves*, JACT 2019.
- Hansen & Gebhart, *Sheaf Neural Networks*, NeurIPS 2020 workshop.
- Barbero et al., *Sheaf Neural Networks with Connection Laplacians*, ICML 2022 workshop.
- Duta et al., *Sheaf Hypergraph Networks*, NeurIPS 2023.
- Bodnar et al., *Weisfeiler and Lehman Go Cellular: CW Networks*, NeurIPS 2021.
- Lim, *Hodge Laplacians on Graphs*, SIAM Review 2020.
- Schaub et al., *Random Walks on Simplicial Complexes and the Normalized Hodge 1-Laplacian*, SIAM Review 2020.
- Platonov et al., *Characterizing Graph Datasets for Node Classification*, NeurIPS 2023.
- Lim et al., *Large Scale Learning on Non-Homophilous Graphs*, NeurIPS 2021.
- Lv et al., HGB / Simple-HGN, KDD 2021.
- Yang et al., SeHGNN, AAAI 2023.
- Errica et al., ICLR 2020.
- Demšar, JMLR 2006.
- Bouthillier et al., MLSys 2021.
- Li et al., HeaRT, NeurIPS 2023.
- Zeng et al., GraphSAINT, ICLR 2020.
- Chiang et al., Cluster-GCN, KDD 2019.
- Hamilton et al., GraphSAGE, NeurIPS 2017.
- Yun et al., GTN, NeurIPS 2019.
- Wang et al., HAN, WWW 2019.
- LeVeque, *Finite Difference Methods for ODEs and PDEs*, SIAM 2007.

---

## 12. Coverage additions (gaps found by a line-by-line re-check of the reviews)

A line-by-line audit of `critical_reviewer_reviews.md` against §1–§11 found the requests below missing or only partly covered. They are binding parts of the plan.

| Gap ID | Review source | What was missing | Required action |
|---|---|---|---|
| G1 | R1 Q2 | Per-baseline **splits, seeds and early stopping** (not just search space and budget). | Appendix table, one row per baseline × task: split protocol, seeds, epochs, patience, monitored metric, search space, number of trials, published-vs-tuned flag. Source of truth: `budget.py`, `search_grids.py`, `published.py`. |
| G2 | R2 C1 (lines 114–116, 134–136) | A concrete example or preliminary analysis where the endpoint view fails under heterophily. | **H11.** IMDB M–A–M and M–D–M: measure the label agreement of (m₁, m₂) stratified by the intermediate node's genre entropy, with bootstrap CIs. Add a toy example (2–3 node types, hand-built) where two paths share endpoints but should send opposite signals. Prediction: agreement depends on the intermediate node. Kill rule: if it does not, rewrite the motivation; do not keep the claim. |
| G3 | R2 W5, R4 S5 | SheafCovPool ablation is on **one dataset** (ENZYMES) with gains inside std. | If SheafCovPool stays anywhere in the paper: ablate it on all graph datasets with 10 folds × 5 seeds and paired tests, and fix its gauge claim (Fact D). Otherwise remove the claim entirely. |
| G4 | R2 W5, R3 Q2/Q3, R4 W7 + AMPSS-protocol comment | AMPSS **selection results and implementation details**, counts and budgets. We drop AMPSS, so the paper must still report them for the *old* method (to justify the replacement) and the analogous quantities for the *new* sampler. | (a) Appendix: old AMPSS candidate count, selected schemas per dataset, `K_S`/`K_inst`/`K_inc`/`ℓ_max` values (F11), and preprocessing time. (b) Answer R4 directly: selection used **training labels only**, split into a fit and a held-out selection part (`make_selector_split(labels=train_labels, holdout_ratio)`, VERIFIED-CODE). The selector seed is separate from the training seeds, so selection is **not repeated per training seed** (VERIFIED-CODE per `CLAUDE.md`; confirm per dataset). (c) For the new sampler, report `ℓ_max`, `r`, paths, closure edges and incidences per batch, and schema-importance tables per dataset. |
| G5 | R4 W5 | The ablation must also separate **type-specific encoders** and **relation-aware contextualization** from the 2-cells. | Add variants: **L** = one shared encoder for all types; **M** = no relation embedding in the restriction-map and edge-initializer inputs. Same parameter budget, 10 seeds. |
| G6 | R4 W6 + heterophily comment | Whether selected or weighted paths **change with heterophily**, plus **homophilic controls of the same graphs**. | **H12.** Compare per-schema mean gate weights between low-MLH (DBLP, ACM) and high-MLH (IMDB, mag-year) data, and along the rewiring sweep in H6 (Spearman ρ between gate mass on each schema and the measured heterophily). Include the rewired-toward-homophily control (p < 0) in H6. |
| G7 | R3 W2, R4 W1/W2 + TU comment | A heterophily statistic for graph-classification datasets, and a justification for using molecular TU data. | Either define and report a per-graph heterophily statistic (stating clearly what plays the role of the node "label") or state that TU graphs test heterogeneity only and remove heterophily claims for graph tasks (as HINPool does). Do the same for link prediction: report endpoint homophily where node labels exist; otherwise say it cannot be measured. |
| G8 | R2 S4 vs. R4 W3 | R2 **did** see our complexity analysis (App. G) and runtime table; R4 says none exists. | Do not write that "reviewers missed it". Move the App. G complexity statement and Table 11 into the main text, and add the end-to-end measurements (§6.7). |
| G9 | R3 S2 | R3 **praised Stage II**: two instances of the same schema need not be equally useful. | Keep instance-level weighting in the new sampler (soft gate, P5) and say explicitly that it continues Stage II. Ablation D tests it. |
| G10 | R4 HetSheaf comment | Options besides re-implementation: compare **only on HetSheaf's native setup**, or label the comparison non-controlled. | Primary: reimplement (§6.3). Additionally report our model under HetSheaf's native protocol (HGB, its sweep size) next to its published numbers. Any remaining quoted cell is marked ‡ "non-controlled" and excluded from bolding and from significance tests. |
| G11 | R4 summary | Gains may come from **preprocessing choices** (e.g., feature construction), not path preservation. | All models get identical input features and feature preprocessing per dataset (state it). Ablation A (endpoint-collapsed) and K (HetSheaf) share our encoder and preprocessing. |
| G12 | R4 end-to-end study | **Parameter count** per dataset; scaling curve **over the number of retained instances**. | Already in §6.7 by name for parameters. Make the instance curve explicit: sweep `r` and plot time and memory vs. incidences per batch. |
| G13 | R1 summary, R2 W1 | MUTAG is the only dataset where we are not best (fact from R1). R2 flags "some baselines show low performance". | Keep MUTAG honestly in the table (if graph tasks stay). The below-chance and anomalous baselines are covered in §6.2. |
| G14 | R4 significance comment | Significance vs. the **closest executable baselines** specifically. | Named comparators: H2SGNN, PSHGCN, SeHGNN, Simple-HGN, and HetSheaf (reimplemented) on HGB; H2G-former, LMSPS and SeHGNN on H2GB. |

---

## Appendix A — Review traceability matrix (every point in the reviews)

Status column: **Covered** = §-reference addresses it fully · **Added** = covered by §12.
"Requires decision" means the response depends on a hypothesis outcome.

### Reviewer 1 (Weak Accept)

| ID | Point (abridged from the review) | Where addressed | Status |
|---|---|---|---|
| R1-S1 | 2-cell construction retains intermediate nodes and relation sequence | Keep; P3 order task protects it | Covered |
| R1-S2 | Construction well defined; proof of consistency | Keep; re-prove under P3 (§5 P3) | Covered |
| R1-S3 | 3 tasks, 11 datasets | §7 scope decision | Covered |
| R1-S4 | AMPSS budgets control operator size | Replaced by bounded per-batch sampling (P5); G4 | Covered + Added |
| R1-S5 | Detailed ablations of AMPSS and SheafCovPool | §8 ablations A–M; G3 | Covered + Added |
| R1-W1 | Scalability not shown; ACM 3.011 s/epoch, 9,260 MB vs GCN 0.109 s | H7, P3, P4, §6.7 | Covered |
| R1-W2 | Baseline search space and budget unclear; rerun under same protocol? | §6.2, §6.3, H8, G1 | Covered + Added |
| R1-W3 | HetSheaf not under same protocol; ± 0 | §6.3, G10 | Covered + Added |
| R1-Q1 | Runtime and memory vs graph size and #instances/incidences | §6.7, G12 | Covered + Added |
| R1-Q2 | Search space, budget, splits, seeds, early stopping per baseline | G1 | Added |
| R1-Q3 | How std combines 10 folds and 5 seeds | §6.1, §6.4 | Covered |

### Reviewer 2 (Weak Reject)

| ID | Point | Where addressed | Status |
|---|---|---|---|
| R2-Sum | Heterophily and path-preservation motivations misaligned with mechanism and evaluation | §3 (Lemmas A/B, Fact C), H1–H6, §7 | Covered; requires decision |
| R2-S1 | Transport-built edge-to-face maps principled; concise proof | Keep; re-prove under P3 | Covered |
| R2-S2 | Lifting instances to 2-cells is the right direction | Keep | Covered |
| R2-S3 | 3 tasks, 4 baseline categories, code released | §6.8 reproducibility | Covered |
| R2-S4 | Complexity analysis and measured efficiency in appendix | G8 | Added |
| R2-W1 | Graph-cls values look like a single split; some baselines low | F12, §6.1, §6.2, G13 | Covered |
| R2-W2 | Baseline tuning undescribed; quoted HetSheaf mixed with own runs | §6.2, §6.3, H8, G1, G10 | Covered + Added |
| R2-W3 | Face term acts on edges only; reaches nodes only via closure edge | Lemma A, Lemma B, Fact C, H1, H3, H4, P2, P6 | Covered; requires decision |
| R2-W4 | Datasets mostly homophilic or effectively homogeneous; no heterophily-degree analysis | H5, H6, §6.6, G6, G7 | Covered + Added |
| R2-W5 | AMPSS and SheafCovPool gains within std on one dataset; AMPSS selections and details missing | P5, G3, G4 | Covered + Added |
| R2-C1 | Example or preliminary analysis of endpoint-view failure (l. 114–116, 134–136) | G2 (H11) | Added |
| R2-C2 | Intro must link each contribution (AMPSS, SheafCovPool, graph cls) to a problem | §7 contributions C1–C3 | Covered |
| R2-C3 | Fig. 3 low readability; symbols differ from text | §7 | Covered |
| R2-C4 | Fig. 4 caption (edge states moved to node stalks) not shown; notation inconsistent | §7 | Covered |
| R2-C5 | Ablation: closure edges only vs. with face term | Ablation B; H1 | Covered |

### Reviewer 3 (Weak Accept)

| ID | Point | Where addressed | Status |
|---|---|---|---|
| R3-Sum | Idea interesting; "more than a straightforward combination"; evaluation and scalability are the concerns | §1, §5, §6 | Covered |
| R3-S1 | Original, well-motivated modeling idea; relation sequence matters | P3 order task, H4, H11 | Covered |
| R3-S2 | AMPSS is task-dependent; Stage II weights instances within a schema | G9 | Added |
| R3-S3 | 3 substantially different tasks | §7 scope | Covered |
| R3-W1 | MLH 0.1277 / 0.2639 / 0.7634: only IMDB heterophilic; use the large benchmarks already cited | H5, §6.6 (H2GB) | Covered |
| R3-W2 | TU setting artificial; no heterophily statistic for TU | G7, §7 option B | Added |
| R3-W3 | Scalability: DFS enumeration, closure edge per instance; no runtime, peak memory, preprocessing time or instance counts | F10, H7, §6.7, G4 | Covered + Added |
| R3-Q1 | Add genuinely heterophilic heterogeneous benchmarks | §6.6 H2GB | Covered |
| R3-Q2 | Per-dataset candidate/selected schemas, instances, closure edges, incidences, preprocessing time, training time, peak memory | §6.7, G4 | Covered + Added |
| R3-Q3 | Sensitivity of `K_S`, `K_inst`, `K_inc` and max path length | G4 (old values reported); new knobs `ℓ_max` and `r` swept (P5) | Covered + Added |

### Reviewer 4 (Weak Reject)

| ID | Point | Where addressed | Status |
|---|---|---|---|
| R4-Sum-a | Setting not consistently established | H5, H6, G7, §7 | Covered + Added |
| R4-Sum-b | Scalability of a 2-cell per instance not shown | §6.7, P3–P5 | Covered |
| R4-Sum-c | Quoted unavailable-code baseline | §6.3, G10 | Covered + Added |
| R4-Sum-d | Path preservation vs. capacity or preprocessing choices | Ablations A–M, equal parameters, G5, G11 | Covered + Added |
| R4-S1..S6 | Limitation addressed is important; 2-cells principled; multiple tasks; AMPSS ablation useful; SheafCovPool some evidence; substantial technical detail | Keep; G3, G9 | Covered |
| R4-W1 | Heterophily only for node cls; none for graph and link tasks | §6.6, G7 | Covered + Added |
| R4-W2 | TU types reinterpreted; single-node-type multi-relation link datasets; narrow the claim | §6.5, §7, G7 | Covered + Added |
| R4-W3 | No end-to-end complexity analysis, runtime table, memory profile, schema/instance counts or large-graph experiment | §6.7, G8, G12 | Covered + Added |
| R4-W4 | HetSheaf numbers not comparable | §6.3, G10 | Covered + Added |
| R4-W5 | 2-cell not isolated from encoders, contextualization, learned maps, AMPSS, weighting, diffusion, readout | Ablations A–K + L, M (G5) | Covered + Added |
| R4-W6 | No evidence that selected paths help under heterophily | H6, G6 (H12) | Covered + Added |
| R4-W7 | Under-specified: schema space, max length, enumeration constraints, instance counts, budgets, selection split, search budget | G1, G4 | Added |
| R4-W8 | Template issues: placeholder venue, generic ACM format, "14 pages" vs 10 | §7 | Covered |
| R4-C1 | Heterophily statistics for all datasets under the views H2S-NN uses | §6.6, G7 | Covered + Added |
| R4-C2 | Justify molecular TU data or add natural heterogeneous graph benchmarks or limit claims | G7, §7 option B | Added |
| R4-C3 | End-to-end scalability: counts, wall-clock, peak memory, parameter count; curves over size, path length, #instances | §6.7, G12 | Covered + Added |
| R4-C4 | Controlled ablation: endpoint-collapsed / no 2-cell diffusion / uniform weights / same #paths / full | Ablations A, B/C, D, E, F | Covered |
| R4-C5 | Treat HetSheaf cautiously: reproduce, native setup, or label non-controlled | §6.3, G10 | Covered + Added |
| R4-C6 | Direct heterophily evidence: stratify, homophilic variants, schemas selected at low vs high heterophily | H6, G6 | Covered + Added |
| R4-C7 | AMPSS protocol: which labels; held-out selection disjoint from test; repeated per split; `K` values | G4(b) | Added |
| R4-C8 | Stds, significance tests, tuning details for all main comparisons; closest executable baselines | §6.4, G1, G14 | Covered + Added |
| R4-C9 | Fix formatting and page count | §7 | Covered |

### Meta-review

| ID | Point | Where addressed | Status |
|---|---|---|---|
| M-1 | Heterophilic setting not established: only IMDB; TU labels as types; gains not related to heterophily degree | H5, H6, §6.6, G7 | Covered + Added |
| M-2 | 2-cell contribution not isolated: face term reaches nodes only via closure edges; ablations do not separate selection, weighting, capacity | §3, H1–H4, ablations A–M, P2 | Covered; requires decision |
| M-3 | No analysis of cost growth with graph size and #instances | §6.7, H7, P3–P5 | Covered |
| M-4 | Protocol questioned: baseline tuning, quoted HetSheaf | §6.2, §6.3, G1, G10 | Covered + Added |
| M-ctx | Reviewers split; both high-confidence reviewers weak reject | Informs priority: R2 and R4 items are P0 | Covered |

**Audit result:** every enumerated point now maps to a section. Fourteen items (G1–G14) were missing or partial in the first version of this spec and were added in §12.
