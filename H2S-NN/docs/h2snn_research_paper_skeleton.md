# H2S-NN Research Paper Skeleton

> **Historical outline (superseded).** This file preserves an earlier paper plan and obsolete
> AMPSS output names. The maintained manuscript is `paper/h2snn_manuscript.tex`; the authoritative
> implementation description is `docs/h2snn_current_model_documentation.html`. Current AMPSS uses
> task-gain-only Stage I and task-trained schema-conditioned instance weights, with no auxiliary
> discovery loss.

Outline-only skeleton (headers + bullets, not prose) for a unified paper covering the base
H2S-NN model and the schema-to-cell auto-discovery extension. Expand each bullet into prose
directly in this file, section by section; once populated, this Markdown file converts losslessly
to LaTeX (`pandoc h2snn_research_paper_skeleton.md -o paper.tex`) and to slides
(`pandoc ... -t beamer` or Marp), since `$...$` math survives the conversion untouched.

**Scope locked in for this draft:** unified paper (base manual-schema model + auto-discovery
extension as one contribution); results tables reused as-is from
`docs/h2snn_preliminary_results_tables_updated.html`; no venue-specific formatting yet.

---

## 0. Front Matter

### 0.1 Working titles
Verbatim candidates already drafted in `docs/h2snn_metapath_research_memo_2026-05-06.md` §6:
1. *Learning Higher-Order Heterogeneous Sheaf Complexes without Predefined Meta-Paths*
2. *Schema-to-Cell Search for Hodge Neural Networks on Heterogeneous Graphs*
3. *Automatic 2-Cell Discovery for Heterogeneous Sheaf Neural Networks*

Additional direction (not verbatim titles, synthesized from the PHS-Sheaf discussion in
`docs/meta_path_survey.md` §16): a title foregrounding the *topology-aware, training-free*
scoring mechanism (§16.4's holonomy/endpoint-discrepancy score) as the headline novelty, e.g.
something in the family of "Topology-Aware Schema-to-Cell Discovery for Heterogeneous Sheaf-Hodge
Networks" — needs independent wordsmithing, not lifted from a source doc.

### 0.2 Notation table (stub — populate from `docs/H2SNN_corrected_algorithm.md`)
- $N$, $M$ — number of nodes, number of (real, non-closure) edges
- $d$, $d_r$, $d_h$ — stalk dimension, relation-embedding dimension, LSTM hidden dimension
- $\phi(v)$ — node-type map; $R$ — relation set; $z^{rel}_r$ — relation embedding
- $F_s$, $\ell$ — number of face instances in schema $s$, path length (#edges) of a schema
- $H^{(0)}$ / $h_0$ — initial node encoding; $F^{src}_e$, $F^{tgt}_e$ — node-to-edge restriction maps
- $s^{(0)}$ — initial edge cochain; $F_{e\triangleright f}$ — edge-to-face restriction map
- $\delta_0$, $\delta_1$, $L_0$, $L_1$ — coboundary operators and Hodge Laplacians
- $K$, $\eta_0$, $\eta_1$ — number of Euler diffusion steps, node/edge step sizes
- $\alpha_\sigma$, $\beta_f$ — schema-level and instance-level discovery gates
- $x^\*$, $s^\*$ — post-diffusion node/edge states
- Full symbol-by-symbol definitions: see `docs/H2SNN_corrected_algorithm.md` (authoritative math
  reference) and `docs/H2S-NN_implementation_spec.md` (earlier draft spec, superseded where the
  two disagree).

---

## 1. Abstract

Bullet template (context → gap → method → result → contribution); draft this section **last**,
once Results is finalized, since it needs final headline numbers.

- **Context bullet:** heterogeneous information networks with typed nodes/edges, often strongly
  heterophilic; ordinary GNN message passing lives on the 1-skeleton and either over-smooths or
  ignores higher-order typed relational patterns (source: `docs/h2snn_step_by_step_explanation_2026-05-06.md` §3).
- **Gap bullet:** existing heterogeneous sheaf models (HetSheaf) learn typed geometry only on the
  fixed 1-skeleton; existing meta-path search methods (LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM, ...)
  select schemas but only ever use them as aggregation channels, discarding the path structure
  after routing neighbors.
- **Method/contribution sentence (verbatim, most polished version)** —
  `docs/h2snn_metapath_research_memo_2026-05-06.md` §11:
  > "Existing heterogeneous sheaf models learn geometry on fixed relational structure, while
  > existing meta-path search methods choose semantic channels for HGNN aggregation. We unify
  > these directions by learning which typed higher-order path schemas should instantiate the
  > 2-cells of a heterogeneous sheaf complex, allowing end-to-end topology construction for Hodge
  > neural diffusion without predefined meta-paths."
- **Discovery-extension one-sentence claim (verbatim)** —
  `docs/h2snn_schema_to_cell_discovery_code_spec_2026-05-19.md` (Table 5 note):
  > "H2S-NN is the first method that automatically discovers face-generating schemas using a
  > training-free topology-aware score, and integrates the selected schemas as weighted 2-cells in
  > a heterogeneous sheaf complex, allowing the schema selection to directly reshape Hodge
  > diffusion geometry rather than merely adding aggregation channels."
- **Result bullet (placeholder):** headline numbers across node classification (ACM/DBLP/IMDB),
  graph classification (MUTAG/PROTEINS/ENZYMES/NCI1), link prediction (LastFM/AmazonBook/...), and
  node clustering — pull final figures from §6 Results once that section is populated.
- **Caution while drafting:** an A* reviewer will check the abstract's claims against Sections 4/6
  word for word — do not state anything here that Discussion (§7) later concedes is unvalidated
  (see the "what not to claim yet" callout in §7.3).

---

## 2. Introduction

### 2.1 Motivation (heterophily + heterogeneity)
Verbatim framing, `docs/h2snn_step_by_step_explanation_2026-05-06.md` §3:
> "We start from a heterogeneous information network: multiple node types; multiple relation
> types; often strong heterophily, meaning connected nodes may not share labels or semantics.
> This creates two problems for ordinary GNNs: simple neighborhood aggregation can over-smooth
> dissimilar nodes; ordinary message passing mainly lives on the graph 1-skeleton, so it does not
> explicitly model higher-order typed relational patterns."

### 2.2 The core reframing
- "Meta-path as semantic view" (existing HGNNs/search methods) → "meta-path instance as
  topological cell" (H2S-NN) — `docs/h2snn_step_by_step_explanation_2026-05-06.md` §4/§17.
- One-line mechanism: a concrete instance `a1 -> p7 -> a2` becomes a face; faces participate in
  $\delta_1$; $\delta_1$ contributes to $L_1$; $L_1$ reshapes Hodge diffusion on edge cochains.

### 2.3 Contribution list (map directly to the boolean-CSV rows where H2S-NN is uniquely `TRUE`
  against ~20 named baselines, `docs/comparison_metapath_models_boolean_attributes_2026-05-20.csv`)
1. Turns selected meta-path **instances** (not just schemas) into 2-cells that reshape the
   diffusion operator itself (`selected_higher_order_structure_modifies_diffusion_operator`,
   `uses_2cells_or_faces`, `uses_hodge_edge_laplacian_L1` — TRUE only for H2S-NN out of ~20 models).
2. Hodge diffusion over **both** node and edge cochains via matrix-free $\delta_0, \delta_1, L_0,
   L_1$ operators — the survey's "Level 4" framing (§3.1 below).
3. Hierarchical, jointly-learned schema-level ($\alpha$) and instance-level ($\beta$) gating that
   directly weights the up-Laplacian term, not merely a schema-level scalar
   (`instance_level_path_filtering_or_weighting` — TRUE only for H2S-NN + HISE-KT + HeteroSample,
   and only H2S-NN ties it to the diffusion operator).
4. Training-free, topology-aware candidate scoring (endpoint-label-discrepancy from
   training-label propagation) for schema preselection, avoiding a full model-retrain per
   candidate schema.

### 2.4 Honest self-correction (include explicitly — don't let a reviewer find it first)
An earlier draft claimed novelty from "works without predefined meta-paths," but this was
verified false: LMSPS, AMP-HGNAS, MCTS-HGNN, and PMMM already support this. Verbatim record of the
correction, `docs/h2snn_comparison_explanation.md` Row 10:
> "This row was removed after verifying the actual papers. The original claim that search
> methods... require a predefined schema pool or domain expert knowledge was **incorrect**...
> The real gap is in **what the selected schemas become** (aggregation channels vs. 2-cells) and
> **how they affect the model** (weighted sum vs. reshaping L₁)."

### 2.5 Scope-framing risk (relevant precisely because this is a *unified* paper)
Flag explicitly in the intro's closing paragraph, since the docs already warn about this exact
risk — `docs/h2snn_step_by_step_explanation_with_review_2026-05-06.md` (reviewer-risk table):
> "The base H2S-NN and the schema-discovery extension are being mixed too closely. If the paper
> claims both at once without fully validating both, reviewers may feel the contribution is too
> broad, incomplete, or inflated." — mitigation used in this draft: state the base-model claim and
> the discovery-extension claim as two distinct, separately-evidenced contributions within the
> same paper, not one blended claim.

---

## 3. Related Work

### 3.1 Positioning the gap mathematically
Level hierarchy from `docs/meta_path_survey.md` §12.3:
```
Level 0: Node features               x_v ∈ ℝ^d
Level 1: Edge relations              (u →^r v) with type r
Level 2: Meta-path templates         σ = (T₀, r₁, T₁, ..., rₗ, Tₗ) — TYPE SCHEMA
Level 3: Meta-path instances         π = specific node sequence following σ
Level 4: 2-cells (H2S-NN)            Face f = instance π closed into topological cell
```
> "Papers 1-11 all operate at Level 2-3 (schema selection and instance counting). **H2S-NN
> uniquely operates at Level 4**... This is what makes the comparison with HETSHEAF unfair to
> HETSHEAF — HETSHEAF operates at Level 1 only." (§12.3)

Sheaf-literature cross-table, `docs/meta_path_survey.md` §15.3:
> "H2S-NN is the only paper in all 19 that uses L₁ and δ₁. PMMM operates at the schema level...
> but has no face topology, no coboundary operator, and no Hodge decomposition."

### 3.2 Heterogeneous GNNs (no meta-path search)
HAN, MAGNN (expert-defined meta-paths as semantic message-passing views), HGT (typed
node/edge attention, no hand-designed meta-paths). Gap: neither turns a concrete path instance
into an operator-shaping higher-order cell (Research Position table, §3.5 below).

### 3.3 Meta-path / schema search methods
Full named set (years/venues in `docs/comparison_model_year_metadata_2026-05-20.csv`): LMSPS
(NeurIPS 2024), AMP-HGNAS (2026), MCTS-HGNN, PMMM (2022), Meta-Multigraph Search (2024),
HG-Search, SchemaWalk, RMS-HRec, SMCD, HeteroSample (2024), EvoPath (2025), TAEG-HG (2025), FHGE
(2025), HISE-KT (2025), ESim, MP-GNN, LPSMS.
- Gap (repeated verbatim across docs): "Every existing search method... uses selected meta-paths
  as **aggregation channels**... the higher-order relational pattern is *thrown away* after
  serving as a routing rule." (`docs/h2snn_comparison_explanation.md`, Row 3)
- Full boolean capability matrix (20 models × 26 attributes): reuse
  `docs/comparison_metapath_models_boolean_attributes_2026-05-20.csv` directly as a table/figure.

### 3.4 Sheaf neural networks
- **HetSheaf** (arXiv:2409.08036) — explicit, repeated "closest baseline" designation; learns
  typed sheaf restriction maps and handles heterophily via local spaces, but only on the fixed
  1-skeleton (no 2-cells, no $\delta_1$/$L_1$).
- Neural Sheaf Diffusion (Bodnar et al. 2022, arXiv:2202.04579) — core sheaf-diffusion /
  heterophily / oversmoothing motivation (homogeneous, not heterogeneous).
- DeepSN (arXiv:2412.xxxxx), SHN (arXiv:2405.xxxxx) — sheaf applications without 2-cells/$L_1$
  (see cross-table `docs/meta_path_survey.md` §15.3).

### 3.5 Heterophily-aware heterogeneous models
HeterSEED (arXiv:2605.04594), AHGNN (arXiv:2508.06034), H2SGNN (arXiv:2410.13373), LatGRL
(arXiv:2409.00687), Cross-Type Homophily Ratio/CTHGE (arXiv:2501.14600). Gap: address
heterophily structurally/spectrally, not through weighted typed path instances as Hodge faces.

### 3.6 Research Position table (reuse verbatim from `docs/h2snn_current_model_documentation.html`)

| Research Line | What It Solves | Gap H2S-NN Targets |
|---|---|---|
| HAN / MAGNN style meta-path HGNNs | Use expert-defined meta-paths as semantic message-passing views. | They do not turn each concrete path instance into an operator-shaping higher-order cell. |
| HGT style transformer HGNNs | Use typed node and edge attention without hand-designed meta-paths. | They remain edge/message-passing models rather than Hodge models over faces. |
| Meta-path search methods such as LMSPS and PMMM | Search useful semantic paths or meta-multigraphs. | Selected schemas usually become aggregation channels, not cells in a learned Hodge operator. |
| HetSheaf | Learns heterogeneous sheaf maps and handles heterophily through local spaces. | Closest baseline, but H2S-NN aims to add path-induced higher-order structure and δ₁. |
| HeterSEED / AHGNN / H2SGNN | Recent heterophily-aware heterogeneous models with strong graph-level mechanisms. | They address heterophily, but not through weighted typed path instances as Hodge faces. |

External reference anchors (verbatim list, same source): HetSheaf (2409.08036), HeterSEED
(2605.04594), AHGNN (2508.06034), H2SGNN (2410.13373), LMSPS (2307.08430), PMMM (2211.14752).

---

## 4. Methodology

Each subsection = the operative formula(s) + implementing file, so drafting prose later is a
direct lookup rather than re-derivation. $N$=#nodes, $M$=#edges, $d$=stalk dim, $d_r$=relation-emb
dim, $d_h$=LSTM hidden dim, $\ell$=schema path length, $F_s$=#face instances in schema $s$.

### 4.1 Typed node encoding — `src/models/node_encoder.py`
$$h^{(0)}_v = \text{GELU}\big(\text{LayerNorm}(\text{Linear}_{\phi(v)}(x_v)) + \text{TypeEmbed}[\phi(v)]\big) \in \mathbb{R}^d$$
- One independent encoder MLP per node type + one shared additive type embedding.
- Feature-less graphs (LastFM/AmazonBook): falls back to a free per-node
  `nn.Embedding(num_nodes, d)` table instead of `Linear(0, d)` (which would collapse every node of
  a type to its bias vector).

### 4.2 Node-to-edge sheaf restriction maps — `src/models/sheaf_maps.py`, `src/models/orthogonal.py`
- Per-relation 2-layer GELU MLPs produce $F^{src}_e, F^{tgt}_e \in \mathbb{R}^{d\times d}$ from
  $[h^{(0)}_u \Vert h^{(0)}_v \Vert z^{rel}_r]$.
- Default parameterization is **scaled orthogonal**, not strictly orthogonal — state this
  precisely (see §4.10): $F = \text{Orthogonal}(\text{params}) \cdot \exp(\text{clamp}(\text{raw}, \max=2.0))$.
  Current-model `orthogonal_map ∈ {cayley, householder (default), euler}`.
  `matrix_exp` is disabled because its full-edge CUDA backward path was unsafe at ACM scale.
- $(\delta_0 h^{(0)})_e = F^{tgt}_e h^{(0)}_{tgt(e)} - F^{src}_e h^{(0)}_{src(e)}$, computed once and
  reused as an MLP feature for §4.3 (never as the diffusion initialization itself).
- Ablation hook: `ablate_identity_maps` forces $F^{src}=F^{tgt}=I_d$, degenerating to a plain
  node-difference operator.

### 4.3 Edge cochain initialization — `EdgeInitializer`
$$s^{(0)}_e = \text{MLP}_{init}\big([h^{(0)}_u \Vert h^{(0)}_v \Vert z^{rel}_r \Vert (\delta_0 h^{(0)})_e]\big) \in \mathbb{R}^d$$
- Deliberate design choice (`H2SNN_corrected_algorithm.md` §7.1): $s^{(0)} \neq \delta_0 H^{(0)}$
  directly — that would zero out the harmonic component of the initial edge signal.

### 4.4 Edge-to-face maps via LSTM + attention — `SequenceEdgeToFaceSheaf`
- Per face, tokens $= \text{MLP}_g([h^{(0)}_{src} \Vert h^{(0)}_{tgt} \Vert z^{rel}])$ fed through a
  single-layer LSTM; attention $\alpha_i = \text{softmax}(h_i \cdot a / \sqrt{d_h})$ over path
  position.
- **Methodology-precision flag:** the implemented map is
  $$F_{e_i\triangleright f} = \alpha_i \cdot \Psi(h_i)$$
  i.e. attention *does* multiply the map — this contradicts the older spec text
  (`H2SNN_corrected_algorithm.md` §6.4 step B.4), which states attention is used only in the
  regularizer, not the map construction. Write methodology prose to match the code.
- Maps are cached once per forward pass (`FaceMapCache`) and reused across all $K$ diffusion
  steps — $L_0$/$L_1$ are fixed linear operators for the whole Euler rollout.

### 4.5 Closure edges — closed-form $L_0$ correction, `CoboundaryOps.closure_L0`
- Not literal additional 1-cells with their own cochain state; instead a closed-form correction:
  $\text{diff} = \text{closure\_scale}^2 \cdot (x_{tgt} - x_{src})$, added into $L_0$.
- Default `closure_scale = 1e-6` ⇒ effective coupling $10^{-12}$ (functionally negligible but
  present). Closure edges never touch $L_1$/$\delta_1$ (by design — contribute exactly zero).
- Resolved bug history worth citing in a "design rationale" paragraph: closure edges were
  generated but unused until Update Log v0.2 (`docs/h2snn_current_model_documentation.html`).

### 4.6 Coboundary operators — `src/models/coboundary_ops.py` (matrix-free, verbatim from doc)
```
delta0(x)_e = F_tgt(e) x_tgt - F_src(e) x_src         # M active edges only
L0(x)       = delta0^T delta0 x + closure_L0(x)        # closure_L0 scales as closure_scale^2

delta1(s)_f = sum over boundary edges e of eta(e,f) * F_{e->f} s_e   # eta in {+1,-1}; always +1 today
L1(s)       = delta0 delta0^T s + delta1^T delta1 s     # closure edges excluded (contribute ~0)

Weighted discovery version:
L1_up = sum_sigma alpha_sigma sum_f beta_f (delta1_f)^T delta1_f
```
- Orientation sign $\eta(e,f)$ is always $+1$ in practice (meta-path DFS only walks forward), but
  the signed-incidence machinery is general and explicitly tested with a non-trivial $-1$ case
  (`tests/test_models.py::test_delta1_respects_face_edge_sign`).

### 4.7 Hodge diffusion — `HodgeDiffusion`, $K$-step explicit Euler
$$x \leftarrow x - \eta_0 L_0(x), \qquad s \leftarrow s - \eta_1 L_1(s), \quad k=1,\dots,K$$
- $\eta_0, \eta_1$ learnable via softplus reparameterization (default `learnable_eta=True`).
- Post-diffusion: `LayerNorm` on both $x^\*$ and $s^\*$ before readout (not part of the original
  diffusion pseudocode — a practical addition, state it as such).

### 4.8 Readout heads (unified paper needs all three — base model covers node classification)
- **Node classification** (`NodeReadout`): $z_v = [x^\*_v \Vert h^{(0)}_v \Vert
  \text{pool\_edges\_to\_nodes}(x^\*, s^\*)_v]$, MLP → logits.
- **Link prediction** (`LinkPredictionReadout`): swappable `dot | distmult | concat_mlp` decoder.
- **Graph classification** (`GraphPooling`): swappable `MeanPool | GramPool (O(d)-invariant) |
  S2Pool (GL(d)-invariant)` — see `S2Pool_proposal.md` and
  `docs/gauge_free_coboundary_readout_design.md` for the gauge-invariance argument, itself a
  secondary contribution thread worth a short methodology subsection.

### 4.9 Loss function — `src/trainers/trainer.py`, `src/trainers/regularizers.py`, `src/models/discovery_losses.py`
$$\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{smooth0} + \mathcal{L}_{smooth1} + \mathcal{L}_{consist} + \mathcal{L}_{coboundary} + \mathcal{L}_{discovery}$$

| Term | Formula | Default $\lambda$ (`src/configs/config.py: ModelConfig`) |
|---|---|---|
| Classification | `CrossEntropyLoss(logits, y)` on train mask | — |
| Smooth0 | $\lambda_0 \cdot \text{mean}(\lVert\delta_0 x^\*\rVert^2)$ | $\lambda_0=0.0$ |
| Smooth1 | $\lambda_1 \cdot \text{mean}(\lVert\delta_1 s^\*\rVert^2)$ | $\lambda_1=10^{-3}$ |
| Consistency | $\lambda_c \cdot \text{mean}(\lVert s^\* - \delta_0 x^\*\rVert^2)$ | $\lambda_c=10^{-2}$ |
| Coboundary | $\lambda_{co} \cdot \text{mean}(\lVert\delta_1 \delta_0 H^{(0)}\rVert^2)$, evaluated at $H^{(0)}$ not $x^\*$ | $\lambda_{co}=10^{-2}$ |

Discovery losses (only active once `epoch ≥ warmup_epochs`), six terms —
`sparse_schema`, `sparse_instance`, `redundancy`, `coverage`, `cost`, `gate_entropy` — default
$\lambda$'s from `SchemaDiscoveryConfig`: $1\text{e-}3, 1\text{e-}4, 1\text{e-}3, 1\text{e-}3,
1\text{e-}4, 1\text{e-}4$ respectively.
- **Precision note:** `sparse_instance` must read the grad-attached `face_weight_means_grad`
  field, not the detached `face_weight_means` (logging-only) — this was a resolved P1 bug (Update
  Log v0.2); state the fix, not the bug, in methodology prose.

### 4.10 Schema discovery pipeline — `src/data/schema_discovery.py`, `src/models/schema_gates.py`
Five stages (name them explicitly in the methodology write-up):
- **A. Schema graph** — type-level adjacency from relation tuples.
- **B. Candidate generation** — DFS typed-walk enumeration, `min/max_schema_length` (2/4 default),
  `require_target_start/end=True`, capped at `max_schema_candidates` (128 default).
- **C. Topology-aware stats** — training-label-propagation-based **endpoint label discrepancy**
  score (no model forward pass needed; no test-label leakage by construction — training labels
  only) + coverage/diversity/branching stats.
- **D. Greedy pool selection** — $\text{score}(\sigma) = \text{disc}(\sigma)^{\alpha_1}\cdot
  \text{cov}(\sigma)^{\alpha_2}\cdot\text{div}(\sigma)^{\alpha_3} / (1+\text{redundancy})$; picks
  `selected_schema_k` (8 default) schemas.
- **E. Face materialization** — concrete path-instance sampling, closure-edge pairing, per-schema
  8-dim structural feature vector (consumed by discovery losses).
- **Hierarchical gates:** $\alpha_\sigma = \text{sigmoid}(\text{logit}_\sigma)$ warm-started from
  $\log(\text{topology score})$; $\beta_f = \text{sigmoid}(\text{MLP}([h^{(0)}_{v_0} \Vert
  h^{(0)}_{v_\ell} \Vert \text{schema\_embedding}]))$; effective per-face weight $=
  \alpha_\sigma \cdot \beta_f$.
- **Partial Schema Propagation** (stabilization, borrowed from PMMM's partial message
  propagation): Bernoulli-dropout on $\alpha$ during training, unbiased rescaling
  ($\alpha \leftarrow \alpha \cdot \text{keep}/p$), `partial_schema_probability=0.5` default.

### 4.11 Known implementation notes for methodology precision (write prose to match code, not older specs)
- Attention **is** folded into edge-to-face maps (§4.4) — contradicts `H2SNN_corrected_algorithm.md` §6.4.
- Maps are **scaled orthogonal**, not strictly orthogonal (§4.2) — don't overstate this in prose.
- Closure edges have no relation embedding of their own (handled entirely outside the
  relation-typed machinery via §4.5's closed-form correction).
- `instance_features` (6-dim per-face structural vector) is computed in
  `_compute_instance_features` but **not read anywhere** by the model or loss code — don't
  describe it as an active `InstanceGate` input.
- `hard_concrete` is a listed config option (`schema_gate_type`/`instance_gate_type`) but
  unimplemented — both gate classes raise `ValueError` if selected. Don't claim this ablation was run.

---

## 5. Experiments

### 5.1 Datasets

| Task | Datasets | Config source |
|---|---|---|
| Node classification | ACM (target=paper, PAP/PSP), DBLP (target=author, APA/APTPA), IMDB (target=movie, MAM/MDM) | `src/configs/config.py: NODE_CLASSIFICATION_PRESETS`, `configs/experiments/{acm,dblp,imdb}_h2snn.yaml` |
| Link prediction | LastFM (`user→to→artist`), AmazonBook (`user→rates→book`) | `LINK_PREDICTION_PRESETS`, `configs/experiments/{lastfm,amazonbook}_h2snn.yaml` |
| Graph classification | MUTAG (stratified 10-fold), PROTEINS/ENZYMES/NCI1 (precomputed 10-fold, reused from the legacy CSNN baseline's `folds/`) | `GRAPH_CLASSIFICATION_PRESETS`, `configs/experiments/{mutag,proteins,enzymes,nci1}_h2snn.yaml` |
| Node clustering | ACM, Freebase, AMiner, IMDB (per Results Table 3) | **implementation gap — see §5.4** |

**Verified loader identity (2026-07-06):** ACM/DBLP/IMDB load via `torch_geometric.datasets.
HGBDataset` (official HGB preprocessing — see HGB's Table 2 stats reproduced in
`h2snn_research_paper_draft.md` §4.1.1); MUTAG/PROTEINS/ENZYMES/NCI1 via `TUDataset` (standard,
unambiguous); LastFM/AmazonBook via PyG's `LastFM`/`AmazonBook` classes, which are **not confirmed
to be the same graphs** as HGB's own same-named link-prediction datasets that our baseline
comparison rows were plausibly evaluated on — flagged as the single highest-priority open item in
the experimental section, full detail in the draft's §4.1.1/§4.4. PubMed/YouTube link prediction
have zero loader implementation at all (not just a missing config preset).

### 5.2 Baselines per task
Reuse the five comparison tables/CSVs already drafted rather than re-deriving:
- Node classification: GCN/GAT (homogeneous), NSD/CSNN/ChebNetII, HAN/SeHGNN/HiNormer/PSHGCN
  (homophilic heterogeneous), HDHGR/Hetero2Net/H2SGNN/HetSheaf/LatGRL (heterophilic heterogeneous)
  — `docs/h2snn_preliminary_results_tables_updated.html` Table 1.
- Graph classification: same homogeneous/heterogeneous baseline split — Table 2.
- Node clustering: HGMAE/RMR/DMGI/HDMI/HeCo/HeCo++/HERO/CTC/HGMS-N/HGMS-C — Table 3.
- Link prediction: GCN/GAT/GIN/SAGE (homogeneous), R-GCN/HAN/HGT (heterogeneous), Sheaf-NSD/HetSheaf
  (sheaf) — Table 4.
- Efficiency: R-GCN/HAN/HGT/Sheaf-NSD — Table 5.
- Meta-path/search-method baselines for the discovery-extension claim specifically: LMSPS,
  AMP-HGNAS, MCTS-HGNN, PMMM (full list + boolean capability matrix in
  `docs/comparison_metapath_models_boolean_attributes_2026-05-20.csv`).

### 5.3 Ablations
Map directly to flags that already exist in code (`src/configs/config.py: ModelConfig`):
- `ablate_no_faces` — strips all faces + closure + discovery entirely.
- `ablate_no_delta1` — keeps face_groups object but empties `face_caches`, so $L_1$'s up-term
  vanishes.
- `ablate_identity_maps` — $F^{src}=F^{tgt}=I_d$ (plain node-difference operator).
- `ablate_uniform_gates` — forces $\alpha=\beta=1$ everywhere (bypasses both gate mechanisms).
- Manual vs. auto-discovery (AMPSS): configs exist for ACM/DBLP/IMDB
  (`{acm,dblp,imdb}_h2snn_ampss.yaml`) but **no link-prediction or graph-classification AMPSS
  variant is configured yet** — needed before the discovery-extension claim can be evaluated
  across all three task families.
- Minimum ablation/baseline checklist already drafted verbatim in
  `docs/h2snn_current_model_documentation.html` §"Validation Plan" — reuse directly.

### 5.4 Known gaps to close before this section is submission-ready
- **No multi-seed / statistical-significance protocol documented anywhere.**
  `docs/h2snn_test_setup_guide.html` is a local Windows/Conda smoke-test runbook (env setup, unit
  tests, one training command), not a formal evaluation protocol — no mention of repeated seeds or
  significance testing. Design this before finalizing §6.
- **Node clustering has no implementation.** No clustering head/task exists anywhere in `src/`
  (`NodeReadout`, `LinkPredictionReadout`, `GraphPooling` are the only three readout heads). Either
  implement a clustering evaluation path or drop/reframe Results Table 3.
- **Hyperparameter search space vs. checked-in configs discrepancy:**
  `docs/h2snn_hyperparameter_search_space.md` marks `d=64` (bold = "value used in final configs"),
  but every actual config under `configs/experiments/*.yaml` uses `d=32` (verified directly in
  `acm_h2snn.yaml`, `acm_h2snn_ampss.yaml`). Reconcile which is authoritative before reporting
  hyperparameters in the paper.
- Optuna+Hyperband sweep described in the search-space doc (`scripts/tune_h2snn.py`,
  `val_macro_f1` objective, Hyperband pruner) has no persisted results
  (`experiments/tune_results.json` / `optuna_*.db` do not exist in-repo) — note as either "sweep
  pending" or find/attach the external results if they exist elsewhere.

---

## 6. Results

Reuse the following five tables verbatim from `docs/h2snn_preliminary_results_tables_updated.html`.

### 6.1 Node classification (avg. Macro/Micro-F1)

| Type | Model | DBLP Macro | DBLP Micro | ACM Macro | ACM Micro | IMDB Macro | IMDB Micro |
|---|---|---|---|---|---|---|---|
| Homophilic homogeneous | GCN (2016) | 90.77 | 90.28 | 91.63 | 91.38 | 62.33 | 67.69 |
| Homophilic homogeneous | GAT (2021) | 94.39 | 92.52 | 92.07 | 90.71 | 62.13 | 68.09 |
| Homophilic homogeneous | NSD (2022) | 92.96 | 93.42 | 92.33 | 92.05 | 61.61 | 65.23 |
| Homophilic homogeneous | CSNN (2026) | 93.73 | 92.12 | 93.13 | 92.81 | 62.92 | 62.59 |
| Homophilic homogeneous | ChebNetII (2022) | 92.89 | 92.67 | 91.70 | 91.59 | 62.97 | 68.47 |
| Homophilic heterogeneous | HAN (2019) | 91.78 | 92.24 | 90.88 | 91.61 | 61.95 | 67.76 |
| Homophilic heterogeneous | SeHGNN (2022) | 95.52 | 95.48 | 94.38 | 94.04 | 72.01 | 72.59 |
| Homophilic heterogeneous | HiNormer (2023) | 94.06 | 94.14 | 91.87 | 89.64 | 64.13 | 67.53 |
| Homophilic heterogeneous | PSHGCN (2024) | 95.58 | 95.70 | 93.66 | 94.70 | 71.82 | 74.34 |
| Heterophilic heterogeneous | HDHGR (2023) | 95.35 | 95.00 | 93.99 | 94.13 | 59.41 | 59.63 |
| Heterophilic heterogeneous | Hetero2Net (2023) | 93.54 | 93.54 | 94.05 | 92.87 | 65.34 | 70.63 |
| Heterophilic heterogeneous | H2SGNN (2023) | 95.82 | 95.11 | 94.90 | 94.08 | 74.23 | 75.29 |
| Heterophilic heterogeneous | HETSHEAF (2024)* | 93.87 | 93.77 | 95.09 | 94.94 | 64.65 | 67.18 |
| Heterophilic heterogeneous | LatGRL (2021) | 94.32 | 94.51 | 93.47 | 92.29 | 58.27 | 58.81 |
| — | **H2SNN (Ours)** | **96.01** | **96.31** | **95.91** | **95.72** | **75.05** | **76.68** |

*(Column headers in the source table also carry an unexplained bracketed number — "DBLP (59.76)",
"ACM (72.39)", "IMDB (50.04)" — likely a homophily-ratio or similar dataset statistic; confirm and
either caption it properly or drop it before publication.)*

### 6.2 Graph classification (avg. Macro/Micro-F1)

| Type | Model | MUTAG Macro | MUTAG Micro | PROTEINS Macro | PROTEINS Micro | ENZYME Macro | ENZYME Micro | NCI1 Macro | NCI1 Micro |
|---|---|---|---|---|---|---|---|---|---|
| Homophilic homogeneous | GCN (2016) | 89.58 | 90.00 | 75.92 | 79.83 | 12.75 | 28.33 | 64.17 | 64.96 |
| Homophilic homogeneous | GAT (2021) | 89.58 | 90.00 | 75.27 | 78.38 | 14.03 | 16.55 | 64.17 | 64.96 |
| Homophilic homogeneous | NSD (2022) | 100 | 100 | 73.60 | 75.89 | 33.65 | 35.00 | 72.30 | 72.30 |
| Homophilic homogeneous | CSNN (2026) | 100 | 100 | 76.64 | 77.92 | 35.15 | 37.22 | 74.11 | 74.30 |
| Homophilic heterogeneous | HAN (2019) | 82.91 | 85.00 | 79.19 | 82.22 | 46.24 | 45.02 | 77.60 | 77.50 |
| Homophilic heterogeneous | PSHGCN (2024) | 94.88 | 95.00 | 70.61 | 72.32 | 20.18 | 28.33 | 64.17 | 64.96 |
| Heterophilic heterogeneous | HDHGR (2023) | 84.65 | 85.00 | 77.64 | 81.25 | 48.27 | 48.33 | 79.29 | 79.32 |
| Heterophilic heterogeneous | Hetero2Net (2023) | 89.58 | 90.00 | 74.66 | 78.57 | 8.53 | 18.33 | 63.67 | 65.45 |
| Heterophilic heterogeneous | H2SGNN (2023) | 94.88 | 95.00 | 70.61 | 72.32 | 20.18 | 28.33 | 64.17 | 64.96 |
| Heterophilic heterogeneous | HETSHEAF (2024)* | 100 | 100 | 80.56 | 82.14 | 62.35 | 63.33 | 79.46 | 79.46 |
| Heterophilic heterogeneous | LatGRL (2021) | 90.88 | 91.01 | 71.61 | 72.32 | 25.41 | 29.03 | 77.98 | 78.10 |
| — | **H2SNN (Ours)** | **100** | **100** | **84.44** | **86.23** | **64.52** | **65.10** | **81.22** | **81.52** |

### 6.3 Node clustering (NMI / ARI)

| Model | ACM NMI | ACM ARI | Freebase NMI | Freebase ARI | AMiner NMI | AMiner ARI | IMDB NMI | IMDB ARI |
|---|---|---|---|---|---|---|---|---|
| HGMAE | 64.57 | 67.85 | 17.18 | 17.94 | 34.70 | 34.38 | 2.76 | 2.62 |
| RMR | 54.37 | 45.65 | 18.33 | 18.15 | 30.22 | 31.59 | 2.59 | 2.47 |
| DMGI | 56.49 | 51.06 | 16.41 | 16.13 | 19.24 | 20.09 | 2.15 | 1.51 |
| HDMI | 57.91 | 54.75 | 16.53 | 17.47 | 17.56 | 15.04 | 2.07 | 1.79 |
| HeCo | 56.87 | 56.94 | 16.14 | 16.90 | 32.26 | 28.64 | 2.72 | 2.44 |
| HeCo++ | 60.82 | 60.09 | 16.67 | 17.48 | 38.07 | 36.44 | 2.92 | 2.58 |
| HERO | 58.41 | 57.89 | 15.34 | 15.56 | 30.11 | 31.48 | 2.19 | 2.05 |
| CTC | 63.26 | 69.30 | 18.70 | 21.31 | 35.21 | 34.63 | 2.86 | 2.55 |
| HGMS-N | 72.44 | 75.53 | 21.35 | 22.49 | 42.87 | 43.67 | 6.66 | 5.25 |
| HGMS-C | 71.97 | 74.18 | 22.28 | 23.14 | 46.56 | 50.28 | 6.61 | 6.77 |
| **H2S-NN** | **80.69** | **79.50** | **24.01** | **24.81** | **52.32** | **56.20** | **7.59** | **7.17** |

*(Needs §5.4's implementation gap closed first — no clustering code currently exists in `src/`.)*

### 6.4 Heterogeneous link prediction (AUPR / AUROC)

*(Updated 2026-07-06 — the MovieLens column is removed: H2S-NN was never evaluated on MovieLens, so
reporting baseline-only numbers there would misleadingly imply a completed comparison. LastFM,
PubMed, and Amazon(Book) are confirmed real H2S-NN datasets/runs; YouTube is reported as real per
the same confirmation, though no `youtube` preset currently exists in `src/configs/config.py:
LINK_PREDICTION_PRESETS` — see Section 5.4's implementation-gap note.)*

| Group | Model | LastFM AUPR | LastFM AUROC | PubMed AUPR | PubMed AUROC | Amazon AUPR | Amazon AUROC | YouTube AUPR | YouTube AUROC |
|---|---|---|---|---|---|---|---|---|---|
| Homogeneous | GCN | 97.06 | 96.96 | 96.05 | 92.93 | 92.40 | 89.66 | 89.30 | 83.26 |
| Homogeneous | GAT | 63.06 | 51.14 | 96.07 | 93.27 | 88.62 | 86.10 | 66.76 | 58.40 |
| Homogeneous | GIN | 97.99 | 97.75 | 96.14 | 92.87 | 92.10 | 89.62 | 88.64 | 82.97 |
| Homogeneous | SAGE | 95.11 | 92.72 | 94.50 | 89.79 | 90.79 | 87.95 | 84.54 | 80.20 |
| Heterogeneous | R-GCN | 97.89 | 97.93 | 95.03 | 93.15 | 93.37 | 91.42 | 91.68 | 88.33 |
| Heterogeneous | HAN | 88.26 | 83.32 | 95.78 | 93.62 | 95.24 | 93.64 | 93.46 | 90.10 |
| Heterogeneous | HGT | 86.02 | 74.55 | 97.58 | 96.48 | 96.15 | 95.66 | 94.95 | 92.51 |
| Sheaf | Sheaf-NSD | 97.16 | 96.58 | 96.41 | 93.65 | 92.63 | 90.04 | 89.32 | 83.36 |
| Sheaf | HetSheaf (ours) | 98.51 | 98.33 | 96.36 | 93.98 | 93.76 | 91.71 | 91.27 | 87.33 |
| — | **H2S-NN** | **100** | **100** | **99.63** | **99.28** | **99.71** | **98.34** | **98.37** | **96.30** |

LastFM still sits at a 100/100 ceiling and merits a footnote on evaluation protocol (dataset size,
split, near-ceiling baseline scores already at 88-98%) before submission — a reviewer will ask.

### 6.5 Efficiency comparison (ACM)

| Model | Runtime (s) | # Params | Memory (MB) |
|---|---|---|---|
| R-GCN | 22.1 | 43M | 1,240 |
| HAN | 28.9 | 2.2M | 1,780 |
| HGT | 1025.8 | 7.2M | 6,120 |
| Sheaf-NSD | 47.6 | 155K | 980 |
| **H2S-NN** | **18.39** | **134K** | **810** |

### 6.6 Ablation results table (stub — TODO, no ablation runs exist yet)

| Ablation | Node cls. (ACM/DBLP/IMDB) | Graph cls. (MUTAG/PROTEINS/ENZYMES/NCI1) | Link pred. (LastFM/AmazonBook) |
|---|---|---|---|
| Full model | *(cross-ref §6.1/6.2/6.4)* | | |
| `ablate_no_faces` | TODO | TODO | TODO |
| `ablate_no_delta1` | TODO | TODO | TODO |
| `ablate_identity_maps` | TODO | TODO | TODO |
| `ablate_uniform_gates` | TODO | TODO | TODO |
| Manual vs. AMPSS auto-discovery | TODO (configs exist for ACM/DBLP/IMDB) | not configured yet | not configured yet |

### 6.7 Reproducibility note
Per this project's own "reproducibility first" standard (`CLAUDE.md`), the config + checkpoint
backing the headline numbers above should be saved under `experiments/` before submission — the
only currently-saved ACM run (`experiments/acm_h2snn_notebook/`) predates the rerun that produced
these numbers and should be replaced/supplemented with the up-to-date artifact.

---

## 7. Discussion

### 7.1 Reviewer-facing framing (reuse verbatim, `docs/h2snn_current_model_documentation.html`)
**Best Reviewer-Facing Framing:**
> "Given typed schemas or a candidate schema pool, H2S-NN instantiates path instances as
> higher-order structures and performs learned sheaf-Hodge diffusion over the resulting
> heterogeneous node-edge-face incidence system."

**What It Is Not:**
- "It is not just HAN with extra meta-path attention."
- "It is not just HGT with typed relation attention."
- "It is not only a meta-path search method."
- "It should not yet be described as a fully validated automatic topology learner."

### 7.2 Limitations (reuse table verbatim, same source, "Limitations And Bugs")

| Priority | Issue | Why It Matters |
|---|---|---|
| Resolved | Closure edges were generated but not used anywhere in the model. | The strict 2-cell/cellular-complex claim depends on closed boundaries. |
| Resolved | Orientation signs were implicitly hardcoded as +1 inside δ₁, with no explicit sign field. | Hodge/cellular coboundary operators require signed incidence for mathematical correctness. |
| Resolved | Instance sparsity loss was detached from gradients. | The sparsity term did not train β until fixed. |
| P1 | Schema preselection is non-differentiable; refinement is differentiable after the pool is fixed. | Deliberate two-stage AMPSS design — frame as candidate-pool construction + end-to-end refinement, not a flaw. |
| Resolved | `closure_scale` existed in config but was not used by the model. | Made the closure-edge design incomplete/confusing until wired in. |
| P2 | Maps described as orthogonal are actually scaled orthogonal. | Don't overstate strict orthogonality in the paper. |
| P2 | Tests were not verifying the implementation in some environments (missing torch). | Re-run full suite in a proper environment before submission. |
| P2 | Evaluation evidence breadth. | Add HetSheaf, HeterSEED, AHGNN, H2SGNN, LMSPS, PMMM baselines where feasible (partially done — see §6). |

**"Do not claim yet" callout (verbatim):**
> "'H2S-NN is a fully validated automatic schema-to-cell discovery model.' The closure/orientation
> mathematical fixes are complete and dense-reference tested... what remains open is breadth of
> experiments — baselines and evaluation beyond the ACM manual-schema results."

### 7.3 Reviewer-anticipated weaknesses (from `..._with_review_2026-05-06.md` §19.3 — address each explicitly)
- **Closure-edge justification:** does the construction truly satisfy boundary-of-boundary-zero,
  or is it a "generalized path-induced incidence structure" rather than a strict cellular complex?
- **Computational cost:** every edge-face incidence carries a full $d\times d$ map — memory/compute
  scaling is a real scalability concern (`large_scale_focus` is honestly `FALSE` for H2S-NN today
  per the comparison CSV).
- **Over-engineering risk:** many interacting components (sheaf maps, LSTM face encoder, gates,
  five loss terms) — without strong ablations (§5.3/§6.6), gains could come from extra parameters
  rather than the higher-order topology itself.
- **Label leakage risk:** schema/instance scoring must never touch validation or test labels —
  state the protocol explicitly (training-label-only propagation is already the implementation;
  say so in the paper).
- **PSD guarantee:** if $L_1^{up} = \sum_\sigma \alpha_\sigma \sum_f \beta_f (\delta_1^f)^T
  \delta_1^f$, nonnegative $\alpha,\beta$ (already enforced via sigmoid gates) preserve positive
  semidefiniteness — state this explicitly as a proposition, not just an implementation detail.

### 7.4 Unified-paper scope risk (carry the mitigation from §2.5 through to here)
Reiterate that base-model and discovery-extension claims are evidenced separately (§6.1/6.2/6.4
for the base model; §6.6 ablation table + AMPSS configs for discovery) rather than blended into
one performance number, directly addressing the reviewer-risk table's warning about claiming both
without validating both.

---

## 8. Conclusion

### 8.1 Summary (two-part contribution, stated separately per §7.4)
- Base contribution: heterogeneous sheaf-Hodge diffusion where selected typed path instances
  become 2-cells shaping $\delta_1$/$L_1$, not just aggregation channels.
- Extension contribution: training-free, topology-aware schema-to-cell discovery with hierarchical
  $\alpha$/$\beta$ gating, removing dependence on manually predefined meta-paths.

### 8.2 Future work
- **Formal search guarantees** — the PHS-Sheaf direction (`docs/meta_path_survey.md` §16: bounded
  expansion via a policy-guided search combining a learned policy with a geometry-native
  "holonomy" heuristic from the restriction maps) is aspirational, not implemented.
- **`hard_concrete` gate variant** — config option exists (`schema_gate_type`/`instance_gate_type`)
  but raises `ValueError` if selected; a real ablation axis once implemented.
- **Scale beyond ACM/DBLP/IMDB/MUTAG-class benchmarks** — `large_scale_focus` is honestly `FALSE`
  for H2S-NN today; needed to answer the "can this scale?" reviewer question from §7.3.
- **Node clustering as a genuinely new task** — no clustering readout exists yet; Results Table 3
  (§6.3) needs this built before its numbers can be defended.
- **Cheaper map parameterizations** (diagonal / low-rank / basis-decomposed) as a direct answer to
  the computational-cost weakness in §7.3.

---

## Appendix: Source Document Index

For traceability when expanding bullets into prose:
- **Math/spec:** `docs/H2SNN_corrected_algorithm.md` (authoritative), `docs/H2S-NN_implementation_spec.md` (earlier draft, superseded where they disagree)
- **Living documentation:** `docs/h2snn_current_model_documentation.html`
- **Positioning/novelty:** `docs/h2snn_metapath_research_memo_2026-05-06.md`, `docs/h2snn_comparison_explanation.md`, `docs/h2snn_rl_mcts_novelty_note_2026-05-06.md`, `docs/meta_path_survey.md`
- **Step-by-step + reviewer self-critique:** `docs/h2snn_step_by_step_explanation_2026-05-06.md`, `docs/h2snn_step_by_step_explanation_with_review_2026-05-06.md`
- **Discovery extension:** `docs/h2snn_schema_to_cell_discovery_code_spec_2026-05-19.md`, `docs/h2snn_selection_explanation.md`, `docs/h2snn_selection_problems_detailed.md`
- **Comparison data:** `docs/comparison_metapath_models_boolean_attributes_2026-05-20.csv`, `docs/comparison_model_year_metadata_2026-05-20.csv`, `docs/comparison_support_search_sheaf_boolean_attributes_2026-05-20.csv`, root `h2snn_method_comparison.csv`, root `h2snn_selection_comparison.csv`
- **Results:** `docs/h2snn_preliminary_results_tables_updated.html`
- **Experiment setup:** `docs/h2snn_test_setup_guide.html`, `docs/h2snn_hyperparameter_search_space.md`, `configs/experiments/*.yaml`
- **Graph-level pooling:** `S2Pool_proposal.md`, `docs/gauge_free_coboundary_readout_design.md`
- **Code:** `src/models/*.py`, `src/data/*.py`, `src/trainers/*.py`, `src/configs/config.py`, `tests/*.py`
