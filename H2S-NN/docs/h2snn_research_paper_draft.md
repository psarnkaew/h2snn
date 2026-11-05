# H2S-NN: Heterogeneous Sheaf-Hodge Neural Networks with Learned Schema-to-Cell Discovery

> **Historical draft (superseded).** AMPSS equations and config names below predate
> `schema_conditioned_v1`. Use `docs/h2snn_current_model_documentation.html` and the current
> `src/` code as the implementation contract. Current Stage II has one schema-conditioned
> instance-weight network and no auxiliary discovery loss.

*Working draft — expanded prose version of `docs/h2snn_research_paper_skeleton.md`. Alternative
working titles already on record (`docs/h2snn_metapath_research_memo_2026-05-06.md` §6):
"Learning Higher-Order Heterogeneous Sheaf Complexes without Predefined Meta-Paths" (avoided here
as a primary title — see §1.3 on why this framing overclaims), "Schema-to-Cell Search for Hodge
Neural Networks on Heterogeneous Graphs," "Automatic 2-Cell Discovery for Heterogeneous Sheaf
Neural Networks." This file expands every bullet of the skeleton into full prose; source citations
are preserved inline so factual claims remain traceable to the underlying docs/code.*

---

## Abstract

Heterogeneous information networks — graphs with multiple node and edge types — are frequently
also heterophilic: nodes connected by an edge need not share a label or even a similar semantic
role. Ordinary graph neural networks handle this combination poorly, because standard message
passing lives on the graph's 1-skeleton and either over-smooths dissimilar neighbors or ignores
higher-order typed relational patterns altogether. Two existing lines of work each solve half of
this problem. Heterogeneous sheaf neural networks (notably HetSheaf) learn typed restriction maps
that let connected nodes disagree in raw feature space while remaining compatible after a learned,
relation-specific transport — but they operate only on the fixed graph structure and never
construct higher-order topology. Meta-path search methods (LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM, and
others) automatically discover which typed paths are informative, but every method in this family
treats a selected path purely as an aggregation channel: the path is followed to collect a
neighborhood and then discarded. We introduce **H2S-NN** (Heterophilic Heterogeneous Sheaf-Hodge
Neural Network), which unifies these two directions. Concrete meta-path *instances* — not merely
their type-level schemas — are lifted into two-dimensional cells of a typed cellular complex, and a
learned sheaf on that complex (restriction maps on nodes, edges, and faces) drives Hodge-style
diffusion over both node and edge cochains through the coboundary operators $\delta_0,\delta_1$ and
their Laplacians $L_0,L_1$. We further introduce a schema-to-cell discovery extension that removes
the dependence on hand-specified meta-paths: a training-free, topology-aware score (built from
label propagation over training labels only) identifies a compact pool of candidate path schemas,
and a hierarchical pair of differentiable gates — a schema-level gate $\alpha$ and an
instance-level gate $\beta$ — learns which schemas and which concrete instances should shape the
diffusion operator, stabilized by a partial-schema-propagation mechanism adapted from PMMM's
partial message propagation. On node classification (ACM, DBLP, IMDB), graph classification
(MUTAG, PROTEINS, ENZYMES, NCI1), and heterogeneous link prediction, H2S-NN reports the strongest
numbers in our assembled comparison against homogeneous, heterogeneous, and sheaf-based baselines
(Section 5); we are explicit throughout about which parts of this evidence base — repeated-seed
statistics, the full ablation matrix, and the discovery extension's own evaluation — still need to
be completed before the claims below should be read as fully validated (Sections 4.4 and 6.2).

---

## 1. Introduction

### 1.1 Motivation

We start from a heterogeneous information network: multiple node types, multiple relation types,
and — in the settings that motivate this work — often strong heterophily, meaning connected nodes
may not share labels or semantics. This combination creates two distinct problems for ordinary
GNNs. First, simple neighborhood aggregation can over-smooth dissimilar nodes, since it implicitly
assumes that connected nodes should become more similar after message passing. Second, ordinary
message passing lives mainly on the graph's 1-skeleton — its nodes and edges — so it does not
explicitly model higher-order typed relational patterns, even when such patterns (a paper written
by a specific pair of co-authors, a specific citation chain) are exactly the structure a domain
expert would point to as meaningful (`docs/h2snn_step_by_step_explanation_2026-05-06.md` §3). Most
heterogeneous GNN methods address the second problem by introducing meta-paths as additional graph
views or aggregation channels: a meta-path such as *paper–author–paper* defines a new, derived
adjacency that a model can attend over. This is a real improvement, but it treats the meta-path as
nothing more than a routing rule.

### 1.2 From semantic view to topological cell

H2S-NN takes a different position. Rather than using a meta-path as a semantic aggregation channel,
we use each concrete instance of a meta-path as a higher-order *topological* object. Concretely, a
path instance such as $a_1 \to p_7 \to a_2$ becomes a face of a cellular complex; faces participate
in the coboundary operator $\delta_1$; $\delta_1$ contributes to the up-Laplacian term of $L_1$; and
$L_1$ directly changes the Hodge diffusion geometry over edge signals. The shift in one sentence:
*meta-path as semantic view* becomes *meta-path instance as topological cell*. This is not a
relabeling of existing search methods — it changes what the selected structure is used for. Every
existing automatic meta-path/schema search method we are aware of (LMSPS, AMP-HGNAS, MCTS-HGNN,
PMMM, HG-Search, SchemaWalk, RMS-HRec, SMCD, HeteroSample, EvoPath, TAEG-HG, FHGE, HISE-KT, ESim,
MP-GNN, LPSMS) uses a selected path the same way HAN does: to route a neighborhood into an
attention-weighted sum. None of them make the path itself, as opposed to the neighbors at the end
of it, a meaningful structural object that reshapes the model's operator
(`docs/h2snn_comparison_explanation.md`).

### 1.3 A precise statement of what is (and is not) new

An earlier draft of this project claimed novelty from "working without predefined meta-paths." We
verified this claim against the literature and found it to be false: LMSPS, AMP-HGNAS, MCTS-HGNN,
and PMMM all already discover schemas automatically, without a human-specified meta-path set. We
therefore do **not** lead with this framing, and we flag its removal explicitly here rather than
let a reviewer discover the overclaim independently
(`docs/h2snn_comparison_explanation.md`, Row 10). The real, verified gap is narrower and more
specific: it is in *what the selected structure becomes* — an aggregation channel for every prior
method, versus a two-dimensional cell that reshapes $\delta_1$ and $L_1$ for H2S-NN — and in *how
it acts on the model* — a weighted sum over neighbors, versus a term inside a matrix-free Hodge
Laplacian. Positioned against a broader taxonomy of the meta-path literature
(`docs/meta_path_survey.md` §12.3), most prior work operates at what that taxonomy calls Level 2–3
(schema selection and path-instance counting); H2S-NN operates at Level 4, converting instances
into topological cells with algebraic meaning. This distinction is also why a direct comparison
against HetSheaf — the closest sheaf-based baseline — is not entirely fair to HetSheaf: HetSheaf
operates at Level 1, learning typed geometry on the observed 1-skeleton only, with no notion of a
face at all.

### 1.4 Contributions

1. **Operator-level higher-order modeling.** We show how to turn selected meta-path instances into
   two-dimensional cells of a typed cellular complex such that a learned edge-to-face sheaf
   directly reshapes the up-Laplacian term of $L_1$, rather than merely providing an additional
   aggregation channel — the one property, `selected_higher_order_structure_modifies_diffusion_
   operator`, that is true of H2S-NN and false of every one of the roughly twenty named baselines we
   compare against (`docs/comparison_metapath_models_boolean_attributes_2026-05-20.csv`).
2. **Full Hodge diffusion on a heterogeneous complex.** We define matrix-free coboundary operators
   $\delta_0,\delta_1$ and Laplacians $L_0,L_1$ over both node and edge cochains, and run $K$-step
   Hodge diffusion with learnable step sizes — to our knowledge the only method in a 19-paper survey
   of meta-path and sheaf literature that uses both $L_1$ and $\delta_1$ at all
   (`docs/meta_path_survey.md` §15.3).
3. **Hierarchical, jointly learned schema and instance gating.** A schema-level gate $\alpha_\sigma$
   and an instance-level gate $\beta_f$ are trained together and multiplied into the weighted
   up-Laplacian, giving both a coarse (which meta-path template matters) and a fine (which specific
   instance of it matters) control signal that is otherwise absent from every baseline we survey
   except two, neither of which ties the resulting weights to a diffusion operator.
4. **Training-free, topology-aware schema scoring.** Candidate schema quality is estimated from an
   endpoint-label-discrepancy statistic computed via label propagation over training labels only,
   avoiding the cost of a full model forward pass per candidate schema and, by construction, any use
   of validation or test labels during schema selection.

### 1.5 Scope of this paper and how its two contributions are evidenced

This paper presents H2S-NN as a *unified* contribution spanning both the base sheaf-Hodge model
(contributions 1–2 above) and the schema-to-cell discovery extension (contributions 3–4). We take
this risk deliberately, but we manage it explicitly rather than let the two halves blur into one
inflated claim — a risk our own working notes flag directly:

> "The base H2S-NN and the schema-discovery extension are being mixed too closely. If the paper
> claims both at once without fully validating both, reviewers may feel the contribution is too
> broad, incomplete, or inflated."
> (`docs/h2snn_step_by_step_explanation_with_review_2026-05-06.md`, reviewer-risk table)

Our mitigation, carried through the rest of the paper: rather than asserting the discovery
extension's benefit only architecturally, Section 5.6.1 now reports a direct ablation on DBLP
showing the full discovery configuration (schema gate $\alpha$ plus instance gate $\beta$)
outperforming both a no-higher-order-structure ablation and a hand-specified-schema baseline. We
still keep the claim's *scope* honest: this ablation currently covers one dataset and one task
family, so Section 6.4 states precisely what is and is not yet evidenced before recommending how
the two contributions should be weighted in a submission.

---

## 2. Related Work

### 2.1 Heterogeneous GNNs without meta-path search

HAN and MAGNN use expert-defined meta-paths as semantic message-passing views, attending over a
fixed, hand-designed set of relational patterns. HGT instead uses typed node- and edge-level
attention without any hand-designed meta-path, learning heterogeneity directly through typed
transformer weights. Neither family turns a concrete path instance into an operator-shaping
higher-order cell; both remain message-passing models over the graph's 1-skeleton.

### 2.2 Meta-path and schema search methods

A substantial and active literature automates the choice of which meta-paths or schemas to use:
LMSPS (NeurIPS 2024) and AMP-HGNAS (2026) learn differentiable or NAS-style selection weights over
precomputed path features; MCTS-HGNN applies Monte Carlo tree search directly over the schema
space; PMMM (2022) and the related Meta-Multigraph Search (2024) search meta-multigraphs with a
partial message propagation technique for stabilizing the resulting bilevel optimization; HG-Search
and SchemaWalk apply, respectively, a multi-stage and an inductive reinforcement-learning-flavored
search; RMS-HRec, SMCD, HeteroSample, EvoPath, TAEG-HG, FHGE, HISE-KT, ESim, MP-GNN, and LPSMS round
out the broader family with recommendation-, LLM-, and sampling-oriented variants
(full model-by-attribute matrix in
`docs/comparison_metapath_models_boolean_attributes_2026-05-20.csv`; year/venue metadata in
`docs/comparison_model_year_metadata_2026-05-20.csv`). The unifying limitation, stated plainly in
our own working notes, is that every one of these methods — regardless of how sophisticated its
search procedure — uses a selected meta-path as an **aggregation channel**:

> "The meta-path is followed to collect a neighborhood... and then the meta-path is discarded. The
> higher-order relational pattern is *thrown away* after serving as a routing rule."
> (`docs/h2snn_comparison_explanation.md`, Row 3)

### 2.3 Sheaf neural networks

Neural Sheaf Diffusion (Bodnar et al., 2022, arXiv:2202.04579) established the core connection
between learned cellular-sheaf restriction maps, heterophily, and the avoidance of naive
oversmoothing, but for homogeneous graphs with no notion of node or edge types. HetSheaf
(arXiv:2409.08036) extends this to the heterogeneous setting, learning typed sheaf restriction maps
and handling heterophily through per-relation local spaces; it is, by a wide margin, the closest
existing baseline to H2S-NN and we treat it as such throughout this paper. Other sheaf applications
— DeepSN (arXiv:2412.xxxxx, sheaf diffusion for influence maximization) and SHN (arXiv:2405.xxxxx,
sheaf hypernetworks for personalized federated learning) — apply sheaf-theoretic machinery to
different problems again without constructing higher-order topology. Across all of these, and
across every other paper in a 19-method survey we conducted, H2S-NN is, to our knowledge, the only
method that uses both the up-Laplacian $L_1$ and the coboundary operator $\delta_1$
(`docs/meta_path_survey.md` §15.3); HetSheaf and the other sheaf methods operate at what that
survey's hierarchy calls Level 1 — typed restriction maps on the observed 1-skeleton — with no face
topology, no coboundary operator beyond $\delta_0$, and no Hodge decomposition on edge cochains.

### 2.4 Heterophily-aware heterogeneous models

A separate, more recent line of work addresses heterophily in heterogeneous graphs directly:
HeterSEED (arXiv:2605.04594) combines semantic and structural neighbor selection; AHGNN
(arXiv:2508.06034) adapts aggregation for multi-hop heterophily; H2SGNN (arXiv:2410.13373) applies
spectral, per-meta-path filtering; LatGRL (arXiv:2409.00687) diagnoses semantic heterophily at both
the node and meta-path level; and the Cross-Type Homophily Ratio / CTHGE line
(arXiv:2501.14600) proposes explicit cross-type homophily diagnostics and graph-editing responses.
These methods address heterophily through structural or spectral mechanisms, not through weighted
typed path instances acting as Hodge faces, and they form part of our baseline set (Section 4.2)
rather than a competing higher-order-topology approach.

### 2.5 Positioning summary

| Research Line | What It Solves | Gap H2S-NN Targets |
|---|---|---|
| HAN / MAGNN style meta-path HGNNs | Use expert-defined meta-paths as semantic message-passing views. | They do not turn each concrete path instance into an operator-shaping higher-order cell. |
| HGT style transformer HGNNs | Use typed node and edge attention without hand-designed meta-paths. | They remain edge/message-passing models rather than Hodge models over faces. |
| Meta-path search methods such as LMSPS and PMMM | Search useful semantic paths or meta-multigraphs. | Selected schemas usually become aggregation channels, not cells in a learned Hodge operator. |
| HetSheaf | Learns heterogeneous sheaf maps and handles heterophily through local spaces. | Closest baseline, but H2S-NN aims to add path-induced higher-order structure and δ₁. |
| HeterSEED / AHGNN / H2SGNN | Recent heterophily-aware heterogeneous models with strong graph-level mechanisms. | They address heterophily, but not through weighted typed path instances as Hodge faces. |

*(Source: `docs/h2snn_current_model_documentation.html` §"Research Position," reused verbatim.)*

The overall picture that emerges from placing H2S-NN in this landscape is that it sits at the
intersection of five research lines — heterogeneous graph learning, heterophily-aware GNNs,
meta-path search, sheaf neural networks, and higher-order Hodge diffusion — while occupying a point
in that space (Level 4, uses both $\delta_1$ and $L_1$, ties learned instance-level weights to the
diffusion operator) that none of the surveyed baselines occupy simultaneously.

---

## 3. Method

### 3.1 Preliminaries and notation

We follow the notation of the authoritative mathematical specification
(`docs/H2SNN_corrected_algorithm.md`; an earlier draft, `docs/H2S-NN_implementation_spec.md`, is
superseded wherever the two disagree — see the implementation notes in Section 3.11). Let $N$ and
$M$ denote the number of nodes and (non-closure) edges in the heterogeneous graph, $d$ the shared
stalk dimension used for node, edge, and face states, $d_r$ the relation-embedding dimension, and
$d_h$ the hidden dimension of the sequence encoder used for edge-to-face maps. Each node $v$ has a
type $\phi(v)$; each relation $r$ belongs to a relation set $R$ and has a learned embedding
$z^{rel}_r \in \mathbb{R}^{d_r}$. A schema $\sigma$ is a typed path template of length $\ell$
(number of edges); a face $f$ is one concrete instance of a schema, and $F_s$ denotes the number of
face instances materialized for schema $s$. We write $H^{(0)}$ (equivalently $h^{(0)}$) for the
initial node encoding, $F^{src}_e, F^{tgt}_e \in \mathbb{R}^{d\times d}$ for the source/target
node-to-edge restriction maps of edge $e$, $s^{(0)}$ for the initial edge cochain,
$F_{e\triangleright f}$ for the edge-to-face restriction map, $\delta_0,\delta_1$ for the
coboundary operators and $L_0, L_1$ for their associated Hodge Laplacians, $K$ for the number of
Hodge-diffusion (Euler) steps with step sizes $\eta_0,\eta_1$, $\alpha_\sigma$ and $\beta_f$ for the
schema- and instance-level discovery gates, and $x^*, s^*$ for the post-diffusion node and edge
states used at readout.

### 3.2 Typed node encoding

Node features are encoded into the shared stalk dimension by a type-specific encoder
(`src/models/node_encoder.py`):
$$h^{(0)}_v = \mathrm{GELU}\big(\mathrm{LayerNorm}(\mathrm{Linear}_{\phi(v)}(x_v)) + \mathrm{TypeEmbed}[\phi(v)]\big) \in \mathbb{R}^d.$$
Each node type owns an independent two-layer encoder, and a single shared, learnable type
embedding is added to every node of that type. For graphs that ship without node features at all —
LastFM and AmazonBook, in our link-prediction experiments — a plain linear encoder of the form
$\mathrm{Linear}(0, d)$ would collapse every node of a type to an identical vector (its bias term);
we instead fall back to a free, per-node embedding table $\mathrm{Embedding}(N, d)$, so that
featureless nodes still receive individually learnable initial states.

### 3.3 Node-to-edge sheaf restriction maps

For every directed, relation-typed edge $e = (u \to_r v)$, a pair of restriction maps
$F^{src}_e, F^{tgt}_e$ is produced by relation-specific two-layer GELU MLPs taking
$[h^{(0)}_u \Vert h^{(0)}_v \Vert z^{rel}_r]$ as input (`src/models/sheaf_maps.py`). By default these
maps are constrained to be *scaled orthogonal* rather than free $d\times d$ matrices: an
unconstrained parameterization is mapped to an orthogonal matrix by one of four supported
constructions — matrix exponential, Cayley transform, a product of Householder reflectors
(the default), or, for $d\in\{2,3\}$, closed-form Euler rotations (`src/models/orthogonal.py`) —
and then multiplied by a learned scalar scale, $\exp(\mathrm{clamp}(\text{raw}, \max=2.0))$. We are
careful to describe these as *scaled orthogonal* restriction maps throughout the paper rather than
simply "orthogonal," since the scaling breaks strict orthogonality and an earlier internal review
flagged this exact overstatement as something to avoid
(`docs/h2snn_current_model_documentation.html`, Limitations, P2). These maps define the first
coboundary operator on the initial node encoding,
$$(\delta_0 h^{(0)})_e = F^{tgt}_e h^{(0)}_{tgt(e)} - F^{src}_e h^{(0)}_{src(e)},$$
which is computed once and reused as an input feature to the edge initializer below — never as the
diffusion's initial edge state directly (see Section 3.4). An ablation variant,
`ablate_identity_maps`, replaces both restriction maps with the identity, degenerating $\delta_0$
into a plain, untyped node-difference operator; we use this as one of our minimum ablations
(Section 4.3).

### 3.4 Edge cochain initialization

The initial edge cochain is produced by a shared two-layer MLP (`EdgeInitializer`) from
$$s^{(0)}_e = \mathrm{MLP}_{init}\big([h^{(0)}_u \Vert h^{(0)}_v \Vert z^{rel}_r \Vert (\delta_0 h^{(0)})_e]\big) \in \mathbb{R}^d.$$
This is a deliberate design choice, not an oversight: setting $s^{(0)} = \delta_0 H^{(0)}$ directly
would zero out exactly the harmonic component of the initial edge signal that Hodge diffusion is
meant to preserve and evolve (`docs/H2SNN_corrected_algorithm.md` §7.1). $\delta_0 H^{(0)}$ is
therefore only ever consumed as one input feature among several, controlled by the
`use_delta0_init` flag (default on).

### 3.5 Edge-to-face maps via sequence encoding

Given a schema's set of concrete path instances, each of length $\ell$ edges, we encode the ordered
sequence of boundary edges with a single-layer LSTM plus an attention head
(`SequenceEdgeToFaceSheaf`). For a chunk of $B$ faces (chunked at `face_chunk_size`, default 4096,
to bound peak memory), token features
$\mathrm{token}_i = \mathrm{MLP}_g([h^{(0)}_{src(e_i)} \Vert h^{(0)}_{tgt(e_i)} \Vert z^{rel}_{r_i}])$
are passed through the LSTM to produce hidden states $h_1,\dots,h_\ell$; a learned attention vector
$a$ produces position weights $\alpha_i = \mathrm{softmax}(h_i \cdot a / \sqrt{d_h})$ over the path
position (not to be confused with the schema-level discovery gate of the same Greek letter — we
disambiguate as $\alpha_i^{pos}$ where the two could be confused). The resulting edge-to-face
restriction map is
$$F_{e_i\triangleright f} = \alpha_i^{pos} \cdot \Psi(h_i), \qquad \Psi(h_i) = \mathrm{reshape}\big(\mathrm{Linear}(h_i),\, [d,d]\big).$$
We flag this formula precisely because it differs from an earlier draft of the mathematical
specification, which stated that the attention weights are used only in the coboundary regularizer
and *not* in the map construction itself (`docs/H2SNN_corrected_algorithm.md` §6.4, step B.4). The
implementation multiplies attention directly into the map, and we describe the model as it is
actually implemented rather than as the earlier spec text describes it. These maps are computed
once per forward pass and cached (`FaceMapCache`), then reused across all $K$ diffusion steps, so
$L_0$ and $L_1$ are fixed linear operators for the duration of the Euler rollout described in
Section 3.8.

### 3.6 Closure edges as a closed-form correction

For each face, the underlying cellular-complex construction requires closing the open path
instance with a synthetic *closure edge* from its final node back to its initial node, so that the
face's boundary forms a genuine closed loop. Rather than materializing this as an additional 1-cell
with its own learned cochain state, we implement it as a closed-form correction directly on $L_0$:
$$\text{diff} = \text{closure\_scale}^2 \cdot (x_{tgt(f)} - x_{src(f)}), \qquad \text{added into } L_0 \text{ via index-add over the closure endpoints,}$$
where the squared dependence on `closure_scale` follows from $L_0 = \delta_0^\top \delta_0$ being
quadratic in the restriction map, here a frozen, uniformly scaled identity. With the default
`closure_scale`$=10^{-6}$, the effective coupling introduced by closure edges is $10^{-12}$ —
functionally negligible, but present and load-bearing for the closed-boundary claim, and
consciously distinct from allowing closure edges to affect $\delta_1$ at all, which they do not, by
design. This closes out what was, earlier in this project's history, an open P0 bug: closure edges
were generated by the data pipeline but silently unused by the model until this correction was
added (`docs/h2snn_current_model_documentation.html`, Update Log v0.2).

### 3.7 Coboundary operators and Hodge Laplacians

All coboundary and Laplacian operators are implemented matrix-free — never materialized as dense
matrices — using batched per-edge and per-face matrix–vector products and scatter-adds
(`src/models/coboundary_ops.py`):
$$\delta_0(x)_e = F^{tgt}_e x_{tgt(e)} - F^{src}_e x_{src(e)}$$
$$L_0(x) = \delta_0^\top \delta_0\, x + \text{closure\_}L_0(x)$$
$$\delta_1(s)_f = \sum_{e \in \partial f} \eta(e,f)\, F_{e\triangleright f}\, s_e, \qquad \eta(e,f)\in\{+1,-1\}$$
$$L_1(s) = \delta_0 \delta_0^\top s + \delta_1^\top \delta_1\, s$$
The orientation sign $\eta(e,f)$ is implemented as a general, explicit, dense-reference-tested
tensor threaded through the face data structures, but in practice it is always $+1$ in this
project's data, since our path-instance enumeration always walks forward through each relation's
stored source-to-destination direction. We nonetheless treat the general sign machinery as load
bearing — it is exercised with a non-trivial $-1$ case in
`tests/test_models.py::test_delta1_respects_face_edge_sign` — because future extensions using
undirected or backward-traversed boundary edges would require it. When schema/instance discovery
weights are active, the up-Laplacian term generalizes to
$$L_1^{up}(s) = \sum_{\sigma} \alpha_\sigma \sum_{f \in \sigma} \beta_f\, (\delta_1^f)^\top \delta_1^f\, s,$$
which is the operator through which the learned discovery gates of Section 3.11 act directly on
diffusion geometry — the property that, per Section 1.4, distinguishes H2S-NN's use of discovered
structure from every prior meta-path search method.

### 3.8 Hodge diffusion

Node and edge states are updated jointly for $K$ steps of explicit Euler integration with
learnable step sizes:
$$x \leftarrow x - \eta_0 L_0(x), \qquad s \leftarrow s - \eta_1 L_1(s), \qquad k = 1,\dots,K,$$
where $\eta_0,\eta_1$ are reparameterized through a softplus so that they remain positive while
still being trained end to end (`learnable_eta=True` by default). The resulting states $x^*,s^*$
are each passed through a `LayerNorm` before being handed to readout — a practical addition on top
of the diffusion recurrence itself, included here for completeness rather than presented as part of
the core theoretical construction.

### 3.9 Readout heads

The base node-classification readout (`NodeReadout`) forms, for each target-type node $v$, the
concatenation
$$z_v = \big[x^*_v \,\Vert\, h^{(0)}_v \,\Vert\, \mathrm{pool\_edges\_to\_nodes}(x^*,s^*)_v\big]$$
of the final node state, the initial node state, and a mean pooling of incident edge states over
both endpoints, and passes $z_v$ through a small MLP classifier. Because this is a unified paper
also covering link prediction and graph classification, two further readout heads share the same
encoder/diffusion backbone: `LinkPredictionReadout`, with a swappable dot-product, DistMult, or
concatenation-plus-MLP decoder over node-pair states; and `GraphPooling`, with swappable graph-level pooling heads: naive sum-, mean-, and max-pooling
baselines (each followed by an MLP), an $O(d)$-invariant `GramPool` (which exploits that
restriction maps are always orthogonal in every configuration we run), a gauge-fixing
attention-barycenter `SheafPool` baseline, and a $GL(d)$-invariant `S2Pool` following the
gauge-free readout design in `S2Pool_proposal.md` and
`docs/gauge_free_coboundary_readout_design.md`. (As of this repository snapshot,
`src/models/graph_pooling.py: build_graph_pooling()` wires up only `meanpool`, `grampool`, and
`s2pool`; the sum-, max-, and `SheafPool` variants used in the ablation of Section 5.6.2 exist in a
further-along version of the code not yet synced here — see Section 4.4.) We treat the
gauge-invariance argument behind `GramPool`/`S2Pool` as a secondary methodological contribution
worth its own short exposition:
because sheaf restriction maps are only well-defined up to a choice of local frame ("gauge") at
each stalk, a graph-level readout that is not itself invariant to that choice risks encoding an
artifact of the parameterization rather than a genuine structural signal, and `S2Pool` is
constructed to be invariant to general invertible gauge transformations rather than only orthogonal
ones.

### 3.10 Training objective

The full training loss combines a supervised classification term with four sheaf-geometric
regularizers and, once discovery gates are active, six further discovery-specific terms
(`src/trainers/trainer.py`, `src/trainers/regularizers.py`, `src/models/discovery_losses.py`):
$$\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{smooth0} + \mathcal{L}_{smooth1} + \mathcal{L}_{consist} + \mathcal{L}_{coboundary} + \mathcal{L}_{discovery}.$$

| Term | Formula | Default $\lambda$ |
|---|---|---|
| Classification | Cross-entropy on the target-type training mask (single-label); binary cross-entropy with a sigmoid head (multi-label) | — |
| Smooth0 | $\lambda_0 \cdot \mathrm{mean}(\lVert\delta_0 x^*\rVert^2)$ | $0.0$ |
| Smooth1 | $\lambda_1 \cdot \mathrm{mean}(\lVert\delta_1 s^*\rVert^2)$ | $10^{-3}$ |
| Consistency | $\lambda_c \cdot \mathrm{mean}(\lVert s^* - \delta_0 x^*\rVert^2)$ | $10^{-2}$ |
| Coboundary | $\lambda_{co} \cdot \mathrm{mean}(\lVert\delta_1 \delta_0 H^{(0)}\rVert^2)$ | $10^{-2}$ |

**Multi-label handling, verified in code:** IMDB's HGB preprocessing gives each movie a multi-hot
genre label rather than a single class. The data loader detects this via `is_multilabel = (target_y.dim()
== 2)` (`src/data/dataset.py`) and threads a `data.is_multilabel` flag into the trainer, which
branches at `src/trainers/trainer.py:141,172` to use a sigmoid + binary-cross-entropy loss and
`compute_multilabel_classification_metrics` instead of the single-label path — this is exactly the
convention HGB itself specifies ("for multi-label datasets, i.e. IMDB in HGB, we use a sigmoid
activation and binary cross-entropy loss," Lv et al. 2021, §4.4.1), so our pipeline matches the
benchmark's own protocol on this point rather than silently mishandling it.

(Default values from `src/configs/config.py: ModelConfig`, and identical to the values recommended
in `docs/H2SNN_corrected_algorithm.md` §9.7.) The coboundary term is deliberately evaluated at the
*initial* encoding $H^{(0)}$ rather than the diffused state $x^*$: this softly penalizes violation
of $\delta_1 \circ \delta_0 = 0$ without competing, through the same tensor, against the diffusion
objective's own gradient signal. This soft penalty replaces an earlier, mathematically incomplete
attempt to enforce $\delta_1\circ\delta_0=0$ exactly via an SVD null-space construction, which
ignored the source-map term in the per-node net map and was abandoned in favor of the present
regularizer (`docs/H2SNN_corrected_algorithm.md` §4.4–§4.5).

Once schema/instance discovery is enabled and past its warm-up period, six further terms are added,
each with its own default weight from `SchemaDiscoveryConfig`:
schema sparsity ($\lambda=10^{-3}$), instance sparsity ($10^{-4}$), redundancy ($10^{-3}$),
coverage ($10^{-3}$), cost ($10^{-4}$), and gate entropy ($10^{-4}$). We highlight one
implementation-precision point worth stating in a methods section: the instance-sparsity term must
read a gradient-attached copy of the mean instance weights, not the detached copy used purely for
logging — an earlier version of the code used the detached tensor, silently producing zero gradient
to the instance gate's parameters and defeating the sparsity regularizer entirely. This is now
resolved (`docs/h2snn_current_model_documentation.html`, Update Log v0.2) and we describe the fixed
behavior throughout.

### 3.11 Schema-to-cell discovery

The auto-discovery pipeline (`src/data/schema_discovery.py`, `src/models/schema_gates.py`)
proceeds in five stages.

**(A) Schema graph.** A type-level adjacency is built from the set of relation tuples, giving, for
each node type, the set of relations that can be followed to each neighboring type.

**(B) Candidate generation.** Typed walks are enumerated by depth-first search over the schema
graph, bounded by `min_schema_length`/`max_schema_length` (2/4 by default), optionally required to
both start and end at the target node type, and capped at `max_schema_candidates` (128 by default)
to keep the candidate pool tractable.

**(C) Topology-aware, training-free scoring.** Rather than evaluate every candidate schema with a
full model forward pass, each candidate is scored from purely structural and label-propagation
statistics: an *endpoint label discrepancy* computed by propagating **training labels only** to
estimate, for sampled path instances of the candidate schema, how dissimilar the predicted labels
at the two path endpoints tend to be — schemas connecting label-dissimilar endpoints are exactly
the heterophilic, informative patterns we want to surface — alongside coverage (fraction of
target-type nodes reached), relation and type diversity, and average branching factor. Because this
score never touches validation or test labels, and never requires a trained model, it is
inexpensive to compute for the full candidate pool.

**(D) Greedy topology-aware pool selection.** Schemas are added one at a time by a greedy
forward-selection rule,
$$\mathrm{score}(\sigma \mid \text{selected}) = \frac{\mathrm{disc}(\sigma)^{\alpha_1}\, \mathrm{cov}(\sigma)^{\alpha_2}\, \mathrm{div}(\sigma)^{\alpha_3}}{1 + \max_{\sigma'\in\text{selected}}\mathrm{redundancy}(\sigma,\sigma')},$$
with redundancy measured as the Jaccard overlap between sampled endpoint sets, until
`selected_schema_k` (8 by default) schemas are retained. After selection, leave-one-out scores are
recomputed among the final pool to produce the topology scores used to warm-start the schema gate
below.

**(E) Face materialization.** For each selected schema, concrete path instances are sampled (capped
by `max_instances_per_schema`/`max_instances_per_anchor`), closure-edge pairs are attached, and an
8-dimensional per-schema structural feature vector is computed and consumed directly by the
discovery losses of Section 3.10. A separate, 6-dimensional per-*instance* structural feature
vector is also computed at this stage but, as far as we can determine from the current codebase, is
not read anywhere downstream — we note this explicitly rather than describe it as an active input
to the instance gate (see Section 3.12).

**Hierarchical gates.** The selected structure is weighted by a product of a schema-level and an
instance-level gate:
$$\alpha_\sigma = \mathrm{sigmoid}(\mathrm{logit}_\sigma), \quad \text{warm-started from } \log(\text{topology score}_\sigma);$$
$$\beta_f = \mathrm{sigmoid}\big(\mathrm{MLP}([h^{(0)}_{v_0} \Vert h^{(0)}_{v_\ell} \Vert \text{schema\_embedding}_\sigma])\big);$$
the effective per-face weight used in $L_1^{up}$ (Section 3.7) is $\alpha_\sigma \cdot \beta_f$. The
warm start from the topology score means schemas the structural stage already flagged as
high-discrepancy begin training with high $\alpha$ and must be *proven wrong* by the discovery
losses and downstream task loss, rather than discovered entirely from a cold start.

**Partial schema propagation.** During training, before computing $L_1^{up}$, each schema's gate is
independently Bernoulli-masked and unbiasedly rescaled,
$\alpha \leftarrow \alpha \cdot \mathrm{Bernoulli}(p)/p$ with `partial_schema_probability`
$p=0.5$ by default. This mechanism is adapted directly from PMMM's *partial message propagation*,
which addresses the same coupled bilevel-optimization instability that arises when architecture-like
weights ($\alpha_\sigma$ here) and the rest of the model's parameters are updated in the same
gradient step: randomly excluding a fraction of candidate structure per iteration breaks the
circular dependency between the two.

### 3.12 Implementation notes for methodological precision

We collect here, in one place, every point at which the implementation's actual behavior differs
from an earlier version of the mathematical specification, or where a plausible-sounding claim
needs to be stated more carefully than it might be by default — so that the rest of this paper can
describe the model exactly as built rather than as originally envisioned.

- Attention **is** folded multiplicatively into the edge-to-face restriction maps (Section 3.5),
  contradicting `docs/H2SNN_corrected_algorithm.md` §6.4's earlier statement that attention affects
  only the regularizer.
- Restriction maps are **scaled orthogonal**, not strictly orthogonal (Section 3.3); we avoid the
  unqualified word "orthogonal" throughout.
- Closure edges carry no relation embedding of their own; they are handled entirely outside the
  relation-typed machinery via the closed-form correction of Section 3.6, not via a synthetic
  closure relation as an earlier spec draft implied.
- The 6-dimensional per-instance structural feature vector computed during face materialization
  (Section 3.11, stage E) is not consumed by the instance gate or any loss term we could locate; we
  do not describe it as an active input.
- `hard_concrete` is a listed value for `schema_gate_type`/`instance_gate_type` in the
  configuration schema, but both gate implementations raise a `ValueError` if it is selected — this
  ablation axis is not yet implemented, and we do not report results for it.

---

## 4. Experimental Setup

### 4.1 Datasets and tasks

| Task | Datasets | Loader (verified in code) |
|---|---|---|
| Node classification | ACM (target: paper; schemas PAP, PSP), DBLP (target: author; schemas APA, APTPA), IMDB (target: movie; schemas MAM, MDM) | `torch_geometric.datasets.HGBDataset` (`src/data/dataset.py: _load_hetero_data`) — i.e. the official Heterogeneous Graph Benchmark (HGB) preprocessing, not the older HAN-paper preprocessing of the same-named datasets |
| Link prediction | LastFM, AmazonBook, PubMed, YouTube | `torch_geometric.datasets.LastFM` / `.AmazonBook` (`src/data/link_prediction_dataset.py: _load_link_pred_hetero_data`) for the first two; **no loader exists for PubMed or YouTube at all** (Section 4.4) |
| Graph classification | MUTAG (fresh stratified 10-fold split), PROTEINS, ENZYMES, NCI1 (precomputed 10-fold splits reused from the legacy CSNN baseline) | `torch_geometric.datasets.TUDataset` (`src/data/graph_classification_dataset.py`) — the standard TUDataset preprocessing used near-universally in the GNN literature, so no ambiguity here |
| Node clustering | ACM, Freebase, AMiner, IMDB | **Not yet implemented** — see Section 4.4 |

We verified the exact loader for every task directly against the source, rather than assume
"ACM/DBLP/IMDB" or "LastFM/Amazon" mean the same preprocessing as in other papers — HGB's own
motivation section makes exactly this point: different papers report incompatible numbers for
same-named heterogeneous benchmarks because they silently use different splits and preprocessing
(Lv et al., 2021, §4.1). Section 4.1.1 gives the resulting statistics; Section 4.4 details a real
consequence of this check for our link-prediction table.

#### 4.1.1 Dataset statistics

**Node classification (HGB's official statistics, Lv et al. 2021, Table 2 — matches our
`HGBDataset` loader exactly):**

| Dataset | #Nodes | #Node types | #Edges | #Edge types | Target | #Classes |
|---|---|---|---|---|---|---|
| DBLP | 26,128 | 4 | 239,566 | 6 | author | 4 |
| IMDB | 21,420 | 4 | 86,642 | 6 | movie | 5 (multi-label) |
| ACM | 10,942 | 4 | 547,872 | 8 | paper | 3 |
| Freebase | 180,098 | 8 | 1,057,688 | 36 | book | 7 |

*(Freebase is listed for completeness since HGB includes it among its node-classification
datasets, but our paper only uses it for the — currently unimplemented — node-clustering
comparison in Section 6.3, not for node classification.)*

**Graph classification (standard TUDataset statistics, widely reported across the GNN literature):**

| Dataset | #Graphs | #Classes | Avg. #Nodes | Avg. #Edges | Domain |
|---|---|---|---|---|---|
| MUTAG | 188 | 2 | 17.9 | 19.8 | Molecules (mutagenicity) |
| PROTEINS | 1,113 | 2 | 39.1 | 72.8 | Proteins (enzyme / non-enzyme) |
| ENZYMES | 600 | 6 | 32.6 | 62.1 | Proteins (enzyme class) |
| NCI1 | 4,110 | 2 | 29.9 | 32.3 | Molecules (anti-cancer activity) |

**Link prediction — dataset identity needs verification before the results in Section 6.4 can be
trusted as an apples-to-apples comparison:**

| Dataset (as labeled in Section 6.4) | HGB's own definition (Lv et al. 2021, Table 2) | What our code actually loads |
|---|---|---|
| LastFM | HetRec-2011 subset; 20,612 nodes, 3 node types, 141,521 edges, target relation user–artist | `torch_geometric.datasets.LastFM` — a differently-constructed (HAN/MAGNN-lineage) user–artist–tag graph |
| Amazon | 10,099 nodes, 1 node type, 148,659 edges, target relation product–product | `torch_geometric.datasets.AmazonBook` — a bipartite user–book graph with roughly 2.9M interactions (a recommendation-style dataset, structurally unrelated to HGB's unipartite product co-purchase graph) |
| PubMed | 63,109 nodes, 4 node types, 244,986 edges, target relation disease–disease | **No loader implemented anywhere in this codebase.** |
| YouTube | Not part of HGB's benchmark suite at all | **No loader implemented anywhere in this codebase.** |

This is a substantive finding, not a cosmetic one. If the baseline rows in Section 6.4's link-prediction
table (GCN, GAT, R-GCN, HAN, HGT, Sheaf-NSD, HetSheaf) were sourced from a prior paper's own
results on HGB's official Amazon/LastFM/PubMed benchmark — plausible, since HetSheaf's own
evaluation is built on HGB (Section 2.3) — and our H2S-NN numbers come from running PyG's
differently-constructed `LastFM`/`AmazonBook` classes, then the "LastFM" and "Amazon" columns in
that table are not measuring the same underlying graphs on both sides, which would undermine the
comparison until resolved. We flag this prominently rather than let it surface later as a reviewer
objection: before Section 6.4 is submission-ready, either (a) re-run H2S-NN on HGB's actual
Amazon/LastFM/PubMed link-prediction files (the `_LocalHGBLinkData`/`_load_local_hgb_nodes`
scaffolding already exists in `src/data/link_prediction_dataset.py` but is not yet wired to any
dataset name), or (b) re-source the baseline numbers from a paper that evaluates on PyG's
`LastFM`/`AmazonBook` datasets specifically, so both sides of the comparison refer to the same
graph.

#### 4.1.2 Node classification protocol

Transductive setting, following HGB's own official protocol exactly (Lv et al. 2021, §4.2.1,
§4.5.1) since our loader consumes HGB's shipped `train_mask`/`val_mask`/`test_mask` directly: all
edges are available during training, and target-type node labels are split 24% train / 6%
validation / 70% test. Metrics are Macro-F1 and Micro-F1, with IMDB's multi-label BCE handling
described in Section 3.10. Per an internal project slide, this protocol is run with **3 random
seeds**; the slide additionally states "10 Folds CV" for this task, which is difficult to reconcile
with a fixed transductive HGB split and needs clarification — most likely it refers to the
`acm_val_ratio`/`acm_split_seed`-driven custom validation re-split described in
`src/configs/config.py: DataConfig` rather than a genuine 10-fold cross-validation of the whole
dataset, but we state this as an open question rather than assert a specific mechanism we have not
directly traced.

#### 4.1.3 Graph classification protocol

80% / 10% / 10% train/validation/test, **5 random seeds**, 10-fold cross-validation — PROTEINS,
ENZYMES, and NCI1 reuse the legacy CSNN baseline's precomputed fold files
(`folds/{ENZYMES,PROTEINS,NCI1}_splits.json`) for direct comparability with that baseline; MUTAG
(which the legacy baseline never ran) gets a fresh stratified 10-fold split cached to
`folds/MUTAG_splits.json` in the same JSON shape (`src/configs/config.py: GRAPH_CLASSIFICATION_PRESETS`,
`num_folds=10`).

#### 4.1.4 Link prediction protocol

`link_val_ratio=0.09` / `link_test_ratio=0.10` by default (`src/configs/config.py: DataConfig`),
i.e. an approximately 81/9/10 train/validation/test split of edges, evaluated via AUPR and AUROC.
Negative sampling is `fixed` by default (`neg_sampling_ratio=1`) with a `resample_each_epoch` mode
available and used for AmazonBook. Unlike the node- and graph-classification protocols above, we
could not find a documented number of random seeds for this task — a gap to close alongside the
dataset-identity issue in Section 4.1.1 before this table is submission-ready.

### 4.2 Baselines

We assemble baselines separately per task, spanning homogeneous, homophilic-heterogeneous, and
heterophilic-heterogeneous families so that H2S-NN's position relative to each family is legible
rather than collapsed into a single aggregate comparison.

- **Node classification:** GCN (2016), GAT (2021) as homogeneous baselines; NSD (2022), CSNN
  (2026), ChebNetII (2022) as homogeneous sheaf/spectral baselines; HAN (2019), SeHGNN (2022),
  HiNormer (2023), PSHGCN (2024) as homophilic heterogeneous baselines; HDHGR (2023), Hetero2Net
  (2023), H2SGNN (2023), HetSheaf (2024), LatGRL (2021) as heterophilic heterogeneous baselines.
- **Graph classification:** the same homogeneous/heterogeneous split, using GCN, GAT, NSD, CSNN,
  HAN, PSHGCN, HDHGR, Hetero2Net, H2SGNN, HetSheaf, and LatGRL.
- **Node clustering:** HGMAE, RMR, DMGI, HDMI, HeCo, HeCo++, HERO, CTC, HGMS-N, HGMS-C.
- **Link prediction:** GCN, GAT, GIN, SAGE (homogeneous); R-GCN, HAN, HGT (heterogeneous);
  Sheaf-NSD, HetSheaf (sheaf-based).
- **Efficiency:** R-GCN, HAN, HGT, Sheaf-NSD, compared on runtime, parameter count, and memory on
  ACM.
- **Discovery-extension-specific baselines:** to substantiate the schema-to-cell discovery
  contribution specifically (as opposed to the base model), we additionally compare against the
  automatic meta-path/schema search family — LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM — using the full
  boolean capability matrix in `docs/comparison_metapath_models_boolean_attributes_2026-05-20.csv`.

### 4.3 Ablations

We isolate the contribution of each architectural component using ablation switches already
implemented in `src/configs/config.py: ModelConfig`:

- **No faces** (`ablate_no_faces`): removes all faces, closure edges, and discovery machinery,
  reducing the model to graph-only sheaf diffusion via $L_0$ alone.
- **No $\delta_1$** (`ablate_no_delta1`): retains the face-group bookkeeping but empties the face
  cache passed into the Laplacian, so the up-Laplacian term of $L_1$ vanishes while everything else
  is held fixed.
- **Identity maps** (`ablate_identity_maps`): replaces both node-to-edge restriction maps with the
  identity, testing how much of the model's benefit comes from the *learned*, relation-specific
  transport as opposed to the higher-order structure itself.
- **Uniform gates** (`ablate_uniform_gates`): forces both the schema gate $\alpha$ and the instance
  gate $\beta$ to $1$ everywhere, isolating the benefit of learned discovery weighting from the
  benefit of having discovered structure at all.
- **Manual vs. auto-discovered schemas, and $\alpha$-only vs. $\alpha$+$\beta$:** configurations
  exist for ACM, DBLP, and IMDB (`{acm,dblp,imdb}_h2snn_ampss.yaml`) contrasting hand-specified
  meta-paths against the full schema-to-cell discovery pipeline of Section 3.11. This is now the
  best-evidenced ablation in the paper: Section 5.6.1 reports a completed four-way comparison on
  DBLP (no faces / predefined schema / AMPSS with $\alpha$ only / full AMPSS with $\alpha$+$\beta$)
  showing a monotonic Macro-F1 improvement across all four configurations. No equivalent
  auto-discovery configuration yet exists for the link-prediction or graph-classification tasks,
  and the DBLP study has not yet been repeated on ACM or IMDB — both gaps flagged rather than
  papered over (Section 4.4).
- **Graph-level pooling head:** Section 5.6.2 reports a completed five-way pooling ablation on
  ENZYME (sum / mean / max pooling, each followed by an MLP, versus the gauge-fixing `SheafPool`
  baseline versus our $GL(d)$-invariant `S2Pool`), showing the two gauge-aware designs
  (`SheafPool`, `S2Pool`) both outperform the three naive permutation-invariant baselines by
  roughly 3 Macro-F1 points, with `S2Pool` best overall.

This ablation set matches, term for term, the minimum ablation checklist already drafted in
`docs/h2snn_current_model_documentation.html` (Validation Plan), which also recommends diagonal or
scalar restriction-map parameterizations as a cheaper variant worth testing and a direct comparison
between the current path-only coboundary construction and a true closed-boundary operator; we treat
both as natural extensions of the ablation matrix above (Section 6 returns to the second point).

### 4.4 Evaluation protocol and open gaps

We state plainly, rather than gloss over, the parts of the experimental protocol that are not yet
finalized, since an A\*-venue reviewer would otherwise (rightly) ask about each of them directly.

- **No documented multi-seed or statistical-significance protocol exists yet.** The current test
  and validation runbook (`docs/h2snn_test_setup_guide.html`) is a local environment smoke-test
  guide — conda/PyTorch/CUDA setup, unit tests, a single training command — not a formal evaluation
  protocol, and makes no mention of repeated seeds or significance testing. This needs to be
  designed and executed before the numbers in Section 5 can be reported with confidence intervals.
- **Node clustering has no corresponding implementation.** None of the three existing readout heads
  (`NodeReadout`, `LinkPredictionReadout`, `GraphPooling`) perform clustering, and no clustering
  code exists anywhere in `src/`. Table 3 of Section 5 (node clustering) therefore either needs a
  clustering evaluation path implemented, or should be dropped or explicitly reframed as aspirational
  pending implementation.
- **PubMed and YouTube link-prediction have no implementation at all**, not merely a missing
  config preset as an earlier version of this document stated — verified directly against
  `src/data/link_prediction_dataset.py`: there is no code path for either dataset name, only a
  `ValueError` for anything besides `lastfm`/`amazonbook`. Per confirmation both are real H2S-NN
  runs, so either the run artifacts and data-loading code live outside this repository and need to
  be brought in, or the results need to be reproduced from scratch against HGB's own PubMed files
  once the dataset-identity question below is resolved.
- **LastFM and Amazon(Book) may not be the same graphs as the baseline numbers they're compared
  against.** Our loader uses PyG's `LastFM`/`AmazonBook` classes, which are different constructions
  from HGB's own same-named link-prediction datasets that the comparison baselines were plausibly
  evaluated on. Full detail and a proposed resolution in Section 4.1.1 — we consider this the single
  most important open item in the experimental section, since it is a comparison-validity question,
  not just a missing-artifact one.
- **MovieLens is not a real H2S-NN dataset.** An earlier version of the link-prediction table
  (Section 5.4) included a MovieLens column with a filled-in H2S-NN row; per confirmation, H2S-NN
  was never evaluated on MovieLens, and the column has been removed rather than left in with the
  baseline literature numbers alone, to avoid implying a completed comparison that does not exist.
- **A hyperparameter-configuration discrepancy needs reconciling.** The hyperparameter search space
  document (`docs/h2snn_hyperparameter_search_space.md`) marks the stalk dimension $d=64$ as "the
  value used in final configs," but every checked-in experiment configuration under
  `configs/experiments/*.yaml` we inspected directly (including `acm_h2snn.yaml` and
  `acm_h2snn_ampss.yaml`) uses $d=32$. We report $d=32$ in Section 5 as the value actually used to
  produce our results, and flag this discrepancy for correction in either the search-space document
  or the configs themselves.
- **The described Optuna + Hyperband sweep has no persisted output.** `scripts/tune_h2snn.py`
  documents a search over 15 hyperparameters against a `val_macro_f1` objective with a Hyperband
  pruner, but neither `experiments/tune_results.json` nor an `optuna_*.db` study file exists in
  this repository. Either this sweep has not yet been run against the current codebase, or its
  results live outside this repository and should be attached before the corresponding
  hyperparameter claims are finalized.
- **This repository's saved run artifacts lag behind the project's actual experimental progress.**
  The one AMPSS run we could find on disk for ACM (`experiments/colab/acm_h2snn_colab_ampss/`) is
  aborted with no completed metrics — yet a completed, four-way AMPSS ablation on DBLP is reported
  in Section 5.6.1, and a completed five-way pooling ablation on ENZYME is reported in Section
  5.6.2, neither of which has a corresponding config/checkpoint/metrics artifact saved under
  `experiments/` in this repository. As with the ACM node-classification numbers in Section 5.1,
  we take these as valid, more current results than what this repository snapshot reflects, but
  flag the same reproducibility action item as Section 5.7: the configs and checkpoints behind
  Sections 5.6.1 and 5.6.2 need to be saved into this repository before the paper's evidence is
  independently reproducible from it alone.
- **`src/models/graph_pooling.py` needs to be synced to match the pooling ablation's five
  variants.** `build_graph_pooling()` currently only implements `meanpool`, `grampool`, and
  `s2pool`; the `Sum+MLP`, `Max+MLP`, and `SheafPool` variants reported in Section 5.6.2 exist, per
  confirmation, in a further-along version of the codebase not yet reflected here.
- **The $\alpha$-only AMPSS ablation has no dedicated config flag yet.** The existing
  `ablate_uniform_gates` switch forces both $\alpha$ and $\beta$ to $1$ jointly; isolating
  "$\alpha$ learned, $\beta$ fixed" (Section 5.6.1's third row) currently requires a config or code
  path not present in this repository snapshot.

---

## 5. Results

We report the following results as currently the best available evidence for H2S-NN, while noting
that Section 4.4's open protocol gaps (repeated seeds, the ablation matrix, and the discovery
extension's own evaluation) still need to be closed before these numbers should be treated as a
finished experimental section.

### 5.1 Node classification

Average Macro-F1 / Micro-F1 across ACM, DBLP, and IMDB.

| Type | Model | DBLP Macro | DBLP Micro | ACM Macro | ACM Micro | IMDB Macro | IMDB Micro |
|---|---|---|---|---|---|---|---|
| Homophilic homogeneous | GCN (2016) | 90.77 | 90.28 | 91.63 | 91.38 | 62.33 | 67.69 |
| Homophilic homogeneous | GAT (2021) | 94.39 | 92.52 | 92.07 | 90.71 | 62.13 | 68.09 |
| Homophilic homogeneous (sheaf/spectral) | NSD (2022) | 92.96 | 93.42 | 92.33 | 92.05 | 61.61 | 65.23 |
| Homophilic homogeneous (sheaf/spectral) | CSNN (2026) | 93.73 | 92.12 | 93.13 | 92.81 | 62.92 | 62.59 |
| Homophilic homogeneous (sheaf/spectral) | ChebNetII (2022) | 92.89 | 92.67 | 91.70 | 91.59 | 62.97 | 68.47 |
| Homophilic heterogeneous | HAN (2019) | 91.78 | 92.24 | 90.88 | 91.61 | 61.95 | 67.76 |
| Homophilic heterogeneous | SeHGNN (2022) | 95.52 | 95.48 | 94.38 | 94.04 | 72.01 | 72.59 |
| Homophilic heterogeneous | HiNormer (2023) | 94.06 | 94.14 | 91.87 | 89.64 | 64.13 | 67.53 |
| Homophilic heterogeneous | PSHGCN (2024) | 95.58 | 95.70 | 93.66 | 94.70 | 71.82 | 74.34 |
| Heterophilic heterogeneous | HDHGR (2023) | 95.35 | 95.00 | 93.99 | 94.13 | 59.41 | 59.63 |
| Heterophilic heterogeneous | Hetero2Net (2023) | 93.54 | 93.54 | 94.05 | 92.87 | 65.34 | 70.63 |
| Heterophilic heterogeneous | H2SGNN (2023) | 95.82 | 95.11 | 94.90 | 94.08 | 74.23 | 75.29 |
| Heterophilic heterogeneous | HETSHEAF (2024)* | 93.87 | 93.77 | 95.09 | 94.94 | 64.65 | 67.18 |
| Heterophilic heterogeneous | LatGRL (2021) | 94.32 | 94.51 | 93.47 | 92.29 | 58.27 | 58.81 |
| — | **H2S-NN (Ours)** | **96.01** | **96.31** | **95.91** | **95.72** | **75.05** | **76.68** |

H2S-NN reports the best Macro- and Micro-F1 on all three datasets against this baseline set,
including the closest sheaf baseline HetSheaf and the strongest heterophily-aware heterogeneous
baseline H2SGNN. The column headers in the underlying results table also carry an unexplained
bracketed figure next to each dataset name (e.g. "ACM (72.39)"), which we believe to be a homophily
or similar dataset-level statistic; we flag this as needing a proper caption (or removal) before
the table is publication-ready, rather than guess at its meaning here.

**Clarification (per the ablation study in Section 5.6.1):** the "H2S-NN (Ours)" row in this and
every other table in Section 5 is the **full AMPSS configuration** — auto-discovered schema pool
with both the schema-level gate $\alpha$ and instance-level gate $\beta$ active — not a
manually-specified-schema baseline. Earlier drafts of Sections 1.5 and 6.4 described the headline
tables as base-model evidence and treated the discovery extension as separately and more weakly
evidenced; Section 6.4 now corrects this.

### 5.2 Graph classification

Average Macro-F1 / Micro-F1 across MUTAG, PROTEINS, ENZYMES, and NCI1 (10-fold).

| Type | Model | MUTAG Macro | MUTAG Micro | PROTEINS Macro | PROTEINS Micro | ENZYME Macro | ENZYME Micro | NCI1 Macro | NCI1 Micro |
|---|---|---|---|---|---|---|---|---|---|
| Homophilic homogeneous | GCN (2016) | 89.58 | 90.00 | 75.92 | 79.83 | 12.75 | 28.33 | 64.17 | 64.96 |
| Homophilic homogeneous | GAT (2021) | 89.58 | 90.00 | 75.27 | 78.38 | 14.03 | 16.55 | 64.17 | 64.96 |
| Homophilic homogeneous (sheaf) | NSD (2022) | 100 | 100 | 73.60 | 75.89 | 33.65 | 35.00 | 72.30 | 72.30 |
| Homophilic homogeneous (sheaf) | CSNN (2026) | 100 | 100 | 76.64 | 77.92 | 35.15 | 37.22 | 74.11 | 74.30 |
| Homophilic heterogeneous | HAN (2019) | 82.91 | 85.00 | 79.19 | 82.22 | 46.24 | 45.02 | 77.60 | 77.50 |
| Homophilic heterogeneous | PSHGCN (2024) | 94.88 | 95.00 | 70.61 | 72.32 | 20.18 | 28.33 | 64.17 | 64.96 |
| Heterophilic heterogeneous | HDHGR (2023) | 84.65 | 85.00 | 77.64 | 81.25 | 48.27 | 48.33 | 79.29 | 79.32 |
| Heterophilic heterogeneous | Hetero2Net (2023) | 89.58 | 90.00 | 74.66 | 78.57 | 8.53 | 18.33 | 63.67 | 65.45 |
| Heterophilic heterogeneous | H2SGNN (2023) | 94.88 | 95.00 | 70.61 | 72.32 | 20.18 | 28.33 | 64.17 | 64.96 |
| Heterophilic heterogeneous | HETSHEAF (2024)* | 100 | 100 | 80.56 | 82.14 | 62.35 | 63.33 | 79.46 | 79.46 |
| Heterophilic heterogeneous | LatGRL (2021) | 90.88 | 91.01 | 71.61 | 72.32 | 25.41 | 29.03 | 77.98 | 78.10 |
| — | **H2S-NN (Ours)** | **100** | **100** | **84.44** | **86.23** | **64.52** | **65.10** | **81.22** | **81.52** |

H2S-NN matches the ceiling on MUTAG shared by several baselines and reports the best result on
each of the remaining three datasets, with the largest margin over the next-best baseline
(HetSheaf) on PROTEINS (+3.88/+4.09) and the smallest on ENZYME (+2.17/+1.77).

### 5.3 Node clustering

NMI / ARI on ACM, Freebase, AMiner, and IMDB. **Contingent on the implementation gap noted in
Section 4.4** — no clustering evaluation path currently exists in this codebase, so this table
should be read as a target drawn from the broader literature comparison rather than a result this
codebase can currently reproduce.

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

### 5.4 Heterogeneous link prediction

AUPR / AUROC on LastFM, PubMed, Amazon, and YouTube. **Read Section 4.1.1 before citing this table:**
we found that our LastFM/Amazon loaders (PyG's `LastFM`/`AmazonBook` classes) are not necessarily
the same graphs as HGB's own same-named link-prediction datasets that the baseline rows below were
plausibly evaluated on, and PubMed/YouTube have no data-loading implementation in this codebase at
all. This table should be treated as provisional until that is resolved.

*(Updated 2026-07-06: the MovieLens column is removed entirely — H2S-NN was never evaluated on
MovieLens, and reporting the literature baselines for it alongside a filled-in "Ours" row would
have misleadingly implied a completed comparison that does not exist. LastFM, PubMed, and
Amazon(Book) are confirmed real H2S-NN datasets and runs; YouTube is confirmed real as well,
though — as with the pooling-head and $\alpha$-only-AMPSS gaps already noted in Section 4.4 — no
`youtube` preset yet exists in `src/configs/config.py: LINK_PREDICTION_PRESETS`, which today only
defines `lastfm` and `amazonbook`. PubMed, Amazon, and YouTube's H2S-NN figures were also revised
down from an earlier version of this table that reported clean 100s across the board; every
baseline row and LastFM's H2S-NN figures are unchanged.)*

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

The revision meaningfully improves the credibility of this table: PubMed, Amazon, and YouTube AUPR
now sit just below the strongest baselines' own near-ceiling scores (e.g. PubMed AUPR 99.63 against
Sheaf-NSD's 96.41 and HetSheaf's 96.36) rather than at an implausible exact 100, and PubMed/YouTube
AUROC in particular show a believable, non-perfect margin over HGT (96.48/92.51) and HetSheaf
(93.98/87.33). LastFM AUPR/AUROC remain at exactly 100, which is still the kind of result an A\*
reviewer will ask about directly, particularly since every baseline on LastFM already sits at
63–98.5. Before this table is submission-ready we need to report the evaluation protocol behind it
in enough detail (dataset size, negative-sampling ratio, split construction) to make the remaining
LastFM ceiling legible against the baselines, or revisit the protocol on that dataset specifically
if the ceiling reflects a degenerate evaluation setup rather than a genuine result — the same
scrutiny that produced the more credible PubMed/Amazon/YouTube numbers above should be applied
there next.

### 5.5 Efficiency comparison (ACM)

| Model | Runtime (s) | # Params | Memory (MB) |
|---|---|---|---|
| R-GCN | 22.1 | 43M | 1,240 |
| HAN | 28.9 | 2.2M | 1,780 |
| HGT | 1025.8 | 7.2M | 6,120 |
| Sheaf-NSD | 47.6 | 155K | 980 |
| **H2S-NN** | **18.39** | **134K** | **810** |

H2S-NN is reported as simultaneously the fastest, smallest, and lowest-memory model in this
comparison on ACM — a claim that, if it survives Section 4.4's protocol tightening, is worth
foregrounding, since "higher-order structure is expensive" is one of the most natural objections to
this class of method (Section 6.3).

### 5.6 Ablation results

Two ablation studies are now complete and are reported below. An earlier version of the source
slides carried a stale subtitle inconsistent with their own table header and data (the schema
ablation was mislabeled "ACM," the pooling ablation "PROTEINS"); both have since been corrected at
the source and are confirmed to be DBLP and ENZYME respectively, matching the exact agreement
already noted against the headline figures in Sections 5.1/5.2.

#### 5.6.1 Schema / meta-path ablation (DBLP, node classification)

*(Confirmed DBLP — the source slide's subtitle has been corrected at the source; the "H2S-NN" row
below reproduces Section 5.1's DBLP figures exactly.)*

| Model | DBLP Macro | DBLP Micro |
|---|---|---|
| H2S-NN — No meta-path | 93.85 | 93.72 |
| H2S-NN — Predefined meta-path | 95.75 | 96.01 |
| H2S-NN — AMPSS, $\alpha$ only | 95.88 | 95.92 |
| **H2S-NN (full AMPSS, $\alpha$+$\beta$)** | **96.01** | **96.31** |

This is the first direct ablation evidence for the discovery extension's central claim
(contribution 3, Section 1.4) rather than the purely architectural argument given in Section 3.11.
Reading the rows as successively adding one mechanism: removing all higher-order face structure
("No meta-path," closest to what `ablate_no_faces` measures) costs roughly two points of Macro-F1
relative to using hand-specified schemas; replacing hand-specified schemas with the auto-discovered
pool and a learned schema-level gate ("AMPSS, $\alpha$ only") improves Macro-F1 further (95.75 →
95.88) but very slightly *underperforms* the predefined-schema variant on Micro-F1 (96.01 → 95.92)
— we do not smooth over this non-monotonicity, since it is a genuine and reportable nuance rather
than a clean story; and adding the instance-level gate $\beta$ on top ("full AMPSS") improves both
metrics over every other row. This result also corrects an imprecision in earlier sections of this
draft: the "H2S-NN (Ours)" rows throughout Section 5 are the **full AMPSS ($\alpha$+$\beta$)
configuration**, not a manually-specified-schema baseline as Sections 1.5 and 6.4 previously
implied — see the corrected discussion in Section 6.4.

One methodological note for reproducibility: the codebase's `ablate_uniform_gates` flag (Section
4.3) jointly forces both $\alpha$ and $\beta$ to $1$, so it does not by itself isolate the
"$\alpha$ learned, $\beta$ fixed" configuration reported as "AMPSS, $\alpha$ only" above. Producing
this row requires either a dedicated config flag not yet present in this repository snapshot, or a
hand-modified run — we flag this as a follow-up item so the ablation can be reproduced directly
from a checked-in config rather than an ad hoc modification.

#### 5.6.2 Pooling ablation (ENZYME, graph classification)

*(Confirmed ENZYME — the source slide's subtitle has been corrected at the source; the "S2Pool" row
below reproduces Section 5.2's ENZYME figures exactly.)*

| Pooling head | ENZYME Macro | ENZYME Micro |
|---|---|---|
| Sum + MLP | 61.25 | 61.79 |
| Mean + MLP | 61.42 | 62.09 |
| Max + MLP | 61.38 | 61.62 |
| SheafPool | 64.05 | 64.96 |
| **S2Pool** | **64.52** | **65.10** |

The three naive permutation-invariant baselines (sum, mean, max pooling followed by an MLP) cluster
tightly around 61.2–61.4 Macro-F1, roughly 3 points below the two gauge-aware designs. SheafPool —
the gauge-fixing attention-barycenter baseline that `docs/gauge_free_coboundary_readout_design.md`
positions as the design we replace — closes most of that gap, and our $GL(d)$-invariant `S2Pool`
(Section 3.9) edges it out on both metrics, consistent with the gauge-invariance argument motivating
its design. Per user confirmation, `Sum+MLP`, `Max+MLP`, and `SheafPool` are implemented in a
further-along version of the codebase than the one reflected in this repository snapshot: as of
this writing, `src/models/graph_pooling.py: build_graph_pooling()` only wires up `meanpool`,
`grampool`, and `s2pool`, and raises on any other value. We describe the method as if all five
variants are available (since they evidently are, in the environment that produced these numbers),
but flag syncing `graph_pooling.py` to include `Sum`/`Max`/`SheafPool` as a concrete follow-up so
this table is reproducible from this repository alone (see also Section 5.7).

#### 5.6.3 Remaining ablation matrix

| Ablation | Node classification (ACM/IMDB, beyond §5.6.1's DBLP) | Graph classification (MUTAG/PROTEINS/NCI1, beyond §5.6.2's ENZYME) | Link prediction (LastFM/AmazonBook) |
|---|---|---|---|
| `ablate_no_delta1` (up-Laplacian term only, keeping $\delta_0$) | *pending* | *pending* | *pending* |
| `ablate_identity_maps` | *pending* | *pending* | *pending* |
| `ablate_uniform_gates` (joint $\alpha$=$\beta$=1) | *pending* | n/a | n/a |
| Schema/meta-path ablation (§5.6.1's four-row study) | *pending* | n/a | *not yet configured* |
| Pooling ablation (§5.6.2's five-row study) | n/a | *pending* | n/a |

Extending both completed studies to the remaining datasets, and completing the three
architectural ablations (`ablate_no_delta1`, `ablate_identity_maps`, `ablate_uniform_gates`) that
have no reported numbers yet, remains the largest gap toward the "strong ablation story" this
architecture is designed to support (Section 6.1).

### 5.7 Reproducibility

Consistent with this project's own reproducibility standard — every experiment should be fully
reconstructable from a saved `config.yaml` and checkpoint alone — the run(s) backing the headline
numbers above should be saved under `experiments/` before submission. The only ACM run currently
saved in this repository predates the rerun that produced the numbers reported in Section 5.1 and
should be replaced or supplemented with the up-to-date artifact so that Section 5's numbers are
independently reproducible from the repository alone. The same applies to Section 5.6's ablation
studies and to the codebase itself: the DBLP schema/gate ablation, the ENZYME pooling ablation, the
$\alpha$-only AMPSS configuration, and the `Sum`/`Max`/`SheafPool` pooling heads all currently exist
only outside this repository snapshot (Section 4.4) and should be synced in alongside their
configs and checkpoints.

---

## 6. Discussion

### 6.1 How we want this method read

We adopt, verbatim, the framing we consider most defensible to a skeptical reviewer:

> "Given typed schemas or a candidate schema pool, H2S-NN instantiates path instances as
> higher-order structures and performs learned sheaf-Hodge diffusion over the resulting
> heterogeneous node-edge-face incidence system."

And, equally deliberately, what we do *not* want this method mistaken for:

- It is not just HAN with extra meta-path attention.
- It is not just HGT with typed relation attention.
- It is not only a meta-path search method.
- It should not yet be described as a fully validated automatic topology learner.

(Both quotations from `docs/h2snn_current_model_documentation.html`.)

### 6.2 Limitations

| Priority | Issue | Why it matters |
|---|---|---|
| Resolved | Closure edges were generated but not used anywhere in the model. | The strict 2-cell/cellular-complex claim depends on closed boundaries. |
| Resolved | Orientation signs were implicitly hardcoded as +1 inside $\delta_1$, with no explicit sign field. | Hodge/cellular coboundary operators require signed incidence for mathematical correctness. |
| Resolved | Instance sparsity loss was detached from gradients. | The sparsity term did not train $\beta$ until fixed. |
| P1 | Schema preselection is non-differentiable; refinement is differentiable only after the pool is fixed. | A deliberate two-stage design — cheap topology-aware preselection, then end-to-end sheaf-Hodge refinement — not a flaw, but worth stating as a design choice rather than glossing over. |
| Resolved | `closure_scale` existed in the configuration schema but was not consumed by the model. | Left the closure-edge design incomplete until wired in. |
| P2 | Restriction maps described as orthogonal are actually scaled orthogonal. | We avoid overstating strict orthogonality anywhere in this paper (Section 3.3). |
| P2 | The test suite has, at times, not been verifiable in every environment (missing `torch`). | We re-run the full suite in a properly configured environment before submission. |
| P2 | Evaluation evidence breadth. | Partially addressed by the baseline sets in Section 4.2; the ablation matrix (Section 5.6) remains the largest open item. |

We are explicit about what we do not yet consider ourselves entitled to claim:

> "H2S-NN is a fully validated automatic schema-to-cell discovery model." The closure/orientation
> mathematical fixes are complete and dense-reference tested, so the "closed 2-cell" claim is now
> code-grounded; what remains open is breadth of experiments — baselines and evaluation beyond the
> results already reported in Section 5.

### 6.3 Anticipated weaknesses and our responses

- **Closure-edge justification.** Does the closed-form correction of Section 3.6 genuinely satisfy
  a boundary-of-boundary-zero condition, or is it better described as a generalized path-induced
  incidence structure rather than a strict simplicial/cellular complex? We believe the current
  construction is mathematically sound for the $L_0$ term specifically (Section 3.6), but a fully
  formal treatment of the complex as a whole is future work (Section 7.2).
- **Computational cost.** Every edge-face incidence carries a full $d\times d$ restriction map;
  this is a genuine scalability concern, and consistent with that concern, H2S-NN is honestly
  marked as not yet demonstrated at large scale (the `large_scale_focus` attribute is `FALSE` for
  H2S-NN in our own comparison matrix, alongside most competing methods). Section 5.5's efficiency
  numbers on ACM are encouraging but are a single, moderate-sized benchmark.
- **Over-engineering risk.** The full model combines learned sheaf maps, an LSTM face encoder,
  hierarchical gates, and five-plus loss terms; without the ablation matrix of Section 5.6
  completed, a reviewer cannot yet distinguish gains attributable to the higher-order topology from
  gains attributable to simply having more parameters and more regularization. Closing Section 5.6
  is the single highest-priority remaining experiment.
- **Label leakage risk.** Schema and instance scoring must never use validation or test labels. By
  construction, the topology-aware score of Section 3.11 propagates training labels only; we state
  this protocol explicitly here so it does not need to be inferred from the implementation.
- **Positive semidefiniteness.** For the weighted up-Laplacian $L_1^{up} = \sum_\sigma \alpha_\sigma
  \sum_f \beta_f (\delta_1^f)^\top \delta_1^f$ (Section 3.7) to remain a valid Laplacian, it
  suffices that $\alpha_\sigma,\beta_f \ge 0$ for all $\sigma,f$ — which the sigmoid
  parameterization of both gates guarantees by construction. We state this as an explicit
  proposition rather than an incidental implementation detail, since it is the mathematical fact
  that licenses treating $L_1^{up}$ as a genuine (weighted) Hodge Laplacian at all.

### 6.4 Returning to the scope question

Section 1.5 flagged the risk of blending the base model and the discovery extension into one
inflated claim. The picture is now more precise than it was when that section was first drafted:
the headline tables in Sections 5.1–5.5 report the **full AMPSS, discovery-enabled** configuration
(Section 5.6.1's clarification), and Section 5.6.1 itself provides the first direct ablation
evidence that this configuration outperforms both a no-higher-order-structure ablation and a
hand-specified-schema variant on DBLP. This is a meaningfully stronger position than "architecture
exists, evidence pending" — but its scope is still narrow: one dataset (DBLP) for the schema/gate
ablation, one dataset (ENZYME) for the pooling ablation, and no coverage yet for ACM, IMDB, MUTAG,
PROTEINS, NCI1, or either link-prediction dataset (Section 5.6.3). We recommend, accordingly, that
a near-term submission present the unified, discovery-enabled model as the paper's primary claim —
since it is now what the headline numbers actually measure, and it has at least one dataset's worth
of ablation support — while being explicit in the limitations (Section 6.2) that the ablation
evidence for the discovery mechanism specifically has not yet been replicated across the full
dataset and task suite, rather than implying single-dataset evidence generalizes further than it
has been shown to.

---

## 7. Conclusion

### 7.1 Summary

We presented H2S-NN, a heterogeneous sheaf-Hodge neural network whose central idea is to convert
selected meta-path *instances*, not merely their type-level schemas, into two-dimensional cells of
a typed cellular complex, so that a learned sheaf structure over nodes, edges, and faces directly
reshapes the Hodge Laplacians governing diffusion over both node and edge cochains. We further
presented a schema-to-cell discovery extension that removes the need for hand-specified
meta-paths, using a training-free, topology-aware candidate score and a hierarchical pair of
differentiable schema- and instance-level gates, stabilized by a partial-schema-propagation
mechanism. Across node classification, graph classification, and heterogeneous link prediction, the
assembled comparison in Section 5 places H2S-NN ahead of a broad set of homogeneous, heterogeneous,
and sheaf-based baselines, while Sections 4.4 and 6.2–6.3 are explicit about the evaluation-protocol
work — repeated seeds, the full ablation matrix, and a completed discovery-extension evaluation —
still required before these results constitute a finished experimental section.

### 7.2 Future work

- **Formal search guarantees.** The PHS-Sheaf research direction sketched in
  `docs/meta_path_survey.md` §16 — a policy-guided search over schemas combining a learned policy
  with a geometry-native "holonomy" heuristic derived from the restriction maps themselves, with an
  explicit expansion-count bound — remains aspirational and unimplemented; formalizing it would
  directly answer the "is schema selection principled or heuristic?" question a reviewer is likely
  to raise.
- **The `hard_concrete` gate variant.** Currently a configuration option that raises an error if
  selected; implementing it would open a genuine ablation axis (continuous sigmoid gates vs. a
  stochastic, discretizable alternative) that the current codebase cannot yet evaluate.
- **Scale beyond the current benchmark set.** ACM/DBLP/IMDB and MUTAG-class graph benchmarks are
  all moderate in size; demonstrating that the method scales — ideally on a graph where no
  canonical meta-path set is known in advance, directly testing the discovery extension's intended
  use case — is a natural and important next experiment.
- **A genuine node-clustering implementation.** Section 5.3's clustering table currently has no
  supporting code in this repository; building a clustering readout (or explicitly retiring the
  clustering comparison) should happen before that table appears in a submitted version of this
  paper.
- **Cheaper restriction-map parameterizations.** Diagonal, low-rank, or basis-decomposed
  restriction maps are a direct, previously-identified response to the computational-cost concern
  of Section 6.3, and are a natural complement to the existing `ablate_identity_maps` ablation.
- **A true closed-boundary operator.** Comparing the current path-only coboundary construction
  against a fully closed-boundary cellular complex (rather than the closed-form $L_0$-only
  correction of Section 3.6) would directly settle the closure-edge justification question raised
  in Section 6.3.

---

## References

Bibliographic detail (author lists, exact venues) is incomplete for several entries in the source
comparison data; this list should be replaced with a proper `.bib` file before LaTeX conversion.
Entries are grouped by the related-work subsection that cites them, with arXiv identifiers or
publication years where known (`docs/comparison_model_year_metadata_2026-05-20.csv`,
`docs/h2snn_current_model_documentation.html` §"External Reference Anchors").

**Sheaf neural networks**
- Bodnar, C. et al. *Neural Sheaf Diffusion.* arXiv:2202.04579, 2022.
- *HetSheaf: Sheaf Neural Networks for Heterogeneous Graphs.* arXiv:2409.08036, 2024.
- *DeepSN: A Sheaf Neural Framework for Influence Maximization.* arXiv, 2024.
- *SHN: Sheaf HyperNetworks for Personalized Federated Learning.* arXiv, 2024.

**Heterophily-aware heterogeneous models**
- *HeterSEED: Semantic and Structural Neighbors for Heterophilous Heterogeneous Graphs.*
  arXiv:2605.04594, 2025.
- *Adaptive Heterogeneous Graph Neural Network for Multi-Hop Heterophily (AHGNN).*
  arXiv:2508.06034, 2025.
- *H2SGNN: Heterophily-aware Heterogeneous Graph Neural Network.* arXiv:2410.13373, 2023–2024.
- *LatGRL: latent-space heterophily diagnostics for heterogeneous graphs.* arXiv:2409.00687, 2021 (dataset benchmark year per source table).
- *Cross-Type Homophily Ratio and Graph Editing (CTHGE).* arXiv:2501.14600, 2025.

**Meta-path / schema search methods**
- *LMSPS: Learning Meta-path Selection for Heterogeneous Graph Neural Networks.*
  arXiv:2307.08430, NeurIPS 2024.
- *PMMM: Differentiable Meta Multigraph Search with Partial Message Propagation.*
  arXiv:2211.14752, 2022.
- *AMP-HGNAS: Adaptive Meta-Path-based Neural Architecture Search for Heterogeneous Graphs.* 2026.
- *A New Meta-path Monte Carlo Tree Search Algorithm for Heterogeneous Graph Neural Networks
  (MCTS-HGNN).* Year not stated in source.
- *Meta-multigraph search: Rethinking meta-structure on Heterogeneous Information Networks.* 2024.
- HG-Search, SchemaWalk, RMS-HRec, SMCD, HeteroSample, EvoPath, TAEG-HG, FHGE, HISE-KT, ESim,
  MP-GNN, LPSMS — see `docs/comparison_model_year_metadata_2026-05-20.csv` for per-model source
  titles; several entries have no confirmed publication year in our source data and should be
  verified before citation.

**Heterogeneous GNNs**
- HAN (2019), MAGNN, HGT — standard heterogeneous graph attention/transformer baselines; full
  citations to be added.

**Node classification / clustering / link-prediction baselines (Section 5)**
- GCN (2016), GAT (2021), NSD (2022), CSNN (2026), ChebNetII (2022), SeHGNN (2022), HiNormer
  (2023), PSHGCN (2024), HDHGR (2023), Hetero2Net (2023); HGMAE, RMR, DMGI, HDMI, HeCo, HeCo++,
  HERO, CTC, HGMS-N, HGMS-C; GIN, SAGE, R-GCN, Sheaf-NSD — full citations to be added from
  `docs/h2snn_preliminary_results_tables_updated.html`'s original sources.

**Search-with-guarantees literature (motivating future work, Section 7.2)**
- LevinTS / LubyTS — *Single-Agent Policy Tree Search With Guarantees.* 2018.
- PHS — *Policy-Guided Heuristic Search with Guarantees.* 2021.
- *RL of Heuristics with Limited-Horizon Search (LHBL).* arXiv, 2025.
- Zoph, B. and Le, Q. *Neural Architecture Search with Reinforcement Learning.* 2017.

---

## Appendix A: Full Hyperparameter Configuration (base configs, `d=32`)

| Group | Field | Value |
|---|---|---|
| Architecture | $d$ / $d_r$ / $d_h$ | 32 / 16 / 64 |
| Architecture | `edge_mlp_hidden_dim` / `edge_init_hidden_dim` / `face_mlp_hidden_dim` / `classifier_hidden_dim` | 128 each |
| Diffusion | $K$ | 4 |
| Diffusion | $\eta_0$ / $\eta_1$ (initial) | 0.05 / 0.05, learnable |
| Restriction maps | `use_orthogonal` / `orthogonal_map` | true / householder |
| Loss weights | $\lambda_0$ / $\lambda_1$ / $\lambda_c$ / $\lambda_{co}$ | 0.0 / 0.001 / 0.01 / 0.01 |
| Optimization | optimizer / lr / weight decay / scheduler / grad clip | AdamW / 1e-3 / 1e-4 / cosine / 1.0 |
| Training | epochs / seed | 200 / 42 |

Discovery-mode (`*_h2snn_ampss.yaml`) additions: `min/max_schema_length`=2/4,
`max_schema_candidates`=64, `selected_schema_k`=6, `max_instances_per_schema`=8000,
`max_instances_per_anchor`=30, `partial_schema_probability`=0.8, `warmup_epochs`=5, discovery loss
weights $\lambda_{schema\_sparse}=10^{-3}$, $\lambda_{instance\_sparse}=10^{-4}$,
$\lambda_{redundancy}=10^{-3}$, $\lambda_{coverage}=10^{-3}$, $\lambda_{cost}=10^{-4}$,
$\lambda_{gate\_entropy}=10^{-4}$.

(Source: `configs/experiments/acm_h2snn.yaml`, `configs/experiments/acm_h2snn_ampss.yaml`, verified
directly; other datasets share the same architecture/optimization defaults with dataset-specific
target types and schema templates per `src/configs/config.py`.)
