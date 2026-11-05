# H2S-NN vs. Existing Methods — Explanation for Each Comparison Row

> **Historical comparison note (superseded for AMPSS implementation details).** Use
> `docs/h2snn_current_model_documentation.html` for the current
> `schema_conditioned_v1` protocol. Current AMPSS uses task-only schema selection and
> schema-conditioned instance weights without auxiliary discovery losses.

**Audience:** A professor who is familiar with graph neural networks (GNNs) but has not read
the H2S-NN project before.

**Purpose:** To explain why each row in `h2snn_method_comparison.csv` exists, what problem it
captures, and why current methods in the literature fail to solve it.

---

## Background — What is the problem we are solving?

In many real-world graphs, nodes are not all the same type. An academic citation network has
**Paper** nodes, **Author** nodes, and **Venue** nodes. The edges are also typed: an author
*writes* a paper, a paper *cites* another paper, a paper is *published at* a venue. These graphs
are called **Heterogeneous Information Networks (HINs)**.

To learn useful node embeddings from a HIN, researchers introduced the concept of a
**meta-path**: a typed sequence of node and edge types that defines a semantic route through
the graph. For example:

- **APA** (Author → Paper → Author) captures co-authorship
- **APCPA** (Author → Paper → Category → Paper → Author) captures shared research topic

Existing heterogeneous GNNs use meta-paths in one of two ways:

1. A human expert writes down the meta-paths before training. The model uses them as
   fixed aggregation channels (like HAN, MAGNN).
2. The model searches for useful meta-paths automatically from a predefined candidate list
   (like LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM).

**H2S-NN goes further than both approaches.** It not only discovers which typed path schemas
are useful, but also lifts those discovered paths into actual higher-order topological objects
(2-cells / faces) that reshape the diffusion operator itself. This is something no prior method
does.

The comparison table captures the capabilities needed to do this correctly. Each row below
explains one capability, why it matters, and what goes wrong in existing methods that lack it.

---

## Row-by-Row Explanation

---

### Row 1 — Works without predefined meta-paths

**What it means:**
The model can accept a raw heterogeneous graph — with node types, edge types, and features —
and produce useful node embeddings without a human first writing down specific meta-path
templates such as "use APA, APCPA, and PAP."

**Who has it and who does not:**
- **HAN**: No. It requires a human to write every meta-path before training begins.
- **HGT**: Yes. It ignores meta-paths entirely and operates directly on observed typed edges.
- **HetSheaf**: Yes. It operates on the typed graph directly without meta-path templates.
- **LMSPS**: Yes. It explicitly states *"without prior, our search space contains all
  target-node-related meta-paths"* — it enumerates all possible paths up to a max hop from
  the type graph automatically.
- **AMP-HGNAS**: Yes. It adaptively generates candidates from the type graph and searches
  among them without requiring manually specified paths.
- **MCTS-HGNN**: Yes. It builds a full meta-path tree from the type graph automatically (all
  paths such as {PP, PA, PI, PPP, PPA, ...}) and uses MCTS to search that tree.
- **PMMM**: Yes. The paper explicitly states *"our PMMM does not require specific prior
  knowledge"* — it uses all relation types from the HIN schema as its search space.
- **H2S-NN**: Yes.

**What this row isolates:**
Only HAN (classic manual-path methods) gets No here. This row correctly captures that the
automatic search methods (LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM) have already solved the
"no manual path required" problem. The remaining rows explain *why they still fall short*.

---

### Row 2 — Automatically discovers schema templates from the type graph

**What it means:**
The model inspects the **schema graph** (a small graph that shows which node types are
connected by which relation types) and proposes candidate typed-path schemas on its own.
No human-provided candidate list is required.

**Why existing methods fail:**
LMSPS, AMP-HGNAS, MCTS-HGNN, and PMMM all perform automatic *search* — but they search
inside a predefined candidate pool. A human still decides which relation types are legal, what
the maximum path length is, and which starting and ending node types are allowed. The search
is automatic only within those human-set boundaries.

This is a meaningful limitation. On datasets with large or unfamiliar schema graphs (many
node types, many relation types), constructing even the candidate pool becomes a manual effort.

**Why it matters:**
True automatic discovery means the model can enumerate, propose, and evaluate typed schemas
entirely from the type graph structure — just as a human expert would, but without requiring
the human. This is especially important for datasets where no "obvious" canonical paths exist.

---

### Row 3 — Constructs higher-order topology (2-cells / faces as topological objects)

**What it means:**
When a meta-path instance is discovered (e.g., a specific Author → Paper → Author triple in
the data), it is treated as a **face** (a 2-cell) in a cellular complex — a genuine topological
object with its own signal space. These faces become part of the mathematical structure that
defines diffusion, not just a list of neighbors to aggregate.

**Why existing methods fail:**
Every search method in the comparison table (LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM) and every
classic method (HAN, MAGNN) uses selected meta-paths as **aggregation channels**. The
meta-path is followed to collect a neighborhood, the neighborhood is averaged or attention-
weighted, and then the meta-path is discarded. The higher-order relational pattern is
*thrown away* after serving as a routing rule.

The difference is like the gap between:
- using a triangle to define "collect three neighbors and average them" (existing methods), and
- treating the triangle as a surface with its own signal that participates in diffusion geometry
  (H2S-NN).

**Why it matters:**
By lifting paths into faces, H2S-NN can model **face-level inconsistency** — situations where
the signals around a triangular pattern conflict with each other. Standard GNNs that treat
meta-paths as channels have no way to represent or resolve this.

---

### Row 4 — Hodge diffusion over node AND edge cochains (uses both δ₀ and δ₁)

**What it means:**
Standard sheaf/graph diffusion operates over **node signals** using the node Laplacian
`L₀ = δ₀ᵀδ₀`. H2S-NN also maintains **edge signals** and uses the full edge Hodge Laplacian
`L₁ = δ₀δ₀ᵀ + δ₁ᵀδ₁`, where `δ₁` is the coboundary operator that connects edge signals to
face signals.

In plain terms: information flows not just between nodes, but also between edges — and the
faces (2-cells) from Row 3 are what makes the `δ₁` term non-trivial.

**Why existing methods fail:**
Even HetSheaf, which is the closest prior work and does use sheaf restriction maps, works only
on the 1-skeleton (the observed graph edges). It computes `L₀` but never defines `δ₁` because
it has no 2-cells. All meta-path search methods (LMSPS, AMP-HGNAS, etc.) are standard HGNNs
with no sheaf or Hodge structure at all.

**Why it matters:**
Without `δ₁`, the model cannot represent how a triangular face pulls its surrounding edges into
agreement. This is the core mathematical mechanism by which H2S-NN propagates information
through higher-order structure — and it is missing from every comparison method.

---

### Row 5 — Orthogonal sheaf restriction maps per relation type

**What it means:**
Each directed edge `(u → v)` of relation type `r` has two learnable `d × d` **orthogonal**
matrices — one for the source node `u` and one for the target node `v`. These matrices rotate
the local coordinate frames of the two endpoint nodes before comparing their signals. This is
the **sheaf structure**: the model learns how meaning transforms as you cross a typed relation.

**Why existing methods fail:**
HAN, HGT, and all meta-path search methods use standard linear transformations or attention
weights that do not enforce any geometric constraint on how signals relate across edges.
They cannot represent the fact that an Author node and a Venue node live in semantically
different spaces — there is no learned rotation between them.

**Why it matters:**
The orthogonality constraint keeps the representation geometrically well-behaved. It prevents
over-smoothing (all nodes collapsing to the same embedding) by ensuring that each relation type
contributes a rotation, not a contraction. HetSheaf also has this property, and H2S-NN
inherits it as a foundation while adding automatic topology construction on top.

---

### Row 6 — Explicitly designed for heterophilic graphs

**What it means:**
A **heterophilic graph** is one where connected nodes often belong to *different* classes or
have very different features. For example, in a social network where fraudsters connect to
honest users, or in a web graph where pages in different categories link to each other,
standard GNN smoothing makes predictions worse.

H2S-NN and HetSheaf are designed to represent **disagreement** across edges (via the
coboundary operator and restriction maps) rather than forcing agreement through averaging.

**Why existing methods fail:**
HAN, HGT, LMSPS, AMP-HGNAS, MCTS-HGNN, and PMMM all ultimately aggregate neighbor features
using some form of weighted average. This is exactly the operation that fails under heterophily
— it pulls a node's embedding toward its neighbors, which are in a different class.

None of the meta-path search methods have any mechanism for representing or resolving
disagreement across edges. They improve *which paths* are followed but not *how disagreement
across those paths is handled*.

**Why it matters:**
The datasets where meta-path learning is most needed (large HINs with diverse relation types)
are often the same datasets where heterophily is a problem. A method that ignores heterophily
will systematically underperform on those datasets.

---

### Row 7 — Handles heterogeneous graphs (multiple typed nodes and edges)

**What it means:**
The model accepts graphs with more than one node type and more than one edge/relation type,
and uses type-specific parameters or transformations to distinguish between them.

**Why all existing methods have this:**
This is the baseline capability for every method in the comparison. A homogeneous GNN (like
plain GCN or GraphSAGE) would get "No" here, but all the compared methods are specifically
designed for heterogeneous graphs. This row is included to confirm that H2S-NN meets the same
basic requirement.

---

### Row 8 — Schema-level importance weights (α)

**What it means:**
The model learns a scalar importance weight `α_σ` for each schema type `σ`. Schemas that are
more informative for the downstream task receive higher weight. Uninformative schemas are
softly suppressed without a hard binary keep/drop decision.

**Why some methods have it and others do not:**
- **HAN** has semantic-level attention that weights each meta-path view. Yes.
- **LMSPS, AMP-HGNAS, PMMM** learn or estimate schema importance during search. Yes.
- **HGT** does not use schemas at all. No.
- **HetSheaf** works at the relation-type level but does not learn schema-level weights. No.
- **MCTS-HGNN** uses discrete tree search scores, not soft learnable weights. No.

**Why it matters:**
Without schema-level weights, the model either treats all schemas equally (wrong: some are much
more informative) or makes a hard discrete selection (expensive and not differentiable). Soft
learned weights allow the model to smoothly adjust schema importance during training.

---

### Row 9 — Instance-level weights that directly shape the Hodge/Laplacian operator (β)

**What it means:**
Every individual path instance `f` (a concrete sequence of nodes matching a schema, e.g. the
specific triple Author₁ → Paper₇ → Author₂) receives a learnable weight `β_f`.
This weight **enters the Hodge operator itself**:

```
L₁^up = Σ_σ α_σ · Σ_{f ∈ I(σ)} β_f · (δ₁^f)ᵀ δ₁^f
```

The topology of the diffusion operator is literally shaped by individual instance importance.

**Why all existing methods fail:**
No prior method has this property. Even HAN, which has node-level attention within each
meta-path view, uses that attention only to weight a sum during aggregation — it does not
change any operator or topological structure. In LMSPS, AMP-HGNAS, MCTS-HGNN, and PMMM,
instance-level filtering is either absent or binary (keep/drop). None of them have a learned
weight per face instance that reshapes a mathematical operator.

**Why it matters:**
Among all path instances that match the same schema, some are highly informative (a rare but
meaningful connection between two specific authors) and others are noise (a generic connection
through a highly-connected hub node). Instance-level weights allow the model to learn this
distinction and let it propagate into the geometry of diffusion.

---

### Row 10 — Removed from comparison table (correction)

This row was removed after verifying the actual papers. The original claim that search methods
(LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM) require a predefined schema pool or domain expert
knowledge was **incorrect**.

- **LMSPS** explicitly states *"without prior"* and auto-generates all target-node-related
  paths from the type graph.
- **PMMM** explicitly states *"does not require specific prior knowledge."*
- **MCTS-HGNN** builds its meta-path tree from the full type graph automatically.
- **AMP-HGNAS** searches adaptively from auto-generated candidates.

The real gap between H2S-NN and these methods is not about removing the need for expert
input — they have already achieved that. The real gap is in **what the selected schemas
become** (aggregation channels vs. 2-cells) and **how they affect the model** (weighted sum
vs. reshaping L₁). Those distinctions are captured in Rows 3, 4, 9, 12, and 13.

---

### Row 11 — Penalizes redundant or overlapping schemas during selection

**What it means:**
When multiple candidate schemas describe nearly the same set of node pairs or carry the same
semantic signal (e.g., two slightly different paths between the same endpoint types), the
selection process should discourage choosing both. A diversity or redundancy penalty is applied
during schema selection.

**Why most methods fail:**
HAN, HGT, HetSheaf, LMSPS, MCTS-HGNN, and PMMM do not explicitly penalize redundancy.
AMP-HGNAS (2026) is the only existing method in the comparison that explicitly addresses
dependency and redundancy among meta-paths — and this is one of its main contributions.

**Why it matters:**
Selecting several near-duplicate schemas wastes computation (more faces to instantiate and
process) and can cause gradient interference during training (highly correlated paths all push
the loss in the same direction, reducing effective update diversity). Redundancy control makes
the selected schema set compact, interpretable, and computationally efficient.

---

### Row 12 — Supports refinement of suboptimal user-provided schemas

**What it means:**
If a user supplies meta-paths as a starting point — but those paths are imperfect, incomplete,
or suboptimal for the specific task — the model can:
1. prune the weak provided schemas,
2. expand to nearby alternatives on the schema graph, and
3. reweight the surviving schemas.

The user-provided schemas are treated as a warm-start hint, not as fixed ground truth.

**Why all existing methods fail:**
In HAN and MAGNN, user-provided meta-paths are fixed inputs that cannot be modified by the
model. In the search methods (LMSPS, AMP-HGNAS, etc.), the search starts from a
human-defined candidate pool — but if the user provides a specific set of "good" paths, the
model has no mechanism to test whether nearby alternatives might be better.

HGT and HetSheaf bypass the problem by never using user-provided meta-paths at all, which is
a different kind of solution.

**Why it matters:**
In practice, domain experts often provide reasonable but not optimal meta-paths. A model that
can accept those paths and improve on them — rather than either accepting them rigidly or
ignoring them entirely — is far more useful in a real research workflow.

---

### Row 13 — Topology-aware scoring of candidate schemas

**What it means:**
When evaluating whether a candidate schema is worth selecting, H2S-NN uses scoring signals
that reflect the **topological utility** of the induced faces — not just downstream task
accuracy. For example:
- Does the schema produce faces that reduce **edge inconsistency** in the Hodge sense?
- Does the schema cover anchor nodes that are underserved by existing selected schemas?
- Does the schema produce a `δ₁` contribution that is informative for diffusion?

**Why all existing methods fail:**
Every meta-path search method evaluates schema quality by training or partially training the
downstream model and measuring validation accuracy. This has two problems:

1. **Expensive:** every candidate evaluation requires a model training run.
2. **Topologically blind:** a schema that improves the diffusion geometry (by correcting
   Hodge inconsistency) but does not immediately improve accuracy may be incorrectly
   discarded.

No existing method has a scoring function that looks inside the Hodge operator to assess
whether a schema is topologically useful.

**Why it matters:**
Topology-aware scoring allows faster, cheaper, and more principled schema evaluation. It also
means the model can discover schemas that are topologically valuable even before their
downstream accuracy benefit is visible — reducing wasted search effort.

---

### Row 14 — End-to-end differentiable training

**What it means:**
All components of the model — schema gates `α`, restriction maps, face weights `β`, and the
downstream task classifier — are optimized jointly by backpropagation in a single training
loop. There is no separate search phase followed by a separate retrain phase.

**Why some methods fail:**
- **LMSPS** uses progressive non-differentiable sampling: it trains on a subset of schemas,
  evaluates validation accuracy, shrinks the pool, and repeats. Each round is a separate
  training run.
- **AMP-HGNAS** uses a two-stage architecture search: search phase first, then retrain on the
  selected architecture.
- **MCTS-HGNN** is inherently discrete: the tree search selects paths and then a separate
  model is trained on them. Gradients do not flow through the search.

**Why it matters:**
Separate search-then-retrain pipelines suffer from **search-evaluation mismatch**: the
architecture or schema set that looks best during search may not remain optimal after full
retraining with the downstream model. End-to-end training avoids this by letting the search
weights and the model weights co-adapt in real time. It also reduces total compute: one
training run instead of many.

---

## Summary: The Three Core Gaps — Explained Simply

First, an important correction from reading the actual papers:
**LMSPS, AMP-HGNAS, MCTS-HGNN, and PMMM already work without predefined meta-paths.**
They all automatically generate every possible path from the graph's type structure and then
search for the useful ones. So "no manual paths required" is NOT what makes H2S-NN special.
The three real gaps are below.

---

### Gap 1 — Existing search methods find good paths but then throw the path away

**Simple analogy:**
Imagine you are studying a city and you discover that the route
"Home → School → Library → Home" is a useful pattern for predicting which neighborhoods
are educational. An existing search method would find this route, then use it to collect
information: it walks the route, gathers the people at the end, and averages their opinions.
After that, the triangular route itself is forgotten. It plays no further role.

H2S-NN does something fundamentally different. It says: **the triangular route itself is a
meaningful object.** It keeps the triangle as a structure in the model, studies whether the
three corners of the triangle agree or disagree with each other, and uses that agreement or
disagreement to adjust how information spreads everywhere in the graph.

**In graph terms:**
Every existing search method (LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM) discovers meta-paths and
then feeds them into the same standard aggregation pipeline that HAN uses — collect neighbors,
take a weighted average, done. The path is just a routing rule.

H2S-NN treats each discovered path instance (e.g., Author₁ → Paper₇ → Author₂) as a
**triangle (face)** that lives inside the model permanently. These triangles form part of the
mathematical operator that controls how all information diffuses. The selected paths do not
just deliver neighbors — they restructure the entire diffusion process.

**Why it matters:**
If you only collect neighbors via a path and then average them, you lose all information about
whether the path itself makes sense. For example, two authors co-authored a paper, but their
research areas are completely different — the triangle is "inconsistent." H2S-NN can detect
and represent this inconsistency. Existing methods cannot — they just average and move on.

---

### Gap 2 — Existing sheaf methods learn on a fixed graph; H2S-NN learns the graph structure itself

**Simple analogy:**
HetSheaf is like a very good furniture arranger. You give it a room (the graph), and it
figures out the best way to arrange the furniture (the sheaf geometry — how node signals
relate across each edge). It does an excellent job with whatever room you give it.

H2S-NN is like an architect who first decides which rooms to build, then arranges the
furniture. It does not just accept the graph as given. It asks: **should there be extra
connections — triangular relationships — that are not in the original graph?** It discovers
which higher-order patterns deserve to exist in the model, builds them in, and then does the
geometry on top of that richer structure.

**In graph terms:**
HetSheaf works on the edges that already exist in the data. Its diffusion only spreads
information along those observed edges. It has no concept of a path instance being a
triangle — it sees only nodes and edges.

H2S-NN adds a second level: **triangles (faces)**. When it finds that Author₁ → Paper₇ →
Author₂ is an important pattern, it creates a triangular structure connecting these three
nodes. Information can now flow around the triangle — not just from node to node along a
single edge. This gives the model access to patterns that are completely invisible to any
method that only looks at the original edges.

**Why it matters:**
Many important relationships in real graphs are not direct edges — they are patterns across
three or more nodes. A model that can only see direct edges will miss these patterns, no
matter how good its geometry is on those edges.

---

### Gap 3 — Existing methods weight paths as importance scores; H2S-NN uses them to reshape the model itself

**Simple analogy:**
Imagine you have five roads leading into a city, and you want to control traffic. One
approach is to put a traffic controller at each road who decides how many cars to let through
(this is what existing methods do — importance weights control how much each path contributes
to the final aggregation). The roads themselves stay the same.

H2S-NN's approach is different: it assigns a weight to each triangular block in the city's
layout, and that weight changes **which streets even exist** in the model. A triangle with
high weight becomes a real, permanent structural feature that reshapes all traffic patterns
everywhere. A triangle with low weight quietly fades out.

**In graph terms:**
In HAN and in the search methods, importance weights (α for schemas, attention for neighbors)
determine how much each path or neighbor contributes to a weighted average. That is all they
do — change the coefficients in a sum.

In H2S-NN, each individual path instance has its own importance weight (β), and that weight
enters the diffusion operator directly. A high-weight triangle pulls the signals of its three
corners toward consistency. A low-weight triangle has little effect. The model is not just
weighting a sum — it is **shaping the geometry that all information flows through.**

**Why it matters:**
Not all instances of the same path type are equally useful. The connection
Author₁ → Paper₇ → Author₂ might be critically important (two major researchers with a
landmark joint paper), while Author₃ → Paper₁₂ → Author₄ might be noise (two people who
accidentally share a workshop paper). Existing methods have no way to distinguish these at
the structural level — they either include all instances equally or drop them all. H2S-NN
learns a different importance for every individual triangle and lets that importance directly
shape how the model behaves.

---

### The one-paragraph version for a professor

The automatic search methods (LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM) have already solved the
"which paths are useful?" problem — they find good paths automatically without any manual
input. But once they find a path, they use it the same way HAN does: follow it to collect
neighbors, aggregate those neighbors, and discard the path. H2S-NN asks a different question:
**what if the path itself — not just the neighbors at the end of it — is a meaningful
structural object?** H2S-NN converts each discovered path instance into a triangle that
permanently lives in the model and reshapes how information diffuses. It is also the only
method that learns how strongly each individual triangle should influence the model, and the
only method that can detect when the three corners of a triangle are in conflict with each
other. None of those properties exist in any prior method.
