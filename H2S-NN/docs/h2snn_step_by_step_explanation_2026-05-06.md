# H2S-NN Step-by-Step Explanation

Date: 2026-05-06

## 1. Purpose of this note

This note explains our H2S-NN pipeline in plain research language, step by step, as one coherent story.

It separates two things that are easy to mix together:

- the **current H2S-NN idea**, where meta-path templates are given manually and their instances become faces in a typed 2D cellular complex;
- the **proposed extension**, where H2S-NN learns which typed schemas should become faces even when good meta-paths are not predefined.

The main goal is to answer a practical question:

> What exactly are we building, what exactly is being learned, and what would the full H2S-NN pipeline do from input graph to final prediction?

---

## 2. One-sentence summary

H2S-NN takes a heterogeneous graph, turns selected meta-path instances into actual 2-cells of a typed cellular complex, learns sheaf restriction maps on nodes, edges, and faces, builds Hodge operators from those maps, diffuses signals on the complex, and predicts labels from the resulting topology-aware representations.

---

## 3. What problem H2S-NN is trying to solve

We start from a heterogeneous information network:

- multiple node types;
- multiple relation types;
- often strong heterophily, meaning connected nodes may not share labels or semantics.

This creates two problems for ordinary GNNs:

- simple neighborhood aggregation can over-smooth dissimilar nodes;
- ordinary message passing mainly lives on the graph 1-skeleton, so it does not explicitly model higher-order typed relational patterns.

Most heterogeneous GNN methods handle higher-order semantics by using meta-paths as extra graph views or aggregation channels.

H2S-NN takes a stronger position:

- a meta-path instance is not just a view;
- it is treated as a higher-order topological object;
- that object participates in the operator that drives diffusion.

That is the core idea.

---

## 4. The main conceptual shift

In a standard meta-path HGNN:

- a meta-path such as `Author -> Paper -> Author` is used to define a semantic neighborhood;
- the model aggregates messages over that view;
- the meta-path is a semantic channel.

In H2S-NN:

- a concrete instance such as `a1 -> p7 -> a2` becomes a face;
- faces participate in `delta_1`;
- `delta_1` contributes to `L_1`;
- `L_1` changes the Hodge diffusion geometry on edge signals.

So the shift is:

> meta-path as semantic view -> meta-path instance as topological cell

That is why the method is not just another meta-path aggregator.

---

## 5. Objects and notation

We begin with a heterogeneous graph:

`G = (V, E, phi, psi, X, Y)`

where:

- `V` is the node set;
- `E` is the directed typed edge set;
- `phi(v)` is the node type of node `v`;
- `psi(e)` is the relation type of edge `e`;
- `X` are node features;
- `Y` are labels on the target node type.

We build a typed 2D cellular complex:

`X = (X^0, X^1, X^2)`

where:

- `X^0` are nodes;
- `X^1` are directed edges;
- `X^2` are faces built from selected meta-path instances.

We then define:

- `delta_0`: node-to-edge coboundary;
- `delta_1`: edge-to-face coboundary;
- `L_0 = delta_0^T delta_0`;
- `L_1 = delta_0 delta_0^T + delta_1^T delta_1`.

Signals live on different cell dimensions:

- node signals on `X^0`;
- edge signals on `X^1`;
- optionally face signals on `X^2`.

For the first implementation, a common hidden dimension `d` is the simplest choice.

---

## 6. Current H2S-NN, step by step

This section describes the current H2S-NN story where meta-path templates are provided manually.

### Step 0. Choose the prediction task

The usual task is node classification on one target node type.

Examples:

- in ACM, predict labels for `Paper`;
- in DBLP, predict labels for `Author`;
- in IMDB, predict labels for `Movie`.

This matters because:

- not all node types need labels;
- the readout is focused on the target type even though the whole heterogeneous graph is used.

### Step 1. Start from the raw heterogeneous graph

We keep the original graph structure:

- all node types remain;
- all relation types remain;
- edge direction is important.

Unlike some meta-path methods, we do not project everything into a single homogeneous graph and throw away the original typed edges.

### Step 2. Compute initial node embeddings

Each node gets an initial embedding.

In the research plan, this role is played by STAGE or another heterogeneity-aware encoder.

Operationally:

- each node feature vector is mapped into a common latent space;
- nodes of different types can use type-specific encoders or a shared encoder conditioned on type.

This produces:

`h_v^(0) in R^d`

for every node `v`.

This stage is important because later sheaf restriction maps are learned from these embeddings.

### Step 3. Choose meta-path templates

In the current H2S-NN design, meta-path templates are manually given.

Examples:

- ACM: `PAP`, `PSP`
- DBLP: `APA`, `APTPA`
- IMDB: `MAM`, `MDM`

At this stage, the model is not yet discovering schemas automatically.

It relies on domain knowledge or prior literature to say:

- these are the semantic path types that seem meaningful;
- we will instantiate them and turn them into faces.

So in the current version, the higher-order candidate set is still externally specified.

### Step 4. Enumerate concrete meta-path instances

For each template, we enumerate or sample concrete path instances in the graph.

Example:

- template `APA`
- instance `a1 -> p7 -> a2`

Each instance is ordered, because order matters for:

- path direction;
- edge sequence;
- position-aware face map learning.

For long templates, full enumeration may be too expensive, so we use budgeted sampling such as:

- at most `K_face` instances per anchor node;
- a small allowed path length such as `2`, `3`, or `4`.

### Step 5. Turn meta-path instances into faces

This is the most important modeling step.

Each selected meta-path instance becomes a 2-cell:

- nodes are 0-cells;
- edges are 1-cells;
- ordered meta-path instances are 2-cells.

This means H2S-NN is not just using path semantics for attention or pooling.

It is explicitly changing the topology of the space on which the model operates.

### Step 6. Close each face topologically

A path by itself is open, but a 2-cell needs a closed boundary.

So for a path instance:

`v0 -> v1 -> ... -> vl`

we add a synthetic closure edge:

`vl -> v0`

This gives the face a closed boundary cycle.

The closure edge is a topological device:

- it makes the face mathematically valid as a 2-cell;
- its diffusion contribution is designed to be negligible or zero;
- it fixes the open-path problem without changing the semantics much.

This correction is important for making the cellular construction mathematically sound.

### Step 7. Build the typed cellular complex

Now we have:

- `X^0 = V`
- `X^1 = original directed edges + closure edges`
- `X^2 = meta-path face instances`

We also store incidence information:

- which nodes belong to each edge;
- which edges lie on the boundary of each face;
- which faces touch each edge.

These incidence structures are needed later to build `delta_0` and `delta_1`.

### Step 8. Define stalks and feature spaces

Each cell gets a stalk, usually `R^d` in the first implementation.

So we have:

- node stalks `F(v) = R^d`
- edge stalks `F(e) = R^d`
- face stalks `F(f) = R^d`

Conceptually, this means:

- each node carries a `d`-dimensional signal;
- each edge carries a `d`-dimensional signal;
- each face carries a `d`-dimensional signal.

The power of the sheaf comes from the fact that these signals are connected by learned linear maps, not by scalar adjacency weights alone.

### Step 9. Learn node-to-edge restriction maps

For every directed edge `e = (u -> v)` with relation type `r`, we learn two maps:

- a source restriction map from node `u` into edge `e`;
- a target restriction map from node `v` into edge `e`.

In practice:

- the map is produced by an MLP;
- the MLP sees embeddings of the two endpoint nodes plus a relation embedding;
- the output is reshaped into a `d x d` matrix.

So H2S-NN does not assume one universal edge operator.

Instead, it learns relation-aware local geometry.

These maps say:

- how node information should be interpreted when entering an edge stalk;
- how source and target contribute differently.

### Step 10. Learn edge-to-face restriction maps

For every edge that lies on the boundary of a face, we learn an edge-to-face map.

This is where ordered path structure matters.

The corrected algorithm uses an LSTM-style sequential encoder:

- read the path as an ordered sequence;
- build a token for each boundary edge using local node and relation information;
- feed the token sequence through an LSTM;
- map each hidden state to a `d x d` edge-to-face restriction matrix.

This gives position-aware face geometry:

- the first edge in a path is not treated the same as the last edge;
- the same relation can play different roles depending on where it appears in the face boundary.

This is one of the places where H2S-NN is genuinely richer than graph-only sheaf models.

### Step 11. Build the node-to-edge coboundary `delta_0`

Once the node-to-edge restriction maps exist, we define:

`(delta_0 x)_e = F_tgt(v -> e) x_v - F_src(u -> e) x_u`

for edge `e = (u -> v)`.

Interpretation:

- if the two endpoint node signals agree after being transported into the edge stalk, the edge discrepancy is small;
- if they disagree, the edge discrepancy is large.

So `delta_0` measures edge-level inconsistency induced by node embeddings under the learned local geometry.

### Step 12. Build the edge-to-face coboundary `delta_1`

For each face `f`, `delta_1` aggregates the boundary edge signals into the face stalk:

`(delta_1 s)_f = sum_{e in boundary(f)} eta_{e,f} F_{e -> f} s_e`

where `eta_{e,f}` is the incidence sign determined by orientation.

Interpretation:

- if the boundary edge signals fit together coherently under the face geometry, the face discrepancy is small;
- if they do not fit, the face discrepancy is large.

So `delta_1` measures whether an edge signal is consistent with the higher-order structure encoded by faces.

### Step 13. Build the Hodge Laplacians

Now define:

- `L_0 = delta_0^T delta_0`
- `L_1 = delta_0 delta_0^T + delta_1^T delta_1`

Interpretation:

- `L_0` acts on node signals;
- `L_1` acts on edge signals;
- the `delta_0 delta_0^T` part ties edge signals back to node-induced discrepancies;
- the `delta_1^T delta_1` part couples edges that co-occur in faces.

This last term is the key higher-order term.

Two edges that share a face can influence each other even if ordinary graph adjacency would not connect them in the same way.

### Step 14. Initialize node and edge states

Node states start from the node encoder:

- `x^(0)` or `H^(0)` from STAGE or a type-specific encoder.

Edge states can be initialized from:

- source node embedding;
- target node embedding;
- relation embedding;
- maybe simple edge features if available.

In practice, a small MLP can build:

`s^(0) in R^(|X^1| x d)`

This gives the edge-space diffusion a learned starting point instead of a hand-designed one.

### Step 15. Run diffusion on nodes and edges

We then apply Hodge-style diffusion.

A simple discrete version is:

- `x^(k+1) = x^(k) - eta_0 L_0 x^(k)`
- `s^(k+1) = s^(k) - eta_1 L_1 s^(k)`

for several steps.

Interpretation:

- node signals are smoothed according to the node Laplacian induced by learned restrictions;
- edge signals are smoothed according to both node-edge consistency and face-induced higher-order consistency.

This is not ordinary graph smoothing.

It is diffusion on a learned sheaf over a typed 2D complex.

### Step 16. Read out node representations

After diffusion, we need node-level vectors for prediction.

A practical readout combines:

- final node embedding;
- initial node embedding;
- pooled information from incident edges.

So each target node gets a representation that includes:

- local node content;
- relation-aware edge content;
- higher-order topological effects propagated through faces.

### Step 17. Predict labels

Feed the final target-node representation into an MLP classifier:

- output logits;
- apply cross-entropy or the task-appropriate loss.

This gives the final supervised prediction.

### Step 18. Train end to end

The model is trained end to end with:

- task loss;
- node smoothness or discrepancy regularization based on `delta_0`;
- edge-face consistency regularization based on `delta_1`;
- consistency terms tying edge states back to node-induced edge states.

So the model is not only optimized for accuracy.

It is also encouraged to build geometrically coherent node-edge-face interactions.

---

## 7. What exactly is learned in the current H2S-NN

In the current version, the following parts are learned:

- node encoders or the downstream projection after STAGE;
- relation embeddings;
- node-to-edge restriction maps;
- edge-to-face restriction maps;
- edge initializers;
- diffusion parameters if a neural diffusion form is used;
- classifier weights.

In the current version, the following part is **not** learned automatically:

- the set of meta-path templates.

That means:

- the face set is induced from a manual schema set;
- the geometry on that fixed higher-order support is learned;
- but the higher-order support itself is not yet discovered from data.

So the current H2S-NN learns **geometry on a fixed higher-order construction**.

---

## 8. Why this is already different from ordinary meta-path HGNNs

Even before adding automatic discovery, H2S-NN is already different from ordinary meta-path methods.

Ordinary meta-path HGNNs usually do this:

- define a meta-path view;
- aggregate messages over that view;
- fuse multiple views.

Current H2S-NN instead does this:

- instantiate concrete meta-paths as faces;
- learn edge-to-face sheaf maps;
- build `delta_1`;
- change `L_1`;
- diffuse on edge cochains with higher-order coupling.

So H2S-NN is already topology-native in a way most HGNNs are not.

---

## 9. Where the current limitation is

The current limitation is clear:

- someone still has to tell the model which meta-path templates to use.

This creates three practical weaknesses:

- if no canonical meta-paths are known, the method is awkward to apply;
- if the manual templates are weak, the whole higher-order construction is limited from the start;
- if the schema space is large, hand-picking good templates becomes unreliable.

That is exactly where the proposed extension enters.

---

## 10. The proposed extension: H2S-NN with automatic schema-to-cell discovery

The proposed extension keeps the H2S-NN core, but learns which higher-order structures should exist.

The key idea is:

> do not only learn sheaf geometry on a fixed face set; learn which typed schemas should generate the face set in the first place

In the lower-risk version, the language of discovery is still meta-path schemas.

So we are not yet discovering arbitrary motifs.

We are discovering:

- which typed path schemas matter;
- which concrete instances of those schemas matter;
- which of those instances should become actual 2-cells.

---

## 11. The proposed extension, step by step

### Step A. Build the schema graph

From the heterogeneous graph, derive a type-level schema graph:

- node types are schema nodes;
- relation types are schema edges;
- valid typed transitions define legal schema walks.

This schema graph is the search space for candidate higher-order structures.

### Step B. Generate candidate typed schemas

Instead of requiring the user to specify `PAP`, `PSP`, `APA`, and so on, we generate valid typed walks automatically up to a small length budget.

Examples:

- `Author -> Paper -> Author`
- `Paper -> Subject -> Paper`
- `Author -> Paper -> Term -> Paper -> Author`

In the simplest first version:

- enumerate valid schema walks up to length `L_max`;
- optionally include manual paths as warm-start candidates if they are known.

So manual meta-paths become optional hints, not strict requirements.

### Step C. Compute cheap schema-level scores

Before spending full H2S-NN computation on every candidate schema, estimate whether a schema looks promising.

Useful schema-level features include:

- instance count;
- anchor coverage;
- type diversity;
- relation diversity;
- overlap with already selected schemas;
- length penalty;
- simple topology-aware utility proxies.

A small scoring model `u_theta(sigma)` can learn to rank candidates.

This stage is important because brute-force evaluation of every schema by full training would be too expensive.

### Step D. Select a compact schema subset

From the candidate pool, keep only a small subset of schemas.

This can be done with:

- top-k selection;
- sparse gates;
- Gumbel-top-k;
- progressive pruning.

The goal is not to keep every valid path.

The goal is to keep the ones that are:

- useful;
- not too redundant;
- not too expensive.

### Step E. Mine concrete instances only for selected schemas

Once a schema subset is selected, we collect path instances from the graph only for those schemas.

This avoids exploding the face set for unpromising candidates.

Again, we use a budget:

- sample at most a fixed number per anchor;
- keep only manageable numbers of instances per schema.

So the pipeline stays scalable.

### Step F. Score path instances

Not every instance of a good schema is equally useful.

For each instance `f`, we compute an instance score `beta_f`.

Useful features can include:

- local rarity or frequency;
- endpoint relevance;
- edge disagreement magnitude from current node-edge signals;
- overlap with other instances;
- label-consistency or structural utility proxies.

This lets H2S-NN distinguish:

- good schema, weak instance;
- good schema, strong instance.

That is a major improvement over schema-only selection.

### Step G. Turn selected instances into faces

Now the chosen schema instances are turned into actual 2-cells exactly as in the base H2S-NN construction.

So the extension does not replace H2S-NN.

It feeds better learned higher-order structure into H2S-NN.

This is why the right phrase is:

> schema-to-cell discovery

rather than just:

> meta-path search

### Step H. Weight the face contribution

The edge Laplacian should now depend on both schema-level and instance-level importance.

A clean formulation is:

`L_1^up = sum_sigma alpha_sigma sum_{f in I(sigma)} beta_f (delta_1^f)^T delta_1^f`

and then:

`L_1 = delta_0 delta_0^T + L_1^up`

where:

- `alpha_sigma` is a schema weight;
- `beta_f` is an instance weight;
- `delta_1^f` is the face-specific contribution of instance `f`.

This is the mathematical center of the extension.

The model is learning not only maps, but also which faces should meaningfully participate in the operator.

### Step I. Train the full model

Now train everything jointly:

- node encoder;
- restriction maps;
- schema scorer or selector;
- instance scorer;
- classifier.

Use a loss of the form:

`L = L_task + lambda_sparse L_sparse + lambda_red L_redundancy + lambda_cov L_coverage + lambda_stab L_stability`

plus any Hodge or consistency regularization already used in base H2S-NN.

### Step J. Output the learned higher-order structure

At the end, the model does not only output predictions.

It also outputs a learned higher-order construction:

- selected schemas;
- selected face instances;
- weights on schemas and instances;
- the resulting weighted 2-complex.

That is the learned higher-order structure.

---

## 12. What the learned higher-order structure actually is

This is the most important clarification.

The learned higher-order structure is **not just a meta-path name** like `APA`.

It is the full object:

- the chosen typed schemas;
- the chosen concrete instances of those schemas;
- the fact that those instances are reified as faces;
- the weights controlling how strongly those faces affect `L_1`.

So:

- a meta-path schema is the language of proposal;
- a face instance is the topological object;
- the weighted face set is the learned higher-order structure.

That is why saying "the model learns meta-paths" is only partially correct.

A better statement is:

> the model learns which typed schemas should be reified as weighted 2-cells in the heterogeneous sheaf complex

---

## 13. Three operating modes we should support

The cleanest design is to let one H2S-NN framework support three modes.

### Mode 1. Manual-schema mode

Use known domain meta-paths directly.

This is the current base version.

It is useful when:

- literature-standard meta-paths exist;
- we want the lowest implementation risk;
- we want strong baselines quickly.

### Mode 2. Schema-refinement mode

Start from manual schemas, but allow the model to:

- prune weak ones;
- expand nearby variants;
- add better alternatives.

This is useful when:

- some priors exist;
- we do not trust them completely.

### Mode 3. Auto-discovery mode

Do not assume any manual meta-path set.

Generate and score candidate schemas from the type graph automatically.

This is the scientifically interesting regime because it addresses the real limitation of the current literature.

---

## 14. What happens when the dataset has no predefined meta-paths

There are two different cases here.

### Case 1. The dataset is heterogeneous, but no one gave us a meta-path set

This is the main target case.

We can still:

- read the node types;
- read the relation types;
- build the schema graph;
- generate valid typed walks automatically.

So in this case, the dataset does have enough structure for automatic schema-to-cell discovery.

### Case 2. The dataset does not even have a usable typed schema

This is harder.

If node and edge types are absent or unreliable, then classic meta-path language is not directly available.

At that point, we need an earlier step such as:

- latent node type discovery;
- latent relation type clustering;
- motif discovery in a homogeneous or weakly typed graph.

That is probably a second paper.

For the first strong H2S-NN extension, it is better to focus on the more tractable and still important case:

- the graph is heterogeneous;
- useful meta-paths are not predefined;
- the model must discover candidate schemas from the type graph.

---

## 15. Why RL or MCTS is not the main story

RL or Monte Carlo search can be useful because schema construction is sequential.

But it should not be the main claim.

Why:

- the field already has RL and MCTS variants for meta-path discovery;
- naive reward design is expensive and unstable;
- the real novelty in H2S-NN is that selected schemas become 2-cells and modify the Hodge operator.

So if we use RL later, its role should be:

- proposal policy;
- search helper;
- exploration mechanism.

The headline should still be:

> automatic higher-order topology learning for H2S-NN

not:

> another RL algorithm for meta-path search

---

## 16. A practical implementation roadmap

The safest implementation order is:

### Phase 1. Base H2S-NN

Implement:

- typed graph loading;
- manual meta-path enumeration or sampling;
- face construction with closure edges;
- node-to-edge maps;
- edge-to-face maps;
- `delta_0`, `delta_1`;
- `L_0`, `L_1`;
- diffusion;
- readout and training.

This gives a correct topology-aware heterogeneous sheaf model.

### Phase 2. Schema scoring and refinement

Add:

- schema graph construction;
- candidate typed walk generation;
- schema-level scoring;
- top-k or sparse schema selection.

At this point, manual meta-paths can become optional warm starts.

### Phase 3. Instance weighting

Add:

- instance sampling budgets;
- instance-level scoring;
- weighted faces in `L_1^up`.

This is where the learned higher-order structure becomes much more data-adaptive.

### Phase 4. Optional search improvements

Only after the above is stable, consider:

- policy-guided schema expansion;
- bounded search;
- heuristic-guided exploration;
- PMMM-style stabilization if bilevel schema selection is used.

This order matters because it keeps the method scientifically clean and implementation-feasible.

---

## 17. The key claim we should make

The strongest clean claim is:

> Existing meta-path methods choose semantic aggregation channels, while H2S-NN learns a higher-order sheaf geometry in which selected meta-path instances become actual 2-cells. Our extension further learns which typed schemas should instantiate those 2-cells, allowing topology construction without requiring a predefined meta-path set.

That claim is much stronger than saying:

- we use meta-paths;
- we use RL to find meta-paths;
- we add a search stage before H2S-NN.

It keeps the real novelty exactly where it belongs:

- in the learned higher-order topology;
- in the learned `delta_1`;
- in the learned `L_1`.

---

## 18. Final takeaway

Current H2S-NN already has a distinctive idea:

- meta-path instances are turned into faces;
- those faces shape Hodge diffusion.

Its current weakness is that the face-generating schema set is still manual.

The natural next step is therefore not "just another meta-path search algorithm."

It is:

> let H2S-NN learn which typed schemas and which concrete path instances should become weighted faces of the heterogeneous sheaf complex

That gives us a model that:

- works when good manual meta-paths exist;
- refines them when they are weak;
- and can discover them automatically when they are missing.

---

## 19. Source anchors in this repository

This note is based mainly on:

- [H2S-NN_implementation_spec.md](<G:\My Drive\H2S-NN_test\H2S-NN_implementation_spec.md>)
- [H2SNN_corrected_algorithm.md](<G:\My Drive\H2S-NN_test\H2SNN_corrected_algorithm.md>)
- [h2snn_metapath_research_memo_2026-05-06.md](<G:\My Drive\H2S-NN_test\docs\h2snn_metapath_research_memo_2026-05-06.md>)
- [h2snn_rl_mcts_novelty_note_2026-05-06.md](<G:\My Drive\H2S-NN_test\docs\h2snn_rl_mcts_novelty_note_2026-05-06.md>)
