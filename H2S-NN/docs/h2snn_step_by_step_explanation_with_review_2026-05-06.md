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


## 19. Reviewer-style strengths and weaknesses

This section summarizes how H2S-NN may be viewed by a critical reviewer for a machine learning, graph learning, or data mining venue.

### 19.1 Overall reviewer assessment

The proposed model has a promising and nontrivial core idea: instead of treating meta-paths only as semantic aggregation channels, H2S-NN turns concrete meta-path instances into 2-cells of a typed cellular complex and uses those cells to shape learned sheaf-Hodge diffusion.

This is a stronger modeling move than ordinary meta-path aggregation because the selected meta-path instances do not merely define additional message-passing views. They become part of the higher-order operator through `delta_1` and the upper part of the edge Hodge Laplacian.

At the same time, the current version still has important weaknesses. The main risks are mathematical soundness of the face construction, computational scalability, dependence on manually selected meta-path templates, and the need to show that the higher-order sheaf construction gives benefits beyond strong heterogeneous graph neural network baselines.

A fair provisional assessment is:

> H2S-NN is potentially strong if the paper clearly formalizes the cellular/sheaf construction, proves basic operator properties, and provides convincing ablations. Without those, reviewers may see the method as conceptually interesting but too complex and insufficiently validated.

### 19.2 Strengths

#### Strength 1. The central idea is distinctive

The strongest contribution is the shift from **meta-path as semantic view** to **meta-path instance as topological cell**.

Most heterogeneous graph neural networks use meta-paths to define semantic neighborhoods, relation-specific views, or attention channels. H2S-NN instead reifies selected path instances as faces. These faces contribute to the coboundary operator `delta_1`, which then changes the edge-space Hodge Laplacian.

This gives the model a clean novelty claim:

> H2S-NN does not merely aggregate over meta-path views; it uses typed path instances to define higher-order cells that modify the diffusion geometry itself.

#### Strength 2. The sheaf formulation is expressive

The model learns restriction maps between nodes, edges, and faces rather than relying only on scalar edge weights or fixed incidence signs.

This is useful for heterogeneous graphs because different node and relation types may not live in the same semantic space. Learned sheaf maps allow the model to decide how information from one type should be interpreted when transported into an edge or face stalk.

In particular:

- node-to-edge maps model relation-aware endpoint compatibility;
- edge-to-face maps model higher-order path consistency;
- `delta_0` measures node-edge disagreement;
- `delta_1` measures edge-face disagreement;
- `L_1 = delta_0 delta_0^T + delta_1^T delta_1` lets edge signals diffuse according to both endpoint consistency and higher-order face structure.

This is a richer operator than ordinary graph message passing.

#### Strength 3. The model addresses heterophily more naturally than simple neighborhood aggregation

Heterogeneous graphs often exhibit heterophily: adjacent nodes may have different labels, types, or semantics. Ordinary graph neural networks can over-smooth such neighborhoods.

H2S-NN has a better inductive bias for this setting because it does not force all neighboring nodes to become similar. Instead, it learns typed restriction maps that define when two signals are compatible after being transported into a common stalk.

This allows the model to represent structured disagreement, not only similarity.

#### Strength 4. Higher-order structure affects the operator, not only the readout

Many models add higher-order information only as extra features, auxiliary views, or attention channels. In H2S-NN, higher-order structures affect the actual diffusion operator through `delta_1^T delta_1`.

This is important because it means the higher-order structure changes how information propagates through the model, rather than merely being appended at the end.

#### Strength 5. The proposed extension has a meaningful path toward automatic topology construction

The extension from manual meta-paths to schema-to-cell discovery is well motivated. Instead of assuming that the correct meta-paths are known in advance, the model can generate candidate typed schemas, score them, select useful ones, mine concrete instances, and turn selected instances into weighted faces.

The strongest version of the claim is not simply that the model learns meta-paths. The stronger claim is:

> H2S-NN learns which typed schemas and concrete instances should be reified as weighted 2-cells in a heterogeneous sheaf complex.

That is more distinctive than ordinary meta-path search.

#### Strength 6. The learned structures may be interpretable

The schema-to-cell extension can output:

- selected schemas;
- selected concrete path instances;
- schema weights `alpha_sigma`;
- instance weights `beta_f`;
- the final weighted higher-order complex.

This gives the model a potentially interpretable output. A user can inspect which typed relational patterns were considered important for prediction.

### 19.3 Weaknesses

#### Weakness 1. Current H2S-NN still depends on predefined meta-path templates

A central limitation of the current H2S-NN formulation is its dependence on manually predefined meta-path templates.

Although H2S-NN uses meta-path instances in a more sophisticated way than ordinary meta-path aggregation methods, the initial choice of which meta-path schemas to instantiate is still externally specified. This means that the higher-order support of the model is only as good as the selected meta-paths.

This creates several problems.

First, predefined meta-paths do not guarantee semantic or predictive usefulness. A meta-path may be standard in the literature, but that does not mean it is optimal for a particular dataset, task, label distribution, or heterophily pattern. If the selected meta-paths are weak, noisy, redundant, or weakly correlated with the target labels, then the resulting face set will also be weak. In that case, the Hodge/sheaf machinery is learning geometry over a suboptimal higher-order support.

Second, this dependence restricts the applicability of the model to datasets where meaningful meta-paths are already known or can be manually designed. Many real heterogeneous graphs have rich type systems but no canonical meta-path set. In such settings, requiring users to provide meta-paths makes the method less automatic and less transferable.

Third, manual meta-path design creates a generalizability bottleneck. The model may perform well on benchmark datasets such as ACM, DBLP, or IMDB because these datasets have well-established meta-paths, but it may lag behind on new domains where the relevant typed relational patterns are unknown, task-specific, or dataset-dependent.

A reviewer may state the concern as follows:

> Although the proposed method introduces an interesting higher-order sheaf construction, its current version still inherits a major weakness of meta-path-based heterogeneous graph neural networks: the higher-order support is determined by manually chosen meta-paths. Since these meta-paths are not guaranteed to be optimal or even useful for the downstream task, the model's performance may depend heavily on external domain knowledge. This limits its generalizability, especially to heterogeneous datasets without predefined or canonical meta-path schemas.

This weakness directly motivates the proposed schema-to-cell discovery extension.

#### Weakness 2. The closure-edge construction needs stronger mathematical justification

The current construction closes each open meta-path instance by adding a synthetic closure edge from the final node back to the initial node.

This is convenient because a face needs a closed boundary. However, reviewers may question whether the closure edge is mathematically and semantically valid.

Important questions include:

- Is the closure edge a real 1-cell or only an auxiliary topological device?
- Does the closure edge participate in diffusion?
- If its contribution is masked or set to zero, is the resulting object still a proper 2-cell?
- Does the construction satisfy the required boundary-of-boundary-zero condition?
- Is the model truly using a cellular complex, or a generalized path-induced incidence structure?

This issue must be formalized carefully. If the paper claims to build a typed cellular complex, then the face construction must be mathematically sound.

#### Weakness 3. The method is computationally heavy

The model has several expensive components:

- meta-path instance enumeration or sampling;
- edge-to-face incidence construction;
- learned node-to-edge restriction maps;
- learned edge-to-face restriction maps;
- Hodge Laplacian construction;
- node and edge diffusion;
- optional schema and instance scoring.

The largest scalability concern is the edge-to-face maps. If every edge-face incidence receives a full `d x d` matrix, memory and computation may grow quickly.

This may limit the model to relatively small benchmark datasets unless the implementation uses efficient approximations, such as:

- diagonal restriction maps;
- low-rank restriction maps;
- basis-decomposed maps;
- relation-specific shared bases;
- sampled faces with strict budgets;
- sparse operator construction.

A reviewer may ask whether the method can scale beyond standard heterogeneous graph benchmarks.

#### Weakness 4. The model may appear over-engineered without strong ablations

H2S-NN contains many interacting components. Without careful ablation studies, reviewers may not know which component actually produces the improvement.

Necessary ablations include:

- remove faces and use only graph-level sheaf diffusion;
- remove `delta_1^T delta_1` from `L_1`;
- replace learned sheaf maps with identity maps;
- replace full matrix maps with scalar or diagonal maps;
- replace edge-to-face LSTM maps with position-independent maps;
- compare manual schemas against random schemas;
- compare learned schema weights against uniform schema weights;
- compare learned instance weights against uniform instance weights;
- compare Hodge diffusion against standard message passing over an expanded graph.

Without these ablations, the method may look like a complicated architecture whose gains could come from extra parameters rather than from the proposed higher-order topology.

#### Weakness 5. The learning objective is not yet fully specified

The proposed extension mentions losses such as sparsity, redundancy, coverage, and stability regularization. These are sensible, but they need exact definitions.

A reviewer will expect clear answers to the following questions:

- How is schema sparsity measured?
- How is redundancy between schemas computed?
- What does coverage mean: anchor-node coverage, relation coverage, type coverage, or face coverage?
- What does stability mean: stable schema selection, stable predictions, or stable operators?
- Are schema weights constrained to be nonnegative?
- Are instance weights constrained to be nonnegative?
- Is the resulting Hodge Laplacian guaranteed to be positive semidefinite?

The last point is especially important. If the weighted upper Laplacian is defined as

`L_1^up = sum_sigma alpha_sigma sum_{f in I(sigma)} beta_f (delta_1^f)^T delta_1^f`,

then nonnegative `alpha_sigma` and `beta_f` help preserve positive semidefiniteness. This should be stated explicitly.

#### Weakness 6. There is a risk of label leakage in schema or instance scoring

The extension proposes scoring schemas and instances using utility proxies. If label-consistency signals are used, the training protocol must be careful.

The model must not use validation or test labels to choose schemas or instances. Otherwise, the reported performance may be invalid.

A clean protocol should specify that:

- training labels may be used for supervised objectives;
- validation labels are used only for model selection;
- test labels are never used in schema or instance selection;
- transductive use of graph structure is allowed only if consistent with the benchmark protocol.

#### Weakness 7. The empirical contribution may be weak if evaluated only on standard meta-path benchmarks

Datasets such as ACM, DBLP, and IMDB are useful, but they may not be sufficient to prove the value of H2S-NN.

These datasets often already have canonical meta-paths. If H2S-NN is evaluated only there, reviewers may conclude that the method mainly benefits from known benchmark-specific meta-path engineering.

The stronger evaluation would include:

- datasets without predefined meta-paths;
- datasets where useful typed schemas are unknown;
- heterophilous heterogeneous graphs;
- synthetic datasets with planted higher-order typed patterns;
- larger graphs where scalability matters.

This is especially important for supporting the generalizability claim.

### 19.4 Suggested reviewer-facing positioning

The safest way to present the model is to separate the base model from the extension.

#### Base H2S-NN claim

> Given a set of typed schemas, H2S-NN instantiates selected path instances as 2-cells and performs learned sheaf-Hodge diffusion over the resulting heterogeneous complex.

#### H2S-NN discovery claim

> H2S-NN can be extended to learn which typed schemas and concrete instances should become weighted faces, reducing dependence on manually predefined meta-paths and improving generalizability to heterogeneous datasets without canonical meta-path sets.

This separation is important because the base model still depends on predefined meta-paths, while the extension directly addresses that limitation.

### 19.5 Most important improvements before submission

Before submitting this work to a strong venue, the most important improvements are:

1. formalize the path-to-face construction rigorously;
2. justify or replace synthetic closure edges;
3. prove basic operator properties, especially positive semidefiniteness under nonnegative face weights;
4. reduce computational cost through diagonal, low-rank, or basis-decomposed restriction maps;
5. add ablations isolating the contribution of `delta_1`, sheaf maps, face weights, and schema discovery;
6. evaluate on datasets where no predefined meta-path set is available;
7. show that automatic schema-to-cell discovery improves generalizability beyond standard benchmark meta-paths.

The key reviewer-facing message should be:

> The current H2S-NN is already distinctive because meta-path instances become operator-shaping 2-cells, but its dependence on predefined meta-paths limits generalizability. The schema-to-cell discovery extension is therefore not an optional add-on; it is the natural mechanism that turns H2S-NN from a topology-aware meta-path model into a more general heterogeneous higher-order learning framework.

---

## 20. Source anchors in this repository

This note is based mainly on:

- [H2S-NN_implementation_spec.md](<G:\My Drive\H2S-NN_test\H2S-NN_implementation_spec.md>)
- [H2SNN_corrected_algorithm.md](<G:\My Drive\H2S-NN_test\H2SNN_corrected_algorithm.md>)
- [h2snn_metapath_research_memo_2026-05-06.md](<G:\My Drive\H2S-NN_test\docs\h2snn_metapath_research_memo_2026-05-06.md>)
- [h2snn_rl_mcts_novelty_note_2026-05-06.md](<G:\My Drive\H2S-NN_test\docs\h2snn_rl_mcts_novelty_note_2026-05-06.md>)

---

## 21. Simple Planning Table

The table below restates the main weaknesses in a compact planning format.

| Problem | Why Reviewers Care | What We Do Next |
|---|---|---|
| The path-to-face construction may not be mathematically solid enough. | If the face construction is not well justified, the whole higher-order topology claim may look artificial or invalid. Reviewers may reject the main idea before even considering the empirical results. | Write a precise formal section for the path-to-face construction. Either prove the closure-edge construction is valid for the operator we use, or weaken the claim and present it as a path-induced higher-order operator rather than a strict classical cellular complex. |
| The base H2S-NN and the schema-discovery extension are being mixed too closely. | If the paper claims both at once without fully validating both, reviewers may feel the contribution is too broad, incomplete, or inflated. | Separate the claims clearly. Present base H2S-NN as one contribution and schema-to-cell discovery as the extension. Only make discovery the headline if it is fully implemented and validated. |
| The model may look over-engineered. | A reviewer may think the performance gain comes from extra parameters or architectural complexity rather than from the higher-order topological idea itself. | Run strong ablations: remove faces, remove `delta_1`, replace learned maps with identity/diagonal maps, remove the LSTM face encoder, compare manual schemas to random schemas, and compare learned weights to uniform weights. |
| The method may be too computationally heavy. | If memory and runtime scale poorly, reviewers may see the method as impractical beyond small benchmarks. | Start with cheaper map parameterizations such as diagonal, low-rank, or basis-decomposed maps. Use strict face budgets and sparse matrix-free operators. Report runtime and memory explicitly. |
| The learning objective for schema discovery is not yet exact enough. | Reviewers expect full mathematical definitions, not only intuitive descriptions like sparsity, redundancy, coverage, and stability. | Write the exact objective now. Define each regularizer precisely, constrain `alpha_sigma` and `beta_f` to be nonnegative, and state why the resulting weighted `L_1` remains positive semidefinite. |
| There is a risk of label leakage in schema or instance scoring. | If schema selection or instance scoring accidentally uses validation or test labels, the reported results will not be trusted. | Lock the protocol early: training labels only for supervised learning, validation only for model selection, and test labels never for schema or instance selection. State this clearly in the method section. |
| The current model still depends on manual meta-path schemas. | This limits generalizability to datasets where good schemas are unknown, weak, or unavailable. | Be honest about this limitation in the base paper. First make the base model mathematically and empirically strong, then add schema-to-cell discovery as the next main extension. |
| The evaluation may be too weak if it uses only ACM, DBLP, and IMDB. | Reviewers may conclude the method only works well on standard benchmark datasets with already-known meta-path schemas. | Add at least one stronger generalization setting: a dataset without canonical meta-paths, a heterophilous heterogeneous dataset, a synthetic dataset with planted higher-order patterns, or a larger dataset for scalability. |

### Recommended order of attack

1. Formalize the path-to-face construction and the closure-edge argument.
2. Simplify the implementation enough that it is computationally credible.
3. Build the ablation suite that proves `delta_1` and face structure matter.
4. Only after that, push hard on schema-to-cell discovery.
