# Problems with Existing Automatic Meta-Path Selection — Detailed Reference

> **Historical design reference (superseded for implementation details).** The problem framing
> remains useful, but field names and objective snippets in this document predate AMPSS
> `schema_conditioned_v1`. Use `docs/h2snn_current_model_documentation.html` and the current
> `src/` implementation for authoritative selection, instance-weight, and ablation semantics.
> Current AMPSS has no schema gate or auxiliary discovery loss.

**Audience:** Professor / researcher familiar with HGNNs. Not assumed to have read H2S-NN.

**Purpose:** Exhaustive explanation of every gap in existing automatic meta-path selection
methods, why it matters, how H2S-NN solves it, and where the solution lives in the codebase.
Structured so that each section maps directly to one slide and one code module.

**Source table:** `h2snn_selection_comparison.csv`
**Compared methods:** LMSPS · AMP-HGNAS · MCTS-HGNN · PMMM · HG-Search · PM-HGNN

---

## What "automatic meta-path selection" means — shared baseline

All six compared methods (LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM, HG-Search, PM-HGNN) already
solve the basic step of **removing manual path design**. Given a heterogeneous graph with node
types {Author, Paper, Venue} and relation types {writes, cites, published-at}, they can
enumerate candidate typed walks from the type graph and identify the useful ones — without a
human writing "use APA and APCPA."

This is the starting point. The seven rows in the comparison table describe **what these
methods still fail to do after that basic step**.

---

## Problem 1 — Hard selection only: no soft schema-level importance score

### The gap

Most methods make a **binary keep/drop decision** per schema. Either a schema survives the
search and is fully used, or it is discarded entirely. There is no in-between.

**Methods with hard selection:**
- **MCTS-HGNN** — Monte Carlo Tree Search scores candidate paths and selects a fixed
  winning subset. A path either makes it through the search tree or does not. No fractional
  contribution is possible.
- **HG-Search** — discrete architecture search; each candidate is either in or out of the
  final architecture.
- **PM-HGNN** — uses a reinforcement learning policy that selects one path per node at
  inference time. The policy outputs a single discrete choice, not a continuous weight.

**Methods with soft selection (learned scalar α per schema):**
- **LMSPS** — learns schema-level soft weights as part of its progressive pruning loop.
- **AMP-HGNAS** — uses learnable Gumbel-softmax weights during architecture search.
- **PMMM** — learns importance scores via a meta-path mixing module.

**H2S-NN** — learns `α_σ = sigmoid(logit_σ)` per schema, fully differentiable, jointly
optimized with model weights.

### Why it matters

A binary decision throws away information. If two schemas are almost equally useful, keeping
one and discarding the other loses the partial contribution of the second. Real schemas exist
on a spectrum of usefulness: some are crucial, some contribute modestly, some are mildly
harmful. Soft continuous weights let the model express this entire spectrum.

Furthermore, hard discrete selection is not differentiable — it cannot be directly optimized
by gradient descent. Methods that use it (MCTS-HGNN, HG-Search) must resort to proxy
objectives or reward signals that are not aligned with the final task loss. This creates a
gap between what is optimized during selection and what the model actually needs.

### H2S-NN implementation

```
src/models/schema_gates.py  →  class SchemaGate
    self.logits: nn.Parameter([num_schemas])         # one learnable scalar per schema
    alpha = sigmoid(self.logits)                     # alpha ∈ (0, 1), soft, differentiable

src/configs/config.py  →  SchemaDiscoveryConfig.schema_gate_type = "sigmoid"
src/models/h2snn_model.py  →  H2SNN._build_face_weights()  # applies alpha to face weights
```

The `logits` are backpropagated through the full task loss. No separate search loop.

### Slide key points

- **What existing methods do:** binary keep/drop (MCTS-HGNN, HG-Search) or soft weights only
  for a few methods (LMSPS, PMMM)
- **What H2S-NN does:** every schema has `α ∈ (0,1)`, jointly optimized, warm-started from
  topology scores so high-discrepancy schemas begin with large logits
- **Visual suggestion:** bar chart of α values after training — shows spectrum, not binary

---

## Problem 2 — No instance-level importance: all path instances treated equally

### The gap

Once a schema (e.g., Author → Paper → Author) is selected, **all concrete path instances**
that match it in the graph are treated with equal importance. The model cannot distinguish:

- Instance A: Author₁ → Paper₇ → Author₂ (two leading researchers, landmark joint paper)
- Instance B: Author₃ → Paper₁₂ → Author₄ (two PhD students, workshop abstract overlap)

Every existing method — LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM, HG-Search, PM-HGNN — has this
limitation. Schema-level selection selects the *type* of path to follow. Instance-level
importance controls *which specific instances* of that path matter. No existing method
separates these two questions.

### Why it matters

Within any selected schema, real graphs contain huge variance in path quality:

- **Hub nodes** create many path instances through highly-connected intermediaries. These are
  often noisy: Author₁ → Survey_Paper → Author₂ via a paper with 500 co-authors gives very
  little information about the specific relationship between Author₁ and Author₂.
- **Specific instances** through rare, meaningful intermediaries carry strong signal.
- **Coincidental instances** arise from random graph structure and should be down-weighted.

If the model assigns equal importance to all instances of a selected schema, it is mixing
high-signal instances with noise at equal proportions. Increasing schema count does not help —
more schemas means more noise instances alongside more signal instances.

The only solution is to learn instance-level importance directly.

### H2S-NN implementation

```
src/models/schema_gates.py  →  class InstanceGate
    input: [h_{v0} || h_{v_L} || schema_embedding]   # shape: [F_s, 2*d + d_r]
    self.net: Linear → ReLU → Linear → sigmoid
    output: beta_f ∈ (0, 1) for each face f          # shape: [F_s]

src/models/h2snn_model.py  →  H2SNN._build_face_weights()
    instance_input = cat([h0[v0], h0[vl], schema_emb])
    beta = self.instance_gate(instance_input)
    weights.append(alpha[schema_idx] * beta)          # product: schema × instance
```

**Key property — adaptive weights:** `beta_f` is recomputed every forward pass as node
embeddings `h0` evolve. As the model learns, its sense of which instances matter updates
continuously. This is impossible with static or precomputed instance weights.

### Slide key points

- **What existing methods do:** after selecting schema σ, use all instances `{f ∈ I(σ)}` with
  equal weight = 1
- **What H2S-NN does:** every face `f` gets its own `β_f` computed from current node
  embeddings — so the model learns to trust specific author–paper–author triples more than
  others
- **Visual suggestion:** histogram of β values for one schema — shows how the model
  differentiates instances

---

## Problem 3 — No joint optimization of schema-level and instance-level selection

### The gap

Even granting that a method has soft schema weights (Problem 1 solved) and instance weights
(Problem 2 solved), no existing method optimizes **both together** in the same training loop.

The reason is structural: in the existing pipeline, schema selection happens first (in a search
phase), and then fixed schemas are given to the model (in a retrain phase). By the time
instance-level processing happens (if it exists at all), the schema weights are already frozen.
The two levels are sequentially decoupled.

### Why it matters

Schema-level and instance-level importance are **correlated**. Whether a schema type is useful
depends on whether its high-quality instances can be found. Whether a specific instance is
useful depends partly on which other schemas are also active — a specific APA instance might
be crucial if no APCPA instances exist, but redundant if they do.

Optimizing them together means:
- If a schema's instances all turn out to be noisy (low β), the schema weight α can drop
  without the user intervening.
- If a single critical instance is discovered (high β), it can pull the schema's α upward.
- Both are driven by the same gradient from the same task loss.

Sequential decoupled optimization misses these cross-level interactions.

### H2S-NN implementation

```
src/models/h2snn_model.py  →  H2SNN.forward()
    # Both alpha and beta are computed inside every forward pass
    face_weights, schema_alpha = self._build_face_weights(data, h0, epoch)
    # face_weights[g] = alpha[schema_idx] * beta   — product of both levels
    # Both contribute to the diffusion operator and therefore to the task loss gradient
    # Both are updated in the same optimizer.step()

src/models/discovery_losses.py  →  compute_discovery_losses()
    sparse_schema = alpha.mean()             # regularizes alpha
    sparse_instance = face_weight_means.mean()  # regularizes beta
    total = sum of all discovery losses + task_loss
```

Both `alpha` (from `SchemaGate.logits`) and the weights inside `InstanceGate.net` receive
gradients from the same backward pass. There is no stage boundary.

### Slide key points

- **What existing methods do:** fix schemas first (search phase), then train model with fixed
  schemas (retrain phase) — α and β never co-adapt
- **What H2S-NN does:** α and β are both `nn.Parameter`/`nn.Module` updated in every gradient
  step alongside restriction maps and classifier weights
- **Key diagram:** single training loop with arrows from task loss back to both α and β

---

## Problem 4 — No redundancy penalty: overlapping schemas are selected independently

### The gap

When evaluating whether to include a schema, most methods ask: "is this schema useful?"
They do not ask: "is this schema useful *given the schemas already selected?*"

This causes redundant schema sets. Consider three schemas on an academic graph:
- **APA**: Author → Paper → Author (co-authorship)
- **APAPA**: Author → Paper → Author → Paper → Author (2-hop co-authorship)
- **APCPA**: Author → Paper → Conference → Paper → Author (shared venue)

All three primarily connect pairs of authors through papers. They carry heavily overlapping
information. Selecting all three wastes capacity, increases computation, and creates gradient
interference during training (many correlated inputs all pushing gradients in the same
direction, making it harder to learn fine-grained distinctions).

**Methods that do NOT handle redundancy:**
- LMSPS, MCTS-HGNN, PMMM, HG-Search, PM-HGNN — all evaluate schemas independently.

**Method that DOES handle redundancy:**
- **AMP-HGNAS (2026)** — explicitly models inter-schema dependency and penalizes overlapping
  schemas. This is one of AMP-HGNAS's primary contributions. H2S-NN also handles this.

### Why it matters

Redundancy has three costs:
1. **Computational:** more faces to instantiate, store, and process per batch
2. **Gradient:** correlated signals reduce the effective information per gradient step
3. **Interpretability:** a model with 8 distinct schemas is easier to interpret than one with
   8 schemas that all say the same thing with slight variation

A diversity-aware selection mechanism ensures the selected pool is compact and complementary.

### H2S-NN implementation — two levels of redundancy control

**Stage D (pre-training, greedy pool selection):**
```
src/data/schema_discovery.py  →  select_schema_pool()
    jaccard_endpoint_overlap(stats_a, stats_b)   # Jaccard on end-node sample sets
    topology_aware_score():
        base = disc^alpha * cov^beta * div^gamma
        redundancy = max Jaccard overlap with already-selected schemas
        score = base / (1 + redundancy)           # penalty for overlap
    # Greedy: add schema with highest diversity-penalized score at each step
```

**During training (learned redundancy regularization):**
```
src/models/discovery_losses.py  →  compute_discovery_losses()
    cov_profile = (start_cov + end_cov) / 2      # coverage proxy per schema
    redundancy = (alpha * cov_profile * alpha).sum() / (alpha.sum()^2)
    # High when many schemas are both active (high alpha) and cover the same nodes
```

Both mechanisms are active: the greedy selection ensures the initial pool is diverse, and the
regularization loss discourages the model from keeping multiple overlapping schemas at high α.

### Slide key points

- **What most existing methods do:** each schema evaluated independently → redundant pool
- **What AMP-HGNAS does:** adds inter-schema dependency penalty (their main contribution)
- **What H2S-NN does:** Jaccard-based diversity in greedy selection PLUS differentiable
  redundancy regularization during training
- **Visual suggestion:** Jaccard overlap matrix of selected schemas — shows how diverse the
  final pool is

---

## Problem 5 — Two-stage pipeline: search then retrain causes search-evaluation mismatch

### The gap

All existing automatic selection methods work in two sequential stages:

**Stage 1 — Schema search:**
Run a search procedure (MCTS tree search, progressive pruning, NAS, RL policy, etc.) to
find good schemas. This uses a simplified, proxy, or partial version of the final model.

**Stage 2 — Model retraining:**
Take the schemas selected in Stage 1 and train the full model from scratch using only those
schemas.

The fundamental problem: **what looks good in Stage 1 may not remain optimal in Stage 2.**

The search stage optimizes a proxy objective (usually validation accuracy with a small proxy
model, or a search-time surrogate). The retrain stage uses a different, larger model trained
to full convergence. The two models have different inductive biases, different learning
dynamics, and different sensitivities to schema choice. A schema that boosted a small proxy
model may add noise to the larger retrained model.

This is called **search-evaluation mismatch** and is a well-known weakness in the neural
architecture search (NAS) literature. Methods affected:

- **LMSPS** — progressive pruning across multiple training rounds; each round is a
  fresh training run on a smaller schema subset.
- **AMP-HGNAS** — explicit two-stage NAS: search phase, then retrain.
- **MCTS-HGNN** — tree search is completely separate from model training. The model trained
  on the MCTS-selected schemas has no gradient connection to the search decisions.
- **PMMM** — searches for meta-path mixtures and then retrains.
- **HG-Search** — discrete architecture search followed by retraining.

**PM-HGNN** partially avoids this by using RL in a single loop, but the RL policy and model
are not jointly differentiable — the policy gradient is distinct from the task gradient.

### Why it matters

The mismatch causes two practical problems:

1. **Wasted computation:** the entire search stage's computation may produce schemas that are
   suboptimal for the final model. You optimized for something you will not use.

2. **Suboptimal schemas:** the final model is trained on schemas chosen by a different model.
   If the true optimal schema set depends on the downstream model's architecture and learned
   representations — which it does — then using a proxy to select it is systematically biased.

End-to-end single-run selection avoids both. The selected schemas are directly optimized for
the exact model they will be used in, updated by the exact loss they will minimize.

### H2S-NN implementation

```
src/trainers/trainer.py  →  Trainer._train_epoch()
    # Inside every batch:
    output = model(data, epoch=epoch)          # computes alpha, beta inside forward
    task_loss = criterion(output.logits, y)
    disc_loss = compute_discovery_losses(output, data, cfg.discovery).total
    loss = task_loss + disc_loss
    loss.backward()                            # single backward: alpha, beta, model all updated
    optimizer.step()                           # single optimizer step: no stage boundary

src/configs/config.py  →  SchemaDiscoveryConfig.warmup_epochs
    # Optional warm-up: gates are frozen for warmup_epochs to let representations stabilize
    # before schema selection begins — but this is one continuous run, not a separate phase
```

There is no `search()` function that returns a fixed schema list. The schema weights `α` and
`β` are ordinary model parameters updated by the same optimizer every step.

### Slide key points

- **What all existing methods do:** search → retrain (two separate training runs)
- **Search-evaluation mismatch:** proxy model ≠ final model → schemas selected for wrong model
- **What H2S-NN does:** α and β are `nn.Parameter` trained with the same `loss.backward()`
  as restriction maps and classifier
- **Key diagram:** single training timeline vs. two-stage timeline — show where gradient flows

---

## Problem 6 — Cannot incorporate domain knowledge as a refinable starting point

### The gap

Suppose a domain expert says: *"For this academic graph, Author→Paper→Author and
Paper→Venue→Paper are probably useful. I'm not certain, but these are my best guesses."*

Every existing method responds in one of two ways:
- **Ignore it entirely** — search methods (LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM) start their
  search from scratch regardless of any hints.
- **Accept it as fixed truth** — manual methods (HAN, MAGNN) take user-provided paths and
  never question them.

Neither is useful. The expert's knowledge is partial and possibly wrong. A good system should:
1. Start from the expert's suggestions
2. Test whether each suggested schema is actually useful
3. Replace weak ones with better nearby alternatives on the schema graph
4. Add complementary schemas not in the original suggestion
5. Do all of this in a single run, without requiring the user to iterate manually

No existing automatic selection method does this.

### Why it matters

In practice, domain experts have partial knowledge. Ignoring it is wasteful — it throws away
real information. Rigidly accepting it is dangerous — it prevents the model from correcting
the expert's mistakes.

A refinement capability bridges manual and automatic methods: it treats human knowledge as a
warm start rather than as ground truth. The model verifies, extends, and corrects expert
knowledge rather than replacing it entirely.

This is especially valuable in applied research where:
- The graph is new and the best schemas are not obvious
- Some schemas are known to be useful but the optimal set is unknown
- Computational budget requires starting close to a good solution

### H2S-NN implementation

```
src/configs/config.py  →  SchemaDiscoveryConfig.mode = "manual" | "auto" | "refine"

src/data/schema_discovery.py  →  generate_schema_candidates()
    # In "refine" mode: start candidates from user-provided schemas,
    # expand to neighbors on the schema graph, score all (original + neighbors),
    # let topology-aware selection and gate training decide which to keep

src/models/schema_gates.py  →  SchemaGate.init_from_scores()
    # User-provided schemas can be warm-initialized with high logits (positive prior)
    # Gate training refines them downward if they prove unhelpful
```

The three modes form a spectrum:
- **manual:** user-specified schemas, fixed pool, only α/β are learned
- **refine:** user schemas as seeds, expanded and scored, pool partially fixed
- **auto:** no seeds, full enumeration from type graph, all of pipeline active

### Slide key points

- **What existing methods do:** binary choice — ignore expert (search methods) or trust
  blindly (HAN/MAGNN) — no middle ground
- **What H2S-NN does:** expert hints → warm-start logits → gate training refines or suppresses
  them → model verifies expert knowledge
- **Three modes:** manual / refine / auto — shown as a slider from "fully human" to "fully
  automatic"

---

## Problem 7 — Schema quality judged only by task accuracy: expensive and topologically blind

### The gap

How do existing methods decide whether a candidate schema is good?

They train a model using that schema and measure validation accuracy. High accuracy → good
schema. Low accuracy → bad schema.

This creates two problems:

**Problem A — Expensive:**
Every candidate evaluation requires training or partially training a model. With dozens or
hundreds of candidate schemas, each evaluation is costly. Methods like MCTS-HGNN perform
dozens of rollouts, each requiring a model forward pass or proxy training. LMSPS runs multiple
training rounds. AMP-HGNAS has a full NAS search stage.

A cheaper scoring signal that does not require gradient steps would dramatically reduce the
cost of schema evaluation.

**Problem B — Topologically blind:**
Validation accuracy measures one thing: does this schema immediately improve prediction scores
on the held-out set? But a schema can be structurally valuable in ways that do not show up as
an accuracy jump — at least not immediately.

For example: a schema that creates faces (triangles) whose endpoints have very different
labels generates a large `δ₁` coboundary signal in the Hodge operator. This means the schema
adds structural inconsistency information to the diffusion. That inconsistency might not improve
accuracy in early training but is exactly the kind of signal the model needs to handle
heterophily. An accuracy-only scorer would discard this schema prematurely.

### H2S-NN implementation

H2S-NN computes a **topology-aware score** before training begins, using only graph structure
and label propagation — no model training required.

```
src/data/schema_discovery.py  →  compute_endpoint_label_discrepancy()
    # For a sampled set of path instances under schema σ:
    start_nodes = path[:, 0]
    end_nodes   = path[:, -1]
    diff = propagated_labels[start_nodes] - propagated_labels[end_nodes]
    disc = mean(||diff||^2)    # high ↔ schema connects heterophilic endpoints
    # Uses label propagation from training labels only — no test leakage

    topology_aware_score(stats, selected_stats):
        disc = endpoint_label_discrepancy        # heterophily signal
        cov  = end_coverage                      # target-node reachability
        div  = type_diversity                    # cross-type schema reward
        redundancy = max Jaccard overlap with selected pool
        score = (disc^α * cov^β * div^γ) / (1 + redundancy)
```

**Why endpoint label discrepancy is topology-native:**
A high discrepancy means the schema systematically connects nodes with different labels —
exactly the condition that creates large `||δ₀ x||²` signals in the Hodge operator. This
score directly predicts whether a schema will contribute meaningful heterophily information to
diffusion, without running any gradient step.

This score is used in two ways:
1. **Pre-training pool selection** — `select_schema_pool()` uses it to rank and greedily pick
   the top-K diverse schemas.
2. **Gate initialization** — `SchemaGate.init_from_scores()` warm-starts logits from these
   scores so high-discrepancy schemas begin training with large positive logits.

### Slide key points

- **What existing methods do:** evaluate schema quality = run (partial) model training = wait
  for accuracy signal — expensive and accuracy-blind to structural value
- **What H2S-NN does:** topology score = `disc × coverage × diversity / (1 + redundancy)` —
  computed from label propagation in seconds, before any gradient step
- **Why topology score predicts usefulness:** schemas that connect heterophilic endpoints
  produce large coboundary signals → directly useful for Hodge diffusion
- **Two uses:** pool selection (greedy K-best diverse set) + gate warm-start (logits ∝ score)
- **Visual suggestion:** scatter plot of topology score vs. final learned α — should correlate

---

## Composite summary: all seven problems in one table

| # | Problem | Methods affected | H2S-NN solution | Code location |
|---|---------|-----------------|-----------------|---------------|
| 1 | Hard selection only | MCTS-HGNN, HG-Search, PM-HGNN | `SchemaGate`: α = sigmoid(logit), fully differentiable | `schema_gates.py` |
| 2 | All instances treated equally | ALL six methods | `InstanceGate`: β per face from current embeddings | `schema_gates.py` |
| 3 | No joint α+β optimization | ALL six methods | Single backward pass updates both α and β | `h2snn_model.py`, `trainer.py` |
| 4 | No redundancy control | LMSPS, MCTS-HGNN, PMMM, HG-Search, PM-HGNN | Jaccard greedy selection + redundancy regularization | `schema_discovery.py`, `discovery_losses.py` |
| 5 | Two-stage pipeline (mismatch) | ALL six methods | α and β are nn.Parameters in one training loop | `trainer.py` |
| 6 | Cannot refine expert knowledge | ALL six methods | manual/refine/auto modes; warm-start logits from hints | `config.py`, `schema_gates.py` |
| 7 | Accuracy-only schema scoring | ALL six methods | Topology score from label propagation; no training needed | `schema_discovery.py` |

---

## Mathematical summary of the full selection mechanism

Let:
- `σ` = a schema (typed walk, e.g. Author→Paper→Author)
- `I(σ)` = set of all path instances of schema σ in the graph
- `α_σ ∈ (0,1)` = schema gate (Problem 1)
- `β_f ∈ (0,1)` = instance gate for face `f` (Problem 2)
- `B_f` = coboundary matrix for face `f` (from the sheaf structure)

The **weighted Hodge-1 up-Laplacian** used in diffusion is:

```
L₁^up = Σ_σ  α_σ · Σ_{f ∈ I(σ)} β_f · B_f^T B_f
```

- `α_σ` and `β_f` are both **learned** (Problems 1–3)
- The schema pool is **diversity-penalized** to reduce redundancy in the Σ_σ sum (Problem 4)
- All of this is **trained end-to-end** in one loop (Problem 5)
- The initial schema pool is **topology-scored**, not accuracy-scored (Problem 7)
- Users can **seed the pool** with domain hints (Problem 6)

This formula is what makes H2S-NN's selection fundamentally different from every existing
method: the selection weights are not just coefficients in a weighted average — they directly
shape the linear operator that defines how information propagates through the entire graph.

---

## Regularization losses that drive selection behavior

All six terms are in `src/models/discovery_losses.py → compute_discovery_losses()`.

| Loss term | Formula | Effect |
|-----------|---------|--------|
| Schema sparsity | `λ₁ · mean(α)` | Pushes unused schemas toward α → 0 |
| Instance sparsity | `λ₂ · mean(β̄_σ)` | Pushes noisy faces toward β → 0 |
| Redundancy | `λ₃ · Σᵢⱼ αᵢαⱼ · overlap_{ij}` | Penalizes active schemas that cover same nodes |
| Coverage reward | `−λ₄ · Σ_σ α_σ · end_cov_σ` | Rewards schemas reaching diverse target nodes |
| Cost | `λ₅ · Σ_σ α_σ · length_σ · log(|I(σ)|)` | Penalizes large expensive active schemas |
| Gate entropy | `λ₆ · H(α)` | Encourages uncertainty early, sharpness later |

All six are added to the task loss and backpropagate into `SchemaGate.logits` and
`InstanceGate.net` every step.

---

## Pipeline overview: what runs before training vs. during training

```
BEFORE TRAINING (src/data/schema_discovery.py)
─────────────────────────────────────────────
Stage A  build_schema_graph()
         → type-graph adjacency: src_type → [(rel_id, rel_name, dst_type)]

Stage B  generate_schema_candidates()
         → DFS on type graph → up to max_candidates typed walks
         → SchemaCandidate(schema_id, name, node_types, relation_ids)

Stage C  estimate_schema_stats()
         → sample path instances per candidate
         → compute: endpoint_label_discrepancy, coverage, diversity, cost

Stage D  select_schema_pool()
         → greedy topology-aware selection of top-K diverse schemas
         → outputs topology_scores for gate initialization

Stage E  enumerate_discovered_faces()
         → materialize all concrete path instances for selected schemas
         → FaceGroup(face_node_index, face_edge_index, closure_edge_index)
         → DiscoveryMetadata(schema_features, topology_scores, instance_features)

DURING TRAINING (src/models/, src/trainers/)
────────────────────────────────────────────
Every forward pass:
  SchemaGate.forward()     → alpha [num_schemas]        (Problem 1)
  InstanceGate.forward()   → beta  [F_s] per group      (Problem 2)
  face_weights = alpha * beta                           (Problem 3)
  L1_up = Σ alpha_σ * Σ beta_f * B_f^T B_f             enters diffusion

Every backward pass:
  task_loss + discovery_losses → gradients into alpha, beta, model  (Problem 5)
  discovery_losses include redundancy, coverage, cost, entropy      (Problem 4)

Initialization (once at model build):
  SchemaGate.init_from_scores(topology_scores)          (Problem 7)
  optional: warm-start from user-provided schemas       (Problem 6)
```
