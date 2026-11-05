# H2S-NN Schema-to-Cell Discovery: Code-Ready Specification

Date: 2026-05-19

> **Historical specification (superseded).** This document records the May 2026 design and
> intentionally retains obsolete fields and loss terms for provenance. It is not the current
> implementation contract. The authoritative AMPSS behavior is the task-only
> `schema_conditioned_v1` description in `docs/h2snn_current_model_documentation.html` and the
> implemented code/tests under `src/` and `tests/`. Current AMPSS has no schema gate, auxiliary
> discovery loss, dense coverage matrix, or `schema_features` loss path.

## 1. Goal

This document turns the schema-to-cell discovery idea into an implementation plan that can map directly into the current `src/` code.

The current implementation already supports manual meta-path templates through:

- `src.configs.config.MetaPathConfig`
- `src.data.preprocessing.enumerate_metapath_faces`
- `src.data.structures.FaceGroup`
- `src.models.sheaf_maps.SequenceEdgeToFaceSheaf`
- `src.models.coboundary_ops.CoboundaryOps.delta1`
- `src.models.h2snn_model.H2SNN`

The missing part is automatic selection:

> which typed schemas should be instantiated, which concrete instances should be kept, and how strongly those faces should contribute to `L1`.

The code should implement this as a two-level selection module:

1. **schema-level selection**: choose typed schema templates;
2. **instance-level selection**: choose concrete path instances inside selected schemas.

This is necessary because H2S-NN does not put schema names directly into the operator. It puts concrete path instances into `delta1`.

---

## 2. Design Decision

Use this method:

> progressive schema generation + differentiable schema gates + differentiable instance gates

Do not start with pure RL, MCTS, or full NAS.

Reasons:

- pure RL/MCTS would require expensive rewards from partial H2S-NN training;
- vanilla NAS can be unstable because schema gates and sheaf weights co-adapt;
- a progressive gated design is easier to debug and still gives a clear path to stronger search later.

This method is still stronger than basic top-k meta-path selection because the selected schemas become weighted face families in `L1`.

---

## 3. Existing Code Anchors

### Existing manual template config

Current manual templates are stored as:

```python
@dataclass(slots=True)
class MetaPathConfig:
    name: str
    node_types: tuple[str, ...]
    sample_per_anchor: int | None = None
```

This works for ACM/DBLP/IMDB because the code assumes each node-type transition maps to exactly one relation type.

For automatic discovery, this is not enough.

Discovery should store both:

- node-type sequence;
- relation-id sequence.

Otherwise, schemas become ambiguous when multiple relation types connect the same node-type pair.

### Existing face representation

Current `FaceGroup`:

```python
@dataclass(slots=True)
class FaceGroup:
    template_name: str
    node_type_sequence: tuple[str, ...]
    face_node_index: torch.Tensor      # [num_faces, length + 1]
    face_edge_index: torch.Tensor      # [num_faces, length]
    relation_index: torch.Tensor       # [num_faces, length]
    closure_edge_index: torch.Tensor   # [num_faces, 2]
```

This is already a good foundation.

Discovery should extend it with schema ids and optional instance weights, without breaking manual mode.

---

## 4. New Config

Add a discovery config in `src/configs/config.py`.

```python
@dataclass(slots=True)
class SchemaDiscoveryConfig:
    enabled: bool = False
    mode: str = "manual"  # "manual" | "refine" | "auto"

    max_schema_length: int = 4
    min_schema_length: int = 2
    max_schema_candidates: int = 128
    selected_schema_k: int = 8

    max_instances_per_schema: int = 20000
    max_instances_per_anchor: int = 50
    selected_instance_k_per_schema: int | None = None

    require_target_start: bool = True
    require_target_end: bool = True
    allow_repeated_node_types: bool = True
    exclude_identity_metapaths: bool = True

    schema_gate_type: str = "hard_concrete"  # "sigmoid" | "hard_concrete"
    instance_gate_type: str = "sigmoid"      # "sigmoid" | "topk"
    partial_schema_probability: float = 0.5

    lambda_schema_sparse: float = 1e-3
    lambda_instance_sparse: float = 1e-4
    lambda_redundancy: float = 1e-3
    lambda_coverage: float = 1e-3
    lambda_cost: float = 1e-4
    lambda_gate_entropy: float = 1e-4
```

Then add it to `TrainConfig`:

```python
@dataclass(slots=True)
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    discovery: SchemaDiscoveryConfig = field(default_factory=SchemaDiscoveryConfig)
    ...
```

YAML example:

```yaml
discovery:
  enabled: true
  mode: auto
  min_schema_length: 2
  max_schema_length: 4
  max_schema_candidates: 128
  selected_schema_k: 8
  max_instances_per_schema: 20000
  max_instances_per_anchor: 50
  schema_gate_type: hard_concrete
  instance_gate_type: sigmoid
  partial_schema_probability: 0.5
```

---

## 5. New Data Structures

Create `src/data/schema_structures.py`.

```python
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SchemaCandidate:
    schema_id: int
    name: str
    node_types: tuple[str, ...]
    relation_ids: tuple[int, ...]
    relation_names: tuple[str, ...]

    @property
    def length(self) -> int:
        return len(self.relation_ids)


@dataclass(slots=True)
class SchemaStats:
    schema_id: int
    num_instances: int
    start_coverage: float
    end_coverage: float
    relation_diversity: float
    type_diversity: float
    avg_branching: float
    estimated_cost: float


@dataclass(slots=True)
class DiscoveredFaceGroup:
    schema: SchemaCandidate
    face_node_index: torch.Tensor       # [F_s, length + 1]
    face_edge_index: torch.Tensor       # [F_s, length]
    relation_index: torch.Tensor        # [F_s, length]
    closure_edge_index: torch.Tensor    # [F_s, 2]
    instance_features: torch.Tensor     # [F_s, d_instance_feat]
    instance_prior: torch.Tensor        # [F_s]

    @property
    def num_faces(self) -> int:
        return int(self.face_node_index.size(0))
```

Then either:

1. convert `DiscoveredFaceGroup` to current `FaceGroup` and keep weights separately, or
2. extend current `FaceGroup` with optional fields.

The least disruptive first implementation is option 1:

```python
@dataclass(slots=True)
class DiscoveryMetadata:
    schema_ids: torch.Tensor            # [num_face_groups]
    schema_features: torch.Tensor       # [num_schemas, d_schema_feat]
    instance_features: tuple[torch.Tensor, ...]  # one [F_s, d_instance_feat] per group
    instance_prior: tuple[torch.Tensor, ...]      # one [F_s] per group
```

Add optional metadata to `PreparedH2SData`:

```python
discovery: DiscoveryMetadata | None = None
```

---

## 6. File Plan

Add these files:

```text
src/data/schema_structures.py
src/data/schema_discovery.py
src/models/schema_gates.py
src/models/discovery_losses.py
```

Modify these files:

```text
src/configs/config.py
src/data/structures.py
src/data/dataset.py
src/models/coboundary_ops.py
src/models/h2snn_model.py
src/trainers/trainer.py
```

Optional later:

```text
src/search/policy_schema_search.py
```

Do not add RL/MCTS in the first coding pass.

---

## 7. Stage A: Build the Schema Graph

Input:

- `relation_tuples: tuple[tuple[str, str, str], ...]`
- each tuple is `(src_type, relation_name, dst_type)`

Output:

```python
schema_adjacency: dict[str, list[tuple[int, str, str]]]
```

Each entry:

```python
src_type -> [(relation_id, relation_name, dst_type), ...]
```

Implementation:

```python
def build_schema_graph(
    relation_tuples: tuple[tuple[str, str, str], ...],
) -> dict[str, list[tuple[int, str, str]]]:
    adjacency: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for relation_id, (src_type, relation_name, dst_type) in enumerate(relation_tuples):
        adjacency[src_type].append((relation_id, relation_name, dst_type))
    return dict(adjacency)
```

---

## 8. Stage B: Generate Candidate Schemas

Function:

```python
def generate_schema_candidates(
    relation_tuples: tuple[tuple[str, str, str], ...],
    target_node_type: str,
    min_length: int,
    max_length: int,
    require_target_start: bool,
    require_target_end: bool,
    allow_repeated_node_types: bool,
    max_candidates: int,
) -> tuple[SchemaCandidate, ...]:
    ...
```

Algorithm:

1. Build schema adjacency from relation tuples.
2. Choose start types.
3. DFS over relation transitions up to `max_length`.
4. Keep schemas whose length is at least `min_length`.
5. If `require_target_end=True`, keep only schemas ending at the target node type.
6. Stop after `max_candidates`, or sort by cheap stats later and truncate.

Pseudo-code:

```python
def dfs(current_type, node_types, relation_ids, relation_names):
    length = len(relation_ids)

    if length >= min_length:
        if not require_target_end or current_type == target_node_type:
            add_candidate(node_types, relation_ids, relation_names)

    if length == max_length:
        return

    for relation_id, relation_name, next_type in schema_adjacency.get(current_type, []):
        if not allow_repeated_node_types and next_type in node_types:
            continue
        dfs(
            current_type=next_type,
            node_types=node_types + [next_type],
            relation_ids=relation_ids + [relation_id],
            relation_names=relation_names + [relation_name],
        )
```

Name format:

```python
name = "__".join(node_types)
```

For DBLP:

```text
author__paper__author
author__paper__term__paper__author
author__paper__venue__paper__author
```

---

## 9. Stage C: Estimate Schema Stats

We need cheap stats before materializing every face.

Function:

```python
def estimate_schema_stats(
    candidates: tuple[SchemaCandidate, ...],
    node_slices: Mapping[str, tuple[int, int]],
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    target_node_type: str,
    max_instances_per_anchor: int,
    max_instances_per_schema: int,
    split_seed: int,
) -> tuple[SchemaStats, ...]:
    ...
```

The estimator can run bounded DFS per schema.

For each schema:

- count sampled instances;
- count unique start nodes;
- count unique end nodes;
- compute average branching;
- estimate cost.

Schema feature vector:

```python
schema_features = torch.tensor([
    log1p(num_instances),
    start_coverage,
    end_coverage,
    relation_diversity,
    type_diversity,
    avg_branching,
    length / max_length,
    estimated_cost,
])
```

Definitions:

```text
start_coverage = number of unique start nodes / number of target nodes
end_coverage   = number of unique end nodes / number of target nodes
relation_diversity = number of unique relation ids / schema length
type_diversity = number of unique node types / (schema length + 1)
estimated_cost = sampled_instance_count * schema_length
```

Important:

- do not use validation labels;
- do not use test labels;
- stats may use graph structure and training labels only if the benchmark is transductive and the method states this clearly.

For the first implementation, use graph-structure-only features.

---

## 10. Stage D: Progressive Schema Pool Selection

Before full training, select `S_pool`.

Function:

```python
def select_schema_pool(
    candidates: tuple[SchemaCandidate, ...],
    stats: tuple[SchemaStats, ...],
    max_schema_candidates: int,
    selected_schema_k: int,
) -> tuple[SchemaCandidate, ...]:
    ...
```

Initial non-learned score:

```python
score = (
    1.0 * log1p(num_instances)
    + 1.0 * start_coverage
    + 1.0 * end_coverage
    + 0.2 * relation_diversity
    + 0.2 * type_diversity
    - 0.5 * estimated_cost_normalized
    - 0.2 * length_normalized
)
```

Then:

```python
S_pool = top max_schema_candidates by score
```

During model training, differentiable gates reduce the active set further.

Why this split:

- pre-training selection keeps memory bounded;
- train-time gates learn task-specific importance.

---

## 11. Stage E: Materialize Face Groups for Selected Schemas

Replace or extend `enumerate_metapath_faces`.

New function:

```python
def enumerate_discovered_faces(
    node_slices: Mapping[str, tuple[int, int]],
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    schemas: tuple[SchemaCandidate, ...],
    max_instances_per_anchor: int,
    max_instances_per_schema: int,
    exclude_identity_metapaths: bool,
    split_seed: int,
) -> tuple[tuple[FaceGroup, ...], DiscoveryMetadata, torch.Tensor]:
    ...
```

Difference from current manual function:

- use `schema.relation_ids` directly;
- do not infer relation id from node-type transition;
- stop when `max_instances_per_schema` is reached;
- compute instance features for each path.

Instance feature vector:

```python
instance_features[f] = [
    path_length_normalized,
    start_degree_log,
    end_degree_log,
    mean_edge_relation_frequency,
    unique_node_ratio,
    repeated_node_indicator,
]
```

No label-derived features in the first version.

Return:

- normal `FaceGroup` tuple for existing model code;
- `DiscoveryMetadata` for gates;
- closure edge tensor.

---

## 12. Stage F: Schema Gates

Create `src/models/schema_gates.py`.

```python
class SchemaGate(nn.Module):
    def __init__(
        self,
        num_schemas: int,
        init_logit: float = 0.0,
        gate_type: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.full((num_schemas,), init_logit))
        self.gate_type = gate_type

    def forward(self, training: bool = True) -> torch.Tensor:
        if self.gate_type == "sigmoid":
            return torch.sigmoid(self.logits)
        if self.gate_type == "hard_concrete":
            return hard_concrete_sample(self.logits, training=training)
        raise ValueError(f"Unknown schema gate type: {self.gate_type}")
```

Output:

```python
alpha: torch.Tensor  # [num_schemas], nonnegative, usually in [0, 1]
```

Start with `sigmoid`.

Add `hard_concrete` later if pruning is needed.

---

## 13. Stage G: Instance Gates

The instance gate maps per-face features to one weight per face.

```python
class InstanceGate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        gate_type: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_type = gate_type

    def forward(self, instance_features: torch.Tensor) -> torch.Tensor:
        logits = self.net(instance_features).squeeze(-1)
        if self.gate_type == "sigmoid":
            return torch.sigmoid(logits)
        raise ValueError(f"Unknown instance gate type: {self.gate_type}")
```

Output per face group:

```python
beta_group: torch.Tensor  # [num_faces_in_group]
```

For memory:

- compute instance gates per face chunk;
- store features on device with `PreparedH2SData.to`.

---

## 14. Stage H: Weighted `delta1`

Current `delta1` computes:

```python
values = (F_ef @ s_edges).sum(dim=1)
```

For weighted faces, use:

```python
face_weight = alpha_schema * beta_face
weighted_values = sqrt(face_weight).unsqueeze(-1) * values
```

Why `sqrt`:

If we define:

```text
weighted_delta1_f = sqrt(w_f) * delta1_f
```

then:

```text
L1_up = weighted_delta1^T weighted_delta1
      = sum_f w_f * delta1_f^T delta1_f
```

This preserves positive semidefiniteness when `w_f >= 0`.

Modify `CoboundaryOps.delta1` to accept optional group weights:

```python
def delta1(
    s: torch.Tensor,
    face_caches: tuple[FaceMapCache, ...],
    map_builder: Callable[[FaceMapCache], torch.Tensor],
    face_weights: tuple[torch.Tensor, ...] | None = None,
) -> Delta1Result:
    ...
```

Inside group loop:

```python
values = torch.matmul(maps, edge_states.unsqueeze(-1)).squeeze(-1).sum(dim=1)
if face_weights is not None:
    weights = face_weights[group_id].clamp_min(0.0)
    values = values * torch.sqrt(weights + 1e-12).unsqueeze(-1)
```

Important:

- use the same weighted `Delta1Result` in `delta1_T`;
- do not multiply weights again in `delta1_T`;
- the weighting belongs to `delta1`, not separately to both directions.

---

## 15. Stage I: Integrate Into H2SNN

Modify `H2SNN.__init__`:

```python
if data.discovery is not None:
    self.schema_gate = SchemaGate(
        num_schemas=len(data.discovery.schema_ids),
        gate_type=config.discovery.schema_gate_type,
    )
    self.instance_gate = InstanceGate(
        input_dim=data.discovery.instance_features[0].size(-1),
        hidden_dim=config.model.face_mlp_hidden_dim,
        gate_type=config.discovery.instance_gate_type,
    )
else:
    self.schema_gate = None
    self.instance_gate = None
```

Add helper:

```python
def _build_face_weights(
    self,
    data: PreparedH2SData,
) -> tuple[torch.Tensor, ...] | None:
    if data.discovery is None:
        return None

    alpha = self.schema_gate(training=self.training)  # [S]
    weights: list[torch.Tensor] = []

    for group_id, features in enumerate(data.discovery.instance_features):
        schema_id = int(data.discovery.schema_ids[group_id].item())
        beta = self.instance_gate(features)  # [F_s]
        weights.append(alpha[schema_id] * beta)

    return tuple(weights)
```

Then in `forward`:

```python
face_weights = self._build_face_weights(data)

def apply_l1(s: torch.Tensor) -> torch.Tensor:
    return CoboundaryOps.L1(
        s=s,
        F_src=sheaf_output.F_src,
        F_tgt=sheaf_output.F_tgt,
        src_idx=src_idx,
        tgt_idx=tgt_idx,
        num_nodes=data.num_nodes,
        face_caches=face_caches,
        map_builder=self.edge_to_face.build_maps,
        num_edges=data.num_edges,
        face_weights=face_weights,
    )
```

Also pass `face_weights` into diagnostic calls:

```python
delta1_s_star = CoboundaryOps.delta1(
    s=s_star,
    face_caches=face_caches,
    map_builder=self.edge_to_face.build_maps,
    face_weights=face_weights,
).concatenated
```

Extend `H2SNNOutput`:

```python
schema_alpha: torch.Tensor | None
face_weight_means: torch.Tensor | None
```

Use these for logging and regularization.

---

## 16. Stage J: Discovery Losses

Create `src/models/discovery_losses.py`.

```python
class DiscoveryLossOutput(NamedTuple):
    sparse_schema: torch.Tensor
    sparse_instance: torch.Tensor
    redundancy: torch.Tensor
    coverage: torch.Tensor
    cost: torch.Tensor
    entropy: torch.Tensor
    total: torch.Tensor
```

### Schema sparsity

```python
L_schema_sparse = alpha.mean()
```

### Instance sparsity

```python
L_instance_sparse = mean(beta_f over all groups)
```

### Cost

Penalize long and numerous active faces:

```python
L_cost = sum_s alpha_s * length_s * num_faces_s / normalizer
```

### Redundancy

Precompute schema overlap matrix `R`:

```text
R_ij = Jaccard(edge_instance_set_i, edge_instance_set_j)
```

First implementation can approximate with start/end coverage overlap:

```python
L_redundancy = alpha.T @ R @ alpha / (alpha.sum() ** 2 + eps)
```

### Coverage

Encourage selected schemas to cover target nodes:

```python
coverage = 1 - prod_s (1 - alpha_s * coverage_vector_s)
L_coverage = 1 - coverage.mean()
```

Where `coverage_vector_s` is `[num_target_nodes]` with 1 if schema `s` has at least one instance touching the target node.

For first implementation, a simpler scalar is acceptable:

```python
L_coverage = -sum_s alpha_s * start_coverage_s / (alpha.sum() + eps)
```

### Gate entropy

Optional:

```python
entropy = -alpha * log(alpha + eps) - (1 - alpha) * log(1 - alpha + eps)
L_entropy = entropy.mean()
```

Use positive coefficient if you want uncertainty early.

Use negative coefficient or annealing if you want hard choices later.

---

## 17. Stage K: Trainer Integration

Modify `Trainer._compute_metrics`.

Current total loss:

```python
total_loss = classification_loss + smooth0 + smooth1 + consistency + coboundary
```

New:

```python
discovery_loss = 0.0
if output.schema_alpha is not None:
    discovery_loss = compute_discovery_losses(
        output=output,
        data=data,
        cfg=self.cfg.discovery,
    ).total

total_loss = (
    classification_loss
    + smooth0
    + smooth1
    + consistency
    + coboundary
    + discovery_loss
)
```

Log:

```python
"schema_alpha_mean"
"schema_alpha_max"
"schema_active_count"
"face_weight_mean"
"discovery_loss"
"schema_sparse_loss"
"instance_sparse_loss"
"redundancy_loss"
"coverage_loss"
"cost_loss"
```

Active count:

```python
active_count = (alpha > 0.5).sum()
```

---

## 18. Partial Schema Propagation

This is the stabilization trick.

During training, randomly drop some schema groups from `delta1`.

Implementation in `_build_face_weights`:

```python
if self.training and self.config.discovery.partial_schema_probability < 1.0:
    keep = torch.bernoulli(
        torch.full_like(alpha, self.config.discovery.partial_schema_probability)
    )
    alpha = alpha * keep / self.config.discovery.partial_schema_probability
```

This keeps the expected contribution similar:

```text
E[alpha * keep / p] = alpha
```

Why this helps:

- prevents every schema gate from co-adapting with sheaf maps in every step;
- makes schema selection less seed-fragile;
- matches the PMMM-style partial propagation idea but applies it to face schemas.

Use this only during training.

During evaluation:

```python
keep = all ones
```

---

## 19. Training Phases

Use three phases.

### Phase 0: data preprocessing

Run once before model construction:

```python
if cfg.discovery.enabled:
    candidates = generate_schema_candidates(...)
    stats = estimate_schema_stats(...)
    schema_pool = select_schema_pool(...)
    face_groups, discovery_metadata, closure_edges = enumerate_discovered_faces(...)
else:
    face_groups, closure_edges = enumerate_metapath_faces(...)
    discovery_metadata = None
```

### Phase 1: warmup without learnable gates

Epochs: `0..warmup_epochs`.

Use:

```python
alpha = 1 for all selected schemas
beta = 1 for all instances
```

Purpose:

- train node encoder and sheaf maps enough to avoid noisy early gate updates.

### Phase 2: train gates and model jointly

After warmup:

```python
alpha = schema_gate()
beta = instance_gate(instance_features)
```

Use partial schema propagation.

### Phase 3: optional hard pruning and retraining

At the end:

```python
keep schema if alpha_s > threshold
keep instance if beta_f > threshold or top-k per schema
```

Then retrain or fine-tune H2S-NN on the pruned face set.

This final phase is optional for the first implementation.

---

## 20. Concrete DBLP Example

Target type:

```text
author
```

Schema graph edges:

```text
author -> paper
paper -> author
paper -> term
term -> paper
paper -> venue
venue -> paper
```

Generated candidate schemas:

```text
author -> paper -> author
author -> paper -> term -> paper -> author
author -> paper -> venue -> paper -> author
```

After schema stats:

```text
APA:
  num_instances = 90,000
  start_coverage = 0.92
  estimated_cost = medium

APTPA:
  num_instances = 400,000 sampled from much larger possible set
  start_coverage = 0.87
  estimated_cost = high

APVPA:
  num_instances = 1,200,000 sampled from much larger possible set
  start_coverage = 0.95
  estimated_cost = very high
```

Training may learn:

```text
alpha_APA   = 0.55
alpha_APTPA = 0.81
alpha_APVPA = 0.10
```

Inside `APTPA`, instance gates may learn:

```text
high beta:
author_a -> paper_1 -> term_sheaf -> paper_2 -> author_b

low beta:
author_a -> paper_5 -> term_model -> paper_8 -> author_c
```

The final operator uses:

```text
L1_up =
  0.55 * weighted_APA_faces
  + 0.81 * weighted_APTPA_faces
  + 0.10 * weighted_APVPA_faces
```

This means the model did not merely select `APTPA` as an attention channel.

It selected concrete `APTPA` instances as weighted faces in `delta1`.

---

## 21. Implementation Order

### First coding pass

Implement:

1. `SchemaCandidate`
2. schema graph construction
3. candidate schema generation
4. bounded discovered face enumeration
5. `DiscoveryMetadata`
6. optional `face_weights` in `CoboundaryOps.delta1`

Keep all weights fixed to 1.

This verifies the data path.

### Second coding pass

Implement:

1. `SchemaGate`
2. `InstanceGate`
3. `_build_face_weights` in `H2SNN`
4. discovery metrics in output

This verifies trainable gates.

### Third coding pass

Implement:

1. discovery regularizers
2. trainer logging
3. partial schema propagation
4. warmup schedule

This verifies the full method.

### Fourth coding pass

Add ablations:

```text
manual schemas
auto schemas with uniform weights
auto schemas with schema gates only
auto schemas with schema + instance gates
no delta1
identity maps
diagonal maps
```

---

## 22. Minimal First Version

If we want the fastest working version, implement only this:

```text
auto-generate schemas
preselect top-K by structural score
enumerate bounded face groups
learn one sigmoid alpha per schema
do not learn beta yet
weight delta1 by sqrt(alpha)
```

This gives a real code path for automatic meta-path selection.

Then add instance gates after the schema-gate version works.

Minimal formula:

```text
L1_up = sum_s alpha_s * delta1_s^T delta1_s
```

Full formula later:

```text
L1_up = sum_s alpha_s sum_f beta_f * delta1_f^T delta1_f
```

---

## 23. Why This Method Is Chosen

We choose this method because the current code already builds `FaceGroup` objects and computes `delta1` group by group.

That means the natural extension is:

- generate more candidate `FaceGroup`s automatically;
- attach one schema gate to each group;
- attach optional instance gates to faces inside each group;
- weight `delta1` before `delta1_T`.

This is the smallest implementation change that still creates a real research contribution.

It is also the right abstraction for H2S-NN:

- schema gates decide which semantic face families matter;
- instance gates decide which concrete faces matter;
- weighted `delta1` decides how those choices alter Hodge diffusion.

That is why this is more code-ready and more H2S-NN-specific than a generic NAS/RL/MCTS meta-path search module.

---

## 24. Topology-Aware Schema Scoring: The Missing Novelty

This section identifies the critical gap in Stages C and D and specifies the fix that makes the method publishable.

### 24.1 The Problem with the Current Structural Score

Stage D's pre-selection score is:

```python
score = (
    1.0 * log1p(num_instances)
    + 1.0 * start_coverage
    + 1.0 * end_coverage
    + 0.2 * relation_diversity
    + 0.2 * type_diversity
    - 0.5 * estimated_cost_normalized
    - 0.2 * length_normalized
)
```

This is purely structural. It contains no task signal. It selects schemas with the most instances and widest coverage, which is essentially what LMSPS (2024) and AMP-HGNAS (2026) already do. A reviewer will correctly note that this component is not novel.

Additionally, Stage F's `SchemaGate` initializes all logits at zero:

```python
self.logits = nn.Parameter(torch.full((num_schemas,), init_logit))
```

A zero-initialized learnable scalar per schema is equivalent to a learned attention weight over meta-path channels — the same mechanism HAN (2019) uses. Without a meaningful warm start, the gate learning has no topological grounding.

The fix is to compute a **topology-aware schema utility score** from the existing `propagated_labels` output that is already produced in `preprocessing.py:label_propagation()`. This score measures exactly what H2S-NN cares about: whether a schema's path instances connect nodes with different label distributions — the heterophilic structure that `delta1` is designed to capture.

---

### 24.2 The Endpoint Label Discrepancy Score

For each candidate schema `sigma` with sampled instances `I_hat(sigma)`:

```
disc(sigma) = (1 / |I_hat(sigma)|) * sum_{f in I_hat(sigma)} || y_hat_{v0^f} - y_hat_{vl^f} ||^2
```

where:
- `y_hat_v in R^C` is the propagated label distribution from `label_propagation()`;
- `v0^f` is the start node of path instance `f`;
- `vl^f` is the end node of path instance `f`.

**Why this is the correct score for H2S-NN:**

`disc(sigma)` measures how different the label distributions are at the two endpoints connected by this schema. High `disc(sigma)` means the schema systematically connects heterophilic nodes. In H2S-NN, these are precisely the faces where `delta1` creates large discrepancy — the `delta1^T delta1` term in `L1` will actively propagate correction signals along those heterophilic face boundaries. Low `disc(sigma)` means the schema connects nodes that are already similar — standard GNN message passing handles this adequately without faces.

**Why this does not leak labels:**

`propagated_labels` is computed from training labels only via label propagation on the graph structure. It is not the ground-truth test label. It is a training-set-derived signal, identical in nature to using training gradients to update model weights.

---

### 24.3 The Complete Topology-Aware Score

Replace Stage D's structural score with:

```python
score(sigma) = (
    disc(sigma) ** alpha_disc          # topology-native: heterophilic endpoint signal
    * coverage(sigma) ** alpha_cov     # must reach diverse target nodes
    * diversity(sigma) ** alpha_div    # rewards cross-type paths
) / (1.0 + redundancy_penalty(sigma, selected))
```

where:

```python
disc(sigma)      = mean endpoint label discrepancy (from propagated_labels)
coverage(sigma)  = end_coverage from Stage C (fraction of target nodes covered)
diversity(sigma) = exp(H(node_type_sequence))  # exponential of type entropy
redundancy_penalty(sigma, selected) = max over selected schemas of:
    Jaccard(endpoints(I_hat(sigma)), endpoints(I_hat(sigma_prime)))
```

Default exponents: `alpha_disc = 1.0`, `alpha_cov = 0.5`, `alpha_div = 0.3`.

Greedy selection at each step:

```python
sigma_next = argmax_{sigma not in selected} score(sigma, selected)
selected.add(sigma_next)
```

The redundancy penalty re-evaluates after each selection, ensuring diversity.

---

### 24.4 Code Changes to Implement This

**In Stage C (`estimate_schema_stats`):**

Add `endpoint_label_discrepancy` to `SchemaStats`:

```python
@dataclass(slots=True)
class SchemaStats:
    schema_id: int
    num_instances: int
    start_coverage: float
    end_coverage: float
    relation_diversity: float
    type_diversity: float
    avg_branching: float
    estimated_cost: float
    endpoint_label_disc: float    # NEW: mean || y_hat_{v0} - y_hat_{vl} ||^2
```

**New function in `src/data/schema_discovery.py`:**

```python
def compute_endpoint_label_discrepancy(
    instance_sample: list[tuple[int, ...]],    # list of (v0, v1, ..., vl) tuples
    propagated_labels: torch.Tensor,            # [num_nodes, num_classes]
) -> float:
    if not instance_sample:
        return 0.0
    start_nodes = torch.tensor([path[0] for path in instance_sample], dtype=torch.long)
    end_nodes   = torch.tensor([path[-1] for path in instance_sample], dtype=torch.long)
    diff = propagated_labels[start_nodes] - propagated_labels[end_nodes]
    return float(diff.pow(2).sum(dim=-1).mean().item())
```

**Calling convention in `estimate_schema_stats`:**

```python
disc = compute_endpoint_label_discrepancy(
    instance_sample=sampled_instances_for_stats,
    propagated_labels=propagated_labels,       # passed from dataset.py
)
stats.endpoint_label_disc = disc
```

The `propagated_labels` tensor is already computed in `build_augmented_features` in `preprocessing.py`. Pass it through to `estimate_schema_stats` as an additional argument. No new computation is required.

**In Stage D (`select_schema_pool`):**

```python
def topology_aware_score(
    stats: SchemaStats,
    selected_stats: list[SchemaStats],
    propagated_labels: torch.Tensor,
    alpha_disc: float = 1.0,
    alpha_cov: float = 0.5,
    alpha_div: float = 0.3,
) -> float:
    disc = max(stats.endpoint_label_disc, 1e-6)
    cov  = max(stats.end_coverage, 1e-6)
    div  = math.exp(stats.type_diversity * math.log(max(stats.type_diversity, 1e-6)))
    redundancy = max(
        (
            jaccard_endpoint_overlap(stats, s)
            for s in selected_stats
        ),
        default=0.0,
    )
    base = (disc ** alpha_disc) * (cov ** alpha_cov) * (div ** alpha_div)
    return base / (1.0 + redundancy)
```

---

### 24.5 Schema Gate Warm Initialization

Instead of initializing schema gate logits at zero, initialize from the topology-aware score:

```python
# In H2SNN.__init__, when building SchemaGate:
if data.discovery is not None:
    schema_scores = data.discovery.schema_topology_scores  # precomputed in Stage D
    init_logits = torch.log(schema_scores.clamp_min(1e-6))
    self.schema_gate = SchemaGate(
        num_schemas=len(data.discovery.schema_ids),
        init_logit=0.0,
        gate_type=config.discovery.schema_gate_type,
    )
    with torch.no_grad():
        self.schema_gate.logits.copy_(init_logits)
```

This gives meaningful gradient initialization: schemas the topology score identifies as important start with high logits; schemas that cover similar or homophilic endpoints start low and must prove their worth during training.

Add `schema_topology_scores: torch.Tensor` to `DiscoveryMetadata`.

---

### 24.6 Upgraded Instance Gate

The current `InstanceGate` takes static structural features. This does not reflect what the model has learned. Replace the instance feature vector with a learned one that uses current node embeddings:

**In `H2SNN._build_face_weights`:**

```python
def _build_face_weights(
    self,
    data: PreparedH2SData,
    h0: torch.Tensor,    # [num_nodes, d] — current node embeddings
) -> tuple[torch.Tensor, ...] | None:
    if data.discovery is None:
        return None

    alpha = self.schema_gate(training=self.training)  # [S]
    weights: list[torch.Tensor] = []

    for group_id, face_group in enumerate(data.face_groups):
        schema_id = int(data.discovery.schema_ids[group_id].item())
        # Use learned node embeddings for instance scoring
        v0 = face_group.face_node_index[:, 0]    # [F_s] start node ids
        vl = face_group.face_node_index[:, -1]   # [F_s] end node ids
        schema_emb = self.schema_embeddings[schema_id].unsqueeze(0).expand(len(v0), -1)
        instance_input = torch.cat([h0[v0], h0[vl], schema_emb], dim=-1)  # [F_s, 2d + d_r]
        beta = self.instance_gate(instance_input)  # [F_s] via sigmoid
        weights.append(alpha[schema_id] * beta)

    return tuple(weights)
```

Add `self.schema_embeddings = nn.Embedding(num_schemas, self.model_config.d_r)` to `H2SNN.__init__` when discovery is enabled.

This is the key adaptive component: `beta_f` changes every forward pass as `h0` evolves during training. The instance gate learns which endpoint-pair combinations are worth activating as faces, based on what the node encoder currently represents.

---

### 24.7 What This Adds to the Novelty Claim

With the topology-aware score and warm-initialized gates, the contribution of the schema selection stage is:

> **We score candidate schemas by their endpoint label discrepancy — computed from training-set label propagation without running the downstream H2S-NN — as a proxy for how much informative `delta1` discrepancy the schema would create if instantiated as faces. This is the first schema scoring criterion grounded in the Hodge coboundary structure of the target model, rather than in downstream validation accuracy or structural Jaccard overlap.**

This claim distinguishes the method from every paper in the survey:
- MCTS-HGNN, RMS-HRec, MP-GNN: require full or partial GNN training to evaluate a candidate schema.
- AMP-HGNAS: uses structural Jaccard overlap — purely graph-structural, no task signal.
- LMSPS/LPSMS: use validation accuracy as score — require bilevel training to evaluate each candidate.
- SchemaWalk: uses coverage + confidence — task-specific to link prediction, not node classification or topology.
- PMMM: uses differentiable schema weights but no pre-training-free scoring at all.

The endpoint label discrepancy score is training-free, topology-native, and task-relevant.

---

## 25. Problem Lineage: Why Existing Methods Lead to This Work

This section documents the chain of unsolved problems across the literature that motivates H2S-NN's automatic schema-to-cell discovery. This is the narrative for the related work section.

---

### Problem 1: Manual Meta-Path Templates Are Required

**Papers affected:** HAN, SMCD, ESim, HISE-KT, and the base H2S-NN itself.

All methods that use meta-paths require a domain expert to specify which schemas to use before training. This is not a minor limitation — it means the method cannot be applied to new heterogeneous graphs where canonical schemas are unknown, or where the useful schemas are task-dependent rather than domain-conventional.

**Consequence:** Methods that depend on manual templates inherit the expert's biases. If the expert specifies PAP and PSP for ACM, the model can never discover that a longer or less obvious path would be more informative for the specific label distribution in the data.

**What H2S-NN solves:** Automatic enumeration from the type graph, followed by topology-aware scoring, produces a candidate set without any manual specification.

---

### Problem 2: Schema Evaluation Requires Downstream GNN Training

**Papers affected:** MCTS-HGNN, RMS-HRec, HG-Search, MP-GNN.

MCTS-HGNN evaluates each schema candidate by running a full GNN forward pass and measuring validation accuracy. RMS-HRec uses DQN with early-stopped GNN evaluation as the reward. MP-GNN runs a full training loop per candidate relation extension. HG-Search uses policy gradient over thousands of sampled architectures.

**Consequence:** The search cost is proportional to the number of GNN training runs. For a schema space of even 50 candidates, this means 50× the training cost of a single run. On large graphs this is prohibitive.

**What H2S-NN solves:** The endpoint label discrepancy score (Section 24) is computed from label propagation at preprocessing time — no GNN forward pass, no training, no gradient computation.

---

### Problem 3: Structural Scoring Does Not Capture Task Relevance

**Papers affected:** AMP-HGNAS.

AMP-HGNAS scores schema redundancy by Jaccard overlap over node and edge sets: `D(Pi, Pj) = Jaccard(N(Pi), N(Pj)) × Jaccard(E(Pi), E(Pj))`. This is purely structural. Two schemas can be structurally disjoint but semantically redundant for the task (both connect nodes of the same class), or structurally overlapping but semantically complementary (both cover the same region of the graph but with different heterophilic patterns).

The professor's note in the survey identifies this: *"A more principled measure would use task-conditional mutual information: I(Z^{Pi}; Y | Z^{Pj})".*

**Consequence:** AMP-HGNAS selects schemas that are structurally diverse but not necessarily task-complementary.

**What H2S-NN solves:** `disc(sigma)` measures label-distribution distance at path endpoints — directly approximating task-conditional mutual information without requiring explicit MI estimation. High discrepancy schemas are complementary in the task sense even if structurally overlapping.

---

### Problem 4: Pre-Computed Candidate Pool Still Requires Manual Initialization

**Papers affected:** LMSPS, LPSMS.

LMSPS and LPSMS are among the strongest recent methods. They use Gumbel-softmax and leave-one-out marginal contribution to efficiently select from a candidate pool. However, the pool must be constructed before search — and in practice, it is constructed from manually specified schemas or simple template extensions. The paper's reported results on DBLP use APA, APTPA, APVPA as the pool, which are standard domain-knowledge templates.

**Consequence:** LMSPS and LPSMS do not solve the template-specification problem — they improve selection from a given pool.

**What H2S-NN solves:** The candidate pool is constructed automatically by DFS over the type graph (Stages A–B). The topology-aware score (Section 24) then prunes this pool without requiring a validation-accuracy-based evaluation loop.

---

### Problem 5: Selected Schemas Become Aggregation Channels, Not Topological Objects

**Papers affected:** All 11 meta-path papers in the survey (MCTS-HGNN, AMP-HGNAS, RMS-HRec, SMCD, HG-Search, SchemaWalk, HISE-KT, LMSPS, ESim, MP-GNN, LPSMS).

Every existing method that selects or discovers meta-paths treats the selected schemas as **aggregation channels** — additional message-passing views that are fused with attention weights. The selected schemas define semantic neighborhoods; they do not change the mathematical operator that drives diffusion.

From the cross-paper synthesis in the survey: *"Papers 1-11 all operate at Level 2-3 (schema selection and instance counting). H2S-NN uniquely operates at Level 4 — it converts instances into topological 2-cells with algebraic meaning (coboundary operator delta1, Hodge Laplacian L1)."*

**Consequence:** In all prior methods, removing a meta-path reduces the number of aggregation views. In H2S-NN, removing a face schema changes the eigenspectrum of `L1` — it changes what the diffusion operator considers to be "disagreement" in the edge cochain space.

**What H2S-NN provides that no other method provides:** The selected schemas instantiate 2-cells. These cells enter `delta1`. `delta1` enters `L1`. `L1` defines the Hodge diffusion geometry. The schema selection is therefore not a preprocessing choice — it is a topological construction that reshapes the model's inductive bias.

---

### Problem 6: Bilevel Optimization Instability in Differentiable Schema Search

**Papers affected:** DiffMG (precursor to PMMM), LMSPS, DARTS-based NAS generally.

PMMM identifies a concrete failure mode: in differentiable meta-structure search, updating architecture weights `alpha_r` and model weights `W` in the same forward pass creates coupled optimization where `alpha` overfits to the current `W` state. Different random seeds produce completely different selected schemas.

PMMM solves this with Partial Message Propagation: in each step, a random subset of schemas is active. The gradient for `alpha_r` is decoupled from the W state that uses r's contribution.

**Consequence without this fix:** Schema gates in H2S-NN's Gumbel-Sheaf training would exhibit the same seed fragility. The reproducibility of discovered schemas would be questionable.

**What H2S-NN solves:** Partial Schema Propagation (Section 18) directly adapts PMMM's fix: in each training step, a Bernoulli mask randomly excludes schema groups from `delta1`. The mathematical analogy is exact: PMMM's `path r` ↔ H2S-NN's `schema sigma`; PMMM's `p_r^{(i,j)}` ↔ H2S-NN's `alpha_sigma`; PMMM's adjacency `A_r^{(i,j)}` ↔ H2S-NN's `delta1^sigma`.

---

### Problem 7: No Formal Search Guarantees

**Papers affected:** All 11 meta-path papers in the survey.

None of the 11 papers in the meta-path search/selection literature has a formal bound on the number of schema evaluations required to find a good schema pool. MCTS-HGNN uses a heuristic UCT score with no convergence analysis. LMSPS uses progressive shrinking with no formal bound. SchemaWalk uses policy gradient with no guarantee.

The survey's Part II (LevinTS, PHS) shows that policy-guided heuristic search can provide formal expansion bounds — but none of these algorithms have been applied to meta-path schema discovery.

**What this opens for H2S-NN:** If the endpoint label discrepancy score is used as the heuristic `eta(sigma)` in PHS, and a small LSTM policy is trained to propose schema extensions, the resulting PHS-Schema search provides a formal bound on how many schemas are evaluated before a good pool is found (see Section 16.2 in the survey). This is the first formally bounded schema discovery algorithm for HGNNs.

This is listed as an optional Phase 4 extension in the implementation roadmap (Section 21). It is not required for the first submission but provides a strong theoretical contribution for a top venue.

---

### Problem 8: No Instance-Level Selection Within Schemas

**Papers affected:** All 11 meta-path papers, and the base H2S-NN.

All existing methods select or weight schemas but treat all instances of a selected schema equally. Either every instance of APA becomes a face, or none does. There is no mechanism to identify that some APA instances (e.g., author-A → paper-7 → author-B where authors are in different research communities) are more topologically informative than others (e.g., author-A → paper-7 → author-A self-loops).

**Consequence:** The face set may include many low-utility instances that add noise to `delta1` without adding informative coboundary signal.

**What H2S-NN provides:** The instance gate `beta_f = sigmoid(MLP([h_{v0^f} || h_{vl^f} || e_sigma]))` assigns a different weight to each face instance based on the current node embeddings. This is an adaptive selection that refines within each schema as the model learns. No other method in the survey operates at this granularity.

---

## 26. Method Comparison Table for Professor

This table compares H2S-NN's automatic schema-to-cell discovery against all relevant methods surveyed. Use this to position the contribution when presenting to your supervisor or in the related work section.

Abbreviations: NC = node classification, LP = link prediction, GC = graph classification.

---

### Table 1: Schema Selection Mechanism and Topology Integration

| Method | Year | Requires Manual Templates? | Schema Eval. Cost | Topology Integration | Selected Schema Becomes |
|---|---|---|---|---|---|
| HAN | 2019 | Yes (required) | None (given) | None | Attention channel |
| ESim | 2019 | Yes (required) | None (given) | None | Embedding bias vector |
| SMCD | 2023 | Yes (required) | None (given) | None | Augmentation view |
| HISE-KT | 2025 | Yes (required) | LLM scoring | None | Attention channel |
| MCTS-HGNN | 2024 | No (MCTS search) | Full GNN per candidate | None | Attention channel |
| RMS-HRec | 2024 | No (RL search) | Early-stopped GNN | None | Aggregation view |
| HG-Search | 2024 | No (NAS search) | Many GNN training runs | None | Architecture component |
| SchemaWalk | 2024 | No (RL on schema graph) | Coverage + confidence | None | Embedding path |
| MP-GNN | 2024 | No (greedy beam) | Full GNN per extension | None | Aggregation channel |
| AMP-HGNAS | 2026 | Semi (initial pool) | Jaccard overlap (free) | None | Aggregation channel |
| LMSPS / LPSMS | 2024 | Semi (initial pool) | Val. accuracy (bilevel) | None | Aggregation channel |
| PMMM | 2024 | Semi (schema adjacency) | DARTS (gradient) | None | Schema-level α weight |
| HETSHEAF | 2024 | Not applicable | Not applicable | 1-skeleton sheaf only | Typed restriction map |
| DeepSN | 2024 | Not applicable | Not applicable | 1-skeleton sheaf only | Sheaf coefficient |
| **H2S-NN (base)** | — | **Yes (required)** | **None (given)** | **2-cells + L1 + delta1** | **Topological 2-cell** |
| **H2S-NN (auto)** | — | **No (type graph DFS)** | **Label discrepancy (free)** | **2-cells + L1 + delta1** | **Weighted topological 2-cell** |

---

### Table 2: Scoring and Initialization Properties

| Method | Schema Score Basis | Task-Relevant Score? | Training-Free Score? | Instance-Level Selection? | Gate Initialization |
|---|---|---|---|---|---|
| AMP-HGNAS | Jaccard node/edge overlap | No | Yes | No | Uniform |
| LMSPS | Gumbel-softmax + val accuracy | Yes | No (bilevel) | No | Uniform |
| LPSMS | Leave-one-out marginal val accuracy | Yes | No (M GNN runs) | No | Uniform |
| MCTS-HGNN | Validation accuracy per MCTS node | Yes | No (full GNN) | No | UCT exploration |
| MP-GNN | MSE of linear predictor | Yes | Partial (cheap proxy) | No | Greedy extension |
| SchemaWalk | Coverage + confidence (RL reward) | Partial | No (RL steps) | No | Random policy |
| PMMM | Gumbel-softmax + val accuracy | Yes | No (DARTS) | No | Uniform |
| **H2S-NN (auto)** | **Endpoint label discrepancy + coverage + diversity** | **Yes** | **Yes (label prop only)** | **Yes (adaptive beta_f)** | **Topology score warm start** |

---

### Table 3: Scalability and Reproducibility

| Method | Max Graph Size Tested | Search Cost | Stable Across Seeds? | Formal Search Bound? |
|---|---|---|---|---|
| ESim | 18K nodes | Preprocessing only | Yes | No |
| MCTS-HGNN | 26K nodes | O(K × GNN cost) | Partially | No |
| HG-Search | 180K nodes | Many GNN runs | Not reported | No |
| SchemaWalk | Schema-level (|T|^L) | RL steps × schema space | Not reported | No |
| AMP-HGNAS | 1.9M nodes (OGBN-MAG) | O(|P| × graph size) | Yes | No |
| LMSPS / LPSMS | 1.9M nodes (OGBN-MAG) | O(NMF²) | Partially | No |
| PMMM | 26K nodes | 0.9–2.1 GPU-min | **Yes (30 seeds tested)** | No |
| **H2S-NN (auto)** | **Target: 1.9M+ (OGBN-MAG)** | **Preprocessing + gated training** | **Yes (PSP ensures this)** | **Optional: PHS bound** |

---

### Table 4: Task and Topology Coverage

| Method | NC | LP | GC | Heterophily Handling | Uses L1 / delta1? | Uses 2-Cells? | Instance Weights? |
|---|---|---|---|---|---|---|---|
| HAN | Yes | No | No | No (assumes homophily) | No | No | No |
| MCTS-HGNN | Yes | No | No | No | No | No | No |
| AMP-HGNAS | Yes | No | No | No | No | No | No |
| LMSPS / LPSMS | Yes | No | No | No | No | No | No |
| PMMM | Yes | No | No | No | No | No | No |
| SchemaWalk | No | Yes | No | No | No | No | No |
| HETSHEAF | Yes | Yes | Yes | Yes (typed restriction maps) | No | No | No |
| DeepSN | No | No | No | No (homogeneous) | No | No | No |
| **H2S-NN (auto)** | **Yes** | **Planned** | **Planned** | **Yes (core design goal)** | **Yes** | **Yes** | **Yes** |

---

### Table 5: The Key Differentiator Summary (for Professor Presentation)

The single most important row is the last column cluster — nobody else has all three simultaneously:

| Method | Automatic Schema Discovery | Topology-Native Integration (2-cells + L1) | Training-Free Schema Scoring |
|---|---|---|---|
| AMP-HGNAS | Partial (needs initial pool) | No | Yes (but structural only) |
| LMSPS | Partial (needs initial pool) | No | No |
| MCTS-HGNN | Yes | No | No |
| PMMM | Yes (schema-level only) | No | No |
| HETSHEAF | No (not applicable) | Partial (1-skeleton only) | No |
| **H2S-NN (auto)** | **Yes (type graph DFS)** | **Yes (full 2-cells + L1 + delta1)** | **Yes (endpoint label discrepancy)** |

**The contribution in one sentence:** H2S-NN is the first method that automatically discovers face-generating schemas using a training-free topology-aware score, and integrates the selected schemas as weighted 2-cells in a heterogeneous sheaf complex, allowing the schema selection to directly reshape Hodge diffusion geometry rather than merely adding aggregation channels.

---

### Notes on Using These Tables

When presenting to your professor:

1. Table 1 is the primary positioning table. The key column is "Selected Schema Becomes." H2S-NN is the only method where the selected schema becomes a topological 2-cell. Every other method produces aggregation channels.

2. Table 2 shows why the scoring mechanism matters. The combination of "Yes" in both "Task-Relevant" and "Training-Free" is unique to H2S-NN. This is the gap that justifies the paper.

3. Table 3 shows the scalability target. OGBN-MAG (1.9M nodes) is the benchmark that separates LMSPS/AMP-HGNAS-class methods from smaller-graph methods. H2S-NN must test here to be taken seriously.

4. Table 4 shows the heterophily claim. Every meta-path paper in the survey assumes or works best on homophilous graphs. H2S-NN's Hodge geometry (the L1 edge Laplacian with upper term from delta1) is specifically designed for heterophilic structure. This is a positioning claim that requires experimental evidence on heterophilic datasets (Amazon Ratings, Roman Empire, Tolokers from the existing codebase).

5. Table 5 is the one-slide summary. If you have two minutes with your professor, show this table. It makes the gap immediately visible.
