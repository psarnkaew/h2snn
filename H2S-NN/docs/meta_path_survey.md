# Survey: Automatic Meta-Path Discovery and Selection for Heterogeneous Graph Neural Networks
> **Written from a professor's perspective for a top-tier CS conference student.**
> Chain-of-thought reasoning is embedded throughout — each paper is dissected for its mathematical core, hidden assumptions, and where it leaves room for a genuinely new contribution.
> **v3 — Added §19 PMMM (differentiable meta multigraph + partial message propagation). §16.8 adds PMMM-inspired stabilization for Gumbel-Sheaf bilevel optimization. §15.1 taxonomy and §16.6 novelty table updated.**

---

## How to Read This Document

For each paper I answer five questions:
1. **What problem do they actually solve?** (not the framing, but the mathematical gap)
2. **How do they solve it?** (key equations, algorithm structure)
3. **What did they achieve?** (concrete numbers, not claims)
4. **Where does it break down?** (the honest weaknesses)
5. **What is left on the table?** (directions for a new contribution)

At the end I give a **cross-paper synthesis** — the conceptual map you need before writing your own paper.

---

## Table of Contents

**Part I — Meta-Path Methods (11 papers)**
1. [MCTS-HGNN — Monte Carlo Meta-path Search](#1-mcts-hgnn)
2. [AMP-HGNAS — Dependency-Aware Architecture Search](#2-amp-hgnas)
3. [RMS-HRec — Reinforcement Learning for Recommendation Meta-paths](#3-rms-hrec)
4. [SMCD — Semantic-Aware Diffusion with Dual Optimization](#4-smcd)
5. [HG-Search — Multi-Stage NAS for HGNNs](#5-hg-search)
6. [SchemaWalk — Inductive Meta-path Learning](#6-schemawalk)
7. [HISE-KT — LLM-Guided Meta-path for Knowledge Tracing](#7-hise-kt)
8. [LMSPS — Long-range Meta-path on Large-scale Graphs](#8-lmsps)
9. [ESim — Meta-path Guided Embedding for Similarity Search](#9-esim)
10. [MP-GNN — Meta-Path Learning for Multi-relational GNNs](#10-mp-gnn)
11. [LPSMS — Relational Subset Selection Reformulation](#11-lpsms)
12. [Cross-Paper Synthesis](#12-cross-paper-synthesis)
13. [Research Opportunity Map](#13-research-opportunity-map)

**Part II — Search with Guarantees (3 papers from `research_paper_reinforce/`)**

12. [LevinTS / LubyTS — Policy Tree Search with Guarantees](#12-levints--lubyts)
13. [LHBL — Limited-Horizon Bellman Learning](#13-lhbl)
14. [PHS — Policy-Heuristic Best-First Search](#14-phs)

**Part III — Sheaf Extensions, NAS Insights, and Meta Multigraph (5 papers from `research_paper02/` + `research_paper/`)**

15. [DeepSN — Sheaf Neural Framework for Influence Maximization](#15-deepsn)
16. [SHN — Sheaf HyperNetworks for Federated Learning](#16-sheaf-hypernetworks-shn)
17. [NAS-RL — Neural Architecture Search with Reinforcement Learning](#17-nas-rl)
18. [NAS Insights — 1000-Paper NAS Survey](#18-nas-insights-from-1000-papers)
19. [PMMM — Differentiable Meta Multigraph with Partial Message Propagation](#19-pmmm)

**Cross-Paper Analysis (v2/v3)**

15\*. [Revised Cross-Paper Synthesis](#15-revised-cross-paper-synthesis-v2)
16\*. [Revised Research Direction: PHS-Sheaf](#16-revised-research-direction-phs-sheaf-think-again)

---

## 1. MCTS-HGNN

**Full title:** *A New Meta-path Monte Carlo Tree Search Algorithm for Heterogeneous Graph Neural Networks*

### 1.1 Problem Formulation

**The gap:** Heterogeneous graphs G = (V, E, A, R) contain multiple node types A and edge/relation types R. A meta-path P = c₁ →^{r₁} c₂ →^{r₂} ··· →^{rₗ} cₗ₊₁ defines a semantic path template. The question is: **which meta-paths should be used?** Traditional methods either hardcode them (domain expert required) or use all of them (semantic noise, computational waste). No principled search strategy existed.

**Mathematical setup:** The meta-path search space is represented as a tree T_P = {Q, I_Q, l_Q} where Q is the set of tree nodes, I_Q are tree edges, and l_Q are weights. Finding optimal meta-paths is equivalent to finding the best leaf nodes in this tree — a sequential decision problem under uncertainty.

### 1.2 Methodology

**Core idea:** Borrow MCTS from game-playing AI (AlphaGo-style) and apply it to the meta-path search space.

The MCTS loop has four stages:

**1. Selection** — Navigate the tree from root to a leaf using the UCT score:
$$\text{score}(\alpha_1, \alpha_2) = \alpha_1 \cdot \frac{1-\nu}{\nu} + \alpha_2 \cdot \eta \cdot \sqrt{\frac{\ln N_t}{\sqrt{n}}}$$
where α₁ balances exploitation (historical performance ν) and α₂ controls exploration (visit-count ratio √(ln N_t / √n)).

**2. Expansion** — Add child nodes representing one more hop in the meta-path. Two expansion modes: single-node (add one node type) or full-depth (expand all children at once).

**3. Simulation** — From the expanded node, randomly walk to a terminal meta-path and evaluate it:
$$h_n(v_i) = \text{AGG}(\{h_{e_j},\ \forall e_j \in \mathcal{N}_S(v_i)\})$$

**4. Backpropagation** — Propagate the reward (validation accuracy) back to parent nodes to update scores.

**Aggregation architecture:** Once meta-paths are chosen, an attention-based aggregation collects embeddings:
$$H^P = X^P \cdot W^P, \quad \hat{y} = \text{MLP}(H), \quad \mathcal{L} = -\sum_i \log \hat{y}_i$$

A semantic fusion Transformer combines embeddings across the selected meta-path set.

**Complexity:** O(K·l·u²) where K = MCTS iterations, l = path length, u = branching factor.

### 1.3 Contributions

- First MCTS-based framework for automatic meta-path discovery in HGNNs
- Exploration-exploitation score that handles variable-depth meta-paths
- Semantic fusion module via Transformer to merge multi-meta-path embeddings

### 1.4 Results

| Dataset | Task | Micro-F1 | vs. best baseline |
|---------|------|----------|------------------|
| DBLP | Node classification | 95.66% | +0.31% over LMSPS |
| IMDB | Node classification | 69.85% | +1.06% over HAE |
| ACM | Node classification | 94.32% | +0.88% over Simple-HGN |

Node clustering NMI: DBLP 90.37%, IMDB 69.46%, ACM 86.72%.

### 1.5 Strengths

- Principled exploration-exploitation tradeoff (not greedy, not random)
- Tree structure naturally represents hierarchical meta-path extensions
- Semantic Transformer fusion captures inter-meta-path interactions
- Consistent improvements across three standard benchmarks

### 1.6 Weaknesses

- **Scalability:** O(K·l·u²) becomes prohibitive on large graphs or deep meta-paths; no experiment beyond DBLP scale
- **Validation dependency:** Reward signal requires full GNN forward pass per MCTS node — computationally expensive
- **No convergence guarantee:** MCTS is heuristic; optimal meta-paths not guaranteed
- **No inter-path dependency modeling:** Treats each meta-path independently; redundant paths can still be selected
- **Limited datasets:** Only three small academic networks tested; no large-scale or heterophilic evaluation

### 1.7 Future Ideas

> **Professor's note:** MCTS is well-motivated but the expensive GNN evaluation per tree node is a bottleneck. A surrogate model (e.g., a learned value network estimating meta-path quality from its type-sequence embedding without full GNN training) would decouple search from evaluation. Combined with a meta-path holonomy score as a geometric proxy for informativeness, you could search purely in type-schema space and only validate the top-K candidates. This would reduce complexity from O(K·GNN_cost) to O(K·embed_cost).

---

## 2. AMP-HGNAS

**Full title:** *Adaptive Meta-Path-based Neural Network Architecture Search for Heterogeneous Graphs*

### 2.1 Problem Formulation

**The gap:** Most HGNNs assume meta-paths are either given (HAN, MAGNN) or that any valid meta-path is useful. Neither is true. Meta-paths can be **redundant** (overlapping graph coverage → adding semantic noise) or **complementary** (independent coverage → additive information). The challenge is to jointly: (a) select a non-redundant meta-path subset and (b) learn optimal aggregation strategy — simultaneously.

**Formal dependency definition:** For meta-paths Pᵢ, Pⱼ, their overlap is:
$$D(P_i, P_j) = \frac{|N(P_i) \cap N(P_j)|}{|N(P_i) \cup N(P_j)|} \times \frac{|E(P_i) \cap E(P_j)|}{|E(P_i) \cup E(P_j)|}$$
High D → highly redundant → keep only one; low D → complementary → keep both.

### 2.2 Methodology

**Two-stage pipeline:**

**Stage I — Meta-path Dependency Selection:**

1. Build dependency graph: edge (Pᵢ, Pⱼ) if D(Pᵢ, Pⱼ) > δ (threshold)
2. Solve **Maximum Independent Set** on this graph → keeps maximally complementary subset S*
3. Approximated by a greedy O(|P|) algorithm (optimal MIS is NP-hard in general)

Meta-path strength (Gumbel-softmax-like):
$$q_k(\alpha_k) = \frac{\exp\left(\frac{\alpha_k + |N(P_k)| + |E(P_k)|}{2\tau}\right)}{\sum_j \exp\left(\frac{\alpha_j + |N(P_j)| + |E(P_j)|}{2\tau}\right)}$$

**Stage II — Adaptive Aggregation via DARTS:**

Three-level architecture search:
- **Message Passing (ℴ¹):** {Direct, Mean, MLP} aggregation variants
- **Feature Aggregation (ℴ²):** {Mean, Max, Attention} across neighbors
- **Meta-path Fusion (ℴ³):** {Mean, Attention} across paths

DARTS bilevel optimization:
$$W \leftarrow W - \eta_W \nabla_W \mathcal{L}_{\text{train}}(W, \Xi;\ B_{\text{trn}}), \quad \Xi \leftarrow \Xi - \eta_\Xi \nabla_\Xi \mathcal{L}_{\text{val}}(W, \Xi;\ B_{\text{val}})$$

Node representation via meta-path k:
$$\hat{X}_k = \prod_{i=0}^{l-1} \hat{A}_{t^i_{c_i, c_{i+1}}} X_{j_t}$$

### 2.3 Contributions

- First explicit inter-meta-path dependency modeling via graph-theoretic redundancy
- Dependency graph + Maximum Independent Set for principled pruning
- Module-level DARTS for simultaneous meta-path and aggregation search
- 35x speedup over LMSPS on OGBN-MAG (1.9M nodes)

### 2.4 Results

| Dataset | Macro-F1 | Micro-F1 | vs. HAN |
|---------|----------|----------|---------|
| DBLP | 95.83 | 96.11 | +4.21 |
| IMDB | 67.29 | 69.51 | +9.55 |
| ACM | 95.15 | 95.13 | +4.26 |
| OGBN-MAG | 56.20 | — | OOM (HAN) |

Training time on OGBN-MAG: **39 seconds** total (Stage I: 29s, Stage II: 10s).

### 2.5 Strengths

- First to quantify and leverage inter-path redundancy as a selection criterion
- Strong scalability: handles 1.9M node graph in 39 seconds
- Module-level DARTS is more expressive than architecture-level search
- Thorough ablation confirming both stages contribute independently

### 2.6 Weaknesses

- **Dependency threshold δ is critical and manually tuned** — no adaptive or data-driven strategy for setting δ; wrong δ either prunes useful paths or keeps redundant ones
- **Greedy MIS may discard complementary paths** — NP-hard exact MIS approximated by greedy; pathological cases exist
- **First-order DARTS approximation** — may miss complex multi-path interactions; known stability issues in the NAS literature
- **Node classification only** — no evaluation on link prediction, graph classification, or recommendation
- **Jaccard-based dependency** — D(Pᵢ, Pⱼ) uses node/edge overlap, which ignores semantic (type-level) similarity; paths can be structurally disjoint but semantically redundant

### 2.7 Future Ideas

> **Professor's note:** The dependency measure D(Pᵢ, Pⱼ) is purely structural (Jaccard over nodes/edges). A more principled measure would use **task-conditional mutual information**: I(Z^{Pᵢ}; Y | Z^{Pⱼ}) — how much additional information does Pᵢ's embedding provide about labels, given Pⱼ's embedding. This is estimable via conditional entropy bounds or variational mutual information estimators. Paths with low conditional MI are genuinely redundant for the task, not just structurally overlapping.

---

## 3. RMS-HRec

**Full title:** *Automatic Meta-Path Discovery for Effective Graph-Based Recommendation*

### 3.1 Problem Formulation

**The gap:** In recommendation systems, meta-paths encode collaborative filtering signals (e.g., User→Movie→Actor→Movie means "users who watched movies with the same actor"). Current recommenders (HERec, MCRec) require these to be manually specified. The gap: **automatic discovery of recommendation-relevant meta-paths using reinforcement learning.**

**Formal setting:** Given a heterogeneous information network (HIN) G = (V, E, A, R) for recommendation, and a base recommender model M (e.g., HERec), find the meta-path set Θ* that maximizes recommendation quality:
$$\Theta^* = \arg\max_\Theta \text{NDCG}@10(\mathcal{M}_\Theta)$$

### 3.2 Methodology

**RL formulation for meta-path discovery:**

- **State s_t:** L2-normalized embedding of current meta-path set: $s_t = \text{Normalize}\!\left(\sum_{\theta \in \Theta_t} E_\theta\right)$
- **Action a_t:** Select a new meta-path from the relation space (four action types covering common HIN relations)
- **Reward R:** Change in NDCG@10 on validation: $R(s_t, a_t) = \text{NDCG}(s_{t+1}) - \text{NDCG}(s_t)$
- **Policy:** Deep Q-Network (DQN) with Bellman update: $Q(s_i, a_i) \leftarrow R(s_i, a_i) + \gamma Q(s_{i+1}, f(s_{i+1}))$

**Key innovation — Early Stopping Approximation:**
Full recommender training to evaluate each candidate meta-path would take hours. Instead, train only a few epochs and stop early when validation loss plateaus (10 consecutive non-improving epochs). This reduces evaluation cost by **100x** while maintaining reward signal fidelity.

**Final recommender architecture (RMS-HRec):**

1. Generate meta-path subgraph for each discovered path Θ
2. GAT-based aggregation per meta-path
3. Attention fusion across paths: $\beta_\theta = \frac{\exp(\phi_\theta)}{\sum_{\theta' \in \Theta} \exp(\phi_{\theta'})},\quad H = \sum_k \beta_\theta^k \cdot H_{\theta_k}$
4. Recommendation score: $\hat{y}(u, i) = h_u^\top h_i$
5. BPR loss: $\mathcal{L}_{\text{rec}} = \sum_{(u,i,j) \in \mathcal{O}} -\ln \sigma(\hat{y}(u,i) - \hat{y}(u,j))$

### 3.3 Contributions

- First RL-based framework for automatic meta-path discovery in recommendation
- **Model-independent:** RMS wrapper applies to any existing meta-path-based recommender
- Early stopping approximation enables practical RL exploration in large action spaces

### 3.4 Results

| Dataset | Metric | RMS | Best manual baseline |
|---------|--------|-----|---------------------|
| Yelp | NDCG@10 | 0.1740 | 0.1633 (Greedy) |
| Douban | NDCG@10 | 0.2400 | 0.2221 (Random) |
| TCL | NDCG@10 | 0.3230 | 0.3058 (Greedy) |

### 3.5 Strengths

- Genuinely model-independent: validated on HERec and MCRec
- Early stopping approximation is a practical and transferable trick
- RL naturally handles the combinatorial meta-path search space
- Discovers transferable patterns across datasets (same meta-paths useful across domains)

### 3.6 Weaknesses

- **Sequential action space limitation:** Only extends meta-paths by one relation at a time; cannot discover structurally non-sequential paths or branching patterns
- **DQN instability:** Q-learning with neural approximators is notoriously unstable; convergence not theoretically guaranteed
- **Reward sparsity:** NDCG@10 rewards are sparse and noisy — many meta-paths have negligible effect; credit assignment is hard
- **Early stopping approximation is a heuristic:** No theoretical bound on how well early-stopped performance approximates fully-trained performance
- **Cold start problem:** RL requires many exploration steps; for new datasets, the agent starts from scratch

### 3.7 Future Ideas

> **Professor's note:** The DQN approach suffers from the standard issues of discrete RL — sparse rewards and slow convergence. A cleaner formulation would be **Bayesian Optimization over meta-path type sequences**: model the NDCG as a Gaussian process over the discrete space of meta-path schemas, and use an acquisition function (Expected Improvement) to select the next meta-path to evaluate. This converges with far fewer evaluations than DQN and provides uncertainty estimates that indicate when the search is complete.

---

## 4. SMCD

**Full title:** *Enhancing Heterogeneous Graph Learning with Semantic-Aware Meta-Path Diffusion and Dual Optimization*

### 4.1 Problem Formulation

**The gap:** Data augmentation for heterogeneous graphs (adding/removing edges, masking features) is typically done without semantic awareness. Random edge removal can destroy semantically critical connections; random addition can introduce noise. Additionally, label scarcity makes supervised training alone insufficient. The paper asks: **how do we augment heterogeneous graph data while preserving meta-path semantics, and how do we leverage unlabeled data jointly?**

### 4.2 Methodology

**Three-component system:**

**Component 1 — Semantic-Aware Meta-Path Perturbation:**

Auxiliary meta-paths are generated from core meta-paths under two constraints:
- *Semantic Association Anchoring:* Auxiliary P' must share at least one edge type with core P
- *Semantic Complementarity:* P' must not completely overlap with P
- *Length Consistency:* |P'| = |P| to maintain comparable semantic scope

Edge perturbation uses semantic similarity: mask edge (u,v) if:
$$\rho_{u,v} \cdot \pi_{u,v} > \theta, \quad \theta \in [0,1]$$
where ρ_{u,v} = path instance count, π_{u,v} = semantic weight (from attention matrix).

**Component 2 — Gaussian Diffusion Augmentation:**

Core meta-path graphs are augmented via a denoising diffusion process:
- Forward process: $q(x_t|x_{t-1}) = \mathcal{N}(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I)$
- Reverse denoising: $p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\ \mu_\theta(x_t,t),\ \Sigma_\theta(x_t,t))$
- Applied iteratively for T=50 steps to augment graph structure while preserving semantic topology

**Component 3 — Dual Optimization:**

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{sp}} + \lambda_2 \mathcal{L}_{\text{auto}} + \lambda_3 \mathcal{L}_{\text{cs}}$$

- $\mathcal{L}_{\text{sp}}$: Supervised cross-entropy on labeled nodes
- $\mathcal{L}_{\text{auto}}$: Self-supervised invariance objective (decorrelation between augmented views)
- $\mathcal{L}_{\text{cs}}$: Class-guided contrastive loss: attract same-class nodes, repel different-class nodes

**Encoder architecture:**
Multi-head graph attention with intra-meta-path attention:
$$\sigma_u^{P_k} = \text{LeakyReLU}\!\left(w_{u,k}^\top [h_u \oplus h_{\text{neighbor}}]\right), \quad \tilde{u}_u^{P_k} = \frac{\exp(\sigma_u^{P_k})}{\sum_{u' \in \mathcal{N}_k} \exp(\sigma_{u'}^{P_k})}$$

### 4.3 Contributions

- Semantic-aware edge perturbation preventing semantic destruction in augmentation
- Diffusion model for heterogeneous graph augmentation (novel application domain)
- Dual optimization (supervised + self-supervised + class-guided) for label-scarce settings
- Robustness: maintains 99.98% accuracy under 50% label noise

### 4.4 Results

| Dataset | Macro-F1 | vs. HAN |
|---------|----------|---------|
| ACM | 96.30 ± 0.01 | +5.34 |
| DBLP | 97.18 ± 0.04 | +4.22 |
| IMDB | 68.59 ± 0.38 | +8.52 |

Noise robustness: under 50% label noise, SMCD maintains 99.98% accuracy vs. HGCMA at 90.89%.

### 4.5 Strengths

- Exceptionally strong performance: 96.30% on ACM, 97.18% on DBLP (among highest reported)
- Semantic awareness in augmentation is well-motivated and novel
- Diffusion process for graph augmentation is a creative and underexplored direction
- Dual objective elegantly handles both labeled and unlabeled data
- Thorough noise robustness experiments that few papers conduct

### 4.6 Weaknesses

- **Meta-paths still manually specified:** The semantic-aware perturbation operates *on* given meta-paths, not discovering them
- **Diffusion overhead:** T=50 denoising steps multiplies forward pass cost; full complexity analysis missing
- **Semantic weight matrix π:** Construction requires edge-level annotations that may not be available in all datasets
- **λ hyperparameters require tuning:** Three loss weights λ₁, λ₂, λ₃ with no principled selection strategy
- **Only three small datasets:** ACM, DBLP, IMDB are well-studied benchmarks; no large-scale or heterophilic evaluation
- **Noise robustness evaluation is suspicious:** 99.98% accuracy under 50% label noise seems unusually high and warrants deeper investigation

### 4.7 Future Ideas

> **Professor's note:** The diffusion augmentation is the most interesting technical piece here. The key limitation is that it operates on fixed meta-path graphs. An extension would be to use the diffusion process **in the sheaf sense** — define a learnable diffusion operator on the heterogeneous graph whose stationary distribution implicitly identifies which meta-path relationships are preserved vs. which are noise. This would unify augmentation with sheaf-theoretic diffusion and allow the meta-paths to emerge from the dynamics rather than being pre-specified.

---

## 5. HG-Search

**Full title:** *HG-Search: Multi-Stage Search for Heterogeneous Graph Neural Networks*

### 5.1 Problem Formulation

**The gap:** Existing HGNNs require manual architecture design (which aggregation functions to use, how many layers, what connectivity patterns). Hyperparameter selection is equally manual. No method jointly optimizes both the **HGNN architecture** (message passing operations, connectivity) and **hyperparameters** (learning rate, hidden dim, dropout) for heterogeneous graphs specifically.

**Key observation:** In the space of all HGNN architectures, the optimal combination of (graph extraction method × neighbor aggregation × semantic fusion × connectivity pattern × hyperparameters) is dataset-dependent and cannot be determined by intuition alone.

### 5.2 Methodology

**General HGNN framework with four modular layers:**

1. **Feature Preprocessing:** $h'_{ij} = W_{A_i} \cdot h_{v_j}$ (project all node types to shared dimension)

2. **Heterogeneous Graph Extraction:** Four paradigms:
   - *Homogenization:* collapse all types to one
   - *Relation-based:* separate subgraph per relation
   - *Meta-path-based:* semantic graph per meta-path
   - *Mixture:* combination

3. **Message Passing (NA) + Semantic Fusion (SF):**
   - NA: $h_i^{(l+1)} = \text{NA}(h_i^{(l)}, \{h_j^{(l)}\}_{j \in \mathcal{N}_i^\phi})$
   - SF: $h_i^{(l+1)} = \text{SF}(\{h_i^{(l+1)}\}_\phi)$

4. **Post-processing:** Final MLP for task output

**Search strategy — Policy Gradient (PG-Search):**

Controller parameterized by θ generates HGNN configurations. Training objective:
$$\mathcal{J}(\theta) = \mathbb{E}_{\alpha \sim \pi_\theta}[R(\alpha)], \quad \nabla_\theta \mathcal{J}(\theta) \approx \frac{1}{B}\sum_b R(\alpha_b) \nabla_\theta \log \pi_\theta(\alpha_b)$$

**Multi-Stage Search (MS-Search):**
1. Pre-search: Joint architecture + hyperparameter initialization via PG-Search
2. HGNN Architecture Search: Optimize architecture while fixing hyperparameter distribution
3. Hyperparameter Search: Fine-tune hyperparameters with fixed architecture

Search space includes: GCNConv, GATConv, SageConv, GINConv for NA; Mean/Max/Sum/Attention for SF; Stack/Skipsum/Skipconcat for connectivity; plus 12 hyperparameter dimensions.

### 5.3 Contributions

- First NAS framework jointly optimizing HGNN architecture and hyperparameters
- Policy gradient controller with dedicated HGNN search space
- Multi-stage decomposition reducing search complexity

### 5.4 Results

| Dataset | Macro-F1 | vs. best manual |
|---------|----------|----------------|
| DBLP | 94.84 ± 0.33 | +1.27 over Simple-HGN |
| IMDB | 66.08 ± 1.45 | Competitive |
| Freebase | 62.79 ± 1.06 | Best reported |
| AMiner | 79.58 ± 2.32 | +5.0 over HAN |

### 5.5 Strengths

- Comprehensive search space covering the full HGNN design space
- MS-Search decomposes search stages, reducing search cost
- Results demonstrate automated design can match or exceed hand-crafted HGNNs
- First work to simultaneously optimize architecture and hyperparameters for HGNNs

### 5.6 Weaknesses

- **Node classification only:** No evaluation on link prediction, graph-level tasks
- **Computational cost of NAS not reported:** Policy gradient can require thousands of evaluations; wall-clock cost unclear
- **Architecture complexity:** Many architectural choices create a vast joint search space; convergence to global optimum not guaranteed
- **Dataset-specific architectures:** No analysis of whether found architectures transfer across datasets
- **No comparison with DARTS-based NAS:** Policy gradient is less efficient than gradient-based NAS

### 5.7 Future Ideas

> **Professor's note:** The most natural next step is to make the search space *sheaf-aware*: include the stalk dimension d, restriction map parameterization family (diagonal/orthogonal/general), and coboundary regularization weight as searchable architecture dimensions. A sheaf-aware NAS framework would jointly discover which aggregation functions, which restriction map families, and which meta-path templates work together — the meta-path selection problem and the architecture selection problem become unified.

---

## 6. SchemaWalk

**Full title:** *Inductive Meta-path Learning for Schema-complex Heterogeneous Information Networks*

### 6.1 Problem Formulation

**The gap (and it is significant):** All existing meta-path methods work on schema-simple HINs — networks with a small number of entity types (e.g., DBLP has 4). Real knowledge graphs like DBPedia have **174 entity types** and hundreds of relation types. In this *schema-complex* regime:
1. The meta-path space is astronomically large (exponential in types × length)
2. Manual specification is infeasible
3. **Inductive setting:** For knowledge graph completion, test edges are never seen during training — path instances that exist in training data may not exist at test time

**Formal definitions:**

- Coverage: $\text{Cov}_{M \to r_q} = \frac{|\{(v_i,v_j) | \text{Conn}_M(v_i,v_j) \wedge r_q(v_i,v_j)\}|}{|\text{all}\ r_q\text{-pairs}|}$
- Confidence: $\text{Conf}_{M \to r_q} = \frac{|\{(v_i,v_j) | \text{Conn}_M(v_i,v_j) \wedge r_q(v_i,v_j)\}|}{|\{(v_i,v_j) | \text{Conn}_M(v_i,v_j)\}|}$

High coverage = meta-path captures many query-relevant pairs. High confidence = captured pairs are actually query-relevant.

### 6.2 Methodology

**RL agent navigating the schema graph (not the instance graph):**

Key insight: Instead of searching instance-level paths (exponentially many), navigate at the **type-schema level** — the schema graph G_s has one node per entity type and one edge per relation type. This compresses the search space from |V|^L (instances) to |T|^L (schema types).

**State:** (current type t_src, relation set R_t, target type t_tgt, path history, query relation r_q)

**Action:** Select next edge in schema graph (outgoing relation from current type)

**Reward:**
$$R(M | r_q, H) = \lambda_1 \cdot \text{Cov}_{M \to r_q} + \lambda_2 \cdot \text{Conf}_{M \to r_q} + \lambda_3 \cdot \mathbf{1}[t_l = t_{\text{tgt}}]$$

**Policy network:** Two-layer LSTM encoder-decoder:
$$D_i = \sigma([r_c \| t_c]),\quad c = [1, \ldots, |A_s|]$$
Policy gradient update: $\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{(t_{\text{src}}, r_q, t_{\text{tgt}})} \left[\hat{R}(M) \nabla_\theta \ln \pi_\theta(A_i | S_i)\right]$

**Multi-relation training:** Cycles through query relations every I_sw iterations, enabling a single agent to discover meta-paths for all relations simultaneously.

### 6.3 Contributions

- First formal definition of inductive meta-path learning problem
- Schema-level path-finding that scales to 174-type knowledge graphs
- Coverage + confidence dual metrics as principled RL reward signal
- Single agent learns meta-paths for all relations jointly

### 6.4 Results

| Dataset | Setting | Hits@1 | MRR |
|---------|---------|--------|-----|
| YAGO26K-906 | Inductive | 0.376 | 0.218 |
| DB111K-174 | Inductive | 0.851 | 0.754 |
| YAGO26K-906 | Transductive | Hits@10: 0.377 | — |
| DB111K-174 | Transductive | Hits@10: 0.875 | — |

Robustness: maintains performance under up to 100% node removal (true inductive generalization).

### 6.5 Strengths

- First method designed for schema-complex settings with 100+ entity types
- Schema-level search is genuinely scalable (|T|^L ≪ |V|^L)
- Coverage + confidence metrics are interpretable and measurable
- True inductive setting evaluation — most methods cannot handle this
- Discovered meta-paths are human-readable and semantically meaningful

### 6.6 Weaknesses

- **KG completion only:** Evaluated purely on link prediction; no node classification or recommendation
- **Reward sparsity at schema level:** Coverage/confidence computed over entire KG — noisy gradient signal
- **LSTM may not capture long-range type dependencies:** For complex schemas, type-level interactions may require more expressive sequence models
- **Multi-relation training with periodic switching** is a heuristic; no convergence analysis
- **No evaluation on downstream tasks:** Discovered meta-paths are validated only intrinsically (coverage/confidence), not by plugging into a full HGNN pipeline

### 6.7 Future Ideas

> **Professor's note:** The schema-level search idea is excellent but the LSTM policy is underpowered for complex type schemas. A more powerful architecture would use a **Transformer over type-relation token sequences** with pre-trained type embeddings from the schema graph structure (e.g., from a type-level GNN). This would capture global schema structure during search, enabling the agent to reason about long-range type dependencies. Additionally, replacing the LSTM reward with a **contrastive schema alignment** objective (align meta-path schema with query relation schema in embedding space) would give a denser and more informative training signal.

---

## 7. HISE-KT

**Full title:** *HISE-KT: Synergizing Heterogeneous Information Networks and LLMs for Explainable Knowledge Tracing with Meta-Path Optimization*

### 7.1 Problem Formulation

**The gap:** Knowledge Tracing (KT) models student performance. The key gap is **explainability**: existing deep KT models (DKT, AKT) predict outcomes but cannot explain why. HIN-based KT can leverage rich relationships (student-question-concept-difficulty) but suffers from (a) noisy meta-path instances and (b) inability to generate natural language explanations. LLM-based KT can explain but lacks structured reasoning about higher-order relationships.

**HIN construction:** Five node types {Students U, Questions Q, Knowledge Concepts K, Difficulty D, Ability A} with four edge types {Q-U, Q-K, Q-D, U-A}.

### 7.2 Methodology

**Module 1 — LLM-Powered Meta-path Quality Scoring:**

Four composite meta-paths: {Q-U-Q, Q-K-Q, Q-D-Q, Q-U-A-U-Q}

Each instance scored by four dimensions:
- **Centrality** (question connectivity): $C_q(P) \propto 1 - \frac{1}{|Q(P)|}\sum_{q \in Q(P)} \text{dist}(q_0, q)$
- **KC Relevance** (knowledge concept match): $R_{KC}(P) \propto \frac{|\{q \in Q(P) | k^* \in KC(q)\}|}{|Q(P)|}$
- **Informativeness** (distinct nodes): $I_{\text{info}}(P) \propto \frac{\text{distinct\_nodes}(P)}{|V \in P|}$
- **Diversity** (node type entropy): $D_{\text{type}}(P) \propto -\frac{1}{\log|L|}\sum_{\ell \in L} p_\ell \log p_\ell$
- **Total:** $S(P) = C_q + R_{KC} + I_{\text{info}} + D_{\text{type}}$

Top-K instances per meta-path are retained; lowest-K are pruned.

**Module 2 — Rule-Based Student Similarity:**

Cross-student similarity via 5-dimensional feature vector + Mahalanobis distance:
$$\mathbf{z}_{u,s} = [|\theta_u - \theta_s|,\ \Delta\text{acc},\ (1+N_Q)^{-c},\ (1+N_K)^{-c},\ (1+f)^{-c}]^\top$$
$$d(u,s) = \sqrt{(\mathbf{z}_{u,s} - \mu)^\top \Sigma^{-1} (\mathbf{z}_{u,s} - \mu)}$$

**Module 3 — LLM Prediction with Explanation:**

Zero-shot LLM (no fine-tuning) receives: (student ability, similar students' performance, meta-path quality scores) and outputs: (binary prediction + 3-sentence explanation).

### 7.3 Contributions

- First to combine HIN reasoning with LLM generation for knowledge tracing
- Principled meta-path quality scoring with four complementary dimensions
- Mahalanobis-based student similarity for cross-student collaborative filtering
- Interpretable predictions with natural language explanations

### 7.4 Results

| Dataset | AUC | vs. DKT |
|---------|-----|---------|
| AssistmentO9 | 0.8703 | +0.0203 |
| Statics2011 | 0.9073 | +0.0573 |
| Freshsub | 0.9482 | Best |

vs. LLM baselines (EPFL, EFKT): 0.8888 vs. 0.6843 on AssistmentO9 (27% improvement).

### 7.5 Strengths

- Novel domain application: educational KT as a heterogeneous graph problem
- Four-dimensional meta-path quality scoring is principled and interpretable
- LLM explanations are genuinely novel and practically useful for educators
- Mahalanobis distance handles correlated similarity dimensions properly

### 7.6 Weaknesses

- **LLM dependency:** Results vary with LLM quality/version; not reproducible without specifying exact LLM and prompt
- **Zero-shot prompting is brittle:** Performance sensitive to prompt format; no systematic prompt optimization
- **Meta-path templates are predefined:** Four templates Q-U-Q, Q-K-Q, Q-D-Q, Q-U-A-U-Q are manually chosen for the education domain
- **Small datasets:** 331-3913 students; educational graphs are tiny compared to web-scale HINs
- **Rule-based filtering:** Mahalanobis similarity is fixed; does not learn task-adaptive student similarity
- **AUC metric only:** No evaluation of explanation quality (human study needed)

### 7.7 Future Ideas

> **Professor's note:** The most impactful extension would be to replace the four manually specified meta-path templates with an LLM-guided schema discovery mechanism: ask the LLM "given student, question, concept, difficulty, ability relationships, which semantic paths are meaningful for predicting student performance?" and use the LLM's structured reasoning to generate candidate schemas automatically. This would make the method truly domain-agnostic and eliminate the last manually specified component.

---

## 8. LMSPS

**Full title:** *Long-range Meta-path Search on Large-scale Heterogeneous Graphs*

### 8.1 Problem Formulation

**The gap (most clearly stated in this paper):** Two empirical observations on DBLP:
1. Removing a single meta-path (e.g., APA) reduces accuracy by only 0.3%
2. Removing an entire class of meta-paths (e.g., all APV-type paths) reduces accuracy by 10.7%

This suggests: **meta-path class (type schema) matters far more than individual instances.** Current methods either (a) enumerate short meta-paths exhaustively (exponential in count as length grows) or (b) use only short paths (losing long-range dependencies). On sparse graphs (OGBN-MAG-5 with average degree 5), long-range paths are essential.

**Formal observation:**
$$|\mathcal{P}_L| \sim \prod_{i=1}^L |\mathcal{T}_E| = |\mathcal{T}_E|^L \quad \text{(exponential growth)}$$

For a graph with 10 edge types and paths of length 6: 10⁶ = 1 million candidate paths.

### 8.2 Methodology

**Pre-computation (SeHGNN-style):**
$$X_k = \begin{cases} X^{c_0} & l = 0 \\ \bar{A}_{c_0,c_1} \bar{A}_{c_1,c_2} \cdots \bar{A}_{c_{l-1},c_l} X^{c_l} & l > 0 \end{cases}$$
where $\bar{A} = D^{-1/2}(A+I)D^{-1/2}$ is the normalized adjacency.

**Progressive Sampling Search — Path Strength via Gumbel-Softmax:**
$$q_k = \frac{\exp\!\left[(\alpha_k + u_k)/\tau\right]}{\sum_j \exp\!\left[(\alpha_j + u_j)/\tau\right]}, \quad u_k \sim \text{Uniform}(0,1)$$

Gumbel noise enables stochastic, differentiable discrete selection.

**Dynamic Candidate Space (progressive pruning):**
$$S_C = \{k \mid q_k \geq \bar{q}_C\}, \quad C = \lfloor\lambda(K - V)\rfloor + V$$
λ decreases from 1 to 0 during training → candidate space shrinks from K to V.

**Sampling Evaluation (discrete meta-path selection):**

After convergence of the continuous relaxation:
1. Uniformly sample M meta-paths from S_C without replacement: $S \sim \text{DiscreteSample}(S_C, M, Q)$
2. Evaluate on validation (200 random samples)
3. Retain the best M paths as S_M

**Final forward pass:**
$$Z = \text{MLP}\!\left(\bigg\|_{k \in S_M} \text{MLP}_k(X_k)\right)$$

MLP instead of Transformer (avoids inductive bias, faster inference).

**Bilevel optimization:**
$$\min_\alpha \mathcal{L}_{\text{val}}(\hat\omega(\alpha), \alpha), \quad \hat\omega(\alpha) = \arg\min_\omega \mathcal{L}_{\text{train}}(\omega, \alpha)$$

### 8.3 Contributions

- First HGNN to systematically leverage long-range (L > 4) meta-paths at scale
- Progressive sampling reduces exponential search to O(K→V) linear shrinking
- Discrete sampling evaluation identifies effective paths without continuous relaxation artifacts
- Generalizable: discovered meta-paths transfer across HGNN architectures

### 8.4 Results

| Dataset | Macro-F1 | vs. SeHGNN |
|---------|----------|-----------|
| DBLP | 95.66 ± 0.22 | +0.21 |
| IMDB | 54.83 ± 0.20 | +0.68 |
| ACM | 94.69 ± 0.36 | +1.69 |
| OGBN-MAG-50 | 54.83% | +3.38% |

On sparse OGBN-MAG-5: **+4.7% over SeHGNN** — confirming long-range dependency value for sparse graphs.

### 8.5 Strengths

- Genuinely addresses scalability: constant memory/time as hop count increases (unlike SeHGNN)
- Strong evidence that **long-range paths matter for sparse graphs** (an underappreciated finding)
- Transferable meta-paths: paths discovered on one backbone architecture improve others
- Clean separation of search (Gumbel-softmax) and evaluation (discrete sampling)

### 8.6 Weaknesses

- **Bilevel optimization instability:** Inner/outer alternation can diverge; no convergence analysis provided
- **Temperature τ is critical but under-analyzed:** Sensitivity analysis only shows results for one τ value
- **Pre-computation memory cost:** Storing X_k for all K candidate paths requires O(K·N·d) memory — expensive for large K
- **Node classification only:** No evaluation on other tasks
- **Comparison with some recent methods missing:** No comparison with AMP-HGNAS or MCTS-HGNN

### 8.7 Future Ideas

> **Professor's note:** The pre-computation bottleneck is the fundamental limitation: you must materialize X_k for all K candidates before search begins. An **on-demand computation** approach would compute X_k only for paths currently in S_C, discarding paths as the candidate set shrinks. This would reduce memory from O(K·N·d) to O(|S_C|·N·d) — a meaningful saving when V ≪ K. The Gumbel-softmax path selection and on-demand computation could be combined in a streaming fashion.

---

## 9. ESim

**Full title:** *Meta-Path Guided Embedding for Similarity Search in Large-Scale Heterogeneous Information Networks*

### 9.1 Problem Formulation

**The gap:** Similarity search in HINs requires comparing nodes that may have no direct connection. PathSim (symmetric meta-path similarity) is the classic approach but requires dense path count matrices — O(|V|²) space. The gap: **scalable embedding-based similarity search that preserves meta-path semantics without dense computation.**

**Setting:** HIN G = (V, E, A, R) with user-specified meta-paths M₁,...,M_K. Goal: learn low-dimensional embeddings x_u ∈ ℝ^d such that similar nodes (under each meta-path) are close in embedding space.

### 9.2 Methodology

**Probabilistic generative model:**

Probability of a path instance P following meta-path M:
$$\Pr(P_{r,\leftarrow E}|u, M) = \prod_{i=1}^L \Pr(v_{i+1}|v_i, M_{s,t})$$

Scoring function:
$$f(u,v,M) = \mu_M + \rho_M^\top x_u + q_M^\top x_v + x_u^\top x_v$$

After decomposition with constraint ρ_M = q_M:
$$f(u,v,M) = (\mu_M - \rho_M^\top q_M) + (x_u + q_M)^\top(x_v + \rho_M)$$

This shifts embeddings by meta-path-specific bias vectors ρ_M, allowing the same x_u to serve multiple meta-paths with different semantic meanings.

**Noise-Contrastive Estimation (NCE):**

Maximizes likelihood ratio of observed paths over noise distribution:
$$\mathcal{L} = \sum_{(u,v,M)} \left[\log \sigma(f(u,v,M)) + K \cdot \mathbb{E}_{v' \sim p_n}[\log(1 - \sigma(f(u,v',M)))]\right]$$

**Pre-computation of path counts (Algorithm 2):**
Dynamic programming: $C(u, l|M) = \sum_{v: (u,v) \in E_M^{(l)}} C(v, l-1|M)$, base case C(u,1|M) = 1.
Enables O(1) sampling per path instance after O(|V|·L) pre-computation.

**Two formulations:**
- **ESim-seq:** Conditional Pr(vᵢ₊₁|vᵢ, M) — each step conditioned on previous node
- **ESim-pair:** Joint Pr(v|u, M) — all path nodes equally conditioned on source

### 9.3 Contributions

- Probabilistic embedding model with meta-path-conditioned bias vectors
- NCE-based optimization scalable to large HINs
- Pre-computation of path counts enabling O(1) online sampling
- Sequential and pairwise formulations capturing different path semantics

### 9.4 Results

| Dataset / Metric | ESim-pair | PTE | PathSim |
|-----------------|-----------|-----|---------|
| DBLP A-P-A AUC | 0.81–0.89 | 0.74–0.80 | 0.52–0.75 |
| Yelp B-R-W-R-B AUC | 0.78–0.88 | 0.63–0.78 | 0.55–0.68 |

Near-linear scaling with network size; training time matches parallelized LINE on comparable hardware.

### 9.5 Strengths

- Theoretically principled probabilistic framework
- Highly efficient: O(1) sampling per path instance after pre-computation
- Flexible: handles diverse path semantics via sequential/pairwise duality
- Meta-path bias vectors ρ_M are interpretable as semantic offsets in embedding space

### 9.6 Weaknesses

- **Meta-paths are still manually specified:** ESim embeds given meta-paths; it does not discover them
- **Evaluated on small networks only:** DBLP (18K nodes) and Yelp (9K nodes) — no large-scale validation
- **No GNN baselines:** Compared against older embedding methods (LINE, PTE, PathSim); no comparison with modern HGNNs
- **Pre-computation scales with path length:** O(|V|·L) pre-computation limits scalability for long paths
- **Static embeddings:** No mechanism for inductive inference on unseen nodes

### 9.7 Future Ideas

> **Professor's note:** The meta-path bias vector ρ_M is the most interesting component and is underutilized. In a sheaf-theoretic reading, ρ_M is exactly a **stalk-level restriction map** — it transforms the shared embedding space into a meta-path-specific semantic space. Connecting ESim's probabilistic framework to sheaf theory would give it a topological interpretation: the bias vectors define a sheaf over the HIN, and the NCE objective becomes sheaf Laplacian smoothness. This connection would enable richer analysis of which meta-paths create compatible (low-curvature) vs. conflicting (high-curvature) embedding subspaces.

---

## 10. MP-GNN

**Full title:** *Meta-Path Learning for Multi-relational Graph Neural Networks*

### 10.1 Problem Formulation

**The gap:** Knowledge graphs and multi-relational networks have **large numbers of relations** (FB15K-237 has 237 relation types). In this regime, meta-path enumeration is infeasible (237^L paths of length L), and manual specification is impossible. The gap: **incrementally discover meta-paths in the presence of many relations using an information-theoretic scoring function that does not require prior specification.**

**Key observation:** If a relation r provides no additional information about node labels beyond what the current meta-path already captures, it should not be added to the path. This is a conditional relevance criterion.

### 10.2 Methodology

**Information-theoretic scoring function:**

Score a candidate relation r for extension by the minimum MSE loss of a simple linear predictor:
$$s_r = \min_{\theta, w} \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2$$

where:
$$\hat{y}_i = \theta^\top h_i^{(0)} + \max_j w_j$$

- θ: learnable weight for node features
- w_j: 1 if node j is on the ground-truth meta-path, 0 otherwise

**Greedy beam search (Algorithm 1):**
```
Initialize mp = [], F1' = 0
While |mp| < L_MAX:
  For each relation r:
    compute s_r (Equation 3)
  r* = argmin_r s_r
  Build MP-GNN with mp + [r*], train, eval on val
  If F1 > F1': mp = mp + [r*], F1' = F1
  Else: discard r*
Return mp
```

**MP-GNN architecture:**

Concatenates embeddings from K meta-paths:
$$h_i^{(l)} = \bigoplus_{k=1}^K h_{(i,k)}^{(l)}$$

RGCN-style message passing per relation in meta-path:
$$h_i^{(l+1)} = \sigma\!\left(W_i^{(l)} h_i^{(l)} + \sum_{j \in \mathcal{N}_i,\, r \in \mathcal{R}^{c_{i,r}}} \frac{1}{|\mathcal{N}_i^{c_{i,r}+1}|} W_r^{(l)} h_j^{(l)}\right)$$

### 10.3 Contributions

- Scoring function for incremental meta-path construction from information-theoretic principles
- Algorithm for meta-path discovery that scales to 237 relations without prior specification
- Demonstrates that node features are essential for multi-relational meta-path discovery (not just graph structure)

### 10.4 Results

**Synthetic experiments:** Perfect F1=1.0 meta-path recovery on synthetic graphs — confirms correctness of scoring.

**Real-world:**

| Dataset | Method | F1 |
|---------|--------|-----|
| DBLP | MP-GNN | 0.94 |
| FB15K-237 (PNC task) | MP-GNN | 0.83 |
| FB15K-237 (EDC task) | MP-GNN | 0.96 |

vs. RGCN (F1=0.74), R-HGNN (F1=0.72) on FB15K-237 PNC. Strong improvement in many-relation regime.

### 10.5 Strengths

- First principled greedy search for meta-paths in many-relation graphs
- Scoring function is computable and interpretable
- Synthetic validation gives strong correctness guarantees
- The "bags" construction for multiple meta-paths is a useful abstraction
- Generalizes to node classification, not just embedding

### 10.6 Weaknesses

- **Greedy search:** Incrementally adds one relation at a time; globally suboptimal combinations not discovered
- **Node features required:** Performance drops significantly without initial node features — limits applicability to feature-free graphs
- **Existential features only:** Scoring captures whether a meta-path exists, not its statistical properties (count, proportion)
- **Scalability unclear:** For L=4 and 237 relations, algorithm requires 237 GNN training runs per step
- **No ablation on scoring function:** Does not validate that the proposed scoring outperforms simpler alternatives (e.g., random, frequency-based)

### 10.7 Future Ideas

> **Professor's note:** The scoring function s_r is based on a linear predictor — a proxy for informativeness. A more powerful approach would use **conditional mutual information** I(Ŷ; r-path | current-meta-path), computed via a variational lower bound. This would capture nonlinear dependencies and interaction effects between relations in the meta-path. The greedy extension could then use the CMI-based score, with theoretical guarantees via submodularity if the full CMI is submodular over the relation set.

---

## 11. LPSMS

**Full title:** *Rethinking Heterogeneous Graph Representation Learning via Relational Subset Selection*

### 11.1 Problem Formulation

**The core reframing (the most conceptually important contribution of all 11 papers):**

Most HGNN research asks "how should we aggregate heterogeneous neighbors?" This paper argues the right question is: **"which relations should we aggregate?"** This shifts from an architecture problem to a combinatorial selection problem.

**Formal statement:**
$$\min_{S \subseteq \mathcal{P}} \mathcal{L}(f(G, S)), \quad |S| \ll |\mathcal{P}|$$

Find a compact subset S of meta-paths (|S| = M ≪ |P|) that minimizes task loss. This is a **subset selection problem**, and its complexity analysis reveals something important:

| Paradigm | Complexity | Bottleneck |
|----------|-----------|-----------|
| Neighbor expansion (HAN) | O(N(rd)^L F) | Exponential in hop L |
| Meta-path enumeration (SeHGNN) | O(NKF²) | Exponential in path count K |
| **Subset selection (LMSPS)** | **O(NMF²)** | **Independent of L** |
| **Marginal contribution (LPSMS)** | **O(NF²)** | **Single path, independent of L** |

### 11.2 Methodology

**LMSPS (Long-range Meta-path Progressive Sampling):**

*Progressive sampling* (Algorithm 1):
- Gumbel-softmax path strength: $q_k = \exp[(\alpha_k+u_k)/\tau] / \sum_j \exp[(\alpha_j+u_j)/\tau]$
- Dynamic candidate set: $S_C = \{k | q_k \geq \bar{q}_C\}$, shrinking via $C = \lfloor\lambda(K-V)\rfloor + V$, $\lambda \in [1,0]$
- Subset sampling: $S \sim \text{UniformSample}(S_C, M)$
- Bilevel optimization alternates between weight update (train loss) and architecture update (val loss)

**LPSMS (Long-range Progressive Single Meta-path Search):**

More radical: optimize one path at a time. **Marginal contribution estimation** via leave-one-out:
$$\Delta_i = \mathcal{L}_{\text{val}}(Z^{(-i)}) - \mathcal{L}_{\text{val}}(Z), \quad Z^{(-i)} = \text{MLP}\!\left(\sum_{k \in S_M \setminus \{i\}} q_k' \cdot \text{MLP}_k(X_k)\right)$$

**Theoretical justification:**

Proposition: Under bounded path interaction assumption:
$$F(S) \approx \sum_{P_i \in S} \phi_i + \varepsilon(S), \quad |\varepsilon(S) - \varepsilon_{S \setminus \{P_i\}}| \leq \delta$$

where φᵢ is the first-order utility of path Pᵢ. This justifies leave-one-out as a first-order approximation to marginal utility — valid when paths interact weakly.

### 11.3 Contributions

- Novel problem framing: HGNN learning as relational subset selection (conceptual contribution)
- LMSPS: O(NMF²) complexity independent of L — enables long-range paths at scale
- LPSMS: O(NF²) single-path evaluation with provable approximation bound
- Transferable meta-paths: schemas discovered for one architecture generalize to others
- Interpretability: learned Δᵢ scores are human-readable marginal importances

### 11.4 Results

| Dataset | LMSPS Macro-F1 | LPSMS Macro-F1 | vs. HAN |
|---------|----------------|----------------|---------|
| DBLP | 95.15 ± 0.22 | 95.78 ± 0.26 | +2.50 |
| IMDB | 64.33 ± 0.48 | 65.00 ± 0.36 | +7.26 |
| ACM | 94.20 ± 0.31 | 94.92 ± 0.25 | +11.59 |
| OGB-MAG | 54.61 | 59.69 | HAN OOM |

LPSMS is **3.6x faster** than LMSPS on DBLP (14.6s vs. 52.1s).

### 11.5 Strengths

- Conceptually the most important: reframes the entire HGNN problem
- Complexity analysis is rigorous and reveals why prior methods fail at scale
- Marginal contribution estimation is principled (connected to Shapley values)
- LPSMS achieves best performance with lowest computational cost
- Meta-path transferability strongly supports the claim that schemas capture dataset-level structure

### 11.6 Weaknesses

- **Weak interaction assumption may fail:** For graphs with strong path coupling (dense, many-relation graphs), the approximation bound δ may be large
- **Pre-computation still required:** Must compute X_k for all K candidate paths before search — O(K·N·d) memory
- **Leave-one-out is O(M) evaluations per epoch:** Not truly O(1) per path; full LLO costs M forward passes
- **Large-scale results mixed with tricks:** OGB-MAG results combine LPSMS with LP (label propagation), MS (multi-scale), Emb (embeddings) — unclear which contributes most
- **Comparison against older baselines:** Missing comparison with AMP-HGNAS, MCTS-HGNN, and SMCD

### 11.7 Future Ideas

> **Professor's note:** The marginal contribution Δᵢ is essentially the **Shapley value** of meta-path Pᵢ if we average over all subsets — but computing the full Shapley is exponential. The leave-one-out approximation is first-order only. A more principled approach would be **KernelSHAP** — a regression-based Shapley estimator that requires only O(M log M) model evaluations to converge, and provides confidence intervals on the importance estimates. This would give rigorous meta-path importance attribution without the full Shapley cost.

---

## 12. Cross-Paper Synthesis

### 12.1 The Fundamental Problem: Four Framings

All 11 papers address the same root challenge — selecting useful meta-paths — but frame it differently:

| Framing | Papers | Core operation |
|---------|--------|---------------|
| **Search** | MCTS-HGNN, HG-Search, LMSPS | Explore a meta-path search space |
| **Selection** | AMP-HGNAS, LPSMS, MP-GNN | Prune from candidate pool using a score |
| **Discovery** | RMS-HRec, SchemaWalk | Learn to generate new meta-paths via RL |
| **Augmentation** | SMCD, ESim | Embed or augment *given* meta-paths more effectively |

**The hybrid gap:** No paper simultaneously (a) discovers meta-paths without manual templates AND (b) operates at the topological level (2-cells, Hodge decomposition). This is the opening for H2S-NN's automatic meta-path extension.

### 12.2 Scalability Landscape

| Paper | Max graph size | Complexity class |
|-------|---------------|-----------------|
| ESim | 18K nodes | O(|V|L) pre-compute |
| MCTS-HGNN | 26K nodes | O(K·GNN_cost) |
| HG-Search | 180K nodes | O(search_iters·GNN_cost) |
| SchemaWalk | Schema-level (|T|^L) | O(|T|^L·RL_steps) |
| AMP-HGNAS | 1.9M nodes | O(|P|·graph_size) |
| LMSPS/LPSMS | 1.9M nodes | O(NMF²) |

**Observation:** The two most scalable methods (AMP-HGNAS, LMSPS) operate by pre-computing path features X_k and then learning selection weights — decoupling graph traversal from learning. This is a pattern worth following.

### 12.3 Where Meta-paths Live in the Mathematical Hierarchy

```
Level 0: Node features               x_v ∈ ℝ^d
Level 1: Edge relations              (u →^r v) with type r
Level 2: Meta-path templates         σ = (T₀, r₁, T₁, ..., rₗ, Tₗ) — TYPE SCHEMA
Level 3: Meta-path instances         π = specific node sequence following σ
Level 4: 2-cells (H2S-NN)           Face f = instance π closed into topological cell
```

**Critical insight:** Papers 1-11 all operate at Level 2-3 (schema selection and instance counting). **H2S-NN uniquely operates at Level 4** — it converts instances into topological 2-cells with algebraic meaning (coboundary operator δ₁, Hodge Laplacian L₁). This is what makes the comparison with HETSHEAF unfair to HETSHEAF — HETSHEAF operates at Level 1 only.

### 12.4 The Meta-Path Automation Spectrum

```
Fully Manual ────────────────────────────────────────── Fully Automatic
    │              │              │              │              │
  HAN           ESim          LMSPS         SchemaWalk     (Open problem)
  MAGNN         SMCD         LPSMS          RMS-HRec
(hardcoded)   (embed/       (select        (generate      (discover from
              augment       from pool)     via RL)        topology itself)
              given)
```

**H2S-NN currently:** Between HAN and ESim (templates hardcoded, instances enumerated). **The proposed extension (Holonomy-Guided Topology Learning)** would place it at the right end — deriving schema importance from the geometry of the learned sheaf.

### 12.5 Shared Weaknesses Across All Papers

1. **Manual templates survive:** Even "automatic" methods (LMSPS, MCTS) still need a candidate pool derived from domain knowledge; they select/rank but do not generate from scratch
2. **Node classification dominance:** 9/11 papers evaluate only on node classification; link prediction and graph classification are underexplored
3. **Small standard benchmarks:** DBLP, IMDB, ACM appear in 8/11 papers — the field has benchmark saturation; new dataset contributions are needed
4. **No geometric grounding:** None of the 11 papers connect meta-path selection to the differential geometry of the learned representation (holonomy, curvature) — this is the unique space H2S-NN can occupy

---

## 13. Research Opportunity Map

> **Professor's perspective:** If you're writing a WSDM paper from this foundation, here is where the genuine contributions lie.

### 13.1 Tier 1 — High Impact, Moderate Difficulty

**A. Schema-Holonomy-Guided Meta-path Selection (direct extension of H2S-NN)**

- *Gap:* All 11 papers select meta-paths using task loss signals (accuracy, NDCG). None uses the geometric structure of the learned sheaf itself.
- *Proposal:* Use holonomy of restriction maps around closed paths as a geometry-native informativeness score. High holonomy = sheaf is "twisted" along that path = face adds new topological content. Differentiable via Gumbel-softmax over schema weights.
- *Novel claim:* "Meta-path schema importance equals sheaf curvature along that schema — no validation set required."
- *Key references to cite:* LMSPS (Gumbel selection), SchemaWalk (schema-level search), Neural Sheaf Diffusion (holonomy background).

**B. Weighted Coboundary Operator (mathematically tight integration)**

- *Gap:* All papers treat meta-path selection as a preprocessing step, separate from the model. H2S-NN can integrate selection *into* the sheaf construction.
- *Proposal:* Replace δ₁ (fixed coboundary) with $$\tilde{\delta}_1 = \sum_\sigma \sqrt{\alpha_\sigma} \cdot \delta_1^\sigma$$, where α_σ are learned schema weights. The Hodge Laplacian becomes $$\tilde{L}_1 = \delta_0\delta_0^\top + \sum_\sigma \alpha_\sigma (\delta_1^\sigma)^\top \delta_1^\sigma$$ — still PSD, still valid Hodge decomposition.
- *Novel claim:* "We are the first to make the 2-cell topology of a cellular sheaf end-to-end learnable."

### 13.2 Tier 2 — Moderate Impact, Lower Difficulty

**C. Long-range Sheaf Diffusion on Sparse Graphs**

- Based on LMSPS finding: sparse graphs benefit most from long-range paths. H2S-NN's L₁ diffusion already captures multi-hop structure through face-edge relationships. Extending face lengths to L=5–8 (currently capped at 4) and applying progressive sampling would be a direct contribution.

**D. Schema Transferability Across Datasets**

- LPSMS shows meta-paths discovered for one architecture transfer to others. H2S-NN could show that *sheaf schemas* (α_σ weights) discovered on one heterogeneous graph transfer to graphs with the same type schema. This would be a strong practical contribution for WSDM's data mining audience.

### 13.3 Tier 3 — Novel but High-Risk

**E. Persistent Homology for Meta-path Relevance**

- Compute persistence diagrams of the filtrated sheaf complex as path lengths grow. Persistent topological features (long-lived homology classes) correspond to meta-paths that create stable, non-trivial topology — automatically the most informative ones.
- High risk: persistent homology computation is expensive; differentiability is non-trivial.

---

## Part II — Search with Guarantees (3 papers from `research_paper_reinforce/`)

> **Why these matter for H2S-NN:** Meta-path schema discovery is a discrete, deterministic search problem. These three papers provide the theoretical machinery to do that search with formal worst-case guarantees — something completely absent from all 11 papers in Part I.

---

## 12. LevinTS / LubyTS

**Full title:** *Single-Agent Policy Tree Search With Guarantees*

### 12.1 Problem Formulation

**The gap:** Monte Carlo Tree Search (MCTS) was designed for adversarial, stochastic domains (games). In single-agent deterministic problems with sparse rewards (goal found only at terminal states), MCTS performs poorly because its UCT score is calibrated for win/loss outcomes, not path quality. The question: **can a learned policy π directly guide tree search with a formal worst-case bound on the number of nodes expanded?**

**Setting:** Deterministic state space with goal set N^g. Policy π assigns probabilities to actions. Search cost = total node expansions.

### 12.2 Methodology

**LevinTS — Policy-Guided Best-First Search:**

Expand nodes in increasing order of:
$$\frac{d_0(n)}{\pi(n)}$$
where d₀(n) = depth of node n, π(n) = policy probability of the path leading to n.

State cuts: skip already-visited states to avoid re-expansion.

**LubyTS — Randomized Restarts:**

Sample trajectories to depth d_max; restart using the Luby sequence {1,2,1,4,1,2,1,8,...} which is optimal for problems where the halting probability is unknown.

### 12.3 Key Theorems

**Theorem (LevinTS guarantee):**
$$N(\text{LevinTS}, N^g) \leq \min_{n \in N^g} \frac{d_0(n)}{\pi(n)}$$
Total expansions bounded by the minimum depth/probability ratio over all solution nodes. If the policy assigns high probability to the solution path, very few nodes are expanded.

**Theorem (LubyTS guarantee):**
$$\mathbb{E}[N(\text{LubyTS}, N^g)] \leq \min_d \left[d + \frac{d}{\pi^+_d}\left(\log_2\frac{d}{\pi^+_d} + 6.1\right)\right]$$
where $\pi^+_d = \sum_{n: d_0(n) \leq d} \pi(n)$ is the cumulative probability mass at depth d.

### 12.4 Results

On 1,000 Sokoban levels (10×10, 4 boxes), policy trained with A3C:

| Method | Solved | Avg. length | Total expansions |
|--------|--------|-------------|-----------------|
| Uniform | 88 | 19 | 94,423,278 |
| MCTS | ~600 | 43 | — |
| LAMA planner | 1,000 | 51.6 | 3,151,325 |
| **LevinTS** | **1,000** | **39.8** | **6,602,666** |
| **LevinTS (ε=0.01)** | **1,000** | **39.5** | **5,026,200** |

LevinTS finds **shorter solutions** than LAMA with **comparable expansions**.

### 12.5 Strengths

- Formal worst-case bound on expansions — unique among all 18 papers reviewed
- State cuts eliminate redundant re-expansion for free
- Works with any learned policy (A3C, PPO, supervised) without modification
- Bound degrades gracefully with policy noise

### 12.6 Weaknesses

- Bound depends heavily on π(n): a poorly trained policy gives a vacuous bound
- Memory grows linearly with expanded nodes (problematic for very deep searches)
- Requires deterministic, observable state space — excludes stochastic graphs
- No mechanism to incorporate a heuristic function alongside the policy

### 12.7 Applicability to Meta-path Schema Search

> **Professor's note:** Meta-path schema search is exactly the setting LevinTS was designed for: deterministic, single-agent, sparse reward (schema utility only known after instantiation and evaluation). The bound N ≤ min_σ [|σ|/π(σ)] tells us that if we can learn a policy π that assigns high probability to useful schemas, LevinTS will find them with very few schema evaluations. The key engineering question is: what is π trained on? Answer: the task loss on validation, using the GNN's current restriction maps to estimate schema utility.

---

## 13. LHBL

**Full title:** *Reinforcement Learning of Heuristics with Limited-Horizon Search*

### 13.1 Problem Formulation

**The gap:** Standard heuristic learning for graph search uses **single-step Bellman updates**: h(s) ← min_{s'} [c(s,s') + h(s')]. This only uses the immediate successor's estimate — ignoring the full subtree structure below. For hard combinatorial problems (Rubik's Cube, large sliding tile puzzles), this leads to slow convergence and systematically underestimated costs.

**Goal:** Learn a heuristic h_θ(s) that minimizes search cost by incorporating multi-step lookahead into the training signal.

### 13.2 Methodology

**Limited-Horizon Bellman Learning (LHBL):**

1. Run search algorithm (A*, GBFS) from start node for N steps
2. Add auxiliary terminal node z; connect all frontier nodes ℓ to z with edge cost c(ℓ,z) = h_θ(ℓ) (current heuristic estimate)
3. Reverse graph edges and run Dijkstra backward from z
4. Assign h_LHBL(v) = shortest path in reversed graph from v to z

**Key equation:**
$$h_{\text{LHBL}}(s) = \min_{\ell \in \text{frontier}} \left[C(s, \ell) + h_\theta(\ell)\right]$$
where C(s,ℓ) = accumulated path cost from s to frontier node ℓ.

This uses the full search subtree to estimate cost-to-go, not just one step.

**Horizon parameter H:** controls depth of lookahead. H=10 is cheapest, H=100 most informative but expensive.

### 13.3 Results

| Domain | Method | % Solved | Avg. length | Expansions |
|--------|--------|---------|-------------|-----------|
| 5×5 Sliding Tile | SSBL | 99.7% | 51.0 | 1,044 |
| 5×5 Sliding Tile | **LHBL(10)** | **100%** | **41.3** | **196** |
| Rubik's Cube | SSBL | 68.8% | — | — |
| Rubik's Cube | **LHBL(10)** | **100%** | — | — |

LHBL(10) reduces expansions by 5× on the sliding tile puzzle and fully solves Rubik's Cube where SSBL fails.

### 13.4 Strengths

- Multi-step lookahead dramatically improves training convergence and final quality
- Auxiliary node construction is computationally clean (single Dijkstra call)
- Works with any base search algorithm (A*, GBFS, Dijkstra)
- Horizon is a principled trade-off knob: short = fast, long = informative

### 13.5 Weaknesses

- SSSP computation required at every training step — O(|V|log|V| + |E|) overhead
- Horizon H must be manually tuned; no adaptive selection strategy
- Assumes cycle-free DAG structure for auxiliary node construction
- No theoretical analysis of convergence rate vs. horizon length

### 13.6 Applicability to Meta-path Schema Search

> **Professor's note:** In the meta-path context, LHBL's insight is directly applicable. The "heuristic" h(σ) for a partial schema σ should not be estimated from the immediate next relation type alone (single-step = SSBL), but from the full rollout of what that schema leads to: what restriction map holonomies it produces, what subgraph it covers, and what task performance results. A limited-horizon rollout (e.g., instantiate σ with H random instances, evaluate their holonomy and coverage) provides a multi-step lookahead heuristic that is dramatically more informative than single-step estimates. This eliminates the need for a full GNN training run per schema evaluation.

---

## 14. PHS

**Full title:** *Policy-Guided Heuristic Search with Guarantees*

### 14.1 Problem Formulation

**The gap:** LevinTS uses policy π alone. Classical heuristic search (A*, GBFS) uses admissible heuristic h alone. Both have limitations: LevinTS ignores geometric structure; A* ignores learned policy. **Can we combine both with a formal guarantee better than either alone?**

**Setting:** Minimize total search loss L(S, n*) = Σ_{n visited} ℓ(n), where ℓ(n) is any non-negative cost function (e.g., node expansion count, time, path cost).

### 14.2 Methodology

**PHS Evaluation Function:**
$$\varphi(n) = \eta(n) \cdot \frac{g(n)}{\pi(n)}$$
where:
- g(n) = accumulated path loss from root to n
- π(n) = policy probability of path to n
- η(n) ≥ 1 = heuristic factor (geometric: how much harder is the remaining search from n?)

Expand nodes in increasing order of φ(n) — lower φ = higher priority.

**Connection to A*:** If h is an A*-admissible heuristic, define:
$$\eta_h(n) = \frac{g(n) + h(n)}{g(n)} \geq 1$$
Then φ_h(n) = [g(n) + h(n)] / π(n) — the PHS-admissible version of A*+policy.

### 14.3 Key Theorem

**Theorem (PHS upper bound on search loss):**
$$L(\text{PHS}, n^*) \leq \frac{g(n^*)}{\pi(n^*)} \cdot \eta^*(n^*) \cdot \sum_{n \in \mathcal{L}(n^*)} \frac{\pi(n)}{\eta^*(n)}$$

where η*(n) = max over descendants of n, and L(n*) = leaves on the solution path.

- If η ≡ 1 (no heuristic): recovers LevinTS bound exactly
- If η > 1 everywhere (good heuristic): the sum shrinks → tighter bound than LevinTS
- **Corollary:** PHS with a good heuristic always has a tighter guarantee than policy alone

**Lower bound (Theorem 4):**
$$\min_S L(S, n^*) \geq \frac{g(n^*)}{\pi(n^*)}$$
This shows the PHS upper bound is within a constant factor of the information-theoretic lower bound — the algorithm is near-optimal.

### 14.4 Results

On three planning domains (Sokoban, The Witness, Sliding Tile 5×5):

| Method | Sokoban (expansions) | Witness (expansions) | Tile 5×5 (%solved) |
|--------|---------------------|---------------------|-------------------|
| WA* (w=1.5) | 3,729 | 18,345 | 22/1000 |
| LevinTS | 2,640 | 520 | — |
| PHS_h | 2,130 | 408 | 1000/1000 |
| **PHS* (η=η_h)** | **1,522** | **408** | **1000/1000** |

PHS* solves all levels in all three domains — the only method to do so — with the fewest expansions on Sokoban.

### 14.5 Strengths

- **Unifies policy guidance and heuristic search in one framework with a formal guarantee**
- Near-optimal bound: within constant factor of information-theoretic minimum
- PHS-admissibility is verifiable and natural (η(n) ≥ 1, non-decreasing φ up the path)
- Works with any combination of learned policy (neural network) and any admissible heuristic
- Consistent improvements over LevinTS across all domains

### 14.6 Weaknesses

- Bound still depends on policy quality: poor π gives a large g(n*)/π(n*) factor
- PHS-admissibility definition is complex; verifying it for new heuristics requires careful analysis
- Learning both policy and heuristic simultaneously creates a chicken-and-egg bootstrapping problem
- Assumes non-negative loss function; does not handle reward-maximization directly

### 14.7 Applicability to Meta-path Schema Search

> **Professor's note:** PHS is the correct framework for H2S-NN's meta-path schema search. The mapping is exact:
> - **Policy π(σ):** Probability that schema σ is useful → trained LSTM over type-token sequence, gradient-updated from task loss
> - **Heuristic η(σ):** Holonomy score of current restriction maps along partial schema σ → ‖Hol_F(σ) - I_d‖_F / max_holonomy (normalized, ≥ 1)
> - **Path cost g(σ):** Length of the schema (number of hops)
> - **Search objective:** Find σ* that minimizes task loss
>
> The PHS guarantee then gives: the schema search terminates having expanded at most [|σ*|/π(σ*)] · η*(σ*) · Σ_{σ on path} π(σ)/η*(σ) schemas. With a well-trained policy and non-trivial holonomy scores, this is far fewer than the exponential enumeration. **This is the first formally bounded meta-path schema discovery algorithm.**

---

## Part III — Sheaf Extensions and NAS Insights (4 papers from `research_paper02/`)

---

## 15. DeepSN

**Full title:** *DeepSN: A Sheaf Neural Framework for Influence Maximization*

### 15.1 Problem Formulation

**Setting:** Influence Maximization (IM): given graph G = (V, E) and budget k, find seed set S* = argmax_{|S|≤k} σ(S; G; θ) where σ is expected influence spread. This is NP-hard; classical algorithms (IMM, CELF) use Monte Carlo approximation.

**The gap:** Existing GNN approaches to IM (GCN, GAT, GraphSAGE) use standard message passing that ignores the geometric structure of how influence propagates differently across different edges. Sheaf theory provides the mathematical framework to encode edge-specific propagation operators.

### 15.2 Methodology

**Cellular Sheaf for IM:**

A sheaf F on G with stalks F(v) ∈ ℝ^d at each node and restriction maps F_{v⊆e}: ℝ^d → ℝ^d at each incident vertex-edge pair.

**Sheaf Laplacian:**
$$\mathcal{L}_F(x_v) = \sum_{e \ni v} F_{v \subseteq e}^\top \cdot \psi_{v,u} \cdot (F_{v \subseteq e} x_v - F_{u \subseteq e} x_u)$$
where ψ_{v,u} ∈ [0,1] are **learnable sheaf coefficients** (the novel parameterization).

**Reaction-diffusion operators:**
- **Pointwise:** $A(X_v(t)) = \Phi^j_v \odot \frac{\kappa^1_v}{\kappa^1_v + |X_v(t)|}$ (captures intrinsic vertex activation dynamics)
- **Coupled:** $Y_v(t) = \sum_{u \in N(v)} (S^j_u - X_u(t)) - \sum_{u \in N(v)} (1 - S^j_u) X_u(t)$ (captures neighborhood influence)

**Full PDE:**
$$\frac{\partial X_v(t)}{\partial t} = -\alpha \Delta_F X_v(t) + \beta A(X_v(t)) + \gamma R(X(t), A, S^j)$$

Discretized: $X(t+1) = X(t) - \alpha \frac{\partial X}{\partial t}$

### 15.3 Contributions

- First application of sheaf theory to influence maximization
- Reaction-diffusion operators model both diffusion and intrinsic vertex activation
- Formal stability proofs (Propositions 1-3 on vertex feature separability)
- Subgraph-based seed optimization reducing combinatorial search

### 15.4 Results

| Dataset | Nodes | Edges | DeepSN vs. best GNN baseline |
|---------|-------|-------|-------------------------------|
| Jazz | 198 | 2,742 | +2.7% accuracy |
| Cora-ML | 2,810 | 7,981 | Competitive |
| Power Grid | 4,941 | 6,594 | Best under LT model |
| Digg | 279,613 | 1,170,689 | Lower MSE by 5.3% |

### 15.5 Strengths

- Mathematically principled: sheaf Laplacian correctly encodes anisotropic diffusion
- Reaction operators model realistic two-stage influence dynamics (exposure → activation)
- Formal separability propositions connect sheaf theory to expressiveness guarantees
- Scales to 279K node graphs

### 15.6 Weaknesses

- **1-skeleton only:** No 2-cells, no δ₁, no L₁ — operates purely on the node-edge sheaf. H2S-NN's cellular complex is strictly richer.
- **Restriction maps via scalar ψ_{v,u}:** Much simpler than H2S-NN's d×d matrix restriction maps; limited expressiveness
- **Limited to two influence states:** Cannot model multi-stage threshold cascades
- **No heterogeneous types:** Ignores node/edge type information entirely

### 15.7 Key Insight for H2S-NN

> **Professor's note:** DeepSN confirms that sheaf theory is useful for dynamic diffusion processes beyond classification — but it stays at the simplest possible sheaf (1-skeleton, scalar coefficients). The contrast with H2S-NN is stark: H2S-NN uses full d×d orthogonal restriction maps, a 2D cellular complex, and Hodge diffusion on both L₀ and L₁. This paper is best cited as "prior work showing sheaf diffusion for dynamic network processes" to position H2S-NN's richer structure as the natural next step.

---

## 16. Sheaf HyperNetworks (SHN)

**Full title:** *Sheaf HyperNetworks for Personalized Federated Learning*

### 16.1 Problem Formulation

**Setting:** Personalized Federated Learning (PFL): M clients with heterogeneous local data want to learn client-specific models θ_m while preserving privacy (no raw data sharing). Minimize:
$$\mathcal{L}_{\text{pfd}}(\theta) = \sum_m p_m \mathcal{L}_m(\theta), \quad p_m = t_m / N$$

**The gap:** Standard GNN-based PFL suffers from over-smoothing (embeddings converge to same value at depth) and heterophily (clients in different clusters are incorrectly smoothed together). Sheaf Laplacians, by learning per-edge restriction maps, can encode directional information flow that prevents both problems.

### 16.2 Methodology

**Sheaf GNN for clients:**
$$X_{t+1} = X_t - \sigma(\Delta_{F(t)}(I \odot W^\ell_\dagger) X_t W^\ell_\ddagger)$$
where Δ_{F(t)} is the normalized sheaf Laplacian with restriction maps F_{v⊆e}: ℝ^d → ℝ^d learned via MLP Φ.

**SHN two-stage pipeline:**

1. **HyperNetwork pretraining:** HN generates personalized client weights from embedding E(m; υ_C). Client relation graph built from cosine similarity of embeddings: $a_{vn} = \cos(x_v, x_n) \in [0,1]$.

2. **Sheaf HyperNetwork:** Sheaf GNN over client relation graph enables personalized weight generation while respecting client similarity geometry. Restriction maps are parameterized via orthogonal MLPs (same approach as Bodnar et al. 2022).

### 16.3 Results

| Dataset | SHN vs. pFedHN | SHN vs. GCN-based PFL |
|---------|----------------|----------------------|
| Cluster CIFAR100 | +2.7% accuracy | +4.1% |
| CompCars | +1.8% | +3.2% |
| PEMS-BAY (traffic) | −5.3% MSE | −8.1% MSE |

### 16.4 Strengths

- Novel domain: sheaves for federated learning is genuinely new
- Privacy-preserving client graph construction (no raw data exchange)
- Formal connection between sheaf over-smoothing resistance and heterophily mitigation
- Orthogonal restriction maps (matching H2S-NN's approach) are well-motivated

### 16.5 Weaknesses

- **1-skeleton only:** Same limitation as DeepSN — no 2-cells, no Hodge structure
- **Cosine similarity graph is heuristic:** k-NN threshold hyperparameter sensitive; no principled client graph construction
- **Sheaf Laplacian regularized ad hoc:** Adds ε·I to ensure positive-definiteness — bypasses proper Hodge decomposition
- **No type heterogeneity:** All clients treated as same "type"; not applicable to heterogeneous entity graphs

### 16.6 Key Insight for H2S-NN

> **Professor's note:** SHN shows sheaves are spreading into new application domains. Importantly, SHN proves (empirically) that orthogonal restriction maps — the same parameterization H2S-NN uses — are robust and generalizable. But SHN still uses only L₀ (node-level Laplacian). The absence of L₁ means SHN cannot capture edge-level harmonic content — which is H2S-NN's core contribution. Cite SHN to demonstrate that orthogonal sheaves are a validated building block; then distinguish H2S-NN via the Hodge decomposition.

---

## 17. NAS-RL (Zoph & Le 2017)

**Full title:** *Neural Architecture Search with Reinforcement Learning*

### 17.1 Problem Formulation

**The founding NAS paper.** Problem: find architecture a ∈ A minimizing validation loss L_val(w*(a), a) where w*(a) = argmin_w L_train(w, a). Architecture is a sequence of discrete design choices (filter sizes, strides, skip connections).

### 17.2 Methodology

**RNN controller generates architecture as token sequence:**

Policy gradient (REINFORCE with baseline b):
$$\nabla_{\theta_c} J(\theta_c) = \frac{1}{m}\sum_{k=1}^m \sum_{t=1}^T \nabla_{\theta_c} \log P(a_{t|a_{1:t-1}}; \theta_c) \cdot (R_k - b)$$
where R_k = validation accuracy of k-th sampled architecture, b = exponential moving average.

**Skip connections** via sigmoid attention: P(layer j is input to i) = σ(v^T tanh(W_prev h_j + W_curr h_i)).

**Distributed training:** 800 GPUs, S=20 parameter servers, K=100 controller replicas, m=8 architectures per batch.

### 17.3 Results

- CIFAR-10: **3.65% error** (better than DenseNet 3.74%)
- Penn Treebank RNN cell: **62.4 perplexity** (3.6× better than prior SOTA)
- Search time: 13 GPU-days (≈10,400 GPU-hours)

### 17.4 Key Weaknesses for Our Context

- **Extremely expensive:** 800 GPUs × 13 days — impractical for academic research
- **REINFORCE has high variance:** Requires many samples to reduce gradient noise; baseline only partially helps
- **NAS survey (§18) shows:** Evolutionary algorithms and DARTS consistently outperform RL-based NAS in head-to-head comparisons
- **No convergence guarantee:** Policy gradient for architecture search has no formal bound

### 17.5 Critical Lesson for H2S-NN

> **Professor's note:** This paper is the reason NOT to use pure RL (DQN, REINFORCE) for meta-path schema search in H2S-NN. The cost is prohibitive, variance is high, and alternatives (DARTS-style gradient methods, PHS with guarantees) are both cheaper and have better theoretical properties. RMS-HRec (§3) made the mistake of adopting DQN from this lineage. H2S-NN should use PHS (§14) for search with guarantees, and Gumbel-softmax (DARTS-like) for end-to-end differentiation.

---

## 18. NAS Insights from 1000 Papers

**Full title:** *Neural Architecture Search: Insights from 1000 Papers*

### 18.1 Problem Formulation

A systematic survey of 1000+ NAS papers (2020–2022) taxonomizing search spaces, search strategies, and performance estimation. Core NAS problem: min_{a∈A} L_val(w*(a), a) within compute budget t.

### 18.2 Key Findings Relevant to Meta-path Schema Search

**Finding 1 — RL is NOT the recommended search strategy:**

The survey documents that evolutionary algorithms (regularized evolution, Real et al. 2019) consistently outperform RL in head-to-head comparisons in well-designed search spaces. DARTS (gradient-based) is most practical for academic compute budgets.

> Direct quote context: "random search is surprisingly competitive for highly engineered search spaces" — meaning if the search space is well-constrained (e.g., type schemas of bounded length), random baselines are hard to beat with RL.

**Finding 2 — Search space design is more important than search strategy:**

The architecture encoding matters more than the optimizer. "Small encoding changes significantly affect NAS performance." For meta-path schemas, this means:
- How we represent a schema (one-hot types? graph structure?) affects what the search finds
- Type-token LSTM encoding (our proposal) is a principled design choice

**Finding 3 — One-shot / DARTS methods are most efficient:**

Train one supernet covering all candidate architectures → continuous relaxation → gradient-based optimization. This is exactly what our weighted coboundary operator ∑_σ α_σ δ₁^σ implements — a sheaf-theory supernet where all schemas are jointly active, with learned combination weights.

**Finding 4 — Cell-based spaces outperform macro-level spaces:**

Searching for reusable cells (= our type schemas σ) rather than full architectures (= individual meta-path instances) is the dominant strategy. This validates the schema-level (not instance-level) design of our approach.

**Finding 5 — MCTS underperforms evolutionary and gradient methods in NAS:**

MCTS-based NAS (Negrinho & Gordon, 2017; Wang et al., 2020b) consistently underperforms DARTS and evolutionary search. This directly challenges MCTS-HGNN (§1) as the right framework — its MCTS backbone is from a deprecated NAS paradigm.

### 18.3 Contribution Map to H2S-NN

| NAS Insight | Implication for H2S-NN |
|-------------|----------------------|
| RL is expensive, high-variance | Use PHS (guaranteed) + DARTS (gradient) instead of DQN/REINFORCE |
| DARTS (one-shot) is most efficient | Weighted coboundary ∑ α_σ δ₁^σ IS a one-shot sheaf supernet |
| Cell-based search (reusable units) is best | Schema-level search (type sequences) is correct granularity |
| Encoding matters most | Type-token LSTM is a principled schema encoding |
| MCTS underperforms in NAS | Do not extend MCTS-HGNN; PHS has formal guarantees and better empirics |
| Random search is competitive baseline | Must include random schema baseline in ablation |

---

## 15. Revised Cross-Paper Synthesis (v2)

*Supersedes §12 from Part I. Now incorporates all 18 papers.*

### 15.1 The Complete Problem Taxonomy

| Framing | Papers (all 18) | Core operation |
|---------|----------------|---------------|
| **Search with guarantees** | LevinTS, LubyTS, PHS | Navigate schema space with formal expansion bounds |
| **RL-based search** | RMS-HRec, SchemaWalk, NAS-RL | Policy gradient / DQN / REINFORCE (expensive, no guarantees) |
| **Gradient-based selection** | AMP-HGNAS, LMSPS, LPSMS, **PMMM** | DARTS / Gumbel-softmax over candidate pool |
| **Tree search (heuristic)** | MCTS-HGNN, HG-Search | MCTS / NAS with policy gradient (now known to underperform) |
| **Sheaf applications** | DeepSN, SHN, HETSHEAF | Sheaf on 1-skeleton (no 2-cells, no L₁) |
| **Augmentation** | SMCD, ESim | Embed/augment given meta-paths more effectively |
| **Domain-specific** | HISE-KT | LLM scoring for education-specific KT |

**The critical observation:** The search-with-guarantees papers (§12–14) are entirely absent from the HetGNN literature. Nobody has applied LevinTS or PHS to meta-path discovery. This is a clean gap.

### 15.2 The NAS Literature's Verdict on Strategy Choice

Based on the 1000-paper survey and empirical evidence:

```
Best (practical + theoretical) ──────────────────── Worst
       │                  │              │              │
  PHS + Gumbel       DARTS/Gumbel    Evolutionary    RL (DQN/
  (our proposal)      (LMSPS)         (random walk)  REINFORCE)
  Formal bounds +    Gradient, no    Competitive     No bounds,
  end-to-end         bounds          baseline        expensive
  differentiable
```

### 15.3 The Sheaf Literature's State (All Applications)

| Paper | Topology used | Restriction maps | L₁/δ₁? | Heterogeneous? |
|-------|--------------|-----------------|--------|---------------|
| Bodnar et al. 2022 (NSD) | 1-skeleton | d×d MLP | No | No |
| HETSHEAF | 1-skeleton | d×d, type-conditioned | No | Yes |
| DeepSN | 1-skeleton | scalar ψ_{v,u} | No | No |
| SHN | 1-skeleton | d×d orthogonal | No | No |
| PMMM | Schema multigraph (1-skeleton) | None (scalar α_r per relation type) | No | Yes |
| **H2S-NN** | **2D cellular complex** | **d×d orthogonal, relation-specific** | **Yes** | **Yes** |

**H2S-NN is the only paper in all 19 that uses L₁ and δ₁.** PMMM operates at the schema level (which relations to use between hyper-node types) but has no face topology, no coboundary operator, and no Hodge decomposition. The Hodge decomposition on edge cochains remains a genuine differentiator.

### 15.4 Why Previous Meta-path Automation Fails

| Method | Fails because |
|--------|-------------|
| MCTS-HGNN | No formal guarantee; MCTS deprecated by NAS survey |
| RMS-HRec, NAS-RL | RL is expensive and high-variance; DQN unstable |
| SchemaWalk | Schema-level RL works but provides no topology integration |
| LMSPS/LPSMS | Gradient-based but assumes candidate pool is given (manual templates still needed for initial set) |
| AMP-HGNAS | Dependency threshold δ is manual; MIS approximation discards useful paths |
| PMMM | Stable DARTS over multi-edges — but still schema-level only; no 2-cells, no L₁, no instance-level topology; λ for derivation threshold is manual |

**What's missing:** A method that (a) searches without manual templates, (b) has formal bounds, (c) integrates discovery directly into the sheaf topology (not as preprocessing), (d) uses gradient-based end-to-end learning for the final combination, and (e) stabilizes bilevel optimization without random seed fragility.

---

## 16. Revised Research Direction: PHS-Sheaf (Think Again)

> **The professor thinks again.** After reading the 7 new papers, the original proposal (schema LSTM + Gumbel-softmax) is correct in spirit but incomplete. The search-with-guarantees papers provide the missing piece: a principled, bounded schema discovery phase. The NAS survey confirms that our gradient-based weighting is the right combination mechanism. The sheaf papers confirm that 2-cells and L₁ are unique to H2S-NN. Here is the refined, complete proposal.

### 16.1 What Changed After Reading the 7 New Papers

**Before (v1 proposal):**
- Enumerate all schemas of bounded length (potentially thousands)
- Learn α_σ via LSTM + Gumbel-softmax over all schemas
- Weighted coboundary ∑_σ √α_σ δ₁^σ

**Problems with v1 (now visible):**
1. Enumerating all schemas is still exponential — same problem as manual templates, just automated
2. Gumbel-softmax over thousands of schemas has high variance (NAS survey warning)
3. No formal guarantee on schema discovery quality

**After (v2 proposal — PHS-Sheaf):**

Two mathematically distinct stages with clean separation of concerns:
- **Stage 1 — PHS-Schema:** Bounded search for a compact schema pool (replaces enumeration)
- **Stage 2 — Gumbel-Sheaf:** Differentiable topology learning over the discovered pool (DARTS-style)

### 16.2 Stage 1: PHS-Schema — Policy-Heuristic Schema Discovery

**Formalize the schema search as a PHS problem (§14):**

- **State space:** Partial type-schema sequences (T₀, r₁, T₁, ..., rₗ, Tₗ) for ℓ ≤ L_max
- **Goal states:** Complete schemas that produce high-utility 2-cells
- **Policy π(σ):** LSTM over type-token sequence, trained from schema utility signal (task loss)
- **Heuristic η(σ):** Holonomy score of current restriction maps along σ:
$$\eta(\sigma) = 1 + \frac{1}{|F_\sigma^{\text{sample}}|} \sum_{f \in F_\sigma^{\text{sample}}} \| \text{Hol}_F(\pi_f) - I_d \|_F$$
  Computed cheaply from K random instances without full GNN training (LHBL-style limited rollout, §13)
- **PHS evaluation function:** $\varphi(\sigma) = \eta(\sigma) \cdot |\sigma| / \pi(\sigma)$
- **Search:** Expand schemas in increasing φ order (LevinTS-style best-first) until M high-scoring schemas are found

**Formal guarantee (from PHS Theorem 1):**
$$L(\text{PHS-Schema}, \sigma^*) \leq \frac{|\sigma^*|}{\pi(\sigma^*)} \cdot \eta^*(\sigma^*) \cdot \sum_{\sigma \in \mathcal{L}(\sigma^*)} \frac{\pi(\sigma)}{\eta^*(\sigma)}$$

Total schemas evaluated is bounded. With a trained policy (π assigns high probability to useful schemas) and holonomy-based η (high for geometrically informative schemas), the bound is tight.

**Holonomy as PHS-admissible heuristic (mathematical validity):**

We need η(σ) ≥ 1 (satisfied by definition above) and φ*(σ) = max_{σ'∈desc(σ)} φ(σ') non-decreasing from child to parent (satisfied because extending a schema with an additional hop can only increase the holonomy, as composition of restriction maps accumulates curvature).

This means the holonomy score is a **valid PHS-admissible heuristic** — the first geometry-native admissible heuristic for combinatorial schema search.

### 16.3 Stage 2: Gumbel-Sheaf — Differentiable Topology Learning

Once PHS-Schema delivers a compact pool S_pool = {σ₁,...,σ_M} (M ≪ total schemas):

**Learn schema importance weights via Gumbel-softmax:**
$$\alpha_\sigma = \frac{\exp\!\left[(\theta_\sigma + u_\sigma)/\tau\right]}{\sum_{\sigma' \in S_{\text{pool}}} \exp\!\left[(\theta_{\sigma'} + u_{\sigma'})/\tau\right]}, \quad u_\sigma \sim \text{Uniform}(0,1)$$

where θ_σ is a learnable scalar per schema, and τ is a temperature that anneals during training.

**Weighted coboundary operator (valid sheaf construction):**
$$\tilde{\delta}_1 = \sum_{\sigma \in S_{\text{pool}}} \sqrt{\alpha_\sigma} \cdot \delta_1^\sigma$$

**Weighted Hodge Laplacian:**
$$\tilde{L}_1 = \delta_0 \delta_0^\top + \tilde{\delta}_1^\top \tilde{\delta}_1 = \delta_0 \delta_0^\top + \sum_\sigma \alpha_\sigma (\delta_1^\sigma)^\top \delta_1^\sigma$$

**Mathematical validity:** Each $(\delta_1^\sigma)^\top \delta_1^\sigma$ is PSD; their convex combination (since α_σ ≥ 0, ∑ α_σ = 1) is also PSD. The Hodge decomposition holds:
$$C^1(G, F) = \text{im}(\delta_0) \oplus \tilde{\mathcal{H}}^1 \oplus \text{im}(\tilde{\delta}_1^\top)$$

The harmonic space H̃¹ is exactly the space of "disagreement" not explained by either node gradients or face boundaries — the learned geometry of the heterogeneous graph.

**End-to-end training:** α_σ weights are trained jointly with restriction maps F^{src}_r, F^{tgt}_r (Stage A) and F_{e▷f} (Stage B) via standard backpropagation. The Gumbel noise provides a discrete relaxation that allows post-training discretization (select top-K schemas by α_σ).

### 16.4 The Complete PHS-Sheaf Algorithm

```
Input: Heterogeneous graph G, node labels Y, max schema length L_max, pool size M
Output: Learned sheaf F, schema weights {α_σ}, node predictions Ŷ

Phase 0 — Warmup (standard H2S-NN, fixed manual templates):
  Train restriction maps F^src_r, F^tgt_r for T_warmup epochs on manual templates
  → Restriction maps are now informative enough to compute meaningful holonomy scores

Phase 1 — PHS-Schema (schema pool construction):
  Initialize policy LSTM π with random weights
  For k = 1 to K_search:
    Run PHS with φ(σ) = η(σ)·|σ|/π(σ) where η(σ) = holonomy_score(σ, F_current)
    Expand top-φ schemas; evaluate utility on validation set
    Update π via policy gradient: ∇_θ log π(σ) · utility(σ)
  S_pool ← top-M schemas by utility

Phase 2 — Gumbel-Sheaf (end-to-end differentiable training):
  Initialize α_σ = 1/M for all σ ∈ S_pool
  For epoch = 1 to T_train:
    Compute α_σ via Gumbel-softmax (temperature τ anneals 1.0 → 0.1)
    Construct ∑_σ α_σ (δ₁^σ)ᵀδ₁^σ → tilde_L1
    Forward pass: x^(T), s^(T) via Hodge diffusion with tilde_L1
    Loss: L = L_cls + λ_diff·L_diff(tilde_delta1) + λ_entropy·H(α)  [entropy regularizer]
    Backprop through α_σ, F^src, F^tgt, F_{e▷f} simultaneously

Output: Sparse α (few schemas dominate), trained restriction maps, node predictions
```

### 16.5 Novel Theoretical Contributions

**Theorem 1 (PHS-Schema Bound):** Let π be the learned schema policy and η be the holonomy-based PHS-admissible heuristic. The PHS-Schema algorithm finds the optimal schema pool with total schema evaluations bounded by:
$$L(\text{PHS-Schema}, \sigma^*) \leq \frac{|\sigma^*|}{\pi(\sigma^*)} \cdot \eta^*(\sigma^*) \cdot \sum_{\sigma \in \mathcal{L}(\sigma^*)} \frac{\pi(\sigma)}{\eta^*(\sigma)}$$
*Proof follows directly from PHS Theorem 1 (§14) with the holonomy admissibility condition verified above.*

**Theorem 2 (Gumbel-Sheaf Validity):** For any schema pool S_pool and weights α_σ ≥ 0 with ∑ α_σ = 1, the operator $\tilde{L}_1 = \delta_0\delta_0^\top + \sum_\sigma \alpha_\sigma (\delta_1^\sigma)^\top \delta_1^\sigma$ is positive semi-definite, and the Hodge decomposition $C^1(G,F) = \text{im}(\delta_0) \oplus \tilde{\mathcal{H}}^1 \oplus \text{im}(\tilde{\delta}_1^\top)$ holds.
*Proof: PSD by convex combination; Hodge decomposition by standard functional analysis on Hilbert spaces.*

**Corollary (Subsumption):** Setting α_σ = 1 for manual templates and 0 otherwise recovers the original H2S-NN exactly. The proposed method strictly generalizes the current model.

### 16.6 Why This Is Novel vs. Every Paper in the Survey

| Property | Closest competitor | PHS-Sheaf |
|----------|-------------------|-----------|
| Formal search bounds | None (all 19 surveyed papers use heuristic/RL search) | PHS Theorem 1: explicit expansion bound |
| Geometry-native heuristic | None | Holonomy score η(σ) from restriction maps |
| End-to-end topology learning | PMMM (schema-level α_r only) | α_σ trained jointly with sheaf maps F^{src}/F^{tgt}/F_{e▷f} |
| 2-cells + L₁ + automatic discovery | None (PMMM has automatic selection but no 2-cells or L₁) | All three simultaneously |
| Subsumes manual templates | No paper addresses this | Special case at α_σ∈{0,1} |
| Hodge decomposition preserved | — | Theorem 2 |
| Stable bilevel optimization | PMMM (PMP random subsampling) | PMP-Sheaf (§16.8): same technique applied to Gumbel-Sheaf α_σ updates |

### 16.7 Experiments That Would Validate This at WSDM

1. **Main results:** PHS-Sheaf vs. H2S-NN-manual on heterophilic benchmarks (Amazon Ratings, Roman Empire, Tolokers). Claim: automatic discovery matches or exceeds manual templates.

2. **Schema interpretability:** After training on ACM, inspect learned α_σ — does the model recover {PAP: high, PSP: high} without being told? (Validates Theorem 1 implicitly.)

3. **Search efficiency ablation:** PHS-Schema vs. random schema enumeration vs. exhaustive enumeration. Report number of schemas evaluated before convergence. Claim: PHS-Schema requires exponentially fewer evaluations.

4. **Holonomy as heuristic quality:** Compare η(σ) scores before and after warmup — shows that restriction map training improves the heuristic, creating a positive feedback loop.

5. **Topology validity check:** Verify L̃₁ eigenvalues ≥ 0 throughout training (validates Theorem 2 empirically). Verify Hodge decomposition on final trained model.

6. **New dataset with no known meta-paths:** Apply to a graph where domain experts cannot specify templates (e.g., a large biomedical KG). PHS-Sheaf should work; manual H2S-NN cannot.

### 16.8 PMMM Insight: Stabilizing Gumbel-Sheaf via Partial Schema Propagation

> **After reading PMMM (§19), v3 update:** PMMM identifies a root cause of DARTS-style instability in heterogeneous graph NAS: coupled bilevel optimization causes α parameters (architecture weights) to overfit to the current state of model weights W, and vice versa, in the same gradient step. Their fix — **partial message propagation (PMP)** — randomly excludes a fraction (1/p) of candidate paths per iteration, breaking the circular dependency.

**The same instability applies to Gumbel-Sheaf's Phase 2.** In Phase 2 of PHS-Sheaf, α_σ (schema weights) are updated on the validation set while F^{src}/F^{tgt}/F_{e▷f} are updated on the training set. If both are updated in the same forward pass, the gradient signal for α_σ is computed under the influence of the current F maps — and vice versa. This is precisely the coupled-optimization problem PMMM identifies for DiffMG.

**Fix — Partial Schema Propagation (PSP) for Gumbel-Sheaf:**

In each Phase 2 training iteration, randomly sample an active schema subset:

$$\tilde{S} = \{\sigma \in S_{\text{pool}} \mid M_\sigma = 1\}, \quad M_\sigma \sim \text{Bernoulli}(1/p), \quad p \in \{2, 3, 4\}$$

Compute the coboundary only over the active subset:

$$\tilde{\delta}_1^{(\text{iter})} = \sum_{\sigma \in \tilde{S}} \sqrt{\alpha_\sigma} \cdot \delta_1^\sigma$$

The gradient for θ_σ (the logit for schema σ) is computed from iterations where σ ∉ $\tilde{S}$ — so the α_σ gradient is never contaminated by the current F-map state that includes σ's own contribution. This mirrors PMMM's Eq.5–6 exactly, transplanted into the sheaf coboundary setting.

**Updated Phase 2 training loop:**

```
Phase 2 — Gumbel-Sheaf with PSP (stable bilevel):
  Initialize α_σ = 1/M for all σ ∈ S_pool
  For epoch = 1 to T_train:
    # Step A: Update model weights F^src, F^tgt, F_{e▷f} on training batch
    Sample active subset S̃ ← {σ | M_σ=1}, M_σ ~ Bernoulli(1/p)
    Compute tilde_L1 over S̃ only
    L_train = L_cls + λ_diff·L_diff(S̃) + λ_co·L_co
    Backprop: update F^src, F^tgt, F_{e▷f} (freeze θ_σ)

    # Step B: Update schema weights α_σ on validation batch  
    Sample different active subset S̃' (excludes schemas updated in Step A)
    Compute tilde_L1 over S̃'
    L_val = L_cls_val + λ_entropy·H(α)
    Backprop: update θ_σ (freeze F maps)
```

**Why this is mathematically sound:** The random subsampling does not change the expected value of the coboundary operator (E[∑_{σ∈S̃} p·α_σ δ₁^σ] = ∑_σ α_σ δ₁^σ with p·Bernoulli(1/p) rescaling), so the Hodge decomposition validity (Theorem 2) is preserved in expectation. The PSD property of $\tilde{L}_1$ holds per-realization since each α_σ ≥ 0.

**Connection to PMMM:** This is not an ad hoc borrowing — the mathematical structure is identical. PMMM's "path" ↔ H2S-NN's "schema σ"; PMMM's "path strength p_r^{(i,j)}" ↔ H2S-NN's "schema weight α_σ"; PMMM's "adjacency 𝒜_r" ↔ H2S-NN's "coboundary δ₁^σ". The analogy is exact, which makes PSP a principled adaptation rather than an engineering trick.

---

## 19. PMMM

**Full title:** *Differentiable Meta Multigraph Search with Partial Message Propagation on Heterogeneous Information Networks*

### 19.1 Problem Formulation

**The gap:** Existing HetGNN methods assume a single message-passing path between any two schema node types. Meta-graphs (DiffMG) allow multiple paths but restrict to exactly one message type per pair of schema types. Neither captures the full expressiveness of simultaneous multi-relational interactions.

**Core insight:** In a heterogeneous graph G = (V, E) with schema S = {T, R}, between any two node types T_i and T_j there may exist multiple distinct relation types or path sequences $\mathcal{R}^{(i,j)} \subseteq \mathcal{R}$. Using all of them simultaneously — the **meta multigraph** — is strictly more expressive than both meta-paths and meta-graphs.

**Problem 1:** Which subset $\hat{S}^{(i,j)} \subseteq \mathcal{R}^{(i,j)}$ of message-passing types should be used? Manual selection requires domain knowledge; exhaustive use adds noise.

**Problem 2:** DARTS-style bilevel optimization over the meta multigraph search space is unstable — different random seeds yield different selected architectures (the "DiffMG reproducibility crisis"). Standard DARTS couples weight optimization and architecture optimization in each gradient step, creating interference.

### 19.2 Methodology

#### A. Meta Multigraph Supernet

The message-passing supernet aggregates across all candidate paths between hyper-nodes:

$$H^{(j)} = \sum_{i < j} \sum_{r \in \mathcal{R}^{(i,j)}} p_r^{(i,j)} \cdot f\!\left(\mathcal{A}_r^{(i,j)},\, H^{(i)}\right) \quad [\text{Eq.1}]$$

where $\mathcal{A}_r^{(i,j)}$ is the adjacency for relation r from type i to j, f(·) is a message-passing function, and the path strength is a softmax over learnable architecture parameters:

$$p_r^{(i,j)} = \frac{\exp\!\left(\alpha_r^{(i,j)}\right)}{\sum_{r' \in \mathcal{R}^{(i,j)}} \exp\!\left(\alpha_{r'}^{(i,j)}\right)} \quad [\text{Eq.2}]$$

**Meta multigraph vs. meta-graph:** A meta-graph restricts to $|\mathcal{R}^{(i,j)}|=1$ (single path type between any schema pair); PMMM allows $|\mathcal{R}^{(i,j)}| > 1$. The multigraph structure captures richer interaction patterns when multiple relation types exist between the same node type pair.

#### B. Partial Message Propagation (PMP)

**The instability problem:** Standard DARTS updates α on the validation set and W on the training set but in the same forward pass. The gradient for α is computed under the influence of W, and vice versa — a coupled optimization that makes α overfit to the current W state.

**PMP solution:** In each forward pass, randomly sample an active subset of paths:

$$S^{(i,j)} = \{r \mid M_r^{(i,j)} = 1,\; r \in \mathcal{R}^{(i,j)}\}, \quad M_r^{(i,j)} \sim \text{Bernoulli}(1/p) \quad [\text{Eq.5}]$$

The propagation becomes:

$$H^{(j)} = \sum_{i < j} \sum_{r \in S^{(i,j)}} p_r^{(i,j)} \cdot f\!\left(\mathcal{A}_r^{(i,j)},\, H^{(i)}\right) \quad [\text{Eq.6}]$$

**Why it works:** The gradient for $\alpha_r^{(i,j)}$ in one step is computed from a forward pass that may exclude path r — breaking the circular dependency. The gradient signal for each α_r is de-correlated from the W state that uses r's contribution.

#### C. Meta Multigraph Derivation

After training, a threshold-based selection extracts the final architecture. The threshold interpolates between max and min path strengths:

$$\tau^{(i,j)} = \lambda \cdot \max_r\!\left\{p_r^{(i,j)}\right\} + (1-\lambda) \cdot \min_r\!\left\{p_r^{(i,j)}\right\} \quad [\text{Eq.7}]$$

The final meta multigraph retains only paths above threshold:

$$\hat{S}^{(i,j)} = \left\{r \mid p_r^{(i,j)} \geq \tau^{(i,j)}\right\}$$

$$H^{(j)} = \sum_{i < j} \sum_{r \in \hat{S}^{(i,j)}} f\!\left(\mathcal{A}_r^{(i,j)},\, H^{(i)}\right) \quad [\text{Eq.8}]$$

### 19.3 Contributions

- First formalization of the **meta multigraph** (directed acyclic multigraph on schema) as the HetGNN architecture search space — strictly more expressive than meta-graph and meta-path
- **Partial message propagation (PMP)** — principled random subsampling that decouples bilevel optimization and eliminates the seed-dependent instability of DiffMG
- Competitive accuracy with search cost of only 0.9–2.1 GPU-minutes
- Identifies and diagnoses DiffMG's instability as a coupled-optimization artifact

### 19.4 Results

| Dataset | Macro-F1 | Micro-F1 | Comparison |
|---------|----------|----------|------------|
| DBLP | 94.69% | 95.40% | Stable across 30 seeds (DiffMG: only ~7/30 seeds work) |
| ACM | 92.76% | 92.65% | +0.7% Macro-F1 over DiffMG |
| IMDB | 61.69% | 64.31% | +4.5% Micro-F1 over DiffMG |

| Dataset | Metric | PMMM |
|---------|--------|------|
| Amazon | AUC | 75.48% |

**Search cost:** 0.9–2.1 GPU-minutes. **Key result:** Consistency across 30 random seeds vs. DiffMG's seed-fragility is the central empirical contribution — demonstrates that PMP solves a structural training artifact, not just a marginal accuracy improvement.

### 19.5 Strengths

- **Stability:** 30-seed reproducibility is a rigorous and practically meaningful benchmark; this is the paper's most important result
- **Elegance:** PMP is a minimal intervention — just a Bernoulli gate on each path per iteration — yet it resolves a fundamental instability
- **Lowest search cost:** 0.9–2.1 GPU-minutes is the cheapest among all 19 papers reviewed
- **Expressiveness hierarchy:** Cleanly establishes meta multigraph > meta-graph > meta-path and demonstrates that the additional expressiveness yields measurable gains on IMDB and ACM
- **Directly adaptable:** The PMP technique is architecture-agnostic — applicable to any supernet with multiple candidate operations per type pair (including H2S-NN's schema pool, see §16.8)

### 19.6 Weaknesses

- **Still 1-skeleton only:** Meta multigraph operates entirely on schema-level message passing between hyper-node types. No 2-cells, no coboundary δ₁, no Hodge Laplacian L₁. The "multi-edge" expressiveness is strictly weaker than H2S-NN's face-based 2-cell topology — a multi-edge ℛ^{(i,j)} cannot encode the position-aware sequential ordering of specific node instances within a path (which H2S-NN's LSTM face restriction maps do capture).
- **No geometric grounding:** Path strength p_r^{(i,j)} is a learned scalar per relation type pair — it encodes no geometric information about the learned representation. Unlike holonomy (which measures how "twisted" the sheaf is along each path), α_r has no algebraic interpretation in terms of the graph's structure.
- **Node classification only:** All experiments use DBLP, ACM, IMDB node classification. No heterophilic benchmarks, no graph classification, no link prediction.
- **λ is manual:** The derivation threshold parameter λ ∈ [0,1] is set to 0.5 without principled justification. Sensitivity to λ is not ablated — the appendix only studies p.
- **Schema-level granularity:** PMMM optimizes which relation types to use between schema node types. Within each relation, all specific node instances are treated identically. Instance-level importance (which specific source-target node pairs carry the most signal) is not modeled — this is exactly what H2S-NN's per-edge restriction maps address.

### 19.7 Position in the Expressiveness Hierarchy

```
Level of abstraction    Method                  What is learned
─────────────────────────────────────────────────────────────────────
Schema (type-level)     HAN, MAGNN              Single meta-path per type pair
Schema (type-level)     DiffMG (meta-graph)     One message type per type pair (DAG)
Schema (type-level)     PMMM (meta multigraph)  Multiple message types per type pair ←
─────────────────────────────────────────────────────────────────────
Instance (node-level)   H2S-NN (2-cell complex) Each specific meta-path instance reified
                                                as a topological face; LSTM restriction
                                                maps encode position-aware node ordering;
                                                L₁ Hodge diffusion on edge cochains    ←
```

**H2S-NN is strictly more expressive than PMMM.** PMMM's α_r^{(i,j)} is one scalar per schema-level relation type pair — it cannot distinguish between different instances of the same relation type. H2S-NN's restriction maps F_{e▷f} are d×d matrices learned per face instance, capturing the specific geometric relationship between each meta-path instance and the graph topology.

### 19.8 Key Insight for H2S-NN (see §16.8 for full integration)

> **Professor's note:** PMMM's partial message propagation is the most practically important finding in this survey for H2S-NN implementation. The Gumbel-Sheaf bilevel optimization (α_σ on validation, sheaf maps on training) has exactly the same coupled-optimization problem that made DiffMG fail across seeds. The PSP (Partial Schema Propagation) technique in §16.8 is a direct transplant of PMMM's Eq.5–6 into the sheaf coboundary setting. The analogy is exact: PMMM's "path r between type pair (i,j)" ↔ H2S-NN's "schema σ in S_pool"; PMMM's α_r^{(i,j)} ↔ H2S-NN's θ_σ; PMMM's 𝒜_r ↔ H2S-NN's δ₁^σ. This should be cited as: "We adapt the partial message propagation stabilization technique of [PMMM] to the cellular sheaf coboundary setting."

---

## Summary Table: All 19 Papers

| # | Paper | Core Method | Best Result | Unique Strength | Fatal Weakness |
|---|-------|-------------|-------------|-----------------|----------------|
| 1 | MCTS-HGNN | Monte Carlo Tree Search | DBLP 95.66% | Exploration-exploitation | Scales poorly; MCTS deprecated |
| 2 | AMP-HGNAS | Dependency graph + DARTS | DBLP 95.83% | Inter-path redundancy | Threshold δ manual |
| 3 | RMS-HRec | Deep Q-Network | TCL NDCG 0.3230 | Model-independent, 100x speedup | DQN instability |
| 4 | SMCD | Diffusion augmentation | DBLP 97.18% | Noise robustness | Meta-paths hardcoded |
| 5 | HG-Search | Policy gradient NAS | DBLP 94.84% | Architecture + hyperparameter | NAS cost unreported |
| 6 | SchemaWalk | RL on schema graph | DB111K Hits@1 0.851 | Truly inductive, 174 types | KG only; no topology |
| 7 | HISE-KT | LLM scoring | Statics AUC 0.9073 | Explainability | LLM dependency |
| 8 | LMSPS | Gumbel-softmax progressive | DBLP 95.66% | Long-range + scalable | Bilevel instability |
| 9 | ESim | NCE probabilistic embedding | DBLP AUC 0.89 | Probabilistic framework | Static, small-scale |
| 10 | MP-GNN | Incremental scoring | FB15K F1 0.83 | Many-relation scalable | Greedy; feature-dependent |
| 11 | LPSMS | Marginal contribution (LLO) | DBLP 95.78% | Conceptual reframing + speed | Weak interaction assumption |
| 12 | LevinTS/LubyTS | Policy tree search | 1000/1000 Sokoban | **Formal expansion bounds** | No heuristic integration |
| 13 | LHBL | Limited-horizon Bellman | 100% Rubik's Cube | Multi-step heuristic learning | SSSP overhead |
| 14 | PHS | Policy + heuristic combined | 1000/1000 all domains | **Near-optimal guarantee** | Bootstrapping complexity |
| 15 | DeepSN | Sheaf reaction-diffusion IM | Digg −5.3% MSE | First sheaf IM | 1-skeleton only; no 2-cells |
| 16 | SHN | Sheaf hypernetworks FL | CIFAR100 +2.7% | Orthogonal maps for PFL | 1-skeleton only; no L₁ |
| 17 | NAS-RL | RNN controller + REINFORCE | CIFAR-10 3.65% error | Founded the field | 800 GPUs; no guarantees |
| 18 | NAS Insights | 1000-paper survey | — | Strategy comparison | No new method |
| 19 | PMMM | Meta multigraph + Partial Message Prop. | DBLP 95.40% Micro-F1 | **PMP stabilization; 30-seed reproducibility** | 1-skeleton only; λ manual |

---

*Document prepared: May 2026 (v3) — extended with 7 papers from `research_paper_reinforce/` and `research_paper02/` (v2), plus PMMM (§19) from `research_paper/` (v3). §16.8 adds PMMM-inspired PSP stabilization for Gumbel-Sheaf bilevel optimization.*

*Reading order for H2S-NN extension:*
*1. §14 (PHS) — understand the guarantee framework*
*2. §11 (LPSMS) — understand the gradient-based selection baseline*
*3. §15 (DeepSN) + §16 (SHN) — confirm that 2-cells/L₁ are genuinely unique to H2S-NN*
*4. §18 (NAS Insights) — understand why RL search is the wrong choice*
*5. §19 (PMMM) — understand partial message propagation and why bilevel stability matters*
*6. §16 (Research Direction) — the complete PHS-Sheaf proposal, including §16.8 PSP stabilization*
