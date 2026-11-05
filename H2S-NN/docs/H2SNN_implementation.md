# H²S-NN: Full Implementation Specification

**Heterogeneous and Heterophily-aware Sheaf Neural Network**  
Step-by-Step Algorithm from Raw Input to Node Classification

*Peeradon Sarnkaew — Department of Mathematics and Computer Science, Chulalongkorn University*

---

## Table of Contents

1. [Overview and Notation](#1-overview-and-notation)
2. [Phase 0: Preprocessing (Run Once, Frozen)](#2-phase-0-preprocessing-run-once-frozen)
   - [Step 0.1 — Soft Label Propagation](#step-01--soft-label-propagation)
   - [Step 0.2 — Typed Structural Descriptors](#step-02--typed-structural-descriptors)
   - [Step 0.3 — Feature Residual (Label-Orthogonal Component)](#step-03--feature-residual-label-orthogonal-component)
   - [Step 0.4 — Heterophily Metrics Computation](#step-04--heterophily-metrics-computation)
   - [Step 0.5 — Meta-Path Instance Enumeration and Closure Edges](#step-05--meta-path-instance-enumeration-and-closure-edges)
3. [Phase 1: Node and Edge Initialisation](#3-phase-1-node-and-edge-initialisation)
4. [Phase 2: Per-Iteration Forward Pass](#4-phase-2-per-iteration-forward-pass)
   - [Step A — Node-to-Edge Restriction Maps](#step-a--node-to-edge-restriction-maps-directed-typed-sheaf)
   - [Step B — Edge-to-Face Restriction Maps](#step-b--edge-to-face-restriction-maps-lstm-sheaf-learner)
   - [Step C — Assemble Hodge Laplacians](#step-c--assemble-hodge-laplacians)
   - [Step D — Edge Cochain Initialisation](#step-d--edge-cochain-initialisation)
   - [Step E — Coupled Hodge Diffusion via Neural ODE](#step-e--coupled-hodge-diffusion-via-neural-ode)
   - [Step F — Classification and Loss](#step-f--classification-and-loss)
5. [Full Training Algorithm (Pseudocode)](#5-full-training-algorithm-pseudocode)
6. [Complexity Analysis](#6-complexity-analysis)
7. [Key Theoretical Guarantees](#7-key-theoretical-guarantees)
8. [Comparison with Prior Methods](#8-comparison-with-prior-methods)

---

## 1. Overview and Notation

### Problem Statement

Given a **Heterogeneous Information Network (HIN)**:

$$\mathcal{G} = (V,\; E,\; \phi,\; \psi,\; \mathbf{X})$$

| Symbol | Meaning |
|--------|---------|
| $V$ | Node set; $\phi: V \to \mathcal{T}_V$ maps each node to its type |
| $E$ | Directed edge set; $\psi: E \to \mathcal{R}$ maps each edge to a relation type |
| $\mathbf{X} = \{x_u \in \mathbb{R}^{d_{\phi(u)}}\}$ | Raw node features (dimension is type-specific) |
| $V^{\mathrm{lab}}_t \subseteq V_{\tau_t}$ | Labeled target nodes with labels $Y_u \in \{1,\ldots,C\}$ |

**Goal:** Predict labels for all unlabeled target nodes.

---

### Key Hyperparameters

| Symbol | Meaning | Typical value |
|--------|---------|---------------|
| $d_0$ | Node stalk dimension | 64–256 |
| $d_1$ | Edge stalk dimension | 64–256 |
| $d_2$ | Face stalk dimension | 32–64 |
| $d_h$ | LSTM hidden dimension | 128 |
| $T$ | ODE integration time | 1.0 |
| $\lambda_{\mathrm{close}}$ | Closure-edge regularisation | $10^{-6}$ |
| $\mathcal{P}$ | Set of meta-path types | dataset-specific |
| $\ell_{\max}$ | Max meta-path length | 2–3 |

---

### Learnable Parameter Set

$$\Theta = \left\{\;
  \underbrace{\mathrm{MLP}^{(\mathrm{src})}_r,\; \mathrm{MLP}^{(\mathrm{tgt})}_r}_{\text{node} \to \text{edge}},\quad
  \underbrace{\mathrm{LSTM},\; \mathbf{a},\; \{P_f\}}_{\text{edge} \to \text{face}},\quad
  \underbrace{\mathrm{MLP}_{\mathrm{init}}}_{\text{edge init}},\quad
  \underbrace{\mathrm{MLP}_{\mathrm{cls}}}_{\text{classifier}}
\;\right\}$$

---

## 2. Phase 0: Preprocessing (Run Once, Frozen)

> Everything in this phase is computed on the **base graph** $\mathcal{G}_0$ and **frozen** before any gradient update. No parameters are learned here.

---

### Step 0.1 — Soft Label Propagation

**Purpose.** Assign a soft class-probability vector $q_u \in \Delta^{C-1}$ to every node $u \in V$, including non-target types. This allows heterophily metrics and feature decomposition to use label information globally across all node types.

**Algorithm.**

1. For labeled target nodes $u \in V^{\mathrm{lab}}_t$: set $q_u = \mathbf{e}_{Y_u}$ (one-hot vector).
2. For all other nodes: initialise $q_u = \mathbf{0}$.
3. Iterate **3 hops** via personalised label diffusion:

$$q_u \;\leftarrow\; \frac{\displaystyle\sum_{r \in \mathcal{R}}\;\sum_{v:(v,u)\in E_r} q_v}{\displaystyle\sum_{r \in \mathcal{R}}\;\sum_{v:(v,u)\in E_r} 1 \;+\; \epsilon}, \qquad \epsilon = 10^{-8}$$

4. Re-normalise: $q_u \leftarrow q_u / \max(1, \|q_u\|_1)$.
5. For labeled target nodes, restore $q_u = \mathbf{e}_{Y_u}$ (override after each hop).
6. **Freeze** $\{q_u\}_{u \in V}$.

> **Implementation note.** Store as a dense matrix $Q \in \mathbb{R}^{|V| \times C}$. Propagation via sparse matrix multiplication: `Q = (A_norm @ Q).clamp(min=0)` repeated 3 times, then override labeled rows. `A_norm` is the row-normalised heterogeneous adjacency (union of all relation types).

**Output:** $\{q_u\}_{u \in V}$, a soft label fingerprint for every node.

---

### Step 0.2 — Typed Structural Descriptors

**Purpose.** Produce a **parameter-free** structural fingerprint for each node under each relation type, capturing typed topology without any learning. Used by the structural heterophily metric $H^{(S)}$.

**For each relation $r \in \mathcal{R}$ and node $u \in V$:**

1. Compute **typed degree**: $\deg_r(u) = |\{v : (u,v) \in E_r\}|$.
2. Compute **typed 2-hop count**: $c^{(2)}_r(u) = |\{w : \exists\, v,\; (u,v)\in E_r \text{ and } (v,w)\in E_r\}|$.
3. Compute **meta-path count profile**: for each meta-path type $M \in \mathcal{P}$ starting from node type $\phi(u)$, count the number of meta-path instances passing through $u$.
4. Concatenate into raw descriptor:

$$\Xi^{(r)}_u = \bigl[\deg_r(u),\; c^{(2)}_r(u),\; \text{meta-path counts}\bigr] \in \mathbb{R}^{d_r}_{\ge 0}$$

5. Log-compress and standardise component-wise:

$$\Xi^{(r),\mathrm{norm}}_{u,k} = \frac{\log(1 + \Xi^{(r)}_{u,k}) - \mu^{(r)}_k}{\sigma^{(r)}_k}$$

where $\mu^{(r)}_k$ and $\sigma^{(r)}_k$ are mean and standard deviation over **all nodes**, computed once on $\mathcal{G}_0$. If $\sigma^{(r)}_k = 0$ (constant component), set that component to 0.

6. Map to probability simplex:

$$z^{(S,r)}_u = \mathrm{softmax}\!\left(\Xi^{(r),\mathrm{norm}}_u\right) \in \Delta^{d_r - 1}$$

7. **Freeze** all $\{z^{(S,r)}_u\}$.

> **Why log-standardise?** Typed degrees can be single digits while 2-hop counts reach thousands. Raw softmax over such values degenerates to near-one-hot. Log-compression ensures each dimension contributes meaningfully.

**Output:** $\{z^{(S,r)}_u\}$ for all $u \in V$, $r \in \mathcal{R}$.

---

### Step 0.3 — Feature Residual (Label-Orthogonal Component)

**Purpose.** Remove the label-predictive direction from raw node features. The remaining residual captures variation that is independent of class identity, and is used by the feature heterophily metric $H^{(F)}$.

**For each node type $\tau \in \mathcal{T}_V$:**

1. Within-type standardise features:

$$\tilde{x}_u = \frac{x_u - \mu^{\mathrm{feat}}_\tau}{\sigma^{\mathrm{feat}}_\tau}, \qquad u \in V_\tau$$

where $\mu^{\mathrm{feat}}_\tau, \sigma^{\mathrm{feat}}_\tau$ are per-dimension mean and std over all nodes of type $\tau$. **Freeze these statistics.**

2. Compute **class prototypes** from labeled target nodes only:

$$\mu_c = \frac{1}{|V_c|}\sum_{u:\, Y_u = c} \tilde{x}_u, \qquad c = 1, \ldots, C$$

**Freeze.**

3. Subtract the label-semantic component:

$$\rho_u = \tilde{x}_u - \sum_{c=1}^{C} q_{u,c}\, \mu_c$$

4. Map to simplex: $z^{(F,r)}_u = \mathrm{softmax}(\rho_u)$. **Freeze.**

**Output:** $\{z^{(F,r)}_u\}$ — label-orthogonal feature atoms for every node.

---

### Step 0.4 — Heterophily Metrics Computation

**Purpose.** Produce scalar diagnostic scores per relation $H^{(a)}_r \in [-1, 1]$ and informativeness score $\mathrm{CTLI}_r \in [0,1]$. These are (i) used as dataset diagnostics, and (ii) **appended to node features** before training.

**For each relation $r \in \mathcal{R}$ and aspect $a \in \{L, S, F\}$:**

**Step 1 — Observed agreement:**

$$A^{(a)}_r = \frac{1}{|E_r|}\sum_{(u,v)\in E_r} \langle z^{(a,r)}_u,\; z^{(a,r)}_v \rangle \in [0, 1]$$

**Step 2 — Degree-weighted baseline marginals** (typed bipartite configuration model):

$$\pi^{(a,r)}_s = \frac{\sum_{u \in V_{\tau_s(r)}} d_r(u)\; z^{(a,r)}_u}{\sum_{u \in V_{\tau_s(r)}} d_r(u)}, \qquad \pi^{(a,r)}_t = \frac{\sum_{v \in V_{\tau_t(r)}} d_r(v)\; z^{(a,r)}_v}{\sum_{v \in V_{\tau_t(r)}} d_r(v)}$$

$$b^{(a)}_r = \langle \pi^{(a,r)}_s,\; \pi^{(a,r)}_t \rangle$$

Apply soft clamp: $b^{(a)}_r \leftarrow \max(\delta, \min(1-\delta,\; b^{(a)}_r))$ with $\delta = 10^{-3}$.

**Step 3 — Piecewise signed normalisation $\Psi$:**

$$H^{(a)}_r = \Psi(A^{(a)}_r;\; b^{(a)}_r) = \begin{cases} \dfrac{A^{(a)}_r - b^{(a)}_r}{1 - b^{(a)}_r} & A^{(a)}_r \ge b^{(a)}_r \\[8pt] \dfrac{A^{(a)}_r - b^{(a)}_r}{b^{(a)}_r} & A^{(a)}_r < b^{(a)}_r \end{cases}$$

This guarantees $H^{(a)}_r \in [-1, 1]$ with $H^{(a)}_r = 0$ when agreement equals chance.

**Step 4 — Graph-level score:**

$$H^{(a)} = \sum_{r \in \mathcal{R}} \omega_r\, H^{(a)}_r, \qquad \omega_r = \frac{|E_r|}{|E|}$$

**Interpretation of scores:**

| Score | Meaning |
|-------|---------|
| $H^{(a)}_r = +1$ | Relation $r$ is maximally homophilic (aspect $a$) |
| $H^{(a)}_r = 0$ | Relation $r$ exhibits only chance-level agreement |
| $H^{(a)}_r = -1$ | Relation $r$ is maximally heterophilic (aspect $a$) |

---

**CTLI — Cross-Type Label Informativeness:**

Measures whether neighbours under relation $r$ are *useful* for predicting the target label, even if dissimilar:

$$\mathrm{CTLI}_r = \frac{H(q_{\mathrm{target}}) - H(q_{\mathrm{target}} \mid q_{\mathrm{neighbour\; under\;} r})}{H(q_{\mathrm{target}})} \in [0, 1]$$

Computed as normalised mutual information between soft label distributions at both ends of relation-$r$ edges.

> **Key diagnostic pair:** $\mathrm{CTLI}_r > 0$ with $H^{(L)}_r < 0$ means the edge is heterophilic *but still informative* — a signal that standard homophily-based models will fail on this relation, but a well-designed model can still extract useful signal.

---

**Append to features.** For every node $u$, append the metric vector:

$$[H^{(L)}_r,\; H^{(S)}_r,\; H^{(F)}_r,\; \mathrm{CTLI}_r]_{r \in \mathcal{R}} \in \mathbb{R}^{4|\mathcal{R}|}$$

to $x_u$, producing augmented feature $\hat{x}_u$. **Freeze metrics.**

**Output:** $\{\hat{x}_u\}$, augmented features; scalar metrics $\{H^{(a)}_r\}$ stored for diagnostic use.

---

### Step 0.5 — Meta-Path Instance Enumeration and Closure Edges

**Purpose.** Enumerate every concrete meta-path instance in the graph to form the face set $\mathcal{X}^2$, and add a closure edge to each face to satisfy the topological boundary requirement.

**For each meta-path type $M = (\tau_0 \xrightarrow{r_1} \tau_1 \xrightarrow{r_2} \cdots \xrightarrow{r_\ell} \tau_\ell) \in \mathcal{P}$:**

1. For every source node $v_0 \in V_{\tau_0}$, perform DFS/BFS following edge types $r_1, r_2, \ldots, r_\ell$ in order.
2. Each complete path $(v_0 \xrightarrow{e_1} v_1 \xrightarrow{e_2} \cdots \xrightarrow{e_\ell} v_\ell)$ becomes one **2-cell** (face): $f = (e_1, e_2, \ldots, e_\ell)$.
3. For each face $f$, add a **closure edge** $\bar{e}_f = (v_\ell \to v_0)$ with synthetic relation type $r_{\mathrm{close}}$.

---

**Closure Edge Construction (Definition 16).**

A 2-cell in a CW-complex requires a **closed** boundary cycle. A meta-path of length $\ell$ is an open walk $v_0 \to v_1 \to \cdots \to v_\ell$, so we close it with $\bar{e}_f$.

The closure edge restriction maps are set **once and permanently frozen**:

$$\mathcal{F}_{v_0 \triangleright \bar{e}_f} = \mathcal{F}_{v_\ell \triangleright \bar{e}_f} = \lambda_{\mathrm{close}} \cdot \mathbf{I}_{d_1}, \qquad \lambda_{\mathrm{close}} = 10^{-6}$$

$$\mathcal{F}_{\bar{e}_f \triangleright f} = \mathbf{0} \in \mathbb{R}^{d_2 \times d_1}$$

> **Why $\mathcal{F}_{\bar{e}_f \triangleright f} = \mathbf{0}$?**  
> The zero edge-to-face map means the closure edge **contributes nothing to the Hodge diffusion** — it carries no semantic content. It exists solely to close the boundary cycle, which is required for $\delta_1 \circ \delta_0 = 0$ to hold identically.  
> The node-to-closure-edge maps are $\lambda \cdot I$ (not zero) to ensure the edge stalk $\mathcal{F}(\bar{e}_f) = \mathbb{R}^{d_1}$ is a proper non-zero space — required for the cochain complex to be well-defined. Since $\lambda = 10^{-6}$, the perturbation to $L_0$ is $O(\lambda^2)$, negligible.

---

**Output of Phase 0:**

- Frozen soft labels $\{q_u\}_{u \in V}$
- Frozen augmented features $\{\hat{x}_u\}_{u \in V}$
- Frozen heterophily scores $\{H^{(a)}_r\}$
- Face set $\mathcal{X}^2 = \{f_1, f_2, \ldots\}$
- Closure edge set $\bar{E}$
- Full 1-cell set $\mathcal{X}^1 = E \cup \bar{E}$

---

## 3. Phase 1: Node and Edge Initialisation

### Step 1.1 — Node Cochain Initialisation

Each node $u$ needs an initial stalk vector $x^{(0)}_u \in \mathbb{R}^{d_0}$. Map the augmented feature $\hat{x}_u$ to the stalk dimension using a **type-specific linear projection**:

$$x^{(0)}_u = \mathrm{LayerNorm}\!\left(W_{\phi(u)}\, \hat{x}_u + b_{\phi(u)}\right), \qquad W_{\phi(u)} \in \mathbb{R}^{d_0 \times (d_{\phi(u)} + 4|\mathcal{R}|)}$$

where $4|\mathcal{R}|$ is the number of appended metric dimensions from Step 0.4.  
Each node type $\tau$ has its own $(W_\tau, b_\tau)$ — these are **learnable** and part of $\Theta$.

### Step 1.2 — Relation-Type Edge Embeddings

A learnable embedding table assigns a vector $x^{(\mathrm{rel})}_r \in \mathbb{R}^{d_r}$ to each relation type $r \in \mathcal{R}$. All edges of type $r$ share the same relation embedding. This is a standard embedding lookup table in $\Theta$.

---

## 4. Phase 2: Per-Iteration Forward Pass

> The following six steps are executed **every training iteration**. All operations are differentiable; gradients flow back through the ODE solve via the adjoint method and through the SVD via implicit differentiation.

---

### Step A — Node-to-Edge Restriction Maps (Directed Typed Sheaf)

**Purpose.** Compute two restriction maps per directed edge — one for the source node and one for the target node — capturing the **asymmetric semantic roles** of the two endpoints under each relation type.

For each real directed edge $e = (u \xrightarrow{r} v) \in E$:

$$\mathcal{F}^{\mathrm{src}}_{u \triangleright e} = \mathrm{MLP}^{(\mathrm{src})}_r\!\left(x_u \;\|\; x_v \;\|\; x^{(\mathrm{rel})}_r\right) \in \mathbb{R}^{d_1 \times d_0}$$

$$\mathcal{F}^{\mathrm{tgt}}_{v \triangleright e} = \mathrm{MLP}^{(\mathrm{tgt})}_r\!\left(x_u \;\|\; x_v \;\|\; x^{(\mathrm{rel})}_r\right) \in \mathbb{R}^{d_1 \times d_0}$$

Each MLP is a **2-layer network** with hidden dimension $d_h$ and ReLU activation. The output is a flattened $d_1 \times d_0$ matrix, reshaped after the final linear layer. There are $2|\mathcal{R}|$ such MLPs in total.

> **Why two maps per edge?**  
> In a standard (undirected) sheaf, each edge–node incidence has one restriction map. Here, the source type $\tau_s(r)$ and target type $\tau_t(r)$ are semantically different (e.g., Author → Paper). A single shared map would collapse this asymmetry. Two separate maps per incidence is a directed cellular sheaf, following the directed sheaf construction adapted to the heterogeneous setting.

**Coboundary operator $\delta_0$** (encodes "disagreement" in the edge stalk):

$$(\delta_0 x)_e = \mathcal{F}^{\mathrm{tgt}}_{v \triangleright e}\, x_v \;-\; \mathcal{F}^{\mathrm{src}}_{u \triangleright e}\, x_u \;\in\; \mathbb{R}^{d_1}$$

---

### Step B — Edge-to-Face Restriction Maps (LSTM Sheaf Learner)

**Purpose.** Produce **position-sensitive** restriction maps from each edge stalk to the face stalk, encoding the sequential order of edges within a meta-path.

For each face $f = (e_1, e_2, \ldots, e_\ell) \in \mathcal{X}^2$:

**Step B.1 — Edge input encodings.**  
For each edge $e_i = (u_i \xrightarrow{r_i} v_i)$ in the face:

$$g(e_i) = \mathrm{MLP}_g\!\left(x_{u_i} \;\|\; x_{v_i} \;\|\; x^{(\mathrm{rel})}_{r_i}\right) \in \mathbb{R}^{d_1}$$

**Step B.2 — LSTM sequential encoding.**  
Run the LSTM across the ordered sequence $(g(e_1), \ldots, g(e_\ell))$:

$$(h_i, c_i) = \mathrm{LSTM}\!\left(g(e_i),\; h_{i-1},\; c_{i-1}\right), \qquad h_i \in \mathbb{R}^{d_h}$$

with $h_0 = c_0 = \mathbf{0}$. The hidden state $h_i$ encodes **sequential position**: the "writes" restriction $\mathcal{F}_{e_1 \triangleright f}$ differs from the "pub_at" restriction $\mathcal{F}_{e_2 \triangleright f}$ (since $h_1 \ne h_2$), capturing ordered relational semantics.

**Step B.3 — Position-sensitive attention weights.**

$$\alpha_i = \frac{h_i^\top\, \mathbf{a}}{\sqrt{d_h}} \in \mathbb{R}$$

where $\mathbf{a} \in \mathbb{R}^{d_h}$ is a learnable query vector (part of $\Theta$).

**Step B.4 — Assemble $M_f$ and enforce $\delta_1 \circ \delta_0 = 0$ via SVD.**

Construct:

$$M_f = \sum_{i=1}^{\ell} \alpha_i \cdot \left(\mathcal{F}^{\mathrm{tgt}}_{v_i \triangleright e_i}\right)^\top \in \mathbb{R}^{d_0 \times d_1}$$

Compute the thin SVD:

$$M_f = U_f \Sigma_f V_f^\top, \qquad U_f \in \mathbb{R}^{d_0 \times r_f}$$

Extract the **orthogonal complement** $U_{f,\perp} \in \mathbb{R}^{(d_0 - r_f) \times d_0}$ (rows orthogonal to the column space of $M_f$).

Set the shared weight matrix:

$$W_f = U_{f,\perp}^\top P_f \in \mathbb{R}^{d_0 \times d_2}$$

where $P_f \in \mathbb{R}^{(d_0 - r_f) \times d_2}$ is a **learnable matrix** (part of $\Theta$). The SVD is **re-computed every forward pass**; gradients flow through $\alpha_i$ (and hence the LSTM) via implicit differentiation.

**Step B.5 — Edge-to-face restriction maps.**

$$\mathcal{F}_{e_i \triangleright f} = \alpha_i\, W_f^\top \in \mathbb{R}^{d_2 \times d_1}, \qquad i = 1, \ldots, \ell$$

$$\mathcal{F}_{\bar{e}_f \triangleright f} = \mathbf{0} \qquad \text{(closure edge, permanently frozen)}$$

> **Why does the SVD ensure $\delta_1 \circ \delta_0 = 0$?**  
> The condition $(\delta_1 \delta_0 x)_f = 0$ expands to:  
> $$\sum_{i=1}^\ell \epsilon(e_i, f)\, \mathcal{F}_{e_i \triangleright f}\, (\delta_0 x)_{e_i} = 0$$  
> Substituting $(\delta_0 x)_{e_i} = \mathcal{F}^{\mathrm{tgt}}_{v_i \triangleright e_i} x_{v_i} - \mathcal{F}^{\mathrm{src}}_{u_i \triangleright e_i} x_{u_i}$ and using $W_f = U_{f,\perp}^\top P_f$ with $U_{f,\perp} \perp \mathrm{col}(M_f)$, the telescoping sum vanishes identically. This is a **hard constraint**, not a soft penalty — it holds at every training step.

**Coboundary operator $\delta_1$:**

$$(\delta_1 s)_f = \sum_{e_i \prec f} \epsilon(e_i, f)\; \mathcal{F}_{e_i \triangleright f}\; s_{e_i} \;\in\; \mathbb{R}^{d_2}$$

where $\epsilon(e_i, f) \in \{+1, -1\}$ is the face orientation sign.

---

### Step C — Assemble Hodge Laplacians

**Hodge-0 Laplacian** (node Laplacian):

$$L_0 = (\delta_0)^\top \delta_0 \in \mathbb{R}^{|V|d_0 \times |V|d_0}$$

Block structure:

$$L_0[v,v] = \sum_{e \ni v} \mathcal{F}_{v \triangleright e}^\top \mathcal{F}_{v \triangleright e}$$

$$L_0[u,v] = -\left(\mathcal{F}^{\mathrm{src}}_{u \triangleright e}\right)^\top \mathcal{F}^{\mathrm{tgt}}_{v \triangleright e} \quad \text{for edge } e = (u \to v), \qquad L_0[u,v] = L_0[v,u]^\top$$

$L_0$ is symmetric positive semi-definite. $\ker(L_0) \cong H^0$ (harmonic node cochains).

Harmonic condition: $L_0 x = 0$ requires $\mathcal{F}^{\mathrm{tgt}}_{v \triangleright e} x_v = \mathcal{F}^{\mathrm{src}}_{u \triangleright e} x_u$ for all edges — nodes may **differ by a learned typed linear transformation** while achieving zero energy. This is the key mechanism preventing over-smoothing.

---

**Hodge-1 Laplacian** (edge Laplacian):

$$L_1 = \underbrace{\delta_0 (\delta_0)^\top}_{L_1^\downarrow \;\text{(down)}} + \underbrace{(\delta_1)^\top \delta_1}_{L_1^\uparrow \;\text{(up)}} \in \mathbb{R}^{|E|d_1 \times |E|d_1}$$

- **$L_1^\downarrow$** couples edges that **share a common node** (down-Laplacian).
- **$L_1^\uparrow$** couples edges that **share a common face** — i.e., edges within the same meta-path instance.

$\ker(L_1) \cong H^1$ (harmonic edge cochains encoding irreducible relational structure).

> **Semantic interpretation of $L_1^\uparrow$:**  
> $L_1^\uparrow[e_i, e_j] = \sum_{f \ni e_i, e_j} \epsilon(e_i,f)\,\epsilon(e_j,f)\, \mathcal{F}_{e_i \triangleright f}^\top \mathcal{F}_{e_j \triangleright f}$  
> For $e_1, e_2$ within the same face $f$ (e.g., Author→Paper→Venue), this adds coupling $\alpha_1\alpha_2 W_f^\top W_f$ that **softens inter-edge repulsion**, encoding the semantic coherence of the complete meta-path as a single unit. This term is absent in all prior sheaf neural networks, which only operate on 1-complexes.

---

### Step D — Edge Cochain Initialisation

> **Critical fix — do NOT set $s^{(0)} = \delta_0 x^{(0)}$.**

**Why this is fatal:** By the Hodge decomposition, $C^1 = \mathrm{im}(\delta_0) \oplus \ker(L_1) \oplus \mathrm{im}(\delta_1^\top)$. If $s^{(0)} = \delta_0 x^{(0)} \in \mathrm{im}(\delta_0)$, then its harmonic component $[s^{(0)}]_{\ker(L_1)} = 0$. Since the ODE $\dot{s} = -L_1 s$ **preserves** the harmonic component at its initial value, we get $s^* = 0$ at convergence — no $H^1$ information is ever captured.

**Correct initialisation (Definition 22):**

$$s^{(0)}_e = \mathrm{MLP}_{\mathrm{init}}\!\left(x_u \;\|\; x_v \;\|\; x^{(\mathrm{rel})}_r \;\|\; (\delta_0 x^{(0)})_e\right) \in \mathbb{R}^{d_1}, \qquad \forall\, e = (u \to v) \in E$$

The term $(\delta_0 x^{(0)})_e$ is included as an **informative input feature** (the initial node disagreement), but the MLP output can have components in all three Hodge subspaces — including the harmonic subspace $\ker(L_1)$.

---

### Step E — Coupled Hodge Diffusion via Neural ODE

**Purpose.** Jointly minimise the three-term Dirichlet energy over node and edge cochains:

$$\mathcal{E}(x, s) = \underbrace{\|\delta_0 x\|^2}_{\mathcal{E}_0:\;\text{node disagreement}} + \underbrace{\|(\delta_0)^\top s\|^2}_{\mathcal{E}_1^\downarrow:\;\text{edge-flow consistency}} + \underbrace{\|\delta_1 s\|^2}_{\mathcal{E}_1^\uparrow:\;\text{face consistency}} = x^\top L_0 x + s^\top L_1 s$$

**Decoupled Hodge ODE system:**

$$\frac{dx}{dt} = -L_0\, x(t), \qquad \frac{ds}{dt} = -L_1\, s(t)$$

with initial conditions $x(0) = x^{(0)}$, $s(0) = s^{(0)}$ from Steps 1.1 and D.

**Numerical integration settings:**

| Setting | Value |
|---------|-------|
| Library | `torchdiffeq.odeint` |
| Solver | `dopri5` (adaptive RK4/5) or `rk4` (fixed step) |
| Integration interval | $t \in [0, T]$, $T = 1.0$ |
| Backpropagation | `adjoint=True` (memory-efficient adjoint method) |

**At convergence (time $T$):**

$$x^* = x(T) \approx \mathrm{proj}_{\ker(L_0)} x^{(0)} \in H^0, \qquad s^* = s(T) \approx \mathrm{proj}_{\ker(L_1)} s^{(0)} \in H^1$$

> **Why does this prevent over-smoothing? (Proposition 8)**  
> For the trivial sheaf ($\mathcal{F}_{v \triangleright e} = I$): $\mathcal{E}_0(x) = 0$ requires $x_v = x_u$ for all edges — classical over-smoothing.  
> For a non-trivial sheaf: $\mathcal{E}_0(x) = 0$ requires only $\mathcal{F}^{\mathrm{tgt}}_{v \triangleright e} x_v = \mathcal{F}^{\mathrm{src}}_{u \triangleright e} x_u$ — nodes may differ by a **learned typed linear transformation** while being in sheaf-agreement.  
> The harmonic space $H^0 = \ker(L_0)$ contains linearly separable representations for $d_0 \ge \lceil C/2 \rceil$ with orthogonal restriction maps. Diffusion converges to a discriminative fixed point, not a constant vector.

---

### Step F — Classification and Loss

$$\hat{Y}_u = \mathrm{MLP}_{\mathrm{cls}}(x^*_u), \qquad \forall\, u \in V^{\mathrm{lab}}_t$$

$$\mathcal{L} = \frac{1}{|V^{\mathrm{lab}}_t|} \sum_{u \in V^{\mathrm{lab}}_t} \mathrm{CrossEntropy}(Y_u,\; \hat{Y}_u)$$

Backpropagate through:
- The ODE solve via the **adjoint method** (gradient of $\mathcal{L}$ w.r.t. $L_0, L_1$, and hence w.r.t. all restriction maps and through them to $\Theta$).
- The SVD via **implicit differentiation** (gradient w.r.t. $\alpha_i$ and the LSTM parameters).

Update $\Theta$ via **Adam**.

---

## 5. Full Training Algorithm (Pseudocode)

```
ALGORITHM: H²S-NN Training

INPUT:
  G         — HIN (V, E, φ, ψ, X)
  P         — set of meta-path types
  V_lab     — labeled target nodes with labels Y
  T         — ODE integration time (default 1.0)
  λ_close   — closure edge regularisation (default 1e-6)

OUTPUT:
  Θ         — trained parameters

════════════════════════════════════════════════
PHASE 0: PREPROCESSING  (run once, then freeze)
════════════════════════════════════════════════

STEP 0.1  Soft label propagation
  for u in V_lab:       q_u = one_hot(Y_u)
  for u not in V_lab:   q_u = 0
  repeat 3 times:
    for u in V:
      q_u = mean of q_v for all v that point to u across all relations
      q_u = q_u / max(1, ||q_u||_1)
    for u in V_lab:     q_u = one_hot(Y_u)   ← restore
  freeze {q_u}

STEP 0.2  Structural descriptors z^(S,r)_u
  for each r in R, u in V:
    Xi_u^r = [deg_r(u), 2-hop_count_r(u), meta-path counts through u]
    Xi_norm = (log(1 + Xi) - mean) / std      ← component-wise, stats from G0
    z^(S,r)_u = softmax(Xi_norm)
  freeze all z^(S,r)

STEP 0.3  Feature residuals z^(F,r)_u
  for each type τ:
    standardise x_u for u in V_τ  →  x̃_u = (x_u - μ_τ) / σ_τ
    freeze (μ_τ, σ_τ)
  for each class c:
    μ_c = mean of x̃_u for labeled u with Y_u = c    ← freeze
  for each u in V:
    ρ_u = x̃_u - Σ_c q_{u,c} * μ_c
    z^(F,r)_u = softmax(ρ_u)
  freeze all z^(F,r)

STEP 0.4  Heterophily metrics
  for each r in R, a in {L, S, F}:
    A^a_r = mean of ⟨z^(a,r)_u, z^(a,r)_v⟩ over (u,v) in E_r
    π^a_s = degree-weighted mean of z^(a,r)_u over source nodes
    π^a_t = degree-weighted mean of z^(a,r)_v over target nodes
    b^a_r = ⟨π^a_s, π^a_t⟩  (clamped to [δ, 1-δ])
    H^a_r = Ψ(A^a_r; b^a_r)   ← piecewise normalisation
  compute CTLI_r = normalised mutual information of soft labels across E_r
  append [H^L_r, H^S_r, H^F_r, CTLI_r for all r] to each x_u → x̂_u
  freeze metrics

STEP 0.5  Cellular complex construction
  X^2 = {}
  Ē   = {}
  for each meta-path type M in P:
    for each starting node v_0 of the right type:
      enumerate all instances (v_0 -e1→ v_1 -e2→ ... -eℓ→ v_ℓ)
      create face f = (e_1, ..., e_ℓ)
      create closure edge ē_f = (v_ℓ → v_0) with type r_close
      set F_{v0 ▷ ē_f} = F_{vℓ ▷ ē_f} = λ_close * I   ← frozen
      set F_{ē_f ▷ f}  = 0                              ← frozen
      X^2 ← X^2 ∪ {f};   Ē ← Ē ∪ {ē_f}
  X^1 = E ∪ Ē

═════════════════════════════════════
PHASE 1: INITIALISATION
═════════════════════════════════════

  for each node u in V:
    x^(0)_u = LayerNorm( W_{φ(u)} * x̂_u + b_{φ(u)} )
  initialise relation embedding table { x^(rel)_r } for r in R

═════════════════════════════════════
TRAINING LOOP
═════════════════════════════════════

for each training iteration:

  ── STEP A: Node → Edge restriction maps ──
  for each real edge e = (u →_r v) in E:
    F^src_{u▷e} = MLP^src_r( x_u ‖ x_v ‖ x^rel_r )   ∈ R^{d1 × d0}
    F^tgt_{v▷e} = MLP^tgt_r( x_u ‖ x_v ‖ x^rel_r )   ∈ R^{d1 × d0}
  compute (δ0 x)_e = F^tgt_{v▷e} x_v - F^src_{u▷e} x_u  for all e

  ── STEP B: Edge → Face restriction maps (LSTM + SVD) ──
  for each face f = (e_1, ..., e_ℓ) in X^2:
    for i = 1 to ℓ:
      g(e_i) = MLP_g( x_{u_i} ‖ x_{v_i} ‖ x^rel_{r_i} )
    (h_1,...,h_ℓ), (c_1,...,c_ℓ) = LSTM( g(e_1),...,g(e_ℓ) )
    α_i = h_i^T a / sqrt(d_h)   for each i
    M_f = Σ_i α_i * (F^tgt_{v_i▷e_i})^T             ∈ R^{d0 × d1}
    U_f, Σ_f, V_f^T = thin_SVD(M_f)
    U_f_perp = orthogonal complement of col(U_f)      ∈ R^{(d0-r_f) × d0}
    W_f = U_f_perp^T * P_f                            ∈ R^{d0 × d2}
    F_{e_i▷f} = α_i * W_f^T                           ∈ R^{d2 × d1}
    F_{ē_f▷f} = 0   (closure edge, frozen)

  ── STEP C: Assemble Laplacians ──
  L0 = (δ0)^T δ0      ← node Laplacian
  L1 = δ0 (δ0)^T  +  (δ1)^T δ1   ← edge Laplacian (down + up)

  ── STEP D: Edge cochain initialisation ──
  for each real edge e = (u →_r v) in E:
    s^(0)_e = MLP_init( x_u ‖ x_v ‖ x^rel_r ‖ (δ0 x^(0))_e )

  ── STEP E: Coupled Hodge ODE ──
  [x*, s*] = odeint( [-L0 x; -L1 s],
                      init=[x^(0); s^(0)],
                      t_span=[0, T],
                      method='dopri5',
                      adjoint=True )
  # x* ∈ ker(L0) ≅ H^0
  # s* ∈ ker(L1) ≅ H^1

  ── STEP F: Classification and update ──
  Ŷ_u = MLP_cls( x*_u )   for u in V_lab
  L = CrossEntropy(Y, Ŷ)
  L.backward()    ← adjoint through ODE, implicit diff through SVD
  Adam.step()

end for
```

---

## 6. Complexity Analysis

| Phase | Operation | Complexity |
|-------|-----------|------------|
| Phase 0 | Soft label propagation | $O(3\|E\|C)$ |
| Phase 0 | Structural descriptors | $O(\|E\| \cdot \|\mathcal{R}\|)$ |
| Phase 0 | Meta-path enumeration | $O(\|V\| \cdot \prod_i \bar{d}_{r_i})$ (bounded by depth cutoff) |
| Step A (per iter.) | Node-to-edge restriction maps | $O(\|E\| \cdot d_1 d_0)$ |
| Step B (per iter.) | LSTM edge-to-face maps | $O(\|\mathcal{X}^2\| \cdot \ell_{\max} \cdot d_h d_1)$ |
| Step B (per iter.) | Dynamic SVD | $O(\|\mathcal{X}^2\| \cdot d_1^2)$ |
| Step C (per iter.) | Laplacian assembly | $O(\|E\| d_1^2 + \|\mathcal{X}^2\| d_1^2)$ |
| Step E (per iter.) | ODE solve | $O(\mathrm{NFE} \cdot (\|V\|d_0^2 + \|E\|d_1^2))$ |
| Step E (per iter.) | Adjoint backprop | $O(\mathrm{NFE} \cdot \|\Theta\|)$ |

where $\mathrm{NFE}$ = number of ODE function evaluations (typically 6–12 for `dopri5`).

SVD cost per face: $O(d_1^2)$ per forward pass. Since $d_1 \ll |V|$, total overhead $O(|\mathcal{X}^2| \cdot d_1^2)$ is negligible compared to the ODE solve.

---

## 7. Key Theoretical Guarantees

### Proposition 1 — Cochain Validity

> For any $\lambda_{\mathrm{close}} > 0$, the construction satisfies $\delta_1 \circ \delta_0 = 0$ at every training step.

*Sketch:* The closure edge contributes zero to faces ($\mathcal{F}_{\bar{e}_f \triangleright f} = \mathbf{0}$), so we only need the real edges to satisfy the identity. The SVD construction of $W_f = U_{f,\perp}^\top P_f$ ensures the telescoping sum over real edges vanishes identically, regardless of the current values of $\alpha_i$ and $P_f$.

---

### Proposition 2 — Harmonic Floor and Linear Separability

> For restriction maps in $O(d_1)$ and $d_0 \ge \lceil C/2 \rceil$ classes, there exist orthogonal restriction maps such that harmonic vectors in $H^0(\mathcal{X};\mathcal{F}) = \ker(L_0)$ are linearly separable by class label.

**Consequence:** Hodge diffusion converges to linearly separable representations rather than a degenerate constant vector — over-smoothing is provably prevented.

---

### Proposition 3 — Over-Squashing via Forman-Ricci Curvature

> After lifting to $\mathcal{X}$, for any edge $e \in E$:  
> $$\mathrm{Ric}_{\mathcal{X}}(e) = \mathrm{Ric}_{\mathcal{G}}(e) + \sum_{f \ni e} w(f) > \mathrm{Ric}_{\mathcal{G}}(e)$$  
> where $w(f) > 0$ is the face weight.

Negative-curvature bottleneck edges become less negative, alleviating over-squashing **without modifying the original edge set** — a principled, semantically-grounded alternative to graph rewiring.

---

### Proposition 4 — Axiomatic Guarantee for Heterophily Metrics

> Under frozen atoms and non-degenerate baseline ($b^{(a)}_r \in [\delta, 1-\delta]$), each of $H^{(L)}_r$, $H^{(S)}_r$, $H^{(F)}_r$ satisfies all five axiomatic properties:

| Property | Description |
|----------|-------------|
| **Max** | Score = $+1$ iff all edges have perfect atom agreement |
| **Min** | Score = $-1$ iff all edges have zero atom agreement |
| **ACB** | Score concentrates at 0 under the typed null model |
| **ECT** | Adding an empty class leaves the score unchanged |
| **Mon** | Adding a same-class edge increases score; cross-class edge decreases it |

---

## 8. Comparison with Prior Methods

| Feature | NSD | HETSHEAF | DSNN | **H²S-NN** |
|---------|-----|----------|------|------------|
| Axiomatic heterophily metrics | ✗ | ✗ | ✗ | ✓ |
| 2D typed cellular complex from meta-paths | ✗ | ✗ | ✗ | ✓ |
| Directed typed sheaf per relation | ✗ | partial | ✓ | ✓ |
| LSTM edge-to-face maps (ordered) | ✗ | ✗ | ✗ | ✓ |
| Dynamic SVD enforcing $\delta_1 \circ \delta_0 = 0$ | n/a | n/a | n/a | ✓ |
| Three-term energy ($\mathcal{E}_0 + \mathcal{E}_1^\downarrow + \mathcal{E}_1^\uparrow$) | ✗ | ✗ | ✗ | ✓ |
| Over-smoothing guarantee | ✓ | ✗ | ✗ | ✓ |
| Over-squashing via curvature (no rewiring) | ✗ | ✗ | ✗ | ✓ |
| Works natively on heterogeneous graphs | ✗ | ✓ | ✗ | ✓ |

---

*End of document.*
