# H²S-NN: Heterophilic Heterogeneous Sheaf Neural Network
## Corrected Algorithm, Mathematical Formulation, and Implementation Specification

> **Purpose of this document:** Single authoritative reference merging and correcting
> `H2S-NN_implementation_spec.md` and `H2SNN_implementation.md`. All mathematical
> errors are identified, explained, and fixed. All pseudocode includes tensor shapes.
> All design decisions are justified. This file serves as the memory anchor for the
> entire project — recover full context by reading §0–§12.

---

## Table of Contents

- §0 Notation
- §1 Problem Setting
- §2 Cellular Complex Construction
- §3 Directed Typed Cellular Sheaf
- §4 Coboundary Operators (with Bug Fix)
- §5 Hodge Laplacians (with Proofs)
- §6 Restriction Map Learning
- §7 Hodge Diffusion
- §8 Complete Forward Pass Algorithm
- §9 Loss Function
- §10 Theoretical Properties
- §11 Implementation Notes (PyTorch + PyG)
- §12 Module Design
- §13 Corrections Table from Original Specs

---

## §0 Notation

| Symbol | Type / Shape | Meaning |
|--------|-------------|---------|
| G = (V, E, φ, ψ, X) | HIN | Heterogeneous Information Network |
| V | set, \|V\| = N | All nodes |
| E | set, \|E\| = M | All typed directed edges |
| T_V | set | Node type vocabulary |
| T_E = R | set | Edge/relation type vocabulary |
| φ: V → T_V | function | Node type assignment |
| ψ: E → R | function | Edge relation type assignment |
| X = {x_v} | dict | Raw node features, x_v ∈ ℝ^{d_raw(φ(v))} |
| x̂_v | ℝ^{d_in(φ(v))} | Pre-processed node features (after Phase 0) |
| P = {π₁,...,π_P} | set | Meta-path template set |
| π = (T₀ →^{r₁} T₁ → ... →^{r_ℓ} T_ℓ) | sequence | Meta-path template of length ℓ |
| X² | set, \|X²\| = F | Set of all meta-path face instances |
| f = (v₀, e₁, v₁, ..., eₗ, vₗ) | tuple | One meta-path face instance |
| ē_f | edge | Synthetic closure edge for face f: vₗ → v₀ |
| X¹ | set, \|X¹\| = M + F | All edges including closure edges |
| d = d₀ = d₁ = d₂ | int | Uniform stalk dimension (can be varied) |
| d_r | int | Relation embedding dimension |
| d_h | int | LSTM hidden dimension |
| C | int | Number of classes (for target node type) |
| K | int | Number of Euler diffusion steps |
| N_τ | int | Number of nodes of type τ |
| F(v) = ℝ^d | vector space | Node stalk |
| F(e) = ℝ^d | vector space | Edge stalk |
| F(f) = ℝ^d | vector space | Face stalk |
| F^src_{u▷e} | ℝ^{d×d} | Source restriction map, edge e = (u→v) |
| F^tgt_{v▷e} | ℝ^{d×d} | Target restriction map, edge e = (u→v) |
| F_{e▷f} | ℝ^{d×d} | Edge-to-face restriction map |
| δ₀ | ℝ^{Md × Nd} | Coboundary: 0-cochains → 1-cochains |
| δ₁ | ℝ^{Fd × Md} | Coboundary: 1-cochains → 2-cochains |
| L₀ | ℝ^{Nd × Nd} | Node Hodge Laplacian = δ₀ᵀδ₀ |
| L₁ | ℝ^{Md × Md} | Edge Hodge Laplacian = δ₀δ₀ᵀ + δ₁ᵀδ₁ |
| H^(0) | ℝ^{N×d} | Initial node embeddings (after encoder) |
| s^(0) | ℝ^{M×d} | Initial edge embeddings |
| x* | ℝ^{N×d} | Final node embeddings (after diffusion) |
| s* | ℝ^{M×d} | Final edge embeddings (after diffusion) |
| η | ℝ (>0) | Euler step size (η < 2/λ_max for stability) |
| λ_close | ℝ (≈ 10⁻⁶) | Frozen scaling for closure edge maps |
| Q | ℝ^{N×C} | Soft label matrix (Phase 0) |
| z^rel_r | ℝ^{d_r} | Relation embedding for type r |
| MLP_r^{src} | ℝ^{2d+d_r} → ℝ^{d²} | Source restriction MLP for relation r |
| MLP_r^{tgt} | ℝ^{2d+d_r} → ℝ^{d²} | Target restriction MLP for relation r |
| MLP_g | ℝ^{2d+d_r} → ℝ^{d_h} | Edge token encoder for LSTM input |
| MLP_F | ℝ^{d_h} → ℝ^{d²} | Shared face map projector |
| MLP_init | ℝ^{3d+d_r} → ℝ^d | Edge cochain initializer |
| MLP_cls | ℝ^{3d} → ℝ^C | Node classifier readout |
| W_τ | ℝ^{d × d_in(τ)} | Type-specific node encoder weight |
| ⊕ | — | Concatenation along feature axis |

---

## §1 Problem Setting

### 1.1 Input: Heterogeneous Information Network

A **Heterogeneous Information Network (HIN)** is:

```
G = (V, E, φ: V → T_V, ψ: E → R, X = {x_v}_{v∈V})
```

- **Multiple node types**: T_V = {Paper, Author, Subject} for ACM, etc.
- **Multiple relation types**: R = {PA, AP, PS, SP} for ACM
  (we treat each direction as a separate relation)
- **Typed raw features**: x_v ∈ ℝ^{d_raw(φ(v))}, dimension varies by type
- All edges are **directed**: e = (u →_r v) with u = src(e), v = tgt(e), rel(e) = r

### 1.2 Meta-paths

A meta-path template is a sequence of node types and relation types:

```
π = T₀ →^{r₁} T₁ →^{r₂} T₂ →^{r₃} ... →^{r_ℓ} T_ℓ
```

An **instance** of π is a concrete walk in G:

```
f = (v₀ →^{r₁} v₁ →^{r₂} v₂ → ... →^{r_ℓ} vₗ)
```

where φ(vᵢ) = Tᵢ and ψ(eᵢ) = rᵢ.

### 1.3 Task

**Node classification** on a specific target node type τ* (e.g., Paper in ACM).
Given a labeled subset V_train ⊂ {v ∈ V : φ(v) = τ*}, predict labels for
V_test ⊂ {v ∈ V : φ(v) = τ*}.

### 1.4 Target Datasets

| Dataset | Node Types | Key Relations | Target Type | Meta-paths Used |
|---------|-----------|---------------|-------------|-----------------|
| ACM | Paper(P), Author(A), Subject(S) | P-A, P-S | Paper | PAP, PSP |
| DBLP | Author(A), Paper(P), Term(T), Venue(V) | A-P, P-T, P-V | Author | APA, APTPA |
| IMDB | Movie(M), Actor(A), Director(D) | M-A, M-D | Movie | MAM, MDM |

Meta-path lengths:
- Length 2 (3 nodes, 2 edges): PAP, PSP, APA, MAM, MDM
- Length 4 (5 nodes, 4 edges): APTPA

For APTPA, **random sampling** of instances per anchor Author node is needed to keep |X²| tractable.

---

## §2 Cellular Complex Construction

### 2.1 Why Lift to a Cellular Complex?

Standard HGNNs treat meta-paths as graph construction tools — they project the HIN to a
homogeneous meta-path graph and lose direct cross-type edges. H²S-NN instead treats
meta-path instances as **actual topological cells**, allowing:

1. **Three-level signal representation**: node signals (0-cochains), edge signals (1-cochains),
   face signals (2-cochains)
2. **Hodge decomposition**: signals decompose into exact + harmonic + coexact parts
3. **Over-squashing mitigation**: 2-cells increase Forman-Ricci curvature (§10.4)

### 2.2 The Three Levels

**0-cells (nodes):** X⁰ = V. Each node v has stalk F(v) = ℝ^d.

**1-cells (edges):** X¹ = E ∪ Ē where Ē is the set of synthetic closure edges (defined below).
- E = all directed typed edges in the original HIN
- Ē = one closure edge per face (added for topological completeness)
- |X¹| = M + F where F = |X²|

**2-cells (faces):** X² = all meta-path instances over all templates in P.
Each face f = (v₀, e₁, v₁, ..., eₗ, vₗ).

### 2.3 Topological Requirement: Why We Need Closure Edges

> **Key issue (Bug 3 in original specs):** A CW-complex 2-cell is, by definition, a map
> from the closed 2-disk D² into the space whose restriction to the boundary circle ∂D²
> maps into the 1-skeleton. This requires the boundary to be a **closed loop**.
>
> A meta-path instance like PAP = (v₀ → v₁ → v₂) is a directed **path** — not a closed
> loop. We cannot attach a 2-disk along an open path.

**Solution:** For each face f = (v₀, e₁, v₁, ..., eₗ, vₗ), introduce a **closure edge**:

```
ē_f = (vₗ →_{r_close} v₀)   [a synthetic directed edge from end-node to start-node]
```

The closed boundary cycle is:

```
∂f = (e₁, e₂, ..., eₗ, ē_f)   [ℓ+1 directed edges forming a closed loop]
```

This is topologically valid — the boundary traces a closed loop in the 1-skeleton.

**Closure edge restriction maps (frozen throughout training):**

```
F^src_{vₗ▷ē_f} = λ_close · I_d  ∈ ℝ^{d×d}
F^tgt_{v₀▷ē_f} = λ_close · I_d  ∈ ℝ^{d×d}
F_{ē_f▷f}      = 0_{d×d}         ∈ ℝ^{d×d}    ← ZERO: no diffusion contribution
```

where λ_close = 10⁻⁶. The closure edge contributes nothing to any Laplacian operator
(since F_{ē_f▷f} = 0), but satisfies the topological requirement. It has negligible
effect on L₀ through the tiny λ_close · I maps.

### 2.4 Orientation Convention

For each 2-cell f with boundary cycle (e₁, ..., eₗ, ē_f), all edges are traversed in their
natural directed orientation. We assign:

```
η(eᵢ, f) = +1  for i = 1,...,ℓ    (path edges)
η(ē_f, f) = +1                     (closure edge)
```

This is consistent because all edges point in the same "winding direction" around the closed
boundary cycle.

### 2.5 Construction Algorithm (Phase 0, Step 0.5)

```
INPUT:  G = (V, E, φ, ψ), meta-path templates P = {π₁,...,π_P}
OUTPUT: X² = list of face dicts, Ē = list of closure edges

X², Ē = [], []
for each template π = (T₀ →^{r₁} ... →^{r_ℓ} T_ℓ) in P:
    instances = enumerate_by_DFS(G, π)
    # DFS from each v₀ ∈ {v : φ(v)=T₀} following relations r₁,...,r_ℓ
    # For APTPA (length=4): sample at most K_face=50 instances per anchor

    for (v₀, e₁, v₁, ..., eₗ, vₗ) in instances:
        ē = DirectedEdge(src=vₗ, tgt=v₀, rel='closure', weight=λ_close)
        f = {
            'path_nodes': [v₀, v₁, ..., vₗ],         # list of ℓ+1 node indices
            'path_edges': [e₁, e₂, ..., eₗ],          # list of ℓ edge indices in E
            'closure_edge': ē,                          # index in Ē
            'template': π,
            'boundary': [e₁,...,eₗ, ē],                # ℓ+1 edges in order
        }
        X².append(f)
        Ē.append(ē)

X¹ = E ∪ Ē   # total edge set, size M + F
```

---

## §3 Directed Typed Cellular Sheaf

### 3.1 Definition

A **directed cellular sheaf** F on (X⁰, X¹, X²) assigns:

1. **Stalks** to every cell:
   - Node stalks: F(v) = ℝ^d for each v ∈ X⁰
   - Edge stalks: F(e) = ℝ^d for each e ∈ X¹
   - Face stalks: F(f) = ℝ^d for each f ∈ X²

2. **Restriction maps** relating adjacent cells:
   - For each directed edge e = (u →_r v) in X¹:
     - **Source map**: F^src_{u▷e} : F(u) → F(e), i.e., F^src_{u▷e} ∈ ℝ^{d×d}
     - **Target map**: F^tgt_{v▷e} : F(v) → F(e), i.e., F^tgt_{v▷e} ∈ ℝ^{d×d}
   - For each edge e in the boundary ∂f of face f:
     - **Edge-to-face map**: F_{e▷f} : F(e) → F(f), i.e., F_{e▷f} ∈ ℝ^{d×d}

### 3.2 Why Directed (vs Classical Undirected Sheaf)?

In classical NSD (Bodnar et al. 2022), undirected edges have a single restriction map
F_{v≤e} per vertex-edge pair. This means if node i ignores its neighbor j (sets F_{i≤e}=0),
it also stops sending information to j — the model cannot independently control
"listening" vs "propagating" behavior.

The directed sheaf (following CSNN, Ribeiro et al. 2026) uses **separate source and target
maps** per edge, enabling:
- F^src_{u▷e} = 0 → u does not propagate into edge e (but v can still listen via F^tgt)
- F^tgt_{v▷e} = 0 → v does not listen from edge e (but u can still propagate)
- Independent control per direction, per relation type

### 3.3 Typed Maps

For a HIN with relation types R, the restriction maps are **typed by relation r = rel(e)**:

- F^src_{u▷e} is output of MLP^src_r([x_u; x_v; z^rel_r])  — different MLP per r
- F^tgt_{v▷e} is output of MLP^tgt_r([x_u; x_v; z^rel_r])  — different MLP per r

This gives the sheaf **heterogeneous semantics**: PAP paths and PSP paths use different
restriction maps, capturing different relational inductive biases.

### 3.4 Cochain Spaces

- **0-cochains**: C⁰(X; F) = ⊕_{v∈X⁰} F(v) ≅ ℝ^{Nd}  [node signals]
- **1-cochains**: C¹(X; F) = ⊕_{e∈X¹} F(e) ≅ ℝ^{(M+F)d}  [edge signals]
- **2-cochains**: C²(X; F) = ⊕_{f∈X²} F(f) ≅ ℝ^{Fd}  [face signals]

In practice, we keep the active edges (E, size M) and closure edges (Ē, size F) separate
for computational efficiency, since Ē contributes negligibly to all operators.

---

## §4 Coboundary Operators

### 4.1 δ₀: 0-cochains → 1-cochains

For node signal x = (x_v)_{v∈V} ∈ ℝ^{Nd}, the coboundary δ₀ maps to edge signals:

```
(δ₀ x)_e = F^tgt_{v▷e} x_v − F^src_{u▷e} x_u   ∈ ℝ^d
```

for each edge e = (u →_r v) ∈ X¹.

**Interpretation:** (δ₀ x)_e measures the "disagreement" between node u's signal projected
into the edge stalk via the source map, and node v's signal projected via the target map.
When (δ₀ x)_e = 0 for all edges, x is in the **kernel of L₀** (the harmonic space).

**Matrix form:** δ₀ ∈ ℝ^{(M+F)d × Nd} is a sparse block matrix with blocks:
```
[δ₀]_{e, v} = +F^tgt_{v▷e}   if v = tgt(e)
[δ₀]_{e, u} = −F^src_{u▷e}   if u = src(e)
[δ₀]_{e, w} = 0               otherwise
```

**Adjoint δ₀ᵀ ∈ ℝ^{Nd × (M+F)d}:** For edge signal s = (s_e)_{e∈X¹} ∈ ℝ^{(M+F)d}:

```
(δ₀ᵀ s)_v =   Σ_{e: tgt(e)=v} (F^tgt_{v▷e})ᵀ s_e
             − Σ_{e: src(e)=v} (F^src_{v▷e})ᵀ s_e
```

**Verification of adjointness:** ⟨δ₀ x, s⟩ = Σ_e sₑᵀ(F^tgt x_{tgt(e)} − F^src x_{src(e)})
= Σ_v xᵥᵀ[(Σ_{tgt=v} (F^tgt)ᵀ sₑ) − (Σ_{src=v} (F^src)ᵀ sₑ)] = ⟨x, δ₀ᵀ s⟩. ✓

### 4.2 δ₁: 1-cochains → 2-cochains

For edge signal s = (s_e)_{e∈X¹} ∈ ℝ^{(M+F)d}, the coboundary δ₁ maps to face signals:

```
(δ₁ s)_f = Σ_{e ∈ ∂f} η(e, f) · F_{e▷f} s_e   ∈ ℝ^d
```

Since η(e, f) = +1 for all boundary edges, and F_{ē_f▷f} = 0 for closure edges:

```
(δ₁ s)_f = Σᵢ₌₁ˡ F_{eᵢ▷f} s_{eᵢ}   [closure edges contribute 0]
```

**Adjoint δ₁ᵀ ∈ ℝ^{(M+F)d × Fd}:** For face signal q ∈ ℝ^{Fd}:

```
(δ₁ᵀ q)_e = Σ_{f: e ∈ ∂f} η(e, f) · (F_{e▷f})ᵀ q_f
           = Σ_{f: e ∈ path(f)} (F_{e▷f})ᵀ q_f
```

(closure edge rows are zero since F_{ē_f▷f} = 0.)

### 4.3 The Coboundary Condition δ₁ ∘ δ₀ = 0

For a valid cellular sheaf complex (chain complex), we require:

```
δ₁ ∘ δ₀ = 0   [the composition of two coboundaries is zero]
```

This is the fundamental axiom enabling the Hodge decomposition (§5.3).

**What it means concretely:** For each face f = (v₀, e₁, v₁, ..., eₗ, vₗ):

```
(δ₁ δ₀ x)_f = Σᵢ₌₁ˡ F_{eᵢ▷f} (δ₀ x)_{eᵢ}
             = Σᵢ₌₁ˡ F_{eᵢ▷f} (F^tgt_{vᵢ▷eᵢ} xᵥᵢ − F^src_{uᵢ▷eᵢ} xᵤᵢ)
```

Collecting terms by node (where uᵢ = vᵢ₋₁ = src(eᵢ)):

```
= Σᵥ [Σ_{i: tgt(eᵢ)=v} F_{eᵢ▷f} F^tgt_{v▷eᵢ} − Σ_{i: src(eᵢ)=v} F_{eᵢ▷f} F^src_{v▷eᵢ}] xᵥ
```

For this to be zero for ALL node signals x, we need the **per-node consistency condition**:

```
Σ_{i: tgt(eᵢ)=v} F_{eᵢ▷f} F^tgt_{v▷eᵢ} = Σ_{i: src(eᵢ)=v} F_{eᵢ▷f} F^src_{v▷eᵢ}   ∀v, ∀f
```

Define the **per-node net map** at node v within face f:

```
N_f(v) = Σ_{i: tgt(eᵢ)=v} αᵢ F^tgt_{v▷eᵢ} − Σ_{i: src(eᵢ)=v} αᵢ F^src_{v▷eᵢ}  ∈ ℝ^{d×d}
```

(where αᵢ are the attention weights from Step B). The condition δ₁∘δ₀=0 is exactly:

```
Σᵥ (F_{e(v)▷f})net N_f(v) = 0   for each face f
```

### 4.4 Critical Bug Fix: Spec 2's SVD Construction is Mathematically Incomplete

> **What Spec 2 claimed:** Build M_f = Σᵢ αᵢ (F^tgt_{vᵢ▷eᵢ})ᵀ ∈ ℝ^{d×d}, take the SVD
> null-space U_{f,⊥}, set W_f = U_{f,⊥}ᵀ P_f, and F_{eᵢ▷f} = αᵢ Wfᵀ.
> This "enforces δ₁∘δ₀=0 identically."
>
> **Why this is wrong:** The construction ensures W_fᵀ [Σᵢ αᵢ F^tgt_{vᵢ▷eᵢ} xᵥᵢ] = 0
> but completely ignores the source map term Σᵢ αᵢ F^src_{uᵢ▷eᵢ} xᵤᵢ.
> For a path (v₀→v₁→v₂), v₁ is both the target of e₁ and the source of e₂.
> The net map N_f(v₁) = α₁ F^tgt_{v₁▷e₁} − α₂ F^src_{v₁▷e₂} must be annihilated,
> but Spec 2 only annihilates α₁ F^tgt_{v₁▷e₁}, leaving α₂ F^src_{v₁▷e₂} unconstrained.
>
> **The corrected M_f** would need to stack all per-node net maps:
> M_f^correct = [N_f(v₀)ᵀ | N_f(v₁)ᵀ | ... | N_f(vₗ)ᵀ] ∈ ℝ^{d × (ℓ+1)d}
> This has variable rank, making batching across faces difficult.

### 4.5 Adopted Fix: Soft Coboundary Regularization

Rather than a hard SVD constraint, we enforce δ₁∘δ₀=0 **softly** via a loss penalty:

```
L_co = λ_co · Σ_{f ∈ X²} ||(δ₁ δ₀ H^(0))_f||²_F
```

evaluated at the initial node embeddings H^(0). This:
- Is fully differentiable (no SVD backward issues)
- Requires no variable-rank batching
- Is standard practice in applied topology papers
- Allows the model to prioritize the classification objective while being guided toward
  a valid chain complex structure

The edge-to-face maps F_{e▷f} are learned via MLP_F from LSTM hidden states (§6.3).

---

## §5 Hodge Laplacians

### 5.1 Node Laplacian L₀

**Definition:**

```
L₀ = δ₀ᵀ δ₀ ∈ ℝ^{Nd × Nd}
```

**Block structure** (for pair of nodes u, v connected by edge e = (u→v)):

Diagonal block for node v:
```
[L₀]_{v,v} = Σ_{e: tgt(e)=v} (F^tgt_{v▷e})ᵀ F^tgt_{v▷e}
            + Σ_{e: src(e)=v} (F^src_{v▷e})ᵀ F^src_{v▷e}
∈ ℝ^{d×d}
```

Off-diagonal block for (u, v) connected by directed edge e = (u→v):
```
[L₀]_{u,v} = −(F^src_{u▷e})ᵀ F^tgt_{v▷e}   ∈ ℝ^{d×d}
[L₀]_{v,u} = −(F^tgt_{v▷e})ᵀ F^src_{u▷e} = [L₀]_{u,v}ᵀ
```

**Note on symmetry:** [L₀]_{v,u} = ([L₀]_{u,v})ᵀ as block matrices (block-symmetry), which
means L₀ is symmetric as a full matrix. Do not confuse block-transpose symmetry with
element-wise commutativity.

**Proposition 1 (L₀ is symmetric PSD):**
For any x ∈ ℝ^{Nd}:
```
xᵀ L₀ x = xᵀ δ₀ᵀ δ₀ x = ||δ₀ x||² ≥ 0
```
Symmetry: (δ₀ᵀ δ₀)ᵀ = δ₀ᵀ (δ₀ᵀ)ᵀ = δ₀ᵀ δ₀. ✓

**Kernel of L₀:**
```
ker(L₀) = {x ∈ ℝ^{Nd} : δ₀ x = 0}
         = {x : F^tgt_{v▷e} xᵥ = F^src_{u▷e} xᵤ  for all edges e = (u→v)}
```

This is NOT the all-constant space (unlike the graph Laplacian). It is the space of
**sheaf-harmonic node signals** — signals where every edge is in "local agreement" under
the learned typed maps. This is the key mechanism preventing over-smoothing:
heterophilic nodes can be in ker(L₀) while having very different raw features, as long
as their difference is in the null space of the restriction maps.

### 5.2 Edge Laplacian L₁

**Definition:**

```
L₁ = δ₀ δ₀ᵀ + δ₁ᵀ δ₁ ∈ ℝ^{Md × Md}
```

(Here we use only the M active edges; closure edges are excluded as they contribute ~0.)

**Two components:**

- **Down-Laplacian**: L₁^↓ = δ₀ δ₀ᵀ
  Couples edge signals through **shared boundary nodes** (two edges sharing a node are
  connected in the down-Laplacian). Measures how much edge e's signal disagrees with
  the "re-lifted" signal from its endpoints via the adjoint maps.

- **Up-Laplacian**: L₁^↑ = δ₁ᵀ δ₁
  Couples edge signals through **shared co-boundary faces** (two edges that both appear
  in the same meta-path instance are connected in the up-Laplacian). This is the
  *meta-path-aware* coupling — edges in the same semantic path share information.

**Proposition 2 (L₁ is symmetric PSD):**
Both δ₀ δ₀ᵀ and δ₁ᵀ δ₁ are of the form AᵀA (or AAᵀ), hence PSD. Their sum is PSD. ✓

**Kernel of L₁:**
```
ker(L₁) = ker(δ₀ᵀ) ∩ ker(δ₁)
         = {s : δ₀ᵀ s = 0} ∩ {s : δ₁ s = 0}
         = "divergence-free" edges that also vanish on all faces
```

This is the **harmonic 1-cohomology** H¹(X; F) — the space encoding irreducible
relational topology not captured by either nodes or faces.

### 5.3 Hodge Decomposition (When δ₁ ∘ δ₀ = 0)

When the coboundary condition holds (or approximately holds via L_co → 0), the
1-cochain space has an orthogonal decomposition:

```
C¹(X; F) = im(δ₀) ⊕ H¹(X; F) ⊕ im(δ₁ᵀ)

where:
  im(δ₀)   = "exact" 1-cochains: edge signals that are coboundaries of node signals
  H¹(X;F)  = "harmonic" 1-cochains: ker(L₁) (the interesting part for H²S-NN)
  im(δ₁ᵀ) = "co-exact" 1-cochains: edge signals that are "pulled back" from faces
```

**Why this matters:**
- Diffusion ds/dt = -L₁ s damps the exact and co-exact components, preserving H¹
- The harmonic edge signals encode the topological/relational structure that cannot be
  reduced to either node information or meta-path information alone
- This is the key representation power claim of H²S-NN over methods that only use
  node signals

### 5.4 Matrix-Free Implementation (Critical for Large Graphs)

> **Do NOT construct L₀ or L₁ as dense matrices.**
> For |V|=8000, d=128: L₀ is (8000×128)² ≈ 10⁹ entries — infeasible.

Always use the **operator form**:

```python
# apply_L0(x): ℝ^{N×d} → ℝ^{N×d}
def apply_L0(x, F_src, F_tgt, edge_index):
    # Step 1: compute δ₀ x  [M, d]
    delta0_x = apply_delta0(x, F_src, F_tgt, edge_index)
    # Step 2: compute δ₀ᵀ (δ₀ x)  [N, d]
    return apply_delta0_T(delta0_x, F_src, F_tgt, edge_index, N=x.shape[0])

# apply_L1(s): ℝ^{M×d} → ℝ^{M×d}
def apply_L1(s, F_src, F_tgt, edge_index, F_ef, face_data):
    down = apply_delta0(apply_delta0_T(s, ...), ...)   # down-Laplacian term
    up   = apply_delta1_T(apply_delta1(s, F_ef, ...), ...)  # up-Laplacian term
    return down + up
```

This is O(M·d²) per application — feasible for all target datasets.

---

## §6 Restriction Map Learning

### 6.1 Phase 0: Preprocessing (Frozen After Computation)

These steps run once before training and produce frozen auxiliary features.

**Step 0.1 — Soft Label Propagation (for heterophily metrics)**

```
Q ∈ ℝ^{N×C},  Q[v] = one_hot(y_v) if v ∈ V_train, else 0
for t = 1, 2, 3:
    Q ← D⁻¹ A Q              # row-normalized adjacency walk
    Q = clamp(Q, min=0)
    Q = Q / max(Q.sum(1, keepdim=True), 1)
    Q[V_train] = one_hot(y_v)  # re-anchor labeled nodes
freeze Q                        # shape: [N, C]
```

**Step 0.2 — Typed Structural Descriptors**

For each node v and relation type r:
```
Ξ^r_v = [deg_r(v), 2-hop_count_r(v), 3-hop_count_r(v)]  ∈ ℝ³
z^(S,r)_v = softmax(normalize(log(1 + Ξ^r_v)))
```
Frozen structural descriptor: z^(S)_v = concat(z^(S,r)_v for r ∈ R) ∈ ℝ^{3|R|}

**Step 0.3 — Feature Residuals (label-independent component)**

```
for each type τ:
    μ_τ = mean of x_v for v with φ(v)=τ;  σ_τ = std
    x̃_v = (x_v - μ_τ) / σ_τ              # standardize
for each class c:
    μ_c = mean of x̃_v for v ∈ V_train ∩ {class c}
for each node v:
    ρ_v = x̃_v - Σ_c Q[v,c] · μ_c         # remove label-predictive direction
    z^(F)_v = LayerNorm(ρ_v)              # ℝ^{d_raw(τ(v))}
```

**Step 0.4 — Heterophily Metrics per Relation Type**

For each relation r ∈ R, compute:

```
# Label heterophily H^L_r ∈ [-1, 1]:
A^L_r(e) = Q[src(e)] · Q[tgt(e)]  (dot product = label agreement for e ∈ E_r)
b^L_r   = Σ_{v} deg_r(v)/|E_r| * (Q[v] · Q[v])  (degree-weighted null baseline)
H^L_r   = piecewise_sign_normalize(mean(A^L_r), b^L_r)

# Structural heterophily H^S_r: same formula but replace Q with z^(S)
# Feature heterophily H^F_r: same formula but use x̃ dot products
# CTLI_r: normalized mutual information between labels and edge incidence pattern
```

**Piecewise sign normalization** maps raw score a with baseline b to [-1, 1]:
```python
def piecewise_normalize(a, b):
    if a >= b:  return (a - b) / (1 - b)   # maps [b, 1] → [0, 1]
    else:       return (a - b) / b          # maps [0, b] → [-1, 0]
    # (handle b=0 and b=1 as edge cases)
```

Five axiomatic properties satisfied: Max, Min, ACB (asymptotic constant baseline),
ECT (empty class tolerance), Mon (monotonicity). See §10.5 for formal statements.

**Step 0.5 — Complex Construction** (see §2.5)

**Final preprocessed features:**
```
x̂_v = concat(x_v, z^(S)_v, [H^L_r, H^S_r, H^F_r, CTLI_r for r in R])
# x̂_v ∈ ℝ^{d_raw(τ(v)) + 3|R| + 4|R|}  =  ℝ^{d_raw(τ(v)) + 7|R|}
```

### 6.2 Phase 1: Type-Specific Node Encoder

```
INPUT: x̂_v ∈ ℝ^{d_in(φ(v))} for each v  (d_in varies by type)
OUTPUT: H^(0) ∈ ℝ^{N×d}  (unified stalk dimension d for all nodes)

H^(0)_v = LayerNorm(W_{φ(v)} x̂_v + b_{φ(v)})

Parameters:
  W_τ ∈ ℝ^{d × d_in(τ)},  b_τ ∈ ℝ^d   for each τ ∈ T_V
  (separate weight matrix and bias per node type)
```

**Why not STAGE?** Full STAGE (Stiefel-Manifold Geometric Encoder) is the Year 2 contribution
requiring Riemannian optimization on the Stiefel manifold. For H²S-NN (Year 1 paper), this
type-specific linear projection is the correct and stated simplification. STAGE can be
substituted as a drop-in improvement.

**Relation embeddings:**
```
z^rel_r = Embedding(r)  ∈ ℝ^{d_r}  for each r ∈ R ∪ {r_close}
```
One learnable embedding per relation type. d_r is a hyperparameter (typically 16–64).

### 6.3 Step A: Node-to-Edge Restriction Maps

**For each relation type r, learn two MLPs:**

```
MLP^src_r : ℝ^{2d + d_r} → ℝ^{d²}   (source restriction)
MLP^tgt_r : ℝ^{2d + d_r} → ℝ^{d²}   (target restriction)
```

Each MLP is a 2-layer network: Linear → ReLU → Linear (with optional layer norm).

**For each edge e = (u →_r v):**
```
inp_e      = concat(H^(0)_u, H^(0)_v, z^rel_r)          # ∈ ℝ^{2d + d_r}
flat_src   = MLP^src_r(inp_e)                              # ∈ ℝ^{d²}
flat_tgt   = MLP^tgt_r(inp_e)                              # ∈ ℝ^{d²}
F^src_{u▷e} = reshape(flat_src, [d, d])                  # ∈ ℝ^{d×d}
F^tgt_{v▷e} = reshape(flat_tgt, [d, d])                  # ∈ ℝ^{d×d}
```

**Vectorized over all edges of relation type r simultaneously:**
```python
# H: [N, d], edge_index_r: [2, M_r], rel_emb_r: [d_r]
src_idx, tgt_idx = edge_index_r  # [M_r] each
inp = torch.cat([H[src_idx], H[tgt_idx],
                 rel_emb_r.unsqueeze(0).expand(M_r, -1)], dim=-1)  # [M_r, 2d+d_r]
F_src_r = mlp_src_r(inp).view(M_r, d, d)   # [M_r, d, d]
F_tgt_r = mlp_tgt_r(inp).view(M_r, d, d)   # [M_r, d, d]
```

**Optional orthogonal parameterization** (recommended, from `models/orthogonal.py`):
```python
# Parameterize F^src as c_s · Q_s where Q_s is orthogonal, c_s > 0 is scale
# This prevents rank collapse and keeps L₀ well-conditioned
orth_s = Orthogonal(d, orthogonal_map='householder')  # from baseline
c_s = F.softplus(mlp_scale_s(inp))        # positive scalar, [M_r]
Q_s = orth_s(mlp_orth_s(inp))             # orthogonal matrix, [M_r, d, d]
F_src_r = c_s.unsqueeze(-1).unsqueeze(-1) * Q_s
```

**Compute coboundary δ₀:**
```python
# Collect all F_src, F_tgt across all relation types
F_src = torch.zeros(M, d, d)  # [M, d, d]
F_tgt = torch.zeros(M, d, d)  # [M, d, d]
# Fill by relation type using edge_type mask
for r, (F_sr, F_tr, idx_r) in zip(R, per_relation_maps):
    F_src[idx_r] = F_sr
    F_tgt[idx_r] = F_tr

# δ₀ H^(0): [M, d]
# F_tgt[e] @ H[tgt_e] - F_src[e] @ H[src_e]
src_idx, tgt_idx = edge_index  # [M] each
delta0_H = (torch.bmm(F_tgt, H[tgt_idx].unsqueeze(-1)).squeeze(-1)  # [M, d]
          - torch.bmm(F_src, H[src_idx].unsqueeze(-1)).squeeze(-1))  # [M, d]
```

**Tensor summary:**

| Tensor | Shape | Notes |
|--------|-------|-------|
| inp per edge | [M_r, 2d+d_r] | Batched over relation type |
| F_src_r | [M_r, d, d] | Source maps for relation r |
| F_tgt_r | [M_r, d, d] | Target maps for relation r |
| F_src (global) | [M, d, d] | All source maps |
| F_tgt (global) | [M, d, d] | All target maps |
| delta0_H | [M, d] | δ₀ H^(0) |

### 6.4 Step B: Edge-to-Face Restriction Maps (LSTM + Attention)

This step learns F_{e▷f} for each edge e in each face f, using an LSTM to capture
the sequential structure of the meta-path.

**Shared parameters (used across all faces and all meta-path types):**
```
MLP_g  : ℝ^{2d + d_r} → ℝ^{d_h}   (edge token encoder)
LSTM   : input_size=d_h, hidden_size=d_h, batch_first=True
a      : ℝ^{d_h}                    (attention query vector)
MLP_F  : ℝ^{d_h} → ℝ^{d²}          (face map projector)
```

**For each face f = (v₀ →^{e₁} v₁ →^{e₂} ... →^{e_ℓ} vₗ):**

```
# B.1: Build LSTM input tokens  [ℓ, d_h]
g(eᵢ) = MLP_g(concat(H^(0)_{src(eᵢ)}, H^(0)_{tgt(eᵢ)}, z^rel_{rel(eᵢ)}))  ∈ ℝ^{d_h}
tokens_f = stack([g(e₁), ..., g(eₗ)])  ∈ ℝ^{ℓ×d_h}

# B.2: LSTM forward pass
h₁,...,hₗ = LSTM(tokens_f)  [ℓ, d_h]   (all hidden states, not just last)
# h_i encodes the "context up to position i" in the meta-path sequence

# B.3: Attention weights (soft, normalized)
α_i_raw = (hᵢ · a) / sqrt(d_h)   ∈ ℝ       # dot-product attention
[α₁,...,αₗ] = softmax([α₁_raw,...,αₗ_raw])  # normalize over path edges

# B.4: Edge-to-face maps  (CORRECTED: direct MLP, no SVD)
F_{eᵢ▷f} = reshape(MLP_F(hᵢ), [d, d])  ∈ ℝ^{d×d}
# Note: the attention weights αᵢ are used in the coboundary regularizer computation
# but NOT directly in the map construction (the LSTM hidden state already encodes position)
```

**Batching across faces:** Group faces by meta-path template type (same sequence length ℓ).
For template with ℓ=2 (PAP, PSP, APA, MAM, MDM), batch all instances together:
```python
# faces_by_type[π]: list of face instances for template π of length ℓ_π
# For template π with ℓ edges:
batch_size_f = len(faces_by_type[π])
tokens_batch = build_tokens(H, faces_by_type[π], rel_emb)  # [batch_f, ℓ, d_h]
h_batch, _ = self.lstm(tokens_batch)                        # [batch_f, ℓ, d_h]
F_ef_batch = self.mlp_F(h_batch).view(batch_f, ℓ, d, d)   # [batch_f, ℓ, d, d]
```

**Closure edges:** F_{ē_f▷f} = 0 (frozen, never passed through LSTM).

**Tensor summary for a template of length ℓ with B_f instances:**

| Tensor | Shape | Notes |
|--------|-------|-------|
| tokens_f | [B_f, ℓ, d_h] | LSTM inputs |
| h_batch | [B_f, ℓ, d_h] | LSTM hidden states |
| α_batch | [B_f, ℓ] | Attention weights |
| F_ef_batch | [B_f, ℓ, d, d] | Edge-to-face maps |

---

## §7 Hodge Diffusion

### 7.1 Edge Cochain Initialization (Step D)

> **Critical design choice (correct in Spec 2):** Do NOT initialize s^(0) = δ₀ H^(0).
>
> **Why:** If s^(0) = δ₀ H^(0), then s^(0) ∈ im(δ₀) — the exact component of C¹.
> The harmonic component H¹(X;F) of s^(0) would be zero. After diffusion,
> s* ≈ proj_{H¹} s^(0) = 0 — the edge signals would carry no harmonic information.
>
> **Correct approach:** Use δ₀ H^(0) as an *input feature* to MLP_init, not as
> the initialization itself. This allows s^(0) to have all three Hodge components.

```
INPUT: H^(0) ∈ ℝ^{N×d}, edge_index ∈ ℤ^{2×M}, edge_type ∈ ℤ^M, delta0_H ∈ ℝ^{M×d}

For each edge e = (u →_r v) in E:
    inp_e = concat(H^(0)_u, H^(0)_v, z^rel_r, delta0_H[e])  ∈ ℝ^{3d + d_r}
    s^(0)_e = MLP_init(inp_e)                                  ∈ ℝ^d

# MLP_init: ℝ^{3d+d_r} → ℝ^d, 2-layer with ReLU
# Shared weights; conditioned on relation type via z^rel_r

For each closure edge ē_f:
    s^(0)_{ē_f} = F^tgt_{v₀▷ē_f} H^(0)_{v₀} - F^src_{vₗ▷ē_f} H^(0)_{vₗ}
                = λ_close (H^(0)_{v₀} - H^(0)_{vₗ})   [from frozen λ_close · I maps]
```

### 7.2 Coupled Hodge Diffusion: K-Step Euler (Step E)

**Mathematical motivation:** The Hodge heat equations:
```
dx/dt = -L₀ x(t),    x(0) = H^(0)
ds/dt = -L₁ s(t),    s(0) = s^(0)
```

L₀ and L₁ are **fixed** during integration (computed once from H^(0) and the learned
restriction maps). This makes both ODEs linear, and Euler discretization is exact in
the limit of small step size.

**K-step Euler (recommended default):**
```
x ← H^(0)   ∈ ℝ^{N×d}
s ← s^(0)   ∈ ℝ^{M×d}

for k = 1, ..., K:
    x ← x - η₀ · apply_L0(x)   # [N, d] → [N, d]
    s ← s - η₁ · apply_L1(s)   # [M, d] → [M, d]

x* = x
s* = s
```

**Stability condition:** For Euler integration of dx/dt = -Lx to be stable:
```
η < 2 / λ_max(L)
```

Since λ_max(L₀) ≤ 2 · max_degree · max_singular_value(F)², we use:
```
η₀ = η₁ = 1 / (1 + max_degree)   [conservative safe bound]
```

Or make η₀, η₁ learnable with constraints (e.g., parametrize as σ(w) · η_max).

**Recommended hyperparameters:** K = 5–8, η = 0.1–0.2. Treat K as a hyperparameter and
tune alongside d and other model params.

**Why not Neural ODE (dopri5)?**
1. `torchdiffeq` is not in requirements.txt
2. The ODE is linear (L₀, L₁ fixed), so Euler with K≥4 gives very similar results
3. Adjoint memory savings only relevant for long integration (T >> 1), not our K=5–8 case
4. Euler is easier to debug and interpretable

**Optional: Neural ODE variant (for ablation)**
```python
# Add to requirements.txt: torchdiffeq>=0.2.3
from torchdiffeq import odeint_adjoint as odeint

# L0_op and L1_op are precomputed linear operators (fixed during odeint)
def hodge_ode(t, state):
    # state: [Nd + Md] flattened
    x = state[:N*d].view(N, d)
    s = state[N*d:].view(M, d)
    return torch.cat([(-L0_op @ x.flatten()).view(N, d).flatten(),
                      (-L1_op @ s.flatten()).view(M, d).flatten()])

state0 = torch.cat([H0.flatten(), s0.flatten()])
stateT = odeint(hodge_ode, state0, torch.tensor([0., 1.0]),
                method='rk4', rtol=1e-3, atol=1e-4,
                adjoint_params=list(model.parameters()))
x_star = stateT[-1, :N*d].view(N, d)
s_star = stateT[-1, N*d:].view(M, d)
```

---

## §8 Complete Forward Pass Algorithm

```
═══════════════════════════════════════════════════════════════
ALGORITHM H²S-NN FORWARD PASS (one training step)
═══════════════════════════════════════════════════════════════

INPUT:
  data.x_dict: dict {τ: ℝ^{N_τ × d_raw(τ)}}  (raw features per node type)
  data.edge_index_dict: dict {(τ₁,r,τ₂): ℝ^{2 × M_r}}
  data.node_type: ℤ^N (global type index)
  data.edge_index: ℤ^{2 × M} (global flattened edge index)
  data.edge_type: ℤ^M (global relation type index)
  data.faces: list of face dicts (from Phase 0.5)
  data.x_hat: ℝ^{N × d_in} (preprocessed features from Phase 0)
  data.train_mask: bool^N, data.y: ℤ^N (labels for target node type)

HYPERPARAMETERS:
  d = 128 (stalk dimension)
  d_r = 32 (relation embedding dim)
  d_h = 128 (LSTM hidden dim)
  K = 5 (Euler steps)
  η = 0.1 (step size)
  λ₀, λ₁, λ_c, λ_co (loss weights)

─────────────────────────────────────────────────────────────
PHASE 1: NODE INITIALIZATION
─────────────────────────────────────────────────────────────

H^(0)_v = LayerNorm(W_{φ(v)} x̂_v + b_{φ(v)})
# input: x̂_v ∈ ℝ^{d_in(φ(v))} (from Phase 0)
# output: H^(0) ∈ ℝ^{N × d}
# parameters: W_τ ∈ ℝ^{d × d_in(τ)}, b_τ ∈ ℝ^d for each τ ∈ T_V

z^rel_r = Embedding(r)  for each r ∈ R
# output: z^rel ∈ ℝ^{|R| × d_r}

─────────────────────────────────────────────────────────────
STEP A: NODE-TO-EDGE RESTRICTION MAPS
─────────────────────────────────────────────────────────────

for each relation type r:
    E_r = edges of type r;  M_r = |E_r|
    src_r, tgt_r = edge_index for E_r   # ℤ^{M_r} each
    inp_r = concat(H^(0)[src_r], H^(0)[tgt_r], z^rel_r.expand(M_r, -1))
    #      ∈ ℝ^{M_r × (2d + d_r)}
    F_src_r = MLP^src_r(inp_r).view(M_r, d, d)   # ℝ^{M_r × d × d}
    F_tgt_r = MLP^tgt_r(inp_r).view(M_r, d, d)   # ℝ^{M_r × d × d}

Assemble global F_src, F_tgt ∈ ℝ^{M × d × d} by filling per-relation blocks.

Compute coboundary:
    src_all, tgt_all = edge_index  # ℤ^M each
    delta0_H = bmm(F_tgt, H^(0)[tgt_all, :, None]).squeeze(-1)    # [M, d]
             - bmm(F_src, H^(0)[src_all, :, None]).squeeze(-1)    # [M, d]
    # delta0_H ∈ ℝ^{M × d}

─────────────────────────────────────────────────────────────
STEP B: EDGE-TO-FACE RESTRICTION MAPS
─────────────────────────────────────────────────────────────

F_ef = dict()   # face_idx → tensor of shape [ℓ, d, d]

Group faces by meta-path template type π (same ℓ):
for each template π (length ℓ):
    B_f = number of instances of π
    # Build LSTM input: [B_f, ℓ, d_h]
    for i = 1,...,ℓ:
        g_i = MLP_g(concat(H^(0)[src(eᵢ)],        # [B_f, d]
                           H^(0)[tgt(eᵢ)],        # [B_f, d]
                           z^rel[rel(eᵢ)]))        # [B_f, d_r]
             # output: [B_f, d_h]
    tokens = stack(g₁,...,gₗ, dim=1)  ∈ ℝ^{B_f × ℓ × d_h}

    h, _ = LSTM(tokens)                 # [B_f, ℓ, d_h]

    # Attention (for coboundary regularizer)
    alpha_raw = (h @ a) / sqrt(d_h)    # [B_f, ℓ]
    alpha = softmax(alpha_raw, dim=-1) # [B_f, ℓ]

    # Face maps (direct from LSTM hidden state)
    F_ef_π = MLP_F(h).view(B_f, ℓ, d, d)  # [B_f, ℓ, d, d]
    store F_ef_π, alpha_π for this template

─────────────────────────────────────────────────────────────
STEP C: ASSEMBLE COBOUNDARY OPERATORS (matrix-free)
─────────────────────────────────────────────────────────────

Define the following as inline functions (closures over F_src, F_tgt, F_ef):

apply_delta0(x):          ℝ^{N×d} → ℝ^{M×d}
    return bmm(F_tgt, x[tgt_all, :, None]).squeeze(-1)
         - bmm(F_src, x[src_all, :, None]).squeeze(-1)

apply_delta0_T(s):        ℝ^{M×d} → ℝ^{N×d}
    out = zeros(N, d)
    out.scatter_add_(0, tgt_all.unsqueeze(-1).expand_as(s),
                     bmm(F_tgt.transpose(-2,-1), s.unsqueeze(-1)).squeeze(-1))
    out.scatter_add_(0, src_all.unsqueeze(-1).expand_as(s),
                    -bmm(F_src.transpose(-2,-1), s.unsqueeze(-1)).squeeze(-1))
    return out

apply_delta1(s):          ℝ^{M×d} → ℝ^{F×d}
    out = zeros(F, d)
    for each face f (index f_idx) with path edges [e₁,...,eₗ]:
        # Vectorized over face batch of same template:
        out[f_idx] = sum_i F_ef[f_idx, i] @ s[eᵢ]
        # = (F_ef_batch @ s[face_edges].unsqueeze(-1)).squeeze(-1).sum(1)
    return out

apply_delta1_T(q):        ℝ^{F×d} → ℝ^{M×d}
    out = zeros(M, d)
    for each face f (index f_idx) with path edges [e₁,...,eₗ]:
        out[eᵢ] += F_ef[f_idx, i].T @ q[f_idx]
        # scatter_add over edge indices
    return out

apply_L0(x):  return apply_delta0_T(apply_delta0(x))
apply_L1(s):  return apply_delta0(apply_delta0_T(s)) + apply_delta1_T(apply_delta1(s))

─────────────────────────────────────────────────────────────
STEP D: EDGE COCHAIN INITIALIZATION
─────────────────────────────────────────────────────────────

for each edge e = (u →_r v) in E:
    inp_e = concat(H^(0)[u], H^(0)[v], z^rel_r, delta0_H[e])  # ℝ^{3d + d_r}
    s^(0)_e = MLP_init(inp_e)                                   # ℝ^d
# Vectorized: [M, 3d+d_r] → [M, d]
s0 = MLP_init(concat(H^(0)[src], H^(0)[tgt],
              z^rel[edge_type],
              delta0_H), dim=-1))   # ∈ ℝ^{M × d}

─────────────────────────────────────────────────────────────
STEP E: COUPLED HODGE DIFFUSION
─────────────────────────────────────────────────────────────

x = H^(0).clone()     # ℝ^{N × d}
s = s0.clone()        # ℝ^{M × d}

for k = 1,...,K:
    L0_x = apply_L0(x)    # ℝ^{N × d}
    L1_s = apply_L1(s)    # ℝ^{M × d}
    x = x - eta_0 * L0_x
    s = s - eta_1 * L1_s

x_star = x   # ℝ^{N × d}    approximate harmonic node signals
s_star = s   # ℝ^{M × d}    approximate harmonic edge signals

─────────────────────────────────────────────────────────────
STEP F: READOUT AND CLASSIFICATION
─────────────────────────────────────────────────────────────

# Edge-to-node pooling: gather incident edge signals for each node
edge_aggr_v = zeros(N, d)
# Mean-pool s_star over all edges incident to v (both as src and tgt)
for e = (u→v):
    edge_aggr[v] += s_star[e]
    edge_aggr[u] += s_star[e]
count[v] = number of incident edges for v
edge_aggr_v = edge_aggr_v / count.unsqueeze(-1).clamp(min=1)
# Or use scatter_mean from torch_scatter for efficiency

# Residual readout: concatenate initial + final + edge signals
z_v = concat(x_star[v], H^(0)[v], edge_aggr_v[v])  ∈ ℝ^{3d}
# Only for target node type (e.g., Paper nodes in ACM)

logits = MLP_cls(z_v[target_mask])   ∈ ℝ^{N_target × C}

OUTPUT:
  logits: ℝ^{N_target × C}
  auxiliary: (delta0_H, s0, s_star, x_star, alpha_per_face) for loss computation
```

---

## §9 Loss Function

### 9.1 Full Loss

```
L = L_cls + λ₀ · L_smooth0 + λ₁ · L_smooth1 + λ_c · L_consist + λ_co · L_co
```

### 9.2 Classification Loss

```
L_cls = CrossEntropyLoss(logits[train_mask], y[train_mask])
```

Standard cross-entropy over labeled target nodes.

### 9.3 Node Smoothness Regularizer

```
L_smooth0 = (1/N) · ||δ₀ x*||²_F
           = (1/N) · ||apply_delta0(x_star)||²_F   ∈ ℝ
```

Encourages the final node embeddings to be "sheaf-smooth" — nodes connected by edges
that have similar signals under the learned restriction maps. This is the standard
Dirichlet energy for sheaves.

**Default weight:** λ₀ = 0.0 (start without, tune if over-fitting).

### 9.4 Edge Smoothness Regularizer

```
L_smooth1 = (1/M) · ||δ₁ s*||²_F
           = (1/M) · ||apply_delta1(s_star)||²_F   ∈ ℝ
```

Encourages the final edge embeddings to map consistently into face stalks.
Equivalent to minimizing the up-Laplacian energy of s*.

**Default weight:** λ₁ = 0.001.

### 9.5 Edge-Node Consistency

```
L_consist = (1/M) · ||s* - δ₀ x*||²_F
           = (1/M) · ||s_star - apply_delta0(x_star)||²_F   ∈ ℝ
```

Encourages the final edge signals to be "consistent" with the final node signals
(i.e., s* ≈ δ₀ x* means edge signals are approximately the coboundary of node signals).

This provides a soft coupling between the two levels of signals, preventing them from
diverging arbitrarily during diffusion.

**Default weight:** λ_c = 0.01.

### 9.6 Coboundary Regularizer (replaces SVD hard constraint)

```
L_co = (1/F) · Σ_{f ∈ X²} ||(δ₁ δ₀ H^(0))_f||²
     = (1/F) · ||apply_delta1(apply_delta0(H0))||²_F   ∈ ℝ
```

Encourages δ₁ ∘ δ₀ ≈ 0 — the coboundary condition of the chain complex.
Evaluated at the INITIAL node embeddings H^(0) (not x* after diffusion, to avoid
gradient conflicts with the diffusion objective).

**Default weight:** λ_co = 0.01. Tune: increase if the Hodge decomposition is important
for your analysis; decrease if it slows convergence on small datasets.

### 9.7 Loss Weights (Recommended Starting Point)

| Weight | Default | Purpose |
|--------|---------|---------|
| λ₀ (node smooth) | 0.0 | Enable for large homophilic components |
| λ₁ (edge smooth) | 0.001 | Small; encourages face consistency |
| λ_c (consistency) | 0.01 | Couples node and edge diffusion |
| λ_co (coboundary) | 0.01 | Enforces chain complex structure |

These are starting points. Run ablation by setting each to 0 independently.

---

## §10 Theoretical Properties

### 10.1 Proposition 1: L₀ is Symmetric Positive Semi-Definite

**Statement:** For any sheaf maps F^src, F^tgt, L₀ = δ₀ᵀδ₀ is symmetric PSD.

**Proof:**
- Symmetry: (δ₀ᵀδ₀)ᵀ = (δ₀)ᵀ(δ₀ᵀ)ᵀ = δ₀ᵀδ₀. ✓
- PSD: ∀x, xᵀL₀x = xᵀδ₀ᵀδ₀x = ||δ₀x||² ≥ 0. ✓

**Corollary:** All eigenvalues of L₀ are real and non-negative. The Euler update
x ← x - η L₀ x = (I - η L₀) x contracts all eigenvectors with eigenvalue > 0
(provided η < 2/λ_max), preserving those in ker(L₀). □

### 10.2 Proposition 2: Diffusion Converges to Harmonic Projection

**Statement:** For η ∈ (0, 2/λ_max(L₀)), the K-step Euler iterate satisfies:

```
x^(K) = (I - η L₀)^K H^(0) →_{K→∞} proj_{ker(L₀)} H^(0)
```

**Proof sketch:** Spectral decomposition L₀ = UΛUᵀ. Then (I - ηL₀) = U(I-ηΛ)Uᵀ.
For eigenvalue λᵢ > 0: |1 - η λᵢ| < 1 (by choice of η), so (1-η λᵢ)^K → 0.
For eigenvalue λᵢ = 0 (harmonic): (1-0)^K = 1. Thus x^(K) → U[:,λ=0] U[:,λ=0]ᵀ H^(0)
= proj_{ker(L₀)} H^(0). □

**Qualification for finite K:** At K=5–8, x^(K) is a smoothed version of H^(0) with
damped non-harmonic components. Full convergence is not needed — the diffusion provides
a learned interpolation between H^(0) and proj_{ker(L₀)} H^(0), controlled by K and η.

### 10.3 Proposition 3: Anti-Over-Smoothing

**Statement (capacity claim):** If d ≥ C (stalk dimension ≥ number of classes) and
orthogonal restriction maps are used, there exist map configurations such that ker(L₀)
contains C distinct class-discriminative directions {μ₁,...,μ_C} with ||μᵢ - μⱼ|| > 0
for i ≠ j.

**Proof sketch:** For each node pair (u, v) connected by edge e:
ker(L₀) requires F^tgt_{v▷e} x_v = F^src_{u▷e} x_u.
This is satisfiable simultaneously for all edges if the sheaf cohomology H⁰(X;F) has
dimension ≥ C, which holds when the sheaf has sufficiently many "globally consistent
sections." With orthogonal maps and d ≥ C, H⁰ has dimension d − rank(L₀), which can
be ≥ C by choosing restriction maps appropriately. □

**Important qualifier:** This is an existence statement. The actual learned maps may or
may not achieve this configuration. The optimization objective (L_cls) guides the model
toward class-discriminative harmonic spaces, but convergence to the ideal configuration
is not guaranteed. In practice, the benefit is empirical: over-smoothing is significantly
reduced compared to standard GNNs on heterophilic graphs (as demonstrated by CSNN).

### 10.4 Proposition 4: Forman-Ricci Curvature Improvement

**Statement:** For any edge e ∈ E in the original HIN, its Forman-Ricci curvature in the
augmented cellular complex X is strictly greater than in the base graph G:

```
Ric_X(e) ≥ Ric_G(e) + |{f ∈ X² : e ∈ ∂f}| · (positive face weight)
```

**Background:** Forman's combinatorial Ricci curvature for CW complexes (Forman 2003)
generalizes the graph Ricci curvature. For a 1-cell e in a CW complex:

```
Ric(e) = w(e) [Σ_{v∈∂e} w(v)/w(e) - Σ_{e'≠e: ∂e'∩∂e≠∅} w(e')/sqrt(w(e)w(e'))
              + Σ_{f: e∈∂f} w(f)/w(e)]
```

The term Σ_{f: e∈∂f} w(f)/w(e) > 0 whenever e is in the boundary of any 2-cell f.
This term is absent in the base graph (which has no 2-cells). Adding 2-cells (meta-path
instances) increases the curvature of all edges participating in at least one face. □

**Consequence for over-squashing:** Higher Forman-Ricci curvature on edge e implies
better long-range information flow across e (Topping et al. 2022, Di Giovanni et al. 2023).
By lifting to the cellular complex, we improve curvature without graph rewiring.

**Qualifier:** The improvement applies to edges participating in faces (i.e., edges in E
that appear in at least one meta-path instance). Isolated edges (not in any meta-path)
have unchanged curvature.

### 10.5 Proposition 5: Axiomatic Heterophily Metrics (Summary)

The five typed heterophily metrics TALH, TASH, TAFH, TACH, TLI satisfy five axiomatic
properties (proofs deferred to the companion metrics paper):

| Property | Meaning |
|----------|---------|
| **Max** | Score = +1 iff all edges have perfect atom agreement between endpoints |
| **Min** | Score = −1 iff all edges have zero atom agreement between endpoints |
| **ACB** (Asymptotic Constant Baseline) | Score concentrates at 0 under the typed null model (random graph with same degree sequence) |
| **ECT** (Empty Class Tolerance) | Adding an empty class leaves score unchanged |
| **Mon** (Monotonicity) | Adding an edge between same-class nodes increases score; adding between different-class nodes decreases it |

No existing heterogeneous graph heterophily metric satisfies all five simultaneously.
H²S-NN uses these as auxiliary features (Phase 0.4) to make the restriction maps
"heterophily-aware" from the start.

---

## §11 Implementation Notes (PyTorch + PyG)

### 11.1 PyG HeteroData Format

ACM, DBLP, IMDB are provided in PyG as `torch_geometric.data.HeteroData`:

```python
from torch_geometric.datasets import ACM, DBLP, IMDB

# ACM example:
data = ACM(root='data/ACM')[0]
data['paper'].x                      # [3025, 1870]
data['author'].x                     # [5835, 1870] (feature-aligned)
data['subject'].x                    # [56, 1870]
data[('paper','to','author')].edge_index   # [2, 9747]
data[('author','to','paper')].edge_index   # [2, 9747]
data[('paper','to','subject')].edge_index  # [2, 3025]

# Labels are on target type:
data['paper'].y         # [3025] class labels
data['paper'].train_mask, val_mask, test_mask
```

**Flattening to H²S-NN format:**
```python
def hetero_to_flat(data: HeteroData, node_types: list, rel_map: dict):
    """Convert HeteroData to flat tensors with global node indexing."""
    # 1. Assign global node offsets
    offset = {}; n = 0
    node_features = []
    node_type_idx = []
    for τ in node_types:
        offset[τ] = n
        n += data[τ].num_nodes
        node_features.append(data[τ].x)
        node_type_idx.append(torch.full((data[τ].num_nodes,), τ_idx, dtype=torch.long))
    x_all = torch.cat(node_features, dim=0)      # [N, d_raw_max] (pad if needed)
    node_type = torch.cat(node_type_idx, dim=0)  # [N]

    # 2. Flatten edge index with global offsets
    edge_index_list = []
    edge_type_list = []
    for (τ_src, r, τ_tgt), r_idx in rel_map.items():
        ei = data[(τ_src, r, τ_tgt)].edge_index.clone()
        ei[0] += offset[τ_src]
        ei[1] += offset[τ_tgt]
        edge_index_list.append(ei)
        edge_type_list.append(torch.full((ei.shape[1],), r_idx, dtype=torch.long))
    edge_index = torch.cat(edge_index_list, dim=1)  # [2, M]
    edge_type = torch.cat(edge_type_list, dim=0)    # [M]

    return x_all, node_type, edge_index, edge_type, offset
```

### 11.2 Meta-path Enumeration

For length-2 meta-paths (most common), use sparse matrix multiplication:

```python
def enumerate_length2_metapath(edge_index_1, edge_index_2, N):
    """
    Enumerate all instances of A→B→C meta-path.
    edge_index_1: [2, M1] edges of type r1 (A→B)
    edge_index_2: [2, M2] edges of type r2 (B→C)
    Returns: face_nodes [F, 3], face_edges [F, 2] (indices into edge_index_1/2)
    """
    # Build adjacency: for each B node, find (A predecessor, C successor) pairs
    from collections import defaultdict
    in_dict = defaultdict(list)   # B → [(A, edge_idx1)]
    out_dict = defaultdict(list)  # B → [(C, edge_idx2)]
    for idx, (a, b) in enumerate(edge_index_1.T.tolist()):
        in_dict[b].append((a, idx))
    for idx, (b, c) in enumerate(edge_index_2.T.tolist()):
        out_dict[b].append((c, idx))

    face_nodes, face_edges = [], []
    for b in set(in_dict.keys()) & set(out_dict.keys()):
        for (a, e1) in in_dict[b]:
            for (c, e2) in out_dict[b]:
                if a != c:  # optionally filter self-loops
                    face_nodes.append([a, b, c])
                    face_edges.append([e1, e2])  # e1 in edge_index_1, e2 in edge_index_2
    return torch.tensor(face_nodes), torch.tensor(face_edges)
```

For APTPA (length 4), use the same approach iteratively (4 steps). Apply instance
sampling (max K_face per anchor) to keep |X²| ≤ 10x|V|.

### 11.3 Vectorized Coboundary Operations

**Key pattern: batch matrix multiply + scatter_add (no loops)**

```python
import torch
from torch_scatter import scatter_add

def apply_delta0(x, F_src, F_tgt, src_idx, tgt_idx):
    """
    x:        [N, d]
    F_src:    [M, d, d]  — source restriction maps
    F_tgt:    [M, d, d]  — target restriction maps
    src_idx:  [M]        — source node indices
    tgt_idx:  [M]        — target node indices
    returns:  [M, d]
    """
    x_src = x[src_idx]   # [M, d]
    x_tgt = x[tgt_idx]   # [M, d]
    # bmm: [M, d, d] @ [M, d, 1] → [M, d]
    out_tgt = torch.bmm(F_tgt, x_tgt.unsqueeze(-1)).squeeze(-1)  # [M, d]
    out_src = torch.bmm(F_src, x_src.unsqueeze(-1)).squeeze(-1)  # [M, d]
    return out_tgt - out_src


def apply_delta0_T(s, F_src, F_tgt, src_idx, tgt_idx, N):
    """
    s:        [M, d]
    returns:  [N, d]
    """
    # (F_tgt)^T @ s  for target nodes
    tgt_contrib = torch.bmm(F_tgt.transpose(-2,-1), s.unsqueeze(-1)).squeeze(-1)  # [M, d]
    # -(F_src)^T @ s  for source nodes
    src_contrib = torch.bmm(F_src.transpose(-2,-1), s.unsqueeze(-1)).squeeze(-1)  # [M, d]
    out = torch.zeros(N, s.shape[-1], device=s.device)
    out = scatter_add(tgt_contrib, tgt_idx, out=out, dim=0)
    out = scatter_add(-src_contrib, src_idx, out=out, dim=0)
    return out
```

### 11.4 LSTM Batching Across Faces

Group faces by template (same length). For each template of length ℓ:

```python
# Suppose template π has B_f instances, each with ℓ path edges
# node_seqs: [B_f, ℓ+1]  — node indices along each path
# edge_seqs: [B_f, ℓ]    — edge indices along each path
# edge_types: [B_f, ℓ]   — relation types

src_nodes = node_seqs[:, :-1]   # [B_f, ℓ]
tgt_nodes = node_seqs[:, 1:]    # [B_f, ℓ]
rel_embs = rel_emb_table[edge_types]  # [B_f, ℓ, d_r]

inp = torch.cat([H[src_nodes],   # [B_f, ℓ, d]
                 H[tgt_nodes],   # [B_f, ℓ, d]
                 rel_embs], -1)  # [B_f, ℓ, 2d+d_r]

tokens = mlp_g(inp)              # [B_f, ℓ, d_h]
h, _ = lstm(tokens)              # [B_f, ℓ, d_h]
alpha = softmax((h @ a_vec) / sqrt(d_h), dim=1)  # [B_f, ℓ]

F_ef = mlp_F(h).view(B_f, ℓ, d, d)  # [B_f, ℓ, d, d]
```

### 11.5 Reuse from CSNN Baseline

| Baseline file | What to reuse | How |
|--------------|--------------|-----|
| `models/orthogonal.py` | `Orthogonal(d, orthogonal_map)` | Optionally parameterize F^src, F^tgt as scaled orthogonal matrices. Prevents rank collapse. |
| `models/sheaf_utils.py` | `batched_sym_matrix_pow()` | If Laplacian normalization needed |
| `utils/exp_utils.py` | `get_optimizer()`, `get_scheduler()`, `get_scores()` | Reuse directly |
| `utils/data_utils.py` | `order_edge_index()` | Use for consistent edge ordering |
| `utils/cosine_scheduler.py` | `get_cosine_schedule_with_warmup()` | Warmup scheduler |
| `exp/run.py` | `Experiment` class skeleton | Extend for HeteroData; keep W&B/fold structure |
| `exp/parser.py` | Argument parsing pattern | Add H²S-NN-specific args |

**Key reuse example (orthogonal restriction maps):**
```python
from models.orthogonal import Orthogonal

class NodeToEdgeSheaf(nn.Module):
    def __init__(self, d, d_r, num_relations, d_hidden, use_orthogonal=True):
        super().__init__()
        self.use_orthogonal = use_orthogonal
        # Per-relation MLPs for parameters
        self.mlp_src = nn.ModuleList([nn.Sequential(
            nn.Linear(2*d + d_r, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d*d if not use_orthogonal else d*(d-1)//2 + 1)
        ) for _ in range(num_relations)])
        self.mlp_tgt = nn.ModuleList([...])
        if use_orthogonal:
            self.orth = Orthogonal(d, orthogonal_map='householder')
    ...
```

---

## §12 Module Design

### 12.1 Directory Layout

```
h2snn/
├── __init__.py
├── preprocessing.py        # Phase 0: label prop, heterophily, complex construction
├── node_encoder.py         # Type-specific linear encoder + relation embeddings
├── sheaf_maps.py           # NodeToEdgeSheaf (Step A) + EdgeToFaceSheaf (Step B)
├── coboundary_ops.py       # apply_delta0, apply_delta0_T, apply_delta1, apply_delta1_T
│                           # apply_L0, apply_L1 (all matrix-free)
├── diffusion.py            # HodgeDiffusion: Euler + optional ODE
├── readout.py              # EdgePooling + NodeClassifier (MLP_cls)
├── losses.py               # L_cls, L_smooth0, L_smooth1, L_consist, L_co
├── h2snn_model.py          # H2SNN main nn.Module (assembles all modules)
├── data_prep.py            # hetero_to_flat(), enumerate_metapath(), FaceDataset
└── config.py               # H2SNNConfig dataclass (all hyperparameters)
```

### 12.2 Key Module Interfaces

```python
# h2snn/config.py
@dataclass
class H2SNNConfig:
    d: int = 128                    # stalk dimension
    d_r: int = 32                   # relation embedding dim
    d_h: int = 128                  # LSTM hidden dim
    d_hidden: int = 256             # MLP hidden dim
    K: int = 5                      # Euler diffusion steps
    eta_0: float = 0.1              # node diffusion step
    eta_1: float = 0.1              # edge diffusion step
    learnable_eta: bool = True      # learn step sizes
    use_orthogonal: bool = True     # orthogonal F^src, F^tgt
    num_relations: int = 4          # |R|
    num_node_types: int = 3         # |T_V|
    num_classes: int = 3            # C
    metapath_templates: list = None # e.g., [('PAP', 2), ('PSP', 2)]
    lambda_0: float = 0.0           # node smoothness weight
    lambda_1: float = 0.001         # edge smoothness weight
    lambda_c: float = 0.01          # consistency weight
    lambda_co: float = 0.01         # coboundary weight
    dropout: float = 0.5
    face_sample_k: int = 50         # max face instances per anchor for long metapaths


# h2snn/sheaf_maps.py
class NodeToEdgeSheaf(nn.Module):
    """Learns F^src_{u▷e} and F^tgt_{v▷e} for all edges.
    Input:  H [N,d], edge_index [2,M], edge_type [M], rel_emb [|R|, d_r]
    Output: F_src [M,d,d], F_tgt [M,d,d], delta0_H [M,d]
    """

class EdgeToFaceSheaf(nn.Module):
    """Learns F_{e▷f} via LSTM + MLP_F for all face-edge pairs.
    Input:  H [N,d], face_node_seqs [F,ℓ+1], face_edge_seqs [F,ℓ], edge_type [F,ℓ], rel_emb
    Output: F_ef [F,ℓ,d,d], alpha [F,ℓ]
    (batched by template type for efficiency)
    """


# h2snn/coboundary_ops.py
class CoboundaryOps:
    """Stateless matrix-free operators. Pass maps as arguments."""
    @staticmethod
    def delta0(x, F_src, F_tgt, src_idx, tgt_idx) -> Tensor:    # [M, d]
    @staticmethod
    def delta0_T(s, F_src, F_tgt, src_idx, tgt_idx, N) -> Tensor: # [N, d]
    @staticmethod
    def delta1(s, F_ef, face_edge_idx, face_mask) -> Tensor:       # [F, d]
    @staticmethod
    def delta1_T(q, F_ef, face_edge_idx, M) -> Tensor:             # [M, d]
    @staticmethod
    def L0(x, ...) -> Tensor:   # = delta0_T(delta0(x))
    @staticmethod
    def L1(s, ...) -> Tensor:   # = delta0(delta0_T(s)) + delta1_T(delta1(s))


# h2snn/diffusion.py
class HodgeDiffusion(nn.Module):
    """K-step Euler or Neural ODE diffusion.
    Input:  x0 [N,d], s0 [M,d], CoboundaryOps, all map tensors
    Output: x_star [N,d], s_star [M,d]
    """


# h2snn/h2snn_model.py
class H2SNN(nn.Module):
    """Main model. Assembles all modules.
    Input:  H2SNNData (flat tensors + face structures)
    Output: logits [N_target, C]
    """
    def __init__(self, config: H2SNNConfig): ...
    def forward(self, data) -> tuple[Tensor, dict]: ...
    # Returns (logits, aux_dict) where aux_dict has delta0_H, s0, x_star, s_star, alpha
```

### 12.3 Training Loop Sketch

```python
model = H2SNN(config)
optimizer = get_optimizer(model, args)  # from utils/exp_utils.py
scheduler = get_scheduler(optimizer, args)

for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()

    logits, aux = model(data)
    L_cls = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    L_s0  = config.lambda_0 * (aux['delta0_x_star']**2).mean()
    L_s1  = config.lambda_1 * (aux['delta1_s_star']**2).mean()
    L_c   = config.lambda_c  * ((aux['s_star'] - aux['delta0_x_star'])**2).mean()
    L_co  = config.lambda_co * (aux['delta1_delta0_H0']**2).mean()
    loss  = L_cls + L_s0 + L_s1 + L_c + L_co

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    # Evaluate every N epochs (standard pattern from exp/run.py)
```

---

## §13 Corrections Table from Original Specs

| # | Location | Original Claim | Error Type | Corrected Version |
|---|----------|---------------|------------|-------------------|
| 1 | Spec 2, Step B | M_f = Σᵢ αᵢ (F^tgt)ᵀ enforces δ₁∘δ₀=0 | **Math error**: ignores F^src terms | Use M_f^correct = stacking of per-node net maps N_f(v); or (preferred) drop SVD, use MLP_F + L_co penalty |
| 2 | Spec 2, Step E | Use torchdiffeq dopri5 Neural ODE | **Missing dependency**: torchdiffeq not in requirements.txt | Use K-step Euler as default; add torchdiffeq as optional |
| 3 | Both specs | 2-cells are meta-path instances (paths) | **Topology gap**: paths aren't 2-disks without closure | Add closure edge ē_f = (vₗ→v₀) with F_{ē_f▷f}=0; closure cycle is topologically valid |
| 4 | Spec 2, Prop 1 | δ₁∘δ₀=0 is enforced identically | **False**: only partial enforcement by Spec 2's construction | "δ₁∘δ₀≈0 enforced via L_co penalty" |
| 5 | Spec 2, Prop 2 | ker(L₀) linearly separable when d₀≥⌈C/2⌉ | **Too strong**: existence vs optimization guarantee | "Capacity claim: there EXIST map configurations with d₀≥C class-discriminative harmonic vectors" |
| 6 | Spec 2, Prop 2 | ODE converges to harmonic projection | **Imprecise for finite T** | "At finite K, x^(K) is a smoothed version of H^(0) with damped non-harmonic components" |
| 7 | Spec 2, Prop 4 | Forman-Ricci curvature claim | **Correct but needs qualifier** | Add: "improvement only applies to edges participating in at least one meta-path face" |
| 8 | Spec 2, Phase 0 | Complex preprocessing pipeline | **Missing**: no mention of face sampling for long metapaths | Add face sampling: max K_face=50 instances per anchor node for APTPA and other length-4+ paths |
| 9 | Spec 1 | STAGE encoder required | **Scope creep**: STAGE is Year 2 | Replace with type-specific linear projection W_τ + LayerNorm (explicit in Spec 2 but not stated clearly in Spec 1) |
| 10 | Both specs | Inconsistent notation for coboundary | **Minor**: η(e,f) orientation signs not defined | All η(e,f)=+1 for directed path faces; formally stated in §2.4 |

---

## Appendix A: Dataset-Specific Notes

### A.1 ACM

```
Node types:    Paper (3025), Author (5835), Subject (56)
Relations:     PA (paper→author), AP (author→paper), PS (paper→subject), SP (subject→paper)
Meta-paths:    PAP = (paper →PA→ author →AP→ paper)  [3025 anchors, ~18k instances]
               PSP = (paper →PS→ subject →SP→ paper) [3025 anchors, ~3k instances]
Labels:        3 classes on Paper nodes (Database, Wireless, Data Mining)
Splits:        train/val/test = 600/300/2125 (per standard benchmark)
```

**Notes:**
- Features are 1870-dim bag-of-words for all node types (same feature space)
- For PSP, subject nodes have few connections → faces are fewer than PAP
- λ_close = 1e-6 is safe since subject nodes already have small degrees

### A.2 DBLP

```
Node types:    Author (4057), Paper (14328), Term (7723), Venue (20)
Relations:     AP, PA, PT, TP, PV, VP
Meta-paths:    APA = (author →AP→ paper →PA→ author)      [~50k instances]
               APTPA = (author→paper→term→paper→author)   [very large, SAMPLE]
Labels:        4 classes on Author nodes (database, data mining, machine learning, info retrieval)
Splits:        train/val/test = 800/400/2857 (standard)
```

**Notes:**
- For APTPA: enumerate but sample max 50 instances per anchor author
- Term nodes have 7723 types → rich structural diversity in faces
- Recommend running APA only first to validate model, then add APTPA

### A.3 IMDB

```
Node types:    Movie (4278), Actor (5257), Director (2081)
Relations:     MA, AM, MD, DM
Meta-paths:    MAM = (movie →MA→ actor →AM→ movie)        [~9k instances]
               MDM = (movie →MD→ director →DM→ movie)     [~4k instances]
Labels:        3 classes on Movie nodes (Action, Comedy, Drama)
Splits:        train/val/test = 400/400/3478 (standard)
```

**Notes:**
- IMDB has relatively low feature quality (keyword bag-of-words)
- The heterophily metrics (Phase 0.4) are especially important here
- MAM typically has higher predictive power than MDM

---

## Appendix B: Complexity Analysis

| Phase/Step | Operation | Time Complexity | Space |
|------------|-----------|----------------|-------|
| Phase 0 | Label propagation (3 hops) | O(3 \|E\| C) | O(NC) |
| Phase 0 | Structural descriptors | O(\|E\| \|R\|) | O(N \|R\|) |
| Phase 0 | Meta-path enumeration (length 2) | O(\|E_r1\| · max_deg) | O(F) |
| Phase 1 | Node encoder | O(N d d_in_max) | O(N d) |
| Step A | Node-to-edge MLPs | O(M (2d+d_r) d_hidden + M d²) | O(M d²) |
| Step A | δ₀ computation | O(M d²) | O(M d) |
| Step B | LSTM per template | O(F ℓ d_h²) | O(F ℓ d²) |
| Step C | apply_L0 (one call) | O(M d²) | O(M d) |
| Step C | apply_L1 (one call) | O((M + F ℓ) d²) | O(M d) |
| Step E | K-step Euler | O(K (M + F ℓ) d²) | O(M d) |
| Step F | Edge pooling + MLP | O(M d + N_target d C) | O(N d) |
| **Total** | **per forward pass** | **O(K (M + F ℓ) d²)** | **O((M + F) d²)** |

For ACM (M≈9k, F≈21k, ℓ=2, d=128, K=5):
- Main term: K (M + F ℓ) d² ≈ 5 × 51k × 16k ≈ 4 × 10⁹ FLOPs
- But most is sparse bmm — actual wall time is ~0.1–0.5s/epoch (comparable to CSNN)

**Memory-critical:** The face structures F_ef require O(F ℓ d²) storage.
For ACM F=21k, ℓ=2, d=128: ~21k × 2 × 128² ≈ 690M floats ≈ 2.7 GB at float32.
**Solution:** Compute F_ef on-the-fly in apply_delta1 (recompute from LSTM each time)
rather than storing. This trades 2.7GB memory for ~2× compute in diffusion loop.

---

*End of H²S-NN Corrected Algorithm Document*
*Last updated: 2026-04-07*
*This document is the single source of truth. When in doubt, this file takes precedence over H2S-NN_implementation_spec.md and H2SNN_implementation.md.*
