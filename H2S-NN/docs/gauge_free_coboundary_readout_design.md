# Gauge-Free Coboundary Readout for Sheaf Graph Classification

## Purpose

This document explains the proposed global readout method for graph classification
with sheaf neural networks.

The goal is not to add many unrelated features. The goal is to solve one specific
readout problem:

**Current SHEAFPOOL performs gauge-compatible readout by gauge-fixing node stalk
embeddings, then applying invariant attention pooling. This is elegant, but it still
compresses the graph through a single attention-weighted barycenter of aligned node
representatives and does not directly read the sheaf coboundary, where local
compatibility is measured.**

The proposed method is:

**Gauge-Free Coboundary Readout**

It reads the graph through the two natural sheaf objects:

```text
C^0(G; F) --delta_0--> C^1(G; F)
```

That means:

- read the node section H in C^0,
- read the edge disagreement delta_0 H in C^1,
- do both without choosing an arbitrary global stalk frame,
- aggregate with permutation-invariant sums.

---

## 1. What SHEAFPOOL Does

SHEAFPOOL starts from node stalk embeddings:

```text
H_i in R^{d x C}
```

where:

- d is the stalk dimension,
- C is the channel dimension,
- i is a node.

Because sheaf node states are only defined up to local basis changes, SHEAFPOOL first
canonicalizes them:

```text
H_i -> Z_i -> Z_tilde_i
```

The steps are:

1. Per-node whitening.
2. Orthogonal Procrustes alignment to a shared learnable anchor.
3. Gauge-invariant attention scores from channel-wise energies.
4. Attention-weighted pooling:

```text
P_G = sum_i alpha_i Z_tilde_i
```

5. Final graph descriptor from channel-wise energies.

Important correction:

SHEAFPOOL is not plain sum pooling. It is **invariant attention-weighted pooling**.
The attention scores are computed from invariant descriptors. The limitation is not
that SHEAFPOOL ignores invariance. The limitation is that, after gauge fixing, it
still compresses the node-stalk distribution into a single weighted barycenter.

Second important correction:

The node projector used in GFCR is closely related to SHEAFPOOL whitening. In the
ideal full-rank, unregularized case:

```text
Z_i = (H_i H_i^T)^(-1/2) H_i
```

so:

```text
Z_i^T Z_i = H_i^T (H_i H_i^T)^(-1) H_i = P_i
```

Therefore, the proposed node descriptor P_i is **the full Gram/projector of the
whitened SHEAFPOOL representation**. The difference is not that GFCR invents a
completely unrelated node invariant. The difference is how the invariant is used:

- SHEAFPOOL uses `diag(Z_i^T Z_i)` to compute invariant attention weights.
- SHEAFPOOL then pools aligned tensors `Z_tilde_i` as the values.
- GFCR uses the full projector `P_i = Z_i^T Z_i` as the readout object itself.
- GFCR applies a learned `phi_0` before summation, rather than forming one
  attention-weighted barycenter of aligned representatives.

---

## 2. The Actual Gap

### Gap 1: SHEAFPOOL computes only a partial quotient statistic for attention

In a sheaf, a node embedding H_i and a locally reparameterized version G_i H_i
can represent the same geometric state:

```text
H_i ~ G_i H_i
```

where G_i is an allowed local stalk basis change.

The true coordinate-free object is the orbit:

```text
[H_i] = {G_i H_i : G_i in G}
```

This is called a quotient object: we identify all coordinate versions that differ
only by a valid local basis change.

SHEAFPOOL partially uses this idea. Whitening produces a representation whose full
Gram:

```text
Z_i^T Z_i
```

is the same row-space projector P_i used by GFCR in the ideal full-rank case.
However, SHEAFPOOL only uses the diagonal of this invariant Gram matrix for
attention weights:

```text
u_i = diag(Z_i^T Z_i)
```

It then still chooses an aligned representative `Z_tilde_i` through Procrustes and
pools those aligned representatives as values.

So the sharper gap is:

**SHEAFPOOL uses a low-dimensional invariant descriptor to choose attention
weights, but its graph value is still an attention-weighted barycenter of
gauge-fixed node representatives.**

Gauge fixing can be useful, but it has risks:

- it depends on full-rank or well-conditioned embeddings,
- Procrustes alignment can be unstable with repeated or near-repeated singular values,
- a learned anchor may impose a non-intrinsic global coordinate convention,
- the exact invariance proof relies on ideal assumptions such as epsilon = 0.

### Gap 2: Attention-weighted barycenters are not distribution-injective

SHEAFPOOL pools:

```text
P_G = sum_i alpha_i Z_tilde_i
```

This is a weighted barycenter. Even with invariant attention weights, many different
node-stalk multisets can map to the same P_G.

Example:

```text
0.5 A + 0.5 B = C
```

where C = (A + B) / 2.

So a graph with two node populations A and B can collapse to the same pooled tensor
as a graph with one averaged population C.

This is exactly the kind of readout bottleneck discussed by adaptive readout and
kernel readout papers: the encoder can be strong, but the readout can still lose
the distributional structure needed for graph classification.

### Gap 3: SHEAFPOOL reads node states, not explicit sheaf disagreement

The core sheaf compatibility object is the coboundary:

```text
(delta_0 H)_e = F_{v -> e} H_v - F_{u -> e} H_u
```

This lives in the edge stalk. It measures whether neighboring node states agree
after being restricted to the same edge space.

SHEAFPOOL uses node embeddings after sheaf message passing, so coboundary information
can influence H_i indirectly. But the final readout does not explicitly read:

```text
{(delta_0 H)_e : e in E}
```

Therefore the classifier is not directly told:

- how compatible the learned section is,
- where disagreement/frustration remains,
- how disagreement is distributed across edges,
- whether the graph is globally easy or difficult to glue.

---

## 3. Research Question

The proposed research question is:

**Can we replace SHEAFPOOL's gauge-fixing attention barycenter with a gauge-free,
permutation-invariant readout of both the sheaf section H and its coboundary
delta_0 H?**

More compactly:

**Can graph classification use a quotient-space readout of C^0 and delta C^0
instead of gauge-fixing and pooling only C^0?**

---

## 4. Proposed Method

### Name

Gauge-Free Coboundary Readout, abbreviated:

```text
GFCR
```

### Inputs

The readout receives:

```text
H       : node stalk embeddings, shape [N, d, C]
edge_index : graph edges, shape [2, E]
F_src   : source restriction maps, shape [E, d_e, d]
F_tgt   : target restriction maps, shape [E, d_e, d]
batch   : node-to-graph assignment, shape [N]
edge_batch : edge-to-graph assignment, shape [E]
```

For the simplest case:

```text
d_e = d
```

but the formulation allows edge stalk dimension d_e to differ from node stalk
dimension d.

### Step 1. Build node quotient descriptors

For each node i, compute:

```text
P_i = H_i^T (H_i H_i^T)^+ H_i
```

where + denotes the Moore-Penrose pseudoinverse.

Shape:

```text
H_i : [d, C]
P_i : [C, C]
```

Interpretation:

P_i is the orthogonal projector onto the row space of H_i. It represents the node
state modulo arbitrary invertible changes of stalk basis.

If:

```text
H_i' = G_i H_i
```

with G_i invertible, then H_i' has the same row space as H_i. Therefore:

```text
P_i' = P_i
```

This gives GL-gauge invariance for node descriptors.

### Step 2. Build edge coboundary descriptors

For each edge e = (u, v), compute the sheaf disagreement:

```text
D_e = F_tgt[e] H_v - F_src[e] H_u
```

Shape:

```text
D_e : [d_e, C]
```

Then compute the edge quotient descriptor:

```text
Q_e = D_e^T (D_e D_e^T)^+ D_e
```

Shape:

```text
Q_e : [C, C]
```

Interpretation:

Q_e is a coordinate-free descriptor of the local disagreement subspace in the edge
stalk. It is invariant to edge-stalk basis changes.

If the edge stalk basis changes by G_e, then:

```text
D_e' = G_e D_e
```

and the row space is unchanged, so:

```text
Q_e' = Q_e
```

### Step 3. Embed node and edge descriptors

Flatten or vectorize the symmetric C x C descriptors. To avoid duplicate entries,
use the upper triangular part:

```text
p_i = vech(P_i)
q_e = vech(Q_e)
```

where:

```text
vech(P_i) in R^{C(C+1)/2}
```

Then apply learned descriptor networks:

```text
z_i = phi_0(p_i)
w_e = phi_1(q_e)
```

where:

```text
phi_0 : R^{C(C+1)/2} -> R^h
phi_1 : R^{C(C+1)/2} -> R^h
```

These networks are shared across all nodes and edges.

### Step 4. Permutation-invariant aggregation

Aggregate by graph using unnormalized sums:

```text
r_0(G) = sum_{i in V_G} z_i
r_1(G) = sum_{e in E_G} w_e
```

The sums are permutation invariant.

Use sum rather than softmax attention as the default because graph classification
often depends on counts, multiplicities, and repeated motifs. A normalized weighted
average can hide those.

Optional size normalization can be tested as an ablation:

```text
mean nodes: r_0 / |V_G|
mean edges: r_1 / |E_G|
sum + log size: [r_0, r_1, log |V_G|, log |E_G|]
```

### Step 5. Graph-level classifier

Concatenate and classify:

```text
r_G = concat(r_0(G), r_1(G))
logits_G = rho(r_G)
```

where:

```text
rho : R^{2h} -> R^{num_classes}
```

---

## 5. Why This Is Not Just "Kernel Readout + Sheaf"

Kernel readout papers operate on ordinary Euclidean node embeddings:

```text
h_i in R^C
```

GFCR operates on quotient descriptors of sheaf cochains:

```text
[H_i] in C^0 / gauge
[(delta_0 H)_e] in C^1 / gauge
```

The contribution is not "use a kernel." The contribution is:

**read graph-level sheaf information directly on the quotient space of the section
and its coboundary, without global gauge fixing.**

Kernel or DeepSets variants can be used inside phi_0 and phi_1 later, but the core
object is the quotient-space C^0 and C^1 readout.

---

## 6. Should Gauge Invariance Be Relaxed?

The final graph prediction should remain invariant to any basis change that is truly
a gauge symmetry of the sheaf model.

However, we should not impose a larger gauge group than the architecture actually
has.

### Recommended rule

Do not relax invariance in the final prediction.

Instead:

```text
match the invariant descriptor to the true gauge group
```

Examples:

1. General invertible gauge:

```text
H_i -> G_i H_i, G_i in GL(d)
```

Use row-space projector:

```text
P_i = H_i^T (H_i H_i^T)^+ H_i
```

2. Orthogonal gauge:

```text
H_i -> Q_i H_i, Q_i in O(d)
```

The full Gram matrix is enough:

```text
K_i = H_i^T H_i
```

This preserves more metric information than the GL projector.

3. No true gauge symmetry:

If the architecture fixes a meaningful coordinate basis and local basis changes are
not a valid symmetry, then enforcing gauge invariance may hurt performance. In that
case, use gauge-invariant readout as a regularized variant, not as a hard requirement.

For HETSHEAF/SHEAFPOOL comparison, the relevant critique is about the setting where
local basis changes are considered valid symmetries. In that setting, exact final
gauge invariance is desirable.

---

## 7. Implementation Plan

### Step A. Implement projector utility

Create a function:

```python
def row_space_projector(H, eps=1e-6):
    # H: [B, d, C]
    # returns P: [B, C, C]
```

Numerically stable options:

1. SVD-based projector:

```text
H = U S V^T
P = V_r V_r^T
```

where V_r contains right singular vectors with singular values above a threshold.

2. Regularized inverse:

```text
P_eps = H^T (H H^T + eps I)^{-1} H
```

This is faster and differentiable, but only approximately GL-invariant for finite eps.

Recommended v1:

Use regularized inverse for training stability, and add an invariance diagnostic test.

### Step B. Implement GFCR module

Create a PyTorch module:

```python
class GaugeFreeCoboundaryReadout(nn.Module):
    def __init__(self, channels, hidden_dim, out_dim, mode="gl_projector"):
        ...

    def forward(self, H, edge_index, F_src, F_tgt, batch, edge_batch):
        ...
```

Forward pass:

1. Compute P_i from H.
2. Compute D_e from restriction maps and endpoint node embeddings.
3. Compute Q_e from D_e.
4. Extract upper triangular vectors from P_i and Q_e.
5. Apply phi_0 and phi_1.
6. Scatter-sum node and edge embeddings by graph id.
7. Concatenate and classify.

### Step C. Add baseline modes

Support these modes for ablation:

```text
node_only        : use r_0 only
edge_only        : use r_1 only
node_plus_edge   : use [r_0, r_1]
orthogonal_gram  : use H_i^T H_i and D_e^T D_e
gl_projector     : use row-space projectors
```

### Step D. Add invariance tests

Test 1: node gauge invariance.

1. Generate random invertible G_i for every node.
2. Transform:

```text
H_i' = G_i H_i
```

3. If restriction maps are also transformed consistently:

```text
F_{i -> e}' = G_e F_{i -> e} G_i^{-1}
```

then logits should match.

Test 2: edge gauge invariance.

1. Generate random invertible G_e for every edge.
2. Transform edge-space residuals through transformed restriction maps.
3. Check that Q_e and logits match.

Test 3: permutation invariance.

1. Randomly permute node order and edge order.
2. Reindex edge_index and batch tensors.
3. Check that logits match.

### Step E. Add performance ablations

Compare:

```text
global_mean_pool
SHEAFPOOL-style aligned attention pooling
GFCR node_only
GFCR edge_only
GFCR node_plus_edge
GFCR orthogonal_gram
GFCR gl_projector
```

Main question:

```text
Does explicit quotient-space coboundary readout improve graph classification
without breaking gauge or permutation invariance?
```

---

## 8. Expected Benefits

### Benefit 1: No arbitrary global stalk frame in the value aggregation

GFCR does not align all nodes to a learned anchor before value aggregation. It uses
the full coordinate-free descriptor `P_i = Z_i^T Z_i` as the value object, then maps
that descriptor into a shared graph feature space.

### Benefit 2: Direct sheaf compatibility signal

GFCR reads:

```text
delta_0 H
```

which directly measures local sheaf agreement/disagreement.

### Benefit 3: Better multiset discrimination than barycentric pooling

The DeepSets-style sum:

```text
sum_i phi_0(P_i)
```

is more expressive than a single weighted average over aligned representatives.

### Benefit 4: Clear invariance guarantees

The descriptors are invariant by construction under the selected gauge group, and
the aggregation is permutation invariant by construction.

---

## 9. Risks and How to Handle Them

### Risk 1: Full GL invariance may remove useful metric information

The row-space projector removes scale and anisotropy. If these are predictive and
not true gauge artifacts, performance can drop.

Mitigation:

Run `orthogonal_gram` and `gl_projector` ablations. Use the descriptor that matches
the model's actual restriction-map family.

### Risk 2: Projectors can be high-dimensional

P_i is C x C, so vech(P_i) has C(C+1)/2 dimensions.

Mitigation:

Keep C moderate or use a low-rank MLP bottleneck in phi_0 and phi_1.

### Risk 3: Regularized inverse gives approximate, not exact, GL invariance

Finite eps can break exact invariance.

Mitigation:

Use SVD projector for evaluation/invariance tests, or compare both SVD and
regularized inverse implementations.

### Risk 4: Edge residuals may be noisy early in training

delta_0 H can be unstable before the sheaf maps become meaningful.

Mitigation:

Use residual readout:

```text
r_G = concat(r_0, lambda_edge * r_1)
```

with learnable or scheduled lambda_edge.

This is still not kitchen-sink: it is simply weighting C^0 versus C^1.

---

## 10. Minimal First Version

For the first implementation, keep it simple:

```text
mode = gl_projector
readout = node_plus_edge
aggregation = sum
projector = regularized inverse
classifier = 2-layer MLP
```

Forward pass:

```text
P_i = H_i^T (H_i H_i^T + eps I)^-1 H_i
D_e = F_tgt[e] H_v - F_src[e] H_u
Q_e = D_e^T (D_e D_e^T + eps I)^-1 D_e

r_0 = scatter_sum(phi_0(vech(P_i)), batch)
r_1 = scatter_sum(phi_1(vech(Q_e)), edge_batch)

logits = rho(concat(r_0, r_1))
```

This first version is enough to test the research claim.

---

## 11. One-Sentence Contribution

**We replace SHEAFPOOL's gauge-fixing attention barycenter with a gauge-free
quotient-space readout of both node sections and sheaf coboundaries, giving a
permutation-invariant graph representation that directly measures sheaf-compatible
and sheaf-incompatible structure without imposing a global stalk frame.**
