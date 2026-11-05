# S2Pool: Sheaf-Aware Readout for H2SNN Graph Prediction

## Purpose

`S2Pool` is a graph-level readout module for an H2SNN-style sheaf neural network. Its goal is to produce a graph representation that is:

1. **basis-free / gauge-invariant**, so the graph prediction does not depend on an arbitrary local coordinate basis of each node stalk;
2. **sheaf-aware**, so the readout uses both node-level information and edge-level sheaf disagreement;
3. **more stable than gauge fixing**, avoiding an explicit global basis alignment step;
4. **more expressive than standard weighted-sum pooling**, because it reads out relational inconsistency through the sheaf restrictions, not only node embeddings.

The module takes the final node stalk representations and sheaf restriction maps from an H2SNN model and returns a graph-level prediction.

---

## Motivation

A standard graph-level readout in a sheaf neural network is nontrivial because each node embedding lives in a **local stalk space**. Directly summing node stalk vectors is generally not well-defined unless all node stalks have already been made comparable in a common basis.

A common workaround is:

1. gauge-fix node stalks into a comparable basis;
2. apply a weighted sum or attention pooling over nodes.

This has several problems:

- **Gauge fixing can be unstable.** Small perturbations may change the selected basis or alignment.
- **Weighted-sum pooling can collapse information.** Opposite or incompatible local signals may cancel after summation.
- **Node-only pooling ignores the sheaf relation.** It summarizes node states but does not explicitly read the consistency or disagreement induced by restriction maps on edges.

`S2Pool` addresses these issues by constructing basis-free descriptors for both nodes and edges, then aggregating them into a graph-level representation.

---

## Inputs

Assume a graph \(G = (V, E)\).

For each node \(i \in V\):

- \(H_i \in \mathbb{R}^{d \times r}\): final node stalk representation from H2SNN.
  - Here, \(d\) is the stalk dimension.
  - \(r\) is the number of channels, heads, or latent columns.
  - If the implementation stores node features as a vector, reshape or project them into matrix form before applying S2Pool.

For each directed or oriented edge \(e = (u, v) \in E\):

- \(F_{u \to e}\): restriction map from node \(u\)'s stalk to the edge stalk.
- \(F_{v \to e}\): restriction map from node \(v\)'s stalk to the edge stalk.

The restriction maps must be compatible with the shape of \(H_u\) and \(H_v\), so that:

\[
F_{u \to e} H_u,\quad F_{v \to e} H_v
\]

are both valid edge-stalk representations with the same shape.

---

## Output

`S2Pool` outputs a graph-level prediction:

\[
\hat{y}_G = \rho(\tau_G),
\]

where:

- \(\tau_G\) is the final graph-level pooled representation;
- \(\rho\) is a prediction MLP.

For graph classification, \(\hat{y}_G\) can be logits.

For graph regression, \(\hat{y}_G\) can be a scalar or vector target.

---

## Core Idea

S2Pool constructs two graph summaries:

1. a **node summary** from basis-free node descriptors;
2. an **edge summary** from basis-free sheaf disagreement descriptors.

The final graph embedding concatenates both summaries:

\[
\tau_G = [\tau_0, \tau_1].
\]

Then a final MLP predicts the graph label:

\[
\hat{y}_G = \rho(\tau_G).
\]

---

## Step 1: Convert Node Stalks to Basis-Free Node Descriptors

For each node \(i\), compute the node Gram descriptor:

\[
P_i = H_i^\top (H_i H_i^\top)^+ H_i.
\]

Here:

- \((\cdot)^+\) denotes the Moore-Penrose pseudoinverse;
- \(P_i\) is a basis-free descriptor of the subspace or projection structure represented by \(H_i\).

Intuition:

- \(H_i\) itself is basis-dependent.
- \(P_i\) removes the arbitrary local basis and keeps the intrinsic subspace information.
- Therefore, pooling \(P_i\)-based descriptors is more stable than pooling raw node stalk coordinates.

Then apply a shared node MLP:

\[
z_i = \phi_0(P_i).
\]

where \(\phi_0\) is shared across all nodes.

---

## Step 2: Compute Sheaf Disagreement on Each Edge

For each edge \(e = (u, v)\), compute the transported node representations on the edge stalk:

\[
F_{u \to e}H_u, \quad F_{v \to e}H_v.
\]

Then compute the sheaf disagreement:

\[
D_e = F_{v \to e}H_v - F_{u \to e}H_u.
\]

Intuition:

- If two adjacent node stalks are consistent under the sheaf restriction maps, \(D_e\) should be small.
- If they disagree, \(D_e\) captures relational incompatibility.
- This makes the readout explicitly sensitive to the sheaf structure, not only to node features.

---

## Step 3: Convert Edge Disagreement to Basis-Free Edge Descriptors

For each edge disagreement \(D_e\), compute:

\[
Q_e = D_e^\top (D_eD_e^\top)^+D_e.
\]

Here \(Q_e\) is a basis-free descriptor of edge-level disagreement.

Then apply a shared edge MLP:

\[
w_e = \phi_1(Q_e).
\]

where \(\phi_1\) is shared across all edges.

---

## Step 4: Aggregate Node and Edge Descriptors

Aggregate node descriptors:

\[
\tau_0 = \sum_{i \in V} z_i.
\]

Aggregate edge disagreement descriptors:

\[
\tau_1 = \sum_{e \in E} w_e.
\]

Then concatenate:

\[
\tau_G = [\tau_0, \tau_1].
\]

Optional implementation variants:

- use mean pooling instead of sum pooling for graph-size normalization;
- concatenate sum and mean pooling if graph sizes vary substantially;
- use attention or Set2Set after basis-free descriptors, but not before basis-free conversion.

Recommended default:

\[
\tau_0 = \frac{1}{|V|}\sum_{i \in V} z_i, \quad
\tau_1 = \frac{1}{|E|}\sum_{e \in E} w_e.
\]

Mean pooling is often safer when graph sizes vary.

---

## Step 5: Graph-Level Prediction

Use a final MLP \(\rho\):

\[
\hat{y}_G = \rho(\tau_G).
\]

For classification:

```python
loss = cross_entropy(logits, labels)
```

For regression:

```python
loss = mse_loss(prediction, target)
```

---

## Pseudocode

```python
class S2Pool(nn.Module):
    def __init__(self, node_desc_dim, edge_desc_dim, hidden_dim, out_dim):
        super().__init__()
        self.node_mlp = MLP(node_desc_dim, hidden_dim, hidden_dim)
        self.edge_mlp = MLP(edge_desc_dim, hidden_dim, hidden_dim)
        self.pred_mlp = MLP(2 * hidden_dim, hidden_dim, out_dim)

    def projection_descriptor(self, X, eps=1e-6):
        """
        Compute basis-free descriptor:
            P = X.T @ pinv(X @ X.T) @ X

        X shape: [d, r]
        P shape: [r, r]
        Return flattened P or its upper triangular entries.
        """
        gram = X @ X.T
        gram_pinv = torch.linalg.pinv(gram, rtol=eps)
        P = X.T @ gram_pinv @ X
        return P

    def forward(self, H, edge_index, F_u_to_e, F_v_to_e, batch):
        """
        H: node stalk tensor.
           Expected shape can be [num_nodes, d, r].

        edge_index: [2, num_edges], where edge_index[0] = source u,
                    edge_index[1] = target v.

        F_u_to_e, F_v_to_e: restriction maps for each edge.
           Expected shape can be [num_edges, d_edge, d_node].

        batch: graph id for each node.
           Shape [num_nodes]. Used for graph-level pooling.
        """
        num_nodes = H.shape[0]
        num_edges = edge_index.shape[1]

        # ----- Node branch -----
        node_features = []
        for i in range(num_nodes):
            P_i = self.projection_descriptor(H[i])
            z_i = self.node_mlp(P_i.flatten())
            node_features.append(z_i)
        node_features = torch.stack(node_features, dim=0)

        tau_0 = scatter_mean(node_features, batch, dim=0)

        # ----- Edge branch -----
        src = edge_index[0]
        dst = edge_index[1]

        edge_features = []
        edge_batch = batch[src]

        for e in range(num_edges):
            u = src[e]
            v = dst[e]

            Hu = H[u]
            Hv = H[v]

            De = F_v_to_e[e] @ Hv - F_u_to_e[e] @ Hu
            Qe = self.projection_descriptor(De)
            we = self.edge_mlp(Qe.flatten())
            edge_features.append(we)

        edge_features = torch.stack(edge_features, dim=0)
        tau_1 = scatter_mean(edge_features, edge_batch, dim=0)

        tau_G = torch.cat([tau_0, tau_1], dim=-1)
        y_hat = self.pred_mlp(tau_G)
        return y_hat
```

---

## Implementation Notes

### 1. Descriptor Flattening

The descriptors \(P_i\) and \(Q_e\) are symmetric projection-like matrices. To avoid redundant entries, use either:

```python
P_flat = P.flatten()
```

or preferably:

```python
P_flat = P[torch.triu_indices(P.shape[0], P.shape[1])]
```

Using the upper triangular entries reduces dimensionality.

### 2. Numerical Stability

The pseudoinverse may be unstable if \(XX^\top\) is ill-conditioned. Practical options:

```python
pinv = torch.linalg.pinv(gram, rtol=1e-5)
```

or Tikhonov regularization:

```python
gram_reg = gram + eps * torch.eye(gram.shape[-1], device=gram.device)
inv = torch.linalg.inv(gram_reg)
P = X.T @ inv @ X
```

Use `pinv` first for correctness. Use regularized inverse only if matrix shapes and rank assumptions are controlled.

### 3. Directed vs Undirected Edges

For undirected graphs, choose a consistent orientation for each edge before computing:

\[
D_e = F_{v \to e}H_v - F_{u \to e}H_u.
\]

Because \(Q_e\) is computed from \(D_e\), changing the sign of \(D_e\) should not change the projection descriptor in the ideal case. Thus the edge descriptor is largely orientation-invariant.

### 4. Graph-Size Normalization

Use `scatter_mean` instead of `scatter_sum` when graphs have different numbers of nodes or edges.

Use `scatter_sum` when graph size itself should influence the target.

### 5. Empty Edge Case

If a graph has no edges, set:

```python
tau_1 = torch.zeros_like(tau_0)
```

or use a learned empty-edge embedding.

---

## Expected Advantages

S2Pool should improve graph prediction for sheaf neural networks because it:

1. avoids unstable explicit gauge fixing;
2. pools basis-free descriptors instead of raw local coordinates;
3. prevents node-only readout from ignoring restriction-map information;
4. captures edge-level sheaf disagreement as a graph-level signal;
5. separates node content \(\tau_0\) and relational inconsistency \(\tau_1\), then lets the prediction MLP combine them.

---

## Minimal Mathematical Specification

Given node stalks \(H_i\), restriction maps \(F_{u \to e}\), \(F_{v \to e}\), and edge \(e = (u, v)\):

\[
P_i = H_i^\top (H_iH_i^\top)^+H_i,
\]

\[
D_e = F_{v \to e}H_v - F_{u \to e}H_u,
\]

\[
Q_e = D_e^\top(D_eD_e^\top)^+D_e,
\]

\[
z_i = \phi_0(P_i),
\]

\[
w_e = \phi_1(Q_e),
\]

\[
\tau_0 = \operatorname{Pool}_{i \in V}(z_i),
\]

\[
\tau_1 = \operatorname{Pool}_{e \in E}(w_e),
\]

\[
\tau_G = [\tau_0, \tau_1],
\]

\[
\hat{y}_G = \rho(\tau_G).
\]

Default pooling operator:

\[
\operatorname{Pool} = \operatorname{mean}.
\]

---

## Recommended File/Module Names

Suggested implementation filename:

```text
s2pool.py
```

Suggested class name:

```text
S2Pool
```

Suggested integration point:

```text
H2SNN encoder -> final node stalks and restriction maps -> S2Pool -> graph prediction
```
