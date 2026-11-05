# H2S-NN: Full Implementation Specification

This document gives a code-ready, end-to-end specification of **H2S-NN** from **input to output** so that another model or engineer can implement it directly.

It follows the research plan's description of H2S-NN as a **static heterophilic heterogeneous graph neural network** that:

1. lifts a heterogeneous information network into a **typed 2D cellular complex**,
2. learns **directed relation-specific sheaf restriction maps**,
3. performs **sheaf / Hodge diffusion** on node and edge cochains, and
4. predicts target node labels with an **MLP classifier trained end-to-end**.

STAGE is used as a plug-in geometric pre-encoder for initial node embeddings.

---

## 1. Problem setting

We begin with a heterogeneous information network

\[
G=(V,E,\tau_V,\tau_E,X,Y_{\mathrm{lab}})
\]

where:

- \(V\) is the set of nodes.
- \(E\subseteq V\times \mathcal R \times V\) is the set of typed directed edges.
- \(\tau_V(v)\) is the node type of node \(v\).
- \(\tau_E(e)=r\) is the relation type of edge \(e\).
- \(X_v\in\mathbb R^{d_{\text{raw}}(\tau_V(v))}\) is the raw feature vector of node \(v\).
- \(Y_{\mathrm{lab}}\) contains labels for the target nodes used in node classification.

The target task is typically **node classification** on one selected target node type, but the model can be adapted to link prediction or affinity prediction later.

The core challenge is that the graph is both:

- **heterogeneous**: multiple node types and edge types,
- **heterophilic**: connected nodes may have different labels or semantics.

Standard message passing often over-smooths under heterophily. H2S-NN is designed to avoid collapsing dissimilar nodes while still enforcing global topological consistency.

---

## 2. High-level model overview

The H2S-NN pipeline is:

1. **Pre-encode each node with STAGE**.
2. **Lift** the heterogeneous graph to a typed **2D cellular complex**.
3. **Learn sheaf restriction maps** from nodes to edges and from edges to faces.
4. **Construct sheaf coboundary operators** \(\delta_0\) and \(\delta_1\).
5. **Build degree-wise Hodge Laplacians** \(L_0\) and \(L_1\).
6. **Diffuse node and edge signals** to obtain harmonic-like representations.
7. **Read out node predictions** using harmonic node embeddings, STAGE embeddings, and pooled edge embeddings.
8. **Train end-to-end** with cross-entropy plus diffusion regularization.

---

## 3. Stage 0: STAGE pre-encoder

The research plan states that **STAGE** is pre-trained once and then plugged into downstream models as a heterogeneity-aware initialization module.

For implementation, treat STAGE as a function

\[
h_v^{(0)} = \mathrm{STAGE}(X_v,\tau_V(v)).
\]

### Practical implementation choices

- Output a common latent dimension for all node types:
  \[
  h_v^{(0)}\in\mathbb R^{d_0}.
  \]
- If STAGE is not yet implemented, use a temporary type-specific linear encoder:
  \[
  h_v^{(0)} = W_{\tau_V(v)}X_v + b_{\tau_V(v)}.
  \]
- Later replace that module with the real STAGE encoder.

### Recommended code interface

```python
class StageEncoder(nn.Module):
    def forward(self, node_features, node_types):
        # returns H0 of shape [num_nodes, d0]
        ...
```

---

## 4. Stage 1: Lift the HIN into a typed 2D cellular complex

H2S-NN lifts the graph to a typed cellular complex

\[
\mathcal X=(X^0,X^1,X^2)
\]

where:

- \(X^0\): 0-cells = nodes,
- \(X^1\): 1-cells = typed directed edges,
- \(X^2\): 2-cells = ordered meta-path instances.

So define

\[
X^0 = V,
\qquad
X^1 = \{e=(u\xrightarrow{r}v)\in E\},
\qquad
X^2 = \{\text{ordered meta-path instances}\}.
\]

### 4.1 Meta-path templates and instances

A meta-path template is a schema-level typed path

\[
\pi = T_0 \xrightarrow{r_1} T_1 \xrightarrow{r_2} \cdots \xrightarrow{r_\ell} T_\ell.
\]

A concrete graph instance of that meta-path is

\[
v_0 \xrightarrow{r_1} v_1 \xrightarrow{r_2} \cdots \xrightarrow{r_\ell} v_\ell.
\]

Each ordered instance becomes one face \(f\in X^2\).

### 4.2 Practical choices for face construction

For a first implementation:

- restrict meta-path lengths to \(\ell\in\{2,3,4\}\),
- choose a small set of templates manually,
- sample at most \(K\) instances per template per seed node,
- store the ordered boundary edge sequence of each face.

### 4.3 Data structures to build

Recommended data structures:

- `nodes`: list of node ids,
- `edges`: list of directed typed edges `(u, r, v)`,
- `faces`: list of face objects containing:
  - `face_id`,
  - `metapath_type`,
  - `boundary_edges = [e1, e2, ..., e_ell]`,
  - `boundary_orientations = [±1, ..., ±1]`.

Also precompute incidence relations:

- node \(\to\) incident edges,
- edge \(\to\) endpoint nodes,
- edge \(\to\) incident faces,
- face \(\to\) boundary edges.

### Recommended code interface

```python
class Typed2DComplexBuilder:
    def build(self, graph, metapath_templates):
        # returns nodes, edges, faces, and incidence structures
        ...
```

---

## 5. Stage 2: Directed typed cellular sheaf

H2S-NN equips the lifted complex with a **cellular sheaf**.

We will maintain three signal spaces:

\[
x_v\in\mathbb R^{d_0},
\qquad
s_e\in\mathbb R^{d_1},
\qquad
q_f\in\mathbb R^{d_2}.
\]

For a first implementation, it is simplest to use

\[
d_0=d_1=d_2=d.
\]

### 5.1 Node-to-edge restriction maps

For each directed typed edge \(e=(u\xrightarrow{r}v)\), learn two maps:

\[
F_{u\triangleright e}^{\mathrm{src},r}\in\mathbb R^{d_1\times d_0},
\qquad
F_{v\triangleright e}^{\mathrm{tgt},r}\in\mathbb R^{d_1\times d_0}.
\]

A concrete parameterization is:

\[
F_{u\triangleright e}^{\mathrm{src},r}
=
\Phi_r^{\mathrm{src}}([h_u^{(0)}\Vert h_v^{(0)}\Vert z_e]),
\]

\[
F_{v\triangleright e}^{\mathrm{tgt},r}
=
\Phi_r^{\mathrm{tgt}}([h_u^{(0)}\Vert h_v^{(0)}\Vert z_e]),
\]

where \(z_e\) is a learned relation / edge-type embedding.

#### Practical notes

- Each MLP outputs \(d_1d_0\) numbers, then reshape to \([d_1,d_0]\).
- Use one pair of MLPs per relation type \(r\), or one shared MLP conditioned on relation embedding.
- To stabilize training, optionally clip or normalize the resulting matrices.

### 5.2 Edge-to-face restriction maps

For each face \(f\) with ordered boundary edges \((e_1,\dots,e_\ell)\), learn

\[
F_{e_i\triangleright f}\in\mathbb R^{d_2\times d_1}.
\]

A practical sequence-aware implementation:

1. Build one token per boundary edge:
   \[
   t_i = [h_{\mathrm{src}(e_i)}^{(0)}\Vert h_{\mathrm{tgt}(e_i)}^{(0)}\Vert \mathrm{emb}(\tau_E(e_i))\Vert \mathrm{emb}(\pi_f)\Vert \mathrm{emb}(i)].
   \]
2. Run a lightweight LSTM over \((t_1,\dots,t_\ell)\):
   \[
   a_i = \mathrm{LSTM}(t_1,\dots,t_\ell)_i.
   \]
3. Project each hidden state to a restriction matrix:
   \[
   F_{e_i\triangleright f} = \Psi(a_i).
   \]

#### Practical notes

- \(\Psi\) can be a linear layer producing \(d_2d_1\) values, then reshape.
- This gives position-aware edge-to-face maps.

### Recommended code interface

```python
class NodeToEdgeSheaf(nn.Module):
    def forward(self, H0, edge_index, edge_types, edge_attr=None):
        # returns source and target restriction matrices for each edge
        ...

class EdgeToFaceSheaf(nn.Module):
    def forward(self, H0, faces, edge_types):
        # returns restriction matrices for each edge-face incidence
        ...
```

---

## 6. Stage 3: Build the sheaf coboundary operators

The sheaf structure induces two coboundary operators:

\[
\delta_0 : C^0(\mathcal X;\mathcal F) \to C^1(\mathcal X;\mathcal F),
\qquad
\delta_1 : C^1(\mathcal X;\mathcal F) \to C^2(\mathcal X;\mathcal F),
\]

with the cochain condition

\[
\delta_1\circ\delta_0 = 0.
\]

### 6.1 Node-to-edge coboundary \(\delta_0\)

For an edge \(e=(u\xrightarrow{r}v)\), define

\[
(\delta_0 x)_e
=
F_{v\triangleright e}^{\mathrm{tgt},r}x_v
-
F_{u\triangleright e}^{\mathrm{src},r}x_u.
\]

This maps node cochains to edge cochains.

### 6.2 Edge-to-face coboundary \(\delta_1\)

For a face \(f\) with oriented boundary \((e_1,\dots,e_\ell)\), define

\[
(\delta_1 s)_f
=
\sum_{i=1}^{\ell}
\eta_{e_i,f}F_{e_i\triangleright f}s_{e_i},
\]

where \(\eta_{e_i,f}\in\{-1,+1\}\) is the signed edge-face incidence.

### Practical implementation notes

Do **not** build dense matrices unless the graph is tiny.

Instead:

- store \(\delta_0\) as a sparse block operator with two nonzero blocks per edge,
- store \(\delta_1\) as a sparse block operator with one nonzero block per boundary edge of each face.

### Recommended code interface

```python
class CoboundaryDelta0:
    def forward(self, x, edge_incidence, F_src, F_tgt):
        ...
    def transpose(self, s, edge_incidence, F_src, F_tgt):
        ...

class CoboundaryDelta1:
    def forward(self, s, face_incidence, F_ef, signs):
        ...
    def transpose(self, q, face_incidence, F_ef, signs):
        ...
```

---

## 7. Degree-wise Hodge Laplacians

Once \(\delta_0\) and \(\delta_1\) are available, define:

\[
L_0 = \delta_0^\top\delta_0,
\qquad
L_1 = \delta_0\delta_0^\top + \delta_1^\top\delta_1.
\]

Interpretation:

- \(L_0\) acts on **node cochains**,
- \(L_1\) acts on **edge cochains**.

### Efficient operator form

Never materialize dense \(L_0\) or \(L_1\) for realistic graphs. Instead implement:

```python
def apply_L0(x):
    return delta0_T(delta0(x))


def apply_L1(s):
    return delta0(delta0_T(s)) + delta1_T(delta1(s))
```

This is the most important efficiency detail.

---

## 8. Diffusion stage

The model diffuses both node and edge signals.

### 8.1 Initial states

Set

\[
x^{(0)} = [h_v^{(0)}]_{v\in V}
\in \mathbb R^{|V|\times d_0}.
\]

Initialize edge states with an edge initializer:

\[
s_e^{(0)} = W_e[h_u^{(0)}\Vert h_v^{(0)}\Vert z_e],
\]

so

\[
s^{(0)}\in\mathbb R^{|E|\times d_1}.
\]

### 8.2 Discrete diffusion solver

For a first implementation, use explicit Euler steps:

\[
x^{(k+1)} = x^{(k)} - \eta_0L_0x^{(k)},
\]

\[
s^{(k+1)} = s^{(k)} - \eta_1L_1s^{(k)},
\]

for \(k=0,\dots,T-1\).

After \(T\) steps, define

\[
x^* = x^{(T)},
\qquad
s^* = s^{(T)}.
\]

These are treated as the final node and edge embeddings after diffusion.

### 8.3 Optional neural diffusion version

If you want a more expressive diffusion layer, use

\[
x^{(k+1)} = x^{(k)} - \eta_0\,\sigma(L_0x^{(k)}W_0),
\]

\[
s^{(k+1)} = s^{(k)} - \eta_1\,\sigma(L_1s^{(k)}W_1).
\]

But the linear Euler form is easier and more stable for the first full implementation.

### Recommended code interface

```python
class H2SDiffusion(nn.Module):
    def __init__(self, num_steps, step_size_node, step_size_edge):
        ...

    def forward(self, x0, s0, delta0, delta1):
        # iteratively produce x_star, s_star
        ...
```

---

## 9. Readout and classifier

The classifier uses three pieces of information per target node:

1. harmonic-like node embedding \(x_v^*\),
2. STAGE embedding \(h_v^{(0)}\),
3. pooled incident edge embedding.

### 9.1 Pool edge information back to nodes

For each node \(v\), transport every incident edge state into the node's stalk
through the corresponding endpoint restriction map, then mean-aggregate:

\[
r_v^{\mathrm{edge}}
=
\frac{1}{d_v}
\left(
\sum_{\mathrm{src}(e)=v}F_{\mathrm{src},e}^{\top}s_e^*
+
\sum_{\mathrm{tgt}(e)=v}F_{\mathrm{tgt},e}^{\top}s_e^*
\right),
\]

where \(d_v\) counts incidences from both observed and closure edges. An isolated
node receives the zero vector. Both endpoint contributions are positive because
this is a context summary, not the oriented coboundary adjoint.

### 9.2 Final node representation

\[
z_v = [x_v^*\Vert h_v^{(0)}\Vert r_v^{\mathrm{edge}}].
\]

### 9.3 Prediction

\[
\hat y_v = \mathrm{MLP}(z_v).
\]

Use a simple 2-layer MLP with dropout and GELU or ReLU.

### Recommended code interface

```python
class NodeClassifier(nn.Module):
    def forward(self, x_star, h0, s_star, edge_index, F_src, F_tgt):
        # endpoint-adjoint edge-to-node pooling, then classification
        ...
```

---

## 10. Loss function

### 10.1 Classification loss

For labeled target nodes \(V_{\mathrm{lab}}\), use cross-entropy:

\[
\mathcal L_{\mathrm{cls}}
=
-\sum_{v\in V_{\mathrm{lab}}}\sum_{c=1}^{C} y_{vc}\log \hat y_{vc}.
\]

### 10.2 Diffusion regularization

A practical diffusion regularizer is

\[
\mathcal L_{\mathrm{diff}}
=
\lambda_0\|\delta_0x^*\|^2
+
\lambda_1\|\delta_1s^*\|^2
+
\lambda_c\|s^* - \delta_0x^*\|^2.
\]

Interpretation:

- \(\|\delta_0x^*\|^2\): node-to-edge discrepancy control,
- \(\|\delta_1s^*\|^2\): edge-to-face smoothness / coherence,
- \(\|s^* - \delta_0x^*\|^2\): consistency between final edge states and node-induced edge states.

### 10.3 Total loss

\[
\mathcal L
=
\mathcal L_{\mathrm{cls}} + \mathcal L_{\mathrm{diff}}.
\]

### Practical note

At the beginning of training, it can help to warm up with only \(\mathcal L_{\mathrm{cls}}\) for a few epochs, then turn on diffusion regularization.

---

## 11. Full algorithm: step by step

The following is the implementation sequence from raw input to prediction.

### Input

- heterogeneous graph \(G=(V,E)\),
- node features \(X\),
- node types \(\tau_V\),
- edge types \(\tau_E\),
- label set on target nodes,
- selected meta-path templates \(\Pi\).

### Output

- node logits \(\hat Y\),
- optionally \(x^*\) and \(s^*\).

### Algorithm

1. **Encode nodes** with STAGE:
   \[
   h_v^{(0)} = \mathrm{STAGE}(X_v,\tau_V(v)).
   \]

2. **Enumerate or sample meta-path instances** from the selected templates.

3. **Build the typed 2D complex**:
   \[
   X^0=V,\qquad X^1=E,\qquad X^2=\text{sampled meta-path instances}.
   \]

4. **Learn node-to-edge restriction maps** for every edge.

5. **Learn edge-to-face restriction maps** for every face-edge incidence.

6. **Assemble the sparse coboundary operators** \(\delta_0\) and \(\delta_1\).

7. **Initialize signals**:
   \[
   x^{(0)}=[h_v^{(0)}]_{v\in V},
   \qquad
   s^{(0)}=\mathrm{EdgeInit}(h_u^{(0)},h_v^{(0)},z_e).
   \]

8. **Diffuse node and edge signals** for \(T\) steps:
   \[
   x^{(k+1)}=x^{(k)}-\eta_0L_0x^{(k)},
   \]
   \[
   s^{(k+1)}=s^{(k)}-\eta_1L_1s^{(k)}.
   \]

9. **Set final embeddings**:
   \[
   x^*=x^{(T)},\qquad s^*=s^{(T)}.
   \]

10. **Pool incident edge embeddings** back to each node.

11. **Form the classifier input**:
   \[
   z_v=[x_v^*\Vert h_v^{(0)}\Vert r_v^{\mathrm{edge}}].
   \]

12. **Predict labels**:
   \[
   \hat y_v=\mathrm{MLP}(z_v).
   \]

13. **Compute total loss**:
   \[
   \mathcal L = \mathcal L_{\mathrm{cls}} + \mathcal L_{\mathrm{diff}}.
   \]

14. **Backpropagate** through all learnable modules.

---

## 12. Minimal tensor specification for code

A follow-up coding model will need exact tensor shapes.

### Node tensor

\[
H^{(0)}\in\mathbb R^{|V|\times d_0}.
\]

### Edge tensor

\[
S^{(0)}\in\mathbb R^{|E|\times d_1}.
\]

### Face tensor

For the first implementation, an explicit persistent face state is optional. You can keep faces implicit and only use \(\delta_1\) and \(\delta_1^\top\).

If explicit face states are later needed:

\[
Q\in\mathbb R^{|X^2|\times d_2}.
\]

### Restriction map tensors

For each edge \(e\):

\[
F_{u\triangleright e}^{\mathrm{src},r}\in\mathbb R^{d_1\times d_0},
\qquad
F_{v\triangleright e}^{\mathrm{tgt},r}\in\mathbb R^{d_1\times d_0}.
\]

For each face-edge incidence:

\[
F_{e\triangleright f}\in\mathbb R^{d_2\times d_1}.
\]

### Sparse incidence format

For \(\delta_0\):

- each edge contributes exactly two nonzero blocks,
- source block is \(-F_{u\triangleright e}^{\mathrm{src},r}\),
- target block is \(+F_{v\triangleright e}^{\mathrm{tgt},r}\).

For \(\delta_1\):

- each face contributes one signed block per boundary edge,
- that block is \(\eta_{e,f}F_{e\triangleright f}\).

---

## 13. Training loop

A practical training loop is:

1. preprocess or cache sampled faces,
2. for each epoch:
   - compute STAGE embeddings,
   - build restriction maps,
   - assemble \(\delta_0,\delta_1\),
   - run diffusion,
   - pool edge embeddings to nodes,
   - classify nodes,
   - compute total loss,
   - backpropagate,
   - validate.

### Stability recommendations

- start with small diffusion depth, for example \(T=3\) to \(8\),
- use gradient clipping,
- use dropout in the classifier,
- optionally normalize restriction maps,
- optionally freeze STAGE at first and unfreeze later.

---

## 14. Inference

Inference is the same as the forward pass, without gradient updates:

1. compute STAGE embeddings,
2. build the typed 2D complex,
3. compute restriction maps,
4. run diffusion,
5. compute logits.

No meta-path projection into separate homogeneous views is required.

---

## 15. Recommended module decomposition for code

A clean codebase can be organized as follows:

- `stage_encoder.py`
- `metapath_lifter.py`
- `sheaf_maps.py`
- `coboundary_ops.py`
- `diffusion.py`
- `decoder.py`
- `losses.py`
- `train_h2snn.py`

Recommended classes:

- `StageEncoder`
- `Typed2DComplexBuilder`
- `NodeToEdgeSheaf`
- `EdgeToFaceSheaf`
- `CoboundaryDelta0`
- `CoboundaryDelta1`
- `H2SDiffusion`
- `NodeClassifier`

---

## 16. Most implementation-friendly final specification

If a coding model wants one concise target specification, use this version first:

\[
h_v^{(0)}=\mathrm{STAGE}(X_v,\tau_V(v)),
\]

\[
(\delta_0 x)_e
=
F_{v\triangleright e}^{\mathrm{tgt},r}x_v
-
F_{u\triangleright e}^{\mathrm{src},r}x_u,
\]

\[
(\delta_1 s)_f
=
\sum_{e\in\partial f}\eta_{e,f}F_{e\triangleright f}s_e,
\]

\[
L_0=\delta_0^\top\delta_0,
\qquad
L_1=\delta_0\delta_0^\top+\delta_1^\top\delta_1,
\]

\[
x^{(k+1)}=x^{(k)}-\eta_0L_0x^{(k)},
\qquad
s^{(k+1)}=s^{(k)}-\eta_1L_1s^{(k)},
\]

\[
z_v=[x_v^*\Vert h_v^{(0)}\Vert \mathrm{Pool}\{s_e^*:e\ni v\}],
\qquad
\hat y_v=\mathrm{MLP}(z_v),
\]

\[
\mathcal L
=
\mathcal L_{\mathrm{CE}}
+
\lambda_0\|\delta_0x^*\|^2
+
\lambda_1\|\delta_1s^*\|^2
+
\lambda_c\|s^* - \delta_0x^*\|^2.
\]

This is concrete enough for an LLM to turn into working PyTorch or PyTorch Geometric code.

---

## 17. Short final summary

**H2S-NN takes a heterogeneous graph, lifts it into a typed 2D cellular complex, learns relation-specific sheaf restriction maps on that complex, diffuses node and edge signals with Hodge-style operators, and predicts node labels from harmonic-like node embeddings augmented with STAGE and pooled edge context.**
