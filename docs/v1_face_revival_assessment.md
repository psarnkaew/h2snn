# Can the v1 2-cell (node→edge→face) come back, with face information reaching nodes and fewer parameters?

2026-11-10 09:00 JST. Asked by the author after the review showed v2 dropped everything the WSDM reviewers praised (R1 S1–S2, R2 S1–S2, R3 S1, R4 S2: path-supported 2-cell, edge-to-face maps by transport, chain-condition proof). Status: assessment only; no design adopted.

## Sources read
- DNSD (Bourgerie et al., arXiv 2605.19021, GFM@ICML 2026 workshop): full paper via fetch (methods, App. A, experiments, parameter table).
- Code `github.com/remibourgerie/deep-neural-sheaf-diffusion`: `models.py` read in full; README read; other files only grepped (no face/cell code anywhere).
- Old manuscript `h2snn_manuscript.tex` lines 1–1000 (method of v1); WSDM reviews in full.

## Facts about DNSD (VERIFIED-SOURCE)
- Graph sheaf only: node stalks, node→edge maps, node states. **No edges states, no faces, no edge-to-face maps.** The node→edge→face structure is H2S-NN v1's own.
- Main idea: replace the Laplacian (difference `F_u x_u − F_v x_v`) by a one-sided "adjacency" message `F_uᵀ F_v x_v`, plus LayerNorm, tanh, per-stalk gate.
- More parameters than NSD (separate source/target map builders per layer): synthetic setting, DNSD diag 5,007 vs NSD diag 2,655 at 16 layers. Diagonal maps match or beat full maps on their benchmarks.

## Numerical check on the v1 construction (VERIFIED-RUN, scratch `dnsd/check_face.py`)
Instance P1–A7–P9 with closure edge P9→P1, extra edge A7–P4, d = 3, scaled-orthogonal maps, A_f random.
| Check | Value | Meaning |
|---|---|---|
| ‖δ₁δ₀‖ | 2.2e-16 | chain condition (manuscript Prop.) holds |
| ‖δ₀ᵀδ₁ᵀ‖ | 2.2e-16 | Lemma A′: face information cannot reach nodes through the signed coupling δ₀ᵀ |
| ‖P δ₁ᵀ‖ at P1 / A7 / P9 / P4 | 3.48 / 2.43 / 1.57 / 0 | with the unsigned edge→node pooling P (v1 readout Eq. pool-transport; DNSD's one-sided idea), face information reaches **every node of the instance, including the intermediate A7**, and nothing else |
| face term on s ∈ im δ₀ | 1.3e-15 | the face sees only edge signal not explained by node states; edge states need their own content (v1: `MLP_init`) |
| ‖L₁ᵘᵖ(A_f) − L₁ᵘᵖ(Q A_f)‖, Q orthogonal | 2.2e-15 | A_f enters only through A_fᵀA_f |

## Consequences
1. **v1 was genuinely 2D** (cochain complex, chain condition). Earlier statement in the HTML ("v1 was not 2D either") is wrong; the reviewers' point (R2 W3, meta-review) was the route to nodes, not the dimension.
2. **The route to nodes can be fixed** by coupling edge→node with the unsigned pooling P at every layer instead of only at the readout. Face information then reaches all instance nodes, not only the endpoints/closure edge. This answers R2 W3 directly; R2 comment 5's ablation (closure-only vs face) becomes runnable.
3. **What the face then contributes** is different from v2 and MAGNN: it measures the inconsistency (circulation) of *edge* states around the instance, not an average of node states. Whether this helps prediction is unknown.
4. **Parameter cut from our own algebra:** v1's `MLP_ℓ` maps (ℓ+1)d + ℓd_r inputs to d² outputs per path length, but only A_fᵀA_f matters. Replace by the scalar weight w_f (already present) or a diagonal metric. Example single linear layer, d = 64, d_r = 16, ℓ = 2: 224×4096 ≈ 0.92M → diagonal 224×64 ≈ 14k → scalar 0. Maps computed once from h⁽⁰⁾ (not per layer as in DNSD).

## Is R2 W3 right about v1? (added 11:40 JST; read against manuscript §3, lines 1–1000)
- **Right in substance.** In v1 the node and edge cochains are diffused separately and "not exchanged between diffusion steps"; face information reaches a prediction only once, at the readout, through the edges incident to the target. Worse than R2 says: the closure edge is in E_ext, so it enters the **node** operator L₀ as a direct P1–P9 coupling (|L₀ block P1–P9| = 0.386 in the check) — an endpoint meta-path edge with holonomy transport (Theorem R). That is the approach the introduction criticises.
- **Wrong in one detail.** "Only through the closure edge" is not exact: the face term also changes the path edge P1–A7, which is incident to the target and pooled at the readout. The intermediate node A7 still never receives face information in v1.
- **Fix that keeps the idea (VERIFIED-RUN, scratch `dnsd/check_noclosure.py`):** keep the closure edge only as the boundary of the 2-cell (δ₁, chain condition); remove it from the node operator (L₀ P1–P9 block → 0) and from the edge→node lift; lift observed edges to nodes at every layer. Face information then reaches P1 / A7 / P9 = 1.74 / 2.43 / 0.78, P4 = 0, only through real path edges. No endpoint shortcut remains in the node computation.
- Ablations that answer R2 comment 5: endpoint holonomy edge only (no face); v1 as submitted; face + lift without closure coupling (proposal); no faces.

**Author position (2026-11-10 11:40):** (i) the face acting on the edge cochain is correct by definition of a 2-cell and stays; (ii) endpoint information inside the face (closure edge state) is acceptable. The reviewer's objection is answered by the route, not by removing endpoint information: the closure edge is no longer the only, or any, route to node states. The "closure without own state" arm is optional (diagnostic), not required.

## DNSD-style messages between dimensions (12:10 JST; VERIFIED-RUN, scratch `dnsd/check_dnsd_face.py`)
Proposal under discussion: DNSD-style messages for node→edge and edge→face, edge and face states, factorized maps shared across layers.
- If node→edge uses DNSD's one-sided form (edge state = target node state transported), the face value equals Σ_i T_{v_i} x_{v_i} exactly (difference 4.4e-16) — the v2 hub cell, i.e. MAGNN's instance encoder. The 2-cell collapses back to the design the novelty review rejected.
- With the coboundary (difference) form, node-induced edge signal gives face value 0; the face sees only edge-state content not explained by nodes.
- Rule: node→edge and edge→face keep the coboundary form (chain condition, curl); DNSD's one-sided form is used for **edge→node** (the lift) and optionally node→node, with LN / odd activation / gate as its depth stabilizers.
- DNSD evidence (paper §5, App. B): Roman Empire 8 layers 83.4 ± 0.9 vs NSD diag 79.1 ± 0.5; 5,007 vs 2,655 parameters is one synthetic pair only, not a general cost statement.

## Risks (must be settled before adopting)
- Memory: edge states and one closure edge per instance return (v1 ACM 9,260 MB; R1 W1, R3 W3, R4 scalability). v2 removed them for this reason.
- Closure edge carries endpoint information; the closure-only ablation must show the face adds something beyond it.
- WISP walks can revisit nodes; v1 chain condition needs simple paths (filter).
- Prior art not checked at formula level: CW Networks (Bodnar et al. 2021, node↔edge↔cell message passing), sheaf models on cell complexes, arXiv 2606.00937 (Cellular Sheaf Neural Operators, unread).
- No evidence of benefit yet; needs a known-answer task whose label depends on edge-state circulation along an instance, with arms: face+P, closure-only, no faces, v2 hub cell.
