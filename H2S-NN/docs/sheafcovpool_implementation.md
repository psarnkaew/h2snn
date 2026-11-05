# SheafCovPool — Implementation Spec (FINALIZED, 2026-07-22)

> **This is the current mechanism spec.** Method: `SheafCovPool`, class `SheafCovPool` in
> [`../src/models/graph_pooling.py`](../src/models/graph_pooling.py), `pooling_type: "sheafcov"`.
> Newcomer walkthrough: [`sheafcovpool_explainer.md`](sheafcovpool_explainer.md). Decisions/history/gate results:
> [`sheafcovpool_design_log.md`](sheafcovpool_design_log.md) §0. Research framing:
> [`sheafcovpool_proposal.html`](sheafcovpool_proposal.html) (see the "FINAL METHOD" card near the top).
>
> **This supersedes every earlier spec in this file's history** — the original Select–Reduce–Connect design and
> the later energy-preserving Galerkin coarsening redesign (kept as Appendix A below for provenance). Both were
> **hierarchical** (they coarsened the graph and re-ran diffusion at a coarser resolution) and both were tested
> and closed as negative on TU benchmarks (§design_log §0). `SheafCovPool` is **not hierarchical** — it is a
> single-level terminal readout, a direct sibling of `GramPool`/`SheafPool` in the same file.

---

## 0. The one principle

`GramPool` already gives an exactly $O(d)$-invariant per-node/per-edge descriptor (the Gram matrix of a small
stack of existing views) but only ever takes its **mean** across nodes. `SheafCovPool` keeps that descriptor
verbatim and adds the two things a mean discards:

1. **Attention** — a learned, gauge-invariant score per node/edge, replacing the uniform mean with a weighted one.
2. **Covariance** — the weighted second moment of the descriptor across nodes/edges, not just its first moment,
   so the readout can represent the *spread* of node/edge states in a graph, not only their average.

Both are computed **without whitening**, which is what makes this safe on H2SNN's low-channel stalks where
SheafPool's whitening step degenerates (§3 below).

---

## 1. What the backbone provides (unchanged from GramPool/SheafPool)

From one `H2SNN.encode(...)` pass, `GraphPooling.forward` already receives everything needed — no new backbone
hooks, no `encode_from_stalk`, no per-level face re-enumeration:

| Field | Shape | Use |
|---|---|---|
| `x_star` | `[N, d]` | node cochain |
| `h0` | `[N, d]` | node input encoding (pre-diffusion) |
| `s_star` | `[E, d]` | edge cochain |
| `delta0_x_star`, `delta0_h0` | `[E, d]` | node→edge disagreement (already computed for the smoothness loss) |
| `F_src`, `F_tgt` | `[E, d, d]` | endpoint restriction maps over the same extended edge domain |
| `edge_index`, `batch`, `num_graphs` | — | batching for scatter/softmax |

`pool_edge_stalks_to_nodes(s_star, edge_index, F_src, F_tgt, num_nodes)` (`src/models/pooling_utils.py`) gives
one more per-node view, exactly as in `GramPool`.

---

## 2. Stage 1 — per-node/per-edge Gram descriptor (identical to `GramPool`)

```
node_stack = stack([x_star, h0, pool_edge_stalks_to_nodes(s_star, edge_index, F_src, F_tgt, N)], dim=-1)  # [N, d, 3]
edge_stack = stack([delta0_x_star, delta0_h0, s_star], dim=-1)                                            # [E, d, 3]

node_gram = node_stack^T @ node_stack   -> flatten -> [N, 9]      # exactly O(d)-invariant
edge_gram = edge_stack^T @ edge_stack   -> flatten -> [E, 9]
```

No pseudo-inverse (unlike `S2Pool`), no whitening (unlike `SheafPool`) — the Gram matrix of an $O(d)$-transformed
stack is unchanged by that transform, so this is already invariant with zero extra machinery.

### 2a. Why the node stack needs transport (correctness fix, 2026-07-29)

The Gram argument only holds if a stack's columns are sections **in one common stalk**, so that a gauge change
moves all of them by the same matrix. The two stacks reach that state differently:

- **Edge stack — already fine.** $(\delta_0 x)_e$, $(\delta_0 h^{(0)})_e$ and $s_e^\star$ all live in edge $e$'s
  stalk, so a gauge change sends each to $R_e(\cdot)$.
- **Node stack — needed fixing.** The original code summed raw $s_e^\star$ coordinates from *different* edge
  stalks into a node vector. Under independent gauges $x_v\mapsto Q_vx_v$, $s_e\mapsto R_es_e$, the sum
  $\sum_e R_es_e$ is not $Q_v(\sum_e s_e)$, so the three Gram entries touching that column ($G_{13},G_{23},G_{33}$
  — half of the six unique entries) changed with the choice of basis. Summing sections across stalks is not a
  well-defined sheaf operation to begin with.

The fix pulls each edge state back through its **own endpoint** restriction map before averaging:

$$\widetilde s_v^\star=\frac{1}{d_v}\Big(\sum_{e:\,\mathrm{src}(e)=v}F_{\mathrm{src},e}^{\mathsf T}s_e^\star
+\sum_{e:\,\mathrm{tgt}(e)=v}F_{\mathrm{tgt},e}^{\mathsf T}s_e^\star\Big).$$

With $F'=R_eFQ_v^{\mathsf T}$ this gives $(F')^{\mathsf T}s'=Q_vF^{\mathsf T}R_e^{\mathsf T}R_es=Q_vF^{\mathsf T}s$,
so all three node columns now move by the same $Q_v$ and $Z_v\mapsto Q_vZ_v$ holds as claimed.

Three points worth stating precisely:

- The requirement is on the **gauge group** ($R_e^{\mathsf T}R_e=I$), *not* on the restriction maps. With
  `use_orthogonal: True` the maps are **conformal-orthogonal** $F=cQ$ with a learned scale, not orthogonal; the
  argument holds for arbitrary $F$ either way. Earlier docs that said the maps are "always orthogonal" were wrong.
- $F^{\mathsf T}$ (adjoint) is used rather than $F^{+}$, so the summary retains the learned conformal incidence
  scale and matches the inner products the Hodge operators already use. Both choices are gauge-equivariant.
- Both endpoints contribute with a `+` sign. This is an incident-edge *context* summary, not the signed
  coboundary adjoint $\delta_0^{\mathsf T}s$. Keeping the sign convention the pre-fix code already used means
  this patch corrects the stalk mismatch **alone**, without simultaneously changing aggregation semantics.

`NodeReadout` and `LinkPredictionReadout` now use this same endpoint-adjoint summary. They do not claim
gauge-invariant logits, but all vectors concatenated by the heads are expressed in the corresponding node stalk.

## 3. Stage 2 — learned projection + gauge-invariant attention

```
u_node = MLP_node(node_gram)   # [N, F]   F = feat_dim = 8
u_edge = MLP_edge(edge_gram)   # [E, F]

score_node = MLP_scorer_node(u_node)                       # [N, 1]
alpha_node = softmax(score_node, index=batch)              # per-graph softmax, sums to 1 within each graph
```

(edges identically, scored/softmaxed per graph via `edge_batch = batch[edge_index[0]]`).

## 4. Stage 3 — first- and second-moment pooling across nodes/edges

For each graph $g$, with $v$ ranging over its nodes:

```
mu_g    =  sum_v  alpha_v * u_v                                          # [F]           first moment (mean)
Sigma_g =  sum_v  alpha_v * (u_v - mu_g) (u_v - mu_g)^T                  # [F, F]        second moment (covariance)
```

implemented via `torch_geometric.utils.scatter(..., reduce="sum")` twice (once for `mu`, once for the
weighted outer product using `mu[batch]` to center) — no explicit loop over graphs. Keep only the upper
triangle of `Sigma_g` (`torch.triu_indices`, includes the diagonal): `F(F+1)/2 = 36` dims for `F=8`. Repeat for
edges with their own `mu_edge_g`, `Sigma_edge_g`.

Both moments are exactly $O(d)$-invariant: they are affine combinations and outer products of the already
gauge-invariant $u_v$, so no new gauge-handling machinery is introduced at this stage.

## 5. Stage 4 — classify

```
summary = concat([mu_node_g, triu(Sigma_node_g), mu_edge_g, triu(Sigma_edge_g)])   # [2*(F + F(F+1)/2)] = 88 dims (F=8)
logits  = MLP_head(summary)                                                         # [num_classes]
```

Trained with cross-entropy only — same discipline as `GramPool`/`SheafPool`. No auxiliary losses, no assignment
regularizer (there is no assignment/selection in this method — every node/edge contributes, weighted by `alpha`).

---

## 6. Exact implementation

Fully implemented in `src/models/graph_pooling.py`:

```python
class SheafCovPool(GraphPooling):
    def __init__(self, d, hidden_dim, num_classes, dropout, feat_dim=8): ...
    def _pool_moments(self, u, scorer, group, num_graphs): ...  # -> (mu [G,F], cov_triu [G, F(F+1)/2])
    def forward(self, x_star, h0, s_star, delta0_x_star, delta0_h0,
                edge_index, F_src, F_tgt, batch, num_graphs): ...
```

`F_src`/`F_tgt` are on the `GraphPooling.forward` interface, not just this class, and `GramPool`, `S2Pool` and
the adapted `SheafPool` all consume the same transported summary. That is deliberate: if only `SheafCovPool`
received the corrected node view, a readout-vs-readout comparison would confound the pooling mechanism with the
correctness fix. `MeanPool` accepts the maps and ignores them (it reads `x_star` only). The hierarchical poolers
in `src/models/s2pool_h2.py` are plain `nn.Module`s on a separate trainer branch and are unaffected.

Registered in `build_graph_pooling(pooling_type="sheafcov", ...)`. Selectable via
`configs/experiments/<dataset>_h2snn.yaml` → `data.pooling_type: sheafcov`. No new `ModelConfig` fields were
needed (unlike the abandoned Galerkin line's `pool_ratio`/`pool_num_levels`/etc. — see Appendix A).

---

## 7. Tests

Implemented in [`../tests/test_graph_pooling.py`](../tests/test_graph_pooling.py) (12 tests, all passing):

| Test | What it pins down |
|---|---|
| `test_hand_computed_endpoint_aggregation` | src/tgt maps applied at the right endpoints, correct incidence divisor |
| `test_transport_is_equivariant_under_independent_local_gauges` | $\widetilde s_v'=Q_v\widetilde s_v$ under independent $Q_v$, $R_e$ and $F'=R_eFQ_v^{\mathsf T}$ |
| `test_raw_coordinate_aggregation_is_not_equivariant` | negative control — the pre-fix aggregation *fails* the assertion above |
| `test_{grampool,sheafcovpool}_invariant_under_independent_local_gauges` | end-to-end logit invariance under local gauges |
| `test_sheafcovpool_invariant_to_node_and_edge_reindexing` | permutation invariance (softmax + sum are order-independent) |
| `test_every_extended_domain_edge_participates` | real *and* closure 1-cells both reach the logits |
| `test_isolated_node_gets_zero_summary`, `test_self_loop_counts_as_two_incidences` | boundary behavior |
| `test_misaligned_inputs_raise_value_error` | the five `ValueError` guards (rank, shape, extended-domain alignment, stalk dim) |
| `test_all_variants_produce_expected_shape_with_no_nans` | shape/finite across **all five** poolers |

The negative control computes the retired raw aggregation inline. It exists because an invariance assertion that
nothing can fail proves nothing — see §7a.

Properties that hold by construction and are not separately asserted: attention sums to one per graph (softmax),
`Sigma_g` is PSD (weighted sum of outer products), and `_pool_moments` returns zeros rather than NaN for an
empty node/edge set (`u.size(0) == 0` branch).

**Cost:** ~18.8K parameters at `d=32` (measured), ~3.6s/epoch on MUTAG — comparable to `GramPool`/`SheafPool`,
no hierarchical overhead. The transport adds $O(Ed^2)$, the same order as the $\delta_0$ application already
running each forward pass.

### 7a. Why an invariance test alone is not enough — the `S2Pool` case

While verifying the transport fix we found that `S2Pool`'s descriptor
$P=X^{\mathsf T}(XX^{\mathsf T})^{+}X$ is the **constant identity** whenever the stack has full column rank,
which is the normal case here (3 views, $d\ge3$). With the thin SVD $X=U\Sigma V^{\mathsf T}$,

$$P=V\Sigma U^{\mathsf T}\big(U\Sigma^{2}U^{\mathsf T}\big)^{+}U\Sigma V^{\mathsf T}
=V\Sigma\Sigma^{-2}\Sigma V^{\mathsf T}=VV^{\mathsf T}=I_3 .$$

Measured: $\max|P-I_3|=3.6\times10^{-7}$ on random full-rank stacks, and logits differ by $1.5\times10^{-8}$
between unrelated batches where `GramPool` and `SheafCovPool` both differ by $\approx2\times10^{-1}$. So
`S2Pool` carries no information from its inputs, every graph gets the same logits, and its advertised
GL($d$)-invariance is invariance-by-constancy. This is exactly why the old gauge test never caught it: a
constant function passes every invariance test ever written. **Do not cite `S2Pool` as a baseline until this is
fixed** — a working variant would have to keep the singular values, as `GramPool`'s $H^{\mathsf T}H$ does,
instead of dividing them out.

---

## 8. Final gate results (MUTAG + ENZYMES, 3 seeds, 40 epochs, macro-F1)

> **Provenance:** these runs predate the 2026-07-29 endpoint-adjoint transport fix (§2a), so they were produced
> with the untransported node view. All three readouts shared that node view, so the comparison between them is
> still like-for-like, but the absolute numbers do not correspond to the current code. Re-run before quoting
> them as evidence about the method as it now stands.

| Dataset | SheafCovPool | GramPool | SheafPool |
|---|---|---|---|
| MUTAG | 0.596 ± 0.050 | 0.595 ± 0.081 | 0.260 ± 0.014 |
| ENZYMES | 0.236 ± 0.144 | 0.166 ± 0.088 | 0.048 ± 0.000 |

Honest reading: ties `GramPool` on MUTAG; nominal but noisy (high-variance, overlapping) improvement on
ENZYMES; clearly dominates `SheafPool` on both (which collapses close to/below the majority-class floor due to
whitening a rank-3-in-`d` stack — see design log §pm2/§3.3 for the root cause). See
`sheafcovpool_design_log.md` §0 for the full discussion and the recommendation on how to frame this in the paper.

---

## Appendix A — superseded specs (historical record only)

The rest of this document (below) is the **energy-preserving Galerkin coarsening** spec that this file
contained before 2026-07-22, itself a rewrite of an even earlier "Select–Reduce–Connect / relearn / inject"
spec. Both were hierarchical methods, built and gated, and closed as negative
(`sheafcovpool_design_log.md` §0, §pm8–§pm9): the apparent positive Galerkin result was traced to a numerical
instability artifact and reversed once fixed. **Do not implement anything below this line — it is not the
method used in the paper.** Kept only so the reasoning and the negative result remain traceable.

### A.1 The one principle (Galerkin, superseded)

Coarsen the sheaf with a **prolongation** `P` (lift coarse cochains to fine) and the **Galerkin operator**

```
L_c = P^*  L  P          (coarse Laplacian on the coarse cochain space)
```

Its defining property:

```
z^*  L_c  z  =  (P z)^*  L  (P z)  =  || δ (P z) ||²      for every coarse cochain z.
```

The coarse quadratic form equals the fine sheaf energy of the lifted signal by construction (Rayleigh-quotient
preservation, as in algebraic multigrid and HiSP), so `L_c` is automatically self-adjoint, PSD, and
energy-faithful — this was verified exactly in code (`rel_err 0.00e+00`). Despite that, the gate result it
produced turned out to be a training-dynamics artifact of a separate bug (see §pm9), not evidence the mechanism
helps.

### A.2 What the backbone provided for this line

| Field | Shape | Use |
|---|---|---|
| `x_star` | `[N, d]` | fine node cochain (coarse signal source) |
| `s_star` | `[E, d]` | fine edge cochain (stage 2) |
| `F_src`, `F_tgt` | `[E, d, d]` | conformal restriction maps — define `L0` and `δ0` |
| `raw_face_curl` | `[F_tot, d]` | unweighted fine curl (stage 2); weighted curl is diagnostic-only |

### A.3 Stage 1 — node-Galerkin (typed)

Type-blocked soft assignment `S ∈ R^{N×K}` (block-diagonal by graph & node type), prolongation `P0 z = S @ z`,
matrix-free coarse operator `L0_c(z) = S^T @ CoboundaryOps.L0(S @ z, F_src, F_tgt, ...)`, a few Euler diffusion
steps on `L0_c`, invariant per-level readout tokens concatenated across levels.

### A.4 Stage 2 — higher-order Galerkin (never reached — stage 1 was closed negative first)

Coarsen `L1 = δ0 δ0^* + δ1^* δ1` via an induced edge prolongation `P1` (product-assignment weights
`S[u,a]·S[v,b]`) and a face prolongation `P2`; diffuse a coarse edge cochain on `L1_c`. This was the intended
"novel part" (typed + higher-order Galerkin coarsening) but was never built — stage 1 alone was tested first per
the project's own staging rule, and it was closed negative before stage 2 was warranted.

### A.5 Why this line was abandoned (summary)

| Version | What it did | Result |
|---|---|---|
| V1 relearn | discard fine sheaf, re-learn on coarse graph | not real coarsening; weak/noisy; 20× slow |
| V1.5 inject | keep surviving maps on gated states | energy-inconsistent → hurt |
| Galerkin (pre-fix) | `L_c=P^* L P` on the existing fine operator | looked energy-preserving AND +0.14 macro-F1 on MUTAG |
| Galerkin (post `1/std` NaN fix) | same operator, numerically stabilized standardization | **result reversed to a tie** — the +0.14 was a training-instability artifact, not the mechanism |

### A.6 Test hooks that were used for this line

- Energy preservation: `z^* L0_c z == ‖δ0(P0 z)‖²` (verified exact).
- PSD: `z^* L0_c z ≥ 0`.
- Permutation equivariance, `O(d)` gauge invariance, type purity, batch isolation — same checks as §7 above,
  adapted to the coarse operator.
