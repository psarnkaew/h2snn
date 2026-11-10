# H2S-NN mini-batch training — design document (method and decisions)

**Status:** decisions M0–M4 agreed with the author on 2026-11-10 (design session, 01:53–03:36 JST). Not yet in the spec: §10 gives the spec text to apply when the mini-batch task card is written (the spec and the card are committed together). Open items are in §12.
**Supersedes:** the "Mini-batch design brief" of `session_handoff.md` (2026-11-09 21:50) for every point decided here.
**Status tags:** VERIFIED-CODE (file:line, branch `part-j`), VERIFIED-RUN (command and numbers in this session), DERIVED, UNVERIFIED, PROPOSAL.

---

## 1. Why, and what "mini-batch" means here
- Spec v1.8/1.9 §6 (F3): ogbn-mag, mag-year, H-Pokec (and IEEE-CIS-G if it does not fit) cannot be trained on the whole graph at once.
- **Receptive field** of a target t = every node whose input can change t's prediction. One application of the operator `A` reaches t's fact neighbours (1 hop) and every node of every cell containing t (up to `ℓ_max = 4` hops); K applications reach ≈ K·ℓ_max hops, i.e. almost the whole graph. No mini-batch can contain it, so **any mini-batch computes an approximation of the full-batch predictor** (as GraphSAGE's neighbour sampling does). Exact equality with full batch is not a goal (M0).
- **Batch size** = number of targets whose loss is summed in one optimizer step. With one scope per target (M1) it changes optimisation and memory only, never the predictor.

## 2. Terms
| Term | Meaning |
|---|---|
| Fact | One stored relation between two nodes, counting the two stored directions of HGB as one (D7). One fact = one oriented 1-cell. ACM: 547,872 rows of `link.dat` → 273,936 facts |
| Fact strength `c_e²` | `c_e = softplus(MLP([h_u, h_v, rel_emb])) + 0.1` (VERIFIED-CODE `sheaf_maps.py:261-273`), shared by both sides of the fact; the fact's term in the Laplacian is `c_e² ‖(rotation)x_u − (rotation)x_v‖²` |
| Cell | One WISP walk (length 2…ℓ_max) = one path 2-cell; weight `w_f m_f` (`w_f = exp(g)`, `m_f` = merged sampling weight) |
| Degree `D_v` | Σ `c_e²` over facts at v + Σ `w_f m_f (1 − 1/|f|)` over cells containing v (VERIFIED-CODE `vertex_path_laplacian.py:241-256`) |
| Context `c_v` | Per (relation, role): mean of the neighbours' `h₀` (VERIFIED-CODE `relational_context.py:28-44`) |
| Scope of t | The small graph cut out to predict t (§4) |

## 3. Training mode per dataset (M0)
- **Full batch** where the dataset fits in GPU memory; **local-scope mini-batch** otherwise. Decided per dataset from a memory measurement (TSUBAME phase 0) before the freeze; the paper states the mode of every dataset.
- Full batch is **not** a value of a tuning grid (it is a different predictor, available only on small datasets: a confound with C6 and an unequal tuning budget).
- The full-batch code path is unchanged (bit-identical).
- C5 reports, on ACM and DBLP, the gap between mini-batch and full batch; it is a reported number, not a criterion that selects the method.

## 4. The scope of a target (M1)
**One scope per target**, not one per batch. Scope of t:
1. node t;
2. **every** fact of t (including self-loop facts of t) and the other endpoint of each;
3. the cells of the WISP walks drawn **from t** (§6), with every node and every fact on them.
Nothing else (no "induced" facts between other scope nodes).

**Example (ACM).** t = paper P1 with facts to authors A1, A2, subject S and ≈ 90 terms (median facts per ACM paper: 93). Walks from P1 give cells such as P1–A1–P9 and P1–S–P20–A5. Scope nodes: P1, A1, A2, S, the terms, P9, P20, A5, …; scope facts: P1's facts + A1–P9, S–P20, P20–A5, …

**Properties (DERIVED):** the scope is a closed 2-complex (every cell brings its facts, every fact its two nodes; the closure edge of a cell is definitional, as in full batch), so the §3 sheaf on it is a complete cellular sheaf (functoriality holds by construction, Prop. 1).

**Size (VERIFIED-RUN, approximate):** own sampler in the device VM (uniform length 2–4, uniform action over stored relations, i.e. untied; 300 anchors; r = 42 / 65 / 12 / 42), HGB raw files. Facts per scope, median / p90 / max, minimal (this design) vs induced:
| | ACM | DBLP | IMDB | Freebase |
|---|---|---|---|---|
| minimal | 158 / 191 / 242 | 76 / 106 / 189 | 24 / 30 / 38 | 73 / 93 / 4,033 |
| induced | 1,154 / 1,567 / 2,148 | 102 / 147 / 380 | 25 / 33 / 41 | 139 / 199 / 12,593 |
A target that is itself a hub has a large scope (Freebase: a book with 5,548 facts); batching by a memory budget handles it (§7).

## 5. What is computed on a scope (M2, M3)
The unchanged §3 model runs on the scope as if the scope were the whole graph:
```
h₀ = TypeEncoder(x) for the scope's nodes                          # featureless types: embedding row of the GLOBAL node id (§11)
c  = RelCtx(h₀) over the scope's facts                              # scope-local
maps, gates, c_e, w_f = exp(g)   from [h₀, c] / h₀ as in full batch # same parameters
D  = degrees of the scope's facts and cells                         # scope-local ("S")
A  = G Δ̂ B on the scope;  x = Σ_{k=0..K} T_k(A − I) h₀ W_k         # K tuned, K ≥ 2
prediction of t = head([x_t, h₀_t])
```
- **Degrees and context from the scope (M3, "S").** Nothing is estimated or guessed; no global work per batch.
  - Consequence (stated as a limitation, §9): a node with many facts outside the scope counts more inside it than in the full graph. Measured hubs (≥ 100 facts) keep ≈ 2 % of their facts in an ACM scope and ≈ 0.3 % in Freebase, i.e. their per-neighbour influence is ≈ √50 ≈ 7× to ≈ 18× larger; the median non-anchor node keeps 3.4 % (ACM), 20 % (DBLP), 22 % (IMDB), 29 % (Freebase) of its facts (VERIFIED-RUN, same approximate run).
- **K (M2):** a tuned hyperparameter, **K ≥ 2** (R9: a degree-1 polynomial gives only a monotone low- or high-pass response; degree 2 can select a middle band). Range fixed with OPEN-5, the same for every arm. Tuned per dataset under that dataset's mode: in a scope, one application of `A` already reaches every scope node, so K > 2 only re-mixes inside the scope, while in full batch it reaches further (DERIVED); a K chosen on full-batch data is not transferred.

## 6. Walks: training and evaluation (M4)
- Walks are always drawn on the **full graph** by the existing WISP sampler (exact action and unvisited-neighbour counts, so `p(ω | t)` and Prop. S are unchanged); the scope is cut afterwards.
- **Per-anchor merge.** A scope contains only the walks drawn from t; cells are merged and `m_f` summed **over t's walks only**. (In full batch a cell drawn from two anchors, e.g. P1–A1–P9 from P1 and from P9, has `m_f` summed over both; in t's scope only t's share enters.) Prop. S holds per anchor: `E[L̂_P,t] = Σ_ω p(ω | t) L_{f(ω)}` (DERIVED from the per-walk estimator).
- **Evaluation draw (E1):** one draw for all anchors, base seed, made once and stored **per walk** (anchor, node sequence, fact sequence, IS weight); each scope takes its anchor's walks. So t's evaluation walks are exactly t's walks in the full-batch evaluation draw, and t's prediction does not depend on the batching. Storage for ogbn-mag ≈ 0.4 GB as int32 (UNVERIFIED estimate). Reason: today one generator serves all anchors in order (VERIFIED-CODE `wisp_sampler.py:148-152`), so drawing per batch would make a target's evaluation depend on which targets share its batch.
- *Correction of an earlier statement in this session:* "evaluation cells identical to full batch" holds for the **walks**; the merged weights differ by the per-anchor merge above.
- **Training draw:** a fresh draw for the batch's anchors, seed from (base seed, epoch, batch index); `proposal: uniform`. Prepared on the CPU in a background worker while the GPU trains (as Part I's overlap sampling).

## 7. Batches
- A batch = several targets; their scopes are placed side by side as one disjoint graph (node ids relabelled), so one call of `encode()` processes the batch. No fact or cell joins two scopes, so every target's prediction equals that of its scope alone (DERIVED; a test, §8).
- Loss: mean over the batch's targets (the per-task losses are unchanged: cross-entropy; IMDB per-genre BCE).
- Evaluation: all evaluation targets, in batches, with the stored evaluation draw.
- **Batch size (OPEN, §12):** targets per step chosen by a memory budget (e.g. a cap on scope nodes + facts + cell incidences per batch) rather than a fixed count, so a hub target does not overflow memory; a target above the budget forms its own batch. The budget is set from the phase-0 memory measurement.

## 8. Guarantees and validation (tests PROPOSED for the card; tolerances fixed when the card is written)
| Property | Status | Test |
|---|---|---|
| Scope is a closed subcomplex; sheaf complete | DERIVED | every cell's facts and every fact's nodes present; functoriality on scope cells |
| Props. 2–7, Lemma N, Theorem R, Prop. K hold on every scope (the scope is a graph) | DERIVED | λ_max(Δ̂) ≤ 2; filter gain ≤ 1 in `M = BG⁻¹`; gate zero ⇒ exact zero blocks (Prop. 6), on random scopes of real data |
| Batch = disjoint union ⇒ each target's prediction equals its scope alone | DERIVED | batch output vs one-scope-at-a-time, same seed |
| Prediction of t independent of batch composition and size (evaluation) | follows from M1 + E1 | evaluate with two batch sizes and shuffled order: identical predictions |
| Whole-graph harness | DERIVED | a scope built with **all** facts and **all** anchors' walks merged as in full batch reproduces the full-batch logits bit for bit (checks the plumbing; not the method) |
| Per-anchor unbiasedness | DERIVED | Monte-Carlo mean of `L̂_P,t` vs exact enumeration on a small graph (as T-H2) |
| t's evaluation walks = t's walks in the full evaluation draw | design | exact equality |
| Featureless types read the global node id | design | mutation: local id → targeted test fails |
| Cost | — | memory and time per batch vs scope size; C5 log–log slopes on the cluster |

## 9. Limitations (paper text, draft)
"On graphs too large for full-batch training, each target is predicted by the same model, with the same parameters, applied to its local scope: its facts and the cells sampled from it. Degrees, relational context and normalisation are those of the scope, so a node with many facts outside the scope weighs more there than in the full graph, and information more than one fact away that no sampled cell carries is not seen. Training and evaluation use the same rule. We report the gap to full-batch training on ACM and DBLP."

## 10. Spec text to apply with the task card (author approves the wording then)
- **R7:** "… no per-edge d×d tensors; a mini-batch applies the same model (same rule, same parameters) to each target's closed local subcomplex (§6), whose degrees, context and normalisation are its own."
- **R10:** "… computed once" holds per complex: one sheaf in full batch; one sheaf per scope in mini-batch.
- **Lemma R (restated):** "A target's scope is a closed subcomplex; the §3 construction on it is a cellular sheaf and Props. 2–7 hold on it. It is not the restriction of the full-graph sheaf (context and degrees are the scope's), and it equals full batch only if the scope covers the receptive field."
- **§3 K row / OPEN-4 closed:** "K is a tuned hyperparameter with K ≥ 2 (R9: degree 1 has a monotone response); range in OPEN-5."
- **§4 limits:** L5 rewritten as §9 above (new L12 if L5 is kept).
- **§6:** training mode per dataset (§3), scope (§4), walks (§6), batches (§7); `K` in the search space with K ≥ 2.

## 11. Implementation notes (VERIFIED-CODE on `part-j`, for the card)
- `H2SNN.encode(x, node_type, edge_index, edge_type, face_groups, num_nodes)` accepts any graph (`h2snn_model.py:1100`); the scope can be passed as a graph with relabelled ids.
- **Featureless types:** `TypeSpecificNodeEncoder` maps a node to its embedding row by its position in the flattened id space minus the type offset (`node_encoder.py:101-104`), and the all-featureless fallback returns the whole embedding table (`h2snn_model.py:728-731`). With relabelled scope ids both would read wrong rows: the encoder must receive global node ids.
- Sampler: `sample_wisp_walks(graph, node_type, anchors, walks_per_anchor, max_length, generator, proposal)` (`wisp_sampler.py:141`) already takes an anchor subset; `merge_wisp_cells` (`:284`) merges across all walks of a draw, so per-anchor merging needs the anchor of each walk.
- Trainer: full-batch only; one optimizer step per epoch (`trainer.py:206-214`); training draw per epoch with seed `cfg.seed + epoch` (`:66-95`).
- Tuning search space today includes `K ∈ {2,…,6}` (`scripts/run_h2snn_wandb_tuning.py:292`), consistent with M2 (K ≥ 2); the range is fixed with OPEN-5.

## 12. Open (next)
1. Batch size and memory budget per batch: needs TSUBAME facts (A100 40 or 80 GB, job time limit, quota; checkpoint/resume across jobs).
2. Fix the test tolerances of §8 in the card (pre-registration).
3. E14 (`r` is 0 or the calibrated value in every config) as the card's first unit.
4. Rules for `r` and `ℓ_max` on datasets without a literature-schema budget (Freebase, H2GB); pre-registered before their first run.
5. Interface general enough for link-prediction anchors (LP card; deferred if LP leaves the IJCAI scope).
6. Pending the author's order when compute is available (M2a): re-run Part K's gate (ii) at the lowest K of the range (β = log 512 and 0, n = 2000, 3 seeds, same criteria; it was run at K = 4).

## 13. Decision log
| # | Decision (author, 2026-11-10) | Reason |
|---|---|---|
| M0 | Full batch where it fits, otherwise local-scope mini-batch; stated per dataset; not a tuning value | Standard practice; closeness to full batch is not a requirement; avoids a dataset-size confound in C6 |
| M1 | One scope per target; minimal scope (t's facts + t's cells) | Prediction well defined and independent of batching; size predictable; multi-hop evidence comes through cells (H); ACM ≈ 7× cheaper than induced |
| M2 | K tuned, K ≥ 2 (03:28; supersedes "K = 2" of 03:01) | R9 lower bound; standard hyperparameter |
| M2a | Gate (ii) re-run waits for the author's order | Author |
| M3 | Degrees and context from the scope ("S") | The scope is a graph, so all of §4 holds verbatim; no new component (§0 rule 1); nothing guessed; no global per-batch work |
| M4 | Evaluation draw stored once (E1); fresh training draw per batch | t's evaluation walks equal its full-batch walks; prediction independent of batching; sampler unchanged |

**Considered and not adopted**
- **Per-batch union scope:** the prediction of t depends on which targets share its batch; batch size changes the model.
- **Full batch as a value of batch size in the grid:** a different predictor; see M0.
- **S+D** (scope degrees rescaled by full-graph fact counts, `D' = D_sub · n_full/n_sub`): closer to full batch only if ρ ≤ √(n_full/n_sub), ρ = mean `c²` of a node's visible facts ÷ mean over all its facts (DERIVED; checked on 100,000 random cases, 0 mismatches). ρ depends on learned maps and data; the option serves closeness, which is not a requirement; Lemma N fails at the scope boundary. Stability holds for any `D' ≥ D_sub` (VERIFIED-RUN: λ_max 1.997 ≤ 2 over 200 random connection Laplacians).
- **R / E** (context from full neighbourhoods; degrees estimated or exact): reading every neighbour of the scope's nodes has no bound per batch on large graphs (hubs of 50,200 facts in Freebase); E needs the full graph's context at every step.
- **T** (receiver-side normalisation `G D⁻¹ L B` + every cell through t + full-neighbourhood context): makes t's first application exact (DERIVED; with `M = B G⁻¹ D` the spectrum is real, λ ≤ 0.955 · 2 max gb, filter gain ≤ 0.92 over 200 random sheaves, VERIFIED-RUN), but changes the full-batch model, scales worse (all anchors drawn every epoch, hub neighbourhoods read) and serves closeness only.
- **Layer-wise neighbour sampling (GraphSAGE):** the model applies one operator K times; sampling per layer gives a different graph per application (not one sheaf).
- **Graph partition (Cluster-GCN):** walks that cross partitions are cut, biasing WISP's estimator (R8).
