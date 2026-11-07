# H2S-NN — Possible reviewer questions and answers

**Date:** 2026-11-07. **Model described:** v2 with D11 (plan `H2S-NN_v2_plan.md` §12).
**Sources:** the WSDM 2027 reviews of paper 850 (`docs/critical_reviewer_reviews.md`), the plan, the D1/D4/D10 reports, and `references/` (HetSheaf, Sheaf Attention Networks, CSNN, DNSD).

## How to use this document
- **Part 1** answers every concern the real reviewers raised. **Part 2** answers questions a reviewer is likely to ask about the new model.
- Each item has a **rebuttal answer** (text you can adapt) and a **status line**. Status tags are the plan's: `VERIFIED-RUN`, `VERIFIED-CODE`, `VERIFIED-SOURCE`, `DERIVED`, `PENDING` (needs an experiment that has not run), `PROPOSAL`.
- **Rule:** do not use an answer whose status is `PENDING` until the result exists. Where an answer needs a number we do not have yet, the text says `[result]`.

---

## Part 1 — Answers to the WSDM 2027 reviews

### 1.1 "The face term reaches nodes only through the closure edges, so it is an endpoint relation in disguise." (R2 W3, R2 comment 5, meta-review)
**This was the most damaging criticism, and it was correct about the submitted model.**

> We agree, and we changed the operator. In the submitted model the 2-cell term acted on edge states and reached nodes only through the closure edge. We prove that this is not a defect of the implementation but a property of every linear Hodge-type flow: for any polynomial or heat flow of the Hodge–Dirac operator on node, edge and face cochains with `δ₁δ₀ = 0`, the face-to-node block is exactly zero (Lemma A′). The revised model therefore couples each path 2-cell **directly to every node on the path** (endpoints and intermediate nodes) through the node→2-cell restriction maps of the same sheaf. These maps are not free parameters: functoriality forces them to be the ordered parallel transport along the path (Prop. 1). The resulting operator is the vertex–2-cell sheaf Laplacian. One diffusion step already moves information between an endpoint and the intermediate node, which an endpoint edge cannot do. We isolate the 2-cell's contribution with the ablations in §1.2.

Status: Lemma A′ `DERIVED` + numerical check (block ≤ 1e-11 for `Dᵏ`, k = 1..8). Direct coupling implemented and tested (D4 test T6: non-zero one-step Jacobian blocks endpoint ↔ intermediate node; mutation "path term on endpoints only" caught) — `VERIFIED-RUN` (`docs/H2S-NN_D4_report.md`).

### 1.2 "The contribution of the 2-cells is not isolated from selection, weighting and capacity." (R4, R2 comment 5, meta-review)
> We added controlled ablations with the same backbone, the same sampled paths and a matched parameter budget. Each row answers one question:
>
> | Variant | Question |
> |---|---|
> | no path cells (`γ = 0`) | Do path cells matter at all? |
> | endpoint meta-path edges (same paths, endpoints only) | Do intermediate nodes matter? (R4's "endpoint-collapsed view") |
> | holonomy closure edges (the submitted model's shortcut) | Does the direct 2-cell coupling add anything over a path-conditioned endpoint edge? (R2 comment 5) |
> | free per-incidence maps instead of forced transport | Does the sheaf condition matter, or would a hypergraph do? |
> | uniform path weights | Does instance weighting matter? (R4) |
> | `h₀`-only readout | Is diffusion used at all? (capacity control) |
>
> The criterion was fixed before running: if `γ = 0` or endpoint edges are within one standard deviation of the full model on the standard and the heterophilic tiers, the 2-cell claim fails, and we would report that rather than add components.

Status: all switches implemented and tested (D4 T11) — `VERIFIED-RUN`. Results — `PENDING` (core-claim screen, plan §12.6; then Tier 0 with 10 seeds and paired tests). Parameter counts per row must be reported.

### 1.3 "Scalability is not demonstrated." (R1 W1/Q1, R3 W3/Q2/Q3, R4, meta-review)
> We rebuilt the model for scale. (i) Restriction maps are factorized into a shared per-relation rotation and k node-dependent Householder reflections and are never stored as d×d matrices (15.75× less map storage at d = 32, k = 2). (ii) The edge cochain, the edge Laplacian and the dense face maps were removed. (iii) The path term costs 2L transport steps per path instead of L(L+1). (iv) Path instances come from a vectorized relation- and direction-constrained random walk with a stated sampling distribution (no enumeration), resampled every epoch. On ACM at d = 32 the model has [302,761] parameters versus 1.62M and peaks at [1.49 GiB]. We report, per dataset, the number of schemas, cells, incidences, sampling time per draw, time per epoch, time-to-accuracy and peak memory, plus scaling curves over graph size (ogbn-mag subgraphs at 10/25/50/100%), path length (ℓ ∈ {2, 3, 4}) and walks per anchor (r ∈ {2, 4, 8, 16}).

Status: map storage, parameter counts, peak memory — `VERIFIED-RUN` (D1, D4, D10 reports; RTX 4060 Laptop 8 GB). The submitted paper's 9,260 MB / 3.011 s per epoch were measured with a different setup: re-measure v1 and v2 **on the same GPU** before putting both in one table. Scaling curves, Tier 2 (ogbn-mag, mag-year, H-Pokec), mini-batch training — `PENDING`.

### 1.4 "The heterophilic setting is not established; only IMDB is strongly heterophilic." (R3 W1, R4, R2 W4, meta-review)
> We agree that DBLP and ACM are mostly homophilic under our own metric. We now (i) add heterophilic heterogeneous benchmarks: Freebase (HGB; MLH 0.550 on the B–B schema) and the H2GB sets (IEEE-CIS-G, ogbn-mag, mag-year, H-Pokec); (ii) run a controlled rewiring of ACM and DBLP that raises heterophily step by step, with a homophilic control, so gains can be plotted against heterophily; (iii) report heterophily per relation and per meta-path for every dataset, under the exact relations the model uses; (iv) recompute IMDB's heterophily with definitions that handle its multi-label targets.

Status: Freebase statistic `VERIFIED-REPO-DOC` (`docs/pyg_dataset_eda_homophily_report.html`). Benchmarks, rewiring and stratified results — `PENDING` (plan §3.3, Tier 1–2; fix spec H5, H6).

### 1.5 "Show a case where reducing a path to its endpoints fails." (R2 comment 1)
> Two instances of the same schema with the same endpoints, for example A₁–P–A₂ and A₁–P′–A₂, are one endpoint edge in an endpoint view but two different 2-cells in our model: their transports pass through different intermediate nodes, so their holonomies differ. We give (i) a synthetic task in which the label is determined by the intermediate node, where endpoint views provably cannot exceed chance and the 2-cell model can, and (ii) on real data, the measured spread of holonomies across instances of the same schema and endpoint pair.

Status: holonomy diversity across instances is tested (D1 acceptance test) — `VERIFIED-RUN`. Synthetic task — `PROPOSAL`, not built.

### 1.6 "Graph-classification datasets are not naturally heterogeneous; link-prediction datasets partly have one node type." (R3 W2, R4)
> We agree. TU datasets with labels read as types are typed graphs, not heterogeneous information networks. We present node classification on heterogeneous benchmarks as the main evidence, report heterophily statistics for every graph- and link-level dataset, and state the graph-level results as a secondary check of the same encoder rather than evidence for the heterogeneous-heterophilic claim. We also found and fixed a loader bug in MUTAG that assigned wrong bond types to 1,771 of 7,442 bonds; all MUTAG results (ours and typed baselines) are rerun.

Status: MUTAG bug and fix — `VERIFIED-RUN` (commit `2322d28`). Rerun results, heterophily statistics for graph/link datasets — `PENDING`. Scope decision for graph classification — author decision.

### 1.7 "Baseline protocol is unclear; HetSheaf numbers are quoted." (R1 W2/W3/Q2, R2 W2, R4)
> Every baseline is rerun under one protocol: the same splits and seeds, the same facts, the same tuning budget per method (search space and number of trials listed in the appendix), and early stopping on validation only. Baselines receive the graph in their standard stored form with reverse relations; our model receives the same facts with each reverse pair tied into one oriented cell. HetSheaf has no public code; its numbers are reported only in a separate, clearly labelled non-controlled table on its native setup, and no conclusion is drawn from them.

Status: same-facts protocol decided (plan §9.5, option A) — implementation in Part F U1. Baseline provenance audit (some Table 1 baseline cells match published values and appear in no run output in the repo — fix spec H8) — **must be resolved before any resubmission**; `PENDING`.

### 1.8 "Standard deviations do not match the stated 10-fold × 5 seeds; some graph-classification values look like single splits." (R1 Q3, R2 W1)
> We recomputed every entry from the raw run outputs; the appendix states for each table the number of splits and seeds, how the standard deviation is taken, and provides the per-run values.

Status: `PENDING`. Do not answer until the recomputation is done.

### 1.9 AMPSS: too many settings, protocol, leakage, which schemas are selected (R2 W5, R3 Q3, R4)
> Schema selection is redesigned (D2). Requirements fixed in advance: no enumeration of instances, path-aware scoring (no endpoint proxies), at most three settings, training labels only, and every selected schema reported per dataset with significance against fixed literature schemas and a comparison with LMSPS. Instances inside a schema are now drawn by a random walk with a stated distribution and an unbiased estimator of the path operator; its two settings (maximum length ℓ and walks per anchor r) get a sensitivity analysis. We also report whether the selected schemas shift as heterophily increases (R4).

Status: D2 mechanism — **open** (plan §2). Walk sampler, unbiasedness test — `VERIFIED-RUN` (D7–D9). Sensitivity and schema shift — `PENDING`.

### 1.10 Introduction does not connect AMPSS and the readout to the problem (R2 comment 2)
> The introduction now derives each component from one stated assumption (Part 2, Q2.1). SheafCovPool is removed: it read edge states that the new model no longer has, and it reduced each node to a 3×3 Gram matrix, which loses information. Graph classification uses sum pooling, which is permutation-invariant and can be injective on multisets (Xu et al., 2019).

Status: readout change — `VERIFIED-CODE` (plan §4.8).

### 1.11 Figures, notation, template (R2 comments 3–4, R4)
> Figures 3–4 are redrawn with the main-text notation; the 2-cell figure shows the node→2-cell maps, the 2-cell state and the closure edge as the boundary only. Template metadata and page count are corrected.

Status: `PENDING` (writing task).

---

## Part 2 — Questions about the new model

### Design rationale (use this order in the paper)

**Q2.1 What do you assume about the data, and how does each component follow?**
> - **A1 Structured disagreement.** Connected nodes are related by a relation- and node-dependent transformation, not by equality; heterophily is systematic difference, not noise. → a cellular sheaf with restriction maps that depend on relation, side and both nodes.
> - **A2 Path semantics.** A metapath instance carries information its endpoints lack (intermediate nodes, relation order). → path-supported 2-cells.
> - **A3 Coherence.** Path-level information must not contradict edge-level information. → node→2-cell maps forced by functoriality (Prop. 1); global sections preserved (Prop. 2).
> - **A4 Factorized trust.** Influence of v on u = receptiveness of u × symmetric compatibility of the pair × reliability of v. Direction comes from node roles; the pair contributes only symmetric compatibility. → listen/broadcast gates on a symmetric sheaf Laplacian.
> - **A5 Geometry from attributes.** The geometry (maps, gates, path weights) can be read from the encoded attributes and types once; diffusion propagates over it.
>
> Properties pursued: P1 the update minimises a stated objective; P2 stable for any step size; P3 coherent across edges and 2-cells; P4 directional influence; P5 linear cost in edges and incidences.

Status: `DERIVED` (plan §12). A5 is the weakest assumption; say so.

**Q2.2 What is the gap with respect to prior work?**
> | Family | Missing |
> |---|---|
> | HAN, HGT, Simple-HGN, SeHGNN | Aggregation assumes similarity (no A1) |
> | MAGNN | Intermediate nodes encoded, but nothing makes paths consistent with edges (no A3) |
> | NSD, HetSheaf | Structured disagreement on edges only (no A2); symmetric influence (no A4) |
> | Sheaf hypergraph networks | Higher-order cells with free maps, which destroy global sections (no A3) |
> | SheafAN / Res-SheafAN | Directional attention on edges; no objective (no P1); no higher-order cells |
> | CSNN | Directional sheaf with complex spectrum; no objective; no higher-order cells |
>
> No previous model combines A1–A4 with P1–P2.

Status: HetSheaf, SheafAN, CSNN, DNSD `VERIFIED-SOURCE` (`references/`). HAN/HGT/Simple-HGN/SeHGNN/MAGNN characterisations — `UNVERIFIED` until each paper is opened and cited.

### 2-cells and sheaf theory

**Q2.3 Isn't a path 2-cell just a hyperedge?**
> The computation has the form of a hyperedge with a hub, but (i) its maps are determined by the boundary edges, not learned freely, (ii) they depend on the order of relations along the path, and (iii) they preserve every global section of the edge sheaf. With free per-incidence maps the kernel generically collapses (Prop. 2), so a sheaf-hypergraph model can contradict the graph sheaf; ours cannot. The "free maps" ablation measures the difference.

Status: `DERIVED`; numerical check (dim ker 3 / 3 / 0) and D4 T4 with negative control — `VERIFIED-RUN`.

**Q2.4 Why not use the Hodge Laplacian of the 2-complex?**
> Lemma A′: under any linear Hodge flow, face information never reaches nodes. The node operator we use is the vertex–2-cell sheaf Laplacian, which is the 0-Laplacian of the pullback sheaf on the star expansion (Prop. 2); we do not call it the Hodge Laplacian.

Status: `DERIVED` + numerical check.

**Q2.5 Where do the 2-cells come from? They are not in the data.**
> Correct: the data is a graph, and the 2-cells are a modelling choice. Each is a metapath instance drawn by a random walk constrained by relation and direction, with a stated distribution; a path and its reversal are one cell, duplicates are merged by summing weights, and the resulting path operator is an unbiased estimate of a stated target. Training resamples every epoch; evaluation uses one fixed draw.

Status: `VERIFIED-RUN` (D8 tests T7–T9, with negative controls).

**Q2.6 Does your model "make the graph a sheaf"?**
> No. Any choice of stalks and maps that composes correctly (vertex → edge → face) is already a cellular sheaf; the transport construction guarantees it. The model diffuses node states toward global sections of that sheaf.

Status: wording rule (plan §4.5). Never write "the model makes the graph a sheaf".

**Q2.7 How are node types, attributes and relation types handled?**
> Each node has its own attributes and a type. A type-specific encoder maps attributes into one shared stalk space (featureless types get a learned vector per node), and a learned type embedding is added. Relation types enter the restriction maps: one rotation per (relation, side) plus node-dependent reflections. All types share the stalk dimension d because the restriction maps must be invertible for the forced transport.

Status: `VERIFIED-CODE` (`src/models/node_encoder.py`).

**Q2.8 Why orthogonal maps? HetSheaf and DNSD find diagonal maps at least as good.**
> Path cells need maps that do not commute: with diagonal maps the transport along a path does not depend on the order of relations, and the relation-order part of A2 is lost. Diagonal maps are an ablation row; if they match or beat orthogonal maps, we drop the relation-order claim.

Status: `DERIVED`; ablation — `PENDING`.

**Q2.9 Reflections can flip a direction; isn't that the heterophily mechanism?**
> Both sides of an edge have the same determinant sign, so every edge transport and every holonomy is a rotation (SO(d)). A transport cannot flip one direction; it can reverse directions in pairs (a 180° rotation in a plane).

Status: `DERIVED` (D1b). Use this wording.

**Q2.10 You use one scale per edge. What is lost?**
> Per-side scales split into a node rescaling, which the degree normalisation cancels exactly, and "scale holonomy" around cycles, which is lost. The latter is exactly what made path transports blow up numerically; with one scale per edge, every transport is orthogonal at any path length. An ablation with per-side scales measures the cost.

Status: `DERIVED` + numerical check (plan §1.7); ablation — `PENDING`.

### Directional influence (the attention question)

**Q2.11 Why is influence not free in each direction, as in attention?**
> Our assumption A4 is that directional trust comes from node roles. We prove that A4 is **equivalent** to the existence of an objective:
>
> **Proposition 4.** Let `w(u←v) > 0` be the influence of v on u, through edges and 2-cells, with transports satisfying `Q_vu = Q_uvᵀ`. The update `x ← x − ηAx` is a gradient flow of a quadratic sheaf energy in a positive per-node metric **if and only if** `w` satisfies detailed balance (`π_u w(u←v) = π_v w(v←u)` for some `π > 0`), **if and only if** `w(u←v) = g_u · s_uv · b_v` with `s` symmetric.
> *Proof.* `A = M⁻¹L` with `L` symmetric forces `m_u w(u←v) = m_v w(v←u)`; conversely set `s_uv = π_u w(u←v)`, `g = 1/π`, `b = 1`. The cycle form is Kolmogorov's criterion.
>
> So D11's gates are not one convenient choice: they are the most general directional influence that keeps an objective function, a real spectrum and known fixed points.

Status: `DERIVED`; numerical check on 100 random sheaves with path cells — D11 form: detailed-balance residual ≤ 2e-14, real spectrum; free per-direction weights: residual ≥ 0.78, max |Im λ| up to 0.32 — `VERIFIED-RUN` (scratch script, to be turned into test T-F3b).

**Q2.12 Why not SheafAN / Res-SheafAN, which also use sheaf maps on edges?**
> The two differ in one modelling assumption and in what can be proved, not in a settled accuracy advantage.
> 1. **The assumption.** Attention lets each receiver weight each neighbour freely (pair-specific direction). Our model assumes direction comes from node roles (A4): receptiveness of the receiver, reliability of the sender, and a symmetric compatibility of the pair.
> 2. **What the assumption buys.** By Prop. 4, A4 holds exactly when the update is gradient descent on a sheaf disagreement energy: real spectrum, characterised fixed points, a readable listen/broadcast gate per node. This is a guarantee and an interpretation, not an accuracy claim; with a few diffusion steps the model does not reach its fixed points.
> 3. **What it may cost.** Pair-specific relevance is what attention models exploit, and attention-based models (e.g. Simple-HGN) are strong on HGB. So A4 may be wrong for real data.
> 4. **How we decide.** We test A4 directly: the same sheaf, the same 2-cells and the same single diffusion block, with the gates replaced by attention over edges, 2-cells and a self-loop. The consequence of each outcome was fixed before running. Separately, SheafAN has no higher-order cells.
>
> HetSheaf reports SheafAN well below Laplacian-based sheaf models on IMDB (54–56 vs 61–63 Macro-F1); this is one data point for non-residual SheafAN in their implementation, not evidence about Res-SheafAN.

Status: Prop. 4 `DERIVED` + numerical check; SheafAN Eq. 3–6 and HetSheaf Table 11 `VERIFIED-SOURCE`; Simple-HGN's HGB standing — cite after opening the HGB paper (`UNVERIFIED` here). A4 test result — `PENDING` (Part F, `coupling: pair_attention`).
**Do not claim:** that attention "cannot stop listening" (SheafAN's adjacency includes self-loops, Eq. 3); that SheafAN is "the worst" family in HetSheaf (true only on IMDB); that only our model combines edges and 2-cells in one rule (a single softmax over self, edges and cells does it too, and also preserves global sections); that our mini-batches are exact (neighbour sampling truncates the receptive field in both models; we only avoid the per-batch softmax renormalisation); that having an energy improves accuracy.

**Q2.13 Why not CSNN's directional sheaf?**
> CSNN gives each node separate send and receive maps and composes them; that operator can have a complex spectrum (their §5), and there is no energy. Our gates give per-node send/receive strength with a real spectrum and an energy.

Status: `VERIFIED-SOURCE` (CSNN); complex-spectrum control in D10 T3 — `VERIFIED-RUN`.

**Q2.14 What can your model not express?**
> Cyclic trust (A over B, B over C, C over A) and pair-specific direction beyond node roles. Excluding them is our assumption A4, not an established fact; we test it directly (Q2.12) and report the outcome either way.

Status: exclusion `DERIVED` (Prop. 4); whether real data needs them — `PENDING`.

**Q2.15 Learned node-wise diffusion rates already exist. What is new about the gates?**
> The gates alone are not the contribution. The contribution is that listen/broadcast cooperation is realised as a sheaf gradient flow in a learned metric, with a stability guarantee and a characterisation (Prop. 4) showing it is the most general such form.

Status: `PROPOSAL`; a literature check on node-wise diffusion rates is required before writing this.

**Q2.16 Do the gates actually learn anything?**
> We report their trajectories and their distribution per node type after training.

Status: **at risk.** D10 smoke run: gates stayed ≈ 0.95 over 20 epochs. The learning fix (Part F U2) is untested. If the gates still do not move, the paper must say they act as fixed hyperparameters.

### Theory, stability, expressivity

**Q2.17 Is the model stable? Does it oversmooth?**
> The normalised operator has eigenvalues in `[0, 2·max g b]`, so any step size in (0, 1) is stable and the energy never increases (Prop. 3 with D11). Diffusion runs a small fixed number of steps and the readout uses `[x_K, h₀]`, so the encoder's own features are always available.

Status: `DERIVED` + numerical checks (D4 T5, D10 T3, Part F T-F3 `PENDING`).

**Q2.18 One linear diffusion block is weak compared with NSD's layers.**
> Correct, this is assumption A5. The `h₀`-only control measures whether diffusion adds anything; if the full model is within one standard deviation of it, we move to a multi-block model with a parameter-matched comparison.

Status: `PENDING` (trigger in plan §12.6).

**Q2.19 Why tie a relation with its reverse (D7)? Is the comparison with baselines fair?**
> HGB stores every fact twice (once per direction) so that message passing reaches both ends. A sheaf Laplacian is symmetric, so storing the fact twice gives two cells with unrelated maps for the same fact, halves the consistent space and adds no direction information (one oriented cell is already direction-aware). We define the inverse relation by exact edge-set equality, with no thresholds or settings. All methods receive the same facts; baselines in their standard stored form.

Status: `VERIFIED-RUN` (all 10 HGB relation pairs are exact mirrors; plan §9.1). The untied ablation is a control, never tuned.

**Q2.20 Was link prediction evaluated without leakage?**
> Splits are made per fact: a held-out link removes the fact in both directions, and we check that no held-out fact appears in the message graph or on any sampled path.

Status: fix `VERIFIED-RUN` (on a fixture, the old split leaked the reverse of 8 of 21 held-out facts; now 0). Earlier link-prediction numbers of all methods may be inflated on datasets with mirror relations: rerun everything.

**Q2.21 What is the complexity?**
> Per diffusion step, time `O((|E| + I)·(d² + k·d))`, where I is the number of 2-cell incidences (2L transport steps per path, each a shared per-relation rotation plus k reflections). Storage for the maps is `O(|E|·k·d + |R|·d²)`: no d×d matrix is stored per edge. Maps are computed once per forward pass. Time is linear in edges and incidences.

Status: `DERIVED` from D1/D9; confirm with the scaling curves (`PENDING`).

---

## Part 3 — What must exist before resubmission (checklist)
- [ ] Core-claim screen and Tier 0 ablations (Q1.2), with parameter counts.
- [ ] A4 test: `pair_attention` vs D11 (Q2.12).
- [ ] Heterophilic benchmarks, rewiring, per-meta-path heterophily (Q1.4).
- [ ] Same-GPU efficiency table and scaling curves (Q1.3).
- [ ] Baseline provenance resolved; every baseline rerun under one protocol (Q1.7, Q1.8).
- [ ] MUTAG and link-prediction reruns (Q1.6, Q2.20).
- [ ] D2 mechanism decided and its selected schemas reported (Q1.9).
- [ ] Literature check on node-wise diffusion rates (Q2.15) and on each HGNN characterisation (Q2.2).
- [ ] Gate behaviour after the learning fix (Q2.16).
