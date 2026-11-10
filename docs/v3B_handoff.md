# v3-B handoff — start here in a new session (finalize the model against the reviews)

Written 2026-11-10 21:50 JST at the end of the v3 design session. English for documents; talk to the author in Thai. **Treat every statement as a lead to re-check**, but the items marked *decided* were agreed by the author in this session.

## 0. Read in this order
1. `claude/working_preferences.md` (rules; no flattery; plain words; formula-level prior-art checks; warning signals stop work).
2. The original reviews: `docs/critical_reviewer_reviews.md` (R1–R4 + meta-review). Read R2 in full — its summary and strengths decide how W3 must be read (§2 below).
3. **`docs/v3B_design.html`** (project copy `claude/v3B_design.html`) — the implementation-ready design of the chosen model. Its §15 lists the open decisions.
4. `docs/v3_A_vs_B.html` (project copy `claude/v3_A_vs_B.html`) — comparison of A (v2 + rewrite) and B, with the proofs in §9 (sheaf, flat cells, kernel, class-separating sections, limits, intermediate nodes).
5. ChatGPT's analysis `C:/Users/Acer/Downloads/H2S-NN v3_ ทาง A กับทาง B.html` §17–22 — use §20 (anchored solve: uniqueness, convergence, injectivity) and §19's limits (−I holonomy cycle, automorphic nodes). Not copied into the repo.
6. Spec v1.10 (`docs/H2S-NN_methodology_spec.md`), `docs/minibatch_design_decisions.md` (M0–M4), `docs/session_handoff.md` (state of the code and data work).
7. Evidence scripts: `docs/v3_evidence/*.py` (numpy; run with `python3 <file>` inside that folder).

## 1. Decisions of this session (author, 2026-11-10)
| # | Decision |
|---|---|
| V1 | **Model = B**: states on nodes, observed facts and 2-cells; path-cell information reaches nodes only through the path's observed facts. |
| V2 | **Closure edge is definitional only**: it makes each path a genuine 2-cell (CW) and carries the proof (δ₁δ₀ = 0 forces transport); no state, no energy term, no route (Theorem 1b). |
| V3 | **No variant C** (no direct node–cell term): R2 praises "2-cells + sheaf-based edge diffusion"; W3 is about the route into nodes ("only through the closure edge"), not about the face acting on edges. |
| V4 | **Propagation = one anchored quadratic solve** (ChatGPT §20): `min ½‖x−h‖² + τ/2 YᵀLY`, head reads `[h_v, x*_v]`. |
| V5 | **Not the Hodge Laplacian of the complex** (Lemma A′: δ₀ᵀδ₁ᵀ = 0 blocks cell→node). Do not call the propagation "Sheaf-Hodge diffusion". |
| V6 | Fair controls share anchor and readout: `hub_A` (v2's L_P), `closure_only` (R2 comment 5), `no_path_cells`, weight ladder, `free_cell_maps`. |
| V7 | Kept from v2: WISP, factorized maps computed once, matrix-free transport, D7 facts, mini-batch M0–M4, all Part J data/protocol fixes. |
| V8 | Empirical toy numbers are algebra checks only; design is decided by proofs and the reviewers' requirements. |

**Superseded:** `claude/v3_redesign_brief.md` (its v3 mixed a signed node→edge with an unsigned lift: not an adjoint pair, face blind to nodes) and the "drop 2D / no 2-cell / no face" decision in `docs/paper_rewrite_brief.md` §5, §7 (06:45). The slides `Claude outputs/progress_2026-11-11.pptx`, `scripts.md` and `H2S-NN/paper/ijcai27/h2snn_ijcai27.tex` were written on that brief (2026-11-09 22:41–22:46) and **do not describe v3-B** — check before the 2026-11-11 meeting.

## 2. How to read the reviews (verified against the original PDFs)
- R2 summary: "path-supported 2-cells **closed by closure edges** and **guarantees a cochain complex** through transport-based edge-to-face maps" — accepted, praised.
- R2 strength 2: "lift … to 2-cells … and **combine them with sheaf-based edge diffusion**" — the face acting on edges is wanted.
- R2 W3 / meta-review: the defect is that face information "is reflected in the node representations **only through the closure edge**". R2 strength 1 (geometry) and W3 (information route) do not contradict: under Hodge propagation the praised δ₁δ₀ = 0 is exactly what blocks the route (Lemma A′). Keep the geometry, change the propagation.
- R3 summary: "more than a straightforward combination of an existing heterogeneous GNN and a sheaf layer" — A (hub) risks reading as HetSheaf + MAGNN; B keeps the 2-cell as a real level.
- Requested ablations: R2 comment 5 (closure-only vs face), R4 (endpoint-collapsed views, path instances without 2-cell diffusion, uniform weights, equal paths, full).

## 3. Status review

### Done (code/data; see `docs/session_handoff.md` for commits)
- Data and protocol: Part J incl. J4–J6 (heterophily ℋ²/MLH, LP datasets, HeaRT negatives, F17 feature rule, loader guard) on `part-j` (last `c41bf28`). Uncommitted doc edits on `part-j` belong to other sessions: do not touch.
- WISP (Parts H, I, K): sampler, unbiased estimator, `w_f = exp(g)`, overlap sampling, gate (ii) met; learned selection not shown (S1′), S2′/S3 deferred to the cluster.
- Maps: factorized `F = R_{ψ,role} H_k(u)` with `c_e` (D1, D12), context `c`, fact layer D7; `PathTransportOperator`, `HolonomyEdgeTerm` exist (v2).
- Mini-batch: designed (M0–M4), not implemented.
- Theory for B: proofs + numpy checks (this session): Theorems 1–9 in `v3B_design.html` §12.

### Still open from the old plan (unchanged by v3-B)
- Mini-batch implementation (card), with E14 (`r` is 0 or calibrated) as first unit.
- WISP for link prediction (LP card); OPEN-5 (tuning budget); freeze; TSUBAME phase 0 (memory, H2GB data, ℋ² reproduction).
- G0b hypotheses (deferred to cluster); experiment-table audit before the freeze.
- Pre-registrations of S2′, S3.
- Paper rewrite (IJCAI-27) and advisor report.

### New work because of v3-B
- Spec v1.11: changes listed in `v3B_design.html` §16 (R1, R4, R9, §3 table, §4 theorems, §5 claims C10 `closure_only`, C11 `hub_A`, §6 arms and cost gate).
- Code: an operator for the node–fact–cell energy (incidence lists, matrix-free UP/DOWN transport on fact states, exact elimination of off-cell facts), PCG forward + implicit backward, new arms, tests (§14 of the design).
- Mini-batch doc: M2 (K ≥ 2 for the filter) becomes PCG tolerance / T_max.
- Paper: 2-cell framing returns; propagation described as an energy on adjacent pairs of the sheaf, with Lemma A′ as the reason v1 failed W3.

### Concerns to settle (warning signals; discuss before engineering)
1. **What fact states add over A (hub)** is unanswered in theory; it is the main scientific risk (R4 asks to separate 2-cells from capacity). `hub_A` vs `full` decides it.
2. **Cost**: state per solve ≈ (N + |E_P| + F)·d; ACM up to ~99 MB per vector if every fact is on a cell (estimate); PCG iterations on large graphs unknown (toy: 30 iterations → 9e-6, condition 27.5). R1, R3 and R4 all asked about scalability.
3. **D-B1/D-B2**: dropping gates (R4) and the filter bank (R9) changes spec claims C4, C4b, C8; the "heterophily needs high frequencies" premise is not proven; class separation can come from the maps (Theorem 4).
4. **Hub dominance** without normalization (D-B3): measure on ACM subject nodes / Freebase.
5. **Prior art at formula level** still open: CW Networks (Bodnar 2021; PDF in `references/CWNetworks_2022.pdf`), CTNN, CCNN (Hajij), sheaf on cell complexes, Sheaf Hypergraph Networks, SGAT (IJCAI 2022), arXiv 2606.00937. Question: is "sheaf energy on the incidence (Hasse) diagram with transport-forced edge→cell maps + anchored solve" known?
6. v2 theory that no longer applies directly (Props. 4–7, E, K, Lemma N, Theorem R in hub form) must be restated or moved to the `hub_A` arm.
7. `householder_k = 2` limits Theorem 4's construction (Cartan–Dieudonné needs up to d reflections): state as a limitation or change.

## 4. First task for the new session: finalize the model against the reviews
1. Re-read the reviews (PDF/markdown) and `v3B_design.html`; produce one table: every praise and every method criticism → the exact equation/component of v3-B that keeps or answers it → status (proved / to test). Flag anything the design does not answer.
2. Decide D-B1…D-B7 with the author (options and recommendations are in the design §15).
3. Run the prior-art check of concern 5 at formula level (write each method's update on P1–A7–P9 / P1–A7–P4 and state the special case where v3-B reduces to it).
4. Only then: spec v1.11 text and the implementation card (units: operator + tests, solver + implicit gradient, arms, cost measurement on ACM/DBLP/IMDB).

## 5. Open question raised by the author (not yet answered)
- For link-prediction datasets without node labels (e.g. LastFM, PubMed after F17), can heterophily be measured from **attribute similarity** instead of labels? Current spec (§1, §6) says "not measurable". Literature has feature-based homophily measures (e.g. attribute/feature similarity along edges); a definition must be fixed before use, and features removed by F17 must not be used. Discuss in the new session.

## 6. Rules
- `part-j` is shared with Claude Code sessions: never edit, commit, stash or reset their work. Write only new files unless the author asks.
- Check mtime (`expectedMtimeMs`) before overwriting any file on the author's computer.
- Specs, cards and prompts in English; replies in Thai.
