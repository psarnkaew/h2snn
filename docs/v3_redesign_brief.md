# v3 redesign brief — start here in a new session

Written 2026-11-10 12:10 JST at the end of a long review session. The author asked for a fresh session because the old one became unreliable. **Treat every claim below as a lead to re-verify, not as settled fact.** Re-run the scratch checks (in `docs/v3_evidence/`) before using any number.

## 1. Why a redesign
- v2 (current code, branch `part-j`) removed everything the WSDM reviewers praised: path-supported 2-cell with closure edge, edge-to-face maps by transport, chain-condition proof (R1 S1–S2, R2 S1–S2, R3 S1, R4 S2; reviews in `docs/critical_reviewer_reviews.md`).
- v2's path cell computes as a hub node (spec Prop. 2) and, with relation-only maps, equals MAGNN's instance encoder (`possible_reviewer_qa.md` Q2.3b). The novelty then reads as HetSheaf + MAGNN (Q2.3c).
- v1's real flaw (R2 W3, meta-review): face information reached node states only at the readout; the closure edge was also in the node operator L₀, i.e. a direct endpoint P1–P9 edge.

## 2. Direction (v3, proposed; not yet adopted)
v1's 2-cell structure + WISP (replaces AMPSS) + v2's memory machinery + a new edge→node route.
- Cochains on nodes, edges (observed + one closure edge per instance), faces (one per instance).
- node→edge and edge→face: **coboundary form** (differences), maps by transport as in v1; chain condition δ₁δ₀ = 0 holds.
- edge→node: **one-sided lift** at every layer (DNSD-style, arXiv 2605.19021; v1's readout pooling moved into every layer), through **observed edges only**. Closure edge stays as the cell boundary, not in L₀ and not in the lift.
- From v2 keep: factorized maps, matrix-free transport, maps computed once and shared across layers, WISP unbiased weights + instance weight, local-scope mini-batch (M0–M4), all data/protocol fixes.
- A_f: only A_fᵀA_f matters → scalar w_f or diagonal, not a d×d MLP output.
- DNSD stabilizers (LayerNorm, odd activation, gate) are candidates for the node side; if adopted, v2's stability props (Prop. 3, 5) no longer apply and must be dropped from the paper.

## 3. Checks done in the old session (re-run before trusting)
Scripts: `docs/v3_evidence/check_face.py`, `check_noclosure.py`, `check_dnsd_face.py` (numpy; toy instance P1–A7–P9 + closure, extra edge A7–P4, d = 3).
| Check | Reported | Meaning |
|---|---|---|
| δ₁δ₀ | 2e-16 | chain condition |
| δ₀ᵀδ₁ᵀ | 2e-16 | Lemma A′: face → node is zero through signed coupling |
| lift Pδ₁ᵀ at P1/A7/P9/P4 (closure excluded) | 1.74 / 2.43 / 0.78 / 0 | face reaches all instance nodes via real edges, nothing else |
| L₀ block P1–P9, closure in / out | 0.386 / 0 | v1 had an endpoint shortcut; removed |
| face on node-induced edge signal (coboundary) | 0 | face sees only edge content not explained by nodes |
| DNSD one-sided node→edge, then edge→face, vs v2 hub sum | 4e-16 | **using DNSD form for node→edge collapses the face to the v2 hub (MAGNN)** |
| L₁ᵘᵖ(A_f) vs L₁ᵘᵖ(QA_f), Q orthogonal | 2e-15 | A_f only via A_fᵀA_f |

## 4. Author decisions so far
- Face acts on the edge cochain: correct by definition, keep.
- Endpoint information inside the face (closure edge state): acceptable. The objection is answered by the route to nodes, not by removing endpoint information.
- Use DNSD as a template for message passing (author's view; scope per §2 rule above is still to be confirmed).
- Drop "2D" wording decision (made earlier for v2) is **reopened**: with v3 the 2-cell is real again.
- Target venue IJCAI-27; slides + scripts for the progress meeting 2026-11-11 (`paper_rewrite_brief.md`, which assumed v2 — update it after the design is settled).

## 5. Gates before adopting v3 (order matters)
1. **Memory/runtime (author's main worry).** Open v1 code (git tag `v1-submitted`) and find where ACM's 9,260 MB came from (suspect d×d maps per edge and per face incidence). Count tensor sizes for v1, v2, v3 from real N, E, I of ACM/DBLP/IMDB. CPU-only small-graph profile for ratios only (local runs are correctness-only, `working_preferences.md`).
2. **Prior art at formula level:** CW Networks (Bodnar et al. 2021), sheaf neural networks on cell complexes, arXiv 2606.00937 (Cellular Sheaf Neural Operators), arXiv 2506.02842, Sheaf Hypergraph Networks, CTNN (`CTNN_copresheaf_notes.md`).
3. **Mechanism + known-answer test.** Mechanism: perturb only a face state, measure change of A7 with the closure lift off; edges from nodes only ⇒ face = 0. Known-answer task where the label depends on edge-state inconsistency around an instance. Arms (same budget): v3; closure-only (endpoint holonomy edge, no face); v1 face + late pooling; no face; DNSD-form node→edge face.

## 6. Open questions for the author
- Which DNSD parts on the node side (adjacency vs Laplacian, LN, tanh, gate)?
- Edge state initialization (v1 `MLP_init`) — the face only sees edge content not explained by nodes, so this choice decides what the face can detect.
- Simple-path filter for WISP walks (chain condition needs simple paths).

## 7. Rules
- `working_preferences.md` in full: plain-language claims, no flattery, check prior art at formula level before claiming, warning signals stop work.
- Claude Code works on branch `part-j` (unit J6, uncommitted LP files): do not touch, commit, stash or reset its work.
- Before writing to the author's computer, check mtime (`expectedMtimeMs`).
- Related docs: `v1_face_revival_assessment.md` (detailed log of this session's reasoning), `possible_reviewer_qa.md` Q2.3b–Q2.3c, `metapath_landscape_2025_2026.html`, spec v1.10, old manuscript `H2S-NN/paper/h2snn_manuscript.tex`.
