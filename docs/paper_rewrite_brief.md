# Paper rewrite + progress slides — brief for a new session

Written 2026-11-10 06:45 JST at the end of the novelty-review session. Read this first, then the files it points to. English for Claude Code / docs; talk to the author in Thai.

## 1. Deliverables (due: slides for the progress meeting 2026-11-11; paper continues after)

| # | Deliverable | Source / template | Output location (new files only) |
|---|---|---|---|
| D1 | IJCAI-27 manuscript, rewritten from the title through the Experiments section (experiments unfinished: keep tables/claims as placeholders `[result]`) | Old paper `H2S-NN/paper/h2snn_manuscript.tex` (+ `references.bib`, `figure/`); IJCAI template `H2S-NN/paper/FormattingGuidelines-IJCAI-27/` (`ijcai27.sty`, `ijcai27.tex`, `named.bst`) | `H2S-NN/paper/ijcai27/h2snn_ijcai27.tex` (copy sty/bst/bib/figures there). Never overwrite `h2snn_manuscript.tex`. |
| D2 | Progress slides | `C:\Users\Acer\Desktop\h2snn\Slides template.pptx` (use its layouts, fonts, colours) | `C:\Users\Acer\Desktop\h2snn\Claude outputs\progress_2026-11-11.pptx` |
| D3 | Speaker script, one section per slide | — | `C:\Users\Acer\Desktop\h2snn\Claude outputs\scripts.md` |

Do D1 and D2/D3 together: one story, the same terms, the same figures. If time is short, D2+D3 come first (meeting is tomorrow).

IJCAI-27 limits: 7 pages body + 2 pages references; anonymized; supplementary appendix allowed (reviewers may skip it, so nothing the argument needs goes only there).

## 2. Style to keep from the old paper (author's instruction)

Keep exactly as in `h2snn_manuscript.tex`: the way figures are designed and referenced, the narrative order, the connectives and sentence style, the section ordering, and the way each equation is introduced and explained (what it computes in words, then the symbol-level detail). Change only what the new model and the review findings require. Read the whole old paper before writing anything.

Add: human tone, critical and precise, clear, concise, informative, academic, easy to follow, consistent terms, no jargon without a one-line definition, no code-level implementation detail (file names, tensor shapes, kernels, hyperparameter defaults go to the appendix or nowhere), every claim reasonable and backed. Rules in `docs/working_preferences.md` apply in full (no self-praise; theory is never the contribution; one plain-language research question with a concrete example first).

## 3. The author's intent (the story the paper must tell)

Author, verbatim (Thai): "การจะมอง heterogeneous heterophilic graph นั้น ควรที่จะพิจารณาถึงความเป็น heterophily บนกราฟดั้งเดิม มิใช่การแตกเป็นหลาย meta-paths และการพิจารณากราฟนั้น meta-paths เป็นอะไรที่สำคัญมากที่ควรนำไปคิดในกระบวนการของกราฟโดยตรง ซึ่งเราจะคิดแค่ meta-path schemas (หมายถึงทุกโหนดที่ใช้ schema เดียวกันหมด จะใช้ shared encoder เดียวกัน) เราต้องมองไปถึงระดับ node instances ซึ่ง meta-path ก็เหมือนเป็นโครงสร้างที่ซ้อนอยู่บนกราฟเช่น 2D-Cellular graph ที่ไม่มีเส้นปิด มันเป็นโครงสร้างที่ฝังในกราฟ ไม่ใช่แค่ข้อมูลเพิ่มเติม"

In English: (1) handle heterophily on the original heterogeneous graph, not on several homogeneous meta-path graphs; (2) meta-paths must enter the graph computation itself; (3) at the level of individual path instances, not schemas (a schema-level method gives every instance of a schema the same treatment); (4) a path instance is a structure laid on the graph, not extra node features.

Running example (use in paper, slides and script): papers P1, P4, P9, author A7, subject S3; instances P1–A7–P9, P1–S3–P9, P1–A7–P4. Splitting by meta-path (PAP, PSP) removes A7 and S3 from the computed graphs; our model keeps them, and A7 receives the message of P1–A7–P9 and can pass it to P4 at the next step.

## 4. What the model computes (plain words; equations in the spec, Sec. "v2")

1. Encode each node by type; add a parameter-free summary of its neighbours per relation.
2. Sheaf on the original graph: each edge gets learned maps that transform a message from one endpoint into the other's space (heterophily: neighbours need not agree directly).
3. WISP samples path instances by random walks on the original graph; no meta-path list is given. Each sampled instance is unbiased-weighted (m_f).
4. Each instance gets a learned scalar weight from its actual nodes and relation sequence (ω_f = m_f · exp g(·)).
5. Each instance adds one term to the operator: its nodes, transformed with the same edge maps along the path, are compared with their common average, and every node of the instance receives the correction (not only the end node).
6. Edge terms + instance terms form one operator, normalized so it is stable for any parameter values; a Chebyshev filter bank with listen/broadcast gates produces the node states; a linear head predicts.

## 5. Claims: allowed / forbidden (settled in the review; see `docs/possible_reviewer_qa.md` Q2.3b, Q2.3c)

Allowed (state plainly):
- Works on the original graph; no meta-path list, no per-meta-path parameters (schema-free sampler).
- Path instances are weighted individually.
- Every node of an instance receives the instance's message, including intermediate nodes (MAGNN writes only to the target).
- Instance messages use the same edge maps as the graph (one geometry for edge and instance evidence).
- The full operator is stable for all parameters (supporting property, not the contribution).

Forbidden:
- "2D computation" or "the cell learns its own state": computationally each instance equals one hub node joined to its nodes (Prop. 2; checked numerically). Instance average with relation-only maps equals MAGNN's rotation encoder (verified to 1e-16).
- "Order of relations is a difference from MAGNN" (withdrawn).
- "Instances add information beyond edges" without a result (Prop. E: with equal weights the gain is mostly reweighting of edge messages; average-case, not measured).
- "The model learns which instance matters": S1′ (Part I) failed; Part K shows it can use correct weights only if given them.
- Theory as novelty.

**Decision (author, 2026-11-10 06:45): drop the "2D" framing.** No "2D", "2-cell", "cellular complex", "face" or "higher-order cell" as a claim anywhere (title, abstract, contributions, method, captions). Call the unit a **path instance**; what the model does with it is "every node of the instance exchanges messages through the edge maps along the path". A figure may shade or outline an instance's nodes to show the group, but the caption must not call it a cell or a 2D structure. Say once in the Method that an instance term is equivalent to a hub node joined to the instance's nodes (Prop. 2), so a reviewer does not discover it. The author's "structure laid on the graph, not extra data" intent is kept as: instances live in the original graph's operator and use its edge maps (not as precomputed features). In the slides, state the change and why: the old 2D story did not hold (v1: Lemma A′ / WSDM R2 W3; v2: hub-node equivalence). Show the author the before/after wording of title, abstract and contributions.

## 6. Experiments section (unfinished)

Write the setup, datasets, protocol and the planned tables, with `[result]` placeholders. Planned comparisons that the novelty answer depends on (Q2.3c): `full` vs `no_path_cells`, `holonomy_edges`, `free_path_maps`, `endpoints`/`uniform` (weight ladder), and the naive-combination baseline (MAGNN-style instance embeddings fed as features to the edge-only sheaf model; not yet in the protocol, needs the author's decision). Known-answer tasks S1/S1′ reported honestly (S1′ failed). Never fill a number that does not exist.

## 7. Terminology (use one term per idea, everywhere)

heterogeneous graph · heterophily · meta-path schema (the type pattern, e.g. PAP) · path instance (one concrete walk, e.g. P1–A7–P9) · instance weight · edge map (sheaf restriction map) · transport (composition of edge maps along an instance) · WISP · original graph. Do not use: 2D, 2-cell, face, cellular complex (decision Sec. 5). Avoid Hodge, coboundary in the main text unless defined and needed.

## 8. Slides and script (D2, D3)

Audience: progress meeting. Order: problem and intent (Sec. 3) → why splitting by meta-path loses information (example) → what the model computes (Sec. 4, one figure) → what changed since the WSDM version and why (v1 closure-edge mechanism could not carry instance information to nodes: Lemma A′ / reviewer R2 W3) → honest findings this week (MAGNN equality, hub-node equivalence, S1′ failure, Part K, data/protocol bugs fixed) → the novelty question (A + B objection) and the experiments that answer it → plan to the IJCAI deadline. Script: what to say per slide, 60–120 words each, plain Thai or English as the author prefers (ask).

## 9. Process rules

- Read before writing: old paper (whole), spec v1.10 v2 sections, Q2.3b/Q2.3c, the HTML literature map `claude/metapath_landscape_2025_2026.html`, `working_preferences.md`.
- Another session (Claude Code) is working in the repo on branch `part-j` (unit J6, uncommitted LP files). Do not touch its files, do not commit/stash/reset. Write only the new files in Sec. 1.
- Check the file's mtime before writing to the author's computer (`expectedMtimeMs`).
- Send drafts early: outline of the paper and slide list first, for the author to approve.
- Last step: compile the LaTeX (pdflatex + bibtex) and check page count; render the slides to images and look at every slide.
