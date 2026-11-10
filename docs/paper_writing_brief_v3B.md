# Paper writing brief — IJCAI-27, model v3-B (for author approval)

Written 2026-11-11 by the Cowork session. Sources of truth: `docs/H2S-NN_methodology_spec.md` v1.11 (equations, theorems, claims), `docs/v3B_prior_art.html` (formula-level comparisons), `docs/H2S-NN_v3B_report.md` (what is implemented and checked). Supersedes `docs/paper_rewrite_brief.md` §4, §5, §7 (v2 model; "no face" rule reversed by the author on 2026-11-11). §2 of that brief (keep the old paper's style) still applies.

Output file (new; existing files untouched): `H2S-NN/paper/ijcai27/h2snn_ijcai27_v3b.tex`, same `ijcai27.sty`, `named.bst`, `references.bib` (new entries appended only to a copy if the author prefers).

## 1. Decisions (author, 2026-11-11)
- Title: **H2S-NN: Heterogeneous Heterophilic Sheaf Neural Network with Path Instances** (the tex title).
- The cell built on one path instance is called a **face** (author, 2026-11-11). Since spec 1.13 the complex is the **incidence poset** nodes < edges < faces: a face sits above the nodes and edges of its path and is never closed into a loop; no closing edge exists, not even in the definition. Sections in scope now: Introduction (with background), Related Work, Preliminaries, Methodology. Experiments, abstract and conclusion later, with `\res{}` placeholders.

## 2. The story in plain words
**Research question (one sentence).** Papers P1 and P9 share the author A7: does it matter *who* A7 is, and should A7 itself learn that P1 and P9 are linked?
**Thesis H.** On heterogeneous graphs, and especially under heterophily, label-relevant information is carried by concrete path instances (their intermediate nodes and the order of their relations) and is lost when an instance is reduced to a relation between its endpoints.
**Idea in one sentence.** Each sampled path instance becomes a face of the same sheaf that already lives on the graph; its maps are fixed by the edges it traverses, its message reaches every node on it, and all propagation is the minimiser of one energy over nodes, edges and faces.
**Running example (everywhere):** papers P1, P4, P9 (different fields), author A7, subject S3; instances P1–A7–P9, P1–S3–P9, P1–A7–P4.

**Arc (each paragraph of the Introduction answers the previous one).**
1. Heterogeneous graphs are often heterophilic; averaging neighbours fails (Fig. 1a).
2. Meta-paths express typed multi-hop relations; schema vs instance; the research question.
3. How prior work answers it, and what each loses: splitting by schema removes A7; instance encoders (MAGNN, MECCH, MF2Vec) keep A7 but write only to the target; sheaf networks (NSD, HetSheaf) handle heterophily but see only edges; networks on cell complexes attach cells to cycles or meshes with free maps, and under Hodge-type diffusion a face cannot reach a node at all (this is also why the submitted model failed reviewer R2 W3: its face reached nodes only through a closing edge; v3-B removes the cause by construction, Prop. 2, without claiming an accuracy gain until C10 is measured). A tuned R-GCN matches many HGNNs, so every new component must be isolated by controlled comparisons.
4. Our answer (the idea sentence above, with Fig. 1c).
5. Contributions (§3).

## 3. Contributions (draft wording; methodological, theory only as support)
1. **Path instances as faces of the graph's own sheaf.** Each instance is a face whose maps are forced by the maps of the edges it traverses, so it adds no free parameter and never contradicts a state that is consistent on the graph. Its message reaches every node of the instance, including intermediate nodes, both directly and through its edges, with no endpoint-to-endpoint shortcut.
2. **One propagation for nodes, edges and faces.** Node representations are the unique minimiser of one energy anchored at the input encoding, for every value of the learned parameters, with no depth or filter hyperparameter. Removing one term of the energy yields an endpoint view, an instance-feature encoder in the style of MAGNN, or no faces at all, so each comparison the question needs is a configuration of the same model on the same sampled instances.
3. **Schema-free sampling with instance-level weights (WISP).** Instances are drawn by random walks from a stated distribution with unbiased weights; each face receives a learned weight from its own nodes and relation sequence, without a meta-path list, and the cost grows linearly with edges and face incidences, not with the number of schemas.

## 4. Terminology and notation (one word, one symbol per idea)
| Concept | Word in the paper | Symbol | Notes |
|---|---|---|---|
| graph | heterogeneous graph | `G=(V,E,\phi,\psi)`, types `\mathcal T`, relations `\mathcal R` | |
| one stored fact (both directions tied) | **edge** | `e \in E` | "fact" never appears; the tie is stated once in Preliminaries |
| type-level pattern | **meta-path schema** (short: schema) | `\pi` | e.g. PAP |
| one concrete walk | **path instance** (short: instance) | `(v_0,e_1,\dots,e_\ell,v_\ell)` | the thing sampled |
| cell built on one instance | **face** | `f \in P_2` | defined once: "a face is a new cell placed above the nodes and edges of one sampled path instance"; never "2-cell of a CW complex", never a closing edge |
| the structure | incidence poset | `P=(P_0,P_1,P_2)=(V,E,\text{faces})`, order `v<e`, `e<f`, `v<f` | sheaf = functor on `P` (Thm 1); cite Curry 2014 / 2019 after a full read |
| vector space on a cell | stalk | `\mathbb R^d` | |
| map node→edge, edge→face, node→face | **restriction map** | `F_{v\inc e}` (learned), `F_{e\inc f}`, `F_{v\inc f}` (forced) | one notation for all levels; "transport" only as a verb |
| learned states | node, edge, face **state** | `x_v, s_e, z_f`; `Y=(x,s,z)` | |
| encoder output | encoding | `h_v` | |
| neighbourhood summary | context | `c_v` | `c_e` of the spec is not used; only `a_e` |
| edge weight | edge weight | `a_e` | |
| face weight | face weight | `\beta_f = w_f m_f`, `w_f=\exp(g(\xi_f))` | `m_f`: sampling weight |
| propagation strength | — | `\tau` | |
| normalized operator | sheaf Laplacian of the complex's incidences | `\hat L` | never "Hodge" |
| output of propagation | propagated state | `x^\star_v` | |
| predictor | — | `p_\theta` | tex used `g_\theta`, which clashes with `g` |
**Forbidden words:** closing edge / closure (except when describing prior Hodge-type models), "2-cell" as a claim, Hodge (except to describe prior models), sheaf-Hodge diffusion, hub, cochain complex in claims, gate, filter bank, Chebyshev, AMPSS, "fact", "2D computation", "anchored solve", "PCG" (one mention of conjugate gradient in the method is enough), code arm names (`hub_A`, `closure_only` …) in the main text.
**Configuration names in the paper (proposal):** Full; No faces; Endpoint edge (face replaced by one edge between its endpoints; never called "closing edge"); Face as feature (MAGNN-type); No node–face term; No edge–face term; Free face maps; weight ladder Uniform / Schema / Endpoints.

## 5. Section plan and page budget (7 pages)
| Section | Pages | Content |
|---|---|---|
| 1 Introduction | 1.0 + Fig. 1 | arc of §2; three contributions |
| 2 Related Work | 0.6 | (a) meta-path HGNNs: precompute/induced graphs (HAN, SeHGNN, LMSPS, heterophily-aware: Hetero²Net, H2GB-line), instance encoders (MAGNN, MECCH, MF2Vec, SGAT) — target-only write-back; (b) sheaf networks: NSD, HetSheaf, Sheaf Hypergraph Networks (unordered, free maps), DSHN (direction by fixed complex phase); (c) topological deep learning: CWN (cells on cycles), CCNN/CTNN (free maps per incidence), cellular sheaf operators on meshes (2606.00937: free maps between levels, layer-wise incidence messages, no direct face→vertex term). One closing sentence per paragraph on the formula-level difference. |
| 3 Preliminaries | 0.7 | heterogeneous graph (edge = tied fact); schema vs instance; heterophily measures (ℋ², MLH; keep current text); cellular sheaves on a graph, extended in four lines to the incidence poset (nodes < edges < faces) and functoriality |
| 4 Methodology | 2.4 + Fig. 2 | 4.1 problem; 4.2 overview (Fig. 2, redrawn); 4.3 encoding and restriction maps; 4.4 faces: sampling (WISP, short) and forced face maps (Prop. 1); 4.5 one energy, term by term on P1–A7–P9 (Prop. 2); 4.6 propagation as its minimiser, prediction (Prop. 3), heterophily in one paragraph (rotations separate classes; low/high-pass at the head); 4.7 face weights; 4.8 special cases (table) and cost |
| 5 Experiments | 1.8 | later |
| 6 Conclusion | 0.2 | later |

**Results in the main text (three propositions, proofs in the appendix):**
- Prop. 1 = spec Thm 1 + Thm 3: given the edge maps, the face maps are forced; the kernel equals the graph's global sections (free face maps collapse it).
- Prop. 2 = spec Thm 8 (+ Lemma A′ as the contrast, one sentence): a face reaches every node of its instance in one operator application; no term couples the endpoints except through the face or a shared edge.
- Prop. 3 = spec Thm 5 + Prop. 10: unique solution for every parameter value; the node-level system has condition number at most `1+τ`, independent of size and degrees.
One sentence each, no proposition: Prop. 7′ (without the node–face term a face pulls no harder than its own edges: motivates the term), Thm 4 (class-separating states under heterophily), Prop. 11 (low- and high-pass at the head), PPR form (L15, credit APPNP).

## 6. Claims
**Allowed:** the four statements of `v3B_prior_art.html` §9; cost "linear in edges and face incidences, `kd` numbers per incidence, memory independent of the number of solver iterations".
**Forbidden until measured:** any gain from faces (C10, C14), from the edge–face coupling (C11), cost numbers on ACM (C13); "first to use intermediate nodes"; "novel solver"; "edge states carry new information"; "the model learns which instance matters".
**Honesty sentences that must appear:** with relation-only maps and faces used as features, the face state equals MAGNN's instance encoder (Method 4.8); edge states add no input information (no edge features); the node-level propagation is a personalised-PageRank resolvent and the contribution is the operator.

## 7. Style
The author's list: human, critical, knowledgeable, clear, concise, informative, academic, easy to follow, consistent, narrative, well organised, no implementation detail, consistent terms, reasonable, no jargon, IJCAI-suitable. Concretely: every equation is introduced by what it computes in words, then symbols, then one sentence on why it is needed; every technical word is defined at first use; every section ends by handing the reader to the next; the running example appears at least once in each of Introduction, Preliminaries and Methodology; file names, tensor shapes, solver settings and defaults go to the appendix; old paper `h2snn_manuscript.tex` is read in full before writing, for sentence style and figure referencing (paper_rewrite_brief §2).

## 8. Open items
1. Figures redrawn by Claude in TikZ (2026-11-11): `figure/h2snn_example_v3b.pdf` (Fig. 1; source `figure/src/fig1_v3b.tex`) and `figure/h2snn_overview_v3b.pdf` (Fig. 2; `fig2_v3b.tex`); the v2 files are untouched.
2. Added to `paper/ijcai27/references.bib` (author's choice, 2026-11-11) from publisher/arXiv pages: `bodnar2021cwn`, `hajij2022tdl`, `hajij2025ctnn`, `gasteiger2019appnp`, `gebhart2023knowledge`, `lee2022sgat`, `park2025mhcl`, `shikhman2026cellular`. DSHN was already there as `mule2025directional`.
3. Reading depth for Related Work: CWN, CTNN, CCNN (definitions), SHN, SGAT, MAGNN at formula level (prior_art §10); 2606.00937 and DSHN read in full HTML on 2026-11-11 (summary above); 2601.21207, 2602.00159, 2512.00242 abstract only — cited at most as "graph/hypergraph sheaf models".
