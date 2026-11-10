# Speaker script — progress meeting, 2026-11-11

Deck: `progress_2026-11-11.pptx` (18 slides). One section per slide. Terms: meta-path schema, meta-path instance, node / edge / face, stalk, restriction map, transport, energy, WISP. Every number that does not exist yet is marked [result].

---

## Slide 1 — Title

Good afternoon. I report on H2S-NN, now titled "Meta-Path-Structured Sheaf Neural Networks for Heterogeneous Heterophilic Graphs". The title changed because the model changed: the model is now version v3-B of the specification. I will state the research question, explain what the model computes, show how the WSDM reviewers' requests shape the paper, say honestly what is still open, and end with decisions I need today. No experimental result of the new model exists yet.

## Slide 2 — Outline

Five parts. Background introduces heterophily on heterogeneous graphs, and the difference between a meta-path schema and a meta-path instance. Problem gives the research question and compares prior work. Method presents meta-path instances as faces of one sheaf, the single energy the model minimises, what changed since WSDM, and what is open. Experiments covers what the reviewers asked for and the comparisons that answer it. Discussion covers the paper, the plan to the January deadline, and the decisions for today.

## Slide 3 — Heterogeneous graphs are often heterophilic

This small graph is the running example: papers, an author and a subject. It is heterogeneous because nodes and relations have several types. It is heterophilic because P1 and P9 share author A7 but belong to different fields, shown by colour. Adjacency does not imply agreement. A meta-path schema is a type pattern, such as paper–author–paper. A meta-path instance is one concrete walk following it, such as P1–A7–P9. Our benchmarks span the range of heterophily: the adjusted measure ℋ² is about 0.74 on IMDB and 0.995 on Freebase, and we add the large H2GB datasets.

## Slide 4 — Research question

Heterogeneous graphs are frequently heterophilic, and existing methods address this indirectly. They split the graph into one homogeneous view per schema and fuse the views. This has two limitations: heterophily is judged on derived graphs instead of the original one, and every pair under the same schema is treated identically, so the particular instances and their intermediate entities are ignored. We ask whether heterophily can be handled directly on the original graph by making meta-path instances part of the structure the model operates on, and whether this gives information that schema-level encoders and instance-level features cannot. These become two goals: one model without per-schema branches, and instances as structure that carry extra information.

## Slide 5 — Splitting by meta-path removes the intermediate nodes

Panel (b) is common practice. The paper–author–paper graph connects P1, P9 and P4; the paper–subject–paper graph connects P1 and P9. Author A7 and subject S3 appear in neither graph, so what they could tell us is gone before learning starts. Panel (c) is our model. The instance P1–A7–P9 is a face on the original graph. All three nodes, including A7, are pulled toward agreement with it, and A7 passes the effect on to P4 through its own edge. The picture is the claim we will test, not a result.

## Slide 6 — Prior work, compared at the level of the update

I compared methods by what their update actually does, not by their abstracts. MAGNN and SGAT use the intermediate nodes of an instance but write only to the target node. Sheaf Hypergraph Networks update intermediate nodes, but the hyperedge has no stalk and its maps are free. CWN and CCNN give faces a stalk, with free maps per incidence. HetSheaf works on edges only. Our statement is narrow: instances are faces of the same sheaf as the graph, their maps are forced by the edge maps, and the identity of the intermediate node enters the geometry. Using intermediate nodes is not new, and I will not claim it. Two recent papers were read at abstract level only, so I make no claim against them.

## Slide 7 — A meta-path instance as a face of the sheaf

The sheaf has three levels. Nodes carry node stalks, edges carry edge stalks, and each sampled meta-path instance is a face, a two-dimensional cell, with its own face stalk. Nodes, edges and faces are the zero-, one- and two-dimensional cells. The face has no closing edge, so the path need not return to its start. Restriction maps connect the levels. Only the node-to-edge maps are learned, and they are direction-aware. The edge-to-face and node-to-face maps are forced: they are the composition of the learned maps along the path. So a face has no parameters except its weight, and the graph's own consistent assignments remain consistent after faces are added. WISP samples the faces with random walks, with no meta-path list.

## Slide 8 — One energy over nodes, edges and faces

The model defines one energy. Each term is a disagreement after transport: between a node and an edge, an edge and a face, and a node and a face. Edge weights and face weights scale the terms; a face's weight is computed from all its nodes and relations. Node representations are the minimiser of this energy, anchored at the encoding, found once per forward pass. The solution is unique for every value of the learned parameters, and a consistent encoding is a fixed point. Every node of a face, including intermediate ones, receives its influence directly. One honest note: the node-level solve is a personalised-PageRank resolvent. The contribution is the operator, not the solver.

## Slide 9 — One face, step by step: P1–A7–P9

Step one: the node-to-edge maps of P1–A7 and A7–P9 are composed along the path, which puts all three nodes in the face's frame. Step two: the face carries a stalk that its nodes and edges must agree with. Step three: P1, A7 and P9 are pulled toward that stalk in proportion to the face weight, so A7 is updated directly. Step four: A7 also lies on the edge A7–P4, so the effect reaches P4 through the edge terms. I state in the paper that with relation-only maps and no edge stalks, a face reduces to MAGNN's encoder. Our maps depend on the pair and the role, so who A7 is matters.

## Slide 10 — Where heterophily is handled

Heterophily is handled in three places. In the restriction maps: neighbours of different classes need only agree after transport, and for any labelling there exist maps that make class-separating assignments consistent. In the face weight: it comes from the whole instance, with no ceiling on one face's pull, and without a branch per schema. At the readout: the head sees a node's own encoding next to its propagated stalk, so both smoothing and its complement are available. What this does not show is that gains on real data come from these mechanisms. That is claim C6, stratifying gain by ℋ², and I have no result yet.

## Slide 11 — What the WSDM reviewers asked for, and the answer

The reviews tell us what the story must contain. Heterophily must be shown and analysed by degree: ℋ² and MLH on every dataset, H2GB added, gain stratified by ℋ². The face reached nodes only through the closure edge: now it is coupled directly to every node, and the closure edge carries no stalk. The face's contribution must be isolated from weighting and capacity: arms on identical sampled faces and parameter budgets. Cost against graph size is needed: a cost gate and scaling curves. Baselines must be rerun under one protocol. I do not repeat the comments about the old selection and pooling modules, because those modules are gone.

## Slide 12 — What changed since WSDM, and what is still open

Changed: the instance is a face with its own stalk coupled to every node, and one anchored energy minimisation replaces Euler steps on a Hodge Laplacian, which provably cannot carry face information to nodes. WISP replaces the old schema selection, and several modules are removed. Every result is rerun. Open: influence is symmetric in strength, with direction only in the maps. We have no example where edge stalks on shared edges are decisive, so we do not claim they add information. Cost is derived, not measured. Propagation is linear. And all claims C10 to C14 are untested.

## Slide 13 — Experiments that decide each claim

Every comparison uses the same sampled faces and matched parameter counts. Full versus no-faces asks whether faces help. The weight ladder, from uniform to schema to endpoints to full, asks whether instance content matters; the claim fails if full is not better than endpoints on two of three HGB datasets. Full versus closure-only tests the reviewers' point that a face may be only its closure. Full versus instance features is the MAGNN-like test: structure versus feature. Covering-only tests whether the node–face term is needed, and the hub arm tests edge stalks. Finally, the gain over endpoints is plotted against ℋ². Ten seeds, paired Wilcoxon tests with Holm correction; consequences were fixed before any run.

## Slide 14 — Datasets and protocol

Node classification is the main task: HGB ACM, DBLP, IMDB and Freebase, and four H2GB datasets that need mini-batch training. We report ℋ² and MLH for each. Link prediction is secondary, on tasks derived from the same graphs plus PubMed and LastFM, with MRR and Hits at 10 against HeaRT hard negatives, split into warm and cold-start pairs. Graph classification goes to the appendix as a generalizability check, with no heterophily claim. All baselines are rerun in one protocol, HetSheaf is mandatory, the tuning budget is identical for every arm, and WISP settings are pre-registered and never tuned.

## Slide 15 — Paper status and plan to the deadline

The paper is IJCAI-27: seven pages including figures, two pages of references, anonymous. The rewrite follows the new story up to the Experiments section, with placeholders for numbers that do not exist. The plan: until November 20, finish the rewrite, run the cost gate on ACM, DBLP and IMDB, implement the instance-feature arm, and finish the link-prediction card. By November 30, audit the experiment table and freeze the specification and code with a commit hash. In early December on TSUBAME, the pipeline gate and the core comparisons run first; they decide whether we keep the full claim or the narrowed one. The abstract is due January 4 and the full paper January 11, 2027. Risks are cluster access, a kill rule firing, and the cost gate.

## Slide 16 — Decisions for today

Four decisions. One: the framing. Meta-path instances are faces of the graph's own sheaf; the terms are node, edge and face, where a face is a two-dimensional cell; we do not use Hodge. Two: narrowed claims agreed now. If full does not beat endpoints, we do not claim that instance content adds information. If full does not beat instance features, faces act as instance encoders and the contribution is the geometry. If the hub comparison fails, edge stalks are not claimed. Three: run the cost gate first, and use fixed-step truncation if it fails. Four: I need the TSUBAME details: A100 memory, job time limit and quota.

## Slide 17 — References

No speech. The thirteen references cited on the slides as [1] to [13]: MAGNN, NSD, HetSheaf, Sheaf Hypergraph Networks, CW Networks, Topological Deep Learning, Copresheaf Topological Neural Networks, SGAT, H2GB, Hetero²Net, HGB, HeaRT, and the adjusted-homophily paper.

## Slide 18 — Appendix: terms

Reference slide for questions. A meta-path schema is a type pattern; a meta-path instance is one walk following it. A cell is a node, an edge or a face, of dimension zero, one and two; a face is one sampled instance. A stalk is the vector space attached to each, and we also use the word for the vector it holds. Restriction maps go from one level to the next: node to edge is learned, edge to face and node to face are forced. Transport is the composition of maps along an instance. Energy is the sum of disagreements at all levels. WISP is the schema-free sampler, and ℋ² is adjusted heterophily.
