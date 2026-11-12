# Speaker script — H2S-NN progress talk, 2026-11-11 (v6)

Deck: `progress_2026-11-11_v6.pptx` (42 slides: 34 main + 3 references + 5 appendix). Same text as speaker notes.

## Slide 1 — H2S-NN: Heterogeneous Heterophilic Sheaf Neural Network with Path Instances

Good afternoon. I will present H2S-NN, our model for graphs that have many node and edge types and in which connected nodes often have different labels. I will present it as I would at IJCAI: the problem, the idea, the method, how we will test it, and what is still open. The model is final and implemented; the large experiments have not run yet, so you will see no accuracy numbers today.

## Slide 2 — Reviewer feedback on the submitted version (WSDM 2027)

How the submitted version was received at WSDM 2027: two weak accepts, two weak rejects, and a reject from the meta-reviewer. Everyone liked the core idea of keeping a concrete path instance as a cell instead of collapsing it into an edge between its ends, and most found the construction principled. The five main concerns: heterophily was not really shown, since only IMDB was strongly heterophilic; the face reached the nodes only through a closing edge between its two ends, which is exactly the endpoint view we criticised; no scalability study; the baseline protocol was unclear and HetSheaf was quoted; and the motivation lacked a concrete failing example. In short, the idea was liked but the evidence was not convincing.

## Slide 3 — Update since last meeting

What changed since the last meeting, in four columns. Contributions: the paper no longer claims AMPSS and SheafCovPool; there is one core idea, path instances as faces with heterophily handled on the original graph, plus WISP as the sampler. Methodology: the closing edge, the K Hodge diffusion steps and the d-by-d maps are gone; faces sit above their paths, propagation is one energy and one solve, and every map is matrix-free, two reflections per edge end, so restriction maps are still there but never built as dense matrices. Path sampling: AMPSS with its schema search and four budgets became WISP. Experiments: from three, four and four datasets with quoted HetSheaf numbers to eight, eight and four datasets with every baseline rerun under one protocol. Status: the model is implemented and passes the theory, gradient and mutation tests, ACM memory went from 9.3 to 2.8 GB, and no accuracy runs have been made yet.

## Slide 4 — Outline

The talk has five parts. Background gives the four ideas you need: typed graphs, heterophily, meta-paths and sheaves. Problem shows what existing methods do with one concrete path and what they lose. Method presents our answer. Experiments states how each claim will be tested. Discussion separates what is new from what is not.

## Slide 5 — A heterogeneous graph: nodes/edges have types

This is the running example for the whole talk. Papers, an author and a subject: nodes have types, and edges have types too, such as "writes" or "has subject". Formally, a heterogeneous graph has a type function for nodes and one for edges. Our task is to predict the field of each paper, shown by colour. Standard benchmarks of this kind are ACM, DBLP, IMDB and Freebase from the HGB benchmark. The point of this slide: a neighbour's type matters, so a model should not treat every neighbour the same way.

## Slide 6 — Meta-paths: a type pattern versus one concrete path

A meta-path schema is a type pattern: paper, author, paper means two papers share an author. An instance is one concrete path that follows it, such as P1, A7, P9. The middle node, A7, is the intermediate node. Our research question is about this node: does it matter who A7 is, and should A7 itself learn that P1 and P9 are linked? Keep this question in mind; every later slide answers part of it.

## Slide 7 — Heterophily: connected nodes have different labels

A GCN updates a node with the average of its neighbours. That works when neighbours share the label, which we call homophily, left. It fails under heterophily, right: neighbours have different labels and averaging mixes them. The simplest measure is edge homophily: the fraction of edges whose two ends have the same label. Near one is homophilic, near zero heterophilic. In our example P1 and P9 share author A7 but are in different fields. So being connected does not mean being similar, and that is the first difficulty.

## Slide 8 — Measuring heterophily: H2[6] on heterogeneous graph

On a typed graph, labels are compared between papers, so heterophily is measured on the paper graphs that meta-paths induce. The H2GB benchmark defines the index H-squared: for each length-two meta-path, build the induced paper graph, measure its adjusted homophily, which corrects for class imbalance, and average one minus that value. Zero means homophilic; around one, heterophilic. The chart shows our values: DBLP is homophilic, IMDB clearly heterophilic, IEEE-CIS-G and Freebase almost fully heterophilic. Freebase is dominated by unlabeled books, which the reference code fills at random; we also report a labeled-only value. With this range we can test whether gains grow with heterophily.

## Slide 9 — Sheaf diffusion: agree inside the edge, stay different

Two neighbours of different classes: u, blue, roughly east; v, pink, roughly north. Each end of the edge has its own map into the edge's space: here u's map rotates by plus thirty degrees and v's by minus sixty. The dashed arrows are the two nodes as the edge sees them. One diffusion step computes the gap inside the edge and moves both nodes, u by F_u-transpose times the gap and v the opposite way, so both original vectors move. Top row: GCN has identity maps, so the gap is just the difference of the two vectors and they merge into one, losing the class. Bottom row: with maps, the dashed arrows meet inside the edge, while u ends almost exactly east and v almost exactly north: the small noise is removed and the classes are kept. That is the whole idea of a sheaf: agreement is required only after the maps.

## Slide 10 — How heterophily is handled today: decompose, then fuse

How do current methods handle a graph that is both heterogeneous and heterophilic? Most of them first decompose it. Step one is the original graph. Step two builds one paper-only graph per meta-path schema: in the PAP view P1 is linked to P9 and P4 because they share an author; in the PSP view P1 is linked to P9 because they share a subject. Notice that A7 and S3 are no longer in either graph. Step three runs a network on each view, and this is where heterophily is handled, for example by separating homophilic and heterophilic neighbours of the derived graph. Step four fuses the views, usually with attention. Hetero2Net, HDHGR and LatGRL follow this pattern for heterophily; HAN and SeHGNN do the same for heterogeneous graphs in general; even MAGNN, which keeps instances, processes each schema separately.

## Slide 11 — Two limitations of decompose-then-fuse

Decompose-then-fuse has two limitations. Left: heterophily is judged and handled on the derived graphs, one homophily value per view, not on the original typed graph; the original edges such as P1 to A7 never enter the computation, and which neighbours count depends on a hand-picked schema list. Right: within one schema, two paper pairs are distinguished only by their endpoints. The PAP view is the matrix product of paper-author and author-paper adjacency, so its entry for P1 and P9 is just the number of shared authors. Whether the path goes through A7 or A2 makes no difference, and A7 itself never learns that P1 and P9 are linked. Both limitations come from replacing the graph by schema views before learning starts. Our two goals answer them one to one.

## Slide 12 — Our view and the research questions

Our view, in four points. Heterophily is handled on the original graph. Meta-paths are part of the structure the model operates on, not extra input. Each sampled path instance becomes a face attached to the nodes and edges it traverses, with no closing edge, so the path does not need to return to its start. And each instance has its own learned weight. This gives two research questions. RQ1: can a single model on the original graph handle heterophily using path instances, without per-schema branches? RQ2: do path instances, used as structure, carry information that schema-level encoders and instance-level features do not? In terms of our example: does it matter who A7 is, and should A7 itself learn that P1 and P9 are linked?

## Slide 13 — Related work: where each family puts the path

Here are the families, compared on the properties our questions need. Decompose-then-fuse methods for heterophily, such as Hetero2Net, HDHGR and LatGRL, and schema-level encoders such as HAN and SeHGNN, work on derived graphs and never see an instance. Instance encoders such as MAGNN and MECCH do use the instance and its intermediate node, but as a feature written into the target node only; A7 is not updated. Sheaf networks such as NSD and HetSheaf stay on the original graph and handle heterophily with learned edge maps, but they only see edges. Cell-complex and sheaf-hypergraph networks update every node of a cell, but their cell maps are free and they do not target typed graphs or heterophily. Our model is the row at the bottom: on the original graph, the instance as a face, every node updated, and heterophily handled by edge maps and their compositions along the path.

## Slide 14 — Contributions

Our contributions, each tied to a research question. One: path instances become faces of the same sheaf that lives on the graph, with maps composed from the edge maps, so every node of the path, including the intermediate one, hears the path, with no closing edge; this serves RQ2. Two: one objective on the original graph handles heterophily through edge maps and meta-path information through faces, without per-schema branches; that is RQ1. Three: WISP samples path instances without a schema list and learns a weight for each. Four: an evaluation designed to answer the reviewers: eight datasets across the heterophily range, every baseline rerun, and every ablation a configuration of the same model.

## Slide 15 — H2S-NN in one picture: what flows, and why each step

The whole model in six steps, with what flows between them. Encode: every node type goes into one d-dimensional space, giving h. Edge maps: each end of each edge gets a map F, so that different classes can agree after the map. Faces: sampled path instances, each with a weight beta, so that each path, including its intermediate node, is kept. Objective: one rule, stay close to h plus tau times the energy E. Solve: the propagated vectors are the minimiser. Predict: a small network reads both the own encoding and the propagated vector. The task loss sits at the end and its gradient flows back through one extra linear solve. The next slides build this up from the simplest propagation rule, showing why each piece is there.

## Slide 16 — What must propagation do? Two requirements

Before any equation, what must propagation do? We use one toy example throughout: four nodes, two-dimensional encodings. ML papers P1 and P4 point east, the biology paper P9 points north, A7 has a learned encoding pointing east, and P1 and P9 carry some noise. Requirement H, for heterophily: P1 and P9 are linked through A7, yet after propagation they must still point different ways; only the noise should go. Requirement M, for meta-paths: the path P1-A7-P9 must reach all three of its nodes, and its effect must depend on who A7 is, not just on the pattern PAP. H is research question 1, M is research question 2. The plan: start from the simplest propagation rule, watch it fail on this example, and fix one term at a time.

## Slide 17 — Step 0: the simplest rule (APPNP) and where it fails

Step zero is the simplest rule we can write: stay close to your encoding, and be equal to your neighbours. Its minimiser is APPNP, personalised PageRank, and one gradient step on it is a GCN layer. On our example, with tau one, requirement H fails: P9 moves from pointing north to pointing halfway between north and east, because A7 links it to two ML papers; the cosine between P1 and P9 jumps from 0.47 to 0.83. Requirement M also fails: the path is only two edges, A7 is just another neighbour, and swapping A7 for any other author changes nothing. What we keep from this rule is its form: a unique minimiser and the anchor to h, which prevents oversmoothing. What we change is the second term: being equal is the wrong notion of agreement.

## Slide 18 — Fix 1: agree after a learned map (requirement H)

Fix one. Instead of comparing x_u and x_v directly, each end of the edge is first mapped into the edge's own space, and agreement is required only there. On the example we let the map of P9 on edge e2 rotate north to east and leave the others as identity. Now requirement H holds: P9 keeps pointing north, its noise is removed, and the cosine between P1 and P9 drops to 0.25. Why: inside e2, P9 and A7 both point east, so there is nothing left to smooth; only the part of the disagreement that the maps cannot explain is removed. In the real model F is a rotation or reflection, so it never shrinks a vector; it is computed from both ends of the edge, and it is learned from the labels through the task loss. We proved that for any labelling such maps exist, so a class-separated state has zero energy and is kept. This is how heterophily is handled on the original graph. Requirement M still fails: the path is still two edges.

## Slide 19 — Fix 2: a state for each path instance (requirement M)

Fix two. Each sampled path instance f gets its own state z_f, and every node on the path must agree with it after its path map T. T is not learned: it is the composition of the edge maps along the path, so A7 is carried through e1 into P1's frame and P9 through e2 and e1. Now requirement M holds. Compare with the two-edge view: A7 now meets the path state directly instead of being just a neighbour; who A7 is matters, because T and the weight beta read A7; the order of relations matters, because T is composed in path order, so P-A-P and P-S-P give different maps; and there is no shortcut between P1 and P9. On the example the face adds one more consistent witness: P9's noise drops further, to 0.12, with no new parameters. Both requirements now hold.

## Slide 20 — WISP: which faces, and how strong each one is

Which paths become faces, and how strongly does each count? First, sampling. From every target node we draw random walks: a uniform length, then at each hop a uniform choice of relation and a uniform unvisited neighbour. On our example the walk P1-A7-P9 has probability one twelfth, P1-S3-P9 one sixth. One walk is one face; m_f counts how often the face was drawn, divided by the number of walks per node, which makes the result unbiased. No schema list is used, and walks are redrawn every epoch. Second, weighting. A small network g reads every node encoding and relation on the path, so two instances of the same schema, through A7 or through A2, can get very different weights. The product is beta_f, the face weight in the objective. Schemas are read out after training, never selected; v1 needed a schema search and four budgets.

## Slide 21 — WISP vs. other ways of choosing meta-paths

Is WISP just random sampling, and is it better than meta-path search? Compared with other papers. HAN uses an expert list and SeHGNN all meta-paths up to k hops; both decide at the schema level. GTN learns a soft mix of edge types end to end, still at the schema level. PMMM and LMSPS search for meta-paths and keep the top k, LMSPS by validation loss; AMP-HGNAS and MCTS-HGNN use architecture search or tree search. AHGNN enumerates all meta-paths up to a length and attends per node, using hop-wise averages. MECCH uses all meta-paths of length K with intermediate nodes averaged into the target only. MF2Vec is closest to us: schema-free walks and intermediate facets, but the path ends up as an edge weight. Our v1 searched schemas with a held-out split. WISP's random walks only propose candidates from a stated distribution; the decision is made per instance by the learned weight, and the instance acts on every node of the path. So: schema-free walks are not new; choosing per instance and updating every node is. Honestly, we have not yet shown that this beats search methods.

## Slide 22 — Putting it together: one objective, one solve

Putting it together. The objective has the anchor, the edge term with maps, and the face term. At the minimiser the gradient is zero, which gives a balance equation for every node: x-star is a weighted average of its own encoding, its neighbours carried through the edge maps into its frame, and the states of the paths it lies on carried back through T. Iterating this balance would be message passing; we instead solve the linear system directly by conjugate gradient, about ten steps, and the answer is unique for any parameters, with tau learned and no depth to tune. The head reads x-star together with h, so the node's own information is kept separate. This is exactly the rule we set out to build: the maps answer requirement H and the faces answer requirement M.

## Slide 23 — What is learned, and what the loss is

What is learned and what the loss is. The learned parts are the type encoders, the edge-map network that outputs two reflection vectors per edge end plus a frame per relation and role, the edge-weight network, the instance-weight network g, one scalar tau, and the head. The face maps T are composed from F, and the sampler and solver have no parameters. The training problem is: minimise the task loss of the head applied to x-star and h, where x-star is the minimiser of the objective. So the objective J is not the loss; it defines the forward pass, and the loss is the usual cross-entropy or binary cross-entropy. The gradient reaches every parameter through one adjoint solve with the same matrix, so memory is proportional to the states and not to the number of solver steps. This is also the answer to 'where do the maps come from': the task loss shapes them, so the model learns what agreement means on each graph.

## Slide 24 — How the method answers the research questions

Finally, how the method maps onto the research questions. RQ1, heterophily on the original graph without per-schema branches, is carried by the edge maps, the anchor to h, the head reading both parts, and the fact that there is one objective on the original graph; it is tested by the gain against H-squared across the eight datasets, by replacing the maps with the identity, and by comparison with Hetero2Net, HetSheaf and a tuned R-GCN. RQ2, path instances as structure, is carried by the face state, the composed maps, the per-instance weight and the fact that every node on the path is updated; it is tested by removing faces, replacing them with closing edges or features, restricting the weight, and by a known-answer task. WISP serves both: candidates without a schema list, selection per instance. Every mechanism was added for one requirement, and every one has an ablation that can remove it.

## Slide 25 — Datasets: three tasks, homophilic to heterophilic

Three tasks. Node classification uses eight datasets: four from HGB, ACM, DBLP, IMDB and Freebase, and four large heterophilic graphs from H2GB: IEEE-CIS-G for fraud, mag-year and ogbn-mag, and H-Pokec. The large ones need mini-batch training, which still has to be designed. Link prediction has eight tasks, four derived from the node graphs so types and labels are real, plus PubMed and LastFM and two large ones. Graph classification uses the four TU datasets of HetSheaf; their types come from labels, so it only shows that the model covers the task, with no heterophily claim.

## Slide 26 — Heterophily metrics: labels (MLH) and features (MDE)

Two heterophily metrics complement H-squared from the background. MLH, from Hetero2Net: for each length-two meta-path, build the graph it induces on target nodes and count the fraction of edges whose ends share a label; average over meta-paths. Near one is homophilic. On our toy PAP graph it is one half. MDE looks at features instead of labels: the average degree-normalised feature difference across meta-path edges; large means linked nodes look different. It depends on feature scale, so it describes a dataset rather than ranking datasets. All three metrics use the same meta-path list, and we first reproduce Hetero2Net's published values.

## Slide 27 — Evaluation metrics and protocol

Metrics and protocol on one slide. Node and graph tasks report Macro-F1, where every class counts equally, and Micro-F1, which equals accuracy for single-label data. Link prediction follows HGB: ROC-AUC and MRR against negatives drawn from the head's two-hop neighbourhood. The protocol is one rule for every method: the same files and splits; Bayesian search with one hundred trials on seed zero using only the validation metric; retrain the best configuration on seeds zero to four; report mean and standard deviation; and test with Welch's t-test, raw and adjusted p-values. Every baseline, HetSheaf included, is rerun this way, which answers concern 4.

## Slide 28 — Cost: how it grows, and what we measured

Cost. Left, how it grows: maps are linear in edges times d, never d-by-d matrices; faces are linear in face incidences; the solve needs about ten passes over edges and incidences; and memory holds the states only, it does not grow with the number of solver steps. v1 built d-by-d maps at cubic cost in d, and meta-path methods such as HAN and SeHGNN build one graph per schema, which can be much denser than the original. Right, measured peak memory on ACM: v1 needed 9.3 GB; the current model 2.8 GB, below HAN and HINormer. The other bars come from the v1 paper's table on a different machine, and ours from a laptop GPU, so read it as an order of magnitude. A fair time comparison will be done on the server; the solver needs 9 to 11 steps on all three HGB datasets.

## Slide 29 — Ablation study: which part carries the gain?

Which part carries the gain? Each row changes exactly one thing, with the same sampled paths, parameter budget and tuning. Removing all faces tests whether path instances add anything. Replacing a face by a closing edge tests whether it is more than its endpoints, reviewer concern 2. Feeding the path only to the head, as MAGNN does, tests structure against features. A uniform or endpoint-only weight tests whether who is on the path matters. These four answer RQ2. Identity edge maps test whether the maps are what handle heterophily, RQ1. Free face maps test the composition. What we write if the full model does not win is fixed before running.

## Slide 30 — Model analysis: when and why it works

Six analyses explain when and why. Gain against H-squared tests whether the benefit grows with heterophily, RQ1. Known-answer synthetic tasks, where the label is decided by the intermediate node, test the mechanism directly. The schema read-out shows which schemas get weight and whether that is stable across seeds. Path length tests sensitivity to l-max. Scaling measures memory and time against graph size and number of faces, concern 3. And a case study of which instances get high weight, by intermediate type and whether the endpoints share a label, gives the qualitative side of RQ2.

## Slide 31 — What is new, and what is not

To be precise about novelty. New: paths are faces of the graph's own sheaf; a face reaches every node of its path without a shortcut; heterophily and meta-path information sit in one objective; and every ablation is a configuration of the same model. Not new: using intermediate nodes, which MAGNN and SGAT do; sheaves on typed graphs, which HetSheaf introduced; and the solver, which is personalised PageRank. With relation-only maps and the path used only as a feature, our face reduces to MAGNN's instance encoder; we state that openly.

## Slide 32 — Then and now: what changed since the submission

Side by side with the submitted version. The story moved from three contributions to two goals. The face used to reach nodes only through a closing edge; now it reaches every node directly. K Hodge steps became one objective and one solve. AMPSS with four budgets became WISP. Three node datasets, one of them heterophilic, became eight across the heterophily range. HetSheaf was quoted; now every baseline is rerun with the same budget. And ACM memory went from 9.3 to 2.8 GB.

## Slide 33 — Limitations and next steps

Limitations we will state. Influence is symmetric in strength: an edge or a face pulls all its members equally, and direction lives only in the maps; that is the price of having an energy with a unique solution. Propagation is linear and all non-linearity is in the head. Edge states carry no input of their own because the benchmarks have no edge features. Paths are simple, up to a maximum length, and sampled. Graph classification is typed by label. And five seeds on fixed splits measure seed variance only. Next steps, in order: the heterophily harness, starting by reproducing Hetero2Net's published values; the protocol code; the mini-batch design for the large graphs; the baseline plan including a HetSheaf adapter; and then the server runs and the analysis part.

## Slide 34 — Conclusion

To conclude. The problem: heterophily is usually handled on per-schema graphs, and paths are reduced to their endpoints. Our idea: stay on the original graph, use learned maps for heterophily, and make each path instance a face. The method is one objective solved once, with WISP sampling and weighting the paths. Next, the two research questions are tested on node, link and graph tasks with ablations fixed in advance. The model is implemented and tested; there are no accuracy numbers yet. Thank you.

## Slide 35 — References

References for the citations on the slides.

## Slide 36 — References (cont.)

References for the citations on the slides.

## Slide 37 — References (cont.)

References for the citations on the slides.

## Slide 38 — Appendix: where WISP can lose, and how we test it

Backup for the WISP question. Where it can be weaker: rare long schemas are sampled rarely, while LMSPS reports gains at five to six hops; the sampling budget is not focused, our adaptive proposal reduced variance by only 4.9 percent, below the 20 percent gate; the weight channel has the capacity to select, shown with an oracle at 0.99, but the learned weight did not yet pick the right instances on a synthetic task; and schema-free walks themselves are not new. Tests: the weight ladder, the expert list on the same model, a proposed run that feeds LMSPS's searched schemas into our model at the same budget, and the path-length curve. If search wins, the claim narrows to no schema list, linear cost and per-instance weights.

## Slide 39 — Appendix: hyperparameters per step (tuned, fixed, learned)

Every knob, step by step. Red: the five things we tune, d, two dropouts, learning rate and weight decay, plus pooling for graph classification. Fixed and never tuned: two reflections per map, maximum walk length 4 and the number of walks, set by a rule; solver tolerance and step limit. Learned: encoders, maps, weights, and tau, the strength of agreement. v1 tuned nine and needed four hand-set budgets; now five tuned plus two fixed path settings, and no depth.

## Slide 40 — Appendix: the full energy (edge states and normalisation)

Backup for questions: the full energy as implemented. Each edge has its own state s_e, the meeting point of its two ends and every face through it; for edges on no face it is eliminated exactly and gives the simpler form used in the main slides. Edge-to-face maps are also composed. The operator is normalised by its diagonal, which bounds the condition number independently of graph size.

## Slide 41 — Appendix: why v1’s face never reached the nodes

Backup: the formal reason behind reviewer concern 2.

## Slide 42 — Appendix: terms used in this talk

Glossary for questions.
