# Speaker script — H2S-NN progress talk, 2026-11-11 (v5)

Deck: `progress_2026-11-11_v5.pptx` (40 slides: 33 main + 3 references + 4 appendix). Same text as speaker notes.

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

## Slide 15 — H2S-NN in one picture: what each step is for

The model in six steps, each with its purpose. Encode puts every node type into one vector space. Edge maps let neighbours of different classes agree after a map instead of being forced to be equal: this is the heterophily part. Faces keep each sampled path instance, including its intermediate node: this is the meta-path part. Step four writes one objective that combines everything: stay close to your own encoding, and agree with neighbours and paths after the maps. Step five: the node vectors we use are the minimiser of that objective. Step six: a small network reads both the node's own encoding and its propagated vector and predicts. Training sends the gradient back through one extra linear solve. The next slides explain steps four, two and three in that order, because the objective is what ties them together.

## Slide 16 — The objective, built up from a GCN in three steps

This is where the energy is used, built up in three steps. Row one: what GCN and APPNP do. Keep each node close to its encoding h, and penalise the difference between neighbours. The minimiser of this is exactly APPNP, personalised PageRank, and one gradient step on it is a GCN-style averaging layer. It pulls neighbours to be equal, which blurs classes under heterophily. Row two: replace the plain difference by the difference after each node's map into the edge, the sheaf from the background. Neighbours now only need to agree after the maps, so different classes can stay apart. Row three: add one term per sampled path instance. The path gets its own state z_f, and every node on the path, including the intermediate one, is asked to agree with it after its own map, weighted by beta_f. Together this is the objective J: stay near h plus tau times the energy E. The propagated node vectors x-star are its minimiser. Same recipe as APPNP; the energy is what changes. Notation is the same on every following slide.

## Slide 17 — Why the edge term keeps different classes apart

Why does this preserve heterophily? Take P1 in machine learning and P9 in biology, both written by A7. Suppose each class has a vector: east for ML, north for biology. If the maps on the edge P1-A7 rotate the ML vector and A7's vector onto the same edge vector, and the maps on A7-P9 do the same for A7 and the biology vector, then the state where every node sits at its class vector has zero energy: nothing to smooth. We proved that such maps always exist for any labelling, even with only two reflections per map. Consequences: a class-separated state costs nothing; if the encoding is already consistent, the solution is the encoding itself; and the solve only removes disagreement that the maps cannot explain. On top of this, the anchor term and the head reading both x-star and h keep each node's own identity. The maps are learned from the labels, so the model learns what agreement means on this graph, on the original edges, without splitting it into views.

## Slide 18 — How a face keeps the meta-path instance

How a face keeps the meta-path instance P1-A7-P9. The face has its own state z_f, the pink circle. Each node is carried into the face's frame by a map T, and T is not new: it is the composition of the edge maps along the path. A7 is carried through edge e1 into P1's frame; P9 is carried through e2 and then e1. At the optimum, z_f is simply the average of the three nodes as the path sees them, and each node in turn is pulled toward z_f. The table compares this with the endpoint view, which turns the path into an edge P1-P9 with a count. The face keeps who A7 is, because the maps and the weight read A7's features; it keeps the order of relations, because T is composed in path order, so PAP and PSP give different maps; A7 itself learns that P1 and P9 are linked; P1 hears P9 through the path state, not through a shortcut; and there is no P1-P9 term at all. So the path is part of the structure, not an extra input.

## Slide 19 — WISP: which faces, and how strong each one is

Which paths become faces, and how strongly does each count? First, sampling. From every target node we draw random walks: a uniform length, then at each hop a uniform choice of relation and a uniform unvisited neighbour. On our example the walk P1-A7-P9 has probability one twelfth, P1-S3-P9 one sixth. One walk is one face; m_f counts how often the face was drawn, divided by the number of walks per node, which makes the result unbiased. No schema list is used, and walks are redrawn every epoch. Second, weighting. A small network g reads every node encoding and relation on the path, so two instances of the same schema, through A7 or through A2, can get very different weights. The product is beta_f, the face weight in the objective. Schemas are read out after training, never selected; v1 needed a schema search and four budgets.

## Slide 20 — WISP vs. other ways of choosing meta-paths

Is WISP just random sampling, and is it better than meta-path search? Compared with other papers. HAN uses an expert list and SeHGNN all meta-paths up to k hops; both decide at the schema level. GTN learns a soft mix of edge types end to end, still at the schema level. PMMM and LMSPS search for meta-paths and keep the top k, LMSPS by validation loss; AMP-HGNAS and MCTS-HGNN use architecture search or tree search. AHGNN enumerates all meta-paths up to a length and attends per node, using hop-wise averages. MECCH uses all meta-paths of length K with intermediate nodes averaged into the target only. MF2Vec is closest to us: schema-free walks and intermediate facets, but the path ends up as an edge weight. Our v1 searched schemas with a held-out split. WISP's random walks only propose candidates from a stated distribution; the decision is made per instance by the learned weight, and the instance acts on every node of the path. So: schema-free walks are not new; choosing per instance and updating every node is. Honestly, we have not yet shown that this beats search methods.

## Slide 21 — Propagation = the minimiser: what it means

What does 'propagation is the minimiser' mean? At the minimiser the gradient is zero, so every node is in balance between the pulls on it. For A7: its own encoding h, its neighbours P1 and P9 carried over through the edge maps, and every path it lies on through the face state. Setting the derivative to zero gives the formula at the bottom: x-star of v is a weighted average of its own encoding, its neighbours transported into its frame, and the states of its paths transported back. One gradient step on the objective is exactly one message-passing layer of this form; the minimiser is what infinitely many such layers converge to, and the anchor to h prevents oversmoothing. We do not iterate layers; we solve the linear system directly with conjugate gradient, about ten steps on the HGB datasets, and the answer is unique for any parameters. The head then reads both x-star and h.

## Slide 22 — Training: forward, loss, and the gradient path

One training epoch. Forward: draw fresh walks, compute maps and weights once, solve for x-star and z-star, apply the head, compute the loss. Backward: instead of unrolling the solver we solve one more system with the same matrix to get lambda, and the gradient for every parameter, maps, weights, tau and encoders, follows from it. So memory does not grow with the number of solver steps. The loss depends only on the task: cross-entropy for node classification, binary per genre for IMDB, binary cross-entropy with one negative per positive for links, and cross-entropy on the pooled node readout for graphs.

## Slide 23 — Datasets: three tasks, homophilic to heterophilic

Three tasks. Node classification uses eight datasets: four from HGB, ACM, DBLP, IMDB and Freebase, and four large heterophilic graphs from H2GB: IEEE-CIS-G for fraud, mag-year and ogbn-mag, and H-Pokec. The large ones need mini-batch training, which still has to be designed. Link prediction has eight tasks, four derived from the node graphs so types and labels are real, plus PubMed and LastFM and two large ones. Graph classification uses the four TU datasets of HetSheaf; their types come from labels, so it only shows that the model covers the task, with no heterophily claim.

## Slide 24 — Heterophily metrics: labels (MLH) and features (MDE)

Two heterophily metrics complement H-squared from the background. MLH, from Hetero2Net: for each length-two meta-path, build the graph it induces on target nodes and count the fraction of edges whose ends share a label; average over meta-paths. Near one is homophilic. On our toy PAP graph it is one half. MDE looks at features instead of labels: the average degree-normalised feature difference across meta-path edges; large means linked nodes look different. It depends on feature scale, so it describes a dataset rather than ranking datasets. All three metrics use the same meta-path list, and we first reproduce Hetero2Net's published values.

## Slide 25 — Evaluation metrics and protocol

Metrics and protocol on one slide. Node and graph tasks report Macro-F1, where every class counts equally, and Micro-F1, which equals accuracy for single-label data. Link prediction follows HGB: ROC-AUC and MRR against negatives drawn from the head's two-hop neighbourhood. The protocol is one rule for every method: the same files and splits; Bayesian search with one hundred trials on seed zero using only the validation metric; retrain the best configuration on seeds zero to four; report mean and standard deviation; and test with Welch's t-test, raw and adjusted p-values. Every baseline, HetSheaf included, is rerun this way, which answers concern 4.

## Slide 26 — Hyperparameters per step: tuned, fixed, learned

Every knob, step by step. Red: the five things we tune, d, two dropouts, learning rate and weight decay, plus pooling for graph classification. Fixed and never tuned: two reflections per map, maximum walk length 4 and the number of walks, set by a rule; solver tolerance and step limit. Learned: encoders, maps, weights, and tau, the strength of agreement. v1 tuned nine and needed four hand-set budgets; now five tuned plus two fixed path settings, and no depth.

## Slide 27 — Cost: how it grows, and what we measured

Cost. Left, how it grows: maps are linear in edges times d, never d-by-d matrices; faces are linear in face incidences; the solve needs about ten passes over edges and incidences; and memory holds the states only, it does not grow with the number of solver steps. v1 built d-by-d maps at cubic cost in d, and meta-path methods such as HAN and SeHGNN build one graph per schema, which can be much denser than the original. Right, measured peak memory on ACM: v1 needed 9.3 GB; the current model 2.8 GB, below HAN and HINormer. The other bars come from the v1 paper's table on a different machine, and ours from a laptop GPU, so read it as an order of magnitude. A fair time comparison will be done on the server; the solver needs 9 to 11 steps on all three HGB datasets.

## Slide 28 — Ablation study: which part carries the gain?

Which part carries the gain? Each row changes exactly one thing, with the same sampled paths, parameter budget and tuning. Removing all faces tests whether path instances add anything. Replacing a face by a closing edge tests whether it is more than its endpoints, reviewer concern 2. Feeding the path only to the head, as MAGNN does, tests structure against features. A uniform or endpoint-only weight tests whether who is on the path matters. These four answer RQ2. Identity edge maps test whether the maps are what handle heterophily, RQ1. Free face maps test the composition. What we write if the full model does not win is fixed before running.

## Slide 29 — Model analysis: when and why it works

Six analyses explain when and why. Gain against H-squared tests whether the benefit grows with heterophily, RQ1. Known-answer synthetic tasks, where the label is decided by the intermediate node, test the mechanism directly. The schema read-out shows which schemas get weight and whether that is stable across seeds. Path length tests sensitivity to l-max. Scaling measures memory and time against graph size and number of faces, concern 3. And a case study of which instances get high weight, by intermediate type and whether the endpoints share a label, gives the qualitative side of RQ2.

## Slide 30 — What is new, and what is not

To be precise about novelty. New: paths are faces of the graph's own sheaf; a face reaches every node of its path without a shortcut; heterophily and meta-path information sit in one objective; and every ablation is a configuration of the same model. Not new: using intermediate nodes, which MAGNN and SGAT do; sheaves on typed graphs, which HetSheaf introduced; and the solver, which is personalised PageRank. With relation-only maps and the path used only as a feature, our face reduces to MAGNN's instance encoder; we state that openly.

## Slide 31 — Then and now: what changed since the submission

Side by side with the submitted version. The story moved from three contributions to two goals. The face used to reach nodes only through a closing edge; now it reaches every node directly. K Hodge steps became one objective and one solve. AMPSS with four budgets became WISP. Three node datasets, one of them heterophilic, became eight across the heterophily range. HetSheaf was quoted; now every baseline is rerun with the same budget. And ACM memory went from 9.3 to 2.8 GB.

## Slide 32 — Limitations and next steps

Limitations we will state. Influence is symmetric in strength: an edge or a face pulls all its members equally, and direction lives only in the maps; that is the price of having an energy with a unique solution. Propagation is linear and all non-linearity is in the head. Edge states carry no input of their own because the benchmarks have no edge features. Paths are simple, up to a maximum length, and sampled. Graph classification is typed by label. And five seeds on fixed splits measure seed variance only. Next steps, in order: the heterophily harness, starting by reproducing Hetero2Net's published values; the protocol code; the mini-batch design for the large graphs; the baseline plan including a HetSheaf adapter; and then the server runs and the analysis part.

## Slide 33 — Conclusion

To conclude. The problem: heterophily is usually handled on per-schema graphs, and paths are reduced to their endpoints. Our idea: stay on the original graph, use learned maps for heterophily, and make each path instance a face. The method is one objective solved once, with WISP sampling and weighting the paths. Next, the two research questions are tested on node, link and graph tasks with ablations fixed in advance. The model is implemented and tested; there are no accuracy numbers yet. Thank you.

## Slide 34 — References

References for the citations on the slides.

## Slide 35 — References (cont.)

References for the citations on the slides.

## Slide 36 — References (cont.)

References for the citations on the slides.

## Slide 37 — Appendix: where WISP can lose, and how we test it

Backup for the WISP question. Where it can be weaker: rare long schemas are sampled rarely, while LMSPS reports gains at five to six hops; the sampling budget is not focused, our adaptive proposal reduced variance by only 4.9 percent, below the 20 percent gate; the weight channel has the capacity to select, shown with an oracle at 0.99, but the learned weight did not yet pick the right instances on a synthetic task; and schema-free walks themselves are not new. Tests: the weight ladder, the expert list on the same model, a proposed run that feeds LMSPS's searched schemas into our model at the same budget, and the path-length curve. If search wins, the claim narrows to no schema list, linear cost and per-instance weights.

## Slide 38 — Appendix: the full energy (edge states and normalisation)

Backup for questions: the full energy as implemented. Each edge has its own state s_e, the meeting point of its two ends and every face through it; for edges on no face it is eliminated exactly and gives the simpler form used in the main slides. Edge-to-face maps are also composed. The operator is normalised by its diagonal, which bounds the condition number independently of graph size.

## Slide 39 — Appendix: why v1’s face never reached the nodes

Backup: the formal reason behind reviewer concern 2.

## Slide 40 — Appendix: terms used in this talk

Glossary for questions.
