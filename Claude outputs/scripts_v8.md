# Speaker scripts — progress_2026-11-11_v8.pptx (all 61 slides)

Notes are read from the v8 file itself, so they match what is on each slide. Slides 38–40 and 59–60 are reference lists, 41 is the backup divider, 61 is the glossary: only a one-line note.

## Slide 1: H2S-NN: Heterogeneous Heterophilic Sheaf Neural Network with Path Instances

Good afternoon. I will present H2S-NN, our model for graphs that have many node and edge types and in which connected nodes often have different labels. I will present it as I would at IJCAI: the problem, the idea, the method, how we will test it, and what is still open. The model is final and implemented; the large experiments have not run yet, so you will see no accuracy numbers today.

## Slide 2: Reviewer feedback on the submitted version

Before the new work, here is how the submitted version was received at WSDM 2027. Two reviewers leaned to accept and two to reject; the meta-reviewer recommended rejection. All of them liked the core idea: keeping each concrete meta-path instance as a higher-order cell instead of collapsing it into a relation between its two ends. Three found the construction principled, with the face maps derived from the edge maps. The breadth of tasks was also appreciated. The next slide lists what they asked us to fix, in their own terms.

## Slide 3: Reviewer feedback: the five main concerns

These are the five main concerns, condensed but in the reviewers' terms. First, the heterophilic setting was not established: by our own measure only IMDB was strongly heterophilic, the graph-classification datasets only became heterogeneous by treating atom labels as types, and we never related gains to the level of heterophily. Second, and most important technically: reviewer 2 showed that in our model the face reached the nodes only through the synthetic closing edge between the two ends, which is the endpoint view we criticised; and the ablations did not separate the face from schema selection, weighting and extra capacity. Third, cost: 9.3 GB on ACM and no scaling study. Fourth, protocol: baseline tuning was not described and HetSheaf was quoted, not rerun. Fifth, clarity. The first three concern the core claim; the rest concern evaluation and writing.

## Slide 4: Update since last meeting

What changed since the last meeting, in four columns. Contributions: the paper no longer claims AMPSS and SheafCovPool; there is one core idea, path instances as faces with heterophily handled on the original graph, plus WISP as the sampler. Methodology: the closing edge, the K Hodge diffusion steps and the d-by-d maps are gone; faces sit above their paths, propagation is one energy and one solve, and every map is matrix-free, two reflections per edge end, so restriction maps are still there but never built as dense matrices. Path sampling: AMPSS with its schema search and four budgets became WISP. Experiments: from three, four and four datasets with quoted HetSheaf numbers to eight, eight and four datasets with every baseline rerun under one protocol. Status: the model is implemented and passes the theory, gradient and mutation tests, ACM memory went from 9.3 to 2.8 GB, and no accuracy runs have been made yet.

## Slide 5: Outline

The talk has five parts. Background gives the four ideas you need: typed graphs, heterophily, meta-paths and sheaves. Problem shows what existing methods do with one concrete path and what they lose. Method presents our answer. Experiments states how each claim will be tested. Discussion separates what is new from what is not.

## Slide 6: A heterogeneous graph: nodes/edges have types

This is the running example for the whole talk. Papers, an author and a subject: nodes have types, and edges have types too, such as "writes" or "has subject". Formally, a heterogeneous graph has a type function for nodes and one for edges. Our task is to predict the field of each paper, shown by colour. Standard benchmarks of this kind are ACM, DBLP, IMDB and Freebase from the HGB benchmark. The point of this slide: a neighbour's type matters, so a model should not treat every neighbour the same way.

## Slide 7: Meta-paths: a type pattern versus one concrete path

A meta-path schema is a type pattern: paper, author, paper means two papers share an author. An instance is one concrete path that follows it, such as P1, A7, P9. The middle node, A7, is the intermediate node. Our research question is about this node: does it matter who A7 is, and should A7 itself learn that P1 and P9 are linked? Keep this question in mind; every later slide answers part of it.

## Slide 8: Heterophily: connected nodes have different labels

A GCN updates a node with the average of its neighbours. That works when neighbours share the label, which we call homophily, left. It fails under heterophily, right: neighbours have different labels and averaging mixes them. The simplest measure is edge homophily: the fraction of edges whose two ends have the same label. Near one is homophilic, near zero heterophilic. In our example P1 and P9 share author A7 but are in different fields. So being connected does not mean being similar, and that is the first difficulty.

## Slide 9: Measuring heterophily: H2[6] on heterogeneous graph

On a typed graph, labels are compared between papers, so heterophily is measured on the paper graphs that meta-paths induce. The H2GB benchmark defines the index H-squared: for each length-two meta-path, build the induced paper graph, measure its adjusted homophily, which corrects for class imbalance, and average one minus that value. Zero means homophilic; around one, heterophilic. The chart shows our values: DBLP is homophilic, IMDB clearly heterophilic, IEEE-CIS-G and Freebase almost fully heterophilic. Freebase is dominated by unlabeled books, which the reference code fills at random; we also report a labeled-only value. With this range we can test whether gains grow with heterophily.

## Slide 10: Sheaf diffusion: agree inside the edge, stay different

Two neighbours of different classes: u, blue, roughly east; v, pink, roughly north. Each end of the edge has its own map into the edge's space: here u's map rotates by plus thirty degrees and v's by minus sixty. The dashed arrows are the two nodes as the edge sees them. One diffusion step computes the gap inside the edge and moves both nodes, u by F_u-transpose times the gap and v the opposite way, so both original vectors move. Top row: GCN has identity maps, so the gap is just the difference of the two vectors and they merge into one, losing the class. Bottom row: with maps, the dashed arrows meet inside the edge, while u ends almost exactly east and v almost exactly north: the small noise is removed and the classes are kept. That is the whole idea of a sheaf: agreement is required only after the maps.

## Slide 11: How heterophily is handled: decompose, then fuse

How do current methods handle a graph that is both heterogeneous and heterophilic? Most of them first decompose it. Step one is the original graph. Step two builds one paper-only graph per meta-path schema: in the PAP view P1 is linked to P9 and P4 because they share an author; in the PSP view P1 is linked to P9 because they share a subject. Notice that A7 and S3 are no longer in either graph. Step three runs a network on each view, and this is where heterophily is handled, for example by separating homophilic and heterophilic neighbours of the derived graph. Step four fuses the views, usually with attention. Hetero2Net, HDHGR and LatGRL follow this pattern for heterophily; HAN and SeHGNN do the same for heterogeneous graphs in general; even MAGNN, which keeps instances, processes each schema separately.

## Slide 12: Two limitations of decompose-then-fuse

Decompose-then-fuse has two limitations. Left: heterophily is judged and handled on the derived graphs, one homophily value per view, not on the original typed graph; the original edges such as P1 to A7 never enter the computation, and which neighbours count depends on a hand-picked schema list. Right: within one schema, two paper pairs are distinguished only by their endpoints. The PAP view is the matrix product of paper-author and author-paper adjacency, so its entry for P1 and P9 is just the number of shared authors. Whether the path goes through A7 or A2 makes no difference, and A7 itself never learns that P1 and P9 are linked. Both limitations come from replacing the graph by schema views before learning starts. Our two goals answer them one to one.

## Slide 13: Our view and the research questions

Our view, in four points. Heterophily is handled on the original graph. Meta-paths are part of the structure the model operates on, not extra input. Each sampled path instance becomes a face attached to the nodes and edges it traverses, with no closing edge, so the path does not need to return to its start. And each instance has its own learned weight. This gives two research questions. RQ1: can a single model on the original graph handle heterophily using path instances, without per-schema branches? RQ2: do path instances, used as structure, carry information that schema-level encoders and instance-level features do not? In terms of our example: does it matter who A7 is, and should A7 itself learn that P1 and P9 are linked?

## Slide 14: Related work: where each family puts the path

Here are the families, compared on the properties our questions need. Decompose-then-fuse methods for heterophily, such as Hetero2Net, HDHGR and LatGRL, and schema-level encoders such as HAN and SeHGNN, work on derived graphs and never see an instance. Instance encoders such as MAGNN and MECCH do use the instance and its intermediate node, but as a feature written into the target node only; A7 is not updated. Sheaf networks such as NSD and HetSheaf stay on the original graph and handle heterophily with learned edge maps, but they only see edges. Cell-complex and sheaf-hypergraph networks update every node of a cell, but their cell maps are free and they do not target typed graphs or heterophily. Our model is the row at the bottom: on the original graph, the instance as a face, every node updated, and heterophily handled by edge maps and their compositions along the path.

## Slide 15: Objectives and contributions

Two objectives follow from the questions. O1: show that one model on the original graph handles heterophily with path instances and without per-schema branches. O2: show that path instances, used as structure, carry information beyond schema-level encoders and beyond instance-level features. Three contributions deliver this. First, path instances become faces of the same sheaf that lives on the graph: their maps are compositions of the edge maps, and every node of the instance receives the face, intermediate nodes included, with no closing edge. Second, node representations are the minimiser of one energy over nodes, edges and faces, which has a unique solution for any parameter values and no depth to tune. Third, WISP samples instances by random walks without a schema list, with unbiased weights, and learns a weight per instance. Because everything is one energy, the comparisons we need are configurations of the same model.

## Slide 16: H2S-NN in one picture

Here is the whole model in six steps. One, the input is the typed graph itself; no meta-path list is given. Two, a type-specific encoder puts every node, whatever its type, into the same d-dimensional space; that vector is h. Three, WISP samples path instances by random walks, and each instance becomes a face with a weight beta. Four, we build a sheaf: each edge gets learned maps for its two ends, and each face gets maps that are compositions of the edge maps along its path. Five, node representations are the minimiser of one energy: stay close to h, and agree with neighbours after the maps. Six, a small MLP reads the node's own encoding and its propagated state and makes the prediction for the task: node, link or graph. Training sends the gradient back, in red, through one extra linear solve rather than through unrolled layers. The next slides go through each step with its equation.

## Slide 17: Step 1–2: encode nodes, learn a map for each edge end

Steps one and two. Each node type has its own linear encoder into a shared d-dimensional space; types without features get a learned vector per node. We also compute a context c: the mean of the neighbours' encodings, separately for each relation and role. Then every end of every edge gets a map. The map is a fixed frame for the relation and the role, times two Householder reflections whose vectors come from a small network that reads both ends of the edge. So the map from A7 into the edge depends on who A7 and P1 are, not only on their types, and source and target roles have different frames, which keeps direction. Because the map is orthogonal, it rotates or reflects and never shrinks a vector. Each edge also gets a positive weight a_e. All of this is computed once per forward pass.

## Slide 18: WISP (1): sample path instances by random walks

WISP, the sampler, step by step on our example. From P1 we first draw a length, uniformly between 2 and l-max; with l-max 4 the length 2 has probability one third. At each hop we choose uniformly among the actions, meaning relation and direction, that still have an unvisited neighbour: from P1 that is author or subject, one half each. Then we choose a uniform unvisited neighbour: from A7 that is P9 or P4, one half each. So P1-A7-P9 has probability one twelfth, P1-A7-P4 also one twelfth, and P1-S3-P9 one sixth. Each walk becomes one face. The weight m_f corrects for how we sampled: walks come from a fifty-fifty mix of p and an optional proposal q, which bounds each correction by 2 over r; with the default q equal to p, m_f is simply the number of walks that hit the face divided by r. On average, the sampled faces reproduce the operator of the walk distribution p exactly. Walks are redrawn every epoch, one fixed draw is used for evaluation, and no schema list or label enters.

## Slide 19: WISP (2): a learned weight for every instance

The second half of WISP is the weight. For each face we collect the encodings of all its nodes and the embeddings of its relations, in order; call this xi. A small network g, one per path length and shared by all schemas, turns xi into a log-weight, so w_f is positive. Multiplying by the sampling correction m_f gives beta_f, the face weight used in the energy. Because g sees who is on the path, two PAP instances, one through A7 and one through A2, can count very differently. Schemas are not selected at all; after training we can read out how much of the total weight each schema received, which gives the interpretability that schema selection used to give. Compared with v1: AMPSS enumerated schemas by depth-first search, selected some, and needed several budgets; WISP has no schema step.

## Slide 20: Step 3: a path instance becomes a face

Step three: what a face is. We keep the path on the original graph and add one new element, the face, above it. Formally the cells are ordered: a node is below the edges that contain it, and an edge and a node are below every face whose path uses them. Every comparable pair is coupled: node with edge in orange, edge with face in green, node with face directly in dashed pink. Each cell has a state that the model solves for: x on nodes, s on edges, z on faces. Important for the reviewers' main criticism: the path is never closed into a loop. There is no edge between P1 and P9, neither in the computation nor in the definition. So A7 is inside the computation: it is coupled to the face directly and through both of its edges.

## Slide 21: Face maps are composed from edge maps

How does a face compare P1, A7 and P9, which live in different frames? It reuses the edge maps. To bring A7 into the face, which uses P1's frame, we apply A7's map into edge e1 and then read the result back in P1's frame with the transpose of P1's map. For P9 we continue through edge e2. In general the node-to-face map T is the product of these steps along the path, and the edge-to-face map R is defined the same way; with these choices every route through the cells gives the same result, which is what makes this a sheaf. What this buys: faces add no parameters; the face map depends on who A7 is, through its edge maps; the order of relations matters; and a state that is consistent on the graph stays consistent after faces are added, so a face adds evidence and never a contradiction. With free per-face maps this last property is lost; we checked that numerically.

## Slide 22: Step 4: one energy over nodes, edges and faces

Step four is the single energy of the model. It has three kinds of terms, each a disagreement after the maps: node with edge in orange, edge with face in green, node with face in pink, scaled by the edge weights a from step two and the face weights beta from WISP. All states are stacked into one vector Y. The edge state s is where both ends of an edge and every face through it must meet; the face state z is where all nodes and edges of one path meet. Because every term is a squared difference, E is a quadratic form with a matrix L. We normalise L by the degrees, giving E-hat with matrix L-hat, so that high-degree nodes do not dominate. Step five minimises E-hat together with an anchor to the encodings.

## Slide 23: Every node of the path hears the face

This answers the reviewers' main technical criticism. In v1 we used Hodge diffusion on a complex with a closing edge. There, the block that sends face information to nodes is exactly zero for any number of diffusion steps; we proved it and checked it numerically. The only route from the face to the nodes was the closing edge between the two ends, which is the endpoint view we argued against. In the new energy, the node-face term puts the face directly into every node's update, including A7, and there is no coupling between P1 and P9 at all except through the face or shared edges. This is structural, by construction. Whether it improves accuracy is a separate question that the experiments must decide.

## Slide 24: Step 5: minimise the energy while staying near the encoding

Step five uses the energy from step four. The objective J has two parts: stay close to the encodings h from step one, and the step-four energy E-hat, weighted by one learned scalar tau. Every symbol here comes from an earlier step: Y is the stacked state vector, P just picks its node part x, L-hat is the matrix of E-hat. Setting the gradient to zero gives one linear system, which we solve with conjugate gradient in about ten steps; the solution is unique for any parameters. If we eliminate the edge and face states exactly, what remains on the nodes is an operator S-hat, and the node solution is the inverse of I plus tau S-hat applied to h. The plot shows what this does per frequency: x-star keeps the smooth part, h minus x-star is the rest, and the head reads both. J defines the forward pass; the training loss on the next slide is a different thing.

## Slide 25: What is learned, and what the loss is

What is learned and what the loss is. The learned parts are the type encoders, the edge-map network that outputs two reflection vectors per edge end plus a frame per relation and role, the edge-weight network, the instance-weight network g, one scalar tau, and the head. The face maps T are composed from F, and the sampler and solver have no parameters. The training problem is: minimise the task loss of the head applied to x-star and h, where x-star is the minimiser of the objective. So the objective J is not the loss; it defines the forward pass, and the loss is the usual cross-entropy or binary cross-entropy. The gradient reaches every parameter through one adjoint solve with the same matrix, so memory is proportional to the states and not to the number of solver steps. This is also the answer to 'where do the maps come from': the task loss shapes them, so the model learns what agreement means on each graph.

## Slide 26: How the method answers the research questions

Finally, how the method maps onto the research questions. RQ1, heterophily on the original graph without per-schema branches, is carried by the edge maps, the anchor to h, the head reading both parts, and the fact that there is one objective on the original graph; it is tested by the gain against H-squared across the eight datasets, by replacing the maps with the identity, and by comparison with Hetero2Net, HetSheaf and a tuned R-GCN. RQ2, path instances as structure, is carried by the face state, the composed maps, the per-instance weight and the fact that every node on the path is updated; it is tested by removing faces, replacing them with closing edges or features, restricting the weight, and by a known-answer task. WISP serves both: candidates without a schema list, selection per instance. Every mechanism was added for one requirement, and every one has an ablation that can remove it.

## Slide 27: Heterophily metrics: labels (MLH) and features (MDE)

Two heterophily metrics complement H-squared from the background. MLH, from Hetero2Net: for each length-two meta-path, build the graph it induces on target nodes and count the fraction of edges whose ends share a label; average over meta-paths. Near one is homophilic. On our toy PAP graph it is one half. MDE looks at features instead of labels: the average degree-normalised feature difference across meta-path edges; large means linked nodes look different. It depends on feature scale, so it describes a dataset rather than ranking datasets. All three metrics use the same meta-path list, and we first reproduce Hetero2Net's published values.

## Slide 28: Datasets (1): node classification, with their heterophily

The eight node-classification datasets with their heterophily. Four from HGB, four large ones from H2GB. The three right-hand columns are the three heterophily measures from the previous slide, all in heterophily form so that one means every linked pair differs. Superscript H marks values from Hetero2Net's table with its meta-path set, which we reproduced within 0.01 on ACM, DBLP and IMDB; G marks values cited from H2GB's table; unmarked values we computed with H2GB's code; a dot means still to compute, and the final table will use one meta-path set for all three measures as the spec requires. Freebase is dominated by unlabelled books that the reference code fills at random, so the labelled-only value is also given. Across the eight sets H-squared runs from 0.17 to 0.995. IEEE-CIS-G shows why H-squared is our primary measure: its MLH looks homophilic at 0.14, but after correcting for class imbalance H-squared is 0.98.

## Slide 29: Datasets (2): link prediction and graph classification

Link prediction has eight tasks under one protocol. Four are derived from the node-classification graphs, so node types and node labels are real: paper-author on ACM, author-paper on DBLP, movie-actor on IMDB, and book-book on Freebase. Two are HGB's own link sets, PubMed and LastFM; LastFM has neither features nor labels and is kept as the one stated exception. Two large tasks, citation on ogbn-mag and friendship on H-Pokec, test scale. In every task, features that are computed from the target relation are removed, because they would leak the answer. Graph classification uses the four TU datasets of HetSheaf. Their node types come from atom or structure labels, which reviewers rightly said is not a natural heterogeneous graph, so this task only shows that the model covers it, with no heterophily claim.

## Slide 30: Evaluation metrics and protocol

Metrics and protocol on one slide. Node and graph tasks report Macro-F1, where every class counts equally, and Micro-F1, which equals accuracy for single-label data. Link prediction follows HGB: ROC-AUC and MRR against negatives drawn from the head's two-hop neighbourhood. The protocol is one rule for every method: the same files and splits; Bayesian search with one hundred trials on seed zero using only the validation metric; retrain the best configuration on seeds zero to four; report mean and standard deviation; and test with Welch's t-test, raw and adjusted p-values. Every baseline, HetSheaf included, is rerun this way, which answers concern 4.

## Slide 31: Cost: how it grows, and what we measured

Cost. Left, how it grows: maps are linear in edges times d, never d-by-d matrices; faces are linear in face incidences; the solve needs about ten passes over edges and incidences; and memory holds the states only, it does not grow with the number of solver steps. v1 built d-by-d maps at cubic cost in d, and meta-path methods such as HAN and SeHGNN build one graph per schema, which can be much denser than the original. Right, measured peak memory on ACM: v1 needed 9.3 GB; the current model 2.8 GB, below HAN and HINormer. The other bars come from the v1 paper's table on a different machine, and ours from a laptop GPU, so read it as an order of magnitude. A fair time comparison will be done on the server; the solver needs 9 to 11 steps on all three HGB datasets.

## Slide 32: Ablation study: which part carries the gain?

Which part carries the gain? Each row changes exactly one thing, with the same sampled paths, parameter budget and tuning. Removing all faces tests whether path instances add anything. Replacing a face by a closing edge tests whether it is more than its endpoints, reviewer concern 2. Feeding the path only to the head, as MAGNN does, tests structure against features. A uniform or endpoint-only weight tests whether who is on the path matters. These four answer RQ2. Identity edge maps test whether the maps are what handle heterophily, RQ1. Free face maps test the composition. What we write if the full model does not win is fixed before running.

## Slide 33: Model analysis: when and why it works

Six analyses explain when and why. Gain against H-squared tests whether the benefit grows with heterophily, RQ1. Known-answer synthetic tasks, where the label is decided by the intermediate node, test the mechanism directly. The schema read-out shows which schemas get weight and whether that is stable across seeds. Path length tests sensitivity to l-max. Scaling measures memory and time against graph size and number of faces, concern 3. And a case study of which instances get high weight, by intermediate type and whether the endpoints share a label, gives the qualitative side of RQ2.

## Slide 34: What is new, and what is not

To be precise about novelty. New: paths are faces of the graph's own sheaf; a face reaches every node of its path without a shortcut; heterophily and meta-path information sit in one objective; and every ablation is a configuration of the same model. Not new: using intermediate nodes, which MAGNN and SGAT do; sheaves on typed graphs, which HetSheaf introduced; and the solver, which is personalised PageRank. With relation-only maps and the path used only as a feature, our face reduces to MAGNN's instance encoder; we state that openly.

## Slide 35: Then and now: what changed since the submission

Side by side with the submitted version. The story moved from three contributions to two goals. The face used to reach nodes only through a closing edge; now it reaches every node directly. K Hodge steps became one objective and one solve. AMPSS with four budgets became WISP. Three node datasets, one of them heterophilic, became eight across the heterophily range. HetSheaf was quoted; now every baseline is rerun with the same budget. And ACM memory went from 9.3 to 2.8 GB.

## Slide 36: Limitations and next steps

Limitations we will state. Influence is symmetric in strength: an edge or a face pulls all its members equally, and direction lives only in the maps; that is the price of having an energy with a unique solution. Propagation is linear and all non-linearity is in the head. Edge states carry no input of their own because the benchmarks have no edge features. Paths are simple, up to a maximum length, and sampled. Graph classification is typed by label. And five seeds on fixed splits measure seed variance only. Next steps, in order: the heterophily harness, starting by reproducing Hetero2Net's published values; the protocol code; the mini-batch design for the large graphs; the baseline plan including a HetSheaf adapter; and then the server runs and the analysis part.

## Slide 37: Conclusion

To conclude. The problem: heterophily is usually handled on per-schema graphs, and paths are reduced to their endpoints. Our idea: stay on the original graph, use learned maps for heterophily, and make each path instance a face. The method is one objective solved once, with WISP sampling and weighting the paths. Next, the two research questions are tested on node, link and graph tasks with ablations fixed in advance. The model is implemented and tested; there are no accuracy numbers yet. Thank you.

## Slide 38: References

References for the citations on the slides.

## Slide 39: References (cont.)

References for the citations on the slides.

## Slide 40: References (cont.)

References for the citations on the slides.

## Slide 41: Backup slide

References for the citations on the slides.

## Slide 42: Training: the loss for each downstream task

The loss depends on the task; the propagation does not. Node classification uses cross-entropy on the labelled training nodes. IMDB is multi-label with five genres, so it uses one binary cross-entropy per genre, and a genre is predicted when its logit is positive; this is how HGB, SeHGNN and HetSheaf treat it. Link prediction scores a pair with a decoder on the two node representations, here x-star and h concatenated, and uses binary cross-entropy with one sampled negative per positive; the decoder is the one of the HGB protocol, the same for every method. Graph classification pools the same node readout over the graph's nodes, with mean, max, add or attention pooling as HetSheaf's baselines use, and a cross-entropy loss. We deliberately add no new readout component, which reviewers had criticised in v1.

## Slide 43: Datasets (1): node classification, small to large

Node classification uses eight datasets. Four come from HGB: ACM, DBLP, IMDB and Freebase, trained on the full graph with HGB's official 24/6/70 splits. Four come from H2GB, the large benchmark for heterophilic heterogeneous graphs: IEEE-CIS-G for fraud, mag-year and ogbn-mag on the same academic graph with different targets, and the social network H-Pokec, with their own splits. The three large ones need mini-batch training, which still has to be designed. The rule for every dataset: it must have node attributes or target labels. This set answers reviewer concern 1: it spans the heterophily range, not just IMDB.

## Slide 44: Datasets (2): link prediction and graph classification

Link prediction has eight tasks under one protocol. Four are derived from the node-classification graphs, so node types and node labels are real: paper-author on ACM, author-paper on DBLP, movie-actor on IMDB, and book-book on Freebase. Two are HGB's own link sets, PubMed and LastFM; LastFM has neither features nor labels and is kept as the one stated exception. Two large tasks, citation on ogbn-mag and friendship on H-Pokec, test scale. In every task, features that are computed from the target relation are removed, because they would leak the answer. Graph classification uses the four TU datasets of HetSheaf. Their node types come from atom or structure labels, which reviewers rightly said is not a natural heterogeneous graph, so this task only shows that the model covers it, with no heterophily claim.

## Slide 45: Heterophily metric (1): MLH, do linked nodes share a label?

Two heterophily metrics are new in the evaluation. The first is MLH, meta-path-based label homophily, from Hetero2Net. Take every meta-path of length two from the target type back to itself, such as PAP. Each defines a graph on target nodes; on it, H-edge is the fraction of edges whose two ends share a label. MLH averages that over the meta-paths. On our example the PAP graph has two edges, P1-P9 with different labels and P1-P4 with the same label, so H-edge is one half. Near one means homophilic, near zero heterophilic. It is the plain label view; the H-squared index from the background slide is the same idea corrected for class imbalance, and that is what we use to stratify results. All three metrics use the same meta-path set, Hetero2Net's list where it exists and its own rule otherwise. Before reporting, we must reproduce Hetero2Net's published values on DBLP, ACM and IMDB.

## Slide 46: Heterophily metric (2): MDE, do linked nodes look alike?

The second metric, MDE, is the meta-path Dirichlet energy, also from Hetero2Net. It looks at features instead of labels: for every edge of a meta-path graph it measures how different the two degree-normalised feature vectors are, then averages. Large MDE means linked nodes look different, which is attribute heterophily. It depends on the scale of the features, so we use it to describe each dataset, not to rank datasets against each other. The table shows where each metric can be computed: all three on node classification except MDE on Freebase, which has no features; on link prediction where the endpoints have labels or kept features; and on graph classification only a one-hop edge homophily with atom type as label. Before reporting anything we must reproduce Hetero2Net's published values within a tolerance fixed in advance.

## Slide 47: Evaluation metrics

These are the metrics, each with its purpose. Macro-F1 averages the F1 score over classes, so rare classes count as much as frequent ones; we average only over classes that occur in the split's true labels, which matters for ogbn-mag with 349 venues. Micro-F1 pools all decisions; for single-label data it equals accuracy. For link prediction we follow HGB: ROC-AUC, the probability that a true link scores higher than a negative drawn from the head's two-hop neighbourhood, and MRR, the mean reciprocal rank of the true link. Which validation metric selects configurations and checkpoints is fixed before running: Macro-F1 for node and graph tasks, ROC-AUC for links.

## Slide 48: Protocol: one rule for every method

The protocol is one rule for every method. All methods read the same files, splits and negatives. Each is tuned with Bayesian search, one hundred trials, on seed zero, using only the validation metric; the best configuration is then retrained on seeds zero to four, and we report mean and standard deviation. For significance we use Welch's t-test, which does not assume equal variances; we compare H2S-NN with the best other method on each metric and report it honestly if a baseline is better, with raw and adjusted p-values. Our own search space is small: dimension, learning rate, weight decay and two dropouts; the walk length, number of walks and solver tolerance are fixed and never tuned. The table maps this to reviewer concern 4. One exception by design: each method keeps the feature preprocessing of its original code.

## Slide 49: Hyperparameters: what is tuned, fixed and learned, per step

Reviewers asked how many choices the model hides. This table lists every knob, step by step. Red entries are the only ones we search: the dimension d, two dropouts, learning rate and weight decay, five in total, plus the pooling type for graph classification. Everything in the third column is fixed once and never tuned: two reflections per map, the maximum walk length 4 on HGB, the number of walks per node, which is calibrated by a fixed rule, the solver tolerance and step limit. The last column is learned by gradient descent; tau, the strength of agreement, is learned rather than tuned. v1 searched nine hyperparameters, including the number of diffusion steps and two step sizes, and needed four hand-set budgets for schemas, instances, incidences and length; now it is five tuned, plus two fixed path settings, the maximum length and the number of walks, which are set by a rule before any run.

## Slide 50: Cost (1): how time and memory grow

How cost grows, step by step. Drawing walks costs nodes times walks per node times the maximum length, on the CPU. Edge maps cost linear in edges times d, because each map is two reflections, never a d-by-d matrix. The instance weights cost linear in the number of faces. One application of the operator is linear in edges plus face incidences; the solve needs T of them, about ten in practice, and so does the backward solve. Memory holds the states of nodes, shared edges and faces, and does not grow with T because we differentiate implicitly. For comparison, v1 built d-by-d maps at cubic cost in d and stored states for every diffusion step. Meta-path methods such as HAN and SeHGNN pay for the edges of each per-schema graph, which can be far denser than the original graph; MAGNN pays for every instance of every schema. These are the standard orders, not measurements.

## Slide 51: Cost (2): measured memory and solver steps

Measured cost. Left: peak GPU memory on ACM. The baseline numbers and v1 come from Table 11 of the submitted paper, measured on that machine; our current number comes from a laptop GPU at dimension 32, so read the comparison as an order of magnitude. v1 needed 9.3 GB; the current model needs 2.8 GB, between H2G-former and HINormer and below HAN. Right: the current model on three datasets. The number of conjugate-gradient steps stays at 9 to 11 while the state grows four times from IMDB to DBLP, which is what the conditioning bound predicts. Time per epoch was 4.3 seconds on ACM, but the laptop was running slow in that session, about two times slower than usual on the model-independent sampling, so a fair time comparison against v1 and the baselines will be measured on the server under the same conditions. Reviewer concern 3 also asks for scaling curves; those are in the analysis plan.

## Slide 52: Ablation study: which part carries the gain?

The ablation study answers one question: which part carries the gain. Each row removes or replaces exactly one component, with everything else held equal: same sampled paths, same parameter budget, same tuning. The first three rows decide RQ2. Removing all faces tests whether path instances add anything. Replacing each face by a closing edge between its ends tests whether a face is more than its endpoints, which is reviewer concern 2. Using the face only as a feature of the classifier is the MAGNN-type instance encoder; if the full model wins, structure beats features. The weight ladder, uniform, schema only, endpoints only, full, tests whether it matters who is on the path. The last three rows test design choices: the direct node-face term, composing face maps from edge maps instead of learning them freely, and orthogonal maps that respect relation order. For every row we wrote down in advance what the paper says if the full model does not win.

## Slide 53: Model analysis: when and why it works

Beyond ablations, six analyses explain when and why the model works. First, gain against heterophily: if handling heterophily with paths is real, the gain should grow with H-squared across the eight datasets; this is RQ1 and reviewer concern 1. Second, known-answer tasks: small synthetic graphs where the label is decided by the intermediate node, so we can check that the mechanism does what we claim, separate from benchmark noise. Third, the schema read-out: which schemas receive weight, and whether that is stable across seeds; this replaces v1's schema selection as the interpretable output. Fourth, sensitivity to the maximum path length. Fifth, scaling of memory, time and solver steps with graph size and number of faces, reviewer concern 3. Sixth, a case study of which instances receive high weight, for example by the type of the intermediate node and whether the endpoints share a label.

## Slide 54: What is new, and what is not

To be precise about novelty. New: path instances become faces of the same sheaf as the graph, with maps composed from the edge maps; a face reaches every node of its path, including A7, without a shortcut edge; heterophily is handled on the original graph without per-schema branches; and because there is one energy, every control is a configuration. Not new: using intermediate nodes, which MAGNN and SGAT already do; sheaf networks on typed graphs, which HetSheaf introduced; and the solver, which is personalised PageRank. There is also a special case we state openly: with relation-only maps and the face used only as a feature, our face equals MAGNN's instance encoder. So the contribution is where the path acts, not the fact that paths are used.

## Slide 55: Then and now (1/2): story and model

A side-by-side comparison with the submitted version. The story changed from 'keep path instances as 2-cells', with three contributions, to two goals: handle heterophily on the original graph, and use paths as structure. The path cell was closed by a synthetic edge; now a face sits above its path with no closing edge, so the face reaches every node of its path directly, where before it reached them only through that edge. K steps of Hodge diffusion became one energy and one solve. AMPSS, which enumerated schemas and needed three budgets, became WISP, which needs no schema list and weights every instance. SheafCovPool was dropped; graph classification uses plain pooling. Measured memory on ACM is about 0.3 times v1 on a laptop GPU at dimension 32; server numbers are still to come.

## Slide 56: Then and now (2/2): evaluation

The evaluation changed more than the model. Node classification goes from three HGB datasets, of which only IMDB was strongly heterophilic, to eight, including four large H2GB graphs. Link prediction drops the single-node-type graphs Amazon and YouTube and uses eight tasks with real node types and leak-free features. Graph classification follows HetSheaf's protocol and is stated as typed by label. Heterophily evidence goes from MLH alone to three metrics plus an analysis of gains against H-squared. Every baseline, HetSheaf included, is rerun with the same tuning budget, and results come with Welch tests. To be clear: none of this has been run yet.

## Slide 57: Limitations and next steps

Limitations we will state. Influence is symmetric in strength: an edge or a face pulls all its members equally, and direction lives only in the maps; that is the price of having an energy with a unique solution. Propagation is linear and all non-linearity is in the head. Edge states carry no input of their own because the benchmarks have no edge features. Paths are simple, up to a maximum length, and sampled. Graph classification is typed by label. And five seeds on fixed splits measure seed variance only. Next steps, in order: the heterophily harness, starting by reproducing Hetero2Net's published values; the protocol code; the mini-batch design for the large graphs; the baseline plan including a HetSheaf adapter; and then the server runs and the analysis part.

## Slide 58: Conclusion

To conclude. The problem: most methods handle heterophily on derived, per-schema graphs, and reduce a path to its two endpoints. Our idea: keep the original graph and make each path instance a face whose maps are composed from the edge maps, reaching every node it passes, with no closing edge. The method is one energy over nodes, edges and faces, solved once, with WISP sampling paths without a schema list and weighting each. The two research questions will be tested on node, link and graph tasks under one protocol, with the controls and their interpretation fixed in advance. The model is implemented and checked for correctness; there are no accuracy numbers yet. Thank you, I am happy to take questions.

## Slide 59: References

References for the citations on the slides.

## Slide 60: References (cont.)

References for the citations on the slides.

## Slide 61: Appendix: terms used in this talk

Glossary for questions.
