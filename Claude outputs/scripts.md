# Speaker script — H2S-NN progress talk (IJCAI-style), 2026-11-11

Deck: `progress_2026-11-11.pptx` (27 slides, about 20 minutes plus questions). The same text is in each slide's speaker notes.
Terms used throughout: heterogeneous graph, heterophily, meta-path (schema), path (instance), restriction map, face, energy, WISP. No accuracy numbers exist yet.

## Slide 1 — Title

Good afternoon. I will present H2S-NN, our model for graphs that have many node and edge types and in which connected nodes often have different labels. I will present it as I would at IJCAI: the problem, the idea, the method, how we will test it, and what is still open. The model is final and implemented; the large experiments have not run yet, so you will see no accuracy numbers today.

## Slide 2 — Outline

The talk has five parts. Background gives the four ideas you need: typed graphs, heterophily, meta-paths and sheaves. Problem shows what existing methods do with one concrete path and what they lose. Method presents our answer. Experiments states how each claim will be tested. Discussion separates what is new from what is not.

## Slide 3 — A heterogeneous graph: nodes and edges have types

This is the running example for the whole talk. Papers, an author and a subject: nodes have types, and edges have types too, such as "writes" or "has subject". Formally, a heterogeneous graph has a type function for nodes and one for edges. Our task is to predict the field of each paper, shown by colour. Standard benchmarks of this kind are ACM, DBLP, IMDB and Freebase from the HGB benchmark. The point of this slide: a neighbour's type matters, so a model should not treat every neighbour the same way.

## Slide 4 — Heterophily: connected nodes often have different labels

A GCN updates a node with the average of its neighbours. That works when neighbours share the label, which we call homophily, left. It fails under heterophily, right: neighbours have different labels and averaging mixes them. The simplest measure is edge homophily: the fraction of edges whose two ends have the same label. Near one is homophilic, near zero heterophilic. In our example P1 and P9 share author A7 but are in different fields. So being connected does not mean being similar, and that is the first difficulty.

## Slide 5 — Meta-paths: a type pattern versus one concrete path

A meta-path schema is a type pattern: paper, author, paper means two papers share an author. An instance is one concrete path that follows it, such as P1, A7, P9. The middle node, A7, is the intermediate node. Our research question is about this node: does it matter who A7 is, and should A7 itself learn that P1 and P9 are linked? Keep this question in mind; every later slide answers part of it.

## Slide 6 — Measuring heterophily on a heterogeneous graph

On a typed graph, labels are compared between papers, so heterophily is measured on the paper graphs that meta-paths induce. The H2GB benchmark defines the index H-squared: for each length-two meta-path, build the induced paper graph, measure its adjusted homophily, which corrects for class imbalance, and average one minus that value. Zero means homophilic; around one, heterophilic. The chart shows our values: DBLP is homophilic, IMDB clearly heterophilic, IEEE-CIS-G and Freebase almost fully heterophilic. Freebase is dominated by unlabeled books, which the reference code fills at random; we also report a labeled-only value. With this range we can test whether gains grow with heterophily.

## Slide 7 — Sheaf networks: compare after a learned map

A GCN implicitly penalises the difference between neighbours, which pulls them to the same vector. That is wrong when their labels differ. A sheaf network gives each end of an edge a learned map, called a restriction map, into a space that belongs to the edge. Neighbours are compared only after their maps: they may stay different, yet agree inside the edge. Summing these penalties gives the sheaf energy; with all maps equal to the identity it is the ordinary graph Laplacian. This is why sheaf networks cope well with heterophily. HetSheaf applies the idea to typed graphs, but only to edges.

## Slide 8 — What happens to A7 in existing methods?

Here is what four families of methods do with the path P1, A7, P9. First, most heterogeneous and heterophilic models split the graph per meta-path into paper-only graphs, as in HAN or SeHGNN; A7 disappears. Second, instance encoders such as MAGNN encode the whole path, including A7, but write the result only into the target node P1; A7 and P9 learn nothing from it. Third, sheaf networks such as NSD and HetSheaf handle heterophily, but they only see edges, never a path. Fourth, one can attach the path as a two-dimensional cell, a face, as cell-complex networks do. But under the standard Hodge diffusion on such a complex, we can prove the face sends exactly zero information to the nodes, whatever the number of steps; the only route left is an extra end-to-end edge, which brings back the endpoint view. So no existing design lets one path update all of its nodes inside propagation.

## Slide 9 — Four requirements, and who meets them

We turn the problem into four requirements. Work on the original graph. Use who the intermediate node is, not only its type. Update every node of the path, including A7. And make the path's maps follow from the edge maps, so a path adds evidence rather than new free parameters. The table checks each family at the level of its update equation. The hypergraph sheaf model comes closest: it treats a path as an unordered set of nodes and updates all of them, but its maps are free and it has no notion of the edges along the path. Our research question: can each path act inside the graph's own propagation, reaching every node it passes through, with maps the graph already has?

## Slide 10 — H2S-NN in one picture

Here is the whole model. One: a type-specific encoder turns every node into a vector h. Two: we sample paths by random walks on the original graph; each walk becomes a face. Three: we write one energy that asks nodes, edges and faces to agree, each after its maps; the picture shows the three levels for the face P1, A7, P9. Four: node representations are the minimiser of that energy; a small MLP reads the encoding and the minimiser and predicts the label. What is learned: the encoders, the edge maps and weights, the face weights, and one scalar tau. The face maps are not learned; they follow from the edges, as the next slides show.

## Slide 11 — A path becomes a face

We keep the path on the original graph and place a new cell, a face, above it. The structure has three levels: nodes, edges above the nodes they join, and faces above the nodes and edges of their path. Each level carries a state: x for nodes, s for edges, z for faces. The face is coupled to every node on it, including A7, and to the edges it uses. A face is never closed into a loop: there is no edge between P1 and P9, not in the computation and not in the definition. That end-to-end edge is exactly the shortcut the Hodge construction needed. A7 is now inside the computation: it receives the face's information and, through its own edge, passes it on to P4.

## Slide 12 — Sampling faces and weighting each one

Which paths become faces? A sampler we call WISP draws random walks from every target node; each walk is a face. No meta-path list is given, and the cost depends on the number of walks, not on the number of possible schemas. Each face gets one weight beta: m is a sampling correction that makes the result unbiased for a stated target distribution, and g is a small network that reads all the nodes and relations of the face. So two paths of the same schema, P1-A7-P9 and P1-A7-P4, can count very differently, depending on who is on them.

## Slide 13 — Face maps are not learned: they follow the edges

How does a face compare the vectors of P1, A7 and P9? It reuses the maps the edges already have. To bring A7 into the face, we go through edge e1: undo A7's map and apply P1's. For P9 we continue through edge e2. So a face map is a composition of edge maps along the path: there is nothing new to learn, and because each edge map is computed from both of its end nodes, the face map depends on who A7 is, not only on its type. The guarantee: any state that is consistent on the graph remains consistent after we add faces, so faces add evidence and never contradict the graph. With free per-face maps, as in copresheaf networks, we verified that this property is lost.

## Slide 14 — One energy over nodes, edges and faces

This is the single energy of the model. It has three kinds of terms, each a disagreement after the maps. Blue: a node and an edge it belongs to. Green: an edge and a face that uses it. Pink: a node and a face that passes through it. Edge weights a and face weights beta scale the terms. The picture on the left shows exactly which pairs are compared for our face: every pair of cells that touch, and nothing else. In particular there is no term between P1 and P9.

## Slide 15 — Every node of the path hears the face

Does the face really reach the nodes? We checked the operator on this example. In the standard Hodge construction the face needs a closing edge between P1 and P9, and the information from the face to any node is exactly zero, whatever the number of diffusion steps; the only link between the endpoints is that closing edge. In our operator the face is coupled to P1, A7 and P9 directly, in a single step, and there is no direct P1-P9 coupling at all. This is a structural property that holds by construction. Whether it improves accuracy is a separate question, claim C10, which the experiments will decide.

## Slide 16 — Propagation = the minimiser of the energy

Instead of stacking layers, we take node representations as the minimiser of one objective. The first term keeps each node close to its encoding h; the second is our energy, scaled by one learned scalar tau. Three consequences. There is exactly one answer for every value of the learned parameters, so training never meets an ill-defined propagation. There is no depth or filter order to tune. And the cost is linear in edges and face incidences; we solve with conjugate gradient and differentiate implicitly, so memory does not grow with the number of solver iterations. The classifier reads both the encoding and the result, which keeps a node's own information separate from its neighbours', which helps under heterophily. To be clear: without faces this is personalised PageRank, as in APPNP, on a sheaf. Our contribution is the operator, not the solver.

## Slide 17 — One model, many controls

Because everything is one energy, every comparison the reviewers asked for is a configuration of the same model: we switch off one kind of term. No faces gives plain sheaf diffusion. Replacing the face by one edge between its endpoints gives the endpoint view. Feeding a face summary only to the classifier gives a MAGNN-type instance encoder; we checked that with relation-only maps this equals MAGNN's encoder to machine precision. Dropping the node-face term shows whether the direct term is needed. Free face maps test whether tying the maps to the edges matters. All arms share the same sampled paths, parameters and budget, so a difference can only come from structure.

## Slide 18 — Datasets: small and large, homophilic to heterophilic

We use two benchmark families. HGB gives ACM, DBLP, IMDB and Freebase, trained on the full graph. H2GB gives four large and strongly heterophilic graphs, which need mini-batch training; each target gets a closed local piece of the same model, so the theory still applies. For link prediction we derive tasks from the same graphs plus PubMed and LastFM. Heterophily ranges from 0.17 to almost 1, so we can read gains against it. Every method, ours included, is rerun under one protocol; no number is copied from another paper, which was a concern of earlier reviewers.

## Slide 19 — Metrics and protocol

For node classification we report macro-F1: the F1 score per class, averaged, so rare classes count as much as frequent ones. IMDB is multi-label with five genres, and we follow the benchmark's treatment. For link prediction we use mean reciprocal rank against hard negatives as in HeaRT: for each true link, its rank among difficult wrong candidates; the reciprocal rewards ranking it first. The protocol is fixed in advance: same splits, the same tuning budget for every method, ten seeds, and paired Wilcoxon tests with Holm correction. A recent study shows that a tuned R-GCN matches many heterogeneous GNNs, so we do not rely on rankings: a component is kept only if it beats its own control under identical settings.

## Slide 20 — Each claim has a deciding experiment

Each claim is paired with the experiment that decides it and with what we will write if it fails, all fixed before running. Does it matter who is on the path: full model against endpoint-only and uniform weights. Does a face beat a shortcut edge: full against the endpoint-edge configuration. Does acting inside propagation beat using the path as a feature: full against the MAGNN-type configuration. Is the direct node-face term needed. And do gains grow with heterophily. If a comparison shows no gain, the paper narrows its claim as written in the last column.

## Slide 21 — Status: implemented and verified

Where are we? The model is implemented and verified: all seven algebra and gradient tests pass, for example the implicit gradient matches finite differences to six times ten to the minus eight. On a two-epoch smoke run on IMDB the solver needs about eight iterations per step, far below its limit, and one epoch takes 1.15 seconds, against 1.33 for the previous version, at 1.3 instead of 0.9 gigabytes. Reviewers worried about cost: the first version used 9.3 gigabytes on ACM, and reading the code shows the cause was a full matrix stored per edge, not the faces. Next are the ACM cost run and then all comparisons on the cluster. There are no accuracy numbers yet.

## Slide 22 — What is new, and what is not

Let me separate what is new from what is not. New: paths become faces of the same sheaf as the graph, with maps that follow the edges; a face reaches every node of the path, including A7, without a shortcut; and because there is one energy, every control is a configuration, not a new model. Not new: using intermediate nodes, which MAGNN and SGAT already do; sampling paths without a schema list; and the solver, which is personalised PageRank. There is also a special case we state openly: with relation-only maps and the face used only as a feature, our face equals MAGNN's instance encoder. So the contribution is where the path acts, inside one sheaf, not the fact that paths are used.

## Slide 23 — Limitations and next steps

Four limitations we state in the paper. Influence is symmetric in strength: an edge or face pulls both sides equally, and direction lives only in the maps. Propagation is linear; all non-linearity is in the classifier. Edge states carry no information of their own, since the benchmarks have no edge features. And the cost depends on how many edges are shared by several faces. The plan: this month the ACM cost run and the method section; in December all comparisons on the cluster with ten seeds; in January results, writing and submission to IJCAI-27. If a deciding comparison fails, we narrow the claim as already written down.

## Slide 24 — Summary

To summarise. First, meta-path instances become faces of the graph's own sheaf, with maps that follow the edges. Second, each face reaches every node on its path, including the intermediate node, without a shortcut, inside one energy and one solve. Third, every control the reviewers asked for is a configuration of the same model, and the experiments are designed so that each claim can fail. Thank you; I am happy to take questions.

## Slide 25 — References

References for the citations on the slides.

## Slide 26 — References (cont.)

References for the citations on the slides.

## Slide 27 — Appendix · Terms used in this talk

Glossary for questions.
