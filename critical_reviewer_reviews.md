WSDM2027 
Peeradon Sarnkaew 
View Reviews
Paper ID
850
Paper Title
H2S-NN: Heterogeneous Heterophilic Sheaf Neural Network
Track Name
Full Paper (Main Track)
Reviewer #1
Questions
1. Summary Please briefly summarize the contributions of this paper and the justification of your overall recommendation.
The paper proposes H2S-NN, a Graph Neural Network (GNN) for heterogeneous graphs that can model heterophilic relationships. Its main idea is to preserve concrete meta-path instances, such as Author-Paper-Author, as higher-order structures instead of reducing them to a direct relationship between the endpoints. It also includes AMPSS (automatic meta-path schema selection and instance weighting) and SheafCovPool (a covariance-based graph pooling method) for selecting useful structures and graph classification.
The experiments cover node classification, graph classification, and link prediction across 11 datasets. H2S-NN reports the best mean result on all datasets except MUTAG in Tables 1-3. The construction is clearly defined and the evaluation covers multiple tasks and datasets. However, the evaluation does not sufficiently demonstrate scalability to larger graphs. The experiments are conducted on relatively small to moderate datasets, and the paper does not show how runtime and memory grow with graph size or with the number of retained meta-path instances. Although AMPSS uses explicit instance and incidence budgets to control this cost, the scalability of the approach remains unclear. This is my main concern. Overall, I recommend Weak Accept.
2. Strengths List at least three strengths of the paper. Consider, where applicable, the originality and significance of the contributions, technical soundness, quality of the empirical or theoretical evaluation, clarity of presentation, reproducibility, and potential impact.
S1. The path-supported 2-cell construction is a novel modeling idea that retains the intermediate nodes and relation sequence of each selected meta-path instance instead of reducing it to a direct endpoint relation.

S2. The mathematical construction is well defined, and the paper provides a proof of the required consistency condition.

S3. The evaluation covers three tasks and 11 datasets, providing evidence across node classification, graph classification, and link prediction.

S4. AMPSS provides automatic meta-path selection and explicit instance and incidence budgets to control the size of the constructed operators.

S5. The paper provides detailed ablations of the proposed components, including AMPSS and the SheafCovPool readout.
3. Weakness. List at least three weaknesses or limitations of the paper. Consider, where applicable, issues related to novelty, significance, technical soundness, experimental or theoretical evaluation, comparison with prior work, clarity, reproducibility, and the validity of the conclusions. Please identify the most consequential concerns.
W1. Scalability is not sufficiently demonstrated. H2S-NN retains concrete meta-path instances as higher-order structures, which can substantially increase the number of cells and incidences that must be processed. AMPSS uses instance and incidence budgets, but the paper does not show how runtime and memory scale with graph size or the number of retained path instances. On ACM, H2S-NN takes 3.011 s/epoch and 9,260 MB, compared with 0.109 s/epoch for GCN. Larger-scale experiments or a scaling analysis would strengthen the practical case for the method.

• W2. Baseline evaluation details are limited. The paper describes the H2S-NN tuning procedure but does not clearly specify the search space and budget for each baseline or whether all baselines were rerun under the same protocol.

• W3. The comparison with HetSheaf is not under the same protocol. Its results are taken from the original paper and reported with zero standard deviation. A same-protocol comparison would make this result easier to assess.
4. Detailed comments to the authors.
Q1. Please report how runtime and memory change with graph size and with the number of retained path instances or incidences.

Q2. Please clarify the search space, tuning budget, splits, seeds, and early stopping used for each baseline.

Q3. Please clarify how the reported standard deviations are computed, particularly the relationship between the 10-fold cross-validation and five random seeds.
8. Overall Recommendation
Weak Accept: Borderline paper, tending to accept
Reviewer #2
Questions
1. Summary Please briefly summarize the contributions of this paper and the justification of your overall recommendation.
This paper proposes a H2S-NN that represents meta-path instances as path-supported 2-cells closed by closure edges and guarantees a cochain complex through transport-based edge-to-face maps, together with a schema selection module and a gauge-invariant graph readout (SheafCovPool). However, it has the weakness that the motivations of heterophily and path structure preservation are misaligned with the actual mechanism and evaluation.
2. Strengths List at least three strengths of the paper. Consider, where applicable, the originality and significance of the contributions, technical soundness, quality of the empirical or theoretical evaluation, clarity of presentation, reproducibility, and potential impact.
1. The design that constructs the edge-to-face maps through the transport of the node-to-edge restrictions has a principled background, and the proof in App. A is also concise.

2. The attempt to lift meta-path instances to 2-cells, instead of reducing them to endpoint relations, and to combine them with sheaf-based edge diffusion is an appropriate direction in heterogeneous sheaf research.

3. The paper has the advantage of addressing node classification, link prediction, and graph classification, comparing against baselines from four categories, and releasing the code anonymously.

4. The paper has the advantage of reporting the computational cost by presenting both a theoretical time and space complexity analysis and a measured efficiency comparison in the appendix.
3. Weakness. List at least three weaknesses or limitations of the paper. Consider, where applicable, issues related to novelty, significance, technical soundness, experimental or theoretical evaluation, comparison with prior work, clarity, reproducibility, and the validity of the conclusions. Please identify the most consequential concerns.
1. The reliability of the experimental results is insufficient. Many of the graph classification values match the values of a single test split rather than the described average over 10-fold × 5 seeds, and some baselines show low performance, so it is difficult to trust the reported performance advantage.

2. The baseline tuning procedure is not described, and values quoted from the original paper (HetSheaf), whose protocol may differ, are mixed with values obtained from the authors' own experiments.

3. The paper claims that it retains the internal path structure in propagation, but the face-induced term acts only on the edge cochain and is reflected in the node representations only through the closure edge, which is an endpoint-level connection. This is not essentially different from the approach that the authors criticize in the introduction.

4. Most of the evaluation datasets are homophilic or effectively homogeneous, and heterophilic heterogeneous benchmarks and an analysis according to the degree of heterophily are lacking.

5. The ablations of AMPSS and SheafCovPool show only gains within the standard deviation on a single dataset, and the selection results and implementation details of AMPSS are not presented, so reproducibility and interpretability are insufficient.
4. Detailed comments to the authors.
1. Since the claim that forms the core motivation of the paper, "whether information should be reinforced or contrasted may depend on the relations and intermediate nodes along the path" (lines 114–116, 134–136), is presented without examples or empirical analysis, the authors should present heterophilic cases in which reduction to an endpoint view actually fails, or a preliminary analysis that shows this.

2. Since the introduction presents only the path reduction problem as the motivation, it does not explain why automatic schema selection (AMPSS) and a gauge-invariant graph readout (SheafCovPool) are needed, or why graph classification is important in the heterogeneous heterophilic setting. Therefore, the introduction should explicitly connect each contribution to the problem it addresses.

3. Figure 3 has the weakness of low readability and low consistency, as it contains symbols that differ from those in the main text.

4. The caption of Figure 4 states that the incident edge states are first moved into their endpoint node stalks, but this does not appear to be clearly shown in the figure, and the notation is also inconsistent with the main text.

5. In relation to Weakness 3, the unique contribution of the 2-cells should be isolated through an ablation that compares a variant that retains only the closure edges with a variant that includes the face term.
8. Overall Recommendation
Weak Reject: Borderline paper, tending to reject
Reviewer #3
Questions
1. Summary Please briefly summarize the contributions of this paper and the justification of your overall recommendation.
This paper studies learning on graphs that are both heterogeneous and heterophilic. The main idea is to avoid collapsing a meta-path instance into an endpoint-level relation. Instead, H2S-NN keeps each selected concrete path instance as a path-supported 2-cell and performs sheaf-Hodge diffusion over node and edge cochains. The method contains three main technical components: AMPSS for task-dependent meta-path schema selection and instance weighting, a sheaf-Hodge construction that introduces compatible node-to-edge and edge-to-face maps, and SheafCovPool for graph-level prediction. I find the basic idea of preserving concrete meta-path instances as higher-order cells interesting, and the construction is more than a straightforward combination of an existing heterogeneous GNN and a sheaf layer. My reservations concern evaluation and scalability. Only one of the node-classification datasets has clearly high reported heterophily, the graph-classification benchmarks are converted into heterogeneous graphs by treating categorical labels as types. Overall, I view this as a technically interesting paper with encouraging results, but some claims should be better supported. I lean toward Weak Accept.
2. Strengths List at least three strengths of the paper. Consider, where applicable, the originality and significance of the contributions, technical soundness, quality of the empirical or theoretical evaluation, clarity of presentation, reproducibility, and potential impact.
1、The main modeling idea is original and well motivated.
Treating a concrete meta-path instance as a path-supported 2-cell preserves the intermediate nodes and relations rather than reducing the path to an endpoint relation. This is a meaningful distinction for heterogeneous graphs, especially when the usefulness of an interaction depends on the relation sequence along the path.
2、AMPSS makes meta-path construction task dependent rather than fully hand-designed.
Stage I selects meta-path schemas using a held-out selection partition, while Stage II further learns different weights for concrete instances of a selected schema. This addresses the fact that two instances of the same schema need not be equally useful.
3、The evaluation covers three substantially different tasks.
The paper evaluates node classification, graph classification, and link prediction, rather than demonstrating the method on only one benchmark family.
3. Weakness. List at least three weaknesses or limitations of the paper. Consider, where applicable, issues related to novelty, significance, technical soundness, experimental or theoretical evaluation, comparison with prior work, clarity, reproducibility, and the validity of the conclusions. Please identify the most consequential concerns.
1、The evidence for the “heterophilic” part of the paper is still limited. This is my main experimental concern. The reported MLH values for DBLP, ACM, and IMDB are 0.1277, 0.2639, and 0.7634, respectively. Thus, among the three node-classification datasets, only IMDB is strongly heterophilic according to the paper’s own metric. The paper would be more convincing with additional genuinely heterophilic heterogeneous datasets, especially since recent large-scale benchmarks in this setting are already cited by the paper.
2、The graph-classification setting is somewhat artificial with respect to graph heterogeneity. PROTEINS, ENZYMES, MUTAG, and NCI1 are standard graph-classification datasets. The paper makes them heterogeneous by interpreting their node and edge labels as node and relation types. This is a valid experimental construction, but it is not equivalent to a naturally heterogeneous information network with semantically distinct entity and relation types. In addition, no heterophily statistic is reported for these graph-classification benchmarks.
3、Scalability is insufficiently studied. AMPSS enumerates meta-path schemas using DFS, retains concrete path instances, and creates one synthetic closure edge per retained instance. The method introduces schema, instance, and incidence budgets precisely because this space can grow quickly. However, the paper gives no runtime, peak-memory, preprocessing-time, or number-of-retained-instance analysis. This is important because the main contribution explicitly operates at the path-instance level rather than the endpoint level.
4. Detailed comments to the authors.
1、Please add more genuinely heterophilic heterogeneous benchmarks. IMDB is the only node-classification dataset with clearly high MLH in the current experiments. Since the paper cites newer large-scale heterogeneous-heterophilic benchmarks, evaluating at least one of them would considerably strengthen the central claim.
2、Please report the computational cost of path-instance construction. I would like to see, for each dataset, the number of candidate schemas, selected schemas, retained path instances, closure edges, and edge-face incidences, together with AMPSS preprocessing time, training time, and peak GPU memory.
3、Please provide a sensitivity analysis for K_S, K_inst，K_inc, and the maximum meta-path length. These are not secondary implementation details. They determine both the amount of higher-order information retained and the size of the augmented cellular complex.
8. Overall Recommendation
Weak Accept: Borderline paper, tending to accept
Reviewer #4
Questions
1. Summary Please briefly summarize the contributions of this paper and the justification of your overall recommendation.
This paper proposes H2S-NN, a Heterogeneous Heterophilic Sheaf Neural Network for node classification, graph classification, and link prediction. The main idea is to preserve concrete meta-path instances rather than collapsing them into endpoint-level meta-path views. The method represents selected path instances as path-supported 2-cells, augments each open path with a synthetic closure edge, learns relation-conditioned sheaf restriction maps, and applies Sheaf-Hodge diffusion over node and edge cochains. It also introduces AMPSS, a two-stage procedure for selecting meta-path schemas and weighting concrete path instances.

The problem is relevant, and the use of higher-order path structure in heterogeneous graphs is potentially interesting. The paper also provides results over several datasets and tasks. However, the claimed setting—heterogeneous heterophilic graph learning—is not consistently established across the evaluations, and the practical scalability of constructing a 2-cell for each selected concrete path instance is not demonstrated. The experimental comparison is also weakened by the use of results quoted from prior work for an unavailable-code baseline and by limited evidence that path-instance preservation, rather than model capacity or preprocessing choices, causes the observed gains. I therefore lean toward rejection.
2. Strengths List at least three strengths of the paper. Consider, where applicable, the originality and significance of the contributions, technical soundness, quality of the empirical or theoretical evaluation, clarity of presentation, reproducibility, and potential impact.
The paper addresses an important limitation of many meta-path-based heterogeneous GNNs: reducing concrete path instances to endpoint-level views can discard intermediate relational structure.
The proposed representation is conceptually interesting. Modeling retained meta-path instances as path-supported 2-cells provides a principled way to incorporate higher-order path information into a sheaf/Hodge-style propagation framework.
The paper considers multiple tasks—node classification, graph classification, and link prediction—rather than evaluating only one benchmark family.
The AMPSS ablation is useful. On DBLP, using path-supported 2-cells and task-aligned meta-path selection improves over no 2-cells, predefined schemas, and partial AMPSS variants.
The paper includes a graph-level readout design, SheafCovPool, that attempts to use both node and edge states. The graph-classification ablation provides some evidence that this readout is better than simple pooling variants.
The manuscript provides substantial technical detail about the construction of restriction maps, cochain spaces, diffusion operators, and the relation between selected paths and 2-cells.
3. Weakness. List at least three weaknesses or limitations of the paper. Consider, where applicable, issues related to novelty, significance, technical soundness, experimental or theoretical evaluation, comparison with prior work, clarity, reproducibility, and the validity of the conclusions. Please identify the most consequential concerns.
The claimed “heterogeneous heterophilic” setting is not consistently demonstrated. For node classification, the paper reports meta-path-based label heterophily values, but DBLP and ACM have relatively low reported heterophily, while only IMDB is strongly heterophilic. For graph classification and link prediction, the paper does not provide comparable heterophily measurements or show that these datasets genuinely instantiate the same heterogeneous-heterophilic setting.
Several evaluation datasets appear only weakly aligned with the central problem formulation. For graph classification, molecular TU datasets are converted into heterogeneous graphs by treating provided node and edge labels as types. This is a reasonable engineering adaptation, but it does not establish that these are naturally heterogeneous heterophilic graphs. Similarly, some link-prediction datasets are single-node-type, multi-relation graphs. The broad claim of effectiveness for heterogeneous heterophilic graph learning should therefore be narrowed or better justified.
Scalability is a major unresolved concern. H2S-NN enumerates meta-path candidates and retains concrete path instances, then creates a path-supported 2-cell for each retained instance. The number of such instances can grow rapidly with graph size, path length, and relation branching. The paper gives no clear end-to-end complexity analysis, runtime table, memory profile, number of selected schemas/instances, or scalability experiment on larger heterogeneous graphs.
The empirical comparison is not fully reliable because HetSheaf results are quoted from the original paper due to unavailable code. Since preprocessing, train/validation/test splits, hyperparameter tuning, feature construction, and evaluation protocols may differ, these numbers should not be treated as directly comparable. This is particularly important because HetSheaf is one of the closest methodological baselines.
The contribution of the core 2-cell construction is not isolated sufficiently. The full model combines type-specific encoders, relation-aware contextualization, learned restriction maps, AMPSS schema selection, instance weighting, Sheaf-Hodge diffusion, and a specialized graph readout. The current ablations do not clearly separate the benefit of preserving concrete path instances from the benefit of additional capacity, selection, weighting, or more expressive restriction maps.
The paper provides no direct analysis showing that the selected path instances are especially useful under heterophily. It would be valuable to show whether AMPSS selects different schemas or weights different path patterns as heterophily increases, and whether the gains are larger on high-heterophily settings than on homophilic controls.
The mathematical exposition is dense, but several practical choices remain under-specified: the candidate schema space, maximum path length, enumeration constraints, selected-instance counts, cost budget values, selection split construction, and hyperparameter search budget.
The paper contains substantial template/formatting issues, including placeholder conference information, a generic ACM reference format, and a statement claiming 14 pages while the submitted PDF has 10 pages. These should be corrected.
4. Detailed comments to the authors.
Please define and verify the target setting consistently for every dataset. Report explicit heterophily statistics for all node-, graph-, and link-prediction datasets, preferably under the exact typed relations or meta-path views used by H2S-NN.
Clarify why molecular graph-classification datasets, where node and edge labels are reinterpreted as types, constitute an appropriate test for a heterogeneous heterophilic graph method. Consider adding naturally heterogeneous graph-classification benchmarks or limiting the claims for this task.
Add an end-to-end scalability study. For each dataset, report the number of candidate schemas, retained schemas, concrete path instances, generated closure edges, wall-clock training time, peak memory, and parameter count. Also provide scaling curves over graph size, maximum path length, and number of retained instances.

Include a stronger controlled ablation for the core claim:

endpoint-collapsed meta-path views with the same backbone and parameter budget;
path instances without 2-cell diffusion;
2-cells with uniform instance weights;
AMPSS with the same number of paths as the full model;
full H2S-NN.

This would make it possible to attribute gains specifically to path-supported 2-cells.

The comparison with HetSheaf should be treated cautiously. If its code is unavailable, please avoid drawing strong direct conclusions from quoted results. Either reproduce it under the same protocol, compare only on its reported native setup, or clearly label the comparison as non-controlled.
Please provide direct evidence for the heterophily motivation. For example, stratify results by meta-path heterophily level, compare to homophilic variants of the same graphs, and visualize or quantify which schemas/path instances AMPSS selects in low- versus high-heterophily conditions.
Clarify AMPSS protocol in more detail. Which labels are used for selection? Is the “held-out selection loss” based only on a validation split that is disjoint from final test evaluation? Is schema search repeated for every split? What are the values of \(K_S\), \(K_{\text{inst}}\), and \(K_{\text{inc}}\)?
Report standard deviations, statistical significance tests, and hyperparameter tuning details for all main comparisons. Five random seeds are useful, but the paper should show whether improvements over the closest executable baselines are statistically reliable.
Correct the formatting metadata and page-count inconsistency before publication.
8. Overall Recommendation
Weak Reject: Borderline paper, tending to reject

Meta-review
This paper proposes H2S-NN, a sheaf neural network for heterogeneous heterophilic graphs. It keeps concrete meta-path instances as path-supported 2-cells rather than collapsing them into endpoint-level relations, and adds AMPSS for meta-path selection and SheafCovPool for graph-level readout. The reviewers find the 2-cell idea novel and the construction principled, and appreciate the evaluation on node classification, graph classification and link prediction.

However, the reviewers are split, both high-confidence reviewers recommend weak reject, and even the positive reviewers note that key claims need better support. First, the heterophilic setting is not established. Only IMDB is strongly heterophilic among the node-classification datasets, the graph-classification benchmarks are standard TU datasets whose labels are reinterpreted as types, and no analysis relates the gains to the degree of heterophily. Second, the contribution of the 2-cells is not isolated. The face term reaches node representations only through the closure edges, and the ablations do not separate it from schema selection, instance weighting and added capacity. Third, the paper does not analyze how the cost grows with graph size and the number of retained path instances. Reviewers also question the evaluation protocol, e.g., baseline tuning and HetSheaf results quoted from the original paper.

I therefore recommend rejection and encourage the authors to address the reviewers’ detailed comments in a future submission.