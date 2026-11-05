# H2S-NN Research Memo: Automatic Meta-path Discovery Beyond HetSheaf

Date: 2026-05-06

## 1. Executive summary

The strongest publishable direction for `H2S-NN` is **not** "another heterogeneous sheaf GNN." `HetSheaf` already occupies that space very well. Its current arXiv version is `2409.08036v2`, updated on **April 16, 2026**, and its core message is clear: learn heterogeneous restriction maps on a typed graph, then do node/link/graph prediction with a 1-complex sheaf model and a gauge-aware pooling layer.

Your real opening is different:

- `HetSheaf` learns **typed sheaf structure on the observed graph**.
- `H2S-NN` can learn **which higher-order typed path schemas should become 2-cells at all**.
- That means the main contribution should be **automatic higher-order topology construction**, not just better type-aware restriction maps.

So the paper should target this problem:

> How can we build a heterogeneous sheaf 2-complex automatically, without relying on manually predefined meta-paths, especially on datasets where useful meta-paths are unclear, numerous, or absent?

That is a real problem in the current literature, and it is meaningfully different from `HetSheaf`.

An important practical clarification is that this capability should live **inside `H2S-NN` itself**. The goal is not to create a separate search tool unrelated to the model, but an `H2S-NN` variant that can:

- use manual meta-paths when they are good,
- refine them when they are weak,
- and discover them automatically when they are missing.

## 2. Where H2S-NN is already different from HetSheaf

From `CLAUDE.md` and the H2S-NN specs, your current model already has a different mathematical center:

- `HetSheaf` works on a heterogeneous **graph / 1-complex**.
- `H2S-NN` lifts a heterogeneous graph into a typed **2-dimensional cellular complex**.
- `H2S-NN` treats meta-path instances as **faces / 2-cells**, not just as extra semantic views.
- `H2S-NN` uses `delta_1`, `L_1 = delta_0 delta_0^T + delta_1^T delta_1`, and Hodge-style diffusion on edges/faces.
- `H2S-NN` therefore has access to **coexact / harmonic edge information** that graph-only sheaf methods do not model directly.

This means your novelty should be framed as:

> Moving from "heterogeneity-aware sheaf diffusion on graphs" to "automatic discovery of higher-order heterogeneous topology for sheaf diffusion."

That framing keeps the paper distinct.

### What you can safely borrow from HetSheaf

- Type-conditioned parameterization of low-level restriction maps.
- Efficient heterogeneous encoders for mixed node/edge types.
- A clean argument that heterogeneity should be represented in the data geometry, not only in the architecture.

### What should *not* be the headline contribution

- "We also use typed restriction maps."
- "We add a better sheaf predictor."
- "We add a pooling layer for graph classification."

Those are too close to the `HetSheaf` story. They are fine as supporting engineering choices, but not as the paper's main claim.

## 3. Literature review from your folders

## 3.1 Folder `research_paper`: core papers on meta-path search and selection

These are the most relevant papers in your first folder for a 2024-2026 positioning.

### A. `Long-range Meta-path Search on Large-scale Heterogeneous Graphs`

- Source in folder: `research_paper/Long-range Meta-path Search on Large-scale Heterogeneous Graphs.pdf`
- Venue/date from paper front page: **NeurIPS 2024**
- Main idea: build a large candidate pool of target-related meta-paths, then use **progressive sampling** to shrink the search space and retrain on the selected paths.
- Why it matters to you: this is the cleanest recent argument that only a small subset of long-range meta-paths is actually useful.
- Limitation: it still assumes a predefined candidate schema pool and remains a **selection** method, not a true topology construction method.

### B. `Rethinking Heterogeneous Graph Representation Learning via Relational Subset Selection`

- Source in folder: `research_paper/Rethinking Heterogeneous Graph Representation Learning via Relational Subset Selection.pdf`
- This is the extended "LMSPS/LPSMS" line.
- Main idea: recast heterogeneous learning as **subset selection**, then estimate path importance through progressive search and leave-one-out marginal contribution.
- Why it matters to you: the paper gives a strong conceptual shift from "aggregate more neighbors" to "select better relations."
- Limitation: still works at the level of meta-path subsets, not at the level of **learned 2-cells** and not with sheaf geometry.

### C. `Adaptive Meta-Path-based Neural Network Architecture Search for Heterogeneous Graphs`

- Source in folder: `research_paper/Adaptive meta-path-based neural network architecture search for heterogeneous graphs.pdf`
- Journal front page: received **November 4, 2025**, accepted **April 23, 2026**, online pre-proof **April 24, 2026**.
- Main idea: dependency-aware meta-path search plus architecture search; explicitly reduces redundancy among path combinations.
- Why it matters to you: this is the most current paper in your folder that treats meta-path search as a joint optimization problem rather than manual preprocessing.
- Limitation: still assumes the main object is a set of meta-path channels for an HGNN, not a higher-order topological complex.

### D. `HISE-KT: Synergizing Heterogeneous Information Networks and LLMs for Explainable Knowledge Tracing with Meta-Path Optimization`

- Source in folder: `research_paper/HISE-KT Synergizing Heterogeneous Information Networks and LLMs for Explainable Knowledge Tracing with Meta-Path Optimization.pdf`
- arXiv front page: `2511.15191v1`, submitted **November 19, 2025**
- Main idea: use an LLM to score/filter meta-path instances and keep higher-quality ones for a downstream reasoning task.
- Why it matters to you: this is useful evidence that **instance quality** matters, not just schema choice.
- Limitation: task-specific, LLM-heavy, and not topology-native.

### E. `Differentiable Meta Multigraph Search with Partial Message Propagation on Heterogeneous Information Networks`

- Source in folder: `research_paper/Differentiable Meta Multigraph Search with Partial Message Propagation on Heterogeneous Information Networks.pdf`
- arXiv front page: `2211.14752v1`, submitted **November 27, 2022**
- Main idea:
  - introduce **Partial Message Meta Multigraph search (PMMM)**;
  - search over a **meta-multigraph** rather than a meta-path or meta-graph;
  - use **partial message propagation** to stabilize differentiable search by randomly activating only a subset of candidate message-passing types per iteration.
- Important technical insight:
  - ordinary differentiable search can be unstable because all candidate paths are jointly optimized and strongly coupled;
  - PMMM reduces this coupling by sampling only part of the candidate paths each iteration so paths are more equally trained;
  - unlike classic meta-graph derivation that keeps only one strongest path per edge, PMMM argues for retaining multiple message-passing types when the graph semantics require it.
- Why it matters to you:
  - this is an early and very relevant precursor to the later 2024 `Meta-multigraph search` journal paper;
  - it strengthens the argument that the search object should not be a single rigid path, but a more expressive higher-order structure;
  - it is especially useful for your design because `H2S-NN` should not assume only one semantic route between two typed states.
- Limitation:
  - still designed for heterogeneous message passing rather than sheaf 2-cell construction;
  - still ultimately searches semantic propagation structure, not topology-native Hodge structure.

### F. Other useful papers in `research_paper`

These are still valuable, even if they are not the exact 2024-2026 anchor papers:

- `A New Meta-path Monte Carlo Tree Search Algorithm for Heterogeneous Graph Neural Networks`
  - Useful for tree-search framing.
  - Weakness: expensive evaluation loop.

- `HG-search: multi-stage search for heterogeneous graph neural networks`
  - Useful as a general HGNN search baseline.
  - Weakness: architecture-search cost and instability.

- `Inductive Meta-path Learning for Schema-complex Heterogeneous Information Networks`
  - Useful because it moves search to the **schema graph**, not only the observed graph.
  - This is conceptually close to what you want.

- `Enhancing Heterogeneous Graph Learning with Semantic-Aware Meta-Path Diffusion`
  - Useful for semantic perturbation and diffusion ideas.
  - Weakness: still operates on given meta-paths.

- `Meta-Path Learning for Multi-relational Graph Neural Networks`
  - Useful for many-relation settings where manual meta-path design breaks down.
  - Weakness: greedy relation extension.

My read of the whole folder is:

- The field has become good at **ranking / selecting / pruning** meta-paths.
- The field is still weak at **constructing higher-order objects automatically** for sheaf or Hodge models.

That gap is exactly where `H2S-NN` can contribute.

## 3.2 Folder `research_paper02`: supporting papers for the design choices

This folder is smaller, but strategically useful.

### A. `DeepSN: A Sheaf Neural Framework for Influence Maximization`

- Source in folder: `research_paper02/DeepSN A Sheaf Neural Framework for Influence Maximization.pdf`
- arXiv front page: `2412.12416v1`, submitted **December 16, 2024**
- Main idea: use sheaf diffusion to model richer propagation patterns than standard GNNs.
- Why it matters to you:
  - supports the claim that sheaf geometry helps when propagation is long-range, asymmetric, and task-dependent;
  - reinforces your case that higher-order propagation choices should be learned, not hand-fixed.

### B. `Sheaf HyperNetworks for Personalized Federated Learning`

- Source in folder: `research_paper02/Sheaf HyperNetworks for Personalized Federated Learning.pdf`
- arXiv front page: `2405.20882v1`, submitted **May 31, 2024**
- Main idea: use sheaf structure when an explicit relation graph may be weak, private, or missing.
- Why it matters to you:
  - directly supports your motivation for datasets where useful meta-path priors are unavailable;
  - suggests a practical idea: infer relational structure first, then optimize higher-order topology on top of it.

### C. `Neural Architecture Search with Reinforcement Learning`

- Source in folder: `research_paper02/NEURAL ARCHITECTURE SEARCH WITH reinforcement learning.pdf`
- This is older background, not part of your 2024-2026 literature target.
- Why it still matters:
  - it is the classical RL search template;
  - it helps explain why naive search is expensive and unstable.

### D. `Neural Architecture Search: Insights from 1000 Papers`

- Source in folder: `research_paper02/Neural Architecture Search Insights from 1000 Papers.pdf`
- Also background rather than a recent anchor.
- Why it matters:
  - it argues for progressive search, weight sharing, low-fidelity evaluation, and search/evaluation consistency;
  - those lessons transfer directly to meta-path search.

My design takeaway from folder 2 is simple:

- Use **sheaf geometry** for expressivity.
- Use **progressive / one-shot / low-cost search**, not brute-force RL, for scalability.

## 3.3 Folder `research_paper_reinforce`: search/RL papers that matter methodologically

This folder does **not** contain heterogeneous-graph meta-path papers directly. Instead, it contains general search / RL papers that are helpful for deciding how an automatic meta-path search component should be designed.

### A. `Single-Agent Policy Tree Search With Guarantees`

- Source in folder: `research_paper_reinforce/Single-Agent Policy Tree Search With Guarantees.pdf`
- arXiv front page: `1811.10928v2`, updated **November 28, 2018**
- Main idea:
  - develop policy-guided tree search algorithms with search-effort guarantees;
  - distinguish between best-first enumeration and sampling-style search;
  - show that plain MCTS is poorly suited to sparse-reward deterministic single-agent problems.
- Why it matters to you:
  - meta-path discovery is also a sparse-reward sequential search problem;
  - this paper supports a move away from naive MCTS toward **policy-guided search with bounded effort**.
- Limitation for your use case:
  - it is not about heterogeneous graphs or meta-paths;
  - by itself it does not create novelty for `H2S-NN`, but it is a good algorithmic inspiration.

### B. `Policy-Guided Heuristic Search with Guarantees`

- Source in folder: `research_paper_reinforce/Policy-Guided Heuristic Search with Guarantees.pdf`
- arXiv front page: `2103.11505v1`, submitted **March 21, 2021**
- Main idea:
  - combine a learned policy and a heuristic in a search algorithm with guarantees;
  - argue that policy and heuristic are complementary rather than redundant.
- Why it matters to you:
  - this is very relevant to schema search on the type graph;
  - you can treat the policy as proposing promising schema extensions and the heuristic as scoring partial schemas cheaply before full Hodge training.
- Strong takeaway:
  - if you want to use RL at all, this paper suggests that **policy-only search is usually not enough**;
  - the strongest design is often **policy + heuristic + bounded search**.

### C. `Beyond Single-Step Updates: Reinforcement Learning of Heuristics with Limited-Horizon Search`

- Source in folder: `research_paper_reinforce/Reinforcement Learning of Heuristics with Limited-Horizon Search.pdf`
- arXiv front page: `2511.10264v1`, submitted **November 13, 2025**
- Main idea:
  - criticize single-step Bellman updates for search problems;
  - use limited-horizon search to build stronger training targets for heuristic learning.
- Why it matters to you:
  - very useful for your reward design problem;
  - instead of evaluating a candidate schema only by a final validation score after full training, you can use a **limited-horizon topology-aware proxy** based on partial search or partial diffusion.
- This is probably the most actionable paper in the folder for H2S-NN.

### Folder-level takeaway

This folder pushes the same conclusion from a different angle:

- pure Monte Carlo or pure policy-gradient search is usually too weak or too expensive on sparse-reward search problems;
- better approaches combine:
  - a learned proposal policy,
  - a cheap heuristic or surrogate score,
  - and a bounded or limited-horizon evaluation mechanism.

That is exactly how I would treat RL or MCTS in `H2S-NN`: as a **search subroutine**, not as the core novelty.

## 4. Recent external papers and what they add

These are the most relevant external additions I would use to update your review to the 2024-2026 landscape.

### 4.1 `Meta-multigraph search: Rethinking meta-structure on heterogeneous information networks`

- Journal: *Knowledge-Based Systems*
- Published: **April 8, 2024**
- DOI: `10.1016/j.knosys.2024.111524`
- Main idea:
  - generalize beyond meta-path and meta-graph into a more expressive **meta-multigraph**;
  - use stable differentiable search (`PMMM`, `PCMM`);
  - explicitly address search instability and search/evaluation mismatch.
- Context:
  - this journal paper is best read as a stronger successor to the earlier PMMM paper in your local folder, `Differentiable Meta Multigraph Search with Partial Message Propagation on Heterogeneous Information Networks`;
  - the earlier PMMM paper contributes the key idea of **partial message propagation** for stabilizing differentiable search, while the later journal version broadens the meta-structure analysis and adds progressive complex-to-concise search.
- Why it matters to H2S-NN:
  - it shows the field is already moving from "fixed meta-paths" to **richer searchable structures**;
  - your paper can go one step further: from searchable meta-structures to searchable **2-cell sheaf complexes**.

### 4.2 `HeteroSample: Meta-path Guided Sampling for Heterogeneous Graph Representation Learning`

- arXiv: `2411.07022`
- Submitted: **November 11, 2024**, revised **December 18, 2024**
- Later appears in IEEE IoT Journal (DBLP entry shows 2025 journal publication).
- Main idea:
  - use meta-path-guided sampling to preserve important semantic and type structure while reducing graph size.
- Why it matters to H2S-NN:
  - useful for your scaling story;
  - automatic face search will need **budgeted instance sampling**, and this paper is a recent supporting reference for that choice.

### 4.3 `EvoPath: Evolutionary meta-path discovery with large language models for complex heterogeneous information networks`

- Journal: *Information Processing and Management*
- Publication date: **January 2025**
- DOI: `10.1016/j.ipm.2024.103920`
- Main idea:
  - combine LLM prompting with an evolutionary buffer to discover meta-paths in schema-rich HINs.
- Why it matters to H2S-NN:
  - supports the claim that manual schema design fails in complex relation spaces;
  - suggests a weak-prior option: use language models only for **proposal generation**, not for the main optimization.
- Caution:
  - do not make the main contribution "LLM generates meta-paths." That is much easier to attack on novelty and robustness grounds.

### 4.4 `Preserving high-order ego-centric topological patterns in node representation in heterogeneous graph` (`TAEG-HG`)

- Journal: *Knowledge-Based Systems*
- 2025 journal paper
- DOI: `10.1016/j.knosys.2025.113067`
- Main idea:
  - automatically learn **task-oriented higher-order ego-centric topological patterns** instead of relying on fixed path templates.
- Why it matters to H2S-NN:
  - this is one of the most relevant "outside the box" recent papers for you;
  - it validates the broader trend that higher-order structure should be **learned from task supervision**, not manually enumerated.
- Important difference:
  - it learns pattern anchors for alignment;
  - you can learn typed path schemas that become **2-cells** for Hodge diffusion.

### 4.5 `FHGE: A Fast Heterogeneous Graph Embedding with Ad-hoc Meta-paths`

- arXiv: `2502.16281`
- Submitted: **February 22, 2025**
- Main idea:
  - make meta-path-guided embedding fast and retraining-free for ad-hoc query-time use.
- Why it matters to H2S-NN:
  - this is useful as a deployment/scalability reference;
  - it strengthens the case that future systems should support **on-demand higher-order semantics**, not only manually fixed training-time paths.

### 4.6 `Is Meta-Path Attention an Explanation? Evidence of Alignment and Decoupling in Heterogeneous GNNs`

- arXiv: `2602.08500`
- Submitted: **February 9, 2026**
- Main idea:
  - shows that meta-path attention weights do **not** reliably equal true meta-path importance;
  - proposes a meta-path-aware explanation protocol and an alignment metric.
- Why it matters to H2S-NN:
  - very important for evaluation;
  - if you learn schema weights `alpha_sigma`, you should **not** claim they are explanations by default.

### 4.7 `Global Meta-path-level Counterfactual Explanation for Heterogeneous Graph Neural Networks by Path Exclusion`

- OpenReview submission to ICLR 2026
- Submitted: **September 17, 2025**, modified **February 11, 2026**
- Main idea:
  - exclude individual meta-paths during the forward pass to measure counterfactual importance;
  - reports accuracy drops of roughly 5-12 percent when critical paths are removed.
- Why it matters to H2S-NN:
  - gives you a much stronger evaluation protocol than attention visualization;
  - you can adapt this idea to **face-schema exclusion**.

## 5. What is still unsolved in 2026

Across the folders and external papers, the unresolved problems are now quite clear.

### 5.1 Candidate pools are still usually manual

Even recent methods like LMSPS and AMP-HGNAS usually start from a predefined schema pool or a handcrafted search space. They automate **selection**, but not fully **discovery from weak prior structure**.

### 5.2 Search is still too tied to validation accuracy

Many methods require repeated downstream training or partial retraining to evaluate path quality. That is expensive and often unstable.

This is also why a direct RL or MCTS formulation is risky for `H2S-NN`: if every action sequence must be rewarded by a full downstream run, search becomes prohibitively costly.

### 5.3 Most methods stay at the schema/view level

They learn which meta-path channels to aggregate, but they do not learn which higher-order patterns should become actual **topological objects** in a sheaf complex.

### 5.4 Search/evaluation mismatch remains a real issue

This is emphasized by meta-multigraph search and by NAS literature more broadly. A path that looks good during a relaxed search phase may not remain good after discrete selection and retraining.

### 5.5 Attention is not a reliable explanation

The 2026 meta-path explanation paper is a warning sign: learned path weights are not automatically faithful explanations.

### 5.6 Datasets without obvious meta-path priors remain awkward

This is exactly the pain point you mentioned. Many current methods still assume the user can say something like "use APA, APCPA, PAP, PSP," which breaks when:

- the schema is unfamiliar,
- the relation vocabulary is large,
- or the domain does not have accepted canonical paths.

### 5.7 Nobody is really doing topology-native meta-path search for sheaf 2-complexes

This is the best opening for `H2S-NN`.

## 6. Recommended paper direction

My recommendation is to make the paper about:

> **Automatic discovery of face-generating schemas for heterogeneous sheaf 2-complexes**

This is more distinctive than "meta-path search for another HGNN," and it is more publishable than "we added a search module to H2S-NN."

### Working paper titles

1. `Learning Higher-Order Heterogeneous Sheaf Complexes without Predefined Meta-Paths`
2. `Schema-to-Cell Search for Hodge Neural Networks on Heterogeneous Graphs`
3. `Automatic 2-Cell Discovery for Heterogeneous Sheaf Neural Networks`

## 7. The concrete method I would recommend

I would recommend a method in the spirit below.

### 7.1 Core idea

Do **not** search directly over node-level path instances first.

Instead:

1. search over **typed schemas** on the schema graph / type graph;
2. instantiate only a budgeted set of promising path instances;
3. convert those instances into weighted 2-cells;
4. run Hodge/sheaf diffusion on the resulting learned 2-complex.

This is the cleanest way to keep the method scalable and distinct from HetSheaf.

Operationally, I would treat manual meta-paths as an **optional warm start**, not as a requirement. That means `H2S-NN` should support three regimes:

- manual-schema mode,
- schema-refinement mode,
- auto-discovery mode.

This is important because your actual motivation is not only "discover paths when none exist" but also "correct bad path choices when given paths are suboptimal."

### 7.2 Search object

Let the heterogeneous graph be

`G = (V, E, phi, psi)`

and let the type graph / schema graph define valid typed transitions.

Instead of assuming a predefined meta-path set `P`, define a search space of typed schemas:

`Sigma = {sigma : sigma is a valid typed walk of length <= L_max on the type graph}`

Examples:

- `Author -> Paper -> Author`
- `Paper -> Venue -> Paper`
- `User -> Item -> Category -> Item`

but the model should generate these automatically from the type graph constraints rather than from manual expert choice.

### 7.3 Progressive schema proposal

Use a schema generator over the type graph:

- constrained random walk,
- beam search,
- or a differentiable next-relation policy.

For each schema `sigma`, compute cheap structural features before full training, for example:

- estimated path count,
- anchor-node coverage,
- relation-type diversity,
- endpoint type entropy,
- overlap with already selected schemas,
- sparsity / length penalty.

Then learn a schema score

`u_theta(sigma)`

and select a sparse subset with Gumbel-top-k or progressive sampling.

This is where LMSPS/LPSMS and AMP-HGNAS help, but your selected objects are not just HGNN channels. They are candidate **face generators**.

If user-provided schemas are available, include them in the initial candidate pool, but do not force them to survive. Let the model:

- keep them if useful,
- prune them if redundant,
- and expand nearby variants if better schemas exist.

### 7.4 Instance selection after schema selection

For each chosen schema `sigma`, mine or sample a budgeted set of path instances `I(sigma)`.

Then learn instance weights

`beta_f = v_theta(f)` for `f in I(sigma)`.

These weights should depend on local features such as:

- endpoint relevance,
- path rarity / frequency,
- edge disagreement magnitude from `delta_0 x`,
- local label consistency proxy,
- overlap with other instances.

This matters because recent work like `HISE-KT` suggests that **instance quality** is as important as schema quality.

### 7.5 Weighted 2-cell construction

For every selected path instance `f`, create a face and include it in the 2-complex with a learnable importance weight.

A clean formulation is:

`L_1^up = sum_{sigma} alpha_sigma sum_{f in I(sigma)} beta_f (delta_1^f)^T delta_1^f`

and

`L_1 = delta_0 delta_0^T + L_1^up`

where:

- `alpha_sigma` = schema-level importance,
- `beta_f` = instance-level importance.

This is the main mathematical point:

> We are not only learning sheaf maps on a fixed graph. We are learning which higher-order cells participate in the sheaf Hodge operator.

That is a much stronger and much more distinct claim than HetSheaf.

### 7.6 Objective

A practical objective would be:

`L = L_task + lambda_sparse L_sparse + lambda_red L_redundancy + lambda_cov L_coverage + lambda_stab L_stability`

with:

- `L_task`: downstream supervised loss;
- `L_sparse`: encourages a compact schema set;
- `L_redundancy`: penalizes highly overlapping schemas/instances;
- `L_coverage`: avoids degenerate selection of one narrow semantic family;
- `L_stability`: discourages large fluctuations in selected schemas across nearby epochs.

If you want a more topology-native criterion, I would add:

- `L_hodge`: reward schemas whose induced faces reduce edge inconsistency or improve useful `L_1` structure;
- optional future extension: a holonomy-like regularizer, but I would keep that as a second paper unless you can formalize it very carefully.

## 8. Why this is different from HetSheaf

This proposed paper differs from `HetSheaf` in the actual problem, not just in the implementation.

### HetSheaf asks:

- how do we learn heterogeneous restriction maps on a typed graph?
- how do we pool sheaf node representations well?

### Your paper would ask:

- how do we discover the higher-order heterogeneous topology itself?
- which typed path schemas deserve to become 2-cells?
- how should those cells enter the Hodge operator?

So the contrast is:

- `HetSheaf`: learn geometry on a fixed 1-skeleton.
- `Auto-H2S`: learn the **higher-order skeleton** that defines the geometry.

That is a strong distinction.

## 9. A lower-risk version and a higher-risk version

## 9.1 Lower-risk version

If you want the most feasible first submission:

- keep path schemas as typed walks only;
- use progressive schema search + instance sampling;
- learn `alpha_sigma` and `beta_f`;
- integrate them into `L_1`.

This is already strong enough for a paper if the experiments work.

## 9.2 Higher-risk, higher-reward version

Move beyond meta-paths entirely and search for **task-oriented higher-order typed patterns**, closer in spirit to `TAEG-HG`, then convert those patterns into admissible 2-cells or small higher-order motifs.

This could be more novel, but it is harder to make mathematically clean and computationally stable.

## 9.3 LLM-assisted version

A third option is to use LLMs only for **schema proposal**, inspired by `EvoPath` and `HISE-KT`:

- LLM proposes candidate typed schemas from text descriptions of node/edge types;
- the actual acceptance/rejection is done by your topology-aware scoring model.

I would only do this if you want an application-heavy or explainability-heavy paper. For a strong ML/DM paper, the topology-learning story is cleaner.

## 9.4 RL / Monte-Carlo-assisted version

If you still want to use RL or Monte Carlo search, I would position it like this:

- the **main novelty** is still automatic higher-order topology learning;
- RL or Monte Carlo is only the **optimization mechanism** for exploring schema space.

The strongest version would be:

- policy-guided schema expansion on the type graph;
- heuristic pruning or bounded best-first search on partial schemas;
- limited-horizon topology-aware rewards instead of full-train rewards for every candidate.

This is much stronger than simply saying:

> "We use reinforcement learning to find meta-paths."

That headline would be too close to prior MCTS/RL meta-path papers and too easy for reviewers to dismiss as a search-algorithm swap.

## 10. Experimental plan

## 10.1 Baselines

You should compare against:

- `H2S-NN` with manual meta-paths
- `HetSheaf`
- strong meta-path HGNN baselines such as `HAN` / `MAGNN`
- search baselines such as `LMSPS/LPSMS`, `AMP-HGNAS`, and if feasible `HG-Search`
- meta-path-free heterogeneous baselines such as `HGT` or similarly strong recent heterogeneous GNNs

## 10.2 Core ablations

At minimum:

1. manual meta-paths vs automatic schema search
2. schema search only vs schema + instance weighting
3. selected schemas used as plain HGNN channels vs selected schemas lifted as sheaf 2-cells
4. task-only scoring vs task + redundancy + coverage
5. fixed top-k vs progressive sampling

The key ablation is:

> Is the gain coming from better path selection alone, or from the fact that selected paths become actual 2-cells inside the Hodge operator?

If you cannot answer that, the paper will look like "meta-path search plus existing H2S-NN."

## 10.3 Datasets

For a convincing paper, I would use:

- canonical heterogeneous datasets: `DBLP`, `ACM`, `IMDB`, `OGBN-MAG`
- at least one large heterogeneous benchmark where schema size is nontrivial
- if possible, one setting where canonical domain meta-paths are unclear or intentionally removed/obscured

That last setting is important because it directly supports your motivation:

> the method is useful when predefined meta-paths are unavailable or unreliable.

## 10.4 Evaluation beyond accuracy

You should report:

- predictive performance
- search cost / wall-clock
- number of selected schemas
- stability across random seeds
- transferability of discovered schemas across backbones
- counterfactual importance of selected schemas/faces

For explanation, do **not** rely only on learned weights.
Use a path-exclusion or face-exclusion protocol inspired by the 2026 explanation papers.

## 11. The actual paper claim I would target

I would aim for a claim like this:

> Existing heterogeneous sheaf models learn geometry on fixed relational structure, while existing meta-path search methods choose semantic channels for HGNN aggregation. We unify these directions by learning which typed higher-order path schemas should instantiate the 2-cells of a heterogeneous sheaf complex, allowing end-to-end topology construction for Hodge neural diffusion without predefined meta-paths.

That claim is:

- novel enough,
- clearly different from `HetSheaf`,
- grounded in a real limitation of current methods,
- and aligned with the mathematical strengths already present in `H2S-NN`.

## 12. Bottom-line recommendation

If I were advising this as a professor, I would say:

1. **Do not** position the next paper as "HetSheaf plus meta-paths."
2. Position it as **automatic higher-order topology learning** for heterogeneous sheaf/Hodge networks.
3. Make the searched object the **2-cell schema**, not just the HGNN meta-path channel.
4. Use recent meta-path search work mainly as scaffolding for:
   - progressive schema search,
   - redundancy control,
   - instance filtering,
   - and evaluation.
5. Keep the main mathematical novelty inside the learned `L_1^up` construction.

And in implementation terms:

6. build `H2S-NN` so it does not break when meta-paths are unavailable;
7. build `H2S-NN` so it is not trapped by poor manual meta-path choices when they are available.

That is the cleanest road to a paper that is both technically grounded and visibly distinct from `HetSheaf`.

## 13. Sources used

### Local project sources

- `CLAUDE.md`
- `HETSHEAF.pdf`
- `meta_path_survey.md`
- `research_paper/Adaptive meta-path-based neural network architecture search for heterogeneous graphs.pdf`
- `research_paper/Differentiable Meta Multigraph Search with Partial Message Propagation on Heterogeneous Information Networks.pdf`
- `research_paper/Long-range Meta-path Search on Large-scale Heterogeneous Graphs.pdf`
- `research_paper/Rethinking Heterogeneous Graph Representation Learning via Relational Subset Selection.pdf`
- `research_paper/HISE-KT Synergizing Heterogeneous Information Networks and LLMs for Explainable Knowledge Tracing with Meta-Path Optimization.pdf`
- `research_paper02/DeepSN A Sheaf Neural Framework for Influence Maximization.pdf`
- `research_paper02/Sheaf HyperNetworks for Personalized Federated Learning.pdf`
- `research_paper02/NEURAL ARCHITECTURE SEARCH WITH reinforcement learning.pdf`
- `research_paper02/Neural Architecture Search Insights from 1000 Papers.pdf`
- `research_paper_reinforce/Single-Agent Policy Tree Search With Guarantees.pdf`
- `research_paper_reinforce/Policy-Guided Heuristic Search with Guarantees.pdf`
- `research_paper_reinforce/Reinforcement Learning of Heuristics with Limited-Horizon Search.pdf`

### External sources

- HetSheaf arXiv page: https://arxiv.org/abs/2409.08036
- HeteroSample arXiv page: https://arxiv.org/abs/2411.07022
- FHGE arXiv page: https://arxiv.org/abs/2502.16281
- Is Meta-Path Attention an Explanation? arXiv page: https://arxiv.org/abs/2602.08500
- Meta-multigraph search (Knowledge-Based Systems, 2024): https://www.sciencedirect.com/science/article/abs/pii/S095070512400159X
- EvoPath (Information Processing and Management, 2025): https://www.sciencedirect.com/science/article/pii/S0306457324002796
- EvoPath DBLP record: https://dblp.org/rec/journals/ipm/LiuCWHFL25
- TAEG-HG / Preserving high-order ego-centric topological patterns... (Knowledge-Based Systems, 2025): https://www.sciencedirect.com/science/article/abs/pii/S0950705125001145
- Global Meta-path-level Counterfactual Explanation by Path Exclusion (OpenReview, ICLR 2026 submission): https://openreview.net/forum?id=qum7m6TmLL
