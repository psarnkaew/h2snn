# RL / Monte-Carlo Meta-Path Search for H2S-NN: Why It Is Interesting but Not Enough

Date: 2026-05-06

## 1. Short answer

Yes, Monte-Carlo search or reinforcement learning is an interesting way to automatically find meta-paths.

But for `H2S-NN`, it is probably **not novel enough as the main contribution** if the paper is only:

> "we use RL/MCTS to search for meta-paths automatically."

That idea already has close neighbors in the literature:

- RL-based meta-path discovery
- MCTS-based meta-path search
- HGNN architecture/meta-structure search
- LLM/evolutionary meta-path proposal

So if you make RL or MCTS the headline, reviewers can easily say:

> this is another search algorithm on top of meta-path selection.

That is the core risk.

## 2. Why RL / MCTS is still attractive

The idea is attractive for good reasons.

### 2.1 Meta-path discovery is naturally sequential

A meta-path is built step by step:

- choose a start type,
- choose the next relation,
- choose the next node type,
- continue until stop.

That looks exactly like:

- a policy rollout,
- a tree search,
- or a sequential decision process.

So your instinct is mathematically reasonable.

### 2.2 It can work without fixed manual path templates

RL/MCTS can search directly on the type graph or schema graph, which is good for datasets where:

- canonical meta-paths are not known,
- the schema is large,
- or expert priors are weak.

### 2.3 It can support variable-length schemas

Unlike a fixed-path candidate pool, a policy search method can stop early or continue deeper. That is useful for discovering:

- short high-signal schemas,
- long sparse schemas,
- or multiple lengths in one framework.

So the attraction is real. The problem is not that RL/MCTS is bad. The problem is that **RL/MCTS alone is not enough of a paper idea anymore**.

## 3. Why it is not enough novelty by itself

## 3.1 There are already close meta-path search precedents

From your paper collection and recent literature, the field already has:

- Monte-Carlo tree search for meta-path search
- reinforcement-learning meta-path discovery
- neural architecture search for heterogeneous graphs
- progressive meta-path search
- meta-multigraph search

So a paper whose main message is:

> "We also search meta-paths automatically with RL or tree search"

will likely look incremental unless the searched object, reward, or mathematical integration is genuinely new.

## 3.2 Search over meta-paths is not the same as novelty in H2S-NN

This is the most important point.

`H2S-NN` is not just another HGNN that consumes meta-path channels. Its real strength is:

- meta-path instances become **2-cells**,
- those 2-cells define `delta_1`,
- `delta_1` defines `L_1^up`,
- and `L_1` changes the actual Hodge/sheaf diffusion geometry.

So if you only contribute:

- a new policy for selecting meta-paths,

then you are improving the **preprocessing/search stage**, but not fully exploiting what makes `H2S-NN` special.

That is why the novelty would feel thin.

## 3.3 Vanilla RL/MCTS is computationally awkward here

If each candidate path or schema must be evaluated by:

- constructing faces,
- learning sheaf maps,
- building `L_1`,
- training the downstream model,
- then reading validation accuracy,

the reward is:

- sparse,
- delayed,
- expensive,
- and noisy.

That is a bad regime for naive RL/MCTS.

This is also why many search papers end up needing:

- pruning,
- progressive shrinking,
- low-fidelity evaluation,
- heuristic scoring,
- or surrogate models.

So even from an optimization perspective, RL/MCTS alone is usually not the right full story.

## 3.4 Reviewers will ask: why not just use LMSPS / AMP-HGNAS / MCTS-HGNN?

If your proposal is mainly search-based, the obvious reviewer question is:

> Why is this not just another search controller on top of existing meta-path methods?

If the answer is only:

- "because we use H2S-NN after selecting the paths,"

that is probably not enough.

You need the answer to be:

> Because we are not only selecting meta-paths for aggregation; we are learning which schemas instantiate the 2-cells of a heterogeneous sheaf complex, and this changes the learned Hodge operator itself.

That is a much better answer.

## 4. What RL / MCTS should be inside H2S-NN

My recommendation is:

> RL or Monte-Carlo search should be a **means**, not the **headline**.

In other words:

- the paper's main novelty = **automatic higher-order topology learning**
- the RL/MCTS component = **one possible optimizer for schema exploration**

That positioning is much stronger.

## 4.1 Important clarification: this should be part of H2S-NN itself

This point is important enough to say plainly:

> `H2S-NN` itself should have a built-in mechanism for handling missing, weak, or non-optimal meta-paths.

In other words, the target is **not**:

- one standalone model called `H2S-NN`, plus
- one unrelated search paper that finds meta-paths for it.

The better target is:

- `H2S-NN` as the core sheaf/Hodge model, and
- an integrated **schema discovery / schema refinement module** inside `H2S-NN`.

So the full story becomes:

1. if good domain meta-paths are available, `H2S-NN` can use them;
2. if the provided meta-paths are mediocre, `H2S-NN` can reweight, prune, or refine them;
3. if no meta-paths are available, `H2S-NN` can discover candidate schemas automatically from the type graph.

That makes the method far more practical and much easier to defend.

## 4.2 The design requirement for H2S-NN

Your model should support three modes, ideally within one unified implementation:

### Mode A. Manual-schema mode

Input:

- user-provided meta-path schemas

Behavior:

- instantiate faces from those schemas
- learn schema weights `alpha_sigma`
- learn instance weights `beta_f`

This is the standard, strongest-prior regime.

### Mode B. Schema-refinement mode

Input:

- user-provided meta-path schemas, but not assumed optimal

Behavior:

- expand nearby alternatives on the schema graph
- drop redundant or weak schemas
- add improved local variants

This directly addresses the case:

> the given meta-paths are not the best.

### Mode C. Auto-discovery mode

Input:

- no meta-paths

Behavior:

- generate schemas on the type graph automatically
- score them
- instantiate only promising schemas as faces

This directly addresses the case:

> we do not have predefined meta-paths.

In my view, this three-mode capability is exactly what would make `H2S-NN` a robust model rather than a fragile research prototype.

## 5. What would be novel enough

Here are three versions, from strongest to weakest.

## 5.1 Strongest version: topology-aware schema-to-cell search

This is the direction I would recommend most.

### Main claim

Learn which typed schemas should instantiate 2-cells in the heterogeneous sheaf complex.

### RL/MCTS role

Use RL or tree search to explore schema space on the type graph.

### Real novelty

The reward is not just downstream accuracy of a meta-path HGNN. Instead, selected schemas become weighted faces and directly change:

- `delta_1`
- `L_1^up`
- `L_1`
- Hodge diffusion

That is a different object from ordinary meta-path search.

### Good headline

`Policy-Guided Schema-to-Cell Search for Heterogeneous Hodge Neural Networks`

or

`Learning Heterogeneous Sheaf 2-Complexes via Topology-Aware Schema Search`

In this version, RL is present, but the novelty is the **learned higher-order sheaf complex**.

## 5.2 Middle version: RL plus topology-aware surrogate reward

If you really want RL to be prominent, then the reward must be more interesting than final validation accuracy.

For example, the reward could combine:

- task loss improvement,
- reduction in edge inconsistency,
- usefulness of induced `L_1^up`,
- coverage of informative regions,
- redundancy penalty,
- stability across nearby epochs.

Then your message becomes:

> We design a topology-aware reward for searching face-generating schemas.

That is better than generic RL meta-path search.

Still, I would keep the paper centered on the topology, not the RL.

## 5.3 Weakest acceptable version: RL as proposal, deterministic scoring as selector

This is lower risk and probably easier to implement.

Use RL/MCTS only to generate candidates:

- propose schema extensions,
- explore uncommon long-range paths,
- diversify candidate pools.

Then use a deterministic scoring module to accept/reject schemas.

This is weaker scientifically than 5.1, but still better than "pure RL solves meta-path search."

## 6. A concrete H2S-NN formulation that makes sense

If I were designing the method, I would use the following structure.

Before the search starts, allow an optional seed set of user-provided schemas:

- if present, treat them as warm-start candidates rather than fixed truths;
- if absent, initialize purely from the type graph.

### Step 1. Search space

Search over partial typed schemas on the type graph:

`sigma_t = (tau_0, r_1, tau_1, ..., r_t, tau_t)`

where each extension is schema-valid.

### Step 2. Policy

A policy proposes the next relation/type extension or stop action:

`pi_theta(a_t | sigma_t)`

This is where RL or MCTS can be used.

### Step 3. Heuristic / surrogate

For each partial or complete schema, compute a cheap score:

- estimated instance count
- target-node coverage
- overlap with already selected schemas
- type entropy
- length penalty
- preliminary topological utility proxy

Call this:

`h_phi(sigma)`

This is important because policy-only search is usually too expensive and unstable.

### Step 4. Instantiate faces only for promising schemas

For selected schemas, sample path instances and turn them into candidate faces.

If manual schemas were provided, they should simply be one subset of the candidate pool. They must not be forced to survive if the model finds better alternatives.

### Step 5. Weighted Hodge construction

Use learned schema weights `alpha_sigma` and instance weights `beta_f`:

`L_1^up = sum_sigma alpha_sigma sum_{f in I(sigma)} beta_f (delta_1^f)^T delta_1^f`

This is the real centerpiece.

### Step 6. Reward

If you use RL, do not rely only on final validation accuracy. Use a mixed reward like:

`R = R_task + gamma_1 R_topo + gamma_2 R_cov - gamma_3 R_red - gamma_4 R_cost`

where:

- `R_task`: downstream performance proxy
- `R_topo`: topology-aware benefit
- `R_cov`: coverage/diversity
- `R_red`: redundancy
- `R_cost`: search or face-construction cost

That gives you a much more `H2S-NN`-specific story.

## 7. Why Monte-Carlo alone is especially risky

Monte-Carlo tree search sounds appealing because schema search is tree-structured. But for this problem, plain MCTS has three issues:

1. reward is sparse and delayed
2. branching factor can be large on schema-rich graphs
3. each rollout can be expensive if it touches actual model training

The search papers in your `research_paper_reinforce` folder are useful exactly because they show how people move beyond naive MCTS:

- policy-guided search
- heuristic-guided search
- limited-horizon evaluation
- bounded search effort

So if you use Monte-Carlo ideas, the modern version should be:

- policy-guided
- heuristic-pruned
- limited-horizon
- topology-aware

not plain MCTS over path sequences.

## 8. What I would say in a paper, honestly

If I were advising this as a professor, I would say the following very directly:

### What not to do

Do not write a paper whose main pitch is:

> We replace manual meta-path design with reinforcement learning / Monte-Carlo search.

That is too easy to compare against prior search papers and too hard to defend as enough novelty.

### What to do instead

Write a paper whose main pitch is:

> We automatically learn the higher-order heterogeneous topology used by a sheaf/Hodge network. We optionally use policy-guided or Monte-Carlo search to explore schema space, but the main novelty is that selected schemas instantiate weighted 2-cells and alter the learned Hodge operator.

That is a strong, defensible story.

## 9. Practical recommendation

If you want the safest high-quality research direction:

1. keep the main contribution as **automatic schema-to-cell construction**
2. use **progressive search or policy-guided search** as the optimizer
3. make the reward or scoring **topology-aware**
4. evaluate not just accuracy, but also:
   - search cost
   - selected schema stability
   - transferability
   - counterfactual importance by removing selected schemas/faces

And operationally, build `H2S-NN` so that it can run in all three regimes:

- no meta-paths available,
- weak meta-paths available,
- good meta-paths available.

That flexibility is not a side detail. It is part of the practical contribution.

If you want to keep RL in the project, my preferred version is:

> **policy-guided heuristic schema search for weighted 2-cell discovery**

That is much better than plain RL meta-path search.

## 10. Bottom line

Monte-Carlo or reinforcement learning is interesting here because schema discovery is sequential and combinatorial.

But for `H2S-NN`, RL/MCTS should be treated as:

- a useful optimization tool,
- a proposal mechanism,
- or a search accelerator,

not the main novelty.

The real publishable novelty is:

> automatically learning which typed higher-order schemas become 2-cells in the heterogeneous sheaf complex, and therefore learning the topology that defines Hodge diffusion itself.

That is where your paper becomes clearly different from both:

- ordinary meta-path search papers, and
- `HetSheaf`.

## 11. Sources used

- [h2snn_metapath_research_memo_2026-05-06.md](<G:\My Drive\H2S-NN_test\docs\h2snn_metapath_research_memo_2026-05-06.md>)
- `research_paper/A New Meta-path Monte Carlo Tree Search Algorithm for Heterogeneous Graph Neural Networks.pdf`
- `research_paper/HG-search multi-stage search for heterogeneous graph neural networks.pdf`
- `research_paper/Long-range Meta-path Search on Large-scale Heterogeneous Graphs.pdf`
- `research_paper/Rethinking Heterogeneous Graph Representation Learning via Relational Subset Selection.pdf`
- `research_paper/Adaptive meta-path-based neural network architecture search for heterogeneous graphs.pdf`
- `research_paper_reinforce/Single-Agent Policy Tree Search With Guarantees.pdf`
- `research_paper_reinforce/Policy-Guided Heuristic Search with Guarantees.pdf`
- `research_paper_reinforce/Reinforcement Learning of Heuristics with Limited-Horizon Search.pdf`
