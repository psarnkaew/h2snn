# Why Existing Automatic Meta-Path Selection Methods Are Not Enough
## Plain-Language Explanation for Each Problem Row

**Audience:** A professor who knows graph neural networks but has not read this project.

**Scope:** This document explains only the **automatic meta-path selection** contribution of
H2S-NN. It compares against other methods that also do automatic selection:
LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM, HG-Search, and PM-HGNN.

---

## What automatic meta-path selection is trying to do

In a heterogeneous graph, you have multiple node types (Author, Paper, Venue) and multiple
relation types (writes, cites, published-at). A meta-path is a sequence of these types that
defines a meaningful route through the graph — for example, Author → Paper → Author (two
authors who co-authored a paper).

Automatic selection means: instead of a human deciding which routes matter, the model figures
this out on its own. All existing methods in this comparison can do that basic step.

**The question is: what problems do they still have after automating that step?**

The seven rows in `h2snn_selection_comparison.csv` each describe one remaining problem.

---

## Row 1 — Most methods can only say "use this schema" or "drop it" — they cannot say "use it a little"

**The problem in plain words:**
Some meta-paths are very important. Some are a little useful. Some are harmful. A good
selection method should express this as a spectrum, not a binary decision. But MCTS-HGNN and
HG-Search make hard discrete choices: a schema either survives the search or it does not.
There is no in-between.

**What "soft learnable weights" means:**
Instead of a hard keep/drop decision, the model learns a number between 0 and 1 for each
schema. A schema with weight 0.9 contributes a lot; a schema with weight 0.1 contributes a
little. This weight is learned automatically during training, not set by a human.

**Who has it:**
LMSPS, AMP-HGNAS, and PMMM all use soft learnable weights (α parameters with softmax or
Gumbel-softmax). MCTS-HGNN and HG-Search use discrete selection — a schema either wins the
search or it does not. PM-HGNN generates one path per node via RL policy — there are no
schema-level weights at all.

**Why soft weights matter:**
A binary decision throws away information. If two schemas are almost equally useful, keeping
one and dropping the other loses the partial contribution of the second. Soft weights let
both contribute proportionally.

---

## Row 2 — All existing methods treat every path of the same type identically

**The problem in plain words:**
Suppose the model selects the schema "Author → Paper → Author." This schema matches thousands
of concrete path instances in the actual graph — every pair of co-authors connected through
every paper. But not all of these are equally useful.

- Author₁ → Paper₇ → Author₂: two leading researchers who co-authored a landmark paper.
  Very informative.
- Author₃ → Paper₁₂ → Author₄: two PhD students who accidentally appeared on the same
  workshop abstract. Probably noise.

All existing methods — LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM, HG-Search, PM-HGNN — treat
these two instances exactly the same once a schema is selected. There is no mechanism to
express that one specific path instance matters more than another.

**What H2S-NN does differently:**
H2S-NN learns a separate importance weight (β) for every individual path instance. The model
can learn that the landmark co-authorship matters a lot and the workshop coincidence matters
very little — automatically, without any human input.

**Why this matters:**
Schema-level selection only tells you which pattern types are useful in general. But in a
real graph, individual instances of the same pattern vary enormously in informativeness.
Ignoring this variation means the model is treating noise and signal as equals inside every
selected schema.

---

## Row 3 — No existing method combines schema-level and instance-level selection together

**The problem in plain words:**
Even if a method decides "Author → Paper → Author is useful" (schema level), it still has no
way to then ask "but which specific Author → Paper → Author triples are the most useful ones?"
(instance level). These are two separate questions, and existing methods can only answer the
first one.

**What H2S-NN does differently:**
H2S-NN has two sets of weights that are optimized at the same time:
- α (alpha): how important is each schema type overall?
- β (beta): how important is each specific path instance within that schema?

Both are learned end-to-end in a single training run. No existing method does both together.

**Why this matters:**
If you only select schema types but treat all instances equally, you are doing half the job.
You found the right category of roads to drive on, but you are still going down all roads in
that category equally — including the ones full of potholes. H2S-NN also learns which
specific roads to prefer.

---

## Row 4 — Most methods select schemas independently without checking if they overlap

**The problem in plain words:**
Imagine selecting three meta-paths: Author→Paper→Author, Author→Paper→Paper→Author, and
Author→Paper→Conference→Paper→Author. All three essentially connect pairs of authors through
papers. They carry nearly the same information. Selecting all three wastes computation and
causes the model to count the same signal three times.

Most methods — LMSPS, MCTS-HGNN, PMMM, HG-Search, PM-HGNN — evaluate each schema
independently. They do not check whether the schemas they choose are redundant with each
other. The model may end up with ten schemas that all say the same thing.

**Who handles this:**
AMP-HGNAS explicitly models dependency between schemas and penalises highly overlapping ones.
This is one of AMP-HGNAS's main contributions. H2S-NN also includes redundancy control.

**Why this matters:**
Redundant schemas do not add information — they just add cost and can create gradient
interference during training (many correlated signals all pushing the model in the same
direction, preventing it from learning fine-grained distinctions).

---

## Row 5 — Every existing method has a separate "search phase" followed by a "retrain phase"

**The problem in plain words:**
All existing automatic selection methods work in two stages:

1. **Search stage:** run the selection process to find good schemas. This uses a simplified
   or proxy version of the model.
2. **Retrain stage:** take the selected schemas and train the real model from scratch.

The problem is that the schemas that looked good during the search stage may not be the best
ones for the real model trained in the retrain stage. The two stages are not aligned — what
the search cares about and what the final model cares about are not the same thing. This is
called **search-evaluation mismatch**, and it is a well-known weakness in architecture search
literature. LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM, and HG-Search all have this problem.

**What H2S-NN does differently:**
Schema selection and model training happen in a single continuous training run. The schema
weights (α and β) and the model weights are updated together by the same gradient at every
step. There is no stage boundary, so there is no mismatch.

**Why this matters:**
Mismatch means extra effort is wasted. You optimise for something in the search stage and
then throw it away when you retrain. Single-run training avoids this waste entirely and
produces schemas that are directly optimised for the final model they will be used in.

---

## Row 6 — No existing method can use prior knowledge as a starting point and then improve it

**The problem in plain words:**
Suppose a domain expert says: "I think Author→Paper→Author and Paper→Venue→Paper are
probably useful for this dataset — I'm not 100% sure, but these are my best guesses."

Every existing automatic selection method ignores this information entirely. They either:
- Start their search from scratch as if the expert said nothing, or
- Accept the given schemas as fixed truth and never question them (like HAN does).

There is no method that says: "OK, I'll start with your two suggested schemas, test whether
they are actually useful, replace them if something better exists nearby, and add more schemas
that complement them."

**What H2S-NN does differently:**
H2S-NN supports three modes in a single framework:
- **Auto mode:** no schemas given — discover everything automatically from the graph.
- **Refinement mode:** user gives some schemas as hints — the model tests them, keeps the
  useful ones, replaces the weak ones, and adds better alternatives.
- **Manual mode:** user gives well-chosen schemas — the model confirms and fine-tunes them.

**Why this matters:**
In a real research workflow, domain experts often have partial knowledge. A method that can
incorporate that knowledge as a soft starting point — rather than ignoring it or blindly
trusting it — is far more practical and efficient.

---

## Row 7 — All existing methods judge schema quality by training a model and measuring accuracy

**The problem in plain words:**
How do existing methods decide whether a schema is good? They train a model using that schema
and measure validation accuracy. If accuracy goes up, the schema is good. If not, it is bad.

This has two problems:

**Problem A — It is expensive.**
Every time you want to evaluate a candidate schema, you need to run (or partially run) a
model training. With dozens or hundreds of candidate schemas, this becomes very costly.
MCTS-HGNN's tree search suffers especially from this — each MCTS rollout requires evaluating
a candidate with a real model forward pass or a proxy MLP.

**Problem B — It is blind to structure.**
Accuracy on a validation set measures one thing: does this schema help the final prediction
score? But a schema might be structurally valuable — for example, it connects nodes that are
currently inconsistent with each other in the graph, and fixing that inconsistency would
improve the model in ways that do not immediately show up as an accuracy jump. The accuracy
signal misses this.

**What H2S-NN does differently:**
H2S-NN also uses graph structural signals to score schemas — for example, whether a schema
produces path instances that reduce inconsistency in the graph structure. This is cheaper to
compute (no full model training required) and captures structural usefulness that accuracy
alone cannot see.

**Why this matters:**
A schema that is structurally valuable but slow to improve accuracy would be incorrectly
discarded by every existing method. H2S-NN can recognise and keep it.

---

## One-paragraph summary for a professor

All six existing automatic selection methods (LMSPS, AMP-HGNAS, MCTS-HGNN, PMMM, HG-Search,
PM-HGNN) successfully remove the need for manual meta-path design. However, they all share
the same underlying limitation: they select which schema *types* to use, then apply all
concrete instances of those schemas equally, using only downstream accuracy to judge quality,
in a two-stage process that creates a mismatch between what is optimised during search and
what the final model actually needs.

H2S-NN's automatic selection addresses all five of these limitations simultaneously: it learns
both schema-level and instance-level importance weights in a single differentiable training
run, penalises redundancy across selected schemas, can incorporate and refine user-provided
domain knowledge, and scores schemas using graph structural signals — not just accuracy.
