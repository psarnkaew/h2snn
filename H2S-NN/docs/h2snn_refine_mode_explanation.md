# How H2S-NN Accepts User-Provided Schemas and Improves Them
## Plain-Language Explanation

---

## The problem this solves

Every existing automatic meta-path selection method has one fixed way of working:
it ignores anything the user knows and discovers everything from scratch.

This is a problem in practice. A researcher studying an academic citation network might say:
*"I think Author → Paper → Author is probably useful. I'm not sure about the others."*

Existing methods cannot use this information at all. They either:
- Require the user to commit fully ("these are the exact paths, no changes") — like HAN, or
- Start completely from scratch as if the user said nothing — like LMSPS, MCTS-HGNN, PMMM.

H2S-NN supports a middle ground: **take the user's suggestion as a starting hint, test it,
keep it if it is good, replace it if something better exists nearby.**

---

## How it works — step by step

### Step 1. The user gives some schema suggestions (optional)

The user can provide a list of meta-path schemas they believe are relevant, for example:

```
User says: try Author → Paper → Author
           and  Author → Paper → Venue → Paper → Author
```

These are the user's best guesses — not guaranteed to be optimal.

### Step 2. The model expands the candidate pool around those suggestions

H2S-NN does not just accept those two schemas. It uses the type graph to automatically
generate **nearby variants** — schemas that are one step longer, one step shorter, or use a
related relation type. For example:

```
User gave:     Author → Paper → Author                        (APA)
Model adds:    Author → Paper → Term → Paper → Author         (APTPA)  ← nearby variant
               Author → Paper → Venue → Paper → Author        (APVPA)  ← nearby variant
               Author → Paper → Author → Paper → Author       (APAPA)  ← one step longer
```

The full candidate pool now contains both the user's schemas and the auto-discovered
neighbors. The user's schemas are not treated as special — they compete equally with all
other candidates.

### Step 3. Every schema in the pool gets a learnable importance weight (α)

Each schema — whether user-provided or auto-discovered — gets one learnable weight α
(alpha) between 0 and 1. A weight close to 1 means "this schema contributes a lot." A
weight close to 0 means "this schema is not useful."

At the start of training, the weights are **initialized from a topology-aware score**,
not from whether the user mentioned the schema. This means a user-provided schema that is
structurally weak starts with a low initial weight, just like any auto-discovered schema
that is structurally weak would.

```
Initial weights (from topology score, before training):
  APA          α = 0.72   ← user gave this, and it scores well
  APTPA        α = 0.61   ← auto-discovered, scores reasonably
  APVPA        α = 0.09   ← auto-discovered, scores low (too many instances, noisy)
  APAPA        α = 0.31   ← user gave this, scores moderate
```

### Step 4. Training adjusts all weights freely — nobody is protected

During training, all α weights are updated by gradient descent together with the rest of
the model. A schema that is genuinely useful will see its α rise. A schema that hurts or
adds nothing will see its α fall toward zero.

**Crucially: the user's schemas have no special protection.** If training discovers that
APA is not actually helpful for this specific dataset and task, its α will drop. If the
auto-discovered APTPA turns out to be far more informative, its α will rise — even though
the user never mentioned it.

```
After training:
  APA          α = 0.55   ← user gave this; still useful, kept
  APTPA        α = 0.81   ← auto-discovered; most useful, kept
  APVPA        α = 0.08   ← effectively dropped (near zero)
  APAPA        α = 0.11   ← user gave this; model decided it adds noise, dropped
```

In this example, the model kept one of the user's two schemas (APA), elevated an
auto-discovered alternative (APTPA), and quietly dropped the other user-provided schema
(APAPA) because it did not help.

### Step 5. Instance-level weights (β) further refine within each kept schema

For every schema that survives (α > threshold), the model also learns a separate weight β
(beta) for every individual path instance within that schema. This means:

- Within APTPA, the specific Author₁ → Paper₇ → Term_X → Paper₂ → Author₂ path gets its
  own β based on how relevant those specific nodes are to the task.
- A noisy or repetitive instance gets a low β and contributes very little.
- An informative instance gets a high β and contributes strongly.

No existing method has this second level of refinement inside each schema.

---

## Why existing methods cannot do this

| Method | Can use user hints? | Protects user schemas? | Can discover better alternatives? |
|---|---|---|---|
| HAN | Yes, but fixed | Yes (rigid) | No |
| LMSPS | No | — | Yes, but from scratch only |
| AMP-HGNAS | No | — | Yes, but from scratch only |
| MCTS-HGNN | No | — | Yes, but from scratch only |
| PMMM | No | — | Yes, but from scratch only |
| **H2S-NN (refine mode)** | **Yes** | **No (they compete freely)** | **Yes, nearby variants** |

The key difference: H2S-NN uses user schemas as a **warm starting point in the search
space**, not as fixed truth and not as ignored information. The user's knowledge helps
focus the initial candidate pool on promising regions, but the model is free to replace
any of those schemas if training evidence supports a better alternative.

---

## The three modes in one sentence each

**Auto mode** (no user input): H2S-NN generates all possible schemas from the type graph
using DFS, scores them with the topology-aware score, and selects a pool — no human input
needed at any step.

**Refine mode** (user provides weak hints): H2S-NN takes the user's schemas, generates
nearby variants, and lets training decide which to keep and which to replace — the user's
schemas compete equally with auto-discovered alternatives.

**Manual mode** (user provides confident schemas): H2S-NN uses the user's schemas directly
as the candidate pool, then learns α and β weights on top of them — the user's knowledge is
respected but still optimized over.

All three modes use the same underlying code. The only difference is how the initial
candidate pool is built.
