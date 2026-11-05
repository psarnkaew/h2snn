# SheafCovPool — How it works, for a newcomer (FINALIZED, 2026-07-22)

> **Method name: `SheafCovPool`.** Spec: [`sheafcovpool_implementation.md`](sheafcovpool_implementation.md) ·
> decisions/history: [`sheafcovpool_design_log.md`](sheafcovpool_design_log.md) · research framing:
> [`sheafcovpool_proposal.html`](sheafcovpool_proposal.html) (see the "FINAL METHOD" card near the top).
>
> **One line:** *SheafCovPool reads out a graph by describing, per molecule, not just the average atom/bond
> state but also how much those states vary — using only quantities that don't depend on each atom's private
> coordinate system.*
>
> This is a **different, simpler method** than the hierarchical "THS-Pool" this file used to describe (grouping
> atoms into super-atoms, zooming out, re-diffusing at a coarser resolution). That hierarchical line was tried
> five different ways and closed as negative on 2026-07-22 — see §5 below for what changed and why. If you're
> looking for the old THS-Pool/Galerkin content, it has been moved to
> [`sheafcovpool_implementation.md`](sheafcovpool_implementation.md) Appendix A as a historical record.

---

## 1. The picture (molecule analogy)

A molecule is a graph: atoms = nodes, bonds = edges. We want one prediction for the **whole molecule**. The
backbone (H2SNN) already gives each atom a state in **its own private coordinate system** (a "stalk") — so raw
atom coordinates from two different atoms aren't directly comparable, only rotation/scale-invariant quantities
built from them are (lengths, inner products).

The simplest way to summarize a molecule is: compute one such invariant number/vector per atom, then **average
it over all atoms**. That's what the existing `GramPool` readout does. It's cheap and safe, but an average
throws away the *shape* of the distribution — a molecule where every atom looks the same and a molecule where
half the atoms are very different from the other half can have the *same average*, even though they're
structurally very different molecules.

**SheafCovPool's idea:** keep the same safe, invariant per-atom descriptor, but summarize it with **two**
statistics instead of one — its (attention-weighted) **average**, and its (attention-weighted) **spread
(covariance)** across the atoms — plus a small learned attention step that lets the model decide which atoms
matter most when computing both.

---

## 2. What "invariant per-atom descriptor" means (concrete)

For atom $v$, take three existing views the backbone already computes — its node state $x_v$, its raw encoded
input $h_v$, and its incident-bond signal pooled onto the atom — and stack them into a small matrix
$H_v \in \mathbb R^{d\times 3}$ ($d$ = stalk dimension, 3 = number of stacked views). The **Gram matrix**

$$g_v = H_v^\top H_v \in \mathbb R^{3\times 3}$$

is built only from inner products of $H_v$'s columns, so it does **not change** if you rotate atom $v$'s private
coordinate system (that's exactly what "gauge-invariant" means here). Flatten it to a 9-dimensional vector —
this is the same descriptor `GramPool` already uses. Bonds get an analogous descriptor from their own three
views (disagreement views instead of raw states).

**One subtlety that took a bug to notice.** Inner products only mean anything if all three columns are written
in the *same* coordinate system. Columns 1 and 2 are: both are quantities of atom $v$, in $v$'s own frame. But
the third column comes from the bonds, and **every bond has its own private frame too** — so adding raw bond
vectors together and then taking inner products against atom $v$'s vectors is comparing numbers written in
different languages. It is not just non-invariant; it is not a meaningful operation at all.

The fix is to translate each bond's signal into atom $v$'s frame *before* averaging, using the restriction map
$F$ that the model already learned for exactly that bond–atom pair (its transpose, $F^\top$, translates back the
other way):

$$\widetilde s_v = \frac{1}{d_v}\sum_{\text{bonds } e \text{ at } v} F_{v \unlhd e}^\top s_e .$$

Now all three columns really are in $v$'s frame, and rotating that frame rotates all three the same way — which
is precisely what makes $g_v$ unchanged. The same reasoning is why the *bond* descriptor needed no fix: its
three views were already all written in that bond's own frame.

---

## 3. The two new ingredients, step by step

1. **Learn a smaller, more useful projection.** Pass the 9-dim Gram descriptor through a small MLP to get an
   8-dim feature $u_v$ per atom (and separately for bonds).
2. **Score each atom with gauge-invariant attention.** A second small MLP turns $u_v$ into a single importance
   score; softmax those scores *within each molecule* so they sum to 1 — this is the same kind of attention
   SheafPool uses, but applied directly to an already-invariant descriptor, with no extra machinery needed.
3. **Pool two moments, not one.**
   - **Mean:** $\mu = \sum_v \alpha_v u_v$ — same idea as `GramPool`'s average, just attention-weighted instead
     of uniform.
   - **Covariance:** $\Sigma = \sum_v \alpha_v (u_v-\mu)(u_v-\mu)^\top$ — a new ingredient. It measures how much
     the atoms' descriptors *disagree with each other* around the mean. Two molecules with the same average atom
     descriptor but very different spreads now get different summaries.
4. **Concatenate and classify.** Take $\mu$ and the upper triangle of $\Sigma$ (avoids double-counting the
   symmetric off-diagonal entries) for both atoms and bonds, concatenate everything, and feed it to a small
   classifier MLP. Trained with plain cross-entropy — nothing else.

**Worked intuition:** picture two 6-atom molecules with the same *average* ring-energy per atom. In molecule A
every atom has nearly that average energy (low spread). In molecule B, half the atoms are near zero and half are
double the average (high spread, same mean). `GramPool`'s mean-only summary cannot tell these apart. SheafCovPool's
$\Sigma$ term can, because it directly measures that spread.

---

## 4. Why not just use SheafPool, which already does something fancier?

SheafPool (HetSheaf) also does attention pooling, but as its *first* step it **whitens** each atom's stacked
matrix — $Z_v = (H_vH_v^\top)^{-1/2}H_v$ — to remove basis ambiguity beyond simple rotations. Whitening needs
enough *columns* (channels) in $H_v$ to be meaningful; H2SNN only stacks 3 views, so the whitened result nearly
collapses to a constant matrix regardless of the input, and the model can't learn anything useful from it. This
was confirmed twice experimentally: SheafPool scores ~0.26 macro-F1 on MUTAG, at/below the majority-class
baseline, both in an earlier sweep and again in the final gate below.

SheafCovPool never whitens — it works directly with the (already invariant, no-whitening-needed) Gram
descriptor `GramPool` uses, so it doesn't inherit SheafPool's failure mode, while still adding a genuinely new
ingredient (the covariance) that `GramPool` lacks.

---

## 5. What changed since the last version of this document (history, briefly)

This project originally set out to build a **hierarchical** method (grouping atoms into super-atoms and
re-diffusing at a coarser resolution, named "S²Pool-H2" then "THS-Pool"). Five variants of that idea were built
and tested on real TU benchmark runs:

| Attempt | Idea | Result |
|---|---|---|
| Terminal higher-order readout | read `s*`/`δ1s*` (edge/ring signal), not just nodes | tied node-only pooling |
| Relearn hierarchy | re-run the whole backbone on a smaller, pooled graph | not real coarsening; noisy weak-positive; 20× slower |
| Geometry injection | keep fine bond-maps, apply to pooled atoms | broke the energy bookkeeping → hurt performance |
| Energy-preserving (Galerkin) coarsening | build the coarse operator so it exactly preserves fine energy | looked like a clean +0.14 macro-F1 win on MUTAG... |
| ...same, after fixing a numerical bug | a `1/std` divide-by-near-zero bug in per-graph normalization was inflating that result | **the win reversed to a tie** once fixed |
| Spectral heat-kernel descriptor | add a diffusion-decay signature as an extra invariant feature | model learned to fully ignore it (identical outputs with/without) |

None of it robustly beat `GramPool`. **SheafCovPool is the method that replaced all of this** — much simpler
(no hierarchy, no coarse operator, no re-running the backbone), and its own gate result is honest about being a
tie-or-modest-improvement over `GramPool`, not a dramatic win:

**Final gate (MUTAG + ENZYMES, 3 seeds, 40 epochs, macro-F1):**

| Dataset | SheafCovPool | GramPool | SheafPool |
|---|---|---|---|
| MUTAG | 0.596 ± 0.050 | 0.595 ± 0.081 | 0.260 ± 0.014 |
| ENZYMES | 0.236 ± 0.144 | 0.166 ± 0.088 | 0.048 ± 0.000 |

Full details of every abandoned attempt: `sheafcovpool_design_log.md` §0 and its changelog (§pm8–§pm11).

---

## 6. Positioning (plausible, honest)

| | invariant, no whitening | adds attention | adds across-node covariance |
|---|---|---|---|
| GramPool | ✓ | ✗ | ✗ |
| SheafPool | ✗ (whitens; collapses on H2SNN) | ✓ | ✗ |
| **SheafCovPool** | **✓** | **✓** | **✓** |

**Claim to make:** SheafCovPool is the readout that gets attention and second-order (distributional)
expressivity onto H2SNN's low-channel stalks *without* SheafPool's whitening failure mode. **Honest caveat:**
on TU-scale graphs (small, local) this mostly shows up as a tie with `GramPool` — the ingredient it adds
(distributional/covariance structure) most plausibly matters on larger or more heterogeneous graphs than TU
provides, which is also why the design log recommends Peptides-func/LRGB as the place to actually test for a
gain, not TU alone.
