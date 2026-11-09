# J5.0 Pre-registration: link-prediction features (J5.1–J5.3), validation files (J5.4), negatives (J5.5), freebase_bb H_edge (J5.6)
Written 2026-11-10, before any J5 computation. Authority: spec v1.9 §6 (Link prediction, Node features), changelog F12–F16. Evidence base: `docs/H2S-NN_partJ_report.md` §10, `j_independent_review.md`, `j_review_response.md`, `j44_record.md`, `j40_data_checks_output.txt`.

## What was looked at before this file (descriptive only, no rule tested)
The node types, feature dimensions and value sets of the three derived tasks (`data/lp_derived/<task>/node.dat`, byte copies of the HGB releases):

| Task (target) | Type (dim, values) | On the target relation? |
|---|---|---|
| acm_pa (rel 2 paper→author) | paper (1902, 0/1); author (1902, 0/1); subject (1902, 0/1); term (none) | paper, author |
| dblp_ap (rel 0 author→paper) | author (334, 0/1, 39 all-zero rows); paper (4231, counts 0–5); term (50, real); venue (none) | author, paper |
| imdb_ma (rel 2 movie→actor) | movie (3489, real); director (3341, fractions and integers ≤ 5); actor (3341, same); keyword (none) | movie, actor |

- IMDB actor values are printed in full Python precision (`0.3333333333333333`, `0.041666666666666664`), consistent with a mean over neighbours.
- The keyword names are in `info.dat` (`"keyword": {"0": "death of wife", …}`).
- PubMed: `link.dat.valid` has rows of relations 0–9; the target is relation 2 (1,942 rows). LastFM: relations 0–2; the target is relation 0 (7,301 rows).
- Known from J4: ACM author = OR of its papers (5,959/5,959; `j40_data_checks_output.txt`). DBLP authors and IMDB actors with identical target-relation neighbour sets have identical features (150/150; 1,563/1,563; random pairs 2/5,000 and 15/5,000).
- Already measured by the reviewer (reused, not recomputed, per the author): acm_pa, binary containment rule, true pair strictly first in 895/895 validation and 995/995 test lists.

## J5.1 Candidate construction rules (fixed here)
**Scope.**
- Every node type with features in each derived task.
- A type with no fact of the target relation cannot be a function of neighbours through it: ACM subject, DBLP term, IMDB director. Such a type is reported as "not on the target relation, not affected", with no rule test.
- Both endpoint types of the target relation are tested, each against the other endpoint.

**Neighbourhood.** N(x) = the other endpoint's nodes linked to x by the target relation in the **release** `link.dat`, with all facts (training, validation and test), because the release's features were built from all of them.

**Vocabulary sources.** g(y) is the vector a neighbour y contributes:
- **V1 (own features):** g(y) = y's feature vector. Also as presence, [g(y) ≠ 0], for count features.
- **V2 (third-type links):** g(y) = y's indicator row over the nodes of a third type W, through a non-target relation. W is: ACM subject or term; DBLP term or venue; IMDB keyword or director.
- **V3 (token vocabulary, IMDB only):** g(y) = the count vector of whitespace-separated tokens over the names (`info.dat`) of y's keywords. Also as presence.
  - Tokens are taken as written. Case-folded tokens are a second variant.

**Aggregators.** Over y ∈ N(x): OR (of g ≠ 0), SUM, MEAN (SUM / |N(x)|; zero vector when N(x) is empty) and MAX.

**Matching.**
- Where g already lives in x's feature space (V1 with equal dimension), the rule is the identity column map.
- Otherwise x's column j is matched to the vocabulary column v whose aggregated column equals x's column j on every node of x's type: the "mapped vocabulary" of the card.
- If several v match a column j that is not all zero, the mapping is unambiguous only if those vocabulary columns are identical over all neighbour nodes; then they are interchangeable on any sub-graph. Otherwise the rule is **ambiguous** and is not counted as exact.
- An all-zero column of x maps to zero.

**Equality.**
- 0/1 and integer features: exact float64 equality of the parsed values.
- Fractional values (MEAN): |difference| ≤ 1e-12 per entry. The release's summation order is unknown, so this allows only rounding; the maximum difference is reported.

**Reported per type:** the rule, and the number of nodes for which it holds exactly with the chosen mapping (all columns equal).

**Exact rule** = holds on **every** node of the type. **Affected type** = one of:
- (i) an exact rule exists;
- (ii) nodes with identical target-relation neighbour sets have identical features in ≥ 95 % of pairs, against ≤ 5 % for random pairs;
- (iii) some candidate holds on > 50 % of its nodes.

**Expected:**
- ACM author: OR of V1, 5,959/5,959 (the known case).
- DBLP author: an exact OR rule over a mapped vocabulary (V1 presence or V2 terms), because the features are binary keyword indicators. Uncertain; the J4 identical-set test says the features are a function of the paper set.
- IMDB actor: MEAN of a token-count vocabulary (V3), from the value pattern. Uncertain.
- No endpoint type whose features are its own (ACM paper, DBLP paper, IMDB movie) has a rule.

**Wrong if:** ACM author does not reach 5,959/5,959 (this would mean the J5.1 code is wrong, since J4.0 found it); or a paper or movie type matches a rule.

**If an affected type has no exact rule:** its features are dropped in the LP task (F12). Its J5.2 scorer uses the candidate with the most exact nodes.

## J5.2 Leak measure (fixed here)
**Lists.** The existing HeaRT validation and test lists (`<task>/heart/heart_{valid,test}_{pos,samples}.npy`, 500 negatives per positive). The positive is scored together with its own 500 negatives.

**Features-only scorer.** For a pair (x, y), with x of an affected type with rule (A, g, mapping m) and y on the other endpoint:
- s(x, y) = |supp(g_m(y)) ∩ supp(f(x))| / |supp(g_m(y))|, with s = 0 when supp(g_m(y)) is empty.
- g_m(y) is y's contribution mapped onto x's columns; f(x) is x's stored feature.
- In words: the share of the candidate's contribution under the rule that is already contained in the kept node's feature. This is the same pair function for both corrupted sides, since every HeaRT pair has one endpoint of each type.
- For acm_pa also the **cosine** of the stored features, as in the review (C2 shared space).

**Rank and metrics.** Rank as HeaRT's evaluator (`vendor/HeaRT/benchmarking/evalutors.py:108-112`): 0.5 · (#neg ≥ pos + #neg > pos) + 1. MRR = mean of 1/rank; Hits@10 = share with rank ≤ 10. An all-tie list has rank 251, so MRR = 1/251 = 0.00398 and Hits@10 = 0.

**Before the fix.**
- dblp_ap and imdb_ma are measured.
- acm_pa: the reviewer's numbers are reused. Binary containment ranks the true pair strictly first in 895/895 and 995/995 lists. Hence fractional containment s gives the positive s = 1 and every negative s < 1 (a negative with s = 1 would have been contained under the binary rule), so MRR = Hits@10 = 1.000 (DERIVED). acm_pa's cosine before the fix is not measured.

**Expected before the fix:**
- acm_pa MRR 1.000 (derived).
- dblp_ap and imdb_ma MRR ≥ 0.9: the rule makes the positive's contribution fully contained, while hard negatives are mostly not.

**Wrong if:** the scorer leaves dblp_ap or imdb_ma near the all-tie level (< 0.1). That would mean the leak is not exploitable through this scorer, and it is reported as such.

## J5.3 Feature-free control and tolerance (fixed here)
**Control (Ctrl-G, "training-graph control").** The same scorer, with f(x) replaced by the rule applied to x's **training** neighbours.
- The training neighbours are computed in the measurement script, independently of the rebuild script: the task's `link.dat` target rows minus the `link.dat.valid` rows. Test rows are not in `link.dat`.
- Ctrl-G reads **no stored feature of the affected type**. It is the information any model gets from the training graph plus the other endpoint's own features.

**Why this control, not a constant.** After the fix, a held-out paper can still resemble its author's training papers. That is legitimate content signal, which every model can reach by message passing over training facts. A constant or random control would call that signal a leak. Ctrl-G separates what the stored features add beyond the training graph: exactly the leak.

**Reference only, not a gate:** the all-tie level (MRR 0.00398, Hits@10 0). For a **dropped** type the scorer has no f(x) and is all-tie by construction; its control is that level.

**Gate (after the fix, validation and test, every affected type):**
- |MRR − MRR_Ctrl-G| ≤ 0.005 and |Hits@10 − Hits@10_Ctrl-G| ≤ 0.005.
- Expected difference exactly 0, because the rebuilt features are the rule on training facts.
- A larger difference means the files still carry information beyond the training graph: a **stop condition** (report, do not tune).

**Also expected:** the after-fix MRR is well below the before-fix MRR. Predicted for acm_pa: < 0.5. This is a prediction, not a gate.

**Bitwise check:** for every node of a rebuilt type, the stored feature (parsed, and as the loader's float32 tensor) equals the rule applied to its training neighbours, computed by a separate code path. A node with no training fact gets the zero vector.

**Mutation checks:**
- Rebuild from all facts instead of training facts → the bitwise check fails, and Ctrl-G is beaten.
- Leave one held-out fact in → caught.

**Text format.** Rebuilt rows are written in the release's own format for that type:
- integers, as `1`/`0` (ACM, DBLP);
- `repr(float)` (IMDB, where `0.0` and `1.0` are written as such).

The task's other rows stay byte-identical, which is checked.

**Where.** Only `data/lp_derived/<task>/node.dat`. The `data/acm`, `dblp`, `imdb`, `freebase` releases are md5-checked unchanged before and after. Each task records what was done in `feature_rebuild.json`.

## J5.4 Validation files, target only
**Change.** `data/PubMed/link.dat.valid` and `data/LastFM/link.dat.valid` keep only their target relation's rows (PubMed relation 2, 1,942 rows; LastFM relation 0, 7,301 rows), byte-identical to today's rows of that relation. The target relations have no stored partner relation.

**Expected:**
- The loader's target train, validation and test positives are identical tensors to before.
- Every non-target relation has all its `link.dat` facts in the message graph: the count equals the `link.dat` rows, where today about 10 % is carved off.

**Wrong if:** a target tensor differs, or a non-target relation is incomplete.

**Mutation:** the file is ignored, so the carve applies → the loader test fails.

## J5.5 HeaRT regeneration
**Inputs and script.**
- acm_pa: author features → C2 cosine.
- PubMed and LastFM: message graph → CN, PPR and L3.
- The same script `scripts/lp/build_heart_negatives.py`, with the same parameters (500, the agg min, PPR 0.15/5e-5) and seed 42.
- dblp_ap, imdb_ma and freebase_bb: unchanged inputs (C2 is not used there: no shared feature space, or features not rebuilt). Their file md5s are shown unchanged.

**Expected:** all J4.4 negative tests pass on the new files; no pool is too small. Random-fill share per task (F16): acm_pa may change from 0 %; PubMed and LastFM about as before.

**Features-only scorer on the new acm_pa lists:** at the Ctrl-G level (same tolerance, 0.005).

**Wrong if:** a J4.4 property fails, or a stop condition of J4.4 occurs.

## J5.6 freebase_bb task-level H_edge (F15)
**No new computation.** The reported value becomes the loop-free one already in the review evidence (`j_review_task_h_edge_without_loops.txt`): fill seed 0 0.8254 (seeds 0.8254–0.8291), labelled-only 0.3529.
- `results.json/md` are updated; chapter 3 only if it shows the value.

**Wrong if:** the stored evidence lacks a value. Then it is recomputed, and that is stated.

## Assumptions and routes
- **Information-route audit (LP files):** after the fix, the held-out facts reach a model through no route:
  - the message graph (J4 guards);
  - endpoint features (rebuilt from training facts, or dropped);
  - the HeaRT lists (training graph and features from the rebuilt files).
- The non-endpoint types (ACM subject and term, DBLP term and venue, IMDB director and keyword) carry no target-relation information, by the scope argument above.
- No `THEORY-CONSTRAINT` is involved: this is data hygiene.
