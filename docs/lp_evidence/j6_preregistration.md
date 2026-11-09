# J6.0 Pre-registration: one feature rule (J6.1), one leak audit per task (J6.2), loader guard (J6.3), negatives (J6.4), cold-start count (J6.5)
Written 2026-11-10, before any J6 computation. Authority: spec v1.10 F17–F20, §6 (Link prediction, Node features). Evidence base: `H2S-NN_partJ_report.md` §11, `j5_independent_review.md`, `j5_review_response.md`, `j51_record.md`.

## Looked at before this file (descriptive only; no probe, no score)
- **Feature presence per task:**
  - PubMed: 200-dim features on all 4 types; no `info.dat`, so the attributes are unnamed.
  - LastFM and freebase_bb: no features on any type.
- **IMDB `info.dat` "Attributes":** columns 0–15 are named numeric attributes (attribute *i* is column *i*):
  - **2** `director_facebook_likes`, **3** `actor_1_facebook_likes`, **4** `actor_3_facebook_likes`, **7** `cast_total_facebook_likes`, **12** `actor_2_facebook_likes`;
  - 16:19 color, 19:67 language, 67:132 country, 132:148 content_rating;
  - 148:3489 keyword-token counts.
- **Non-binary columns (more than 2 distinct values, as text) of the endpoint types:**
  - IMDB movie: 690 of 3,489 (columns 0–15 and 674 token columns);
  - DBLP paper: 297 of 4,231 (word counts);
  - PubMed disease: 200 of 200;
  - ACM paper: 0 of 1,902.

## J6.1 The rule (F17), LP task files only
Starting from the HGB release `node.dat` (so the J5 rebuilt values disappear):
- acm_pa: remove the author block (type 1).
- dblp_ap: remove the author block (type 0).
- imdb_ma: remove the actor block (type 2), and movie columns 3, 4, 7 and 12 (3,489 → 3,485 columns; the order of the rest is kept).
- freebase_bb: no features; nothing removed (still recorded, so the J6.3 guard covers it).

A removed block becomes an empty fourth column, as the release writes a featureless type. The script `scripts/lp/apply_feature_rule.py` writes `<task>/feature_rule.json`, with the removed types and columns (and their names), the resulting widths and the `node.dat` md5. J5's stale `feature_rebuild.json` files are deleted. `rebuild_lp_features.py` stays in the repo, marked "superseded by F17", and is not used.

**Gate:**
- the removed blocks and columns are absent;
- every other row is byte-identical to the release;
- every kept movie value is equal, as text, to the release's value in the same column;
- the node-classification releases are unchanged (md5).

## J6.2 The audit (F18): design and pass criterion
### (i) Attribute inventory
For both endpoint types of each of the 6 tasks, every named attribute (or unnamed block) is classified as **own**, or **about the other endpoint / computed from the target relation**, with its source:
- `info.dat` names;
- J5.1 results (rules);
- the dataset documentation (PubMed: the HNE README, cited if readable, else UNVERIFIED).

**Pass:** no attribute classified as "about the other endpoint / target relation" remains after J6.1.

### (ii) Fingerprint probe (non-binary columns of both endpoint types, after J6.1)
- **Column families:**
  - each **named** non-binary attribute is one family. IMDB movie: columns 0, 1, 2, 5, 6, 8, 9, 10, 11, 13, 14 and 15 after removal.
  - each **unnamed block** is one family, over its non-binary columns: IMDB movie token counts, DBLP paper word counts, PubMed disease 200-dim.
  - ACM, LastFM and freebase_bb have no non-binary feature column. Their probe has no family, which is recorded, not counted as evidence.
- **Fingerprint of node x in family F (cap c):** R_F,c(x) = the set of (column, exact text value) pairs of x in F, where:
  - the value is not a literal zero (`0`, `0.0`);
  - the value occurs in at most c nodes of x's type in that column.
  - c ∈ {5, 50, ∞}; all three are run.
- **Probe score** on a HeaRT pair (u, v), with N_tr = training neighbours through the target relation (`link.dat` minus `link.dat.valid`; test facts are not in `link.dat`):
  - **cross-type**, x the endpoint carrying F and y the other: s(x, y) = |R(x) ∩ ∪_{z ∈ N_tr(y)} R(z)|, i.e. x shares rare exact values with y's training partners. This is the J5 reviewer's probe, generalised.
  - **same-type** (PubMed): s(u, v) = shared(u, N_tr(v)) + shared(v, N_tr(u)).
- **Ranking:** on the stored HeaRT validation and test lists, with HeaRT's rank (mean of the optimistic and pessimistic ranks); MRR and Hits@10.
- **Ctrl-G (training-graph control), feature-free.** Each probe family recodes "x is similar to y's training partners". The graph's version of that is a typed length-3 path count:
  - For each relation r incident to x's type, with r taken from the training message graph (the target's training facts and all non-target facts), g_r(x, y) = Σ_{z ∈ N_tr(y)} |N_r(x) ∩ N_r(z)|. Same-type: symmetrised as above.
  - r includes the target relation itself (then g is the bipartite L3 count, undirected and unnormalised).
  - **Ctrl-G per split = the best r** (highest MRR; its Hits@10 is taken with it).
  - **Why this control:** a feature family that only re-encodes a graph relation (e.g. `director_facebook_likes` ≈ the director's identity) can be recovered from that relation by any graph model, and should not count as a leak. A family that identifies the held-out partner (the actor-likes columns) has no training-graph equivalent and should beat every r.
- **Pass criterion (per task):** for every family F, every cap c and both splits, MRR_F,c − MRR_Ctrl-G ≤ **0.01** and Hits@10_F,c − Hits@10_Ctrl-G ≤ **0.02**.
  - The tolerance is fixed below the J5 leak's margin over its control (+0.04–0.06 MRR, +0.10–0.20 Hits@10).
  - **Any excess is a FAIL:** stop, report the family and columns, remove nothing.
- **Power check (positive control, calibration of this instrument).** This is not a re-measurement of the J5 leak's size: the audit is useless unless it detects a known leak.
  - **Synthetic:** a unit test injects a column whose value is the other endpoint's id-derived number (an attribute of the partner) → FAIL. An own random column → PASS.
  - **Real:** the same probe on the **release** movie columns 3, 4, 7 and 12 (the J5 leak), against the same Ctrl-G on the imdb_ma lists, must FAIL the criterion.
  - **If the real positive control passes, the audit is underpowered.** Stop and report; do not tune.
- **Expected:**
  - the positive control fails;
  - every real family passes, including `director_facebook_likes`, which should be close to the director-path control;
  - the PubMed embeddings pass (exact shared values should be rare).
  - DBLP word counts and IMDB token counts are content similarity, expected near their term or keyword path control. **Uncertain:** if one exceeds the tolerance, that is a stop, reported for the author.

## J6.3 Loader guard (F20)
**Change** in `src/data/link_prediction_dataset.py`: for every derived task (acm_pa, dblp_ap, imdb_ma, freebase_bb), the loader reads `<task>/feature_rule.json` and refuses to load when:
- the file is missing;
- the `node.dat` md5 differs from the recorded one;
- a removed type carries features;
- a type's width differs from the recorded width.

**Tests:**
- the rule-applied toy task loads;
- missing record → refused;
- a re-run of `build_derived_lp.py` (release node.dat restored) → refused;
- a hand-edited value → refused.

The existing toy tests that load derived tasks apply the rule after building. **Mutation:** guard removed → the re-run test fails.

## J6.4 Negatives
- `build_heart_negatives.py`: acm_pa gets `features=False`, since its author features are removed, so C2 no longer applies; PubMed is the only task with C2.
- **Regenerate acm_pa only.** The other five are rebuilt into a scratch subfolder and must be md5-identical to the kept files:
  - imdb_ma's movie columns changed, but imdb_ma has no C2;
  - PubMed's inputs did not change.
- **Gate:** the J4.4 checks on the new acm_pa files, and the J4.4 test module.
- **Expected:** acm_pa random fill stays about 23 %. Cold authors already score 0 everywhere, and the cosine of the other positives disappears, so the share may rise.

## J6.5 Cold-start count (F19)
- **Cold positive:** at least one endpoint has **no training fact of the target relation** (`link.dat` target rows minus `link.dat.valid`). Counted per task and split, with the side (source, destination, both).
- **Random-fill share:** from each `heart/meta.json`.
- **Expected:**
  - acm_pa test: at least the 461 author-cold positives (J5.5), plus some of the 73 paper-cold ones;
  - same-type tasks: few.
- A count that differs from the expectation is recorded, not a stop.

## Routes (information-route audit, LP files after J6)
- **Message graph:** the J4 guards.
- **Endpoint features:** the F17 removals, then the J6.2 audit.
- **HeaRT lists:** the training graph, plus C2 only on PubMed (own features).

No `THEORY-CONSTRAINT` is involved.
