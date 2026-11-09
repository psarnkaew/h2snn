# J5.1 Which LP endpoint features encode the target relation (spec v1.9 F12)
2026-11-10. Candidates, equality and the "affected" criterion are those of `j5_preregistration.md` (J5.1), fixed before this run.

**Command (VERIFIED-RUN):** `python scripts/lp/lp_feature_rules.py --tasks acm_pa dblp_ap imdb_ma --json-out ../docs/lp_evidence/j51_feature_rules.json`, run from `H2S-NN/`, CPU.
- Output: `j51_feature_rules_output.txt`.
- Neighbourhoods are those of the release `link.dat` (all facts); the release md5s match `split_meta.json`.

| Task | Type (dim) | On target? | Identical neighbour set → identical features (random pairs) | Best rule: nodes exact | Decision |
|---|---|---|---|---|---|
| acm_pa | paper (1902) | yes | 0/102 (0/5,000) | none; V1/or 531/3,025, papers whose authors have that one paper | not affected |
| acm_pa | **author** (1902) | yes | 2,365/2,365 (0/5,000) | **OR of its papers' features, identity map: 5,959/5,959**. Also exact: MAX, presence, and OR over the papers' terms (V2:r6, i.e. paper features = term indicators) | **rebuild** |
| acm_pa | subject (1902) | no | — | — | not affected |
| dblp_ap | **author** (334) | yes | 150/150 (2/5,000) | **no exact rule**: OR over the papers' terms (V2:r1) 133/334 columns, 491/4,057 nodes | **drop** |
| dblp_ap | paper (4231) | yes | 20/8,627 (0/5,000) | none: 0/14,328 | not affected |
| dblp_ap | term (50) | no | — | — | not affected |
| imdb_ma | movie (3489) | yes | 43/151 (0/5,000) | none: 0/4,932 | not affected |
| imdb_ma | **actor** (3341) | yes | 1,563/1,563 (15/5,000) | **MEAN of its movies' features over 3,341 mapped columns (V1/mean): 6,124/6,124**, max \|diff\| 2.2e-16 (rounding, ≤ 1e-12 pre-registered), 0 ambiguous | **rebuild** |
| imdb_ma | director (3341) | no (movie–director) | — | — | not affected |

**IMDB actor.** The 3,341 mapped columns are a column subset of the movie features. The other 148 movie columns are the numeric attributes, country and content rating of `info.dat`. The keyword-token candidates (V3) come close but are not exact (5,967/6,124), so the rule is V1/mean.

## Against the pre-registration
- **Positive control met:** ACM author 5,959/5,959.
- **No own-feature type matches a rule** (paper, movie): as expected.
- IMDB actor = MEAN: the pre-registered guess was keyword tokens (V3); the exact rule found is V1 (the movies' own token columns). The expectation said "uncertain"; the type is affected and rebuilt either way.
- **DBLP author: MISS** against the expectation of an exact rule.
  - The features are a function of the paper set (150/150 identical), so the type is affected (criterion ii). But no pre-registered rule reproduces them.
  - Diagnosis (`j51_dblp_diagnosis.py`, output `j51_dblp_diagnosis_output.txt`): over the 201 unmatched columns, with the best term column, the disagreements are mostly "rule 1, author 0" (17,884 cells against 1,112 "author 1, rule 0"); the median column agrees on 99.3 % of authors.
  - This fits a keyword set selected from the author's papers (a count threshold or a per-author top-k), not an OR. Such a rule was not pre-registered and is **not tested now**: adding a candidate after seeing the data is excluded by the card's rules.
  - Per F12 the type is **dropped** in the LP task. If the author wants a thresholded rule tested, that is a new pre-registered unit.

## Verification
- **Known-answer tests:** `tests/test_lp_feature_rules.py` (4) cover identity OR, a permuted subset of columns, the MEAN of keyword-token counts, and own features with no rule. OK.
  - The first version of the permuted test used a 5-column permutation (equal dimension). The pre-registration fixes the identity map for equal dimensions, so the test was changed to a 3-of-8 permuted subset. Its first toy was too small: the ambiguity flag fired correctly, and the toy was enlarged.
- **Mutation check:** `j51_mutation_check.py` → `j51_mutation_output.txt`, **4/4 caught** (OR as SUM, MEAN divided twice, column matching disabled, every candidate called exact).
  - The "matching disabled" mutation was **not caught** at first. The per-node fallback (best agreement) made a rule count as exact while columns were unmatched, and a rebuild from that mapping would have dropped columns.
  - Fixed: exact now also requires every column matched (`Rule.exact`). Then 4/4 caught.
