# J5.3 Rebuild or drop the LP endpoint features (spec v1.9 F12)
2026-11-10. LP task files only (`data/lp_derived/<task>/node.dat`). The node-classification releases are untouched (41 files, md5 identical before and after). The data folder is gitignored, so the evidence is the scripts, the md5 values and the outputs below.

**Command (VERIFIED-RUN):** `python scripts/lp/rebuild_lp_features.py --rules ../docs/lp_evidence/j51_feature_rules.json --tasks acm_pa dblp_ap imdb_ma`, run from `H2S-NN/`, CPU. Output: `j53_rebuild_output.txt`. Per task, `feature_rebuild.json` records the rule, the column map and the counts.

| Task | Type | Action | Rule (J5.1) | Nodes | Without a training fact (zero vector) | Rows changed | node.dat md5 before → after |
|---|---|---|---|---|---|---|---|
| acm_pa | author | **rebuilt** from 8,059 training facts | OR of its papers' features | 5,959 | 847 | 1,566 | `01b4942d…` → `5eed223b…` |
| dblp_ap | author | **dropped** (featureless: learned embedding) | none exact | 4,057 | — | 4,057 | `efcf8247…` → `63493833…` |
| imdb_ma | actor | **rebuilt** from 11,971 training facts | MEAN of its movies' 3,341 token columns | 6,124 | 773 | 1,915 | `7954f977…` → `42c9e113…` |
| freebase_bb | — | unchanged (Freebase has no node features) | — | — | — | 0 | `a7d65c59…` |

**Rule choice and format:**
- ACM has six equivalent exact rules (OR and MAX, on features, their presence, or the paper→term links). Applied to the training facts, all six give identical features (`other_exact_rules_agree_on_training`: all true).
- Values are written in the release's format: integers for ACM, `repr` for IMDB. Every release row of the rebuilt type prints back to its own text with that format.
- `split_meta.json`'s output md5 of node.dat is updated, with a pointer to `feature_rebuild.json`.
- `build_derived_lp.py`'s docstring now states that the rebuild runs after the build.

## Gate 1: bitwise (`j53_bitwise_check.py` → `j53_bitwise_check_output.txt`: ALL PASS)
- **acm_pa author:** 5,959/5,959 nodes equal to the rule on training facts. The rule is computed node by node in `lp_feature_leak.rule_on_training`, a separate code path from the rebuild's sparse aggregation. The loader's float32 tensor is also equal.
- **imdb_ma actor:** 6,124/6,124, and the loader tensor is equal.
- **dblp_ap author:** no stored feature, and no `x` in the loader.
- The rows of every other type are byte-identical to the release (4,983 / 22,071 / 15,296 rows).
- The node-classification releases: 41 files, 0 changed (`j53_nc_release_md5_before.txt`).

## Gate 2: features-only scorer at the Ctrl-G level, tolerance 0.005
Command: `lp_feature_leak.py … --tasks acm_pa dblp_ap imdb_ma --cosine` → `j53_leak_after_output.txt`. These are the existing J4.4 lists; acm_pa's are regenerated in J5.5.

| Task | Split | Scorer MRR / Hits@10 | Ctrl-G MRR / Hits@10 | Difference | Before (J5.2) |
|---|---|---|---|---|---|
| acm_pa | valid | 0.0206 / 0.0547 | 0.0206 / 0.0547 | **+0.0000 / +0.0000** | 1.000 / 1.000 |
| acm_pa | test | 0.0195 / 0.0472 | 0.0195 / 0.0472 | **+0.0000 / +0.0000** | 1.000 / 1.000 |
| imdb_ma | valid | 0.0566 / 0.0632 | 0.0566 / 0.0632 | **+0.0000 / +0.0000** | 0.9105 / 0.9662 |
| imdb_ma | test | 0.0439 / 0.0562 | 0.0439 / 0.0562 | **+0.0000 / +0.0000** | 0.9226 / 0.9777 |
| dblp_ap (dropped) | valid | 0.0040 / 0.0000 (all ties) | 0.0234 / 0.0402 | at the all-tie level (its pre-registered control) | 0.0856 / 0.2596 |
| dblp_ap (dropped) | test | 0.0040 / 0.0000 (all ties) | 0.0244 / 0.0423 | at the all-tie level | 0.0855 / 0.2592 |

**Gate met.** The difference is exactly 0, as expected for features that equal the rule on training facts. The prediction "acm_pa after < 0.5" holds (0.02).

**ACM cosine on the stored features** (C2 heuristic, shared space) after the fix: MRR 0.0722 / 0.0557, Hits@10 0.1385 / 0.1176 on the old lists. This is content similarity, not containment of a held-out paper.

## Mutation checks (`j53_mutation_check.py` → `j53_mutation_output.txt`): 2/2 caught
- **Rebuild from all release facts (test included):** the toy bitwise tests fail. On a scratch copy of the real acm_pa, the scorer beats Ctrl-G by 0.979 (validation) and 0.981 (test) MRR.
- **Validation facts kept in the training neighbourhood:** the toy bitwise tests fail.
- The rebuild script is restored byte-identical, and the real task file's md5 is unchanged across the run (`5eed223b…`).

## Tests
`tests/test_lp_feature_rebuild.py` (5): OR and MEAN rebuilt bitwise from training facts; a type without an exact rule is dropped; other rows are byte-identical and the release untouched; the rebuild is idempotent.

Affected modules run after the rebuild: `test_lp_feature_rebuild`, `test_lp_feature_rules`, `test_lp_feature_leak`, `test_lp_derived`, `test_lp_valid_files`, `test_heart_negatives`: **81 tests OK**.
