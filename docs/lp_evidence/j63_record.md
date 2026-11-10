# J6.3 Loader guard (F20)
2026-11-10. CPU only; no training.

## What changed
- **`src/data/link_prediction_dataset.py`** (the only `src/` file edited).
  - The four derived tasks (acm_pa, dblp_ap, imdb_ma, freebase_bb) carry `"feature_rule": True`.
  - `_load_local_hgb_link_data` calls `_assert_feature_rule` before reading the nodes. It refuses a task when:
    - `feature_rule.json` is missing;
    - `node.dat`'s md5 differs from the record. This catches a re-run of the split builder, which restores the release features, or a hand edit;
    - a node type has rows of different widths;
    - the per-type widths differ from the record, or a removed type has features. This holds even when the md5 in the record is forged.
- **Tests that build toy derived tasks and then load them now apply the rule after the build:** `test_lp_derived.py`, `test_lp_valid_files.py`, `test_heart_negatives.py`. This is the order the real pipeline now requires.
- **`scripts/lp/build_heart_negatives.py`:** acm_pa's `features` flag is now `False`.
  - With the author block removed (F17), C2 (cosine in one shared feature space) no longer applies. PubMed is the only C2 task.
  - **Carded under J6.4 but committed here**, because the toy HeaRT tests now apply the rule and load featureless authors. The acm_pa lists are regenerated in J6.4.

## Tests (VERIFIED-RUN)
- **New test file `tests/test_lp_feature_rule.py`, 7 tests.** It covers:
  - block removal keeps every other row;
  - single-column removal;
  - a task with the rule applied loads.

  The loader refuses each of:
  - a missing record;
  - a split-builder re-run;
  - a hand-edited value;
  - a removed type with features under a forged md5.
- **`tests.test_lp_feature_rule`, `test_lp_derived`, `test_lp_valid_files`, `test_heart_negatives`:** 76 tests, OK.

## Mutation check (`j63_mutation_check.py` → `j63_mutation_output.txt`)

| Mutation | Result |
|---|---|
| Guard call removed | caught |
| md5 check removed | caught |
| Width/removed-type check removed | caught |

3/3 caught. The loader was byte-identical after each restore.
