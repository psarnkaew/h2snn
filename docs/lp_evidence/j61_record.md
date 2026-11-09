# J6.1 The F17 rule applied to the derived LP tasks
2026-11-10. The LP task files only. The data folder is gitignored; the evidence is the script, the md5s and the outputs.

**Command (VERIFIED-RUN):** `python scripts/lp/apply_feature_rule.py`, run from `H2S-NN/`, CPU, starting from the release `node.dat`. Output: `j61_apply_output.txt`. Each task has its `feature_rule.json`.

| Task | Removed | Widths after (type: dim) | node.dat md5 after |
|---|---|---|---|
| acm_pa | author block (5,959 rows → featureless) | paper 1902, author 0, subject 1902, term 0 | `68d9b987…` |
| dblp_ap | author block (4,057 rows) | author 0, paper 4231, term 50, venue 0 | `63493833…` (= J5's drop, as expected) |
| imdb_ma | actor block (6,124 rows); movie columns 3, 4, 7, 12 (`actor_1_`, `actor_3_`, `cast_total_`, `actor_2_facebook_likes`; names checked against `info.dat`) | movie 3,485, director 3,341, actor 0, keyword 0 | `b7eff936…` |
| freebase_bb | nothing (no features) | all 0 | `a7d65c59…` (= the release) |

**What else changed:**
- `split_meta.json`'s output md5 of node.dat is updated, and its J5 key is replaced by `node_dat_feature_rule`.
- J5's `feature_rebuild.json` files are deleted.
- `rebuild_lp_features.py` is marked superseded (kept, unused).
- `build_derived_lp.py`'s docstring points to the rule script.

## Gate (`j61_check.py` → `j61_check_output.txt`: ALL PASS; own parser)
- The removed blocks are featureless on every row (5,959 / 4,057 / 6,124).
- imdb_ma: every one of the 4,932 movie rows has 4 fewer values. The kept values are text-equal, in order, to the release's.
- Every other row is byte-identical to the release.
- `feature_rule.json` and `split_meta.json` carry the actual md5.
- The node-classification releases: 41 files, 0 changed.
- `director_facebook_likes` (movie column 2) is kept, as F17 says.
