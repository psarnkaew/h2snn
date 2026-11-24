# Ops review: card 20 into v3b (combine-c20)

2026-11-23, ops session. Base `v3b` = `0b9873d`. Card 20 head `ccb7eef` (code commit `6726f57`).

## Branch
- `combine-c20` = `ccb7eef`. Card 20 is built on `0b9873d`, so this is a fast-forward with no merge commit.
- Card 19 (MLH/MDE script, another session) is not included. At 19:00 its evidence was uncommitted and it had no report. It touches disjoint files (analysis only, no run path), so it can follow on its own.
- Code change against `0b9873d` in `src/`: `_h2gformer_h2gb.py` only (+14/−1). The B1a subset view gets the same `pestat_Hetero_Metapath` tensors as `data`.

## Full suite
- **Job 18603** on `ccb7eef` (lab, CPU): 1,003 tests ran, 1 error, 1 skipped.
- **The error:** `test_card17_ieee_cis.MetapathCacheTests.test_second_seed_reuses_the_seed0_cache` raised `AttributeError: 'NoneType' object has no attribute 'loader'` in `share_metapath_with_subset_view`.
- **Cause: test only.** The card 17 test stubs `val_subset_loader` to return `(None, ids)`. The real `val_subset_loader` (`minibatch.py:224-240`) always returns `eval_loader(view, ...)`, never None. Card 20's own targeted run did not include this module.
- **Fix:** `f1fc158` (tests only, one line). The card 17 test also stubs `share_metapath_with_subset_view`. `src/` is identical to `ccb7eef`. Pushed to `combine-c20` as a fast-forward.
- **Rerun:** job 18604 on `f1fc158`, `test_card17_ieee_cis` + `test_card20_h2gformer_subset_metapath`: 5 tests OK (20:09–20:10).
- **Ancestry:** `0b9873d` is an ancestor of `f1fc158`, so `v3b` fast-forwards.

## Recommendation and decision
- Recommend `f1fc158409ad7d122ee67c42c897208282b9fd65` for `v3b`. The author approved the move on 2026-11-23 ("ย้ายได้เลย suite ผ่านแล้วรัน H2G-former ต่อ").
- Runs at `0b9873d` stay valid. Card 20 shows the first training loss is identical, and the change only affects the H2G-former B1a validation input.

## After the move (ops)
- H2G-former published seed 0 on mag-year, oag-chem, ogbn-mag, PDNS and H-Pokec: gpu_1, 12 h, from the `f1fc158` checkout, into the `0b9873d` output directory.
- Wave 2 (seeds 1–4) for the four datasets that already have the MetaPath2Vec cache from wave 1.
- H-Pokec seeds 1–4 after seed 0 has built its cache. No H-Pokec cache exists yet, because wave 1's task was deleted while building it.
- Tuned H2G-former (b2: Freebase NC, Freebase (B–B), ogbn-mag (P–P), H-Pokec (U–U)) goes on node_q through the ops TSUBAME automation.
