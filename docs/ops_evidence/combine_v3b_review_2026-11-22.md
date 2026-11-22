# Ops review: combine-v3b (cards 12 + 14 + 15 + 16), 2026-11-22

Branch `combine-v3b`, head `e52b0252d8f4c21b7b39cf7f505c34079da47fc0`, fast-forward from `v3b` `66e9349`.

- `bab7b54` merge-c12-c14 → `007fa82` merge card16 `15e06cf` (no-ff) → `674b80e` merge card15 `2e4881c` (no-ff, no conflict).
- `b0c7714` (ops): `configs/experiments/acm_pp_h2snn_v3b.yaml` epochs 40 → 500 (model session request); TSUBAME cell lists acm_pa → acm_pp (`scripts/tsubame/make_cell_list.py` phase baselines-1 and `H2SNN_CELLS`; `host_memory.py` acm_pp "no record"); `tests/test_tsubame_cell_list.py`, `tests/test_tsubame_phases.py`.
- Full suite on `b0c7714`, lab job 18293 (CPU, clean checkout, vendor with built sparse_tools): 998 run, 1 skipped, **3 failures**, all from the intended acm_pa → acm_pp swap: `test_card5_protocol_hooks` T4, `test_card7_tsubame` T9, `test_card8_phases` byte-identity guards compare the cell lists with older captures.
- `e52b025` (ops, tests only): those three guards read the captures' acm_pa as acm_pp; every other byte is compared as before.
- Targeted run on `e52b025`, job 18296: `test_card5_protocol_hooks`, `test_card7_tsubame`, `test_card8_phases`, `test_tsubame_cell_list`, `test_tsubame_phases`, `test_tsubame_host_memory`: 28 tests OK.
- acm_pp data copied to TSUBAME `$WORK/h2snn-data/LP/lp_derived/acm_pp` (10 files, sha256 identical).

Recommendation: approve `e52b0252d8f4c21b7b39cf7f505c34079da47fc0` for the single `v3b` move.
