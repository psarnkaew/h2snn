# Merge of cards 12 and 14 (S12), 2026-11-22

Inputs: card14 3b2f0f2 (branch start), card12 8c13e8b, design-docs cd54121, base origin/v3b 66e9349. v3b is not touched.

## Merge commits

| Commit | What |
|---|---|
| 186ce2a | `git merge --no-ff 8c13e8b` (card 12), no conflicts |
| ec4c607 | `git merge --no-ff cd54121` (design-docs, docs only), no conflicts |
| 65b9309 | re-captures + T4 test pointer + evidence job scripts |
| (this commit) | report and job logs |

Worktree `/mnt/data/peeradon/h2snn-merge`. `vendor/` is a copy of the card14 worktree's (not a symlink). `data` links to `/mnt/data/peeradon/h2snn/H2S-NN/data`.

## Re-captured files

Both captures were made from ec4c607 in job 18275 (g2n01, i9-10900K, CPU only, environment E, 1 thread), with the same scripts as before (`docs/legacy_evidence/v3b_golden.py capture`, as in `lr1_job.sh`; `docs/card5_evidence/capture_base.py`). Script: `docs/merge_c12_c14_evidence/capture.sh`, log `capture_18275.log`.

- **`docs/legacy_evidence/v3b_golden_cluster.pt`** (sha256 cf296000…). Card 14 replaced the CPU generators with a counter-based SplitMix64 stream, so every WISP draw changed. The cells, logits, losses, gradients and solver statistics that depend on those draws changed with it. Compared with the 66e9349 capture (job 18280, `golden_keys.py`):
  - the same 23 configurations and the same skipped one (`nc_map_family_dense`);
  - all 1238 old tensors are present: 870 changed (the count card 14 reported) and 368 are bit-identical;
  - the only new keys are the 44 `*.pcg_zero_rhs_scopes` (card 12's stats field, 2 per configuration).
- **`docs/card5_evidence/base_capture_ec4c607.json`** (new name; `base_capture_894fd74.json` is kept as history). The `hgb_full_batch` and `acm_minibatch_flag` metrics changed in their losses, because of the same random stream. Accuracy and F1 are unchanged.
  - `baselines_1_cell_list` differs from 894fd74. Card 8 C5 changed it on purpose, and T4 compares it with the 1646efd capture, not with this file.
  - `test_T4_records_equal_the_base_commit` now reads the new file. What it compares is unchanged.
- `test_v3b_golden` is unchanged. It is still bit-exact and ignores only `pcg_zero_rhs_scopes`.

## Full suite

Job 18281 (g2tn01, TITAN RTX, `--mem` 39G = 29.7 GiB (card 14 suite peak) + 30 %; `WANDB_MODE` unset) ran on 65b9309: **993 tests, OK, 0 skipped**. Wall time 1 h 00 m. Peak RSS 28.4 GiB. Log: `docs/merge_c12_c14_evidence/suite_18281.log`.

## Smoke epochs

Card 14's `smoke.py` ran on ec4c607, whose src is identical to the head. `--mem` was each run's card 14 peak + 30 %.

| Run | Job | GPU | Wall time | Peak host RSS | Peak GPU | Result |
|---|---|---|---|---|---|---|
| PDNS `balmy-sweep-7` | 18276 | RTX 3090 | 5 m 53 s | 6.1 GiB | 11.97 GiB | loss 0.396 finite, 29/29 parameters moved, **no `PCGConvergenceError`** |
| NCI1 `astral-sweep-1`, lr 3e-4 | 18277 | RTX 3090 | 11 m 24 s | 1.4 GiB | 0.02 GiB | loss 0.684 finite, 158/163 moved (the same 5 structurally unused tensors as on 66e9349, card 14 report) |
| ACM `acm_h2snn_v3b.yaml` | 18278 | TITAN RTX | 12 s | 1.3 GiB | 1.67 GiB | loss 1.110 finite, 36/36 moved |

Logs: `docs/merge_c12_c14_evidence/job_<id>.log`.

## src diff against 66e9349

```
$ git diff --stat 66e9349 -- H2S-NN/src
 H2S-NN/src/data/path_sampler.py                    |  76 ++++++--
 H2S-NN/src/data/scopes.py                          | 194 ++++++++++++++-------
 H2S-NN/src/data/wisp_sampler.py                    | 170 ++++++++++++------
 H2S-NN/src/data/wisp_tasks.py                      |  90 ++++------
 H2S-NN/src/models/anchored_solve.py                |  46 ++++-
 .../src/trainers/graph_classification_trainer.py   |  31 ++--
 H2S-NN/src/trainers/link_prediction_trainer.py     |   8 +-
 H2S-NN/src/trainers/minibatch_lp.py                |  33 ++--
 H2S-NN/src/trainers/minibatch_trainer.py           |  24 ++-
 9 files changed, 452 insertions(+), 220 deletions(-)
```

These are card 14's 8 files and card 12's `anchored_solve.py`. Nothing in `configs/` changed. The only test file changed is `tests/test_card5_protocol_hooks.py` (the capture file name).

Note: the author's test rule of 2026-11-22 (one full suite on the final combined branch) arrived while job 18281 was already running, so its result is kept here.

`sha` is the tested head; the commit that adds this report adds only docs on top of it.

branch=merge-c12-c14 sha=65b93090cab9bd9595e8e78933fda9198ee70c6b base=66e9349b79021784a08b37c5f8be0798c0260a04
