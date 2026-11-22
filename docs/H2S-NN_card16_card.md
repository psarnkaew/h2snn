# Card 16 — Binary F1 on the two-class H2GB datasets (IEEE-CIS-G, PDNS)

Written 2026-11-22 by the model design session; approved by the author the same day. Small evaluation change; no method change, no change to selection.

## Why
H2GB's `metric_best: f1` on IEEE-CIS and PDNS is the **binary F1 of class 1** (`H2GB/graphgym/logger.py:220-226` picks `classification_binary` for ≤ 2 classes; `H2GB/logger.py:155` `f1_score(true, pred)`). The published IEEE-CIS-G numbers we copy (register, corrected 2026-11-22) are therefore binary F1, while we report Macro-F1. The columns must match.

## Change
- On every two-class node-classification dataset (IEEE-CIS-G, PDNS), every method we run (H2S-NN and every baseline, tuned or published-configuration) **also reports** binary F1 of class 1 (`sklearn f1_score(y, ŷ, average="binary", pos_label=1)`, ŷ = argmax of the `C = 2` logits), on validation (B1a subset and full) and test, next to Macro- and Micro-F1.
- Checkpoint and configuration selection are **unchanged** (validation Macro-F1 on the B1a subset; spec 1.28 option 1). The paper's table shows binary F1 on these two datasets; the difference in selection signal goes in the appendix difference table.
- Where: the shared metric function(s) used by the H2S-NN trainers and by `src/baselines/metrics.py`; the result records the campaign writes. Nothing else.

## Base and git
Branch `card16` from `merge-c12-c14` (after S12's merge report), so it merges with cards 12 and 14 into `v3b`. Stage by explicit path, push `card16`, never merge into `v3b`.

## Tests (only these)
- One known-answer test: a fixed 2-class prediction gives the hand-computed binary F1, Macro-F1 and Micro-F1.
- One smoke epoch on PDNS (H2S-NN) showing the new key in the records.
- The full suite by `sbatch`.

## Report
`docs/H2S-NN_card16_report.md`: files changed, test results. Last line `branch=card16 sha=<sha> card=docs/H2S-NN_card16_card.md base=<merge-c12-c14 sha>`. Estimate ≈ 2–3 h.
