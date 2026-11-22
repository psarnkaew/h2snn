# Card 16 — Binary F1 on IEEE-CIS-G and PDNS; epoch budget of tuned cells (500 / patience 40)

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
`docs/H2S-NN_card16_report.md`: files changed, test results. Last line `branch=card16 sha=<sha> card=docs/H2S-NN_card16_card.md base=<merge-c12-c14 sha>`. Estimate ≈ ½ day (both parts).

## Part 2 — Epoch budget of tuned cells (author decision 2026-11-22, via ops; evidence `docs/ops_evidence/epoch_budget_2026-11-22/`, 45ad4ea)
- `src/configs/config.py` `PROTOCOL_BUDGETS`: HGB NC 300/30 → **500/40**; LP 40/40 → **500/40**; H2GB NC 500/None → **500/40**; GC (hetsheaf) unchanged (validation loss, patience 50, final weights; 500 epochs from the configs, card 13). Monitor and checkpoint unchanged in kind: best validation metric (HGB NC Macro-F1, H2GB NC B1a-subset Macro-F1, LP ROC-AUC).
- H2GB loops evaluate every `eval_period` epochs (4; OAG 2): early stopping is checked at those evaluations, and patience counts **epochs** (40 epochs = 10 or 20 evaluations without improvement).
- The baseline budget path (`src/baselines/budget.py`) follows the table; the H2S-NN configs that set patience explicitly (H2GB 500, card 5) are set to follow the table; tests that pin the old budgets are updated to the new values.
- Published-configuration cells are **unchanged** (their source's epochs and patience).
- Each run record keeps the epoch it stopped at, so ops can report per cell the share of trials that reach the 500 cap.
- Tests: the existing budget tests with the new values; one test that H2GB patience counts epochs over evaluation points (eval_period 4, patience 40 → stops after 10 evaluations without improvement); the full suite.
