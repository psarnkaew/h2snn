# H2S-NN imdb_ma LP sweep on TSUBAME (ops evidence, 2026-11-22)

For the model session's card 12 L follow-up (imdb_ma: 29 % of validation-positive actors isolated, no inversion).
Source: first TSUBAME `h2snn` chain, checkout `v3b` `e5bdf10`, run directory `$WORK/h2snn-out/e5bdf10…/experiments/h2snn/h2snn_link_prediction_imdb_ma_tune_*`
(H100 `gpu_1`, 20–21 Nov). Sweep complete: 100 trials, 40 epochs each (LP budget: 40 epochs, patience 40).
Script: `lp_sweep_curves.py` (read-only). Trial score = `val_auroc` of `best_test_metrics.json` (the sweep objective).

## Summary
- Best trial `vital-sweep-81`: val AUROC 0.7307 at epoch 38, lr 7.99e-4, test (random negatives) AUROC in `trials.tsv`.
- Median trial `morning-sweep-40` (rank 50): val AUROC 0.7141 at epoch 28, lr 1.85e-3.
- Best epoch = 0 in **1 of 100** trials (acm_pa: 67 of 69). Best-epoch median 31; 24 trials peak at the last epoch (39).
- Val AUROC over trials: min 0.513, Q1 0.704, median 0.714, max 0.731.

## Curves (from `curves.tsv`; AUROC, loss)
| trial | epoch | train AUROC | val AUROC | train loss | val loss |
|---|---|---|---|---|---|
| best | 0 | 0.438 | 0.515 | 0.711 | 0.695 |
| best | 10 | 0.667 | 0.593 | 0.672 | 0.685 |
| best | 20 | 0.778 | 0.679 | 0.609 | 0.637 |
| best | 30 | 0.810 | 0.723 | 0.565 | 0.609 |
| best | 38 | 0.824 | **0.731** | 0.542 | 0.604 |
| median | 0 | 0.438 | 0.517 | 0.708 | 0.694 |
| median | 10 | 0.772 | 0.659 | 0.603 | 0.645 |
| median | 20 | 0.829 | 0.709 | 0.510 | 0.652 |
| median | 28 | 0.851 | **0.714** | 0.471 | 0.729 |
| median | 39 | 0.897 | 0.698 | 0.399 | 0.937 |

Train and validation AUROC rise together from about 0.44/0.52 at epoch 0; no inversion (validation never falls below its epoch-0 value after the first epochs).
The median trial overfits after epoch ~28 (val loss up from 0.645 to 0.937, val AUROC 0.714 → 0.698). The best trial (lower lr) is still improving at epoch 38–39.
