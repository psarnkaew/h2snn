# H2S-NN on the H2GB NC datasets: first tuning trials (ops, 2026-11-21)

Source: the `h2snn` tuning chains on TSUBAME at `v3b` `e5bdf10` (started 2026-11-20, cancelled 2026-11-21, decisions log `a353a79`). All trials ran on one NVIDIA H100 (94 GB), resource type `gpu_1`. Read-only extraction by `h2gb_summary.py` from each trial's `metrics.jsonl` and `run.log`. Chance levels from `h2gb_chance.py` on the lab copy of the data (split seed 0; B1a subset as `src/evaluation/validation_subset.py`).

Files:
- `summary.tsv`: one row per trial: batch, eval period, s/epoch (median of log-timestamp differences / epochs, first interval dropped), last epoch, best-validation epoch and value, how the trial ended.
- `<dataset>_furthest.tsv`: per evaluated epoch of the trial that reached the most epochs: train loss, val loss, val accuracy, val Macro-F1, val Micro-F1. Train metrics other than the loss are not computed by the trainer (logged as 0).
- `chance_levels.json`: class counts and chance levels on the B1a subset and on the whole validation split.

"Val" is the B1a validation subset (the selection metric). `end`: `running/killed` = stopped by the chain cancellation, not by the trial.

## Per dataset

| Dataset | Classes | Batch | s / epoch | h / 500 epochs | Epochs reached | Best val epoch | Best val Macro-F1 | Val acc at best | Chance Macro-F1 (prior / uniform / majority) | Chance acc (majority) | Published number to compare |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ogbn-mag | 349 | 1024 | 56.9 | 7.9 | 456 (killed) | 424 | 0.0355 | 0.243 | 0.0027 / 0.0019 / 0.0001 | 0.022 (largest val class 0.064) | Macro-F1 (Hetero²Net Table III, not H2GB): GCN 25.14, GAT 22.94, R-GCN 27.01, HAN 8.94, HINormer 25.80, Hetero²Net 33.28 |
| mag-year | 5 | 256 | 113–121 | 15.7–16.8 | 48 (T_max), 164 (killed) | 132 | 0.2173 | 0.312 | 0.200 / 0.196 / 0.078 | 0.243 | none (H2GB reports accuracy) |
| oag-cs | 3,514 (731 in subset) | 256 | 128.3 | 17.8 | 196 (killed) | 192 | 0.0090 | 0.101 | 0.0010 / 0.0004 / 0.0000 | 0.019 | none (H2GB reports accuracy) |
| oag-eng | 3,957 (589 in subset) | 1024 | 26.6–33.7 | 3.7–4.7 | 499 (finished), 316 (killed) | 300 | 0.1201 | 0.336 | 0.0012 / 0.0003 / 0.0000 | 0.006 | none (H2GB reports accuracy) |
| oag-chem | 2,986 (589 in subset) | 256 | 301.4 | 41.9 | 88 (killed) | 86 | 0.0137 | 0.115 | 0.0013 / 0.0004 / 0.0001 | 0.025 | none (H2GB reports accuracy) |
| IEEE-CIS-G | 2 (positive share 0.093) | 32 | 311.0 | 43.2 | 84 (killed) | 20 | 0.4878 | 0.906 | 0.499 / 0.401 / 0.476 | 0.907 | H2GB's binary F1 of the positive class (not Macro-F1; register to be corrected): GCN 28.79, GAT 28.51, R-GCN 31.44, HAN 23.15, H2G-former 31.55. Chance positive-class F1: all-positive 0.171, prior-random 0.083 |
| H-Pokec | 2 | 256 | 220.8 | 30.7 | 116 (killed) | 32 | 0.6352 | 0.635 | 0.500 / 0.500 / 0.337 | 0.509 | none (H2GB reports accuracy) |
| PDNS | 2 (positive share 0.66) | 32 / 256 / 1024 | 1,050 / 129–176 / 44.6 | 146 / 18–25 / 6.2 | 0–16 (10 of 11 trials stopped by `T_max`) | 0–16 | 0.8752 | 0.890 | 0.500 / 0.486 / 0.399 | 0.663 | none (H2GB reports binary F1 / accuracy) |

## What the curves show
- **IEEE-CIS-G: at chance.** Val accuracy is 0.9066 at every evaluation, equal to the majority share, and val Macro-F1 is constant at 0.4755: the model predicts the majority class for every node.
- **mag-year: barely above chance.** Macro-F1 0.217 against 0.200 for a random predictor; accuracy 0.31 against 0.24 for the majority class.
- **ogbn-mag, oag-cs, oag-eng, oag-chem: learning, slowly.** Macro-F1 is 9–100× the chance level and train and val loss still fall at the last epoch reached (oag-eng's finished trial: val acc 0.011 → 0.245 at epoch 248 → 0.266 at 499). The absolute Macro-F1 is small partly because the metric averages over 589–1,314 classes present in the split. oag-chem's val loss spiked to 22.0 at epoch 42.
- **H-Pokec: above chance** (0.635 vs 0.50), still improving slowly.
- **PDNS: learns fast** (0.875 Macro-F1 by epoch 16) but 10 of 11 trials stopped on `T_max` within 16 epochs.
- **Time budget:** at the batch sizes drawn, oag-chem, IEEE-CIS-G and H-Pokec need 31–43 h per 500-epoch trial and PDNS at batch 32 about 146 h; these exceed one 24 h job and the tuner has no epoch-level resume.
