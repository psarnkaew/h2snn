# Published-number register (spec 1.26 §6 "Published numbers as placeholders")

**What this file is.** It lists every result cell that holds a **published** number because we have not run that (dataset, task, method) cell ourselves. It is project memory, not the paper. The author decided this on 2026-11-19 (decisions log).

**Rules (spec 1.26).**
- A placeholder enters no statistical test, and no claim rests on it.
- Before submission, each row is either replaced by our run (fill `replaced by`) or kept as published by an explicit author decision (fill `kept: decision row`).
- Whoever puts a published number into a table (the paper session or any agent) adds the row here **in the same commit**.

**Columns.**
- `protocol differences`: how the source differs from spec §6, for example split, metric (Macro-F1 vs accuracy), checkpoint selection (B1a subset), number of seeds, test negatives (LP), or tuning budget.
- `status`: `placeholder`, `replaced`, or `kept`.

| dataset | task | method | metric | value (mean ± std) | source (paper, table, page) | protocol differences | status | replaced by / kept: decision row |
|---|---|---|---|---|---|---|---|---|
| ACM | NC | GCN | Macro-F1 | 92.17±0.24 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| DBLP | NC | GCN | Macro-F1 | 90.84±0.32 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| IMDB | NC | GCN | Macro-F1 | 57.88±1.18 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| Freebase | NC | GCN | Macro-F1 | 27.84±3.13 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| IEEE-CIS-G | NC | GCN | Macro-F1 | 28.79±1.07 | Lin et al. 2025 (H2GB), Table 3, p.7 | H2GB authors' runs, 5 seeds, H2GB hyperparameters, V100; metric is H2GB's F1, which is Macro-F1 on IEEE-CIS-G (spec 1.14 H4); full validation set (not the B1a subset) | placeholder | — |
| ACM | NC | GAT | Macro-F1 | 92.26±0.94 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| DBLP | NC | GAT | Macro-F1 | 93.83±0.27 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| IMDB | NC | GAT | Macro-F1 | 58.94±1.35 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| Freebase | NC | GAT | Macro-F1 | 40.74±2.58 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| IEEE-CIS-G | NC | GAT | Macro-F1 | 28.51±0.45 | Lin et al. 2025 (H2GB), Table 3, p.7 | H2GB authors' runs, 5 seeds, H2GB hyperparameters, V100; metric is H2GB's F1, which is Macro-F1 on IEEE-CIS-G (spec 1.14 H4); full validation set (not the B1a subset) | placeholder | — |
| ACM | NC | R-GCN | Macro-F1 | 91.55±0.74 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| DBLP | NC | R-GCN | Macro-F1 | 91.52±0.50 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| IMDB | NC | R-GCN | Macro-F1 | 58.85±0.26 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| Freebase | NC | R-GCN | Macro-F1 | 46.78±0.77 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| IEEE-CIS-G | NC | R-GCN | Macro-F1 | 31.44±0.96 | Lin et al. 2025 (H2GB), Table 3, p.7 | H2GB authors' runs, 5 seeds, H2GB hyperparameters, V100; metric is H2GB's F1, which is Macro-F1 on IEEE-CIS-G (spec 1.14 H4); full validation set (not the B1a subset) | placeholder | — |
| ACM | NC | HAN | Macro-F1 | 90.89±0.43 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| DBLP | NC | HAN | Macro-F1 | 91.67±0.49 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| IMDB | NC | HAN | Macro-F1 | 57.74±0.96 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| Freebase | NC | HAN | Macro-F1 | 21.31±1.68 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| IEEE-CIS-G | NC | HAN | Macro-F1 | 23.15±0.43 | Lin et al. 2025 (H2GB), Table 3, p.7 | H2GB authors' runs, 5 seeds, H2GB hyperparameters, V100; metric is H2GB's F1, which is Macro-F1 on IEEE-CIS-G (spec 1.14 H4); full validation set (not the B1a subset) | placeholder | — |
| ACM | NC | MAGNN | Macro-F1 | 90.88±0.64 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| DBLP | NC | MAGNN | Macro-F1 | 93.28±0.51 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| IMDB | NC | MAGNN | Macro-F1 | 56.49±3.20 | Lv et al. 2021 (HGB), Table 3, p.7 | HGB authors' own runs: 5 seeds, HGB hyperparameters (no 100-trial Bayesian search), same 24/6/70 split | placeholder | — |
| ACM | NC | HetSheaf | Macro-F1 | 94.30±0.24 | Braithwaite et al. 2024 (HetSheaf), Table 2, p.9 | HetSheaf authors' runs: 150-run sweep, seeds 42-46, canonical HGB 24/6/70 split; Freebase not reported | placeholder | — |
| DBLP | NC | HetSheaf | Macro-F1 | 94.93±0.36 | Braithwaite et al. 2024 (HetSheaf), Table 2, p.9 | HetSheaf authors' runs: 150-run sweep, seeds 42-46, canonical HGB 24/6/70 split; Freebase not reported | placeholder | — |
| IMDB | NC | HetSheaf | Macro-F1 | 64.51±0.68 | Braithwaite et al. 2024 (HetSheaf), Table 2, p.9 | HetSheaf authors' runs: 150-run sweep, seeds 42-46, canonical HGB 24/6/70 split; Freebase not reported | placeholder | — |
| ACM | NC | NSD | Macro-F1 | 92.07±2.09 | Braithwaite et al. 2024 (HetSheaf), Table 2, p.9 | HetSheaf authors' runs: 150-run sweep, seeds 42-46, canonical HGB 24/6/70 split; Freebase not reported | placeholder | — |
| DBLP | NC | NSD | Macro-F1 | 92.91±1.21 | Braithwaite et al. 2024 (HetSheaf), Table 2, p.9 | HetSheaf authors' runs: 150-run sweep, seeds 42-46, canonical HGB 24/6/70 split; Freebase not reported | placeholder | — |
| IMDB | NC | NSD | Macro-F1 | 61.94±0.34 | Braithwaite et al. 2024 (HetSheaf), Table 2, p.9 | HetSheaf authors' runs: 150-run sweep, seeds 42-46, canonical HGB 24/6/70 split; Freebase not reported | placeholder | — |
| ACM | NC | SeHGNN | Macro-F1 | 92.95±0.38 | Braithwaite et al. 2024 (HetSheaf), Table 23, p.52 | HetSheaf matched-protocol re-run: 100-run random search, best validation configuration, HGB 24/6/70 split, HetSheaf seeds | placeholder | — |
| DBLP | NC | SeHGNN | Macro-F1 | 93.81±0.29 | Braithwaite et al. 2024 (HetSheaf), Table 23, p.52 | HetSheaf matched-protocol re-run: 100-run random search, best validation configuration, HGB 24/6/70 split, HetSheaf seeds | placeholder | — |
| IMDB | NC | SeHGNN | Macro-F1 | 66.87±0.11 | Braithwaite et al. 2024 (HetSheaf), Table 23, p.52 | HetSheaf matched-protocol re-run: 100-run random search, best validation configuration, HGB 24/6/70 split, HetSheaf seeds | placeholder | — |
| IEEE-CIS-G | NC | H2G-former | Macro-F1 | 31.55±0.92 | Lin et al. 2025 (H2GB), Table 3, p.7 | H2GB authors' runs, 5 seeds, H2GB hyperparameters, V100; metric is H2GB's F1, which is Macro-F1 on IEEE-CIS-G (spec 1.14 H4); full validation set (not the B1a subset) | placeholder | — |
| PubMed | LP | GCN | ROC-AUC | 80.48±0.81 | Lv et al. 2021 (HGB), Table 4, p.7 | HGB authors' runs, 5 seeds, HGB hyperparameters; HGB 2-hop negative test | placeholder | — |
| LastFM | LP | GCN | ROC-AUC | 59.17±0.31 | Lv et al. 2021 (HGB), Table 4, p.7 | HGB authors' runs, 5 seeds, HGB hyperparameters; HGB 2-hop negative test | placeholder | — |
| PubMed | LP | R-GCN | ROC-AUC | 78.29±0.18 | Lv et al. 2021 (HGB), Table 4, p.7 | HGB authors' runs, 5 seeds, HGB hyperparameters; HGB 2-hop negative test | placeholder | — |
| LastFM | LP | R-GCN | ROC-AUC | 57.21±0.09 | Lv et al. 2021 (HGB), Table 4, p.7 | HGB authors' runs, 5 seeds, HGB hyperparameters; HGB 2-hop negative test | placeholder | — |
| LastFM | LP | MAGNN | ROC-AUC | 56.81±0.05 | Lv et al. 2021 (HGB), Table 4, p.7 | HGB authors' runs, 5 seeds, HGB hyperparameters; HGB 2-hop negative test | placeholder | — |
