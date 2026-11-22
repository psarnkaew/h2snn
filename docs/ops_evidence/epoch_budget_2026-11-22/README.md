# Epoch budget check: does the cap or early stopping end the trials? (ops evidence, 2026-11-22)

Question (author): HGB and H2GB set epochs and patience per model and dataset; is our shared budget a problem for LP, NC, GC?

## Sources
- HGB paper (Lv et al., KDD 2021, arXiv 2112.14936v1 HTML): no epoch or patience number for any task. Appendix C: "For training epoch, we will use early stop mechanism based on the evaluation on validation set to promise fully training."
- H2GB paper (arXiv 2407.10916v2 HTML): node classification only; no epoch number in the text (configs on GitHub; we run its yaml: 500 epochs, no early stopping).
- HGB LP code (`vendor/HGB/LP/benchmark/methods/`): `baseline/run_new.py` 40 / 40 (README: LastFM), `baseline/run_dist.py` 300 / 40 (README: amazon, PubMed), MAGNN 100 / 5–10.
- HGB NC code (`vendor/HGB/NC/benchmark/methods/`): `baseline`, `GNN`, `HGT` 300 / 30; GTN 40; HAN patience 100; MAGNN 100 / 5–10; RSHN 50.
- Spec (tuned cells): NC HGB 300 / 30; LP 40 / 40; H2GB 500, no early stopping; GC 500 / 50 on validation loss (card 13).

## H2S-NN trials of the first TSUBAME chain (`v3b` `e5bdf10`, H100)
| Sweep | Trials | Epochs run (max / median) | Trials at the cap | Best epoch (median / max) | Best in the last 5 epochs |
|---|---|---|---|---|---|
| NC acm | 100 | 254 / 51.5 | 0 (cap 300) | 20.5 / 223 | 0 |
| NC dblp | 100 | 207 / 73.5 | 0 | 42.5 / 176 | 0 |
| NC imdb | 100 | 138 / 49 | 0 | 18 / 107 | 0 |
| NC freebase | 27 | 168 / 82 | 0 | 51 / 137 | 0 |
| LP dblp_ap | 100 | 40 / 40 | 100 (cap 40) | 39 / 39 | **85** |
| LP imdb_ma | 100 | 40 / 40 | 100 | 31 / 39 | 24 (best at epoch 39) |
| LP lastfm | 17 | 40 / 40 | 17 | 39 / 39 | **16** |
| LP pubmed | 22 | 40 / 40 | 22 | 8 / 39 | 4 |
| NC oag-eng (H2GB) | 1 | 500 (fixed) | — | 442 | — |
| GC (4 datasets) | 0 finished | — | — | — | — |

## Reading
- **LP:** the cap ends every trial and most trials are still improving (dblp_ap 85 of 100, lastfm 16 of 17 best in the last 5 epochs). 40 / 40 is one HGB script's default, not a benchmark rule; patience = epochs disables early stopping.
- **NC (HGB):** early stopping ends every trial well before 300; no trial reaches the cap. No change needed.
- **NC (H2GB):** a fixed 500 epochs without early stopping is H2GB's own protocol; one finished trial (best at 442 of 500) is too little to judge; recheck after the restart.
- **GC:** HetSheaf's protocol (500, patience 50 on validation loss) is in place since card 13; no finished trial yet; recheck after the restart.
