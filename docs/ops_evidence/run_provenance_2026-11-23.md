# Run provenance: GPU per cell and mixed run SHAs (ops, 2026-11-23)

Rule (2026-11-20): one cell = one GPU type, for all of its tuning trials and final seeds.
Code for every cell below: v3b 0b9873d, unless a row says otherwise.
The model session asked for these lists (reply on the H2G-former fix, card 20, 2026-11-23). This file is the ops source; the register can point to it.

## H2S-NN (ours)

| Task | Dataset | GPU | Site |
|---|---|---|---|
| NC | ACM | RTX 3090 24 GB | lab |
| NC | IMDB | RTX 3090 24 GB | lab |
| LP | ACM (P–P) | RTX 3090 24 GB | lab |
| NC | DBLP | TITAN RTX 24 GB | lab |
| LP | IMDB (M–A) | TITAN RTX 24 GB | lab |
| LP | LastFM | RTX A6000 48 GB | lab |
| LP | PubMed | RTX A6000 48 GB | lab |
| GC | MUTAG, ENZYMES, PROTEINS | RTX 2080 Ti 11 GB | lab |
| NC | ogbn-mag, mag-year, OAG-CS, OAG-ENG, OAG-CHEM, IEEE-CIS-G, PDNS | H100 94 GB (gpu_1) | TSUBAME |
| LP | Freebase (B–B), DBLP (A–P) | H100 94 GB (gpu_1) | TSUBAME |
| NC | Freebase, H-Pokec | H100 94 GB (node_q) | TSUBAME |
| LP | ogbn-mag (P–P), H-Pokec (U–U) | H100 94 GB (node_q) | TSUBAME |

- DBLP (A–P) LP failed the lab fit check: 95 % of 24 GB on TITAN RTX. It runs on H100 only.
- Lab cells passed a fit check: largest probed configuration ≤ 75 % of GPU memory.
- NCI1 GC is not scheduled yet (open decision).

## Baseline tuning b1 (tuned configurations, small datasets)

Cells that passed the lab fit check run on the lab type below. Cells that did not fit (OOM) run on H100.

| Task | Dataset | Model | GPU |
|---|---|---|---|
| LP | acm_pp | csnn | RTX 3090 (lab) |
| LP | acm_pp | gat | H100 (TSUBAME gpu_1) |
| LP | acm_pp | gcn | RTX 3090 (lab) |
| LP | acm_pp | h2gformer | H100 (TSUBAME gpu_1) |
| LP | acm_pp | h2sgnn | RTX 3090 (lab) |
| LP | acm_pp | han | TITAN RTX (lab) |
| LP | acm_pp | hetero2net | RTX 3090 (lab) |
| LP | acm_pp | hetsheaf | H100 (TSUBAME gpu_1) |
| LP | acm_pp | nsd | H100 (TSUBAME gpu_1) |
| LP | acm_pp | pshgcn | TITAN RTX (lab) |
| LP | acm_pp | rgcn | RTX 3090 (lab) |
| LP | dblp_ap | csnn | TITAN RTX (lab) |
| LP | dblp_ap | gat | RTX 3090 (lab) |
| LP | dblp_ap | gcn | TITAN RTX (lab) |
| LP | dblp_ap | h2gformer | H100 (TSUBAME gpu_1) |
| LP | dblp_ap | h2sgnn | TITAN RTX (lab) |
| LP | dblp_ap | han | RTX 3090 (lab) |
| LP | dblp_ap | hetero2net | TITAN RTX (lab) |
| LP | dblp_ap | hetsheaf | H100 (TSUBAME gpu_1) |
| LP | dblp_ap | nsd | H100 (TSUBAME gpu_1) |
| LP | dblp_ap | pshgcn | RTX 3090 (lab) |
| LP | dblp_ap | rgcn | TITAN RTX (lab) |
| LP | imdb_ma | csnn | RTX 3090 (lab) |
| LP | imdb_ma | gat | TITAN RTX (lab) |
| LP | imdb_ma | gcn | RTX 3090 (lab) |
| LP | imdb_ma | h2gformer | H100 (TSUBAME gpu_1) |
| LP | imdb_ma | h2sgnn | RTX 3090 (lab) |
| LP | imdb_ma | han | TITAN RTX (lab) |
| LP | imdb_ma | hetero2net | RTX 3090 (lab) |
| LP | imdb_ma | hetsheaf | H100 (TSUBAME gpu_1) |
| LP | imdb_ma | nsd | RTX 3090 (lab) |
| LP | imdb_ma | pshgcn | TITAN RTX (lab) |
| LP | imdb_ma | rgcn | RTX 3090 (lab) |
| LP | lastfm | csnn | TITAN RTX (lab) |
| LP | lastfm | gat | RTX 3090 (lab) |
| LP | lastfm | h2gformer | H100 (TSUBAME gpu_1) |
| LP | lastfm | h2sgnn | RTX 3090 (lab) |
| LP | lastfm | han | TITAN RTX (lab) |
| LP | lastfm | hetero2net | RTX 3090 (lab) |
| LP | lastfm | hetsheaf | H100 (TSUBAME gpu_1) |
| LP | lastfm | nsd | H100 (TSUBAME gpu_1) |
| LP | lastfm | rgcn | TITAN RTX (lab) |
| LP | pubmed | csnn | RTX 3090 (lab) |
| LP | pubmed | gat | H100 (TSUBAME gpu_1) |
| LP | pubmed | h2gformer | RTX 3090 (lab) |
| LP | pubmed | h2sgnn | TITAN RTX (lab) |
| LP | pubmed | han | RTX 3090 (lab) |
| LP | pubmed | hetero2net | TITAN RTX (lab) |
| LP | pubmed | hetsheaf | H100 (TSUBAME gpu_1) |
| LP | pubmed | nsd | H100 (TSUBAME gpu_1) |
| LP | pubmed | pshgcn | H100 (TSUBAME gpu_1) |
| NC | acm | h2gformer | H100 (TSUBAME gpu_1) |
| NC | dblp | h2gformer | H100 (TSUBAME gpu_1) |
| NC | imdb | chebnetii | RTX 3090 (lab) |
| NC | imdb | csnn | TITAN RTX (lab) |
| NC | imdb | h2gformer | H100 (TSUBAME gpu_1) |
| NC | imdb | hetsheaf | H100 (TSUBAME gpu_1) |

Early tuning (CSNN and HetSheaf on ACM/DBLP NC, `cells_tuning_early.txt`, code e52b025): H100 (gpu_1). It was moved from node_f before it started.
Published configurations and H2GB/large-dataset cells: H100 on TSUBAME (node type is in each run's provenance `gpu_name`).

## Published cells with mixed run SHAs (training identical)

The 2d822a4 runner passed `--resume` to `run_baseline_published.py`, so lines exited with code 2 at start (fixed by card 10 A3.3).
Those seed lines were rerun from the 0b9873d checkout into the same 2d822a4 output directory.
For these cells, training is identical between the two SHAs: cross_entropy, no early stopping; the other changes are dry-run/fit only.
Seeds of one cell that finished at different SHAs are listed below.

| Cell (published config) | Output dir SHA | Seeds at 2d822a4 | Seeds at 0b9873d |
|---|---|---|---|
| GCN oag-chem NC | 2d822a4 | 0, 1, 2, 3 | 4 (rerun queued/running 2026-11-23) |

The other part a/b cells, all GAT/GCN/R-GCN/HAN on mag-year/pdns and HAN/R-GCN on oag-chem, failed with exit 2 on every seed. They run entirely at 0b9873d in the 2d822a4 output directory, so the run SHA differs from the directory name, but all seeds share one SHA.
Source: `logs/prov/*.jsonl` of the 2d822a4 output directory on TSUBAME. The `sha` is in each start event; ok = an end event with exit 0.
