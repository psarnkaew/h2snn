# Card 2 D2 — dry run on real data: summary

One epoch per cell at the largest memory point of its search space plus one validation pass (card 2 Rule), W&B project
`h2snn-dryrun` group `d2`. Full table: `cells.md` / `cells.csv` (one row per cell and attempt; no metric value).
Driver `d2_dryrun.py`; array `d2_array.sbatch`; workers inside the author's allocation 16795 `d2_alloc_worker.sh`,
`d2_alloc_worker2.sh` (claim files keep two workers off the same cell); lists `d2_cells_*.txt`; search spaces
`d2_spaces.py` / `d2_spaces.json`; table `d2_table.py`. Records and logs (with the trainers' console lines) stay in
`/mnt/data/peeradon/dryrun_scratch` (outside the repository). Code at `e419370` (D1) for every run.
Five records were written by hand from `sacct` and the job log where the process was killed before it could write one
(`record_source` field): h-pokec (host OOM), han / h2gformer / hetero2net Freebase NC (segfault, host OOM ×2), han
freebase_bb step-down (segfault).

## Counts (163 cells: 149 HGB / LP / GC + 14 H2GB NC; status at the largest point on a 24 GB GPU)

| status | H2S-NN | baselines | total |
|---|---|---|---|
| OK | 14 | 104 | 118 |
| OOM (GPU) | 5 | 23 | 28 |
| host OOM | 1 (h-pokec, 30G) | 2 (h2gformer, hetero2net Freebase NC, 24G) | 3 |
| stop condition | 2 (enzymes, nci1: log-weight range > 30 at epoch 0) | – | 2 |
| blocked: baseline code (session 1) | – | 12 | 12 |

PCG: no solve reached `T_max` (`pcg_not_converged` 0 in every H2S-NN record); max iterations per solve 7–15.
Large-LP cells (`ogbn-mag_pp`, `h-pokec_uu`) are out of D1–D4 (design-session review).

## Vendor commits (`d2_vendor_commits.tsv`)
All 16 copied clones are at their `VENDORS.tsv` commit. Changes in the clones: H2GB has the author's 3 known
modifications (`ieee_cis_dataset.py`, `pokec_dataset.py`, `examples/use_dataset.py`); HGB has an untracked
`CLONED_COMMIT.txt`. Nothing else.

## OOM at 24 GB → one memory axis one grid step lower (author, 2026-11-16)
The A6000 is held by another user (2-day job), so the 24-cell and 4-cell A6000 retries (16859, 16883) were cancelled.
Instead each OOM cell ran once at its largest point with one axis one grid step lower, on a TITAN RTX
(`d2_cells_stepdown_24gb.txt`; the driver refuses any value that is not the next grid value). Axis per method, from the OOM
frame: H2S-NN `r` (`wisp_walks_per_anchor` 64→32: number of cells); NSD / CSNN stalk `d` (sheaf Laplacian size);
HINormer `len_seq` (score tensor is quadratic in it, `HINormer/model.py:147`); GAT / HAN `num_heads`; PSHGCN `K`
(polynomial terms); H2GFormer `metapath_pe` True→False where the OOM is in the MetaPath2Vec pre-step
(`_h2gformer_encoders.py:211`; its width is `min(128, hidden/2)`, so `hidden_dim` 512→256 does not change it: tried once on
lastfm, still OOM, kept as attempt `24GB-stepdown-hidden`), `hidden_dim` 512→256 on freebase_bb (OOM in `gt_layer.py:646`).

| result | cells |
|---|---|
| fits at 24 GB (16) | H2S-NN dblp_ap, pubmed, lastfm (r ≤ 32); H2GFormer acm, dblp, imdb, acm_pa, dblp_ap, imdb_ma, lastfm (metapath_pe off), freebase_bb (hidden ≤ 256); CSNN freebase NC, freebase_bb (d ≤ 3); GAT acm_pa (heads ≤ 4); NSD pubmed (d ≤ 4); PSHGCN pubmed (K ≤ 2) |
| still OOM (11) | H2S-NN freebase NC (> 23.2 GiB + 2 MiB), freebase_bb (> 22.7 + 0.07); NSD acm, freebase NC, acm_pa, freebase_bb (d 4); HINormer acm, dblp, imdb (len_seq 100); GAT freebase_bb (heads 4, > 16.2 + 6.0); PSHGCN freebase_bb (K 2, > 23.2 + 0.2) |
| error (1) | HAN freebase_bb (heads 4): segfault after `torch_sparse/matmul.py:97` (CSR `torch.sparse.mm`); at heads 8 it was a plain OOM |

Per-cell numbers: `cells.md`, section "OOM at 24 GB".

## Pending when D2 was committed (queued, not waited for)
- A6000 retry of the 5 H2S-NN OOM cells at the full point: array 16889.
- h-pokec H2S-NN at 120G host memory on g3an01: 16875.
- han / h2gformer / hetero2net Freebase NC with 40G host memory on a 3090: 16882.

## Blocked: baseline code (for session 1; card 1d unit H3b, design-docs `795ec7a`)
The 11 cells of H3b (HAN NC imdb; HINormer NC freebase; HAN and H2SGNN LP imdb_ma; H2SGNN LP freebase_bb; ChebNetII NC
ogbn-mag, mag-year, oag-cs, oag-eng, oag-chem; SeHGNN NC ieee-cis-g), not fixed here, re-run after card 1d is on `v3b`.
**New since H3b:** HAN NC freebase — segfault (core dumped) right after `vendor/HGB/NC/benchmark/methods/HAN/utils.py:213`
builds the book–book sparse feature tensor (`sacct` 16797_56: FAILED 11:0, host 2.5 GiB, 52 s); and the HAN freebase_bb
step-down segfault above. Both in sparse code on Freebase; for session 1.

## Times worth a decision (not a stop)
H2S-NN H2GB NC: one full validation pass takes 544 s (ogbn-mag), 3,394 s (mag-year), 449 s (oag-chem); test pass of
mag-year 3,069 s. At 500 epochs with `eval_period` 4 that is 125 validation passes per trial (mag-year ≈ 118 h of
validation alone per trial).
