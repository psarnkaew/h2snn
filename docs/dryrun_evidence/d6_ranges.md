# Card 2 D6.3 — OOM cells: the largest grid point that fits a 24 GB GPU (decision 1)

From D2's largest point, the axis of D2's step-down one grid step at a time; at its minimum the next memory axis (previous axes kept at their minimum); every other axis at its largest value. TITAN RTX / RTX 3090 (24 GB), no A6000. 'axis [full range] → fitting range'. HAN freebase_bb (segfault at heads 4) is session 1's (not laddered).

| cell | axis: full range → fitting range | result | GPU peak GiB | from |
|---|---|---|---|---|
| h2snn|node_classification|freebase | r [8, 16, 32, 64] → ≤ 8 | fits | 19.66 | D6.3 ladder step 2 (+ D2's step) |
| h2snn|link_prediction|dblp_ap | r [8, 16, 32, 64] → ≤ 32 | fits | 20.15 | D2 step-down (1 step) |
| h2snn|link_prediction|freebase_bb | r [8, 16, 32, 64] → ≤ 8 | fits | 19.21 | D6.3 ladder step 2 (+ D2's step) |
| h2snn|link_prediction|pubmed | r [8, 16, 32, 64] → ≤ 32 | fits | 17.41 | D2 step-down (1 step) |
| h2snn|link_prediction|lastfm | r [8, 16, 32, 64] → ≤ 32 | fits | 17.22 | D2 step-down (1 step) |
| nsd|node_classification|acm | d [2, 3, 4, 5] → ≤ 3 | fits | 16.68 | D6.3 ladder step 1 (+ D2's step) |
| hinormer|node_classification|acm | len_seq [20, 30, 50, 80, 100, 150] → ≤ 80 | fits | 17.82 | D6.3 ladder step 1 (+ D2's step) |
| h2gformer|node_classification|acm | metapath_pe [True, False] → False | fits | 13.93 | D2 step-down (1 step) |
| hinormer|node_classification|dblp | len_seq [20, 30, 50, 80, 100, 150] → ≤ 50 | fits | 12.94 | D6.3 ladder step 2 (+ D2's step) |
| h2gformer|node_classification|dblp | metapath_pe [True, False] → False | fits | 7.89 | D2 step-down (1 step) |
| hinormer|node_classification|imdb | len_seq [20, 30, 50, 80, 100, 150] → ≤ 50 | fits | 15.57 | D6.3 ladder step 2 (+ D2's step) |
| h2gformer|node_classification|imdb | metapath_pe [True, False] → False | fits | 4.38 | D2 step-down (1 step) |
| nsd|node_classification|freebase | d [2, 3, 4, 5] → ≤ 2; hidden_channels [32, 64] → ≤ 32 | fits | 16.69 | D6.3 ladder step 3 (+ D2's step) |
| csnn|node_classification|freebase | d [2, 3, 4] → ≤ 3 | fits | 19.26 | D2 step-down (1 step) |
| gat|link_prediction|acm_pa | num_heads [4, 8] → ≤ 4 | fits | 11.06 | D2 step-down (1 step) |
| nsd|link_prediction|acm_pa | d [2, 3, 4, 5] → ≤ 3 | fits | 16.69 | D6.3 ladder step 1 (+ D2's step) |
| h2gformer|link_prediction|acm_pa | metapath_pe [True, False] → False | fits | 13.76 | D2 step-down (1 step) |
| h2gformer|link_prediction|dblp_ap | metapath_pe [True, False] → False | fits | 7.71 | D2 step-down (1 step) |
| h2gformer|link_prediction|imdb_ma | metapath_pe [True, False] → False | fits | 3.74 | D2 step-down (1 step) |
| gat|link_prediction|freebase_bb | num_heads [4, 8] → ≤ 4; hidden_dim [64, 128, 256] → ≤ 128 | fits | 17.52 | D6.3 ladder step 1 (+ D2's step) |
| nsd|link_prediction|freebase_bb | d [2, 3, 4, 5] → ≤ 2; hidden_channels [32, 64] → ≤ 32 | fits | 16.72 | D6.3 ladder step 3 (+ D2's step) |
| csnn|link_prediction|freebase_bb | d [2, 3, 4] → ≤ 3 | fits | 19.35 | D2 step-down (1 step) |
| han|link_prediction|freebase_bb | num_heads [4, 8] → ≤ 4 | error (no D6.3 ladder: blocked: baseline code (for session 1) at ) |  | D2 step-down |
| pshgcn|link_prediction|freebase_bb | K [1, 2, 3] → ≤ 1; hidden [128, 256] → ≤ 128 | fits | 13.65 | D6.3 ladder step 2 (+ D2's step) |
| h2gformer|link_prediction|freebase_bb | hidden_dim [64, 128, 256, 512] → ≤ 256 | fits | 18.75 | D2 step-down (1 step) |
| nsd|link_prediction|pubmed | d [2, 3, 4, 5] → ≤ 4 | fits | 19.73 | D2 step-down (1 step) |
| pshgcn|link_prediction|pubmed | K [1, 2, 3] → ≤ 2 | fits | 10.61 | D2 step-down (1 step) |
| h2gformer|link_prediction|lastfm | metapath_pe [True, False] → False | fits | 3.95 | D2 step-down (1 step) |

Not laddered (no grid axis changes the failing allocation, or not a memory failure):

- **H2S-NN Freebase NC**: the card's first step, mini-batch at B 1024, is refused by the config (`src/configs/config.py:1088-1094`: training mode fixed per dataset, mini-batch outside H2GB only as the acm/dblp analysis flag); spec 1.17's "Freebase mini-batch if full batch does not fit" is not implemented. **Stop item** (a new setting). The full-batch ladder above (r ≤ 8) is information only.
- **H²GFormer, Hetero²Net Freebase NC**: D2 retry at 40G host (16882_2, 16882_3, RTX 3090): host fits (peak 35.3 GiB), CUDA OOM moving the graph to the GPU (one 25.27 GiB tensor, `torch_geometric/data/data.py:364` from `h2gformer.py:568` / `hetero2net.py:264`, before any model axis applies): no 24 GB grid point fits.
- **HAN Freebase NC**: segmentation fault again at 40G host (16882_1, vendor HGB NC): session 1's (card 1e).
- **HAN freebase_bb LP**: segmentation fault at heads 4 (D2 step-down): session 1's (card 1e).
