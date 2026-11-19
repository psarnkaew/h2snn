| Cell | hidden_channels | d | layers | Status | Peak GPU GiB | Site (OOM: tried to allocate) | GPU, node, job |
|---|---|---|---|---|---|---|---|
| acm | 16 | 5 | 8 | OOM | 21.72 | `src/baselines/training.py:89` (1.78 GiB) | RTX 3090, g3n01, 17118 |
| acm | 8 | 5 | 8 | OK | 12.82 |  | RTX 3090, g3n01, 17118 |
| freebase | 16 | 5 | 8 | OOM | 22.75 | `models/sheaf_gnn/transductive/disc_models.py:560` (2.61 GiB) | RTX 3090, g3n01, 17119 |
| freebase | 8 | 5 | 8 | OK | 19.64 |  | RTX 3090, g3n01, 17119 |
| acm_pa | 16 | 5 | 8 | OOM | 21.69 | `src/baselines/training.py:239` (1.77 GiB) | RTX 3090, g3n01, 17120 |
| acm_pa | 8 | 5 | 8 | OK | 12.85 |  | RTX 3090, g3n01, 17120 |
| freebase_bb | 16 | 5 | 8 | OOM | 22.6 | `models/sheaf_gnn/transductive/disc_models.py:560` (2.54 GiB) | RTX 3090, g3n01, 17121 |
| freebase_bb | 8 | 5 | 8 | OK | 19.59 |  | RTX 3090, g3n01, 17121 |
| pubmed | 16 | 5 | 8 | OK | 12.22 |  | RTX 3090, g3n01, 17122 |
| ogbn-mag_pp | 16 | 5 | 8 | OOM | 18.95 | `models/sheaf_gnn/sheaf_models.py:1236` (16.88 GiB) | RTX 3090, g3n01, 17123 |
| ogbn-mag_pp | 8 | 5 | 8 | OOM | 21.14 | `models/sheaf_gnn/sheaf_models.py:1237` (8.44 GiB) | RTX 3090, g3n01, 17123 |
| ogbn-mag_pp | 8 | 4 | 8 | OOM | 17.17 | `models/sheaf_gnn/sheaf_models.py:1237` (7.23 GiB) | RTX 3090, g3n01, 17123 |
| ogbn-mag_pp | 8 | 3 | 8 | OOM | 19.81 | `models/sheaf_gnn/sheaf_models.py:1238` (12.06 GiB) | RTX 3090, g3n01, 17123 |
| ogbn-mag_pp | 8 | 2 | 8 | OOM | 15.81 | `models/sheaf_gnn/sheaf_models.py:1238` (9.65 GiB) | RTX 3090, g3n01, 17123 |
| ogbn-mag_pp | 8 | 2 | 7 | OOM | 15.81 | `models/sheaf_gnn/sheaf_models.py:1238` (9.65 GiB) | RTX 3090, g3n01, 17123 |
| ogbn-mag_pp | 8 | 2 | 6 | OOM | 15.81 | `models/sheaf_gnn/sheaf_models.py:1238` (9.65 GiB) | RTX 3090, g3n01, 17123 |
| ogbn-mag_pp | 8 | 2 | 5 | OOM | 15.81 | `models/sheaf_gnn/sheaf_models.py:1238` (9.65 GiB) | RTX 3090, g3n01, 17123 |
| ogbn-mag_pp | 8 | 2 | 4 | OOM | 15.81 | `models/sheaf_gnn/sheaf_models.py:1238` (9.65 GiB) | RTX 3090, g3n01, 17123 |
| ogbn-mag_pp | 8 | 2 | 3 | OOM | 15.81 | `models/sheaf_gnn/sheaf_models.py:1238` (9.65 GiB) | RTX 3090, g3n01, 17123 |
| ogbn-mag_pp | 8 | 2 | 2 | OOM | 15.81 | `models/sheaf_gnn/sheaf_models.py:1238` (9.65 GiB) | RTX 3090, g3n01, 17123 |
| h-pokec_uu | 16 | 5 | 8 | error | 12.13 | `models/sheaf_gnn/transductive/disc_models.py:571` AssertionError:  | RTX 3090, g3n01, 17124 |
