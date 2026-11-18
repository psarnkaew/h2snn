# Card 1e F3.2 — HAN on Freebase: ranges from the F0 evidence (no ladder run)

Author decision 2026-11-18 ("1e: a, a, b", second correction). No GPU job, no HAN model code change.

| Cell | Grid before | Range now | Evidence | Status |
|---|---|---|---|---|
| HAN NC `freebase` | `num_heads` {4, 8}, `hidden_dim` {64, 128, 256} | `num_heads` **{4}**, `hidden_dim` {64, 128, 256} (full) | heads 8: GPU memory exhausted in DGL `edge_softmax`, 1.01 GiB free before the last of the 7 meta-path graphs (140.0M edges) (`p6_f0_han_nc_cuda_mem.txt`, job 17160) | heads 4 fitting is **DERIVED, not measured** (halves the edge × head tensors) |
| HAN LP `freebase_bb` | `num_heads` {4, 8} | **N/A** | CUDA OOM at heads 4 on a 24 GB GPU (8 graphs, 202.6M edges; `p6_f0_han_lp.txt`, job 17156) | reason: "OOM on a 24 GB GPU at num_heads 4, the grid minimum; the width is a secondary memory term (F0 item 3)" |

Where it is recorded:
- `H2S-NN/src/baselines/adapters/hgb_han.py`: `search_space("node_classification", "freebase")` sets `num_heads` to `[4]`. `freebase_bb` is removed from `SUPPORTED_LP_DATASETS`.
- `H2S-NN/src/baselines/registry.py`: `HAN_FREEBASE_LP_NA` is the HAN row's `unsupported` entry for (`link_prediction`, `freebase_bb`).
- `H2S-NN/tests/test_baselines.py`: the count of runnable LP cells goes from 76 to 75.

If a heads-4 NC trial OOMs on TSUBAME, it is recorded and reported (no A6000; any further step-down is a design-session decision).
