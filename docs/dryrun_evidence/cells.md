# Card 2 D2 — dry run, one row per cell and attempt

Status per cell at the largest point on a 24 GB GPU: OK 118, OOM 28, blocked: baseline code 12, host OOM 3, stop condition 2; cells 163.

Times in seconds; memory in GiB (GPU: `torch.cuda.max_memory_allocated` / reserved; host: peak RSS of the process). H2S-NN: train = one epoch (draw + steps), val = one full validation pass; PCG = max iterations of any scope in any solve (fwd / bwd in training, fwd in validation). Baselines: train = from the end of the bundle build to the first `.eval()` (model construction + the epoch), val = from there to the epoch report (validation; and test where the loop evaluates it there). No metric value is recorded.

| cell | attempt | status | note / error (first file:line) | GPU (node) | GPU peak | host peak | build | train | val | PCG tr f/b, val | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h2snn|node_classification|acm | 24GB | OK |  | TITAN RTX (g2tn01) | 7.26 | 1.68 | 3.43 | 4.95 | 1.04 | 7/6, 9 |  |
| h2snn|node_classification|dblp | 24GB | OK |  | TITAN RTX (g2tn01) | 9.83 | 2.02 | 6.02 | 5.74 | 1.50 | 8/7, 11 |  |
| h2snn|node_classification|imdb | 24GB | OK |  | TITAN RTX (g2tn01) | 7.40 | 1.88 | 5.36 | 5.40 | 1.26 | 9/8, 12 |  |
| h2snn|node_classification|freebase | 24GB | OOM | needs > 21.84 GiB allocated + 704.00 MiB requested; `src/models/sheaf_energy.py:173 (down)` CUDA out of memory. Tried to allocate 704.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 295.56 MiB is free. I | TITAN RTX (g2tn01) | 21.84 | 13.91 | 32.34 | 32.57 |  |  |  |
| h2snn|node_classification|freebase | 24GB-stepdown | OOM | needs > 23.23 GiB allocated + 2.00 MiB requested; `src/models/map_operators.py:198 (_rotate)` CUDA out of memory. Tried to allocate 2.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 3.31 MiB is free. Inclu | TITAN RTX (g2tn01) | 23.23 | 7.47 | 16.93 | 27.66 |  | 8/8,  |  |
| h2snn|link_prediction|acm_pa | 24GB | OK |  | TITAN RTX (g2tn01) | 15.90 | 1.87 | 8.33 | 9.80 | 2.63 | 8/7, 11 |  |
| h2snn|link_prediction|dblp_ap | 24GB | OOM | needs > 23.15 GiB allocated + 66.00 MiB requested; `src/models/map_operators.py:213 (_reflect)` CUDA out of memory. Tried to allocate 66.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 17.56 MiB is free. Inc | TITAN RTX (g2tn01) | 23.15 | 4.34 | 19.79 | 19.50 |  | 10/9,  |  |
| h2snn|link_prediction|dblp_ap | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 20.15 | 3.90 | 15.61 | 17.60 | 3.49 | 9/9, 12 |  |
| h2snn|link_prediction|imdb_ma | 24GB | OK |  | TITAN RTX (g2tn01) | 12.78 | 2.52 | 11.13 | 13.38 | 2.42 | 10/9, 13 |  |
| h2snn|link_prediction|freebase_bb | 24GB | OOM | needs > 21.40 GiB allocated + 816.00 MiB requested; `src/models/map_operators.py:523 (adjoint_prefix)` CUDA out of memory. Tried to allocate 816.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 799.56 MiB is free. I | TITAN RTX (g2tn01) | 21.40 | 14.11 | 37.40 | 32.21 |  |  |  |
| h2snn|link_prediction|freebase_bb | 24GB-stepdown | OOM | needs > 22.70 GiB allocated + 68.00 MiB requested; `src/models/map_operators.py:213 (_reflect)` CUDA out of memory. Tried to allocate 68.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 69.31 MiB is free. Inc | TITAN RTX (g2tn01) | 22.70 | 7.58 | 22.85 | 26.42 |  | 8/7,  |  |
| h2snn|link_prediction|pubmed | 24GB | OOM | needs > 22.92 GiB allocated + 56.00 MiB requested; `src/models/map_operators.py:201 (_rotate)` CUDA out of memory. Tried to allocate 56.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 41.56 MiB is free. Inc | TITAN RTX (g2tn01) | 22.92 | 3.60 | 14.64 | 18.12 |  | 9/8,  |  |
| h2snn|link_prediction|pubmed | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 17.41 | 2.47 | 10.05 | 29.18 | 3.19 | 9/8, 12 |  |
| h2snn|link_prediction|lastfm | 24GB | OOM | needs > 22.96 GiB allocated + 60.00 MiB requested; `src/models/map_operators.py:213 (_reflect)` CUDA out of memory. Tried to allocate 60.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 49.56 MiB is free. Inc | TITAN RTX (g2tn01) | 22.96 | 2.16 | 10.37 | 16.67 |  | 9/8,  |  |
| h2snn|link_prediction|lastfm | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 17.22 | 1.62 | 6.53 | 57.41 | 3.56 | 10/8, 14 |  |
| h2snn|graph_classification|mutag | 24GB | OK |  | TITAN RTX (g2tn01) | 0.05 | 1.54 | 2.24 | 143.74 | 8.87 | 12/11, 14 |  |
| h2snn|graph_classification|proteins | 24GB | OK |  | TITAN RTX (g2tn01) | 0.90 | 1.60 | 2.73 | 596.81 | 35.60 | 13/11, 9 |  |
| h2snn|graph_classification|enzymes | 24GB | stop condition | `src/models/h2snn_model.py:740 (_encode_anchored)` stop condition log_weight_range: 30.1465 > 30 at epoch 0 (spec v1.15 §6) | TITAN RTX (g2tn01) | 0.12 | 1.48 | 2.74 | 131.60 |  | 14/12,  | {"condition": "log_weight_range", "value": 30.146507263183594, "limit": 30.0, "epoch": 0.0, "class": "LogWeightRangeStopError"} |
| h2snn|graph_classification|nci1 | 24GB | stop condition | `src/models/h2snn_model.py:740 (_encode_anchored)` stop condition log_weight_range: 32.0897 > 30 at epoch 0 (spec v1.15 §6) | TITAN RTX (g2tn01) | 0.07 | 1.51 | 3.87 | 955.82 |  | 11/11,  | {"condition": "log_weight_range", "value": 32.089717864990234, "limit": 30.0, "epoch": 0.0, "class": "LogWeightRangeStopError"} |
| gcn|node_classification|acm | 24GB | OK |  | TITAN RTX (g2tn01) | 0.14 | 1.59 | 4.67 | 7.76 | 0.03 |  |  |
| gat|node_classification|acm | 24GB | OK |  | TITAN RTX (g2tn01) | 1.23 | 1.65 | 4.66 | 7.56 | 0.07 |  |  |
| chebnetii|node_classification|acm | 24GB | OK |  | TITAN RTX (g2tn01) | 0.10 | 1.47 | 6.34 | 7.35 | 0.04 |  |  |
| nsd|node_classification|acm | 24GB | OOM | needs > 18.38 GiB allocated + 3.60 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch_sparse/spmm.py:28 (spmm)` CUDA out of memory. Tried to allocate 3.60 GiB. GPU 0 has a total capacity of 23.64 GiB of which 1.32 GiB is free. Inclu | TITAN RTX (g2tn01) | 18.38 | 1.33 | 4.83 |  |  |  |  |
| nsd|node_classification|acm | 24GB-stepdown | OOM | needs > 20.15 GiB allocated + 2.40 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch/autograd/graph.py:769 (_engine_run_backward)` CUDA out of memory. Tried to allocate 2.40 GiB. GPU 0 has a total capacity of 23.64 GiB of which 1.57 GiB is free. Inclu | TITAN RTX (g2tn01) | 20.15 | 1.46 | 4.64 |  |  |  |  |
| csnn|node_classification|acm | 24GB | OK |  | TITAN RTX (g2tn01) | 6.64 | 1.57 | 4.74 | 8.54 | 0.17 |  |  |
| han|node_classification|acm | 24GB | OK |  | TITAN RTX (g2tn01) | 3.35 | 1.76 | 4.67 | 13.80 | 0.16 |  |  |
| sehgnn|node_classification|acm | 24GB | OK |  | TITAN RTX (g2tn01) | 4.26 | 2.85 | 6.58 | 18.27 | 0.09 |  |  |
| hinormer|node_classification|acm | 24GB | OOM | needs > 21.10 GiB allocated + 2.03 GiB requested; `vendor/HINormer/model.py:152 (forward)` CUDA out of memory. Tried to allocate 2.03 GiB. GPU 0 has a total capacity of 23.64 GiB of which 635.56 MiB is free. Inc | TITAN RTX (g2tn01) | 21.10 | 1.34 | 5.93 |  |  |  |  |
| hinormer|node_classification|acm | 24GB-stepdown | OOM | needs > 22.44 GiB allocated + 924.00 MiB requested; `vendor/HINormer/model.py:153 (forward)` CUDA out of memory. Tried to allocate 924.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 455.31 MiB is free. I | TITAN RTX (g2tn01) | 22.44 | 1.37 | 6.02 |  |  |  |  |
| pshgcn|node_classification|acm | 24GB | OK |  | TITAN RTX (g2tn01) | 8.13 | 1.75 | 4.67 | 12.79 | 0.29 |  |  |
| h2gformer|node_classification|acm | 24GB | OOM | needs > 18.40 GiB allocated + 5.19 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch/autograd/graph.py:769 (_engine_run_backward)` CUDA out of memory. Tried to allocate 5.19 GiB. GPU 0 has a total capacity of 23.64 GiB of which 4.32 GiB is free. Inclu | TITAN RTX (g2tn01) | 18.40 | 1.60 | 4.66 |  |  |  |  |
| h2gformer|node_classification|acm | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 13.93 | 1.65 | 4.59 | 8.30 | 0.12 |  |  |
| hetero2net|node_classification|acm | 24GB | OK |  | TITAN RTX (g2tn01) | 0.60 | 1.62 | 4.74 | 9.31 | 0.05 |  |  |
| h2sgnn|node_classification|acm | 24GB | OK |  | TITAN RTX (g2tn01) | 4.30 | 1.73 | 4.71 | 12.36 | 0.06 |  |  |
| gcn|node_classification|dblp | 24GB | OK |  | TITAN RTX (g2tn01) | 0.27 | 1.83 | 3.17 | 5.92 | 0.03 |  |  |
| gat|node_classification|dblp | 24GB | OK |  | TITAN RTX (g2tn01) | 2.49 | 1.93 | 3.15 | 6.46 | 0.08 |  |  |
| chebnetii|node_classification|dblp | 24GB | OK |  | TITAN RTX (g2tn01) | 0.20 | 1.78 | 4.55 | 5.77 | 0.03 |  |  |
| nsd|node_classification|dblp | 24GB | OK |  | TITAN RTX (g2tn01) | 17.15 | 1.89 | 3.02 | 28.92 | 0.27 |  |  |
| csnn|node_classification|dblp | 24GB | OK |  | TITAN RTX (g2tn01) | 4.07 | 1.90 | 2.97 | 6.49 | 0.10 |  |  |
| han|node_classification|dblp | 24GB | OK |  | TITAN RTX (g2tn01) | 2.54 | 4.09 | 3.06 | 19.32 | 0.13 |  |  |
| sehgnn|node_classification|dblp | 24GB | OK |  | TITAN RTX (g2tn01) | 1.55 | 4.09 | 4.85 | 22.78 | 0.06 |  |  |
| hinormer|node_classification|dblp | 24GB | OOM | needs > 20.08 GiB allocated + 2.72 GiB requested; `vendor/HINormer/model.py:153 (forward)` CUDA out of memory. Tried to allocate 2.72 GiB. GPU 0 has a total capacity of 23.64 GiB of which 1.47 GiB is free. Inclu | TITAN RTX (g2tn01) | 20.08 | 1.63 | 4.35 |  |  |  |  |
| hinormer|node_classification|dblp | 24GB-stepdown | OOM | needs > 21.75 GiB allocated + 1.21 GiB requested; `vendor/HINormer/model.py:152 (forward)` CUDA out of memory. Tried to allocate 1.21 GiB. GPU 0 has a total capacity of 23.64 GiB of which 991.31 MiB is free. Inc | TITAN RTX (g2tn01) | 21.75 | 1.63 | 4.33 |  |  |  |  |
| pshgcn|node_classification|dblp | 24GB | OK |  | TITAN RTX (g2tn01) | 6.86 | 4.09 | 3.04 | 17.78 | 0.17 |  |  |
| h2gformer|node_classification|dblp | 24GB | OOM | needs > 18.57 GiB allocated + 5.19 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch/autograd/graph.py:769 (_engine_run_backward)` CUDA out of memory. Tried to allocate 5.19 GiB. GPU 0 has a total capacity of 23.64 GiB of which 4.18 GiB is free. Inclu | TITAN RTX (g2tn01) | 18.57 | 1.86 | 3.02 |  |  |  |  |
| h2gformer|node_classification|dblp | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 7.89 | 1.89 | 3.01 | 8.65 | 0.11 |  |  |
| hetero2net|node_classification|dblp | 24GB | OK |  | TITAN RTX (g2tn01) | 1.25 | 1.88 | 3.07 | 9.29 | 0.07 |  |  |
| h2sgnn|node_classification|dblp | 24GB | OK |  | TITAN RTX (g2tn01) | 1.43 | 4.10 | 3.07 | 16.64 | 0.06 |  |  |
| gcn|node_classification|imdb | 24GB | OK |  | TITAN RTX (g2tn01) | 0.21 | 1.69 | 1.18 | 5.55 | 0.03 |  |  |
| gat|node_classification|imdb | 24GB | OK |  | TITAN RTX (g2tn01) | 2.01 | 1.75 | 1.15 | 5.69 | 0.08 |  |  |
| chebnetii|node_classification|imdb | 24GB | OK |  | TITAN RTX (g2tn01) | 0.16 | 1.60 | 2.47 | 4.89 | 0.04 |  |  |
| nsd|node_classification|imdb | 24GB | OK |  | TITAN RTX (g2tn01) | 7.18 | 1.71 | 1.13 | 13.60 | 0.15 |  |  |
| csnn|node_classification|imdb | 24GB | OK |  | TITAN RTX (g2tn01) | 2.13 | 1.75 | 1.23 | 5.55 | 0.07 |  |  |
| han|node_classification|imdb | 24GB | error | blocked: baseline code (for session 1) at src/baselines/adapters/hgb_han.py:153 (fit_task); `src/baselines/adapters/hgb_han.py:153 (fit_task)` too many values to unpack (expected 11) | TITAN RTX (g2tn01) | 0.00 | 3.18 | 1.18 |  |  |  |  |
| sehgnn|node_classification|imdb | 24GB | OK |  | TITAN RTX (g2tn01) | 5.70 | 4.53 | 2.99 | 21.14 | 0.10 |  |  |
| hinormer|node_classification|imdb | 24GB | OOM | needs > 20.94 GiB allocated + 3.31 GiB requested; `vendor/HINormer/model.py:152 (forward)` CUDA out of memory. Tried to allocate 3.31 GiB. GPU 0 has a total capacity of 23.64 GiB of which 183.56 MiB is free. Inc | TITAN RTX (g2tn01) | 20.94 | 1.56 | 2.50 |  |  |  |  |
| hinormer|node_classification|imdb | 24GB-stepdown | OOM | needs > 21.93 GiB allocated + 1.47 GiB requested; `vendor/HINormer/model.py:153 (forward)` CUDA out of memory. Tried to allocate 1.47 GiB. GPU 0 has a total capacity of 23.64 GiB of which 803.31 MiB is free. Inc | TITAN RTX (g2tn01) | 21.93 | 1.56 | 2.41 |  |  |  |  |
| pshgcn|node_classification|imdb | 24GB | OK |  | TITAN RTX (g2tn01) | 5.75 | 3.18 | 1.17 | 17.26 | 0.14 |  |  |
| h2gformer|node_classification|imdb | 24GB | OOM | needs > 18.72 GiB allocated + 5.19 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch/autograd/graph.py:769 (_engine_run_backward)` CUDA out of memory. Tried to allocate 5.19 GiB. GPU 0 has a total capacity of 23.64 GiB of which 4.65 GiB is free. Inclu | TITAN RTX (g2tn01) | 18.72 | 1.71 | 1.14 |  |  |  |  |
| h2gformer|node_classification|imdb | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 4.38 | 1.74 | 1.15 | 8.35 | 0.09 |  |  |
| hetero2net|node_classification|imdb | 24GB | OK |  | TITAN RTX (g2tn01) | 1.18 | 1.67 | 1.15 | 9.30 | 0.05 |  |  |
| h2sgnn|node_classification|imdb | 24GB | OK |  | TITAN RTX (g2tn01) | 0.50 | 3.15 | 1.14 | 15.24 | 0.03 |  |  |
| gcn|node_classification|freebase | 24GB | OK |  | TITAN RTX (g2tn01) | 1.77 | 1.45 | 7.32 | 10.31 | 0.06 |  |  |
| gat|node_classification|freebase | 24GB | OK |  | TITAN RTX (g2tn01) | 16.83 | 1.54 | 7.34 | 11.15 | 0.54 |  |  |
| chebnetii|node_classification|freebase | 24GB | OK |  | TITAN RTX (g2tn01) | 1.25 | 1.42 | 8.59 | 9.17 | 0.05 |  |  |
| nsd|node_classification|freebase | 24GB | OOM | needs > 19.18 GiB allocated + 10.45 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch_sparse/spmm.py:28 (spmm)` CUDA out of memory. Tried to allocate 10.45 GiB. GPU 0 has a total capacity of 23.64 GiB of which 1.50 GiB is free. Incl | TITAN RTX (g2tn01) | 19.18 | 1.42 | 7.15 |  |  |  |  |
| nsd|node_classification|freebase | 24GB-stepdown | OOM | needs > 14.07 GiB allocated + 6.97 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch_sparse/spmm.py:28 (spmm)` CUDA out of memory. Tried to allocate 6.97 GiB. GPU 0 has a total capacity of 23.64 GiB of which 6.80 GiB is free. Inclu | TITAN RTX (g2tn01) | 14.07 | 1.42 | 7.02 |  |  |  |  |
| csnn|node_classification|freebase | 24GB | OOM | needs > 22.33 GiB allocated + 1.38 GiB requested; `vendor/csnn/models/CSNNconv.py:143 (message)` CUDA out of memory. Tried to allocate 1.38 GiB. GPU 0 has a total capacity of 23.64 GiB of which 129.56 MiB is free. Inc | TITAN RTX (g2tn01) | 22.33 | 1.42 | 7.06 |  |  |  |  |
| csnn|node_classification|freebase | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 19.26 | 1.52 | 7.36 | 10.17 | 0.43 |  |  |
| han|node_classification|freebase | 24GB | error | blocked: baseline code (for session 1) at ; `vendor/HGB/NC/benchmark/methods/HAN/utils.py:213 (last line in the log before the fault)` process ended with a segmentation fault (core dumped) after vendor/HGB/NC/benchmark/methods/HAN/utils.py:213 built the b | TITAN RTX (g2tn01) |  | 2.48 |  |  |  |  |  |
| hinormer|node_classification|freebase | 24GB | error | blocked: baseline code (for session 1) at src/baselines/training.py:81 (target_logits); `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/dgl/nn/pytorch/conv/graphconv.py:408 (forward)` There are 0-in-degree nodes in the graph, output for those nodes will be invalid. This is harmful for some applications, | TITAN RTX (g2tn01) | 0.70 | 1.45 | 8.38 |  |  |  |  |
| h2gformer|node_classification|freebase | 24GB | host OOM |  | TITAN RTX (g2tn01) |  | 24.00 |  |  |  |  |  |
| hetero2net|node_classification|freebase | 24GB | host OOM |  | TITAN RTX (g2tn01) |  | 24.00 |  |  |  |  |  |
| gcn|link_prediction|acm_pa | 24GB | OK |  | TITAN RTX (g2tn01) | 1.35 | 1.58 | 8.96 | 7.25 | 0.07 |  |  |
| gat|link_prediction|acm_pa | 24GB | OOM | needs > 17.67 GiB allocated + 4.23 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch/autograd/graph.py:769 (_engine_run_backward)` CUDA out of memory. Tried to allocate 4.23 GiB. GPU 0 has a total capacity of 23.64 GiB of which 1.73 GiB is free. Inclu | TITAN RTX (g2tn01) | 17.67 | 1.52 | 8.91 |  |  |  |  |
| gat|link_prediction|acm_pa | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 11.06 | 1.61 | 8.88 | 7.35 | 0.11 |  |  |
| nsd|link_prediction|acm_pa | 24GB | OOM | needs > 18.38 GiB allocated + 3.57 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch_sparse/spmm.py:28 (spmm)` CUDA out of memory. Tried to allocate 3.57 GiB. GPU 0 has a total capacity of 23.64 GiB of which 1.35 GiB is free. Inclu | TITAN RTX (g2tn01) | 18.38 | 1.48 | 8.81 |  |  |  |  |
| nsd|link_prediction|acm_pa | 24GB-stepdown | OOM | needs > 20.14 GiB allocated + 2.38 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch/autograd/graph.py:769 (_engine_run_backward)` CUDA out of memory. Tried to allocate 2.38 GiB. GPU 0 has a total capacity of 23.64 GiB of which 1.59 GiB is free. Inclu | TITAN RTX (g2tn01) | 20.14 | 1.61 | 9.10 |  |  |  |  |
| csnn|link_prediction|acm_pa | 24GB | OK |  | TITAN RTX (g2tn01) | 6.73 | 1.73 | 8.90 | 7.15 | 0.19 |  |  |
| han|link_prediction|acm_pa | 24GB | OK |  | TITAN RTX (g2tn01) | 4.56 | 1.78 | 9.19 | 7.12 | 0.17 |  |  |
| pshgcn|link_prediction|acm_pa | 24GB | OK |  | TITAN RTX (g2tn01) | 1.69 | 1.68 | 9.29 | 8.08 | 0.09 |  |  |
| h2gformer|link_prediction|acm_pa | 24GB | OOM | needs > 18.36 GiB allocated + 5.19 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch/autograd/graph.py:769 (_engine_run_backward)` CUDA out of memory. Tried to allocate 5.19 GiB. GPU 0 has a total capacity of 23.64 GiB of which 4.36 GiB is free. Inclu | TITAN RTX (g2tn01) | 18.36 | 1.64 | 8.94 |  |  |  |  |
| h2gformer|link_prediction|acm_pa | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 13.76 | 1.73 | 8.91 | 8.84 | 0.14 |  |  |
| hetero2net|link_prediction|acm_pa | 24GB | OK |  | TITAN RTX (g2tn01) | 1.07 | 1.64 | 9.29 | 5.01 | 0.09 |  |  |
| h2sgnn|link_prediction|acm_pa | 24GB | OK |  | TITAN RTX (g2tn01) | 1.40 | 1.65 | 9.45 | 6.48 | 0.13 |  |  |
| gcn|link_prediction|dblp_ap | 24GB | OK |  | TITAN RTX (g2tn01) | 1.45 | 3.89 | 22.25 | 7.81 | 0.09 |  |  |
| gat|link_prediction|dblp_ap | 24GB | OK |  | TITAN RTX (g2tn01) | 11.58 | 3.77 | 22.10 | 7.65 | 0.16 |  |  |
| nsd|link_prediction|dblp_ap | 24GB | OK |  | TITAN RTX (g2tn01) | 17.16 | 3.75 | 22.31 | 30.76 | 0.31 |  |  |
| csnn|link_prediction|dblp_ap | 24GB | OK |  | TITAN RTX (g2tn01) | 4.46 | 3.87 | 17.65 | 6.95 | 0.12 |  |  |
| han|link_prediction|dblp_ap | 24GB | OK |  | TITAN RTX (g2tn01) | 9.55 | 3.89 | 17.82 | 8.69 | 0.44 |  |  |
| pshgcn|link_prediction|dblp_ap | 24GB | OK |  | TITAN RTX (g2tn01) | 2.85 | 3.77 | 20.06 | 7.35 | 0.11 |  |  |
| h2gformer|link_prediction|dblp_ap | 24GB | OOM | needs > 18.57 GiB allocated + 5.19 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch/autograd/graph.py:769 (_engine_run_backward)` CUDA out of memory. Tried to allocate 5.19 GiB. GPU 0 has a total capacity of 23.64 GiB of which 4.18 GiB is free. Inclu | TITAN RTX (g2tn01) | 18.57 | 3.87 | 20.35 |  |  |  |  |
| h2gformer|link_prediction|dblp_ap | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 7.71 | 3.75 | 18.06 | 8.07 | 0.13 |  |  |
| hetero2net|link_prediction|dblp_ap | 24GB | OK |  | TITAN RTX (g2tn01) | 2.55 | 3.87 | 18.07 | 5.39 | 0.16 |  |  |
| h2sgnn|link_prediction|dblp_ap | 24GB | OK |  | TITAN RTX (g2tn01) | 4.31 | 3.87 | 16.86 | 7.98 | 0.34 |  |  |
| gcn|link_prediction|imdb_ma | 24GB | OK |  | TITAN RTX (g2tn01) | 0.81 | 2.39 | 9.03 | 6.04 | 0.06 |  |  |
| gat|link_prediction|imdb_ma | 24GB | OK |  | TITAN RTX (g2tn01) | 5.14 | 2.39 | 8.61 | 5.88 | 0.11 |  |  |
| nsd|link_prediction|imdb_ma | 24GB | OK |  | TITAN RTX (g2tn01) | 7.15 | 2.36 | 7.72 | 12.43 | 0.17 |  |  |
| csnn|link_prediction|imdb_ma | 24GB | OK |  | TITAN RTX (g2tn01) | 2.39 | 2.37 | 7.87 | 6.31 | 0.08 |  |  |
| han|link_prediction|imdb_ma | 24GB | error | blocked: baseline code (for session 1) at src/baselines/adapters/_target_block.py:44 (target_node_range); `src/baselines/adapters/_target_block.py:44 (target_node_range)` imdb_ma: target node types ['actor', 'movie'] are not contiguous in the flattened layout, so a single-range block crop w | TITAN RTX (g2tn01) | 0.00 | 2.39 | 8.49 |  |  |  |  |
| pshgcn|link_prediction|imdb_ma | 24GB | OK |  | TITAN RTX (g2tn01) | 2.15 | 2.39 | 7.91 | 5.62 | 0.07 |  |  |
| h2gformer|link_prediction|imdb_ma | 24GB | OOM | needs > 18.44 GiB allocated + 5.19 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch/autograd/graph.py:769 (_engine_run_backward)` CUDA out of memory. Tried to allocate 5.19 GiB. GPU 0 has a total capacity of 23.64 GiB of which 4.94 GiB is free. Inclu | TITAN RTX (g2tn01) | 18.44 | 2.36 | 7.68 |  |  |  |  |
| h2gformer|link_prediction|imdb_ma | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 3.74 | 2.36 | 8.61 | 8.12 | 0.09 |  |  |
| hetero2net|link_prediction|imdb_ma | 24GB | OK |  | TITAN RTX (g2tn01) | 1.61 | 2.36 | 7.84 | 4.72 | 0.09 |  |  |
| h2sgnn|link_prediction|imdb_ma | 24GB | error | blocked: baseline code (for session 1) at src/baselines/adapters/_target_block.py:44 (target_node_range); `src/baselines/adapters/_target_block.py:44 (target_node_range)` imdb_ma: target node types ['actor', 'movie'] are not contiguous in the flattened layout, so a single-range block crop w | TITAN RTX (g2tn01) | 0.00 | 2.36 | 7.68 |  |  |  |  |
| gcn|link_prediction|freebase_bb | 24GB | OK |  | TITAN RTX (g2tn01) | 5.24 | 3.49 | 159.43 | 10.36 | 0.35 |  |  |
| gat|link_prediction|freebase_bb | 24GB | OOM | needs > 14.66 GiB allocated + 12.02 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch_geometric/nn/conv/gat_conv.py:409 (message)` CUDA out of memory. Tried to allocate 12.02 GiB. GPU 0 has a total capacity of 23.64 GiB of which 7.54 GiB is free. Incl | TITAN RTX (g2tn01) | 14.66 | 3.49 | 141.00 |  |  |  |  |
| gat|link_prediction|freebase_bb | 24GB-stepdown | OOM | needs > 16.21 GiB allocated + 6.01 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch_geometric/nn/conv/gat_conv.py:409 (message)` CUDA out of memory. Tried to allocate 6.01 GiB. GPU 0 has a total capacity of 23.64 GiB of which 2.24 GiB is free. Inclu | TITAN RTX (g2tn01) | 16.21 | 3.48 | 139.94 |  |  |  |  |
| nsd|link_prediction|freebase_bb | 24GB | OOM | needs > 19.13 GiB allocated + 10.14 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch_sparse/spmm.py:28 (spmm)` CUDA out of memory. Tried to allocate 10.14 GiB. GPU 0 has a total capacity of 23.64 GiB of which 1.68 GiB is free. Incl | TITAN RTX (g2tn01) | 19.13 | 3.46 | 141.57 |  |  |  |  |
| nsd|link_prediction|freebase_bb | 24GB-stepdown | OOM | needs > 21.18 GiB allocated + 3.99 GiB requested; `vendor/neural-sheaf-diffusion/models/sheaf_models.py:50 (forward)` CUDA out of memory. Tried to allocate 3.99 GiB. GPU 0 has a total capacity of 23.64 GiB of which 23.31 MiB is free. Incl | TITAN RTX (g2tn01) | 21.18 | 3.46 | 139.23 |  |  |  |  |
| csnn|link_prediction|freebase_bb | 24GB | OOM | needs > 22.37 GiB allocated + 1.33 GiB requested; `vendor/csnn/models/CSNNconv.py:143 (message)` CUDA out of memory. Tried to allocate 1.33 GiB. GPU 0 has a total capacity of 23.64 GiB of which 479.31 MiB is free. Inc | TITAN RTX (g2tn01) | 22.37 | 3.46 | 142.14 |  |  |  |  |
| csnn|link_prediction|freebase_bb | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 19.35 | 3.46 | 136.48 | 11.16 | 0.70 |  |  |
| han|link_prediction|freebase_bb | 24GB | OOM | needs > 22.02 GiB allocated + 1.13 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch/nn/functional.py:1677 (leaky_relu)` CUDA out of memory. Tried to allocate 1.13 GiB. GPU 0 has a total capacity of 23.64 GiB of which 459.31 MiB is free. Inc | TITAN RTX (g2tn01) | 22.02 | 6.55 | 141.59 |  |  |  |  |
| han|link_prediction|freebase_bb | 24GB-stepdown | error | blocked: baseline code (for session 1) at ; `torch_sparse/matmul.py:97 (last line in the log before the fault)` process ended with a segmentation fault (core dumped) after torch_sparse/matmul.py:97 (torch.sparse.mm on a CSR tensor)  | TITAN RTX (g2tn01) |  |  |  |  |  |  |  |
| pshgcn|link_prediction|freebase_bb | 24GB | OOM | needs > 23.22 GiB allocated + 176.00 MiB requested; `vendor/PSHGCN/hgb-lp/pshgcn.py:104 (forward)` CUDA out of memory. Tried to allocate 176.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 85.31 MiB is free. In | TITAN RTX (g2tn01) | 23.22 | 3.49 | 141.62 |  |  |  |  |
| pshgcn|link_prediction|freebase_bb | 24GB-stepdown | OOM | needs > 23.22 GiB allocated + 176.00 MiB requested; `vendor/PSHGCN/hgb-lp/pshgcn.py:104 (forward)` CUDA out of memory. Tried to allocate 176.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 85.31 MiB is free. In | TITAN RTX (g2tn01) | 23.22 | 3.49 | 140.37 |  |  |  |  |
| h2gformer|link_prediction|freebase_bb | 24GB | OOM | needs > 19.56 GiB allocated + 1.95 GiB requested; `vendor/H2GB/H2GB/layer/gt_layer.py:646 (forward)` CUDA out of memory. Tried to allocate 1.95 GiB. GPU 0 has a total capacity of 23.64 GiB of which 1.49 GiB is free. Inclu | TITAN RTX (g2tn01) | 19.56 | 3.46 | 142.77 |  |  |  |  |
| h2gformer|link_prediction|freebase_bb | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 18.75 | 3.46 | 141.69 | 713.13 | 0.49 |  |  |
| hetero2net|link_prediction|freebase_bb | 24GB | OK |  | TITAN RTX (g2tn01) | 12.33 | 3.46 | 141.54 | 6.57 | 0.60 |  |  |
| h2sgnn|link_prediction|freebase_bb | 24GB | error | blocked: baseline code (for session 1) at /mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch/cuda/__init__.py:892; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch/cuda/__init__.py:892 (synchronize)` CUDA error: an illegal memory access was encountered | TITAN RTX (g2tn01) | 0.00 | 4.25 | 141.81 |  |  |  |  |
| gcn|link_prediction|pubmed | 24GB | OK |  | TITAN RTX (g2tn01) | 1.55 | 1.81 | 21.79 | 7.01 | 0.12 |  |  |
| gat|link_prediction|pubmed | 24GB | OK |  | TITAN RTX (g2tn01) | 18.52 | 1.81 | 21.73 | 7.15 | 0.29 |  |  |
| nsd|link_prediction|pubmed | 24GB | OOM | needs > 19.68 GiB allocated + 2.70 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch_sparse/spmm.py:28 (spmm)` CUDA out of memory. Tried to allocate 2.70 GiB. GPU 0 has a total capacity of 23.64 GiB of which 1.41 GiB is free. Inclu | TITAN RTX (g2tn01) | 19.68 | 1.78 | 22.01 |  |  |  |  |
| nsd|link_prediction|pubmed | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 19.73 | 1.78 | 21.02 | 36.28 | 0.48 |  |  |
| csnn|link_prediction|pubmed | 24GB | OK |  | TITAN RTX (g2tn01) | 7.51 | 1.78 | 22.21 | 6.27 | 0.23 |  |  |
| han|link_prediction|pubmed | 24GB | OK |  | TITAN RTX (g2tn01) | 7.15 | 1.87 | 21.71 | 7.65 | 0.34 |  |  |
| pshgcn|link_prediction|pubmed | 24GB | OOM | needs > 23.35 GiB allocated + 62.00 MiB requested; `vendor/PSHGCN/hgb-lp/pshgcn.py:126 (forward)` CUDA out of memory. Tried to allocate 62.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 57.31 MiB is free. Inc | TITAN RTX (g2tn01) | 23.35 | 1.81 | 22.05 |  |  |  |  |
| pshgcn|link_prediction|pubmed | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 10.61 | 1.81 | 21.39 | 6.70 | 0.36 |  |  |
| h2gformer|link_prediction|pubmed | 24GB | OK |  | TITAN RTX (g2tn01) | 9.41 | 1.79 | 21.69 | 8.94 | 0.23 |  |  |
| hetero2net|link_prediction|pubmed | 24GB | OK |  | TITAN RTX (g2tn01) | 2.75 | 1.78 | 21.67 | 5.08 | 0.15 |  |  |
| h2sgnn|link_prediction|pubmed | 24GB | OK |  | TITAN RTX (g2tn01) | 2.60 | 1.83 | 22.02 | 6.73 | 0.33 |  |  |
| gcn|link_prediction|lastfm | 24GB | OK |  | TITAN RTX (g2tn01) | 0.66 | 1.33 | 2.23 | 5.61 | 0.13 |  |  |
| gat|link_prediction|lastfm | 24GB | OK |  | TITAN RTX (g2tn01) | 9.50 | 1.33 | 2.23 | 5.75 | 0.19 |  |  |
| nsd|link_prediction|lastfm | 24GB | OK |  | TITAN RTX (g2tn01) | 14.61 | 1.45 | 2.35 | 23.20 | 0.32 |  |  |
| csnn|link_prediction|lastfm | 24GB | OK |  | TITAN RTX (g2tn01) | 3.43 | 1.48 | 2.25 | 5.52 | 0.18 |  |  |
| han|link_prediction|lastfm | 24GB | OK |  | TITAN RTX (g2tn01) | 6.33 | 1.48 | 2.22 | 6.28 | 0.24 |  |  |
| pshgcn|link_prediction|lastfm | 24GB | OK |  | TITAN RTX (g2tn01) | 1.77 | 1.37 | 2.64 | 5.61 | 0.15 |  |  |
| h2gformer|link_prediction|lastfm | 24GB | OOM | needs > 18.36 GiB allocated + 5.19 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch/autograd/graph.py:769 (_engine_run_backward)` CUDA out of memory. Tried to allocate 5.19 GiB. GPU 0 has a total capacity of 23.64 GiB of which 5.02 GiB is free. Inclu | TITAN RTX (g2tn01) | 18.36 | 1.43 | 2.28 |  |  |  |  |
| h2gformer|link_prediction|lastfm | 24GB-stepdown-hidden | OOM | needs > 18.35 GiB allocated + 5.19 GiB requested; `/mnt/home/peeradon/.conda/envs/h2snn-b/lib/python3.12/site-packages/torch/autograd/graph.py:769 (_engine_run_backward)` CUDA out of memory. Tried to allocate 5.19 GiB. GPU 0 has a total capacity of 23.64 GiB of which 5.05 GiB is free. Inclu | TITAN RTX (g2tn01) | 18.35 | 1.40 | 2.23 |  |  |  |  |
| h2gformer|link_prediction|lastfm | 24GB-stepdown | OK |  | TITAN RTX (g2tn01) | 3.95 | 1.46 | 2.19 | 8.70 | 0.17 |  |  |
| hetero2net|link_prediction|lastfm | 24GB | OK |  | TITAN RTX (g2tn01) | 1.10 | 1.34 | 2.18 | 4.81 | 0.14 |  |  |
| h2sgnn|link_prediction|lastfm | 24GB | OK |  | TITAN RTX (g2tn01) | 1.30 | 1.40 | 2.25 | 5.62 | 0.20 |  |  |
| gcn|graph_classification|mutag | 24GB | OK |  | TITAN RTX (g2tn01) | 0.02 | 1.26 | 0.07 | 0.70 | 0.86 |  |  |
| gat|graph_classification|mutag | 24GB | OK |  | TITAN RTX (g2tn01) | 0.17 | 1.28 | 0.06 | 1.01 | 1.14 |  |  |
| nsd|graph_classification|mutag | 24GB | OK |  | TITAN RTX (g2tn01) | 0.02 | 1.47 | 0.06 | 7.26 | 1.14 |  |  |
| csnn|graph_classification|mutag | 24GB | OK |  | TITAN RTX (g2tn01) | 0.03 | 1.47 | 0.06 | 11.49 | 1.67 |  |  |
| han|graph_classification|mutag | 24GB | OK |  | TITAN RTX (g2tn01) | 0.04 | 1.33 | 0.06 | 11.42 | 1.94 |  |  |
| pshgcn|graph_classification|mutag | 24GB | OK |  | TITAN RTX (g2tn01) | 0.09 | 1.38 | 0.07 | 73.82 | 5.49 |  |  |
| h2gformer|graph_classification|mutag | 24GB | OK |  | TITAN RTX (g2tn01) | 0.19 | 1.45 | 0.06 | 9.85 | 0.28 |  |  |
| hetero2net|graph_classification|mutag | 24GB | OK |  | TITAN RTX (g2tn01) | 0.05 | 1.30 | 0.06 | 6.36 | 0.96 |  |  |
| h2sgnn|graph_classification|mutag | 24GB | OK |  | TITAN RTX (g2tn01) | 0.02 | 1.34 | 0.06 | 10.15 | 1.18 |  |  |
| gcn|graph_classification|proteins | 24GB | OK |  | TITAN RTX (g2tn01) | 0.03 | 1.28 | 0.24 | 3.10 | 1.33 |  |  |
| gat|graph_classification|proteins | 24GB | OK |  | TITAN RTX (g2tn01) | 0.28 | 1.31 | 0.25 | 4.88 | 1.30 |  |  |
| nsd|graph_classification|proteins | 24GB | OK |  | TITAN RTX (g2tn01) | 0.17 | 1.48 | 0.24 | 27.66 | 5.13 |  |  |
| csnn|graph_classification|proteins | 24GB | OK |  | TITAN RTX (g2tn01) | 0.10 | 1.49 | 0.25 | 43.81 | 4.38 |  |  |
| han|graph_classification|proteins | 24GB | OK |  | TITAN RTX (g2tn01) | 0.05 | 1.34 | 0.25 | 10.28 | 1.51 |  |  |
| pshgcn|graph_classification|proteins | 24GB | OK |  | TITAN RTX (g2tn01) | 0.03 | 1.37 | 0.25 | 10.33 | 1.27 |  |  |
| h2gformer|graph_classification|proteins | 24GB | OK |  | TITAN RTX (g2tn01) | 0.20 | 1.44 | 0.24 | 19.47 | 1.37 |  |  |
| hetero2net|graph_classification|proteins | 24GB | OK |  | TITAN RTX (g2tn01) | 0.04 | 1.30 | 0.25 | 9.42 | 1.46 |  |  |
| h2sgnn|graph_classification|proteins | 24GB | OK |  | TITAN RTX (g2tn01) | 0.03 | 1.31 | 0.24 | 11.10 | 1.93 |  |  |
| gcn|graph_classification|enzymes | 24GB | OK |  | TITAN RTX (g2tn01) | 0.02 | 1.26 | 0.14 | 1.81 | 1.25 |  |  |
| gat|graph_classification|enzymes | 24GB | OK |  | TITAN RTX (g2tn01) | 0.17 | 1.32 | 0.14 | 2.73 | 1.08 |  |  |
| nsd|graph_classification|enzymes | 24GB | OK |  | TITAN RTX (g2tn01) | 0.04 | 1.49 | 0.14 | 16.39 | 2.94 |  |  |
| csnn|graph_classification|enzymes | 24GB | OK |  | TITAN RTX (g2tn01) | 0.04 | 1.48 | 0.13 | 26.42 | 2.91 |  |  |
| han|graph_classification|enzymes | 24GB | OK |  | TITAN RTX (g2tn01) | 0.03 | 1.40 | 0.14 | 7.79 | 1.16 |  |  |
| pshgcn|graph_classification|enzymes | 24GB | OK |  | TITAN RTX (g2tn01) | 0.02 | 1.37 | 0.14 | 7.62 | 1.02 |  |  |
| h2gformer|graph_classification|enzymes | 24GB | OK |  | TITAN RTX (g2tn01) | 0.16 | 1.46 | 0.13 | 13.90 | 0.70 |  |  |
| hetero2net|graph_classification|enzymes | 24GB | OK |  | TITAN RTX (g2tn01) | 0.02 | 1.31 | 0.14 | 7.09 | 1.15 |  |  |
| h2sgnn|graph_classification|enzymes | 24GB | OK |  | TITAN RTX (g2tn01) | 0.02 | 1.30 | 0.13 | 8.02 | 1.10 |  |  |
| gcn|graph_classification|nci1 | 24GB | OK |  | TITAN RTX (g2tn01) | 0.02 | 1.28 | 0.93 | 11.17 | 2.51 |  |  |
| gat|graph_classification|nci1 | 24GB | OK |  | TITAN RTX (g2tn01) | 0.17 | 1.31 | 0.93 | 17.32 | 2.86 |  |  |
| nsd|graph_classification|nci1 | 24GB | OK |  | TITAN RTX (g2tn01) | 0.04 | 1.49 | 0.93 | 66.18 | 11.05 |  |  |
| csnn|graph_classification|nci1 | 24GB | OK |  | TITAN RTX (g2tn01) | 0.04 | 1.55 | 0.92 | 147.15 | 14.93 |  |  |
| han|graph_classification|nci1 | 24GB | OK |  | TITAN RTX (g2tn01) | 0.03 | 1.42 | 0.95 | 25.19 | 3.58 |  |  |
| pshgcn|graph_classification|nci1 | 24GB | OK |  | TITAN RTX (g2tn01) | 0.02 | 1.40 | 0.96 | 23.05 | 2.66 |  |  |
| h2gformer|graph_classification|nci1 | 24GB | OK |  | TITAN RTX (g2tn01) | 0.16 | 1.44 | 0.93 | 47.70 | 4.72 |  |  |
| hetero2net|graph_classification|nci1 | 24GB | OK |  | TITAN RTX (g2tn01) | 0.02 | 1.34 | 0.94 | 22.90 | 3.12 |  |  |
| h2sgnn|graph_classification|nci1 | 24GB | OK |  | TITAN RTX (g2tn01) | 0.02 | 1.29 | 0.91 | 29.89 | 3.13 |  |  |
| h2snn|node_classification|ogbn-mag | 24GB | OK |  | TITAN RTX (g2tn01) | 4.36 | 10.30 | 40.15 | 97.92 | 543.56 | 7/8, 10 |  |
| h2snn|node_classification|mag-year | 24GB | OK |  | TITAN RTX (g2tn01) | 4.44 | 20.60 | 36.94 | 71.30 | 3393.97 | 7/8, 10 |  |
| h2snn|node_classification|oag-cs | 24GB | OK |  | TITAN RTX (g2tn01) | 6.03 | 10.52 | 46.62 | 73.73 | 137.24 | 7/8, 9 |  |
| h2snn|node_classification|oag-eng | 24GB | OK |  | TITAN RTX (g2tn01) | 5.58 | 7.37 | 39.42 | 50.49 | 154.83 | 7/8, 10 |  |
| h2snn|node_classification|oag-chem | 24GB | OK |  | TITAN RTX (g2tn01) | 5.98 | 14.93 | 70.50 | 92.14 | 449.39 | 7/8, 10 |  |
| h2snn|node_classification|ieee-cis-g | 24GB | OK |  | TITAN RTX (g2tn01) | 8.75 | 10.31 | 35.94 | 38.20 | 77.33 | 7/7, 9 |  |
| h2snn|node_classification|h-pokec | 24GB | host OOM |  | TITAN RTX (g2tn01) |  | 30.00 |  |  |  |  |  |
| h2snn|node_classification|pdns | 24GB | OK |  | TITAN RTX (g2tn01) | 3.53 | 14.32 | 55.68 | 99.18 | 78.98 | 12/11, 15 |  |
| chebnetii|node_classification|ogbn-mag | 24GB | error | blocked: baseline code (for session 1) at src/baselines/bundles.py:375 (_base_config); `src/configs/config.py:1023 (resolved)` data.batch_size (B) is required in mini-batch training. | TITAN RTX (g2tn01) | 0.00 | 0.65 | 0.00 |  |  |  |  |
| chebnetii|node_classification|mag-year | 24GB | error | blocked: baseline code (for session 1) at src/baselines/bundles.py:375 (_base_config); `src/configs/config.py:1023 (resolved)` data.batch_size (B) is required in mini-batch training. | TITAN RTX (g2tn01) | 0.00 | 0.65 | 0.00 |  |  |  |  |
| chebnetii|node_classification|oag-cs | 24GB | error | blocked: baseline code (for session 1) at src/baselines/bundles.py:375 (_base_config); `src/configs/config.py:1023 (resolved)` data.batch_size (B) is required in mini-batch training. | TITAN RTX (g2tn01) | 0.00 | 0.65 | 0.00 |  |  |  |  |
| chebnetii|node_classification|oag-eng | 24GB | error | blocked: baseline code (for session 1) at src/baselines/bundles.py:375 (_base_config); `src/configs/config.py:1023 (resolved)` data.batch_size (B) is required in mini-batch training. | TITAN RTX (g2tn01) | 0.00 | 0.65 | 0.00 |  |  |  |  |
| chebnetii|node_classification|oag-chem | 24GB | error | blocked: baseline code (for session 1) at src/baselines/bundles.py:375 (_base_config); `src/configs/config.py:1023 (resolved)` data.batch_size (B) is required in mini-batch training. | TITAN RTX (g2tn01) | 0.00 | 0.65 | 0.00 |  |  |  |  |
| sehgnn|node_classification|ieee-cis-g | 24GB | error | blocked: baseline code (for session 1) at src/baselines/bundles.py:375 (_base_config); `src/configs/config.py:1023 (resolved)` data.batch_size (B) is required in mini-batch training. | TITAN RTX (g2tn01) | 0.00 | 0.68 | 0.00 |  |  |  |  |

## OOM at 24 GB: one memory axis one grid step lower (author, 2026-11-16)

Largest point of the cell's search space with ONE axis one grid step lower (the axis that dominates memory in the OOM trace); everything else unchanged. A measurement for the author's decision, not a restriction of the space.

| cell | axis stepped down | result at 24 GB | GPU peak | GPU (node) |
|---|---|---|---|---|
| h2snn|node_classification|freebase | data.wisp_walks_per_anchor = 32 | still OOM (needs > 23.23 GiB allocated + 2.00 MiB requested) | 23.23 | TITAN RTX (g2tn01) |
| h2snn|link_prediction|dblp_ap | data.wisp_walks_per_anchor = 32 | fits at 24 GB with data.wisp_walks_per_anchor <= 32 | 20.15 | TITAN RTX (g2tn01) |
| h2snn|link_prediction|freebase_bb | data.wisp_walks_per_anchor = 32 | still OOM (needs > 22.70 GiB allocated + 68.00 MiB requested) | 22.70 | TITAN RTX (g2tn01) |
| h2snn|link_prediction|pubmed | data.wisp_walks_per_anchor = 32 | fits at 24 GB with data.wisp_walks_per_anchor <= 32 | 17.41 | TITAN RTX (g2tn01) |
| h2snn|link_prediction|lastfm | data.wisp_walks_per_anchor = 32 | fits at 24 GB with data.wisp_walks_per_anchor <= 32 | 17.22 | TITAN RTX (g2tn01) |
| nsd|node_classification|acm | d = 4 | still OOM (needs > 20.15 GiB allocated + 2.40 GiB requested) | 20.15 | TITAN RTX (g2tn01) |
| hinormer|node_classification|acm | len_seq = 100 | still OOM (needs > 22.44 GiB allocated + 924.00 MiB requested) | 22.44 | TITAN RTX (g2tn01) |
| h2gformer|node_classification|acm | metapath_pe = False | fits at 24 GB with metapath_pe = False | 13.93 | TITAN RTX (g2tn01) |
| hinormer|node_classification|dblp | len_seq = 100 | still OOM (needs > 21.75 GiB allocated + 1.21 GiB requested) | 21.75 | TITAN RTX (g2tn01) |
| h2gformer|node_classification|dblp | metapath_pe = False | fits at 24 GB with metapath_pe = False | 7.89 | TITAN RTX (g2tn01) |
| hinormer|node_classification|imdb | len_seq = 100 | still OOM (needs > 21.93 GiB allocated + 1.47 GiB requested) | 21.93 | TITAN RTX (g2tn01) |
| h2gformer|node_classification|imdb | metapath_pe = False | fits at 24 GB with metapath_pe = False | 4.38 | TITAN RTX (g2tn01) |
| nsd|node_classification|freebase | d = 4 | still OOM (needs > 14.07 GiB allocated + 6.97 GiB requested) | 14.07 | TITAN RTX (g2tn01) |
| csnn|node_classification|freebase | d = 3 | fits at 24 GB with d <= 3 | 19.26 | TITAN RTX (g2tn01) |
| gat|link_prediction|acm_pa | num_heads = 4 | fits at 24 GB with num_heads <= 4 | 11.06 | TITAN RTX (g2tn01) |
| nsd|link_prediction|acm_pa | d = 4 | still OOM (needs > 20.14 GiB allocated + 2.38 GiB requested) | 20.14 | TITAN RTX (g2tn01) |
| h2gformer|link_prediction|acm_pa | metapath_pe = False | fits at 24 GB with metapath_pe = False | 13.76 | TITAN RTX (g2tn01) |
| h2gformer|link_prediction|dblp_ap | metapath_pe = False | fits at 24 GB with metapath_pe = False | 7.71 | TITAN RTX (g2tn01) |
| h2gformer|link_prediction|imdb_ma | metapath_pe = False | fits at 24 GB with metapath_pe = False | 3.74 | TITAN RTX (g2tn01) |
| gat|link_prediction|freebase_bb | num_heads = 4 | still OOM (needs > 16.21 GiB allocated + 6.01 GiB requested) | 16.21 | TITAN RTX (g2tn01) |
| nsd|link_prediction|freebase_bb | d = 4 | still OOM (needs > 21.18 GiB allocated + 3.99 GiB requested) | 21.18 | TITAN RTX (g2tn01) |
| csnn|link_prediction|freebase_bb | d = 3 | fits at 24 GB with d <= 3 | 19.35 | TITAN RTX (g2tn01) |
| han|link_prediction|freebase_bb | num_heads = 4 | error: `torch_sparse/matmul.py:97 (last line in the log before the fault)` process ended with a segmentation fault (core dumped) after torch_sparse/matmul.py:97 (torch.sparse. |  | TITAN RTX (g2tn01) |
| pshgcn|link_prediction|freebase_bb | K = 2 | still OOM (needs > 23.22 GiB allocated + 176.00 MiB requested) | 23.22 | TITAN RTX (g2tn01) |
| h2gformer|link_prediction|freebase_bb | hidden_dim = 256 | fits at 24 GB with hidden_dim <= 256 | 18.75 | TITAN RTX (g2tn01) |
| nsd|link_prediction|pubmed | d = 4 | fits at 24 GB with d <= 4 | 19.73 | TITAN RTX (g2tn01) |
| pshgcn|link_prediction|pubmed | K = 2 | fits at 24 GB with K <= 2 | 10.61 | TITAN RTX (g2tn01) |
| h2gformer|link_prediction|lastfm | metapath_pe = False | fits at 24 GB with metapath_pe = False | 3.95 | TITAN RTX (g2tn01) |

## Queued, not run when this table was built (pending)

| cell | attempt | job |
|---|---|---|
| h2snn|node_classification|freebase | A6000 | 16889 |
| h2snn|link_prediction|dblp_ap | A6000 | 16889 |
| h2snn|link_prediction|freebase_bb | A6000 | 16889 |
| h2snn|link_prediction|pubmed | A6000 | 16889 |
| h2snn|link_prediction|lastfm | A6000 | 16889 |
| h2snn|node_classification|h-pokec | A6000-120G (g3an01) | 16875 |
| han|node_classification|freebase | 24GB-40G (3090) | 16882 |
| h2gformer|node_classification|freebase | 24GB-40G (3090) | 16882 |
| hetero2net|node_classification|freebase | 24GB-40G (3090) | 16882 |
