# Card 2 D6.4 — central-point timing (decision B2a): one epoch + one full validation pass

Central point: each numeric grid axis at the middle of its grid (lower middle for an even count), log-uniform at the geometric middle, uniform at the middle, categoricals at the adapter's default (H2S-NN: the YAML's); H2S-NN on H2GB NC at B 256. Times in s; `val targets` = targets of one full validation pass; ms/target = validation time per target (to scale to the B1a subset). Baselines' train = model construction + the epoch. No metric.

| cell | status | GPU (node) | GPU peak | host peak | train epoch | val pass | val targets | ms/target | val timed by |
|---|---|---|---|---|---|---|---|---|---|
| h2snn|graph_classification|enzymes | OK | TITAN RTX (g2tn01) | 0.03 | 1.53 | 149.54 | 10.50 | 60 | 175.06 | val_pass phase |
| h2snn|graph_classification|mutag | OK | TITAN RTX (g2tn01) | 0.02 | 1.52 | 49.92 | 3.10 | 18 | 171.96 | val_pass phase |
| h2snn|graph_classification|nci1 | stop condition `src/models/h2snn_model.py:746 (_encode_anchored)` | TITAN RTX (g2tn01) | 0.02 | 1.50 | 587.36 |  | 411 |  |  |
| h2snn|graph_classification|proteins | OK | TITAN RTX (g2tn01) | 0.10 | 1.57 | 299.49 | 22.41 | 111 | 201.86 | val_pass phase |
| h2snn|node_classification|h-pokec | OK | TITAN RTX (g2tn01) | 1.08 | 15.40 | 208.89 | 3940.92 | 408160 | 9.66 | val_pass phase |
| h2snn|node_classification|ieee-cis-g | OK | GeForce RTX 2080 Ti (g2n01) | 1.18 | 6.38 | 27.29 | 20.42 | 14423 | 1.42 | val_pass phase |
| h2snn|node_classification|mag-year | OK | TITAN RTX (g2tn01) | 0.41 | 9.89 | 162.16 | 827.60 | 184097 | 4.50 | val_pass phase |
| h2snn|node_classification|oag-chem | OK | GeForce RTX 2080 Ti (g2n01) | 0.61 | 14.47 | 145.79 | 103.14 | 59664 | 1.73 | val_pass phase |
| h2snn|node_classification|oag-cs | OK | GeForce RTX 2080 Ti (g2n01) | 0.59 | 10.05 | 117.22 | 33.30 | 19551 | 1.70 | val_pass phase |
| h2snn|node_classification|oag-eng | OK | GeForce RTX 2080 Ti (g2n01) | 0.46 | 6.95 | 62.99 | 38.25 | 28524 | 1.34 | val_pass phase |
| h2snn|node_classification|ogbn-mag | OK | TITAN RTX (g2tn01) | 0.41 | 9.75 | 164.45 | 113.36 | 64879 | 1.75 | val_pass phase |
| h2snn|node_classification|pdns | OK | GeForce RTX 2080 Ti (g2n01) | 0.81 | 14.32 | 266.95 | 27.55 | 17450 | 1.58 | val_pass phase |
| csnn|graph_classification|enzymes | OK | TITAN RTX (g2tn01) | 0.02 | 1.47 | 13.62 | 1.85 | 60 | 30.76 | first eval → epoch report (val + test where the loop evaluates both) |
| gat|graph_classification|enzymes | OK | TITAN RTX (g2tn01) | 0.02 | 1.29 | 1.95 | 1.19 | 60 | 19.81 | first eval → epoch report (val + test where the loop evaluates both) |
| gcn|graph_classification|enzymes | OK | TITAN RTX (g2tn01) | 0.02 | 1.26 | 1.43 | 1.06 | 60 | 17.64 | first eval → epoch report (val + test where the loop evaluates both) |
| h2gformer|graph_classification|enzymes | OK | TITAN RTX (g2tn01) | 0.02 | 1.43 | 12.85 | 0.53 | 60 | 8.85 | first eval → epoch report (val + test where the loop evaluates both) |
| h2sgnn|graph_classification|enzymes | OK | TITAN RTX (g2tn01) | 0.02 | 1.27 | 7.41 | 1.07 | 60 | 17.77 | first eval → epoch report (val + test where the loop evaluates both) |
| han|graph_classification|enzymes | OK | TITAN RTX (g2tn01) | 0.02 | 1.35 | 7.80 | 1.18 | 60 | 19.74 | first eval → epoch report (val + test where the loop evaluates both) |
| hetero2net|graph_classification|enzymes | non-finite `src/models/anchored_solve.py:95 (check_finite_loss)` | TITAN RTX (g2tn01) | 0.02 | 1.34 |  |  | 60 |  |  |
| nsd|graph_classification|enzymes | OK | TITAN RTX (g2tn01) | 0.02 | 1.44 | 13.22 | 2.75 | 60 | 45.80 | first eval → epoch report (val + test where the loop evaluates both) |
| pshgcn|graph_classification|enzymes | OK | TITAN RTX (g2tn01) | 0.02 | 1.36 | 8.46 | 1.06 | 60 | 17.71 | first eval → epoch report (val + test where the loop evaluates both) |
| csnn|graph_classification|mutag | OK | TITAN RTX (g2tn01) | 0.02 | 1.47 | 7.82 | 1.15 | 18 | 63.95 | first eval → epoch report (val + test where the loop evaluates both) |
| gat|graph_classification|mutag | OK | TITAN RTX (g2tn01) | 0.02 | 1.29 | 0.75 | 0.93 | 18 | 51.52 | first eval → epoch report (val + test where the loop evaluates both) |
| gcn|graph_classification|mutag | OK | TITAN RTX (g2tn01) | 0.02 | 1.26 | 0.57 | 0.92 | 18 | 51.10 | first eval → epoch report (val + test where the loop evaluates both) |
| h2gformer|graph_classification|mutag | OK | TITAN RTX (g2tn01) | 0.02 | 1.43 | 10.24 | 0.21 | 18 | 11.89 | first eval → epoch report (val + test where the loop evaluates both) |
| h2sgnn|graph_classification|mutag | OK | TITAN RTX (g2tn01) | 0.02 | 1.33 | 8.75 | 1.33 | 18 | 73.88 | first eval → epoch report (val + test where the loop evaluates both) |
| han|graph_classification|mutag | OK | TITAN RTX (g2tn01) | 0.02 | 1.39 | 11.52 | 2.00 | 18 | 111.33 | first eval → epoch report (val + test where the loop evaluates both) |
| hetero2net|graph_classification|mutag | OK | TITAN RTX (g2tn01) | 0.02 | 1.56 | 705.69 | 1.06 | 18 | 58.87 | first eval → epoch report (val + test where the loop evaluates both) |
| nsd|graph_classification|mutag | OK | TITAN RTX (g2tn01) | 0.02 | 1.42 | 6.48 | 1.08 | 18 | 60.23 | first eval → epoch report (val + test where the loop evaluates both) |
| pshgcn|graph_classification|mutag | OK | TITAN RTX (g2tn01) | 0.02 | 1.36 | 7.10 | 1.08 | 18 | 59.84 | first eval → epoch report (val + test where the loop evaluates both) |
| csnn|graph_classification|nci1 | OK | TITAN RTX (g2tn01) | 0.02 | 1.52 | 67.67 | 7.16 | 411 | 17.43 | first eval → epoch report (val + test where the loop evaluates both) |
| gat|graph_classification|nci1 | OK | TITAN RTX (g2tn01) | 0.02 | 1.32 | 12.20 | 2.79 | 411 | 6.79 | first eval → epoch report (val + test where the loop evaluates both) |
| gcn|graph_classification|nci1 | OK | TITAN RTX (g2tn01) | 0.02 | 1.28 | 8.83 | 2.17 | 411 | 5.29 | first eval → epoch report (val + test where the loop evaluates both) |
| h2gformer|graph_classification|nci1 | OK | TITAN RTX (g2tn01) | 0.02 | 1.45 | 37.53 | 3.35 | 411 | 8.16 | first eval → epoch report (val + test where the loop evaluates both) |
| h2sgnn|graph_classification|nci1 | OK | TITAN RTX (g2tn01) | 0.02 | 1.30 | 24.74 | 2.95 | 411 | 7.17 | first eval → epoch report (val + test where the loop evaluates both) |
| han|graph_classification|nci1 | OK | TITAN RTX (g2tn01) | 0.02 | 1.37 | 23.26 | 3.64 | 411 | 8.86 | first eval → epoch report (val + test where the loop evaluates both) |
| nsd|graph_classification|nci1 | OK | TITAN RTX (g2tn01) | 0.02 | 1.44 | 45.89 | 8.16 | 411 | 19.86 | first eval → epoch report (val + test where the loop evaluates both) |
| pshgcn|graph_classification|nci1 | OK | TITAN RTX (g2tn01) | 0.02 | 1.38 | 20.64 | 2.48 | 411 | 6.02 | first eval → epoch report (val + test where the loop evaluates both) |
| csnn|graph_classification|proteins | OK | TITAN RTX (g2tn01) | 0.03 | 1.50 | 21.43 | 2.33 | 111 | 21.01 | first eval → epoch report (val + test where the loop evaluates both) |
| gat|graph_classification|proteins | OK | TITAN RTX (g2tn01) | 0.05 | 1.30 | 3.39 | 1.23 | 111 | 11.04 | first eval → epoch report (val + test where the loop evaluates both) |
| gcn|graph_classification|proteins | OK | TITAN RTX (g2tn01) | 0.02 | 1.28 | 2.55 | 1.30 | 111 | 11.74 | first eval → epoch report (val + test where the loop evaluates both) |
| h2gformer|graph_classification|proteins | OK | TITAN RTX (g2tn01) | 0.04 | 1.42 | 16.44 | 0.96 | 111 | 8.68 | first eval → epoch report (val + test where the loop evaluates both) |
| h2sgnn|graph_classification|proteins | OK | TITAN RTX (g2tn01) | 0.02 | 1.33 | 10.00 | 1.51 | 111 | 13.63 | first eval → epoch report (val + test where the loop evaluates both) |
| han|graph_classification|proteins | OK | TITAN RTX (g2tn01) | 0.03 | 1.36 | 10.21 | 1.82 | 111 | 16.42 | first eval → epoch report (val + test where the loop evaluates both) |
| hetero2net|graph_classification|proteins | non-finite `src/models/anchored_solve.py:95 (check_finite_loss)` | TITAN RTX (g2tn01) | 0.02 | 1.49 |  |  | 111 |  |  |
| nsd|graph_classification|proteins | OK | TITAN RTX (g2tn01) | 0.03 | 1.43 | 21.34 | 4.69 | 111 | 42.25 | first eval → epoch report (val + test where the loop evaluates both) |
| pshgcn|graph_classification|proteins | OK | TITAN RTX (g2tn01) | 0.02 | 1.37 | 9.83 | 1.31 | 111 | 11.78 | first eval → epoch report (val + test where the loop evaluates both) |
| chebnetii|node_classification|ieee-cis-g | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.76 |  |  |  |  |  |
| csnn|node_classification|ieee-cis-g | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| gat|node_classification|ieee-cis-g | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| gcn|node_classification|ieee-cis-g | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| h2gformer|node_classification|ieee-cis-g | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| han|node_classification|ieee-cis-g | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| hetero2net|node_classification|ieee-cis-g | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.76 |  |  |  |  |  |
| hinormer|node_classification|ieee-cis-g | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| nsd|node_classification|ieee-cis-g | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| sehgnn|node_classification|ieee-cis-g | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| chebnetii|node_classification|mag-year | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.76 |  |  |  |  |  |
| csnn|node_classification|mag-year | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| gat|node_classification|mag-year | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| gcn|node_classification|mag-year | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| h2gformer|node_classification|mag-year | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| han|node_classification|mag-year | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| hetero2net|node_classification|mag-year | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| hinormer|node_classification|mag-year | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| nsd|node_classification|mag-year | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| sehgnn|node_classification|mag-year | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| chebnetii|node_classification|oag-chem | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| csnn|node_classification|oag-chem | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| gat|node_classification|oag-chem | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| gcn|node_classification|oag-chem | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| h2gformer|node_classification|oag-chem | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| han|node_classification|oag-chem | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| hetero2net|node_classification|oag-chem | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| hinormer|node_classification|oag-chem | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| nsd|node_classification|oag-chem | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| sehgnn|node_classification|oag-chem | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| chebnetii|node_classification|oag-cs | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| csnn|node_classification|oag-cs | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| gat|node_classification|oag-cs | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| gcn|node_classification|oag-cs | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| h2gformer|node_classification|oag-cs | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| han|node_classification|oag-cs | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| hetero2net|node_classification|oag-cs | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| hinormer|node_classification|oag-cs | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| nsd|node_classification|oag-cs | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| sehgnn|node_classification|oag-cs | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| chebnetii|node_classification|oag-eng | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.76 |  |  |  |  |  |
| csnn|node_classification|oag-eng | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| gat|node_classification|oag-eng | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| gcn|node_classification|oag-eng | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| h2gformer|node_classification|oag-eng | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| han|node_classification|oag-eng | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| hetero2net|node_classification|oag-eng | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| hinormer|node_classification|oag-eng | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| nsd|node_classification|oag-eng | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| sehgnn|node_classification|oag-eng | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| chebnetii|node_classification|ogbn-mag | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| csnn|node_classification|ogbn-mag | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| gat|node_classification|ogbn-mag | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| gcn|node_classification|ogbn-mag | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| h2gformer|node_classification|ogbn-mag | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| han|node_classification|ogbn-mag | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| hetero2net|node_classification|ogbn-mag | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.76 |  |  |  |  |  |
| hinormer|node_classification|ogbn-mag | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| nsd|node_classification|ogbn-mag | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| sehgnn|node_classification|ogbn-mag | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| chebnetii|node_classification|pdns | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| csnn|node_classification|pdns | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| gat|node_classification|pdns | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| gcn|node_classification|pdns | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| h2gformer|node_classification|pdns | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| han|node_classification|pdns | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| hetero2net|node_classification|pdns | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| hinormer|node_classification|pdns | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |
| nsd|node_classification|pdns | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.77 |  |  |  |  |  |
| sehgnn|node_classification|pdns | blocked: baseline code `src/configs/config.py:1107 (resolved)` | TITAN RTX (g2tn01) | 0.00 | 0.79 |  |  |  |  |  |

Not run (design session 2026-11-17: H-Pokec runs only what B2a needs, H2S-NN's central point): array tasks 16924_3 to 16924_11 cancelled while pending (60G each, one at a time, blocking g2tn01's second GPU): the central lines of GCN, GAT, ChebNetII, NSD, CSNN, HAN, HINormer, H²GFormer, Hetero²Net on H-Pokec NC (each would stop in `bundles._base_config` (H3b) like the other H2GB NC baseline cells above).
