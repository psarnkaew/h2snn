# H2S-NN GC trials: seconds per epoch on the H100 (ops, 2026-11-21)

Source: the `h2snn` tuning chains at v3b `e5bdf10` on TSUBAME (list `cells_tuning_h2snn_gpu1.txt`, resource `gpu_1`: one H100 and the CPU share of one GPU slot). Read-only extraction by `gc_speed.py` from each trial's `run.log` (median of the time between consecutive epoch lines, the first interval dropped). `gc_trials.tsv` has one line per trial. `end` = `tau` (stop condition), `early-stop` (patience 50 on the validation loss) or `killed/running` (chain cancelled).

| Dataset | Trials | s / epoch (median per trial) | Epochs of scored trials | h per scored trial |
|---|---|---|---|---|
| MUTAG | 5 | 60–73 (130 in one τ trial) | 77, 142, 143 (early stop) | 1.3–2.6 |
| PROTEINS | 2 | 175, 497 | 115 (early stop) | ≈ 5.6 |
| ENZYMES | 4 | 142–366 | 100 (early stop) | ≈ 4.3 |
| NCI1 | 3 | 2,800–3,600 (47–60 min) | none (2 τ, 1 killed at epoch 4) | ≥ 40 at 51 epochs |

Batch size 1, so the epoch time scales with the number of training graphs and is likely bound by per-graph work on the CPU side, not by the GPU. `gpu_1` gets only a small CPU share; a `node_f` slot (a quarter of a node) has more cores. That is untested.
