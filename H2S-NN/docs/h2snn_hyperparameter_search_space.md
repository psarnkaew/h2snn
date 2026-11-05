# H2SNN W&B Hyperparameter Search Space

This document describes the primary search implemented by
`scripts/run_h2snn_wandb_tuning.py`. The runner uses W&B Bayesian search with
Hyperband early termination and maximizes the selected validation monitor. Always run
`--validate_only` before creating a sweep; validation does not contact W&B or train a
model. The separate `scripts/tune_h2snn.py` entry point is not assumed to have an
identical search space.

## Protocol decisions

- Node and link prediction use one complete-graph forward/backward optimizer step per
  epoch. Query minibatching and subcomplex sampling are not implemented or tuned in the
  primary protocol.
- Graph classification uses the implementation's fixed batch size of 32 independent
  graphs, and every graph remains a complete node-edge-face complex. Batch size is not tuned.
- The base YAML defines the experiment protocol. Stage-I AMPSS support, materialization
  budgets, relation semantics, data splits, optimizer family, scheduler, and structural
  switches are not silently changed by the primary sweep.
- Every repeated seed gets an isolated sweep and every trial runs in a fresh worker
  process.
- Node and link trainers enforce the configured `early_stopping_patience`; patience is
  fixed within a sweep rather than treated as a model hyperparameter.

## Primary search

| Parameter | Search | Applicability | Meaning |
|---|---|---|---|
| `d` | node/link: `{16, 32}`; graph: `{16, 32, 64}` | all tasks | Stalk dimension; override explicitly with `--d_values` |
| `K` | `{2, 3, 4, 5, 6}` | all tasks | Number of Hodge-diffusion steps |
| `eta_0` | log-uniform `[0.005, 0.3]` | all tasks | Initial node-cochain step size |
| `eta_1` | log-uniform `[0.005, 0.3]` | all tasks | Initial edge-cochain step size |
| `layer_dropout` | `{0.0, 0.1, 0.2, 0.3}` | all tasks | Dropout between diffusion steps |
| `encoder_dropout` | `{0.0, 0.1, 0.2, 0.3}` | datasets with input features | Omitted for the all-featureless identity-embedding fallback (`lastfm`, `amazonbook`) because it is inactive there |
| `classifier_dropout` | `{0.0, 0.1, 0.2, 0.3}` | node/graph, and link with a fixed `concat_mlp` decoder | Omitted for link `dot`/`distmult` and decoder-family sweeps because those branches do not consume it |
| `lr` | `{1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2}` | all tasks | Initial learning rate |
| `weight_decay` | `{0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3}` | all tasks | Optimizer weight decay; zero is retained as a control |
| `sheafcov_feat_dim` | `{4, 8, 16}` | graph classification with fixed `data.pooling_type: sheafcov` | Learned post-Gram feature width; override with `--sheafcov_feat_dims` |

Changing `d` changes only `model.d`. The runner no longer silently rewrites `d_r`,
`d_h`, edge/face MLP widths, or classifier width. A coupled width-scaling study must be
declared separately rather than hidden inside the `d` override.

## Why the raw Gram size is fixed

SheafCovPool constructs three semantic stalk-valued views for each node and each edge,
so the raw descriptor is

\[
G = Z^\top Z \in \mathbb{R}^{3\times3}.
\]

The three diagonal values are view energies and the three distinct off-diagonal values
are pairwise alignments. The order is determined by the view definition, not model
capacity. A `5x5` or `7x7` Gram matrix would require additional gauge-compatible views
and would change the readout architecture. Padding, repeating, or linearly expanding
the three existing views adds no invariant information.

Capacity is tuned after this descriptor through `model.sheafcov_feat_dim = F`. The
node/edge means and upper-triangular covariances produce a graph summary of width
`F^2 + 3F`: 28, 88, and 304 for `F` equal to 4, 8, and 16. The raw flattened input
remains nine entries for compatibility.

## Optional ablation spaces

These switches are supported by the runner but absent from the three primary Colab
notebooks:

| Flag | Added parameters | Purpose |
|---|---|---|
| `--include_pooling_type` | `pooling_type in {meanpool, grampool, sheafcov}` | Graph-readout ablation; auto AMPSS excludes `meanpool`, and this flag is mutually exclusive with `--sheafcov_feat_dims` |
| `--include_decoder_type` | `decoder_type in {dot, distmult, concat_mlp}` | Link-decoder ablation; classifier dropout is held fixed because it is inactive for two branches |
| `--include_discovery` | `discovery_warmup_epochs in {0,5,10}` and `discovery_operator_anneal_epochs in {3,5,10}` | Stage-II AMPSS schedule study; requires an auto-discovery base config |

`S2Pool` is excluded from the current pooling sweep because its descriptor is degenerate
for the normal full-column-rank three-view stack. When `--include_discovery` is used,
set `--hyperband_min_iter` above the largest warmup-plus-anneal combination; the current
space therefore requires at least 21 epochs before Hyperband may terminate a trial.

Stage-I candidate generation, schema/instance/incidence budgets, and selector seeds are
frozen protocol inputs rather than Stage-II hyperparameters. Pooling and decoder
families should be reported as ablations, not mixed into the primary model-capacity
search.

## Fixed or removed dimensions

The following remain in the resolved base configuration and are not W&B dimensions:

- `learnable_eta`, `use_delta0_init`, and `orthogonal_map: fasthpp`;
- optimizer name, scheduler, and gradient clipping;
- `d_r`, `d_h`, edge/face hidden widths, and classifier hidden width;
- `closure_scale`, `face_sample_k`, and `num_label_prop_steps`;
- graph batch size and the full-graph node/link training context.

Legacy entropy/redundancy coefficients, schema or instance sparsity gates,
partial-schema dropout, auxiliary discovery losses, old `lambda_*` objectives, and
unsupported orthogonal-map choices are rejected. They must not be restored from
historical configs or W&B state.

## Preflight commands

Node classification, complete graph:

```powershell
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts\run_h2snn_wandb_tuning.py `
    --config configs\experiments\acm_h2snn_ampss.yaml `
    --exp_name acm_h2snn_ampss_node_fullbatch_primary_v5 `
    --num_runs 5 --n_trials 30 --d_values 16 32 --validate_only
```

Link prediction, complete outer-training graph:

```powershell
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts\run_h2snn_wandb_tuning.py `
    --config configs\experiments\lastfm_h2snn.yaml `
    --exp_name lastfm_h2snn_link_fullbatch_primary_v5 `
    --num_runs 5 --n_trials 30 --d_values 16 32 --validate_only
```

Graph classification with fixed SheafCovPool:

```powershell
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts\run_h2snn_wandb_tuning.py `
    --config configs\experiments\mutag_h2snn.yaml `
    --exp_name mutag_h2snn_graph_sheafcov_primary_v5 `
    --num_runs 5 --n_trials 40 --d_values 16 32 64 `
    --sheafcov_feat_dims 4 8 16 --validate_only
```

Remove `--validate_only` only after the resolved space is correct. Tuning state version
5 fingerprints the base configuration and search space. Start without `--resume`; use
it only for the exact interrupted state and original W&B sweep ID. Old state directories
are intentionally incompatible.

## Resource-conscious verification

1. Run the three `--validate_only` commands.
2. Run unit tests for configuration, sweep routing, and SheafCov widths.
3. Optionally run one short single-trial smoke check on the smallest dataset and
   smallest `d`; do not treat it as benchmark evidence.
4. Profile a real full forward/backward step before deciding whether node/link
   minibatching is necessary.

Do not claim completed tuning results until traceable W&B runs and repeated-seed final
evaluations exist.
