# M3a-A0 findings (VERIFIED-CODE, branch part-m3a at the merge commit b58900c, 2026-11-15)

Paths relative to `H2S-NN/`.

## Where the LP decoder is chosen today

- Config field: `src/configs/config.py:652` `decoder_type: str = "dot"  # "dot" | "distmult" | "concat_mlp"`;
  parsed at `:750`; validated at `:938-939` (`{"dot", "distmult", "concat_mlp"}`, else `Unsupported data.decoder_type`).
- Model: `src/models/h2snn_model.py:669-677` builds `LinkPredictionReadout(decoder_type=config.data.decoder_type, ...)`;
  `src/models/link_readout.py:42-57` (dot: nothing; distmult: `relation_matrices`; concat_mlp: an MLP), `:68-79` (scoring).
- Tuner `scripts/run_h2snn_wandb_tuning.py`:
  - `:93` `decoder_type` is in `CONDITIONAL_PARAMETER_NAMES`;
  - `:132` `--decoder_type` with `choices=["dot", "distmult", "concat_mlp"]` (sets the base config's `data.decoder_type`, `:291-292`);
  - `:133` `--include_decoder_type` (store_true);
  - `:270-275` a proposed `decoder_type` is accepted in `{"dot", "distmult", "concat_mlp"}`;
  - `:297-345` `build_sweep_config`: raises if `include_decoder_type` and not LP (`:312-313`); `decoder_type` enters the space only with
    `--include_decoder_type`, as `{"values": ["dot", "distmult", "concat_mlp"]}` (`:341-342`); otherwise the base config's decoder is used for every trial;
  - `:379-380` a trial's `decoder_type` is written to `data.decoder_type`;
  - `:503-511` `_build_sweep_for_run` passes `decoder_type=base_cfg.data.decoder_type` and `include_decoder_type=args.include_decoder_type`.
- Run scripts: `scripts/run_h2snn_runs.py:38` and `scripts/train_h2snn.py:23` `--decoder_type` with `choices=["dot", "distmult", "concat_mlp"]`
  (applied at `run_h2snn_runs.py:67-68`, `train_h2snn.py:42-43`).
- GC tuning script `scripts/run_graph_classification_repeated_cv_wandb_tuning.py:59-60`: `--decoder_type` (no choices) and
  `--include_decoder_type`; neither is read anywhere else in that script (grep: lines 59-60 only), so both are dead flags for GC.

## Which dropout the LP head applies with dot / DistMult

- `classifier_dropout` is consumed in exactly two places (grep over `src/`): the NC head `src/models/h2snn_model.py:688` and the LP readout's
  argument `:675`, which `LinkPredictionReadout` uses **only** inside the `concat_mlp` MLP (`src/models/link_readout.py:48-54`).
  The dot and DistMult branches (`:42-47`, `:68-75`) and the endpoint projection `Linear + GELU` (`:39-41`) have no dropout.
- Hence with `dot`/`distmult` the value of `classifier_dropout` has **no effect** on an LP model. The tuner's rule
  (`run_h2snn_wandb_tuning.py:330-337`: classifier dropout in the LP space only when `decoder_type == "concat_mlp"` and no `--include_decoder_type`)
  is consistent with this. After W2 the `concat_mlp` branch is unreachable, so `classifier_dropout` is never in the LP space.
- The encoder dropout (`encoder_dropout`, `h2snn_model.py:518`) is independent of the decoder and is unchanged.

## Every baseline's LP decoder values

All read from each adapter's LP `search_space`; the shared head `src/baselines/heads.py:58-64` (`build_link_decoder`) accepts exactly `dot` and `distmult`.

| Baseline (registry name) | File:line | Values |
|---|---|---|
| gcn, gat (`hgb_gnn`) | `src/baselines/adapters/hgb_gnn.py:129` | dot, distmult |
| han (`hgb_han`) | `src/baselines/adapters/hgb_han.py:124` | dot, distmult |
| nsd | `src/baselines/adapters/nsd.py:226` | dot, distmult |
| csnn | `src/baselines/adapters/csnn.py:177` | dot, distmult |
| pshgcn | `src/baselines/adapters/pshgcn.py:156` | dot, distmult |
| h2gformer | `src/baselines/adapters/h2gformer.py:219` | dot, distmult |
| hetero2net | `src/baselines/adapters/hetero2net.py:178` | dot, distmult |
| h2sgnn | `src/baselines/adapters/h2sgnn.py:171` | dot, distmult |

No baseline differs from {dot, DistMult} (no report-only finding). A1's test reads these from the registry at run time.

## Scope builder (A2)

- `src/data/scopes.py:52-63` `scope_members`: `incident = ((src == target) | (dst == target)).nonzero()` over all facts: O(E) per target.
- `:110-175` `build_scope_batch` calls it once per target (`:124`).
- Callers: `src/trainers/minibatch_trainer.py:119`, `src/trainers/minibatch_lp.py:76`; tests patch `build_scope_batch` with a
  3-argument spy (`tests/test_partMBfix_flag.py:96-103`, `tests/test_partMB_trainer.py:146-152`, `tests/test_partMB_link.py:104-110`), so the call
  signature `(graph, targets, cells)` is kept and the index is built inside `scopes.py` for the graph given.

## Transport code used by `build_cell_group` (A3)

- `src/models/sheaf_energy.py:230-241` `build_cell_group` builds `PathTransportOperator(maps.src, maps.tgt, ..., shared_scale=True,
  face_edge_sign=group.face_edge_sign)` and `FactCellTransport(transport, ...)` (R comes from the same transport).
- `src/models/map_operators.py:472-475` `validate_face_orientation(..., face_edge_sign)` runs first; `:485-497` the hop sign selects the
  near/far maps per hop (`reversed_hop = face_edge_sign[:, j] < 0`; rows swapped with `_mix_rows`); `:499-509` `_step`/`_step_T` use them.
- `tests/test_partMBfix_functoriality.py` computes the reference `F_near/F_far` from the maps and `face_edge_sign` itself (l.74-79), so a
  transport that misreads the sign but passes the orientation check is measured by the residual assertion (l.89-90).
