# Card 2 D6.2 — per-adapter code check, `6ba6dec` → merged head `0944edb` (design-docs `c065dfc` + v3b `5ccc792`)

Amended D6.2 (author, 2026-11-16): `git diff 6ba6dec 0944edb -- <adapter and the shared modules it imports>` for the
adapters of `d5_ready_baselines.txt` (101 cells) and of the D2 step-down cells; re-run only where the cell's path changed.
Changed files under `H2S-NN/src/` (24; every other `src/` file is identical, `git diff --name-only`): the ones below
plus files no ready / step-down cell imports (`_chebnetii_h2gb.py`, `_h2gb_network.py`, `_h2gformer_h2gb.py`,
`_hetero2net_h2gb.py`, `_pshgcn_mag.py`, `_sehgnn_h2gb.py`, `large_lp_training.py`: H2GB NC / large-LP only).
Unchanged and imported by these cells: `campaign.py`, `protocol.py`, `training.py`, `heads.py`, `metrics.py`,
`progress.py`, `search_grids.py`, `_homogeneous_lp.py`, `_han_lp.py`, `_h2sgnn_lp.py`, `_pshgcn_lp.py`,
`_polynomial_lp.py`, `_gc_typed.py`, `_h2gformer_encoders.py`, `_hgb_vendor.py`, `_target_block.py`, `h2sgnn.py`,
`pshgcn.py`, `src/data/**`, `src/trainers/**`.

| file | lines removed or changed | used by (ready / step-down cells) | result for these cells' paths |
|---|---|---|---|
| `adapters/hgb_gnn.py` (GCN, GAT) | 2: `supports` gains `or dataset in H2GB_DATASETS` / `LARGE_LP_DATASETS` | NC acm/dblp/imdb/freebase, LP, GC; GAT acm_pa step-down | add-only for this cell's path (new branches guarded by `bundle.dataset in H2GB_DATASETS` / `LARGE_LP_DATASETS`, new search space only `if dataset in H2GB_DATASETS`) |
| `adapters/csnn.py` | 2: `supports` as above | NC, LP, GC; CSNN freebase / freebase_bb step-down | add-only for this cell's path (same guards) |
| `adapters/nsd.py` | 2: `supports` as above | NC, LP, GC; NSD pubmed step-down | add-only for this cell's path |
| `adapters/hinormer.py` | 1: `supports` as above | (no ready cell; HINormer HGB NC still OOM, D6.3) | add-only for this cell's path (`_fit_h2gb` only `if bundle.dataset in H2GB_DATASETS`; new search space only for H2GB) |
| `adapters/h2gformer.py` | 2: `supports` as above | LP pubmed (ready); 8 step-down cells (NC acm/dblp/imdb, LP acm_pa/dblp_ap/imdb_ma/lastfm/freebase_bb) | add-only for this cell's path (H2GB / large-LP branches guarded) |
| `adapters/hgb_han.py` | 2: `supports` as above | NC acm/dblp, LP, GC | add-only for this cell's path |
| `adapters/sehgnn.py` | 2: `H2GB_RUNNABLE` set widened; `h2gb.node_inputs(..., MAG_RANDOM_TYPES)` in `_fit_h2gb` | NC acm/dblp/imdb | add-only for this cell's path (both lines on the H2GB branch only) |
| `adapters/chebnetii.py` | 9: `H2GB_DATASETS` widened; search-space `return` → `space = …; return space` (H2GB branch); H2GB fit body restructured (forward / projected branches) | NC acm/dblp/freebase/imdb | add-only for this cell's path (every changed line inside the H2GB branches, `chebnetii.py:122-137` (search space) and `_fit_h2gb` `chebnetii.py:231-290`; HGB NC path unchanged) |
| `adapters/hetero2net.py` | 12: `supports`; NC search space `return {` → `space = …` (+ batch size only for H2GB); **HGB NC evaluation** → `evaluation_logits` (card 1d H2: validation from `train_data`) | LP and GC (ready); NC acm/dblp/imdb held back | LP, GC: add-only for this cell's path. **NC (HGB): changed** — already held back (card 1d D2), not in the ready list |
| `bundles.py` | 0 (adds `LargeLinkBundle`, returned only `if dataset in LARGE_LP_TASKS`) | every baseline cell | add-only for this cell's path |
| `registry.py` | 28: H2GB NC / large-LP pending reasons and wired-cell sets | every baseline cell (`iter_cells`, `unsupported_reason`) | add-only for this cell's path (only H2GB NC and large-LP rows change; HGB NC / LP / GC rows identical) |
| `minibatch.py` | 14: `seed_rows` / `evaluate_split` take the labels | none (H2GB NC only) | not on these cells' paths |
| `runtime/vendor_paths.py` | 0 (adds `pshgcn_mag` root) | every adapter | add-only for this cell's path |
| `configs/config.py` | 9: arm switches (`ablation.fact_maps_identity`, `tuning.arm`) and their tuning validation (card 4) | every cell (TrainConfig) | add-only for this cell's path (new keys default off; the old refusals kept for the main table) |
| `models/h2snn_model.py`, `map_operators.py`, `sheaf_maps.py` | 13: `build` takes `*factors_and_rest`; arm "maps = I" (`IdentityMapOperator`) | H2S-NN step-down cells (LP dblp_ap, pubmed, lastfm) | full model (`identity_maps` False): the same operators built from the same tensors in the same order (`h2snn_model.py:760-796`); bit-exact golden / mb0 in the D6.9 suite |

**Result:** no ready cell and no step-down cell has a changed path; **nothing is re-run** for D6.2.
`d6_ready_baselines.txt` = `d5_ready_baselines.txt` carried over (101 lines, same format); Hetero²Net HGB NC stays held back.

Found while preparing D6.4 (not a D6.2 cell): every H2GB NC baseline cell still stops in `bundles._base_config`
(`bundles.py:388` → `config.py:1107`, "data.batch_size (B) is required in mini-batch training") on the merged `v3b`
— the H3b row of card 1d, not yet on `v3b`. Recorded in D6.4 as "blocked: baseline code (H3b)".
