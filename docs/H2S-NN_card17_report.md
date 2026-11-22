# Card 17 report — IEEE-CIS-G: five H2GB cells as published-configuration runs (weighted cross-entropy)

Card: `docs/H2S-NN_card17_card.md` (design-docs 49911a4). Branch `card17` from `origin/v3b` = `e52b0252d8f4c21b7b39cf7f505c34079da47fc0` (v3b has not moved since; no merge needed). Worktree `/mnt/data/peeradon/h2snn-card17`, `vendor/` copied from the e52b025 run tree, `data` → `/mnt/data/peeradon/h2snn/H2S-NN/data`.

**Result:** all steps (1, 2, 3, 3b, 3c, 4) are done. The card's tests pass and the mutation is caught. The GPU smoke passes for GCN and H2G-former. No transcribed key changes training in a way the shared loop cannot run, so there was no stop. Two points need a reviewer's eye:
- Findings 1 and 2 below: two edits outside the card's literal file scope, both needed for the cells to run their yaml.
- Finding 3: a loss normalisation difference from H2GB's binary path, which is consistent with the stated difference that already exists.

## Diff stat (`git diff e52b025...card17 -- H2S-NN`)

```
 configs/baselines/gat/node_classification/ieee-cis-g.yaml        |  88 +++
 configs/baselines/gcn/node_classification/ieee-cis-g.yaml        |  86 +++
 configs/baselines/h2gformer/node_classification/ieee-cis-g.yaml  | 102 +++
 configs/baselines/han/node_classification/ieee-cis-g.yaml        |  88 +++
 configs/baselines/paper_filled_cells.tsv                         |   5 -
 configs/baselines/rgcn/node_classification/ieee-cis-g.yaml       |  86 +++
 src/baselines/adapters/_h2gb_network.py                          |   7 +-
 src/baselines/adapters/_h2gformer_h2gb.py                        |  51 +-
 src/baselines/h2gb_published.py                                  |  34 +-
 tests/test_card17_ieee_cis.py                                    | 148 +++
 tests/test_card8_phases.py                                       |   2 +-
 11 files changed, 678 insertions(+), 19 deletions(-)
```
Evidence (outside `H2S-NN/`): `docs/card17_evidence/` (generator, sbatch files, logs, smoke JSONs). The reserved files are untouched (`anchored_solve.py`, the `src/data/` samplers and scopes, `src/trainers/`, `src/configs/config.py`, `src/baselines/budget.py`).

## Steps

1. **Filled rows.** Removed the five `ieee-cis-g` rows (old lines 6, 11, 16, 21, 28). Nothing else in the file changed.
2. **Published files.** Card 8 committed no generator, so I rebuilt its rule in `docs/card17_evidence/gen_published.py`:
   - `params` = `yaml_published_values`, `lines` = `yaml_key_lines`, and `source` = the yaml path plus the H2GB commit from `vendor/VENDORS.tsv`.
   - Output uses `yaml.safe_dump(sort_keys=False, width=120)`.
   - Before writing, job **18379** re-rendered all 30 existing H2GB files and checked them byte for byte: 30 IDENTICAL, 0 differing. It then wrote the five new files.
   - Each new file's `notes` cites card 17.
3. **Weighted loss.** New `h2gb_criterion(args)` in `h2gb_published.py`:
   - `cross_entropy` returns `F.cross_entropy` itself, so existing cells are bit-identical.
   - `weighted_cross_entropy` with a list computes `F.cross_entropy(logits, y, weight=tensor(w, dtype=logits.dtype, device=logits.device))`. The dtype is float because `torch.tensor([1, 5])` would be int64.
   - `weighted_cross_entropy` without a list (`None` or missing) is refused, and so is any other loss.
   - `fit_h2gb_published` uses it in place of the old `!= "cross_entropy"` refusal. Evaluation, B1a selection and metrics are unchanged.
3b. **Featureless types.** `configure_from_yaml` now sets `cfg.share["dim_in"] = {t: width if x else None for t in data.node_types}`, as H2GB's `loader.py:197-203` does. Types with `x` keep their width, so datasets whose types all have `x` get the same dict as before.
   - Other consumers: in `src/baselines`, `cfg.share["dim_in"]` is only passed to H2GB's `HomoGNNModel`, `HeteroGNNModel` and `GTModel`. Their encoders (`raw_encoder.py:91-117`, `hetero_raw_encoder.py`, `hetero_label_encoder.py:25-32`) read `dim_in[t]` only for types with `x` and use the keys only for featureless ones, which is the same pattern as H2GB.
   - `adapters/h2gformer.py` (the non-H2GB path) sets its own `dim_in` and is not affected.
   - The smoke builds both GCN and H2G-former on IEEE-CIS-G, which has 11 featureless types (`card1`–`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `ProductCD`). The `KeyError 'card1'` from card 18 is gone.
3c. **MetaPath2Vec once per dataset.**
   - The cache key is now `<cache>/<dataset>/dim{dim_pe}` with no seed.
   - When the cache is missing, `attach_h2gb_metapath` trains H2GB's `preprocess_Metapath` at seed 0, inside `_Seed0`. `_Seed0` saves the python, numpy, torch CPU and CUDA RNG states, calls `seed_everything(0)`, and restores the states afterwards. A run therefore draws the same randomness whether or not the cache existed.
   - When the cache exists, nothing changes. Every seed and trial, tuned or published, reuses the same embedding.
4. **Status.** `cell_status` returns `published` for the five cells; they were added to `H2GB_PUBLISHED_CELLS` (finding 1). `make_cell_list.py --phase published` now writes 43 lines (38 before). These are the five new lines, from job 18384 with `--out out/cells_published.txt`; there `<out>` was `/mnt/data/peeradon/h2snn-card17/H2S-NN/out`, and ops sets its own:
```
scripts/run_baseline_published.py --task node_classification --models gat --datasets ieee-cis-g --final_seeds 0 1 2 3 4 --output_dir <out>/experiments/baselines --state_dir <out>/state --wandb_project h2snn_baseline_t4
scripts/run_baseline_published.py --task node_classification --models gcn --datasets ieee-cis-g --final_seeds 0 1 2 3 4 --output_dir <out>/experiments/baselines --state_dir <out>/state --wandb_project h2snn_baseline_t4
scripts/run_baseline_published.py --task node_classification --models h2gformer --datasets ieee-cis-g --final_seeds 0 1 2 3 4 --output_dir <out>/experiments/baselines --state_dir <out>/state --wandb_project h2snn_baseline_t4
scripts/run_baseline_published.py --task node_classification --models han --datasets ieee-cis-g --final_seeds 0 1 2 3 4 --output_dir <out>/experiments/baselines --state_dir <out>/state --wandb_project h2snn_baseline_t4
scripts/run_baseline_published.py --task node_classification --models rgcn --datasets ieee-cis-g --final_seeds 0 1 2 3 4 --output_dir <out>/experiments/baselines --state_dir <out>/state --wandb_project h2snn_baseline_t4
```

## The five generated files

| file | source yaml | keys transcribed |
|---|---|---|
| `gcn/node_classification/ieee-cis-g.yaml` | `IEEE-CIS-GCN.yaml` | 33 |
| `gat/node_classification/ieee-cis-g.yaml` | `IEEE-CIS-GAT.yaml` | 34 (+ `gnn.attn_heads`) |
| `rgcn/node_classification/ieee-cis-g.yaml` | `IEEE-CIS-RGCN.yaml` | 33 |
| `han/node_classification/ieee-cis-g.yaml` | `IEEE-CIS-HAN.yaml` | 34 (+ `gnn.attn_heads`) |
| `h2gformer/node_classification/ieee-cis-g.yaml` | `IEEE-CIS-H2Gformer.yaml` | 41 |

All are from H2GB commit d3b49db.
- **Keys transcribed:** `dataset.node_encoder*`, `dataset.edge_encoder*`, `train.{sampler, neighbor_sizes, iter_per_epoch, batch_size, eval_period}`, `model.{type, loss_fun, loss_fun_weight}`, the `gnn.*` or `gt.*` block, `posenc_Hetero_Label.{enable, dim_pe}` (H2G-former only) and `optim.*`.
- **Values:** `loss_fun: weighted_cross_entropy` and `loss_fun_weight: [1, 5]` in all five; `optim` is adamW, lr 1e-3, wd 1e-5, 500 epochs, cosine with 5 warm-up epochs, clipping on, accumulation 32.
- **Encoders:** GCN and GAT use `node_encoder_name: Raw`; R-GCN and HAN use `Hetero_Raw`; H2G-former uses `Hetero_Raw+Hetero_Label`, which matches `H2GB_STACKS["ieee-cis-g"]`.
- **Excluded keys** (as `EXCLUDED_KEYS`): `out_dir`, `metric_best`, `seed`, `wandb.*`, `num_threads`, `num_workers`, `dataset.{dir, format, name, task, task_type, task_entity, transductive}`, `train.{mode, ckpt_period, tqdm, persistent_workers, pin_memory}`, `val.*`, `model.{edge_decoding, graph_pooling}`.
- **Not transcribed:** no key was left out for any reason other than `EXCLUDED_KEYS`. The commented-out `#optim` blocks are comments, not keys.
- The plan check passes: `iter_per_epoch` 256 × `batch_size` 32 is one 8,192-target epoch, as `published_plan` requires (the smoke ran the real check first).

## Findings

1. **Outside "loss only" in `h2gb_published.py`.** The five `(model, "ieee-cis-g")` pairs were added to `H2GB_PUBLISHED_CELLS`. Without them, `is_h2gb_published` is False, so the adapters run the tuned path even with the published file in place. This is a 2-line addition next to the existing set.
2. **Outside "cache key" in `_h2gformer_h2gb.py`.** The H2G-former published path called the loop with a hard-coded `criterion=F.cross_entropy` and never checked `model.loss_fun`. Without a change, the IEEE-CIS H2G-former cell would silently train unweighted. It now uses `h2gb_criterion(args)` on the published path and `F.cross_entropy` on the tuned path (unchanged). The seed-0 / RNG-restore wrapper (`_Seed0`, step 3c) is also in this file.
3. **Normalisation (existing stated difference, for the record).**
   - H2GB on a binary task builds a 1-logit head and uses `BCE(pred, y, weight=w[y])`, which is a plain mean: Σ w_y·l / N.
   - The card's form, `nll_loss(log_softmax)` with `weight`, is the weighted mean Σ w_y·l / Σ w_y over the step's targets.
   - The per-target losses are identical, because a 2-logit softmax is a sigmoid of the logit difference. The per-step scale differs by N / Σ w_y, which depends on how many positives a batch has.
   - The 2-logit cross-entropy replacing H2GB's binary BCE is already stated in `_h2gb_network.py` (module docstring). The card asked for the multi-class form, so I implemented it as written. I did not treat this as a stop.

## Tests and mutation

- **Card tests** (`tests/test_card17_ieee_cis.py`):
  - `test_five_cells_published_and_resolve`: status, file exists, params and lines round-trip with `yaml_published_values` / `yaml_key_lines`, `loss_fun_weight == [1, 5]`.
  - `test_weighted_loss_known_answer`: equals `nll_loss(log_softmax, weight)` and a by-hand weighted mean, and differs from unweighted.
  - `test_cross_entropy_unchanged_and_no_weight_refused`: `cross_entropy` is `F.cross_entropy` itself; no list is refused.
  - `test_second_seed_reuses_the_seed0_cache`: two `fit` calls at seeds 0 and 1 on ogbn-mag, with a synthetic MAG graph and a stub `preprocess_Metapath`. The stub is called once, with `torch.initial_seed() == 0`, at `<cache>/ogbn-mag/dim{d}`, and that is the only cache directory.
- **CPU job 18380** (16G, 1 h) on uncommitted card 17 code: 57 of 58 OK. The one failure was `test_card8_phases` asserting 38 published lines, which is the intended +5. I updated it to 43 (`NC 38 + LP 5`).
- **CPU job 18384** (16G, 1 h) on bf3a9df: **58 tests OK**. Modules: test_card17_ieee_cis, test_card8_published, test_baselines_h2gb_classa, test_partP_h2gformer, test_tsubame_cell_list, test_card8_phases, test_card7_tsubame, test_card5_protocol_hooks. Max RSS 3.3 GiB.
- **Mutation `drop_weight`** (`weight=weight` dropped in `h2gb_criterion`): `test_weighted_loss_known_answer` fails, so the mutation is **CAUGHT** (jobs 18380 and 18384, `mut_*_drop_weight.log`). The file was restored and the diff is clean.
- No full suite was run; ops runs it at review.

## GPU smoke (IEEE-CIS-G, seed 0, published files)

Script: `docs/card17_evidence/smoke_ieee.py`. The adapter's published path runs unchanged, with three smoke-only cuts:
- `published_plan` runs with its real 256 × 32 check, then is shortened to 1 epoch of 4 steps.
- Evaluation runs every epoch.
- The val and test masks are capped at their first 512 targets.

| job | node, GPU | --mem | model | loss per step | params moved | Macro-F1 (val / test) | Micro-F1 (val / test) | time |
|---|---|---|---|---|---|---|---|---|
| 18382 (measuring) | g2tn01, TITAN RTX | 48G | GCN | 1.49, 1.52, 1.39, 1.30 | 16 tensors | 0.114 / 0.056 | 0.123 / 0.059 | 56 s |
| | | | H2G-former | 0.56, 0.74, 0.68, 0.81 | 189 tensors | 0.466 / 0.483 | 0.871 / 0.936 | 57 s |
| **18383 (final)** | g3n01 | **15G** | GCN | same as 18382 | 16 | same | same | 49 s |
| | | | H2G-former | same as 18382 | 189 | same | same | 51 s |

- In both runs, both models use the criterion `weighted_cross_entropy`, all losses are finite, and the record has Macro-F1 and Micro-F1.
- Four steps say nothing about final quality. GCN's low F1 after 4 steps under the [1, 5] weights is expected.
- **Peak memory:** sacct MaxRSS was 11.1 GiB in the measuring job 18382 (process `ru_maxrss` 8.8 GiB), so the final `--mem` is 11.1 × 1.3 = 14.5 → **15G**. Job 18383's MaxRSS was 8.5 GiB. CUDA max allocated was 0.06 GiB (GCN) and 0.32 GiB (H2G-former) at 4 steps.

branch=card17 sha=bf3a9df6945a2d9fe43869ec559329c68a2e4f01 card=docs/H2S-NN_card17_card.md base=e52b0252d8f4c21b7b39cf7f505c34079da47fc0
