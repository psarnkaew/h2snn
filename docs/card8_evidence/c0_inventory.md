# Card 8 C0 — Inventory (no code)

Branch `card8` (base `v3b` fc0ebaa + design-docs b5adecf). Read for this unit: the card; `docs/published_number_register.md`;
`src/baselines/{registry,published,campaign,budget,search_grids,minibatch}.py`; `scripts/run_baseline_published.py`;
the 16 existing `configs/baselines/` files; H2GB `configs/<dataset>/*-{GCN,GAT,HAN,H2Gformer,RGCN,HINormer}.yaml`
(train/optim/gnn/gt blocks) and `H2GB/train/custom_train.py:69-170, 255-293`; HGB `NC/benchmark/methods/{MAGNN,RGCN}`
(README, `run_Freebase.py`), HGB `LP/benchmark/methods/{GNN,RGCN,MAGNN,baseline}/README.md`; SeHGNN `ogbn/Readme.md`;
PSHGCN `ogbn-mag/README.md`, `ogbn-mag/main.py`, `hgb-lp/README.md`, `hgb-lp/training.py`; Hetero2Net, H2SGNN,
HINormer READMEs; H2SGNN/PSHGCN `config.yaml` block names. Not read in full: H2GB model modules (only the class names
the yamls name), MAGNN `model/` and `utils/` (C3).

## 1. Filled cells

`H2S-NN/configs/baselines/paper_filled_cells.tsv` (task, dataset, method, register line): every register row whose
metric is the table metric (NC Macro-F1, LP ROC-AUC, GC Macro-F1). 81 cells: NC 52, LP 5, GC 24 (the 24 GC Micro-F1
rows are not table-metric cells). Every row mapped without guessing (dataset and method names map one to one to the
registry names; R-GCN → `rgcn`, MAGNN → `magnn`, HiNormer → `hinormer`, Hetero²Net → `hetero2net`); no duplicate
(task, dataset, method). No stop.

## 2. Published configurations of empty cells

| cells | status | source (file:line) | note |
|---|---|---|---|
| HAN × {mag-year, oag-cs, oag-eng, oag-chem, h-pokec, pdns} | published | `H2GB/configs/<D>/<D>-HAN.yaml` (train: l.23–; optim l.57–65; oag-cs 54–62, oag-eng/oag-chem 56–64, Pokec 74–82) | H2GB's own `HeteroGNNModel` (our adapter runs it) |
| R-GCN × same 6 | published | `<D>-RGCN.yaml` (optim l.56–64; oag-chem 55–63, Pokec 73–81) | H2GB's `HomoGNNModel`, `layer_type: RGCN` (card C2) |
| H2G-former × {ogbn-mag, mag-year, oag-cs, oag-eng, oag-chem, h-pokec, pdns} | published | `<D>-H2Gformer.yaml` (optim l.70–78 ogbn-mag, 69–77 mag-year, 64–72 oag-cs/oag-eng, 68–76 oag-chem/PDNS, 84–92 Pokec) | H2GB's own `GTModel` |
| GCN, GAT × {mag-year, oag-cs, oag-eng, oag-chem, h-pokec, pdns} | published ‡ | `<D>-GCN.yaml`, `<D>-GAT.yaml` (`gnn:` l.43; optim 56–64/57–65; Pokec 73–81/74–82) | ‡ see §4 |
| HINormer × {mag-year, oag-eng, pdns} | published ‡ | `<D>-HINormer.yaml` (`gnn:` l.43, optim 54–62) | ‡ see §4; no yaml for oag-cs, oag-chem, IEEE-CIS, Pokec |
| PSHGCN imdb (NC) | published | existing `configs/baselines/pshgcn/node_classification/imdb.yaml` (PSHGCN `hgb/config.yaml` IMDB block l.23) | |
| H2SGNN imdb (NC) | published | existing `configs/baselines/h2sgnn/node_classification/imdb.yaml` (H2SGNN `config.yaml` IMDB block l.42) | |
| PSHGCN ogbn-mag | published ‡ | PSHGCN `ogbn-mag/README.md:13` (`python main.py --stage 1 --bias`), rest `main.py:186-208` argparse | README runs the script for this dataset with no other flag (card rule); ‡ see §4 |
| MAGNN freebase (NC) | published | HGB `NC/benchmark/methods/MAGNN/run_Freebase.py:17-19` (dropout 0.5, lr 0.005, wd 0.001), `:209-227` (feats-type 2, hidden 64, heads 8, attn-vec 128, RotatE0, epoch 100, patience 5, batch 8, samples 100) | the file is Freebase-specific (its own defaults); MAGNN's README (l.6-10) lists only DBLP/ACM/IMDB_new commands — stated |
| PSHGCN lastfm (LP) | published | existing `configs/baselines/pshgcn/link_prediction/lastfm.yaml` (PSHGCN `hgb-lp/README.md:5`, `training.py:175-196`) | |

Differences from the card's expected list (not a stop):
- **SeHGNN ogbn-mag → `tune`.** The README's no-extra-embedding command (`SeHGNN/ogbn/Readme.md:8-10`) trains 4 stages
  (`--stages 300 300 300 300`) with label features (`--label-feats --num-label-hops 2`) and pseudo-label enhancement
  (`--threshold 0.75 --gama 10`). Our adapter runs stage 0 without label features (`_sehgnn_h2gb.py:17-18`, card
  baselines-h2gb C4), so the command's inputs are not ones our adapter supports (card rule: "else tuned").
- **GAT LP pubmed, lastfm → `tune`.** HGB's LP GNN README runs only `homoGNN.py --data amazon --model GCN`
  (`LP/benchmark/methods/GNN/README.md:2`); no GAT command for PubMed/LastFM, argparse defaults only.
- **R-GCN, MAGNN LP:** pubmed and lastfm R-GCN and lastfm MAGNN are filled; MAGNN pubmed has no script
  (`LP/.../MAGNN` has `run_DBLP/IMDB/LastFM.py` only) → `tune`.
- Other repositories checked for an empty cell: Hetero²Net README (DBLP, IMDB, ACM, MAG, RCDD; all filled or not
  ours), H2SGNN/PSHGCN `config.yaml` (DBLP, ACM, IMDB, AMiner), HINormer README (its Freebase is another subgraph,
  register note) → no further published cell.
- H2GB yaml oddity (not ours to fix): `Pokec/Pokec-GAT.yaml:61` says `layer_type: GCN` under the GAT file name.

## 3. Final cell table

Status: `filled` (register number; never run), `published` (5 seeds at the source configuration), `tune` (100 trials
seed 0, then seeds 0–4), `probe` (1-trial dry run at the smallest grid point on the H100, C5), `N/A (reason)`.
GC: no baseline cell is run (GC option (a), decisions 2026-11-20): every baseline GC cell is `filled` (HetSheaf
Table 4: GCN, GAT, NSD, R-GCN, HAN, HetSheaf) or leaves the table; MAGNN is not in GC (amendment b5adecf).

#### NC (Macro-F1)
| method | acm | dblp | imdb | freebase | ogbn-mag | mag-year | oag-cs | oag-eng | oag-chem | ieee-cis-g | h-pokec | pdns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gcn | filled | filled | filled | filled | filled | published ‡ | published ‡ | published ‡ | published ‡ | filled | published ‡ | published ‡ |
| gat | filled | filled | filled | filled | filled | published ‡ | published ‡ | published ‡ | published ‡ | filled | published ‡ | published ‡ |
| chebnetii | filled | filled | tune | tune | tune | tune | tune | tune | tune | tune | tune | tune |
| nsd | filled | filled | filled | tune | tune | tune | tune | tune | tune | tune | tune | tune |
| csnn | tune | tune | tune | tune | tune | tune | tune | tune | tune | tune | tune | tune |
| rgcn | filled | filled | filled | filled | filled | published | published | published | published | filled | published | published |
| han | filled | filled | filled | filled | filled | published | published | published | published | filled | published | published |
| magnn | filled | filled | filled | published | probe | probe | probe | probe | probe | probe | probe | probe |
| sehgnn | filled | filled | filled | filled | tune | tune | tune | tune | tune | tune | N/A (OOM, author 2026-11-16) | tune |
| hinormer | filled | filled | filled | tune | filled | published ‡ | tune | published ‡ | tune | tune | tune | published ‡ |
| pshgcn | filled | filled | published | tune | published ‡ | tune | tune | tune | tune | tune | tune | tune |
| h2gformer | tune | tune | tune | tune | published | published | published | published | published | filled | published | published |
| hetero2net | filled | filled | filled | tune | filled | tune | tune | tune | tune | tune | tune | tune |
| h2sgnn | filled | filled | published | tune | tune | tune | tune | tune | tune | tune | tune | tune |
| hetsheaf | filled | filled | filled | tune | tune | tune | tune | tune | tune | tune | tune | tune |

#### LP (ROC-AUC)
| method | acm_pa | dblp_ap | imdb_ma | freebase_bb | pubmed | lastfm | ogbn-mag_pp | h-pokec_uu |
|---|---|---|---|---|---|---|---|---|
| gcn | tune | tune | tune | tune | filled | filled | tune | tune |
| gat | tune | tune | tune | tune | tune | tune | tune | tune |
| nsd | tune | tune | tune | tune | tune | tune | tune | tune |
| csnn | tune | tune | tune | tune | tune | tune | tune | tune |
| rgcn | tune | tune | tune | tune | filled | filled | tune | tune |
| han | tune | tune | tune | N/A (OOM, author 2026-11-18) | tune | tune | tune | tune |
| magnn | tune | tune | tune | tune | tune | filled | probe | probe |
| pshgcn | tune | tune | tune | tune | tune | published | tune | tune |
| h2gformer | tune | tune | tune | tune | tune | tune | tune | tune |
| hetero2net | tune | tune | tune | tune | tune | tune | tune | tune |
| h2sgnn | tune | tune | tune | tune | tune | tune | tune | tune |
| hetsheaf | tune | tune | tune | tune | tune | tune | tune | tune |

Counts: NC 180 cells = 52 filled, 22 published, 16 published ‡, 81 tune, 8 probe, 1 N/A; LP 96 cells = 5 filled,
1 published, 87 tune, 2 probe, 1 N/A.

## 4. ‡ Published cells our runner cannot execute as the file states (forward note for C1)

Checked here because C1 depends on it; the C1 stop is reported in the card report.
- **GCN, GAT on H2GB (12 cells).** The yaml configures H2GB's `HomoGNNModel` (`gnn:` block, e.g.
  `mag-year-GCN.yaml:43-55`: `layers_pre_mp 0`, `layers_mp 4`, `layers_post_mp 1`, `dim_inner 512`, `batch_norm True`,
  `act relu`, `agg mean`, `use_linear False`). Our GCN/GAT on H2GB run **HGB's** GCN/GAT (`HGB/NC/benchmark/methods/GNN/GNN.py:10,58`:
  per-type input `Linear`, `GraphConv`/`GATConv` stack, ELU from our call, no batch norm, no post-MP layer) on H2GB's
  sampler (`adapters/hgb_gnn.py:277-316`, `minibatch.fit_homogeneous_on_hgt`; author decision S1, class A). `batch_norm`,
  `act`, `layers_post_mp`, `layers_pre_mp`, `agg` have no counterpart in that model.
- **HINormer on H2GB (3 cells).** The yaml configures H2GB's `HINormerModel` (`hin` sampler, `neighbor_sizes [50]`,
  `gnn.dim_inner 32`, `layers_mp 1`, `batch_norm True`: `mag-year-HINormer.yaml:23-53`); our adapter runs the original
  HINormer instead (registry `H2GB_NC_ADAPTER_PENDING["hinormer"]`: H2GB's model has its transformer commented out;
  card 1c F1, decision G-1). No key of that `gnn:` block maps onto the original HINormer's arguments.
- **PSHGCN ogbn-mag (1 cell).** `--batch-size 10000` (`main.py:203`), early stopping `--patience 100` per stage
  (`main.py:174, 204`), an epoch = one pass over the training rows. Our H2GB loop (`minibatch.minibatch_plan`,
  `train_index_minibatch`) accepts B ∈ {32, 256, 1024}, defines an epoch as 8,192 targets and has no early stop.
- Executable with code changes inside `src/baselines/` (not a stop): H2GB's `optim` keys (`optimizer adamW`,
  `base_lr`, `weight_decay`, `max_epoch 500`, `scheduler cosine_with_warmup` + `num_warmup_epochs 5`,
  `clip_grad_norm True` with H2GB's default value 1.0 (`H2GB/config/optimizers_config.py:28`), `batch_accumulation 32`
  with H2GB's step rule `custom_train.py:154-159`, scheduler stepped once per epoch `:288-293`), `train.batch_size 32`
  × `iter_per_epoch 256` = 8,192 targets (= our epoch), `eval_period 4`; HAN/R-GCN/H2G-former model keys through
  H2GB's own cfg (H2G-former's `overrides` today sets `gt.dropout` and `gnn.dropout` from one key; the yaml has 0.5 and
  0.0, so the published path must pass them separately).
