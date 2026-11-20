# Card 8 report: published configurations, R-GCN and MAGNN adapters, TSUBAME lists

**Verdict: stopped at C1** (card stop condition "a published source key cannot be executed exactly"). C0 is done.
C1–C6 were not started: no code, no config file, no job. An author decision is needed on 16 cells (§3).

Branch `card8` = `v3b` fc0ebaa + merge of design-docs b5adecf (merge commit 8be533e). Untracked `data.tar*` and
`docs/lab-gpu-cluster.html` in the work tree were left alone.

## 1. Units

| unit | commit | job | result |
|---|---|---|---|
| C0 inventory | b18f3ec | none (no code) | done: `docs/card8_evidence/c0_inventory.md`, `H2S-NN/configs/baselines/paper_filled_cells.tsv` |
| C1 published configs | none | none | **stop** (§3) |
| C2–C6 | none | none | not started (after a stop the card says do not continue) |

## 2. C0 result (details and file:line in `docs/card8_evidence/c0_inventory.md`)

Filled cells: 81 (NC 52, LP 5, GC 24); every register row mapped to one cell without guessing.

| task | cells | filled | published | published ‡ | tune | probe | N/A |
|---|---|---|---|---|---|---|---|
| NC (15 methods × 12) | 180 | 52 | 22 | 16 | 81 | 8 | 1 (SeHGNN h-pokec) |
| LP (12 methods × 8) | 96 | 5 | 1 | 0 | 87 | 2 | 1 (HAN freebase_bb) |
| GC | baseline cells not run (option (a)) | 24 | — | — | — | — | rest |

Published (22 + 1): HAN, R-GCN × 6 H2GB datasets (H2GB yaml); H2G-former × 7 (H2GB yaml); PSHGCN imdb, H2SGNN imdb
(existing files); MAGNN freebase (`run_Freebase.py`); LP PSHGCN lastfm (existing file).

Differences from the card's expected list (no stop):

| cell | card expected | C0 | reason |
|---|---|---|---|
| SeHGNN ogbn-mag | published if inputs supported | tune | README command needs label features and 4 self-training stages (`SeHGNN/ogbn/Readme.md:8-10`); our adapter is stage 0 without label features |
| GAT LP pubmed, lastfm | check | tune | HGB LP GNN README has only `--data amazon --model GCN`; argparse defaults only |
| MAGNN LP pubmed | where empty | tune | no MAGNN PubMed script in HGB LP |
| MAGNN NC freebase | published | published (stated) | `run_Freebase.py` is Freebase-specific, but MAGNN's README does not list it |

## 3. Stop: published sources our runner cannot execute as written (C1)

| cells | source keys not executable | our model | file:line |
|---|---|---|---|
| GCN, GAT × {mag-year, oag-cs, oag-eng, oag-chem, h-pokec, pdns} (12) | `gnn.batch_norm True`, `gnn.act relu`, `gnn.layers_post_mp 1`, `gnn.layers_pre_mp 0`, `gnn.agg mean`, `gnn.use_linear` (H2GB `HomoGNNModel`) | HGB's GCN/GAT (no batch norm, ELU, no post-MP layer), decision S1 class A | `H2GB/configs/<D>/<D>-GCN.yaml:43-55` (GAT similar; Pokec `gnn:` l.60); `HGB/NC/benchmark/methods/GNN/GNN.py:10,58`; `src/baselines/adapters/hgb_gnn.py:277-316` |
| HINormer × {mag-year, oag-eng, pdns} (3) | the whole `gnn:` block of H2GB's `HINormerModel`, `train.sampler hin`, `neighbor_sizes [50]` | the original HINormer (H2GB's model not used, card 1c F1, G-1) | `<D>-HINormer.yaml:23-53`; `registry.py` `H2GB_NC_ADAPTER_PENDING["hinormer"]` |
| PSHGCN ogbn-mag (1) | `--batch-size 10000`, early stop `--patience 100`, epoch = one pass over the training rows | our H2GB loop: B ∈ {32, 256, 1024}, 8,192 targets per epoch, no early stop | `PSHGCN/ogbn-mag/main.py:174, 199-204`; `src/baselines/minibatch.py` `minibatch_plan`, `train_index_minibatch` |

The rest of C1 is executable with changes inside `src/baselines/`: H2GB `optim` keys (AdamW, cosine with 5 warm-up
epochs stepped per epoch, clipping at H2GB's default 1.0, accumulation 32 with H2GB's step rule
`custom_train.py:154-159`), batch 32 × 256 iterations = our 8,192-target epoch, eval period 4. H2G-former's published
path must set `gt.dropout` and `gnn.dropout` separately (0.5 and 0.0 in the yaml; `overrides` today sets both from one key).

### Options for the author

| option | what | for | against |
|---|---|---|---|
| A | ‡ cells become `tune` (the yaml configures a different model from the one we run) | no new code; keeps decision S1 and G-1 | 12 GCN/GAT cells lose a published configuration |
| B | GCN, GAT published cells run H2GB's own `HomoGNNModel` (`layer_type` GCN/GAT), as the card already plans for R-GCN | "follow the source's code exactly"; the same code path as H2GB R-GCN (C2), so little extra code | the GCN/GAT rows then use two implementations (HGB's on tuned cells, H2GB's on published cells); departs from S1 for these cells |
| C | PSHGCN ogbn-mag: allow B = 10,000, its epoch and early stop for this one published cell | runs their command | a second H2GB loop variant for one cell |

Recommendation: **B for GCN/GAT** (H2GB's published configuration belongs to H2GB's model, which the card already
brings in for R-GCN), **A for HINormer** (H2GB's HINormerModel was rejected as broken, G-1) and **A for PSHGCN
ogbn-mag** (one cell; a second loop for it is not worth the preparation time).

## 4. Deviations

None in code (no code written). C0 judgement calls are listed in §2.

## 5. Open questions

1. Options A/B/C above for the 16 ‡ cells (decides C1, C4 and C5 counts).
2. MAGNN NC freebase counted as published from a script its README does not list: confirm.
3. `Pokec-GAT.yaml:61` says `layer_type: GCN` (H2GB's file). If option B is chosen, run it as written (GCN) or treat
   the cell as `tune`?

branch=card8 sha=b18f3ec04e282ccd26b07b4617161df9cf47e216 card=docs/H2S-NN_card8_published_rgcn_magnn_card.md base=fc0ebaa605280f2d987c4848cc3109c7d3b66680
