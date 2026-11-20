# Card 8 report: published configurations, R-GCN and MAGNN adapters, TSUBAME lists

**Verdict: stopped again in C1**, this time on Amendment 3 ("if a value cannot be found or executed exactly, stop as
C1"). R-GCN × LastFM LP cannot run with HGB's stated configuration under our DGL. C0 is updated for Amendments 1 and 3.
The tested part of C1 is committed and labelled partial. C2b–C6 were not started.

Branch `card8`: `v3b` fc0ebaa + design-docs b5adecf, 790ff68 (amendments 1–2) and 377f16e (amendment 3), all merged.

## 1. Units

| unit | commit | job | result |
|---|---|---|---|
| C0 inventory | b18f3ec | — | done (first stop, ‡ cells) |
| C0 update, amendment 1 | 1000b0c | — | 11 GCN/GAT H2GB cells published (H2GB `HomoGNNModel`); GAT h-pokec, HINormer ×3, PSHGCN ogbn-mag tuned |
| C0 update, amendment 3 | 5683753 | — | 5 LP rows out of `paper_filled_cells.tsv` (76 filled: NC 52, GC 24); LP sources with file:line (inventory §6) |
| C1 published configs | c6bfd39 (**partial**) | 17963 (CPU, 24G) | 23 tests OK; mutation "drop scheduler from the H2GB mapping" caught. The 5 LP cells of amendment 3 are not done |
| C2b, C2–C6 | — | — | not started |

Jobs 17959 (36G, never scheduled: the nodes had ≈32G free next to another user's jobs; cancelled) and 17962 (2 failures,
both test-side: the warm-up's first rate is `max(1e-6, 0) × lr = 1e-9`, `extra_optimizers.py:194-196`; an existing
fan-out test counted 4 × 8 yamls and now counts 5 × 8 with R-GCN) were replaced by 17963. Logs are in `docs/card8_evidence/`.

## 2. What C1 (partial) contains

- `src/baselines/h2gb_published.py`: 30 published H2GB-yaml cells (GCN 6, GAT 5, HAN 6, R-GCN 6, H2G-former 7).
  - **Setup:** H2GB's cfg is set from the yaml (`configure_from_yaml`), its own network is built (`HomoGNNModel` / `HeteroGNNModel` / `GTModel`), and so are its optimizer and scheduler (`create_optimizer`, `create_scheduler`).
  - **Step rule:** H2GB's (`custom_train.py:106-159, 288-293`): gradients summed over `batch_accumulation` batches, a step when `(it+1) % acc == 0 or it+1 == len`, clipping at 1.0, one scheduler step per epoch.
  - **Kept from our protocol:** the data, evaluation and logging keys, listed in `EXCLUDED_KEYS`.
- Wiring: `minibatch.SourceSchedule` (opt-in; the tuned path is unchanged), `hgb_gnn`/`hgb_han`/`_h2gformer_h2gb` published branches.
- 30 config files (dotted H2GB keys, a yaml line for each) and `magnn/node_classification/freebase.yaml` (`run_Freebase.py` lines).
- `published.refuse_source_schedule`, called in `campaign.fit_cell`: a tuned run carrying an optimizer, scheduler, clipping, accumulation or LP batch key is refused.
- `cell_status.py`: the cell statuses (filled, N/A, probe, published, tune).
- Published runner: base seed 0 by default, `--final_seeds`, `--state_dir`, and filled cells are skipped.

## 3. Stop: amendment 3 LP cells

| cell | issue | file:line |
|---|---|---|
| R-GCN × lastfm | HGB's command `link_predict.py --dataset=LastFM` → width 64, `n_bases` 10, `RelGraphConv(..., "bdd", ...)`; DGL 2.4 `TypedLinear` raises "Input and output sizes must be divisible by num_bases" (64 % 10 = 4) | `RGCN/README.md:24`, `link_predict.py:30, 364, 379`; `dgl/nn/pytorch/linear.py` (bdd branch) |
| GCN × pubmed, lastfm (not a stop by itself) | HGB validates every 10 batches inside an epoch, with early stopping after 3 validations on the validation loss; our shared loop validates once per epoch. Running it exactly needs an intra-epoch validation option in the shared LP loop (published runs only) | `GNN/homoGNN.py:46-65`, `utils.py:159` |
| MAGNN × lastfm (not checked) | settings transcribable (`run_LastFM.py:18-20, 21-25, 71, 225-236`); executability depends on the MAGNN LP adapter (C3), including its training masks (`use_masks`) | |

R-GCN × pubmed is executable as stated (README width 60).

### Options (author)

| option | R-GCN × lastfm | for / against |
|---|---|---|
| A | `tune` (100 trials; its search space already respects divisibility) | no invented value; but the cell is no longer "HGB's configuration" |
| B | published with `n_bases` that divides 64 (e.g. 8) or with width 60 (HGB's PubMed width), stated as a deviation | keeps 5 seeds without tuning; but one value is ours, not HGB's |
| C | published with a basis regulariser instead of "bdd" | different model; not recommended |

Recommendation: **A** for R-GCN × lastfm (an exact HGB run is impossible, and a chosen value is not "HGB's
configuration"). For GCN × pubmed/lastfm: allow an intra-epoch validation option (every 10 steps, patience counted in
validations, selection on ROC-AUC as ours) for published LP runs.

## 4. Uncommitted drafts (not part of any commit)

- `H2S-NN/src/baselines/adapters/rgcn.py` (C2 draft, untracked)
- C2b drafts (`training.py`, `heads.py`, `budget.py`, test file) in the session scratchpad

None was tested. Both continue after the author's decision.

## 5. Deviations

- C1 is committed partial, after the stop, so the tested H2GB/MAGNN-Freebase work is not left uncommitted. Until C1
  completes, the 5 amendment-3 LP cells have status `tune` in `cell_status` (no published file yet). C5 reads that
  status, and C5 has not run.
- Existing test `test_baselines_minibatch.FanoutTests.test_fanouts_equal_h2gb_yaml`: count 32 → 40 (R-GCN's 8 yamls
  added to `H2GB_CONFIG_METHODS` and checked equal to the yaml).
- Interim test jobs used `--mem=24G` (36G could not be scheduled); C6 keeps 36G.

branch=card8 sha=c6bfd39886a3ca72c49987ee78b7bfd2e7275029 card=docs/H2S-NN_card8_published_rgcn_magnn_card.md base=fc0ebaa605280f2d987c4848cc3109c7d3b66680
