# Card 8 report: published configurations, R-GCN and MAGNN adapters, TSUBAME lists

**Verdict: done.** C0–C6 are complete after amendments 1–5. One CPU job (18015, 36G, no GPU) ran 242 targeted tests:
all OK, none skipped. Every card mutation is caught. H2S-NN is untouched:
`git diff fc0ebaa -- H2S-NN/src/models H2S-NN/src/trainers H2S-NN/configs/experiments` is empty. No real-data run.

Branch `card8` = `v3b` fc0ebaa plus merges of design-docs b5adecf, 790ff68, 377f16e and 3489b8e.

## 1. Units

| unit | commit | job | result |
|---|---|---|---|
| C0 inventory (+ amendments 1, 3, 4, 5) | b18f3ec, 1000b0c, 5683753, 0bc171a | — | 73 filled cells (NC 49, GC 24); table in `docs/card8_evidence/c0_inventory.md` |
| C1 published configurations | c6bfd39, 9b4c55d | 17963, 17987 | 23 + 13 tests OK; mutation "scheduler dropped from the H2GB mapping" caught |
| C2b LP step size, DistMult | d68f9c5 | 17989 | 16 OK; mutations "one step per epoch" and "diagonal DistMult" caught |
| C2 R-GCN | ba4cdea | 17997 | 8 OK |
| C3 MAGNN | 49517de | 18002 | 19 OK; mutation "instances on the full graph" caught |
| C4 registry, search spaces | 836ad74 | 18013 | 119 OK |
| C5 TSUBAME lists | 31eed94 | 18014 | 13 OK; mutation "no filled exclusion" caught |
| C6 targeted job | a44218d | 18015 | 242 OK, 0 skipped; all 5 mutations caught |

## 2. Cell statuses and lists

| task | filled | published | tune | probe | N/A |
|---|---|---|---|---|---|
| NC (15 × 12) | 49 | 33 | 89 | 8 | 1 |
| LP (12 × 8) | 0 | 5 | 88 | 2 | 1 |
| GC | 24 | — | — | — | baseline cells not run (option (a)) |

| phase | lines | `node_f` tasks |
|---|---|---|
| `h2snn` | unchanged (byte-identical to fc0ebaa, tested) | — |
| `baselines-early` | 4 (byte-identical) | 1 |
| `baselines-1` (rebuilt: tune cells only) | 61 | 16 |
| `baselines-2` (new) | 112 | 28 |
| `published` (new: `run_baseline_published.py … --final_seeds 0 1 2 3 4`) | 38 | 10 |
| `probe` (new: 1 trial, `--dry_run --grid_point min`) | 10 | 3 |

The tuning lines (4 + 61 + 112 = 177) equal the 89 + 88 tune cells.

## 3. Adapters changed by C2b

There is no adapter-specific LP epoch loop. Every small-LP adapter trains through the shared `training.train_link_prediction` with `**budget_kwargs(LINK_PREDICTION, params)`, so the batch of 8,192 positives per step applies to all of them:
- `hgb_gnn.py` (GCN, GAT);
- `_homogeneous_lp.py` (NSD, CSNN);
- `hgb_han.py`, `pshgcn.py`, `h2sgnn.py`, `hetero2net.py`, `h2gformer.py`, `hetsheaf.py`;
- `rgcn.py`, `magnn.py` (new).

A static test checks this. Large LP (`fit_large_lp`) was already per 8,192 and is unchanged. The shared `DistMultDecoder` is now a full `[R, d, d]` matrix per relation (HGB's init), and every LP baseline uses it.

## 4. Decisions and deviations

- **Published H2GB cells (30):** each runs H2GB's own model and the yaml's optimizer, cosine schedule, clipping, accumulation, epochs, eval period and fan-out.
  - Data, evaluation and logging keys stay ours (`h2gb_published.EXCLUDED_KEYS`; protocol option 1).
  - On the binary datasets the network has `C` outputs with cross-entropy (the existing stated difference).
- **GCN × PubMed/LastFM LP (amendment 4):** run in our loop with validation once per epoch. HGB's patience of 3 validations is applied as 3 epochs.
- **MAGNN:**
  - ACM/DBLP/IMDB NC are filled and not wired: ACM's preprocessing adds self-loops and IMDB uses full-batch `MAGNN_nc`. The registry carries the reason.
  - Off HGB, MAGNN's instances are enumerated by our code in MAGNN's format; this is tested equal to HGB's `get_adjlist_pkl`.
  - The LP sampler re-implements `parse_adjlist_LastFM` for generic endpoint types, because HGB's version hard-codes the LastFM types and `.to('cuda')`.
  - When a whole batch has no instance of a meta-path, each target gets one self-instance (MAGNN LP's own rule). Without this, MAGNN's code crashes on an empty reshape (`base_MAGNN.py:150`).
  - Large LP (probe cells) uses MAGNN's NC layer on each sampled subgraph, without the training mask.
- **R-GCN:**
  - HGB's `EntityClassify` is rebuilt on HGB's own `BaseRGCN` with its three builders verbatim, because `entity_classify.py` calls `pynvml.nvmlInit()` at import.
  - The LP width grid is {40, 60, 80, 120}, because `"bdd"` needs width divisible by `n_bases`; HGB's default 64 is left out.
  - Tuned R-GCN cells on H2GB use H2GB's own R-GCN with the tuned keys, as HAN's class-A path (there are no such cells today).
- **GC option (a):** every baseline GC cell is unsupported with that reason, and the code is kept. The GC lane notebooks are no longer generated; the old files are kept because the repo rules say notebooks are not deleted.
- **Existing tests updated, with reasons in comments:**
  - registry counts (runnable 271);
  - the contract and published-runnable tests skip GC option (a);
  - `SearchSpaceCoverageTests` checks only files of tuned cells;
  - the large-LP method list gains R-GCN and MAGNN;
  - `test_fanouts_equal_h2gb_yaml` expects 40 yamls instead of 32;
  - the two TSUBAME list tests expect tune cells only.
- **Bugs found and fixed in C4:**
  - The R-GCN/MAGNN vendor blocks left HGB's `utils` and `model` modules in memory; this broke HAN in one process. The blocks now also push their HGB root, and a regression test covers it.
  - The PSHGCN LastFM yaml from C2b did not parse.
- **Memory:** jobs touching `test_partP_freebase` need 32G (peak 28 GB, the known Freebase-test peak). Interim jobs used 24–32G; C6 used 36G.

## 5. Open (not blocking)

- `collect.py` / `status.py` read published lines through `lines.py`; `collect.py` does not yet put `published_summary.json` into `results.csv` (ops).
- The probe lines decide MAGNN N/A on the H100. Every first TSUBAME line is the real-data check of its cell.

branch=card8 sha=a44218d974814cf356c1bb0ef31268d017a77f87 card=docs/H2S-NN_card8_published_rgcn_magnn_card.md base=fc0ebaa605280f2d987c4848cc3109c7d3b66680
