# Card 3 Part B report — Freebase NC baselines; PSHGCN / H2SGNN on H2GB NC and large LP

Card: `docs/H2S-NN_hetsheaf_freebase_card.md` Part B, including the B2 amendment of 2026-11-18. Base: v3b `a8828a8`
plus design-docs `1497290` (merge `250fc39`). Slim scope: no real-data run, no dry run, no metric. No A6000 or g3an01
was used. `minibatch.py` is not edited.

## Units
| Unit | Commit | Content |
|---|---|---|
| B0 | `d37abbb` | Preconditions: vendor file:line of each dataset branch, Freebase schema, batching choices (`freebase_baselines_evidence/b0_preconditions.md`). |
| B1 | `29ec019`, fix `2c588ef` | Wiring and Slim tests (below). Stated deviations are in `freebase_baselines_evidence/deviations.md`. |
| B2 | (this commit) | Full suite on g2n01, `--mem 32G`, pinned kernel env: **750 tests OK** at `2c588ef` (job 17658, `b2_suite_17658.txt`). |

## B1 wiring
- **Freebase NC.**
  - SeHGNN: its own Freebase branch (`_freebase_inputs`), without the dense eye.
  - PSHGCN and H2SGNN: the F-1 generic branch (`adapters/_hgb_generic.py`). Parallel relations are merged, and
    missing reverses are completed (author decision 2026-11-18, D1).
- **H2GB NC: shared loops only, no own loop.**
  - PSHGCN on 6 datasets: their ogbn-mag decoupled scheme made generic (D3), trained by `train_index_minibatch`.
  - H2SGNN on 8 datasets: H2GB's `hgt` sampler, trained by `train_minibatch_node_classification`.
- **Large LP** (`ogbn-mag_pp`, `h-pokec_uu`): both models use the shared `fit_large_lp`.
- **Registry.** 275 runnable cells, all with an adapter. The pending entries for PSHGCN/H2SGNN and the Freebase
  unsupported entries are removed.

## B1 tests (`tests/test_freebase_baselines.py`; logs `b1_tests.txt`, `b1_tests_dev3.txt`)
- **ACM-branch equivalence.** PSHGCN: `adjs`, `lis` and `lis_t` are exactly equal. H2SGNN: `adjs`, meta-path names
  and operators are exactly equal. For the H2SGNN operators, the comparison is on the target block: their ACM crop is
  the constant `target_max = 3024` (`processing.py:87`), equal to ours on real ACM but not on the 20-paper synthetic
  release. Diagnostic: `b1_h2sgnn_equiv_diag.*`.
- **Reverse completion.**
  - It is a no-op where reverses are stored.
  - On a one-direction toy graph, the operators get both directions and a symmetric same-type key.
  - Mutation: with the transpose dropped, the toy test fails (`b1_reverse_mutation.*`).
- **One step per adapter path is finite and moves the parameters:**
  - Freebase-shaped release: all three models, including a parallel relation;
  - H2GB: PSHGCN target rows, where batching is exact, and an H2SGNN sampled batch.
- **Large LP held-out route.** PSHGCN's per-node precomputation reads the message graph only; a negative control is
  included.

## Items for review
1. **D1 reverse completion** (author decision). Citations checked at the pinned SeHGNN: `hgb/utils.py:450,
   461-469`. The prompt's `446,459-467` is a few lines off; line 446 is a `print`. PSHGCN `hgb-lp`:
   `scripts/data_loader.py:44, 514-517`. B0's citation is corrected.
2. **Freebase K explosion.** The generic PSHGCN term lists on Freebase have 64 keys. This is a cost for the tuning
   budget, not a correctness item.
3. **SeHGNN Freebase published values out of the shared grids.** The published lr and weight decay are 3e-5. No
   `freebase.yaml` is committed, because the reachability test would fail (D4). Author's choice: widen the grids or
   leave Freebase without a published point.
4. **PSHGCN decoupled scheme on non-MAG H2GB** (D3): K = 2 and no label part. The H-Pokec PSHGCN override (HGB
   space) does not apply to the decoupled search space.
5. **Large-LP optimiser.** It is the shared single AdamW, not the vendors' per-group optimisers.
6. **Freebase row order.** HGB's `node.dat` is not in id order (61,021 inversions). Labels and splits are mapped by
   `hgb_target_rows` / `project_bundle_split`, and a mismatch raises.
7. **The design-docs merge brought in `H2S-NN/paper/` files.** I did not edit them.
8. **B2 ran twice.**
   - Run 17655 at `29ec019`: 10 failures in `test_budget_table`, because `h2sgnn` lacked `READS_BATCH_PLAN`. Its
     H2GB path already read `minibatch_plan`; only the flag was missing. Fixed in `2c588ef`.
   - Run 17658 at `2c588ef`: 750 tests OK.

branch=freebase-baselines sha=<see final message> card=docs/H2S-NN_hetsheaf_freebase_card.md base=a8828a8c002e17714cbd588d147b8c9f128fca1f
