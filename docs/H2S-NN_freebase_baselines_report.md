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

B0–B2 head (reviewed and accepted): `59822692b1182a5e958ee07dd13213a5f3c0a1dd` on base `a8828a8`.

## B3 — review follow-up (card section "Part B — B3", design-docs `ca98dba`): **stopped at B3.4**
| Item | Commit | Result |
|---|---|---|
| B3.1 merges | `d2f5282` (design-docs `ca98dba`), `2e0b48b` (v3b `9b7ae0b`) | Only the two named conflicts. `test_baselines.py` keeps both sides' comments; `link_prediction_dgl.ipynb` comes from the generator (`--check` OK). Registry after the merge: **274 / NC 155 / LP 79 / GC 40** (job 17668, `b3_1_merge.txt`). The auto-merged `sehgnn.py` and `registry.py` carry exactly v3b's own hunks. |
| B3.2 B1a wiring | `34689e8` | `h2sgnn.py`: `val_subset_loader(..., run_seed=bundle.metadata.get("seed"))` passed as `val_subset_loader=` and `val_subset_ids=`. `pshgcn.py` (H2GB path): `run_seed=` passed to `train_index_minibatch`. `minibatch.py` and `validation_subset.py` are not edited. |
| B3.3 test + mutation | `f953cbe` | `B1aSubsetTests`, run on a synthetic graph registered as `pdns` with seed 1, for both paths. The ids evaluated during training equal `validation_subset("pdns", val ids, 1)`; test ids appear only in the final test pass; the record has `val_subset_macro_f1` and the matching fingerprint. Mutation (`b3_mutation.py`): PSHGCN without `run_seed` makes its test fail (`ValueError`: PDNS needs `run_seed`); H2SGNN still passes; the real code passes both. |
| B3.4 targeted job | (this commit) | **Stop: OOM.** Job 17676 (g2tn01, CPU, `--mem 16G` as prescribed) was killed at MaxRSS 16.0 GiB after 17 min, `OUT_OF_MEMORY` (`b3_4_tests_17676.txt`, `b3_4_sacct_17676.txt`). Unittest printed no summary, so there is no pass/fail result for the 8 modules. The mutation, run after it in the same job, gave the expected outcome (`mutation_exit=0`). |
| B3.4 rerun, 32G (ops 2026-11-18) | `d0af22b` | Job **17686** (g2n01, CPU, `--mem 32G` as an `sbatch` override; the script is unchanged): **130 tests OK**; mutation as expected. Peak MaxRSS **28,210,588 KiB (26.9 GiB)** (`b3_4_tests_17686.txt`, `b3_4_sacct_17686.txt`). |
| v3b merge + rerun, 32G | `9ed2f2b` (merge of v3b `0b0219b`, no conflict) | Job **17690** (g2n01, CPU, 32G) on the merged head: **130 tests OK**; mutation as expected. Peak MaxRSS **28,211,052 KiB (26.9 GiB)** (`b3_4_tests_17690.txt`, `b3_4_sacct_17690.txt`). |

**Stop item for review.** The 8 modules together need more than 16G in one process. This is consistent with B1: the
same group of tests was OOM-killed at 16G and passed at 32G (`b1_tests_dev3.txt`), and B2 ran at 32G. Proposed:
rerun `b3_4_job.sh` unchanged except `--mem 32G` (or split the modules over two 16G jobs). I did not rerun it,
because the prompt fixes 16G.

branch=freebase-baselines sha=<see final message> card=docs/H2S-NN_hetsheaf_freebase_card.md base=0b0219bba24276396fcbd8af8f9b8a59b610c748
