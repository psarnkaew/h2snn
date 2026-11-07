# G0 — pipeline-correctness gate: pre-registration

Written 2026-11-08, before any G0 script was run. Nothing below was changed after seeing a result;
later additions are appended under a dated heading.

## Sources (opened)
- Reference numbers: HetSheaf (`H2S-NN/references/HETSHEAF.pdf`, Table 2, p. 9, rendered page read):
  R-GCN macro-F1 ACM 93.27 ± 0.82, DBLP 94.30 ± 0.23, IMDB 59.67 ± 0.21 (micro 93.19 ± 0.86,
  94.70 ± 0.20, 62.36 ± 0.20); five seeds 42–46; hyperparameters chosen by a 150-run Bayesian
  sweep per model (App. C.1.1). The selected configuration is **not published** (only the search
  space, Table 5).
- Reference implementation: `H2S-NN/vendor/Heterogeneous-Sheaf-Neural-Networks` @ 6964e1e
  (`hetsheaf/models/gnn_baselines/RGCN.py`, `hetsheaf/datasets/hgb.py`, `hetsheaf/datasets/utils.py`,
  `hetsheaf/node_classification/node_classifier.py`, `hetsheaf/scripts/run_gnn_nc.py`,
  configs `nc_config.yaml`, `model/rgcn.yaml`, `trainer/trainer.yaml`, `callbacks/sheaf_nc.yaml`).
- Official HGB split rule: HGB's `utils/data.py` as vendored in `H2S-NN/vendor/HINormer` @ a29fcac
  (lines 21–32): test = every node in `label.dat.test`; validation = the first `int(0.2·n)` of an
  **unseeded** `np.random.shuffle` of the `label.dat` nodes; train = the rest. HGB therefore fixes
  the test set and the train∪val pool, not a single train/val partition.

## Baseline choice
R-GCN, not SimpleHGN: SimpleHGN's official code is DGL-based and DGL does not import in `h2snn`
(D11 U5 probe); R-GCN runs with PyG's `RGCNConv`, its implementation is vendored, and the spec
(§6) names its HetSheaf number as the reference.

"Published configuration" = the committed defaults of the vendored repo, the only configuration on
record: `RGCN(layers=2, hidden_channels=32, dropout=0.2, input_dropout=0, initial_dropout=0,
output_dim=32)` in **embedding mode** (`num_nodes > 0` is always passed by `run_gnn_nc.py`, so no
node features are used), linear decoder, AdamW lr 1e-3, weight decay 5e-4, cosine LR (T_max 1000,
eta_min 1e-6, per epoch), max 200 epochs, early stopping on validation loss with patience 100,
test with the weights at the end of training (`trainer.test(classifier, datamodule)` without a
checkpoint path), seed 42. Deviation stated up front: these may differ from the sweep winner
(UNVERIFIED); the control arm below measures what this configuration gives on their own pipeline.

## (a) Split check — measurements and expectations
For ACM, DBLP, IMDB, compare our `build_dataset` (config `configs/experiments/<ds>_h2snn.yaml`)
against HGB's own loader (`vendor/HINormer/utils/data_loader.py`, which parses the raw files
independently of PyG):
1. our test node set == HGB `labels_test` mask (exact);
2. our train ∪ val == HGB `labels_train` mask (exact); train ∩ val = ∅; |val| vs HGB's `int(0.2 n)`;
3. labels equal on every labelled node (ACM/DBLP class index; IMDB multi-hot vector, exact);
4. per-split sizes and class counts; local index of target type == HGB global id (target type 0).
Expected: 1–3 hold exactly; |val| may differ by 1 (we round, HGB floors). The train/val partition
necessarily differs from any particular HGB run (theirs is unseeded) — reported, not a failure.
Wrong if: any of 1–3 fails.

## (b) Reference baseline — arms
- **O (our pipeline):** our `build_dataset` (PyG HGBDataset → `flatten_edges` → tied fact layer),
  our masks, our labels, our metric code (`src/utils/misc.py`), our selection rule (test at the
  epoch of best validation macro-F1). R-GCN needs messages in both directions: each fact
  `(u, r, v)` is given as relation `r` plus its reverse as relation `r + R` (this reconstructs the
  stored HGB edges exactly for tied relations). Also reported: test at the final epoch (their rule).
- **P (their pipeline, control):** PyG HGBDataset on the same raw files with their transform chain
  (`Constant`, `ToUndirected`, `SubsetTrainSplit(val_ratio=0.1)` with a fresh `torch.Generator()`,
  `RemoveSelfLoops`, `NormalizeFeatures`, `RemoveDuplicatedEdges`, `to_homogeneous`), their split,
  final-epoch test. Lightning is not installed, so the loop is re-implemented without it (one
  full-batch step per epoch, validation each epoch, same optimizer/scheduler/early stopping).
- Same model code (vendored `RGCN.py` imported by file path), same config, seed 42, one run per
  arm and dataset, GPU.

## Gate rule (fixed now)
Band = published mean ± 2 published std (macro-F1): ACM [91.63, 94.91], DBLP [93.84, 94.76],
IMDB [59.25, 60.09].
- **PASS** if arm O is inside the band on ACM **and** DBLP (the two datasets the spec names).
  IMDB is reported against its band but does not decide the gate (multilabel; metric conventions
  differ more across codebases).
- If O misses: look at P.
  - P inside the band and O outside → **FAIL, our pipeline**: diagnose by swapping one component
    at a time between O and P (split, graph construction, selection rule, metric) until the gap is
    located; no fix in G0.
  - P also outside the band, and |O − P| ≤ 2 published std → **INCONCLUSIVE (configuration)**:
    the unpublished sweep winner, not our pipeline, explains the miss; reported as such.
  - P outside and |O − P| > 2 published std → **FAIL**, diagnose as above.
- Secondary, reported for every dataset: |O − P| (same model, same config, different pipeline).

Wrong if: the gate passes yet a component swap shows a > 2 std effect of our pipeline (would mean
the pass is luck); or O ≫ band (would suggest leakage — check test labels are never used).

Weakness stated now: one seed, std estimates from 5 seeds; a 2-std band is a coarse instrument.
It detects a 6–10 point pipeline error, which is what the gate is for; it does not certify
sub-point agreement.

## Addendum 2026-11-08 (written after the six gate runs, before any diagnostic run)
Gate outcome per the rule above (`g0_rgcn_acm_output.txt`, `g0_rgcn_dblp_imdb_output.txt`):
ACM — O 89.04 (best val) / 88.47 (final), P 91.20 (final): both outside the band, |O − P| = 2.73 > 1.64 → **FAIL, diagnose**.
DBLP — O 92.08 / 91.96, P 91.93: both outside, |O − P| = 0.03 ≤ 0.46 → INCONCLUSIVE (configuration).
IMDB (not gate-deciding) — O 50.74 / 51.03, P 55.30: |O − P| = 4.27.
Metric ruled out already: our sklearn code and torchmetrics agree to < 1e-6 on every final epoch.
Diagnostic swaps (one component each, same model/config/seed 42, final-epoch test = the rule P uses):
 S1 split: O's graph with P's split (10 % val);  S2 graph: P's graph with O's split (20 % val).
 If S2 moves O toward P: decompose the graph difference into (g1) ToUndirected's duplicate reverse
 relation types, (g2) self-loop removal, (g3) duplicate-edge removal, one at a time.
Expected (stated now): the split explains part (12.5 % more training labels in P); the graph explains
the rest. Caveat fixed now: single seed; a swap effect below ~1 point is not distinguishable from
seed noise without more seeds (multi-seed runs need approval).
