# H2S-NN — D12 report (task card Part G)

**Status:** G0 only (2026-11-08). G1–G4 not started; they wait for the author's reply to this section.
Branch `d12-encoder` (off `tier0-screen-d11`). Evidence: `docs/D12_evidence/`. Hardware: RTX 4060 8 GB, Windows, env `h2snn` (torch 2.6, PyG 2.8.0, torchmetrics 1.9.0).

## G0 — Pipeline-correctness gate

### 1. Summary
- **(a) Splits: PASS.** For ACM, DBLP and IMDB, our test set is exactly HGB's `label.dat.test`, train ∪ val is exactly `label.dat`, the validation share is HGB's 20 % (same sizes as HGB's `int(0.2 n)`), and all 3025 / 4057 / 4573 labels match. HGB's own loader was the reference.
- **(b) Reference baseline: outside the band on every dataset; per the pre-registered rule, ACM = FAIL (diagnosed), DBLP = INCONCLUSIVE.** R-GCN (HetSheaf's code and committed config, seed 42) on our pipeline scored ACM 89.04 and DBLP 92.08, against bands [91.63, 94.91] and [93.84, 94.76].
- **Diagnosis:** I found no defect in our data. The gap between our pipeline and HetSheaf's comes from two HetSheaf choices (a 10 % validation share instead of HGB's 20 %, and `ToUndirected`, which duplicates every relation). The rest of the gap to the published number is the unpublished sweep configuration: HetSheaf's own pipeline also misses with that configuration.
- **What this means for D12:** on the identical split, labels, metric and selection rule, a feature-free R-GCN beats H2S-NN (D11 screen, seed 42) by **5.0 points on ACM and 9.9 on DBLP**. The 6–10-point deficit therefore lies in the model, training or feature path, not in splits, labels, metric or selection.

### 2. Files changed (no `src/` or `tests/` change)
`docs/D12_evidence/g0_preregistration.md` (rules, written before any run; dated addendum written before the diagnostics) · `g0_splits.py` + `g0_splits.json`/`_output.txt` (a) · `g0_rgcn.py` + `g0_rgcn_<ds>_<arm>.json` (per-epoch curves) + `g0_rgcn_*_output.txt`, `g0_swaps_output.txt`, `g0_graph_decomp_output.txt` (b) · this report.

### 3. Deviations from the card
- **Published configuration.** HetSheaf selected R-GCN with a 150-run sweep and publishes only the search space (App. C.1.1, Table 5; VERIFIED-SOURCE). I used the only configuration on record, the repo's committed defaults (2 layers, hidden 32, dropout 0.2, embedding mode, AdamW 1e-3 / 5e-4, cosine, 200 epochs, early stopping on validation loss with patience 100, test with final weights). Whether this matches the sweep winner is UNVERIFIED. Arm P was added to measure what this configuration gives on HetSheaf's own pipeline.
- **Lightning not installed.** HetSheaf's training loop was re-implemented without it: same optimizer, scheduler, early stopping and loss; metric checked against torchmetrics.
- **SimpleHGN not run.** Its code is DGL-based, and DGL does not import in `h2snn`.
- **Extra runs.** Ten single-seed diagnostic runs (swaps). There were no multi-seed runs.

### 4. Verification evidence
`python ../docs/D12_evidence/g0_splits.py` (from `H2S-NN/`): `test_equal: true`, `train_union_val_equal_label_dat: true`, `train_val_disjoint: true`, `label_mismatch_count: 0` on all three datasets; sizes are 726/181/2118 (ACM), 974/243/2840 (DBLP) and 1097/274/3202 (IMDB). The test class counts equal HGB's. `python ../docs/D12_evidence/g0_rgcn.py` also checked that our tied facts plus their reversals reproduce the stored HGB edge multiset exactly (`reconstructs_stored_edges: true`; 547,872 / 239,566 / 86,642 edges).

### 5. Measurements (test macro-F1 ×100, seed 42, RTX 4060; O = our pipeline, P = HetSheaf's pipeline, final-epoch weights unless marked)
| | Band (pub. ± 2 std) | O best-val | O | P | O graph + P split | P graph + O split | P graph without `ToUndirected` + O split | H2S-NN full / h₀-only (D11, s42) |
|---|---|---|---|---|---|---|---|---|
| ACM | [91.63, 94.91] | 89.04 | 88.47 | 91.20 | 89.77 | 90.61 | 89.71 | 84.01 / 83.77 |
| DBLP | [93.84, 94.76] | 92.08 | 91.96 | 91.93 | — | — | — | 82.18 / 76.98 |
| IMDB | [59.25, 60.09] | 50.74 | 51.03 | 55.30 | 52.81 | 53.96 | 51.89 | not run |

**Diagnosis, one component at a time:**
- **Metric:** ruled out. sklearn and torchmetrics agree to < 1e-6.
- **Split:** worth +1.3 (ACM) and +1.8 (IMDB). HetSheaf's `SubsetTrainSplit(val_ratio=0.1)` trains on 817 ACM nodes instead of 726. Their paper says "canonical HGB 24/6/70" (App. C.3.1), but their code gives 27/3/70. That is a HetSheaf paper–code discrepancy (VERIFIED-SOURCE vs VERIFIED-CODE `hetsheaf/datasets/hgb.py:55`).
- **Graph:** worth +2.1 (ACM) and +2.9 (IMDB), all of it from `ToUndirected`. HGB already stores both directions, so the transform gives every direction a second relation type: 14 relations instead of 8 on ACM, with twice the edges. Removing self-loops or duplicates changed nothing (90.56 / 90.61 vs 90.61).
- **Features:** not involved (embedding mode).

**Noise caveat:** HetSheaf's graph without `ToUndirected` is our graph minus 16 self-loops, yet it scored 89.71 against 88.47. Single-seed noise or ordering noise is therefore about 1 point, so the individual swap effects (1.3–2.9) are only 1–3× noise.

### 6. Self-review checklist
- Split ids compared with an independent parser (HGB's loader, not PyG): PASS.
- Labels compared on every labelled node, IMDB multi-hot exactly: PASS.
- Tolerance fixed before any run: PASS (`g0_preregistration.md`). The diagnostic plan was added after the gate runs, before the diagnostics, and is marked as such.
- Test labels never used for selection: PASS. Selection uses validation loss or validation macro-F1 only; test is logged per epoch for reporting.
- Arm O exercises our loader, split, labels and metric code: PASS. It does **not** exercise our trainer (`src/trainers/trainer.py`) or the feature path: NOT CHECKED.
- Seed noise for this configuration: NOT CHECKED (needs approved multi-seed runs). One near-duplicate pair suggests about 1 point.
- Independent sub-agent review: not done for G0; planned for G4.

### 7. Assumptions imposed for theory and reality checks
No THEORY-CONSTRAINT was touched.
- **Reality check (protocol):** published numbers on HGB are not protocol-equivalent. HetSheaf's R-GCN uses a different validation share from HGB's and relation-duplicated inputs, and is selected by a 150-run sweep. A "within 2 std of published" gate tests the configuration as much as the pipeline. This supports spec §6 rule "baselines all rerun under this protocol; no quoted numbers".

### 8. Open items
- **Feature path unverified.** `h₀`-only is 15 points below the feature-free R-GCN on DBLP (76.98 vs 92.08). Our features are raw: HetSheaf row-normalises, and HGB's recommended `feats_type` for ACM is 2 (as transcribed in `src/baselines/bundles.py:44`; HGB source not opened). A features-using reference on our pipeline (e.g. HetSheaf's GCN, ACM 92.25) would test this. It is not run because the card allows one baseline.
- **Our trainer is unverified as a component.** R-GCN was trained by the gate script's loop, not by `trainer.py`.
- **Number reproduction not achieved on any pipeline.** It needs either HetSheaf's selected R-GCN configuration (UNVERIFIED whether it can be obtained) or HGB's own R-GCN/SimpleHGN with their commands (needs an HGB clone and working DGL).
- **D11 screen incomplete.** 19 of 45 runs are done; the driver is not running and was not restarted. Stray empty `C:\c\…` folder (from D11) is still present.

### 9. Decisions needed from the author
1. Should G0 count as passed on the data leg (splits, labels, edges, metric verified exactly) with the reproduction leg recorded as "not achievable without the sweep winner"? If not, which route: (a) ask HetSheaf's authors for the selected configurations, (b) set up HGB's R-GCN/SimpleHGN, or (c) approve 5 seeds of arms O and P to measure noise?
2. Should a feature-path check (one features-using baseline on our pipeline, plus `h₀`-only with row-normalised features) run before G1? R-GCN without features outscores every D11 variant on the same pipeline, so the deficit may be partly in how H2S-NN uses features and the graph rather than in the sheaf design.
3. Which split should the protocol cite when comparing to HetSheaf's paper numbers (C7): HGB's 24/6/70, which we use, or HetSheaf's code split of 27/3/70?

**Author decisions (2026-11-08):**
1. G0 is passed on the data leg. Reproduction is recorded as not achievable, and the internal reference is R-GCN on our pipeline.
2. The feature-path check runs as G0b (below).
3. The protocol cites HGB's 24/6/70 split. HetSheaf is rerun as baseline C7; its published numbers appear only in a labelled note.

## G0b — Gap localization ladder (2026-11-08)

### 1. Summary
- 34 runs, all first-attempt successes: L0 has 5 seeds and L1–L4 have 3 seeds each, on ACM and DBLP. The reading rules were fixed in `D12_evidence/g0b_preregistration.md` before any code change or run, and were evaluated by script (`g0b_summary.md`).
- **(i) does not fire.** Row-normalised features change `h₀`-only by −0.2 on both datasets, so feature preprocessing is not the defect. **(ii) does not fire.** The learned sheaf does not hurt: L4 is 0.6 (ACM) and 1.3 (DBLP) above L3, within noise.
- **(iii) fires on both datasets.** Sheaf-free H2S-NN is **5.3 (ACM) and 9.8 (DBLP) points below** R-GCN, with train macro-F1 about 100 (ACM) and 97.5 (DBLP). It is therefore not underfitting. ACM is labelled overfitting by the rule. DBLP is unlabelled (train 97.5 is under the 98 cut), although its validation score is 8.3 below L0's.
- **(iv) fires on DBLP only.** GCN scores 80.2 on DBLP, below its band edge of 87.3. On ACM it scores 92.2, inside its band.
- **Localization:** the gap sits in the propagation and training path shared by every H2S-NN variant, not in the sheaf and not in feature scaling. On identical data and raw features, GCN reaches 92.2 on ACM, while sheaf-free H2S-NN reaches 84.5, only 0.8 above its own `h₀`-only head.

### 2. Files changed
- `src/configs/config.py`: `ablation.identity_maps` (needs `no_path_cells`; rejected with `h0_only_readout` or `pair_attention`) and `data.feature_normalization` (`none` | `row`, node classification only). Both defaults are off. Both are rejected by `reject_never_tuned_settings`: `identity_maps` in a base config, and both in search spaces.
- `src/models/map_operators.py`: `IdentityMapOperator`. `src/models/h2snn_model.py`: uses it when the switch is on; `node_to_edge` is still built (same state dict) but not called. `src/data/dataset.py`: applies PyG `NormalizeFeatures` per node type when `row`.
- `tests/test_gap_ladder.py` (new, 10 tests) and `tests/test_v2_model.py` (the every-ablation test now gives `identity_maps` its required companion).
- `D12_evidence/g0_rgcn.py`: eval-mode train F1, a GCN option and a file prefix. Its defaults still reproduce G0: L0 seed 42 = 89.04 / 92.08, unchanged.
- `D12_evidence/g0b_*`: driver, logs, per-run JSON and summary.

### 3. Deviations
- **Train F1.** `trainer.py` logs training-mode train F1 (dropout on, before the step). The ladder instead reloads each run's `best.pt` and evaluates in eval mode. The reloaded validation and test F1 equal the logged ones exactly (difference 0, all 24 runs).
- **Suite.** The full suite was run once (294 tests: 24 pre-existing `test_baselines` failures plus 1 new failure). The new failure was the every-ablation test setting `identity_maps` alone, which is invalid by design; it was fixed by passing the required companion switch. Only the affected tests were then rerun (14 OK), not the whole suite.
- **Stray folder.** Deleting `C:\c` was refused by the harness (protected system path). It contains only empty folders and needs a manual `Remove-Item C:\c -Recurse`.

### 4–5. Verification and measurements
- **Hardware:** RTX 4060.
- **Selection:** test macro-F1 at the best validation macro-F1 epoch, as mean ± sample std. Train F1 is eval mode at that epoch.

| Rung | ACM test | ACM val / train | DBLP test | DBLP val / train |
|---|---|---|---|---|
| L0 R-GCN, feature-free (5 seeds) | **89.72 ± 0.52** | 90.2 / 99.8 | **91.56 ± 1.37** | 91.4 / 100 |
| L1 GCN (HetSheaf code), raw features | 92.23 ± 0.16 (band 90.6–93.9) | 95.9 / 95.7 | 80.22 ± 0.18 (band 87.3–94.4) | 79.6 / 93.8 |
| L2 `h₀`-only, raw | 83.63 ± 0.17 | 86.8 / 100 | 77.76 ± 0.86 | 78.5 / 94.3 |
| L2 `h₀`-only, row-normalised | 83.46 ± 1.24 | 88.3 / 98.5 | 77.52 ± 0.56 | 78.7 / 96.8 |
| L3 sheaf-free H2S-NN | 84.46 ± 0.54 | 87.6 / 99.95 | 81.73 ± 0.50 | 83.2 / 97.5 |
| L4 full D11 | 85.09 ± 1.67 | 88.9 / 100 | 83.05 ± 0.78 | 84.2 / 99.5 |

- **Reproducibility (VERIFIED-RUN):** L2-raw ACM seed 42 = 83.77, and L4 ACM = 85.09 ± 1.67, both identical to the D11 screen. `feature_normalization: row` gives unit row sums for every featured ACM type; the featureless type stays 0.

**Evidence behind (iii)** (from `metrics.jsonl` at the selected epoch):
- The selected epochs are early: L3 about 20 on ACM; L4 16–58. Train F1 is already about 100 there, so the `[x_K, h₀]` head fits the training nodes within roughly 20 epochs and validation peaks at that point.
- Propagation adds only 0.8 (ACM) and 4.0 (DBLP) over `h₀`-only.
- The learned step size is η = 0.35–0.46 (L3) with K = 4.
- In L4, the listen-gate mean falls from its 0.6 initialisation to 0.34–0.45: the full model learns to propagate *less*.

**Hypotheses** (UNVERIFIED; each can be tested by one switch on L3; not run because G0b stops here):
- **h1 (ego shortcut):** the `h₀` half of the readout memorises from raw, high-dimensional features before propagation becomes useful.
- **h2 (reach and strength):** four first-order steps with η ≈ 0.45 are a weak low-pass compared with GCN's two full aggregations. G2's filter bank targets this.
- **h3 (trainer settings):** our training setup differs from L0/L1's: weight decay 1e-4 vs 5e-4 (VERIFIED from the configs); scheduler horizon and dropout settings were not compared.

**DBLP (iv):** the feature-free R-GCN (91.6) beats both GCN (80.2) and `h₀`-only (77.8). DBLP's signal is mostly structural, and its author features are weak. HetSheaf's GCN number used constant-plus-row-normalised features and the 27/3/70 split. The cause of our GCN's DBLP deficit is UNVERIFIED: row normalisation did not help `h₀`-only, but it was not tried for GCN.

### 6. Self-review
- Pre-registration before code or runs: PASS. Rules applied by script: PASS. Determinism versus D11: PASS (identical values).
- Checkpoint reload consistency: PASS (difference 0).
- Identity maps equal the dense normalised graph Laplacian ⊗ I on 5 random multigraphs within 1e-12, with a control (learned maps differ by > 1e-3): PASS.
- State dict unchanged: PASS. Sheaf-free logits ignore the map module, with a control: PASS. Sheaf-free model still propagates (non-vacuous): PASS.
- With 3 seeds, differences under about 2 points (L4 vs L3, L2-row vs L2-raw) are not resolved. The rules use 2-point margins, so the four verdicts do not depend on them.

### 7. Assumptions imposed for theory and reality checks
No THEORY-CONSTRAINT was added or touched. `identity_maps` forbids path cells only because transports were not defined for it in this unit (scope, not theory). Reality check: on identical data, the strongest simple models (R-GCN without features, GCN with features) beat every H2S-NN variant. Both use full neighbourhood aggregation with no ego shortcut in the head.

### 8. Open items
h1–h3 untested; the cause of GCN's DBLP deficit; IMDB not in the ladder (as carded); the stray `C:\c` folder.

### 9. Decisions needed from the author
1. **Rule (iii) points at propagation or training, not at the sheaf.** G2 (filter bank) addresses h2 only. Should I run a short, pre-registered check of h1–h3 on L3 before G1? For example: readout `x_K` only; K = 8; HetSheaf-like regularisation; 3 seeds each, ACM and DBLP, about 1 hour. Or start G1/G2 as planned?
2. **Rule (i) did not fire,** so `feature_normalization` stays `none` for every model. Should the C7 baselines nonetheless get their original preprocessing (HetSheaf: constant column plus row normalisation)? Rule (iv) found the baseline feature path suspect on DBLP.
