# H2S-NN gap analysis report (2026-11-25)

Analysis session, following `docs/analysis_brief_h2snn_gap_2026-11-25.md`.
This report is analysis only. Nothing in the spec, code, configs or result directories was changed, and no job was submitted.
One CPU-only graph-statistics script was run on the lab login node (about 20 s): label propagation over ACM meta-paths (§1.4).

**Paths.**
- `<lab-out>` = lab `h2snn-out/0b9873d0c19455c8df56fa152492188d8918b62d/experiments`.
- `<t4-out:SHA>` = TSUBAME `h2snn-out/<SHA>/experiments` (read through `ssh t4`).
- "Ours" = our pipeline. "Published" = `docs/published_number_register.md`.

## 0. Summary

| Cell | Is the gap real? | Main cause | Confidence |
|---|---|---|---|
| ACM NC | **Partly.** Against published numbers, 3–5 points; about 2 of those are a pipeline-wide offset that our own baseline reruns show too. Against same-pipeline baselines, **1.2–2.3 points**, which is real (≈ 2–3 seed std). | (a) Validation selection on ACM is close to noise: 181 validation nodes, and validation and test have Spearman 0.09 across 100 trials. (b) The model fits the 726 training nodes within about 10 epochs, so the checkpoint is picked in the fast transient phase, with diffusion time τ still at its initial value. (c) The search therefore locked onto l_max = 4; l_max = 3 is better on test. | Medium–high for (a)(b); medium for (c) |
| ACM P–P LP | **The variance is one seed.** Seeds 0, 2, 3 and 4 give 0.957 ± 0.001, at or above every baseline. Seed 1 gives 0.848. | Seed 1 is a training failure (a bad basin), not a split or test problem. The path-weight range collapses, validation AUROC plateaus at 0.84 and early stopping fires at epoch 77. The same failure appears in about 8% of nearby tuning trials. | High |
| DBLP NC | Tied with the same-pipeline baselines (H2G-former 93.33, HetSheaf 92.82). | — | — |
| IMDB NC | Tied at the top. Our H2G-former rerun gives **64.61 ± 0.56** (3 seeds), slightly above H2S-NN's 64.04 and not in the brief's table. | — | — |
| H2GB (early) | Early validation is weak on IEEE-CIS-G, mag-year, oag-cs and PDNS, and strong on DBLP A–P LP. | Not diagnosed: few trials, and the validation metric's comparability needs checking. | Low |

## 1. ACM NC

### 1.1 Numbers: validation and test, same pipeline

Final seeds, macro-F1 (val → test, drop):

| Model | Source | Seeds | Val | Test | Val−test drop | Best epoch |
|---|---|---|---|---|---|---|
| H2S-NN | `<lab-out>/h2snn/h2snn_node_classification_acm/final_seed*.json` | 5 | 93.62 | **88.78 ± 0.64** | **4.84** | 7–12 |
| H2G-former | `<t4-out:0b9873d>/baselines/h2gformer_node_classification_acm` | 3 | 94.03 | 90.60 ± 0.55 | 3.43 | 59–100 |
| CSNN | `<t4-out:e52b025>/baselines/csnn_node_classification_acm` | 3 | 92.72 | 89.96 ± 0.66 | 2.76 | 64–75 |
| HetSheaf | `<t4-out:e52b025>/baselines/hetsheaf_node_classification_acm` | 2 of 3 (seed 2 collapsed: best epoch 0, test 17.0) | 94.69 | 91.13 | 3.56 | 123–134 |

- **On validation, H2S-NN is level with the baselines.** Its best-trial validation (95.63) equals HetSheaf's (95.48) and is above H2G-former's (95.02) and CSNN's (94.36). The gap appears only on test.
- **Every model loses 3–5 points from validation to test on ACM**, so part of the drop belongs to the split or the selection, not to H2S-NN. H2S-NN loses 1.3–2 points more than the others.
- **The published numbers sit about 2–3 points above our pipeline for HetSheaf on all three HGB datasets:**

| Dataset | HetSheaf ours | HetSheaf published |
|---|---|---|
| ACM | 91.1 | 94.30 |
| DBLP | 92.82 | 94.93 |
| IMDB | 62.41 | 64.51 |

The 2–5 point gap to published ACM numbers (GCN 92.17, … HetSheaf 94.30) is therefore partly a protocol offset: the authors' own sweeps and tricks, and the HGB hyperparameters. It is not specific to H2S-NN.

**Confidence:** high for the numbers. Medium for "offset" as the explanation: we have no same-pipeline GCN/GAT/R-GCN/HAN on ACM NC to confirm it directly.

### 1.2 Validation cannot rank configurations on ACM

From `<lab-out>/h2snn/h2snn_node_classification_acm/tuning_trials_run0.json` (100 trials, 0 failed):

**Rank agreement between validation and test is near zero on ACM:**

| Cell | Spearman rank correlation (val, test) |
|---|---|
| ACM NC | 0.09 (Pearson 0.14) |
| DBLP NC | Pearson 0.85 |
| IMDB NC | Pearson 0.95 |
| ACM LP | Pearson 0.99 |

The ACM baselines show 0.99–1.00, but their trial ranges are wider. H2S-NN's ACM trials all lie between 89.2 and 95.6 on validation, which lowers its correlation.

**The validation set is coarse.** It has 181 nodes (726 / 181 / 2118 train/val/test, 24/6/70). One node is worth 0.55 points.
- The top 5 trials span 0.6 validation points, about one node.
- Their test scores range from 88.7 to 90.1.

**Over the 100 trials, test macro-F1 has median 89.4, 90th percentile 90.7 and maximum 91.4.** Even oracle selection on test would land below published GCN (92.17). On ACM, this search space, and not only the selection, is limited to about 89–91.

**The search concentrated on l_max = 4** (75 of 100 trials), because its mean validation is 0.66 points higher (93.74 vs 93.08), about 1.2 validation nodes. On test it is the other way round:

| `wisp_max_length` | Trials | Mean val | Mean test | Max test |
|---|---|---|---|---|
| 3 | 13 | 93.08 | **90.17** | **91.36** |
| 4 | 75 | 93.74 | 89.44 | 91.02 |
| 5 | 10 | 90.41 | 88.80 | 90.47 |

Four of the five highest test scores have l_max = 3. I use this only to say *where* the gap is. It must not be used to pick a configuration (see §4).

**Confidence:** high that ACM selection is noise-dominated. Medium that l_max = 3 is truly better (13 trials).

### 1.3 Training dynamics: the model fits fast and is selected in the transient phase

From the final-seed `metrics.jsonl` and `run.log` (seed 0 run: `…/final_seeds/h2snn_node_classification_acm_tune_final_seed0_final_seed0/`):

- **Training accuracy reaches 1.00 by epoch 22.** Validation loss falls to 0.23 at epoch 11, then climbs to 0.85–0.90 by epoch 40.
- **Best epochs are 7–12**, and early stopping fires at epochs 47–52, so the 500-epoch cap is never the limit.
- **Validation is unstable at the selected epoch.** It moves 2.2–2.9 points per epoch within ±5 epochs of the best epoch in 4 of 5 seeds; DBLP moves 0.2–0.9.
  - Seed 0, epochs 7 → 8 → 9: validation 93.96 → 95.63 → 92.76, while test goes 88.58 → 89.49 → 89.83.
  - The best epoch is a validation spike: a *winner's curse* across epochs and trials. That explains why validation is high, not why test is low. Test is smooth and peaks at about 89.9 around epoch 12.
- **At the selected epoch the diffusion has barely trained.** τ (learned diffusion time, initialised at 1.0) is 0.89–1.05 on ACM against 6.2–6.9 on DBLP. Training macro-F1 at selection is 0.92–0.99, and in 4 seeds it is below validation.
  - So the reported ACM number mostly reflects the first ~10 epochs of encoder + classifier, not a converged sheaf diffusion.
- **Strong regularisation is preferred but does not help test.**
  - The selected weight decay is 0.1; 3 of the top 10 trials use 0.1 or 0.8 (upper edge 0.8).
  - Classifier dropout in the top 10 is 0.67–0.87.
  - Mean test per weight-decay value stays within 88.4–89.7 with no trend.

**Confidence:** high for the observations. Medium for "fast memorisation of 726 nodes, selected in the transient phase, is H2S-NN's specific extra loss". The baselines peak at epochs 59–134 with smaller validation-to-test drops, but I have not read their per-epoch curves.

### 1.4 Ruled out or unlikely

- **The validation split is not structurally easier than test.** I ran label propagation from the training labels over each meta-path (CPU only, no model):

| Meta-path | Val macro-F1 | Test macro-F1 |
|---|---|---|
| P-A-P | 86.6 | 88.4 |
| P-P | 90.0 | 88.3 |
| P-S-P | 65.9 | 68.5 |
| P-T-P | 17.8 | 17.4 |

  - The class balance is the same across train, validation and test.
  - So the 3–5 point validation-to-test drop that every model shows is selection inflation, not a harder test set.
- **Not a solver problem.** PCG converges in every logged epoch (`pcg_converged` = 1, forward iterations 6–19).
- **Tuning health is fine.**
  - 0 failed trials in all five lab cells.
  - Best trials are at index 20 (ACM NC), 35, 73, 29 and 33 of 100, so the search was not still improving at the end.
- **Featureless "term" type (a candidate, not shown).**
  - ACM has 1,902 term nodes with no features, and papers have on average 84.6 term edges against 3.3 author edges.
  - The term meta-path carries no label signal (P-T-P F1 ≈ 17).
  - WISP chooses the action (relation, direction) uniformly, not the edge. So term hops are about 1/5 to 1/3 of first steps, not 95%.
  - A wasted walk budget is possible, but the learned path weights should down-weight these walks. To confirm it, read the per-signature weights (`src/analysis/wisp_readouts.py`) of the selected checkpoint.
  - Confidence: low.
- **Feature normalisation (open).** H2S-NN uses `feature_normalization: none`; HetSheaf's pipeline applies PyG `NormalizeFeatures` (row-normalised bag-of-words). This is not tested. It is cheap to check on validation (proposal P4).

## 2. ACM P–P LP: variance

Per seed (`<lab-out>/h2snn/h2snn_link_prediction_acm_pp/final_seed*.json`):

| Seed | Val AUROC | Test AUROC (random) | Best / stop epoch |
|---|---|---|---|
| 0 | 0.9536 | 0.9572 | 363 / 403 |
| **1** | **0.8425** | **0.8481** | **37 / 77** |
| 2 | 0.9524 | 0.9570 | 264 / 304 |
| 3 | 0.9501 | 0.9551 | 232 / 272 |
| 4 | 0.9531 | 0.9583 | 248 / 288 |

- **Without seed 1: 0.9569 ± 0.0013.** That beats CSNN 0.9534, HAN 0.9531 and H2SGNN 0.9435 (all 5 seeds, lab) and matches NSD 0.9576 (3 seeds, TSUBAME). With it: 0.9351 ± 0.0487.
- **Seed 1 is low on validation as well,** so this is a training failure, not a test-split effect.
- **The split is fixed across seeds** (`acm_split_seed` 42 in `src/data/link_prediction_dataset.py`), and every baseline's seed 1 is normal (CSNN 0.9569, R-GCN 0.9503, …). So the cause is in training (initialisation or walk draws), not in the data.
- **Signature (from `metrics.jsonl`):**
  - In the healthy seeds, the loss sits on a plateau at about 0.69–0.65 until epoch ~50, then drops. Meanwhile the path log-weight range (`path_log_w_range`) grows to 12–19, and validation AUROC climbs from 0.74 to 0.95.
  - Seed 1 leaves the plateau early (loss 0.67 → 0.57 between epochs 20 and 30), and its path-weight range **collapses from 6.5 to 3.9** and stays at 4–6.5.
  - Validation AUROC then sits at 0.83–0.84 from epoch 30 to 76 while validation loss rises. The run has taken a shortcut that does not use path selection.
  - Longer patience would very likely not have rescued it: validation AUROC was flat and validation loss was rising.
- **Recurrence.** Among the 49 tuning trials with the same decoder and width (distmult, d = 64), 4 ended below 0.90 validation AUROC (8%). One of them shows the same signature: validation 0.836, best epoch 61, path-weight range 7.9.
  - At an 8% stall rate, the chance that at least 1 of 5 seeds stalls is 1 − 0.92⁵ ≈ 34%. Seed 1 is therefore not an extraordinary event.
- **Related observation:** IMDB M–A LP seed 1 is also low (test 0.666 against 0.706–0.728), but its path-weight range is normal (12.9). That is a different cause and stays open until the IMDB M–A baselines finish.

**Confidence:** high that the variance is one stalled run. Medium for the mechanism: path-weight collapse is a correlate, not a proven cause.

## 3. DBLP, IMDB and H2GB

- **DBLP NC.**
  - Same-pipeline results: H2S-NN 93.09 ± 0.67 vs H2G-former 93.33 (3 seeds), HetSheaf 92.82, CSNN 91.18. That is a tie.
  - Validation and test agree (Pearson 0.85) and the best epochs are 32–64.
  - The search sits at range edges: in the top 10 trials, `lr` is 0.006–0.010 (edge 0.01) and `encoder_dropout` is 0.72–0.89 (edge 0.9).
- **IMDB NC.**
  - H2G-former, ours: 64.61 ± 0.56 (`<t4-out:0b9873d>/baselines/h2gformer_node_classification_imdb`, 3 seeds). PSHGCN and H2SGNN: 63.57.
  - "At the top" should read "tied at the top".
  - The search sits at range edges: `d` = 64 in all of the top 10 (upper edge), `lr` near 0.01, `encoder_dropout` 0.84–0.90.
- **What differs about the winning datasets.**
  - On DBLP and IMDB, validation tracks test, and the model trains for 15–64 epochs with τ moving (DBLP τ ≈ 6.5).
  - On ACM it does neither. ACM has the smallest training set relative to the signal: the P-A-P and P-P label propagation alone scores ≈ 88–90. In that regime a high-capacity model memorises within a few epochs.
  - I have not computed a heterophily measure. `src/analysis/heterophily.py` exists for that.
- **H2GB (TSUBAME `0b9873d`, H2S-NN trial run directories, best validation so far):**

| Cell | H2S-NN trials | H2S-NN best val | Baseline val (final-seed mean) |
|---|---|---|---|
| IEEE-CIS-G NC | 34 | 0.587 | GAT 0.656, H2G-former 0.653, R-GCN 0.650 |
| mag-year NC | 24 | 0.334 | H2G-former 0.487 |
| oag-cs NC | 4 | 0.036 | ChebNetII 0.074, PSHGCN 0.071 |
| PDNS NC | 9 | 0.935 | H2G-former 0.964 |
| DBLP A–P LP | 39 | **0.876** | NSD 0.865, HetSheaf 0.830 |

  - Read this with care. The published-config baselines may compute the validation metric differently, so confirm that metric and split are identical before using these numbers (for example, ogbn-mag H2S-NN 0.055 from 1 trial against H2G-former 0.417 looks like a metric mismatch, not a model gap).
  - Confidence: low. It is worth a dedicated look once more trials exist.

## 4. Possible changes, ranked

Every change to H2S-NN, its search space or its selection rule after seeing these test results is a **deviation**. The author decides it with the model session, and the paper reports it. Items marked *diagnostic* use validation only. All lab runs need the author's OK.

| # | Change | Kind | Cost | What it addresses or confirms |
|---|---|---|---|---|
| P1 | Report the ACM LP and ACM NC comparisons honestly as they are. In the table, separate "our pipeline" from "published", and state the pipeline offset (HetSheaf ours vs published: −2.1 to −3.2 on all three HGB datasets). For ACM LP, keep the pre-registered mean ± std and add a footnote on the stalled seed 1 (median, or mean without it). | Reporting | 0 | Most of the apparent ACM NC gap; the ACM LP std |
| P2 | *Diagnostic:* rerun the top 10 ACM NC trials by validation with 3 seeds each and rank them by mean validation. This tests whether a less noisy validation choice changes the configuration (for example to l_max = 3). | Diagnostic, lab | ~30 runs × ~1.5 min ≈ 1 GPU-hour | Confirms §1.2. If it picks a different configuration, the fix is a selection-rule deviation applied to every model |
| P3 | *Diagnostic:* for the selected ACM NC configuration, log per-epoch validation and test with checkpoint selection on validation loss, or on validation macro-F1 smoothed over 5 epochs. Compare with the current spike-picking rule. | Diagnostic, lab | 5 runs, < 15 min | Confirms §1.3 (winner's curse across epochs). Changing the rule is a protocol deviation for all models |
| P4 | *Diagnostic:* ACM NC with `feature_normalization` set to row-normalise, and/or with dropout on the raw input features, on the selected configuration plus a 20–30-trial mini-sweep. Read validation only. | Diagnostic, then deviation | ~1 GPU-hour | Fast memorisation of 726 nodes (§1.3); the gap to HetSheaf's preprocessing |
| P5 | ACM LP stall guard. Either (a) a restart rule: if validation AUROC is below threshold X at epoch E, restart with seed + 1000, applied to every model; or (b) k = 3 initialisations per seed, picking the best by validation. | Deviation (protocol) | +1 run per stalled seed (~400 epochs) | §2. Option (a) must be pre-declared and applied to all models, otherwise it is not fair |
| P6 | *Diagnostic:* read the WISP per-signature weights of the selected ACM NC and ACM LP (seed 1 vs seed 0) checkpoints (`src/analysis/wisp_readouts.py`). This is CPU only, on existing checkpoints. | Diagnostic, no training | Minutes, CPU | Tests the term-hub hypothesis (§1.4) and the LP path-weight collapse (§2) |
| P7 | Widen the search ranges that sit on an edge: DBLP/IMDB `lr` above 0.01, `encoder_dropout` above 0.9, IMDB `d` above 64, IMDB M–A `lr` below 1e-4 and `d` below 16. | Deviation (search space) | Full re-sweep per cell | Possible small gains on DBLP and IMDB; does not fix ACM |
| P8 | H2GB: check that validation metric and split are identical between H2S-NN trials and the published-config baselines, then repeat the table in §3 when each cell has ≥ 20 trials. | Analysis | Reading only | Whether the early weakness on IEEE-CIS-G and mag-year is real |

**Recommended order:** P1 (free), then P6 (free, CPU), then P2 + P3 together (about 1.5 GPU-hours on the lab, validation only). P4 and P5 only if the author and the model session accept a declared deviation.

## 5. Not done or open

- Per-epoch curves of the ACM baselines (on TSUBAME, in the trial directories) were not read; that would complete the comparison in §1.3.
- I did not verify that IMDB NC `test_acc` (0.316) is exact-match accuracy for every model.
- I did not compute a heterophily measure.
- IMDB M–A LP baselines are still running, so the IMDB M–A seed-1 drop (0.666) stays open.
