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

## 6. Round 2 (author's request, 2026-11-25): which hyperparameter or which part of the model?

Source: the 100 tuning trials per NC cell (`tuning_trials_run0.json`) and their `metrics.jsonl`, plus the code under `src/models/` and `src/baselines/search_grids.py`.
"Per SD" means the change in macro-F1 points for a one-standard-deviation change of that hyperparameter. The estimates come from a joint linear regression over the 100 trials; the inputs are log lr, d, encoder dropout, classifier dropout, log weight decay, l_max and log₂ walks.

### 6.1 Learning rate: not too high. Lowering the floor to 1e-6 goes the wrong way

| Cell | lr 1e-4–3e-4 | 3e-4–1e-3 | 1e-3–3e-3 | 3e-3–1e-2 | Effect on test, per SD |
|---|---|---|---|---|---|
| ACM (test) | 88.42 (n = 6, best epoch 83) | 89.05 (n = 11) | **89.66** (n = 37) | 89.47 (n = 46, best epoch 10) | +0.18 |
| DBLP (test) | 85.63 (n = 2, best epoch 263) | — | 91.32 (n = 8) | **92.97** (n = 90) | +0.66 |
| IMDB (test) | 40.95 (n = 2) | 53.24 (n = 3) | 59.36 (n = 4) | **62.97** (n = 91) | +3.22 |

- **Lower lr is worse on every dataset.** It trains slowly (best epoch 83–263) and scores lower. Trials below 1e-4 would sit even lower, so widening the range down to 1e-6 would only waste trials.
- **On DBLP and IMDB the upper edge (1e-2) is binding**: the top-10 trials have lr 0.006–0.0099.
- **On ACM, lr hardly matters** for test (+0.18 per SD). The fast fit on ACM (§1.3) is therefore not caused by a too-large lr.

### 6.2 Hidden dimension d: limiting on IMDB only

| Cell | Effect of d on test, per SD | By d |
|---|---|---|
| ACM | +0.05 | d = 16: 88.05, d = 32: 89.48, d = 64: 89.47 (saturated) |
| DBLP | +0.02 | No effect |
| IMDB | **+1.11** | d = 64 in all of the top 10 (upper edge) |

In H2S-NN, d is both the stalk dimension and the entire node-representation width: the classifier reads x* ∈ ℝ^d. The baselines' hidden width goes up to 256 (`search_grids.HIDDEN`), while H2S-NN stops at 64. On IMDB, d = 128 is worth testing. On ACM, width is not the bottleneck.

### 6.3 Dropout: the strongest test predictor on ACM, under-explored by the search and capped by the range

| ACM encoder dropout | Trials | Val | Test |
|---|---|---|---|
| 0.0–0.3 | 15 | 92.53 | 88.77 |
| 0.3–0.5 | 53 | 93.38 | 89.17 (the selected trial has 0.448) |
| 0.5–0.7 | 26 | 93.23 | 90.10 |
| 0.7–0.9 | **6** | 94.12 | **90.48** |

- **On ACM, encoder dropout has the largest effect on test** (+0.59 per SD, the biggest coefficient). Yet the search sampled 0.7–0.9 only 6 times, because the noisy validation (§1.2) did not reward it.
- **On DBLP and IMDB the upper edge (0.9) is binding**: 85 and 73 of 100 trials sit in 0.7–0.9, and its effect on test is +0.96 and +0.44 per SD.
- **What H2S-NN lacks compared with the baselines** (`src/models/node_encoder.py`):
  - `encoder_dropout` acts after `Linear(d_t → d) + LayerNorm + GELU`, on the d-dimensional h0, not on the raw features.
  - The baselines also tune *input-feature dropout* (`search_grids.INPUT_DROP`, 0–0.9; PSHGCN publishes 0.85 for ACM).
  - ACM paper and author features are 1,902-dim bag-of-words. One `Linear(1902 → 32)` per type, trained on 726 labels, memorises within about 10 epochs (§1.3). Input dropout is the standard regulariser for this, and H2S-NN does not have it.
  - The featureless term type gets a free `nn.Embedding(1902, d)`, which adds 1,902 × d parameters trained only through the loss.
- **Weight decay** has no effect (−0.03 per SD on ACM test).
- **`l_max`** is negative on ACM (−0.18 test, −0.34 val per SD); l_max = 3 is best (§1.2).

### 6.4 Model component: the sheaf diffusion helps ACM, but the checkpoint is taken before it has trained

Across the 100 ACM trials, at the selected epoch (`metrics.jsonl`, train block):

| Correlation with | Test | Val |
|---|---|---|
| τ (diffusion time) | **+0.45** | −0.08 |
| Path log-weight range | +0.28 | −0.32 |
| Best epoch | −0.24 | **−0.50** |

- Trials whose diffusion grew (larger τ) generalise better. Validation does not see this, and it actually prefers *early* checkpoints (−0.50).
- So the selection systematically picks checkpoints where τ ≈ 1.04 (its initial value). On DBLP τ ≈ 6.1, and there validation and test agree on τ (+0.37 and +0.21).
- **The model:** h0 = TypeEncoder(x) → one anchored solve (PᵀP + τL̂)Y* = Pᵀh0 → MLP head on x*.
  - On ACM the linear encoder alone fits the training labels before τ and the path weights have moved.
  - The head therefore learns on an almost unsmoothed h0, which validation (181 nodes) rewards and test (2,118 nodes) does not.
- **Conclusion:** the problem on ACM is not a broken component. It is an **ordering problem**: the encoder memorises faster than the diffusion trains, and nothing in the search space slows the encoder down (no input dropout, encoder dropout capped at 0.9 and rarely sampled, τ_init fixed at 1.0).

**Confidence:** medium–high for the correlations. Causation needs the diagnostics in §6.5.

### 6.5 Diagnostic plan (lab, needs the author's OK)

Everything uses the selected ACM configuration, 3 seeds per point, and config overrides only (no code change). Results are read on validation first. Test is reported separately and labelled as diagnosis. ACM takes about 1.6 s per epoch and about 60 epochs per run, so about 2 minutes per run.

| # | Question | Runs | Change |
|---|---|---|---|
| D1 | Does more encoder dropout close the gap? | `encoder_dropout` ∈ {0.7, 0.85, 0.9, 0.95} × 3 = 12 | config |
| D2 | Does a larger initial τ help (does diffusion train in time)? | `tau_init` ∈ {3, 6} × 3 = 6 | config (fixed card value, not a search key) |
| D3 | Which component matters on ACM? | Arms `h0_only` (no diffusion), `S2` (identity fact maps), `ego_head` ([x*, h0]), `direction_blind_maps`, `endpoints` × 3 = 15 | config (existing ablation switches) |
| D4 | Does width help? | d = 128 on ACM and on IMDB × 3 = 6 | config |
| D5 | Is l_max = 3 better on validation with seeds averaged? | l_max = 3 × 3 = 3 | config |

Total: about 42 runs, about 1.5–2 GPU-hours on one lab GPU.

**Input-feature dropout** (the clearest gap compared with the baselines) needs a code change in `node_encoder.py`, so it is a proposal for the model session, not a diagnostic. If D1 helps strongly, that is the evidence for it.

Adopting any of these after seeing test results is a deviation. That decision belongs to the author and the model session.

## 7. Overfit or underfit? (author's question, 2026-11-25)

Source: final-seed `metrics.jsonl` for each cell, plus the IMDB M–A tuning trials grouped by d, decoder and lr.
Caveat: the training metrics are probably logged in train mode, with dropout on, so training scores may be slightly underestimated.

| Cell | Train at the best epoch | Val | Later in the run | Verdict |
|---|---|---|---|---|
| ACM NC | Macro-F1 0.92–0.95 (seed 2: 0.99); train loss 0.29–0.46 | 0.92–0.96 | Train 1.00 and train loss 0.001 by epoch ~20; val loss 0.23 (minimum at epoch 10–13) → 0.75–0.91 | **Overfits fast.** The good window lasts only about 3–5 epochs. The checkpoint is picked at its start, while the model is still underfit, and 10 epochs later it is overfit. |
| ACM LP, seeds 0/2/3/4 | AUROC 0.99 | 0.95 | Val loss minimum 0.41–0.43 at epoch 77–112, rising to 0.64–1.4, while val AUROC keeps rising slowly | Overfits in calibration only (over-confident scores); the ranking still improves. Not a limiting factor (0.957 is top). |
| ACM LP, seed 1 | AUROC 0.87 (0.94 at the last epoch) | 0.84, flat from epoch 30 to 76 | Val loss 0.54 → 0.87 | Stuck in a poor optimum: underfit compared with the other seeds, and overfitting inside that optimum (§2). |
| IMDB M–A LP | AUROC 0.81–0.85; train loss ≈ 0.5 | 0.68–0.74 | Flat | **Looks underfit, but is overfit-limited.** In the trials, larger or faster models reach train AUROC 0.95–0.97 while val *falls* (d = 64: 0.65–0.70; d = 16 with lr < 1e-3: 0.726). The search escapes overfitting by going to the smallest d (16) and the lowest lr (1e-4–2e-4, the lower edge). |
| IMDB NC | Macro-F1 0.67–0.83 | 0.66–0.68 | Train 0.92–0.94; val loss 0.49 → 0.88–1.10 | Overfits after about epoch 14. |
| DBLP NC (for comparison) | 0.99 | 0.94–0.97 | Val loss 0.17–0.22 → 0.26–0.31 | Mild overfit; the window is wide (best epoch 32–64). |

**Reading.**
- None of the weak cells is capacity-limited. Each is limited by generalisation: the model can fit the training labels and edges, and the extra fit does not transfer.
- The fixes therefore lie in regularisation and in when to stop or select, not in more capacity or longer training:
  - input-feature dropout or a higher dropout range;
  - a lower lr floor *for IMDB M–A only* (1e-5, since the lower edge 1e-4 binds there);
  - checkpoint selection on validation loss or smoothed validation;
  - IMDB M–A: also d = 8.
- The exception is IMDB NC, where d = 64 binds (§6.2): there, more width *and* more dropout could both help.
- H2GB cells: not checked (curves not read).

## 8. Every dataset: rank and training curves (author's request, 2026-11-25)

**Sources.**
- Lab: `<lab-out>`.
- TSUBAME (read only): every H2S-NN trial run directory in `<t4-out:0b9873d>/h2snn/*_tune_run0_*` (`metrics.jsonl`, `best_test_metrics.json`, `config.yaml`), and the baselines in `<t4-out:0b9873d>` and `<t4-out:2d822a4>`.

**H2GB caveat.** For H2GB, H2S-NN has 1–37 finished trials per cell, and the number shown is the test score of the *best-validation trial* (one seed). The baselines show the mean over 3–5 final seeds.

### 8.1 Rank among the numbers produced by our pipeline (test)

| Cell | H2S-NN | Best of ours | H2S-NN rank |
|---|---|---|---|
| ACM NC | 88.78 | HetSheaf 91.1 (2 seeds), H2G-former 90.60 | 4 / 4 |
| DBLP NC | 93.09 | H2G-former 93.33 | 2 / 4 |
| IMDB NC | 64.04 | H2G-former 64.61 | 2 / 7 |
| ACM P–P LP | 0.935 (0.957 without seed 1) | NSD 0.958 | 10 / 10 (2 / 10 without seed 1) |
| DBLP A–P LP | 0.868 (trial) | NSD 0.876 | 2 / 4 |
| IMDB M–A LP | 0.706 | HetSheaf 0.778 | 3 / 3 (lab baselines still running) |
| PDNS | 0.933 (9 trials) | H2G-former 0.961 | 3 / 6 |
| IEEE-CIS-G | 0.550 (34 trials) | R-GCN 0.612 | 8 / 10 |
| H-Pokec | 0.685 (5 trials) | R-GCN 0.789 | 3 / 4 |
| mag-year | 0.336 (24 trials) | H2G-former 0.487 | 2 / 2 |
| oag-cs | 0.032 (4 trials) | H2G-former 0.094 | 7 / 8 |
| oag-eng | 0.090 (7 trials) | H2G-former 0.221 | 4 / 6 |
| oag-chem | 0.072 (1 trial) | H2G-former 0.147 | 3 / 6 |
| ogbn-mag | 0.058 (1 trial) | H2G-former 0.381 | 2 / 2 |

H2S-NN is first in no cell. It is close to the top on DBLP NC, IMDB NC, DBLP A–P LP and ACM LP (without seed 1), and clearly behind on ACM NC, IMDB M–A and every H2GB NC cell.

**Against the best published numbers** (author, 2026-11-25):

| Dataset | Best published | H2S-NN | Gap | HetSheaf ours − published (pipeline offset) |
|---|---|---|---|---|
| ACM | H2SGNN 94.47 | 88.78 | −5.7 | −3.2 |
| DBLP | H2SGNN 95.19 | 93.09 | −2.1 | −2.1 |
| IMDB | SeHGNN 67.11 | 64.04 | −3.1 | −2.1 |

- Only the HetSheaf rerun measures the pipeline offset directly. For H2SGNN on ACM and DBLP, and for SeHGNN, we have no rerun on our pipeline.
- Our H2SGNN on IMDB gives 63.57, below H2S-NN; the H2SGNN paper reports no IMDB result.
- So on DBLP and IMDB the gap to the best published number is about the size of the measured offset. On ACM about 2.5–3.5 points remain after the offset.

### 8.2 Two different regimes in the curves

**(A) HGB, full batch (ACM, IMDB, Freebase NC, all lab LP): overfit (§7).**
- Freebase NC, the best trial:
  - train macro-F1 0.975 against val 0.30;
  - val loss lowest at epoch 7 (1.48), rising to 5.5 by the end;
  - all top-3 trials chose d = 16, the smallest value.

**(B) H2GB, minibatch: underfit, and stopped while still improving.**

| Cell | Train loss at the end | Val loss at the end | Curve at stop |
|---|---|---|---|
| PDNS | 0.153 | 0.154 | Val loss still at its minimum and val F1 still rising when patience stopped the run at epoch 156 |
| H-Pokec | 0.580 | 0.566 | Val F1 still rising (0.6885 at the last evaluation) when the run stopped at epoch 312; lr chosen 1e-4–3e-4 |
| oag-eng / oag-cs / oag-chem | 3.5 / 3.8 / 3.0 | 3.07 / 4.09 / 3.34 | Slow monotone decrease; best epoch close to the last evaluation |
| mag-year | 1.285 | 1.298 | Plateau from about epoch 48 (val F1 0.29–0.32) |
| ogbn-mag | 3.17 | 2.96 | Plateau from about epoch 24 |
| IEEE-CIS-G | 0.19 | 0.28 | Val macro-F1 0.48–0.52 almost the whole run, close to all-majority (0.4755; accuracy 0.905) |

Train loss and val loss are almost equal, so there is no generalisation gap. The model has not fitted the training set.

**Protocol differences found in the code** (`src/trainers/minibatch_trainer.py`, `src/baselines/h2gb_published.py`):

1. **Early stopping.**
   - H2S-NN stops after 40 epochs without a gain in validation-subset macro-F1. That is only 10–20 evaluations of a noisy metric (`bad_epochs += period`, period 2–4).
   - The published-config baselines run with `patience=None`, so the full H2GB epoch count. Their best epochs are 300–499.
2. **Schedule.** The baselines use H2GB's optimizer and scheduler (warm-up and schedule); H2S-NN uses constant lr (`scheduler: none`).
3. **Loss on IEEE-CIS-G.**
   - The published-config baselines train with `weighted_cross_entropy` (fixed class weights).
   - H2S-NN uses unweighted `cross_entropy` (`minibatch_trainer.py:134`).
   - The tuned baselines on the same plain protocol score like H2S-NN (PSHGCN 0.570, ChebNetII 0.538; H2SGNN and HINormer 0.475, which is all-majority).
   - So most of the IEEE-CIS-G gap is the loss and protocol, not the model.
4. **Capacity on large datasets.** The representation is d ≤ 64, with a single linear encoder. Featureless types are free embeddings trained only through minibatches (ogbn-mag authors, Freebase). The H2GB NC cells with many classes (oag-*, ogbn-mag) underfit the most.

### 8.3 What follows

| Regime | Problem | What to change (proposals; each is a deviation the author and the model session decide) |
|---|---|---|
| HGB, full batch | Overfit, plus noisy selection | Input-feature dropout or a higher dropout range; selection on val loss or smoothed val; τ_init; a lower lr floor for IMDB M–A only (§6, §7) |
| H2GB, minibatch | Underfit, plus stopping too early, plus loss and schedule different from the baselines | Patience in *evaluations*, or on val loss (or no early stopping, as for the published-config baselines); an lr schedule; weighted CE on IEEE-CIS-G (the same loss as the baselines); d up to 128–256 on oag-*, ogbn-mag and mag-year |
| Reference | The gap to published is measured only through HetSheaf | Rerun H2SGNN (ACM, DBLP) and SeHGNN (ACM, DBLP, IMDB) on our pipeline (lab, small datasets) to measure the real gap to the strongest methods |

**Diagnostics** (need the author's OK):
- Lab: D1–D5 from §6.5.
- TSUBAME: 1–2 H2S-NN runs on PDNS and on IEEE-CIS-G with the baselines' protocol (no early stopping, weighted CE on IEEE). These cost TSUBAME points.

## 9. Is the solver (T_max) a cause? (author's question, 2026-11-25)

**Author's decision (2026-11-25):** do not rerun the published methods (H2SGNN, SeHGNN). The "Reference" row of §8.3 is withdrawn.

**Code** (`src/models/anchored_solve.py`):
- PCG runs with `solver_max_iterations` (T_max) = 100 and tolerance 1e-3 for the first 10 epochs, then 1e-4.
- `solve_anchored(..., strict=True)` *raises* `PCGConvergenceError` when any scope reaches T_max without converging.
- So a run that finished converged on every step.

**Evidence that T_max is not hit:**

| Cells | Epochs checked | Largest forward iterations | Epochs at T_max | Not converged |
|---|---|---|---|---|
| Lab, all 5 cells (`metrics.jsonl` train block, 105 runs each) | ≈ 75,000 | 56 (ACM LP); ACM NC 20, DBLP NC 47, IMDB NC 29, IMDB M–A 30 | 0 | 0 |
| TSUBAME full-batch: DBLP A–P LP, Freebase NC, Freebase B–B LP | ≈ 19,600 | 72, 32, 31 | 0 | 0 |

- No `PCGConvergenceError` appears in any run log on the lab or on TSUBAME.
- **Gap: the minibatch NC trainer does not log τ or the PCG statistics** (its train block holds only `loss` and `steps`). For PDNS, IEEE-CIS-G, H-Pokec, mag-year, oag-* and ogbn-mag we know only that nothing raised.

**A related solver limit that does bite: the stop condition τ > 50** (`tau_stop`, spec v1.16 §6) ends the run, and the trial then has no result.

| Cell | Trials stopped | Epoch | lr | Batch |
|---|---|---|---|---|
| ogbn-mag | 2 of 5 | 8, 295 | 5.7e-4, 6.9e-4 | 32, 1024 |
| oag-cs | 2 of 8 | 1, 6 | 8.1e-3, 8.2e-3 | 32 |
| oag-chem | 1 of 3 | 9 | 9.4e-3 | 256 |
| oag-eng | 1 of 10 | 100 | 4.3e-3 | 32 |
| IEEE-CIS-G | 1 of 37 | 11 | 7.4e-3 | 32 |
| PDNS | 1 of 12 | 20 | 9.5e-3 | 256 |
| H-Pokec U–U LP | 1 of 8 | 0 | 8.8e-3 | full batch |
| MUTAG (lab) | 2 | 34, 68 | — | — |

- Every stopped trial has τ_init = 1 and `scalar_lr_scale` = 10, so θ_τ (log τ) learns at 10 × lr.
- With a minibatch of 32–256 targets there are many Adam steps per epoch. At 10 × lr ≈ 0.04–0.09, θ_τ can travel from 0 to log 50 ≈ 3.9 within one epoch. This explains the stops at epochs 0–11.
- **Effect on the results:**
  - (i) These trials are lost: 2 of 5 on ogbn-mag, 2 of 8 on oag-cs.
  - (ii) The search is steered away from high lr exactly on the cells that underfit (§8.2).
  - (iii) If τ wants to grow without bound, the diffusion is pushing towards heavy smoothing, which may itself limit fitting on these datasets. This is not shown, because τ is not logged in the completed minibatch runs.

**Verdict.**
- T_max = 100 is **not** a cause in any cell where it can be checked: at most 72 iterations, never at the cap.
- The τ dynamics in minibatch mode (scalar lr × 10, many steps per epoch, the τ > 50 stop) is a **plausible contributor** to the H2GB underfit and to the lost trials. Confidence: medium for the lost trials (observed), low for the underfit (not observable without logging).

**Proposals** (for the model session; changes to the code or to fixed values are deviations):
- Log τ and the PCG statistics in the minibatch trainer. This is logging only and does not change results.
- In minibatch mode, use `scalar_lr_scale` = 1, or set it per optimizer step rather than per epoch.
- *Diagnostic (TSUBAME, needs OK):* one PDNS run with τ logged, to see whether τ stays near 1 or climbs in a completed run.

## 10. Diagnostic results D1–D5 (lab job 18974, 2026-11-25; author OK)

**Setup.**
- Code: checkout `0b9873d`, run on an RTX 3090 (the GPU type of the original runs).
- Each variant uses the selected final configuration of the cell with **one field changed**, and runs seeds 0–2.
- Outputs are in lab `h2snn-analysis/gapdiag_2026-11-25/`, outside every result directory.
- **Control:** ACM seed 0 with the original settings reproduces the original final seed exactly (val 95.626, test 89.492, best epoch 8).
- Reference: the original final seeds 0–2.

| Cell | Variant | Val | Test | Best epoch | τ at the best epoch |
|---|---|---|---|---|---|
| ACM | Selected (enc. dropout 0.448), original seeds 0–2 | 94.26 ± 1.22 | 89.07 ± 0.69 | 8, 8, 12 | ≈ 1.0 |
| ACM | **D1** enc. dropout 0.7 | 92.89 ± 1.19 | 89.82 ± 0.62 | 11, 11, 9 | 1.26 |
| ACM | **D1** enc. dropout 0.85 | 93.81 ± 0.52 | 90.65 ± 1.21 | 11, 18, 67 | 2.23 |
| ACM | **D1** enc. dropout 0.9 | 94.24 ± 1.41 | 92.04 ± 0.39 | 44, 43, 28 | 3.14 |
| ACM | **D1** enc. dropout 0.95 | **94.97 ± 0.96** | **93.16 ± 0.40** | 51, 55, 67 | 4.70 |
| ACM | D2 τ_init 3 | 93.64 ± 1.36 | 89.54 ± 0.24 | 7, 8, 8 | 2.52 |
| ACM | D2 τ_init 6 | 93.28 ± 0.94 | 90.05 ± 0.77 | 7, 13, 8 | 4.71 |
| ACM | D3 `h0_only` (no diffusion) | 91.69 ± 0.84 | **86.01 ± 0.81** | 13, 10, 14 | — |
| ACM | D3 `S2` (identity fact maps) | 92.45 ± 0.26 | 89.54 ± 0.25 | 13, 15, 14 | 0.86 |
| ACM | D3 `ego_head` | 92.79 ± 1.90 | 88.77 ± 0.26 | 10, 9, 13 | 1.09 |
| ACM | D3 `direction_blind_maps` | 93.04 ± 0.90 | 90.02 ± 0.31 | 13, 9, 11 | 0.93 |
| ACM | D3 `endpoints` | 93.20 ± 0.55 | 89.49 ± 0.29 | 9, 10, 11 | 0.93 |
| ACM | D4 d = 128 | 93.33 ± 0.59 | 88.71 ± 0.73 | 7, 5, 14 | 0.90 |
| ACM | D5 l_max = 3 | 93.81 ± 0.95 | 89.60 ± 0.29 | 15, 8, 10 | 1.04 |
| IMDB | Selected (d = 64), original seeds 0–2 | 66.73 ± 0.84 | 63.59 ± 0.67 | 15, 18, 14 | — |
| IMDB | **D4** d = 128 | 66.05 ± 0.45 | **65.60 ± 0.46** | 30, 31, 10 | 1.14 |

**Findings.**

1. **Encoder dropout is the main cause of the ACM gap (confirmed).**
   - Test rises monotonically with encoder dropout: 89.07 → 89.82 → 90.65 → 92.04 → **93.16** (+4.1, about 6 seed std).
   - At 0.9–0.95, the best epoch moves from 8 to 28–67 and τ grows from ≈ 1 to 3–4.7. This is the mechanism of §6.4: once the encoder cannot memorise the 726 labels, the diffusion has time to train.
   - Validation moves the same way (94.97 at 0.95, the highest), so the effect is visible without test.
   - 93.16 is above every same-pipeline ACM baseline (HetSheaf 91.1, H2G-former 90.60), above published GCN/GAT/R-GCN/HAN, and below published H2SGNN (94.47), PSHGCN (94.35) and SeHGNN (94.05).
   - **The best value, 0.95, lies outside the shared search range (upper edge 0.9).**
2. **The diffusion helps; the learned maps do not (on ACM).**
   - `h0_only` loses 3.1 points (86.01 vs 89.07), so the sheaf diffusion contributes.
   - Identity fact maps (`S2`), direction-blind maps and `endpoints` are within noise of the full model (89.5–90.0). On ACM, the learned SO(d) maps add nothing measurable.
3. **τ_init alone does not help** (89.5 / 90.1). τ starts larger, but the encoder still memorises by epoch 7–13. Overfitting of the encoder is the cause, not the speed of τ.
4. **Width:**
   - ACM d = 128 shows no gain (88.71), as §6.2 predicted.
   - **IMDB d = 128: test +2.0** (65.60 vs 63.59), above H2G-former ours (64.61) and published HetSheaf (64.51), below SeHGNN (67.11).
   - On IMDB, validation does *not* show it (66.05 vs 66.73). Selection on validation would therefore not have picked d = 128; this result is weaker evidence than D1.
5. **l_max = 3:** +0.5, within noise (§1.2 overstated it).

**What this means for the protocol** (decision for the author and the model session):
- Adopting encoder dropout 0.95, or d = 128, after seeing test is a deviation.
- The clean route is a re-tune with a widened range. For example: dropout up to 0.95–0.99, plus input-feature dropout; d ∈ {16, …, 128}.
- To be fair, the same widening must apply to the baselines' shared ranges (`search_grids.DROPOUT` caps every method at 0.9), or the paper must state that only H2S-NN's range was widened.
- Cost on the lab: one 100-trial ACM sweep takes about 3–4 GPU-hours (≈ 2 min per trial); IMDB about 2× that.

## 11. Does the anchor matter? (author's question, 2026-11-25)

**What the anchor is.**
- The anchored solve is Y* = argmin ‖PY − h0‖² + τ YᵀL̂Y, that is (PᵀP + τL̂)Y* = Pᵀh0 (`src/models/anchored_solve.py`).
- The anchor term ‖PY − h0‖² keeps each node's state close to its own encoding h0, and τ sets how strongly the diffusion pulls against it:
  - τ → 0 gives Y* = h0 (`h0_only`);
  - τ → ∞ gives full smoothing, where the anchor no longer matters.
- `anchored` is the only propagation left (Chebyshev was removed), so there is no "un-anchored" arm. Anchor strength is controlled through τ alone.

| Setting | Effective anchor | ACM test | τ at the best epoch |
|---|---|---|---|
| `h0_only` (D3) | Infinite: no diffusion | 86.01 | — |
| Selected configuration | Strong (τ ≈ 1) | 89.07 | ≈ 1.0 |
| D2 τ_init 3 / 6 | Weaker from the start | 89.54 / 90.05 | 2.5 / 4.7 |
| D1 encoder dropout 0.95 | Weaker, learned | **93.16** | 4.7 |
| 100 ACM trials | — | Correlation(test, τ) = +0.45 | — |
| DBLP selected | — | 93.09 (validation and test consistent) | ≈ 6 |

**Reading.**
- **Too strong an anchor hurts:** removing the diffusion costs 3.1 points, and the checkpoint selected on ACM sits at τ ≈ 1.
- **A weaker anchor by itself does not fix it:**
  - D2 τ_init 6 reaches the same τ (4.7) as D1 encoder dropout 0.95, but scores 90.05 against 93.16.
  - The solve is linear in h0, so it smooths whatever h0 holds. When the encoder has memorised the 726 labels, h0 is overfit and smoothing it does not recover the test signal.
  - Encoder dropout makes h0 an unreliable anchor during training, which forces the model to use its neighbours. τ then grows *because of* this, not as the cause.
- **Too weak an anchor stops the run:** τ > 50 ended 9 H2GB trials (§9). Whether completed minibatch runs drift towards weak anchoring cannot be checked, because τ is not logged there.
- **Walk anchors (the other "anchor": WISP walks per anchor):**
  - This setting matters little on ACM (−0.09 per SD on test), but helps on DBLP (+0.61) and IMDB (+0.60).
  - On IMDB, 64 walks (the upper edge) appear among the top trials. Widen this range in any re-tune.

**Verdict.** The anchor matters through how it interacts with h0, not on its own. The fix lies in the quality of h0 (regularising the encoder), not in a stronger or weaker anchor.

**Optional check.** Run D1 at 0.95 with τ frozen at 1, 3 runs (about 6 min on the lab). This shows whether the gain needs τ to grow. It needs a small code switch (θ_τ trainable on or off), so it goes to the model session.
