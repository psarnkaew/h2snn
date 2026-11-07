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

**Author decision (2026-11-08):** proceed with G1–G3, then add G3b (functional check). The decisions above that were not addressed remain open.

## G1–G3b — Relational context, filter-bank encoder, cost gate, functional check (2026-11-08)

Pre-registrations: `D12_evidence/g1_g3_preregistration.md` and `g3b_preregistration.md`, both written before the code or runs they cover. Commits: G1 `cdd2fb8`, G2 `3760864`, G3 `e1e91ec`, G3b (this section).

### 1. Summary
- **G1:** `model.geometry_context` (default on). `c` is the per-(relation, role) mean of the neighbours' `h₀` over facts. `[h₀, c]` feeds the map MLPs, the gate heads and the path encoder. T-G7 passes exactly. With the switch off, the D11 golden outputs are reproduced bit for bit (13 cases, 844 comparisons).
- **G2:** `model.encoder` (default `filter_bank`). T-G1…T-G6 and T-G8 pass, and the mutation check catches 3/3.
- **G3 PASS:** the filter bank costs **+0.3 % time and +0.04 % peak memory** over `first_order` (gate: 25 % / 10 %).
- **G3b (a):** a2, a3 and a4 pass. a1 passes 5 of its 6 checks. **Check 2 fails on its pre-registered tolerance:** `filter_norm_sum` reached 1.0000017 > 1 + 1e-6. The cause is float32 rounding in the diagnostic: the same weights give 1 + 4.7e-8 in float64. The gain stayed ≤ 0.951 throughout.
- **G3b (b) PASS:** on the synthetic task the full model scores 0.992 on all 3 seeds; `h₀`-only scores 0.19–0.29 (chance is 0.25).

### 2. Files changed
- **G1:**
  - `src/models/relational_context.py` (new).
  - `src/models/sheaf_maps.py`: `input_dim`; an exact factored first layer (`gathered_linear`, `mlp_on_blocks`); shared-role symmetrisation passed as a callable.
  - `src/models/h2snn_model.py`: `geometry_inputs`, and geometry widths for the maps, gates, path and endpoint encoders and free maps.
  - `src/configs/config.py`: the switch and its tuning rejection.
  - `tests/test_filter_bank.py` (new).
  - D11 pins in `tests/test_cooperative.py` and `tests/test_v2_model.py`; `docs/heads_evidence/v2_golden.py` names `geometry_context: false` and `encoder: first_order`.
- **G2:**
  - `src/models/filter_bank.py` (new).
  - `h2snn_model.py`: `apply_operator`, `propagate`, `operator_diagnostics`; θ_η exists only under `first_order`.
  - `config.py`: the `encoder` switch. The rejection matches the exact key, so `encoder_dropout` stays tunable.
  - `trainer.py` and `link_prediction_trainer.py` log `operator_diagnostics`.
  - Tests: 9 D11-behaviour tests now name `encoder: first_order`; 2 task-head tests accept either encoder's scalar.
- **G3 / G3b:** evidence scripts only (`D12_evidence/g3_*`, `g3b_*`).

### 3. Deviations
- **Clenshaw evaluation.** The filter uses Clenshaw's recurrence: the same polynomial as the card's forward recurrence, but with K applications of A instead of K(K+1)/2. T-G6 checks it against the dense forward-recurrence reference.
- **Factored first layer (not in the card).** Concatenating `[h₀, c]` per incidence created tensors of width 2d(1+2R) per edge and per cell, which broke R7; the existing memory test T8 caught it. The first linear layer is now evaluated per node and then gathered. This is exact (≤ 1e-12, tested). The context-off path keeps D11's concatenation, so the goldens stay bit-identical.
- **Layer dropout** is applied once to the filter output (default 0).
- **`pair_attention` under `filter_bank`.** The filter is applied to A = I − P. At initialisation this equals D11's attention update with η = η₀. There is no gain guarantee for this arm, because P is not self-adjoint.
- **`endpoint_edges` under context.** The endpoints' `c` contains the intermediates' `h₀` as neighbourhood means. This information is not path-specific, but it reaches the arm. T11 is pinned to context off; see decision 2.
- **Lemma N premise (T-G5).** The lemma needs degree > 0, no self-loop fact, and no repeated node inside a cell.
  - ACM has 8 self-loop facts (`cite`), on 8 papers; DBLP and IMDB have none.
  - No evaluation-draw cell repeats a node (0 of about 370k).
- **a1 was executed twice.** The first execution crashed in my post-processing after training finished (a device mismatch in the script). `metrics.jsonl` holds both runs, which agree to about 1e-6. The numbers below are from the second run.

### 4–5. Verification and measurements
**Tests and suites**
- `tests/test_filter_bank.py`: 17 tests pass.
- T-G1: 60 fixtures; imaginary part ≤ 1e-9; spectrum within [−1e-9, 2·max g b + 1e-9].
- T-G2: 50 fixtures with Σ‖V‖ in [0.3, 5]; M-gain ≤ 1 + 1e-9. The control (no normalisation) exceeds 1.
- T-G3: exact zero blocks; the control at gates 0.5 gives a non-zero block (> 1e-8).
- T-G4: ≤ 1e-10 for η ∈ {0.25, 0.5, 0.9} and K ∈ {2, 5}.
- T-G5: more than 100 nodes checked, within 1e-12.
- T-G6: ≤ 1e-10.
- T-G7: exact.
- T-G8: non-zero gradients, including the map-MLP columns that read `c`.
- The gain diagnostic matches the exact M-gain within 1e-4.
- Mutations (`g2_mutation_output.txt`): without normalisation the gain is 2.91 (caught); the recurrence on A gives a T-G4 error of 1.53 (caught); swapped gates give a non-zero block u←j (caught).
- Full suite: G1 300 tests, G2 311 tests. After the fixes, only the 24 pre-existing `test_baselines` failures remain. The first runs failed golden-output tests: a `per_side_scale` gradient-order change, fixed by keeping one shared input tensor. They also failed one T6 control (pinned).

**G3 (ACM, RTX 4060; forward + backward; mean of 3 runs after a warm-up)**

| Arm | Time | Peak memory | Parameters |
|---|---|---|---|
| first_order | 1.471 s | 1611.8 MiB | 532,649 |
| filter_bank | 1.475 s | 1612.5 MiB | 537,768 |

The extra parameters are (K+1)·d² − θ_η = 5,119. For information (not gated), the context itself costs +39 % time and +5.4 % memory over geometry from `h₀` alone.

**G3b**
- **a1 (ACM, 5 epochs, defaults):**
  - Loss falls from 1.101 to 0.894. Filter gain is 0.936–0.950 and λ_max(A) is 0.51–0.72 at every epoch. `filter_norm_sum` reaches 1.0000017 (FAIL at 1e-6; see the summary).
  - Every W_k moved: relative change 0.066–1.17, with the last term W₄ the largest. The cap became active: Σ‖V‖ ended at 1.067.
  - γ moved from 1.000 to 1.0015. The gate heads changed by 0.84 and 0.61 (relative); mean g fell from 0.61 to 0.54.
  - ‖∂L/∂c‖ = 1.4e-3. Reloaded logits are equal on CPU (`torch.equal`); on GPU the maximum difference is 1.5e-7.
- **a2 (DBLP):** the model has path lengths 2 and 4; no errors.
- **a3 (cap active from the start):** gradients finite; gain 0.988 → 0.962. The first loss equals a1's exactly, because W = V/2 is the D11 initialisation.
- **a4:** all 8 comparison switches run 2 epochs without error, with finite losses.
- **(b) synthetic task (test accuracy, seeds 42 / 43 / 44):**
  - full: 0.992 / 0.992 / 0.992
  - `h₀`-only: 0.19 / 0.23 / 0.29
  - `first_order`: 1.00 / 0.99 / 0.90
  - `no_path_cells`: 1.00 / 0.99 / 0.25 (seed 44 never learned: best epoch 3, stopped at epoch 104)
  - Every test target lies on a PAPV cell.

### 6. Self-review
- PASS: every card test with its tolerance; mutation check 3/3; D11 bit-identity; both suites; G3.
- FAIL: a1 check 2 (tolerance, diagnosed above).
- Non-vacuity: every operator test has a control; T-G7 checks both directions.
- **The synthetic task does not test the path cells:** `no_path_cells` solves it on 2 of 3 seeds, because the label is 3 hops away and K = 4. It shows that information reaches a node through a typed path; it does not show that cells add information (C2).
- NOT CHECKED: link prediction and graph classification under the filter bank beyond one-step unit tests; IMDB smoke (not carded).

### 7. Assumptions imposed for theory and reality checks
- **THEORY-CONSTRAINT: the norm cap Σ‖W_k‖₂ ≤ 1 (Prop. 5).** It forbids any filter with gain above 1. Possible cost: the cap became active within the first epochs of a1 (Σ‖V‖ reached 1.067), so training pushes against it. Whether it costs accuracy is untested; C8 is the test.
- **THEORY-CONSTRAINT: the frequency response needs the real spectrum of the gated self-adjoint operator (Prop. 4).** The attention arm has none, and it is reported without a bound.
- Reality check: on ACM, the context accounts for most of the cost (+39 % time).

### 8. Open items
- The a1 check 2 tolerance.
- `no_path_cells` seed 44 failing on the synthetic task (not diagnosed; not gated).
- G0b's h1–h3 are still untested; the filter bank addresses h2 only.
- No real-data accuracy for the D12 model yet (by design).
- IMDB has a relation named `movie__>actorh__actor` (from HGB's `info.dat` via PyG): cosmetic, out of scope.
- The stray `C:\c` folder.

### 9. Decisions needed from the author
1. **a1 check 2.** Should I accept the float32 diagnosis, or record the tolerance as wrong and re-check with a float64 diagnostic? I propose 1e-5 for float32 diagnostics, the same as the gain criterion; I did not change it.
2. **C2 under context.** With `geometry_context` on, the `endpoint_edges` arm's maps see intermediates through the endpoints' neighbourhood means (not path-specific). Is that the intended C2 comparison, or should C2 run its two arms with context off?
3. **A known-answer task that needs the cells.** Should I build one where the label depends on which intermediate a specific instance passes through, with the same endpoint relations? That would test H/C2 mechanistically, which the current task cannot.
4. **Next step.** Choose between G4 (final report ≤ 2 pages) and the still-open G0b check of h1–h3.
