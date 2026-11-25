# Handoff to the model session: H2S-NN gap findings and proposed changes (2026-11-25)

From: the analysis session (author's request, 2026-11-25: "send the nonlinearity-between-diffusion-steps question to the model session as well").
Evidence: `docs/H2S-NN_gap_analysis_report.md` (§1–§12). Section numbers below refer to it.

Nothing here has been applied.
- Every change below, if adopted after these test results, is a **deviation**. The author decides it with the model session, and the paper reports it.
- Diagnostics ran on the lab from checkout `0b9873d` (job 18974; control run reproduces the original exactly). Outputs: lab `h2snn-analysis/gapdiag_2026-11-25/`.

## 1. What the evidence says (short)

- **HGB cells (full batch) overfit.** On ACM the one-layer encoder memorises the 726 training labels by about epoch 10. The selected checkpoint sits at τ ≈ 1, where the diffusion has barely trained (§1.3, §6.4, §7).
- **Encoder dropout is the main lever on ACM** (D1, §10). Test macro-F1 by `encoder_dropout`:

| Encoder dropout | Test |
|---|---|
| 0.448 (selected) | 89.07 |
| 0.7 | 89.82 |
| 0.85 | 90.65 |
| 0.9 | 92.04 |
| **0.95** | **93.16 ± 0.40** |

  - Validation rises with it, the best epoch moves to 50–67 and τ grows to 4.7.
  - 0.95 lies outside the shared search range (upper edge 0.9).
- **Diffusion helps, the learned maps do not on ACM** (D3):

| Variant | ACM test |
|---|---|
| `h0_only` | 86.01 (−3.1) |
| `S2` (identity fact maps) | 89.54 |
| Direction-blind maps | 90.02 |
| `endpoints` | 89.49 |

  Identity, direction-blind and endpoint maps are all within noise of the full model.
- **τ_init alone does not help** (D2: 89.5 / 90.1). The solve is linear in h0 and smooths whatever h0 holds, so the fix is h0 quality, not anchor strength (§11).
- **Width:**
  - IMDB d = 128 gives test 65.60 vs 63.59. Validation does not show it, so this is weaker evidence.
  - ACM d = 128 shows no gain (D4).
- **H2GB cells (minibatch) underfit.** Train loss ≈ val loss. Runs stop on patience while validation is still improving (PDNS, H-Pokec, oag-*), or plateau (mag-year, ogbn-mag) (§8.2).
- **H2GB protocol differs from the published-config baselines** (§8.2):

| | H2S-NN | Published-config baselines |
|---|---|---|
| Early stopping | Patience 40 epochs = 10–20 evaluations | None (`patience=None`) |
| lr schedule | Constant | H2GB scheduler |
| Loss on IEEE-CIS-G | Plain cross-entropy | Weighted cross-entropy |

  The tuned baselines on the plain protocol score like H2S-NN on IEEE-CIS-G (PSHGCN 0.570, ChebNetII 0.538).
- **Solver:**
  - T_max = 100 is never reached; at most 72 iterations (§9).
  - The τ > 50 stop ended 9 H2GB trials, mostly in epochs 0–11, with `scalar_lr_scale` = 10 and many minibatch steps per epoch.
  - The minibatch NC trainer logs neither τ nor PCG statistics.
- **ACM LP variance** is one stalled seed: path-weight range collapsed, early stop at epoch 77. The other 4 seeds give 0.957 ± 0.001 (§2).

## 2. Proposed changes, ranked

| # | Change | Kind | Where | Evidence | Note |
|---|---|---|---|---|---|
| M1 | **Input-feature dropout** before the type encoders, tuned over the baselines' shared `INPUT_DROP` range (0–0.9) | Code + search space | `src/models/node_encoder.py`, `scripts/run_h2snn_wandb_tuning.py` | D1: ACM needs the self-feature path broken. Baselines tune it (PSHGCN 0.85 on ACM) | Strongest candidate; brings H2S-NN in line with the baselines' search space |
| M2 | Widen `encoder_dropout` beyond 0.9 (e.g. to 0.95 or 0.99) | Search space | Shared `src/baselines/search_grids.DROPOUT` | D1: 0.95 > 0.9 > lower | **Fairness:** the range is shared by every method. Either widen it for all methods, or state in the paper that only H2S-NN's was widened |
| M3 | Search-space edges: d up to 128 (IMDB); `wisp_walks_per_anchor` above 64 (IMDB); lr above 1e-2 (DBLP/IMDB NC); lr below 1e-4 and d = 8 (IMDB M–A) | Search space | Tuning script | §6.1–6.3, §7, D4 | Re-sweep of the affected cells |
| M4 | LP head dropout | Code | `src/models/link_readout.py` | IMDB M–A is limited by overfitting (§7) | Only where LP overfits |
| M5 | Minibatch training protocol: count patience in evaluations, or select on val loss (or no early stopping, as the published configs); an lr schedule; weighted CE on IEEE-CIS-G (the same class weights as the baselines) | Code + protocol | `src/trainers/minibatch_trainer.py` | §8.2 | The fairness argument goes the other way here: H2S-NN currently gets *less* than the published-config baselines |
| M6 | Log τ and PCG statistics in the minibatch trainer | Logging only | `minibatch_trainer.py` | §9 | Needed to diagnose H2GB; no change to results |
| M7 | `scalar_lr_scale` in minibatch mode: 1, or scaled per optimizer step | Code / fixed value | Optimizer groups | §9 (τ > 50 stops) | Recovers lost trials on ogbn-mag and oag-* |
| M8 | Checkpoint selection on validation loss or smoothed validation | Protocol (all methods) | Trainers | §1.2–1.3 (ACM validation is noise-dominated) | Optional; it would have to apply to every method |
| M9 | ~~`feature_normalization: row`~~ | Config (exists) | — | D6: 88.90 vs 89.07, no gain (§4) | **Dropped** |

**Not recommended:**
- more classifier dropout (already tuned over 0–0.9 and not at the edge);
- dropout in the map MLPs (maps are not the lever on ACM);
- a nonlinear transport map (it breaks SO(d), the gauge argument and Thm 1, and D3 shows maps are not the lever) (§12).

## 3. Design question for the model session: nonlinearity between diffusion steps

**Today:** h0 = TypeEncoder(x) (one layer) → **one** linear anchored solve (PᵀP + τL̂)Y* = Pᵀh0 → MLP head on x*.
- The maps are nonlinear functions of h0, but the propagation is a single linear filter of h0. There is no nonlinearity between diffusion steps.

**Why consider it:**
- The H2GB NC cells underfit (train loss ≈ val loss, plateaus on mag-year and ogbn-mag), and the oag-* and ogbn-mag cells have many classes.
- A single linear filter of a one-layer encoding may lack capacity there. H2G-former, the strongest baseline on H2GB, is a multi-layer nonlinear model.
- On ACM (overfit) more depth is unlikely to help; the evidence there points to regularisation.

**Options to weigh** (no evidence yet for any):
- (a) **K stacked solves with a nonlinearity:** h¹ = σ(W·x*⁽⁰⁾), then a second anchored solve on h¹, and so on.
  - Each solve keeps its SO(d) maps and theory.
  - Cost is about K× solves and backward solves.
  - Questions: maps recomputed per layer or shared, and τ per layer.
- (b) **A nonlinear block on x* before the head.** Cheap, but nearly what the MLP head already does, so it probably adds little.
- (c) **A deeper encoder before the solve.** It increases the capacity of h0, but on HGB it would worsen the memorisation found in D1, so it would need input dropout (M1) with it.

**Suggested first test, if the session wants one:** option (a) with K = 2 on one underfit H2GB cell (PDNS or mag-year) and on ACM (to check there is no harm), at the selected configurations, 3 seeds.
- This is a model change, so the spec, a card and a deviation note come first.

## 4. D6 (feature normalisation) result

Lab job 19001: 3 ACM runs with `data.feature_normalization: row` at the selected configuration (seeds 0–2).

| Variant | Val | Test | Best epoch | τ at the best epoch |
|---|---|---|---|---|
| Selected (original seeds 0–2) | 94.26 ± 1.22 | 89.07 ± 0.69 | 8, 8, 12 | ≈ 1.0 |
| Row normalisation | 93.12 ± 0.86 | 88.90 ± 0.45 | 16, 16, 17 | 1.14 |

**Row normalisation gives no gain.** The encoder still fits early (best epoch about 16, τ ≈ 1). Feature scale is not the cause. M9 is dropped.

## 5. Implications for the paper (for the author)

- **Maps on ACM.** The learned SO(d) maps show no measurable gain over identity maps on ACM (D3). Any claim that the maps matter needs evidence from datasets where they do, for example the heterophilous H2GB cells once those runs are comparable.
- **ACM LP.** The ACM LP std (0.049) comes from one stalled seed. Report it as pre-registered, with a note.
