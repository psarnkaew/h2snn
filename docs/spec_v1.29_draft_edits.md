# Spec 1.29 — draft edits (AC-1 – AC-9), DRAFT for author approval

Written 2026-11-20 by the model session. Base: spec v1.28 (`design-docs` `7c099a8`; 1.28 did not touch §5 C2/C2b/C14/C15 or §6 "Ablation statistics", "Ablation table", "Timing and analysis-only settings"). **Source:** the author's item-by-item review of pre-registration v2 on 2026-11-20 (decisions D1–D8 in `docs/ablation_analysis_preregistration_v3.md`). **The method is unchanged, and so is `src/`:** the freeze holds; the changes reach code only in `scripts/` (card 9).

Where this draft and pre-registration v3 differ, v3 wins.

---

## AC-1 Header
- Replace `**Version:** 1.28 (2026-11-20).` with `**Version:** 1.29 (2026-11-20).`
- In the same paragraph, replace `each computes only what `docs/ablation_analysis_preregistration_draft.md` fixes` with `each computes only what `docs/ablation_analysis_preregistration_v3.md` fixes (1.29)`.
- Insert before `**Changes from 1.27 (to 1.28;`:
> **Changes from 1.28 (to 1.29; author review of the ablation pre-registration, 2026-11-20, draft `docs/spec_v1.29_draft_edits.md` AC-1–AC-9; the method and `src/` are unchanged):** (AC1) The ablation study and model analysis are exactly those of `docs/ablation_analysis_preregistration_v3.md` (replaces v2). (AC2) Ablation seeds 1–5; no p-values and no multiplicity adjustment; "`full` better on a dataset" = mean paired difference > 0 and `full` higher in at least 4 of 5 seeds. (AC3) Configuration (a) only (`full`'s tuned hyperparameters, one change per row); separate tuning of the arms and the capacity rule are dropped; parameter counts are reported. (AC4) The ablation table has 8 rows: `uniform` added, with claim C2-w. (AC5) C2's kill-rule consequence and C14's fallback rewritten; the reference to a "§7 option B" removed. (AC6) Schema read-out without a permutation null; 5 of 5 seeds for "favour". (AC7) Perturbation study: drop and gap defined with fixed rules; HetSheaf and CSNN not waited for. (AC8) Face-effect and two-regime analyses: fixed 4-of-5 rules, scale-free margin, node energy divided by ‖x*_v‖².

## AC-2 §5 C2 row
Replace the whole row with:
> | **C2 (thesis H), instance weights on identical WISP cells** (`endpoints`' `g` matched to `full`'s parameter count within 5 %; identical evaluation cells) | **P2 (core)** `full` > `endpoints`; `full` > `no_path_cells` (C2-cells); `full` > `uniform` (C2-w, 1.29). All configuration (a), seeds 1–5; rule of §6 "Ablation statistics" (1.29) | **Kill rule:** `full` not better than `endpoints` on ≥ 3 of the 4 ablation datasets (ACM, DBLP, IMDB, Freebase). `full` not better than `no_path_cells` on ≥ 3 of 4: warning signal (stop and discuss). C2-w: not better than `uniform` on ≥ 3 of 4 | Kill rule: no claim that the face weight depends on which node is in the middle of the face; the paper claims only what passed among C2-cells, C2-w, C14 and C15; stop and discuss before writing. C2-w: no claim that learned per-face weights help; read with S3 (§6 known-answer table) |

## AC-3 §5 C2b row
Replace the evidence and falsification cells with:
> | C2b Schema read-out is stable and meaningful | Mean pairwise Kendall τ-b of the rankings of `ρ_σ = I_σ / c_σ` over seeds 1–5 of `full` (10 pairs; length-2 sub-path schemas with mean coverage ≥ 1 % of the cell mass, at most the ten largest by coverage; fewer than 4 → n/a); no permutation null (1.29); "the learned face weights favour σ" only at 5 of 5 seeds (descriptive; S2′ not used) | Not stable (mean τ-b ≥ 0.5) on ≥ 3 of 4 ablation datasets | The read-out is reported without interpretation; no interpretability claim. (1.23: `ρ_σ`, not `I_σ`, because `I_σ` follows WISP's coverage and is stable even for constant `g`) |

## AC-4 §5 C14 and C15 rows
- **C14:** replace `Not better on ≥ 3 of 4 ablation datasets (decided on the separately tuned `instance_features`) | Cells act as instance encoders; the contribution is the geometry (Thm 1, 4) and the theory |` with `Not better on ≥ 3 of 4 ablation datasets (configuration (a), 1.29) | No claim that diffusing over faces beats using them as features of the head; the remaining contribution is what C2, C2-cells, C2-w and C15 support (1.29; theory is never the contribution) |`.
- **C15:** replace `scale `c_e` kept; separately tuned); 2×2 with `no_path_cells` and S0 reported; mechanism: N2, N2b (§6)` with `scale `c_e` kept; configuration (a), 1.29); 2×2 with `no_path_cells` and S0 reported; mechanism: two-regime analysis R1–R3 (pre-registration v3 §4.4)`. The falsification and consequence cells are unchanged.

## AC-5 §5 note under the table
Replace the italic paragraph `*(1.23; 1.27) The statistics of C2, C14 and C15 … do not apply to the claims.*` with:
> *(1.23; 1.29) C2, C2-w, C14 and C15 are decided by the rule of §6 "Ablation statistics": `full` is better than a row on a dataset when the mean paired difference over seeds 1–5 is > 0 and `full` is higher in at least 4 of the 5 seeds. No p-value is computed. Every row runs with `full`'s tuned configuration and one change; the parameter count of every row is reported. The main-table protocol (seeds 0–4, Welch) does not apply to the claims.*

## AC-6 §6 "Ablation statistics"
Replace the whole bullet with:
> - **Ablation statistics (1.23; 1.29, author 2026-11-20).** Datasets: HGB ACM, DBLP, IMDB, Freebase (two homophilic, two heterophilic; Freebase full batch if it fits at the largest configuration, else mini-batch). Seeds 1–5 (the tuning seed 0 excluded; `full`'s main-table runs of seeds 1–4 reused, seed 5 run); runs paired by seed (same WISP draws and batch order in every row). Configuration (a) only: every row runs with `full`'s tuned configuration plus its one change; no row is tuned on its own. Rule: `full` is better than a row on a dataset when the mean of the paired differences of test Macro-F1 is > 0 and `full` is strictly higher in at least 4 of the 5 seeds; no p-value and no multiplicity adjustment. Reported: mean ± sample std per row, the five paired differences, their mean, the 4-of-5 count, and the parameter count of every row (total and the removed or added part). Every ablation run keeps its best checkpoint and resolved configuration. LP and GC have no ablation (stated limitation). The ablation study and analyses are exactly those of `docs/ablation_analysis_preregistration_v3.md` (1.29), fixed before the runs: schema read-out, perturbation study, face effect F1–F3 with the rule-chosen case study, two-regime analysis R1–R3, complexity, timing and the cost-vs-accuracy figure.

## AC-7 §6 "Ablation table"
- Replace `- **Ablation table (1.27):** 7 rows on the four ablation datasets, seeds 1–10: `full`, `no_path_cells`, S2, S0, `endpoints`, `instance_features` and the **fixed-literature-schema row** (1.18; 1.27: configuration (a), seeds 1–10, reported, not a claim; previously a main-table row on seeds 0–4):` with `- **Ablation table (1.27; 1.29):** 8 rows on the four ablation datasets, configuration (a), seeds 1–5: `full`, `no_path_cells`, S2, S0, `uniform` (1.29; every face the same learned weight `exp(L·tanh(b/L))`, `g` unused), `endpoints`, `instance_features` and the **fixed-literature-schema row** (1.18; reported, not a claim; previously a main-table row on seeds 0–4):`.
- Replace `if its mean test Macro-F1 is higher than `full`'s on a dataset with Welch p < 0.05 (raw, 5 seeds; the main-table test), the paper says so` with `if it is better than `full` on a dataset by the rule of "Ablation statistics" (roles swapped; 1.29), the paper says so`.

## AC-8 §6 "v3-B arms"
Replace `` `uniform / schema / endpoints` (C2 ladder); `` with `` `uniform` (1.29: ablation row, C2-w), `schema` (C2 ladder, withdrawn) and `endpoints` (C2); ``.

## AC-9 §6 "Timing and analysis-only settings"
- Replace `each with its configuration from the original graph (or its published configuration, listed), seeds 1–5.` with `each with its configuration from the original graph (or its published configuration, listed), seeds 1–5; HetSheaf and CSNN are not waited for: if their configurations are not ready when the other perturbation runs are complete, their rows are left out and the omission is stated (1.29). The drop, gap and expectation rules are those of pre-registration v3 §4.2.`
- After `stated in the paper; never from the main or ablation runs.` insert ` Time per epoch only, at a fixed 8,192 training targets per epoch; one-off preprocessing time and total training time are not reported (1.29, author).`

---

**Not changed:** §4 (Prop. 12), C1, C5, C7 (1.28), the main-table protocol, the freeze. §7 `OPEN-5` keeps its historical reference to the 1.23 draft.

**Decisions log:** on approval, one row "Ablation pre-registration v3 and spec 1.29 approved" records D1–D8. It also notes that the row "Ablation plan v2 approved" (2026-11-19) still says 20/40/60/80 %, superseded by the row "Card 7 Phase A review; perturbation levels" (0.2–0.6); the log is append-only, so the old row is not edited.
