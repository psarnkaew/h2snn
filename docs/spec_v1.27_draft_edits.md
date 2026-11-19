# Spec 1.27 — draft edits (AB-1 – AB-9), APPROVED and APPLIED 2026-11-19

Written 2026-11-19 by the model session. Base: spec v1.26. **Source:** the author-approved ablation plan v2 (`docs/ablation_plan_v2_2026-11-19.html`; decisions log rows of 2026-11-19) and its pre-registration `docs/ablation_analysis_preregistration_v2.md` (draft, approved with this file). **The method is unchanged.** 1.27 changes the claims (§5), the ablation and analysis protocol (§6), and adds one corollary to §4.

---

## AB-1 Header
- Replace `**Version:** 1.26 (2026-11-19).` with `**Version:** 1.27 (2026-11-19).`
- Insert before `**Changes from 1.25 (to 1.26;`:
> **Changes from 1.26 (to 1.27; author decisions of 2026-11-19 "Ablation plan v2 approved" and "Card 7", draft `docs/spec_v1.27_draft_edits.md` AB-1–AB-9; the method is unchanged):** (AB1) The ablation study and model analysis are exactly those of `docs/ablation_analysis_preregistration_v2.md` (which replaces the 1.23 pre-registration). §5 keeps C1, C2 (without the ladder and the `closure_only` control), C2b (descriptive, without S2′), C5 (restated: complexity and timing), C7, C14 and C15. It withdraws C6, C10, C11, C12, C13, C16, C17, C18 and the `h₀`-only sanity row. (AB2) §4 adds Prop. 12 (two regimes), a corollary of Prop. 11 and Thm 4. (AB3) §6: the ablation table has 7 rows, and the literature meta-path row moves into it (configuration (a), seeds 1–10). The perturbation study raises heterophily by adding typed different-class facts and lowers it by removal, with targets 20/40/60/80 %. The face-effect and two-regime analyses are added. N1–N6, N2b, U3 and the scaling curve are withdrawn.

## AB-2 §1: ℋ² wording
Replace `ℋ² remains the stratification variable of C6` with `ℋ² remains the heterophily descriptor of each dataset (C6 withdrawn in 1.27; the perturbation study and the analyses use MLH and the per-node η_v of the pre-registration v2)`.

(The "Node classification" bullet of §6 says Freebase is "included without a heterophily threshold, because C6 needs datasets across the heterophily range". Replace `because C6 needs datasets across the heterophily range` with `so that the ablation datasets span homophilic and heterophilic graphs`.)

## AB-3 §4: new row after Prop. 11
> | **Prop. 12 (two regimes; 1.27; corollary of Prop. 11 and Thm 4)** | (i) If every transport is `I` (S2's operator), the solve is `x* = (I + τŜ_I)⁻¹h₀` with `Ŝ_I` the normalised Laplacian of the weighted graph the facts and cells induce on the nodes: a low-pass filter of the personalised-PageRank form (Prop. 11 with `Q = I`), which smooths neighbouring encodings. (ii) For any labelling with `C` classes there are TD-1 maps (rank `m ≥ 2`) whose class-separating section is a fixed point of the solve and is linearly separable on `x*` (Thm 4). Both are points of the same parameter family; which one a fact takes is learned (measured by R1 of the pre-registration v2). No claim that training reaches (ii) | DERIVED (from Prop. 11, Thm 4) |

## AB-4 §5: rows kept and restated
- **C2:** in the "Evidence" cell, replace `` `full` > `no_path_cells`; ladder pairs `schema` vs `uniform`, `endpoints` vs `schema` reported (not kill rules). Structural control `closure_only` (Theorem R). `` with `` `full` > `no_path_cells` (C2-cells). `` In the claim title, replace `weight ladder on identical WISP cells** (`uniform ⊂ schema ⊂ endpoints ⊂ full`; ` with `instance weights on identical WISP cells** (`.
- **C2b:** replace `; and S2′ | Not stable (mean τ-b ≥ 0.5 and adjusted p < 0.05) on ≥ 3 of 4 ablation datasets, or S2′ fails |` with ` (descriptive; 1.27: S2′ not used) | Not stable (mean τ-b ≥ 0.5 and adjusted p < 0.05) on ≥ 3 of 4 ablation datasets |`.
- **C5:** replace the whole row with:
> | C5 Scalable (1.27) | Big-O of the WISP draw, map build, one PCG iteration and memory (|E|, face incidences, `d`, `m`); timing runs on ogbn-mag (8,192 targets per epoch) and IMDB for every main-table method: time per epoch, peak GPU memory, parameter count, on one H100 | — (descriptive) | — |
- **C15:** unchanged. The 2×2 with `no_path_cells` and S0 stays reported.

## AB-5 §5: rows withdrawn
Delete the rows C6, C10, C11, C12, C13, C16, C17, C18 and "Sanity (1.23)". After "Removed in 1.11: … Removed in 1.23: C3 (relation order)." add: ` Withdrawn in 1.27 (author, ablation plan v2): C6 (and its title rule), C10, C11, C12, C13, C16, C17, C18 and the h₀-only sanity row; their arms stay in the frozen code and are not run.`

Replace `*(1.23) The statistics of C2–C17 are those of §6 "Ablation statistics";` with `*(1.23; 1.27) The statistics of C2, C14 and C15 are those of §6 "Ablation statistics";`.

## AB-6 §6 "Ablation statistics"
- Replace `C6 also on the eight H2GB datasets (`full` and `endpoints` only). ` with nothing.
- Replace `Holm per claim over its datasets (m = 4; C6 m = 12).` with `Holm per claim over its 4 datasets.`
- Replace `` `endpoints`, `closure_only`, `instance_features`, `no_path_cells` and S2 are also tuned on their own `` with `` `endpoints`, `instance_features`, `no_path_cells` and S2 are also tuned on their own ``.
- Replace `Analyses N1–N6, N2b, N4b, the perturbation study, U3 and C5's figures: `docs/ablation_analysis_preregistration_draft.md` (§4), fixed before the runs.` with `The ablation study and analyses are exactly those of `docs/ablation_analysis_preregistration_v2.md` (1.27), fixed before the runs: schema read-out, perturbation study, face effect F1–F3 with the rule-chosen case study, two-regime analysis R1–R3, complexity and timing.`

## AB-7 §6 "Ablation table"
Replace `- **Ablation table:** exactly the comparisons of §5 (C2, C2b, C10–C18) plus `h₀`-only (sanity) and the **fixed-literature-schema row** (1.18; reported, not a claim): the main model with its tuned configuration for that dataset and seed,` with `- **Ablation table (1.27):** 7 rows on the four ablation datasets, seeds 1–10: `full`, `no_path_cells`, S2, S0, `endpoints`, `instance_features` and the **fixed-literature-schema row** (1.18; 1.27: configuration (a), seeds 1–10, reported, not a claim; previously a main-table row on seeds 0–4): the main model with its tuned configuration for that dataset and seed,`. The rest of the bullet (list, budget, walk rule) is unchanged.

## AB-8 §6 "Timing and analysis-only settings"
- Replace `Every time or memory number of C5, C13, U3 and the cost-vs-accuracy figure` with `Every time or memory number of C5 (1.27: ogbn-mag and IMDB)`, and drop the parenthesis about the cost-vs-accuracy figure. If the author wants that figure kept, AB-8 keeps it instead.
- Replace the N2b sentence and the perturbation sentence (from `Analysis N2b (` to `never with the main table.`) with:
> The perturbation study (1.27; ACM and DBLP, IMDB optional) moves heterophily h = 1 − MLH to 20/40/60/80 %: it raises h by adding typed facts of the `M₂` relations that create different-class meta-path pairs, with the partner drawn uniformly over the other classes, and lowers it by removing facts (at most 50 % of a relation's facts); it never adds same-class facts. Every level has a random control with equal counts per relation, and unreached levels are reported. Methods: H2S-NN `full`, `no_path_cells` (and S2 on the targeted graphs for R3), GCN, HAN, SeHGNN, Hetero²Net and HetSheaf, each with its configuration from the original graph (or its published configuration, listed), seeds 1–5. All labels are used to build the graphs; they are semi-synthetic and are compared across methods only, never with the main table. The face knockout of F3 is an evaluation-only change (card 5 hook) and is refused in training and tuning.

## AB-9 §6 "Schema read-out and coverage"
Replace `Analysis N4b: per dataset, Spearman` with `(1.27: part of Q3 of the pre-registration v2) Per dataset and seed, Spearman`.

---

**Not changed:**
- the main tables (C7), seeds 0–4 with Welch;
- the known-answer rows;
- the method, solver, tuning and environment.

**When applied:**
- back up v1.26 to `docs/spec_backups/`;
- mark `docs/ablation_analysis_preregistration_draft.md` as superseded (one header line);
- mark the pre-registration v2 as approved.

**Applied 2026-11-19** (author: approve both; keep the cost-vs-accuracy figure). AB-4 C5 and AB-8 keep the cost-vs-accuracy figure on ogbn-mag. Backup: `docs/spec_backups/H2S-NN_methodology_spec_v1.26.md`.
