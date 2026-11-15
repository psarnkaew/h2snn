# Spec v1.23 — draft edits (ablation and analysis pre-registration), for the author's approval

**Status:** DRAFT, awaiting approval of the wording. Decisions: `docs/experiment_settings_decisions.md`, rows of 2026-11-15/16 from "Ablation statistics and seeds (B-1)" to "Pre-registration draft §8 and sweep method" (author); the full text of the analyses is `docs/ablation_analysis_preregistration_draft.md` (approved by the author 2026-11-16), which becomes a spec appendix file by reference (X-13). Written against spec **v1.22** on disk (md5 `7b44dd2c9b10bf620a8e36d8d3a9b922`, 2026-11-16); every anchor below occurs exactly once (checked by script, §Check). Apply on `design-docs` only after the author's approval; back up to `docs/spec_backups/H2S-NN_methodology_spec_v1.22.md`. **The method is unchanged** (the new arms S2/S0 are ablation configurations, not part of the model). Not included: the environment line for `h2snn-b` (handoff §6 small open item; separate approval).

---

## X-1 §1 Thesis H: no order claim (B-5)
**Replace:** `their intermediate nodes and the order of their relations — and is lost`
**with:**
> their intermediate nodes and their relations — and is lost

## X-2 §5 claims table
**X-2a. Replace the row beginning** `| **C2 (thesis H), weight ladder on identical WISP cells**` **with:**
> | **C2 (thesis H), weight ladder on identical WISP cells** (`uniform ⊂ schema ⊂ endpoints ⊂ full`; `endpoints`' `g` matched to `full`'s parameter count within 5 %; identical evaluation cells) | **P2 (core)** `full` > `endpoints` (decided on the separately tuned `endpoints`), `full` > `no_path_cells`; ladder pairs `schema` vs `uniform`, `endpoints` vs `schema` reported (not kill rules). Structural control `closure_only` (Theorem R). Statistics: §6 "Ablation statistics" (1.23) | **Kill rule:** `full` not better than `endpoints` on ≥ 3 of the 4 ablation datasets (ACM, DBLP, IMDB, Freebase). `full` not better than `no_path_cells` on ≥ 3 of 4: warning signal (stop and discuss) | No claim that instance-conditioned weighting adds information beyond transports and states; narrow the paper (§7 option B); read with S3 (§6 known-answer table) |

**X-2b. Replace the row beginning** `| C2b Schema read-out is stable and meaningful |` **with:**
> | C2b Schema read-out is stable and meaningful | Mean pairwise Kendall τ-b of the rankings of `ρ_σ = I_σ / c_σ` over seeds 1–10 of `full` (length-2 sub-path schemas with mean coverage ≥ 1 % of the cell mass, at most the ten largest by coverage; fewer than 4 → n/a); permutation null (10,000 within-seed shuffles), Holm m = 4; and S2′ | Not stable (mean τ-b ≥ 0.5 and adjusted p < 0.05) on ≥ 3 of 4 ablation datasets, or S2′ fails | The read-out is reported without interpretation; no interpretability claim. (1.23: `ρ_σ`, not `I_σ`, because `I_σ` follows WISP's coverage and is stable even for constant `g`) |

**X-2c. Delete the row beginning** `| C3 Relation order matters |` (B-5: order is not a difference from MAGNN, most literature meta-paths are palindromes, the diagonal arm confounds order with rotation capacity).

**X-2d. Replace the row beginning** `| C5 Scalable |` **with:**
> | C5 Scalable | ogbn-mag subgraphs (random paper fraction 1/64 … 1 plus neighbours), fixed configuration (`d` 32, `ℓ_max` 3, `r` 16), full batch and mini-batch, timed on one GPU type: memory and time vs incidences (log–log), mini-batch time per step vs graph size, PCG iterations vs `N`; per-dataset cost table (12 NC); mini-batch vs full-batch gap on ACM/DBLP | Full-batch memory or time slope > 1.15, or PCG-iteration slope vs `N` > 0.1 (mini-batch step-time slope > 0.15 and a gap > 1 Macro-F1 point are reported, not falsifying) | State the limit; fall back to fixed-`T` truncation (polynomial form) |

**X-2e. Replace the row beginning** `| C6 Heterophily |` **with:**
> | C6 Heterophily | P3 within each of the 12 NC datasets: per seed (1–10), the least-squares slope of the per-node gain (correct(`full`) − correct(`endpoints`), configuration (a)) on per-node meta-path heterophily centred within class; Wilcoxon, Holm m = 12. Supporting: cross-dataset Spearman (ℋ², mean gain); the edge-removal perturbation study on ACM/DBLP (§6) | Slope not positive and significant on ≥ 6 of 12 datasets, or significantly negative on any | Remove "heterophilic" from the title |

**X-2f. In the row beginning** `| C9 Relational context helps (R10) |` **replace** `Not significantly better on ≥ 2 of 3` **with** `Not better on ≥ 3 of 4 ablation datasets`.

**X-2g. In the row beginning** `| **C10 Face ≠ closure**` **replace** `Not significantly better on ≥ 2 of 3 HGB` **with** `Not better on ≥ 3 of 4 ablation datasets (decided on the separately tuned `closure_only`)`.

**X-2h. In the row beginning** `| **C11 The fact–cell coupling adds over the hub form**` **replace** `| Not significantly better |` **with** `| Not better on ≥ 3 of 4 ablation datasets |`.

**X-2i. In the row beginning** `| **C12 The node–cell term is needed**` **replace** `` `covering_only` not worse, and S1′-B passes without the term `` **with** `` `covering_only` not worse on ≥ 2 of 4 ablation datasets (i.e. `full` better on at most 2), and S1′-B passes without the term ``.

*Reason for X-2i's wording:* "not worse" is the complement of "`full` better on ≥ 3 of 4"; stated explicitly so that C12 uses the same per-dataset rule as the other claims.

**X-2j. In the row beginning** `| **C14 2-cell diffusion vs instance features**` **replace** `| Not significantly better |` **with** `| Not better on ≥ 3 of 4 ablation datasets (decided on the separately tuned `instance_features`) |`.

**X-2k. Insert after the C14 row:**
> | **C15 Learned transports matter** (1.23; Thm 4 in practice; R4 "more expressive restriction maps") | `full` vs S2 (maps = I, separately tuned); 2×2 with `no_path_cells` and S0 reported; mechanism: N2, N2b (§6) | `full` not better on both IMDB and Freebase (the heterophilic ablation datasets; a tie on ACM/DBLP is allowed) | The heterophily mechanism is not attributed to the transports (Thm 4 stays a capacity result); warning signal (stop and discuss) |
> | **C16 Direction-aware maps** (1.23) | `full` vs `direction_blind_maps`. Same-type relations exist on ACM (paper–cite–paper) and Freebase (8 of 36 relation types), none on DBLP, IMDB | `full` not better on both ACM and Freebase | "Direction-aware" is not a selling point; direction is mostly carried by node types |
> | **C17 Composition of edge maps** (1.23; Thm 3 control) | `full` vs `free_cell_maps`, two-sided | — (no kill rule). Readings: `full` better on ≥ 3 of 4 → composition helps; no difference → "no difference found", kept for Thm 3; `free_cell_maps` better on ≥ 3 of 4 → composition has a cost | The cost is stated; composition kept for theory |
> | Sanity (1.23) | `full` vs `h₀`-only | `full` not better on ≥ 3 of 4 | Warning signal (stop and discuss); Freebase (no features) `h₀`-only near chance is also a leak control |

## X-3 §5 notes under the table
**Replace:** `Removed in 1.11: C4, C4b (gates), C8 (filter bank).`
**with:**
> Removed in 1.11: C4, C4b (gates), C8 (filter bank). Removed in 1.23: C3 (relation order).

**Replace:** `*(1.14) The comparisons and statistics of C2–C14 belong to the discussion/analysis part and are restated when that part is designed; the main-table protocol of §6 (5 seeds, Welch) does not define them.*`
**with:**
> *(1.23) The statistics of C2–C17 are those of §6 "Ablation statistics"; "better on a dataset" means Holm-adjusted p < 0.05 and median paired difference > 0 in favour of `full`. If `full` beats an arm with fewer parameters whose separately tuned `d` is 64 (top of the space), the paper states that capacity is not excluded as an explanation for that claim. The main-table protocol (5 seeds, Welch) does not apply to the claims.*

## X-4 §6 Tuning: arms
**Replace:** `Ablation arms are out of scope here.`
**with:**
> Ablation arms: §6 "Ablation statistics" (1.23).

## X-5 §6 Seeds and statistics
**Replace:** `The statistics of the §5 claims (C2–C14) are restated when the discussion part is designed.`
**with:**
> The statistics of the §5 claims are in "Ablation statistics" (1.23).

## X-6 §6 Ablation table: datasets, seeds, configurations (new bullet + edited opening)
**X-6a. Replace:** `- **Ablation table:** exactly the comparisons of §5 (C2, C3, C9, C10–C14) plus `h₀`-only (sanity) and the`
**with:**
> - **Ablation statistics (1.23, author decisions B-1–B-3, B-13).** Datasets: HGB ACM, DBLP, IMDB, Freebase (two homophilic, two heterophilic; Freebase full batch if it fits at the largest configuration, else mini-batch); C6 also on the eight H2GB datasets (`full` and `endpoints` only). Seeds 1–10 (the tuning seed 0 excluded; `full`'s main-table runs of seeds 1–4 reused); runs paired by seed (same WISP draws and batch order in every arm). Test: Wilcoxon signed-rank on the 10 paired differences of test Macro-F1, two-sided; Holm per claim over its datasets (m = 4; C6 m = 12). Reported: mean ± sample std per arm, raw and adjusted p, mean paired difference with a bootstrap 95 % CI. Configurations: every arm runs with `full`'s tuned configuration plus its one change ((a)); `endpoints`, `closure_only`, `instance_features`, `no_path_cells` and S2 are also tuned on their own ((b): 100 Bayesian trials on seed 0, space restricted to the keys that act in the arm) and their claims are decided on (b), with (a) reported beside; if (a) and (b) disagree, both are reported and the paper follows (b). Every ablation run keeps its best checkpoint and resolved configuration. A parameter table lists every arm (total and the removed or added part). LP and GC have no ablation (stated limitation). Analyses N1–N6, N2b, N4b, the perturbation study, U3 and C5's figures: `docs/ablation_analysis_preregistration_draft.md` (§4), fixed before the runs.
> - **Ablation table:** exactly the comparisons of §5 (C2, C2b, C9–C17) plus `h₀`-only (sanity) and the

**X-6b. Insert after the Ablation-table bullet's last sentence** `if it is not better, the paper says only that a list is not needed there.`
> The appendix pair (`g` reads `[h₀, c]` in `full` and `endpoints`) runs with configuration (a) on seeds 1–10 and is reported without a claim (1.23).

## X-7 §6 Schema read-out: C2b on `ρ_σ`
**X-7a. Replace:** `Kendall τ-b of the `I_σ` rankings of the length-2 sub-path schemas between seeds (C2b; the same set as the top ten; fixed before results, author 2026-11-13).`
**with:**
> Kendall τ-b of the `I_σ` and `c_σ` rankings between seeds, descriptive. C2b's statistic is computed on the ablation runs of `full` (seeds 1–10) on the rankings of `ρ_σ` (§5 C2b; 1.23 replaces the `I_σ` statistic of 1.18 before any result).

**X-7b. Replace:** `only if `ρ_σ > 1` on 5 of 5 seeds; otherwise the values are reported with no statement about selection. No significance test (5 seeds; descriptive).`
**with:**
> only if `ρ_σ > 1` on 10 of 10 seeds of the ablation runs of `full` (1.23); otherwise the values are reported with no statement about selection. No significance test (descriptive). Analysis N4b: per dataset, Spearman between `ρ_σ` and the label homophily of σ's induced target-type graph (descriptive; positive → the weights favour homophilous schemas; near zero → they do not follow homophily).

## X-8 §6 Known-answer rows
**Replace:** `- *(1.14: deferred by the author to the discussion part; pre-registration drafts in `docs/known_answer_preregistration_draft.md`.)* **Known-answer rows (1.7; run with the protocol, pre-registered under §0 rule 6 before the freeze):**`
**with:**
> - **Known-answer rows (1.7; 1.23: run on the cluster after M3b and before any tuning; pre-registered under §0 rule 6 in `docs/known_answer_preregistration_draft.md`; settings: AdamW without scheduler or clipping, seeds 1–3; S1′-B as approved 2026-11-11, a `full` + oracle miss is a warning signal; S3: length-4 chain, relevance `t_P ⊕ t_B ⊕ t_Q = 0`, invisible to every fact window and to `endpoints`; S2′ option (a)):**

**Then replace:** `if S3 passes and C2's kill rule fires, H is not supported through the weight channel. Either way the paper is narrowed as §5 states; no component is added.`
**with:**
> if S3 passes and C2's kill rule fires, H is not supported through the weight channel; if S3 fails and C2's kill rule does not fire, the real-data gain is claimed but not that `g` learns relevance no fact can see (S3 reported as a limit) (1.23). When the kill rule fires the paper is narrowed as §5 states; no component is added.

## X-9 §6 WISP settings: proposal variance before tuning; U3 without E11
**X-9a. Replace:** `; the U3 calibration curve keeps E11's rule.`
**with:**
> ; the U3 curves do not use it either (1.23).

**X-9b. Replace:** `the variance gate is re-measured on a fully trained checkpoint on the cluster, and `adaptive` is used only if it then reaches 20 %.`
**with:**
> the variance gate is re-measured **before any tuning** (1.23), on ACM, seed 1, a fixed central configuration (`d` 32, `ℓ_max` 3, `r` 16, lr 1e-3, weight decay 1e-4, dropout 0.5) trained to early stopping; `adaptive` is used only if it reaches 20 %, and the result is fixed for every later run.

**X-9c. Replace:** `Analysis curve (not tuning): accuracy vs `ℓ_max ∈ {2, 3, 4, 5, 6}` (U3), with `r` calibrated per `ℓ_max`; DBLP at `ℓ_max = 2` is excluded (no `r` reaches the budget, 1.7).`
**with:**
> Analysis curves (U3; 1.23, not tuning): one at a time around `full`'s tuned configuration, `ℓ_max` ∈ {2, …, 6} at the tuned `r` and `r` ∈ {8, 16, 32, 64, 128} at the tuned `ℓ_max` (6 and 128 are analysis-only values), on the four ablation datasets, seeds 1–5; recorded per point: Macro-F1, walks per length, time per epoch and peak memory (one GPU type), OOM as a point. A drop of more than 2 points at `ℓ_max` 6 below the best point on ≥ 3 of 4 datasets is reported as "drops at large `ℓ_max` at fixed `r`", never as "long paths are harmful".

## X-10 §6 WISP gate (i) closed
**Replace:** `**Status (1.7):** (ii) is met (E12, Part K `6ac401f`); (i) stands as recorded for G0b (hypotheses deferred to the cluster by the author's decision).`
**with:**
> **Status (1.23):** (ii) is met (E12, Part K `6ac401f`); (i) is closed by author decision (2026-11-16): G0b's gap was measured on the v1/v2 model; its h2 (reach of K first-order steps) does not exist in v3-B, h3 (trainer settings) is closed by the shared tuned protocol, and h1 (ego shortcut) is answered by the `full` vs `h₀`-only sanity comparison. No untuned v3-B vs R-GCN comparison is run.

## X-11 §6 v3-B arms: S2 and S0
**Replace:** `` `uniform / schema / endpoints` (C2 ladder); `direction_blind_maps`; `h₀`-only.``
**with:**
> `uniform / schema / endpoints` (C2 ladder); `direction_blind_maps`; `h₀`-only; (1.23) **S2** = `full` with every fact map `F = I` (hence every transport `I`; cells, `g`, `a_e`, `β_f`, `τ`, solver and head unchanged; the geometry network unused) and **S0** = `no_path_cells` with `F = I` (C15; an identity map operator under `anchored`, own card). No dense or diagonal map family is built (C3 removed).

## X-12 §6 timing rule and analysis exceptions (new bullet, inserted after the "Cost gate (1.11 …)" bullet)
> - **Timing and analysis-only settings (1.23).** Every time or memory number of C5, C13, U3 and the cost-vs-accuracy figure (ogbn-mag: test Macro-F1 vs time per epoch and vs peak memory, every main-table method at its tuned configuration) comes from separate timing runs on one GPU type, stated in the paper; never from the main or ablation runs. Analysis N2b (`τ` fixed ∈ {1, 10, 100}, `full` vs S2, DBLP and IMDB, seeds 1–3) lifts the `τ > 50` stop and raises `T_max` to 500 for its runs only. The edge-removal perturbation study (ACM, DBLP; at most 50 % of a relation's facts removed; a density-matched random-removal control at every level; seven methods with their configurations tuned on the original graph; seeds 1–5) uses all labels to choose removals and is compared across methods only, never with the main table.

## X-13 §7 OPEN-5
**Replace:** `` `OPEN-5` Closed for the main tables in 1.14 (§6 Tuning); still open for ablation arms (discussion part).``
**with:**
> `OPEN-5` Closed for the main tables in 1.14 (§6 Tuning) and for ablation arms in 1.23 (§6 "Ablation statistics"; analyses fixed in `docs/ablation_analysis_preregistration_draft.md`, which is frozen with this spec).

## Proposed changelog entry (prepended above the 1.22 entry; version line → 1.23, date 2026-11-16)
> **Changes from 1.22 (to 1.23; ablation and analysis design, author decisions of 2026-11-15/16 in `docs/experiment_settings_decisions.md` (B-1 … B-13, B2, mechanism analyses, sweep), pre-registration `docs/ablation_analysis_preregistration_draft.md`, draft `docs/spec_v1.23_draft_edits.md`; the method is unchanged):** (X1) thesis H without relation order; C3 removed. (X2) Claims decided on four HGB datasets (ACM, DBLP, IMDB, Freebase) with "≥ 3 of 4"; seeds 1–10 paired by seed, Wilcoxon, Holm per claim; arms with `full`'s configuration plus one change, and `endpoints`, `closure_only`, `instance_features`, `no_path_cells`, S2 also tuned on their own (claims decided on those). (X3) New claims C15 (learned transports; arms S2/S0 with maps = I), C16 (direction-aware maps), C17 (composition); `h₀`-only sanity. (X4) C2b on `ρ_σ` rankings (τ-b ≥ 0.5, permutation null); C6 decided within datasets (per-node slope, ≥ 6 of 12); C5 criteria (slopes 1.15 / 0.1) on ogbn-mag subgraphs; timing on one GPU type. (X5) Known-answer tasks before tuning with protocol settings; S3 design; full C2 × S3 table. (X6) WISP gate (i) closed; proposal variance measured before tuning; U3 one-at-a-time without E11. (X7) Analyses N1–N6, N2b, N4b and the edge-removal perturbation study pre-registered.

## Check
Anchors checked with a script on spec v1.22 (md5 above), 2026-11-16: every anchor occurs exactly once, except `| Not significantly better |`, which occurs twice (rows C11 and C14); X-2h and X-2j therefore replace it only inside the named row. The X-12 insertion point `- **Cost gate (1.11;` occurs once.

## X-14 §6 Seeds and statistics: Holm per task (author 2026-11-16)
**Replace:** `and both raw and adjusted values are given; which adjustment (Holm or Bonferroni) is decided after the results (author).`
**with:**
> and both raw and adjusted values are given; the adjustment is Holm, fixed before the results (1.23, author), applied per task: NC (24 tests), LP (28) and GC (8) are separate families.

Anchor checked: occurs once in spec v1.22 (md5 `7b44dd2c9b10bf620a8e36d8d3a9b922`). Changelog addition: "(X8) main-table multiplicity: Holm per task, fixed before the results."

## X-15 §6 Graph classification: heterophily statement and type mixing (author 2026-11-16)
**Replace:** `MUTAG's test set is 20 graphs. No heterophily claim.`
**with:**
> MUTAG's test set is 20 graphs. No heterophily claim and no heterophily value (1.23): labels are graph-level, so ℋ² and MLH are undefined, and MDE is not reported (constant features under this protocol; MUTAG and NCI1 have no attributes). The GC dataset table reports the number of node and edge types and the type-mixing ratio (share of edges joining nodes of different types) as a heterogeneity statistic, never as heterophily.

Anchor checked: occurs once in spec v1.22. Changelog addition: "(X9) GC: no heterophily value; type-mixing ratio reported."
