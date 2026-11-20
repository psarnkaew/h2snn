# H2S-NN — Ablation and analysis pre-registration v3 (APPROVED 2026-11-20; binding with spec 1.29)

**Status:**
- Written 2026-11-20 by the model session from the author's item-by-item review of v2 on 2026-11-20 (decisions D1–D8 below; summary page https://claude.ai/artifact/UF5TbzQuJRMM9UnyAA5evT).
- Approved by the author on 2026-11-20. It supersedes `docs/ablation_analysis_preregistration_v2.md` (d3c8d44), which stays as history.
- It becomes binding with spec 1.29 (`docs/spec_v1.29_draft_edits.md`) and is frozen before the first run it governs. No run governed by v2 has started.

**Scope rule (author, 2026-11-19; kept):** the ablation study and model analysis are exactly the items below. Nothing else is run, reported or claimed.

**Code:** the arms exist in frozen `src/` (`v3b` ≥ `fc0ebaa`; `uniform` is `path_weight_input: uniform`). The changes of v3 touch only `scripts/` (card 9), so the H2S-NN freeze is unchanged.

**Decisions of 2026-11-20 (author):**

| # | Decision | Reason given |
|---|---|---|
| D1 | Seeds 1–5; no p-values, no Holm; a fixed 4-of-5 rule decides "better on a dataset" | Time; Wilcoxon cannot reach p < 0.05 at n = 5 (minimum 0.0625), and a rule fixed in advance is still needed for the kill rule and the warnings |
| D2 | Configuration (a) only: `full`'s tuned hyperparameters, one change per row. Configuration (b) (separate tuning of the arms) is dropped | Standard practice in the field; saves 1,600 trials |
| D3 | Add the `uniform` row (every face the same learned weight) | Tests whether the face weight `g` helps at all; `endpoints` only tests whether `g` needs the middle node |
| D4 | The C2 kill rule and the C14 fallback are rewritten (§3); C15 kept as it is | v2 pointed to a "§7 option B" that no longer exists, and its C14 fallback named theory as the contribution |
| D5 | Schema read-out: no permutation null; "favour σ" needs 5 of 5 seeds; wording "the learned face weights", never "the sampler"; the homophily link is shown as a scatter | WISP's proposal is fixed, `g` is what learns; 4–10 schemas per dataset make one Spearman number unstable |
| D6 | Perturbation: "loses less" and "grows" get fixed definitions (§4.2); HetSheaf and CSNN are not waited for | No claim depends on those two rows |
| D7 | Face effect: fixed net-fix rule; scale-free margin; F3 and case-study wording (§4.3) | Margins of two models are on different scales; ACM has no G0 node |
| D8 | Two-regime: fixed rules for R1–R3; node energy divided by ‖x*_v‖²; R1 is read together with R3 (§4.4). Scalability unchanged: time per epoch only (author) | Raw `e_v` grows with the norm of `x*_v` |

---

## 0. Questions and where each is answered

| # | Question (author's table, advisor meeting 2026-11-19) | Answered by | § |
|---|---|---|---|
| Q1 | Does the model generalize? | Main tables (spec §6); not part of this document | — |
| Q2 | Do faces and each component matter? | Ablation table, 8 rows | 2, 3 |
| Q3 | Do meta-paths relate to heterophily? Do the learned face weights favour meaningful meta-paths, beyond what random walks cover? | Schema read-out | 4.1 |
| Q4 | Is the model related to the heterophily level? Is it robust when heterophily changes? | Perturbation study | 4.2 |
| Q5 | Is it scalable? | Complexity and timing | 4.5 |
| A1 | Advisor: how does a face change the embedding so that the prediction becomes right? | Face effect F1–F3 and case study | 4.3 |
| A2 | Advisor: show, beyond accuracy, that one model works under homophily and heterophily (proof; energy vs performance and representation) | Two-regime analysis R1–R3 and Prop. 12 (spec 1.27) | 4.4 |

---

## 1. Common settings

| Item | Setting |
|---|---|
| Datasets | HGB ACM, DBLP (homophilic), IMDB, Freebase (heterophilic). Node classification. Freebase is full batch if the largest configuration fits on one H100 94 GB (card 5 U3), else mini-batch (spec 1.26) |
| Metric | Test Macro-F1 (IMDB: the mean per-genre F1, spec §6). Checkpoints are selected on validation Macro-F1 |
| Seeds | **1–5** (seed 0 is the tuning seed of `full` and is excluded, because the rows are not tuned and seed 0 would favour `full`). Runs are paired by seed: the same seed gives the same WISP draws and batch order in every row. `full`'s main-table runs of seeds 1–4 are reused; `full` seed 5 is run |
| Configuration | **(a) only:** `full`'s tuned configuration for that dataset, with the row's one change. No row is tuned on its own. Stated in the paper: "hyperparameters fixed to `full`'s tuned values" |
| "`full` better than row X on a dataset" | Both: (1) the mean over seeds 1–5 of the paired difference `full − X` is > 0; (2) `full` is strictly higher than X in **at least 4 of the 5 seeds**. No p-value and no multiplicity adjustment are computed or reported |
| Reported per comparison | Mean ± sample std (ddof = 1) per row; the five paired differences; their mean; the 4-of-5 count |
| Capacity | The parameter count of every row (total, and the part removed or added) is reported beside the table. No capacity rule decides a claim |
| Analyses | Descriptive. Expectations and the rule that reads each are written below before any run. An outcome that differs is reported as it is, and the paper makes no claim on that point. "4 of 5 seeds" in §4 means the per-seed statement holds in at least 4 of seeds 1–5 |
| Machine | TSUBAME H100. Ablation and analysis runs start after `full` is tuned. Every run keeps its best checkpoint and resolved configuration |
| Not done | Ablation on LP and GC (stated limitation). Baselines never appear in the ablation table |

---

## 2. The ablation table (8 rows; each is `full` with one change; all configuration (a))

Example used throughout: papers P1 (AI) and P9 (DB) share author A7, and the walk P1–A7–P9 is one face; P4 (AI) is another paper of A7.

| Row | One change from `full` (on P1–A7–P9) |
|---|---|
| `full` | — (tuned) |
| `no_path_cells` | `β_f = 0`: no faces, so the model is an edge sheaf. P1 meets A7 only through its own fact |
| S2 | Every fact map is `I`, so every transport is `I`. Faces, `g`, `a_e`, `β_f`, `τ`, solver and head are unchanged |
| S0 | `no_path_cells` with maps = I |
| `uniform` (added in v3) | Every face gets the same weight `w = exp(L·tanh(b/L))`, one learned scalar `b` shared by all faces; `g` is unused (`path_weight_input: uniform`). P1–A7–P9 and P1–A7–P4 weigh the same |
| `endpoints` | `g` reads P1, P9 and the relations only; A7 is invisible to the face weight. Hidden width is matched to `full`'s `g` within 5 % |
| `instance_features` | `β_f = 0` in the solve; `Σ_f β_f mean_v T_{v,f} h_v` is concatenated to the head input (the MAGNN-type instance encoder) |
| Literature meta-paths | Faces are drawn only along HGB's published meta-paths (`path_source: schemas`; spec §6 list and walk budget), instead of by random walks |

**Code status:** every row exists in `src/` (v3b `fc0ebaa`). Out-of-scope arms (`closure_only`, `hub_A`, `covering_only`, `schema`, `direction_blind_maps`, `free_cell_maps`, `ego_head`, `h0_only_readout`) stay in the frozen code and are not run.

---

## 3. Claims (fixed before any run)

"Better" is the rule of §1.

| Claim | Comparison | Holds if | If not |
|---|---|---|---|
| **C2 (thesis H; kill rule)** | `full` vs `endpoints` | Better on ≥ 3 of 4 | Kill rule: the paper makes no claim that the face weight depends on which node is in the middle of the face. It claims only what passed among C2-cells, C2-w, C14 and C15. Stop and discuss before writing |
| **C2-cells** | `full` vs `no_path_cells` | Better on ≥ 3 of 4 | Faces add nothing over the edge sheaf. **Warning signal:** stop and discuss |
| **C2-w** (added in v3) | `full` vs `uniform` | Better on ≥ 3 of 4 | The paper makes no claim that learned per-face weights help; faces are presented as equally weighted cells |
| **C14** | `full` vs `instance_features` | Better on ≥ 3 of 4 | The paper makes no claim that diffusing over faces beats using them as features of the head. The remaining contribution is what C2, C2-cells, C2-w and C15 support |
| **C15** | `full` vs S2 | Better on **both** IMDB and Freebase. A tie on ACM and DBLP is allowed | The paper drops the claim that its heterophily mechanism comes from the transports (Thm 4 stays a capacity result). **Warning signal** |
| 2×2 (reported) | (1) `full − no_path_cells`; (2) S2 − S0; (3) `full` − S2; (4) `no_path_cells` − S0 | — | If the mean of (1) exceeds the mean of (2) on a dataset, faces and sheaf maps are complementary there; reported, not claimed |
| Literature row (reported) | `full` vs literature meta-paths | — | If the literature row is better than `full` on a dataset (the §1 rule with the roles swapped), the paper says the expert-chosen meta-paths still add information there and narrows its statement to "no list is needed"; otherwise it says only that a list is not needed there. It never claims WISP finds better paths than the list |

---

## 4. Analyses

### 4.1 Schema read-out (Q3)
Computed from `full`'s runs of seeds 1–5 (configuration (a); seeds 1–4 are the main-table runs). No training.

**Quantities:**
- `c_σ`: coverage, the share of the face mass in sub-path schema σ with every `w_f = 1`;
- `I_σ`: the learned share;
- `ρ_σ = I_σ / c_σ`. Under `uniform`, `ρ_σ = 1` for every σ, so a departure from 1 is the effect of `g`.

**Set:** length-2 schemas with mean coverage ≥ 1 %, at most the ten largest by coverage. Fewer than 4 gives "n/a".

**Stability:** the mean pairwise Kendall τ-b of the `ρ_σ` rankings over seeds 1–5 (10 pairs). Stable = mean τ-b ≥ 0.5. No permutation null. The τ-b of the `c_σ` rankings is reported beside it.

**Literature check:** the number of seeds (of 5) with `ρ_σ > 1` for each HGB published meta-path. The paper says "the learned face weights favour σ" only at 5 of 5.

**Heterophily link:** a scatter of `ρ_σ` against the label homophily of σ's induced target-type graph (spec §1 reference code), all seeds, per dataset; the per-seed Spearman is printed beside it. Positive means the weights favour homophilous schemas; near zero means they do not follow homophily. No single-number claim (4–10 schemas per dataset).

**Wording (fixed):** the paper says "the learned face weights", never "the sampler learns": WISP's proposal is fixed (spec §7) and `g` weighs the faces after they are drawn.

**Reading:** descriptive. If the rankings are not stable, the read-out is reported without interpretation. S2′ is not used.

### 4.2 Perturbation study (Q4)
**Graphs:** ACM (h = 0.262 now) and DBLP (h = 0.128). IMDB is not used. Heterophily is h = 1 − MLH ∈ [0, 1] over the spec §1 `M₂` set, computed with the reference code. ℋ² is reported beside it at every level.

**Targets:** h ∈ {0.20, 0.30, 0.40, 0.50, 0.60}, each within ±0.01, plus the original graph as a reference point.

**Ceiling (DERIVED, checked against card 7 job 17814):** an intermediate node with n target neighbours makes a clique in the meta-path graph, and at most (1 − 1/C)·n/(n−1) of its pairs can be different-class. For large n this is 1 − 1/C: ≈ 0.667 for ACM (C = 3) and 0.75 for DBLP (C = 4). Edge addition reached 0.638 and 0.748. Higher levels would need fabricated intermediate nodes, which are not used. Higher heterophily is covered by the real heterophilic datasets of the main table. The figure draws 1 − 1/C as a dashed line.

**Generator rules (card 7 V3, accepted):** additions are split across the `M₂` relations in proportion to their size. For an author or term fact, a paper of another class is drawn uniformly, and the new fact links the labelled paper to one of that paper's neighbours. Unreached levels are written but not run. Every run logs the fingerprint of the edge list it loaded, and the manifest checks it against the generator's fingerprint.

| Direction | Rule |
|---|---|
| Raise h | **Add** facts of the `M₂` relations already in the graph, with typed endpoints, that create different-class meta-path pairs. The partner of the labelled endpoint is drawn uniformly over all other classes, so an added fact says only "not my class". No duplicates and no self-loops |
| Lower h | **Remove** facts that create different-class pairs, at most 50 % of a relation's facts. Same-class edges are never added: that would build test labels into the graph |
| Control | Every targeted graph has a random graph with the same number of facts added or removed per relation, drawn uniformly. The effect of heterophily is the targeted result minus the random result |

**Reachability:** a level not reachable under the 50 % cap is reported as unreached, with the closest value. The cap is never raised after results.

**Methods:**
- H2S-NN `full` and `no_path_cells` (configuration (a); the original-graph points reuse the ablation runs of seeds 1–5);
- GCN, HAN, SeHGNN, Hetero²Net, HetSheaf and CSNN, each with its configuration tuned on the original graph, or its published configuration (listed); never retuned. GCN and HAN use HGB's published run commands; SeHGNN and Hetero²Net use their published configurations; HetSheaf and CSNN use their ACM/DBLP tuning. **HetSheaf and CSNN are not waited for:** if their configurations are not ready when the other perturbation runs are complete, their rows are left out of the figure and the omission is stated;
- seeds 1–5; the six baselines also run on the original graphs (seeds 1–5, same launcher and configuration source) as their reference points;
- in addition, H2S-NN S2 on the targeted graphs only, for R3's perturbation axis.

**Labels:** all labels are used to build the graphs. The graphs are semi-synthetic, and curves are compared across methods only, never with the main table.

**Definitions (fixed now):**
- `h_max`: the highest targeted level reached on the dataset (0.6 on both, card 7).
- **Drop** of method m at seed s: `drop_m(s) = F1_m,random(h_max, s) − F1_m,targeted(h_max, s)`, i.e. [F1 on the original graph − F1 targeted] minus [F1 on the original graph − F1 random], the loss caused by heterophily beyond the loss caused by the same number of added or removed facts.
- **Gap** at a graph G and seed s: `gap(G, s) = F1_full(G, s) − F1_no_path_cells(G, s)`.

**Expected (fixed now):**
- E1: H2S-NN `full` loses less than each of GCN, HAN and SeHGNN: the mean of `drop_full` is below the mean of `drop_m`, and `drop_full(s) < drop_m(s)` in at least 4 of 5 seeds, on **both** ACM and DBLP. The comparisons with Hetero²Net, CSNN and HetSheaf are reported (no expectation fixed).
- E2: `gap` grows with h: `gap(targeted h_max, s) > gap(original, s)` in at least 4 of 5 seeds, on both ACM and DBLP.

If either expectation fails, it is reported as it is, and the paper makes no robustness claim on that point.

### 4.3 Face effect (A1)
Uses `full` and `no_path_cells`, configuration (a), the 4 datasets, seeds 1–5, and card 5 hooks (b) and (d).

- **η_v** (amended 2026-11-20 by the author, before any run): the mean, over the `M₂` paths on which v has at least one labelled target-type neighbour, of the share of those neighbours (distinct, u ≠ v) whose label differs from v's — the node-level counterpart of h = 1 − MLH. IMDB: per path, the mean over genres of the binary disagreement, then the mean over paths. Nodes with no labelled neighbour on any path are excluded and counted.
- **η groups:** G0 = {η_v = 0}; G1–G3 = tertiles of η_v over the test targets with η_v > 0, value bins (−∞, t1], (t1, t2], (t2, ∞). The same rule on every dataset; an empty or small G0 is reported with its count, never merged (ACM: G0 is empty). **G_low** = the lowest non-empty group (G0, else G1).
- **F1, flips:** per test node, wrong→right and right→wrong from `no_path_cells` to `full`, per η group. **Net fix rate** of a group = (wrong→right − right→wrong) / group size; mean and range over seeds.
- **F2, margin** (single-label datasets ACM, DBLP, Freebase): margin_v = distance from x*_v to the nearest other-class centroid − distance to its own-class centroid, with centroids from the test x* of the same model and seed. **Scale-free margin:** margin_v divided by the mean pairwise distance between that model's class centroids (same seed), so that the two models are compared on the same scale. Reported for both models by η group; the raw margin is reported beside it.
- **F3, knockout:** on trained `full`, seeds 1–3, every cell through test node v gets `β_f = 0` at evaluation only. Report the share of correct predictions that turn wrong, and the scale-free margin change, by η group. At most 2,000 test nodes per (dataset, seed), drawn with a fixed seed.
- **Expected (rules fixed now), per dataset:**
  - F1: the net fix rate of all test nodes is > 0 in at least 4 of 5 seeds; and the net fix rate of G3 exceeds that of G_low in at least 4 of 5 seeds;
  - F2: the mean scale-free margin difference (`full` − `no_path_cells`) is > 0 in at least 4 of 5 seeds; and it is larger in G3 than in G_low in at least 4 of 5 seeds;
  - F3: knockout turns a non-zero share of correct predictions wrong in each of seeds 1–3.
- **Wording (fixed):** F1 answers "would a model without faces do worse"; F3 answers only "does the trained model rely on the faces through v" (the knockout puts the model in a state it was not trained in). The paper states them separately.
- **Case study (fixed rule):**
  - candidates: ACM seed 1 test nodes with a different-class neighbour (fact or `M₂` meta-path) whose F3 knockout turns right into wrong;
  - pick: the largest scale-free margin drop; on a tie, the smaller id;
  - the figure shows the node, the different-class neighbour, the 3 cells through the node with the largest `w_f` (ids, types, schema), a 2-D PCA (fitted on all test x* of `full`) of h₀, x* with the knockout and x* of `full` with class centroids, and the true-class probability with and without the knockout;
  - the caption says the node was "chosen by a fixed rule: the largest margin drop", not that it is typical, and points to F1–F3 for the overall picture.

### 4.4 Two-regime analysis (A2)
**Theory:** Prop. 12 (spec 1.27, a corollary of Prop. 11 and Thm 4).
- With every transport equal to `I`, the solve is the resolvent `(I + τŜ_I)⁻¹h₀` of the weighted node graph: a low-pass smoothing of the personalised-PageRank form (homophily regime).
- For any labelling, there are TD-1 maps whose class-separating section is a fixed point, separable by a linear classifier on `x*` (heterophily regime).
- Both are points of one parameter family, and training chooses per fact.

**Measurements** (no training; card 5 hooks (a), (c), (d); seeds 1–5):

| Part | Measure | Expected (rule fixed now) |
|---|---|---|
| **R1** transports | For cells whose two endpoints are labelled targets, the endpoint transport composed from the fact transports along the cell; ‖T − I‖_F split by same-class vs different-class endpoint pairs (medians, AUROC). On ACM P–P and Freebase B–B, also the fact-level ‖Q_e − I‖_F. The learned τ per run | AUROC > 0.5 (different-class pairs farther from `I`) in at least 4 of 5 seeds, per dataset |
| **R2** energy vs performance | Node energy, primary: `ẽ_v = e_v / ‖x*_v‖²` (hook (a)); raw `e_v` reported beside. AUROC of `ẽ_v` separating wrong from correct predictions; Spearman(`ẽ_v`, η_v) | AUROC > 0.5 (wrong predictions carry more energy) in at least 4 of 5 seeds, per dataset; the Spearman is reported (heterophilic nodes can still have low energy) |
| **R3** energy vs representation | `full` and S2: normalised energy Ẽ = E / ‖x*‖²; a linear probe (logistic regression, `C = 1`, trained on training targets, scored on test targets) on x* and on h₀; the within/between-class distance ratio on x* and h₀ | On IMDB and Freebase: (probe Macro-F1 of `full` − of S2, on x*) is larger in G3 than in G_low in at least 4 of 5 seeds. At low η both models are expected to separate classes better on x* than on h₀ (reported) |

**Two axes:**
1. Nodes, by η group (G0–G3), on the 4 datasets.
2. Graphs, by perturbation level (§4.2), with `full` and S2 on the targeted graphs.

**Wording (fixed):** R1 alone shows only that the maps rotate differently by class; the paper links rotation to correct predictions through R3 (removing the rotation, S2, loses the class separation at high η), and never from R1 alone.

### 4.5 Scalability (Q5)
- **Complexity:** big-O of the WISP draw, the map build, one PCG iteration and memory, in |E|, the face incidences, d and m = 4. Compared with meta-path enumeration (HAN, SeHGNN) and d² per edge (NSD). Written in the paper; no run.
- **Timing:** on ogbn-mag (8,192 training targets per epoch) and IMDB, for every main-table method at its tuned configuration (the card 2 central point if not tuned, flagged). Time per epoch only (author, 2026-11-20): one-off preprocessing time and total training time are not reported. The paper states "time per epoch at a fixed 8,192 training targets per epoch".
- **Cost-vs-accuracy figure (author, 2026-11-19: kept):** on ogbn-mag, test Macro-F1 (main table) vs time per epoch and vs peak memory (timing runs), one point per main-table method at its tuned configuration. No extra training. A method that does not fit is shown as N/A.
  - Measures: time per epoch (median of epochs 2–4 after one warm-up), peak GPU memory, parameter count.
  - Separate timing runs on one H100.

---

## 5. Run budget (estimates; all on TSUBAME after `full` is tuned)

| Block | Runs |
|---|---|
| Ablation (a): 7 rows × 4 datasets × 5 seeds + `full` seed 5 × 4 | 144 small |
| Ablation (b) | none (D2) |
| Perturbation: 8 methods × 2 datasets × 5 levels × 2 kinds × 5 seeds + S2 on the targeted graphs (50) + baseline reference points (60) | ≈ 910 small (≈ 690 without HetSheaf and CSNN) |
| §4.1, §4.3, §4.4 | read from checkpoints (F3 is evaluation only) |
| Timing | ≈ 28 runs, a few hours on one H100 |

## 6. Changed from v2 (author, 2026-11-20)
- Seeds 1–10 → 1–5; Wilcoxon, Holm, bootstrap CI and raw/adjusted p removed; the 4-of-5 rule decides (D1).
- Configuration (b) and the capacity rule removed; parameter counts reported (D2).
- `uniform` row and claim C2-w added (D3).
- C2 kill-rule consequence and C14 fallback rewritten (D4).
- §4.1: no permutation null; 5 of 5; wording; scatter (D5).
- §4.2: drop and gap defined; E1, E2 rules; HetSheaf and CSNN not awaited (D6).
- §4.3: net fix rate, G_low, scale-free margin, F1–F3 rules, F3 and caption wording (D7).
- §4.4: R1–R3 rules; `ẽ_v`; R1 read with R3 (D8).

## 7. Removed from v1 (author, 2026-11-19; unchanged)
- **Arms:** the C2 ladder `schema`, C10 `closure_only`, C11 `hub_A`, C12 `covering_only`, C16 `direction_blind_maps`, C17 `free_cell_maps`, C18 `ego_head`, `h₀`-only. (`uniform` returns in v3 as its own row, not as part of a ladder.)
- **Analyses:** C6 on the 12 NC datasets and its title rule; N1–N4, N2b, N2c; U3; the C5 scaling curve and the mini-batch vs full-batch gap.
