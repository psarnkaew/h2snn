# H2S-NN — Ablation and analysis pre-registration v2 (SUPERSEDED 2026-11-20 by `docs/ablation_analysis_preregistration_v3.md` and spec 1.29; history)

**Status:**
- Written 2026-11-19 by the model session from the author-approved plan `docs/ablation_plan_v2_2026-11-19.html` (decisions log rows of 2026-11-19).
- When the author approves it, it supersedes `docs/ablation_analysis_preregistration_draft.md` (v1), which stays as history.
- It becomes binding with spec 1.27 (`docs/spec_v1.27_draft_edits.md`) and is frozen before the first run it governs.

**Scope rule (author, 2026-11-19):** the ablation study and model analysis are exactly the items below. Nothing else is run, reported or claimed.

**Code:** the arms exist in `src/`; the hooks come from card 5; the scripts from card 7.

---

## 0. Questions and where each is answered

| # | Question (author's table, advisor meeting 2026-11-19) | Answered by | § |
|---|---|---|---|
| Q1 | Does the model generalize? | Main tables (spec §6); not part of this document | — |
| Q2 | Do faces and each component matter? | Ablation table, 7 rows | 2, 3 |
| Q3 | Do meta-paths relate to heterophily? Does the sampler learn meaningful meta-paths, not just random walks? | Schema read-out | 4.1 |
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
| Seeds | 1–10 (seed 0 is the tuning seed). Runs are paired by seed: the same seed gives the same WISP draws and batch order in every arm. `full`'s main-table runs of seeds 1–4 are reused |
| Test | Wilcoxon signed-rank on the 10 paired differences of test Macro-F1, two-sided (`scipy.stats.wilcoxon`). The minimum attainable p at n = 10 is 0.00195 |
| Multiplicity | Holm per claim over its 4 datasets |
| Passes on a dataset | Holm-adjusted p < 0.05 and median paired difference > 0 in favour of `full` |
| Reported per comparison | Mean ± sample std per arm, raw and adjusted p, median difference, mean paired difference with a bootstrap 95 % CI (10,000 resamples, percentile, fixed seed) |
| Configuration (a) | `full`'s tuned configuration for that dataset, with the row's one change |
| Configuration (b) | The arm tuned on its own: plain Bayesian search, 100 trials, seed 0, validation metric, with the space restricted to the keys that act in the arm. Then seeds 1–10. Arms: `endpoints`, `no_path_cells`, `instance_features`, S2. Their claims are decided on (b), with (a) reported beside. If (a) and (b) disagree, both are reported and the paper follows (b) |
| Capacity reading | If `full` beats an arm with fewer parameters whose tuned `d` is 64 (top of the space), the paper states that capacity is not excluded as an explanation for that claim |
| Analyses | Descriptive. Expectations are written below before any run. An outcome that differs is reported as it is, and the paper makes no claim on that point |
| Machine | TSUBAME H100. Ablation and analysis runs start after `full` is tuned. Every run keeps its best checkpoint and resolved configuration |
| Not done | Ablation on LP and GC (stated limitation). Baselines never appear in the ablation table |

---

## 2. The ablation table (7 rows; each is `full` with one change)

Example used throughout: papers P1 (AI) and P9 (DB) share author A7, and the walk P1–A7–P9 is one face.

| Row | One change from `full` (on P1–A7–P9) | Config |
|---|---|---|
| `full` | — | tuned |
| `no_path_cells` | `β_f = 0`: no faces, so the model is an edge sheaf. P1 meets A7 only through its own fact | (a)+(b) |
| S2 | Every fact map is `I`, so every transport is `I`. Faces, `g`, `a_e`, `β_f`, `τ`, solver and head are unchanged | (a)+(b) |
| S0 | `no_path_cells` with maps = I | (a) |
| `endpoints` | `g` reads P1, P9 and the relations only; A7 is invisible to the face weight. Hidden width is matched to `full`'s `g` within 5 % | (a)+(b) |
| `instance_features` | `β_f = 0` in the solve; `Σ_f β_f mean_v T_{v,f} h_v` is concatenated to the head input (the MAGNN-type instance encoder) | (a)+(b) |
| Literature meta-paths | Faces are drawn only along HGB's published meta-paths (`path_source: schemas`; spec §6 list and walk budget), instead of by random walks | (a) |

**Code status:** every row exists in `src/` (v3b `894fd74`). Out-of-scope arms (`closure_only`, `hub_A`, `covering_only`, `schema`, `uniform`, `direction_blind_maps`, `free_cell_maps`, `ego_head`, `h0_only_readout`) stay in the frozen code and are not run.

---

## 3. Claims (fixed before any run)

| Claim | Comparison | Holds if | If not |
|---|---|---|---|
| **C2 (thesis H; kill rule)** | `full` vs `endpoints` (decided on (b)) | Passes on ≥ 3 of 4 | Kill rule: no claim that instance-conditioned weighting adds information beyond endpoints; narrow the paper (spec §7 option B) |
| **C2-cells** | `full` vs `no_path_cells` (b) | Passes on ≥ 3 of 4 | Faces add nothing over the edge sheaf at tuned settings. **Warning signal:** stop and discuss |
| **C14** | `full` vs `instance_features` (b) | Passes on ≥ 3 of 4 | The cells act as instance encoders; the contribution is the geometry and the theory |
| **C15** | `full` vs S2 (b) | Passes on **both** IMDB and Freebase. A tie on ACM and DBLP is allowed | The paper drops the claim that its heterophily mechanism comes from the transports (Thm 4 stays a capacity result). **Warning signal** |
| 2×2 (reported) | (1) `full − no_path_cells`; (2) S2 − S0; (3) `full` − S2; (4) `no_path_cells` − S0, all on (a) | — | If (1) > (2), faces and sheaf maps are complementary; reported, not claimed |
| Literature row (reported) | `full` vs literature meta-paths (a) | — | If it beats `full` clearly, the paper says the expert-chosen schemas still add information, and random walks are recommended when no meta-path list exists |

---

## 4. Analyses

### 4.1 Schema read-out (Q3)
Computed from the main-table runs (seeds 0–4) and `full` (a) (seeds 1–10). No training.

**Quantities:**
- `c_σ`: coverage, the share of the face mass in sub-path schema σ with every `w_f = 1`;
- `I_σ`: the learned share;
- `ρ_σ = I_σ / c_σ`.

**Set:** length-2 schemas with mean coverage ≥ 1 %, at most the ten largest by coverage. Fewer than 4 gives "n/a".

**Stability:**
- the mean pairwise Kendall τ-b of the `ρ_σ` rankings over seeds 1–10 (45 pairs);
- a permutation null (schema labels shuffled within each seed, 10,000 draws), Holm m = 4;
- stable = τ-b ≥ 0.5 and adjusted p < 0.05;
- the τ-b of the `c_σ` rankings is reported beside it.

**Literature check:**
- the number of seeds (of 10) with `ρ_σ > 1` for each HGB published meta-path;
- the paper says "the learned weights favour σ" only at 10 of 10.

**Heterophily link:** per dataset and seed, Spearman between `ρ_σ` and the label homophily of σ's induced target-type graph (spec §1 reference code). Positive means the weights favour homophilous schemas; near zero means they do not follow homophily.

**Reading:** descriptive. If the rankings are not stable, the read-out is reported without interpretation. S2′ is not used: its card 4 result was on the 1.24 model and it is not a paper claim.

### 4.2 Perturbation study (Q4)
**Graphs:** ACM (h = 0.262 now) and DBLP (h = 0.128). IMDB is not used. Heterophily is h = 1 − MLH ∈ [0, 1] over the spec §1 `M₂` set, computed with the reference code. ℋ² is reported beside it at every level. (Amended 2026-11-19 by the author: h, not ℋ², sets the levels because it has fixed bounds; the earlier "ACM ≈ 40 %" was the Hetero²Net three-path set.)

**Targets:** h ∈ {0.20, 0.30, 0.40, 0.50, 0.60}, each within ±0.01, plus the original graph as a reference point.

**Ceiling (DERIVED, checked against card 7 job 17814):** an intermediate node with n target neighbours makes a clique in the meta-path graph, and at most (1 − 1/C)·n/(n−1) of its pairs can be different-class. For large n this is 1 − 1/C: ≈ 0.667 for ACM (C = 3) and 0.75 for DBLP (C = 4). Edge addition reached 0.638 and 0.748. Higher levels would need new intermediate nodes with at most C neighbours, i.e. fabricated nodes, which are not used. Higher heterophily is covered by the real heterophilic datasets of the main table (IMDB, Freebase, H2GB). The figure draws 1 − 1/C as a dashed line.

**Generator rules (card 7 V3, accepted):** additions are split across the `M₂` relations in proportion to their size. For an author or term fact, a paper of another class is drawn uniformly, and the new fact links the labelled paper to one of that paper's neighbours. Unreached levels are written but not run. Every run logs the fingerprint of the edge list it loaded, and the manifest checks it against the generator's fingerprint.

| Direction | Rule |
|---|---|
| Raise h | **Add** facts of the `M₂` relations already in the graph, with typed endpoints, that create different-class meta-path pairs. The partner of the labelled endpoint is drawn uniformly over all other classes, so an added fact says only "not my class". No duplicates and no self-loops |
| Lower h | **Remove** facts that create different-class pairs, at most 50 % of a relation's facts. Same-class edges are never added: that would build test labels into the graph |
| Control | Every targeted graph has a random graph with the same number of facts added or removed per relation, drawn uniformly. The effect of heterophily is the targeted result minus the random result |

**Reachability:** a level not reachable under the 50 % cap is reported as unreached, with the closest value. The cap is never raised after results.

**Methods:**
- H2S-NN `full` and `no_path_cells`;
- GCN, HAN, SeHGNN, Hetero²Net, HetSheaf and CSNN (added 2026-11-19 as the homogeneous heterophily model, so the categories are homophily-assuming, meta-path, heterogeneous heterophilic, heterogeneous sheaf and homogeneous sheaf), each with its configuration tuned on the original graph, or its published configuration if not tuned (listed); never retuned. GCN and HAN use HGB's published run commands (`vendor/HGB/NC/benchmark/methods/GNN/README.md`, HAN `main.py` defaults); SeHGNN and Hetero²Net use their published configurations; HetSheaf has only sweep files, so it waits for its ACM/DBLP tuning; CSNN uses a published ACM/DBLP configuration if one exists, else it waits for its ACM/DBLP tuning;
- seeds 1–5;
- in addition, H2S-NN S2 on the targeted graphs only, for R3's perturbation axis.

**Labels:** all labels are used to build the graphs. The graphs are semi-synthetic, and curves are compared across methods only, never with the main table.

**Expected (fixed now):**
- H2S-NN loses less Macro-F1 than GCN, HAN and SeHGNN as h rises; the comparisons with CSNN and HetSheaf are reported (no expectation fixed);
- `full − no_path_cells` grows with h.

If either expectation fails, it is reported as it is, and the paper makes no robustness claim.

### 4.3 Face effect (A1)
Uses `full` and `no_path_cells`, configuration (a), the 4 datasets, seeds 1–10, and card 5 hooks (b) and (d).

- **η_v** (amended 2026-11-20 by the author, before any run; reason: on ACM the union of the `M₂` neighbours is dominated by P–T–P, a median of 3,024 neighbours per paper, so η_v took 7 distinct values; card 7 report, job 17862): the mean, over the `M₂` paths on which v has at least one labelled target-type neighbour, of the share of those neighbours (distinct, u ≠ v) whose label differs from v's — the node-level counterpart of h = 1 − MLH, which also averages over `M₂`. IMDB: per path, the mean over genres of the binary disagreement, then the mean over paths. Nodes with no labelled neighbour on any path are excluded and counted.
- **η groups** (amended 2026-11-20; replaces quartiles, which leave empty bins under ties, e.g. 1,280 of 1,820 DBLP test authors have η_v = 0): G0 = {η_v = 0}; G1–G3 = tertiles of η_v over the test targets with η_v > 0, value bins (−∞, t1], (t1, t2], (t2, ∞). The same rule on every dataset; an empty or small G0 is reported with its count, never merged. Every "η_v quartile" below means an η group.
- **F1, flips:** per test node, wrong→right and right→wrong from `no_path_cells` to `full`, per η_v quartile (mean and range over seeds).
- **F2, margin** (single-label datasets): margin_v = distance from x*_v to the nearest other-class centroid − distance to its own-class centroid, with centroids from the test x* of the same model. Reported for both models by η_v quartile.
- **F3, knockout:** on trained `full`, seeds 1–3, every cell through test node v gets `β_f = 0` at evaluation only. Report the share of correct predictions that turn wrong, and the margin change, by η_v quartile. At most 2,000 test nodes per (dataset, seed), drawn with a fixed seed.
- **Expected:** faces fix more nodes than they break, more so at high η_v; the margin is larger with faces, more so at high η_v; knockout turns a non-zero share of correct predictions wrong.
- **Case study (fixed rule):**
  - candidates: ACM seed 1 test nodes with a different-class neighbour (fact or `M₂` meta-path) whose F3 knockout turns right into wrong;
  - pick: the largest margin drop; on a tie, the smaller id;
  - the figure shows the node, the different-class neighbour, the 3 cells through the node with the largest `w_f` (ids, types, schema), a 2-D PCA (fitted on all test x* of `full`) of h₀, x* with the knockout and x* of `full` with class centroids, and the true-class probability with and without the knockout.

### 4.4 Two-regime analysis (A2)
**Theory:** Prop. 12 (spec 1.27, a corollary of Prop. 11 and Thm 4).
- With every transport equal to `I`, the solve is the resolvent `(I + τŜ_I)⁻¹h₀` of the weighted node graph: a low-pass smoothing of the personalised-PageRank form (homophily regime).
- For any labelling, there are TD-1 maps whose class-separating section is a fixed point, separable by a linear classifier on `x*` (heterophily regime).
- Both are points of one parameter family, and training chooses per fact.

**Measurements** (no training; card 5 hooks (a), (c), (d)):

| Part | Measure | Expected |
|---|---|---|
| **R1** transports | For cells whose two endpoints are labelled targets, the endpoint transport composed from the fact transports along the cell; ‖T − I‖_F split by same-class vs different-class endpoint pairs (medians, AUROC). On ACM P–P and Freebase B–B, also the fact-level ‖Q_e − I‖_F. The learned τ per run | Same-class pairs closer to `I` than different-class pairs (AUROC > 0.5) |
| **R2** energy vs performance | Node energy `e_v` (hook (a)): AUROC of `e_v` separating correct from wrong predictions; Spearman(`e_v`, η_v) | Lower energy for correct predictions; heterophilic nodes can still have low energy |
| **R3** energy vs representation | `full` and S2 (a): normalised energy Ẽ; a linear probe (logistic regression, `C = 1`, trained on training targets, scored on test targets) on x* and on h₀; the within/between-class distance ratio on x* and h₀ | At low η both models separate classes better on x* than on h₀. At high η, `full` keeps the separation while S2 loses it |

**Two axes:**
1. Nodes, by η group (G0–G3), on the 4 datasets.
2. Graphs, by perturbation level (§4.2), with `full` and S2 on the targeted graphs.

### 4.5 Scalability (Q5)
- **Complexity:** big-O of the WISP draw, the map build, one PCG iteration and memory, in |E|, the face incidences, d and m = 4. Compared with meta-path enumeration (HAN, SeHGNN) and d² per edge (NSD). Written in the paper; no run.
- **Timing:** on ogbn-mag (8,192 training targets per epoch) and IMDB, for every main-table method at its tuned configuration (the card 2 central point if not tuned, flagged).
- **Cost-vs-accuracy figure (author, 2026-11-19: kept):** on ogbn-mag, test Macro-F1 (main table) vs time per epoch and vs peak memory (timing runs), one point per main-table method at its tuned configuration. No extra training: it combines the main-table results with the timing runs.
  - Measures: time per epoch (median of epochs 2–4 after one warm-up), peak GPU memory, parameter count.
  - Separate timing runs on one H100.

---

## 5. Run budget (estimates; all on TSUBAME after `full` is tuned)

| Block | Runs |
|---|---|
| Ablation (a): 6 rows × 4 datasets × 10 seeds + `full` seeds 5–10 × 4 | ≈ 264 small |
| Ablation (b): 4 arms × 4 datasets × 100 trials, then × 10 seeds | 1,600 trials + 160 small |
| Perturbation: 8 methods × 2 datasets × 5 levels × 2 kinds × 5 seeds + S2 on the targeted graphs (50) | ≈ 850 small |
| §4.1, §4.3, §4.4 | read from checkpoints (F3 is evaluation only) |
| Timing | a few hours on one H100 |

If time runs short, the first cut is configuration (b): (a) only, stated as a limitation. The author decides before the runs.

## 6. Removed from v1 (author, 2026-11-19)
- **Arms:** the C2 ladder (`schema`, `uniform`), C10 `closure_only`, C11 `hub_A`, C12 `covering_only`, C16 `direction_blind_maps`, C17 `free_cell_maps`, C18 `ego_head`, `h₀`-only.
- **Analyses:**
  - C6 on the 12 NC datasets, with its consequence "remove *heterophilic* from the title";
  - N1–N4, N2b, N2c;
  - U3;
  - the C5 scaling curve and the mini-batch vs full-batch gap;
  - the parameter table, except where §1's capacity reading needs a count.
