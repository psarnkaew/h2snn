# H2S-NN — Ablation and analysis pre-registration (DRAFT)

**Status:** DRAFT for the author (2026-11-16, ablation design session, branch `design-ablation`). It becomes binding when the author approves it and its content enters spec v1.23; it is frozen with the spec and code hash before the first run it governs. Nothing here changes the spec, a card or code by itself.
**Authority:** spec v1.22 §1, §5, §6; `docs/experiment_settings_decisions.md` rows of 2026-11-15/16 "Ablation statistics and seeds (B-1)" through "Remaining round-2 analyses (B2)" (author decisions); `docs/known_answer_preregistration_draft.md`; `docs/open_items_ablation_legacy_2026-11-15.md`; reviews of the WSDM 2027 submission (`docs/Reviewer_review.pdf`, R1–R4).
**Tags:** VERIFIED-CODE / VERIFIED-RUN / VERIFIED-SOURCE, DERIVED, UNVERIFIED, PROPOSAL (= not yet decided by the author).
**Scope:** the comparisons and analyses that explain *why* the model works (the "discussion" part). The main tables (C7: every method, seeds 0–4, Welch) are governed by spec §6 and are unchanged.

---

## 0. Research question and what must be shown

**Plain question.** Heterogeneous graphs often link nodes of different classes. Existing methods first cut the graph into one view per meta-path schema and then fuse the views, so (a) heterophily is judged on derived graphs, and (b) two node pairs that follow the same schema are told apart only by their endpoints; the nodes in between are ignored. H2S-NN keeps each sampled path instance as a cell on the original graph and solves one energy problem over it.
**Example.** Papers P1 (class AI) and P9 (class DB) share author A7. Does it matter who A7 is, and can P1 and P9 stay different although they are linked?

**Goal (paper).** (i) A single model on the original graph handles heterophily with path instances and without per-schema branches; (ii) path instances, used as structure, carry information that schema-level encoders and instance-level features do not.

**Goal → evidence map** (every sentence of the goal has a test; § = section of this document):

| Sentence of the goal | Evidence | § |
|---|---|---|
| Path instances carry information beyond endpoints | C2 (kill rule), C2 ladder pairs, S3, N3 | 3, 4, 5 |
| A cell is not an endpoint shortcut ("no closing edge") | C10 (`closure_only`) | 3 |
| Structure beats instance features (MAGNN-type) | C14 (`instance_features`) | 3 |
| Cells couple through edges, not as hyperedges | C11 (`hub_A`), C12 (`covering_only`, S1′-B) | 3, 5 |
| Learned transports keep different-class neighbours apart | C15 (S2), N2, N2b | 3, 4 |
| Direction-aware maps | C16 (`direction_blind_maps`) | 3 |
| A path acts through the composition of its edges' maps | C17 (`free_cell_maps`) | 3 |
| Relational context in the maps | C9 (`geometry_context: false`) | 3 |
| Handles heterophily on the original graph | C6 (within-dataset), controlled sweep, N1 | 3, 4 |
| No per-schema branches; schemas still readable | Fixed-literature-schema row, C2b, N4, N4b | 3, 4 |
| Scales | C5, C13, cost figure, per-dataset table, U3 | 4 |
| Propagation helps at all (sanity) | `h₀`-only | 3 |

Removed: **C3 (relation order)**, author 2026-11-15 (B-5): order is not a difference from MAGNN (withdrawn 2026-11-10), most literature meta-paths are palindromes, and the diagonal-map arm confounds order with rotation capacity. Spec v1.23 deletes "and the order of their relations" from thesis H.

---

## 1. Common settings (every comparison below unless stated)

| Item | Setting | Source |
|---|---|---|
| Ablation datasets | HGB ACM, DBLP, IMDB, Freebase (ℋ² 0.399, 0.173, 0.737 per genre, 0.995 / labelled-only 0.882; Part J, VERIFIED-RUN). Two homophilic, two heterophilic. Freebase full batch if it fits at the largest configuration, else mini-batch (spec §6 rule) | B-3 |
| Task, metric | Node classification; test Macro-F1 (IMDB: mean per-genre F1, spec §6); checkpoints and configurations selected on validation Macro-F1 | spec §6 |
| Seeds | 1–10 (seed 0, the tuning seed, excluded). `full`'s main-table runs of seeds 1–4 are reused | B-1 |
| Pairing | By seed: the same seed gives the same WISP draws and batch order in every arm (`src/data/wisp_sampler.py:394, 467`, VERIFIED-CODE); initial weights differ where parameter sets differ | B-1 |
| Test | Wilcoxon signed-rank on the 10 paired differences, two-sided (`scipy.stats.wilcoxon`); minimum attainable p at n = 10 is 0.00195 (VERIFIED-RUN) | B-1 |
| Multiplicity | Holm per claim over its datasets (m = 4 on the ablation family; m = 12 for C6) | B-1, B-3 |
| "Passes on a dataset" | Holm-adjusted p < 0.05 and median paired difference > 0 (in favour of `full`) | B-1 |
| Default claim rule | Passes on ≥ 3 of 4 datasets (stricter than v1.22's "≥ 2 of 3"; never loosened). C15 and C16 use their own rules (§3) | B-3 |
| Reported per comparison | Mean ± sample std per arm, raw and adjusted p, mean paired difference with a bootstrap 95 % CI (10,000 resamples of the 10 differences, percentile) | B-1 |
| Configuration (a) | `full`'s tuned configuration for that dataset with the arm's one change | B-2 |
| Configuration (b) | The arm tuned on its own: plain Bayesian search, 100 trials, seed 0, validation metric, space restricted to keys that act in that arm (e.g. no `ℓ_max`, `r` for `no_path_cells`); then seeds 1–10. Arms: `endpoints`, `closure_only`, `instance_features`, `no_path_cells`, S2 | B-2, B-4 |
| Which configuration decides | Arms with (b): the claim is decided on (b); (a) is reported beside; if they disagree both are reported and the paper's statement follows (b). Every other arm: (a) only, stated as a limitation | B-2 |
| Checkpoints | Every ablation run keeps its best checkpoint and resolved configuration; the evaluation draw is rebuilt from the base seed (spec §6 requirement) | N1–N4 |
| Timing | Every time or memory number in §4.6 comes from separate timing runs on **one GPU type** (stated); never from the main or ablation runs, which run on mixed nodes (2080 Ti, 3090, A6000, TITAN RTX) | B-12 |
| Not in this document | LP and GC have no ablation (stated limitation); baselines never appear in the ablation table | B-3 |

---

## 2. Arms (each is `full` with one change; same draw, anchor, head and budget)

| Arm | One change from `full` (on P1–A7–P9) | Status in code (v3b `d6f3b88`) | Config |
|---|---|---|---|
| `full` | — | exists | tuned |
| `endpoints` | `g` reads `v₀`, `v_ℓ` and the actions only (A7 invisible to the weight); hidden width matched to `full`'s `g` within 5 % (`endpoint_hidden_width`, `h2snn_model.py:350`) | exists (`path_weight_input`) | (a)+(b) |
| `schema` | `g` reads the actions only (every PAP instance has the same weight) | exists | (a) |
| `uniform` | `w_f = exp(b)` for every cell | exists | (a) |
| `no_path_cells` | `β_f = 0` (no cells) | exists | (a)+(b) |
| `closure_only` | each cell replaced by one holonomy edge `v₀`–`v_ℓ` with weight `β_f` (`HolonomyEdgeTerm`) | exists | (a)+(b) |
| `instance_features` | `β_f = 0` in the solve; `Σ_f β_f mean_v T_{v,f} h_v` concatenated to the anchor's head input | exists | (a)+(b) |
| `hub_A` | fact–cell term dropped, fact states eliminated (cells couple to nodes only) | exists | (a) |
| `covering_only` | node–cell term dropped | exists | (a) |
| **S2** (maps = I) | every fact map `F = I`, hence every transport `I`; cells, `g`, `a_e`, `β_f`, `τ`, solver, head unchanged; geometry network unused | **new** (identity map operator under `anchored`; own card) | (a)+(b) |
| **S0** | `no_path_cells` with maps = I | **new** (same operator) | (a) |
| `direction_blind_maps` | the two roles of a relation share one frame and a symmetrised scale (`sheaf_maps.py:128`) | exists | (a) |
| `free_cell_maps` | node→cell maps free per incidence instead of composed from the edge maps (Thm 3 control) | exists | (a) |
| geometry context off | maps and fact weights read `h₀` instead of `[h₀, c]` (`geometry_context: false`) | exists | (a) |
| `h₀`-only | no propagation; head reads `h₀` (`h0_only_readout`) | exists | (a) |
| Fixed-literature-schema row | path source = HGB's published meta-path list (`path_source: schemas`), spec §6 | exists | spec §6 (seeds 0–4, Welch; reported, not a claim) |
| Appendix pair | `g` reads `[h₀, c]` (`path_weight_context: true`) in `full` and `endpoints` (spec §6) | exists | **PROPOSAL:** (a), seeds 1–10, reported without a claim |

The parameter table (§4.8) lists every arm's count.

---

## 3. Claims: evidence, criterion and consequence (fixed before any run)

Notation: "better on d" = passes on dataset d in the sense of §1.

| Claim | Comparison | Holds if | If not |
|---|---|---|---|
| **C2 (thesis H; kill rule)** | `full` vs `endpoints` (decided on (b)) | better on ≥ 3 of 4 | Kill rule fires: no claim that instance-conditioned weighting adds information beyond transports and states; narrow the paper (spec §7 option B). Read with S3 (§5) |
| C2 ladder pairs (reported) | `schema` vs `uniform`; `endpoints` vs `schema` (a), Holm m = 4 per pair | — (not a kill rule) | Readings: schema-level weights carry information (HAN/SeHGNN-type); endpoints add to the schema |
| C2 (cells) | `full` vs `no_path_cells` (b) | better on ≥ 3 of 4 | Cells add nothing over the edge sheaf at tuned settings; warning signal (stop and discuss) |
| **C2b schema read-out** | Stability of `ρ_σ = I_σ / c_σ` rankings across seeds (§4.4) and S2′ | stable on ≥ 3 of 4 **and** S2′ passes | Read-out reported without interpretation; no interpretability claim |
| **C6 heterophily** | Within-dataset slope of per-node gain (`full` − `endpoints`) on per-node heterophily, 12 NC datasets (§4.1) | positive and significant on ≥ 6 of 12, significantly negative on none | Remove "heterophilic" from the title (spec) |
| C9 relational context | `full` vs geometry context off (a) | better on ≥ 3 of 4 | Drop `c` (spec) |
| **C10 face ≠ closure** | `full` vs `closure_only` (b) | better on ≥ 3 of 4 | The cell equals its holonomy edge in practice; the W3 fix is structural only (spec) |
| **C11 fact–cell coupling** | `full` vs `hub_A` (a) | better on ≥ 3 of 4 | The coupling is kept for Thm 3, not as a source of gains (spec) |
| **C12 node–cell term** | `full` vs `covering_only` (a) and S1′-B (§5) | as spec: falsified if `covering_only` is not worse **and** S1′-B passes without the term | Revert to covering pairs; report the ceiling as harmless (spec) |
| **C14 diffusion vs instance features** | `full` vs `instance_features` (b) | better on ≥ 3 of 4 | Cells act as instance encoders; the contribution is the geometry and the theory (spec) |
| **C15 learned transports (new)** | `full` vs S2 (b); 2×2 with `no_path_cells` and S0 reported | better on **both** IMDB and Freebase (heterophilic); a tie on ACM/DBLP is allowed | The paper drops the claim that its heterophily mechanism comes from the transports (Thm 4 stays a capacity result); **warning signal** |
| **C16 direction-aware maps (new)** | `full` vs `direction_blind_maps` (a). Same-type relations exist only on ACM (paper–cite–paper) and Freebase (8 of 36 types, e.g. book–book 202,674 facts); none on DBLP, IMDB (`link.dat`, VERIFIED-RUN) | better on **both** ACM and Freebase | "Direction-aware" is not a selling point; direction is mostly carried by node types |
| **C17 composition (new)** | `full` vs `free_cell_maps` (a), two-sided | three fixed readings: `full` better on ≥ 3 of 4 → composition helps; no difference → "no difference found" (never "equal"), kept for Thm 3; `free_cell_maps` better on ≥ 3 of 4 → composition has a cost, stated, kept for theory | no kill rule |
| Sanity | `full` vs `h₀`-only (a) | better on ≥ 3 of 4 | Warning signal (stop and discuss). Freebase has no features: `h₀`-only near chance there is also a leak control |
| C5, C13 | §4.6 | §4.6 | spec |

**Capacity reading (all claims, B-13).** If `full` beats an arm with fewer parameters and that arm's tuned `d` is 64 (top of the space), the paper states for that claim that capacity is not excluded as an explanation.

---

## 4. Analyses

### 4.1 C6 within datasets (N1)
- **Per-node heterophily** `η_v` = share of `v`'s target-type neighbours through the meta-paths of `M₂` (spec §1 list per dataset) with a label different from `v`'s (IMDB: mean over genres of the binary disagreement); nodes without such neighbours are excluded from the slope and counted.
- **Class centring:** `η̃_v = η_v − mean{η_u : y_u = y_v, u in the test split}` (removes the class-imbalance route, e.g. IEEE-CIS fraud nodes are heterophilic almost by class).
- **Per-node gain** `γ_v` = 1[`full` correct on v] − 1[`endpoints` correct on v] (IMDB: mean over genres).
- **Statistic:** per seed, the least-squares slope of `γ` on `η̃` over test targets; Wilcoxon over seeds 1–10, Holm m = 12. Runs: `full` and `endpoints` at configuration (a) on all 12 NC datasets.
- **Rule:** §3 (≥ 6 of 12 positive and significant, none significantly negative).
- Reported beside (supporting, not deciding): gain per quartile of `η̃` with bootstrap CIs; `full − no_path_cells` per quartile on the four ablation datasets; cross-dataset Spearman between ℋ² and mean gain over 12 datasets.
- Test labels are used only in this post-hoc analysis; nothing feeds back into training.
- **Known risk (stated now):** datasets whose nodes are almost all heterophilic have a narrow `η̃` range, so a true effect can miss significance.

### 4.2 Controlled heterophily sweep (N5)
- **Graphs:** ACM and DBLP (single-label, fully labelled; IMDB multi-label and Freebase 80 % unlabelled excluded).
- **Rewiring:** degree-preserving swaps of fact endpoints within one relation (P1–A7, P2–A9 → P1–A9, P2–A7), drawn uniformly among swaps that move the meta-path homophily toward the target, until the target is reached; every node's degree per relation and every fact count is unchanged. Relations swapped: those of the dataset's `M₂` meta-paths (ACM: paper–author, paper–subject, paper–term; DBLP: author–paper), proportionally to their share of `M₂` instances. **PROPOSAL** (card detail): one fixed rewiring seed per level; the swap rule is unit-tested (degrees, fact counts, achieved homophily).
- **Levels:** MLH (meta-path label homophily, Hetero²Net's definition, spec §1) ∈ {20, 35, 50, 65, 80} % (current DBLP 87 %, ACM 60 %, VERIFIED-RUN); ℋ² measured and reported at each level with the spec §1 reference code.
- **Methods:** H2S-NN `full`, `endpoints`; GCN, HAN, SeHGNN, Hetero²Net, HetSheaf; each with its configuration tuned on the original graph (no retuning); seeds 1–5.
- **Label use:** all labels are used to rewire (as cSBM-type sweeps do); curves are compared across methods only, never with the main table (stated).
- **Readings fixed now (descriptive):** Macro-F1 vs MLH per method — expected: H2S-NN loses less than GCN, HAN and SeHGNN at low homophily; `full − endpoints` vs MLH — expected to grow as homophily falls; range (max − min) per method. If not as expected, reported as is, and the paper makes no robustness claim. The sweep supports C6; it does not decide it.
- **Prior work (one search, 2026-11-16; UNVERIFIED completeness):** homophily sweeps are common for homogeneous GNNs (synthetic graphs over many edge-homophily levels) and hypergraphs; for heterogeneous graphs the closest found is HDHGR (WWW'23), which rewires toward homophily as a method, not as an evaluation sweep. A 2024 ICML paper notes that equal heterophily ratios can hide different patterns; our swaps are uniform among moving swaps, which is stated as the pattern used.

### 4.3 Mechanism analyses on the ablation checkpoints
- **N2 transport separation.** For path-connected target pairs (u, v) in a cell, distance after transport `‖T_{u,f} x*_u − T_{v,f} x*_v‖` and plain distance `‖x*_u − x*_v‖`, split by same-class vs different-class pairs; `full` vs S2. Linear probe (logistic regression, fixed `C = 1`, training targets → test targets) on `x*` only, and the within/between-class distance ratio on `x*`. Expected: in `full` different-class pairs agree after transport while staying apart before it; in S2 they are pulled together. Descriptive.
- **N2b non-collapse under fixed `τ`.** `τ` fixed ∈ {1, 10, 100} (retrained), `full` vs S2, DBLP and IMDB, seeds 1–3; probe accuracy and distance ratio on `x*`; t-SNE illustrative only. Protocol exceptions (stated): the `τ > 50` stop is lifted and `T_max` = 500 for these runs; `τ` = 1000 is not run. Expected: S2's probe accuracy falls as `τ` grows, `full`'s stays or falls less; otherwise reported as is and no non-collapse claim.
- **N3 intermediate-node intervention — CORRECTED, needs the author's approval (see note).** On each trained checkpoint, the `h₀` of every **intermediate** cell node is replaced, **in `g`'s input only**, by the `h₀` of a random node of the same type (fixed seed); the graph, the maps, the edge terms and the endpoints are untouched. Measured: mean `|Δ log w_f|` over cells and the share of test targets whose predicted class changes. Exact control: `endpoints` gives 0 (its `g` never reads intermediates; VERIFIED-CODE `EndpointEncoder`, `h2snn_model.py:617`). Rule fixed now: in `full`, mean `|Δ log w_f|` < 0.1 on ≥ 3 of 4 datasets is a warning signal (stop and discuss: the trained weights do not use intermediates). *Note:* the version approved on 2026-11-15 swapped intermediate features in the whole model with `closure_only` as a zero control; that control is wrong — A7 is a one-hop neighbour of P1, so its features reach P1 through the edge terms and the fact maps in every arm, `closure_only` included (its holonomy map is composed of fact maps that read A7). The corrected version isolates the weight channel; the value of intermediates through the solve is what C2 and C10 measure.
- **N4 learned weights by instance type.** Distribution of `log w_f` by (endpoint labels equal / different) and by the intermediate's class where labelled; descriptive.
- **N4b schema weights vs schema homophily.** Per dataset, Spearman between `ρ_σ` and the homophily of σ's induced target-type graph, seeds 1–10; in the sweep, `ρ_σ` of the rewired schemas per level. Two fixed readings: positive → the weights favour homophilous schemas; near zero → the weights do not follow homophily (the transports handle heterophilous schemas). Descriptive.
- **N6 error analysis and case study.** Per-class and per-`η̃`-quartile errors of `full` vs `endpoints`; one real-data case (R2 comment 1): a test target that `endpoints` misclassifies and `full` classifies correctly, with its top cells by `w_f` and their intermediates. The case is chosen by a fixed rule (the target with the largest `w_f` spread among cells that share both endpoints, on ACM, seed 1), not by hand.

### 4.4 C2b schema read-out (B-8)
- Set: length-2 sub-path schemas with mean coverage ≥ 1 % of the cell mass, at most the ten largest by coverage (fixed by coverage); fewer than 4 → "n/a".
- Statistic: mean pairwise Kendall τ-b of the `ρ_σ` rankings over seeds 1–10 (45 pairs) of `full`; permutation null (schema labels shuffled within each seed, 10,000 draws), Holm m = 4. Stable: mean τ-b ≥ 0.5 and adjusted p < 0.05.
- Reported beside: τ of `I_σ` and of `c_σ` (shows how much stability is coverage); the literature-path reading of spec §6 becomes `ρ_σ > 1` on 10 of 10 seeds.
- Why `ρ_σ` (DERIVED): `I_σ` follows WISP's coverage, so its ranking is stable even when `g` is constant.

### 4.5 U3 sensitivity (B-11)
- One-at-a-time around `full`'s tuned configuration: `ℓ_max` ∈ {2, …, 6} at the tuned `r`; `r` ∈ {8, 16, 32, 64, 128} at the tuned `ℓ_max`; four ablation datasets, seeds 1–5; `ℓ_max` 6 and `r` 128 are analysis-only values. No E11 calibration.
- Recorded per point: Macro-F1, walks per length, time per epoch, peak memory (timing on the one GPU type), OOM as a point.
- Readings: a drop > 2 points at `ℓ_max` 6 below the best point on ≥ 3 of 4 datasets is reported as "drops at large `ℓ_max` at fixed `r`", never "long paths are harmful"; `r` expected to rise then saturate. No kill rule.

### 4.6 C5 / C13 scalability and cost (B-12)
1. **Cost-vs-accuracy figure** on ogbn-mag: test Macro-F1 vs time per epoch (8,192 targets) and vs peak memory, every main-table method at its tuned configuration, a few timed epochs. ACM vs v1 (9,260 MB, 3.011 s/epoch) is a C13 line.
2. **Scaling curve:** ogbn-mag subgraphs from a random fraction of papers ∈ {1/64, 1/32, …, 1} plus their neighbours (fixed seed); fixed configuration (`d` 32, `ℓ_max` 3, `r` 16); full batch and mini-batch; a few timed epochs. Criteria: full-batch memory or time vs incidences, log–log slope > 1.15 → super-linear; mini-batch time per step vs graph size slope > 0.15 → reported; PCG iterations vs `N` slope > 0.1 → growing with `N`; OOM recorded as the full-batch limit. C5 falsified → spec consequence (state the limit; fixed-`T` truncation).
3. **Per-dataset table** (12 NC, from the main runs): walks, cells, incidences, `|E_shared|`, distinct schemas drawn, WISP draw time, time per epoch, peak memory, parameters, PCG iterations. The paper states that v3-B has no schema enumeration and no closure edges (R3/R4 asked for v1's AMPSS counts).
4. **Mini-batch vs full-batch gap** (spec 1.22 analysis flag), ACM and DBLP, seeds 1–5: a Macro-F1 gap > 1 point on either is stated, and the H2GB results carry that note.

### 4.7 Not done
Graph clustering (no claim needs it). C3. Ablations on LP and GC. A v3-B vs R-GCN untuned comparison (G0b; gate (i) closed by author decision, B-10).

### 4.8 Parameter table (B-13)
Every arm on the four ablation datasets at `full`'s configuration: total and the part removed or added (`g`, geometry network, head input); share of `g` for `uniform` and `schema`. "Within 5 %" applies to `endpoints` only. No padding.

---

## 5. Known-answer tasks (B-9; details in `known_answer_preregistration_draft.md`, to be updated)
- **S1′-B** (C12, v3-B capacity): as approved 2026-11-11. A `full` + oracle miss at log 512 is a warning signal.
- **S3** (learnability of `g`): length-4 chain P–A–B–C–Q, relevance iff `t_P ⊕ t_B ⊕ t_Q = 0`, 64 cells per target; arms `full`, `endpoints`, `uniform`, `full` + oracle; fact-input probe audit.
- **S2′** (C2b): option (a) of 2026-11-11.
- **Changes from the draft:** AdamW without scheduler or clipping (protocol settings), seeds 1–3; reachability recomputed (scalar group at constant 1e-2 over 100 epochs moves ≈ 1.0).
- **C2 × S3 reading table:**

| | S3 passes | S3 fails |
|---|---|---|
| C2 kill rule not fired | H supported through the weight channel on real and synthetic data | The real-data gain is claimed; the claim that `g` learns relevance no fact can see is not; S3 reported as a limit |
| C2 kill rule fired | The channel can learn, but the data carry no instance-level signal beyond endpoints (spec) | The channel did not learn selection; H untested through the weights (spec) |

---

## 6. Order of work and prerequisites
1. M3b (data, other session).
2. **Proposal variance** on ACM before any tuning: seed 1, central configuration (`d` 32, `ℓ_max` 3, `r` 16, lr 1e-3, weight decay 1e-4, dropout 0.5), trained to early stopping; `adaptive` only if variance reduction ≥ 20 %; the result is fixed for every later run (B-10).
3. **Known-answer** S1′-B, S3, S2′ (cluster). A warning signal stops the plan.
4. New arms S2/S0 (identity map operator), analysis code (N1–N6, N2b, N4b, C2b, parameter table), sweep generator, subgraph sampler, timing harness: cards written by the other design session after spec v1.23.
5. Spec and code frozen (hash recorded).
6. Main tuning (spec §6) → `full`'s configurations.
7. Ablation runs (a); tuning (b) for the five arms; C6 runs on the H2GB datasets; sweep; U3; N2b.
8. Timing runs (one GPU type); analyses; report.

**Run counts (estimates):**

| Block | Runs |
|---|---|
| Ablation (a): 15 arms × 4 datasets × 10 seeds, minus 16 reused `full` runs | ≈ 584 (small, full batch; Freebase possibly mini-batch) |
| Ablation (b): 5 arms × 4 datasets × 100 trials, + 5 × 4 × 10 seed runs | 2,000 trials + 200 runs |
| C6 on the 8 H2GB datasets: `endpoints` 10 seeds + `full` seeds 5–10 | 128 large mini-batch runs |
| Sweep: 7 methods × 2 datasets × 5 levels × 5 seeds | 350 small |
| U3: 4 datasets × 8 new points × 5 seeds | 160 |
| N2b: 2 arms × 3 `τ` × 2 datasets × 3 seeds | 36 |
| Known-answer | ≈ 36 small synthetic |
| Timing and scaling | hours on one GPU type |

Per-run time is UNVERIFIED until the first tuning trials; H2GB runs (500 epochs × 8,192 targets) dominate.

---

## 7. Spec v1.23 edits implied (to be drafted after approval)
§1 thesis H without "order"; §5 table: C2 rule "≥ 3 of 4" on four HGB datasets, C2b on `ρ_σ`, C3 removed, C6 within-dataset rule, C15–C17 added, `h₀`-only sanity; §5 note (1.14) replaced by the statistics of §1 here; §6 ablation bullet (datasets, seeds 1–10, configurations (a)/(b)), schema read-out (`ρ_σ`, seeds), known-answer settings, WISP gate (i) closed, proposal variance before tuning, U3 without E11, C5 criteria, timing on one GPU type; §6 v3-B arms list + S2/S0; §7 OPEN-5 closed for arms.

## 8. Open points for the author
1. **N3 correction** (§4.3): approve the weight-channel intervention with the `endpoints` zero control and the 0.1 threshold, or drop N3.
2. Appendix pair (`path_weight_context: true`): configuration (a), seeds 1–10, reported without a claim (PROPOSAL).
3. Sweep detail: which relations are swapped and in what proportion (§4.2 PROPOSAL).
4. N6 case-selection rule (§4.3).
