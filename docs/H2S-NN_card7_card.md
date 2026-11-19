# Task card 7 — Ablation and analysis code (plan v2): launchers, statistics, perturbation, read-outs, face effect, two-regime analysis, timing

**Owner (design):** model session. **Executor:** S6, worktree `/mnt/data/peeradon/h2snn-model`.
**Branch:** new branch `card7` from `v3b` `894fd74`. Then `git merge --no-ff` of `design-docs` at the SHA given in the prompt. Merge, never rebase.

## Scope rule (author, 2026-11-19): exactly the plan, nothing else
The ablation study and model analysis consist **only** of the items in `docs/ablation_plan_v2_2026-11-19.html` (author-approved; decisions log row of 2026-11-19 "Ablation plan v2 approved"). Implement, list and report only those.

**In scope:**
- **Ablation table, 7 rows:** `full`, `no_path_cells`, S2 (maps = I), S0 (`no_path_cells` + maps = I), `endpoints`, `instance_features`, and the literature meta-path row (`path_source: schemas`).
- **Q3 schema read-out:** coverage `c_σ`, learned share `I_σ`, `ρ_σ`, ranking stability, and `ρ_σ` vs schema homophily.
- **Q4 perturbation** to heterophily 20/40/60/80 %.
- **Face effect:** F1, F2, F3 and the rule-chosen case study.
- **Two-regime analysis:** R1, R2, R3.
- **Q5 timing** on ogbn-mag and IMDB.

**Out of scope:** every other arm or analysis of the old pre-registration:
- arms `closure_only`, `hub_A`, `covering_only`, `schema`, `uniform`, `direction_blind_maps`, `free_cell_maps`, `ego_head`, `h0_only_readout`;
- C6 on H2GB;
- N2, N2b, N2c, N3, N4;
- U3;
- the scaling curve;
- the mini-batch vs full-batch gap.

Do not delete these arms from `src/`, because the code is frozen. Remove them from every manifest, list and default of the scripts. If something in the plan needs one of them, stop and report.

**Binding (in this order):**
1. The plan above.
2. Spec v1.26 §6 (protocol: datasets, metrics, seeds, tuning, environment, published-number placeholders).
3. `docs/experiment_settings_decisions.md` rows of 2026-11-19.
4. Card 5 (`docs/H2S-NN_card5_card.md`) U2, for the hook API, once card 5 is on `v3b`.
5. `docs/working_preferences.md`.

**Mode:**
- Code in `scripts/` and tests only. No change in `src/`: the H2S-NN freeze covers `src/`, and card 5 owns the hooks.
- Small synthetic or HGB checks only, plus one smoke run per launcher on ACM at a tiny budget.
- No ablation, perturbation or timing run at full budget. These runs start on TSUBAME after `full` is tuned.

## Phases
- **Phase A (start now, base `894fd74`):** V0–V4 and V7. These units need no card-5 hook.
- **Phase B (after card 5 is on `v3b`; the prompt or a follow-up message gives the SHA):**
  1. merge `origin/v3b` at that SHA;
  2. V5, V6 and V8;
  3. tests, suite and report.
  Never import from the unmerged `card5` branch.
- If Phase A is done and card 5 is not yet on `v3b`, write the Phase A part of the report, push, and stop. The work resumes on a follow-up prompt.

## Units

### V0 — Worktree, merge, scope list
- Create the branch, merge `design-docs`, and record HEAD.
- List the files you will touch, with a one-line plan per file.

### V1 — Ablation launchers for the 7 rows (`scripts/ablation/arms.py`, `sweeps.py`, `seed_runs.py`)
- **`arms.py`.** Write exactly the 7 configurations (a): `full`'s tuned configuration with the one change of each row. The literature meta-path row becomes a configuration (a) with seeds 1–10, reported without a claim. This supersedes the card 4 note that called it a main-table row; spec 1.27 will record the change.
- **`sweeps.py`.** Configuration (b) for exactly 4 arms: `endpoints`, `no_path_cells`, `instance_features` and S2. `closure_only` is removed. Each uses 100 trials, seed 0 and the launcher's arm-tuning mode, as today.
- **`seed_runs.py`.**
  - The manifest holds the 7 rows (a) with seeds 1–10, and the 4 arms (b) with seeds 1–10 after their sweeps.
  - `full` (a) runs seeds 5–10 only; seeds 1–4 are looked up from the main table, as today.
  - Freebase mode is an input (`--freebase_mode`), set by card 5 U3.
- Every row keeps its best checkpoint and `config.yaml`. F1–F3 and R1–R3 read them.

### V2 — Statistics (`scripts/ablation/stats.py`)
Input: the run outputs of V1 (test Macro-F1 per arm, dataset and seed; IMDB is the mean per-genre F1, as recorded).

**Output per comparison of `full` with a row and dataset:**
- mean ± sample std per arm;
- paired differences by seed;
- Wilcoxon signed-rank, two-sided (`scipy.stats.wilcoxon`);
- Holm over the 4 datasets per claim;
- the median difference;
- a bootstrap 95 % CI of the mean difference (10,000 resamples, percentile, fixed seed).

**Claim rules (plan v2):**

| Claim | Comparison | Holds if | If not |
|---|---|---|---|
| C2-cells | `full` vs `no_path_cells` | ≥ 3 of 4 | warning |
| C2 | `full` vs `endpoints` | ≥ 3 of 4 | kill rule |
| C14 | `full` vs `instance_features` | ≥ 3 of 4 | — |
| C15 | `full` vs S2 | **both** IMDB and Freebase | warning |

- **"Passes on a dataset"** means: adjusted p < 0.05 and median difference > 0.
- **Decided on (b) where (b) exists;** (a) is reported beside it.
- **S0 and the literature row:** reported only.
- **2×2 table, four differences** (descriptive):
  1. `full − no_path_cells`;
  2. S2 − S0;
  3. `full` − S2;
  4. `no_path_cells` − S0.

Writes a JSON and a Markdown table.

### V3 — Perturbation graphs and their run manifest (`scripts/perturbation/`)
**Generator `make_graphs.py`.** Graphs: ACM and DBLP; IMDB behind `--include_imdb`, where heterophily is the mean over genres. The heterophily level is h = 1 − MLH over the dataset's length-2 meta-paths `M₂` (spec §1 list). It is computed with the reference functions of `scripts/heterophily/compute_heterophily.py`; do not write a second implementation.

| Step | Rule |
|---|---|
| Targets | h ∈ {0.20, 0.40, 0.60, 0.80}, each within ±0.01 when reachable |
| Raise h | **Add** facts of the `M₂` relations that are already in the graph (typed endpoints) and that create different-class meta-path pairs. The labelled endpoint's partner is drawn **uniformly over all other classes**. No duplicate and no self-loop. |
| Lower h | **Remove** facts that create different-class pairs, at most 50 % of each relation's facts. Never add same-class edges. |
| Random control | For every targeted graph, a graph with the same number of facts added or removed per relation, drawn uniformly at random, typed the same way. |
| Seeds | One fixed seed per (dataset, level, kind). |

**Recorded per graph:**
- the reached h and ℋ² (reference code);
- the number of facts added and removed per relation;
- the seed;
- a fingerprint of the edge list.

A level that cannot be reached under the 50 % cap is recorded as unreached with the closest value. Never raise the cap.

**Output format:**
- Each graph is written in the dataset's own file format to `data/NC/<DATASET>_pert/<kind>_h<level>/`.
- Every method must load it through its normal loader with only the data directory changed.
- If some loader cannot do that, stop and report which loader and why.

**Manifest `manifest.py`.**
- **Method rows, 5 seeds (1–5), every targeted and random graph:**
  - H2S-NN `full` and `no_path_cells`;
  - GCN, HAN, SeHGNN, Hetero²Net, HetSheaf.
- **R3 rows (perturbation axis):** H2S-NN S2 on the **targeted** graphs only.
- **Configurations:**
  - each method's configuration tuned on the original graph, read from the main-table output;
  - if a baseline has none yet, its published configuration (flag `--published_config`), listed in the manifest;
  - never retuned.
- **Launchers:** the existing fixed-configuration paths: `run_h2snn_runs.py` for H2S-NN, and the baseline launcher's fixed-config or final-seed path. If a launcher has no fixed-config path, stop and report.

### V4 — Schema read-out, Q3 (`scripts/schema_readout.py` and a small analysis script)
- **Inputs:** the runs of the main table (seeds 0–4) and of `full` (a) (seeds 1–10).
- **Per dataset:**
  - `c_σ`, `I_σ`, `ρ_σ`, keeping the existing functions;
  - for length-2 schemas with coverage ≥ 1 %, the top 10 by coverage.
- **Stability:**
  - mean pairwise Kendall τ-b of the `ρ_σ` rankings over seeds 1–10;
  - a permutation null, shuffling schema labels within each seed, 10,000 draws, Holm m = 4;
  - stable = mean τ-b ≥ 0.5 and adjusted p < 0.05;
  - the τ-b of the `c_σ` rankings reported beside it;
  - fewer than 4 schemas → "n/a".
- **Literature check:** for each HGB published meta-path, the number of seeds with `ρ_σ > 1`.
- **Heterophily link:** Spearman between `ρ_σ` and the homophily of σ's induced target-type graph (reference code), per seed.
- **Figures (matplotlib, PDF):**
  - a bar chart of `ρ_σ` for the top 10;
  - a scatter of `c_σ` vs `I_σ` with the diagonal.

### V5 — Face effect F1–F3 and the case study (Phase B; uses card 5 hooks (b) and (d))
- **η_v.**
  - Definition: the share of `v`'s target-type neighbours through the `M₂` meta-paths that have a different label (IMDB: the mean over genres of the binary disagreement).
  - Nodes without such neighbours are excluded and counted.
  - Quartiles of η_v are taken over the test targets of each dataset.
- **F1, flips.** For every test node, per seed (1–10) and dataset, compare the predictions of `full` and `no_path_cells` (configuration (a)). Count wrong→right and right→wrong per η_v quartile, giving the mean and range over seeds.
- **F2, margin.** Single-label datasets only (ACM, DBLP, Freebase); IMDB is reported in F1 only.
  - margin_v = distance from x*_v to the nearest other-class centroid − distance to its own-class centroid.
  - Centroids are computed from the test x* per class of the same model.
  - Report it for `full` and `no_path_cells`, by η_v quartile.
- **F3, knockout.** On `full` (a), seeds 1–3, use hook (b): for each test node v, knock out the cells through v, then record whether v's prediction changes from right to wrong and the margin change.
  - Each knockout is one evaluation forward of the whole evaluation graph (full batch on HGB). Read node v only.
  - At most 2,000 test nodes per (dataset, seed), drawn with a fixed seed when there are more.
- **Case study (rule; no hand choice).**
  - **Data:** ACM, seed 1.
  - **Candidates:** test nodes that have a fact to a labelled neighbour of another class (P–P cite) or a different-class meta-path neighbour, and whose F3 knockout turns right into wrong.
  - **Pick:** the largest margin drop; on a tie, the smaller node id.
  - **Figure data (JSON) plus one PDF figure:**
    - the node, the different-class neighbour, and the 3 cells through the node with the largest `w_f`, with their node ids, types and schema;
    - 2-D PCA, fitted on all test x* of the `full` run, of h₀, x* with the knockout, and x* of `full`, with the class centroids;
    - the probability of the true class with the knockout and without it.

### V6 — Two-regime analysis R1–R3 (Phase B; uses card 5 hooks (a), (c), (d))
- **R1, transports.**
  - For each cell whose two endpoints are labelled target nodes, compose the endpoint transport from the fact transports along the cell (hook (c); use `Q` or `Qᵀ` per hop direction, as the model does).
  - Report ‖T − I‖_F split by same-class vs different-class endpoint pairs, with the medians and the AUROC of separating the two groups.
  - On ACM (P–P) and Freebase (B–B), also report the fact-level ‖Q_e − I‖_F for facts with two labelled endpoints.
  - Report the learned τ of every run.
- **R2, energy vs performance.**
  - Hook (a) gives e_v.
  - Report the AUROC of e_v separating correct from wrong test predictions, and the Spearman correlation between e_v and η_v, per dataset and seed.
- **R3, energy vs representation.** For `full` and S2 (configuration (a)):
  - Ẽ;
  - a linear probe (logistic regression, `C = 1`, fixed solver and max_iter, trained on the training targets and scored on the test targets), on x* and on h₀;
  - the within/between-class distance ratio on x* and on h₀ (single-label datasets).
- **Two axes:**
  1. Nodes: by η_v quartile on the four ablation datasets.
  2. Graphs: perturbation level, using the V3 runs (`full` and S2 on the targeted graphs).

### V7 — Timing, Q5 (`scripts/timing/`)
- **Datasets:** ogbn-mag (8,192 training targets per epoch) and IMDB.
- **Methods:** every main-table method at its tuned configuration, or at the card 2 central point when it is not tuned yet (flagged).
- **Per method and dataset:**
  - time per epoch: the median of epochs 2–4 after one warm-up epoch;
  - peak GPU memory (`max_memory_allocated`);
  - parameter count.
- **Code to reuse:** `scripts/v3b_cost_smoke.py` for H2S-NN, and the card 2 D6 central-point timing path for the baselines. Find it; if there is none, stop and report.
- **Phase A:** the script plus a CPU or lab-GPU smoke on IMDB for H2S-NN and one baseline, with 1 epoch. The real timing runs on one H100 later.

### V8 — TSUBAME phases (Phase B; same pattern as card 5 U4)
Add phases to `scripts/tsubame/make_cell_list.py`:
- `ablation`: the V1 manifest (b) tunes and seed runs;
- `perturbation`: the V3 manifest;
- `timing`: the V7 runs.

Rules: existing phases stay byte-identical. The ops session owns the TSUBAME card; say so in the report.

## Tests (one file per unit where possible) and mutations
| # | Test |
|---|---|
| T1 | `arms.py` writes exactly the 7 configurations, `sweeps.py` exactly the 4 (b) arms, and no manifest contains an out-of-scope arm |
| T2 | `stats.py` on synthetic numbers with known p (scipy) and a hand-checked Holm example; the claim rules (≥ 3/4; C15 both IMDB and Freebase); the 2×2 differences |
| T3 | Perturbation on a toy typed graph: the reached h is within ±0.01 of the target, or the level is marked unreached at the cap; added facts are typed, new, not self-loops, and come only from `M₂` relations; no other relation is touched; removal ≤ 50 % per relation; the random control has equal counts per relation; h uses the reference function (same value as `compute_heterophily.py` on the toy) |
| T4 | A perturbed toy graph loads through the normal loader of H2S-NN and one baseline with only the data directory changed |
| T5 | Q3: τ-b and the permutation p on a toy ranking (perfectly stable → τ = 1; shuffled → p large); the literature count |
| T6 | F1/F2/F3 on a tiny trained fixture: flips are counted correctly from crafted predictions; the margin sign is right; F3 with hook (b) changes only the chosen node's cells (Phase B) |
| T7 | R1 composition: the composed endpoint transport equals the product of the fact transports in path order (≤ 1e-12); R2/R3 on crafted inputs (Phase B) |
| T8 | Timing script: smoke on IMDB, 1 epoch, writes every field |
| T9 | TSUBAME phases: line counts equal the manifests; existing phases byte-identical (Phase B) |

**Mutations** (each must make a test fail; report which):
1. The perturbation adds same-class edges when raising h (T3).
2. `stats.py` applies Holm over all claims instead of per claim (T2).

## Suite, report, rules
- **Suite.** In Phase B, one full suite in the worktree itself (sbatch, one GPU not shared, environment E, `--mem 35G`). Golden and mb0 must be unchanged; `src/` is untouched, so any difference is a stop.
- **Report.**
  - **File:** `docs/H2S-NN_card7_report.md`.
  - **Short format:** files per unit; test table; mutations; the perturbation reachability table (dataset × level: reached h, ℋ², facts added and removed, or unreached); manifest row counts (ablation, perturbation, timing); deviations; decisions for the author.
  - **Phase A section** first; Phase B appended.
  - **Last line:** `branch=card7 sha=<full head sha> card=docs/H2S-NN_card7_card.md base=894fd7421ce8581d38b967906bde5593e881518a`, or the evidence-commit form.
- **Stop and report if:**
  - anything needs a change in `src/`;
  - a method's loader cannot read a perturbed graph by data directory;
  - a launcher has no fixed-config path;
  - the plan needs an out-of-scope arm;
  - anything in the spec contradicts this card.
- **Not stop conditions:** unreachable perturbation levels (report them); baselines without tuned configurations (use `--published_config` and list them).
- **Jobs.** sbatch only; one GPU not shared; `--mem` = recorded peak + 30 %; never the A6000 or g3an01. Job scripts print `git rev-parse HEAD`, hostname and `$SLURM_JOB_ID`.
- **Git.**
  - One commit per unit; push `card7` after each unit; never push another branch.
  - Never stash, reset, rebase, amend or force-push. Stage files by name.
  - Do not merge into `v3b`.
- **Estimate.** Phase A ≈ 1.5 days. Phase B ≈ 1–1.5 days after card 5 is on `v3b`.
