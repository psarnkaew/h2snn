# Task card 9 — Ablation and analysis scripts to pre-registration v3 (spec 1.29)

**Owner (design):** model session. **Executor:** the author assigns (S6 wrote card 7; its worktree `/mnt/data/peeradon/h2snn-model` is the natural place).
**Branch:** new branch `card9` from `v3b` `fc0ebaa605280f2d987c4848cc3109c7d3b66680`. Then `git merge --no-ff` of `design-docs` at the SHA given in the prompt. Merge, never rebase.

## What changed and why
On 2026-11-20 the author reviewed the ablation pre-registration item by item and approved **`docs/ablation_analysis_preregistration_v3.md`** with **spec 1.29** (`docs/spec_v1.29_draft_edits.md`; decisions log row "Ablation pre-registration v3 and spec 1.29 approved"). The card 7 scripts implement v2. This card brings them to v3. No run has started, so nothing is re-run.

The changes, in one table (D-numbers are the decisions in v3's header):

| # | v2 (card 7 code) | v3 (this card) |
|---|---|---|
| D1 | Seeds 1–10; Wilcoxon, Holm, bootstrap CI; pass = Holm p < 0.05 and median > 0 | Seeds 1–5; no p-value, no Holm, no bootstrap; `full` better than X on a dataset = mean paired difference > 0 **and** `full` strictly higher in ≥ 4 of 5 seeds |
| D2 | Configuration (a) and (b) (4 arms tuned on their own) | (a) only; no `tune` rows, no (b) rows, no capacity rule; parameter counts reported |
| D3 | 7 rows | 8 rows: `uniform` added (`ablation.path_weight_input: uniform`), claim C2-w |
| D5 | Schema read-out: seeds 1–10, permutation null + Holm | Seeds 1–5 of `full` (a); no null; stable = mean τ-b ≥ 0.5; literature check out of 5; ρ_σ vs schema homophily as a scatter |
| D6 | Perturbation: figure only | Drop, gap and the rules E1, E2 computed per seed; HetSheaf and CSNN may be absent |
| D7 | Face effect: raw margin; no rules | Net fix rate, G_low, scale-free margin, the F1–F3 rules |
| D8 | Two-regime: raw `e_v`; no rules | `ẽ_v = e_v / ‖x*_v‖²` primary (raw beside); the R1–R3 rules |

## Scope rule
- Exactly pre-registration v3. Nothing else is implemented, run or reported.
- **Code in `scripts/` and `tests/` only.** No change in `src/` or `configs/`: the H2S-NN freeze (`365a757`, unchanged at `fc0ebaa`) must hold. Check at the end: `git diff fc0ebaa -- H2S-NN/src H2S-NN/configs` is empty. Because `src/` does not change, no golden or mb0 compare is needed.
- Out-of-scope arms stay in frozen `src/` and in no manifest: `closure_only`, `hub_A`, `covering_only`, `schema`, `direction_blind_maps`, `free_cell_maps`, `ego_head`, `h0_only_readout`.
- `sweeps.py` and `variance_gate.py` stay in the tree and are not called by any manifest or phase.

**Binding (in this order):**
1. `docs/ablation_analysis_preregistration_v3.md`.
2. Spec 1.29 §5 (C2, C2b, C14, C15 and the note under the table), §6 "Ablation statistics", "Ablation table", "Timing and analysis-only settings".
3. `docs/experiment_settings_decisions.md` row of 2026-11-20 "Ablation pre-registration v3 and spec 1.29 approved".
4. Card 7 (`docs/H2S-NN_card7_card.md`) for everything this card does not change.
5. `docs/working_preferences.md` (slim preparation: correctness checks only).

**Mode:** correctness only. Unit tests on fixtures and synthetic data, one mutation per new rule, and the full suite in a SLURM job. No smoke run on real data, no performance run, no metric reported as a result.

## Units

### W0 — Branch, merge, scope list
- Create `card9` from `fc0ebaa`; merge `design-docs` at the prompt's SHA; record HEAD.
- List the files you will touch, one line each.

### W1 — Rows and manifest (`scripts/ablation/arms.py`, `seed_runs.py`)
- `arms.ARM_CHANGES`: add `"uniform": {"ablation": {"path_weight_input": "uniform"}}`. Eight rows: `full`, `no_path_cells`, `S2`, `S0`, `uniform`, `endpoints`, `instance_features`, `literature`.
- `arms.SEEDS = (1, 2, 3, 4, 5)`. `REUSED_FULL_SEEDS = (1, 2, 3, 4)` unchanged. `full` (a) runs seed 5 only.
- Remove `TUNED_ARMS` use from the manifest: `seed_runs.build_manifest` emits only `runs` rows of configuration (a); no `tune` row and no (b) row. Remove the `tuning_base` input if nothing else needs it (say so in the report).
- Docstrings: v3, 8 rows, seeds 1–5, (a) only.
- Expected manifest per dataset: 8 `runs` rows; `full` = base seed 5, 1 run; every other row = base seed 1, 5 runs.

### W2 — Parameter table (`scripts/ablation/param_table.py`)
- The 8 rows. Under `uniform`, `g` is absent and one scalar `b` remains; report the `g` part as that count, and say how it was read from the model.

### W3 — Ablation rule (`scripts/ablation/stats.py`)
- `paired`: per comparison report the seeds, mean ± sample std (ddof = 1) per row, the paired differences, their mean, and `wins` = number of seeds with `full − X > 0` (strict; a tie is not a win).
- Remove Wilcoxon, Holm, the bootstrap CI and every p-value from the output.
- `better(dataset)` = `mean_difference > 0 and wins >= 4`, computed only when all 5 seeds are present on both sides; otherwise `None` (incomplete), never `False`.
- Claims: C2-cells (`no_path_cells`, 3 of 4, warning signal), C2 (`endpoints`, 3 of 4, kill rule), **C2-w (`uniform`, 3 of 4, "no claim that learned per-face weights help")**, C14 (`instance_features`, 3 of 4), C15 (S2, IMDB and Freebase, warning signal). All on configuration (a).
- Reported only: S0; the 2×2 (four mean paired differences; "complementary" flag per dataset = mean of `full − no_path_cells` > mean of `S2 − S0`); the literature row with the rule **with roles swapped** (literature better than `full` = mean of `literature − full` > 0 and literature higher in ≥ 4 of 5 seeds).
- `collect`: accept only `<dataset>__ablation_a_<row>` run names; a (b) name raises with a clear message. `KNOWN_ROWS` includes `uniform`.
- Markdown table: per claim and dataset, the means ± std, the five differences, mean difference, wins, `better`; then the claim verdict.

### W4 — Schema read-out (`scripts/analysis/q3_schema.py`)
- Inputs: `full`'s configuration-(a) runs of seeds 1–5 only (seeds 1–4 are the main-table runs). Main-table seed 0 is not read.
- `STABILITY_SEEDS = (1, 2, 3, 4, 5)`: the mean pairwise Kendall τ-b over the 10 pairs; `stable = mean_tau_b_rho >= 0.5`. Remove the permutation null, its draws and Holm. Keep the τ-b of the `c_σ` rankings beside it.
- Literature check: per HGB meta-path, seeds (of 5) with `ρ_σ > 1`; a flag `favour = (count == 5)`.
- Heterophily link: a scatter figure of `ρ_σ` against schema homophily (all seeds, one marker per seed and schema) per dataset, plus the per-seed Spearman in the JSON. No pooled single-number verdict.
- Output and docstrings use "learned face weights", never "the sampler learns".

### W5 — Perturbation rules (`scripts/perturbation/`, new `rules.py` or inside `figure.py`)
- Per-seed values are needed. H2S-NN: `best_test_metrics_run<i>.json` of each run. Baselines: find the per-seed test Macro-F1 the launcher writes. **If a baseline writes only a mean and std, stop and report** (do not change a baseline adapter in this card).
- `h_max` = the highest targeted level reached per dataset (from `graph.json`).
- `drop_m(s) = F1_m,random(h_max, s) − F1_m,targeted(h_max, s)`.
- `gap(G, s) = F1_full(G, s) − F1_no_path_cells(G, s)`; the original-graph gap uses the ablation (a) runs of seeds 1–5.
- **E1** per baseline m in {GCN, HAN, SeHGNN}: `mean(drop_full) < mean(drop_m)` and `drop_full(s) < drop_m(s)` in ≥ 4 of 5 seeds; holds when true on both ACM and DBLP. Hetero²Net, CSNN and HetSheaf: the same numbers, reported, no verdict.
- **E2:** `gap(targeted h_max, s) > gap(original, s)` in ≥ 4 of 5 seeds, on both ACM and DBLP.
- A method with no rows (HetSheaf, CSNN not ready) is listed as `absent` in the output and the figure; it never raises.

### W6 — Face effect (`scripts/analysis/face_effect.py`, `mechanism_common.py`)
- Seeds 1–5 for F1 and F2; F3 seeds stay 1–3.
- **Net fix rate** per η group = (wrong→right − right→wrong) / group size; also for all test nodes.
- **G_low** = the lowest non-empty of G0, G1 (per dataset and seed).
- **Scale-free margin:** `margin_v / D`, with `D` = the mean pairwise Euclidean distance between the class centroids of that model and seed (the same test-x* centroids `mc.margins` uses). Report raw and scale-free.
- F3 margin change and the case-study pick use the scale-free margin (the knockout keeps the base centroids, so `D` is the base model's).
- Rules per dataset (output booleans and counts):
  - F1a: net fix rate of all test nodes > 0 in ≥ 4 of 5 seeds;
  - F1b: net fix rate of G3 > that of G_low in ≥ 4 of 5 seeds;
  - F2a (ACM, DBLP, Freebase): mean scale-free margin difference (`full` − `no_path_cells`) > 0 in ≥ 4 of 5 seeds;
  - F2b: that difference larger in G3 than in G_low in ≥ 4 of 5 seeds;
  - F3: right→wrong share > 0 in each of seeds 1–3.

### W7 — Two-regime (`scripts/analysis/two_regime.py`)
- R2: `ẽ_v = e_v / ‖x*_v‖²` primary (the same `x*` the hook returns); raw `e_v` beside. AUROC (wrong vs correct) and Spearman with η_v on both.
- Rules per dataset (output booleans and counts over seeds 1–5):
  - R1: cell-level AUROC (different-class farther from `I`) > 0.5 in ≥ 4 of 5 seeds;
  - R2: AUROC of `ẽ_v` > 0.5 in ≥ 4 of 5 seeds;
  - R3 (IMDB, Freebase): (probe Macro-F1 on x* of `full` − of S2, same seed) in G3 > in G_low in ≥ 4 of 5 seeds. R3 needs a `full` run and an S2 run of the same dataset and seed; a missing partner gives `None`.

### W8 — TSUBAME ablation phase (`scripts/tsubame/make_cell_list.py`)
- `--phase ablation` writes no tuning list (there are no `tune` rows) and one seed line per `runs` row. Remove the "(b) row waits for its sweep" logic. The ops session reads this in the report.

### W9 — Tests, mutations, suite
- Update the card 7 tests that assert v2 behaviour (seeds 1–10, (b) rows, Wilcoxon, Holm, the permutation null). Say per test what changed.
- New tests (one per new mechanism):
  - **T1 rule:** a crafted long table: 4 wins with positive mean → better; 3 wins → not better; 4 wins with a negative mean → not better; one tie counts as no win; one missing seed → `None`. C2-w present; a (b) run name raises.
  - **T2 `uniform` row:** the emitted `uniform.yaml` passes `TrainConfig.from_dict`; on the existing small fixture, one training step of the `uniform` model has finite loss, `g` is absent, `b` moves, and all faces carry one weight.
  - **T3 manifest:** 8 rows per dataset, seeds as W1, no `tune` row.
  - **T4 scale-free margin:** multiplying every x* by 3 leaves the scale-free margins unchanged (raw margins triple).
  - **T5 `ẽ_v`:** multiplying x* (and the energy terms accordingly) by 3 leaves `ẽ_v` unchanged; on a fixture where wrong nodes carry larger `ẽ_v`, R2 = true.
  - **T6 perturbation rules:** crafted per-seed numbers where E1 holds on ACM but not DBLP → E1 false; E2 true case; a method with no rows → `absent`.
  - **T7 q3:** seeds 1–5 only (seed 0 ignored); no permutation key in the output; `favour` only at 5 of 5.
- Mutations (run each against its target test only): (m1) `wins >= 3`; (m2) drop `mean_difference > 0`; (m3) margin not divided by `D`; (m4) `ẽ_v` = raw `e_v`; (m5) E1 on one dataset only. Each must fail its test.
- Full suite in one SLURM job; compare with the card 7 reference (879 tests): every change in the count explained.

## Run rules
- Every job through SLURM (`srun`/`sbatch`, `--mem`, `--time`); the login node is for git and editing. Never request the A6000 or `g3an01` (`--exclude=g3an01`).
- Every job script prints `git rev-parse HEAD`, `hostname` and `$SLURM_JOB_ID`.
- Tests never reach real data.

## Git
- Commit per unit. Push only `card9`. Never push, merge or fast-forward `v3b`. Never rebase, amend or force-push.
- The report (`docs/card9_report.md` on `card9`) ends with: `branch=card9 sha=<full head sha> card=docs/H2S-NN_card9_card.md base=fc0ebaa605280f2d987c4848cc3109c7d3b66680`. After that line, no further push without a new prompt.

## Stop conditions (stop, push what is committed, report)
- Any change seems needed in `src/` or `configs/`.
- The `uniform` arm does not build or train under TD-1 maps.
- A baseline gives no per-seed test values (W5).
- A rule of pre-registration v3 is ambiguous for the code, or the code shows a fact that contradicts it.
- The suite has a failure that this card did not cause and cannot explain.

## Report (short: tables and decisions)
1. Files touched, per unit.
2. The v2 → v3 behaviour per script (one line each).
3. Tests: new, changed, mutations caught; suite count and job id.
4. `git diff fc0ebaa -- H2S-NN/src H2S-NN/configs` (must be empty).
5. Notes for the ops session (W8).
6. Questions.
