# Card 9 report — ablation and analysis scripts to pre-registration v3 (spec 1.29)

Card: `docs/H2S-NN_card9_card.md`. Binding: `docs/ablation_analysis_preregistration_v3.md`.

## W0 — branch, merge, scope

- `origin/v3b` = `fc0ebaa605280f2d987c4848cc3109c7d3b66680` (checked); `origin/design-docs` contains `b9a2a2f0d6d6a8df255d476108b73e458f7cc693` (checked).
- `card9` created from `fc0ebaa`; `git merge --no-ff b9a2a2f` → `313eb23`.

Files this card touches (one line each):

| Unit | File |
|---|---|
| W1 | `H2S-NN/scripts/ablation/arms.py` |
| W1 | `H2S-NN/scripts/ablation/seed_runs.py` |
| W2 | `H2S-NN/scripts/ablation/param_table.py` |
| W3 | `H2S-NN/scripts/ablation/stats.py` |
| W4 | `H2S-NN/scripts/analysis/q3_schema.py` |
| W5 | `H2S-NN/scripts/perturbation/rules.py` (new) |
| W5 | `H2S-NN/scripts/perturbation/figure.py` (absent methods) |
| W6 | `H2S-NN/scripts/analysis/face_effect.py` |
| W6 | `H2S-NN/scripts/analysis/mechanism_common.py` |
| W7 | `H2S-NN/scripts/analysis/two_regime.py` |
| W8 | `H2S-NN/scripts/tsubame/make_cell_list.py` |
| W9 | `H2S-NN/tests/test_card7_ablation.py`, `test_card7_stats.py`, `test_card7_q3.py`, `test_card7_tsubame.py` (v2 → v3 assertions) |
| W9 | `H2S-NN/tests/test_card9_v3.py` (new: T1–T7) |
| W9 | `docs/card9_evidence/` (job scripts and logs) |
| — | `docs/card9_report.md` (this file) |

## 1. Files touched, per unit

| Unit | Commit | Files |
|---|---|---|
| W1 | `e04bfdc` | `scripts/ablation/arms.py`, `seed_runs.py`, `ablation_array.sbatch` (comment only), `tests/test_card7_ablation.py` |
| W2 | `aa42761` | `scripts/ablation/param_table.py` |
| W3 | `108b06c` | `scripts/ablation/stats.py`, `tests/test_card7_stats.py` |
| W4 | `c4dd259` | `scripts/analysis/q3_schema.py`, `tests/test_card7_q3.py` |
| W5 | `05c1367` | `scripts/perturbation/rules.py` (new), `scripts/perturbation/figure.py`, `tests/test_card9_v3.py` (T1, T6, T7) |
| W6 | `de636fc` | `scripts/analysis/mechanism_common.py`, `face_effect.py`, `tests/test_card7_mechanism.py` |
| W7 | `391bb58` | `scripts/analysis/two_regime.py` |
| W8 | `263d85e` | `scripts/tsubame/make_cell_list.py`, `tests/test_card7_tsubame.py` |
| W9 | `f2c37ad` + report commit | `tests/test_card9_v3.py` (T2–T5), `docs/card9_evidence/*` |

All paths under `H2S-NN/`. Nothing in `src/` or `configs/`.

## 2. v2 → v3 behaviour per script

| Script | v2 (card 7) | v3 (card 9) |
|---|---|---|
| `arms.py` | 7 rows, seeds 1–10 | 8 rows (`uniform` = `ablation.path_weight_input: uniform`), `SEEDS = (1..5)`; `TUNED_ARMS` kept only because `sweeps.py` (uncalled) and the card 4 `tuning.arm` test read it |
| `seed_runs.py` | `tune` + (a) + (b) rows; `--tuning_base` | (a) `runs` rows only: `full` base seed 5 × 1 run, others base seed 1 × 5 runs; `--tuning_base` removed (nothing else needed it); `tuned_config()` kept because `run_row.py` uses it for the timing phase |
| `param_table.py` | `g` = path encoder + gate | the same, plus `path_log_weight`: under `uniform` the model builds no encoder/gate and holds one scalar `b`, so `g` = 1; read from `model.named_parameters()` by name prefix; `g_form` says which form was counted; `delta_g_vs_full` added |
| `stats.py` | Wilcoxon, Holm, bootstrap, (b) decides | per comparison: seeds, mean ± sd (ddof 1), the 5 differences, mean, `wins` (strict); `better` = mean > 0 and wins ≥ 4, `None` unless all 5 seeds on both sides; claims C2-cells, C2, **C2-w**, C14, C15 on (a); S0 reported; literature with roles swapped (`literature_better`); 2×2 + `complementary`; `collect` accepts only `<ds>__ablation_a_<row>`, raises on a (b) name, an unknown row or a seed outside 1–5; no p-value anywhere |
| `q3_schema.py` | seeds 1–10, permutation null + Holm | seeds 1–5 only (seed 0 ignored and listed); stable = mean τ-b ≥ 0.5 over 10 pairs (`None` if < 5 seeds); `favour` = 5 of 5; ρ_σ-vs-homophily scatter (one marker per seed × schema) + per-seed Spearman (statistic only); wording "learned face weights" |
| `perturbation/rules.py` (new) | — | `h_max`, `drop_m(s)`, `gap(G, s)` per seed; E1 per GCN/HAN/SeHGNN (mean + ≥ 4 of 5, on both ACM and DBLP; false as soon as one is false); Hetero²Net/CSNN/HetSheaf reported without verdict; E2; methods without a runnable row → `absent` |
| `perturbation/figure.py` | — | `absent` per dataset in the JSON and as text in the panels |
| `face_effect.py` | seeds 1–10, raw margin, no rules | seeds 1–5 (F3 1–3); net fix rate per group and `all`; raw + scale-free margins; F2 difference; F3 and case pick on scale-free margin; rules F1a, F1b, F2a, F2b (ACM, DBLP, Freebase), F3 |
| `mechanism_common.py` | — | `centroid_scale` (D), `scale_free_margins`, `g_low`, `four_of_five`; `arm_of` returns `uniform` (before it would have returned `full` for a uniform run) and raises on other non-`full` `path_weight_input` |
| `two_regime.py` | R2 on raw e_v, no rules | R2 primary on ẽ_v = e_v / ‖x*_v‖² (x* from hook (a), same rows), raw beside (`R2.raw`); rules R1, R2, R3 (IMDB, Freebase; S2 partner needed, else `None`) |
| `make_cell_list.py` | tuning list + seed list, (b) waits | `--phase ablation` writes `cells_ablation_seeds.txt` only (one line per `runs` row); a `tune` or (b) row raises |

**Decisions made in code (for review):**
- "4 of 5" rules (stats, F, R, E) give `None` when any of seeds 1–5 is missing, even when the outcome is already fixed; E1/E2 "on both datasets" and F3 "each of seeds 1–3" are false as soon as one part is false.
- Baseline per-seed values (W5, not a stop): every launcher's summary (`summarize_cell`) stores `units`, one record per seed with `test_macro_f1`. Seed = `final_seed` (tuned) or unit `run<i>` → i-th seed of the row (`build_units`: base_seed + i). A summary without `units` raises `NoPerSeedValues`.
- H2S-NN per-seed files are `<output_dir>/<exp>_run<i>/best_test_metrics_run<i>.json`, beside (not inside) `<output_dir>/<exp>/` (`TrainConfig.exp_dir`).
- Schema homophily stays the reference code's `h_edge` of G_σ (edge label homophily), as in card 7.

## 3. Tests

**Changed card 7 tests (v2 → v3):**

| Test | Change |
|---|---|
| `test_card7_ablation`: `..._seven_rows` → `..._eight_rows` | `uniform` row; `SEEDS == (1..5)` |
| `test_card7_ablation`: manifest, Freebase mode | no `tune`/(b) rows, seeds end at 5, new `build_manifest` signature; every row's file checked |
| `test_card7_ablation`: `OUT_OF_SCOPE` | `uniform` removed (it is a v3 row) |
| `test_card7_stats` | removed Wilcoxon, Holm (×2), bootstrap tests (code removed); claim rules on 5 seeds with C2-w; new S0 + literature test; 2×2 adds `complementary`; JSON has no p key (8 → 5 tests) |
| `test_card7_q3` | stability on seeds 1–5 without p; "shuffled → large p" → alternating rankings (τ-b = −0.2, not stable); Holm test → stable-rule + figures; literature counts out of 5 |
| `test_card7_mechanism` | flip counts carry `net_fix_rate` and `all`; case pick reads `scale_free_margin_change`; end-to-end checks `all`, the scale-free change and the rule keys |
| `test_card7_tsubame` T9 | seed lines only (16 = 2 × 8), a v2 manifest raises, `main` writes only `cells_ablation_seeds.txt` |

**New:** `tests/test_card9_v3.py`, 15 tests: T1 (3), T2 (2), T3 (1), T4 (2, with the F rules), T5 (2, with R1–R3), T6 (3), T7 (2).

**Mutations** (job 18008, snapshot of `f2c37ad`; each against its target only; all restored identical):

| Mutation | Target | Result |
|---|---|---|
| m1 `wins >= 3` | T1 `test_T1_rule_edge_cases` | FAIL (caught) |
| m2 no `mean_difference > 0` | T1 `test_T1_rule_edge_cases` | FAIL (caught) |
| m3 margin not divided by D | T4 `test_T4_scaling_...` | FAIL (caught) |
| m4 ẽ_v = raw e_v | T5 `test_T5_scaling_...` | FAIL (caught) |
| m5 E1 on ACM only | T6 `test_T6_e1_needs_both_...` | FAIL (caught) |

**Suite:** job 18009 (g2n01, 2080 Ti, `f2c37ad`, clean tree): `Ran 903 tests` → **OK**, no skip.
Count vs card 7's 879 (loader count, job 18010): 4d00154 (card 7 suite head) 879 → fc0ebaa 891 (+12 from merges after card 7's suite: tsubame collect/final-seeds/lines/host-memory/phases tests and one card 7 mechanism test) → card9 903 (+15 new card 9 tests, −3 removed stats tests).

## 4. Freeze

`git diff fc0ebaa605280f2d987c4848cc3109c7d3b66680 -- H2S-NN/src H2S-NN/configs` is **empty** (also printed by job 18009). No golden/mb0 compare (not needed, card scope rule). The `design-docs` merge brought paper `.tex` changes; this card wrote none.

## 5. Notes for the ops session (W8)

- `--phase ablation` now writes **no tuning list**: one file `cells_ablation_seeds.txt` (default name; `--out` overrides), one `run_row.py` line per manifest `runs` row: per dataset 8 lines (`full` seed 5 alone; 7 rows × seeds 1–5). 4 datasets → 32 lines, 144 runs (v3 §5).
- Manifests must be rebuilt with the new `seed_runs.py` (no `--tuning_base`); a v2 manifest makes `make_cell_list.py` raise.
- The W&B project `h2snn_ablation_t4` is no longer used by any list.
- `run_row.py --tuned_config` is unchanged (no ablation line uses it now; the timing phase uses `--optional_tuned_config`).

## 6. Questions

1. `scripts/schema_readout.py` (the per-run read-out, not in this card's list) still has its own `--permutations` Kendall test across runs. `q3_schema.py` does not read it. Should it be removed in a later card, or left as it is?
2. `None` while a seed is missing, even when the 4-of-5 outcome is already settled: is this conservative choice what the author wants?
3. Schema homophily = edge label homophily `h_edge` of G_σ (unchanged from card 7). Please confirm this is the "label homophily" of v3 §4.1.

branch=card9 sha=c6e3e7c7bd15248eba0130bf9cdc0651999767bf card=docs/H2S-NN_card9_card.md base=fc0ebaa605280f2d987c4848cc3109c7d3b66680
