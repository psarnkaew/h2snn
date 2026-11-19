# Card 7 — ablation and analysis code: report (S6)

Card `docs/H2S-NN_card7_card.md`; plan `docs/ablation_plan_v2_2026-11-19.html`; spec **1.27** (bound after the design-session update of 2026-11-19; the card is otherwise unchanged); pre-registration v2. Evidence: `docs/card7_evidence/`. Every job ran by `sbatch`, env `h2snn-b`, no A6000 / g3an01. **No file under `src/` changed.** Stop conditions: none fired.

## Phase A (V0–V4, V7; 2026-11-19)
`origin/v3b` is still `894fd74` (card 5 not on `v3b`): Phase B (V5, V6, V8, suite) waits for the follow-up prompt.

### Files per unit
| Unit | Commit | Files |
|---|---|---|
| V0 | `235ce3f` (merge design-docs `10db463`), `5045152` | `docs/card7_evidence/v0_scope.md` (file list + plan) |
| V1 | `7ee3db7` | `scripts/ablation/{arms,sweeps,seed_runs,dry_run,param_table}.py`, `tests/test_card7_ablation.py` (new), `tests/test_card4_ablation_driver.py` |
| V2 | `bfd4cad` | `scripts/ablation/stats.py` (new), `tests/test_card7_stats.py` (new) |
| V3 | `0e7f91d`, evidence `84d7efe` | `scripts/perturbation/{make_graphs,manifest}.py` (new), `tests/card7_fixtures.py`, `tests/test_card7_perturbation.py` (new) |
| merge | `ba1cc87` | design-docs `96a8a95` (spec 1.27, pre-registration v2, V7 cost-vs-accuracy item) |
| V4 | `57d94d0` | `scripts/analysis/q3_schema.py` (new), `tests/test_card7_q3.py` (new) |
| V7 | `9df8a93` | `scripts/timing/{time_methods,cost_vs_accuracy}.py` (new), `tests/test_card7_timing.py` (new) |

### Tests (CPU jobs, worktree, environment E)
| # | Job | Result |
|---|---|---|
| T1 | 17803 | 10 OK (5 card 7 + 5 card 4 driver): `arms.py` writes exactly the 7 rows, `sweeps.py` exactly the 4 (b) arms (100 trials, seed 0, launcher space), the ablation manifest holds 4 `tune` + 7 (a) + 4 (b) rows with seeds 1–10 (`full` (a) 5–10), Freebase mode is an input, no out-of-scope arm in any manifest (also checked on the perturbation manifest, T3) |
| T2 | 17807 | 8 OK: Wilcoxon = scipy (10 positive differences: exact p = 2/1024); Holm hand example (0.02, 0.03, 0.06, 0.06); Holm per claim (m = 4); ≥ 3/4 rule, C15 needs IMDB **and** Freebase, wrong-sign significance fails; missing dataset → undecided; 2×2 differences; bootstrap CI fixed |
| T3 | 17810 (9 OK), 17811 (with T1: 14 OK) | DBLP- and ACM-shaped toy releases in HGB format): reached within ±0.01 or unreached with the bound also missing; added facts typed, new, no self-loop, partner of another class; DBLP's non-M₂ relations (paper–term, paper–venue) byte-identical; removal ≤ 50 % and only different-class facts; random control equal counts per relation; written graph's reference `analyse` h = search h; manifest rows (7 methods, S2 on targeted only, seeds 1–5, tuned / published) |
| T4 | 17810, 17811 | OK: the perturbed toy loads through H2S-NN (`data.root` = the graph's data root; +added facts) and through HAN's own HGB loader from the baseline bundle's default `data` root with the working directory at the graph folder (edge count = written `link.dat` lines) |
| T5 | 17816 | 7 OK: stable rankings → τ-b = 1, p = 1/(draws+1); shuffled → p > 0.05; < 4 schemas → n/a; Holm over datasets, stable rule; top-10 by mean coverage ≥ 1 % (length 2 only); literature count (ρ = 1 is not > 1; absent = no); schema → target meta-path (reversed hop via the mirrored type) |
| T8 | 17817 | 4 OK: median of epochs 2–4, every field written (1 epoch → `time_per_epoch_seconds: null`); baseline path with a stubbed `fit_cell` (training time ends at the first `.eval()`, first optimizer's parameters); cost-vs-accuracy on crafted inputs (points, missing, central flag, PDF) |
| T8 smoke | 17818 (RTX 3090, g3n01) | IMDB, 1 epoch, central point: H2S-NN (`v3b_cost_smoke.run`) 963.5 MB peak, 640,302 parameters; GCN (D6 fit path) 180.9 MB, 2,574,853 parameters; every field written; no metric read |

Failed first versions kept as records: 17809 (toy too homophilous; test asked for the bound graph instead of the closest), 17815 (negative error bar in the Q3 figure). Job 17813 (generation) was cancelled before it started (my `--datasets` argument error; empty log removed).

### Mutations (job 17812, snapshot, each on its target test)
| Mutation | Target | Result |
|---|---|---|
| 1 Raising h draws the partner from the node's own class | T3 `test_raising_adds_only_typed_new_different_class_facts` | caught |
| 2 Holm over all claims (m = 16) instead of per claim | T2 `test_holm_is_per_claim_over_four_datasets` | caught (0.03125 ≠ 0.0078125) |

### Perturbation reachability (job 17814, CPU, 21 min, peak 8.2 GB; reference h = 1 − MLH over M₂, `compute_heterophily.analyse` on the written files)
Original graphs: ACM h **0.2621** (ℋ² 0.3994); DBLP h 0.1278 (ℋ² 0.1732); IMDB h 0.3247 (ℋ² 0.7373). Facts per relation: ACM cite 5,343, author 9,949, subject 3,025, term 255,619; DBLP author–paper 19,645; IMDB director 4,932, actor 14,779, keyword 23,610.

| dataset | level | targeted h (ℋ²) | random h (ℋ²) | facts (per relation, targeted = random counts) |
|---|---|---|---|---|
| ACM | 0.20 | 0.1978 (0.3008) | 0.2605 (0.3970) | removed: cite 417, author 777, subject 236, term 19,970 |
| ACM | 0.40 | 0.4086 (0.6202) | 0.3392 (0.5153) | added: 1,002 / 1,865 / 567 / 47,929 |
| ACM | 0.60 | 0.6011 (0.9046) | 0.4996 (0.7524) | added: 5,343 / 9,949 / 3,025 / 255,619 |
| ACM | 0.80 | **unreached**, closest 0.6376 (0.9576) | 0.5730 (0.8609) | added: 10,686 / 19,898 / 6,050 / 511,238 |
| DBLP | 0.20 | 0.1977 (0.2675) | 0.1693 (0.2294) | added 307 |
| DBLP | 0.40 | 0.3969 (0.5346) | 0.3039 (0.4109) | added 1,535 |
| DBLP | 0.60 | 0.6080 (0.8168) | 0.4643 (0.6267) | added 4,297 |
| DBLP | 0.80 | **unreached**, closest 0.7477 (1.0096) | 0.7122 (0.9602) | added 39,290 |
| IMDB (optional) | 0.20 | **unreached** at the 50 % cap, 0.2383 (0.5310) | 0.2903 | removed 2,465 / 7,388 / 11,802 |
| IMDB | 0.40 | 0.4019 (0.9139) | 0.3601 | added 2,466 / 7,390 / 11,805 |
| IMDB | 0.60, 0.80 | **unreached**, closest 0.4301 / 0.4284 | 0.4193 / 0.4196 | added 14,796 / 44,337 / 70,830 |

The addition bound (`--max_add_ratio 4`) is not what stops 0.80: h peaks and then falls as facts are added (h at the bound: ACM 0.6312, DBLP 0.7182, IMDB 0.4295). Graphs (85–90 MB per dataset, unchanged files hard-linked): `H2S-NN/data/NC/{ACM,DBLP,IMDB}_pert/<kind>_h<level>/`, each with `graph.json` (seed, α, counts, reached h, ℋ², h at the bound, SHA-256 of `link.dat`).

### Manifest row counts (job 17819, placeholder `full` = the experiment files; nothing run)
| Manifest | Rows | Runs |
|---|---|---|
| Ablation (4 datasets) | 16 `tune` (b) + 28 (a) + 16 (b) | 1,600 trials + 264 (a) runs (`full` 5–10 only) + 160 (b) runs |
| Perturbation, ACM + DBLP, reached levels (0.2/0.4/0.6 × targeted/random) | 90 when every baseline has a configuration (7 methods × 12 graphs + S2 × 6) | 450 (seeds 1–5) |
| — today, `--published_config` | 54 (+ 9 IMDB) | GCN, HAN, HetSheaf: no tuned and no published configuration on ACM/DBLP/IMDB (`v3_manifest_published_missing.json`); SeHGNN, Hetero²Net: published |
| Timing | 14 methods × 2 datasets = 28 | `time_methods.py --list`: H2S-NN + the 13 registry NC baselines, same list on ogbn-mag and IMDB |

### Deviations (each one line)
- `tests/test_card4_ablation_driver.py` changed: it expected `ego_head` and no literature row; now the 7 rows (the literature row compared with the resolved `full`).
- The literature row's YAML is the resolved configuration from the existing builder `fixed_literature_schema_config` (only `data.path_source` and the run name differ), the other rows are `full`'s file + one key.
- Perturbed-graph layout: `<...>/<kind>_h<level>/data/NC/<name>/raw/<NAME>/`. The baseline launchers have no data-root option (`bundle_overrides` in `src/` returns `{}`), so their rows run with the working directory at `<kind>_h<level>/` and read `./data`; nothing in `src/` or the launchers changed.
- Readings of the generator rules (card silent): one scalar α sets every M₂ relation's count, `round(α · facts_r)`; for a target–x relation the partner q (uniform over the other classes) fixes x as a uniform neighbour of q; IMDB "class" = label set; addition bound 4× facts per relation (not binding, see above).
- Unreached levels are written (closest graph) but get no manifest rows unless `--include_unreached`.
- H2S-NN on ogbn-mag (mini-batch) is timed through the trainer (`train_epoch` wrapped): `v3b_cost_smoke.py` is full-batch only. Baseline epoch time = previous epoch report → first `.eval()`; baseline parameter count = the first optimizer's parameters.
- T8's "smoke on IMDB" is a lab-GPU job, not a unit test (tests never reach real data); the unit test uses stubs.
- The 1.26 §6 text still listed the old arms when the card started; the plan was followed, and spec 1.27 (merged `ba1cc87`) now agrees with it.

### Decisions for the author
- **ACM's current h is 0.262, not ≈ 40 %.** The plan's "≈ 40 %" matches ACM's ℋ² (0.399). With h = 1 − MLH (card, spec §1), ACM's 0.20 is a removal and 0.40/0.60 are additions. Confirm the definition (h, not ℋ²).
- **0.80 is unreached** on ACM (closest 0.638) and DBLP (0.748); IMDB reaches only 0.40 (its per-genre binary h saturates near 0.43; 0.20 not reachable under the cap). Recommendation: report 0.20/0.40/0.60 on ACM and DBLP; do not run IMDB.
- Confirm the generator readings above (proportional α; partner via q's neighbour) and "no runs on unreached levels".
- GCN, HAN and HetSheaf have no published HGB configuration in our adapters: their perturbation rows need their main-table sweeps first (SeHGNN, Hetero²Net can use published ones with `--published_config`).
- Confirm the timing list = H2S-NN + the 13 registry NC baselines on both datasets (28 timing runs).
- V8 (TSUBAME phases) is Phase B; the ops session owns the TSUBAME card.

The `sha` below is the evidence commit (`84d7efe`: code, tests and evidence of Phase A); this report commit sits on top of it and changes only this file.

branch=card7 sha=84d7efe11e8209fc3289a0b1e6dfa82a9b17aa7c card=docs/H2S-NN_card7_card.md base=894fd7421ce8581d38b967906bde5593e881518a
