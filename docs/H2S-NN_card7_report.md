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

## Phase A+ (amendments 1, 3, 4, 7 without a card-5 hook; 2026-11-19)
Design-docs `ea80f75` (contains `c5db2fc`, `c551123`) merged `0a77d6b`. `origin/v3b` is `59ab582` (card 6 H3; card 5 not on `v3b`): Phase B (V5, V6, V8, suite) still waits. Only `scripts/` and `tests/` changed; no file under `src/` or `configs/`.

### Files per unit
| Unit | Commit | Files |
|---|---|---|
| merge | `0a77d6b` | design-docs `ea80f75` |
| A+1 (amendment 1) | `2a2f9bd` | `scripts/perturbation/make_graphs.py` (levels, `LEVEL_SEED_INDEX`, edge fingerprint, C, ceiling, `--levels`, `--backfill`), `fingerprint.py` (new), `figure.py` (new) |
| A+2 (amendment 3) | `e2a8f11` | `scripts/perturbation/run_checked.py` (new) |
| A+3 (amendments 4, 7) | `9192f78` | `scripts/perturbation/baseline_configs.py` (new), `run_hgb_command.py` (new), `manifest.py` |
| A+4 (tests) | `9400456` | `tests/test_card7_amendments.py` (new), `tests/test_card7_perturbation.py` (manifest test) |
| A+5 (evidence) | `0dc936b` | `docs/card7_evidence/aplus_*`, `targeted_1783{6,7}.txt` |

### What was built
- **Levels:** `LEVELS = (0.20, 0.30, 0.40, 0.50, 0.60)`; seed index fixed per level (0.20/0.40/0.60/0.80 keep Phase A's 0–3, 0.30 → 4, 0.50 → 5), so 0.20/0.40/0.60 stay as generated (job 17838's backfill re-checked every Phase A `link.dat` SHA-256: unchanged). Only 0.30 and 0.50 were generated. Every `graph.json` now holds h, ℋ², C, the ceiling 1 − 1/C and the edge fingerprint.
- **Fingerprint (amendment 3):** the set of facts as unordered (node type, local index) pairs in PyG `HGBDataset`'s numbering, SHA-256 of the sorted pairs. Generator side: parsed from the written `info/node/link.dat` without PyG. Run side: the `PreparedH2SData` the loader returned (H2S-NN's `build_dataset`, also every baseline bundle's `bundle.data`). `run_checked.py` runs the launcher in-process with `build_dataset` wrapped and checks, at every load, (1) data root = the graph's data root (before loading; catches a wrong working directory), (2) `link.dat` SHA-256 = the generator's (the file the vendor loaders such as HAN's read under the same root), (3) loaded fingerprint = the manifest's `fingerprint` column. Each check is a JSON line in `<run>/fingerprint_checks.jsonl`; a mismatch, or a launcher that never loads, fails the run before training.
- **Figure:** `figure.py` collects each manifest row's summary (test Macro-F1 mean ± std) against the reached h, one panel per dataset × kind, dashed vertical line at 1 − 1/C (from `graph.json`), optional hollow reference points at the original h (`--reference`).
- **Baselines (amendments 4, 7):** source fixed per method (`baseline_configs.CONFIG_SOURCE`). GCN/HAN run at the mapped HGB command through `run_hgb_command.py` (the published launcher's loop: `build_units`, `fit_cell` with source "published" so the HGB epochs/patience are the budget, `summarize_cell`); SeHGNN/Hetero²Net through `run_baseline_published.py`; HetSheaf/CSNN through `run_baseline_tuning.py --final_seeds 1 2 3 4 5` once their ACM/DBLP seed-0 sweep exists, until then rows marked "needs tuned config" with no command.
- **Manifest:** ACM and DBLP only (IMDB graphs and Phase A's 0.80 folders stay on disk, listed under `excluded_graphs`), CSNN added, columns `h`, `h2_index`, `fingerprint`, `summary`; every command wrapped by `run_checked.py`.

### Levels reached (job 17838, CPU, 3 min, peak 5.5 GB; reference h = 1 − MLH over M₂, ℋ² beside it; every level reached within ±0.01)
Ceiling 1 − 1/C: ACM 0.6667 (C = 3), DBLP 0.7500 (C = 4). Original: ACM h 0.2621 (ℋ² 0.3994), DBLP h 0.1278 (ℋ² 0.1732).

| dataset | level | targeted h (ℋ²) | random h (ℋ²) | facts (targeted = random counts per relation) | job |
|---|---|---|---|---|---|
| ACM | 0.20 | 0.1978 (0.3008) | 0.2605 (0.3970) | removed: cite 417, author 777, subject 236, term 19,970 | 17814 |
| ACM | 0.30 | 0.2917 (0.4446) | 0.2774 (0.4230) | added: 167 / 311 / 95 / 7,988 | 17838 |
| ACM | 0.40 | 0.4086 (0.6202) | 0.3392 (0.5153) | added: 1,002 / 1,865 / 567 / 47,929 | 17814 |
| ACM | 0.50 | 0.4959 (0.7497) | 0.3949 (0.5976) | added: 2,004 / 3,731 / 1,134 / 95,857 | 17838 |
| ACM | 0.60 | 0.6011 (0.9046) | 0.4996 (0.7524) | added: 5,343 / 9,949 / 3,025 / 255,619 | 17814 |
| DBLP | 0.20 | 0.1977 (0.2675) | 0.1693 (0.2294) | added 307 | 17814 |
| DBLP | 0.30 | 0.3007 (0.4059) | 0.2398 (0.3246) | added 844 | 17838 |
| DBLP | 0.40 | 0.3969 (0.5346) | 0.3039 (0.4109) | added 1,535 | 17814 |
| DBLP | 0.50 | 0.5028 (0.6752) | 0.3803 (0.5135) | added 2,609 | 17838 |
| DBLP | 0.60 | 0.6080 (0.8168) | 0.4643 (0.6267) | added 4,297 | 17814 |

Edge fingerprints and pair counts per graph: `aplus_reachability.md`.

### Manifest row counts (job 17839; placeholder `full` = the experiment files; no baseline sweep output; nothing run)
| | Rows | Runs (seeds 1–5) |
|---|---|---|
| H2S-NN `full`, `no_path_cells` (20 graphs each) | 40 | 200 |
| H2S-NN S2 (10 targeted graphs) | 10 | 50 |
| GCN, HAN (HGB command), SeHGNN, Hetero²Net (published): 20 graphs each | 80 | 400 |
| HetSheaf, CSNN: "needs tuned config" (20 graphs each) | 40 | 200 |
| **Total** | **170** | **850** (= amendment 7's 800 + 50); runnable now 130 rows / 650 runs |

### Baseline configuration mapping (amendment 4; `baseline_configs.py`, values read back from the vendor files by the tests)
| model | HGB setting | HGB value | adapter key | adapter value |
|---|---|---|---|---|
| GCN | `--lr` | 1e-3 (README) | `lr` | 1e-3 |
| GCN | `--weight-decay` | 1e-6 (README) | `weight_decay` | 1e-6 |
| GCN | `--feats-type` | ACM 0 (README); DBLP 3 (`run.py` default) | `feats_type` | ACM 0; DBLP 3 |
| GCN | `--hidden-dim` | 64 (default) | `hidden_dim` | 64 |
| GCN | `--num-layers` | 2 (default) | `num_layers` | 2 |
| GCN | `--dropout` | 0.5 (default) | `dropout` | 0.5 |
| GCN | `--epoch` / `--patience` | 300 / 30 (defaults) | `epochs` / `early_stopping` | 300 / 30 |
| HAN | `lr` | 0.005 (`utils.default_configure`) | `lr` | 0.005 |
| HAN | `weight_decay` | 0.001 | `weight_decay` | 0.001 |
| HAN | `hidden_units` | 8 | `hidden_dim` | 8 |
| HAN | `num_heads` | [8] (one HAN layer) | `num_heads` | 8 (adapter builds `[num_heads]`) |
| HAN | `dropout` | 0.6 | `dropout` | 0.6 |
| HAN | `--num_epochs` / `--patience` | 200 / 100 (`main.py` defaults) | `epochs` / `early_stopping` | 200 / 100 |
| SeHGNN, Hetero²Net | — | published | `configs/baselines/<model>/node_classification/{acm,dblp}.yaml` | as transcribed |
| HetSheaf | — | none (sweep files only) | — | needs tuned config |
| CSNN | — | none published for ACM/DBLP | — | needs tuned config |

No counterpart key (reported, nothing added in `src/`): the optimizer (HGB Adam with L2 decay; adapters AdamW, fixed in code); the early-stopping criterion (HGB validation loss, HAN loss + accuracy; ours validation Macro-F1, protocol); HGB's unseeded validation carve-out (ours deterministic, adapter deviation 1); GCN `--repeat` (ours: seeds as units); HAN `--num_layers` (parsed by `main.py`, unused by its model).

**CSNN published configuration, where looked (none for ACM/DBLP):** `vendor/csnn/exp/scripts/` (amazon, enzymes, minesweeper, questions, roman, squirrel, tolokers); `vendor/Cooperative-Sheaf-NN/README.md` (points to those scripts); `references/cooperative_sheaf_neural_network.pdf` (arXiv, 3 Feb 2026: roman-empire, amazon-ratings, minesweeper, tolokers, questions, small heterophilic sets, Penn94; "ACM" occurs only as the ACM-GCN baseline and in citations); `configs/baselines/csnn/` (graph classification only).

### Tests (CPU, worktree, environment E)
| Job | Head | Result |
|---|---|---|
| 17836 | `0a77d6b` + uncommitted (dev check) | 22 OK (`test_card7_amendments`, `test_card7_perturbation`) |
| 17837 | `9400456` | **27 OK** (+ `test_card7_ablation`), peak 0.47 GB |

`test_card7_amendments` (13): levels and seeds (Phase A seeds kept, 20 distinct); generator fingerprint = loaded H2S-NN graph (DBLP toy, + added facts = + pairs), ACM toy with self-mirrored citations; backfill adds the fields and refuses a changed `link.dat`; **run check:** H2S-NN (`run_h2snn_runs.py --dry_run` through `run_checked.py`, every load logged and matching, summary written) and GCN (`run_hgb_command.py` from the graph folder: the `./data` route, one check, matching, record `configuration: hgb_command` with the mapped parameters); a wrong graph fails on the fingerprint, a wrong working directory fails on the data root before loading, the loader is restored; a launcher that never loads fails; GCN/HAN values equal the vendor README / argparse / `default_configure` values, every mapped key is an adapter key, sources (SeHGNN/Hetero²Net published; GCN/HAN/HetSheaf/CSNN not; no CSNN ACM/DBLP script); figure: points/missing collected from a crafted manifest, one dashed line per dataset × kind at 1 − 1/C. Manifest test: 8 methods per graph, S2 on targeted only, sources per method, every command through the check with the graph's fingerprint, IMDB and 0.80 excluded, 34 rows / 30 runnable on the toy, no out-of-scope arm.

### Deviations (each one line)
- GCN/HAN have a new launcher `scripts/perturbation/run_hgb_command.py` (the published launcher's loop with the mapped parameters) instead of files in `configs/baselines/`, which would have turned their main-table cells into published ones.
- The configuration source is fixed per method as amendment 4 lists it; a later GCN/HAN main-table sweep would not replace the HGB command in the perturbation rows.
- For baselines whose vendor code re-reads the raw files (HAN), the edge fingerprint is of the bundle's graph; the vendor's own parse is covered by checks 1–2 (same root, same `link.dat` bytes), not fingerprinted separately.
- The fingerprint ignores link types and weights (a fact = an unordered endpoint pair); the `link.dat` SHA-256 covers the exact file.

### Decisions for the author
- **Reference point at the original graph:** the budget (800 + 50) has no runs on the original graphs. `figure.py --reference` takes the main-table summaries (seeds 0–4); alternatively runs on the original graphs at seeds 1–5: 8 methods (H2S-NN `full`, `no_path_cells` + 6 baselines) × 2 datasets × 5 seeds = 80 runs (+ 10 for S2). Which one?
- HetSheaf and CSNN rows (40 rows, 200 runs) wait for their ACM/DBLP seed-0 sweeps in the main-table campaign.

The `sha` below is the Phase A+ evidence commit (`0dc936b`: code, tests and evidence of Phase A+; Phase A's was `84d7efe`); the report commit on top of it changes only this file.

branch=card7 sha=0dc936be3e2421a80c8d2caa2a9a3021497823f5 card=docs/H2S-NN_card7_card.md base=894fd7421ce8581d38b967906bde5593e881518a
