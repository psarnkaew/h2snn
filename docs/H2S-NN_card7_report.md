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

(Phase A+ evidence commit `0dc936b`; report commit `e5dfd0b`.)

## Phase B (V5, V6, V8, amendment 8; 2026-11-19)
Merges (`--no-ff`, no rebase): design-docs `4ffa753` (amendments 1–8) → `10969a8`; `v3b` `365a757` (card 5, H2S-NN freeze) → `359162e`. Only `scripts/`, `tests/` and `docs/card7_evidence/` changed: `git diff 365a757 HEAD -- src configs` is empty. The card 5 hooks are used as they are (`src/analysis/mechanism_hooks.py`). The ops session owns the TSUBAME card: V8 adds three phases to `make_cell_list.py` and one wrapper script; no existing phase, list or slot-script behaviour changes (T9: byte-identical).

### Files per unit
| Unit | Commit | Files |
|---|---|---|
| merges | `10969a8`, `359162e` | design-docs `4ffa753`, `v3b` `365a757` |
| B1 (amendment 8) | `cf8ef7b` | `scripts/perturbation/make_graphs.py` (`--original`, `write_original`), `manifest.py` (`baseline_rows`, reference rows), `figure.py` (reference points, `--h2snn_reference_table`); `tests/test_card7_amendments.py`, `test_card7_perturbation.py` (updated) |
| B2 (V5) | `2ff1a32` | `scripts/analysis/mechanism_common.py` (new), `scripts/analysis/face_effect.py` (new) |
| B3 (V6) | `7fc71b5` | `scripts/analysis/two_regime.py` (new) |
| B4 (V8) | `8640fb2` | `scripts/tsubame/make_cell_list.py` (phases `ablation`, `perturbation`, `timing`), `scripts/tsubame/run_row.py` (new) |
| B5 (tests) | `2b6ce79` | `tests/test_card7_mechanism.py` (new: T6, T7), `tests/test_card7_tsubame.py` (new: T9) |
| B6 (evidence) | `4d00154` | `docs/card7_evidence/phaseb_*`, `targeted_1784{8}.txt`, `targeted_1785{4,5,6}.txt`, `suite_job.sh` |
| B7 (suite) | `ad59bfe` | `docs/card7_evidence/suite_17858.txt` |

### What was built
- **Amendment 8.** `make_graphs.py --original` writes each original graph in the perturbed graphs' layout, `data/NC/<DATASET>_pert/original/` (release files hard-linked; `link.dat` SHA-256 checked equal to the release's; `graph.json` of kind `original` with the reference h and ℋ², the edge fingerprint, C, the ceiling). `manifest.py` adds the six baselines on it: seeds 1–5, the same configuration source and checked launcher as their perturbation rows, run from the original folder (`./data`), `graph_kind` `original`. No H2S-NN rows: `figure.py --h2snn_reference_table` takes `full`, `no_path_cells` and S2 from the ablation configuration (a) table of `stats.py collect` at seeds 1–5 (`full`'s seeds 1–4 there are the main table's runs, configuration (a) by construction; a reference with fewer than 5 seeds is listed as missing). The baselines' reference points come from the manifest's `original` rows. Main-table summaries are never read; the old free-form `--reference` JSON is removed.
- **Shared helpers (`mechanism_common.py`).** A run is reloaded from its folder (`config.yaml` + `checkpoints/best.pt`, `load_trained_model`: the run's own graph and evaluation draw); its row is read from the configuration (`arm_of`). One evaluation *view* = one forward: full batch = the whole evaluation graph; mini-batch = the trainer's own evaluation draw and scope batches (`MiniBatchNodeTrainer.evaluation_cells` / `_batch`, `EVAL_BATCH_TARGETS`). Every analysis forward starts cold (`_warm_start` cleared) so no result depends on the forwards before it. **η_v** is computed with the heterophily reference code (`h2gb_metapaths`, `iter_metapath_chunks`, `labeled_mask`; Freebase symmetrised; IMDB the mean over genres of the binary disagreement): distinct labelled u ≠ v with (u, v) in the union of the M₂ meta-path graphs (the card's wording; see decisions). Quartiles over each graph's test targets, bins (−∞, q25], (q25, q50], (q50, q75], (q75, ∞); nodes without a labelled neighbour excluded and counted.
- **V5 (`face_effect.py`).** F1 flips (`full` vs `no_path_cells`, same seed, every test node) per quartile; F2 margins per quartile for both models (own test-x* centroids; ACM, DBLP, Freebase); F3 with hook (b) on `full` seeds 1–3, at most 2,000 test nodes per (dataset, seed) drawn with `default_rng(0)`, one whole-graph forward per node (mini-batch: the node's own scope, base and knockout alike), node v read only: right→wrong, prediction change, margin change (base centroids). Case study by the card's rule (ACM, seed 1; candidates with a different-class fact or M₂ neighbour that the knockout turns right→wrong; largest margin drop, ties → smaller id): the node, its different-class neighbour, the 3 cells through it with the largest w_f (hook (d)), the 2-D PCA (fitted on all test x* of `full`) of h₀, knockout x*, `full` x* and the centroids, P(true class) with and without the knockout; JSON + PDF.
- **V6 (`two_regime.py`).** R1: for every cell of the evaluation forward(s) with two labelled target ends, T = Q₁⋯Q_L with Q_e (hook (c)) on +1 hops and Q_eᵀ on −1 hops, in path order (as `PathTransportOperator`); ‖T − I‖_F same vs different class (medians, AUROC of separating different-class cells), also by start-node quartile and cell length; fact level (ACM P–P, Freebase B–B) ‖Q_e − I‖_F; τ per run. Cells and facts are processed 2,048 / 8,192 at a time (one hook (c) forward per chunk). R2: e_v (hook (a)) at the test targets; AUROC of e_v for wrong vs correct (positive = wrong), Spearman with η_v, per quartile. R3 (`full`, S2): Ẽ, linear probe (`LogisticRegression(C=1, lbfgs, max_iter=1000)`, one-vs-rest per genre on IMDB; trained on the training targets, macro-F1 on the test targets) on x* and h₀, within/between distance ratio on x* and h₀. Node axis: the original graphs by quartile; graph axis: every perturbed-graph run (tag, h and ℋ² from its `graph.json`). Mean and range over seeds per (dataset, graph, arm).
- **V8.** `--phase ablation --manifest <one per dataset>`: `cells_ablation_tuning.txt` (one H2S-NN tuning line per `tune` row: 100 trials, seed 0, own exp_name and state directory, W&B `h2snn_ablation_t4`) and `cells_ablation_seeds.txt` (one line per `runs` row; a (b) row first writes the arm's tuned file from its sweep's `best_tuned_run0.json` and exits 3 = "waiting" while that file does not exist). `--phase perturbation --manifest`: one line per runnable row (perturbed graphs and the original-graph rows); "needs tuned config" rows are comments. `--phase timing`: one line per (main-table method, dataset), H2S-NN with its tuned file from the `h2snn` phase when that sweep has finished (else the launcher's flagged central point). Rows run through `scripts/tsubame/run_row.py`, because the slot script runs `python <line> --resume` from `H2S-NN/` and the row launchers have no `--resume`, the baselines must start in their graph's folder, and a tuned-baseline row first copies its sweep's trials file. The wrapper drops the trailing `--resume`, skips a row whose summary file exists, sets the working directory, does the copy / tuned file, and runs the launcher.

### Original graphs (amendment 8; job 17859; reference `analyse` on the written folder)
| dataset | h | ℋ² | C | ceiling 1 − 1/C | edge fingerprint (pairs) | `link.dat` |
|---|---|---|---|---|---|---|
| ACM | 0.2621 | 0.3994 | 3 | 0.6667 | `f68aabb4bc28` (273,911) | = release |
| DBLP | 0.1278 | 0.1732 | 4 | 0.75 | `b9c3276f5940` (119,783) | = release |

### Manifest and cell-list counts (job 17859; placeholder `full` = the experiment files; no baseline sweep output; nothing run)
| | rows | runs | runnable now | cell-list lines |
|---|---|---|---|---|
| ablation (4 datasets; Freebase `full_batch` as a placeholder, card 5 D1) | 16 `tune` + 44 `runs` | 16 sweeps × 100 trials; 44 seed rows (`full` (a) seeds 5–10; seeds 1–4 looked up) | all | 16 tuning + 44 seeds |
| perturbation incl. reference | 182 (170 + 12 `original`) | 910 (850 + 60) | 138 rows / 690 runs | 138 (+ 44 comments) |
| timing | 28 | 28 | 28 | 28 |

"Needs tuned config": HetSheaf and CSNN on ACM and DBLP, 44 rows / 220 runs (40 perturbation + 4 reference rows).

### Tests (CPU, worktree, environment E)
| Job | Tree | Result |
|---|---|---|
| 17848 | `359162e` + B1 (uncommitted) | 22 OK (`test_card7_amendments`, `test_card7_perturbation`) |
| 17854 | + B2–B5 (dev) | 2 problems in the tests themselves: the ACM toy cannot reach h 0.60 at `max_add_ratio` 4 (fixture; 2.0 as the other card 7 tests); a T9 check keyed on configuration "tuned", which H2S-NN `full` rows also carry |
| 17855 | dev | 21 of 22; T7's crafted probe data put a test point on the wrong side (test data fixed) |
| 17856 | dev = the content of `2ff1a32`–`2b6ce79` (no file changed after the job) | **56 OK**: T6, T7, T9 (22 new) + `test_card7_amendments`, `test_card7_perturbation`, `test_card5_protocol_hooks`, peak 0.9 GB |

`test_card7_mechanism` (T6, T7; fixture: a toy ACM release with runs of `full`, `no_path_cells` and S2 at seeds 1–2 trained by `run_h2snn_runs.py --dry_run` and reloaded from their folders, plus one `full` run on a perturbed toy graph):
- **T6:** flips counted from crafted predictions per quartile and for `none`; margin sign and value (nearest other centroid, empty class); quartile bins; the case-study rule (largest drop, tie → smaller id, no neighbour → no candidate) and the fixed F3 draw; **F3 with hook (b) zeroes exactly the cells through the chosen node and leaves every other β_f unchanged**, the node read equals a manual knockout forward, and a cold forward afterwards repeats the base logits bit for bit; η_v equals a brute-force count over the reference meta-path graphs; the mini-batch views give the trainer's own evaluation ids, labels and logits; F1–F3 end to end on the trained runs (F1 recounted from the two runs, F3 only on the listed seeds, margin change = knockout − base, the case-study path on a forced drop: a different-class neighbour, ≤ 3 cells through the node in w_f order, PDF).
- **T7:** composition = Q₁ Q₂ᵀ Q₃ … on crafted orthogonal matrices (≤ 1e-12; flipped signs and reversed order differ); **composition from hook (c) = the model's own `PathTransportOperator.apply_holonomy` on every cell (float64, ≤ 1e-12, −1 hops present)**; `r1_cells` with chunks smaller than a group gives the same distances; fact level computed; R2 on crafted inputs (AUROC 1 / 0, Spearman 1, per quartile); R3 on crafted inputs (distance ratio 0.1, probe macro-F1 1 single- and multi-label); R1–R3 end to end on both axes (R1/R2 on `full` only, R3 on `full` and S2, `no_path_cells` skipped, the perturbed-graph run tagged with its h, Ẽ equal to hook (a)).

`test_card7_tsubame` (T9): ablation lines = 8 tuning + 22 seed lines for two manifests, each parsed by its launcher's own `parse_args` (100 trials, seed 0, state directory, project; (b) rows carry the tuning base, the sweep's best file and the tuned file); perturbation lines = the manifest's runnable rows (20 on the toy, incl. 5 reference rows), comments for "needs tuned config", every wrapped launcher parsed, copies only on tuned-baseline rows; timing = 28 lines (14 per dataset); **`baselines-1` and `h2snn` lists and files byte-identical to `365a757`'s script**; `run_row.py`: `--resume` dropped, working directory, copy, done-skip, (b) tuned file waits (exit 3) then is written by the launcher's merge, optional tuned file; one GCN reference line run as the slot runs it (`python run_row.py … --resume` from `H2S-NN/`): the fingerprint check matches from the original folder, the summary is written, a second leg reports "row complete".

**Mutations:** the card lists none for Phase B. The T7 tests check that flipping the hop signs or reversing the order changes the composed transport.

### Real-data smoke (job 17860, CPU, 4 min, peak 2.7 GB; 1-epoch runs, no metric reported)
ACM `full`, `no_path_cells` and S2 (configuration (a) of the experiment file), seed 1, `--dry_run`: trained and reloaded. `face_effect.py` (F3 on 20 nodes): F1, F2, F3 written; the case-study rule found no candidate (no right→wrong among 20 nodes after one epoch). `two_regime.py`: R1 on 64,764 labelled-end cells and 5,343 P–P facts, R2 on 2,118 test nodes, R3 for both arms; 145 s on CPU.

### Suite
**Job 17858** (g3n01, RTX 3090 24 GB, sbatch, one GPU not shared, environment E, `--mem 35G`, peak RSS 28.4 GiB (MaxRSS 29,778,244 KiB; the B7 commit message's "29.8 GB" is that KiB figure read as GB); 38 min; in the worktree itself at the clean head `4d00154`, vendor complete, `vendor/sparse_tools` built in place; `src/` vs `v3b` `365a757`: no difference):
- golden v3b: `compared 1238 tensors in 23 configurations; differing: none` (`nc_map_family_dense` SKIPPED as in every earlier gate: the legacy-removal card removed `map_family: dense`);
- mb0 reference: `compared 136 tensors; differing: none`;
- `python -m unittest discover -s tests -v`: `Ran 879 tests in 2276.416s` → **OK** (no failure, error or skip), including T6, T7, T9 and the T9 byte-identity of the existing phases.

Log `docs/card7_evidence/suite_17858.txt` (commit `ad59bfe`).

### Deviations (each one line)
- V8 rows run through a new wrapper `scripts/tsubame/run_row.py` (`--resume`, working directory, trials copy, tuned file); `sweep_slots.sh` and the existing phases are unchanged. A finished row is recognised by its summary file; an interrupted row runs again from its first seed.
- The ablation sweeps' W&B project `h2snn_ablation_t4` and output `<out>/experiments/ablation` are my choice (the card names none). A (b) seed line waits for `best_tuned_run0.json`; as for the `h2snn` final seeds, run the seeds list after the tuning list has finished.
- Mini-batch only (Freebase if card 5 D1 chooses it): R3's probe needs training-target x*, which the evaluation draw (validation + test targets) does not cover, so the training targets use one draw with the run seed; R1's fact level is not computed (it needs the whole graph); F3 reads base and knockout on the node's own scope.
- R1 uses the cells of the evaluation forward(s): full batch = every cell of the evaluation draw (anchors include training targets); mini-batch = the test batches' scopes.
- Undefined details fixed as: margins and distances Euclidean; distance ratio = mean distance of a test node to its class centroid / mean distance between class centroids; R2 AUROC with positive = wrong; IMDB "correct" = the predicted genre set equals the true set, and "same class" in R1 = identical genre sets.
- Case-study figure: the different-class neighbour shown is a fact neighbour if there is one, else an M₂ neighbour, the smallest id in each case; cells carry hook (d)'s raw relation ids and schema key.
- `figure.py --reference` (free JSON) is replaced by the manifest's reference rows plus `--h2snn_reference_table` (amendment 8).

### Decisions for the author
1. **η_v definition.** The card's wording ("the share of v's target-type neighbours through the M₂ meta-paths") is implemented as distinct neighbours over the union of the M₂ meta-path graphs. On real ACM this degenerates: P–T–P gives a paper a median of 3,024 meta-path neighbours, so η_v ≈ 1 − Σp_c² for almost every paper (job 17862, 2,118 test targets):

   | ACM η_v | quartile edges | bins Q1–Q4 | distinct values |
   |---|---|---|---|
   | union of M₂ neighbours (implemented) | 0.650 / 0.672 / 0.680 | 727 / 714 / 676 / 1 | 7 |
   | mean over M₂ of the per-path share (as h averages over M₂) | 0.200 / 0.300 / 0.406 | 530 / 530 / 528 / 530 | 1,207 |

   Per path: P–cite/ref–P paths 0.13–0.20 mean, P–A–P 0.14, P–S–P 0.35, P–T–P 0.67. The per-path mean is a one-function change in `mechanism_common.eta` (scripts only). Keep the union reading, or switch?
2. **Quartiles with ties.** DBLP has one M₂ path (A–P–A), so both readings agree, but 1,280 of the 1,820 test authors with a labelled co-author have η_v = 0 (and 1,020 have none): the value bins are 1,280 / 0 / 99 / 441. Keep value quartiles (counts reported), or use e.g. an "η = 0" group plus tertiles of η > 0, or equal-size rank groups?
3. **Freebase mode** (card 5 D1) is still open; the ablation manifest takes it as `--freebase_mode`, and the analysis follows the run's own mode.
4. HetSheaf and CSNN rows (44 rows, 220 runs incl. the 4 reference rows) still wait for their ACM/DBLP seed-0 sweeps.
5. For the ops session: the TSUBAME data copy needs `data/NC/ACM_pert/` and `DBLP_pert/` (perturbed and `original` folders) for the perturbation phase, and the three new lists are generated there after the `h2snn` phase has tuned `full`.

No stop condition fired: nothing in `src/` or `configs/` changed (`git diff 365a757 HEAD -- src configs` empty), every loader reads a perturbed or original graph by its data directory, every launcher has a fixed-config path, no arm outside ablation plan v2 was added, and the spec and card agree. Not merged into `v3b`. The sha below is the B7 evidence commit (the suite ran at its parent `4d00154`, which holds every Phase B code and test file); this report is the commit after it.

branch=card7 sha=ad59bfe5c928cf1afa506d7e7bed015f5e2c24ee card=docs/H2S-NN_card7_card.md base=894fd7421ce8581d38b967906bde5593e881518a
