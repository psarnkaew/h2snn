# Card 15 report — acm_pp (ACM paper–cite–paper) replaces acm_pa in the LP table

Card `docs/H2S-NN_card15_card.md` (design-docs cd54121), base `66e9349`, branch `card15`. Amendments in this run:
author test rule 2026-11-22 (no full suite for card 15; targeted modules + two smoke epochs); model-session decision
2026-11-22, option (b) (split acm_pp by unordered pair, self-loops out of the positives).

**Result.** acm_pp is built by unordered pair and is **not flagged** by the card 12 rule. It is wired into the loader,
config, registry and every LP adapter, and acm_pa is out of `registry.DERIVED_LP`. The heterophily values are
written. Both smoke epochs pass. All targeted test modules pass, and `RealSplitTests_acm_pp` passes unchanged.
No stop condition fired: rel 0 has an exact reverse (rel 1), and no reserved file was touched.

## Split (step 1; job 18287, final)

`DerivedTask("acm", 0, "unordered_pair")`, seed 42, val 0.09, test 0.10. The target is rel 0 (paper-cite-paper).
Its exact reverse is rel 1 (paper-ref-paper).

| | pairs | rows (rel 0) |
|---|---|---|
| all (non-loop) | 5,310 | 5,335 (25 pairs cite each other) |
| train | 4,301 | 4,322 |
| valid | 478 | 481 |
| test | 531 | 532 |
| self-loops | – | 8, kept in `link.dat`, in no split, never positives |

- `link.dat` loses 1,064 rows: the 532 test rows of rel 0 and their 532 rel 1 rows.
- No test pair is left in `link.dat`, either as rel 0 in either direction or as rel 1 reversed.
- **F17 removes nothing.** `feature_rule.json` lists no type and no column; widths are paper / author / subject 1,902 and term 0. Paper features are keywords, not derived from citations. Both endpoints are papers, so no column describes "the other endpoint".
- **HGB test negatives:** 532 2-hop and 532 random (`hgb_negatives/`). No head lacks 2-hop candidates.

**Builder change (option b), `scripts/lp/build_derived_lp.py`, `build_task_by_pair`.**
- The function used to refuse a relation with a stored reverse. It now accepts a relation whose stored reverse is exact (more than one exact reverse raises).
- A test pair's target rows (both directions) and its stored-reverse rows leave `link.dat` together.
- Without a reverse (freebase_bb), the code path and `split_meta.json` are as before.
- **Other tasks unchanged:** acm_pa, dblp_ap, imdb_ma and freebase_bb were rebuilt into job scratch. Every `info.dat`, `link.dat`, `link.dat.test` and `link.dat.valid` has the same md5 as on disk (job 18287 log, section 2).

**Loader spec** (`src/data/link_prediction_dataset.py`, `acm_pp`): `exclude_self_loop_positives` and `split_unit: unordered_pair`, as for freebase_bb.

**Superseded first build (by fact; job 18274, commit 62cc9eb):** 5,343 facts, 4,328 / 481 / 534. 2 self-loops fell into test. 4 validation and 2 test positives had their reverse pair in the training message graph (mutual citations). This broke `RealSplitTests_acm_pp`, so the model session chose option (b).

## Degree check (step 2; card 12 L scripts copied unchanged, md5 8b48851 / db67f54)

| task | head / tail | features | other rel. head / tail | val pos deg-0 tail / head | val neg deg-0 tail / head | AUROC all: train / val / test 2-hop / test random | AUROC target pair: train / val | (i) | (ii) | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| **acm_pp** | paper / paper | yes / yes | 3 / 3 | 0.000 / 0.000 | 0.000 / 0.000 | 0.5921 / 0.5611 / 0.5752 / 0.5999 | 0.7837 / 0.7167 | no | no | **no** |
| dblp_ap (ref.) | author / paper | no (emb.) / yes | 0 / 2 | 0.000 / 0.090 | 0.000 / 0.090 | 0.5148 / 0.4855 / – / 0.4808 | 0.5519 / 0.4200 | no | yes | no |

- **Degrees (message graph, all relations), share deg-0 / median.**
  - Tail: validation positives 0.000 / 100; validation negatives 0.000 / 93.
  - Head: 0.000 / 96.
  - dblp_ap reproduces card 12 exactly.
- **Leakage.** No validation or test positive is in the message graph, stored or reversed (by-fact build: 4 / 2 reversed). The 42 reversed training positives are the 21 mutual pairs inside train, as for freebase_bb. Negatives that are known positives: train 8 / 4,322, validation 1 / 481.
- **Evidence:** `docs/card15_evidence/L/`. The first (by-fact) build was not flagged either: AUROC 0.588 / 0.583.

## Files changed (vs the merge commit 03ea4a1)

- **Data scripts.**
  - `scripts/lp/build_derived_lp.py`: the acm_pp task, plus the by-pair change above.
  - `scripts/lp/apply_feature_rule.py`: the acm_pp F17 rule, which removes nothing.
  - `scripts/lp/write_hgb_negatives.py`: the acm_pp target.
- **Loader and config.**
  - `src/data/link_prediction_dataset.py`: the acm_pp spec.
  - `src/configs/config.py`: `LINK_PREDICTION_PRESETS` target `("paper", "rel_0", "paper")`, `resample_each_epoch`; `DATASET_TASK_TYPES`.
- **Registry.** `src/baselines/registry.py`: `DERIVED_LP = ("acm_pp", "dblp_ap", "imdb_ma", "freebase_bb")`. No acm_pa line is generated any more.
- **Adapters.** acm_pp was added next to acm_pa in each supported-dataset set: `csnn`, `h2gformer`, `h2sgnn`, `hetero2net`, `hetsheaf` (and its `_HGB_NC_LP_DATASETS`), `hgb_gnn`, `hgb_han`, `magnn`, `nsd`, `pshgcn`, `rgcn`. `_h2gformer_encoders.DERIVED_SOURCE` now has `acm_pp → acm`. The acm_pa entries stay.
- **Experiment config.** `configs/experiments/acm_pp_h2snn_v3b.yaml` is acm_pa's with `dataset_name`, `target_relation` and `exp_name` changed (step 4).
- **Notebooks.** The five LP baseline notebooks were regenerated by `scripts/generate_baseline_notebooks.py`; the only change is acm_pa → acm_pp.
- **Heterophily and smoke scripts.**
  - `scripts/heterophily_stats.py`: `LP_LOCAL` and `TASK_H_EDGE` entries for acm_pp.
  - `scripts/partn_smoke.py`: `LP_TARGETS` entry for acm_pp.
- **Tests.** acm_pp added to the lists of derived tasks in `test_data_layout`, `test_hgb_oracle`, `test_lp_derived` (`RealSplitTests_acm_pp`), `test_partP_h2gformer` and `test_baselines_hetsheaf`. Two other fixes:
  - `test_hgb_oracle`: the reverse-relation lookup now excludes the target itself. A paper–paper target found 2 candidates.
  - `test_partP_derived_lp.FitTests`: now runs on a tiny acm_pp built from the synthetic ACM release, because acm_pa left the registry. A tiny dblp_ap does not work here: it has no 2-hop test set, so `test_2hop_auroc` is None.
  - `test_lp_valid_files.test_derived_tasks_evaluate_81_9_10` is left as it was. It compares loader rows with by-fact counts, so freebase_bb and acm_pp do not belong in it.
- **Evidence.** `scripts/diagnostics/card15/` (job scripts, the copied L scripts, `smoke_one_epoch.py`), `docs/card15_evidence/`, and the acm_pp records in `docs/heterophily_evidence/`.
- **Not changed.**
  - `scripts/tsubame/{make_cell_list,host_memory}.py` and their tests (ops session; the cell list still names acm_pa).
  - The HeaRT negatives (`build_heart_negatives.py`; nothing in `src/` reads them).
  - The J5-era scripts `lp_feature_rules.py` and `rebuild_lp_features.py` (superseded by F17).
  - The reserved files (`anchored_solve.py`, the path/WISP samplers, `scopes.py`, `src/trainers/`).

## Adapter choices for an ACM paper–paper target (each adapter's own rule)

- **MAGNN** (`magnn.py:287-311`, `lp_sides`): both sides start at paper. Each side takes the ACM listed paths (HGB table: PAP, PSP, PcPAP, PcPSP, PrPAP, PrPSP). A path is masked when its first or last hop is the target relation (`magnn.py:266`): PcPAP, PcPSP, and PrPAP, PrPSP, whose rel 1 hop resolves to rel 0 (next section).
- **H2G-former** (`_h2gformer_encoders.py:105-120`): through `DERIVED_SOURCE`, the ACM table; it is used only when `metapath_pe` is on.
- **HAN and H2SGNN** (`_h2sgnn_lp.py:92-137`): every length-2 meta-path whose outer ends are target types. For acm_pp the block holds papers only (P–X–P).
- **HetSheaf:** the HGB-NC branch (`hgb_nc_transform`).
- **Hetero2Net (local LP), GCN, GAT, NSD, CSNN, PSHGCN, R-GCN:** no meta-paths; the relation graph only.

## Heterophily (step 5; job 18287, the card 6 harness as for acm_pa)

| task | task-level H_edge | MDE per endpoint type (training graph) |
|---|---|---|
| acm_pp | **0.1203** (1 − label homophily of paper-cite-paper; all papers labelled, 1 fill seed; 5,343 edges, 8 loops) | paper: **0.7369** |
| acm_pa (old record) | – (authors unlabelled) | author: – (F17); paper: 0.8654 |

- **Why H_edge exists for acm_pp.** The harness defines `TASK_H_EDGE` as "the LP tasks whose two endpoint types carry labels". Both acm_pp endpoints are papers, so acm_pp qualifies.
- **Why MDE differs from acm_pa.** MDE uses Hetero²Net's ACM list (PAP, PSP, PTP). The difference comes from PAP: acm_pp's training graph keeps every authorship, while acm_pa held its test authorships out (PAP 2.157 vs 2.543).
- **Harness version.** The PSP value now comes from the H2c-1 closed form (the harness changed after the acm_pa records). It equals the chunked value of the old records: 0.0427226 in both.
- **No change from the rebuild.** Both values are the same for the by-fact and the by-pair builds: H_edge uses the full relation, and MDE's paths do not use citations.

## Tests

| job | what | result |
|---|---|---|
| 18283 → 18287 | smoke, 1 epoch on acm_pp (`partn_smoke.py` checks, epochs = 1): H2S-NN and GCN | final run 18287: losses finite, no stop condition, no collapse, parameters moved (H2S-NN 34/34, GCN 12/12) |
| 18284 | full suite (submitted before the author's test rule; ran to the end, 54 min, 28.2 GiB peak) | 987 tests, 16 failures, all from the acm_pa → acm_pp switch: 5 stale notebooks, 9 `FitTests` (tiny acm_pa no longer in the registry), oracle reverse lookup, `RealSplitTests_acm_pp` (mutual citations) |
| 18285 | 21 modules listing acm_pa / acm_pp (LP, derived-task, registry, adapter, tsubame), notebooks regenerated | 265 tests: 9 `FitTests` (tiny dblp_ap, no 2-hop set) + `RealSplitTests_acm_pp` failed, rest OK |
| 18286 | `test_partP_derived_lp` (FitTests on tiny acm_pp) | 5 OK |
| 18287 | after option (b): `test_lp_derived`, `test_hgb_oracle`, `test_lp_valid_files`, `test_heart_negatives`, `test_data_layout`, `test_partP_derived_lp` | 88 tests; `RealSplitTests_acm_pp` OK; 1 failure: acm_pp in the by-fact row-count test (removed, see above) |
| 18288 / 18289 | `test_lp_valid_files` | 18288 read the test file before the edit reached the node (NFS; traceback line from the old file); 18289: 11 OK |

**Memory.** Measured peaks were 1.3 GB (build, smoke), 2.9 GB (heterophily) and 3.8 GB (option b job). Each later job asked for at most its peak + 30 %, rounded up: 6–8 G. The full suite asked for 36 G (card 8 suite peak 27.7 GiB + 30 %).

## MAGNN: ACM cited-by meta-paths read in the citing direction (no fix)

**Mechanism.**
- The LP graph keeps one relation per tied pair. For ACM, the fact layer keeps `paper-rel_0-paper` (cite) and drops its exact reverse `paper-rel_1-paper` (ref). MAGNN builds its graph from these kept relations (`TypedGraph.from_global(... data.relation_tuples ...)`, `magnn.py:588`).
- The ACM listed paths PrPAP and PrPSP (`config.py:527-532`) start with a `paper-rel_1-paper` hop. `TypedGraph.hop` (`magnn.py:195-210`) does not find rel 1. It is not a `rev_` name, so it falls to the fallback at `magnn.py:205-209`: the one remaining relation between the two types. That is rel 0, returned **forward**, because `self.relations[r][0] == a` holds for any paper–paper relation.
- So the "cited-by" hop walks the citing direction. PrPAP and PrPSP become copies of PcPAP and PcPSP: same edges, same edge-type ids. MAGNN gets two duplicated schemas instead of the reverse direction.

**What it affects.**
- **MAGNN LP: acm_pa and acm_pp.** Both use `listed_paths` for derived tasks (`magnn.py:300-304`). This is a silent duplication, no error. Other LP tasks are not affected:
  - DBLP and IMDB have no same-type relation in their listed paths.
  - freebase_bb has no stored reverses (nothing is dropped), and its `-` hops resolve through `rev_`.
  - LastFM uses MAGNN's own lists.
- **MAGNN NC: not affected.** ACM NC is not wired for MAGNN (`registry.py:174`, `:215`: published number). Freebase NC uses MAGNN's own lists.
- **H2S-NN: not affected.** Its fixed-literature-schema row resolves hops through the fact layer (`src/data/schemas.py:89`, `layer.resolve`: a dropped inverse becomes its kept partner with the direction flipped).
- **H2G-former, related but different (code reading only, not run).**
  - On acm_pa / acm_pp with `metapath_pe = True` (in its search space, `h2gformer.py:229-231`), the ACM walk schema uses hops that the LP walk graph does not have: `paper-rel_1-paper` and `paper-rel_2-author`. The graph holds the kept relations plus their `rev_` copies (`_h2gformer_encoders.py:159-165`).
  - So `metapath2vec_embeddings` would raise `KeyError` (`_h2gformer_encoders.py:192-195`), not duplicate paths.
  - ACM NC for H2G-former uses PyG's HGBDataset, which stores all eight relations, so NC is not affected.

## Other notes

- `scripts/tsubame/make_cell_list.py:62,94` and `host_memory.py:48` still list acm_pa; that belongs to the ops session. The TSUBAME cell list and dashboard need acm_pp added before any acm_pp run there.
- No acm_pp HeaRT negatives were built (spec 1.28: LP random only; nothing in `src/` reads HeaRT files).
- Data written to the shared root: `data/LP/lp_derived/acm_pp/` (the by-pair build, F17 record, `hgb_negatives/`). acm_pa's folder stays on disk, untouched.

## Jobs

| job | node / GPU | what | wall | peak |
|---|---|---|---|---|
| 18274 | g2n01 (CPU) | by-fact build (superseded) | 19 s | 1.3 GB |
| 18279 | (CPU) | degree check, by-fact build (superseded) | < 1 min | — |
| 18282 | g3n01 (CPU) | heterophily, by-fact build | 49 s | 2.9 GB |
| 18283 | g2n01, 2080 Ti | smoke, by-fact build | 34 s | 1.3 GB |
| 18284 | g2n01 | full suite | 54 min | 28.2 GiB |
| 18285 | g2n01 | targeted modules + notebook regeneration | 8.8 min | 4.8 GiB |
| 18286 | — | `test_partP_derived_lp` | 1.3 min | 0.5 GB |
| 18287 | g3n01, 3090 | option (b): build, md5 check, degree check, heterophily, smoke, targeted tests | 4.4 min | 3.8 GiB |
| 18288 / 18289 | g3n01 | `test_lp_valid_files` | < 1 min | 1.1 / 2.9 GB |

branch=card15 sha=0dff044491b9057554c18a2e2702b7dedb78d68d card=docs/H2S-NN_card15_card.md base=66e9349b79021784a08b37c5f8be0798c0260a04
