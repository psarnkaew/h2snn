# Card 6 — Heterophily harness (M2, M3 heterophily part): ℋ², MLH, MDE for every dataset

Written 2026-11-17 by the design session; author asked to start now (decision 8 of the "Card 2 review" row: a small M2-harness card; Part J's `scripts/heterophily/compute_heterophily.py` is **not** used — reasons in `dry-run` `docs/dryrun_evidence/d4_stop.md`). For session 4. Branch `heterophily` from `v3b` `1867681dc197ab536ad5cf1fbede4a9df9abffab`, worktree `/mnt/data/peeradon/h2snn-ablprep`. Report `docs/H2S-NN_heterophily_report.md`; evidence and outputs `docs/heterophily_evidence/`.

**Binding:** spec v1.24 §1 heterophily paragraphs (MLH, ℋ², multilabel IMDB, meta-path set and graph, MDE) and `docs/H2S-NN_partM_card.md` M2 (and M3's "heterophily of these graphs" line). Where the two differ, the spec wins. **GC:** spec 1.23 (GC paragraph) — no heterophily value for GC; the GC table reports node/edge type counts and the type-mixing ratio only (spec §1's older GC line "1-hop `H_edge`, label = atom type" is superseded; the design session fixes the spec text in 1.25).

**Author 2026-11-17:** a dataset (or a measure on it) that cannot be computed within the limits below is left blank ("–") with the reason; it is not a stop.

## Scope (rows of the output table)
| Task | Datasets | ℋ² | MLH | MDE |
|---|---|---|---|---|
| NC, HGB | acm, dblp, imdb (per genre, then mean), freebase | yes | yes | yes ("–" freebase: no target features) |
| NC, H2GB | ogbn-mag, mag-year, oag-cs, oag-eng, oag-chem, ieee-cis-g, h-pokec, pdns | yes | yes | target type with features |
| LP | acm_pa, dblp_ap, imdb_ma, freebase_bb, ogbn-mag P–P, h-pokec U–U | task-level `H_edge` of the target relation where both endpoint types carry labels | — | per endpoint type that keeps features (F17 files), training graph |
| LP | pubmed, lastfm | "–" (no labels) | "–" | per endpoint type with features |
| GC | mutag, proteins, enzymes, nci1 | "–" (spec 1.23) | "–" | "–"; type counts + type-mixing ratio |

Graphs, splits and features come from the **project's own loaders on `v3b`** (the graphs the trainers see; symmetrised as they build them); record the loader file:line per dataset. OAG files by their verified identity (card 1 / Part M3 check).

## Reference code (called, never edited; no re-implementation)
- MLH, `label_homophily`, MDE `attribute_homophily`: `vendor/Hetero2Net/hetero2net/homophily.py:4-25`; `M₂` = Hetero²Net's predefined list where it has one (DBLP APA; ACM PAP, PSP, PTP; IMDB MDM, MAM, MKM; ogbn-mag PFP, PAP), else `get_metapaths` (`:51-63`). mag-year gets its own `get_metapaths` set.
- ℋ²: `vendor/H2GB/H2GB/calcHomophily.py` `calcHomophily` with its own `extract_metapath` rule; unlabeled targets with H2GB's random fill (`:110-113`), 5 fill seeds (spread reported), labeled-only value beside it.
- `G_P` (meta-path graph) built as each reference builds it; if a reference function takes `G_P` as input, the harness builds `G_P` by sparse matrix product of the loader's typed adjacencies, with de-duplication and self-pair handling as the reference code (record the convention per measure).

## Decisions taken in this card (design session; the author may override in review)
- **D-a, MDE on large graphs.** `attribute_homophily` loops over edges in Python (`homophily.py:9-12`), infeasible on graphs of 10⁷–10⁸ edges. The harness uses the reference function **unchanged** wherever `|E(G_P)| ≤ 2·10⁶`; above that, an **equal-output chunked call** (`src/analysis/heterophily.py`): the same expression per edge, `‖x_r/√deg_src(r) − x_c/√deg_dst(c)‖² / 4` with the reference's `degree(...).clamp(min=1)`, mean over edges, float32 as the reference, edges in chunks. Gate **H0-eq:** on every HGB `G_P` (acm, dblp, imdb meta-paths) the chunked value equals the reference value to `rtol 1e-5` (float32 summation order); a failed gate stops the unit. Both values recorded on those graphs.
- **D-b, limits.** One CPU-only job per (dataset, measure) — **no GPU requested** (GPUs are the cluster's bottleneck). `--mem` ≤ 56G on the 64 GB nodes; wall time ≤ 12 h. Exceeding either → "–" with the reason (peak memory / time reached). g3an01 is not used (author rule 2026-11-17).
- **D-c, no published value**: no comparison column, no tolerance, no cited number in the output (spec 1.15).

## Units (one commit each)
### H0 — Harness and tests (local CPU, no real data)
`src/analysis/heterophily.py` (wrappers + chunked MDE + `G_P` builder + GC type statistics) and `scripts/heterophily_stats.py` (`--dataset`, `--task`, `--measure`, writes one JSON record per call: value(s), per-meta-path values, per-genre values (IMDB), fill seeds and spread, labeled-only value, `M₂` used, code path + vendor commit, harness commit, peak host memory, wall time).
Tests `tests/test_heterophily_harness.py`:
| Test | Expectation |
|---|---|
| Hand graph (4 targets, 2 classes, one meta-path) | MLH, `H_edge`, MDE equal hand-computed values exactly |
| Meta-path sets | `M₂` for acm, dblp, imdb equals Hetero²Net's lists; ogbn-mag PFP, PAP; H2GB's rule includes a same-type relation where one exists (synthetic schema) |
| Chunked MDE = reference | random graphs (several sizes, isolated nodes, multi-edges) `rtol 1e-5`, all chunk sizes |
| IMDB multilabel | per-genre binary labels, mean over genres; exact-match rule only as a separate reproduction field |
| GC type-mixing | hand graph value exact |
| **Mutation** | chunked MDE without the degree clamp → equality test fails; `M₂` from `get_metapaths` for acm instead of the list → set test fails |
Suite: the new tests + golden/mb0 `differing: none` (nothing else touched).

### H1 — HGB NC, local LP, GC (CPU jobs)
All rows of these tasks; gate H0-eq on the HGB `G_P`s first. Output into `docs/heterophily_evidence/stats.json` (append) and `table.md`.

### H2 — H2GB NC and large LP (CPU jobs)
The 8 H2GB NC datasets and the 2 large LP tasks; D-b limits; blanks with reasons. One array, throttled to 3 concurrent jobs.

### H3 — Report
Table (dataset × measure, with "–" and reasons), per-meta-path detail in `stats.json`, cost per row. Last line `branch=heterophily sha=<full head> card=docs/H2S-NN_heterophily_harness_card.md base=1867681dc197ab536ad5cf1fbede4a9df9abffab`.

## Files
`src/analysis/heterophily.py` (+ `src/analysis/__init__.py` if missing), `scripts/heterophily_stats.py`, `tests/test_heterophily_harness.py`, `docs/heterophily_evidence/**`, the report. **Never** `scripts/heterophily/compute_heterophily.py`, `src/baselines/**`, `src/trainers/**`, `src/config.py`, vendors. Anything else: stop and ask.

## Rules and stop conditions
- No model is trained; no metric. Values are dataset descriptors.
- Stop: H0-eq fails; a reference function cannot be called without editing it; a loader would need a change.
- Git: branch from the pinned SHA; one commit per unit; stage by explicit path; never stash/reset others' work; push only `heterophily`; wait for review.
