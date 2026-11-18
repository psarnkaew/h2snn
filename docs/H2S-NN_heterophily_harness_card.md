# Card 6 — Heterophily harness (M2, M3 heterophily part): ℋ², MLH, MDE for every dataset

Written 2026-11-17 by the design session; author asked to start now (decision 8 of the "Card 2 review" row: a small M2-harness card; Part J's `scripts/heterophily/compute_heterophily.py` is **not** used — reasons in `dry-run` `docs/dryrun_evidence/d4_stop.md`). For **session 5 (S5, new)**. Branch `heterophily` from `v3b` `1867681dc197ab536ad5cf1fbede4a9df9abffab`, own worktree `/mnt/data/peeradon/h2snn-heterophily` (S4 keeps `/mnt/data/peeradon/h2snn-ablprep` for card 5); `H2S-NN/data` a symlink to `/mnt/data/peeradon/h2snn/H2S-NN/data`, vendor clones (git-ignored) copied from `/mnt/data/peeradon/h2snn/H2S-NN/vendor/`. Report `docs/H2S-NN_heterophily_report.md`; evidence and outputs `docs/heterophily_evidence/`.

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

### H2b — Closed form on complete meta-path graphs (added 2026-11-18, author; decisions log row "Card 6 H2b")
**Why:** the H2 logs show that on OAG the paper–field–paper `G_P` is complete: oag-eng 370,624 edges per row = its paper count (job 17388_13), oag-cs 546,704 per row = its paper count (17388_15), so the reference pass enumerates n² pairs (≈ 3·10¹¹) for a value that depends only on class shares and feature moments.
**Rule.** For one meta-path `src→mid→dst` with both ends the target type: `G_P` is complete (every ordered pair, self-pairs included, as the harness's `G_P` keeps them) if some `mid` node is adjacent to all n targets on both steps (exact check from the two adjacencies; the union of the step's stored relations as now). Then, with the labels after the same fill (`fill_like_h2gb` / the vendor fill) and the same features as the reference call:
- MLH and H2GB `h_edge`: Σₖ pₖ², pₖ = nₖ / n; H2GB `h_insensitive` = 0 and `h_adj` = 0 (every node has degree n); return every value the reference returns, in its form.
- MDE (Hetero²Net `attribute_homophily`, degrees n): (mean_u ‖x_u‖² − ‖x̄‖²) / (2n), float64.
No hub node → the reference path as before (no other shortcut). The record states `gp_complete: true` and the hub node id per meta-path. Definitions are unchanged; only the computation differs (DERIVED, checked by the design session with Hetero²Net's own functions on a complete graph n = 37: MLH 0.2607743 both, MDE 0.0603155 both).
**Gate H2b-eq (before any real value is written):** on synthetic graphs with a hub (several sizes, labels with a filled `-1` part, features d ≥ 3), closed form = the reference functions called unchanged (`label_homophily`, `attribute_homophily`, `calcHomophily` all three returned values) to rtol 1e-5; a graph without a hub is refused by the closed form. **Mutation:** drop self-pairs (n(n−1) pairs) → the gate fails. **Real-data check:** let 17388_15 (oag-cs MLH, reference path) finish; the closed form on oag-cs must equal it to rtol 1e-5, else stop.
**Jobs:** cancel 17388_13 (oag-eng MDE) and 17388_14 (oag-eng ℋ²) and the pending tasks of 17388 for ogbn-mag, mag-year and OAG; rerun them through H2b (closed form where a hub exists, the reference elsewhere). At most **2 CPU jobs at once**, `--mem` = measured host peak + 30 % (≥ 16G), so the GPU nodes keep RAM for GPU jobs. A non-hub meta-path whose reference pass would exceed 12 h may run the reference code unchanged on one 24 GB GPU (its device follows the inputs; `chunk_size` is its own argument and does not change the value), one GPU job at a time, after S2's jobs, never the A6000; else "–" with the reason.

### H2c — The two remaining timeouts, same formula, CPU (added 2026-11-18, author; decisions log row "Card 6 H2c")
After H2b, two cells timed out on CPU: **h-pokec ℋ²** (seed 0 alone > 12 h; 10 of 16 meta-paths done; no hub) and **ieee-cis-g MDE** (the transaction–card3–transaction path projected at ≈ 35 h; no hub). The author asked for the fastest route **at the same formula**. Design-session analysis: the delay is one long serial job, not CPU speed; a lab GPU adds queue waits (≈ 25 h seen), a 24 GB memory risk for the whole `G_P`, and new device code, so **no GPU job**. No vendor file is edited. Neither item blocks H3: write H3 as instructed and add the H2c values in a later report section "H2c".

**H2c-1 — ieee-cis-g MDE: exact closed form on a disjoint union of cliques (CPU).**
1. Check on the real graph: every transaction has at most one card3 edge (step 1), so `G_P = A₁A₂` has 0/1 entries and is a disjoint union of cliques (one per card3 value). Report the max card3 edges per transaction and the counts; state whether the harness's `G_P` holds self-pairs.
2. If yes: per clique of size `n`, with `S₁ = Σ x`, `S₂ = Σ ‖x‖²` (float64), the energy sum of `mde_chunked`'s expression is `(2 n S₂ − 2‖S₁‖²) / (4 d)`, `d = n` with self-pairs, `d = n − 1` without; MDE of the path = total / edges. (DERIVED; design session checked numerically, relative difference 1.5e-16 in both cases.) A generalisation of H2b (H2b = one clique holding every target).
3. Test: on synthetic one-mid-per-target graphs (with and without self-pairs as the `G_P` builder makes them) the closed form equals `mde_chunked` and the reference `attribute_homophily` within 1e-6. **Mutation:** drop the self-pair term → fails.
4. **Real-data check:** apply it to every other ieee-cis-g meta-path already finished with `mde_chunked` that satisfies the same condition: closed form vs the recorded value within 1e-5 relative (if none qualifies, one such path of another dataset already computed). Any mismatch → stop.
5. If the condition does not hold: ieee-cis-g MDE is reported "–" (one path ≈ 35 h on CPU); no other route.

**H2c-2 — h-pokec ℋ²: one meta-path per task, in parallel (CPU).**
1. The existing per-meta-path computation (the split path of `h2_measure` / `_per_path_calc`, tested equal to the whole-graph `calcHomophily` call in `c74bb20`) with a `--metapath-index` option in the harness (harness only; the computation is unchanged). Seed-0 fill exactly as now (`fill_like_h2gb`, same seed; assert the labels used equal the fill). Each task writes its per-path row; a collect step takes the mean over the non-empty paths as `calcHomophily` does.
2. Test: on a synthetic graph, per-index calls + collect equal the single whole-graph call. **Mutation:** collect divides by all paths including empty ones → fails.
3. **Real-data check:** the timed-out whole-graph run printed per-meta-path results for 10 paths (seed 0). The split rows of those 10 paths equal the printed `h_edge` / `h_insensitive` / `h_adj` exactly (or to the printed precision). Any mismatch → stop.
4. Jobs: one CPU array task per meta-path (16), no `--gres`, `--mem` = recorded peak of the whole-graph run + 30 % (≥ 16G), **at most 8 tasks at once** (this overrides H2b's 2-job limit for H2c-2 only). A task over the time limit → h-pokec ℋ² is reported "–" with the per-path times.

Report section "H2c": the checks, job ids, peaks, and which code path produced each value. In the paper, both values are labelled as computed by an exactly equivalent form (appendix names the code path), as H2b.

### H3 — Report
Table (dataset × measure, with "–" and reasons), per-meta-path detail in `stats.json`, cost per row. Last line `branch=heterophily sha=<full head> card=docs/H2S-NN_heterophily_harness_card.md base=1867681dc197ab536ad5cf1fbede4a9df9abffab`.

## Files
`src/analysis/heterophily.py` (+ `src/analysis/__init__.py` if missing), `scripts/heterophily_stats.py`, `tests/test_heterophily_harness.py`, `docs/heterophily_evidence/**`, the report. **Never** `scripts/heterophily/compute_heterophily.py`, `src/baselines/**`, `src/trainers/**`, `src/config.py`, vendors. Anything else: stop and ask.

## Rules and stop conditions
- No model is trained; no metric. Values are dataset descriptors.
- Stop: H0-eq fails; a reference function cannot be called without editing it; a loader would need a change.
- Git: branch from the pinned SHA; one commit per unit; stage by explicit path; never stash/reset others' work; push only `heterophily`; wait for review.
