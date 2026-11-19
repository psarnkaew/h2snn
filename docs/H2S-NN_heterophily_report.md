# Card 6 — Heterophily harness: report (H3)

Session S5, 2026-11-19. Card: `docs/H2S-NN_heterophily_harness_card.md` (H0–H3, H2b, H2c at design-docs `4c7c287`).
Branch `heterophily`, merged with `v3b` `894fd742` (TD-1 model change; it contains `87c2e4a`) as `d0b9f62`,
no conflict. Outputs: `docs/heterophily_evidence/table.md` (generated), `stats.json` (per meta-path detail),
`records/` (one JSON per dataset × measure, with job, node, harness commit, wall time and peak memory).

## 1. Gates on the merged head `d0b9f62`

| Job | Content | Result |
|---|---|---|
| 17826 | `tests.test_heterophily_harness` (H0, H0-eq, H2b-eq, H2c) + mutations M3a/M3b/M3c | 22 tests OK; M3a, M3b, M3c killed |
| 17827 | `tests.test_heterophily_harness` + mutations M4 (clique form without self-pairs), M5 (collect divides by every path) | 22 tests OK; M4, M5 killed |

Both are CPU jobs (16G, no `--gres`); logs `h2b_gate_17826.txt`, `h2c/logs/h2c_gate_17827.txt`. The `FAILED
(errors=2)` line in 17826 is the M3c mutant's output, not the suite. `stats.json` / `table.md`: job 17828 on `d0b9f62`.

## 2. Node classification (heterophily form: 0 = homophilic)

| Dataset | ℋ² | MLH | MDE | Fill seeds (ℋ² / MLH) |
|---|---|---|---|---|
| acm | 0.3994 | 0.4038 | 0.7369 | all labeled |
| dblp | 0.1732 | 0.1278 | 0.8004 | all labeled |
| imdb | 0.7373 [0.7343, 0.7387]; lab. 0.7181 | 0.3247 [0.3230, 0.3254]; lab. 0.3120 | 0.5928 | 0–4 / 0–4 |
| freebase | 0.9953 [0.9944, 0.9956]; lab. 0.8821 | 0.8458 [0.8445, 0.8459]; lab. 0.4724 | – (target type has no features) | 0–4 / 0–4 |
| ogbn-mag | 0.8772 | 0.8731 | 0.004011 | all labeled |
| mag-year | 0.9652 | 0.7717 | 0.004011 | all labeled |
| oag-cs | 0.96519 [0.96518, 0.96519]; lab. 0.8851 | 0.98026; lab. 0.9163 | 5.379 | 0–4 / 0–4 |
| oag-eng | 0.87283 [0.87283, 0.87283]; lab. 0.8144 | 0.91230; lab. 0.8637 | 9.202 | 0–4 / 0–4 |
| oag-chem | 0.88550; lab. 0.8854 | 0.92136; lab. 0.9213 | 6.150 | **0 only** / 0–4 |
| ieee-cis-g | 0.9846 | 0.1352 | 9.83e-15 | all labeled |
| h-pokec | 0.94877 (no lab.) | 0.3851; lab. 0.3851 | 0.005758 | **0 only** / 0–4 |
| pdns | 0.78669; lab. 0.0499 | 0.4849 [0.4840, 0.4850]; lab. 0.1231 | 0.001995 | **0 only** / 0–4 |

`[a, b]` = range over fill seeds 0–4 (primary value: seed 0). "all labeled": every target has a label, so the fill
does not act and one pass is run. `table.md` prints ieee-cis-g MDE as `0.0000` (4 decimals); the value is
9.830e-15 (`records/nc__ieee-cis-g__mde.json`).

**MDE scale.** MDE uses the raw target features, so its scale differs across datasets (it scales with the square
of the feature scale; e.g. ieee-cis-g row norms ≈ 0.0145, OAG features give values > 1). Compare MDE within a
dataset (across meta-paths, or a dataset's graph variants), not across datasets.

LP and GC tables: `table.md` (unchanged since H1/H2).

## 3. Per dataset: code path, skipped meta-paths, reuse

| Dataset | Measure | Job | Code path | Empty-G_P paths skipped | Reused from |
|---|---|---|---|---|---|
| ogbn-mag, mag-year | ℋ², MLH, MDE | 17661/17615, 17617/17618, 17634/17640 | H2b closed form (1/3, 1/2, 1/2 G_P) | 0 | – |
| oag-cs | ℋ² | 17664_2 | H2b (1/46 G_P) | **30 of 46** | – |
| oag-cs | MLH | 17388_15 (record job 17494) | reference `label_homophily` | 0 | **H2** (H2b check 17585: rel. diff 8.4e-13) |
| oag-cs | MDE | 17629 | H2b (1/2 G_P) | 0 | – |
| oag-eng | ℋ² | 17664_1 | H2b (1/46 G_P) | **30 of 46** | – |
| oag-eng | MLH | 17388 (record job 17446) | reference `label_homophily` | 0 | **H2** |
| oag-eng | MDE | 17627 | H2b (1/2 G_P) | 0 | – |
| oag-chem | ℋ² | 17766 | H2b (1/46 G_P) | **30 of 46** | – |
| oag-chem | MLH, MDE | 17625, 17631 | H2b (1/2 G_P) | 0 | – |
| ieee-cis-g | ℋ², MLH | 17443, 17438 | reference | 0 | **H2** |
| ieee-cis-g | MDE | 17739 | **H2c-1** clique closed form (11/11 G_P) | 0 | – |
| h-pokec | ℋ² | 17740 (16 tasks) + collect 17768 | **H2c-2** one meta-path per task | 0 | – |
| h-pokec | MLH, MDE | 17393, 17398 | reference / D-a chunked | 0 | **H2** |
| pdns | ℋ² | 17721 | H2b | 0 | – |
| pdns | MLH, MDE | 17389, 17390 | reference / D-a chunked | 0 | **H2** |
| acm, dblp, imdb, freebase | all | 17205–17245 | reference (H0-eq: chunked = reference) | 0 | H1 |

The 30 skipped OAG meta-paths are the `paper–PF_in_Li–field–rev_PF_in_Lj–paper` paths with i ≠ j; their G_P is
empty, and `calcHomophily` skips them (`denomin.sum() == 0`) and averages over the 16 non-empty ones; the harness does
the same (H2b, c74bb20). Lists: `records/nc__oag-*__h2.json` → `h2b.empty_gp_skipped`.

**Seed 0 only (ℋ²):** pdns (author decision B; 5 seeds need ≈ 11.7 h; 17388_18 timed out after 22/25 passes),
oag-chem (17664_3 timed out in seed 2 after 11.5 h, no value; rerun 17766 with seed 0, 7.8 h), h-pokec (H2c-2,
`--fill-seeds 0`). No spread for these three. Seed spread on the other OAG datasets: ≤ 1.2e-5.
Timed-out records: `logs/*_timeout.json`.

## 4. H2c

### H2c-1 — ieee-cis-g MDE by the clique closed form

- **Condition:** each target has at most one mid node on step 1, and step 2 is the transpose of step 1. Then G_P is
  a disjoint union of cliques. All 11 M₂ paths meet it; e.g. card3: 172 transactions without an edge, 103 cliques,
  9.54e9 G_P edges. Survey: `h2c/check/h2c1_survey.json`.
- **Formula:** per clique of size n, energy = (2n·S₂ − 2‖S₁‖²)/(4n), where d = n because the G_P builder keeps
  self-pairs. MDE = Σ energy / Σ n² (float64).
- **Tests:** the closed form equals `mde_chunked` and `attribute_homophily` (1e-6, with and without self-pairs);
  a graph that is not a union of cliques is refused. Mutation M4 is killed (17734, 17737, 17827).
- **Real-data check (17738):** the only finished chunked path that meets the condition is acm paper–subject–paper:
  closed form 0.04272257 vs recorded 0.04272257 (rel. 1.2e-7), edges equal. Every other finished path is refused.
- **Cancellation check (17769):** the closed form equals the cancellation-free Σ‖x−μ‖²/2 per clique within ≈ 1e-9
  relative on all 11 paths. The small value is real: rows have norm ≈ 0.0145 and cliques have up to 89,299 nodes.
- **Result:** 9.830e-15 (job 17739, 35 s, 14.4 GB). The clique form is not used in the HGB gate (`both=True`), so
  H0-eq is unchanged.

| Path (transaction–X–transaction) | MDE | Cliques | G_P edges |
|---|---|---|---|
| card1 | 4.87e-14 | 8,499 | 2.14e8 |
| card2 | 1.51e-14 | 482 | 8.12e8 |
| card3 | 1.46e-15 | 103 | 9.54e9 |
| card4 | 1.56e-15 | 4 | 1.00e10 |
| card5 | 3.50e-15 | 106 | 4.10e9 |
| card6 | 1.51e-15 | 3 | 1.04e10 |
| ProductCD | 2.05e-15 | 4 | 6.49e9 |
| addr1 | 2.71e-14 | 255 | 3.44e8 |
| addr2 | 1.51e-15 | 72 | 6.54e9 |
| P_emaildomain | 2.31e-15 | 59 | 4.09e9 |
| R_emaildomain | 3.31e-15 | 60 | 4.22e9 |

### H2c-2 — h-pokec ℋ², one meta-path per task

- **Run:** `--metapath-index i` runs the existing per-path computation on one meta-path. `h2_collect` averages over
  the paths with a row (the non-empty ones). Array 17740 ran 16 tasks, 8 at a time: CPU, 16G, peak 7.39 GB, longest
  task 4.9 h. Collect job 17768 wrote the value **0.94877**.
- **Test:** parts plus collect equal the whole-graph call. Mutation M5 is killed (17737, 17827).
- **Labeled-only diagnostic not run (`--no-labeled-only`):** only 163 of 1,632,803 targets are unlabeled, and the
  diagnostic would push the longest path past 11.5 h.
- **Correction of my earlier statement:** I said the timed-out run 17388_7 did "10 of 16 meta-paths in > 12 h". In
  fact it finished all 16 paths of seed 0 (≈ 7 h 10 min) and was 63 % into seed 1. Because the harness captures
  `calcHomophily`'s stdout, the only per-path values in that log are tqdm's `h_edge` rounded to 3 decimals.
- **Real-data check (17770):** each part's seed-0 `h_edge`, rounded to 3 decimals, against the value printed by
  17388_7 for the same index. **16 of 16 equal**, so this covers all 16 paths, not 10. `h_adj` (which gives ℋ²)
  was never printed, so it is not compared.

| i | Meta-path (user–…–user) | h_edge (part) | printed | h_adj | task wall |
|---|---|---|---|---|---|
| 0 | has_friend–user–has_friend | 0.59809 | 0.598 | 0.19187 | 4.92 h |
| 1 | life_style | 0.58975 | 0.59 | 0.03734 | 11.5 min |
| 2 | music | 0.59686 | 0.597 | 0.03161 | 6.3 min |
| 3 | cars | 0.51179 | 0.512 | 0.01313 | 2.4 min |
| 4 | politics | 0.56433 | 0.564 | 0.08667 | 1.6 min |
| 5 | relationships | 0.59083 | 0.591 | 0.00300 | 18.3 min |
| 6 | art_culture | 0.66308 | 0.663 | 0.04507 | 1.7 min |
| 7 | hobbies_interests | 0.52662 | 0.527 | 0.00286 | 10.3 min |
| 8 | science_technologies | 0.95388 | 0.954 | 0.12050 | 2.1 min |
| 9 | computers_internet | 0.50579 | 0.506 | 0.00763 | 2.6 min |
| 10 | education | 0.63470 | 0.635 | 0.02859 | 2.2 min |
| 11 | sport | 0.57147 | 0.571 | 0.10140 | 2.5 min |
| 12 | movies | 0.67288 | 0.673 | 0.05895 | 3.1 min |
| 13 | travelling | 0.57146 | 0.571 | 0.02330 | 1.8 min |
| 14 | health | 0.63713 | 0.637 | 0.04489 | 1.9 min |
| 15 | companies_brands | 0.63330 | 0.633 | 0.02281 | 1.8 min |

ℋ² = 1 − mean(h_adj) = 0.94877. Detail: `h2c/check/h2c2_hpokec.json`, parts in `records/parts/`.

## 5. Commits (branch `heterophily`)

| Commit | Content |
|---|---|
| 9917d95 | H2c-1 code, tests, M4 (gate 17734) |
| b35ddda | H2c-2 code, tests, M5 (gate 17737) |
| 50e71c6 | H2c-1 evidence (17739, 17769) |
| 8d47b7c | H2c-2 evidence (17740, 17768, 17770) |
| b3513a6 | H2/H2b evidence (17388, 17585, 17615, 17664, 17721) |
| d6c202d | oag-chem ℋ² seed 0 (17766) |
| d0b9f62 | merge `v3b` 894fd74 |
| (this report) | gates 17826/17827, stats.json / table.md (17828), this report |

Rules kept: CPU-only jobs, no `--gres`, `--mem` ≤ 56G, `--time` ≤ 12 h, g3an01 excluded. Files outside the card's
list: none (the merge brings v3b's own changes). No vendor file edited. The H2c code was written in a separate
worktree while 17664/17721 were running from this checkout.

branch=heterophily sha=d0b9f62872a026c8928f3dc58b8d0affba882ec0 card=docs/H2S-NN_heterophily_harness_card.md base=894fd7421ce8581d38b967906bde5593e881518a
