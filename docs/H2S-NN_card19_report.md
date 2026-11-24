# Card 19 report — Heterophily on H2GB's meta-path set (spec 1.31 option A)

Ops session, 2026-11-23 → 2026-11-24. Card: `docs/H2S-NN_card19_card.md` (design-docs 877e08b). Branch `card19` from `v3b` `0b9873d`; `v3b` and the freeze do not move. Evidence: `docs/heterophily_evidence/h2gb_set/`.

## Summary
- **Gates:** G1, G2 and G3 pass.
- **Step 5:** passes.
- **MLH:** taken from card 6's ℋ² records. No graph was loaded.
- **MDE:** computed on H2GB's set for 8 NC datasets and 4 LP tasks.
- **One deviation from the card, approved by the author on 2026-11-24:** a new exact closed form for paths whose `G_P = A Aᵀ`.
  - It is an inclusion–exclusion over shared mid sets.
  - Without it, the large OAG field–field paths (up to 1.6e11 edges) could not finish in the 24 h job limit under D-a.
  - On 5 paths it equals D-a: relative difference ≤ 1.6e-8, with identical edge counts.
  - The code is in the allowed script; `src/` is unchanged.
- **Large changes in MDE from 1.30 to 1.31:**
  - **OAG (×10–15):** H2GB's set adds the cross-type author paths (e.g. write_last → write_first), whose MDE is 83–238.
  - **ACM:** gains the cite/ref paths.
  - **PDNS:** gains the domain–domain paths.
- **Step 4:** H2GB Table 2's MLH was not produced by the released `calcHomophily` on the ℋ² graph. For MAG and OAG it equals our set minus the paper–paper citation path. The paper must not claim to reproduce it.

## Script and tests
- **`H2S-NN/scripts/heterophily/h2gb_set_measures.py`:**
  - modes `--measure mlh|mde`, `--task nc|lp`, `--metapath-index`, `--collect`, `--table`, `--gates`, `--overlap-cliques`;
  - imports `src.analysis.heterophily` and card 6's `scripts/heterophily_stats.py` unchanged;
  - exits with code 3 on a card STOP condition;
  - records carry the commit, job, node, threads, vendor commits, wall time and peak RSS.
- **`H2S-NN/tests/test_card19_h2gb_set.py`:** 3 tests, `Ran 3 tests OK` (job 18778_1, commit 6584e8b).
  - (a) toy cite/ref + writes: the mini-graph MDE equals `het.mde_reference` on each hand-built `G_P`;
  - (b) MLH from a hand-made ℋ² record;
  - (c) added with the author-approved closed form: on a toy with overlapping cliques (k max 5), the closed form equals the reference in value and edge count; a 3-occurrence chunk size gives the same value (chunk carry).
- **Commits on `card19`:**
  - c3e9312: script + tests (a), (b);
  - 694475e: closed form + test (c);
  - 6584e8b: memory fix of the closed form (16G OOM: gathers bounded by occurrences);
  - dfc9a8a: evidence.
- **Records by commit:**
  - MLH: c3e9312.
  - MDE: c3e9312 for the jobs before 2026-11-24 15:20; 6584e8b for the later jobs.
  - The script changes after c3e9312 only add the `--overlap-cliques` route; without the flag the route is unchanged.

## Gates
**G1: PASS.** The new route equals card 6 MDE (≤ 1e-6 relative). Job 18479, commit c3e9312.

| case | card 6 | card 19 | rel. diff |
|---|---|---|---|
| NC DBLP | 0.8004068732 | 0.8004068732 | 0 |
| NC IMDB | 0.5927698861 | 0.5927698861 | 0 |
| NC IEEE-CIS-G | 9.8304e-15 | 9.8304e-15 | 0 |
| LP dblp_ap (paper) | 0.0385793977 | 0.0385793977 | 2.05e-10 |
| LP imdb_ma (movie) | 0.5168308044 | 0.5168307927 | 2.27e-8 |

The LP gate sets equal card 6 `m2`:
- dblp_ap: term / venue / author;
- imdb_ma: director / actor / keyword.

**G2: PASS** (±1e-4). Job 18477.

| dataset | expected | card 19 |
|---|---|---|
| ACM | 0.2621 | 0.262088 |
| DBLP | 0.1278 | 0.127778 |
| IMDB | 0.3247 | 0.324674 |
| Freebase | 0.8464 | 0.846442 |

**G3: PASS.** `git diff 0b9873d card19 -- H2S-NN/src H2S-NN/configs H2S-NN/scripts/perturbation H2S-NN/vendor` is 0 lines. `vendor/` was copied with rsync and is not a symlink.

**STOP conditions:** none triggered.
- Every dataset's H2GB enumeration equals its card 6 h2 record (`card6_h2_paths_equal: true` in every MDE record).
- Empty paths are as expected: OAG 30 of 46 each, none elsewhere.

## Node classification (step 1, `table.md`)
- MLH and ℋ² are in heterophily form (first fill seed).
- MDE uses raw features and is the mean over non-empty paths.
- 1.30 = card 6 (Hetero²Net lists); 1.31 = H2GB's set.

| Dataset | ℋ² | MLH 1.30 | MLH 1.31 | MDE 1.30 | MDE 1.31 | paths | empty |
|---|---|---|---|---|---|---|---|
| ACM | 0.399 | 0.4038 | 0.2621 | 0.7369 | 2.1873 | 7 | 0 |
| DBLP | 0.173 | 0.1278 | 0.1278 | 0.8004 | 0.8004 | 1 | 0 |
| IMDB | 0.737 | 0.3247 | 0.3247 | 0.5928 | 0.5928 | 3 | 0 |
| Freebase | 0.995 | 0.8458 | 0.8464 | – | – (no features) | 8 | 0 |
| ogbn-mag | 0.877 | 0.8731 | 0.8443 | 0.0040 | 0.0033 | 3 | 0 |
| mag-year | 0.965 | 0.7717 | 0.7490 | 0.0040 | 0.0033 | 3 | 0 |
| oag-cs | 0.965 | 0.9803 | 0.9640 | 5.3789 | 63.0844 | 46 | 30 |
| oag-eng | 0.873 | 0.9123 | 0.8681 | 9.2019 | 93.9512 | 46 | 30 |
| oag-chem | 0.885 | 0.9214 | 0.8757 | 6.1501 | 88.5552 | 46 | 30 |
| IEEE-CIS-G | 0.985 | 0.1352 | 0.1352 | 9.83e-15 | 9.83e-15 | 11 | 0 |
| H-Pokec | 0.949 | 0.3851 | 0.3861 | 0.0058 | 0.0056 | 16 | 0 |
| PDNS | 0.787 | 0.4849 | 0.3918 | 0.0020 | 0.0336 | 5 | 0 |

Notes:
- Freebase MLH over its fill seeds is [0.8453, 0.8466]; labeled-only 0.4775.
- PDNS labeled-only MLH is 0.0247.
- ogbn-mag and mag-year share the graph and the features, so their MDE is the same, as in 1.30.

## Link prediction (step 3b, appendix)
MDE on the training graph, per endpoint type with features.

| Task | endpoint type | paths (empty) | MDE 1.30 | MDE 1.31 |
|---|---|---|---|---|
| acm_pp | paper | 7 (0) | 0.7369 | 2.5863 |
| dblp_ap | paper | 3 (0) | 0.0386 | 0.0386 (gate) |
| imdb_ma | movie | 3 (0) | 0.5168 | 0.5168 (gate) |
| pubmed | disease | 4 (0) | 0.0780 | 0.0605 |
| ogbn-mag_pp | paper | 3 (0) | 0.0040 | 0.0035 |
| h-pokec_uu | user | 16 (0) | 0.0058 | 0.0057 |
| dblp_ap (author), imdb_ma (actor), freebase_bb, lastfm | – | – | – | – (features removed, F17) |

## ACM per-path MDE
NC, job 18484_1.

| # | meta-path | MDE | `G_P` edges | card 6 (1.30) | route |
|---|---|---|---|---|---|
| 0 | paper-cite-paper-cite-paper | 4.7349 | 9,402 | – | reference |
| 1 | paper-cite-paper-ref-paper | 1.2052 | 44,587 | – | reference |
| 2 | paper-ref-paper-cite-paper | 2.4257 | 14,540 | – | reference |
| 3 | paper-ref-paper-ref-paper | 4.7349 | 9,402 | – | reference |
| 4 | paper-to-author-to-paper | 2.157297 | 29,767 | 2.157297 (same) | reference |
| 5 | paper-to-subject-to-paper | 0.042723 | 2,217,089 | 0.042723 (same) | D-a + reference |
| 6 | paper-to-term-to-paper | 0.010652 | 9,150,595 | 0.010652 (same) | D-a + reference |

- The three paths shared with card 6 equal card 6 exactly, both in value and in edges.
- The four added paper–paper paths raise the mean from 0.737 to 2.187.

For reference, PDNS per path:
- apex-apex 0.00429
- apex-similar 0.07257
- resolves 0.00199 (the 1.30 value)
- similar-apex 0.07257
- similar-similar 0.01641

## Deviation: closed form for `G_P = A Aᵀ` (author approved 2026-11-24)
**Problem.**
- OAG-CS mp15 (PF_in_L1–L1) is a 6.0e10-edge `G_P`.
- It reached 19 % of the first of D-a's two passes in 9.5 h; projected total ≈ 80 h, more than the 24 h limit.
- OAG-Eng mp01 (L1–L1) and OAG-Chem mp08 (L2–L2) and mp15 (L1–L1, 1.6e11 edges) would also time out.
- Tasks 18484_55 and _87 were cancelled; _140 and _147 were cancelled while still pending.

**Method.** When step 2 is the transpose of step 1, two targets are adjacent iff they share a mid node. With F(p) the mid set of target p, n_S = #{q : S ⊆ F(q)} and y = x/√d:
- degree d_p = Σ_{∅≠S⊆F(p)} (−1)^{|S|+1} n_S;
- Σ_{(r,c)∈G_P} y_r·y_c = Σ_S (−1)^{|S|+1} ‖Σ_{q⊇S} y_q‖²;
- energy sum = (2 Σ_{d_p>0} ‖x_p‖² − 2 Σ y_r·y_c)/4;
- edges = Σ d_p.

It is exact and uses float64 throughout. Cost is Σ_p (2^{k_p} − 1) subset occurrences. On OAG L1–L5, k max ≤ 8, so this is ≤ 2.5e7 occurrences per path. Paths that are not `A Aᵀ` (e.g. PP_cite–PP_cite, the cross-type author paths) keep their card 6 route: H2b, H2c-1, the reference or D-a.

**Gate against D-a.** Jobs 18778 and 18849, commit 6584e8b.

| path | D-a | closed form | rel. diff | edges (both) | D-a time → closed form |
|---|---|---|---|---|---|
| OAG-CS mp08 L3–L3 | 0.4197597905 | 0.4197597973 | 1.6e-8 | 1,018,670,723 | 46 min → 18 s |
| OAG-CS mp22 L2–L2 | 0.0619278716 | 0.0619278725 | 1.4e-8 | 8,544,360,190 | 7.4 h → 67 s |
| OAG-CS mp29 L5–L5 | 1.3508129967 | 1.3508130108 | 1.0e-8 | 54,068,616 | 5 min → 22 s |
| OAG-CS mp36 L4–L4 | 0.7719248987 | 0.7719249074 | 1.1e-8 | 153,002,334 | 12 min → 17 s |
| OAG-Chem mp01 L3–L3 | 0.2000284953 | 0.2000284984 | 1.5e-8 | 3,891,915,033 | 2.8 h → 24 s |

The difference comes from D-a's float32 energies.
- The reported values are the D-a ones wherever D-a finished.
- The closed form is used only for the four paths below. Their values are in `records/parts/` with `code_path` "closed form (overlapping cliques …)".

| path | `G_P` edges | MDE | wall | peak |
|---|---|---|---|---|
| OAG-CS mp15 L1–L1 | 59,786,663,432 | 0.0053989 | 16 s | 11.4 GB |
| OAG-Eng mp01 L1–L1 | 21,635,792,402 | 0.0121248 | 20 s | 8.4 GB |
| OAG-Chem mp08 L2–L2 | 31,052,541,764 | 0.0252735 | 91 s | 16.5 GB |
| OAG-Chem mp15 L1–L1 | 161,262,093,106 | 0.0027831 | 26 s | 16.4 GB |

## Step 4: where H2GB Table 2's MLH comes from (read-only)
**Code.** The only MLH code in H2GB (vendor commit d3b49db) is `H2GB/calcHomophily.py`:
- Meta-paths are enumerated by `extract_metapath` at `H2GB/calcHomophily.py:91-105` (called at `:105`).
- The stricter condition `# if src != dst and src == cur:` at `:100` has been commented out since the file's first commit f12538f (2024-06-05). The active condition is `if src == cur:` at `:101`, so same-type relations (e.g. paper–paper citation) are included. The fill step is at `:113`.
- `h_edge = float(nomin.sum()/denomin.sum())` at `:165`, appended at `:178`, overall print at `:184`.
- It is run by `run/calcHomo.sh:18` (`python -m H2GB.calcHomophily`).

**Paper.** Eq. (1) and Table 2 define MLH = Agg(H_edge(G_P) | P ∈ M_k) with Agg ∈ {mean, max}. §3.3 uses the mean.

**Findings** (Table 2 MLH vs ours):
1. **MAG/OAG.** Table 2 equals the mean h_edge over H2GB's set **with the paper–paper citation path removed**, within 1e-4:

   | dataset | Table 2 | ours without citation path |
   |---|---|---|
   | ogbn-mag | 0.8731 | 0.8731 |
   | mag-year | 0.7718 | 0.7717 |
   | oag-cs | 0.9623 | 0.9623 |
   | oag-eng | 0.8689 | 0.8689 |
   | oag-chem | 0.8724 | 0.8723 |

   This is consistent with the stricter `src != dst` rule at `:100` being active when Table 2 was produced.
2. **IEEE-CIS-G and PDNS** match the full set: 0.1352 = 0.1352; PDNS 0.3916 vs 0.3918, keeping the same-type apex/similar paths.
3. **H-Pokec is unexplained:** Table 2 0.3922 vs:
   - full set 0.3861;
   - without has_friend 0.3851;
   - edge-weighted 0.4236;
   - median 0.4092.
4. Our ℋ² matches Table 2 on every dataset.

**Conclusion.** Table 2's MLH was not produced by the released `calcHomophily` on the ℋ² graph. Our MLH 1.31 follows the released code. The paper should not say it reproduces Table 2's MLH.

## Step 5: perturbation consistency (read-only, no regeneration)
Job 18478, node g2n01, `step5_check.py` → `step5.json`.
- The generator's `compute_heterophily.h2gb_metapaths` (via `ref.load_hgb`) lists the same paths as `het.h2gb_metapaths`:
  - ACM: 7 paths (cite/ref × cite/ref + author, subject, term);
  - DBLP: 1 path (author–paper–author).
- Card 7's original-graph h (`data/NC/{ACM,DBLP}_pert/original/graph.json`) equals the step 1 MLH:
  - ACM 0.26208780539590504;
  - DBLP 0.12777827769279226;
  - absolute difference 0 for both.

## Jobs, wall time and peak memory
- Lab cluster, sbatch only.
- Settings: CPU only, 8 CPUs, `--exclude=g3an01,g2tn01`, 16G unless stated.
- Array 18484: 177 tasks, `%2`, 24 h, `list_runs.txt`.
- Time limits used: 24 h for array 18484 and job 18842; 4 h for arrays 18751, 18778 and 18849.

| job | what | result |
|---|---|---|
| 18474 | tests (pytest) | failed: no pytest in h2snn-b; `run.sbatch` switched to unittest |
| 18476 | tests (a), (b) | OK |
| 18477 | MLH, 12 datasets | 0.07 s, 0.39 GB |
| 18478 | step 5 | pass |
| 18479 | G1 (5 cases) | dblp 2 s / 0.9 GB, imdb 22 s / 3.4 GB, ieee 80 s / 14.4 GB, dblp_ap 196 s / 3.7 GB, imdb_ma 27 s / 1.7 GB |
| 18484 | MDE runs (177 tasks) | 171 records; 4 cancelled (closed form instead); 2 OOM (OAG-Chem mp37/mp45, H2c-1 at 16G) |
| 18740, 18744 | k distribution of OAG PF relations (scratch, not evidence) | k max ≤ 8 |
| 18751 | closed form, first version | tests OK, 3 gate paths OK; 4 OOM at 16G (unbounded gather); superseded |
| 18778 | closed form (6584e8b): tests + 4 gate paths + 4 production paths | all OK |
| 18842, 18849 | OAG-Chem mp37, mp45 at 32G; OAG-Chem mp01 closed-form gate | OK; 17 s, 24.6 GB each; 24 s |
| 18851 | collect, table, gates | gates exit 0 |

Per dataset (MDE; sum and longest are wall time over its records):

| task / dataset | records | sum | longest | peak |
|---|---|---|---|---|
| NC ACM | 1 | 7 min | 7 min | 12.0 GB |
| NC ogbn-mag | 1 | 2.05 h | 2.05 h | 6.3 GB |
| NC mag-year | 1 | 2.08 h | 2.08 h | 6.3 GB |
| NC PDNS | 1 | 1.02 h | 1.02 h | 7.9 GB |
| NC H-Pokec | 16 | 14.3 h | 11.6 h (has_friend) | 7.4 GB |
| NC oag-cs | 46 | 10.5 h | 7.4 h (mp22 D-a) | 13.7 GB |
| NC oag-eng | 46 | 3.2 h | 2.1 h | 11.8 GB |
| NC oag-chem | 46 | 6.9 h | 2.8 h (mp01 D-a) | 24.6 GB (32G retries) |
| LP acm_pp / pubmed | 1 / 1 | 1 min / 1 min | – | 3.2 / 2.2 GB |
| LP ogbn-mag_pp | 1 | 1.94 h | 1.94 h | 5.3 GB |
| LP h-pokec_uu | 16 | 12.1 h | 9.4 h | 6.6 GB |

- Cancelled runs are not counted in the table: 18484_55 ran 9.5 h; 18484_87 ran 1 h.
- Logs: `docs/heterophily_evidence/h2gb_set/logs/`. Lists: `list_*.txt`.
- The closed-form gate records are in `overlap_gate/records/parts/`.

branch=card19 sha=dfc9a8a342ac3380c59109adef845d1dbdbd2be6 card=docs/H2S-NN_card19_card.md base=0b9873d0c19455c8df56fa152492188d8918b62d
