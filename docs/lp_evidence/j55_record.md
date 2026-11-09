# J5.5 HeaRT negatives regenerated where an input changed (spec v1.9 F12, F13, F16)
2026-11-10. Same script, parameters and seed as J4.4: `build_heart_negatives.py`, 500 per positive, agg min, PPR 0.15/5e-5, seed 42, pair heuristic as coded (F14).

**Command (VERIFIED-RUN):** `python scripts/lp/build_heart_negatives.py --tasks acm_pa pubmed lastfm`. Output: `j55_build_output.txt`.
- **acm_pa:** the author features feed C2 cosine (J5.3).
- **PubMed, LastFM:** the message graph feeds CN, PPR and L3 (J5.4).

| Task | Input changed | Positives val / test | Files (md5 before → after, `j55_heart_md5_{before,after}.txt`) |
|---|---|---|---|
| acm_pa | features (C2) | 895 / 995 (pos files unchanged) | `heart_{valid,test}_samples.npy` changed |
| pubmed | message graph | 1,942 / 8,528 (pos files unchanged) | samples changed |
| lastfm | message graph | 7,301 / 18,567 (pos files unchanged) | samples changed |
| dblp_ap, imdb_ma, freebase_bb | none (no C2: no shared feature space, or no features; graphs unchanged) | — | **all files unchanged** (md5 identical) |

**The unchanged tasks are reproducible from today's inputs.** Rebuilt into a scratch subfolder (`--out-subdir heart_j55_unchanged_check`), every `.npy` of dblp_ap, imdb_ma and freebase_bb is md5-identical to the kept file (`j55_unchanged_rebuild_output.txt`: 12/12 IDENTICAL). The scratch subfolder was then deleted.

## Gates
- **J4.4 properties on the new files:** `j44_real_checks.py` → `j55_real_checks_output.txt`, **all 12 files of the 6 tasks pass**:
  - shape;
  - positives aligned with the loader;
  - no positive of any split in either orientation;
  - corrupted endpoint of the right type;
  - no duplicate in a list;
  - halves;
  - no self-pair.
  - `tests/test_heart_negatives`: 22 OK.
- **No pool too small (C4).** The smallest eligible pool is 1,281 and the smallest fill pool 1,459, against 250 needed.
- **Features-only scorer on the new acm_pa lists** (`j55_acm_leak_new_lists_output.txt`): **equal to Ctrl-G** (difference +0.0000).

  | Split | Scorer and Ctrl-G MRR | Hits@10 | Cosine MRR / Hits@10 |
  |---|---|---|---|
  | valid | 0.0205 | 0.0547 | 0.0671 / 0.1307 |
  | test | 0.0194 | 0.0472 | 0.0534 / 0.1146 |

## Random-fill share per task (F16; `j55_random_fill_share.py` → `j55_random_fill_share_output.txt`)
| Task | valid | test | vs J4.4 |
|---|---|---|---|
| dblp_ap | 54.1 % | 53.5 % | unchanged (files unchanged) |
| imdb_ma | 32.9 % | 32.7 % | unchanged |
| **acm_pa** | **23.5 %** (421 half-lists) | **23.2 %** (461) | **was 0 %** |
| freebase_bb | 7.1 % | 7.5 % | unchanged |
| lastfm | 7.2 % | 6.7 % | same counts (1,097 / 2,597 half-lists) |
| pubmed | 0 % | 0 % | unchanged |

**acm_pa 0 % → 23 % differs from the share in spec F16 ("ACM 0").**
- **Diagnosis** (`j55_acm_cold_start.py` → `j55_acm_cold_start_output.txt`): the randomly filled half-lists number exactly the positives whose kept **author has no training fact**: 421/895 and 461/995.
  - Such an author is cold: it has no L3 path. Its rebuilt feature is the zero vector, so every cosine is 0, and PPR from it reaches no paper above zero. So every candidate paper scores 0, and HeaRT fills the half-list uniformly.
  - Before the fix, the leaked feature (the OR of all the author's papers, held-out ones included) gave these authors non-zero cosine to the papers. Their lists were ranked by the leak.
- **Also measured:** 73 kept papers have no training fact. They are not random-filled: a paper keeps its own features and its other relations.
- This is the honest cold-start case, not a defect. It is reported per F16 and listed for the author.

## Not changed
The RA-variant files (`heart_variant_ra/`) stay on disk, unused (F14). They were built in J4 on the old PubMed message graph.
