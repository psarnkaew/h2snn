# J4.1 Heterophily conventions of spec v1.8 §1: record (2026-11-09)
Pre-registration: `j4_preregistration.md`. Values: `results.md`, `results.json` (now under the J4.1 conventions; the J0.5 versions are in git history at `3c63df0`). Run: `j41_run_log.txt`.

## What changed in `scripts/heterophily/compute_heterophily.py`
- **IMDB per genre** (`analyse_multilabel`):
  - `calcHomophily` runs on each binary genre label `1(g ∈ Y_v)`, with −1 for the 359 unlabelled movies, which it fills itself, on a clone.
  - Fill seed `100·s + g`; primary s = 0.
  - MLH and ℋ² are means over the 5 genres and over `M₂`.
  - Own float64 code cross-checks every (genre, meta-path) to 1e-6.
  - Exact label-vector match is kept only as the reproduction row; Jaccard (Zhao Def. 1) on labelled-only pairs is the appendix row.
- **Labelled-only aggregates** (MLH, ℋ²) are reported as diagnostics for every partly labelled dataset.
- **Freebase:** the symmetrised graph is primary; "as loaded" is the flagged comparison (`--freebase-as-loaded`).
- **Every value carries a `convention` string.**

## Tests (VERIFIED-RUN)
- `python -m unittest tests.test_heterophily_metrics` → `Ran 17 tests … OK`: the 11 of J0.3, plus 6 new tests:
  - a multi-label hand toy (per genre: MLH 13/48, ℋ² 407/630; exact match 11/24; Jaccard 35/48);
  - the fill/diagnostic tests (single-label and per genre);
  - the fully labelled case.
- **Mutations** (`j41_mutation_check.py` → `j41_mutation_output.txt`): **7/7 caught**, and the file is byte-identical after restore.
  - The 4 J0.3 mutations are carried over. The wrapper mutation was updated to the new `labels` argument.
  - "Skip the fill" (labelled-only as primary) is caught in both paths.
  - "Exact match instead of per genre" is caught.

## Pre-registered checks against the run
| # | Check | Result |
|---|---|---|
| 1 | ACM, DBLP equal J0.5 | **PASS**, difference 0.0 on MLH, ℋ² and the reproduction row (`j41_compare_with_j05.txt`). Both are fully labelled (`fill_seeds = [0]`, no diagnostic), so the fill does not touch them |
| 2 | Freebase symmetrised, seed 0 = J0.5's flagged row | **PASS**, difference 0.0; the 5-seed lists are identical |
| 3 | IMDB per-genre MLH < exact-match 0.763 | **PASS**: per genre 0.284–0.378; IMDB MLH 0.3247 |
| 3 | Labelled pairs: per-genre agreement ≥ exact match | **PASS** on every genre and meta-path (asserted in code, table in `results.md`) |
| 3 | IMDB fill spread ≤ 0.02 | **PASS**: ℋ² 0.7366–0.7393 (0.0027); MLH 0.3243–0.3255 |
| 4 | Own = reference to 1e-6 | **PASS** (asserted for every genre and meta-path; the run did not raise) |
| 5 | Jaccard ≥ exact match | **PASS** (asserted): 0.6191 ≥ 0.4158, 0.4001 ≥ 0.1780, 0.3917 ≥ 0.1424 |

## Values (heterophily form unless marked)
| Dataset | Convention | MLH | ℋ² | Labelled-only diagnostic MLH / ℋ² | Reproduction row (homophily) |
|---|---|---|---|---|---|
| ACM | fully labelled | 0.2621 | 0.3994 | — | 0.5962 (published 0.60) |
| DBLP | fully labelled | 0.1278 | 0.1732 | — | 0.8722 (published 0.87) |
| IMDB | per genre, fill seed 0 | 0.3247 (0.3243–0.3255) | **0.7373** (0.7366–0.7393) | 0.3120 / 0.7181 | 0.2370 exact match (published 0.24) |
| Freebase | symmetrised, fill seed 0 | 0.8464 (0.8453–0.8466) | **0.9953** (0.9944–0.9956) | 0.4775 / 0.8821 | — (no published value) |

- **IMDB appendix:** Jaccard label homophily 0.4703 (homophily form, labelled-only pairs, mean over `M₂`).
- **Freebase, symmetrised:** Hetero²Net's set is no longer empty (book→X→book through the `rev_` relations). Its exact-match value of 0.1542 has no published counterpart and is not used.

## Diagnoses (values that differ from J0.5, by design)
- **IMDB MLH moves from 0.763 to 0.325.** This is the convention change, not a bug. Under exact match, two movies agree only when all five genre bits are equal; per genre, each bit is compared separately.
- The ℋ² of 0.737 says that, after correcting for genre frequency, agreement per genre is only modestly above chance.
