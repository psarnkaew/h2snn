# J4.0 Pre-registration: heterophily conventions (J4.1) and Freebase task-level H_edge (J4.3)
Written 2026-11-09 before any J4 code. Authority: spec v1.8 §1 (Heterophily), changelog F7, F10, F11. Evidence base: `results.md`/`results.json` (J0.5), `j02_reference_code.md`.

## Sources quoted (VERIFIED-SOURCE; downloaded to the session scratchpad, outside the repo)
**Zhao et al., "Multi-label Node Classification On Graph-Structured Data", TMLR 09/2023, arXiv 2304.10398v4, §3 "Label homophily", Definition 1:**
> "Given a multi-label graph G, the label homophily h of G is defined as the average of the Jaccard similarity of the label set of all connected nodes in the graph: h = 1/|E| Σ_{(i,j)∈E} |𝓛(i) ∩ 𝓛(j)| / |𝓛(i) ∪ 𝓛(j)|."

(The PDF text layer drops the set symbols; the formula is read from the rendered line.)

**H2GB `calcHomophily`** (`vendor/H2GB/H2GB/calcHomophily.py`, commit `d3b49db`, re-read today):
- `:110-113`: `num_classes = label.max() + 1; mask = label == -1; label[mask] = torch.randint(0, num_classes, size=(mask.sum(),))`. This is the fill, in place, using the global RNG.
- For a binary genre label (0/1, −1 unlabelled), `num_classes` = 2, so an unlabelled movie gets 0 or 1 with probability ½ each.

## J4.1 Conventions to implement (fixed now; none is chosen after seeing values)
| Item | Convention | Source |
|---|---|---|
| (a) IMDB labels | For each genre g = 0…4: `y_g[v] = Y_v[g]` for the 4,573 labelled movies; `−1` for the 359 unlabelled movies (all-zero rows in PyG). `H_edge`, `H_adj` per meta-path from `calcHomophily` on `y_g`. IMDB MLH = mean over genres of (mean over `M₂` of `H_edge`); ℋ² = the same with `H_adj`. Heterophily form `1 − h`. Per-genre values are reported | spec §1 "Multilabel (IMDB)" |
| (b) Fill | `calcHomophily`'s own fill, on a clone (J0.5 wrapper). **Fill seed s ∈ {0,…,4}; the primary value is s = 0.** Single-label datasets: `torch.manual_seed(s)` before the call, as in J0.5. IMDB: genre g under fill seed s uses `torch.manual_seed(100·s + g)`, so each genre gets an independent fill and every (s, g) seed is distinct. The spread is min–max over s | spec §1 "Unlabeled target nodes" |
| (b) Diagnostic | Labelled-only row (Freebase, IMDB): `G_P` restricted to pairs whose two endpoints are labelled, computed by own float64 code. Diagnostic only, never primary | spec §1 |
| (c) Freebase graph | **Primary = symmetrised** as H2GB's loaders do (`symmetrise_like_h2gb`, `mag_dataset.py:103-113`: same-type relations `to_undirected`, cross-type relations get `rev_`). The "as loaded" row is kept as a flagged comparison | spec §1 "Meta-path set and graph" |
| (d) Meta-paths | H2GB's full `M₂` for every dataset. The Hetero²Net set and exact match stay only as the reproduction rows (ACM 0.596, DBLP 0.872, IMDB 0.237) | spec §1 |
| (e) Appendix | Zhao Definition 1 on each `G_P` of `M₂` (as `calcHomophily` builds it: coalesced, self-pairs kept, J = 1 on a self-pair), mean over `M₂`. **Only pairs with both endpoints labelled**, because the Jaccard of two empty label sets is 0/0 (undefined), and the 359 unlabelled movies have empty sets. Reported in homophily form, like the source | Zhao Def. 1 |

**Confirm:** ACM and DBLP have no unlabelled target (count of `−1` = 0), so the fill does not touch them, and their J4 values must equal J0.5's exactly.

## Expected results and what would show the implementation wrong
1. **ACM, DBLP:** MLH and ℋ² equal J0.5 to 1e-12 (0.262/0.399, 0.128/0.173). *Wrong if* they differ, which would mean a convention leaked into fully labelled data.
2. **Freebase symmetrised, seed 0:** equals J0.5's flagged row (MLH 0.846, ℋ² 0.995) to 1e-12. Spread over 5 seeds small (≤ 0.01 on ℋ²). Labelled-only diagnostic about 0.89, as computed in J0.5 for "as loaded"; the symmetrised value is not yet known. *Wrong if* seed 0 differs from J0.5.
3. **IMDB per genre:**
   - The binary per-genre `H_edge` is **lower** (heterophily form) than the exact-match MLH 0.763, because two movies agree on one genre far more often than on the whole set. Expected for every genre; with the fill it is not a theorem (the exact-match row treats unlabelled movies as all-zero rows, the per-genre row fills them).
   - **Hard check, labelled-only pairs:** per-genre agreement ≥ exact-match agreement on the same pairs (exact match implies agreement on every genre). *Wrong if* this fails for any genre and meta-path.
   - ℋ² is not predicted.
   - The fill spread is small (7.3 % of movies filled; expected ≤ 0.02).
4. **Own code vs reference:** per meta-path and genre, own float64 `h_edge` and `h_adj` equal `calcHomophily`'s to 1e-6 (the J0 cross-check). *Wrong if* any differs.
5. **Jaccard (appendix):** on labelled-only pairs, mean J ≥ the fraction of exact matches on the same pairs (J = 1 exactly when the sets are equal, J ≥ 0 otherwise). *Wrong if* this inequality fails.

## Tests (`tests/test_heterophily_metrics.py`, added before the real run)
- **Hand toy, multi-label.** The P-A-P/P-cite-P toy with 2-genre labels. The pre-computed per-genre `h_edge`, `h_adj`, the means, and the exact-match value differ, so the three rules give different numbers.
- **Mutation "skip the fill" (labelled-only as the primary value):** caught by a toy with unlabelled nodes where the filled reference value and the labelled-only value differ.
- **Mutation "exact match instead of per genre":** caught by the hand toy.
- **Jaccard hand values** on the toy.
- **Existing 11 tests stay green.**

## J4.3 task-level `H_edge` (Freebase BOOK-and-BOOK)
- **Graph:** relation 0 (BOOK-and-BOOK), all stored rows, symmetrised like (c) (`to_undirected`, coalesced, self-loops kept), on books only.
- **Value:** `H_edge` = Hetero²Net `label_homophily(edge_index, y)` (the reference edge-homophily function) on the labels after `calcHomophily`'s fill (`h2gb_random_fill`, seed s), cross-checked with own code to 1e-6. Fill seeds 0–4, primary s = 0.
- **Labelled-only diagnostic:** edges whose two endpoints are labelled.
- **Expected:** heterophily form, near the Freebase meta-path values for the filled row (dominated by random labels: about 1 − 1/7 ≈ 0.86 if labels were uniform over 7 classes; the class shares are not uniform, so not a sharp prediction). The labelled-only value is lower.
- **Wrong if:** the reference and own code differ, or the value changes between two runs with the same seed.

## Not stop conditions
A value that differs from the expectations above (except the exactness checks 1, 2 and 4) is recorded with a diagnosis, not adjusted.
