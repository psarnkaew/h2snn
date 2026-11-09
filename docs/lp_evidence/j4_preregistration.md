# J4.0 Pre-registration: loader fixes (J4.2), Freebase link prediction (J4.3), HeaRT negatives (J4.4), IMDB protocol (J4.5)
Written 2026-11-09 before any J4 code. Authority: spec v1.8 §6 (IMDB protocol, Link prediction), changelog F6, F8–F10. Evidence base: `inventory.md`, `j13_loader.md`, `j14_heart.md`.

## Sources quoted (VERIFIED-SOURCE)
### HeaRT
Code `Juanhui28/HeaRT` at `c447cbff4c493b60d14b6544c3c39d3b9c5ddff0`, read in the session scratchpad.

| Parameter | Value | file:line |
|---|---|---|
| Negatives per positive | `--num-samples` default **500**; `k = args.num_samples // 2` = **250 per corrupted side** | `heart_negatives/create_heart_negatives.py:270`, `:123`, `:174` |
| Two sides | 250 pairs `(source, *)`, then 250 pairs `(*, target)`, concatenated | `:220-226` |
| Heuristic 1 | `--cn-metric` default `RA`: `reciprocal_degree = 1 / adj.sum(dim=0)`, `cn_scores = A @ (A * reciprocal_degree)` | `:268`, `:69-75` |
| Heuristic 2 | PPR, Andersen push: `--alpha` 0.15 (teleport), `--eps` 5e-5; push while `r[v] >= alpha*eps*out_degree[v]`, LIFO queue | `:273-274`; `calc_ppr.py:19`, `:76-108` |
| Heuristic 3 | Cosine feature similarity, `sklearn.metrics.pairwise.cosine_similarity` on every non-OGB dataset | `:82-91`, `:241-246` |
| Per-heuristic rank | `rankdata(row[row > 0], method='min')`, reversed so 1 = largest; zeros get max+1, filtered (−1) max+2 | `:94-116` |
| Combination | `--agg` default `min`: `np.min([cn_ranks, ppr_ranks, (feat_ranks)], axis=0)` | `:269`, `:124`, `:141-143` |
| Selection | if `max_greater_zero = num_nodes − #all-zero − #masked ≥ k`: `torch.topk(-combined, k)`; else take the top `max_greater_zero`, then `np.random.choice(zero_ids, k − max_greater_zero)` (**with** replacement, default) | `:145-162` |
| Mask | kept endpoint's neighbours in `adj_t` (validation) or `train_valid_adj` (test), plus both endpoints of the pair | `:178-206` |
| Graph | Planetoid: training positives in both directions; self-loops skipped when reading | `hard_utils.py:110-111`, `:119-123` |
| Seeds | `random`, `numpy`, `torch` seeded 42 | `hard_utils.py:14-19` |
| Metric | `--metric` default `MRR`; HeaRT_small reports Hits@1, 3, 10, 100 (`k_list = [1, 3, 10, 100]`); rank = mean of optimistic and pessimistic rank | `benchmarking/HeaRT_small/main_gnn_CoraCiteseerPubmed.py:278`; `main_gae_CoraCiteseerPubmed.py:109`; `benchmarking/evalutors.py:96-112` |

Paper (arXiv 2306.10453v3, read in J1.4): K = 500 per positive (§4.3); main tables use Hits@10 for Cora/Citeseer/Pubmed and Hits@20 for OGB.
- **K of Hits@K is a metric choice for the later trainer card.** It is not needed for the negative files. We record HeaRT's: Hits@10 on the small datasets.

### L3 (Kovács et al., "Network-based prediction of protein interactions", Nat. Commun. 10, 1240 (2019), section "Degree-normalized L3 predictions", Eq. (1))
> "we assign a degree-normalized L3 score to each node pair, X and Y: p_XY = Σ_{U,V} a_XU a_UV a_VY / √(k_U k_V), (1) where k_U is the degree of node U and a_XU = 1 if proteins X and U interact, and zero otherwise."

**Our use** (spec v1.8 C1): `a` = the target relation's **training** facts as an undirected bipartite graph (binary, de-duplicated); `k` = degree in that graph.
- For a kept source x (type S) and a candidate y (type D): `p = [B D_D^{-1/2} Bᵀ D_S^{-1/2} B]_{x,y}`, where B is the S×D training biadjacency.
- The paths are x–U(D)–V(S)–y.
- The formula is symmetric, so the kept-destination side uses the same matrix, read by column.

## J4.2 Loader fixes
**Inventory of consumers (VERIFIED-CODE):**
- `_load_local_hgb_link_data` is called by `build_link_prediction_dataset` (`link_prediction_dataset.py:784`) and `tests/test_paths_and_facts.py:1210-1218` (which checks a ρ-partner's validation = the partner's, flipped).
- `_split_edge_index_hgb_by_head` is called by the loader (`:382`) and `tests/test_data.py:655-664`.
- `_LOCAL_HGB_LINK_DATASETS` is read by the loader (`:346`, `:783`, `:811`) and named in `config.py:266`, `:275` (comments only).
- Baselines reach the loader only through `build_link_prediction_dataset`.

**Fact found while reading (VERIFIED-CODE, not in J1.3's record).** The current carve runs on **every** relation of `link.dat`, not only the target. Each relation's carved validation leaves the message graph (`:373-396`), and all of them feed `hgb_adjacency_edge_index` (`:835-881`). So about `link_val_ratio` of every non-target relation is absent from the message graph today.

**Change:**
- **(a)** When `link.dat.valid` exists, the validation facts of each relation **in the file** are the file's rows; that relation's training facts are the rest of its `link.dat` rows. Its ρ-partner takes the same split flipped, as now, and a self-inverse relation takes both orientations.
  - A relation **absent** from the file has no validation facts, so all its facts stay in the message graph.
  - Every file row must be a `link.dat` row (HGB format), otherwise raise.
  - Without the file: the current carve, unchanged.
- **(b)** Guard: raise if a relation r′ ≠ target exists whose endpoint types are the target's reversed, while ρ(target) is `None` (the J1.3 reverse-only leak).
- **(c)** `scripts/lp/write_valid_files.py` writes `data/PubMed/link.dat.valid` and `data/LastFM/link.dat.valid` from the current carve, with the presets' `link_val_ratio` (0.10, `config.py:244`, `:255`) and `acm_split_seed` 42.
  - The files contain **every relation's** carved rows, so that reading them reproduces today's split exactly ("no numeric change").
  - **Consequence:** PubMed and LastFM keep today's carve of their non-target relations. The derived tasks, whose file lists only the target relation, keep all non-target facts. **This asymmetry is recorded for the author (§9), not resolved here.**

**Expected / wrong if:**
- PubMed, LastFM: train/val/test positives and negatives identical with and without the file (*wrong if* any tensor differs).
- Derived tasks: loader validation = `link.dat.valid` (895 / 1,768 / 1,330), so the evaluated split is 81/9/10 ±0.5 pp (*wrong if not*).
- The reverse-only fixture raises (*wrong if it loads*).
- Mutation (a), file ignored → the 81/9/10 test fails. Mutation (b), guard removed → the fixture test fails.
- The existing loader tests (`test_paths_and_facts`, `test_data`, `test_lp_derived`) stay green.

## J4.3 Freebase BOOK-and-BOOK (`freebase_bb`)
**Data (VERIFIED-RUN, `j40_data_facts.py`, output `j40_data_checks_output.txt`):** relation 0 has 202,674 rows, 15,036 self-loops, 187,638 non-loop rows of which 120,206 (64.1 %) have their reverse stored, giving **127,535 unordered non-loop pairs**.
- No other book→book relation exists, so guard (b) does not apply.
- ρ(rel_0) = `None`, so the D7 fact layer treats (u, v) and (v, u) as two facts. That is why the split is by unordered pair (F10).

**Build** (`scripts/lp/build_derived_lp.py`, new task):
1. Pairs `{min(u,v), max(u,v)}` of non-loop rows, sorted.
2. `numpy.default_rng(42).permutation`.
3. Test = first `round(0.10 n)` = 12,754 pairs; validation = next `round(0.09 n)` = 11,478; train = 103,303.
4. Rows written:
   - `link.dat.test` = the **stored rows** of the test pairs (1 or 2 per pair; **positives are rows as stored**, as HGB lists rows; PubMed's HGB test likewise lists both orientations of each pair);
   - `link.dat` = the release minus the test rows (both orientations);
   - `link.dat.valid` = the stored rows of the validation pairs, which also stay in `link.dat`;
   - self-loops stay in `link.dat` and appear in no split.
5. Node and info files are copied unchanged.

**Loader:** register `freebase_bb` (8 node types from `info.dat`: book, film, music, sports, people, location, organization, business), plus a dataset flag that excludes self-loops from the training positives. They stay in the message graph as stored facts (card J4.3).

**Tests / wrong if:**
- No unordered pair has rows in two parts.
- No self-loop in any split's positives (train included).
- Proportions 81/9/10 ±0.5 pp by pair.
- Same seed → byte-identical files.
- **Mutation "split by directed fact"** → the reciprocal-leak test fails: a held-out pair's reverse row would be in `link.dat`, which the test checks for every held-out pair.
- The loader's leak checks pass on the built task.

**Not decided by me:** whether the trainer should score (u, v) and (v, u) of a two-row pair as one positive. The files keep rows as stored; noted for the LP trainer card.

## J4.4 HeaRT negatives
**Tasks and heuristics** (spec v1.8 C1–C4):

| Task | Target | Heuristics | Feature cosine |
|---|---|---|---|
| pubmed | disease–disease rel_2 (self-inverse) | RA + PPR | yes (all PubMed types share one 200-d space; same type) |
| freebase_bb | book–book rel_0 | RA + PPR | no (no features) |
| acm_pa | paper→author rel_2 | L3 + PPR | **yes**: verified below |
| dblp_ap | author→paper rel_0 | L3 + PPR | no (334 vs 4,231 dims) |
| imdb_ma | movie→actor rel_2 | L3 + PPR | no (3,489 vs 3,341 dims) |
| lastfm | user→artist rel_0 | L3 + PPR | no (no features) |

**ACM feature space (VERIFIED-RUN, `j40_acm_feature_vocab.py` on the HGB release, output `j40_data_checks_output.txt`):** every author's binary vector equals the element-wise OR of the vectors of the papers they wrote: **5,959/5,959** authors (random-author control: 3/5,959). So paper and author features use one vocabulary, and C2 applies to ACM.

**Finding (VERIFIED-RUN, reported for the author; out of J4's scope to change).** Endpoint features encode the target relation, held-out facts included:
- **ACM:** author = OR of **all** own papers, test papers included.
- **DBLP:** authors with identical paper sets have identical features, 150/150 pairs (random pairs: 2/5,000).
- **IMDB:** actors with identical movie sets have identical features, 1,563/1,563 (random: 15/5,000; `j40_feature_dependence.py`).

A model can read held-out links from these features. This is listed under decisions for the author.

**Graphs:**
- **Message graph:** RA and PPR run on the loader's training message graph (`_load_local_hgb_link_data`'s training edges of every relation, after J4.2). It is made homogeneous over all node types, symmetrised, binary, de-duplicated, with **self-loops removed**, as HeaRT's data reading drops them (`hard_utils.py:110-111`).
- **L3** runs on the target relation's training facts only.
- **Built through the loader**, so C3 (validation = the fixed file) holds by construction.

**Procedure per positive (a, b), per side:**
- **Candidates** = every node of the required type. For example, when the destination is corrupted, every node of the target's destination type; the corrupted endpoint always has the target relation's type on that side.
- **Mask** (score −1): any candidate c with (a, c) or (c, a) a target-relation positive of **any** split (train, validation, test). This is stricter than HeaRT (training neighbours only), as the card requires. Also masked: c = a and c = b (HeaRT `:205-206`). This also excludes self-loop negatives.
- **Rank** each heuristic exactly as `rank_score_matrix`; combine by min; select exactly as `rank_and_merge_node`, with `num_nodes` = the number of candidates of that type.
- **Random fill (C4):** `rng.choice(pool, size, replace=False)`, where pool = zero-score candidates not already selected, so no list holds a duplicate. HeaRT's code can repeat a node, both through replacement and through a top-k part that already took zero-score nodes; that is the C4 change.
  - `rng = numpy.default_rng(42)` per task and split, consumed in positive order.
  - **If a pool is smaller than needed → stop J4.4 and report.**
- **Tie order** in the top-k: `torch.topk` exactly as HeaRT. It is deterministic on CPU, which the same-seed test checks.

**Files.** Under `data/lp_derived/<task>/heart/` (PubMed, LastFM: `data/<name>/heart/`), in raw HGB node ids:
- `heart_{valid,test}_pos.npy` (n×2, rows in the loader's order);
- `heart_{valid,test}_samples.npy` (n×500×2, HeaRT's layout);
- `meta.json` (parameters, counts of filled entries, md5).

**Expected:**
- All pools ≥ 250. The smallest type is LastFM users (1,892) minus an artist's listeners.
- Random fill only in a minority of lists, except possibly where PPR mass is concentrated.
- Counts are reported.

**Tests (`tests/test_heart_negatives.py`; wrong if any fails):**
- No negative is a positive of any split, in either orientation.
- Every corrupted endpoint has the required type.
- No duplicate within a list.
- Same seed → identical files.
- RA, L3 and rank/merge equal hand values on a toy.
- PPR push agrees with power-iteration PPR within the push's error bound.
- **Mutation "heuristics on the full graph"** (held-out facts included) → a targeted test fails. The test checks that the heuristic graph holds no validation or test fact and that negatives equal those from the training-only graph.

## J4.5 IMDB protocol (spec v1.8 §6)
**Expected (VERIFIED-CODE pointers to check):**
- The node-classification split indices come from `label.dat` / `label.dat.test`, so the 359 unlabelled movies are in no index (`src/data/dataset.py`).
- The H2S-NN trainer uses BCE (`trainer.py:251-253`) and `compute_multilabel_classification_metrics` (`trainer.py:260-261`), with a default binarisation that must be `threshold` (logit > 0) (`src/utils/misc.py:93-130`).
- Baselines go through `src/baselines/metrics.py` (`MULTILABEL_PROTOCOLS`, default to check) or their own code.

**Wrong if:**
- an unlabelled movie appears in train, validation or test indices, or in the loss mask;
- a default is `oracle_topk`.

**Adapters:** inventoried (`file:line`) and **reported, not changed** if any uses top-k or its own metric.
