# J0 pre-registration: heterophily measures (spec v1.7 §1; task card Part J, J0)
Written 2026-11-09 on `part-j` (after `24fe8f7`), before any J0 code, and before the reference code was read.

## 1. Inventory
**Existing code touched by J0: none.** J0 adds new files only:
- `H2S-NN/scripts/heterophily/compute_heterophily.py`;
- `H2S-NN/tests/test_heterophily_metrics.py`;
- the outputs in this folder.

No `src/` file changes, so there are no consumers to list.

**Data read (VERIFIED-RUN, `ls` / `head` / `wc` on 2026-11-09):**
- HGB raw node-classification releases: `H2S-NN/data/{acm/acm/raw/ACM, dblp/dblp/raw/DBLP, imdb/imdb/raw/IMDB, freebase/freebase/raw/Freebase}/` (`node.dat`, `link.dat`, `label.dat`, `label.dat.test`, `info.dat`).
- Test labels are present in every release:

| Dataset | Labeled target nodes (`label.dat` + `label.dat.test`) | Target type size |
|---|---|---|
| ACM | 907 + 2,118 = 3,025 | 3,025 papers |
| DBLP | 1,217 + 2,840 = 4,057 | 4,057 authors |
| IMDB | 1,371 + 3,202 = 4,573 (multilabel) | 4,932 movies |
| Freebase | 2,386 + 5,568 = 7,954 | 40,402 books |

The type sizes come from `cut -f3 node.dat | uniq -c`. The same files are loaded by `src/data/dataset.py:29-43` through PyG `HGBDataset`.

**Length-2 target→target meta-paths visible in `info.dat`** (DERIVED from the relation lists; the exact set used is taken from the reference code in J0.2):
- DBLP: author–paper–author (APA) only. Authors connect only to papers (relations 0 and 3).
- ACM (paper target): PAP, PSP, PTP, and the paper–paper relations cite/ref composed twice (PPP variants). Whether these count is a reference-code question.
- IMDB (movie target): MDM, MAM, MKM.
- Freebase (book target): the relations are directed and mostly one-way (36 relation types, no stored reverses). The length-2 set depends on whether the reference code adds reverse edges.

**Published values to reproduce** (VERIFIED-SOURCE, PDFs in `H2S-NN/references/metapath_search/`):
- **Hetero²Net** (Li et al., TPAMI 47(9) 2025, the version on disk) **Table II**, "MLH (Label)", homophily form, "calculated over the target node type": DBLP 0.87, ACM 0.60, IMDB 0.24 (MAG 0.21, RCDD 0.45). Eq. (7) defines `MLH_A = (1/|𝒫|) Σ_P H(G_P)`, the mean over all length-2 meta-paths from A to A, where "H can be either edge-level or node-level".
- **Source inconsistency, recorded before any computation.** Hetero²Net Table VII (length 2) lists the columns DBLP, IMDB, ACM, MAG with MLH 0.87, 0.60, 0.24, 0.21. That is, ACM and IMDB are swapped relative to Table II. DBLP (0.87) agrees in both tables. The spec cites "Table 1" (the arXiv numbering); the TPAMI PDF on disk has Table II.
- **H2GB** (Lin et al., KDD 2025) **Table 2**, heterophily form; cited, not recomputed:

| Dataset | `H_edge` | `H_node` | `H_adj` | MLH | ℋ² |
|---|---|---|---|---|---|
| ogbn-mag | 0.9205 | 0.9539 | 0.9312 | 0.8731 | 0.8773 |
| mag-year | 0.7909 | 0.7946 | 0.9977 | 0.7718 | 0.9654 |
| IEEE-CIS-G | 0.5917 | 0.5839 | 1.3151 | 0.1352 | 0.9846 |
| H-Pokec | 0.5663 | 0.5667 | 1.1350 | 0.3922 | 0.9488 |

  - ℋ² = `Agg(H_adj(G_P) | P ∈ M_k)`, with length-2 meta-paths and mean aggregation (§3.3.1).
  - MLH = `Agg(H_edge(G_P))` with Agg ∈ {mean, max}. The aggregation used in Table 2 is not stated in the paper, so J0.2 must read it from the code.

## 2. Measurement and expected result
- **J0.3 tests (`unittest`):**
  1. **Toy graph,** computed by hand in the test file before running: two node types, two meta-paths, known `H_edge` and `H_adj`. Ours equals the hand values, and the reference functions equal them too.
  2. **Random graph:** ours equals the reference on one random heterogeneous graph (fixed seed), to 1e-12.
  3. **Mutation:** dropping the class-imbalance correction of `H_adj` makes a targeted test fail.
- **J0.4 gate (validation, before any new number):** DBLP MLH (APA), homophily form = 1 − heterophily, computed on the HGB DBLP release with the reference conventions. **Pass iff the value lies in [0.86, 0.88]** (0.87 ± 0.01).
- **ACM and IMDB:** reported next to both published readings (Table II and Table VII). They are **not** a gate: the source is self-inconsistent.

## 3. Results that would show the implementation wrong
- Any J0.3 test fails, or the mutation is not caught.
- Our function differs from the reference function on the random graph.
- DBLP MLH outside [0.86, 0.88] after the diagnosis allowed by the card:
  - dataset version (node and edge counts against Table II: 26,128 nodes, 239,566 edges);
  - meta-path set;
  - de-duplication, symmetrisation and self-loops, but only *as the reference code does them*.
- A definition is never changed to fit. After 3 diagnosed attempts, J0 stops (card stop condition).

## 4. Rule 6 (reachability and information-route audit)
- **Not applicable.** J0 trains nothing and predicts no label. The J0.4 gate is a deterministic recomputation of a published statistic, with no optimizer and no label route to audit.
- **The risk that matters is a convention mismatch.** Under multiplicity, self-loops or the labeled subset, the same definition gives different numbers. It is handled by taking every convention from the reference code (J0.2, items a–d) and recording it before J0.4 runs.

## 5. Conventions open until J0.2 (to be read from code, not chosen)
- **(a)** MLH aggregation in H2GB Table 2 (mean or max).
- **(b)** For meta-path-induced graphs: de-duplication (multiplicity), symmetrisation, self-loops (every author with at least one paper has an APA self-loop), and how edges to unlabeled targets are treated (IMDB, Freebase).
- **(c)** The meta-path set: all length-2 target→target paths, or listed subsets.
- **(d)** Multilabel handling (IMDB). If it is in neither codebase, J0 stops and lists the candidates without choosing.
