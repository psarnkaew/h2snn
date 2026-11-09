# J0.2 Reference code for MLH and ℋ² (2026-11-09)

## Where the code came from
| Code | Source | Commit | Location |
|---|---|---|---|
| H2GB | `git clone https://github.com/junhongmit/H2GB` (the docs site junhongmit.github.io/H2GB is named in the paper's abstract) | `d3b49db63ad66f8a3282dbe699d50ce10ed3062a` | `H2S-NN/vendor/H2GB/` |
| Hetero²Net | `git clone https://github.com/EdisonLeeeee/Hetero2Net` (URL printed in the paper, TPAMI p. 7861) | `49133947f33ba46ccd3efb869727f8aef3d7254a` | `H2S-NN/vendor/Hetero2Net/` |

- I cloned both repositories. I did not pip-install them, so the `h2snn` environment is unchanged.
- Both folders were the empty clone targets named in `CLAUDE.md`. They are left untracked and not committed.
- `torch_sparse` 0.6.18 now imports in `h2snn` (VERIFIED-RUN), so H2GB's function can be called directly.

## Functions behind the published tables (VERIFIED-CODE)
**H2GB Table 2: `H2GB/H2GB/calcHomophily.py::calcHomophily(data, task_entity, chunk_size=200)`.**
- README `:85-90` names it as the ℋ² function and says "Heterophily = 1 − Homophily".
- `run/calcHomo.sh` runs it on each benchmark config through `python -m H2GB.calcHomophily`, which loads each dataset through H2GB's own loaders.
- `build/lib/H2GB/calcHomophily.py` is byte-identical (`diff`).
- `calcHomoHomophily.py` is a homogeneous variant: it converts the graph with `to_homogeneous`, has no meta-paths, and is not used for Table 2.

**Hetero²Net Table II (MLH): `hetero2net/homophily.py`.**
- `label_homophily(edge_index, y)` `:16-24`: edge homophily.
- `label_homophily_matrix(edge_index, y)` `:39-48`: per-node homophily.
- `get_metapaths(start, edge_types, ...)` `:51-63`.
- **The repository has no script that builds `G_P` or computes MLH.** `get_metapaths` and `label_homophily` are called nowhere (`grep`). How Table II's `G_P` was built is therefore **UNVERIFIED** from Hetero²Net's code.

## (a) MLH aggregation in H2GB Table 2: mean
- `calcHomophily` returns the per-meta-path lists and prints `sum(h_edges) / len(h_edges)` (`:185`) and `sum(h_adjs) / len(h_adjs)` (`:187`).
- There is no max anywhere in the file. Hetero²Net Eq. (7) is also a mean.
- **Conclusion:** both MLH and ℋ² are means over meta-paths. Heterophily form = 1 − homophily, a linear map, so the mean commutes with it.

## (b) The meta-path-induced graph `G_P`, from H2GB (the only code that builds one)
- **Product:** `adj_1 = SparseTensor(edge_index of P[0])` and `adj_2 = SparseTensor(edge_index of P[1])`, both with `value=None` (`:123-128`). `G_P` = the `coo()` pattern of `adj_1[chunk] @ adj_2` (`:146-150`).
- **De-duplication: yes.** The product is coalesced, so each ordered pair `(u, v)` appears once however many intermediates connect it. Multiplicity is dropped.
- **Self-loops: kept.** Nothing removes the diagonal. Every target node with at least one `P[0]` neighbour has a loop `(u, u)`, which always counts as same-label.
- **Symmetrisation:** none inside the function; it uses the stored edge types as given.
  - H2GB's dataset loaders symmetrise at load time, e.g. `H2GB/datasets/mag_dataset.py:103-113`: a same-type relation is made undirected with `to_undirected`, and a cross-type relation gets a `rev_<rel>` reverse.
  - Hetero²Net loads HGB data through PyG `HGBDataset` with `T.ToUndirected()` **commented out** (`hetero2net/dataset.py:16`), i.e. as stored.
  - HGB ACM, DBLP and IMDB already store every relation in both directions (VERIFIED-RUN: an `awk`/`diff` of the pair sets shows each listed relation is exactly the reverse of its partner):
    - ACM 0 = rev(1) (cite/ref), 2 = rev(3), 4 = rev(5), 6 = rev(7);
    - DBLP 0 = rev(3), 1 = rev(4), 2 = rev(5);
    - IMDB 0 = rev(1), 2 = rev(3), 4 = rev(5).
    - ACM cite also contains 8 self-loops.
  - HGB Freebase stores 36 one-way relations.
- **Edges counted:** `num_edges` = directed pairs of `G_P`, loops included (`:152`). `deg` = in-degree in `G_P` (`:153`).
- **`h_edge`** = fraction of `G_P` pairs whose endpoints share a label (`:155-158, 165`).
- **`h_adj`** = `(h_edge − Σ_k p̄_k²) / (1 − Σ_k p̄_k²)`, with `p̄_k = Σ_{v: y_v=k} indeg(v) / num_edges` (`:170-173`). This is Platonov et al.'s adjusted homophily.
- **Unlabeled targets:** labels equal to −1 are **filled with a random class**, `label[mask] = torch.randint(...)` (`:110-113`).
  - The global RNG is not seeded inside, and the fill mutates `data[task_entity].y` in place.
  - PyG `HGBDataset` stores unlabeled single-label targets as −1 and unlabeled IMDB movies as an all-zero row (`torch_geometric/datasets/hgb_dataset.py:181,183`, PyG 2.8.0).
- **Node homophily:** not computed per meta-path by `calcHomophily`.
  - H2GB's homogeneous variant uses PyG `homophily(edge_index, y, method='node')` (`calcHomoHomophily.py:165`): the per-node mean over in-edges, then the mean over **all** nodes, with nodes without in-edges counted as 0.
  - Hetero²Net's `label_homophily_matrix(...)` is the same per-node quantity (`:39-48`) and also handles multilabel.

## (c) The meta-path set: the two references differ when a relation joins a type to itself
- **H2GB:** every 2-hop sequence of stored edge types from the target back to the target, **same-type relations included**. The filter `# if src != dst and src == cur:` is commented out (`calcHomophily.py:100-101`).
- **Hetero²Net:** `get_metapaths` **skips same-type relations** (`if src == dst: continue`, `homophily.py:60`). This matches the paper's lists (§VI-A): DBLP {APA}, ACM {PAP, PSP, PTP}, IMDB {MDM, MAM, MKM}.
- **Effect per dataset (DERIVED from the relation lists):**

| Dataset | Effect |
|---|---|
| DBLP | No same-type relation, so both references give {APA} |
| IMDB | No same-type relation, so both give {MDM, MAM, MKM} |
| ACM | H2GB adds the 4 paper–paper compositions (cite/ref × cite/ref) to PAP, PSP, PTP: 7 paths against 3 |
| Freebase, as stored | H2GB finds only book–and–book–and–book; Hetero²Net finds none. With H2GB's loader symmetrisation, both find many |

## (d) Multilabel (IMDB): defined by Hetero²Net for `H_edge` and `H_node`, by neither codebase for `H_adj`
- **Hetero²Net:** equal labels means the label vectors are **identical** (`(y[row] == y[col]).all(dim=1)`, `homophily.py:20-21, 43-44`).
- **H2GB:** none. `calcHomophily` assumes 1-D integer labels (`label.max() + 1`, `label.bincount`, `:111, 133-134`); a 2-D label tensor fails at `bincount`.
- **Consequence:**
  - **MLH (IMDB) is computable** with Hetero²Net's convention.
  - **ℋ² (IMDB) is not:** adjusted homophily needs class shares `p̄_k`, and no reference code defines them for multilabel targets.
  - Per the card and spec §1, I do not choose. Candidates for the author:
    1. Label sets as classes (label powerset; consistent with Hetero²Net's exact match).
    2. Per-label binary `H_adj`, averaged over the 5 labels.
    3. A Jaccard-weighted agreement with an analogous chance correction.
  - J0 is **not** stopped as a whole, because (d) is defined for MLH. Only IMDB's ℋ² is left uncomputed. This is my reading of the card's stop rule, stated here for the author.

## Conventions used in J0.3–J0.5 (each traced above)
1. `G_P` exactly as H2GB builds it: stored edge types, coalesced product, loops kept.
2. The graph exactly as PyG `HGBDataset` loads it (as Hetero²Net does). Freebase is additionally reported under H2GB's loader symmetrisation, flagged for the author.
3. `h_edge`, `h_adj` (single-label): **H2GB's `calcHomophily` called directly.** `H_node` and the multilabel `h_edge` (IMDB): Hetero²Net's `label_homophily_matrix` / `label_homophily` on the same `G_P`.
4. Meta-path sets:
   - **H2GB set** for ℋ² and MLH-H2GB;
   - **Hetero²Net set** for MLH compared with Table II.
5. Unlabeled targets:
   - **Single-label (Freebase):** H2GB's random fill, seeded, with the spread over seeds reported.
   - **IMDB:** PyG's zero rows as Hetero²Net receives them, plus a labeled-only diagnostic. Neither reference states which subset Table II used, so this is flagged.
