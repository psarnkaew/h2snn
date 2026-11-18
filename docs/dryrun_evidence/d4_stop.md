# Card 2 D4 — heterophily statistics: STOPPED (the script does not follow M2)

Card D4: "Run `scripts/heterophily/compute_heterophily.py` as spec §1 and card M2 define it ... If the script does not follow
M2 (e.g. it re-implements a formula), stop this unit and report what differs." Read in full at `e419370` (821 lines) against
`docs/H2S-NN_partM_card.md` M2 and the reference files `vendor/Hetero2Net/hetero2net/homophily.py` (commit 4913394) and
`vendor/H2GB/H2GB/calcHomophily.py` (d3b49db). The script is Part J's (unit J0, HGB node classification); M2 specifies a
different harness (`scripts/heterophily_stats.py` + a module under `src/analysis/`). **No job was run; no value computed.**

| # | M2 requires | `compute_heterophily.py` does (file:line) |
|---|---|---|
| 1 | Datasets: HGB NC 4, the local LP tasks (task-level `H_edge`, MDE on the F17 files), TU GC 4 (1-hop `H_edge` on bonds, label = atom type), H2GB NC (8 incl. PDNS) | HGB NC only: `--datasets choices=list(HGB_DATASETS)` (`:756-757`), loader `load_hgb` = PyG `HGBDataset` (`:380-383`); LP only as the Freebase book-book task-level `H_edge` (`--task-h-edge-freebase`, `:767-780`, `:774`); no TU, no H2GB loader |
| 2 | MLH **and MDE** from Hetero²Net `homophily.py` (MDE on raw features, "–" without target features) | MDE is never computed: Hetero²Net's `attribute_homophily` / `attribute_homophily_matrix` (`homophily.py:4-37`) are not called anywhere; no feature measure in `DatasetResult` (`:345-372`) |
| 3 | Hetero²Net's `M₂` = its **predefined list** where it has one (DBLP APA; ACM PAP, PSP, PTP; IMDB MDM, MAM, MKM; ogbn-mag PFP, PAP), else `get_metapaths` | Always `get_metapaths` (`hetero2net_metapaths`, `:130-148`); the predefined lists are not used. The headline MLH is the mean over **H2GB's** meta-paths (`mlh_h2gb_heterophily_mean`, `:478`), not Hetero²Net's `M₂` |
| 4 | **No published value is used** (no comparison column, no tolerance) | Published values are written into the output: `CITED_H2GB_TABLE2` (`:619-624`), `CITED_HETERO2NET_TABLE2` (`:625`), emitted as `"cited"` in the JSON (`:809-810`) |
| 5 | Reference code unchanged, **no re-implementation of the formulas** | Own code computes reported values where the reference is not called: node homophily from own float64 counts when `G_P` exceeds `--node-ref-max-edges` (`h_node = counts.h_node`, `:450`) and for every IMDB genre (`"h_node": own.h_node`, `source_h_node="own, per genre ..."`, `:546, 568`); `G_P` itself rebuilt by own code (`iter_metapath_chunks`, `:151-166`, "the product sits inside that function"). ℋ² and per-meta-path `h_edge` do come from `calcHomophily` (`:91-122`) with its fill on a clone (5 fill seeds, labelled-only beside it), as M2 asks |

Consequence: running the script "as M2 defines it" is not possible without new code (loaders for LP/GC/H2GB, the MDE
call, the predefined lists, removing the cited values). Writing that harness is a change of the script, not a run;
D4 is reported as a stop item for the design session (a card for the M2 harness, or an explicit acceptance of
Part J's script for the HGB NC rows only). Note for that card: Hetero²Net's `attribute_homophily` is a Python loop over
edges (`homophily.py:9-12`), which is impractical on the H2GB graphs (12–77 M edges) without a vectorised equal-output
call — a decision, since M2 forbids re-implementing the formula.
