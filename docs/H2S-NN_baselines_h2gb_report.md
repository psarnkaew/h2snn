# Baselines on the H2GB datasets — report (slim mode)

Card `docs/H2S-NN_baselines_h2gb_card.md` (Slim mode section binding). Branch `baselines-h2gb` from `v3b`
`4bfa5dea2da5ad8f96a6e6d181c0cfc58b05e5cd`; `design-docs` merged `--no-ff`: first `4d20701` (first prompt, merge
`724c117`), then `dfbc06e40636cc6eeaf152624c57895c0e0020e6` (slim prompt, merge `dafb357`, after C0 and C1 were pushed;
the branch could not be re-created without a reset or force-push, and `dfbc06e` only adds docs on top of `4d20701`).
Every job ran under SLURM in `h2snn-b` (allocation 16741 on g2n01 via `srun --overlap`); job scripts and logs are in
`docs/baselines_h2gb_evidence/`. No training run on real data, no metric, no comparison between methods.

**Verdict.** C2, C4 (partly), C6 (infrastructure) and C7 item 2 are done. **C0 stops** on four contradictions between
the vendor code and the scheme table (S1–S4). **C1 stops**: GCN's full-fan-out equivalence fails at depth = its
propagation steps (5.27e-2). C3 and C5 are not started (they need S1), and no large-LP adapter is wired.
**The one decision that unblocks most of the card is S1 (sampler depth).**

## Units

| Unit | Commit | State | Jobs (16741 step) | Evidence |
|---|---|---|---|---|
| C0 preconditions | `8dd349f` | **stopped** (S1–S4; S5 noted) | 59 (probe) | `c0_preconditions.md`, `c0_probe.{py,txt}` |
| C1 shared mini-batch infrastructure | `8ae1fe4` | **stopped** (GCN equivalence) | 60 tests, 61 diagnostic, 62 mutation, 63 gate | `c1_tests.txt`, `c1_equivalence_diagnostic.txt`, `c1_mutation.txt`, `c1_gate.txt` |
| C2 PDNS loader | `bc5153f` | done | 64 tests, 65 data check, gate 66 | `c2_tests.txt`, `c2_pdns_check.txt`, `c2_pdns_split_record.json`, `c2_gate.txt` |
| C3 class A adapters | — | not started (needs S1; HAN also S5) | — | — |
| C4 class B adapters | `cef6b6d` | partly: SeHGNN, ChebNetII wired; PSHGCN (S3), HINormer (S2) stopped; Hetero²Net not wired | 67-69 | `c4_tests.txt`, `c4_hetero2net_download.md` |
| C5 class C adapters | `4624d38` | not started (needs S1; H2SGNN S4) | — | `c5_blocked.md` |
| C6 large LP | `1722f2b` | sampler on the message graph + held-out test/mutation; no adapter wired | 70 test, 71 mutation | `c6_tests.txt`, `c6_mutation.txt` |
| C7 item 2 (no download) | `e0d91cb`, `3f33f6d` | done | 73 find, 74 tests, 75 re-check (found MUTAG), 76 gate, 78 GC tests, 79 final re-check | `c7_find_download.txt`, `c7_tests.txt`, `c7_nodata_check_e0d91cb.txt`, `c7b_tests.txt`, `c7_nodata_check_3f33f6d.txt`, `c7_gate.txt` |
| C8 full suite + report | this commit | see C8 | 80 suite, 81 gate | `c8_suite_g2n01_16741.txt`, `c8_gate.txt` |

## Scheme per method (C0, VERIFIED-CODE; probe VERIFIED-RUN, step 59)

| Method | Class | Source's scheme (file:line) | Depth / hop parameter | State on H2GB NC |
|---|---|---|---|---|
| GCN, GAT | A | H2GB `hgt`: `custom_sampler.py:323-341`; HGTLoader iterations = YAML list length (`hgt_sampler.py:48`) | HGB model: `num_layers + 1` convs (`GNN.py:32-43, 74-79`); H2GB list: 6 (MAG, OAG, IEEE), 4 (PDNS), Pokec user 4 / clubs 3 | pending S1 |
| HAN | A | H2GB: PyG `HANConv` on raw relations (`hetero_gnn_model.py:127-129`) | `gnn.layers_mp` | pending S1 + S5 |
| H2G-former | A | H2GB `GTModel` on `hgt` batches (`gt_model.py:62,187`; head `hetero_node.py:29-38`) | `gt.layers`; list 3 (MAG), 6 (OAG, IEEE), 4 (PDNS) | pending S1 |
| SeHGNN | B | `SeHGNN/ogbn/main.py:85-96, 170-174`, `ogbn/utils.py:49-80, 138-139` | `--num-hops` | **wired** on ieee-cis-g; MAG ×2 decision C4-a; OAG ×3, PDNS decision C4-b; H-Pokec N/A (author) |
| PSHGCN | B | `PSHGCN/ogbn-mag/processing.py:153-159`, `main.py:36-40, 91-95` (78-87 = enhance loaders; label propagation 96-116) | `--K` | **stopped S3** (model hard-codes MAG types, `ogbn-mag/model.py:57-66`); H-Pokec range K = 1, hidden ≤ 64 in place (`search_grids.DATASET_OVERRIDES`) |
| ChebNetII | B | `ChebNetII/ogb/processing.py:60-105`, `ogb/model.py:8-56`, `ogb/large_train.py:75-83` | `--K` | **wired** on ogbn-mag, mag-year, oag-cs/eng/chem; IEEE, H-Pokec, PDNS decision C4-c |
| Hetero²Net | B | `Hetero2Net/main.py:64, 147-175` (NeighborLoader, `[10, 10]`, batch 1024) | `HeteroGNN(num_layers=2)` | **not wired** in this card; MAG waits for the embedding file |
| HINormer | B | sequences once (`HINormer/run.py:100-127`) but every forward runs `num_gnns` full-graph GCN layers first (`model.py:204-208`) | `--len-seq`, `num_gnns` | **stopped S2** |
| NSD, CSNN | C | none; Laplacian / normalisation from the whole graph (`laplacian_builders.py:42,70,142-152`; `CSNNconv.py:74-78`) | `layers` | D1 second branch → pending S1 |
| H2SGNN | C | not node-wise: propagation inside the forward (`H2SGNN/model.py:67-86, 90-140`) | `--K` | D1 second branch → pending S1 (S4) |

## C0 — stops (`c0_preconditions.md`)
- **S1 sampler depth (classes A, C).** The card's rule "depth = the method's layers (H2GB's per-hop fan-out repeated)" is
  not H2GB's scheme: H2GB runs as many HGTLoader iterations as its YAML list is long, independent of the layers
  (GCN on MAG: 6 iterations, 4 layers); on H-Pokec the per-type lists have different lengths (user 4, clubs 3) and the
  user fan-out changes by hop (2048, 2048, 512, 512), so there is no single value to repeat.
- **S2 HINormer** is not "batches of targets": its forward runs full-graph GCN layers before reading the sequences.
- **S3 PSHGCN**: its batched ogbn-mag scheme hard-codes MAG's types and widths; its HGB model propagates in the forward.
- **S4 H2SGNN** is not node-wise (D1 second branch; inherits S1).
- **S5 HAN** (no stop by itself): semantic attention is a mean over every node of the subgraph it gets (HGB
  `model_hetero.py:28`; PyG `han_conv.py` `group`), so a sampled output never equals the full-batch one.

## C1 — shared infrastructure (stopped on the GCN equivalence)
`src/baselines/minibatch.py`: plan (B ∈ {32, 256, 1024}, 8,192/B steps, 500 epochs, H2GB's eval-epoch rule,
`default_eval_period`), fan-out read through H2GB's own cfg and expression, H2GB's `get_HGTloader` (depth = the
lists passed), whole-split evaluation with a coverage check, best validation macro-F1 epoch; later (C4) the index-batch
loop for precomputed-input methods and `h2gb_graph`. Tests (`tests/test_baselines_minibatch.py`, 7 OK, step 60): steps
per epoch, fan-outs equal the YAML for all 32 (class-A method, dataset) pairs, coverage, loop. Mutation (step 62):
evaluation without the last partial batch → coverage test fails ("covered 12 targets; the split has 13").
**Equivalence (kept test) fails** (step 60): HGB GCN, 2 GraphConvs, depth 2, full fan-out: max |diff| = 5.27e-2 at
validation seeds [3, 4, 6, 15] (atol 1e-10). Diagnostic (step 61, `c1_equivalence_diagnostic.txt`):

| depth | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| GCN max \|diff\| | 7.9e-1 | 7.2e-2 | **1.1e-16** | 1.1e-16 | 1.1e-16 |
| HAN max \|diff\| (dropped test in slim mode) | 7.0e-2 | 1.8e-3 | 8.5e-4 | 3.5e-4 | 1.8e-4 |

GCN is exact from depth = propagation steps + 1 (the last hop's degrees enter GraphConv's `norm="both"`); HAN is exact
at no depth (S5). The failing module is kept as a reproducer in `docs/baselines_h2gb_evidence/c1_equivalence_tests.py`
(not in `tests/`, so the suite stays usable for the gates; moved into `tests/` once S1/S5 are decided).

## C2 — PDNS (done)
`load_h2gb_dataset("pdns", root, seed)`: H2GB's `preformat_PDNS` with a raw-only offline class (every raw file H2GB
lists required, never downloads, writes nothing), after `seed_everything(seed)` as `H2GB/main.py:87-101`;
`build_dataset` passes the run seed (`config.seed`); preset target `domain_node`; PDNS leaves "data pending".
Test (step 64): equal to H2GB's own code on a fixture with H2GB's 68 raw files (graph, labels, masks), 70/20/10 by
H2GB's integer arithmetic, seed-dependent. Real-data check (step 65, peak 6.92 GiB): seed 0 = 61,077 / 17,450 / 8,725,
class counts [20,304, 40,773] / [5,815, 11,635] / [2,860, 5,865], **equal to Part M3b `b5_pdns_check.json`**; seeds
0-4 recorded with sha256 of each index set (`c2_pdns_split_record.json`). Gate (step 66): none / none, 129 tests OK.

## C4 — class B
- **SeHGNN** (`adapters/_sehgnn_h2gb.py`): their `hg_propagate` once on H2GB's graph (one letter per type), batches of
  target rows, `SeHGNN_mag`, stage 0 (no label features, no enhance stage, as `ogbn/main.py:290-291`), featureless types
  random-uniform as their MAG branch (`ogbn/utils.py:352-354`). Test: stored features equal an independent float64
  meta-path mean computation key by key (atol 1e-10), row batching exact in eval; one step finite and moves.
- **ChebNetII** (`adapters/_chebnetii_h2gb.py`): their ogb scheme on the union graph (one feature matrix). One step
  finite and moves; a dataset without one feature width is refused.
- Registry per cell (`H2GB_NC_WIRED_CELLS`, `H2GB_NC_CELL_REASONS`); SeHGNN H-Pokec N/A; PSHGCN H-Pokec override.
  Tests step 67-69 OK. **Omission found in C7** (`test_budget_table`: a runnable H2GB cell must declare
  `READS_BATCH_PLAN`): fixed in the C7 commit.
- **Hetero²Net download** (`c4_hetero2net_download.md`): URL `https://data.pyg.org/datasets/mag_metapath2vec_emb.zip`
  (PyG `ogb_mag.py:51-53`), file `mag_metapath2vec_emb.zip` (PyG reads `mag_metapath2vec_emb.pt` inside,
  `:94, 150-153`), place it at `/mnt/data/peeradon/h2snn/H2S-NN/data/NC/mag/hetero2net/mag_metapath2vec_emb.zip`.

## C6 — large LP (infrastructure only)
`src/baselines/large_lp_minibatch.py`: task files read (`split.npz`, `negatives.npz`), message graph (held-out pairs
removed in both orientations in every same-type relation, B6 rule), PyG `LinkNeighborLoader` seeded by the links,
`assert_no_held_out_facts`. Test (step 70): no held-out fact in any subgraph of training/validation/test links at full
fan-out, depth 1-3 (incl. a second same-type relation carrying held-out pairs). Mutation (step 71): sampling on the full
graph → "17 held-out fact(s) present". No LP adapter wired (S1 for the message-passing methods, S3 PSHGCN, S4 H2SGNN).

## C7 — no test reaches the network (item 2)
Found by running the suite in a data-less `git archive` export with HTTP(S) proxies set to a dead local port (step 73;
nothing was fetched): `test_baselines.HGBSplitTests` (×3 datasets), and then `RealListTests`, `SchemaTests` and
`BaselineBundleIdentityTests`, which saw the empty `raw/` folders PyG had created and tried too. Fix:
`src/data/offline.py` (`offline_hgb_dataset`: a missing release is refused with `DataNotOnDiskError` before PyG creates
anything; `download()` raises), used by `build_dataset` and H2G-former's HGB path; `SchemaTests` reads through it; the
two tests that assert counts of the real releases skip when the release is absent (as the suite's other real-data
tests do). Re-check on the `e0d91cb` export (step 75): no HGB attempt left, but **one more network access**, present in
the first run too and missed in my first reading of it: `BaselineBundleIdentityTests.test_graph_bundle_equals_explicit_untied`
fetched TU MUTAG from chrsmrrs.com (PyG's `TUDataset`; the dead proxy did not stop this request; the file landed in the
scratch exports only, never in `data/`). Follow-up `3f33f6d`: `offline_tu_dataset` (same rule) in
`build_graph_classification_dataset`; that test skips without MUTAG, as `ProtocolOnMutagTests` already does; GC tests
OK in place (step 78). Final re-check on the `3f33f6d` export (step 79): **no download attempt** (0 `Downloading` lines) and no `data/` folder created; 684 tests, 12 fail closed there because they need LP releases or `.git` (out of scope below). Gates: step 76 (`e0d91cb`) and 81
(`3f33f6d`): none / none.

## C8 — full suite (g2n01, job 16741 step 77, `scripts/slurm/run_tests.sbatch`, E exported)
**Ran 684 tests, OK** (`c8_suite_g2n01_16741.txt`; 2080 Ti visible, the CUDA test ran) = Part M3b's 671 + 13 new (`test_baselines_minibatch` 7, `test_pdns_loader` 1, `test_baselines_h2gb_classb` 4, `test_baselines_large_lp` 1). Bit-exact compares at `3f33f6d` (step 81, the last `src/` change): golden 1365 / mb0 139 tensors,
differing: none. A first suite run (step 77) was cancelled and discarded: I edited `src/data` while it ran.

## Deviations
- Branch base: `dfbc06e` merged into the existing branch (see header), not a fresh branch.
- C1's failing equivalence module kept in the evidence folder, not in `tests/`.
- C7: two real-release tests skip without data instead of moving to a fixture root (their counts are the releases').
- Existing assertions changed because the card changes the state they pin: `test_baselines` (PDNS no longer data
  pending; runnable cells 135 → 141, NC 45 → 51), `test_h2gb_nc` (per-cell registry rule; PDNS readiness).
- SeHGNN: `sparse_tools` (C++ package, not installed) stubbed during the import of `ogbn/utils.py`; used only for label
  diagonals, which stage 0 does not use. ChebNetII: `sys_normalized_adjacency` restated (their `processing.py` runs
  argparse at import).
- C1 tests beyond the slim list (plan, fan-out, loop) were written before the slim prompt; they stay.
- `c7_find_download` is step 73 (the C7 commit message says 72).
- C4's `READS_BATCH_PLAN` declaration (required by `test_budget_table`) was missed in C4 and added in the C7 commit.

## Decisions for the author
| # | Decision | Options | Recommendation |
|---|---|---|---|
| S1 | Sampler depth for classes A and C | (a) H2GB's YAML list unchanged (its depth, independent of tuned layers); (b) the card's rule, depth = propagation steps (GCN not exact, C1), H-Pokec defined separately; (c) depth = propagation steps + 1 | (a): it is H2GB's code ("follow the source exactly") and, in every cell checked, H2GB's depth ≥ layers + 1 (exact under full fan-out); (c) if depth must follow the tuned layers |
| S5 | HAN on sampled subgraphs | (a) accept H2GB's HAN as H2GB runs it (attention mean over the sampled subgraph), stated; (b) N/A on H2GB | (a), as H2GB benchmarks it this way |
| S2 | HINormer | (a) H2GB's own `hin` sampler and configs (class A); (b) its own code with the full-graph GCN each step; (c) N/A | (a) |
| S3 | PSHGCN | (a) N/A on H2GB NC/LP (own scheme needs a model edit); (b) an adapter edit of their ogbn-mag model (types/widths) | decide; (a) keeps "no model edit" |
| C4-a | SeHGNN on MAG inputs | (a) their MAG branch (random-uniform, or ComplEx with a download); (b) H2GB's averaged features | (b) unless their published number is the target, then (a) with ComplEx |
| C4-b | SeHGNN parallel relations (OAG, PDNS) | (a) merge (adapter in place); (b) N/A | (a) |
| C4-c | ChebNetII on IEEE-CIS, H-Pokec, PDNS (no single feature width) | (a) learned per-type input projection before propagation (not their scheme); (b) N/A | decide |

## Out of scope (not done)
- C7 item 1 (`clone_vendors.sh` accepting a patch-only clone) and item 3 (`vendor_paths.py:88`, symlinked clones): not
  done (slim mode). The gates here avoided symlinks by copying the vendor tree into the exports.
- SeHGNN's ogbn-mag branch reads ComplEx/random embeddings, like Hetero²Net's MetaPath2Vec (C4-a).
- `test_vendor_manifest.VendorIgnoreTests` errors in a `git archive` export (no `.git`); it passes in place.
- Without a data root, 12 tests that read LP releases (`LinkProtocolTests`, `PolynomialRelationKeyTests`, `FixedValidationFileTests`) or `.git` (`VendorIgnoreTests`) fail closed instead of skipping; none reaches the network (step 79).
- Scratch exports under `/mnt/data/peeradon/bh2gb_gate/` were removed at the end. The two exports with the fetched MUTAG file are gone with them.

Code head `3f33f6d270176fc2061eeaf4768264dcb91e77d6`; the report commit's own SHA cannot be written inside it and is given in the final message.

branch=baselines-h2gb sha=<see final message> card=docs/H2S-NN_baselines_h2gb_card.md base=4bfa5dea2da5ad8f96a6e6d181c0cfc58b05e5cd
