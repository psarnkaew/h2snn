# Baselines on the H2GB datasets, part 3 (card 1c) — report (slim mode)

Card `docs/H2S-NN_baselines_h2gb_part3_card.md`; branch `baselines-h2gb-3` from `v3b` `2ef9cac14ae03136efb07ad0c71233697ae138b1`,
then `git merge --no-ff 93092977c5045c5c9b851d412c81306312d52b23` (clean). Jobs: allocation 16741 on g2n01 (`srun --overlap`),
env `h2snn-b`; scripts and logs in `docs/baselines_h2gb_evidence/` (prefix `p3_`). Before each test or mutation the job printed
the sha256 of the edited files (`p3_sha.sh`) and it was compared with the login node's. No training run on real data, no
metric. No file outside the card's list changed (`src/baselines/**`, baseline tests, `notebooks/baselines/`, evidence), so no
golden/mb0 compare was required.

**Verdict.** On the H2GB NC datasets, 77 of 96 baseline cells are now runnable (all of GCN, GAT, HAN, H2G-former, HINormer,
ChebNetII, NSD, CSNN; SeHGNN 7, Hetero²Net 6). On the large LP tasks 13 of 14 cells with an LP path are runnable. Remaining:
PSHGCN on MAG (**F2 stop item**: `sparse_tools` needs `sortednp`, which needs numpy < 2), Hetero²Net on MAG ×3 (waiting for the
author's file), PSHGCN elsewhere and H2SGNN (card 3 Part B), SeHGNN H-Pokec (N/A, author). Two findings matter beyond this
card: H2GB's own MetaPath2Vec preprocessing could not run in `h2snn-b` (fixed here, F6), and the HGB Hetero²Net adapter lets
validation labels reach validation targets (not fixed; decision below).

## Units
| Unit | Commit | Jobs (16741 step) | State |
|---|---|---|---|
| F0 preconditions | `fcd6102` | — | done (`p3_f0_preconditions.md`) |
| F1 HINormer (G-1) | `8ee1fff` | 105 | done |
| F2 PSHGCN MAG (G-2) | `17422cd` | 107 build, 108 sortednp, 109 restore, 110-111 tests | **stop item (label part)**; width part done |
| F3 SeHGNN MAG (G-4) | `c1eb838` | 112 | done |
| F4 Hetero²Net (G-5 amended) | `067f71b` | 113-122 tests, 125 mutation | done (MAG waits for the file) |
| F5 ChebNetII IEEE/H-Pokec (G-6) | `d3c432f` | 126 | done |
| F6 H2G-former mag-year (G-7) | `08751b5` | 132 (cancelled externally), 137-138 | done; sklearn shim (finding) |
| F7 large LP (G-8, G-9, G-10) | `bcdcc8f`, `cf289b0` (notebooks) | 140 tests, 141 notebooks, 143 mutation, 145 registry | done; Hetero²Net ogbn-mag P–P waits |
| F8 suite and report | this commit | suite (below) | see F8 |

## Cell table (`p3_f8_cells.txt`)
R = runnable; W = waiting for the author's named file; B = moved to card 3 Part B; S = stop item; — = N/A.

| Method | ogbn-mag | mag-year | oag-cs | oag-eng | oag-chem | ieee-cis-g | h-pokec | pdns | ogbn-mag P–P | H-Pokec U–U |
|---|---|---|---|---|---|---|---|---|---|---|
| GCN, GAT, NSD, CSNN, HAN, H2G-former | R | R | R | R | R | R | R | R | R | R |
| HINormer | R | R | R | R | R | R | R | R | — | — |
| ChebNetII | R | R | R | R | R | R | R | R | — | — |
| SeHGNN | R | R | R | R | R | R | — (author) | R | — | — |
| Hetero²Net | W | W | R | R | R | R | R | R | W | R |
| PSHGCN | S (F2) | S (F2) | B | B | B | B | B | B | B | B |
| H2SGNN | B | B | B | B | B | B | B | B | B | B |

W: `<data root>/NC/mag/hetero2net/mag_metapath2vec_emb.zip` from `https://data.pyg.org/datasets/mag_metapath2vec_emb.zip`
(sha256 written beside it at first use; a different file is refused).

## Per unit (what, file:line)
- **F1 HINormer (G-1).** The original code as on HGB: its sequence routine (`run.py:100-122`, here seeded by the target ids),
  its model (`model.py:198-217`: full-graph GCN, then the sequences); class B: the shared batch size and 8,192 targets per
  epoch imposed on its training step (`run.py:155` trains all targets in one forward). H2GB's `HINormerModel` is not used
  (registry reason). No label input (`model.py:198-217`), so no route test. Test: one step.
- **F2 PSHGCN MAG (G-2).** `sparse_tools` `0596f00` cloned as `SeHGNN/Readme.md:22-27` says and built (CPU extensions
  `_spspmm_diag_cpu.so`, `_spspmm_diag_unsym_cpu.so`). `setup.py develop` runs an isolated PEP 517 build without torch on
  this setuptools, so the build used `pip install -e . --no-build-isolation --no-deps`. Its import needs `sortednp`
  (`sparse_tools/utils.py:5`, not declared). `sortednp` 0.5.0 requires numpy < 2 and fails to import with numpy 2.4.6
  (step 108). Environment restored (step 109: neither installed, `pip check` as before), so there is **no lock line**. Width
  part done: `_pshgcn_mag.build_model` (S3 (b)), equal to their class with MAG's widths (float64, exact; test).
- **F3 SeHGNN MAG (G-4).** Author / field / institution take random-uniform(−0.5, 0.5) `embed_size` inputs, as their README's
  command without extra embeddings builds them (`ogbn/utils.py:352-354`), replacing H2GB's averaged features. Test: inputs +
  one step.
- **F4 Hetero²Net (G-5 amended).** Their lists follow no single rule (F0: target-centred, other-type-centred and open paths),
  so non-MAG datasets run with `metapaths = None` and `torch.eye` for featureless types, as their FreeBase branch
  (`hetero2net/dataset.py:48-53`). NeighborLoader `[10, 10]`. Training and validation batches come from `train_data`
  (validation and test labels hidden in `y_emb`), test batches from `test_data` (`main.py:125-175, 202-206`); one step =
  their `train_mini_batch`. Tests: route test (validation outputs unchanged when validation/test labels change; test
  outputs unchanged when test labels change), one step. Mutation: validation on `test_data`, caught (step 125).
- **F5 ChebNetII IEEE-CIS / H-Pokec (G-6).** HGB's per-type projection first (sparse identity block = embedding table), their
  Chebyshev bases of the projected features in every forward over the whole graph. Equal to the per-block form by linearity
  (part 2 test). Test: one step with an identity block.
- **F6 H2G-former mag-year (G-7).** H2GB's `preprocess_Metapath` dispatched to ogbn-mag's schema (`encoding_generator.py:263-280`)
  through a name shim; same graph.
- **F7 large LP.**

  | Methods | ogbn-mag P–P | H-Pokec U–U |
  |---|---|---|
  | GCN, GAT, NSD, CSNN (local-LP encoders on HGB's reduction of each batch) | link sampler, H2GB's per-hop list | H2GB `hgt` with its per-type lists, seeded by the step's endpoints (G-8) |
  | HAN, H2G-former (H2GB's own networks, whose head reads the seed rows; H2G-former raw encoder only) | `hgt` seeded by endpoints | `hgt` seeded by endpoints |
  | Hetero²Net (its HeteroGNN + disentanglement term) | waits (file) | link sampler `[10, 10]` |

  Evaluation computes each endpoint once. Notebooks regenerated with the unchanged generator (four LP lanes).

## Tests and mutations (slim)
| Check | Step | Result |
|---|---|---|
| One step finite and moves: HINormer, SeHGNN MAG inputs, Hetero²Net, ChebNetII (identity block), H2G-former mag-year (incl. H2GB's MetaPath2Vec on a MAG-typed graph) | 105, 112, 122, 126, 138 | OK |
| PSHGCN width model = their class with MAG widths (float64, exact) | 111 | OK |
| Hetero²Net label route | 122 | OK; mutation caught (125) |
| Held-out facts over the three LP sampler configurations (link H2GB list, link [10, 10], hgt seeded by endpoints); one LP step per method (7) | 140 | OK; mutation (full graph) caught in all three (143: 76 / 74 / 76 held-out facts) |
| Registry, notebooks (generator `--check`), budget table | 145 | OK |

## F8 — full suite
`scripts/slurm/run_tests.sbatch` at `cf289b0` on g2n01 (job 16741; 2080 Ti visible): **Ran 703 tests, OK** in 2,474 s (`p3_f8_suite_g2n01_16741.txt`) = card 1b's 696 + 7 new (HINormer 1, PSHGCN width 1, SeHGNN MAG 1, Hetero²Net 2, ChebNetII identity block 1, H2G-former mag-year 1; the two large-LP tests were rewritten, not added). Existing tests pass; `test_baselines.py` has its cell counts updated (runnable 193 → 225: NC 103 → 122, LP 54 → 67) and its GCN large-LP assertion (now runnable).

## Findings (beyond this card)
1. **H2GB's MetaPath2Vec preprocessing could not run in `h2snn-b`.** Its per-epoch probe calls
   `LogisticRegression(multi_class="auto")` (`H2GB/loader/embedding/metapath2vec.py:229-238`), which sklearn 1.9 removed.
   This affected every H2G-former cell with MetaPath2Vec, including the five part 2 wired; part 2's tests did not reach
   that code. Fixed (F6) with a factory shim active only during preprocessing: `"auto"` is sklearn's default behaviour,
   any other value is refused. The probe's accuracy is only printed and never selects the embedding.
2. **HGB Hetero²Net NC adapter: validation labels reach validation targets (not fixed).** `src/baselines/adapters/hetero2net.py`
   takes validation metrics from `eval_data`, whose `y_emb` hides only test labels, while their code evaluates validation on
   `train_data` (validation hidden: `Hetero2Net/main.py:205-206, 218-219`); `HeteroGNN` embeds every node's `y_emb`,
   the target's own included (`models.py:105-113`). Validation macro-F1, and with it model selection, can be inflated on
   ACM/DBLP/IMDB/Freebase; test labels stay hidden. Fixing it changes an existing HGB baseline output (a stop condition
   here), so it is listed for a fix card. F4's mutation shows the effect on H2GB batches.
3. Memory/time to watch in the dry run (not fixed, by the rules): Hetero²Net's FreeBase-branch `torch.eye` for H-Pokec's 15
   club types (dense, about 20k × 20k each); HINormer's full-graph GCN in every step and its per-node Python sequence
   routine on the MAG/OAG graphs; ChebNetII's whole-graph propagation per step on IEEE-CIS/H-Pokec.

## Deviations
- F2: `pip install -e . --no-build-isolation --no-deps` instead of `setup.py develop` (same editable install; isolation
  removed because the isolated build has no torch); then uninstalled again (stop item).
- F7: commit `bcdcc8f` says the notebook regeneration changed nothing; that was wrong (a stale `git status` on the login
  node). `cf289b0` adds the regenerated notebooks and states the correction. No amend.
- Step 132 was cancelled from outside this session (0.9 GB RSS; not an OOM); the tests were re-run (137, 138).
- HAN and H2G-former on ogbn-mag P–P use H2GB's `hgt` sampler seeded by endpoints instead of the link sampler: their head
  returns the seed rows (`hetero_node.py:29-38`), and `hgt` is H2GB's sampler for these networks (S1).

## Decisions for the author
| # | Item | Options |
|---|---|---|
| 1 | PSHGCN label part on MAG (F2) | (a) build `sortednp` from source against numpy 2 (an environment change); (b) load `sparse_tools`' compiled diagonals without its `utils` import (an adapter shim); (c) run PSHGCN MAG without the label part (stated deviation); (d) N/A |
| 2 | HGB Hetero²Net validation route (finding 2) | fix card: evaluate validation on the training copy as their code does (changes the HGB NC outputs of that baseline) |
| 3 | Memory items (finding 3) | decide after the dry run |

The report commit's own SHA cannot be written inside it; it is given in the final message.

branch=baselines-h2gb-3 sha=<see final message> card=docs/H2S-NN_baselines_h2gb_part3_card.md base=2ef9cac14ae03136efb07ad0c71233697ae138b1
