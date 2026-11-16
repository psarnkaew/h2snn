# Baselines on the H2GB datasets, part 5 (card 1d continuation) — report (slim mode)

Card `docs/H2S-NN_baselines_h2gb_part4_card.md` as amended at design-docs `44f0e59` (`.gitignore` line, unit H3b);
branch `baselines-h2gb-5` from `7df909df4487ca8deed03aace71ccce34d5eeb69`, then `git merge --no-ff 44f0e59` (clean).
Jobs by `sbatch` (allocation 16741 had expired), env `h2snn-b` (nothing installed or removed); evidence
`docs/baselines_h2gb_evidence/p5_*`. Before each test or mutation, the job printed the sha256 of the edited files, and it
matched the login node's. No training run on real data, no metric. No `src/` change outside `src/baselines/`, so no
golden/mb0 compare.

**Verdict.** H1 is done: `sparse_tools` is a vendor clone built in place, and PSHGCN runs on ogbn-mag and mag-year. H3b:
3 of the 5 dry-run blockers are fixed, each with a test that reproduces the dry run's error. 2 are stop items, because
they cannot be reproduced off real data. H4: see below.

## Units
| Unit | Commit | Jobs | State |
|---|---|---|---|
| merge design-docs `44f0e59` | `8c180cf` | — | done |
| H1 `sparse_tools` vendor + PSHGCN MAG label part (D1-b) | `d23d455` | 16891 build, 16895 tests, 16899 mutation, 16902 notebooks | done |
| H3b dry-run blockers | `e26d659` | 16906 repro, 16910 before fix, 16911 after | 3 fixed, 2 stop items |
| H4 suite and report | this commit | suite (below) | see H4 |

## H1: `sparse_tools` and the PSHGCN ogbn-mag label part
- **Vendor.** `VENDORS.tsv` row `sparse_tools` `0596f00` (clone, no licence file). `.gitignore` gets `/vendor/sparse_tools/`.
- **Build.** `scripts/setup/build_sparse_tools.sh` runs `setup.py build_ext --inplace` inside the clone. Job 16891 on
  g2tn01 built both `.so` files, and `pip freeze` was identical before and after.
- **Stub.** `_pshgcn_mag.pshgcn_mag_modules()` puts a stub `sortednp` in `sys.modules` only while it imports
  `sparse_tools`, PSHGCN's `processing`/`utils`/`model` from the clones, and removes the stub and `sparse_tools.*`
  afterwards.
  - Any public attribute of the stub raises "sortednp is not available".
  - Dunder lookups answer `AttributeError`, so that `pushd_sys_path` can introspect modules (`getattr(module,
    "__file__", None)`, `vendor_paths.py:91`). The first version raised on every name and broke that call; the tests
    caught it.
- **Label part.**
  - Diagonals: PFFP = ABA(P–F), PPPP = AAA(P–P), PAAP = ABA((A–P)ᵀ) (`processing.py:134-150`).
  - Labels: training rows only (`main.py:96-101`).
  - Self-effect removal: `main.py:110-115`.
  - Keys `PPPP, PAAP, PP, PFFP` (K = 2).
  - A/F/I inputs: random-uniform 256 (seeded).
  - No call into `sparse_tools.utils`.
- **Cells.** PSHGCN NC ogbn-mag and mag-year are wired.

## H3b: dry-run blockers
On this branch, I checked each blocker first. None was already fixed by cards 1b/1c.

| Cell(s) | Finding | Outcome |
|---|---|---|
| HAN NC imdb | HAN's IMDB loader returns HGB's `dl` as a 12th value (`HAN/utils.py:333`), used only by its test scorer. HGB runs IMDB with `main_multi.py`, whose model is `model_hetero_multi.HAN` (sigmoid semantic attention; `HAN/README.md`). | **fixed** as the source: first 11 values; `model_hetero_multi` for imdb |
| HAN LP, H2SGNN LP imdb_ma | IMDB lays out movie, director, actor, keyword, so the movie–actor block is two ranges. No source code exists for this (neither ships LP); the limit was our shared target-block helper. | **fixed**: block = sorted ids of the endpoint types (no other type), global→local map for crops, scatter by index |
| ChebNetII NC ogbn-mag, mag-year, oag-cs, oag-eng, oag-chem; SeHGNN NC ieee-cis-g | `build_bundle` validates a `TrainConfig`, and these datasets' NC mode is mini-batch, which requires `data.batch_size`. `fit_cell` never passed it (`READS_BATCH_PLAN` is declared by 10 adapters but read nowhere). | **fixed**: `fit_cell` passes the run's B (`params["batch_size"]`, default 256 as every `minibatch_plan` call) as `data.batch_size` for mini-batch NC cells. This also unblocks the other H2GB NC baseline cells wired in cards 1b–1d, which take the same path. |
| HINormer NC freebase | HINormer only removes self-loops (`run.py:95-96`); its `GraphConv` keeps DGL's zero-in-degree check, which DGL 0.9 has too. Its README runs Freebase, so either the authors' `adjM` has no isolated node and ours differs, or the source fails as well. Only the real release can say which. | **stop item** (no fix) |
| H2SGNN LP freebase_bb | CUDA illegal memory access (dry-run job 16795). On synthetic `freebase_bb` built by the project's pipeline, with the dry run's parameters and `CUDA_LAUNCH_BLOCKING=1` on a TITAN RTX, there is no error at 60 nodes or 20,500 nodes / 175,826 message edges (job 16906, `p5_h3b_h2sgnn_freebase_bb_repro.*`). | **stop item**: not reproduced off real data |

## Cell delta (registry)
| Cell | Before (part 4) | Now |
|---|---|---|
| PSHGCN ogbn-mag, mag-year (NC) | pending (H1 stopped) | runnable |
| HAN NC imdb; HAN/H2SGNN LP imdb_ma; ChebNetII ×5 and SeHGNN ieee-cis-g (NC) | listed runnable, failed in the dry run | listed runnable, failing path fixed (tests) |
| HINormer NC freebase; H2SGNN LP freebase_bb | listed runnable, failed in the dry run | unchanged; stop items |

Runnable cells: 228 → 230 (NC 124 → 126). Notebooks: `--check` reports all 13 match the generator (job 16902), so no
regeneration was needed.

## Tests and mutations (slim)
| Check | Job | Result |
|---|---|---|
| H1: the diagonals equal their dense formulas, exactly (float32 kernels; a fixture with power-of-two degrees); label route; no `sortednp`/`sparse_tools` left in `sys.modules`; one step finite and parameters move; plus vendor manifest, baselines, budget, h2gb_nc | 16895 | 107 OK |
| H1 mutation: every paper row (validation included) in the label one-hot | 16899 | caught ("held-out labels reach PP") |
| H3b on the pre-fix code: each new test hits the dry run's error (unpack 11; "not contiguous" for HAN and H2SGNN; "data.batch_size (B) is required" for all 6 cells) | 16910 | reproduces |
| H3b on the fixed code (+ baselines, failed trials, part P derived LP and Freebase, part O reload, search spaces, W&B) | 16911 | 130 OK |

## H4: full suite
`scripts/slurm/run_tests.sbatch` ran at `e26d659` on g3n01 (job 16912; RTX 3090 visible): **Ran 712 tests, OK**, in
1,578 s (`p5_h4_suite.txt`). That is part 4's 705 plus 4 new (H1) and 3 new (H3b). `test_baselines.py` changed only its
cell counts (runnable 228 → 230, NC 124 → 126).

## Deviations
- Evidence files are named `p5_*` (the card's file list says `p4_*`), because this continuation has its own report
  (`part5`).
- The stub also answers dunder lookups with `AttributeError` (above). Without that, `pushd_sys_path` cannot run while
  the stub is installed.
- One pre-fix H3b run (not kept) skipped the sha256 check and showed the batch-size test passing on the old code: the
  compute node still saw the fixed `campaign.py` (NFS). Rerun with the check (16910), it fails as expected.

## Decisions for the author / design session
| # | Item | Options |
|---|---|---|
| 1 | HINormer NC freebase | (a) count, on the real release, the nodes with no non-self edge in HINormer's `adjM` and in our view (a data check, no training), then decide; (b) add self-loops (HGB's GNN convention), a deviation from HINormer's `run.py` |
| 2 | H2SGNN LP freebase_bb | card 2 re-runs the cell after the merge with `CUDA_LAUNCH_BLOCKING=1` on real data to name the kernel |

The report commit's own SHA cannot be written inside it; the final message gives it.

(Part 5 report line, as reviewed: `branch=baselines-h2gb-5 sha=ba0e9d61f20ffdff4a6db617fcb1d95c440f8065 card=docs/H2S-NN_baselines_h2gb_part4_card.md base=7df909df4487ca8deed03aace71ccce34d5eeb69`.)

## H5 — review follow-up (author decision 1a)
Card section "H5" at design-docs `21d9196`; evidence `docs/baselines_h2gb_evidence/p5_h5_*`.

- **Merges.** The branch started at `ba0e9d6` (= origin). I merged design-docs `21d9196` (`9c65d3f`), then v3b
  `5ccc792` (`454a3b5`). Neither conflicted, and nothing was rebased.
- **Fix (`cf5de23`).** `hinormer.allow_isolated_nodes` sets `allow_zero_in_degree` on every `GraphConv` in
  `model.GCNLayers`. It uses DGL 2.4's setter, which writes `_allow_zero_in_degree`.
  - It applies at both places a HINormer model is built: `build_model` (H2GB path) and `fit_task` (HGB path).
  - It applies on every dataset.
  - The graph is unchanged and no self-loop is added. Vendor code is not edited.
  - An isolated node receives no message, and its degree normalisation is already clamped at 1 (`model.py:85`).
- **Tests** (`tests/test_baselines_hinormer_isolated.py`, synthetic two-type graph):

  | Check | Job | Result |
  |---|---|---|
  | (i) isolated target node, pre-fix code | 16913 | raises DGL's "There are 0-in-degree nodes in the graph" |
  | (i) fixed code: finite outputs through `fit_task` (HGB path) and `build_model` (H2GB path); (ii) no isolated node: same-seed outputs with and without the flag bit-identical (`torch.equal`); + `test_baselines_h2gb_classb`, `test_baselines` | 16914 | 101 OK |
  | Mutation: flag dropped, run against (i) only | 16917 | caught (DGL 0-in-degree error) |

- **Notebooks.** No cell claim changed, so nothing was regenerated.
- **Suite.** `scripts/slurm/run_tests.sbatch` ran at `cf5de23` on g2n01 (job 16918; its own RTX 2080 Ti, `--gres=gpu:1`):
  **Ran 728 tests, OK**, in 1,597 s (`p5_h5_suite.txt`). That is the merged base's 726 plus 2 new. No `src/` change
  outside `src/baselines/`, so no golden/mb0 compare.
- **Not here:** the H2SGNN freebase_bb rerun (decision 2) goes to card 2 / S2.

The report commit's own SHA cannot be written inside it; the final message gives it.

branch=baselines-h2gb-5 sha=<see final message> card=docs/H2S-NN_baselines_h2gb_part4_card.md base=ba0e9d61f20ffdff4a6db617fcb1d95c440f8065
