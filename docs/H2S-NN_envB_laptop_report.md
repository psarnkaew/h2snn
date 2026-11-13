# Report — Option B check on the laptop (E0–E6)

**Branch** `env-b-check` from `part-mb` at `da3e1f535c20d3c2d364712a073b1ccfb6a62d9e` (hash present after `git fetch`; VERIFIED-RUN `git rev-parse HEAD` in `e0_baseline.txt`). One commit per unit (E3 has an addendum commit E3b). Evidence: `docs/envB_evidence/`. Date 2026-11-14. `v3b` and `part-mb` untouched; spec and cards untouched; `h2snn` only read (pip freeze, suite, import checks).

## Verdict

- **Torch 2.4.1 itself is compatible** with every required library that can be tested on Windows, and with the full suite: in `h2snn-b` the suite gives the same result as in `h2snn` (torch 2.6.0) except one test whose library (`torch_householder`) the card did not install (E4). No bit-exact test differs.
- **Pre-registered expectation 1 failed as stated:** with the CS2 pins alone, pip *replaces torch 2.4.1 by 2.15.0*. Cause: an unpinned chain `performer-pytorch 1.1.4 → local-attention 1.11.2 → hyper-connections ≥ 0.4.3 (needs torch ≥ 2.5)`. One extra pin, `hyper-connections==0.4.2`, fixes it; `pip check` is then clean (E2).
- **The DGL 2.4.0 wheel could not be inspected at all.** `data.dgl.ai` is not just unreachable over TLS: plain HTTP shows a Namecheap page "**Domain registration has expired**" (E1). The same host serves every `repo.html` index (torch-2.4 and torch-2.6, also the 2.5.0 pin). No wheel path was given in the prompt (placeholder left unfilled) and none is on the laptop. So Python tag, `numpy` and `torchdata` constraints and the graphbolt torch version of DGL 2.4.0 are **not answered**.
- **E5: torch 2.4.1 and 2.6.0 give identical initial weights and outputs on this machine** (golden tests OK, `mb0 compare`: `differing: none`). By the card's rule, the cluster's 1-ulp difference is platform (compiler, BLAS, CPU), not torch version.
- No performance run, no metric, no method code, tolerance, test or requirement file changed.

## E0 — Preconditions and baseline (commit `123a291`)

**Done.** `docs/envB_evidence/e0_baseline.txt`. GPU: **NVIDIA GeForce RTX 4060 Laptop, 8 GB, driver 591.44** → CUDA index used. Existing `h2snn`: Python 3.12.x, torch 2.6.0+cu124, **dgl 2.2.1**, **torch_householder 1.0.1**, **torchdata 0.11.0**, torch_scatter/sparse/cluster `+pt26cu124`.

**Deviation.** `requirements/linux-lock.txt` is **not on `part-mb`** (it is a CS2 file of branch `cluster-setup`). I read it with `git show origin/cluster-setup:H2S-NN/requirements/linux-lock.txt` and the CS2 report the same way (read-only). `linux-lock-b.txt` is derived from that file.

## E1 — DGL 2.4.0 Linux wheel (commit `56f1009`)

**Done as far as possible.** `e1_dgl_index.txt` (VERIFIED-RUN):

| Probe | Result |
|---|---|
| `curl https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html` (also `--ssl-no-revoke`, `--tlsv1.2`) | `curl: (35) schannel: failed to receive handshake` |
| python urllib (`h2snn`) | `SSL: UNEXPECTED_EOF_WHILE_READING` (same as the cluster's CS0 result) |
| `http://data.dgl.ai/` | 200, a Namecheap parking page: "Domain registration has expired. … Renew this domain"; host resolves to 104.219.250.37 |
| `http://data.dgl.ai/wheels/…/repo.html` (4 paths) | 404 |
| `https://www.dgl.ai/pages/start.html` | same TLS failure (not further probed) |
| PyPI `dgl` | latest **2.2.1**, only `win_amd64` (cp38–cp312); `Requires-Dist: numpy>=1.14.0, scipy>=1.1.0, networkx>=2.1, requests>=2.19.0, tqdm, psutil>=5.8.0, torchdata>=0.5.0, pandas` |

**E1 answers.** Python tag: not answered. `numpy` constraint: not answered for 2.4.0 (PyPI 2.2.1 has `numpy>=1.14.0`, no upper bound). `torchdata`: `>=0.5.0` on PyPI 2.2.1; not answered for 2.4.0. Graphbolt torch version: not answered. **Deviation:** the prompt's `<<AUTHOR: path …>>` was not filled; a recursive search of `C:\Users\Acer` for `dgl*.whl` found nothing. I did not guess a path.

## E2 — The environment `h2snn-b` (commit `df6e475`)

**Done.** `conda create -n h2snn-b python=3.12` needed no administrator rights (Python 3.12.16). torch 2.4.1 exists for cp312 `win_amd64` on the cu124 index (`torch-2.4.1+cu124-cp312-cp312-win_amd64.whl`, 2506 MB); installed with `pip install --no-cache-dir`. Commands and all outputs: `e2_commands.txt` (it contains conda progress-bar noise; no output was edited).

| Step | Result |
|---|---|
| torch 2.4.1 (cu124 index) | installed; `torch.cuda.is_available()` True |
| `torch_scatter==2.1.2`, `torch_sparse==0.6.18`, `torch_cluster==1.6.3` from `data.pyg.org/whl/torch-2.4.0+cu124.html` | **all three exist for cp312 win_amd64** (`+pt24cu124`) — expectation 2 holds; nothing compiled. This step also pulled numpy 2.5.4 / scipy 1.18.1, replaced by the pins in the later step |
| `torch_geometric==2.8.0` | installed |
| all other pins of the lock (except torch, PyG extensions, `torch_householder`, `dgl`) | **dry-run replaces torch by 2.15.0** (see below); with `hyper-connections==0.4.2` added: installed, torch stays 2.4.1+cu124 |
| `dgl==2.2.1` + `torchdata==0.8.0` | installed |
| `pip check` | **No broken requirements found.** |

**Finding (expectation 1 not met as stated).** `pip install --dry-run -r <the 19 other exact pins>` ends in `Would install … torch-2.15.0 torch-einops-utils-0.2.6 …`. Chain (PyPI metadata, read-only): `performer-pytorch==1.1.4` requires `local-attention>=1.1.1`; `local-attention 1.11.2` requires `hyper-connections>=0.1.8`; `hyper-connections` 0.4.3 and later require `torch>=2.5` (0.4.8+ also `torch-einops-utils`, which needs `torch>=2.5`). `hyper-connections 0.4.2` is the newest release with `torch>=2.3`. The existing `h2snn` has `hyper-connections 0.4.11` because it runs torch 2.6. The torch-2.6 lock leaves this chain unpinned, which is harmless on 2.6 but a trap on 2.4.

**Candidate lock** `H2S-NN/requirements/linux-lock-b.txt` (new file; `linux-lock.txt` and `check_env.py` untouched). Differences from the CS2 lock: `torch==2.4.1`; PyG index `torch-2.4.0+cu124`; `dgl==2.4.0+cu124` with index `torch-2.4/cu124` (**untested**); added `hyper-connections==0.4.2` (reason in a comment); added `torchdata==0.8.0` (reason in a comment, see E3). Header comments say all of this.

**Installed versions vs the lock** (`e2_versions.py` → `e2_versions.txt`): 26 of 28 lines equal (torch and the three PyG extensions with a local build tag only). Differences: **`dgl`** (lock 2.4.0+cu124, installed 2.2.1 — expected) and **`torch_householder`** (not installed in `h2snn-b`; the card excludes it from E2 — expected differences were "only DGL").

## E3 — Import matrix (commits `c1992fc`, `234e49d`)

`e3_imports.py` → `e3_import_matrix_h2snn-b.txt` (fresh interpreter per library, `python -W default`).

| Library | Locked | Installed in `h2snn-b` | Import | Warnings |
|---|---|---|---|---|
| torch | 2.4.1 | 2.4.1+cu124 | ok; CUDA tensor op ok on RTX 4060 | none |
| torch_geometric | 2.8.0 | 2.8.0 | ok; GCNConv ok | DeprecationWarning "'torch-cluster' is no longer necessary…" (known, harmless) |
| torch_scatter / torch_sparse / torch_cluster | 2.1.2 / 0.6.18 / 1.6.3 | `+pt24cu124` each | ok; scatter_add, SparseTensor matmul, radius_graph, fps ok | none |
| **dgl** | 2.4.0+cu124 (Linux) | 2.2.1 (Windows CPU, the only one) | **bare `import dgl` fails:** `FileNotFoundError: Cannot find DGL C++ graphbolt library at …\dgl\graphbolt\graphbolt_pytorch_2.4.1.dll` | — |
| torchdata | (unpinned in CS2) | 0.8.0 | ok; `torchdata.datapipes` ok | UserWarning: datapipes/dataloader2 deprecated |
| numpy / scipy / pandas / sklearn | 2.4.6 / 1.18.0 / 3.0.3 / 1.9.0 | equal | ok | no NumPy-1.x/2.x message anywhere |
| ogb, pytorch_lightning, wandb, torchmetrics | 1.3.6 / 2.6.5 / 0.28.0 / 1.8.2 | equal | ok (Lightning + torchmetrics Accuracy call ok; OGB Evaluator ok) | — |
| umap | 0.5.12 | equal | ok | ImportWarning: Tensorflow not installed |
| performer_pytorch | 1.1.4 | equal | ok; a Performer forward ok | DeprecationWarning (distutils `LooseVersion`) |

**DGL, two routes.**
1. *Bare `import dgl`* fails in `h2snn-b` **and in the existing `h2snn`** (`graphbolt_pytorch_2.6.0.dll`; control in `e3_dgl_graphbolt_control.txt`). The Windows 2.2.1 wheel ships graphbolt DLLs for torch 2.1.0–2.3.0 only. This is a Windows-wheel limit, not a torch-2.4.1 result.
2. *The project's own route* (E3b, `e3b_dgl_via_shim.py/.txt`): `src/baselines/runtime/_dgl_compat.py` (existing code, unmodified) stubs graphbolt. Through it, in `h2snn-b` (torch 2.4.1, numpy 2.4.6, dgl 2.2.1) and in `h2snn`: homogeneous and heterograph message passing, `GraphConv`, `GATConv` run; **`-W default` printed no warning**. This explains why no DGL-backed test differs in E4.

**Expectation 3, checked as far as possible.** `torchdata`: DGL's graphbolt imports `torchdata.datapipes` (VERIFIED-CODE in the installed dgl 2.2.1: `graphbolt/base.py:20`, `dataloader.py:7-8`, `neighbor_sampler.py:8`, `item_sampler.py:11`, `minibatch_transformer.py:5`); `torchdata` 0.10+ has no `datapipes` module (`h2snn` has 0.11.0: `ModuleNotFoundError: No module named 'torchdata.datapipes'`). So a graphbolt-enabled DGL needs `torchdata<=0.9`; 0.8.0 is the torch-2.4 release → candidate pin (UNVERIFIED for 2.4.0's own code). NumPy 2: the 2.2.1 Windows CPU wheel runs with numpy 2.4.6 without any message for the APIs tried; **DGL 2.4.0 Linux with numpy 2.4.6 is untested** (no wheel, E1), so no `numpy<2` decision can be made from this card.

## E4 — Full suite (commit `7b191de`)

`cd H2S-NN; python -m unittest discover -s tests -v`, same commit (`c1992fc`, docs-only difference from `part-mb`), same machine, one after the other. Logs `e4_suite_h2snn-b.txt`, `e4_suite_h2snn.txt`; per-test comparison `e4_diff.py` → `e4_per_test_diff.txt`.

| Env | Result |
|---|---|
| `h2snn` (torch 2.6.0) | Ran 674 — FAILED (failures=2, errors=2, skipped=7) = Part MB-fix reference |
| `h2snn-b` (torch 2.4.1) | Ran 674 — FAILED (failures=4, errors=2, skipped=7) |

Per-test difference: **one test, two subtests** — `test_baselines.DglCudaGateTests.test_only_dgl_backed_models_are_gated` (`model='csnn'`, `model='nsd'`). Class: **library missing** (`torch_householder` not installed in `h2snn-b`). First error line: `AssertionError: 'missing requirement group sheaf (torch_householder) -- install requirements/baselines-sheaf.txt' is not None` (`tests/test_baselines.py:1394`). The same test fails on the cluster for the same reason (CS5). The 4 common non-passing tests are the known laptop set (notebooks with outputs; missing vendor clones `sehgnn, sehgnn_data, pshgcn, pshgcn_lp, chebnetii`; PSHGCN-dependent `test_colliding_vendor_module_names_do_not_leak` and `test_vendor_model_init_is_reproducible_under_seed_everything`). Skipped lines (empty vendor clones) are identical. **No bit-exact capture differs.** Nothing was fixed. Expectation 4 holds (the exception it names did not occur). Wall-clock (1653 s vs 2432 s) is not comparable across the two runs. `e4_diff.py` counts 673 distinct ids against unittest's 674 (cause not investigated); the totals above are unittest's own.

**Not run in `h2snn-b`:** whether `torch_householder==1.0.1` installs next to torch 2.4.1 on Windows (the card excludes it). It is a lock line and backs NSD/CSNN, so it remains untested for 2.4.1 here; the cluster card should test it.

## E5 — Initial weights, torch 2.4.1 vs 2.6.0 (commit `90cfe1a`)

`e5_initial_weights.txt` (VERIFIED-RUN, both envs, same commands from `H2S-NN/`):

| Env | `tests.test_task_heads tests.test_cooperative` | `mb0_reference.py compare mb0_reference.pt` |
|---|---|---|
| `h2snn-b` (2.4.1+cu124) | Ran 30, **OK** (incl. the golden node-classification test and `test_T1_symmetric_reproduces_the_golden_outputs_bit_for_bit`) | `compared 139 tensors; differing: none` |
| `h2snn` (2.6.0+cu124), control | Ran 30, OK | `compared 139 tensors; differing: none` |

**Verdict by the result rule fixed in the card:** golden tests pass and the compare says `differing: none` → **torch 2.4.1 and 2.6.0 give identical initial weights and outputs on this machine**; the cluster's 1-ulp difference at initialisation (CS4/CS5) is therefore platform (compiler, BLAS/MKL, CPU), not torch version. Limit: one machine (this CPU, Windows wheels); it does not test the Linux wheels, and `mb0 compare` prints only the first five names (`none` = empty list).

## Compatibility summary

| Question | Answer |
|---|---|
| Does torch 2.4.1 break any tested library or test? | No. |
| What breaks the pins? | `hyper-connections` chain → needs `hyper-connections==0.4.2`. |
| Do the PyG extension wheels exist for torch 2.4? | Yes (Windows cp312 verified; Linux not checked here). |
| Is DGL 2.4.0 compatible with torch 2.4.1, numpy 2.4.6, torchdata? | **Unknown:** the wheel index domain has expired and no wheel was supplied. |

## Decisions for the author

| # | Decision | Options | Recommendation | Reason |
|---|---|---|---|---|
| 1 | Where does the DGL wheel come from? `data.dgl.ai` has expired (both the 2.4 and the 2.6 index die with it) | (a) supply the `dgl-2.4.0+cu124-cp312-…manylinux…x86_64.whl` file (any copy you hold, or another trusted source) and I run E1(b) on it before any cluster change; (b) if you hold the `dgl-2.5.0+cu124` wheel (CS card, decision 1), supply that and inspect it instead; (c) build DGL from the `dmlc/dgl` clone | (a) or (b): inspect the actual file first | Option B was chosen on the premise that the 2.4.0 CUDA wheel is obtainable; the premise is unverified, so the cluster should not move to torch 2.4.1 until the file is in hand and its graphbolt library, Python tag and `numpy`/`torchdata` constraints are read |
| 2 | torch 2.4.1 vs 2.6.0 for the cluster | (a) torch 2.4.1 with `linux-lock-b.txt`; (b) keep 2.6.0 | decide after #1 | Nothing in the laptop tests favours either torch version; the deciding fact is DGL |
| 3 | `hyper-connections==0.4.2` in the lock | adopt (needed under 2.4.1) / drop if (b) in #2 | adopt with 2.4.1 | Without it pip upgrades torch to 2.15.0 |
| 4 | `torchdata==0.8.0` in the lock | pin / leave unpinned | pin | 0.10+ has no `datapipes`; DGL 2.2.1's graphbolt imports it (2.4.0 unverified) |
| 5 | `numpy` pin | keep 2.4.6 / `numpy<2` | keep for now; decide with the 2.4.0 wheel's metadata | No NumPy-1.x message for DGL 2.2.1 on Windows; 2.4.0 Linux untested; changing the pin affects every other library |
| 6 | Golden captures (`v2_golden.pt`, `mb0_reference.pt`) | re-capture on the cluster (CS decision 3a) | yes, independent of #2 | E5 shows the torch version is not the cause, so the 1-ulp mismatch will persist on Linux whichever torch is chosen |
| 7 | Lock transitive packages | pin `local-attention` and its chain explicitly / leave | consider, for the final lock | An unpinned transitive chain changed the resolver outcome between torch 2.6 and 2.4 |

## Out-of-scope findings (listed, not fixed)

- `linux-lock.txt` and `check_env.py` live only on `cluster-setup`, not on `part-mb`; the card says they exist on this base.
- Existing `h2snn` has `torchdata 0.11.0`, which lacks `torchdata.datapipes`; harmless on the laptop only because the shim stubs graphbolt.
- Windows `dgl 2.2.1` has no graphbolt for torch ≥ 2.4: the Windows DGL stays on the shim route whatever torch is chosen.
- `pip install --dry-run` downloads the full 2.5 GB torch wheel; `e2_commands.txt` contains conda progress-bar escape noise.
