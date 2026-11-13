# Report — Option B check on the laptop (E0–E6)

**Branch** `env-b-check` from `part-mb` at `da3e1f535c20d3c2d364712a073b1ccfb6a62d9e` (hash present after `git fetch`; VERIFIED-RUN `git rev-parse HEAD` in `e0_baseline.txt`). One commit per unit (E3 has an addendum commit E3b). Evidence: `docs/envB_evidence/`. Date 2026-11-14. `v3b` and `part-mb` untouched; spec and cards untouched; `h2snn` only read (pip freeze, suite, import checks).

## Verdict

- **Torch 2.4.1 itself is compatible** with every required library that can be tested on Windows, and with the full suite: in `h2snn-b` the suite gives the same result as in `h2snn` (torch 2.6.0) except one test whose library (`torch_householder`) the card did not install (E4). No bit-exact test differs.
- **Pre-registered expectation 1 failed as stated:** with the CS2 pins alone, pip *replaces torch 2.4.1 by 2.15.0*. Cause: an unpinned chain `performer-pytorch 1.1.4 → local-attention 1.11.2 → hyper-connections ≥ 0.4.3 (needs torch ≥ 2.5)`. One extra pin, `hyper-connections==0.4.2`, fixes it; `pip check` is then clean (E2).
- **No DGL 2.4.0 wheel could be inspected; the author later supplied the 2.4.0 SOURCE tree (`H2S-NN/vendor/dgl-2.4.0`, E1b).** `data.dgl.ai` is not just unreachable over TLS: plain HTTP shows a Namecheap page "**Domain registration has expired**" (E1); it serves every `repo.html` index (torch-2.4, torch-2.6, the 2.5.0 pin). The source tree answers two of the card's questions and changes the plan: **DGL 2.4.0 declares `torch<=2.4.0`** (so torch **2.4.1 is excluded**; the lock must say 2.4.0) and **does not need `torchdata`** (so my `torchdata==0.8.0` pin is dropped). Python tag and the graphbolt libraries of the official wheel remain unknown (a source tree has neither).
- **torch 2.4.0 itself was not tested on the laptop;** E2–E5 ran with 2.4.1. Torch 2.4.0 and 2.4.1 are patch releases, but this is a gap: a repeat of E2–E5 with 2.4.0 is Decision 1 below.
- **`torch_householder` (CSNN/NSD) installs next to torch 2.4.1 and passes `pip check`, but cannot be imported on the laptop** (JIT compile needs MSVC `cl`, absent; E7). Its one C++ source is CPU-only, so the Linux cluster needs `g++` and `ninja` only.
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

## E1b — the DGL 2.4.0 files supplied by the author (commit with E7)

`e1b_dgl_2.4.0_source_inspection.txt` (read-only; nothing installed or built; `H2S-NN/vendor/dgl-2.4.0` is untracked and was not staged). **It is a source tree, not a wheel** (2112 files, `python/dgl/_ffi/libinfo.py:108` `__version__ = "2.4.0"`, no `.whl/.so/.dll`; `third_party/` with its 13 git submodules is absent, so it cannot be built as it stands). Answers to the E1 questions:

| Question | Answer (VERIFIED-CODE, path relative to the tree) |
|---|---|
| Python tag | none (set by whoever builds the wheel) |
| `numpy` constraint | `numpy>=1.14.0`, no upper bound (`python/setup.py:224`) |
| `torchdata` constraint | **none**: not in `install_requires`; no `import torchdata` anywhere under `python/`; GraphBolt carries its own datapipes copy (`python/dgl/graphbolt/datapipes/__init__.py:1`, "mostly copied from torchdata==0.7.1") |
| `torch` constraint | **`torch<=2.4.0`** (`python/setup.py:233`): **torch 2.4.1 violates it** |
| graphbolt library | looked up by the exact torch version before `+`: `libgraphbolt_pytorch_<ver>.so` (`python/dgl/graphbolt/__init__.py:52-56`; `dgl_sparse` likewise `python/dgl/sparse/__init__.py:19-22`); one library is built per torch version (`graphbolt/CMakeLists.txt:47`, `graphbolt/build.sh:40-42`). Which versions the official wheel contains: unknown (UNVERIFIED) |

Consequences: (1) with `dgl==2.4.0` in the lock, **torch 2.4.1 cannot be pinned**; pip would downgrade torch to 2.4.0 or fail, and `pip check` would fail. The candidate lock now says `torch==2.4.0`. (2) The pre-registered concern about `torchdata` does not apply to 2.4.0; the `torchdata==0.8.0` pin is dropped (it stays true for the Windows 2.2.1 build). (3) The NumPy-2 concern for 2.4.0 cannot be settled from source (only a lower bound is declared). (4) A wheel still has to be built (needs the submodules, a CUDA 12.4 toolchain and network access to GitHub) or obtained; this source tree alone does not give the cluster a wheel.

## E7 — `torch_householder` next to torch 2.4.1 (added at the author's request)

`e7_torch_householder.txt` (VERIFIED-RUN, `h2snn-b`; this changes `h2snn-b` after E4):

| Check | Result |
|---|---|
| `pip install --no-cache-dir torch_householder==1.0.1` | **ok**: only an sdist on PyPI (no wheel); pip built a wheel and added `ninja 1.13.2`; `pip check`: No broken requirements found |
| `import torch_householder` (CPU) | **fails**: the module JIT-compiles `householder.cpp` at import (`torch.utils.cpp_extension.load`) and torch calls `where cl`: `CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1` (no MSVC on the laptop; `gcc` from Strawberry exists but torch on Windows requires `cl`; no `nvcc`) |
| `tests.test_baselines.DglCudaGateTests` after the install | Ran 7, **OK** (the two E4 subtests csnn/nsd now pass; they only check that the package is present) |

So on the laptop CSNN/NSD cannot run for the same reason as in `h2snn` (CSNN one-epoch tests skip with "no `cl`"). **Not tested: whether `householder.cpp` compiles against torch 2.4.x headers on Linux.** The source is a single CPU file (`sources=[householder.cpp]`, no CUDA), so the cluster test needs only `g++` and `ninja`: install `torch_householder==1.0.1` in a job, `python -c "import torch_householder"` (compiles on import), then the CSNN and NSD one-epoch tests. That test belongs in the rewritten cluster card (CS7-1).

## E2 — The environment `h2snn-b` (commit `df6e475`)

**Done.** `conda create -n h2snn-b python=3.12` needed no administrator rights (Python 3.12.16). torch 2.4.1 exists for cp312 `win_amd64` on the cu124 index (`torch-2.4.1+cu124-cp312-cp312-win_amd64.whl`, 2506 MB); installed with `pip install --no-cache-dir`. Commands and all outputs: `e2_commands.txt` (it contains conda progress-bar noise; no output was edited).

| Step | Result |
|---|---|
| torch 2.4.1 (cu124 index) | installed; `torch.cuda.is_available()` True |
| `torch_scatter==2.1.2`, `torch_sparse==0.6.18`, `torch_cluster==1.6.3` from `data.pyg.org/whl/torch-2.4.0+cu124.html` | **all three exist for cp312 win_amd64** (`+pt24cu124`) — expectation 2 holds; nothing compiled. This step also pulled numpy 2.5.4 / scipy 1.18.1, replaced by the pins in the later step |
| `torch_geometric==2.8.0` | installed |
| all other pins of the lock (except torch, PyG extensions, `torch_householder`, `dgl`) | **dry-run replaces torch by 2.15.0** (see below); with `hyper-connections==0.4.2` added: installed, torch stays 2.4.1+cu124 |
| `dgl==2.2.1` + `torchdata==0.8.0` | installed (the torchdata pin was my choice for the 2.2.1 build; dropped from the lock after E1b) |
| `pip check` | **No broken requirements found.** |

**Finding (expectation 1 not met as stated).** `pip install --dry-run -r <the 19 other exact pins>` ends in `Would install … torch-2.15.0 torch-einops-utils-0.2.6 …`. Chain (PyPI metadata, read-only): `performer-pytorch==1.1.4` requires `local-attention>=1.1.1`; `local-attention 1.11.2` requires `hyper-connections>=0.1.8`; `hyper-connections` 0.4.3 and later require `torch>=2.5` (0.4.8+ also `torch-einops-utils`, which needs `torch>=2.5`). `hyper-connections 0.4.2` is the newest release with `torch>=2.3`. The existing `h2snn` has `hyper-connections 0.4.11` because it runs torch 2.6. The torch-2.6 lock leaves this chain unpinned, which is harmless on 2.6 but a trap on 2.4.

**Candidate lock** `H2S-NN/requirements/linux-lock-b.txt` (new file; `linux-lock.txt` and `check_env.py` untouched). Differences from the CS2 lock: `torch==2.4.0` (**revised after E1b**: DGL 2.4.0 needs `torch<=2.4.0`; the laptop ran 2.4.1); PyG index `torch-2.4.0+cu124` (now the exact torch version); `dgl==2.4.0+cu124` with index `torch-2.4/cu124` (**untested**); added `hyper-connections==0.4.2` (reason in a comment). `torchdata` is **not** pinned (revised after E1b: DGL 2.4.0 does not use it). Header comments say all of this.

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

**Expectation 3, checked as far as possible.** `torchdata`: DGL's graphbolt imports `torchdata.datapipes` (VERIFIED-CODE in the installed dgl 2.2.1: `graphbolt/base.py:20`, `dataloader.py:7-8`, `neighbor_sampler.py:8`, `item_sampler.py:11`, `minibatch_transformer.py:5`); `torchdata` 0.10+ has no `datapipes` module (`h2snn` has 0.11.0: `ModuleNotFoundError: No module named 'torchdata.datapipes'`). So a graphbolt-enabled DGL 2.2.1 needs `torchdata<=0.9`. **Revised by E1b:** DGL 2.4.0 does not import torchdata at all, so no torchdata pin is needed for it. NumPy 2: the 2.2.1 Windows CPU wheel runs with numpy 2.4.6 without any message for the APIs tried; **DGL 2.4.0 Linux with numpy 2.4.6 is untested** (no wheel; its source declares only `numpy>=1.14.0`), so no `numpy<2` decision can be made from this card.

## E4 — Full suite (commit `7b191de`)

`cd H2S-NN; python -m unittest discover -s tests -v`, same commit (`c1992fc`, docs-only difference from `part-mb`), same machine, one after the other. Logs `e4_suite_h2snn-b.txt`, `e4_suite_h2snn.txt`; per-test comparison `e4_diff.py` → `e4_per_test_diff.txt`.

| Env | Result |
|---|---|
| `h2snn` (torch 2.6.0) | Ran 674 — FAILED (failures=2, errors=2, skipped=7) = Part MB-fix reference |
| `h2snn-b` (torch 2.4.1) | Ran 674 — FAILED (failures=4, errors=2, skipped=7) |

Per-test difference: **one test, two subtests** — `test_baselines.DglCudaGateTests.test_only_dgl_backed_models_are_gated` (`model='csnn'`, `model='nsd'`). Class: **library missing** (`torch_householder` not installed in `h2snn-b`). First error line: `AssertionError: 'missing requirement group sheaf (torch_householder) -- install requirements/baselines-sheaf.txt' is not None` (`tests/test_baselines.py:1394`). The same test fails on the cluster for the same reason (CS5). The 4 common non-passing tests are the known laptop set (notebooks with outputs; missing vendor clones `sehgnn, sehgnn_data, pshgcn, pshgcn_lp, chebnetii`; PSHGCN-dependent `test_colliding_vendor_module_names_do_not_leak` and `test_vendor_model_init_is_reproducible_under_seed_everything`). Skipped lines (empty vendor clones) are identical. **No bit-exact capture differs.** Nothing was fixed. Expectation 4 holds (the exception it names did not occur). Wall-clock (1653 s vs 2432 s) is not comparable across the two runs. `e4_diff.py` counts 673 distinct ids against unittest's 674 (cause not investigated); the totals above are unittest's own.

**Afterwards (E7):** `torch_householder==1.0.1` installs next to torch 2.4.1 (`pip check` clean) but cannot be imported on the laptop (JIT build needs MSVC); with it installed the two gate subtests pass. The suite logs above were taken before that install.

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
| Is DGL 2.4.0 compatible with torch 2.4.1, numpy 2.4.6, torchdata? | **torch 2.4.1: no** (source declares `torch<=2.4.0`); torchdata: not needed; numpy: only a lower bound is declared; the Linux wheel itself is still unavailable and untested. |

## Decisions for the author

| # | Decision | Options | Recommendation | Reason |
|---|---|---|---|---|
| 1 | torch version for the cluster, given DGL 2.4.0 needs `torch<=2.4.0` | (a) `torch==2.4.0`: first repeat E2–E5 on the laptop in a new env `h2snn-b240` (2.5 GB download, one suite ≈ 30–40 min); (b) keep 2.6.0 and find a DGL for it; (c) go ahead with 2.4.0 on the cluster without the repeat | (a) | The laptop evidence is for 2.4.1, which DGL 2.4.0 forbids; the repeat is cheap next to a failed cluster setup |
| 2 | How the cluster gets a DGL wheel (the index domain has expired; only the 2.4.0 source tree is in hand) | (a) build `dgl 2.4.0+cu124` on the cluster from `vendor/dgl-2.4.0` + its 13 submodules (needs CUDA 12.4 toolkit, cmake, GitHub access; a build card); (b) supply a prebuilt `dgl-2.4.0` cu124 cp312 manylinux wheel; (c) CPU-only DGL built the same way | (b) if you can find one; else (a) as its own card | A source tree gives no wheel; building is a separate piece of work whose result must then be inspected like E1(b) |
| 3 | `hyper-connections==0.4.2` in the lock | adopt / drop if torch stays 2.6 | adopt with torch 2.4.x | Without it pip upgrades torch to 2.15.0 |
| 4 | `torchdata` pin | none (revised) | none for DGL 2.4.0 | E1b: DGL 2.4.0 does not use torchdata |
| 5 | `numpy` pin | keep 2.4.6 / `numpy<2` | keep for now; decide with the 2.4.0 wheel's metadata | No NumPy-1.x message for DGL 2.2.1 on Windows; 2.4.0 Linux untested; changing the pin affects every other library |
| 6 | Golden captures (`v2_golden.pt`, `mb0_reference.pt`) | re-capture on the cluster (CS decision 3a) | yes, independent of #1 | E5 shows the torch version is not the cause, so the 1-ulp mismatch will persist on Linux whichever torch is chosen |
| 7 | Lock transitive packages | pin `local-attention` and its chain explicitly / leave | consider, for the final lock | An unpinned transitive chain changed the resolver outcome between torch 2.6 and 2.4 |

## Out-of-scope findings (listed, not fixed)

- `linux-lock.txt` and `check_env.py` live only on `cluster-setup`, not on `part-mb`; the card says they exist on this base.
- Existing `h2snn` has `torchdata 0.11.0`, which lacks `torchdata.datapipes`; harmless on the laptop only because the shim stubs graphbolt.
- Windows `dgl 2.2.1` has no graphbolt for torch ≥ 2.4: the Windows DGL stays on the shim route whatever torch is chosen.
- `pip install --dry-run` downloads the full 2.5 GB torch wheel; `e2_commands.txt` contains conda progress-bar escape noise.
