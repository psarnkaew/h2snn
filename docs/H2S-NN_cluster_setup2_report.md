# Report — Cluster setup 2 (CS7-0 to CS7-6)

**Branch** `cluster-setup2`. It was created from `cluster-setup` at `d8f5813b19865121c99e9e296a25ca7789eb5b91` and merged with `design-docs` at `2d9ea70f878b474d1bd979ff370730630230bea7`. Both hashes were checked after `git fetch`, and the merge had no conflict.
**Card** `docs/H2S-NN_cluster_setup2_card.md`.
**Evidence** `docs/cluster_evidence2/`: each job script sits next to its output, and each output starts with `git rev-parse HEAD`, the hostname and the job id (plus the step id and GPU where relevant).
**Rules followed:**
- One commit per unit, staged by explicit path, and pushed after each unit.
- No other branch was touched.
- The environment `h2snn` (torch 2.6.0) was not modified.

Date 2026-11-15.

**Where the jobs ran.** The author held an interactive allocation, **job 16721** on g2n01 (RTX 2080 Ti, 6 CPUs, 32 GB, 6 h), and asked that this session use it. Most units therefore ran as SLURM steps inside it, with `srun --jobid=16721 --overlap bash <script>`. Every output records `job=16721 step=<n>`. Work that needed another node or GPU type ran as separate `sbatch` jobs:

| Job | Node | Purpose |
|---|---|---|
| 16722 | g2n01, CPU | build of `h2snn-b` |
| 16723 | g3an01, A6000 | DGL check on sm_86 |
| 16724 | g3an01 | CS7-3 node B |
| 16725 | g3an01 | CS7-5 node B |

Nothing ran on the login node except git, file reads/edits, `sha256sum` of the DGL file and job submission.

## Verdict

- **All four pre-registered expectations hold.**
  1. Torch is `2.4.1+cu124` after every install step. `check_env` reports 0 differences. `pip check` prints exactly one line, DGL's `torch<=2.4.0`.
  2. DGL 2.4.0 imports with numpy 2.4.6 with no NumPy-1.x message. It loads the three `*_pytorch_2.4.1.so` libraries and passes 14/14 known-answer checks on sm_75 and on sm_86, all with diff 0.
  3. `torch_householder` compiles with g++ 14.2. Its output is orthogonal to 6.4e-16 and equals the plain-torch product to 3.0e-16.
  4. Every DGL-, Householder- and PSHGCN/ChebNetII-backed CS5 failure, error and skip now passes.
- **Full suite (CS7-6):** 697 tests. **FAILED (failures=3)**, 0 errors, 0 skips. The three failures are exactly the expected set: the two Windows-golden bit-exact tests and the notebook check. The float64 scope test is green.
- **CS7-3 measurement:** the captures are bit-identical when repeated on node A. They **differ on node B** (g3an01): golden 412/1136 tensors (max 1.9e-6), mb0 39/139 (max 4.8e-7).
  - The two nodes use different CPU kernel paths: `torch.__config__.show()` reports AVX2 on g2n01 and AVX512 on g3an01.
  - Initial weights and data fixtures are identical across the nodes; only forward and backward outputs differ.
  - By the card's rule, **bit-exact references are not portable across the cluster's CPUs**. The author decides (Decision 1).
- No method code was changed, no tolerance loosened and no test skipped. There was no performance run and no metric. The Windows references `v2_golden.pt` and `mb0_reference.pt` are untouched.

## CS7-0 — Preconditions and branch (commit `64211fb`)

**Done.** VERIFIED-RUN on login01; evidence in `cs7_0_preconditions.txt`, committed with CS7-1 because the merge commit carries the CS7-0 message.

- `origin/cluster-setup` = `d8f5813b…`. `origin/design-docs` = `2d9ea70f…`.
- Merge `--no-ff`: no conflict.
- DGL file: **291 228 305 bytes**, sha256 **`0267a1a7…c867a8`**. Both equal the card.
- `sinfo`:

| Node | GPU | State |
|---|---|---|
| g2n01 | `2080ti:1` | idle |
| g3n01 | `3090:2` | mix |
| g3an01 | `a6000:1` | idle |
| g2tn01 | `titan_rtx:2` | **down** |

**Deviation:** g2tn01, the card's example node B, was down, so node B in CS7-3 and CS7-5 is **g3an01**. It has a different CPU model from g2n01 (i9-10900X vs i9-10900K).

## CS7-1 — Environment `h2snn-b` (commit `ed3ea99`)

### (a) Lock and install script

- `H2S-NN/requirements/linux-lock-b.txt`: the `-f https://data.dgl.ai/...` line is removed. `dgl==2.4.0+cu124` stays as a check line, with the card's comment ("installed by conda from the dglteam package by scripts/setup/install_env_b.sh; pip must not install this line"). The DGL comment keeps the package URL and sha256, and the script reads that value.
- New `H2S-NN/scripts/setup/install_env_b.sh ENV_NAME DGL_FILE` runs these steps:
  1. torch `2.4.1+cu124` from the cu124 index, with `--no-cache-dir`.
  2. `torch_scatter`, `torch_sparse` and `torch_cluster` from the torch-2.4.0+cu124 PyG index. Versions are read from the lock, and only binary wheels are allowed.
  3. The lock filtered by `grep -v '^dgl=='` into a temporary file, then `pip install -r`.
  4. Torch is checked after every install step, and the script exits 1 unless it is `2.4.1+cu124`.
  5. The DGL file's sha256 is checked against the lock, with exit 1 on a mismatch.
  6. `conda install -n ENV --no-deps --offline -y DGL_FILE`.
  7. `pip check`, with its exit code ignored.

  The script refuses an empty `ENV_NAME`, `base` and `h2snn`.
- New `H2S-NN/tests/test_env_b_install.py`, with 4 tests that only read files:
  - The lock has no `data.dgl.ai` line, exactly one `dgl==` line, `hyper-connections==0.4.2`, `torch==2.4.1` and a sha256.
  - The script filters `dgl` before pip and never gives pip the raw lock. It checks sha256 with an exit before `conda install`. It passes `--no-deps`, refuses an empty name before `conda activate`, and never names `h2snn` as a target.
  - Each checker is also exercised on in-test mutations.
- **Tests (VERIFIED-RUN, 16721 step 2, `cs7_1a_tests.txt`):** Ran 4, OK.
- **Mutations** were applied to the real script and restored (`restored: identical`):
  - sha256 check removed → `AssertionError: … ['sha256 is not checked before conda install']`
  - `dgl` not filtered (the `grep -v` line replaced by `cp`) → `['dgl is not filtered from the lock']`
- **Deviation:** the first run of these tests in job 16722 failed (`cs7_1ab_build.txt`). This was a fault in my test, not in the script: its pattern `conda install` matched the line `echo "== step 6: conda install DGL …"` before the command itself. I fixed the test (pattern `^\s*conda install\b`) and reran it in step 2.
  - During that rerun the mutation edited `install_env_b.sh` for about one second while job 16722 was executing it (blocked inside pip).
  - The file was restored byte-identical before pip returned. The 16722 log shows steps 3 → 5 → 6 → 7 in order and `install_exit=0`.
  - Later mutations were applied only to files no running job used.

### (b) Build

VERIFIED-RUN job **16722** (g2n01, CPU, `--mem=16G --time=2:00:00`); log `cs7_1ab_build.txt`.

- `conda create -n h2snn-b python=3.12 -y` gave Python 3.12.16. Then `install_env_b.sh h2snn-b <DGL file>` ran: **`install_exit=0 seconds=899`**.
- Torch after each step: `[1] torch 2.4.1+cu124`, `[2] …`, `[3] …`, `[6] …`.
- Step 3 replaced the numpy 2.5.4 / scipy 1.18.1 that the PyG step had pulled in with the pinned **numpy 2.4.6 / scipy 1.18.0**.
- `torch_householder` was built from its sdist (`torch_householder-1.0.1-cp312-cp312-linux_x86_64.whl`).
- sha256 `lock=…c867a8 file=…c867a8 size=291228305`. Conda: `NEW packages: dgl <unknown>/linux-64::dgl-2.4.0.th24.cu124-py312_0`.
- `python scripts/setup/check_env.py --lock requirements/linux-lock-b.txt` → **`differences: 0; notes: 4`**. The notes are the local tags `torch 2.4.1+cu124` and `torch_cluster/scatter/sparse +pt24cu124`. `check_env_exit=0`.
- `conda list -n h2snn-b dgl` → `dgl 2.4.0.th24.cu124 py312_0`.
- `pip freeze` → `h2snn-b_pip_freeze.txt` (105 lines).

**`pip check` output** (expectation 1), the same after the install, in 16722 and after the full suite:
```
dgl 2.4.0+cu124 has requirement torch<=2.4.0, but you have torch 2.4.1+cu124.
```
Exactly one line, the accepted one.

**Limit.** The script runs `pip check` only after the DGL step. "Clean before DGL" is therefore inferred, not observed: `pip check` lists every broken requirement, and the only line names DGL.

### (c) DGL known-answer checks

`cs7_dgl_check.py`, run with `python -W default`.

| | RTX 2080 Ti, sm_75 (16721 step 3, g2n01) | RTX A6000, sm_86 (job 16723, g3an01) |
|---|---|---|
| torch / numpy / scipy | 2.4.1+cu124 / 2.4.6 / 1.18.0 | same |
| `dgl.__version__` / `importlib.metadata.version("dgl")` | 2.4.0+cu124 / 2.4.0+cu124 | same |
| "compiled using NumPy 1.x" / `_ARRAY_API` messages | none (`grep -c` = 0) | none |
| loaded `libgraphbolt_pytorch_2.4.1.so`, `libdgl_sparse_pytorch_2.4.1.so`, `libtensoradapter_pytorch_2.4.1.so` (`/proc/self/maps`) | PASS (all three) | PASS |
| (i) 6-node graph `update_all(copy_u, sum)` vs `index_add_` | PASS, max\|diff\| 0.000e+00 | PASS, 0.000e+00 |
| (i) `(copy_u, mean)` vs sum / in-degree | PASS, 0.000e+00 | PASS, 0.000e+00 |
| (ii) 3-type heterograph `multi_update_all` sum vs per-relation sum | PASS, 0.000e+00 | PASS, 0.000e+00 |
| (iii) `GraphConv` (identity weight, no bias, `norm='none'`) vs plain sum | PASS, 0.000e+00 | PASS, 0.000e+00 |
| (iii) `GraphConv` / `GATConv` forward and backward finite | PASS / PASS | PASS / PASS |
| (iv) graph from numpy and from `scipy.sparse` == graph from torch (`torch.equal` on `edges()`) | PASS / PASS | PASS / PASS |
| (v) `import dgl.graphbolt`, `import dgl.sparse` | PASS / PASS | PASS / PASS |
| `dependencies.dgl_cuda_report()` | `(True, 'DGL has CUDA support')` | same |
| **Summary** | **14/14** | **14/14** |

The tolerance was rtol = atol = 1e-6 (fixed by the card). DGL printed two notices at import: "DGL backend not selected… Assuming PyTorch", which writes `~/.dgl/config.json`, and a graphbolt `GBWarning` about the CUDA allocator (out-of-scope finding 2).

### (d) torch_householder

VERIFIED-RUN, 16721 step 4 (g2n01); `cs7_1d_householder.txt`.

- `TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b` did not exist before the run.
- `import torch_householder` JIT-compiled the extension in **10.9 s** with **g++ (Debian 14.2.0-19) 14.2.0** and ninja 1.13.2.
- Compiler line, from `build.ninja`:

```
cxx = c++
cflags = -DTORCH_EXTENSION_NAME=torch_householder_cpp ... -isystem .../envs/h2snn-b/.../torch/include ... -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17
ldflags = -shared -L.../torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python
```

- The known answer follows the convention read from the source (VERIFIED-CODE `torch_householder/householder.py:59-84`, `householder.cpp:28-37`): with `tau=None`, each column is normalised to a unit vector u_j, and Q = H₀H₁…H₄·eye(8,5) with H_j = I − 2u_ju_jᵀ.
- For a float64 parameter of shape (8, 5):
  - **max|QᵀQ − I| = 6.405e-16** (< 1e-10): PASS
  - **max|Q − plain-torch product| = 3.036e-16** (atol 1e-10): PASS

### (e) The CS5 failures in `h2snn-b`

VERIFIED-RUN, 16721 step 5 (g2n01, 2080 Ti); `cs7_1e_cs5_rerun.txt`. The run covered `tests.test_baselines`, `test_partP_freebase`, `test_partP_derived_lp`, `test_partPfix_data_root`, `test_check_env` and `test_env_b_install`.

Result: **Ran 112 tests — FAILED (failures=1)**. The only failure is `test_notebooks_are_committed_clean` (`AssertionError: 1 is not None`, the known notebook check). There are 0 errors and 0 skips.

| CS5 row | Now |
|---|---|
| `dgl` not installed: `DglCudaGateTests` (all tests, incl. `test_cuda_dgl_on_linux_is_healthy`); one-epoch gcn, gat, han, hinormer (NC Freebase), han (LP), han (data-root) | **pass** |
| `torch_householder` not installed: `test_only_dgl_backed_models_are_gated` (csnn, nsd); one-epoch nsd, csnn (NC and LP) | **pass** |
| `DglCudaGateTests` and the conda build's version string | no failure; `dgl.__version__` = metadata = `2.4.0+cu124` |

**Deviation:** the PSHGCN/ChebNetII clones of CS7-2 were made on the login node while this run was in progress, so this run already saw them. The vendor-clone tests also passed here. CS7-2 reran them in their own job.

## CS7-2 — The four vendors; ignore rules (commit `37c1afc`)

Clones were made on login01 with git only (`cs7_2_clones.txt`). The latest default-branch commits come from `git ls-remote --symref <url> HEAD` (all on `main`):

| Vendor | URL | Commit | Licence (LICENSE file) | kind |
|---|---|---|---|---|
| PSHGCN | https://github.com/ivam-he/PSHGCN | `f87406ce56ba3780d19aa540d3d1081ff085cca5` | no licence file (none found anywhere in the clone) | clone |
| ChebNetII | https://github.com/ivam-he/ChebNetII | `ded6c18cbe9673234071031767d17826ad632aca` | no licence file | clone |
| Cooperative-Sheaf-NN | https://github.com/andrerg02/Cooperative-Sheaf-NN | `a1a12f78f3548605b375760779a704f3c2c77a09` | no licence file | clone |
| hgbi | — | — | — | **not-needed** |

- `H2S-NN/vendor/VENDORS.tsv` is updated as in the table.
- `H2S-NN/tests/test_vendor_manifest.py`. These are stated test changes:
  - `KINDS` now includes `not-needed`.
  - `test_every_row_has_a_commit` exempts `not-needed` rows.
  - New `test_not_needed_rows_carry_no_url_or_commit`.
  - New `VendorIgnoreTests.test_every_clone_directory_is_ignored`. It is driven by `VENDORS.tsv` through `git check-ignore --no-index`: every `kind=clone` directory must be ignored, `in-repo` vendors and `VENDORS.tsv` must not be.
- `H2S-NN/.gitignore`: `/vendor/<name>/` for every `kind=clone` row except HGB, which keeps its existing `vendor/HGB/*` rule because its `CLONED_COMMIT.txt` is tracked. After this, the vendor clones no longer appear in `git status`.

`clone_vendors.sh` full run: every clone `PRESENT` (the three new ones `CLONED` first), plus:
```
REFUSED     H2GB: dirty clone (modified tracked files) at /mnt/data/peeradon/h2snn/H2S-NN/vendor/H2GB
```
This is not a stop condition. It is reported and was not cleaned. `git -C vendor/H2GB status` shows `examples/use_dataset.py` modified, and also `H2GB/datasets/ieee_cis_dataset.py` and `H2GB/datasets/pokec_dataset.py` (Decision 4).

**GPU job** (VERIFIED-RUN, 16721 step 6, g2n01, 2080 Ti; `cs7_2_vendor_tests.txt`):
- `available_vendors()`: **`all_true True false: []`**.
- `test_vendor_manifest`: Ran 6, OK.
- **Mutation** (`/vendor/PSHGCN/` deleted from `.gitignore`, then restored) → `AssertionError: Lists differ: ['clone not ignored: PSHGCN'] != []`, FAILED. Restored identical, then OK.
- Per vendor:

| CS5 test | Vendor | Result |
|---|---|---|
| `test_every_lp_adapter_runs_one_epoch` | pshgcn (LP) | pass (was skipped) |
| `test_every_nc_adapter_runs_one_epoch` | chebnetii (NC) | pass (was skipped) |
| `test_all_declared_vendor_clones_are_present` | pshgcn, pshgcn_lp, chebnetii | pass (was failing) |
| `test_colliding_vendor_module_names_do_not_leak` | PSHGCN vs H2SGNN `processing.py` | pass (was `FileNotFoundError`) |
| `test_vendor_model_init_is_reproducible_under_seed_everything` | PSHGCN | pass (was `FileNotFoundError`) |

Ran 5 tests, OK.

## CS7-3 — Golden and mb0 captures across two node types (commit `74c8eb9`)

**Scripts that produced the references.**
- `docs/heads_evidence/v2_golden.py`, named in `golden_capture.txt`.
- `docs/partMB_evidence/mb0_reference.py`.

**Deviation:** `v2_golden.py capture` cannot run any more, because its fixture builder `tests.test_models` no longer exists (stated in the script). I used its **`recapture <ref> <out>`** mode. It re-records every output on the fixtures stored inside the Windows reference and only reads that file. `mb0_reference.py save` was used as is.

The captures ran in `h2snn-b`, CPU only (`CUDA_VISIBLE_DEVICES=""`), with `OMP_NUM_THREADS=MKL_NUM_THREADS=1` (`torch.get_num_threads()` = 1). Script: `cs7_3_job.sh`. Comparison: `cs7_tensor_diff.py`, which uses `torch.equal` per tensor and reports every differing name.

| Node | CPU | torch CPU kernels | Job |
|---|---|---|---|
| A: g2n01 (2080 Ti node) | Intel i9-10900K @ 3.70GHz, no avx512f | `CPU capability usage: AVX2` | capture 16721 step 7; again 16721 step 8 |
| B: g3an01 (A6000 node) | Intel i9-10900X @ 3.70GHz, avx512f | `CPU capability usage: AVX512` | job 16724 |

Both nodes report the same build in `torch.__config__.show()`: GCC 9.3, oneAPI MKL 2022.2, C++17. The full output is in each log.

New files: `docs/cluster_evidence2/v2_golden_nodeA.pt` (sha256 `086bfc58…`) and `docs/cluster_evidence2/mb0_reference_nodeA.pt` (sha256 `c193e9ae…`).

| File | Tensors | A again: differing / max\|diff\| | B: differing / max\|diff\| |
|---|---|---|---|
| golden (`v2_golden_nodeA.pt`; 13 cases, 844 record entries + stored fixtures) | 1136 | **0 / 0** | **412 / 1.907e-06** |
| mb0 (`mb0_reference_nodeA.pt`) | 139 | **0 / 0** | **39 / 4.768e-07** |

**Which tensors differ on B.** The full list is in `cs7_3_check_g3an01.txt`, one `DIFF <name>: max|diff|=…` line per tensor.
- **Golden:**
  - All 13 `eval/logits` and all 13 `train/logits` differ, plus 386 `grad/*` tensors.
  - **None of the `init/*` tensors** (initial weights) differs, and none of the stored data fixtures.
- **mb0:**
  - `eval_logits_1`, `eval_logits_2` and `h0` differ, plus 36 `grads.*` (node_encoder 16, node_to_edge 9, path_encoder 4, readout 3, path_gate 2, relation_embedding 1, theta_tau 1).
  - Every WISP draw and `train_loss` is identical.

**Result rule (card):** the captures differ on B, so **bit-exact references are not portable across the cluster's CPUs**. The author decides between pinning tests to one node type and a stated tolerance in a stated precision (Decision 1).

Further reading (DERIVED from the table): the RNG and initialisation are portable between the two nodes; only arithmetic differs. This matches torch selecting AVX2 vs AVX512 CPU kernels. The Windows references are untouched.

## CS7-4 — `mb0_reference.py compare` exit code (commit `260e286`)

- `docs/partMB_evidence/mb0_reference.py compare` now prints `compared N tensors; differing: K|none`, then one line `  DIFF <name>` for every differing tensor (previously the first 5 only), and calls `sys.exit(1 if bad else 0)`. `mb0_reference.pt` is untouched.
- New `H2S-NN/tests/test_mb0_reference_compare.py`:
  - It saves a fresh reference on the running node, so the unperturbed compare is exact on any node.
  - The unperturbed reference → exit 0 and `differing: none`.
  - A copy with `eval_logits_1[0] += 1.0` → exit 1, and the only DIFF line is `DIFF .eval_logits_1`.
- VERIFIED-RUN, 16721 step 9 (`cs7_4_compare_exit.txt`): Ran 2, OK.
- **Mutation** (the `sys.exit` line deleted) → `AssertionError: 0 != 1 : compared 139 tensors; differing: 1`, FAILED. Restored identical, then OK.

## CS7-5 — The scope test in float64 (commit `94c40bf`)

`tests/test_partMB_scopes.py::ScopeTests::test_featureless_node_in_two_scopes_has_its_global_h0`. This is a stated test change, approved 2026-11-15:
- `H2SNN(...).double().eval()`, `graph.x.double()`, `batch.data.x.double()`. Every other line is kept, including the two copy/scope assertions.
- `torch.equal` is replaced by `torch.allclose(scoped, whole[batch.global_node_id], rtol=0, atol=1e-12)`, plus a dtype check (`float64`).
- **Negative control** in the same test: A6's first scoped row is replaced by the encoding of author A7 (`whole[7]`). The test then asserts `allclose` is False and the max difference is > 1e-6.

The model runs in float64 without any change: `node_encoder.py:91-95` allocates `hidden` in `x.dtype`, and line 111 casts the embeddings `.to(x.dtype)` (VERIFIED-CODE). No method code was changed. The test prints the measured max|diff|.

| Node | CPU kernels | Job | Module | max\|diff\| (float64) |
|---|---|---|---|---|
| g2n01 (i9-10900K) | AVX2 | 16721 step 10 | Ran 5, OK | **0.000e+00** |
| g3an01 (i9-10900X) | AVX512 | 16725 | Ran 5, OK | **0.000e+00** |

**Mutation** (`node_encoder.py:109`: `global_node_id[rows] + 1`, on g2n01): `AssertionError: False is not true`, FAILED. The test printed max|diff| = 3.145e+00. The file was restored identical and `git diff` is clean.

## CS7-6 — Full suite, frozen environment (this commit)

VERIFIED-RUN: `docs/cluster_evidence2/cs7_6_run_tests_b.sbatch`. It is a copy of `scripts/slurm/run_tests.sbatch` that activates `h2snn-b` and sets `TORCH_EXTENSIONS_DIR`; its header asks for GPU, `--mem=32G` and `--time=4:00:00`. It ran as **16721 step 11** (g2n01, RTX 2080 Ti, driver 550.163.01) at git `94c40bf`. Output: `cs7_6_full_suite.txt`.

**`Ran 697 tests in 1098.737s — FAILED (failures=3)`**: 0 errors, 0 skips.

| Failure | First error line | Expected by the card |
|---|---|---|
| `test_baselines.BaselineNotebookTests.test_notebooks_are_committed_clean` | `AssertionError: 1 is not None` | yes |
| `test_cooperative.SymmetricIsThePreviousModelTests.test_T1_symmetric_reproduces_the_golden_outputs_bit_for_bit` | `591 != 0 : ['[toy_acm] MISMATCH init/node_encoder.encoders.0.0.weight max|diff|=2.980e-08', …` | yes (Windows golden) |
| `test_task_heads.NodeClassificationGoldenTests.test_outputs_and_gradients_are_bit_identical_to_the_pre_removal_capture` | `591 != 0 : [… same first line]` | yes (Windows golden) |

Every other test is green, including the float64 scope test. No NumPy-1.x message appears in the log.

### Classification against CS5 (job 16615, `h2snn`, torch 2.6.0)

CS5: 689 tests, failures=15, errors=12, skipped=2. The +8 tests are the new ones: `test_env_b_install` 4, `test_vendor_manifest` +2, `test_mb0_reference_compare` 2.

| CS5 cause | CS5 count | Now |
|---|---|---|
| Known notebook check | 1 F | **still fails** (expected) |
| Vendor clones missing (`test_all_declared_vendor_clones_are_present`, `…colliding…`, `…seed_everything`) | 1 F + 2 E | **pass** (CS7-2) |
| `dgl` not installed (`DglCudaGateTests`, one-epoch gcn/gat/han/hinormer/han LP/han data-root) | 7 F + 6 E | **pass** (CS7-1) |
| `torch_householder` not installed (gate csnn/nsd; one-epoch nsd, csnn NC/LP) | 2 F + 4 E | **pass** (CS7-1) |
| 1-ulp platform difference: `test_task_heads` golden, `test_cooperative` T1 golden | 2 F | **still fail** (expected; Windows capture; 591 mismatches, CS5 had 593) |
| 1-ulp platform difference: `test_partMB_scopes` featureless/global_h0 | 1 F | **pass** (CS7-5, float64) |
| Manifest test `test_every_row_has_a_commit` (4 unresolved URLs) | 1 F | **pass** (CS7-2) |
| Skips: one-epoch pshgcn (LP), chebnetii (NC) | 2 S | **run and pass** (CS7-2) |

The 591 golden mismatches include `init/*` tensors, so the Windows capture differs from the cluster already at initialisation. Between the two cluster nodes the `init/*` tensors are identical (CS7-3).

### mb0 compare, run in the same job

Both compares used CPU and 1 thread, as in the CS7-3 captures.

| Reference | Result | Exit |
|---|---|---|
| Windows `docs/partMB_evidence/mb0_reference.pt` | `compared 139 tensors; differing: 38`, then 38 DIFF names (`.eval_logits_1`, `.eval_logits_2`, `.h0`, `.grads.theta_tau`, 34 more `.grads.*`) | **1** |
| `docs/cluster_evidence2/mb0_reference_nodeA.pt` (g2n01) | `compared 139 tensors; differing: none` | **0** |

### Frozen environment

- **`H2S-NN/requirements/linux-lock-b.freeze.txt`** is the `pip freeze` of `h2snn-b` after the suite: 105 package lines. Its header records the date (2026-11-15T09:52:31+09:00), node (g2n01), job (16721 step 11) and git (`94c40bf`).
- It is identical to the freeze taken after the build (`h2snn-b_pip_freeze.txt`).
- It contains one line that pip cannot install from: `packaging @ file:///home/conda/feedstock_root/...`, the conda-provided package of the env's Python. The header says this, and that `dgl` comes from the conda package.

## Decisions for the author

| # | Decision | Options | Recommendation | Reason |
|---|---|---|---|---|
| 1 | Bit-exact references (`v2_golden.pt` tests, `mb0_reference.pt`). CS7-3: not portable between AVX2 (g2n01) and AVX512 (g3an01) nodes | (a) re-capture on one node type and run the bit-exact tests only there; (b) a stated tolerance in a stated precision (e.g. float64 as in CS7-5); (c) before choosing, one short job: pin the kernel path on every node (`ATEN_CPU_CAPABILITY=avx2`, `MKL_ENABLE_INSTRUCTIONS=AVX2`), capture on A, compare on B. **UNTESTED**: MKL and oneDNN may still dispatch differently | (c), then (a) with the pinned path if (c) gives 0 differences; otherwise choose between (a) and (b) | The initial weights already agree across nodes. The forward/backward differences match the AVX2/AVX512 kernel choice that torch reports, so pinning the path may make one capture valid on every node; this is not shown yet |
| 2 | `h2snn-b` as the default (`run_tests.sbatch`, CLAUDE.md/AGENTS.md, replacing `linux-lock.txt`) | approve / not yet | approve, as the planned later step | All expectations hold; the full suite fails only on the three expected tests |
| 3 | `linux-lock-b.freeze.txt` | keep as a record (as written) / turn into an install constraint file | keep as a record | The `packaging @ file://` line and the conda-installed `dgl` make it unusable as a pip input without editing |
| 4 | `vendor/H2GB` is a dirty clone: `examples/use_dataset.py`, `H2GB/datasets/ieee_cis_dataset.py`, `H2GB/datasets/pokec_dataset.py` modified | move the author's changes under `scripts/` (M3b open item) / keep | decide in M3b | `clone_vendors.sh` refuses H2GB while it is dirty; I did not touch it |
| 5 | PSHGCN, ChebNetII, Cooperative-Sheaf-NN have no licence file | note | never commit their files (now git-ignored) | As for HGB |
| 6 | The Windows golden files | keep as history (working preferences) | keep | Re-capture follows Decision 1 |

## Out-of-scope findings (listed, not fixed)

1. `requirements/baselines-dgl.txt` still names the dead `data.dgl.ai` index (the card already lists it as out of scope).
2. Importing `dgl.graphbolt` prints a `GBWarning`: graphbolt enables an experimental CUDA allocator setting (`expandable_segments`) unless `PYTORCH_CUDA_ALLOC_CONF` is set. This may matter for memory and timing in later GPU runs.
3. DGL's first import writes `~/.dgl/config.json`, choosing the pytorch backend. This is harmless and sits in home.
4. `torch.utils.checkpoint` prints `FutureWarning: torch.cpu.amp.autocast(...) is deprecated` under torch 2.4.1 (from torch's own code).
5. `v2_golden.py capture` cannot run (`tests.test_models` was removed); only `recapture` works.
6. Building `torch_householder` from its sdist spends several minutes in pip's isolated "Installing build dependencies" step (1.0.1 has no wheel). An install without network access would need `--no-build-isolation`. Not changed here.
7. The `h2snn` environment (torch 2.6.0) still lacks DGL and `torch_householder`. It is unchanged by this card.

branch=cluster-setup2 sha=<see final message> card=docs/H2S-NN_cluster_setup2_card.md base=2d9ea70f878b474d1bd979ff370730630230bea7
