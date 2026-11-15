# Report — Cluster setup 3 (CS8-0 to CS8-5)

**Branch** `cluster-setup3`. It was created from `v3b` at `8bf2e8f2f3e4024f7c5afc66fc54157d0b3e5e68` and merged with `design-docs` at `c085395167b8ea524c0ff8ce37f6ec0ddc484629`. Both hashes were checked after `git fetch`, and the merge had no conflict.
**Card** `docs/H2S-NN_cluster_setup3_card.md`.
**Evidence** `docs/cluster_evidence3/`: each job script sits next to its output. Each output starts with `git rev-parse HEAD`, the hostname, the job id (and step id), the CPU model (`lscpu`), `torch.backends.cpu.get_cpu_capability()` and `torch.get_num_threads()`.
**Rules followed:**
- One commit per unit, staged by explicit path, pushed after each unit; no other branch touched.
- `h2snn` not modified.
- `src/` not changed.
- The Windows references and the CS7 files in `docs/cluster_evidence2/` were not overwritten.

Date 2026-11-15.

**Where the jobs ran.** The author's interactive allocation **job 16721** (g2n01, RTX 2080 Ti, 6 CPUs, 32 GB) was used for steps on g2n01: `srun --jobid=16721 --overlap`, recorded as `job=16721 step=<n>`. The author asked for this. Work that needed 8 CPUs, another node or a clean comparison ran as `sbatch --nodelist=…` jobs.

**Mutations.** Each was applied only while no job of mine was running, and was restored (byte-identical, checked with `cmp`) before the next submission. The script and test mutations of CS8-3 ran inside a single job, one step after another.

## Verdict

- **Rule R2 applies (CS8-1).** With the pinned kernel environment E and **1 thread**, captures are bit-identical on node A again (g2n01, i9-10900K) and on node B (g3an01, i9-10900X, AVX512): golden 0/1136 tensors, mb0 0/139. With the test job's default thread count (8) they differ from the 1-thread capture: golden 43/1136 (max 4.768e-07) and mb0 7/139 (max 1.863e-09), with identical difference lists on both nodes. Node C: none (see CS8-0).
- **CS8-2 done under R2.**
  - New cluster references `v2_golden_cluster.pt` and `mb0_reference_cluster.pt`.
  - The golden tests and `mb0_reference.py compare` assert E and compare with 1 thread.
  - With E they pass on both node types. Without E they fail with the pinned-kernel message (compare exits 2).
  - Both mutations behave as the card requires.
- **CS8-3 and CS8-4 done.**
  - `h2snn-b` is the default: `run_tests.sbatch` exports `TORCH_EXTENSIONS_DIR` and E, `check_env.py` defaults to `linux-lock-b.txt`, and CLAUDE.md/AGENTS.md are updated.
  - `baselines-dgl.txt` no longer names the dead index.
- **CS8-5: full suite with the new default on both node types: 697 tests, FAILED (failures=1).** The only failure on each node is the notebook check, as expected under R1/R2. The two golden tests, the mb0 compare test and the float64 scope test are green on both. `mb0_reference.py compare` against `mb0_reference_cluster.pt` gives **`differing: none`, exit 0** on both nodes.

## CS8-0 — Preconditions and branch (merge commit `e0a7ca1`)

**Done.** VERIFIED-RUN `git fetch`:
- `origin/v3b` = `8bf2e8f2…`. `origin/design-docs` = `c0853951…`.
- `git checkout -b cluster-setup3 8bf2e8f…`, then `git merge --no-ff c085395…`: no conflict.

Evidence `cs8_0_preconditions.txt`, committed with CS8-1 because the merge commit carries the CS8-0 message.

`sinfo` and CPU per node. `scontrol show node` lists no CPU model (`AvailableFeatures=(null)`), so the source is one tiny CPU job per node that was up (`cs8_0_cpuinfo_job.sh`):

| Node | GPU | State | CPU (`lscpu`) | avx2 / avx512f | capability without E / with E | Job |
|---|---|---|---|---|---|---|
| g2n01 | 2080ti:1 | mix (author's allocation) | i9-10900K @ 3.70GHz | yes / no | AVX2 / AVX2 | 16721 step 12 |
| g3an01 | a6000:1 | idle | i9-10900X @ 3.70GHz | yes / **yes** | **AVX512 / AVX2** | 16726 |
| g3n01 | 3090:2 | mix | i9-10900K @ 3.70GHz | yes / no | AVX2 / AVX2 | 16727 |
| g2tn01 | titan_rtx:2 | **down** | — | — | — | — |

E has the expected effect: g3an01 reports `AVX2` with E.

## CS8-1 — Does E make captures portable? (commit `4fc33df`)

Setup:
- E is exported before Python starts: `ATEN_CPU_CAPABILITY=avx2`, `MKL_CBWR=AVX2`, `ONEDNN_MAX_CPU_ISA=AVX2`.
- CPU only (`CUDA_VISIBLE_DEVICES=""`), env `h2snn-b`. Script `cs8_1_job.sh`.
- Captures: `v2_golden.py recapture <Windows ref> <out>` (the fixtures come from the Windows file) and `mb0_reference.py save <out>`.
- Comparison: `docs/cluster_evidence2/cs7_tensor_diff.py`.

Steps:
1. **Capture on A = g2n01** with 1 thread (`OMP_NUM_THREADS=MKL_NUM_THREADS=1`; torch reports `threads 1`, `cpu_capability AVX2`) in 16721 step 13. Outputs: `docs/cluster_evidence3/v2_golden_E.pt` (sha256 `b97d9ad7…445f`) and `mb0_reference_E.pt` (sha256 `83f5f57d…1700`).
2. **Compare on A again** (job 16728) and on **B = g3an01** (job 16729). Each job was `sbatch --cpus-per-task=8` and ran twice: with 1 thread, then with the torch default (`threads 8`).
3. **Node C: none.** g3n01 has the same CPU model as A (i9-10900K), and g2tn01 is down.

| File | Node | Threads | `get_cpu_capability()` | Differing | max\|diff\| |
|---|---|---|---|---|---|
| golden (1136 tensors) | A g2n01 | 1 | AVX2 | **0** | 0 |
| golden | A g2n01 | 8 | AVX2 | 43 | 4.768e-07 |
| golden | B g3an01 | 1 | AVX2 | **0** | 0 |
| golden | B g3an01 | 8 | AVX2 | 43 | 4.768e-07 |
| mb0 (139 tensors) | A g2n01 | 1 | AVX2 | **0** | 0 |
| mb0 | A g2n01 | 8 | AVX2 | 7 | 1.863e-09 |
| mb0 | B g3an01 | 1 | AVX2 | **0** | 0 |
| mb0 | B g3an01 | 8 | AVX2 | 7 | 1.863e-09 |

The 8-thread differences are only gradients (golden: `grad/node_encoder.encoders.*.1.*` and similar). The two nodes list the same names with the same values (`diff` of the DIFF lines: identical).

**Rule: R2.** There are 0 differences on every node with 1 thread, but differences with 8 threads. So E plus 1 thread is used inside the bit-exact checks.

## CS8-2 — Cluster references and the bit-exact tests (commit `b500b7c`)

### New and changed files

**New reference files:**
- `docs/heads_evidence/v2_golden_cluster.pt`, a copy of `v2_golden_E.pt` (sha256 equal: `b97d9ad7fd7c4f4bd0cd78e6df824621112743fb0919c6bfcc4fa9814826445f`).
- `docs/partMB_evidence/mb0_reference_cluster.pt`, a copy of `mb0_reference_E.pt` (`83f5f57d4feee19624f5bb9e38e935032813546c7a159c4f523d36d183c91700`).
- The Windows files are unchanged.

**Provenance:**
- `docs/heads_evidence/golden_capture.txt` now records: node, CPU, E, 1 thread, job 16721 step 13, git `e0a7ca1…`, sha256, and the CS8-1 portability result. The Windows line is kept and named as history.
- The docstring of `docs/partMB_evidence/mb0_reference.py` records the same.

**Tests** (stated test changes):
- New helper `H2S-NN/tests/pinned_kernels.py`:
  - `assert_kernel_path_pinned(case)` asserts `torch.backends.cpu.get_cpu_capability() == "AVX2"` and `os.environ.get("MKL_CBWR") == "AVX2"`. Otherwise it fails with "bit-exact test needs the pinned kernel environment: run through scripts/slurm/run_tests.sbatch" (followed by the observed values).
  - `one_thread()` sets `torch.set_num_threads(1)` and restores the previous count afterwards.
- `tests/test_task_heads.py` and `tests/test_cooperative.py`:
  - `GOLDEN_REFERENCE` now points to `v2_golden_cluster.pt` (path only).
  - Before comparing, each golden test calls `assert_kernel_path_pinned(self)` and runs `golden.compare` inside `one_thread()`.
  - In `test_cooperative.py` the non-vacuity control test reads the same constant, so it now also reads the cluster file. It uses only the fixtures and key set, which are identical to the Windows file's.

**`mb0_reference.py`:**
- `compare` exits **2** with the same message unless the kernel path is pinned.
- `references()` runs with 1 thread, restored afterwards, in both `save` and `compare`.
- The card names `mb0_reference.py`. Applying 1 thread to `save` as well keeps future saves on the same terms as the reference.

### Runs (VERIFIED-RUN, `cs8_2_job.sh`, `--cpus-per-task=8`)

| Job | Node | E | Golden tests (`NodeClassificationGoldenTests`, `SymmetricIsThePreviousModelTests`) | `test_mb0_reference_compare` | `mb0 compare mb0_reference_cluster.pt` |
|---|---|---|---|---|---|
| 16730 | g2n01 | yes (AVX2, 8 threads) | Ran 5, **OK** | Ran 2, OK | `differing: none`, exit 0 |
| 16731 | g3an01 | yes (AVX2, 8 threads) | Ran 5, **OK** | Ran 2, OK | `differing: none`, exit 0 |
| 16732 | g3an01 | **no** (AVX512) | FAILED (failures=2): both golden tests `AssertionError: False is not true : bit-exact test needs the pinned kernel environment: run through scripts/slurm/run_tests.sbatch (cpu_capability=AVX512, MKL_CBWR=None)` | FAILED (2): `2 != 1` / `2 != 0` with the message | the message, **exit 2** |

### Mutations (g3an01, `tests/test_task_heads.py`, restored after each)

| Mutation | Job | Result |
|---|---|---|
| The test reads the Windows `v2_golden.pt` again, with E | 16733 | FAIL `AssertionError: 591 != 0 : ['[toy_acm] MISMATCH init/node_encoder.encoders.0.0.weight max|diff|=2.980e-08', …` |
| `assert_kernel_path_pinned(self)` removed, run **without E** | 16734 | the mutated test FAILs on **mismatches**, `AssertionError: 412 != 0 : ['[toy_acm] MISMATCH eval/logits max|diff|=5.960e-08', …`, not on the message. In the same run the unmutated T1 test fails with the message. This shows that the assertion turns a silent platform difference into a clear error |

## CS8-3 — `h2snn-b` as the default (commit `2613a7d`)

**`scripts/slurm/run_tests.sbatch`:**
- Runs `conda activate h2snn-b`.
- Exports `TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b` and the three E lines (`export ATEN_CPU_CAPABILITY=avx2`, `export MKL_CBWR=AVX2`, `export ONEDNN_MAX_CPU_ISA=AVX2`).
- Prints the CPU model (`lscpu`), `get_cpu_capability()` and threads, in addition to job, node, git and GPU.
- Its header explains E.

**`tests/test_slurm_scripts.py`** (stated test change):
- The checker requires `conda activate h2snn-b` (plain `h2snn` no longer passes), each of the three E export lines and the `TORCH_EXTENSIONS_DIR` line.
- `test_checker_catches_each_violation` exercises each new rule.

**`scripts/setup/check_env.py`:** the default lock is `requirements/linux-lock-b.txt`, and the docstring is updated.

**`tests/test_check_env.py`** (stated test change): `test_repository_lock_parses` now asserts:
- the default lock is `linux-lock-b.txt`;
- `torch` 2.4.1, `dgl` 2.4.0+cu124 and `hyper-connections` 0.4.2;
- 28 pins.

**`requirements/linux-lock.txt`:** first line `# superseded by linux-lock-b.txt (2026-11-15); env h2snn (torch 2.6.0) only`.

**`H2S-NN/CLAUDE.md` and `H2S-NN/AGENTS.md`:**
- The cluster section is identical in both (checked with `diff`). It names `h2snn-b`, `linux-lock-b.txt`, `install_env_b.sh`, the freeze record, E and why, the bit-exact references and the pinned-kernel message. It also adds: use of an author allocation with `srun --jobid --overlap`; every job script prints HEAD, hostname and job id; **never mutate a file a running job reads**; push and approval-by-SHA rules; and the **report-line rule** (`branch= sha= card= base=`; nothing pushed after it).
- "Do not commit or push unless asked" → "Commit and push only as the task card and `docs/working_preferences.md` say (push rules 2026-11-14, approval by SHA 2026-11-15)". In CLAUDE.md this is under "Hard rules"; in AGENTS.md it is the last bullet of "Evidence and conduct rules", where the phrase appears.

VERIFIED-RUN, 16721 step 14 (g2n01; `cs8_3_default_env.txt`):
- `bash -n` exit 0.
- `tests.test_slurm_scripts tests.test_check_env`: Ran 11, **OK**.
- `python scripts/setup/check_env.py` (no `--lock`): `lock …/requirements/linux-lock-b.txt; differences: 0; notes: 4`, exit 0.
- **Mutation** (`export MKL_CBWR=AVX2` deleted) → `AssertionError: Lists differ: ['missing export MKL_CBWR=AVX2'] != []`, FAILED. Restored identical, then OK.
- On login01, `sbatch --test-only scripts/slurm/run_tests.sbatch` → accepted (`Job 16735 to start at 2026-11-15T23:01:08 … on nodes g3an01`). That start estimate was wrong: job 16736 started at once.

**Deviation (beyond the listed edits):** the "Testing" paragraph of CLAUDE.md and AGENTS.md said that `test_task_heads.py` compares with `v2_golden.pt`. After CS8-2 that sentence was false, so it now names `v2_golden_cluster.pt`, says it needs E, and names the Windows file as history. No other paragraph was changed.

## CS8-4 — Stale items (commit `b0b2086`)

`requirements/baselines-dgl.txt`:
- Removed `-f https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html` and `dgl==2.5.0+cu124; sys_platform == "linux"`.
- The Linux comment now says that DGL 2.4.0+cu124 comes from the conda package through `scripts/setup/install_env_b.sh` (URL and sha256 in `linux-lock-b.txt`), and that pip must not install it.
- The Windows line `dgl==2.2.1; sys_platform == "win32"` is unchanged.

No test parses this file; `test_baselines.py:1381` only checks that the file name appears in a message.

VERIFIED-RUN, 16721 step 15 (g2n01, 2080 Ti; `cs8_4_dgl_requirements.txt`): `RequirementGroupTests`, `DglCudaGateTests` and `SplitInstallNotebookTests` ran 14, **OK**.

## CS8-5 — Full suite on two node types (this commit)

The suite is `scripts/slurm/run_tests.sbatch` (the new default: `h2snn-b`, E, `TORCH_EXTENSIONS_DIR`) at git `b0b2086`.

| Node | How | CPU / kernels / threads | GPU | Result |
|---|---|---|---|---|
| g3an01 | `sbatch --nodelist=g3an01 scripts/slurm/run_tests.sbatch`, **job 16736** (log copied from `logs/slurm-16736.out` to `cs8_5_suite_g3an01.txt`) | i9-10900X / AVX2 (with E) / 8 | RTX A6000 | **Ran 697 tests in 1108.092s — FAILED (failures=1)** |
| g2n01 | `srun --jobid=16721 --overlap bash scripts/slurm/run_tests.sbatch` (step in the author's allocation; `cs8_5_suite_g2n01.txt`) | i9-10900K / AVX2 / **6** | RTX 2080 Ti | **Ran 697 tests in 1065.580s — FAILED (failures=1)** |

- **On both nodes the only failure is `test_baselines.BaselineNotebookTests.test_notebooks_are_committed_clean`** (`AssertionError: 1 is not None`). There are 0 errors and 0 skips. This is the expected result under R1/R2.
- `test_task_heads` golden, `test_cooperative` T1 golden, `test_mb0_reference_compare` and the float64 scope test are `ok` on both nodes. "pinned kernel" appears 0 times in either log.
- **Compared with CS7-6** (job 16721 step 11, before E): 697 tests and 3 failures there, 1 failure now. The two golden tests turned green; the notebook check is unchanged.

**mb0 compare against `mb0_reference_cluster.pt`** (`cs8_5_mb0_job.sh`, E exported, `--cpus-per-task=8`; the script compares with 1 thread):

| Node | Job | Result | Exit |
|---|---|---|---|
| g2n01 (i9-10900K) | 16737 | `compared 139 tensors; differing: none` | 0 |
| g3an01 (i9-10900X) | 16738 | `compared 139 tensors; differing: none` | 0 |

**Deviation:** the card asks for `sbatch` on g2n01. The cluster's only 2080 Ti was held by the author's allocation 16721, so the author agreed that g2n01 work runs inside it. As a step, the SBATCH header lines are not applied: the step had the allocation's 6 CPUs (torch reports 6 threads, not 8), and the output went to the evidence file instead of `logs/`. The bit-exact checks force 1 thread themselves, so this does not affect them.

## Decisions for the author

| # | Decision | Options | Recommendation | Reason |
|---|---|---|---|---|
| 1 | The bit-exact checks run with E and 1 thread (R2, implemented) | keep / later capture at a fixed thread count > 1 | keep | It is exact on both CPU types now. With E the 8-thread results were also identical across the two nodes (CS8-1, observed, not part of the rule), but they depend on the thread count, so a fixed count would have to be enforced |
| 2 | A future third CPU type (g2tn01 is down; its CPU was i7-9700 in CS5) | re-run CS8-1 checks there when it is up | re-run `cs8_1_job.sh check` there once | Portability is shown for two CPU models only (node C: none) |
| 3 | `baselines-dgl.txt` now has no Linux line | keep / add a pointer line | keep | pip cannot install DGL on Linux any more; the conda route is the only one. On a fresh Linux env, `pip install -r requirements.txt` therefore leaves the DGL group missing, and the group check reports it |

## Out-of-scope findings (listed, not fixed)

1. `src/baselines/runtime/dependencies.py:43` and `:62-63` (also the Linux message at `:280-283`) still tell the user to reinstall `requirements/baselines-dgl.txt` "with --force-reinstall to get the +cu124 wheel" and name `dgl-2.5.0+cu124`. This is stale after CS8-4 and lives in `src/`, which this card does not change.
2. The CLAUDE.md "Dependencies" section (line ~1016) still describes the Linux pin `dgl==2.5.0+cu124` against DGL's wheel index. It is outside the cluster section the card authorises.
3. `mb0_reference.py` calls `torch.load(path)` without `weights_only`, so torch 2.4.1 prints a `FutureWarning`.
4. `sbatch --test-only` predicted a start at 23:01 for a job that started at once. Its estimates are not reliable on this cluster.
5. Still open from CS7 and assigned to M3b by the card: the graphbolt `GBWarning` / `PYTORCH_CUDA_ALLOC_CONF`, the dirty `vendor/H2GB` clone, and `run_use_dataset.sbatch`. Not touched.

branch=cluster-setup3 sha=<see final message> card=docs/H2S-NN_cluster_setup3_card.md base=8bf2e8f2f3e4024f7c5afc66fc54157d0b3e5e68
