# Task card — Option B check on the laptop: does torch 2.4.1 work with every required library of H2S-NN?

**Why.** The official DGL wheels support PyTorch up to 2.4 (Linux CUDA 12.4 from 2.4; https://www.dgl.ai/pages/start.html); the Windows wheel on PyPI stops at DGL 2.2.1 (cp312 win_amd64). The pin `dgl==2.5.0+cu124` for torch 2.6 in `requirements/linux-lock.txt` was an unverified design-session pin and is wrong. Option B (author, 2026-11-14): the cluster environment moves to **torch 2.4.1+cu124 with DGL 2.4.0+cu124**. This card tests, in a **new** laptop environment, whether torch 2.4.1 is compatible with all other required libraries and with our test suite, before the cluster is changed.
**What this card cannot test:** DGL 2.4.0 on Linux + CUDA (the laptop is Windows). It tests everything else and inspects the Linux wheel's metadata; the cluster card (CS7-1, rewritten after this report) tests the Linux DGL.
**Binding:** `docs/working_preferences.md`; `H2S-NN/AGENTS.md`/`CLAUDE.md` (Windows section); `docs/H2S-NN_cluster_setup_report.md` (CS2 version table); spec v1.22 untouched.
**Branch:** `env-b-check` from `part-mb` at the commit given in the prompt. **Report:** `docs/H2S-NN_envB_laptop_report.md`; evidence `docs/envB_evidence/`.

## Rules
- Correctness only: no performance run, no metric as a result, no comparison between methods.
- **The existing environment `h2snn` is never modified** (no install, no uninstall, no upgrade). All installs go into the new environment `h2snn-b`. Existing files `requirements/linux-lock.txt` and `check_env.py` are not edited; the candidate lock is a new file.
- Forbidden: loosening tolerances, deleting or skipping tests, changing method code, editing the spec or cards, committing wheels, `data/`, vendor clones, `Claude outputs/`, `H2S-NN/paper/`. Stage by explicit path only (never `git add -A`).
- Install with `pip install --no-cache-dir`; record every command and its output in `docs/envB_evidence/`.
- **Stop:** creating the environment needs administrator rights; torch 2.4.1 has no cp312 wheel for this machine (report the index answer). **Not stop conditions:** a library that does not install or import (that is the finding: report the exact error), a test failing in `h2snn-b` (classify it).

## Pre-registered expectations (fixed before the run)
1. All pins of the CS2 table except torch, the three PyG extensions and DGL install unchanged next to torch 2.4.1 and `pip check` is clean. Consequence if not: the conflicting pair is named in the report; the author decides which pin moves.
2. `torch_scatter==2.1.2`, `torch_sparse==0.6.18`, `torch_cluster==1.6.3` exist for torch 2.4.0 on `https://data.pyg.org/whl/torch-2.4.0+cu124.html` (the index name says 2.4.0; the wheels are used with 2.4.1). Consequence if a wheel is missing on Windows: record it; do not compile.
3. DGL 2.4.0 needs `torchdata` (PyPI metadata of the latest DGL on PyPI lists `torchdata>=0.5.0`), which is **not** in our lock, and the community reports a NumPy-2 incompatibility for DGL wheels (our lock pins numpy 2.4.6). Both are tested in E1/E3; the result decides whether the lock gets `torchdata==<x>` and/or a `numpy<2` pin (a change of the numpy pin affects every other library and is a decision for the author).
4. With torch 2.4.1 the full suite gives the same results as with torch 2.6.0 in `h2snn` except tests that are bit-exact against captures made with 2.6.0.

## Units

### E0 — Preconditions and baseline
`git fetch`; check the base hash; create `env-b-check`. Record in `docs/envB_evidence/e0_baseline.txt`: `nvidia-smi` (GPU or none), `conda info`, Python and `pip freeze` of the existing `h2snn` (read only), and which DGL/torch_householder it has.

### E1 — Inspect the DGL 2.4.0 Linux wheel without installing it
(a) Fetch the index page `https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html` (read-only; from the laptop it is reachable) and record the DGL 2.4.0 file names and their Python/platform tags (is cp312 / `py3-none` listed?). (b) If the author's wheel path is given in the prompt (`<<AUTHOR>>`), do not install it: list its contents (`python -m zipfile -l`), print its `METADATA` `Requires-Dist` lines, and list the graphbolt `.so` file names (they carry a torch version). Record the answers to: Python tag; `numpy` constraint; `torchdata` constraint; torch version of the graphbolt library.

### E2 — The new environment
`conda create -n h2snn-b python=3.12`; then, in this order: `torch==2.4.1` (index `https://download.pytorch.org/whl/cu124` if E0 shows an NVIDIA GPU, else the CPU index; record which); the three PyG extensions at the pins above from `https://data.pyg.org/whl/torch-2.4.0+cu124.html`; `torch_geometric==2.8.0`; every other pin of `requirements/linux-lock.txt` except `torch`, the PyG extensions, `torch_householder` and `dgl`; then what exists for DGL on Windows (`dgl==2.2.1` from PyPI, as the laptop has) and `torchdata` (the newest 0.8.x). Run `pip install --dry-run` first and keep the resolver output. Write the candidate lock `H2S-NN/requirements/linux-lock-b.txt`: same file with `torch==2.4.1`, the PyG index `torch-2.4.0+cu124`, DGL `dgl==2.4.0+cu124` and the DGL index `torch-2.4/cu124`, plus any pin E2/E3 show to be necessary (each with the reason in a comment). **Tests:** `pip check` output; the installed versions of every lock line (a small script, `docs/envB_evidence/e2_versions.txt`) with the differences from `linux-lock-b.txt` (expected: only DGL).

### E3 — Import matrix
In `h2snn-b`, in a fresh interpreter per library: import torch, torch_geometric, torch_scatter, torch_sparse, torch_cluster, dgl, torchdata, numpy, scipy, pandas, sklearn, ogb, pytorch_lightning, wandb, torchmetrics, umap, performer_pytorch; print each version; `torch.cuda.is_available()`; a CUDA tensor op if a GPU exists. Run with warnings visible (`python -W default`) and record every warning, in particular "A module that was compiled using NumPy 1.x cannot be run in NumPy 2" and any DGL/graphbolt message. One tiny functional call per extension (scatter, sparse matmul, a DGL graph with message passing).

### E4 — Full suite in `h2snn-b`
Run the whole unittest suite in `h2snn-b` (same command as the Part MB-fix suite; GPU if present). Reference: Part MB-fix (674 tests; failures 2, errors 2, skips 7: `docs/H2S-NN_partMBfix_report.md`). Run the same suite in `h2snn` at the same commit and write the **per-test** difference (a test that is ok in one and not in the other). Classify each difference: library missing, bit-exact capture, other (first error line). Nothing is fixed.

### E5 — Does torch 2.4.1 vs 2.6.0 change the initial weights? (measurement)
The cluster's golden test fails at 1 ulp already at initialisation. Same machine, same OS, same compiler family: in `h2snn-b` run `tests.test_task_heads` and `tests.test_cooperative` (the golden tests, captured on this laptop with torch 2.6.0) and `docs/partMB_evidence/mb0_reference.py compare` (read-only compare). **Result rule (fixed before the run):** golden tests pass and the compare says `differing: none` → torch 2.4.1 and 2.6.0 give identical initial weights and outputs on this machine, so the cluster's 1-ulp difference is platform (compiler, MKL, CPU), not torch version; any mismatch → report its tensors and `max|diff|`; the torch version matters too and every capture must be redone on the cluster's final environment.

### E6 — Report
Per unit: done, VERIFIED-RUN commands with outputs, deviations. Then: a compatibility table (library, locked version, installed in `h2snn-b`, import ok, warnings), the E1 answers, the E4 per-test difference, the E5 verdict, "Decisions for the author" (options, recommendation, reason), out-of-scope findings. Commit the candidate lock, scripts and evidence (one commit per unit); push `env-b-check` after the report.
