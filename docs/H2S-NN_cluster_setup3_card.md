# Task card — Cluster setup 3 (CS8): pinned CPU kernel path, cluster references for the bit-exact tests, `h2snn-b` as the default, stale items

**Binding:** `docs/working_preferences.md` (all; in particular "Review, approval and branch flow (2026-11-15)"); `docs/H2S-NN_cluster_setup2_card.md` and `docs/H2S-NN_cluster_setup2_report.md` (CS7, approved at `8bf2e8f`); `H2S-NN/AGENTS.md`/`CLAUDE.md`. Spec v1.22 is the design authority and is not touched; this card changes no method code.
**Author decisions (CS7 review, 2026-11-15):** `h2snn-b` becomes the default environment; before choosing between "bit-exact tests on one node type" and "a stated tolerance", measure whether pinning the CPU kernel path makes captures portable across the cluster's CPUs; the freeze file stays a record; the Windows references stay as history (never overwritten).
**Why.** CS7-3: captures made on g2n01 (i9-10900K, torch reports AVX2) differ on g3an01 (i9-10900X, AVX512) in forward/backward outputs (golden 412/1136 tensors, max 1.9e-6; mb0 39/139, max 4.8e-7) while initial weights are identical. The bit-exact tests (`test_task_heads` golden, `test_cooperative` T1 golden, the mb0 compare) exist to catch any change of full-batch outputs during refactors (Part M3a's gate). A tolerance would hide small real changes; a capture that is valid on every node keeps the check exact.
**Why a re-capture on current code is valid:** the Windows golden passed on the laptop at Part MB-fix with the current method code (674-test run), CS7 changed no method code, and on the cluster `init/*` already differs from Windows (platform). So the cluster capture records the same computation; it is not a new baseline of new code.
**Branch:** `cluster-setup3`, created from `v3b` at `8bf2e8f2f3e4024f7c5afc66fc54157d0b3e5e68`, then `git merge --no-ff` of the pinned `design-docs` SHA (given in the prompt; a conflict is a stop). **Report:** `docs/H2S-NN_cluster_setup3_report.md`; evidence `docs/cluster_evidence3/`.

## Rules
- Correctness only; no performance run, no metric, no comparison between methods. Everything in `h2snn-b`; `h2snn` is not modified or removed. All computation in SLURM jobs; every job script (kept in `docs/cluster_evidence3/`) prints `git rev-parse HEAD`, `hostname`, `$SLURM_JOB_ID`, the CPU model (`lscpu | grep 'Model name'`), `torch.backends.cpu.get_cpu_capability()` and `torch.get_num_threads()`.
- **Never mutate a file that a running job reads** (bash reads scripts while it runs; CS7-1a). Mutations are applied only when no job of yours is running, and restored before the next submission.
- Forbidden: loosening tolerances, deleting or skipping tests, silencing exceptions, changing method code (`src/`), editing the spec or cards, overwriting `docs/heads_evidence/v2_golden.pt`, `docs/partMB_evidence/mb0_reference.pt` or the CS7 files in `docs/cluster_evidence2/`.
- Git: one commit per unit, explicit paths; push `cluster-setup3` after each unit; never push or move any other branch; never force-push, rebase, amend or stash. The author's untracked files (`H2S-NN/scripts/slurm/run_use_dataset.sbatch`, `docs/lab-gpu-cluster.html`) and the dirty `vendor/H2GB` clone are not touched.

## Pinned kernel environment `E` (fixed now)
```
ATEN_CPU_CAPABILITY=avx2     # torch's own CPU kernels: AVX2 path on every node
MKL_CBWR=AVX2                # MKL conditional numerical reproducibility: AVX2 code path
ONEDNN_MAX_CPU_ISA=AVX2      # oneDNN (mkldnn) kernels capped at AVX2
```
Set before Python starts (exported in the job script). Expected effect: `torch.backends.cpu.get_cpu_capability()` reports `AVX2` on g3an01 too.

## Units

### CS8-0 — Preconditions and branch
`git fetch`; check `origin/v3b` = `8bf2e8f…` and `origin/design-docs` = the pinned SHA; create the branch and merge as above. Record `sinfo` (node, GPU, state) and, for every node that is up, its CPU model and flags `avx2`/`avx512f` (one tiny CPU job per node, or `scontrol show node` if it lists the CPU; state the source).

### CS8-1 — Does `E` make captures portable? (measurement; pre-registered)
CPU only (`CUDA_VISIBLE_DEVICES=""`), `h2snn-b`, with `E` exported. Scripts as in CS7-3 (`docs/heads_evidence/v2_golden.py recapture <Windows ref> <out>`, `docs/partMB_evidence/mb0_reference.py save <out>`), comparison with `docs/cluster_evidence2/cs7_tensor_diff.py` (every differing name, max |diff|).
1. Capture both on node A = g2n01 with 1 thread → `docs/cluster_evidence3/v2_golden_E.pt`, `mb0_reference_E.pt`.
2. Compare against them on: A again; B = g3an01; and C = one more node with a different CPU model if one is up (g3n01 or g2tn01; else say none). On every node, twice: with 1 thread and with the thread count of the test job (`--cpus-per-task=8`, torch default).
3. Table per file × node × threads: differing tensors, max |diff|, `get_cpu_capability()`.

**Result rule (fixed now):**
- **R1** — 0 differences on every node with both thread counts → `E` is adopted for every bit-exact check (CS8-2 with `E`, threads as the suite).
- **R2** — 0 differences on every node with 1 thread, not with 8 → `E` plus 1 thread inside the bit-exact checks (`torch.set_num_threads(1)` in those tests and in `mb0_reference.py`, restored after; a stated test change, not a tolerance).
- **R3** — any difference across nodes with 1 thread → **stop after CS8-1 and report**; CS8-2 is not done; the author decides between bit-exact tests pinned to one node type and a stated tolerance. CS8-3 and CS8-4 still run.

### CS8-2 — Cluster references and the bit-exact tests (only under R1 or R2)
- New files (the Windows files stay): `docs/heads_evidence/v2_golden_cluster.pt` and `docs/partMB_evidence/mb0_reference_cluster.pt` = the CS8-1 captures (copy, then check sha256 equal). Record in `docs/heads_evidence/golden_capture.txt` and in the mb0 script's docstring: node, CPU, `E`, threads, job, git SHA, sha256; the Windows file is named as history.
- `tests/test_task_heads.py` and `tests/test_cooperative.py` read `v2_golden_cluster.pt` (stated test change: path only). Before comparing, each bit-exact test asserts that the kernel path is pinned: `torch.backends.cpu.get_cpu_capability() == "AVX2"` and `os.environ.get("MKL_CBWR") == "AVX2"`, else fails with the message "bit-exact test needs the pinned kernel environment: run through scripts/slurm/run_tests.sbatch". Under R2 they also run with 1 thread (restored after).
- `docs/partMB_evidence/mb0_reference.py compare` performs the same check (exit 2 with that message if not pinned).
- **Tests:** with `E`, both golden tests pass on g2n01 and on g3an01; without `E` they fail with the message (shown once, not by editing the test). **Mutations:** a test reads the Windows file again → fails on g3an01 or g2n01 with mismatches; the pinned-kernel assertion removed and the test run without `E` on g3an01 → fails on mismatches, not on the message (this shows the assertion is what turns a silent platform difference into a clear error).

### CS8-3 — `h2snn-b` as the default
- `scripts/slurm/run_tests.sbatch`: activates `h2snn-b`; exports `TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b` and `E` (under R3: `E` is still exported, harmless); prints CPU model, `get_cpu_capability()`, threads. `tests/test_slurm_scripts.py` checks `conda activate h2snn-b`, the three `E` lines and `TORCH_EXTENSIONS_DIR` (stated test change). **Mutation:** one `E` line removed → the script test fails.
- `scripts/setup/check_env.py`: default lock `requirements/linux-lock-b.txt`; `tests/test_check_env.py` updated for the default (stated). `requirements/linux-lock.txt` stays, with a first-line comment "superseded by linux-lock-b.txt (2026-11-15); env h2snn (torch 2.6.0) only".
- `H2S-NN/CLAUDE.md` and `H2S-NN/AGENTS.md` (this card authorises the edit; identical text in both): the cluster section names `h2snn-b`, `linux-lock-b.txt`, `scripts/setup/install_env_b.sh`, `E` and why; the "Hard rules" line "Do not commit or push unless asked" becomes "Commit and push only as the task card and `docs/working_preferences.md` say (push rules 2026-11-14, approval by SHA 2026-11-15)"; add the report-line rule (`branch= sha= card= base=`; nothing pushed after it) and the "never mutate a file a running job reads" rule.

### CS8-4 — Stale items
- `requirements/baselines-dgl.txt`: remove the dead `-f https://data.dgl.ai/...` line and the 2.5.0 pin; Linux: a comment that DGL 2.4.0+cu124 comes from the conda package via `install_env_b.sh` (pip must not install it); Windows line unchanged. If a test reads this file, update it (stated).
- Nothing else (the `GBWarning`/`PYTORCH_CUDA_ALLOC_CONF`, the H2GB patch and `run_use_dataset.sbatch` belong to M3b).

### CS8-5 — Full suite on two node types, report
`sbatch scripts/slurm/run_tests.sbatch` (the new default) once on g2n01 and once on g3an01 (`--nodelist`). Expected under R1/R2: on both nodes the only failure is the notebook check; under R3: also the two golden tests (Windows/cluster mismatch, unchanged). Then `mb0_reference.py compare` against `mb0_reference_cluster.pt` on both nodes (under R1/R2: `differing: none`, exit 0). Report: per unit done / VERIFIED-RUN (jobs, nodes, CPUs) / deviations; the CS8-1 table and which rule applies; both suite results; "Decisions for the author"; out-of-scope findings; the report line.

## Stop conditions
CS8-0 hash mismatch or merge conflict; R3 (stop CS8-2 only); a bit-exact test or the mb0 compare cannot be made to read the pinned environment without changing `src/` (report file:line); the card is wrong or ambiguous. **Not stop conditions:** a node being down (use another of a different CPU or report "none"); differences in CS8-1 (that is the measurement).

## Time (estimate)
CS8-0/1: 1 h (queue); CS8-2: 45 min; CS8-3/4: 45 min; CS8-5: 2 × 20 min of compute plus queue. About 3–4 h; longer if g3an01 is busy.
