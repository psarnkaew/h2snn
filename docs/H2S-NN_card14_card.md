# Card 14 — Whole H2S-NN data pipeline on the GPU (WISP sampling, cells, scopes, GC draws)

Written 2026-11-22 by the model design session. Approved by the author the same day ("run everything on the GPU; test on the lab cluster"). This card changes `src/`: the freeze is lifted for the files named below, and the change goes into spec 1.30. No method change. Report: `docs/H2S-NN_card14_report.md`.

## Why
The runs are CPU-bound. On the lab, PDNS shows 4 cores at 100 % with GPU ≈ 0 % and ≈ 14 s per step (2.8 s on the H100), mag-year ≈ 300 s per epoch (90–125 s on the H100), and NCI1 ≈ 17 min per epoch (47–60 min on the H100 `gpu_1`). The time goes to WISP sampling, cell merging and scope building, which run on CPU tensors with Python loops. Examples:
- `src/data/scopes.py:182`, one loop iteration per target with `.tolist()`;
- `src/data/wisp_tasks.py:103`, one loop iteration per graph;
- the `.cpu()` calls in `src/data/wisp_sampler.py`.

Nothing in the method needs the CPU. At the current speed, 100 trials do not fit for H2GB (≈ 15 h per trial on mag-year) or GC (NCI1 > 40 h per trial). The C5 cost-vs-accuracy figure would also show this overhead as the method's cost.

## Goal
Every tensor operation of an H2S-NN training or evaluation step runs on the model's device. This covers:
- the WISP draw (walks, proposal, importance ratios);
- canonical cells and merging;
- the face groups;
- the scope batch (members, local indices, incidence);
- the LP endpoint scopes;
- the per-graph GC draws.

No Python loop over targets, graphs, scopes or walks is allowed in a step. Loops over a small, fixed set are allowed: hop index ≤ `ℓ_max`, path length, relation types. The prepared graph (edge index, CSR, action graph, incidence index) is built once per run and kept on the device. The only host syncs allowed in a step are the ones PCG already has (convergence check) plus logging.

## Preconditions
- Worktree `/mnt/data/peeradon/h2snn-card14`, branch `card14` from `origin/v3b` `66e9349b79021784a08b37c5f8be0798c0260a04`; `git merge` the design-docs SHA given in the prompt. Card 12 runs in parallel on `anchored_solve.py`. Do not touch that file; the two branches merge into `v3b` separately.
- Read: this card; `src/data/{path_sampler,wisp_sampler,wisp_tasks,scopes}.py`; `src/trainers/{minibatch_trainer,minibatch_lp,link_prediction_trainer}.py`; the GC training loop; the model's `_encode_anchored` and how it receives cells and scopes; the tests of these modules; spec §3.2 (WISP) and §6 "WISP settings"; card 11 report §B (runtime notes); `docs/lab-gpu-cluster.html`.
- Runs: `sbatch` only, one GPU, `--mem` = measured peak + 30 %. Use the lab 3090 / TITAN RTX; use an A6000 only if nothing else fits, and say so in the report. Every run states GPU, job id, wall time, peak GPU and host memory.

## Amendment 2 (author, 2026-11-22) — binding; supersedes Amendment 1 and everything below it
Do **both**, in one pass: (1) move the whole H2S-NN data pipeline to the device (remove `.cpu()`, CPU generators, host round-trips); (2) vectorise the Python loops over targets, graphs, scopes and walks (first of all the per-target loop of `build_scope_batch`, `src/data/scopes.py:182`; the per-graph loop of `src/data/wisp_tasks.py:103`). Results need not equal the old code (random stream and summation order change; the lines rerun). No method change; the seed schedule is kept.

**Tests (only these, keep it short):** (a) each rewritten function gives the old function's output on one fixed input (canonical sort), old functions kept under `tests/reference/`; (b) one-epoch smoke on ACM, PDNS, mag-year and NCI1 (finite loss, parameters move); (c) the full suite by `sbatch`. Nothing else.

**Speed:** one table, old vs new on the lab (3090 / TITAN RTX), 20 steps + one evaluation pass, mag-year, PDNS, NCI1, ACM: s/step, GPU utilisation, peak GPU memory. Target ≥ 5× on mag-year, PDNS, NCI1; if missed, report the profile of what is left (do not stop work for it).

**Rules:** files as listed in Amendment 1; not `anchored_solve.py`; nothing under `configs/`; copy `vendor/`; `sbatch` only; one commit (plus one if needed), push `card14`, never merge into `v3b`. Report `docs/H2S-NN_card14_report.md`: what moved and what was vectorised, the speed table, test results; last line `branch=card14 sha=<sha> card=docs/H2S-NN_card14_card.md base=66e9349b79021784a08b37c5f8be0798c0260a04`. Estimate ≈ 1–2 days.

## Amendment 1 (author, 2026-11-22) — supersedes the Goal's "no Python loop" requirement and the units, rules, stop conditions and report below
The author does not require results equal to the old code: the random stream (CPU vs CUDA generator) and GPU summation order may change every number, and the H2S-NN lines rerun anyway. Only the speed and correctness of the move matter. **Do not start until the author says so.**

### S1 — Move to the device, as is (≈ 1–2 h)
Remove the host copies (`.cpu()`, `torch.Generator()` on the CPU, `.tolist()` / `int(...)` where they only exist to work on the host) so that the WISP draw, cells, face groups, scopes, LP endpoint scopes and GC draws run on `graph.device`, without restructuring the code. Measure on the lab (3090 / TITAN RTX), old vs new, 20 training steps plus one evaluation pass, on mag-year `dandy-sweep-1`, PDNS `balmy-sweep-7`, NCI1 `astral-sweep-1` (lr 3e-4) and ACM full batch: seconds per step, GPU utilisation, peak GPU memory. If every one of mag-year, PDNS and NCI1 is **≥ 5× faster per step** and ACM is not slower, go to the tests and stop there.

### S2 — Only if S1 misses the gate: profile, then vectorise the hot loops
Profile (`torch.profiler`, CPU and CUDA) the configurations that miss the gate; rewrite only the loops that take the time (expected: the per-target loop of `build_scope_batch`, `src/data/scopes.py:182`; per-step overhead on GC) as batched tensor operations. Measure again as in S1. If the gate is still missed, report the profile and stop.

### Tests (only these)
1. **Same function on the same input** — for every function rewritten in S2 (not for S1's device moves): on a fixed input (a fixed `WispDraw` and target list on ACM, one mini-batch fixture, one GC graph), the new output equals the old function's output after a canonical sort. This catches index errors that would not crash but would silently train on wrong cells. The old functions are kept only as test references under `tests/reference/`.
2. **Smoke** — one epoch each of ACM, mag-year (or ogbn-mag), PDNS and NCI1 on the GPU: finite loss, parameters move, no host sync errors.
3. The full suite (`run_tests.sbatch`; W&B tests without `WANDB_MODE=disabled`, as card 12 found).
No distribution test, no determinism test, no mutation test, no comparison of metrics with the old code.

### Rules
- No method change; the seed schedule is unchanged (only the random stream differs).
- Files allowed under `src/`: `src/data/{path_sampler,wisp_sampler,wisp_tasks,scopes}.py`, `src/trainers/{minibatch_trainer,minibatch_lp,link_prediction_trainer}.py`, the GC trainer, and the model file only where it moves tensors between devices. Do not touch `src/models/anchored_solve.py` (card 12). Nothing under `configs/`. Another file: stop and report.
- `sbatch` only (no tests on the login node), one GPU, `--mem` = measured peak + 30 %; copy (not symlink) the git-ignored `vendor/` into the worktree (card 12 found that symlinks break `vendor_paths`).
- No tuning, no performance runs. Git: one commit for S1 (+ tests), one for S2 if needed; stage by explicit path; push `card14`; never stash, reset, rebase, commit others' changes or merge into `v3b`.

### Stop conditions
S1 and S2 both miss the gate; a file outside the list must change; a run needs more than 6 h.

### Report (`docs/H2S-NN_card14_report.md`)
What moved; the speed table (old vs S1, and S2 if run) against the gate; the profile if S2 ran; test results; anything unexpected. Last line `branch=card14 sha=<sha> card=docs/H2S-NN_card14_card.md base=66e9349b79021784a08b37c5f8be0798c0260a04`.

### Estimate
S1 + tests ≈ ½ day; with S2 ≈ 1–3 days.

## Original units (superseded by Amendment 1; kept for the record)

## Units (one commit each; push `card14` after each)

### G0 — Profile (before any change; report, no code change under `src/`)
On the lab GPU, profile 20 training steps plus one evaluation pass with `torch.profiler` (CPU and CUDA), plus wall time per phase. Phases:
- WISP draw;
- cell merge;
- face groups;
- scope build;
- `encode` / PCG forward;
- backward;
- optimizer;
- host↔device copies.

Configurations, all from the card 11 inputs except ACM:
- ogbn-mag mini-batch;
- mag-year `dandy-sweep-1`;
- PDNS `balmy-sweep-7`;
- NCI1 `astral-sweep-1` at lr 3e-4;
- ACM full batch.

Output: a table of the share of wall time per phase, with the top 10 functions by self CPU time. The table decides the order of G1. If more than half the step time is in PCG or backward (GPU work), **stop and report**: that would contradict card 11.

### G1 — Move the pipeline to the device (src change)
- All of the above runs on `graph.device`. Replace the per-target, per-graph and per-scope loops with batched tensor operations: segment / `index_select` / `scatter` / sort-based grouping, and padded or CSR walk tensors.
- Random draws use a `torch.Generator` on the device, seeded by the existing seed functions (`training_draw_seed`, `graph_draw_seed`, the fixed evaluation draw seed). The seed schedule is unchanged. The random stream changes; this is accepted.
- GC: the training batch size stays 1 graph per optimizer step (HetSheaf protocol). The WISP draws of all graphs of an epoch may be made in one batched device call at the start of the epoch, with each graph's draw identical to drawing it alone from its own seed. Test 4 checks this.
- The trainers keep their interfaces and config keys. No new config key, and no CPU fallback path in `src/`. The old CPU functions are kept only as test references under `tests/reference/`.
- Files allowed: `src/data/{path_sampler,wisp_sampler,wisp_tasks,scopes}.py`, `src/trainers/{minibatch_trainer,minibatch_lp,link_prediction_trainer}.py`, the GC trainer, and the model file only where it moves tensors between devices. Any other file under `src/`: **stop and report**.

### G2 — Tests (CPU and lab GPU)
1. **Deterministic parts are exactly equal.** For a fixed `WispDraw` given as input, the new cell merge, face groups and scope batch must equal the old CPU reference exactly: same tensors after a canonical sort, on ACM, IMDB, one mini-batch H2GB fixture and one GC graph.
2. **The draw has the same distribution.** Draw from the old and new samplers with many seeds on ACM and on a small mini-batch fixture. Compare:
   - path-type (signature) frequencies, using a chi-square test at α = 0.001;
   - the walk-length distribution;
   - the cell counts per anchor;
   - the mean importance ratio per signature.

   These tolerances are fixed now; do not loosen them. Also a known-answer check: on a 6-node graph, every signature's empirical frequency matches the proposal's exact probability within 3 standard errors.
3. **Determinism.** Two runs with the same seed on the same GPU give an identical draw and identical scope batches.
4. **GC batched draw.** Each graph's draw from the epoch-level batched call is identical to drawing that graph alone with its seed.
5. **Training smoke test.** One epoch each of ACM full batch, ogbn-mag, PDNS and NCI1, compared with the old code:
   - loss finite;
   - parameters move;
   - validation metric within the seed-to-seed range of the old code. Measure this range from 3 seeds of the old code at 1 epoch; it is not a performance claim.
6. **Mutations** (each must fail a test; revert after):
   - drop one walk direction;
   - off-by-one in the scope local index;
   - use the same seed for every graph in GC;
   - skip the importance ratio.
- Run the full suite (`run_tests.sbatch`).

### G3 — Speed and utilisation on the lab (gate)
Same five configurations as G0, old vs new code, same GPU type, 20 steps + one evaluation pass. Report:
- seconds per step and per epoch;
- GPU utilisation (`nvidia-smi dmon` or the profiler);
- host CPU %;
- peak GPU memory.

**Gate:** at least **5× faster per step on mag-year, PDNS and NCI1**, and not slower on ACM. Peak GPU memory must fit the H100 rule of spec 1.26 at the largest search point (state the measurement). If the gate is missed, report the new G0-style profile of the remaining time and stop. Do not optimise further without the design session.

## Rules
- No method change. Same draw distribution, same cells, same scopes, same seed schedule; only the random stream differs.
- `git diff --stat 66e9349 -- H2S-NN/src` lists only the files allowed in G1. No change under `configs/`.
- Do not tune; no performance runs; nothing here is a result.
- Git: one commit per unit, stage by explicit path, push `card14`. Never stash, reset, rebase, commit others' changes or merge into `v3b`; the fast-forward needs the author's approval after the design and ops reviews.

## Stop conditions
- G0 shows GPU work dominating.
- A test of G2 fails and the fix would change the method or the draw distribution.
- A file outside the G1 list must change.
- The G3 gate is missed.
- A run needs more than 6 h.

## Report (`docs/H2S-NN_card14_report.md`)
- G0 profile tables.
- The G1 design: what moved, and which loops were replaced by what.
- G2 test and mutation results.
- G3 speed table against the gate.
- Anything unexpected.
- The evidence line: `branch=card14 sha=<sha> card=docs/H2S-NN_card14_card.md base=66e9349b79021784a08b37c5f8be0798c0260a04`.

## Consequences (for the author; not the agent's task)
- The random stream changes, so the finished H2S-NN tuning on HGB ACM, DBLP and IMDB reruns (≈ 2–3 h each on the H100). The baselines are not affected.
- Card 12 (PCG) and card 14 merge into `v3b` together with spec 1.30, then the H2S-NN lines restart.

## Estimate
- G0: ½ day.
- G1: 2–3 days.
- G2: 1 day.
- G3: ½ day.
- Total ≈ 4–5 days.

It takes longer if G0 shows a different bottleneck or the scope builder needs a new grouping design. To save time, do G1 in the order the G0 profile gives and run G3 on mag-year first.
