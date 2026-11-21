# Card 12 — PCG zero right-hand-side rule (src fix) and the degree check on every LP task

Written 2026-11-21 by the model design session after the card 11 report (`card11` `37fc77a`); approved by the author the same day. Unit P changes `src/` (the freeze is lifted for this one rule only; spec 1.30). Unit L is evidence only. Report: `docs/H2S-NN_card12_report.md`.

## Why
- **P (card 11 B).** On PDNS every run fails at epoch 1, step 172: the backward right-hand side of one scope is ≈ 0 (‖b_s‖ = 3.4e-17, median 1.7e-3), the per-scope stopping test `‖r_s‖ ≤ tol·‖b_s‖` asks for an absolute residual below fp32 round-off, and PCG reaches `T_max` although the scope's system is well conditioned (cond ≈ 6). mag-year (failed at epoch 49 on the H100) is presumed to fail the same way; this card measures it.
- **L (card 11 A).** acm_pa is ill-posed under our LP protocol: the tail type (author) has no features (F17) and no relation other than the target, so holding out a fact leaves 47 % of validation-positive authors isolated, the degree-only score inverts (train 0.684, validation 0.448) and GCN fails too. Other LP tasks may share this (imdb_ma: actors have no features under F17 and only the movie relation). The author decides which tasks stay after seeing L.

## Preconditions
- Worktree `/mnt/data/peeradon/h2snn-card12`, branch `card12` from `origin/v3b` `2d822a4eb48867de4476b97159430a64d8468808`; `git merge` the design-docs SHA given in the prompt (this card); copy the card 11 scripts only: `git checkout 37fc77a60c1caff38b54bb0a4a2a88c20ab60df4 -- H2S-NN/scripts/diagnostics/card11` (not its evidence). Link the git-ignored `vendor/` clones as in card 11.
- Read: this card; `src/models/anchored_solve.py` (`pcg`, `solve_anchored`, `AnchoredSolve`, the stats object); `tests/test_anchored_solve.py`; the card 11 report §B and §A1; spec §6 "Solver settings".
- Runs: `sbatch` only, one GPU, `--mem` = measured peak + 30 %; prefer 3090 / TITAN RTX, no A6000 unless nothing else fits (state it). Every run states GPU, job id, wall time, peak memory.

## Unit P — zero right-hand-side rule in `pcg` (src change)

**Rule (spec 1.30 text, fixed now).** In `pcg`, after `b_norm` is computed: a scope `s` with `‖b_s‖ ≤ ρ · max_s ‖b_s‖`, `ρ = 1e-10` (a constant, not a config key, never tuned), is **solved exactly by `y_s = 0`**: its rows of `y` are set to 0, it is marked converged with 0 iterations, and its reported relative residual is 0. If `max_s ‖b_s‖ = 0`, every scope gets `y = 0`. Reason: `M` is SPD and block-diagonal over scopes (Thm 5), so `b_s = 0 ⇒ y_s = 0` exactly; for `‖b_s‖ ≤ 1e-10 · max` the error `‖M_s⁻¹ b_s‖ ≤ ‖b_s‖/λ_min` is below fp32 resolution of the call. Forward and backward use the same `pcg`, so both are covered. A full batch is one scope (ratio 1), so full-batch runs never trigger the rule.

**Implementation constraints.**
- When no scope triggers, the code path must be **bit-identical** to `2d822a4` (e.g. compute the mask; apply `torch.where` / row masking only if `mask.any()`).
- Count triggered scopes per call, and add the count to the solver stats as `pcg_zero_rhs_scopes`. This is a log field only and has no effect on training.
- Nothing else changes: tolerance schedule, `T_max`, Jacobi, warm start, the `strict` raise.

**Tests (CPU, `tests/test_anchored_solve.py`).**
1. Two scopes, scope 1 with `b = 0` and a non-zero warm start `y0` → `y` rows of scope 1 = 0, converged, iterations 0; scope 0's `y` is bit-identical to the old `pcg` on scope 0 alone.
2. Scope 1 with `‖b‖ = 1e-17`, others ~1e-3 → no `PCGConvergenceError` with `strict`; old code raises (reproduce with the fixture).
3. All `b = 0` → `y = 0`, no division by zero, no NaN.
4. No scope below `ρ` → output bit-identical to the old `pcg` (compare against a copy of the old function in the test file).
5. Backward: `AnchoredSolve` gradient on a fixture whose loss gives one scope a zero `dL/dY` → finite gradients, no raise.
- **Mutations** (each must make a test fail; revert after): remove the rule; set `ρ = 0`; set the triggered rows to `y0` instead of 0.
- The full suite passes (`run_tests.sbatch`, as card 8).

**Run checks (`gates`).**
- **P1 identity.** ACM full batch, 1 epoch, CPU pinned as card 11 job 18070. Old (`2d822a4`) vs new must have bit-identical losses and all parameters equal. `pcg_zero_rhs_scopes` = 0.
- **P2 PDNS** `balmy-sweep-7` (card 11 inputs), with the card 11 probe. Run until 3 h or the end of epoch 3. Gate: passes epoch 1 step 172 with no `PCGConvergenceError`. Report the following:
  - the number of triggered scopes per epoch and their ‖b_s‖;
  - the iteration max/p95 per epoch;
  - whether training loss and validation Macro-F1 to the failure point equal card 11 B2a, which they should up to the first trigger.
- **P3 mag-year** `dandy-sweep-1` with the probe, run to the end of epoch 55. On the lab this takes ≈ 300 s/epoch, about 4.6 h. If the queue makes it longer than 6 h, stop and report the trend.
  - Gate: no `PCGConvergenceError` up to epoch 55.
  - Report the first epoch with a triggered scope; this confirms or refutes the card 11 inference.
  - If mag-year still fails with no zero-RHS scope, **stop and report**. That is a different mechanism.

## Unit L — degree check on every LP task (evidence only)
Run card 11's `a1_data_checks.py` (no change except the task list; add tasks by argument) on **acm_pa, dblp_ap** (references), **imdb_ma, lastfm, pubmed, freebase_bb, h-pokec_uu, ogbn-mag_pp** (CPU jobs; the two large tasks may need more memory, measure). Per task:
- the head and tail types; for each type, whether it has input features (yes/no, or a learned embedding);
- the number of relations each endpoint type has besides the target;
- the tail (and head) message-graph degree of train / validation / test positives and of each split's negatives: share with degree 0, median, mean;
- the degree-only AUROC on train / validation / test random, with all relations and with the target pair only;
- leakage counts as in card 11 A1.

**Flag rule (fixed now; the report applies it, the author decides).** A task is **flagged** when both of the following hold:
- (i) the share of degree-0 tails (or heads) among validation positives is ≥ 2× that among validation negatives and ≥ 0.2;
- (ii) the degree-only AUROC (all relations) is > 0.5 on train and < 0.5 on validation.

The report lists the flagged tasks. Do not change any task, split or feature.

## Rules
- `src/` changes only in `src/models/anchored_solve.py` (and its stats field) and `tests/test_anchored_solve.py`. Nothing under `configs/` changes. `git diff --stat 2d822a4 -- H2S-NN/src` lists only that file.
- Diagnostic scripts go in `scripts/diagnostics/card12/`; evidence in `docs/card12_evidence/`. No checkpoints are committed.
- Nothing here is a result. Do not tune.
- Git: make one commit per unit (P code+tests, P runs, L), stage by explicit path, and push `card12` after each. Never stash, reset, rebase or commit others' changes. **Never merge into `v3b`**: the fast-forward needs the author's approval after the design and ops reviews.

## Stop conditions
- P1 not bit-identical.
- P2 still fails, or fails at a different place.
- P3 fails with no triggered scope.
- Any other `src/` change seems needed.
- A run needs more than 6 h (report the trend).

## Report (`docs/H2S-NN_card12_report.md`)
- **P:** the diff; tests and mutation results; P1–P3 tables.
- **L:** the per-task table and the flag column.
- **Other:** anything unexpected.

End with the evidence line `branch=card12 sha=<sha> card=docs/H2S-NN_card12_card.md base=2d822a4eb48867de4476b97159430a64d8468808`.

## Estimate
P code and tests ≈ 2 h; P1 15 min; P2 ≈ 2–3 h wall; P3 ≈ 5 h wall; L ≈ 1–2 h. P2, P3 and L run in parallel, so the total is ≈ 1 day of wall time. It takes longer if the lab queue is busy.

## After this card (model session, not the agent)
- Spec 1.30: add the rule to §6 "Solver settings", plus the author's decision on the flagged LP tasks.
- GC: no model change. The τ guard stays, and ops confirms that a trial stopped by the guard returns its best validation score to the Bayesian search.
