# Card 12 report — PCG zero right-hand-side rule (P) and the degree check on every LP task (L)

Card: `docs/H2S-NN_card12_card.md` (design-docs `5109a44`, amendments 1–2 in `47fe21f`). Branch `card12` = base
`2d822a4` + merge `5109a44` + card 11 scripts from `37fc77a` (scripts only) + merge `47fe21f`.
Commits: P code + tests `cbe87c5`, L `95c34f9`, P runs `3742fab`. `git diff --stat 2d822a4 -- H2S-NN/src` lists
only `H2S-NN/src/models/anchored_solve.py` (43 insertions, 3 deletions); nothing under `configs/` changed.
Scripts: `H2S-NN/scripts/diagnostics/card12/`. Evidence: `docs/card12_evidence/{P,L}/`. Run outputs and
checkpoints stay outside the repository (`/mnt/data/peeradon/card12_runs`). Nothing here is a result.

Stop conditions: none hit. P1 is bit-identical; P2 passes epoch 1 step 172 and fails nowhere else; P3 was dropped
(amendment 2); no further `src/` change was needed; no run came near 6 h. Before amendment 1 the agent stopped once,
before committing, on the golden-test conflict (below).

## P — zero right-hand-side rule

### Diff (`git diff 2d822a4 -- H2S-NN/src/models/anchored_solve.py`)

```diff
@@ class SolveStats:
     per_scope_forward: list[int] = field(default_factory=list)
+    # scopes solved by the zero right-hand-side rule (spec 1.30); a log field, no effect on training
+    forward_zero_rhs_scopes: int = 0
+    backward_zero_rhs_scopes: int = 0
@@ def as_dict
             "pcg_converged": float(self.forward_converged and self.backward_converged),
+            "pcg_zero_rhs_scopes": float(self.forward_zero_rhs_scopes
+                                         + self.backward_zero_rhs_scopes),
@@ after segment_dot
+# Zero right-hand-side rule (spec 1.30, card 12): a scope with ||b_s|| <= ZERO_RHS_RATIO * max_s ||b_s||
+# is solved exactly by y_s = 0. A constant, not a config key, never tuned.
+ZERO_RHS_RATIO = 1e-10
+
+def zero_rhs_mask(b_norm: torch.Tensor) -> torch.Tensor:
+    """`[S]` bool: the scopes the zero right-hand-side rule solves by `y_s = 0`
+    (`||b_s|| <= ZERO_RHS_RATIO * max_s ||b_s||`; every scope when `max_s ||b_s|| = 0`)."""
+    return b_norm <= ZERO_RHS_RATIO * b_norm.max()
+
+def count_zero_rhs_scopes(b, scope_rows, num_scopes) -> int:
+    """Number of scopes of `b` that `pcg` solves by the zero right-hand-side rule (stats only)."""
+    return int(zero_rhs_mask(segment_dot(b, b, scope_rows, num_scopes).sqrt()).sum())
@@ def pcg (docstring: the rule, M SPD and block-diagonal over scopes, Thm 5)
     y = y0.clone()
+    b_norm = segment_dot(b, b, scope_rows, num_scopes).sqrt()
+    zero_rhs = zero_rhs_mask(b_norm)
+    any_zero_rhs = bool(zero_rhs.any())
+    if any_zero_rhs:
+        y = torch.where(per_row(zero_rhs), torch.zeros_like(y), y)
     r = b - apply_m(y)
     zt = jacobi_inverse.unsqueeze(-1) * r
     p = zt.clone()
     rz = segment_dot(r, zt, scope_rows, num_scopes)
-    b_norm = segment_dot(b, b, scope_rows, num_scopes).sqrt()
     threshold = tolerance * b_norm
     r_norm = segment_dot(r, r, scope_rows, num_scopes).sqrt()
     converged = r_norm <= threshold
+    if any_zero_rhs:
+        converged = converged | zero_rhs
@@ end of pcg
     relative = r_norm / torch.where(b_norm > 0, b_norm, torch.ones_like(b_norm))
+    if any_zero_rhs:
+        relative = torch.where(zero_rhs, torch.zeros_like(relative), relative)
     return y, iterations, converged, relative
@@ AnchoredSolve.forward
         stats.per_scope_forward = iterations.tolist()
+        stats.forward_zero_rhs_scopes = count_zero_rhs_scopes(
+            system.right_hand_side(h0), scope_rows, num_scopes)
@@ AnchoredSolve.backward
             stats.backward_residual = float(residual.max())
+            stats.backward_zero_rhs_scopes = count_zero_rhs_scopes(
+                grad_y, ctx.scope_rows, ctx.num_scopes)
@@ __all__: + ZERO_RHS_RATIO, count_zero_rhs_scopes, zero_rhs_mask
```

(Docstring lines condensed here; the full diff is in commit `cbe87c5`.) Design notes:
- **Bit identity.** `b_norm` is a pure function of `b`, so moving it before the first residual changes no value.
  Every new branch runs only if `mask.any()`. With no triggered scope, the arithmetic is the old one, plus one
  `max`/compare and a host sync.
- **Triggered scope.** Its rows of `y` are set to 0 before the first residual, so `r_s = b_s` (≈ 0). It is
  marked converged, so `alpha = beta = 0` and the rows stay exactly 0 (iterations 0). Its relative residual is
  reported as 0. All scopes trigger when `max_s ‖b_s‖ = 0`, which gives `y = 0` with no division and no NaN.
- **`pcg`'s signature and 4-tuple return are unchanged.** The card 11 probe and other tests wrap or unpack it.
  The count for `pcg_zero_rhs_scopes` is therefore taken in `AnchoredSolve` with the rule's own `zero_rhs_mask`
  on the same right-hand side. On the GPU, `segment_dot` uses `index_add` (atomics), so a scope lying exactly at
  the 1e-10 boundary could in principle be counted differently from `pcg`'s own decision. This affects the log
  only; in P2 the triggered scopes sat between 1e-13 and 1e-18 of the maximum, far from the boundary.

### Tests (CPU, `tests/test_anchored_solve.py`, class `ZeroRightHandSideTests`)

Fixture: a block-diagonal SPD system, one dense block per scope applied block by block, `M = I + τ L` with L's
spectrum in [0.85, 5]. `_pcg_2d822a4` is a verbatim copy of the old `pcg`.

| # | test | checks |
|---|---|---|
| 1 | `test_P1_zero_scope_gets_zero_and_other_scope_is_unchanged` | 2 scopes, `b_1 = 0`, non-zero `y0` (fp32 and fp64): scope 1 rows = 0, converged, 0 iterations, residual 0. Scope 0 is **bit-identical** to the old `pcg` on scope 0 alone with the same per-scope reduction (`scope_rows` of scope 0, an empty second scope). With one scope (`.sum()` in place of `index_add`) the last bits differ, so that variant is checked to `≤ 10·tol·max|y|`, with equal iteration counts |
| 2 | `test_P2_tiny_scope_does_not_raise_where_the_old_pcg_does` | fp32, scope norms (1e-3, **3.4e-17**, 1e-3), denormals flushed. Old `pcg` + strict check raises `PCGConvergenceError`; new does not (also through `solve_anchored`, strict) |
| 3 | `test_P3_all_zero_rhs` | 1 and 3 scopes, `b = 0`: `y = 0`, converged, 0 iterations, residual 0, finite |
| 4 | `test_P4_no_trigger_is_bit_identical_to_the_old_pcg` | fp32 / fp64 × {1 scope, 3 scopes of mixed scale, a scope at 2e-10 of max} × {zero, warm start}: all four outputs `torch.equal` to the old `pcg`; count 0 |
| 5 | `test_P5_backward_with_a_zero_gradient_scope` | `AnchoredSolve` with `dL/dY_1 = 0`: finite gradients for h0, the operator parameter and θ_τ; `h0.grad` rows of scope 1 = 0; `backward_zero_rhs_scopes` = 1, forward 0, `pcg_zero_rhs_scopes` = 1 |
| 5b | `test_P5b_backward_with_a_tiny_gradient_scope` | `‖dL/dY_1‖ = 3.4e-17` (fp32, flushed): with the old `pcg` the backward raises; with the rule it does not, gradients are finite |
| 5c | `test_P5c_forward_counts_triggered_scopes` | forward with `h0` of scope 1 = 0: count 1, 0 iterations, rows 0 |

**Mechanism (amendment 1).** The old `pcg` fails on the fixture **only with `torch.set_flush_denormal(True)`**.
With denormals kept (the CPU default), fp32 PCG is scale-invariant and the old code converges at ‖b_s‖ = 1e-17
(prototype: 1e-18, 3.4e-17 and 1e-16 reach T_max with flushing; 1e-17, 1e-15 and 1e-12 converge). At the PDNS
scale the entries are about 1e-19, so the per-scope dot products (`bᵀb`, `rᵀr`, `pᵀq`, about 1e-38 and below)
underflow when denormals are flushed. ‖b_s‖ then comes out ≈ 0 or exactly 0 and the relative test becomes
unreachable. The rule covers both cases. The fixture keeps flushing on (only inside tests 2 and 5b, restored
afterwards), and the test file says why.

**Mutations** (`p_mutations.py`; each one runs on a temporary copy of `src/` and `tests/`, the repository is not
touched; an unmutated control passes first):

| mutation | caught by |
|---|---|
| remove the rule (`zero_rhs` all False) | P1, P2, P3, P5b |
| ρ = 0 | P2, P5b |
| triggered rows = `y0` instead of 0 | P1, P3 |

**Golden test (amendment 1).** `tests/test_v3b_golden.py` ignores only keys whose last component is
`pcg_zero_rhs_scopes`. Every captured tensor keeps its bit-identity check (0 differing in job 18139, before the
exclusion), and there is no re-capture.

**Suite jobs** (logs in `docs/card12_evidence/P/`):

| job | what | GPU / node | wall | result |
|---|---|---|---|---|
| 18137 | full suite, first try | 3090 / g3n01 | cancelled | 85 errors from symlinked `vendor/` (below); cancelled |
| 18139 | full suite + mutations | 3090 / g3n01 | 51 min 37 s | 961 tests: failures 2, errors 2 = the golden new key + 3 W&B tests (job ran with `WANDB_MODE=disabled`); mutations 3/3 caught |
| 18240 | `tests.test_wandb_logging` without `WANDB_MODE` | TITAN RTX / g2tn01 | 41 s | 7 OK |
| 18241 | `test_v3b_golden`, `test_anchored_solve`, `test_wandb_logging` (golden exclusion in place) + mutations | 3090 / g3n01 | 3 min 11 s | 17 OK, 0 skipped; mutations 3/3 caught, control passes |

Host memory of the suite jobs was not recorded by `sacct` (the batch step's MaxRSS is empty for 18139; the
request was 32G as in `run_tests.sbatch`).

### P1 identity (job 18242, CPU, pinned E, 1 thread, g3n01 — GPU 3090 allocated, unused; 1 min 7 s, peak 3.1 GB)

`p1_identity.py`: one ACM full-batch epoch (`acm_h2snn_v3b.yaml`, `epochs: 1`), each run in its own process from
`seed_everything`. `new` is this worktree; `old` is a temporary copy of `src/` with `anchored_solve.py` from
`2d822a4` (md5 `917168894004…` equals `git show 2d822a4:…`).

| comparison | train loss | val loss | parameters (36 tensors) | `pcg_zero_rhs_scopes` |
|---|---|---|---|---|
| old vs new | **bit-identical** (`3ff1e5a0e0000000` = 1.1185616254806519) | **bit-identical** (`3ff1449e80000000`) | **all equal** | 0 over 4 solves (also 0.0 in the logged stats) |
| new vs new (control) | bit-identical | bit-identical | all equal | 0 |

**Gate holds.**

### P2 PDNS `balmy-sweep-7` (job 18243, TITAN RTX, g2tn01; 1 h 06 min; host peak 14.9 GB, GPU peak 0.48 GiB)

`run_p.py` is card 11's `run_from_config.py` plus two additions. The zero-RHS probe wraps `pcg` outside the card 11
probe and logs each call's triggered scopes and their ‖b_s‖. `--stop_after_step 1:172` stops the run cleanly at
the first training solve of epoch 1 step 173 (amendment 2). No config key changed (`p2_pdns_changes.json`:
`[]`). `--mem` was 21G (card 11 peak 16.0 GB + 30 %). The card 11 B2a reference files (37fc77a) were used for the
comparison.

**Gate: passes epoch 1 step 172, no `PCGConvergenceError`.** Outcome line:
`stop: epoch 1 step 173 reached (passed step 172)`. The card 11 failing solve (backward, step 172, scope 27,
‖b_s‖ = **3.41e-17**, max over the step's scopes 9.0e-2) is now solved by the rule. That solve took 13 PCG
iterations for its other scopes, against card 11's 100 iterations and relative residual 7.0e5.

Per epoch (`p2_per_epoch.tsv`; 32 scopes per step):

| epoch | dir | solves | iterations max / mean / p95 | max rel. residual | unconverged | triggered scopes |
|---|---|---|---|---|---|---|
| 0 | fwd | 256 | 15 / 10.72 / 14 | 1.0e-3 | 0 | 0 |
| 0 | bwd | 256 | 15 / 10.69 / 13 | 1.0e-3 | 0 | **11** (steps 133–238) |
| eval (after epoch 0) | fwd | 8 | 20 / 18.75 / 19.6 | 1.0e-4 | 0 | 0 |
| 1 (to step 172) | fwd | 173 | 17 / 13.40 / 14 | 1.0e-3 | 0 | 0 |
| 1 (to step 172) | bwd | 173 | **15** / 13.20 / 15 | 1.0e-3 | 0 | **21** (steps 28–172) |

Triggered scopes and their ‖b_s‖ (`p2_triggers.tsv`), all in backward solves and never in a forward solve:

- epoch 0: 3.99e-13, 3.87e-12, 1.30e-12, 3.45e-13, 2.04e-13, 4.07e-13, 1.31e-14, 3.68e-13, 1.92e-12, 1.86e-13,
  1.29e-14;
- epoch 1: 2.43e-15, 2.31e-14, 1.63e-12, 5.53e-14 and 7.21e-12 (step 67, two scopes), 5.20e-12, 1.54e-15,
  1.67e-16, 7.67e-12, 6.80e-12, 2.55e-12, 3.82e-12, 1.01e-13, 3.21e-12, 3.57e-15, 7.04e-13, 7.16e-13, 4.37e-12,
  3.21e-13, 4.37e-16, **3.41e-17** (step 172).
- Per-call max ‖b_s‖: 0.05–0.10; median: 2e-3–1e-2.

**Against card 11 B2a** (`p2_vs_card11.tsv`; 866 solves matched in call order within epoch / mode / direction, up
to card 11's failure):
- **Iterations.** Max iterations and the unconverged count are equal in every solve except the failing one. Mean
  iterations differ only on the calls where the rule fired (the triggered scope now counts 0 instead of ~6–8).
- **Before the first trigger (epoch 0 step 133).** 690 solves are equal in every logged field. The rest differ
  only in the last digit of the relative residual or τ (≤ 1e-5), starting at epoch 0 step 9, which is before any
  trigger. This is the GPU run-to-run nondeterminism card 11 B1 measured (job 18065/18096: backward
  atomics). It is not the rule.
- **Epoch 0 metrics.** Train loss 0.39954816 here vs 0.39954813 in card 11: equal to 7 digits, not bit-equal.
  This epoch also contains 11 triggered backward scopes and the GPU nondeterminism above. Validation Macro-F1
  0.8725895195738922 is **identical**. Epoch 1 was not finished in either run, so there is no epoch-1 metric to
  compare.

So the answer to "do loss and Macro-F1 equal B2a up to the first trigger" is: yes up to GPU last-bit noise. The
run follows B2a solve for solve (iterations, τ to 1e-5, the same failing step and scope) until the rule takes
over at step 172.

Wall time: epoch 0 took 39 min here, against ≈ 63 min in card 11 B2a, which shared g2tn01 with B3.

### P3 — dropped (amendment 2): mag-year is checked in its real H100 run after the merge.

## L — degree check on every LP task (evidence only)

**Method and jobs.**
- `a1_data_checks.py` (card12 copy). The tasks are given by argument (`--task NAME=CONFIG`). acm_pa and dblp_ap
  use card 11's reference trial configs; the other six use `configs/experiments/<task>_h2snn_v3b.yaml`.
- **Change beyond the task list (needed, not in the card's wording).** The two large tasks have their own data
  object, so `run_large_task` takes the same measurements on `build_large_link_dataset(cfg)`, the trainer's own
  large-LP data. It uses the epoch-0 training negatives `hgb_train_negatives(train_pos, dst_range, seed + 0)`
  (the H2S-NN large-LP loop's draw) and `np.isin` for membership. It has no known-positive list; the column
  counts negatives that are task positives instead.
- `l_flags.py` applies the card's flag rule mechanically.
- Jobs (CPU, `--exclude=g2tn01,g3an01`):

| job | tasks | node | wall | host peak |
|---|---|---|---|---|
| 18249 | acm_pa, dblp_ap, imdb_ma, lastfm, pubmed, freebase_bb | g2n01 | 43 s | 3.6 GB |
| 18250 | h-pokec_uu | g2n01 | 9 min 35 s | 7.9 GB |
| 18251 | ogbn-mag_pp | g3n01 | 4 min 11 s | 6.5 GB |
| 18258 | summary (`l_flags.py`) | g2n01 | < 1 s | — |

The large tasks were first requested with 48G, an unmeasured first run; the peaks above are now measured.

### Per-task table (`docs/card12_evidence/L/l_summary.{tsv,md}`)

Features: "yes" means the type has input features. "no (Sequential)" means a learned per-node embedding of that
type (`TypeSpecificNodeEncoder`: Embedding + LayerNorm). "no (NoneType)" means no type of the graph has
features, and the model uses one learned per-node `identity_embedding`. Other relations are the kept relations
touching the type, besides the target pair. The AUROC is the degree-only score `log(1+deg_u)+log(1+deg_v)`.

| task | head / tail | features head / tail | other rel. head / tail | val pos deg-0 tail / head | val neg deg-0 tail / head | AUROC all relations: train / val / test 2-hop / test random | AUROC target pair: train / val / test random | (i) | (ii) | **flag** |
|---|---|---|---|---|---|---|---|---|---|---|
| acm_pa | paper / author | yes / no (emb.) | 3 / **0** | **0.470** / 0.000 | 0.124 / 0.000 | **0.684 / 0.448** / 0.320 / 0.458 | 0.685 / 0.470 / 0.484 | yes | yes | **FLAGGED** |
| dblp_ap | author / paper | no (emb.) / yes | 0 / 2 | 0.000 / 0.090 | 0.000 / 0.090 | 0.515 / 0.486 / — / 0.481 | 0.552 / 0.420 / 0.419 | no | yes | no |
| imdb_ma | movie / actor | yes / no (emb.) | 2 / **0** | **0.289** / 0.000 | 0.123 / 0.000 | 0.755 / 0.599 / — / 0.598 | 0.759 / 0.593 / 0.591 | **yes** | no | no |
| lastfm | user / artist | no (all featureless) | 1 / 1 | 0.116 / 0.000 | 0.146 / 0.000 | 0.862 / 0.802 / 0.568 / 0.819 | 0.878 / 0.794 / 0.812 | no | no | no |
| pubmed | disease / disease | yes / yes | 3 / 3 | 0.065 / 0.065 | 0.005 / 0.065 | 0.726 / 0.758 / 0.746 / 0.795 | 0.743 / 0.743 / 0.776 | no | no | no |
| freebase_bb | book / book | no (all featureless) | 7 / 7 | 0.027 / 0.026 | 0.024 / 0.026 | 0.690 / 0.668 / 0.599 / 0.665 | 0.695 / 0.666 / 0.664 | no | no | no |
| h-pokec_uu | user / user | yes / yes | 15 / 15 | 0.009 / 0.000 | 0.021 / 0.000 | 0.700 / 0.691 / 0.441 / 0.691 | 0.703 / 0.693 / 0.693 | no | no | no |
| ogbn-mag_pp | paper / paper | yes / yes | 2 / 2 | 0.000 / 0.000 | 0.000 / 0.000 | 0.626 / 0.615 / 0.615 / 0.615 | 0.671 / 0.655 / 0.655 | no | no | no |

"—": dblp_ap and imdb_ma have no test 2-hop negatives (0 candidates), as in card 11.

Message-graph degree, all relations (share deg-0 / median / mean; negatives keep the head):

| task | tail: train pos | tail: val pos | tail: test pos | tail: val neg | tail: test random neg | head: val pos (= val neg) |
|---|---|---|---|---|---|---|
| acm_pa | 0.000 / 2 / 3.56 | 0.470 / 1 / 2.49 | 0.463 / 1 / 2.35 | 0.124 / 1 / 1.34 | 0.154 / 1 / 1.33 | 0.000 / 94 / 92.8 |
| dblp_ap | 0.000 / 8 / 8.57 | 0.000 / 7 / 7.57 | 0.000 / 7 / 7.46 | 0.000 / 8 / 8.11 | 0.000 / 8 / 8.12 | 0.090 / 6 / 14.2 |
| imdb_ma | 0.000 / 3 / 6.24 | 0.289 / 2 / 5.34 | 0.290 / 2 / 5.34 | 0.123 / 1 / 1.92 | 0.122 / 1 / 2.01 | 0.000 / 8 / 7.40 |
| lastfm | 0.000 / 30 / 75.0 | 0.116 / 27 / 69.2 | 0.108 / 30 / 73.5 | 0.146 / 1 / 5.42 | 0.153 / 1 / 4.72 | 0.000 / 43 / 48.6 |
| pubmed | 0.000 / 19 / 1846 | 0.065 / 33 / 1589 | 0.001 / 38 / 1506 | 0.005 / 2 / 4.04 | 0.006 / 2 / 4.72 | 0.065 / 33 / 1589 |
| freebase_bb | 0.000 / 17 / 464 | 0.027 / 15 / 451 | 0.028 / 15 / 453 | 0.024 / 6 / 14.2 | 0.024 / 6 / 14.0 | 0.026 / 23 / 605 |
| h-pokec_uu | 0.000 / 40 / 55.9 | 0.009 / 39 / 55.0 | 0.009 / 39 / 54.9 | 0.021 / 11 / 24.3 | 0.021 / 11 / 24.3 | 0.000 / 65 / 109 |
| ogbn-mag_pp | 0.000 / 35 / 68.6 | 0.000 / 34 / 67.5 | 0.000 / 34 / 68.1 | 0.000 / 22 / 31.5 | 0.000 / 22 / 31.5 | 0.000 / 39 / 82.0 |

Train / test-2-hop negatives and the target-pair degrees are in `a1_degree.tsv` per task directory.

**Leakage** (`a1_leakage.tsv`):
- **Held-out positives.** Validation and test positives in the message graph: **0 for every task**, stored or
  reversed, any relation. They also never appear among another split's positives.
- **Training positives** stay in the message graph by construction: stored (dblp_ap, lastfm, freebase_bb,
  h-pokec_uu, ogbn-mag_pp), reversed (acm_pa, imdb_ma), or half stored and half reversed for pubmed
  (disease–disease; 16 366 of 32 167 each way).
- **Negatives that are known positives:**

| task | train | val |
|---|---|---|
| acm_pa | 4 / 8 059 | 1 / 895 |
| dblp_ap | 27 / 15 913 | 2 / 1 768 |
| imdb_ma | 5 / 11 971 | 0 / 1 330 |
| lastfm | 197 / 66 966 | 27 / 7 301 |
| pubmed | 1 059 / 32 167 | 53 / 1 942 |
| freebase_bb | 1 391 / 152 021 | 160 / 16 894 |
| h-pokec_uu | 1 441 / 18.1 M (task positives either way) | 145 / 2.0 M |
| ogbn-mag_pp | 389 / 4.4 M | 49 / 486 k |

  HGB does not filter these. Duplicates inside test 2-hop negatives: lastfm 613, h-pokec_uu 925, others ≤ 65.

**Flagged by the card's rule: acm_pa only.** The author decides which tasks stay.

## Other (unexpected or worth knowing)

1. **The rule fires far more often than the one failing scope.** On PDNS it fired on 32 backward scopes in 1.7
   epochs, with ‖b_s‖ from 3.4e-17 to 7.7e-12. In card 11 the old `pcg` converged on all of them except step
   172, since those values are above the underflow range. They now get `y_s = 0` instead of a solution of size
   ~1e-12, which is below fp32 resolution of the call (rule rationale), and the training trajectory is the same
   as B2a's up to last-bit noise. Forward solves never triggered, because the forward right-hand side is `h0`.
2. **imdb_ma meets condition (i) but not (ii).** 29 % of validation-positive actors are isolated against 12 % of
   negatives; actors have no features and no relation other than movie, the acm_pa pattern. But the degree-only
   AUROC does not invert (validation 0.599), so it is not flagged. The same isolation exists, but it is weaker
   than on acm_pa (29 % vs 47 %), and the movie end carries features.
3. **dblp_ap.** The target-pair-only AUROC inverts (0.552 / 0.420), as card 11 found. The rule uses all
   relations (0.515 / 0.486), and condition (i) does not hold (head deg-0 is 0.090 for positives and negatives
   alike), so dblp_ap is not flagged.
4. **h-pokec_uu test 2-hop degree AUROC is 0.441** (train 0.700). 2-hop negatives have higher tail degree than
   positives (median 53 vs 39). This concerns the hard test set, not the flag rule.
5. **Environment.**
   - Symlinked `vendor/` clones (card 11's setup) make `src/baselines/runtime/vendor_paths._module_root` resolve
     module files through the symlink, so they no longer match this worktree's `VENDOR_ROOTS`. H2GB is then
     imported twice (`KeyError: Module with 'GraphormerBias' already defined`): 85 suite errors in job 18137.
     Copying the git-ignored clones (251 MB) fixed it (job 18140: the three affected modules, 22 OK).
   - `/vendor/H2GB/` in `.gitignore` matches directories only, so a symlink there is *not* ignored.
6. **Process.**
   - Early in the card the agent ran three short unittest/prototype commands on the login node, against CLAUDE.md.
     Everything after that went through sbatch.
   - The first L submission placed ogbn-mag_pp on g3an01 (CPU only, A6000 node). It was cancelled within
     seconds and resubmitted with `--exclude=g2tn01,g3an01`.
   - No A6000 was used for any run.

branch=card12 sha=3742fab6e26a3a7cd391d9c8cd26f317d3070d47 card=docs/H2S-NN_card12_card.md base=2d822a4eb48867de4476b97159430a64d8468808
