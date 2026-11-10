# Mini-batch design — v3-B addendum (M2 → solver tolerance and `T_max`; per-scope PCG)

**Written:** 2026-11-11 (Part L, U6). **Amends:** `docs/minibatch_design_decisions.md` (M0–M4, 2026-11-10) for the v3-B propagation of spec v1.11 (G3, G10). That document is not edited; where the two differ, this addendum applies. Status tags as in the card: VERIFIED-CODE (file:line on branch `v3b`), DERIVED, UNVERIFIED.

## 1. What changes
| Item of `minibatch_design_decisions.md` | Under v3-B |
|---|---|
| §5 M2: `K` tuned, `K ≥ 2` (Chebyshev depth) | **Superseded.** There is no `K`: propagation is one anchored solve `(PᵀP + τL̂) Y* = Pᵀh₀` (spec R13). Its precision knobs are the PCG relative tolerance and `T_max`, both fixed by the card (§3) and never tuned: tolerance 1e-3 for the first 10 epochs, then 1e-4; `T_max = 100`, a stop condition (the solve raises `PCGConvergenceError`, VERIFIED-CODE `src/models/anchored_solve.py` `solve_anchored`). |
| §5 "one application of `A` already reaches every scope node, so K > 2 only re-mixes" | Moot: the solve is exact to tolerance on the scope whatever its diameter. |
| §5 M3 (degrees and context from the scope) | Unchanged, now for `D_v, D_e, D_f` of all three levels (`SheafEnergyOperator.degree`). |
| §7 batches as one disjoint graph | Unchanged, with one addition: **PCG inner products and stopping are per scope** (below). |
| §8 row "batch = disjoint union ⇒ each target's prediction equals its scope alone" | Holds only with per-scope PCG (DERIVED): with one global inner product, `α` and `β` would mix the scopes and the iterate of one scope would depend on the residuals of the others. |
| §10 spec text "§3 K row / OPEN-4 closed: K ≥ 2" | Replaced by spec v1.11 (G10): OPEN-4 closed as "no `K`". |
| §12 open item 6 (re-run Part K gate (ii) at the lowest K) | Moot (no `K`); the gate (ii) re-run, if the author still wants it, would be under the anchored solve. |

## 2. Per-scope PCG (M1 requires it)
The batch system is block-diagonal over scopes, so PCG on the batch equals PCG on each scope **iff** every scalar of the iteration is taken per scope:
```
α_s = <r, z̃>_s / <p, q>_s,   β_s = <r_new, z̃_new>_s / <r, z̃>_s,   stop_s: ‖r‖_s ≤ tol · ‖b‖_s
```
with `<·,·>_s` the segment sum over the rows of scope `s` (node, fact-state and cell rows alike). A converged scope gets `α_s = β_s = 0` and stops changing; the loop ends when every scope has converged or at `T_max` (then the whole batch raises). Implementation: `pcg(..., scope_rows, num_scopes)` and `segment_dot` in `src/models/anchored_solve.py` (VERIFIED-CODE); the full-batch model passes one scope (`scope_rows = None`), which is the plain inner product.

**Status.** The per-scope path is implemented but **UNVERIFIED by a test**: the card lists no multi-scope test (the "batch = one scope at a time, bit-identical" check of `v3B_final_model.html` §2.1 belongs to the mini-batch card), so none was written. The Jacobi preconditioner is scalar per row and therefore trivially per scope.

## 3. Warm start and the evaluation draw (M4 interaction)
The evaluation draw is fixed (E1), so the model keeps `Y*` of the last evaluation solve and starts the next epoch's evaluation solve from it (`H2SNN._warm_start`, used only in eval mode and only when the system has the same row count). Training draws change every epoch, so training solves start from the D-B7 initial guess. In mini-batch, the stored `Y*` would have to be per scope (per target) since batches are re-cut; that storage is the mini-batch card's decision.

## 4. What the mini-batch card must add for v3-B
1. `scope_rows` built from the batch's relabelled ids: node rows by target, fact-state rows by the scope of the fact's source node, cell rows by the scope of `v₀`.
2. The bit-identity test (batch vs one scope at a time, same tolerance).
3. Per-scope warm-start storage for evaluation, or none (then evaluation solves start cold; a cost, not a correctness issue).
4. The elimination (`EliminatedOperator`) is per complex and applies unchanged to a scope (facts on ≤ 1 cell of the *scope* are eliminated; a fact shared by two cells of different scopes does not exist, since cells never cross scopes).
