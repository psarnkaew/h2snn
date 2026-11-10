# Task card — v3-B: three-level sheaf energy, anchored solve, arms (Part L)

**Written:** 2026-11-11 (design session). **Design authority:** `docs/H2S-NN_methodology_spec.md (v1.11)` (binding), then `docs/v3B_final_model.html` §1–2 (equations and completeness list), `docs/v3B_decisions.html` (why each decision), `docs/v3B_prior_art.html` (wording rules), `docs/minibatch_design_decisions.md` (M0–M4; M2 superseded by this card's solver settings). Author decisions D-B0(b), D-B1(a), D-B2(a), D-B3(b), D-B4(b), D-B5, D-B6, D-B7 were taken on 2026-11-10 23:43 JST and are not re-opened here.
**Scope:** correctness and function only. **No performance run, no tuning, no training beyond smoke.** Every test is algebraic (dense vs matrix-free, finite differences, known identities) on tiny fixtures, plus a 2-epoch smoke on IMDB. **The tests listed in §2 are the complete set; add no others.**
**Status tags for the report:** VERIFIED-RUN (command + numbers), VERIFIED-CODE (file:line), DERIVED, UNVERIFIED, DEVIATION (anything not as specified here, with reason).

---

## 0. Preconditions and git
- Repo root `C:\Users\Acer\Desktop\h2snn` (code in `H2S-NN/`). Current branch `part-j`, HEAD `c41bf28`. The working tree has **uncommitted changes that belong to other sessions** (`H2S-NN/AGENTS.md`, `H2S-NN/CLAUDE.md`, `docs/H2S-NN_v2_plan.md`, `docs/possible_reviewer_qa.md`, `docs/session_handoff.md`, `docs/working_preferences.md`, `H2S-NN/paper/*`, untracked `references/*.pdf`, `Claude outputs/`, `vendor/H2GB/`). **Never stage, commit, stash, reset or edit them.** Stage by explicit path only.
- Create branch `v3b` from `c41bf28` (`git checkout -b v3b c41bf28` is fine with the dirty tree; the dirty files stay unstaged). Record `git rev-parse HEAD` in the report.
- One commit per unit (U1–U6), message `Lx: <unit title>`. No push, no merge, no rebase until the author has read the report.
- Baseline: `python -m unittest discover -s tests` once before U1 and record the count (expected: failures only in `test_baselines`, as in Part J).
- Python/torch as in the repo; CPU is enough for U1–U4; U5 uses the RTX 4060 (8 GB) and may fall back to CPU with reduced `r`.
- New code lives in new files; existing v2 code paths (`filter_bank.py`, gates, `VertexPathSheafLaplacian`) are **not deleted** in this card (they stay reachable only under the legacy config `propagation: chebyshev`, and their tests must still pass). Removal is a later housekeeping card after the author approves.

## 1. Specification (what U1–U4 implement)

Notation: nodes `v` (N), facts `e` (E, D7 fact layer), cells `f` (F, WISP draws), `d` = stalk dim. Learned once per forward: `h₀`, `c`, `F_{v◁e}` (orthogonal, `FactorizedMapOperator`), `a_e = c_e²`, `β_f = w_f m_f`, `τ = exp(θ)`.

Transport (existing `PathTransportOperator`): `T_{v_0,f} = I`, `T_{v_i,f} = T_{v_{i-1},f} F_{v_{i-1}◁e_i}ᵀ F_{v_i◁e_i}` (signed hops as implemented). Fact→cell map `R_{e_i,f} = T_{v_{i-1},f} F_{v_{i-1}◁e_i}ᵀ`.

Energy (spec §3 Eq. E) and operator `L = ∇²E` on `Y = (x, s, z)`, applied matrix-free:
```
(L Y)_v = Σ_{e∋v} a_e F_{v◁e}ᵀ (F_{v◁e} x_v − s_e)  +  Σ_{f∋v} β_f T_{v,f}ᵀ (T_{v,f} x_v − z_f)
(L Y)_e = Σ_{v∈e} a_e (s_e − F_{v◁e} x_v)           +  Σ_{f∋e} β_f R_{e,f}ᵀ (R_{e,f} s_e − z_f)
(L Y)_f = β_f [ Σ_{e∈f} (z_f − R_{e,f} s_e)  +  Σ_{v∈f} (z_f − T_{v,f} x_v) ]
D_v = Σ_{e∋v} a_e + Σ_{f∋v} β_f ;  D_e = 2 a_e + Σ_{f∋e} β_f ;  D_f = β_f (2 ℓ_f + 1)        # scalars (FᵀF = I)
L̂ = D^{-1/2} L D^{-1/2}       # every diagonal block of L̂ is I
```
Matrix-free fact↔cell transport (new; mirrors `transport_sum` / `adjoint_prefix`):
```
UP   (facts → cell f):   u ← 0;  for i = ℓ..1:  u ← Q_i u + F_{v_{i-1}◁e_i}ᵀ s_{e_i}   ;  Σ_i R_{e_i,f} s_{e_i} = u
DOWN (cell f → facts):   t ← z_f; for i = 1..ℓ:  r_{e_i} += F_{v_{i-1}◁e_i} t ;  t ← Q_iᵀ t
with Q_i = F_{v_{i-1}◁e_i}ᵀ F_{v_i◁e_i}  (the signed step of PathTransportOperator)
```
Elimination (Prop. 7, 7′; exact): facts on 0 cells → direct node–node term weight `â_e/2`; facts on exactly 1 cell `f` → direct term `â_e²/(2â_e+β̂_f)` plus node–cell weight `â_e β̂_f/(2â_e+β̂_f)` added at both endpoints (hatted = after normalisation, i.e. apply the elimination to `L̂`'s weights; `D` is computed from the full `L` first). Fact states remain only for `E_shared` (facts on ≥ 2 cells).

Anchored solve (spec R13): `M = PᵀP + τ L̂`, `M Y* = Pᵀ h₀`, `P` = node selector.
```
PCG(M, b = Pᵀh₀, Y₀, J, tol, T_max, scope_id):
  J = diag(1+τ on node blocks, τ on fact and cell blocks)        # exact Jacobi after normalisation
  Y₀ = (h₀, s₀, z₀) with s₀_e = ½(F_u h_u + F_v h_v), z₀_f = mean over f of {R_{e,f} s₀_e, T_{v,f} h_v}
  r = b − M Y₀ ; zt = J⁻¹ r ; p = zt
  loop t = 1..T_max:
     q = M p
     α = <r, zt>_scope / <p, q>_scope            # inner products are per-scope segment sums (M1)
     Y += α p ; r_new = r − α q
     converged_scope = ‖r_new‖_scope / ‖b‖_scope ≤ tol   # per-scope; converged scopes are masked (α := 0)
     stop when all scopes converged
     zt_new = J⁻¹ r_new ; β = <r_new, zt_new>_scope / <r, zt>_scope ; p = zt_new + β p
  return Y*, iterations per scope
```
Full batch = one scope. Backward (implicit): solve `M λ = Pᵀ g` with the same PCG, then `∂ℒ/∂θ = −τ ∇_θ [λᵀ L̂(θ) Y*]` by autograd of the scalar with `λ, Y*` detached; `∂ℒ/∂h₀ = P λ` (+ the direct head path); `∂ℒ/∂τ = −λᵀ L̂ Y*`. Implement as `torch.autograd.Function` (`AnchoredSolve.apply(h0, params…)`), where the forward runs under `no_grad` and the backward recomputes `λᵀ L̂ Y*` with grad enabled.

Head: `ŷ_v = MLP([h₀_v, x*_v])` (existing). Tolerance schedule (fixed, never tuned): relative tol 1e-3 for epochs 1–10, then 1e-4; `T_max = 100`; warm start on the evaluation draw from the previous epoch's `Y*` (training draws change, so no warm start there).

Arms (config `model.propagation: anchored`, `model.arm:`): `full`, `covering_only` (drop node–cell term; `D` accordingly), `hub_A` (eliminate all `s_e`; drop fact–cell term; node–cell term stays = v2 `L_P` with weight `β_f`), `closure_only` (no cells; `HolonomyEdgeTerm` with weights `β_f`), `no_path_cells` (`β_f = 0`), `instance_features` (`β_f = 0` in the solve; `Σ_{f: anchor v} β_f · mean_{u∈f} T_{u,f} h₀_u` concatenated to the head input of anchor `v`), `free_cell_maps` (`R_{e,f}` free orthogonal per incidence; test-only), `uniform / schema / endpoints` (existing C2 ladder on `g`'s input).

## 2. Units

| Unit | Deliverable | Gate (all must pass; tolerances fixed here) |
|---|---|---|
| **U1 operator** | `src/models/sheaf_energy.py`: `IncidenceComplex` (node–fact `[2,E]`, fact–cell `[I_ef, 5] = (f, i, e, u, v)`, node–cell `[I_nc, 2]`, per-fact cell count), `FactCellTransport` (UP/DOWN above, built on `PathTransportOperator`'s signed steps), `SheafEnergyOperator.apply(Y) -> L Y`, `.degree()`, `.apply_normalized(Y)`; `tests/test_sheaf_energy.py` | T1 functoriality `R_{e_i,f} F_{v◁e_i} = T_{v,f}` at both ends of every incidence on a real WISP draw (IMDB, r = 2, 50 anchors): max ‖·‖ ≤ 1e-5 fp32. T2 matrix-free vs dense `L` on a random fixture (N = 12, E = 20, F = 6, d = 4, fp64): ≤ 1e-12. T3 symmetry `<Y₁, L Y₂> = <L Y₁, Y₂>` ≤ 1e-12 and PSD (min eig ≥ −1e-10); needed for PCG to be valid. |
| **U2 elimination** | `EliminatedOperator` (Prop. 7/7′) producing the reduced `M` on `(x, s_shared, z)`; `|E_shared|` statistics | T7 `x*` and `z*` from the full-state solve vs the eliminated solve (dense, fp64) ≤ 1e-10 on fixtures where facts lie on 0, 1 and 2 cells. |
| **U3 solver** | `src/models/anchored_solve.py`: batched per-scope PCG, Jacobi, warm start, tolerance schedule, `AnchoredSolve` autograd Function | T9 PCG vs dense solve (fp64, tol 1e-10) ≤ 1e-8. T10 implicit gradient vs central finite differences (fp64, ε = 1e-6) on every parameter group incl. `θ_τ`: relative error ≤ 1e-6. |
| **U4 model + arms** | `h2snn_model.py`: `propagation: anchored` path (config-validated), `τ` parameter in its own optimizer group (10× base lr), arms as configs; `tests/test_v3b_model.py` | T14 every arm runs forward + backward on the fixture; loss finite; every parameter group receives a nonzero gradient where it should (and **zero** where it should not: `hub_A` has no fact-state path). |
| **U5 smoke run** | `scripts/v3b_cost_smoke.py`; JSON | On **IMDB only** (`d = 32`, `r = 12`, `ℓ_max = 4`), arm `full`, **2 epochs** of the normal training step (forward + backward + optimizer step, full batch; if OOM halve `r`). Gate: loss is finite in both epochs; loss of epoch 2 differs from epoch 1; every parameter group (incl. `θ_τ`) receives a nonzero gradient; PCG converged (< `T_max`) in every solve. Record (report only, no gate): peak `max_memory_allocated`, time per epoch, PCG iterations fwd/bwd, `|E_P|`, `|E_shared|`, eliminated share, τ. No accuracy number. **Not in this card (deferred):** ACM/DBLP cost runs and the cost gate vs v1/v2 (memory ≤ 3× v2, time ≤ 4× v2); they must be run before claim C13 is written as a result. |
| **U6 docs + report** | `docs/minibatch_v3B_addendum.md` (M2 → tol/T_max; per-scope PCG), `docs/H2S-NN_v3B_report.md` | Report format in §4. No edits to `docs/minibatch_design_decisions.md` or any other existing doc. |

Order: U1 → U2 → U3 → U4 → U5 → U6. U2 may be skipped **only** if `|E_shared|/|E_P| ≥ 0.9` on IMDB in a quick count (then elimination saves nothing; report the count and the decision as DEVIATION).

## 3. Rules
- Never tuned / never in a search space: `tol`, `T_max`, the tolerance schedule, `τ` init, `ℓ_max`, `r`, `proposal`, `path_weight_input`.
- No training runs beyond the 2-epoch IMDB smoke. No accuracy numbers in the report (if a smoke prints one, it is omitted).
- Deviations from this card are allowed only to make a test pass for a reason you can state; each is logged as DEVIATION with file:line.
- fp64 for algebraic tests; fp32 for smokes.
- **Stop and write the report** (do not continue to later units) if: a gate of U1–U4 fails after one honest fix attempt; PCG hits `T_max` on any fixture; `τ > 50` or max log-weight > 30 in a smoke; OOM on IMDB even at `r` = 4; a theorem check contradicts the spec (e.g. T2 or T7 mismatch, or a non-finite loss in the smoke).
- Not stop conditions (report only): PCG iteration count, memory and time numbers in U5.

## 4. Report format (`docs/H2S-NN_v3B_report.md`)
1. Branch, base hash, commits per unit. 2. Per unit: what was built (file:line), each test with its number and tolerance, PASS/FAIL, DEVIATIONs. 3. U5 numbers for IMDB (memory, time, iterations fwd/bwd, |E_P|, |E_shared|, eliminated share, τ) next to v2 H6 (IMDB 889 MB / 1.33 s); no v1 comparison (v1 was measured on ACM). 4. Judgment calls made, each marked "decision for the author" with options. 5. What was not done and why. 6. Time spent per unit.

## 5. Time estimate (estimate; longer if the signed-hop handling of `PathTransportOperator` needs refactoring for fact states, or if the 4060 OOMs and the smokes must move to CPU)
U1 2 h · U2 1 h · U3 2 h · U4 1.5 h · U5 0.5 h · U6 1 h ⇒ about 7–9 h of agent time (estimate), 1–2 sessions.

---

## 6. Ready-to-paste prompt for Claude Code

```text
You are implementing Part L (v3-B) of H2S-NN in the repo at C:\Users\Acer\Desktop\h2snn (code in H2S-NN/).

BINDING DOCUMENTS, in this order of authority:
1. docs/H2S-NN_v3B_card.md  (this task card: units, tests, tolerances, rules, stop conditions)
2. docs/H2S-NN_methodology_spec.md (v1.11)  (design authority; §3 equations, §4 theorems, §6 solver settings and arms)
3. docs/v3B_final_model.html §1–2, docs/v3B_decisions.html, docs/minibatch_design_decisions.md (M0–M4; M2 is superseded by the card)
4. docs/working_preferences.md and H2S-NN/AGENTS.md / CLAUDE.md for working rules.
Read 1 and 2 fully before writing code. Read the existing code you build on: src/models/map_operators.py (PathTransportOperator, FactorizedMapOperator), src/models/vertex_path_laplacian.py, src/models/sheaf_maps.py, src/data/wisp_sampler.py, src/models/h2snn_model.py (encode()).

AUTHOR DECISIONS ALREADY TAKEN (do not re-open): energy on all comparable pairs (node–fact, fact–cell, node–cell) with one weight β_f per cell; no gates; no filter bank / K; symmetric normalisation on all three levels; τ = exp(θ) learned in its own optimizer group; PCG + implicit gradient with per-scope inner products; householder_k stays 2; closure edges carry no state and have no code path; v2 code paths stay in place behind the legacy config and are not deleted.

SCOPE AND ORDER: U1 operator → U2 elimination → U3 solver → U4 model + arms → U5 2-epoch IMDB smoke → U6 docs + report. Each unit's gate (tests with the tolerances in the card) must pass before the next unit starts.

WHAT MAY RUN: unit tests on tiny fixtures (fp64), the T1 functoriality check on one real WISP draw (IMDB, r = 2, 50 anchors), and a 2-epoch smoke on IMDB only in U5 (d = 32; r = 12). Write only the tests listed in the card, no others. No training runs, no tuning, no accuracy numbers anywhere in the report. If the GPU runs out of memory, halve r until it fits and record it.

STOP CONDITIONS (stop, write the report, do not continue): a U1–U4 gate fails after one honest fix attempt; PCG reaches T_max = 100 on any fixture; τ > 50 or max log-weight > 30 in a smoke; OOM on IMDB at r = 4; a theorem check contradicts the spec. NOT stop conditions (report only): PCG iteration count, memory and time numbers in U5.

GIT: create branch v3b from commit c41bf28 (git checkout -b v3b c41bf28). The working tree contains uncommitted changes of other sessions (AGENTS.md, CLAUDE.md, docs/H2S-NN_v2_plan.md, docs/possible_reviewer_qa.md, docs/session_handoff.md, docs/working_preferences.md, paper/, references/*.pdf, "Claude outputs/", vendor/H2GB/): never stage, commit, stash, reset or edit them; stage new/changed files by explicit path only. One commit per unit, message "Lx: <title>". Never push, merge or rebase. Record git rev-parse HEAD at start and after each commit.

NEW FILES: src/models/sheaf_energy.py, src/models/anchored_solve.py, tests/test_sheaf_energy.py, tests/test_anchored_solve.py, tests/test_v3b_model.py, scripts/v3b_cost_smoke.py, docs/minibatch_v3B_addendum.md, docs/H2S-NN_v3B_report.md. Changes to existing files only in h2snn_model.py (new propagation path + arms + τ group) and configs (new validated keys). Do not edit any other existing doc.

REPORT: docs/H2S-NN_v3B_report.md in the format of card §4; every number with the command that produced it; every deviation tagged DEVIATION with file:line; every judgment call listed as "decision for the author" with options. Then stop.
```
