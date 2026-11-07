# G1–G3 — pre-registration (written 2026-11-08, before any G1/G2 code change)

Authority: methodology spec v1.2 §3–§4; task card Part G. Nothing below is changed after a result;
additions go under a dated heading.

## G1 — relational context `c` and `model.geometry_context`
**Change.** `model.geometry_context: bool` (default `true`). `c = RelCtx(h₀)`: for every node, every
fact relation `r` (compact ids after the fact layer) and role (0 = the node is the stored source, 1 =
target), the mean of the other endpoint's `h₀` over the facts in that slot; zeros if none. A
shared-role relation (ρ(r) = r, or every relation under `direction_blind_maps`) has one role: both
incidences go to slot 0, slot 1 stays zero, so `c` does not depend on the stored orientation.
`c ∈ R^{N × 2R·d}`; geometry inputs `[h₀, c]` replace `h₀` in: the map MLPs (factorized and dense
families: reflector, scale), the gate heads, the path encoder (and its ablation variants:
`EndpointEncoder`, `FreeIncidenceMaps`). Unchanged: the readout `[x, h₀]`; the `pair_attention`
score network (keeps `h₀`; C4b arm unchanged — its path latents come from the shared path encoder).
`false` = D11 exactly (same parameters, same creation order, same outputs).
**Tests (`tests/test_filter_bank.py`).** T-G7 (exact): perturbing one neighbour `u` of `v` changes
`v`'s maps on an edge not incident to `u`, and `v`'s gates, under `true`; leaves them bit-identical
under `false`. `c` equals a loop reference; a self-inverse relation's `c` is unchanged when its
facts' stored orientation is flipped (control: a non-shared relation's `c` changes); facts only: with
a tied fact layer over stored edges in both directions, `c` equals the loop reference over the
stored relation `r` alone (no double counting through ρ(r)). D11 golden outputs unchanged under
`geometry_context: false` (+ `encoder: first_order` after G2).
**Wrong if** any of these fails, or the context-off model is not bit-identical to D11.

## G2 — filter-bank encoder and `model.encoder`
**Change.** `model.encoder: filter_bank | first_order` (default `filter_bank`). Module
`src/models/filter_bank.py`: `V ∈ R^{(K+1)×d×d}`, initialised `V_k = θ_k I`, θ = Chebyshev
coefficients of `(1 − η₀(1 + ξ))^K` (`numpy.polynomial.chebyshev.poly2cheb`, float64),
η₀ = `model.eta_init`; `W_k = V_k / max(1, Σ_j ‖V_j‖₂)` (`torch.linalg.matrix_norm(ord=2)`);
`x = Σ_k T_k(A − I)(h₀ W_k)` evaluated by Clenshaw's recurrence on `L = A − I` (vector-valued
coefficients `h₀ W_k`; K applications of `A`, never materialized; mathematically the card's
`T_k` recurrence — T-G6 checks it against a dense `Σ T_k(L) (I ⊗ W_kᵀ)` built from materialized
`A`). `A x = g ⊙ Δ̂(b ⊙ x)` (gates skipped when absent), shared with the first-order update so
`first_order` stays bit-identical to D11. Layer dropout is applied once to the filter output
(default `layer_dropout = 0`). `θ_η` exists only under `first_order`; `V` is in the normal
optimizer group; gates and γ stay in the scalar group. `pair_attention` under `filter_bank`: the
same filter with `A = I − P` (equals D11's attention update with η = η₀ at initialisation); no gain
guarantee for this comparison arm (P is not self-adjoint) — stated, not hidden.
Diagnostics (logged every `log_every` epochs under `filter_bank`, replacing `eta_lambda_max`):
`filter_norm_sum = Σ‖W_k‖₂` (Prop. 5's certified bound, valid when σ(A) ⊂ [0, 2]),
`lambda_max_A` (power iteration on the similar symmetric form; lower bound), `filter_gain`
(power iteration of `F*F` in the metric `M = BG⁻¹`, 30 steps; lower bound of the M-gain).
**Tests** (card tolerances): T-G1 spectrum of `A` real, in `[0, 2·max g b]` (≥ 50 fixtures, 1e-9);
T-G2 M-gain ≤ 1 + 1e-9 for random `V`, control without normalization > 1 on some fixture; T-G3
exact zero blocks with `g_u = 0` / `b_v = 0`, control gates 0.5 > 1e-8; T-G4 filter at the D11
init equals `first_order` for η ∈ {0.25, 0.5, 0.9}, K ∈ {2, 5} (1e-10); T-G5 Lemma N diagonal
blocks of `Δ̂` = I for nodes without self-loop facts and not repeated inside a cell (1e-12); T-G6
matrix-free = dense (1e-10); T-G8 non-zero gradients to every `V_k`, gate heads, γ, map MLPs.
Fixtures float64 with path cells and signed hops. Mutations (`D12_evidence/g2_mutation_check.py`):
M1 normalization removed → T-G2 fails; M2 recurrence on `A` instead of `A − I` → T-G4 fails;
M3 gates as `B Δ̂ G` → T-G3 fails.
Data facts documented (not tested): self-loop facts per dataset, and cells with a repeated node
(Lemma N's premises).
**Expected**: all pass at the first attempt except possibly tolerance-free numerics of T-G2 (power
method not used there: exact dense norms). **Wrong if** any fails after 3 diagnosed attempts (stop).

## G3 — cost gate
ACM, `configs/experiments/acm_h2snn.yaml` (d = 32, K = 4, defaults otherwise, i.e. geometry
context on in both arms), seed 42, GPU, training mode: one forward + backward (loss = cross-entropy
on the training nodes), 1 warm-up then 3 repeats; time with `torch.cuda.synchronize`, peak memory
`torch.cuda.max_memory_allocated` after `reset_peak_memory_stats` per repeat; parameter count.
**Gate:** filter_bank vs first_order overhead ≤ 25 % time (mean of 3) and ≤ 10 % peak memory.
Informational, not gated: the same with `geometry_context: false`.
**Expected:** < 10 % time, < 5 % memory (same number of operator applications; K+1 extra `N×d×d`
products). **Wrong if** exceeded → stop (card stop condition).
