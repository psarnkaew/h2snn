# Spec v1.12 — draft edits (item 5), for the author's approval

**Status:** DRAFT. Nothing here has been applied to `docs/H2S-NN_methodology_spec.md`. The spec on disk was checked before drafting (v1.11, md5 `9ff34b6c699841b12f59bf37de3c4ae8`, last modified 2026-11-10 15:59 JST); the old text below is copied from that file. When approved, the edits are applied in one pass with a changelog entry (G11–G16 below) and the file is re-read immediately before writing, in case another session changed it.
**Written:** 2026-11-11, design session. **Sources:** report §4 decisions (all (a), 2026-11-11 03:02), report §7 (L7 cost), report §8 (L8 correctness), this session's numerical check of Thm 4 (below).

Each edit: where, old text (exact), new text, reason, status tag.

---

## E-1. §6: stale cost-gate line (filter bank vs first-order)
**Where:** §6, line 204.
**Old:**
> - **Cost gate (before the protocol):** filter bank vs first-order on ACM at the default `d`: peak memory and time per forward + backward, parameter count; pre-registered overhead ≤ 25% time and ≤ 10% memory.

**New:** delete the line. (The v3-B cost gate is the 1.11 line further down; the 1.11 "Removed from the protocol" line already lists this gate as removed.)
**Reason:** internal contradiction; the filter bank does not exist in v3-B.

## E-2. §6: stale ablation-table line
**Where:** §6, line 219.
**Old:**
> - **Ablation table:** exactly the comparisons of §5 (C2–C4b, C8, C9) plus `h₀`-only (sanity) and the fixed-literature-schema row (interim D8 sampler; reported, not a claim).

**New:**
> - **Ablation table:** exactly the comparisons of §5 (C2, C3, C9, C10–C14) plus `h₀`-only (sanity) and the fixed-literature-schema row (interim D8 sampler; reported, not a claim).

**Reason:** C4, C4b, C8 were removed in 1.11 (§5 "Removed in 1.11"); C10–C14 were added and were missing here.

## E-3. §5 C13 and §6 cost gate: results and v2 references
**Where:** §5 row C13 (line 192) and §6 cost-gate line (line 228).
**Old (C13):**
> | **C13 Cost** | Peak memory and time per epoch on ACM/DBLP/IMDB vs v2 (1,661 / 2,705 / 889 MB) and v1 (ACM 9,260 MB, 3.0 s); PCG iterations at tol 1e-4; `|E_shared|`. Pre-registered gate: memory ≤ 3× v2, PCG ≤ 40 iterations, time ≤ 4× v2 | Gate missed | Fixed-`T` truncation; report |

**New (C13):**
> | **C13 Cost** | Peak memory and PCG iterations at tol 1e-4 on ACM/DBLP/IMDB (`d = 32`), with `|E_shared|` and the eliminated share; reference v1 (ACM 9,260 MB, 3.011 s/epoch, v1 paper Table 11; v1's `d`, precision and GPU as recorded in report §8, else stated as unknown). Time per epoch vs v1 and the baselines is measured on the server under the protocol. Pre-registered gate (measured 2026-11-11, report §7): memory ≤ 3× v2 H6, PCG ≤ 40 iterations: **met** (memory 1.71× / 1.85× / 1.51×; PCG 9 / 11 / 10 at tol 1e-4; ACM 0.30× v1) | Gate missed on the server, or time per epoch growing super-linearly (C5) | Fixed-`T` truncation; report |

**Old (§6 cost gate):**
> - **Cost gate (1.11; before the protocol, on ACM, DBLP, IMDB, one epoch, `d = 32`):** peak memory ≤ 3× v2's H6 numbers; PCG iterations ≤ 40 at tol 1e-4; time per epoch ≤ 4× v2. `|E_shared|` and the eliminated share are reported. A miss switches the solver to fixed-`T` truncation (`T` = the iteration count of the gate) and is reported; the method is not otherwise changed.

**New (§6 cost gate):**
> - **Cost gate (1.11; measured 2026-11-11 on an RTX 4060 Laptop 8 GB, `d = 32`, report §7):** peak memory ≤ 3× v2's H6 numbers and PCG iterations ≤ 40 at tol 1e-4: **met** on ACM, DBLP, IMDB. Time was within 4× v2 but the machine ran about 1.9× slower than during H6 (model-independent draw time), so no time ratio against v2 is reported (author decision 2026-11-11: v2 is not used further). Time per epoch is reported against v1 and the baselines from the server runs. A later miss of the memory or PCG bound on the server switches the solver to fixed-`T` truncation (`T` = the iteration count of the gate) and is reported; the method is not otherwise changed.

**Reason:** author decision 2026-11-11 (no paired v2 timing); report §7 results; v1 figures are from the v1 paper, not a measurement of ours.
**Open point for the author:** the "Falsified if" column of C13 needs a server-side criterion; the draft uses "gate missed on the server, or super-linear time (C5)". Alternative: keep C13 as met and move all time statements to C5.

## E-4. `hub_A` wording: no equality with v2 is claimed
**Where:** §6 v3-B arms line (line 226).
**Old (fragment):**
> `hub_A` (no fact states: eliminate all `s_e`, fact–cell term dropped; = v2's `L_P` under the anchored solve);

**New (fragment):**
> `hub_A` (fact–cell term dropped and every fact state eliminated exactly: the hub form, in which cells couple only to nodes; its relation to v2's `L_P` is not claimed and not tested);

**Reason:** the equality was never tested (T16 not run) and v2 is no longer used; a claim without a test is removed (§0 rule 1 spirit). Under author decision 4(a), the difference between `full` and `hub_A` is the fact–cell coupling, not the existence of fact states (fact states carry no input information, L14; their elimination is exact).
**Also (§8 positioning row "Reduction of v3-B to it", Sheaf Hypergraph Nets):** "`hub_A` + free maps + `β = 1/δ_e`" stays (it is a reduction statement about the hub form, not about v2).

## E-5. C11 wording: fact–cell coupling (author decision 4(a), 2026-11-11)
**Where:** §5 row C11 (line 190).
**Old:**
> | **C11 Fact states add over the hub form** (meta-review "capacity") | `full` vs `hub_A` (fact states eliminated everywhere; same parameters) | Not significantly better | Fact states are the computational form that preserves the sheaf, not a source of gains (L14) |

**New:**
> | **C11 The fact–cell coupling adds over the hub form** (meta-review "capacity") | `full` vs `hub_A` (same draw, maps, parameters; `hub_A` drops the fact–cell term) | Not significantly better | The fact–cell coupling is the form that keeps the three-level sheaf (Thm 3), not a source of gains; fact states carry no input information (L14) and their elimination is exact (Prop. 7′) |

**Reason:** handoff §2 consequence of decision 4(a): "Claim C11 must say fact–cell coupling"; never write "edge states carry new information".

## E-6. Thm 4: the production map family suffices (finding of L8, DEVIATION L8b-a)
**Where:** §4 row Thm 4 (line 150), §4 limit L4′ (line 166).
**Old (Thm 4, statement and status):**
> For any labelling `y` and distinct unit vectors `a_1..a_C`, maps that are `I` or one reflection per incidence make `x*_v = a_{y(v)}` a global section, hence a fixed point of the solve; under normalisation the section is `D_v^{1/2} a_{y(v)}`, separable by homogeneous linear classifiers. Needs `k ≥ 1` reflection, not Cartan–Dieudonné (correction of `v3B_design.html` §15 D-B6) | DERIVED; checked 2e-15 on 29/48 heterophilic edges |

**New:**
> For any labelling `y` and distinct unit vectors `a_1..a_C`, there are orthogonal fact maps with `F_{u◁e} a_{y(u)} = F_{v◁e} a_{y(v)}` on every fact; then `x*_v = a_{y(v)}` is a global section, hence a fixed point of the solve; under normalisation the section is `D_v^{1/2} a_{y(v)}`, separable by homogeneous linear classifiers. Such maps exist with `I` or one reflection per incidence, and also inside the production family `F = R_{ψ,role} H₂H₁` (`k = 2`, determinant fixed by `R`): choose `F_{v◁e}` freely and `H₂H₁` at `u` as the plane rotation taking `a_{y(u)}` to `R_{ψ,role}ᵀ F_{v◁e} a_{y(v)}` (two reflections: `H₁` along `a − b`, `H₂` along any vector orthogonal to `b`). Not Cartan–Dieudonné (correction of `v3B_design.html` §15 D-B6). Existence only: whether the map MLP learns these maps is not claimed | DERIVED; checked 2e-15 on 29/48 heterophilic edges (one reflection); L8 F1 ≤ 1.2e-14 with dense maps on 14/20 heterophilic facts, production operator, transports and solver; two-reflection construction checked numerically, `max ‖H₂H₁a − b‖ = 3.9e-14`, det = +1, d ∈ {2, 3, 4, 8, 32}, 200 random pairs each |

**Old (L4′):**
> **L4′** `k = 2` reflections do not cover O(d) (Thm 4 needs only one).

**New (L4′):**
> **L4′** `k = 2` reflections do not cover O(d); Thm 4 needs only a map taking one vector to another per incidence, which `k = 2` provides (a plane rotation), so the theorem holds for the production family.

**Reason:** L8 (DEVIATION L8b-a) showed that `FactorizedMapOperator` (det +1 per incidence for fixed `R`) cannot realise "I or one reflection" next to each other; the theorem's conclusion does not need that construction. The old text could be read as "the model cannot express Thm 4's sections", which is false.
**Optional test (not required):** F1 with `FactorizedMapOperator`, Householder vectors set directly (bypassing the MLP), same tolerance as L8 F1 (≈ 30 min for the coding agent).

## E-7. Consequential edits of the Part L decisions (all (a), 2026-11-11)
These follow from decisions already taken; listed so the spec matches the code.
| Where | Old | New | Decision |
|---|---|---|---|
| G6 (line 10) | "`τ = e^θ`, own optimizer group, init 1" | "`τ = e^θ`, in the scalar optimizer group (10× base lr, no weight decay, shared with the path-gate bias `b`), init 1" | 2(a) |
| §3 table row "Learned τ" (line 107) | "Learned `τ = e^θ` (own group, init 1)" | "Learned `τ = e^θ` (scalar group, init 1)" | 2(a) |
| §6 solver settings (line 227) | "`θ` init 0, own optimizer group at 10× the base learning rate" | "`θ` init 0, scalar optimizer group at 10× the base learning rate" | 2(a) |
| §3 table row "PCG + implicit gradient" (line 108) and §6 solver settings (line 227) | "Jacobi preconditioner `J = diag(1+τ, τ, τ)` is exact-scalar after normalisation" / "Jacobi `J = diag(1+τ, τ, τ)`" | "Jacobi preconditioner: the exact scalar diagonal of the system solved (`diag(1+τ, τ, τ)` on the full-state system; on the eliminated system the node and cell scalars include the Schur corrections `1 − Σ_{e∈E_elim} w²`)" | 1(a), DEVIATION U3-a |
| §3 table row "Fact states and cell states" | "State `(N + |E_shared| + F)·d` after Prop. 7′ elimination" | unchanged; add "(default kept set `E_shared`; author decision 4(a): not every fact is eliminated)" | 4(a) |
| §6 solver settings | — | add "evaluation solves use the final tolerance 1e-4 from epoch 0; the backward solve raises at `T_max` as the forward does" | 5(a), 10(a) |

## Proposed changelog entry (to prepend when applied)
> **Changes from 1.11 (to 1.12; after Part L units L7–L8, report §7–§8; author decisions of 2026-11-11):** (G11) C13 and the cost gate record the L7 result (memory and PCG met on ACM, DBLP, IMDB); no time ratio against v2 (author decision); time vs v1 and baselines on the server; (G12) stale §6 lines removed (filter-bank cost gate; ablation table now C2, C3, C9, C10–C14); (G13) `hub_A` is described as the hub form; no equality with v2's `L_P` is claimed; (G14) C11 is about the fact–cell coupling (decision 4(a)); (G15) Thm 4 and L4′: the production family `k = 2` provides the class-separating sections (existence; L8 DEVIATION L8b-a); (G16) the spec states the Part L decisions 1(a), 2(a), 4(a), 5(a), 10(a) (Jacobi on the eliminated system, θ_τ in the scalar group, kept set `E_shared`, evaluation tolerance, backward stop).
