# Spec v1.25 — draft edits (part Z: relational context `c` removed; restriction maps → TD-1), for the author's approval

**Status:** APPROVED by the author and APPLIED (2026-11-18; spec v1.25). (Z-1–Z-9: `c` removed; Z-10–Z-19: maps → TD-1, m = 4). Decision: `docs/experiment_settings_decisions.md`, row of 2026-11-18 "Relational context `c` removed" (author). The other pending 1.25 items (B1a, B2a, OOM range rule 1, §1 GC heterophily line) are drafted with card 5 and are added to this file then; part Z is independent of them.

**Scope:** spec v1.24 (`docs/H2S-NN_methodology_spec.md`) and the pre-registration that is frozen with it (`docs/ablation_analysis_preregistration_draft.md`). The positioning table (§8, row "Identity of intermediates in geometry … yes (pair-conditioned maps)") needs no edit: it now holds exactly. Change logs of earlier versions are not edited.

---

## Z-1 Header: change line for 1.25
**Insert** (above "Changes from 1.23 (to 1.24)"), merged with card 5's items when those are drafted:
> **Changes from 1.24 (to 1.25; author decision of 2026-11-18 in `docs/experiment_settings_decisions.md`):** (Z1) the relational context `c` is removed from the method: the geometry (maps `F_{v◁e}`, fact weights `a_e`) reads each endpoint's `h₀` only, as `g` already did (1.4); R10 restated; C9, its arm and the appendix pair are withdrawn. Reasons: the map claim (pair conditioning carries the identity of intermediate nodes) holds exactly only without `c`; graph structure reaches the representation only through the sheaf solve, which keeps C2's attribution clean; the evidence for `c` (D12 G0/G1) was measured on v1/v2. Accepted without a test (author).

## Z-2 §2 requirements, row R10
**Replace the row beginning** `| R10 | The geometry (restriction maps, fact weights, path weights) depends on each node's **typed neighbourhood**` **with:**
> | R10 | The geometry (restriction maps, fact weights, path weights) is a function of the **incidence alone** (its nodes' encodings `h₀`, relations and roles; no neighbourhood summary), and is computed **once** | Graph structure enters the representation only through the sheaf solve; one sheaf per forward pass keeps R5 and R7 and makes every theorem of §4 apply |

## Z-3 §3 pipeline block
**Delete the line:**
> `c   = RelCtx(h₀)                                                # per (relation, role): mean of neighbours' h₀`

**In the line beginning** `F_{v◁e}, a_e, β_f = w_f m_f` **replace** `computed once from [h₀, c] / h₀` **with** `computed once from h₀`.

## Z-4 §3 component table
**Delete the row beginning** `| Relational context `c` | R10 |`.

## Z-5 §3 "Heterogeneity" bullet
**Replace** `type encoder, relation-and-role frames, relational context `c`, path cells` **with** `type encoder, relation-and-role frames, path cells`.

## Z-6 §3.2 (c) Weights
**Delete** ` (1.4: `h₀`, not `geom = [h₀, c]`)` (no `geom` exists any more; the 1.4 change log keeps the history).

## Z-7 §5 claims table
**Delete the row beginning** `| C9 Relational context helps (R10) |`. Claim numbers are not reused.

## Z-8 §6 Ablation table bullet
**Replace** `exactly the comparisons of §5 (C2, C2b, C9–C17)` **with** `exactly the comparisons of §5 (C2, C2b, C10–C17)`.
**Delete the sentence** `The appendix pair (`g` reads `[h₀, c]` in `full` and `endpoints`) runs with configuration (a) on seeds 1–10 and is reported without a claim (1.23).`

## Z-9 Pre-registration (`docs/ablation_analysis_preregistration_draft.md`, frozen with the spec)
- Claim map: **delete the row** `| Relational context in the maps | C9 (`geometry_context: false`) | 3 |`.
- Arms table: **delete the rows** `| geometry context off | … |` and `| Appendix pair | … |`.
- Decision rules: **delete the row** `| C9 relational context | `full` vs geometry context off (a) | better on ≥ 3 of 4 | Drop `c` (spec) |`.
- Closed-items line of 2026-11-16: **append** ` Withdrawn 2026-11-18 (author): the appendix pair and C9, with `c` (spec 1.25 Z1).`
- Holm families are per claim (m = 4 datasets), so no other family changes; check any run-budget line that counts the `geometry_context_off` arm or the appendix pair, and reduce it.

---

## Not spec text: what the change touches (for the card)

**Code** (`v3b`, H2S-NN): remove `model.geometry_context` and `ablation.path_weight_context` (config keys, guards at `src/configs/config.py:67, 74, 235-238, 300-304, 331, 1346, 1408-1410, 1459, 1492, 1524-1526, 1616-1619`), `src/models/relational_context.py`, and the `[h₀, c]` path in `src/models/h2snn_model.py` (`_geometry_input` and the `factored_inputs` flags that follow `geometry_context`); arm `geometry_context_off` in `scripts/ablation/arms.py:45`; `scripts/known_answer/s3_audit.py` (imports `RelationalContext`), `scripts/v3b_cost_smoke.py`, `CLAUDE.md:888`. Tests: `tests/test_relational_context.py` (delete), `tests/test_v2_model.py` (T11-ctx, appendix-pair cases), `tests/test_wisp.py` (34 references). `endpoint_hidden_width` is unaffected (`g` reads `h₀` in both arms). Golden and mb0 fingerprints change by design: regenerate them and record the old/new hashes in the report. A config that still sets either key must be refused with a message that names spec 1.25 Z1.

**Results obtained with `c`** (open, author): card 4 known-answer (S1′-B, S3, S2′) and the card 2 central-point timings that fix B2a's trial count `N`. Either rerun on the `c`-free model before the freeze, or report them as obtained on the model with `c` (deviation) and the timings as upper bounds.

**Paper** (`h2snn_ijcai27_v3b.tex`; text proposed in chat, not edited here): line 149 sentence defining `c_v`; Eq. (maps) line 155 `u = MLP([h_v, h_{v'}, ζ_r, ρ])`; line 159 add that the geometry is a function of the incidence alone; line 222 "No relational context" sentence; line 355 ablation row.

**Other docs:** `docs/minibatch_design_v3B.md:34` (scope example mentions `c`) — a note that the difference no longer exists, at the next edit of that file.

---

# Part Z (continued): restriction maps → TD-1 (Z-10 – Z-19)

**Decisions:** `docs/experiment_settings_decisions.md`, rows of 2026-11-18 "Restriction maps: SO(d) kept; instance part → TD-1" and "TD-1 rank m; pre-implementation diagnostic" (author). Analysis: artifact "H2S-NN Advisor Review" §2.5–2.15. Applies on top of Z-1–Z-9 (the text below assumes `c` is already removed).

**The new maps (for reference in every item below):**
```
F_{src◁e} = c_e · I
F_{tgt◁e} = c_e · R_r · exp(U_r C_e U_rᵀ)            (relations with ρ(r) ≠ r)
F_{src◁e} = c_e · exp(−U_r C_e U_rᵀ / 2),  F_{tgt◁e} = c_e · exp(+U_r C_e U_rᵀ / 2)   (ρ(r) = r; C_e antisymmetric in the endpoints)
R_r ∈ SO(d)   one FastHPP frame per relation (d reflections, d a power of two ⇒ det +1)
U_r ∈ ℝ^{d×m} orthonormal (thin QR of a learned d×m matrix), m = 4
C_e ∈ so(m)   linear in one per-edge MLP over [h_src, h_tgt, rel_emb] (last layer zero-initialised ⇒ C_e = 0 at init); the same MLP gives c_e = softplus(·) + ε
exp(U C Uᵀ) = I + U (exp(C) − I) Uᵀ     (UᵀU = I; exp(C) is 4×4)
Q_e = F_srcᵀ F_tgt / c_e² ∈ SO(d)
```

## Z-10 Header: extend the 1.25 change line of Z-1
**Append to the change line drafted in Z-1:**
> (Z2) Restriction maps: transports stay in SO(d) (not O(d), diagonal or general; reasons in the decisions log); the instance part changes from two Householder reflections per incidence to a rank-4 rotation in a learned per-relation subspace, `F_{tgt◁e} = c_e R_r exp(U_r C_e U_rᵀ)`, `F_{src◁e} = c_e I` (one frame per relation; the second frame was a gauge), m = 4 fixed; Thm 4 and L4′ restated; S2 and `direction_blind_maps` re-expressed; a type-only read-out is added to the mechanism analyses. Reason: the Householder pair is scale-invariant and degenerate at the identity; exp is the identity's own chart (Jacobian I at C = 0). Accepted without a pre-implementation diagnostic (author).

## Z-11 §3 pipeline block
**In the line beginning** `F_{v◁e}, a_e, β_f = w_f m_f` **replace** `# one sheaf: orthogonal factorized maps, fact weights, cell weights` **with** `# one sheaf: SO(d) maps (relation frame × rank-4 instance rotation), fact weights, cell weights`.

## Z-12 §3 component table, maps row
**Replace the row beginning** `| Factorized orthogonal fact maps, one weight per fact (D1, D1a) |` **with:**
> | SO(d) fact maps: relation frame × instance rotation, one weight per fact (D1a; 1.25 TD-1) | R2, R7 | `F_{src◁e} = c_e I`, `F_{tgt◁e} = c_e R_r exp(U_r C_e U_rᵀ)`; `R_r` one FastHPP frame per relation; `U_r ∈ ℝ^{d×4}` orthonormal per relation; `C_e ∈ so(4)` linear in one per-edge MLP over `[h_src, h_tgt, rel]` (zero-initialised output), which also gives `c_e`; relations with `ρ(r) = r`: no frame, `C_e` antisymmetric in the endpoints, `F = c_e exp(∓U_r C_e U_rᵀ/2)`; `a_e = c_e²`; 17 numbers per fact | Transports `Q_e ∈ SO(d)` (L4); instance part a rank-4 rotation, exactly `I` at `C_e = 0` (L4′) |

## Z-13 §3 "Direction" bullet
**Replace** `separate frames per role (`R_{ψ,src} ≠ R_{ψ,tgt}`), a role code in the reflection input, adjoint transport on backward hops` **with** `one frame per relation applied on the target role, so `Q_e = R_r exp(U_r C_e U_rᵀ)` differs from its reverse `Q_eᵀ`; the instance MLP reads the endpoints in stored order; adjoint transport on backward hops`.
**Replace** `Ablation: `direction_blind_maps`.` **with** `Ablation: `direction_blind_maps` (1.25: every relation is treated as one with `ρ(r) = r`: no frame, `C_e` antisymmetric in the endpoints).`

## Z-14 §3 "Heterogeneity" bullet (after Z-5)
**Replace** `relation-and-role frames` **with** `relation frames (one per relation)`.

## Z-15 §4 Thm 4 row
**Replace the row beginning** `| Thm 4 (class-separating sections under heterophily) |` **with:**
> | Thm 4 (class-separating sections under heterophily; 1.25, TD-1 maps) | For any labelling `y` with `C` classes there are distinct unit vectors `a_1..a_C` and TD-1 maps of any rank `m ≥ 2` with `F_{u◁e} a_{y(u)} = F_{v◁e} a_{y(v)}` on every fact; then `x*_v = a_{y(v)}` is a global section, hence a fixed point of the solve; under normalisation the section is `D_v^{1/2} a_{y(v)}`, separable by homogeneous linear classifiers (`argmax_c ⟨a_c, ·⟩`). Construction: a plane `P`, `a_c` at angle `θ_c = 2πc/C` in `P` (points on a circle are in convex position), `U_r ⊇ P`, `R_r = I`, `C_e` the rotation of `P` by `θ_{y(src)} − θ_{y(tgt)}` (antisymmetric in the endpoints, as `ρ(r) = r` requires). For given `a_1..a_C` the same holds when `m ≥ dim span(a_1..a_C)`. Weaker than 1.24, which held for any given `a_1..a_C`. Existence only: whether the map MLP learns these maps is not claimed | DERIVED (sketch, design session 2026-11-18); written proof and numerical check on the TD-1 operator to be redone (as the 1.24 checks: transports, solver, dense maps) |

## Z-16 §4 limits, L4′
**Replace** `**L4′** `k = 2` reflections do not cover O(d); Thm 4 needs only a map taking one vector to another per incidence, which `k = 2` provides (a plane rotation), so the theorem holds for the production family.` **with:**
> **L4′** (1.25) Transports lie in SO(d), not O(d): maps from one continuous family give `det Q = +1`, a det −1 transport would need a discrete per-edge choice, and for `d ≥ 2` SO(d) takes any unit vector to any other. The instance part is a rank-4 rotation inside the relation's subspace `U_r`: 6 numbers per fact, not a free rotation plane in ℝᵈ. Thm 4 holds with its 1.25 construction.

## Z-17 §5 C15 row
**Replace** `` `full` vs S2 (maps = I, separately tuned) `` **with** `` `full` vs S2 (maps = I: `R_r = I` and `C_e ≡ 0`, scale `c_e` kept; separately tuned) ``.

## Z-18 §7 OPEN-5
**Replace** `` `proposal`, `path_weight_input`, solver tolerance, `T_max` and `τ`'s initialisation are never in a search space `` **with** `` `proposal`, `path_weight_input`, solver tolerance, `T_max`, `τ`'s initialisation and the map rank `m` (= 4, 1.25) are never in a search space ``.

## Z-19 Pre-registration §4.3 (mechanism analyses)
**Insert after the N2b bullet:**
> - **N2c type-only read-out (1.25).** On `full`'s checkpoints (configuration (a), seeds 1–10, the four ablation datasets), evaluate once more with `C_e ≡ 0` (relation frames only; no training, same evaluation draw); report the change in test Macro-F1 per dataset with its paired spread. Reading fixed now: a drop means the instance rotations carry information beyond the relation frames; no drop means the maps act through their relation frames only, and the "identity of intermediates in the geometry" sentence (§8 positioning) is weakened to the cells' weights. No claim, no kill rule.

---

## Not spec text: what Z-10 – Z-19 touch (for the card)

- **Code:** `src/models/sheaf_maps.py` (`FactorizedNodeToEdgeSheaf` → TD-1 builder: per-relation frame and `U_r`, one per-edge MLP giving `C_e` and `c_e`, antisymmetrisation for `ρ(r) = r`); `src/models/map_operators.py` (new operator: src side scale only, tgt side `U`-projection, 4×4 rotation, `U`-lift, frame GEMM; `apply_T`, inverses, `select`, `stored_elements`); `PathTransportOperator` unchanged in interface; `identity_maps` (S2) and `direction_blind_maps` re-expressed; `householder_k` removed from config (refused with a message naming 1.25 Z2); new config key for the type-only read-out (evaluation only, refused in tuning).
- **Tests:** orthogonality and det +1 of every transport; `C_e = 0` gives `Q_e = R_r`; antisymmetry for `ρ(r) = r` (stored orientation does not matter); closed form `I + U(exp C − I)Uᵀ` vs `torch.matrix_exp` on the d×d matrix; adjoint and inverse identities; Thm 4 construction checked numerically on the operator; gradient check through the 4×4 exp; mutation checks. Golden and mb0 fingerprints change by design (old/new hashes in the report).
- **Reruns before the freeze (bundled with Z-1–Z-9):** card 4 known-answer (S1′-B, S3, S2′), the card 2 central-point timings (B2a's `N`), `scripts/v3b_cost_smoke.py` cost table (predicted: solve ≈ 0.5×, build ≈ 0.2×, stored geometry 257 → 17 numbers per edge).
- **Paper:** Eq. (maps) and the sentence after it; Thm 4 statement and its appendix proof; L4′ wording ("SO(d) transports"); variants paragraph (identity maps, direction-blind maps).

---

## Z-20 (found while applying; same decision as Z-1) two stale `c` references
- §4 L5: **replace** `degrees, relational context and normalisation are the scope's` **with** `degrees and normalisation are the scope's`.
- §6 WISP settings: **replace** the sentence beginning `Context: C2 runs with `geometry_context: true`` and the following "Appendix pair" sentence **with** `Context: maps and fact weights read `h₀` only (1.25 Z1) and `g` reads `h₀` only (1.4, R12) in every arm, so `endpoints` sees no intermediate through `g`.`

**Applied 2026-11-18** (author approval of Z-1–Z-19 the same day): spec v1.24 → v1.25 (backup `docs/spec_backups/H2S-NN_methodology_spec_v1.24.md`); pre-registration edits Z-9 and Z-19 applied to `docs/ablation_analysis_preregistration_draft.md`. Card-5 items move to 1.26.

---

# Part Z (continued 2): the head reads `x*` only (Z-21 – Z-34) — APPROVED by the author and APPLIED (2026-11-18, spec v1.25)

**Decision:** `docs/experiment_settings_decisions.md`, row of 2026-11-18 "Readout: head reads `x*` only" (author). Applies on top of spec v1.25 as committed (`8acd764`); same version 1.25 (no implementation has started).

## Z-21 Header: extend the 1.25 change line
**Insert before** ` The 1.25 items planned with card 5` **:**
> (Z3) The head reads `x*` only (was `[x*, h₀]`): the anchor keeps node locality inside the solve and Thm 4 makes a class-separating section linearly separable on `x*`; R11 withdrawn, R9′ and Prop. 11 restated; the 1.24 readout becomes the arm `ego_head` (claim C18).

## Z-22 §2 R9′ row
**Replace the row beginning** `| R9′ | Propagation gives, per stalk coordinate,` **with:**
> | R9′ | Propagation is the resolvent `x* = (I+τŜ)⁻¹h₀`: a low-pass response in the sheaf spectrum, whose strength the one learned `τ` sets against the anchor (1.25: no high-pass is formed at the head) | Heterophily is carried by the transports: a sheaf-smooth signal can be anti-aligned across classes (Thm 4), so separation is available at λ = 0 without a high-pass; band-pass is **not** required and is a designated extension (L13) |

## Z-23 §2 R11 row and the withdrawn line
**Delete the row beginning** `| R11 | A node's own features reach the nonlinear head separately from propagated ones |`.
**Replace** `Withdrawn in 1.11: **R4** (asymmetric influence) → limitation L1′; **R9** (band-selective filter bank) → R9′.` **with** `Withdrawn in 1.11: **R4** (asymmetric influence) → limitation L1′; **R9** (band-selective filter bank) → R9′. Withdrawn in 1.25: **R11** (ego/neighbour separation at the head) → the head reads `x*` only; the ego input is the arm `ego_head` (C18).`

## Z-24 §3 pipeline block
**Replace** `y   = head([x*, h₀])                                            # MLP` **with** `y   = head(x*)                                                  # MLP (1.25: x* only)`.

## Z-25 §3 component table, readout row
**Replace the row beginning** `| Readout `[x*, h₀]` | R11, R9′ |` **with:**
> | Readout `x*` (1.25) | R13, R9′ | The anchor keeps each node's own encoding in `x*`; a class-separating section is linearly separable on `x*` (Thm 4) | — |

## Z-26 §3 task readouts
**In the paragraph beginning** `**Task readouts and losses (1.15).**` **replace** `NC: `[x*, h₀]` → Linear(2d, 128)` **with** `NC: `x*` → Linear(d, 128)`; **replace** `LP: per endpoint Linear(2d, d) → GELU` **with** `LP: per endpoint Linear(d, d) → GELU`; **replace** `GC: `[x*, h₀]` of every node → pooling` **with** `GC: `x*` of every node → pooling`.

## Z-27 §3 "Heterophily" bullet
**Replace** `is handled at three levels, each with a proof of what it can express: rotations in the maps (nodes of different classes agree after transport without being equal, Thm 4), instance-conditioned trust in cells (`β_f`, R12), and ego separation plus the complementary high-pass at the head (R11, Prop. 11).` **with** `is handled at two levels, each with a proof of what it can express: rotations in the maps (nodes of different classes agree after transport without being equal, Thm 4) and instance-conditioned trust in cells (`β_f`, R12); the anchor keeps each node's own encoding in the solve (R13). Since 1.25 the head reads `x*` only; ego separation at the head is the arm `ego_head` (C18).`

## Z-28 §4 Prop. 11 row
**Replace the row beginning** `| **Prop. 11 (low- and high-pass at the head)** |` **with:**
> | **Prop. 11 (resolvent response; 1.25)** | `x* = (I+τŜ)⁻¹h₀` and `h₀ − x* = τŜx*` (the transported disagreement). With the head reading `x*` only (1.25), the output is low-pass in Ŝ's spectrum; the arm `ego_head` adds `h₀`, whose first linear layer then spans `x*` and `τŜx*`. The per-coordinate frequency reading is exact only when the head's weights commute with Ŝ; the identity `h₀ − x* = τŜx*` holds always. Band-pass needs a second `τ` (L13) | DERIVED |

## Z-29 §5 new claim C18 (insert after the C17 row)
> | **C18 Ego input at the head is not needed** (1.25) | `full` (head reads `x*`) vs `ego_head` (head reads `[x*, h₀]`, the 1.24 readout), configuration (a), seeds 1–10 | `ego_head` better on ≥ 3 of 4 ablation datasets (Wilcoxon, Holm m = 4) | The paper states that the ego input improves accuracy and that the 1.25 readout costs accuracy there; the main table is not changed (tuned on `full`) |

## Z-30 §6 ablation table bullet
**Replace** `exactly the comparisons of §5 (C2, C2b, C10–C17)` **with** `exactly the comparisons of §5 (C2, C2b, C10–C18)`.

## Z-31 §6 v3-B arms bullet
**Replace** `` `direction_blind_maps`; `h₀`-only; `` **with** `` `direction_blind_maps`; `h₀`-only; (1.25) `ego_head` (the head reads `[x*, h₀]`); ``.

## Z-32 Pre-registration (`docs/ablation_analysis_preregistration_draft.md`)
- Claim map: **insert after** the row `| Propagation helps at all (sanity) | `h₀`-only | 3 |` the row `| The head needs no ego input (1.25) | C18 (`ego_head`) | 3 |`.
- Arms table: **insert after** the `h₀`-only row the row `| `ego_head` (1.25) | head reads `[x*, h₀]` (the 1.24 readout); everything else as `full` | **new** (switch) | (a) |`.
- Arms table: **replace** in the `direction_blind_maps` row `the two roles of a relation share one frame and a symmetrised scale (`sheaf_maps.py:128`)` **with** `every relation is treated as one with `ρ(r) = r`: no frame, antisymmetric `C_e`, symmetrised scale (1.25)`.
- Decision rules: **insert after** the C17 row the row `| **C18 ego input (1.25)** | `full` vs `ego_head` (a) | `ego_head` better on ≥ 3 of 4 | The paper states that the ego input improves accuracy; main table unchanged |`.
- Budget: one more arm × 4 ablation datasets × seeds 1–10 at configuration (a) (no separate tuning).

## Z-33 Not spec text (card)
Model-change card unit **M2b**: head inputs `x*` only (NC `Linear(d, 128)`, LP `Linear(d, d)`, GC pooling over `x*`); `instance_features` keeps its concatenation to the head input (`[x*, instance feature]`); new switch `ablation.ego_head: bool = False` (refused in a main-table tuning base config; an arm of configuration (a)); `h0_only_readout` unchanged; tests: head input widths, `ego_head` reproduces the 1.24 head input, golden list gains `ego_head`.

## Z-34 (found while applying; same decision) GC readout in §6
§6 graph-classification bullet: **replace** `Readout: `[x*, h₀]` pooled over the graph's nodes` **with** `Readout: `x*` pooled over the graph's nodes (1.25)`.

**Applied 2026-11-18** (author approval of Z-21–Z-33 the same day): spec v1.25 now includes Z1–Z3 (change line updated); pre-registration Z-32 applied.

---

# Part Z (continued 3): bounded cell weights (Z-35 – Z-38) — DRAFT, awaiting approval with the card review

**Decision:** `docs/experiment_settings_decisions.md`, row of 2026-11-18 "Model change: `R_r` init, H2GB timing, bounded `g`" (author, option c1).

## Z-35 Header: extend the 1.25 change line
**Insert before** ` The 1.25 items planned with card 5` **:**
> (Z4) Bounded cell weights: `log w_f = L·tanh(g(ξ_f)/L)`, `L = 10` fixed, so the log-weight range is at most 2L = 20 < 30 by construction (D6b: the range grew with optimizer steps past 30 at lr 1e-3 and 1e-4 on nci1); the stop stays as a guard and cannot fire.

## Z-36 §3 component table, WISP instance weight row
**Replace the row beginning** `| **WISP instance weight** `w_f = exp(g(ξ_f))` |` **with:**
> | **WISP instance weight** `w_f = exp(L·tanh(g(ξ_f)/L))`, `L = 10` (1.25) | R12 | Positive per-cell weights within a ratio of `e^{2L}` ≈ 5·10⁸ (no practical ceiling; node–cell term); contains uniform, schema-only and endpoint weights as restrictions (C2 ladder) | One encoder per path length; log-weight range logged; the stop at 30 is unreachable under the bound |

## Z-37 §3.2 (c) Weights
**Replace** `**(c) Weights.** `w_f = exp(g(ξ_f))`;` **with** `**(c) Weights.** `w_f = exp(L·tanh(g(ξ_f)/L))`, `L = 10` fixed and never tuned (1.25; identical to `exp(g)` up to 0.2 % at initialisation, where `g = b = log 0.5`);`.

## Z-38 §6 failed trials and OPEN-5
- **Replace** `(`T_max`, `τ > 50`, log-weight range > 30, non-finite `g`)` **with** `(`T_max`, `τ > 50`, log-weight range > 30 (unreachable since 1.25), non-finite `g`)`.
- OPEN-5: **replace** `τ`'s initialisation and the map rank `m` (= 4, 1.25) are never in a search space` **with** `τ`'s initialisation, the map rank `m` (= 4, 1.25) and the weight bound `L` (= 10, 1.25) are never in a search space`.
