# Thm 4 (spec v1.25): class-separating sections with TD-1 maps — proof

Design session (model), 2026-11-18. Spec row: §4 Thm 4 (1.25, Z-15). Numerical check: `docs/thm4_td1_evidence/check_thm4_td1.py` and its output `check_thm4_td1.out` (ALL_CHECKS_PASS).

## Setting

- **Graph.** A graph with facts `e = (u → v)` of relation `r`, and a labelling `y : V → {1, …, C}`, `C ≥ 2`.
- **Maps.** The TD-1 orthogonal factors of spec 1.25 are
  - `ρ(r) ≠ r`: `F_{u◁e} = I`, `F_{v◁e} = R_r exp(A_e)`;
  - `ρ(r) = r`: `F_{u◁e} = exp(−A_e/2)`, `F_{v◁e} = exp(A_e/2)`, with `C_e(u, v) = −C_e(v, u)`;
  - where `A_e = U_r C_e U_rᵀ`, `U_rᵀU_r = I_m`, `C_e ∈ so(m)` and `R_r ∈ SO(d)`.
- **Scales.** The positive scales `c_e` are common to both endpoints of a fact and enter the energy as `a_e = c_e²`, so they do not change which states are sections.
- **Faces.** A face `f` is a simple walk `v_0, e_1, v_1, …, e_ℓ, v_ℓ`. Its forced maps (Thm 1) are
  - `Q_j = F_{v_{j−1}◁e_j}ᵀ F_{v_j◁e_j}`;
  - `T_{v_0} = I`, `T_{v_j} = T_{v_{j−1}} Q_j`;
  - `R_{e_j,f} = T_{v_{j−1}} F_{v_{j−1}◁e_j}ᵀ`.
- **Energy.** Eq. (E) is
  ```
  E(Y) = ½ Σ_e a_e Σ_{v∈e} ‖F_{v◁e}x_v − s_e‖² + ½ Σ_f β_f [ Σ_{e∈f} ‖R_{e,f}s_e − z_f‖² + Σ_{v∈f} ‖T_{v,f}x_v − z_f‖² ],
  ```
  with `a_e, β_f > 0`.
- **Degrees.** `D_v = Σ a_e + Σ β_f`, `D_e = 2a_e + Σ β_f`, `D_f = β_f(2ℓ+1)` (scalar, because the factors are orthogonal).
- **Normalised operator.** `L̂ = D^{−1/2} L D^{−1/2}`.
- **Solve.** `(PᵀP + τL̂) Y = Pᵀh₀`.

## Statement

For any labelling `y` with `C ≥ 2` classes there are distinct unit vectors `a_1, …, a_C` and TD-1 maps of any rank `m ≥ 2` such that `F_{u◁e} a_{y(u)} = F_{v◁e} a_{y(v)}` on every fact. Consequently:

- **(i)** `x_v = a_{y(v)}` extends to a global section, so `E = 0` and `LY = 0`.
- **(ii)** With `h₀_v = D_v^{1/2} a_{y(v)}`, the anchored solve returns `x*_v = D_v^{1/2} a_{y(v)}`.
- **(iii)** `argmax_c ⟨a_c, x*_v⟩ = y(v)` for every node, so the classes are separated by a homogeneous linear classifier.

**Corollary.** For given distinct unit vectors `a_1, …, a_C` the same holds whenever `m ≥ max(2, dim span(a_1, …, a_C))`.

## Lemmas

**Lemma 1 (low-rank exponential).** If `UᵀU = I_m` and `C ∈ so(m)`, then `exp(UCUᵀ) = I + U(exp(C) − I)Uᵀ`. It acts as `U exp(C) Uᵀ` on `range(U)` and as the identity on `range(U)^⊥`.
*Proof.* `(UCUᵀ)^k = U C^k Uᵀ` for `k ≥ 1`, because `UᵀU = I`. Summing the series gives the result. The identity holds on `range(U)^⊥` because `Uᵀw = 0` there. ∎

**Lemma 2 (circle).** Let `p_1, p_2` be orthonormal and `a_c = cos θ_c p_1 + sin θ_c p_2` with `θ_c = 2π(c−1)/C`. Then `⟨a_c, a_{c'}⟩ = cos(θ_c − θ_{c'})`. This equals 1 if and only if `c = c'` and is strictly smaller otherwise. Hence `argmax_c ⟨a_c, λ a_y⟩ = y` for every `λ > 0`. More generally, for any distinct unit vectors, `⟨a_c, a_y⟩ < 1 = ⟨a_y, a_y⟩` for `c ≠ y` (equality in Cauchy–Schwarz forces `a_c = a_y`). ∎

**Lemma 3 (sections extend to faces).** If `F_{u◁e} x_u = F_{v◁e} x_v` on every fact and every factor is orthogonal, define
- `s_e = F_{u◁e} x_u`;
- `z_f = x_{v_0}` for each face.

Then every term of (E) vanishes.

*Proof.*
- **Node–fact terms.** Zero by the definition of `s_e`.
- **Transports.** For hop `j` of a face, `Q_j x_{v_j} = F_{v_{j−1}◁e_j}ᵀ F_{v_j◁e_j} x_{v_j} = F_{v_{j−1}◁e_j}ᵀ F_{v_{j−1}◁e_j} x_{v_{j−1}} = x_{v_{j−1}}` by orthogonality. By induction `T_{v_j} x_{v_j} = T_{v_{j−1}} x_{v_{j−1}} = … = x_{v_0} = z_f`.
- **Node–cell terms.** Zero by the transport identity above.
- **Fact–cell terms.** `R_{e_j,f} s_{e_j} = T_{v_{j−1}} F_{v_{j−1}◁e_j}ᵀ F_{v_{j−1}◁e_j} x_{v_{j−1}} = T_{v_{j−1}} x_{v_{j−1}} = z_f`, so they are zero too.

Because `E(Y) = ½ YᵀLY` with `L ⪰ 0`, `E = 0` implies `LY = 0`. ∎

## Proof of the statement

**Construction.**
- Pick orthonormal `p_1, p_2 ∈ ℝᵈ` and the `a_c` of Lemma 2.
- For every relation take `R_r = I`, and `U_r` orthonormal with `U_r e_1 = p_1`, `U_r e_2 = p_2` (any `m ≥ 2`, any completion).
- Let `J ∈ so(m)` be the generator with `J e_1 = e_2`, `J e_2 = −e_1`, zero elsewhere, and set `C_e = (θ_{y(u)} − θ_{y(v)}) J` for the stored fact `u → v`.

By Lemma 1, `exp(φ U_r J U_rᵀ)` rotates the plane `P = span(p_1, p_2)` by `φ` and fixes `P^⊥`. So it maps `a_c` to the vector at angle `θ_c + φ`.

**Facts with `ρ(r) ≠ r`.**
- `F_{u◁e} a_{y(u)} = a_{y(u)}`.
- `F_{v◁e} a_{y(v)} = exp((θ_{y(u)} − θ_{y(v)}) U_r J U_rᵀ) a_{y(v)}` is the vector at angle `θ_{y(v)} + θ_{y(u)} − θ_{y(v)} = θ_{y(u)}`, which is `a_{y(u)}`.

**Facts with `ρ(r) = r`.**
- **Agreement.** Both `exp(−A_e/2) a_{y(u)}` and `exp(A_e/2) a_{y(v)}` are the vector at angle `(θ_{y(u)} + θ_{y(v)})/2`.
- **Antisymmetry.** `C_e(v, u) = (θ_{y(v)} − θ_{y(u)})J = −C_e(u, v)`, as `ρ(r) = r` requires.
- **Orientation does not matter.** Storing the fact as `v → u` gives `F'_{v◁e} = exp(−(−A_e)/2) = F_{v◁e}` and `F'_{u◁e} = F_{u◁e}`, so the operator is the same.

**Agreement and (i).** So `F_{u◁e} a_{y(u)} = F_{v◁e} a_{y(v)}` holds on every fact. Every transport `R_r exp(A_e)` is in `SO(d)`, because the exponential of a skew matrix has determinant `e^{tr A} = 1`. Lemma 3 then gives (i).

**(ii) Normalised solve.**
- Let `Ŷ = D^{1/2} Y`. Then `L̂ Ŷ = D^{−1/2} L Y = 0`.
- If `h₀ = PŶ` (i.e. `h₀_v = D_v^{1/2} a_{y(v)}`), then `(PᵀP + τL̂) Ŷ = PᵀPŶ = Pᵀh₀`, so `Ŷ` solves the anchored system.
- The solution is unique because `PᵀP + τL̂ ≻ 0` (Thm 5). Hence `x* = PŶ = h₀`.

**(iii) Separation.** `x*_v = λ_v a_{y(v)}` with `λ_v = D_v^{1/2} > 0`. Lemma 2 gives (iii) with weight rows `a_1, …, a_C`.

Eliminating fact and face states (Prop. 7/7′) is exact, so it does not change `x*`. ∎

**Proof of the corollary.**
- **Subspace.** Let `k = dim span(a_c)`, `m ≥ max(2, k)`, and choose an orthonormal `U` whose range contains `span(a_c)`. Write `b_c = Uᵀa_c`; these are unit vectors in `ℝᵐ`.
- **Generator for a non-antipodal pair (`b_s ≠ −b_t`).**
  - `w_1 = b_t`, `w_2 = (b_s − ⟨b_s, b_t⟩b_t)/‖·‖`, `φ = atan2(⟨b_s, w_2⟩, ⟨b_s, w_1⟩)`.
  - `B_{st} = φ(w_2w_1ᵀ − w_1w_2ᵀ)` (zero if `b_s = b_t`). Then `exp(B_{st}) b_t = b_s`.
  - Exchanging `s` and `t` gives the same plane with the opposite orientation and the same angle, so `B_{ts} = −B_{st}`.
- **Generator for an antipodal pair.** Use a rotation by `π` in a fixed plane containing `b_t`, chosen per unordered pair, with `B_{ts} = −B_{st}`.
- **Maps.** Set `C_e = B_{y(u)y(v)}` and `R_r = I`. Then:
  - `ρ(r) ≠ r`: `F_{v◁e} a_{y(v)} = U exp(B) b_{y(v)} = a_{y(u)}`, using Lemma 1 and `a = UUᵀa` on `range(U)`.
  - `ρ(r) = r`: `exp(−A/2) a_{y(u)} = exp(A/2) a_{y(v)}`, which is equivalent to `exp(A) a_{y(v)} = a_{y(u)}`.
- **Conclusion.** The rest is as above, with the general part of Lemma 2. ∎

## Scope and remarks

- **Weaker than 1.24.** In 1.24 a free per-incidence plane rotation (two reflections) allowed any given `a_1..a_C` in `ℝᵈ`. In 1.25 the vectors are either chosen (the circle construction, any `C`, any `m ≥ 2`) or must span at most `m = 4` dimensions. Whether TD-1 maps with nontrivial `R_r` reach given vectors spanning more than `m` dimensions is not claimed.
- **`C` larger than `d`.** The circle construction does not depend on `C`: ogbn-mag (349 classes, `d ≤ 64`) is covered. The class margin is `1 − cos(2π/C)`, which is 1.6·10⁻⁴ at `C = 349`. This is an existence statement, not a robustness one.
- **Existence only.** As in 1.24, `C_e` here is a value per fact. Whether the MLP over `[h_src, h_tgt, rel]` produces these values (it cannot when two nodes with equal `h₀` have different labels) is not claimed.

## Numerical check (`check_thm4_td1.out`, float64, CPU)

**Setup.** The dense `L` is assembled term by term from (E). The toy graph has 42 nodes, 97 facts and 40 faces of length 2–4; relations include one with `ρ(r) = r`; random heterophilic labels (about 80 % of facts join different classes); random `a_e, β_f`; `τ = 7`; `d = 8`, `m = 4`.

| Case | Result |
|---|---|
| Construction, `C = 5` | `E` at section 6.2e-15; max \|LY\| 1.5e-14; degree blocks scalar to 1.2e-14 and equal to the spec formulas to 3.6e-15; anchored matrix min eigenvalue 0.36; max \|x* − h₀\| 1.8e-15; argmax accuracy 1.0; transports orthogonal to 7e-15, det 1 |
| Stored orientation of `ρ(r) = r` facts flipped | max \|L − L′\| = 0 |
| Corollary, 6 given random unit vectors spanning 4 dims | `E` 4.2e-15; \|x* − h₀\| 2.2e-15; accuracy 1.0; `‖B_{st} + B_{ts}‖` 1.1e-15 |
| Negative control: one fact rotated by 0.3 rad | `E` = 0.0136, \|x* − h₀\| = 0.016 (detected) |
| Negative control: identity maps (graph diffusion), same labels | `E` = 179.4; argmax accuracy 0.57 |
| Production size `d = 64`, `m = 4`, `C = 349` | residuals 2.0e-11 (distinct roles), 1.3e-12 (`ρ(r) = r`); scaled argmax 1.0; closed form vs `matrix_exp` 2.9e-14; `R_r exp(·)` orthogonal to 2e-15, min det 1 − 7e-15 |

The check on the production operator (`src/`, transports and solver) is part of the implementation card.
