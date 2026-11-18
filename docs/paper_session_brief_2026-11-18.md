# Paper session brief — bring the IJCAI-27 text to model 1.25 (2026-11-18)

Written by the model session for a new **paper session**. Read this first, then the sources in §2. The paper file is `H2S-NN/paper/ijcai27/h2snn_ijcai27_v3b.tex`; its text still describes model 1.24.

## 1. Rules (author)
- **Text goes to chat first.** Show proposed LaTeX in a code block with short notes. Edit the `.tex` only when the author says so (e.g. "แก้ลงไฟล์เลย"). Compiling to check is fine.
- **Build.** TinyTeX at `~/.TinyTeX/bin/x86_64-linux`; outDir `H2S-NN/paper/build` (gitignored). On "Extra }, or forgotten \endgroup" or undefined refs, run `latexmk -C -outdir=<build>` and rebuild.
- **Wording.**
  - The structure is an **incidence poset** (nodes < edges < faces), never a "2-cell" or CW complex.
  - Use "heterophily" wording only.
  - One word per idea, as in `docs/paper_writing_brief_v3B.md` §4: edge, face, path instance, restriction map, stalk, state.
- **Scope.**
  - Change only the parts the author names.
  - Do not trim for page count piecemeal. The 11-page overflow is cut later in one pass.
  - Do not redraw Fig. 1/2 until the content pass is finished and the author asks.
- **Replies** to the author are in Thai; the paper is in English.
- **Git (shared worktree `/mnt/data/peeradon/h2snn-design`, branch `design-docs`, used by the model and ops sessions too).**
  - Stage only your own files by name; never `git commit -a` or `git add .`.
  - Commit and push right after an edit the author approved.
  - Never rebase, reset, stash or check out another branch.
  - If `index.lock` exists, wait and retry.
  - Files you did not edit belong to someone else; leave them.
- **What the paper session owns.** The paper session owns the `.tex`, `references.bib` and `figure/`. It never edits the spec, cards, pre-registrations or the decisions log. Questions about the method go to the model session (through the author).

## 2. Sources of truth (in this order)
1. `docs/H2S-NN_methodology_spec.md` **v1.25**, as follows:
   - the header change line (Z1–Z4), the §3 component table and §3.2;
   - theorem rows Thm 4 and Prop. 11;
   - requirement rows R9′, R10 and R12, and the "Withdrawn in 1.25" line (R11);
   - claims C17 and C18.
2. `docs/experiment_settings_decisions.md`, the rows of 2026-11-18. These give the reasons for every change: c removed, SO(d) kept, TD-1, m = 4, head reads `x*`, `free_cell_maps`, solver unchanged, bounded `g`, the S6 review, and weight decay considered but not adopted.
3. `docs/thm4_td1_proof.md`: the proof behind the heterophily sentence. It is checked numerically in `docs/thm4_td1_evidence/`.
4. `docs/H2S-NN_model_change_td1_card.md` §0: the exact equations.
5. `docs/H2S-NN_model_change_td1_report.md` on branch `model-td1` (`git show origin/model-td1:docs/H2S-NN_model_change_td1_report.md`): the measured cost and tests.
6. `docs/ablation_analysis_preregistration_draft.md`: the C18 `ego_head` rows, and N2c, the type-only read-out in §4.3.
7. `docs/paper_writing_brief_v3B.md`: style, story and terminology. Its model parts (v1.11) are superseded by 1–6.

## 3. The model now (1.25), in one place
```
h_v      = per-type encoder (unchanged)
geometry reads h only: no relational context c (no neighbour means)
fact e = (u -> v), relation r:
  F_{u<e} = c_e I,   F_{v<e} = c_e R_r exp(U_r C_e U_r^T)          (roles differ)
  F_{u<e} = c_e exp(-U_r C_e U_r^T / 2), F_{v<e} = c_e exp(+... / 2)  (symmetric relation, C antisymmetric in u,v)
  R_r in SO(d): one learned frame per relation with two distinct roles; NO frame for a relation that is its own reverse (rho(r) = r)
  U_r: d x 4, orthonormal (QR) per relation
  C_e in so(4) (6 numbers) and c_e from ONE per-edge MLP over [h_u, h_v, rel_emb]; C_e = 0 at init
  transports Q_e = F_u^T F_v / c_e^2 in SO(d)
faces: forced maps (unchanged, Prop. forced); weight beta_f = w_f m_f with
  w_f = exp(L tanh(g(xi_f)/L)), L = 10 fixed          (was exp(g))
solve: anchored, (P^T P + tau L^) Y = P^T h; x* = (I + tau S^)^{-1} h  (unchanged; tau one learned scalar)
head:  reads x* only (was [x* || h]);  [x*, h] is the variant "ego head" (C18)
```

## 4. What in the current `.tex` is now wrong (line numbers of `d24fe65`)
| Line | Now says | Change to (source) |
|---|---|---|
| 142 (Fig. 2 caption) | "an MLP head turns into a prediction" | Still true; check that nothing says the head also reads `h`. |
| 145 | "a prediction head reads them together with the encodings" | The head reads the node representations only (spec Z3, R11 withdrawn). |
| 149 | `c_v` = mean of neighbours per relation/direction | **Delete** the `c_v` sentence (Z1). Reason, if a sentence is wanted: graph structure enters the representation only through the solve. |
| 155–159 | Eq. maps: `u = MLP([h_v, c_v, h_{v'}, c_{v'}, ζ_r, ρ])`, `R_{r,ρ} H_1 H_2`, Householder reflections | TD-1 maps (§3 above; spec §3 maps row). One frame per relation, since a second frame is a gauge. Instance part `exp(U_r C_e U_rᵀ)` with rank 4. Say "SO(d)", not "orthogonal". The pair-conditioning sentence stays: the MLP reads both endpoints. Edge weight `a_e = c_e²` reads the two endpoint encodings and the relation, with no `c`. |
| 190 | `w_f = \exp g(\xi_f)` | `w_f = \exp(L\tanh(g(\xi_f)/L))`, `L=10` (Z4). One clause on why: it keeps the weight range finite during long training. |
| 206 | `\hat y_v = MLP([x^\star_v \concat h_v])` with the `zhu2020beyond` ego-separation reason | `\hat y_v = MLP(x^\star_v)`. The anchor already keeps each node's own encoding inside the solve, and `h − x^\star = τ\hat S x^\star` (Prop. 11). Drop or move the `zhu2020beyond` reason to the `ego head` variant. |
| 214 | "maps of the form of Eq. admit configurations in which nodes of different classes are consistent … (existence only)" | Still true for TD-1 maps of any rank `m ≥ 2` and any number of classes (Thm 4, `docs/thm4_td1_proof.md`). Keep "existence only". |
| 222 (variants, Maps) | "Direction-blind maps gives the two roles of a relation a common frame"; "No relational context …" | Direction-blind: every relation is treated as symmetric (no frame, antisymmetric `C`; spec §3.1 "Direction" and C16; S2 is the identity-maps arm of C15). **Remove "No relational context"**. **Add "Ego head"**: the head reads `[x^\star, h]` (C18). If maps are discussed in the mechanism paragraph, add the type-only read-out (`C_e = 0` at evaluation, N2c). |
| 355 (ablation table) | row "No relational context" | Remove it and add a row "Ego head" (C18). The row set must match spec §6 and the pre-registration. |
| 402 (mechanism) | transport distance, intermediate-node replacement, schema ranking | Add the type-only maps read-out (N2c) if the author wants it in the text. |
| Appendix (proofs 388–392) | Props. forced / reach / unique | These are unchanged, but check every use of the map form and of `c`. |
| Cost paragraph (Discussion) | — | Report **measured** numbers only. On ACM at d 32: train step 0.55×, backward 0.43×, peak memory 0.77×, map storage per fact 129 → 17 floats. **Do not** claim that one operator application is 2× faster: it measured 1.08× per PCG iteration. |

Also grep the whole `.tex` for `c_v`, `context`, `Householder`, `H_1`, `R_{r,\rho}`, `\concat h`, `\exp g` and "orthogonal matrix" before saying a section is done.

## 5. Short justifications the text may need (from the decisions log)
- **Why SO(d), not O(d), diagonal or general maps.**
  - The model sees only `F_uᵀF_v`. Any continuous family gives det = +1, and for d ≥ 2, SO(d) maps any vector to any other vector of the same norm.
  - Diagonal or general maps break the forced face maps' inverse, scalar degrees and memory at scale.
  - Two sentences at most.
- **Why TD-1.**
  - Householder pairs are scale-invariant and degenerate at the identity; `exp` is a smooth chart at the identity, with Jacobian I at `C = 0`.
  - A large rotation is carried by `R_r`.
  - Rank 4 is fixed and never tuned.
- **Why the head reads `x*` only.** The anchor keeps node locality inside the solve, and a class-separating section is linearly separable on `x*` (Thm 4). The ego input is tested as a variant (C18).
- **Solver.** Unchanged (implicit, anchored, PCG). No new solver discussion is planned.

## 6. State outside the paper
- **Code.** Model 1.25 is implemented on branch `model-td1` (S6). Follow-up M8 is running (docs fixes, merge of `v3b` `87c2e4a`, full suite). After review the author fast-forwards `v3b`.
- **Results.** No main-table result exists yet: tuning runs on TSUBAME after card 5. Every result cell stays `\res{}`.
- **Spec v1.26 / card 5.** Not written yet (B1a, PDNS validation subset, Freebase NC mini-batch). Nothing in it changes the method text above.

**Corrections (model session, 2026-11-18, after the paper session's check):** §3 frame line (no frame for `ρ(r) = r`) and the §4 direction-blind source (§3.1 and C16, not S2) fixed; spec §3.2 ACM example changed to the bounded weight (Z-39, editorial).
