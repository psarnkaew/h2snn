# SheafCovPool — Design & Decision Log (living document)

> **FINALIZED METHOD NAME: `SheafCovPool`** (class `SheafCovPool`, `src/models/graph_pooling.py`,
> canonical `pooling_type: "sheafcov"`; legacy `"sheafcov"` remains accepted). This is the pooling method
> going into the manuscript. Full spec:
> [`sheafcovpool_implementation.md`](sheafcovpool_implementation.md) · committed proposal:
> [`sheafcovpool_proposal.html`](sheafcovpool_proposal.html) · newcomer walkthrough:
> [`sheafcovpool_explainer.md`](sheafcovpool_explainer.md).
>
> **Naming history (context, not current guidance):** this pooling project was originally scoped as a
> **hierarchical** coarsening method named "S²Pool-H2", then renamed **THS-Pool** (Typed Hierarchical Sheaf
> Pooling) on 2026-07-21. That hierarchical/coarsening line was tested exhaustively (relearn, geometry
> injection, energy-preserving Galerkin coarsening, spectral descriptor — §pm8–pm10) and **closed as negative**
> on 2026-07-22 (§0): none of it robustly beat a plain invariant readout on TU graphs. **`SheafCovPool` is a
> different, single-level (non-hierarchical) method that replaced it as the finalized proposal** — do not use
> "THS-Pool" to refer to the current method; it names the abandoned hierarchical exploration, kept below only
> as a record of what was tried and why it didn't work.
>
> **Purpose of this file.** Single source of truth for the pooling project, so decisions, findings, and open
> questions survive context loss. **Always update this file** when a decision is made, an experiment lands, or
> a position changes. Newest changes go in the Changelog (bottom) and are reflected in the relevant section.
>
> **Companion docs:** [`sheafcovpool_proposal.html`](sheafcovpool_proposal.html) (the committed proposal — why/what) ·
> [`sheafcovpool_implementation.md`](sheafcovpool_implementation.md) (mechanism spec — how) ·
> `scratchpad/measure_faces.py` (the κ / faces experiment) ·
> memory `faces-needed-molecular-gc.md`.
>
> **Last updated:** 2026-07-22.

---

## 0. FINAL DECISION (2026-07-22) — TU pooling investigation CLOSED

After the node-Galerkin positive result (pm8) was traced to a numerical-instability artifact and reversed
under a stability fix (pm9), and a spectral-descriptor readout also failed to add signal (pm10), the
hierarchical/coarsening line of the THS-Pool investigation is **closed as negative** on TU-scale graphs: no
tested mechanism (relearn hierarchy, geometry injection, energy-preserving Galerkin coarsening, spectral
heat-kernel descriptor) robustly beat a plain invariant terminal readout (GramPool) on MUTAG/ENZYMES.

Separately, reading the CSNN predecessor paper (pm11) clarified that the model family's own graph-classification
precedent is **Peptides-func/LRGB (long-range)**, not TU — TU graphs are too small/local to exercise the
regime (over-squashing, long-range dependence) where sheaf/cooperative structure has a plausible advantage. This
reframes the whole TU investigation as testing on a benchmark type where a null result is expected regardless of
readout sophistication.

**Adopted final pooling proposal: `SheafCovPool`** (`src/models/graph_pooling.py`, canonical `pooling_type:
"sheafcov"`) — a single-level (non-hierarchical), gauge-invariant readout that improves on GramPool by adding
(a) a learned projection of the existing per-node/per-edge Gram descriptor scored by gauge-invariant attention,
and (b) the **across-node/across-edge covariance** of that projection (not just its mean), capturing the spread
of node/edge states rather than only their average. Unlike SheafPool, it needs no whitening, so it does not
collapse on H2SNN's single-channel stalks (SheafPool: ~0.26 macro-F1 on MUTAG, near/below majority baseline,
confirmed again in the final gate below).

**Final gate (MUTAG + ENZYMES, 3 seeds, 40 epochs, macro-F1, mean ± population std):**

| Dataset | sheafcov (final) | grampool (baseline) | sheafpool (known-broken) |
|---|---|---|---|
| MUTAG | 0.596 ± 0.050 [0.525, 0.635, 0.627] | 0.595 ± 0.081 [0.708, 0.525, 0.552] | 0.260 ± 0.014 |
| ENZYMES | 0.236 ± 0.144 [0.049, 0.400, 0.258] | 0.166 ± 0.088 [0.048, 0.257, 0.194] | 0.048 ± 0.000 |

**Honest read:** MUTAG is an exact tie with GramPool (sheafcov has lower variance, no accuracy gain). ENZYMES
shows a nominal +0.07 over GramPool but both arms are high-variance (3 seeds, small dataset) and the intervals
overlap — this is *consistent with*, not clean proof against, the same "ties on TU" pattern seen throughout this
investigation. SheafPool's collapse is reconfirmed. **Decision:** ship `SheafCovPool` as the paper's
pooling readout — it is at worst tied with GramPool and strictly dominates SheafPool on this backbone — but do
not claim a proven TU performance win; if graph classification stays in the paper's main results, position TU
numbers as a sanity/no-regression check and, if time permits, evaluate on Peptides-func/LRGB where the
architecture's actual hypothesized advantage (long-range, higher-order sheaf structure) can be tested. No further
pooling-mechanism iteration planned — this closes the pooling design thread.

---

## 1. Project context

- **Backbone: H2SNN** (`src/`) — heterophilic **heterogeneous sheaf-Hodge** network. Typed nodes/edges, typed
  meta-path faces (2-cells), learned conformal restriction maps ($R=\lambda O$), Hodge diffusion of a node
  cochain `x*` **and** an explicit edge cochain `s*`. Exposes `x*, s*, δ0x*, δ1s*, F_src/F_tgt`.
- **Task in scope: graph classification on TU benchmarks only** — MUTAG (7 atom types, ~188 graphs),
  PROTEINS (3 SSE types, ~1113), ENZYMES (3 SSE, 6-class, ~600), NCI1 (~37 atom types, ~4110).
- **Problem being solved:** the legacy terminal `S2Pool` readout is provably degenerate
  ($P(X)=X^\top(XX^\top)^\dagger X = I_3$ for full-rank stacks). Need a principled pooling that summarizes —
  and ideally coarsens — the typed node/edge/face cochain structure for graph classification.

---

## 2. Target novelty — "the empty cell" (SUPERSEDED framing — kept as history, see §0)

> This section's novelty target and prior-art table describe the **hierarchical THS-Pool line, which was
> abandoned** (§0). It is kept verbatim as a record of the reasoning at the time. `SheafCovPool`'s own
> positioning (single-level, no whitening, adds attention + across-node covariance to GramPool) is in §0 and
> in `sheafcovpool_implementation.md`.

Prior art coverage:

| Method | Role | Ranks | Sheaf | Typed | Hierarchy | Gauge handling |
|---|---|---|---|---|---|---|
| **SheafPool** (HetSheaf) | terminal readout | node | yes | via backbone | no | general (whiten+anchor+attention) |
| **HiSP** (2026) | hierarchical pool | node | yes | no (homogeneous) | yes | spectral (Galerkin) |
| **HINPool / MVPool** | hierarchical pool | node | no | yes / no | yes | n/a |
| **THS-Pool (ours)** | hierarchical pool | **node+edge+face** | yes | **yes** | **yes** | O(d) energies |

**Uncontested ground = the intersection: typed + hierarchical + higher-order (edge/face) sheaf coarsening.**
Nobody occupies it. This is the defensible novelty **iff** the mechanism genuinely pools the edge/face ranks and
coarsens the sheaf (see §6 lean-vs-sheaf-true).

---

## 2.5 Theoretical footing & reframing (2026-07-21) — READOUT is the bet, hierarchy is conditional

Honest assessment of *why* we'd expect a gain — two parts with very different footing:

- **Higher-order readout — provable information asymmetry (STRONG footing).** In H2SNN, `x` and `s` diffuse
  **independently** (`diffusion.py`) and `x0=h0` is face-independent, so **`x*` contains zero face information**;
  the face/edge signal lives entirely in `s*` / `δ1 s*`. SheafPool and any node-only readout read node stalks
  (`x*`), so **`H2SNN+SheafPool` is provably blind to faces.** Our readout reads `s*` / `δ1 s*` too → on tasks
  where faces carry signal (high κ) we access task-relevant information the node-only baseline *structurally
  cannot*. This is the cellular / beyond-1-WL expressivity story (rings/cycles). **Caveat:** information access ≠
  accuracy gain (could be noise); it should materialize *iff* κ is high.
- **Hierarchy — WEAK footing on TU molecules.** No theorem that hierarchical > global pooling; "Rethinking
  pooling" shows it frequently isn't; the receptive-field argument barely applies on ~18–30-node graphs.

**Reframing (DECIDED):** the **core bet is the higher-order sheaf READOUT** (node+edge+face), not the hierarchy.
The hierarchy and sheaf-true coarsening (§6) are **demoted to conditional extensions**, built only if the readout
test (§8 Gate B) shows the higher-order rank pays off. The plan (§8–§9) is rebuilt readout-first.

We cannot *assume* a performance gain — but the information asymmetry is a real, provable reason to expect one
exactly where κ is high, which is stronger footing than "we hope."

---

## 3. Key empirical findings so far

### 3.1 κ experiment (faces carry *energy*) — DONE
Diagnostic: $\kappa = \dfrac{\|\delta_1 s^\star\|^2}{\|\delta_1 s^\star\|^2 + \|\delta_0^\ast s^\star\|^2}$ =
fraction of edge-Hodge ($L_1$) energy from the face/upper term. Setup: grampool, discovery off (clean typed
2-hop faces), single holdout split.

| Dataset | node types | κ init → trained | reading |
|---|---|---|---|
| MUTAG | 7 atom (rings) | 0.17 → **0.41** ↑ | training **amplifies** faces |
| PROTEINS | 3 SSE | 0.23 → **0.042** ↓ | training **suppresses** faces |
| NCI1 | ~37 atom | *not completed* | expected to track MUTAG |

**Conclusion:** the model *adaptively* routes energy through faces when typed cyclic structure is rich
(molecular rings) and suppresses them otherwise — with no supervision. This reversed an earlier "defer faces"
stance.

### 3.2 Critical caveat — energy ≠ accuracy (UNPROVEN)
κ shows faces are **used** (help fit the objective), **not** that they improve **test accuracy**. The accuracy
evidence we have is weak:
- MUTAG: acc 0.789 (ON) = 0.789 (OFF); AUROC 0.93 vs 0.77 but **single 19-graph split → untrustworthy**.
- PROTEINS: acc 0.688 vs 0.696; AUROC 0.734 vs 0.736 → **effectively null**.

**So "faces improve performance" is NOT established.** The faces→accuracy ablation (rank ladder) is the
load-bearing, unrun experiment. It is a *core experiment*, not "analysis."

### 3.3 Phase-0 readout sweep results (2026-07-21) — CAUTIONARY
MUTAG + ENZYMES, 3 seeds, 80 epochs, no tuning. Test macro-F1 (mean):

| | meanpool (node) | grampool ON (node+edge+face) | grampool OFF | sheafpool |
|---|---|---|---|---|
| MUTAG (κ≈0.07–0.58) | 0.73 | 0.737 | 0.625 | 0.26 (invalid) |
| ENZYMES (κ≈0.34) | 0.285 | 0.298 | 0.229 | 0.05 (invalid) |

1. **Faces help the edge readout (partial win):** grampool ON > OFF on 3/3 ENZYMES seeds, 2/3 MUTAG (+0.07..0.11).
2. **BUT higher-order readout ties node-only:** grampool ON ≈ meanpool on both. The information asymmetry does
   NOT translate to a net win over trivial mean pooling. **Key concern; SheafPool-independent.**
3. **κ-stratification broke:** ENZYMES is κ≈0.34 (moderate), not low — poor low-κ control (PROTEINS κ=0.042 was
   the real one, dropped).
4. **SheafPool arm invalid (diagnosed):** whitening a d×C=32×3 rank-deficient stalk → diag(Zᵀ Z)=(G+εI)⁻¹G ≈ I
   (near-constant energies) + unstable `eigh` gradients (29 degenerate ε-eigenvalues). SheafPool needs many
   feature channels; H2SNN's single-channel stalk (faked into 3 views) doesn't provide them. Gate A unanswered.

Caveats: 3 seeds, high variance, tiny/noisy MUTAG test, no tuning, crude Gram readout, no low-κ control.
**Decision (2026-07-21):** proceed to method implementation despite the weak readout signal — the method's value
is now an empirical question, with the burden on the **selection/coarsening** (the readout alone tied). Readout =
in-house trace-norm Gram (SheafPool does not fit H2SNN's single-channel stalk).

---

## 4. Invariance audit (our method)

| Property | THS-Pool | vs SheafPool |
|---|---|---|
| Permutation-invariant (node/edge/face reindex) | **Yes** (set-based scores, symmetric-sum readout) | same |
| Gauge-invariant | **Yes, but O(d) only** (energy descriptors) | weaker — SheafPool handles diagonal/general via whitening |
| Conformal scale ($R=\lambda O$) | declared as sheaf **data**, carry $\log\lambda$; consistent under O(d) gauge. Trace-normalized Gram removes it if treated as gauge. | SheafPool removes it automatically via whitening |
| Higher-rank invariance (edge/face energies) | **Yes** (extra) | SheafPool has no edge/face ranks |

Net: **we are narrower on the gauge group (O(d)) but broader on ranks (node+edge+face).**

---

## 5. Readout decision (SETTLED)

- **Legacy S2Pool projector** ($X^\top(XX^\top)^\dagger X$) = **degenerate ($I_3$)** → discard; keep only as a
  negative-control baseline.
- **GramPool** ($X^\top X$) = O(d)-invariant, non-degenerate, but scales by $\lambda^2$ under conformal maps.
- **✅ In-house readout = trace-normalized Gram energies, per rank (node/edge/face)** — conformal-invariant,
  cheap, multi-rank. Not a strict subset of SheafPool (SheafPool is node-only).
- **SheafPool = a strong BASELINE to beat, not a borrowed component.** (Rationale: its whitening buys
  general-gauge robustness we don't need for conformal maps; it's node-only; and it needs the same view-stacking
  hack on H2SNN's single-channel state anyway.)
- **Open sub-question (Gate A):** is trace-norm Gram empirically competitive with SheafPool terminally? If not,
  adopt whitening and cite. SheafPool published TU numbers (macro-F1): MUTAG 100, PROTEINS 80.6, ENZYMES 62.4,
  NCI1 79.5 — the bar.

---

## 6. Method — PIVOTED (2026-07-21): THS-Pool = **Typed Higher-Order Galerkin Sheaf Pooling**

**The method is now energy-preserving Galerkin coarsening** — the version the proposal (§10.3) always specified
but that we never actually built. See `sheafcovpool_implementation.md` (rewritten) and `sheafcovpool_explainer.md` for
the full spec.

**Why the earlier versions failed, and why Galerkin is different (the deep reason):**
- **V0 readout** — tied node-only pooling (Phase-0).
- **V1 relearn** — discards the fine sheaf and re-learns on the coarse graph: *not coarsening at all*; weak/noisy;
  and 20× slow (re-runs the whole backbone + face DFS per level).
- **V1.5 inject** — ad-hoc "keep the surviving maps" on gated states: *energy-inconsistent* → **hurt** (0.729).
- **Galerkin** — coarse operator `L_c = P* L P` **equals the fine sheaf energy of the lifted coarse signal by
  construction** (`z* L_c z = ‖δ (Pz)‖²`). It cannot be energy-inconsistent (removes V1.5's failure mode), it does
  **not** re-run the backbone (just matrix-free `L` applications on the *existing* fine maps → **lighter** than
  relearn), and it is genuinely *coarsening the sheaf* (faithful to the proposal).

**Novelty:** HiSP (2026) does Galerkin sheaf coarsening for **node-only, homogeneous** graphs. THS-Pool extends it
to **typed + higher-order (`L_1`)** — the empty cell. Grounding: HiSP (local spectral prolongation + Galerkin),
Loukas (coarsening guarantees), MinCut/DiffPool (learnable assignment), Bodnar CW/simplicial (higher-order WL).

**Staged plan (committed 2026-07-21):**
1. **Node-Galerkin (typed)** — learnable type-aware assignment `S` → prolongation `P0` → `L_{0,c}=P0* L0 P0` →
   diffuse coarse `x_c` on `L_{0,c}` → invariant readout. Tractable HiSP extension; tells us fast whether
   energy-preserving coarsening behaves better than relearn/inject did.
2. **Higher-order** — edge/face prolongations `P1,P2`, `L_{1,c}=P1* L1 P1` (coarsen the curl term). The novel,
   hard part; build only after stage 1.

Everything below in this section on lean-vs-sheaf-true is **history** (superseded).

### 6-old. lean vs sheaf-true (SUPERSEDED — kept as record)

Committed skeleton (both variants): Select–Reduce–Connect. Type-aware Top-$K$ per node type, scored by
O(d)-invariant sheaf energies (node $\|x^\star\|^2$, edge disagreement $\varepsilon_e=\|\delta_0x^\star\|^2$,
face curl $\|\delta_1s^\star\|^2$, $\log\lambda$, degree) → induced typed sub-complex → re-encode via
`encode_from_stalk` → multi-rank trace-norm Gram readout → cross-level concat → **single cross-entropy loss**.
Blockers resolved in the implementation spec (`encode_from_stalk` refactor; **re-enumerate** faces on the coarse
graph; ranks are **read out per level, not transported**).

- **Lean version** (current spec): select nodes + **re-learn** a fresh sheaf per level. *Risk:* the sheaf is
  decorative — this is "SheafPool-minus-whitening + a hierarchy"; it does **not** occupy the empty cell and is
  incremental. HiSP's whole point is that node-assignment alone is insufficient to coarsen a sheaf.
- **✅ Recommended: sheaf-true version.** Add ONE principled mechanism: **coarsen the sheaf by restriction** —
  *inject* the surviving fine restriction maps into the coarse block instead of re-learning, and use **typed
  restriction-map composition** ($Q=R_{v\to e}^\top R_{u\to e}$) for reconnections through dropped nodes. This
  genuinely coarsens the typed sheaf, occupies the empty cell, and differentiates from HiSP (untyped, spectral)
  and SheafPool (no hierarchy) simultaneously. Cost: needs the `inject` backbone path + conformal-scale handling.
- **Decision status:** recommend sheaf-true; **not yet ratified.** Build only after Phase-0 de-risk (§8).

---

## 7. Contributions framing (OPEN — needs final call)

Two candidate contributions:

1. **Methodological (differentiator):** the first *typed, hierarchical, higher-order* sheaf coarsening (the
   empty cell). — **KEEP (primary).**
2. **Analytical (insurance):** *"When and why does higher-order sheaf structure help typed graph classification?"*
   — the κ characterization (adaptive amplify-on-rings / suppress-on-coarse-types, unsupervised; not
   characterized by SheafPool/HiSP/HINPool). Robust even if the pooler only ties.

**Status of the analytical contribution:** 2026-07-21 — user leans **DROP it as a headline** ("we already know
faces affect performance"). My honest position recorded for the record:
- Dropping it as a *headline contribution* is reasonable → method-forward paper.
- **BUT** the premise "faces affect performance" is **not yet true** (§3.2: we showed energy, not accuracy).
- Regardless of framing, the **faces→accuracy ablation is mandatory** in any higher-order pooling paper.
- The analytical framing is *insurance*: if the pooler only ties SheafPool, the κ story is still a publishable
  finding. Dropping it removes that fallback and raises the bar (method must robustly win).

**➡ Unresolved decision for the user:** keep the κ analysis as a co-contribution/insurance, or go pure method
and accept the higher risk? (Either way, run the faces→accuracy ablation.)

---

## 8. Plan — readout-first, staged go/no-go

**Principle for limited time: design experiments to be informative WHETHER OR NOT we win.** Use the κ separation
(high-κ *should* help, low-κ *should not*) so any outcome is a publishable result, not just a win/lose.

- **Phase 0 — Higher-order READOUT test (no new architecture; the primary gate).** Terminal readout on a
  *selective* high-κ vs low-κ dataset pair (§9); node-only vs node+edge+face.
  - **Gate B (KILLER):** does node+edge+face beat node-only on **accuracy**, and does the gain **track κ** (helps
    on high-κ, not on low-κ)?
    - Clean separation → higher-order readout confirmed + κ becomes a *tested accuracy* claim → proceed.
    - Helps nowhere → higher-order thesis empirically dead on these datasets → pivot/stop.
  - **Gate A (readout form):** trace-norm Gram ≈ SheafPool terminally? (else adopt whitening + cite)
- **Phase 1 — Hierarchy (CONDITIONAL — only if Gate B passes).** `encode_from_stalk` + Top-$K$ select + induced
  sub-complex + re-diffuse + multi-rank readout + CE.
  - **Gate C:** 2-level hierarchy beats matched 1-level terminal? (else the paper is the higher-order *readout*,
    no hierarchy)
- **Phase 2 — Sheaf-true coarsening (CONDITIONAL — only if Phase 1 lives).** Inject surviving maps + typed path
  composition. **Gate D:** beats lean hierarchy?

**Non-negotiable rigor (AAAI/NeurIPS):** baselines incl. `H2SNN+SheafPool`, HINPool, HiSP, a standard pooler;
rank-ladder ablation; multi-seed, mean±std + effect sizes; **stratify gains by κ**; MUTAG's 19-graph test is
unreliable (lean on NCI1/PROTEINS/ENZYMES). Pooling is a skeptical subfield ("Rethinking pooling") — must beat
simple baselines robustly.

---

## 9. Immediate experiment — readout sweep (RUNNING since 2026-07-21)

**Scope:** small TU datasets only — **MUTAG** (high-κ, 0.41) + **ENZYMES** (low-κ expected, 3 SSE types;
κ measured by the sweep). PROTEINS and NCI1 deferred (slow). Terminal readouts, **no hierarchy yet**.
**No hyperparameter tuning** — all configs share each dataset's YAML hyperparameters (fair, no cherry-picking;
may handicap SheafPool — light per-method tuning deferred to the paper stage).

**Configs (4), discovery off, 3 seeds, 80 epochs, checkpoint per run:**

| Label | pooling | reads | role |
|---|---|---|---|
| `meanpool_node` | meanpool | node-only `x*` | node-only floor |
| `grampool_facesON` | grampool | node+edge (`s*`, face-influenced) | our readout, faces on |
| `grampool_facesOFF` | grampool + `ablate_no_faces` | node+edge, faces removed | **face isolation** |
| `sheafpool` | SheafPool (whiten+align+attention) on the **same** 3-view stacks | node+edge | **readout-form baseline** |

**What each comparison isolates:**
- `grampool_facesON` − `grampool_facesOFF` → **face-rank contribution** (Gate B), to be **stratified by κ**
  (larger gap expected on MUTAG than ENZYMES/PROTEINS).
- `meanpool` vs `grampool` → does using the edge/higher-order info help at all.
- `grampool` vs `sheafpool` → **Gate A**: is our simple Gram readout as good as SheafPool's whitening+attention on
  identical inputs.

**Honest limitation (fast follow-up):** SheafPool here is fed the **same 3-view inputs** as grampool
(apples-to-apples readout-MECHANISM test), so it is *not* the node-only SheafPool of the paper. The pure
node-only SheafPool competitor (H_i=[x*,h0], C=2) is a follow-up to directly test "node+edge (ours) vs node-only
(SheafPool)" — the cleanest "is our path better" comparison.

**Decision rule:** face gap (ON−OFF) positive and larger on high-κ than low-κ → higher-order + κ-tracking
confirmed → build Phase 1. And grampool ≥ sheafpool → simple Gram readout suffices (else adopt whitening + cite).

**Harness:** `scratchpad/run_readout_sweep.py` (reuses backbone + `graph_pooling.py`; SheafPool added as
`src/models/graph_pooling.py::SheafPool`, smoke-tested — eigh/SVD stable, no NaN). Results →
`scratchpad/readout_sweep_results.jsonl` (resumable). **Deferred:** hierarchy, sheaf-true coarsening, node-only
SheafPool, NCI1.

---

## 10. Open decisions — CLOSED 2026-07-22 (see §0)

**Decided 2026-07-21:**
- ✅ **Core bet = higher-order READOUT**, not the hierarchy (§2.5). Hierarchy + sheaf-true coarsening are
  conditional extensions, built only if the readout test (Gate B) passes.
- ✅ **Run the selective higher-order readout test first** (§9) before any further method design.

**Resolved 2026-07-22 (final, §0):**
1. **Contributions framing (§7):** resolved as *positioning claim, not performance guarantee* — ship
   `SheafCovPool` as the paper's readout (ties GramPool, dominates SheafPool); do not claim a proven
   TU win. Hierarchy/coarsening line closed negative (Galerkin's apparent win was a numerical artifact, §pm9;
   spectral descriptor added nothing, §pm10).
2. **Lean vs sheaf-true (§6):** moot — hierarchy line closed, no Phase 1 needed.
3. **Readout form (Gate A, §5):** resolved — `SheafCovPool` (attention + across-node covariance on the
   plain Gram descriptor), not SheafPool whitening (confirmed broken again in the final gate) nor plain
   trace-norm Gram alone (superseded by the covariance variant).
4. **High-κ arm choice:** moot for the TU chapter; relevant again only if/when Peptides-func/LRGB is pursued
   (§0's suggested next step, not yet started).

---

## 11. Changelog

- **2026-07-29** — **Correctness fix: the node view stack was never local-gauge invariant.** The incident-edge
  column was built by summing raw `s*` coordinates drawn from *different* edge stalks, so under independent
  gauges `x_v -> Q_v x_v`, `s_e -> R_e s_e` it did not co-transform with `x*`/`h0`; the three Gram entries
  touching it (`G13, G23, G33` — half of the six unique node entries) depended on the arbitrary choice of basis.
  Summing sections across stalks is not a well-defined sheaf operation to begin with. Fixed by
  `pool_edge_stalks_to_nodes`, which transports each edge state through its own endpoint restriction map
  (`F_src^T` / `F_tgt^T`, unsigned at both endpoints) before averaging; `(R_e F Q_v^T)^T R_e s = Q_v F^T s`
  needs only the **gauge group** to be orthogonal, not the maps — which are conformal-orthogonal `F = cQ`, so the
  long-standing "restriction maps are always orthogonal" claim in the docstrings and docs was wrong and is
  corrected. The edge stack needed no change: its three views already share one stalk. All four incident-edge
  poolers (GramPool, S2Pool, adapted SheafPool, SheafCovPool) take the same corrected summary, so readout
  comparisons stay unconfounded. `NodeReadout`/`LinkPredictionReadout` intentionally left on the old helper —
  they claim no gauge invariance, and migrating them would move node-classification and link-prediction numbers.
  **Consequence for §0 and the §8 gate table: those runs predate this fix.** The three readouts shared the same
  (wrong) node view, so their relative ordering is still like-for-like, but the absolute numbers no longer
  correspond to the code.
- **2026-07-29** — **`S2Pool` found degenerate; do not use it as a baseline.** Its descriptor
  `P = X^T(XX^T)^+X` equals the constant `I_3` for any full-rank stack (3 views, `d >= 3`): with `X = U S V^T`,
  `P = V S U^T (U S^2 U^T)^+ U S V^T = V V^T = I`. Measured `max|P - I| = 3.6e-07`; logits differ by `1.5e-08`
  across unrelated batches where GramPool/SheafCovPool differ by `~2e-01`. Its GL(d) invariance is
  invariance-by-constancy, which is precisely why the old gauge-invariance test passed on it for months — a
  constant function passes every invariance test. Not fixed (out of scope); recorded so nobody quotes it.
  Methodological lesson now written into the test suite: every invariance assertion needs a negative control
  proving it *can* fail.
- **2026-07-22 (final)** — **Adopted `SheafCovPool` as the final pooling proposal; closed the TU pooling
  investigation** (see §0). Implemented `SheafCovPool` (`src/models/graph_pooling.py`,
  canonical `pooling_type: "sheafcov"`): attention-weighted mean **and across-node/edge covariance** of the existing
  gauge-invariant Gram descriptor (no whitening — sidesteps SheafPool's collapse on single-channel stalks).
  Final gate (MUTAG+ENZYMES, 3 seeds, 40ep): **ties GramPool on MUTAG (0.596 vs 0.595), nominal +0.07 but noisy
  on ENZYMES (0.236 vs 0.166), dominates SheafPool on both (0.260/0.048)**. No further pooling-mechanism
  iteration planned.
- **2026-07-21 (pm11)** — **Read the CSNN predecessor paper** (`cooperative_sheaf_neural_network.pdf`).
  Confirms H2SNN's lineage is unambiguously sheaf-based (directed cellular sheaves, in/out-degree Laplacians),
  using multi-channel stalks `X ∈ R^{nd×h}` unlike H2SNN's single-channel design (a deliberate simplification,
  orthogonal to "being a sheaf" — but the reason SheafPool's whitening degenerates on H2SNN). **CSNN's own graph
  classification benchmark is Peptides-func/Peptides-struct (Long Range Graph Benchmark)**, not TU — used
  explicitly to demonstrate mitigation of over-squashing/long-range under-reaching. This reframes the whole TU
  pooling investigation: TU graphs are too small/local to exercise the regime where sheaf/cooperative structure
  has a plausible advantage, so a null result there is expected regardless of readout sophistication.
- **2026-07-21 (pm10)** — **Tested `SheafSpectralPool`** (gauge+permutation-invariant heat-kernel diffusion-decay
  descriptor of `L0`, power-iteration normalized, fully detached; `src/models/s2pool_h2.py`). Smoke-tested,
  wired via `ths_spectral` + `pool_use_spectral`. **Gate (MUTAG, 2 seeds, 40ep): `local_only` and
  `local+spectral` produced byte-identical per-seed metrics** — the trained model fully ignored the spectral
  channel. **Negative result**, reported without hedging (the identical-per-seed values are unambiguous evidence
  of zero added signal, not just "no improvement").
- **2026-07-21 (pm9)** — **Root-caused and fixed an ENZYMES NaN/degenerate-collapse bug, which REVERSES the pm8
  positive result.** Chain: coarse diffusion diverging on larger graphs → tried spectral-norm step scaling (made
  it worse, immediate NaN) → isolated failure to training batch 1 regardless of fix → detached `F_src`/`F_tgt`
  in `_L0c` (no effect) → detached `assign` too (no effect) → **root cause: `_standardize_per_graph`'s
  `1/std` blows up on near-zero-variance graphs**, and only the `fine_coarse` variant backprops through it (via
  `_assign`), explaining why only the coarse arm NaN'd. **Fix:** `scale = (var[batch].sqrt() +
  EPS).clamp_min(1e-2).detach()`; verified all batches finite for a full epoch. **Re-ran the MUTAG Gate C
  post-fix: fine+coarse 0.756±0.085 ties fine-only 0.745±0.080** — the earlier pm8 "+0.14, beyond noise" result
  was an artifact of the (buggy/unstable) pre-fix dynamics, not a real effect. ENZYMES post-fix: fine+coarse
  0.251±0.105 (highly unstable per-seed) nominally above fine-only 0.164±0.017 but **losing to grampool
  0.303±0.147**. User asked whether this could be a hyperparameter issue; explicitly declined an HP sweep and
  redirected to two alternatives instead (global Galerkin readout / invariant descriptor) — see pm10.
- **2026-07-21 (pm8)** — **Built node-Galerkin (stage 1) and it WORKS.** `GalerkinSheafPool`
  (`src/models/s2pool_h2.py`): typed soft assignment `S`, matrix-free `L0_c = S^T L0(S z)`, bounded coarse
  diffusion, log-scaled invariant readout. **Energy preservation verified EXACT** (`z^T L0_c z == ‖δ0(Sz)‖²`,
  rel_err 0, PSD). Fast: **3.3 s/epoch (~13× faster than relearn)**, 11K params, stable (fixed a readout-energy
  spike). **Gate (MUTAG, 3 seeds, macro-F1): fine+coarse 0.846±0.023 vs fine-only 0.707±0.052 → +0.14 on 3/3
  seeds, beyond noise** (matched readout, only the coarse level toggled). Also beats grampool 0.635 and old
  relearn 0.713. **First clean positive in the whole pooling investigation** — energy-preserving coarsening helps
  where relearn (noisy) / inject (hurt) did not. Next: confirm on ENZYMES + more seeds; then higher-order `L1`.
- **2026-07-21 (pm7)** — **PIVOT: THS-Pool → Typed Higher-Order Galerkin Sheaf Pooling** (§6). Committed to the
  Galerkin redesign as a genuine research thrust: energy-preserving coarse operators `L_c = P* L P` (the version
  the proposal §10.3 always specified but we never built), which removes V1.5's energy-inconsistency failure and
  is *lighter* than relearn (no per-level backbone re-run). Staged: node-Galerkin (typed) first, then higher-order
  `L_1`. Rewrote `sheafcovpool_implementation.md`, `sheafcovpool_explainer.md`; updated proposal. Relearn/inject code
  (`s2pool_h2.py`) superseded (kept as negative-ablation record).
- **2026-07-21 (pm6)** — **Built + tested V1.5 (inject surviving maps).** Backbone `injected_maps` path +
  `pool_inject_maps` config; smoke OK. **One-seed comparison (MUTAG seed 0, macro-F1):** L2_relearn (V1) **0.835**
  > L1_terminal 0.774 > L2_inject (V1.5) **0.729**. **Geometry injection HURT** — re-learning coarse maps beats
  carrying fine ones down. Since the cheap geometry step (V1.5) underperforms, full V2 (path composition) is very
  unlikely to help → **do not build V2.** Net evidence across experiments: readout tied node-only (Phase-0), plain
  hierarchy weak-positive within noise (Gate C V1), geometry-carrying negative (V1.5). Only the lean re-learn
  hierarchy (V1) has any positive signal, and it is noise-level. Direction under review (pivot vs write-up).
- **2026-07-21 (pm5)** — **Built THS-Pool V1** (`src/models/s2pool_h2.py` + `encode_from_stalk` refactor + trainer
  multi-level driver + config `pool_num_levels`/`pool_ratio`). Smoke-tested on MUTAG (learns, no NaN). Profiling:
  the per-level CPU face-enumeration DFS dominates (~20× slower than terminal). **Gate C (MUTAG, 3 seeds,
  face_k=20, 60ep, macro-F1):** L2-hierarchy 0.713±0.089 vs L1-terminal 0.677±0.080 vs grampool 0.673±0.029 →
  **weak positive** (L2>L1 by +0.036 on 2/3 seeds, but within noise; MUTAG test = 19 graphs). Sanity: L1 readout
  ≈ grampool (no regression). Next per decision: escalate to V1.5 (inject surviving maps) — the real bet — since
  V1 is at worst a noisy tie, and V1 tying does not rule out geometry-carrying.
- **2026-07-21 (pm4)** — **Renamed method S²Pool-H2 → THS-Pool** (Typed Hierarchical Sheaf Pooling) across all
  doc content (filenames later canonicalized). Added newcomer walkthrough `docs/sheafcovpool_explainer.md` (analogy + 6-step worked
  example + novelty). **Recorded decision: build V1 (lean) first, V2 (sheaf-true) conditional on Gate C** — V1 is
  the most-favorable, most-feasible test of whether the hierarchy helps; V2 (constrained geometry) can't rescue a
  hierarchy that V1 (free re-learning) already ties.
- **2026-07-21 (pm3)** — Phase-0 sweep results logged (§3.3): CAUTIONARY — higher-order readout **ties**
  node-only; SheafPool arm invalid (diagnosed). User chose to proceed to method implementation. **Started build:**
  `encode_from_stalk()` refactor landed in `src/models/h2snn_model.py` (behavior-preserving, verified
  `encode` vs `encode_from_stalk` diff ~1e-7) — blocker #1 resolved in code. Next: `src/models/s2pool_h2.py`
  (V1 SRC pooler) + driver loop.
- **2026-07-21 (pm2)** — Implemented `SheafPool` readout on the H2SNN backbone
  (`src/models/graph_pooling.py`, apples-to-apples on grampool's 3-view inputs; smoke-tested, numerically stable)
  and a resumable multi-seed sweep runner (`scratchpad/run_readout_sweep.py`). **Launched the Phase-0 readout
  sweep on GPU** over small datasets (MUTAG/ENZYMES/PROTEINS × {meanpool, grampool ON/OFF, sheafpool} × 5 seeds,
  80 epochs). NCI1 deferred (large). §9 updated to the actual running configs; node-only SheafPool noted as a
  follow-up.
- **2026-07-21 (pm)** — **Reframed: higher-order READOUT is the core bet; hierarchy demoted to conditional
  extension.** Added theoretical footing (§2.5): `x*` is provably face-independent (x/s diffuse independently), so
  node-only pooling / `H2SNN+SheafPool` is blind to faces → the readout has a provable information asymmetry,
  while the hierarchy has weak footing on small graphs. Rebuilt the plan (§8) readout-first and the immediate
  experiment (§9) as a selective **high-κ vs low-κ** terminal test designed to be informative either way. Decided
  to run this test before further method design.
- **2026-07-21** — Created this living log. Captured: empty-cell novelty target; κ findings + the energy≠accuracy
  caveat; invariance audit; readout decision (trace-norm Gram in-house, SheafPool as baseline, projector
  discarded); lean-vs-sheaf-true (recommend sheaf-true); contributions-framing tension (analysis dropped as
  headline per user, but faces→accuracy ablation mandatory and unproven); staged go/no-go plan; one-day test
  design. Fixed 3 implementation blockers earlier (see `sheafcovpool_implementation.md`).
- **2026-07-15** — κ experiment run (MUTAG, PROTEINS); proposal rewritten to committed single-method form;
  implementation spec written.
