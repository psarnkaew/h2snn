# Known-answer pre-registrations S1′-B, S3, S2′ under v3-B — DRAFT (item 6)

**Status:** DRAFT for the author. Not yet a frozen pre-registration: it becomes one when the author approves it and it is committed before any S-run. Spec §0 rule 6 applies: every task states (a) **reachability** and (b) an **information-route audit**, per arm, including memorisation, written from the code.
**Written:** 2026-11-11, design session. **Authority:** spec v1.11 (C2, C2b, C12, §6 known-answer rows, E13), task card Part K (K0 pre-registration, `docs/WISP_evidence/k_preregistration.md`), Part K report, Part L report §7–§8.
**Tags:** VERIFIED-CODE, VERIFIED-RUN, DERIVED, UNVERIFIED, PROPOSAL.

## 0. What changed since Part K (why these must be re-registered)
1. **The propagation changed.** Gate (ii) (capacity of the weight channel) was met in Part K under v2 (gated Chebyshev filter bank, `K = 4`). v3-B uses one anchored solve on the three-level energy. **Capacity under v3-B is not shown**; S1′-B's `full` arm re-establishes it.
2. **The receptive field is the whole connected component.** The solve is a PPR resolvent; a fact route is no longer limited to `K` applications. Every route audit below uses components, not hop counts.
3. **Context routes.** Maps and fact weights read the geometry `[h₀, c]` in every arm (E10; `geometry_context: true`). `c_v` is, per (relation, role), the mean of `v`'s neighbours' `h₀` (VERIFIED-CODE `src/models/relational_context.py:3, 28-44`). Hence a fact `(u, v)` sees, through its map MLP and its weight `c_e = softplus(MLP([h_u, h_v, rel_emb])) + ε` (VERIFIED-CODE `src/models/sheaf_maps.py:261-273`, `h` = geometry), every node within one hop of `u` or `v`, as per-relation means. **A window of four consecutive path nodes is visible to one fact.** The Part K audit of S1′ did not list this route (DERIVED; it was harmless there because no arm learned the 6-bit parity, β = 0 stayed at chance).

Common settings (all three tasks; PROPOSAL, fixed before running): `propagation: anchored`, solver values of spec §6 (never tuned), `d = 32`, `tau_init = 1`, AdamW lr 1e-3, scalar group 1e-2 (θ_τ, `b`), cosine, 100 epochs, grad clip 1.0, validation-accuracy selection, `resample_paths: false` (one draw = training and evaluation, as in Part K), n = 2000 targets split 50/25/25 (1,000 training targets), seeds 42, 43, 44. The oracle lives in the evidence script, not in `src/`.
**Reachability of the scalars** (DERIVED, as K0): one scalar in the scalar group moves at most Σ_t lr_t = 0.01 · 50.5 ≈ 0.505 in 100 cosine epochs. So `log τ ∈ [−0.5, 0.5]` (τ ∈ [0.60, 1.66]) and `b` shifts every cell by at most e^{±0.5}. Neither can supply a selection ratio; selection must come from the oracle's fixed β (S1′-B) or from `g` (S3, S2′).

---

## S1′-B — is the node–cell term needed? (claim C12)
**Question.** On a task whose label is carried by one cell among 64 per target, does a strongly weighted cell inform its target **only** when the node–cell term exists (`full`), and stay capped under covering pairs only (`covering_only`)? This tests Prop. 7′'s ceiling in a known-answer setting.

**Graph.** `h5_synthetic.graph_s1p(n = 2000)` unchanged (VERIFIED-CODE `docs/WISP_evidence/h5_synthetic.py:86-121`): each target paper P has 8 authors A, each A writes 8 cpapers Q (64 cells P–A–Q per target, all of schema `writes`·`writes_c`); exactly one Q\* per target carries class y; the 63 others carry an independent random class; decoys D (complement bits) write A's cpapers; 6-bit codes with `b_P ⊕ b_A ⊕ b_Q = 0` only for Q\*; 4 noise dims on P and Q. ℓ_max = 2, r = 640 (coverage 1.0 in Part I/K).

**Arms (18 runs).** `full` + oracle and `covering_only` + oracle, β ∈ {0, log 64, log 512}, seeds 42–44. Oracle: `w_f = exp(b + β·o_f)`, `o_f` the known relevance, β a constant buffer, `b` learned (scalar group), `g` unused; `β_f = w_f m_f` as in production.

**Idealised check (VERIFIED-RUN, this session; script `docs/known_answer_evidence/s1pb_idealised_check.py`).** Same cell structure, `d = 1`, identity maps, `a_e = 1`, exact sparse solve of `(PᵀP + τL̂)Y = Pᵀh`, readout sign(x*_P), n = 400:
| β | `full` acc (corr) | `covering_only` acc (corr) |
|---|---|---|
| 0 | 0.47 (0.01) | 0.49 (0.01) |
| log 64 | 0.82 (0.69) | 0.55 (0.18) |
| log 512 | 1.00 (0.95) | 0.55 (0.18) |
| log 4096 | 1.00 (0.99) | 0.55 (0.19) |
`covering_only` at log 512 over τ ∈ {0.6, 1.66, 10, 50}: corr 0.23 / 0.14 / 0.04 / 0.02 (acc ≤ 0.57); seeds 0–2 at τ = 1: acc 0.55 / 0.58 / 0.61. Same pattern with 32 authors × 2 cpapers. Reading (DERIVED): under covering pairs the relevant cell reaches P only through the fact state `s_{PA*}`, which P mixes with its seven other facts at weight `a_e`; increasing β saturates the cell's pull on `s_{PA*}` but not P's share of it, which is Prop. 7′'s ceiling generalised to shared facts.

**Reachability.** β is fixed; the selection ratio e^β needs no learning. What must be learned: maps that keep a common class direction across relevant cells (dispersion, K0 definition) and the head. Many parameters at lr 1e-3; Part K's v2 runs aligned them (dispersion 0.87 → 0.42–0.52 at log 512). Under v3-B, UNVERIFIED.

**Information-route audit (per arm).**
1. *Own features:* none (y independent of `b_P`). *Memorisation:* P's 4 noise dims + 6 bits are unique; 1,000 training targets; every arm can fit training labels. The test split is the check.
2. *Facts (both arms, whole component):* Q's class reaches P through P–A and A–Q facts, mixed over 64 cpapers with one carrying y: idealised SNR ≈ 1/√63 (≈ 0.55). Transports cannot select (orthogonal: they rotate, never rescale).
3. *Context (both arms):* `c_A` (relation `writes_c`) = mean of A's 8 cpapers' `h₀`, which for A\* includes y/8 in the class slot; the P–A fact map and weight see `c_A` and `h_P`. A fact weight that grows with "A is the relevant author" needs the 3-way parity from means, which no arm learned in Parts H–K. Bounded by the β = 0 control.
4. *Cells:* `full`: node–cell term `β_f ‖T_{P,f}x_P − z_f‖²` gives P a direct pull of weight β\* (idealised 0.82 / 1.00 at log 64 / log 512). `covering_only`: only via `s_{PA*}` (ceiling above).
5. *Decoys:* add A's complement bits, not the label.

**Expected (fixed now).** `full` + oracle: ≥ 0.9 on 3/3 seeds at log 512 (log 64: ≈ 0.8, reported). `covering_only` + oracle: ≤ 0.65 at every β, every seed. β = 0: ≤ 0.65 in both arms.

**Criterion and consequences (fixed now).**
- **Leak:** β = 0 ≥ 0.9 in either arm → the task leaks; no verdict; redesign.
- **Capacity under v3-B:** `full` + oracle < 0.9 on any seed at log 512 → v3-B capacity is not shown at default settings. **This is a warning signal (working preferences): stop and discuss before further engineering.** Report relevant share of `D_P`, dispersion, τ.
- **C12 known-answer part:** passes iff `full` ≥ 0.9 on 3/3 at log 512 **and** `covering_only` ≤ 0.65 at every β and seed. If `covering_only` ≥ 0.9 at log 512 on 3/3, the "S1′-B passes without the term" half of C12's falsification holds; C12 is then decided by the cluster comparison as written in §5. Anything between is reported as is, criterion not loosened.

**Not stop conditions:** `log 64` results; `∂L/∂β` sign (reported as in K0).

---

## S3 — can `g` learn instance-level relevance? (C2, learnability of the weight channel)
**Spec text (E13):** "instance-level learnability: the relevant cell is fixed by a simple function of its intermediate and its anchor that the endpoints cannot see."

**Problem with the spec text (DERIVED).** Any function of (anchor P, intermediate A) on a length-2 cell P–A–Q is visible to the P–A fact's weight `c_e` and map, which read `[h_P, c_P, h_A, c_A]`. The `uniform` and `endpoints` arms keep fact weights, so they could select the relevant author through `c_e` without `g`. With one cpaper per author the fact route alone would select the relevant cell completely; with 8 per author it would lift the SNR from 1/√63 to about 1/√7. The control arms would not be controls. The same holds for any relevance that depends on nodes within a window of four consecutive path nodes (§0 item 3).

**Proposal (PROPOSAL; needs the author's approval).** Make relevance a **pairwise-invisible** function of three nodes that no single fact (with context) can see together, and keep it as simple as possible: a **1-bit three-way parity**.
- **Cell length 4:** P –r₁– A –r₂– B –r₃– C –r₄– Q (types P, A, B, C, Q; distinct relations, one schema only, so no schema route).
- **Relevance:** `t_P ⊕ t_B ⊕ t_Q = 0` (1-bit tags on P, the middle node B, and the end Q). Distances: P–B 2, B–Q 2, P–Q 4. A fact window (u's neighbour, u, v, v's neighbour) spans 3 edges, so it contains at most two of {P, B, Q} (DERIVED). `endpoints` sees P and Q only. Only `full` (g reads all `h₀` on the cell) sees all three.
- **Per target:** 64 cells (e.g. 4 A × 4 B × 4 C chains ending in 64 distinct Q, or 64 disjoint chains; to be fixed by the card so that every intermediate is shared only within one target), exactly one relevant (its Q carries y), the 63 others have parity 1 and an independent random class.
- **Bit balance:** tags assigned so that, for every node, the per-relation context means of tags are independent of whether the node lies on the relevant cell (e.g. each B has equally many neighbours of each tag). This must be checked on the generated graph by an explicit audit (below).

**Arms.** `full`, `endpoints`, `uniform` (C2 ladder on identical cells), and `full` + oracle (fixed β = log 512) as the capacity reference; seeds 42–44; n = 2000.

**Reachability.** The relevant cell needs a log-weight gap ≈ log 512 ≈ 6.2 over the irrelevant ones (from S1′-B's dose-response under v3-B; replace by S1′-B's measured value once known). `g` has one projection + GELU + linear per length; Part I raised log w by ≈ 4.4 and K2b by ≈ 6.95 in 100 epochs (v2), so the range is reachable in principle. The function (3-input XOR of ±1 tags) is representable by a one-hidden-layer GELU network (DERIVED). Whether it is learned is the test.

**Information-route audit.**
1. *Own features / memorisation:* as S1′-B (noise dims unique; 1,000 training targets).
2. *Facts:* every fact weight and map sees at most two of {P, B, Q}; relevance is independent of any two (pairwise-invisible by construction), so no fact weight can select (DERIVED; **to be verified on the generated graph** by fitting, on the training targets, a probe that predicts relevance from each fact's input `[h_u, c_u, h_v, c_v, rel]`: accuracy must be at chance; fixed solver and λ as K0).
3. *Transports:* orthogonal; sign flips along the chain can make the composite depend on the parity, but they cannot rescale; the 63 random classes keep their noise (SNR 1/√63) (DERIVED).
4. *Cells:* `uniform`: equal weights, SNR ≈ 1/√63. `endpoints`: g sees P, Q (their tags' XOR is independent of relevance), so no selection. `full`: g can compute the parity.
5. *Context:* covered by item 2's probe.

**Expected (fixed now).** `full` + oracle ≥ 0.9 (3/3). `uniform`, `endpoints` ≤ 0.65 on every seed. `full`: open (this is the question).
**Criterion and consequences (fixed now).**
- Any control ≥ 0.9, or the fact-input probe above chance → leak; no verdict; redesign.
- `full` + oracle < 0.9 → capacity not shown on this graph; stop at the diagnosis.
- `full` ≥ 0.9 on 3/3 → `g` learns instance-level relevance that no endpoint or fact view contains: the weight channel is learnable (the S3 statement of spec §6).
- `full` ≤ 0.65 on all seeds → not shown at default settings; with the C2 kill rule, the paper reading of spec §6 applies ("the weight channel did not learn selection; H is untested through the weights"). Between: reported as is.

---

## S2′ — schema read-out without relation-specific fact routes (C2b)
**Spec text (E13):** "every carrier attaches by the same relation, so only cell weights can separate schemas."

**Problem (DERIVED).** Under v3-B no graph can make cell weights the only way to separate schemas:
1. Schemas differ in at least one relation or in the order of relations. At the first position where two schemas differ, the facts differ in relation, and fact weights and maps read `rel_emb`, so the fact route can weight one schema's facts more.
2. If two schemas use the same relations in a different order, relation-conditioned orthogonal transports do not commute: the label-carrying schema's signal arrives at the target in a different direction from the distractor's, and a linear head can project the distractor out (up to `d` distractor directions).
Part K's K2b miss (`uniform` solved it through the relation-specific fact `d`) is one instance of (1).

**Options for the author.**
- **(a) Recommended:** keep C2b's criterion as a statement about the read-out, not about necessity: rerun K2b's graph (`i4_s2_unconfounded.graph_s2u(n = 2000)`, r = 200, coverage 1.0) under v3-B; criterion unchanged (σ\* strictly highest mean learned log-weight per cell among full-walk schemas on 3/3 seeds; `uniform` cannot rank it first by construction). Report `uniform`'s accuracy as "selection not required on this graph". The paper's C2b wording becomes: "when one schema carries the label, the read-out ranks it first", without "only the weights could find it".
- **(b)** Design a graph with more than `d` distractor schemas of permuted relation order (to block the projection route) and equal relation multisets at every position (to block the fact-weight route). Possible in principle; large, and its audit is hard to make complete.
- **(c)** Drop S2′ and keep C2b on real data only (Kendall τ across seeds, as 1.10).

**Reachability (option a).** K2b reached a σ\* − sibling gap of ≈ 0.14–0.21 in mean log-weight and σ\* ≈ 6–7 absolute under v2; the criterion needs only a strict gap > 0.
**Route audit (option a).** As K2b's (`k_preregistration.md` K2b), plus the context route (§0 item 3): `c_{M3}` contains C\*'s class through relation `d`.

---

## Open decisions for the author
**Decided 2026-11-11:** S1′-B approved as written; S2′ option (a). **Later the same day the author deferred all three tasks** (S1′-B, S3, S2′; prompt KA1 not run). S3 design choice still open.
1. ~~Approve S1′-B as written (graph unchanged; 18 runs).~~ Approved.
2. S3: accept the proposed 3-node, 1-bit, length-4 design in place of the spec text, or keep the spec text knowing the controls leak through fact weights.
3. ~~S2′: option (a), (b) or (c).~~ (a).
4. Where these run: locally (8 GB; S1′-B at n = 2000 needed 1.9 GB under v2, v3-B UNVERIFIED) before the mini-batch card, or on the cluster with the protocol (spec §6 today).
5. The spec text to change with these (E13 wording of S3 and S2′) goes into the v1.12 edits after approval.
