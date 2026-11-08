# WISP (task card Part H): pre-registration of H1–H6

Written 2026-11-08 on branch `wisp`, before any H1 code. Authority: methodology spec v1.3 (§2 R2,
R8, R12; §3.2; §4 Props. S, K, R, E; §5 C2, C2b; §6 WISP settings and gate). Nothing below is
changed after a result; additions get a dated heading. **Status of H5 (S1, S2): pending the
author's answers to the three H0 decisions at the end. S1/S2 are not run before those answers.**

## Author overrides of Part H (message of 2026-11-08), recorded
1. **H3 variance gate:** the fixed model is the 5-epoch ACM smoke checkpoint; 20 training draws per
   proposal (card: 50).
2. **H4:** ΔL_σ with T = 3 evaluation draws.
3. **H5:** the synthetic graphs are sized so that one run takes under 1 minute, with at most 100
   epochs.
   - 3 seeds for `full` and for an added **oracle-weight arm** (w_f set from the known answer,
     everything else trained, as in D12 G3c). The oracle arm separates capacity from learning;
     it is reported alongside and does not change the pass criteria.
   - 1 seed for every other arm. 2 more seeds are run only if a control unexpectedly reaches the
     pass level.
4. **H6:**
   - every ladder value on ACM only, 2 epochs each;
   - DBLP and IMDB with `full` only, 2 epochs each, plus sampling statistics;
   - LP and GC 1 epoch each, on one dataset each;
   - cost on ACM only.
5. **Test suite:** the full suite after H2, H6 and H7; only the affected tests after H1, H3, H4, H5.
6. **No empirical claims.** No number from this card is a performance result. Real-data accuracy
   is logged only, never reported or compared. The spec §6 WISP gate stays closed (G0b h1–h3 are
   deferred to the cluster).
7. **Unchanged:** correctness tests T-H1…T-H11 and the 5 mutation checks.

## Definitions fixed now (from spec §3.2 and the card; where they leave a choice, the choice is stated)
- **Actions.** An action is a (relation, direction) pair, with direction ±1, or 0 for a
  self-inverse relation (D8's `HalfEdges` convention). A_j is the set of actions with at least one
  fact from v_{j−1} to a node not yet on the walk.
- **N_j** is the set of **facts** of action a_j from v_{j−1} to unvisited nodes, so |N_j| counts
  facts. With no duplicate facts this is exactly "uniform unvisited neighbour". VERIFIED-RUN in H0
  (`h0_data_facts.json`): ACM, DBLP and IMDB have 0 duplicate facts. ACM's 25 reciprocal cite pairs
  are distinct actions. The fixtures include a parallel pair so the convention is tested.
- **A self-loop fact** leads to a visited node, so it is never available.
- **p(ω), q(ω) and the IS ratio** are computed in float64, from exact counts taken from per-node
  action-degree tables minus the visited nodes' fact counts.
- **Draw seeds** (unchanged from D8):
  - evaluation: one draw from `p` only, seed = `cfg.seed`;
  - training draw of epoch e: seed `cfg.seed + e`, from `½p + ½q` (`q = p` under `proposal: uniform`).
  - The evaluation draw does not depend on the ladder arm, the proposal or the model.
- **Merge** is D8's: canonical fact sequence, summed weights. With `q = p` every ratio is exactly 1,
  so `m_f = Σ 1/r`, as in D8.
- **Grouping** is by length, one `FaceGroup` per length. A per-cell relation sequence and signature
  id are kept, with a signature table.
- **`wisp_walks_per_anchor` default rule (calibration, not tuning; no labels, no model):** the
  smallest r whose ACM evaluation-draw incidence count (Σ cells × (length + 1), seed 42,
  `ℓ_max = 4`) is ≥ the `schemas` run's 403,122. DBLP (862,788) and IMDB (143,052) use the same
  rule in H6, per spec §6. r and the counts are reported.
- **`ℓ_max = 4`** on HGB (spec §6). The synthetic tasks state their own ℓ_max below.

## H1: sampler with q = p (T-H1, T-H2 for q = p, T-H4, T-H9, T-H10)
- **Expect:** all pass, at the card's tolerances.
- **T-H1 setup:** enumerate every simple walk from the anchors of a fixture with ≥ 3 node types, a
  self-inverse relation, a reciprocal pair, a parallel pair, a dead end and a hub. Run 200k draws;
  every walk's frequency must be within 4 s.e. of p(ω). The discarded mass must equal the
  enumerated dead-end mass within 4 s.e.
  - At about 100 walks, the family-wise false-failure probability is ≈ 100 × 6e-5 ≈ 0.6 %.
- **T-H10:** sampling time at r ∈ {r₀, 2r₀, 4r₀} on ACM, linear fit R² ≥ 0.98. The code has loops
  over hops (≤ ℓ_max) and over visited positions (≤ ℓ_max) only.
- **Wrong if:** any walk outside 4 s.e.; a drawn walk repeats a node; the dead-end mass mismatches;
  the evaluation draw differs between arms; R² < 0.98.

## H2: `path_weight: exp`, the ladder, per-cell relation embeddings (T-H5…T-H8; D12 goldens)
- **Expect:** all pass. `sigmoid_gamma` + `schemas` reproduces the D12 goldens bit for bit. The
  golden case `uniform_path_gate` becomes `path_weight_input: uniform` under `sigmoid_gamma` and
  must stay bit-identical.
- **Capacity rule for `endpoints`:** the hidden width H_e of the per-length first layer is the
  integer that minimises |P_endpoints(H_e) − P_full(H)|, with P the parameter count of `g` summed
  over lengths 2..ℓ_max. The test checks |ΔP|/P_full ≤ 5 %.
- **Projections** are built for every length in 2..ℓ_max under `wisp`, not only for the lengths in
  one draw (inventory §7).
- **T-H7 and the context.** Under `geometry_context: true`, an intermediate's h₀ enters its
  neighbours' c, and the endpoints are neighbours of the intermediate in a length-2 cell. So
  "perturb an intermediate's h₀ ⇒ `endpoints` w unchanged" holds only with the context off. T-H7
  therefore runs two ways:
  - (a) context on: perturb the intermediate's row of the **geometry input** fed to g. Exact
    invariance is expected for `uniform`, `schema` and `endpoints`; `full` changes.
  - (b) `geometry_context: false`: perturb the intermediate's **h₀**. The same expectations hold.

  This is the checklist item "endpoints sees no intermediate except through c", made testable.
- **Wrong if:** any golden differs; exp weights over e^{−10}…e^{10} break Prop. 2, 3 or 5 beyond
  1e-9; a θ_γ parameter exists under `exp`; |ΔP| > 5 %.

## H3: adaptive defensive proposal (T-H2 biased q, T-H3; variance gate)
- **Tests:** T-H2 with a strongly biased q (IS ratio dropped ⇒ control fails); T-H3 (every ratio ≤ 2
  for random and near-degenerate q; a per-step mixture control exceeds 2 somewhere).
- **Variance gate (reduced).**
  - Fixed model: the 5-epoch ACM smoke checkpoint (`path_source: wisp`, `path_weight: exp`,
    `full`, `proposal: uniform`, seed 42).
  - q_len and q_act come from that model's Σ w_f m_f per length and per (node type, action) on
    one uniform training draw (seed 43), detached.
  - 20 training draws per proposal at equal r, seeds 1000…1019, the same seeds for both proposals.
  - 16 Rademacher probe matrices z ∈ {±1}^{N×d}, fixed seed.
  - Statistic: V(q) = mean_draws mean_probes ‖(L̂_P^(i) − L̄_q) z‖², with L̂_P = Σ w_f m_f L_f
    applied matrix-free and L̄_q the mean over that proposal's draws.
  - **Gate:** V(adaptive) ≤ 0.8 · V(uniform). Otherwise the default stays `uniform` and the report
    says so.
- **Expected: UNVERIFIED, no confident prior.** After 5 epochs, w_f may still be near exp(b) with
  little spread, and q then mostly redistributes mass toward frequent (type, action) pairs. That
  lowers variance only where the weighted mass is concentrated. My guess is that the gate is
  missed (reduction < 20 %).
- **Wrong if (T-H3):** any ratio > 2 + 1e-12.

## H4: read-outs (T-H11)
- **Expect:** pass.
  - Σ_π I_π = Σ_σ I_σ = 1 within 1e-10.
  - The pair decomposition reproduces each cell's energy within 1e-10 (Prop. K, uniform κ).
  - ΔL_σ of an absent schema is exactly 0.
- **ΔL_σ:** T = 3 evaluation draws from p, seeds `cfg.seed + 7919·t`, t = 0, 1, 2 (t = 0 is the
  evaluation draw).
- **Kendall τ** has a permutation null with 1000 permutations, fixed seed.
- **Isolation:** read-outs run on a frozen checkpoint; the read-out module imports nothing from the
  training code.

## H5: known-answer tasks (PENDING the H0 decisions; written so the author can choose)
**S1 as carded.**
- Each target P has PAP-like instances (P, A, Q), with bits b on P, A and Q. Relevance:
  b_P ⊕ b_A ⊕ b_Q = 0. Relevant instances carry class y, irrelevant ones 1 − y, so endpoint class
  counts are identical across classes.
- Contexts are balanced so that no 1-hop mean reveals b_A:
  - each A writes P, Q and Q̄ = Q with the bit and class flipped;
  - each Q has a second, decoy author with bit 1 − b_A.
- ℓ_max = 2.
- **Card expectation:** `full` ≥ 0.9 and `endpoints` near chance.
- **My prediction (DERIVED, before any run): `endpoints` will likely NOT stay near chance.**
  - The card's rank argument covers products of **positive per-edge weights**. But a cell's
    transport is a product of **orthogonal** edge maps, each conditioned on its edge's two
    endpoints (D1; spec R2).
  - A Householder reflection of the class coordinate, conditioned on b_P ⊕ b_A on edge (P, A) and
    on b_A ⊕ b_Q on edge (A, Q), makes Q's class arrive at P multiplied by (−1)^{b_P⊕b_A⊕b_Q}.
  - Relevant instances (class y, sign +) and irrelevant ones (class 1 − y, sign −) then all vote
    for y. Parity is exactly what sign-composing transports express.
  - `endpoints` keeps those transports; only its weight is endpoint-only. So do `schema`,
    `uniform` and the fact-edge route of `no_path_cells`.
  - More generally, for bits any 3-node relevance pattern is separable by a rank-1
    (scale × sign) product of per-edge factors.
  - So on a low-cardinality S1, C2's `full` vs `endpoints` contrast is not representational in this
    model family.
- **Alternative S1′:** high-cardinality key matching, D12 G3c v2 (6-bit keys, 16 class-balanced
  co-papers per author). There the transports can route only with crosstalk. But D12 found that
  `full` did not *learn* w_f at default settings (0.504 / 0.58 / 0.536), while the oracle w_f
  reached 1.0 / 0.996 / 1.0. My prediction for S1′ is that `full` misses 0.9 and the oracle
  passes.

**S2 as carded.**
- A rare planted length-4 schema σ\* among ≥ 6 relation types carries the label.
- **Expectation:** "I_σ ranks σ\* first on every seed".
- **Problem (DERIVED):** by Prop. K's pair accounting, a planted length-4 cell gives its own weight
  to each of its 10 pairs, so each of σ\*'s sub-path schemas receives at least the planted cells'
  σ\* weight, plus whatever other cells add. So I_σ(σ\*) can never be strictly first among all
  sub-path schemas.
- **Testable versions (PROPOSALS):**
  - (i) σ\* ranked first by I_π, the full-walk schema share, which is in the spec;
  - (ii) σ\* first by I_σ among length-4 sub-path schemas;
  - (iii) σ\* first by lift, I_σ(trained) / I_σ(at initialisation, constant g). This is not in the
    spec.

## H6: smoke and cost (reduced; measurement, gates as carded)
- **Expect:** every listed run finishes with finite losses.
- **Max log-weight ≤ 30;** otherwise stop and diagnose, no clamping.
- **Sampling:** ≤ 25 % of epoch time on ACM, DBLP and IMDB.
- **Peak memory:** ≤ 1.25 × the `schemas` run on ACM at matched incidences.
- **My prior:** the sampling-time share is the gate most at risk on ACM: about 100k walks × 4 hops
  of visited-count lookups per epoch, against a forward + backward of about 1.5 s (G3).
  UNVERIFIED.
- Accuracy is logged, never reported.

## H0 decisions needed from the author (these block H5 only; H1–H4 can proceed)
1. **S1:** run the card's parity S1 with my prediction recorded (a likely miss on `endpoints`,
   which would be a finding about transports), or replace it with S1′ (key matching, where the
   expected miss is `full` learning w_f, with the oracle as capacity evidence), or run both?
   - Either way, C2/P2 should be read with the derivation above in mind: in this model the
     intermediates already shape every arm's *transport*. The ladder isolates only the *weight*.
2. **S2 criterion:** (i), (ii) or (iii) above, or a combination?
3. **Branch base:** `wisp` is created at `19a9d36`, not `1c106d2`.
   - `19a9d36` is your docs-only commit on top of `1c106d2`: specs, plan, task cards, AGENTS.md,
     CLAUDE.md. `src/` and `tests/` are identical to `1c106d2`.
   - Branching at `1c106d2` would have removed Part H and your AGENTS/CLAUDE edits from the working
     tree. Those two files are now committed, so my D2-row edits to them stay uncommitted, as
     instructed.
   - OK?


## Addendum 2026-11-08: the author's H0 decisions (written before any H1 code and before H5)
Decisions 1–3 of H0 are answered. H5 below replaces the H5 section above wherever they differ.

**A. Branch base.** `19a9d36` is confirmed.

**B. S2 criterion.** The read-outs run on the evaluation draw at the selected checkpoint.
- **Primary:** σ\* has the highest *mean learned log-weight per cell*, mean_f g(ξ_f) over the cells
  whose full-walk schema is π, among all full-walk schemas. This is frequency-free.
- **Secondary:** σ\* has the highest lift I_π(trained) / I_π(initialisation), where the
  initialisation has constant g, so I_π(init) is the share of Σ m_f.
- Raw I_π and I_σ are reported as well.
- The spec's C2b criterion "top I_σ" is impossible by Prop. K (H0 derivation). This is noted in the
  report; the spec is not edited.

**C. What the C2 ladder isolates (for the report).** The ladder isolates only the **weight
channel**. Intermediates also act through:
- **states:** the cell couples every node to the cell mean. The `endpoint_edges` arm (Theorem R)
  tests this;
- **transports:** a cell's transports are products of edge maps conditioned on intermediates. They
  are present in every ladder arm.

The spec is not edited.

**D. S1, the card's parity task: expectation changed. Expected now: `full` ≥ 0.9 AND `endpoints`
≥ 0.9.** It demonstrates the transport channel, not the weight channel.
- **Derivation:**
  - A cell's P←Q coupling is w_f · T_Pᵀ T_Q, and T_Pᵀ T_Q is a product of the orthogonal-times-scale
    edge maps of (P, A) and (A, Q), each conditioned on its edge's two endpoints.
  - A Householder reflection of the class coordinate, conditioned on b_P ⊕ b_A on (P, A) and on
    b_A ⊕ b_Q on (A, Q), delivers Q's class to P multiplied by (−1)^{b_P⊕b_A⊕b_Q}.
  - Relevant instances (class y) arrive with sign + and irrelevant ones (class 1 − y) with sign −,
    so all of them vote for y.
  - The card's rank argument concerns positive per-edge weights, and is right for those.
- **Graph** (node types `paper`, `author`, `cpaper`; relations author→paper, author→cpaper;
  ℓ_max = 2, so every cell is P–A–Q):
  - 1000 targets P with bit b_P, label y (500/500), and 4 N(0, 1) noise features.
  - Each P has 2 authors A of its own (feature: bit b_A).
  - Each A writes P and two cpapers: Q (bit b_P ⊕ b_A, class y; relevant) and Q̄ (bit 1 ⊕ b_P ⊕ b_A,
    class 1 − y; irrelevant). Class marker one-hot in 2 dims, plus 4 noise features.
  - Each target therefore has 4 instances, 2 relevant and 2 irrelevant, and identical class
    counts.
  - Split 50/25/25, graph seed 0, split seed 1.
- **Arms:**
  - `full` and `oracle_weight` (w_f = 1 on relevant cells, 0 on the rest): 3 seeds each;
  - `endpoints`, `schema`, `uniform`, `no_path_cells`: 1 seed each;
  - diagnostic `endpoints` with `geometry_context: false`, 1 seed. Its weight sees only P and Q,
    with no context; if it passes, the parity is carried by the transports.
- **Budget:** ≤ 100 epochs; selection on validation accuracy; model block of `acm_h2snn.yaml` with
  `path_source: wisp`, `path_weight: exp`, `wisp_max_length: 2`, and `r` large enough to draw every
  instance (coverage reported).
- **Wrong if** `full` or `endpoints` < 0.9 on any of its seeds.
- Not gated: `schema`, `uniform` and `no_path_cells` are reported. I expect `schema` and `uniform` to
  pass too, since they have the same transports, and `no_path_cells` likely as well, through the
  2-hop fact route.

**E. S1′, the selection task for the weight channel.** It follows the author's specification
except for three deviations, each needed to avoid a leak that is certain or structural:
- **(i) n = 64 instances per target, 1 relevant (author: ~10, 1–2 relevant).**
  - With random classes on the irrelevant instances, the plain majority over all instances
    already scores Φ-level 0.62 (k = 1, n = 10) or 0.75 (k = 2, n = 10), above chance + 0.1.
    These are exact binomial values (H0 notes).
  - More generally, any fixed statistic that weights the relevant instance coherently and the
    irrelevant ones with class-independent signs has signal-to-noise ≈ k/√(n − k).
  - At n = 64, k = 1 this is 0.126, so the counting baseline is ≈ 0.55.
- **(ii) Relevance is parity over 6-bit vectors:** relevant iff b_P ⊕ b_A ⊕ b_Q = 0 in {0,1}⁶.
  - A single bit cannot make 1 of 64 instances relevant without a tell. Random bits make half of
    them relevant. Constrained bits make the relevant instance the odd one out among its author's
    co-papers, which the author's context mean exposes to a single edge.
  - With 6-bit vectors, the relevant instance is constructed and every irrelevant b_Q is uniform
    over the 63 values with XOR ≠ 0.
  - It is still a parity (per coordinate), and it does not factorise over edges.
- **(iii) One decoy co-author D_A per author, with bits ¬b_A,** writing all of A's cpapers. Each
  cpaper's context is then mean(h₀_A, h₀_D), whose pre-LayerNorm bit sum is constant. Without the
  decoy, the context of Q is h₀_A, and the `endpoints` weight would see b_A.
  - D is 3 hops from the target, outside ℓ_max = 2.
  - Residual risk: LayerNorm is nonlinear, so the sum of two normalised vectors is not exactly
    constant.

**S1′ graph:**
- 500 targets P with a 6-bit b_P (feature = the 6 bits as ±1) and 4 noise features.
- 8 authors per target, its own (feature: the 6 bits of b_A).
- 8 cpapers per author (features: class one-hot (2), the 6 bits of b_Q, 4 noise).
- One relevant instance per target, at a uniformly chosen author and cpaper: b_Q = b_P ⊕ b_A,
  class = y.
- The 63 others: b_Q uniform with b_P ⊕ b_A ⊕ b_Q ≠ 0, class uniform in {0, 1}, independent of y.
- 500 decoys. Split 50/25/25.

**Signal-to-noise argument for the controls** (`endpoints`, `uniform`, `schema`, `no_path_cells`):
- An instance's contribution to P is w_f · (product of positive per-edge scales) · (product of
  orthogonal edge maps) · Q's state.
- Orthogonal maps can rotate or flip, but they cannot suppress an instance. Positive per-edge
  scale products can suppress only factorised patterns, and the 6-bit parity indicator is not
  one: rank argument per coordinate.
- So, without an instance-conditioned w_f, P sees all 64 instances with weights that do not single
  out the relevant one.
  - Any single fixed linear statistic then has SNR ≈ 1/√63, an accuracy of ≈ 0.55.
  - Routing into orthogonal channels conditioned on bits (the transports' only selection
    mechanism) would need one channel per pattern of the 64 to be exact; d = 32 < 64, so it would
    carry crosstalk.
- **Stated limitation:** this bounds the transports only up to the channel capacity. A d-channel
  routing is not excluded representationally (as in D12 G3c). Whether the controls stay near
  chance is therefore empirical.

**S1′ criteria:**
- `full` ≥ 0.9 on every seed, OR, if it misses, `oracle_weight` ≥ 0.9 on every seed (capacity
  shown, learning not achieved at default settings).
- Each control is expected to be "near chance": ≤ 0.65 (counting baseline 0.55 + 0.1).
- A control "passes" if it reaches ≥ 0.9 (the pass level). Then 2 more seeds are run; if it still
  passes, the task leaks. It is redesigned and not reported.
- A control between 0.65 and 0.9 is reported as a miss of "near chance", with diagnosis.
- **My prediction:** `full` misses, because learning a 3-way 6-bit parity in g from one relevant
  instance in 64 is harder than D12 G3c's key match, which was not learned. The oracle passes, and
  every control is ≤ 0.65.
- **Budget:** ≤ 100 epochs, under 1 minute per run; `wisp_max_length: 2`; r large enough for full
  coverage (reported).

**F. S2 graph (fixed now).**
- **Node types:** target T (label y, 4 noise features); carrier C (class one-hot + 4 noise);
  intermediates M1, M2, M3 (4 noise each).
- **Relations (7):**
  - a: T–M1, b: M1–M2, c: M2–M3, d: M3–C;
  - distractors e: T–C, g: M1–C, h: M2–C.
- **Planted schema σ\* = a b c d (length 4).**
  - Each T has its own chain T–M1–M2–M3–C\* with class(C\*) = y.
  - Each T also has 4 e-edges to carriers of random class.
  - Each M1 and each M2 has 2 g/h-edges to random-class carriers.
- **Sampling:** `wisp_max_length: 4`, anchors T, r large enough to draw σ\* for every target
  (coverage reported). Under p, σ\* is rare, about 1/48 of a target's walks: length 1/3, then
  action choices.
- 600 targets; `full`, 3 seeds; ≤ 100 epochs.
- **Expectation:** primary (B) on every seed; secondary reported. Accuracy is reported as a
  functional check only.

## Correction 2026-11-08 (approved by the author; written before any H5 run)
- **S1′ decoy count:** "500 decoys" in addendum E is wrong. The rule there is one decoy co-author per
  author, so the graph has 500 × 8 = **4,000 decoys**. The rule is unchanged; only the count is
  corrected.
- **Run size:** the author asks for each H5 run to stay under about 1 minute. If a graph above makes
  a run take longer, its size is reduced (fewer targets; the per-target structure, the rules and
  the criteria stay as written). The final sizes and the measured run times are stated in the
  report.
- **S2 σ\* identification:** σ\* is the canonical full-walk key of the cells that are exactly a
  planted chain T_i–M1_i–M2_i–M3_i–C\*_i (either orientation). The script asserts that all such cells
  share one key. Ranks are by strict count (rank 1 = no schema scores higher).
- **Author override of the stop conditions (message of 2026-11-08):** a missed S1/S1′/S2
  expectation is recorded with numbers and diagnosis (with the oracle arm for S1/S1′); it does not
  stop the card. The leak rule for S1′ controls stays.

## Log 2026-11-09 (after results; recorded for the order of events)
- **Stop conditions:** the work stopped at the missed H3 gate (`ffb7295`). The author then lifted
  the H3/H6-gate stop conditions (message of 2026-11-08, after that commit) and asked for H4–H7 to
  continue with the safer defaults. That is a change of stop conditions made after a result. The
  gates, their thresholds and the default rule ("keep the safer default") are unchanged.
- **H6 `r` for DBLP and IMDB:** the first H6 runs used ACM's r = 42, against the rule above
  (lines 49–52). After the independent review (MAJOR-1), r was calibrated by the rule
  (`h6_calibrate_r.json`: DBLP 65, IMDB 12) and both runs were repeated. The repeats are the H6
  record; the r = 42 runs are kept in the JSON and marked as superseded in the report.
