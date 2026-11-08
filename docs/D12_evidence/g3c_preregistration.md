# G3c — known-answer task for path-specific binding: pre-registration

Written 2026-11-08, before any G3c run and before the generator was executed (author decision 3 on
G1–G3b). Nothing below is changed after a result; additions get a dated heading.

**Question.** Does the per-instance cell weight `w_f` bind information carried by the intermediate
node of a path (the author's lead bit) to information at the far endpoint (the co-paper's class)?
The endpoint edges, the fact edges alone and h0 must not be able to do this.

## Common protocol (every design)
- **Pipeline.** Production pipeline end to end: fact layer, walk sampler, model, trainer. Only the
  dataset loader is replaced, as in G3b(b). Script: `docs/D12_evidence/g3c_binding.py`; results in
  `g3c_results.json`.
- **Graph.** Node types `paper` and `author`; one relation `(author, writes, paper)`, stored with its
  reverse `(paper, written_by, author)` and tied by the fact layer. Schema PAP only, so every cell
  has length 2. The graph is fixed (generator seed 0) across all runs.
- **Labels and split.** 1000 target papers with labels y ∈ {0, 1}, exactly 500 of each. Targets are
  split 50/25/25 with split seed 1: 500 train, 250 val, 250 test. Co-papers are unlabelled.
- **Class encoding.** The class appears only in the co-paper's feature dims 0–1, as one-hot(class).
  Targets have zeros there. No other feature of any node depends on any class.
- **Model and training.** Model block of `acm_h2snn.yaml`: d 32, d_r 16, K 4, hidden 128,
  `filter_bank`, `geometry_context: true`, cooperative gates. AdamW, lr 1e-3, cosine schedule, up
  to 200 epochs, early-stopping patience 100. The checkpoint is selected on validation accuracy;
  the reported metric is test accuracy at the selected epoch.
- **Path cells.** Walk sampler with `face_sample_k` (r) = 400 walks per anchor; the default is 50,
  raised so that every length-2 instance of a target is drawn (coverage is measured). The draw is
  fixed: seed = run seed, `resample_paths: false`.
- **Seeds.** 42, 43, 44. They vary the model initialisation and the cell draw.
- **Arms** (switches only, never tuned): `full` (defaults); `endpoint_edges`; `no_path_cells`;
  `h0_only_readout`; `uniform_path_gate` (w_f = 1 for every cell).
- **Pass criterion.** Chance = 0.5. **Pass iff** full test accuracy ≥ 0.90 on every seed **and**
  every run of endpoint_edges, no_path_cells and h0_only has test accuracy ≤ 0.60 (chance + 0.1).
  uniform_path_gate is reported, not gated; it tests whether w_f is what binds.
- **Leak rule.** If any control run (endpoint_edges, no_path_cells, h0_only) exceeds 0.60, the design
  leaks. No result is reported for it, only the leak evidence, and the task is redesigned. Runs are
  made control-first, in the order no_path_cells, endpoint_edges, h0_only, full, uniform_path_gate,
  with seeds 42, 43, 44 within each arm. A design stops at its first leaking control run.
- **False-leak rate.** With 250 test targets, a chance-level run exceeds 0.60 (≥ 151 correct) with
  binomial probability ≈ 6e-4; over 9 control runs, ≈ 0.6%.
- **Reported per run.**
  - Test and val accuracy, selected epoch, and train accuracy (train mode) at that epoch.
  - Coverage: the fraction of test targets whose relevant cell (P, L_P, Q\*_P) and distractor cell
    (P, N_P, Q'_P) are both in the draw.
  - For arms with a path gate (full, endpoint_edges): mean w_f at the selected checkpoint over the
    test targets' relevant cells, their distractor cells, and all cells.
- **If full fails while the controls stay near chance,** diagnose before anything else:
  1. train accuracy (fit vs no fit);
  2. coverage;
  3. w_f on relevant vs distractor cells;
  4. the uniform_path_gate result.

  Classify as bug vs. budget/learnability, with evidence.

## Design v1: the author's design, literal
- **Authors.** Each target P_i has two authors: L_i (lead = 1) and N_i (lead = 0); 2000 authors in
  all. Author features: one dim, the lead bit (1.0 / 0.0).
- **Co-papers.** Each author has exactly one co-paper. Q\*_i (class y_i) is written by L_i only;
  Q'_i (class 1 − y_i) by N_i only. The classes are complementary: a model that sees both
  co-papers' classes without knowing which author each came through sees {0, 1} for every target,
  so it is at chance.
- **Paper features** (10 dims): [one-hot class marker (2), N(0, 1) noise (8)]; targets have a zero
  marker.
- **Facts:** (L_i, P_i), (N_i, P_i), (L_i, Q\*_i), (N_i, Q'_i).
- **Label:** Label(P_i) = class of the co-paper reached through the lead author = y_i.

**Prediction, stated before running: v1 is expected to LEAK.** The lead bit is a property of a
single node, so binding needs no path cell:
- (a) The author's listen and broadcast gates are heads on [h0, c]. They can close the non-lead
  author: Prop. 6 gives exact zero blocks when g or b is 0. Every arm with diffusion has them.
- (b) The edge maps of (P, A) see the lead bit and A's context c_A. c_A is the mean over A's two
  papers, so it contains Q's class marker.
- (c) In the endpoint arm, Q's context is the h0 of its only author, so the endpoint edge (P, Q)
  sees the lead bit.

If v1 does not leak, v1 is the G3c result and v2 is not run. If v1 leaks, v2 is run.

## Design v2: contingent, run only if v1 leaks (fixed now)
**Principle.** No single node, no adjacent pair with its 1-hop means, and no endpoint pair with its
1-hop means determines the label. Only the triple (P, A, Q), which the path encoder sees, does.
- **Keys.** Each target P_i has a key k_i, drawn uniformly from the 64 codes of 6 bits and written
  as ±1 in paper feature dims 2–7. Keys are drawn independently of y, so they carry no label
  information.
- **Authors.** L_i (lead = 1) and N_i (lead = 0) for every target i; 2000 authors. Author features
  are the lead bit only: no noise, so no author's identity can be matched through context means.
- **Paper features** (16 dims): [one-hot class marker (2), key (6), N(0, 1) noise (8)]. Targets:
  [0, 0, k_i, noise].
- **Co-papers.** Every co-paper has exactly two authors: one lead-type and one non-lead-type.
  - Q\*_i: key k_i, class y_i, written by L_i and N_σ(i).
  - Q'_i: key k_i, class 1 − y_i, written by N_i and L_τ(i).
  - σ and τ are permutations with σ(i) ≠ i, τ(i) ≠ i, k_σ(i) ≠ k_i and k_τ(i) ≠ k_i.
  - Fillers: every author gets 14 more co-papers, so 16 in all; 14 000 fillers. Each filler is
    shared by one lead-type and one non-lead-type author, never L_i with N_i of the same target.
    Its class is chosen so that every author's 16 co-papers are exactly 8 of class 0 and 8 of
    class 1. Its key is uniform among the codes other than the keys of both its authors' targets.
- **Label:** Label(P_i) = class of the co-paper that is reached through the lead author and carries
  P_i's key = class(Q\*_i) = y_i.
- **Invariants asserted by the generator** (a failure is a generator bug, not a design change):
  1. every author has 16 co-papers, 8 per class;
  2. among L_i's co-papers exactly one (Q\*_i) carries k_i, and among N_i's exactly one (Q'_i);
  3. every co-paper has one lead-type and one non-lead-type author;
  4. no co-paper is written by both authors of one target;
  5. labels are 500/500.

**Why each control should fail** (the argument the run tests):
- **h0_only:** target features are key and noise only.
- **Node gates:** the lead bit identifies L_P, but L_P's 16 co-papers are 8/8 by class. Closing N_P
  therefore leaves a class-balanced set, and the key still has to be matched.
- **Context:**
  - c_P is the mean of {lead 1, lead 0}, the same for every target;
  - c_A is the mean of A's 17 papers, with keys and class markers summed separately, so there is no
    key–class binding;
  - c_Q is the mean of {lead 1, lead 0}.
- **Endpoint edges:** (P, Q) sees k_P and k_Q, so it finds Q\*_P and Q'_P, which both carry k_P and
  have complementary classes. The intermediate's lead bit is not in g_P or g_Q (both contexts are
  constant), so the endpoint arm sees a balanced pair.
- **Fact edges alone:** P's 1-hop neighbours are its two authors, which carry only the lead bit.
  Reaching y requires selecting, through the author's state (a sum over 16 co-papers), the co-paper
  whose key equals k_P. That is bilinear key routing through the edge maps of (A, Q) and (P, A).
- **full:** w_f is a one-hidden-layer function of [g_P, g_A, g_Q] (path encoder plus gate), so it can
  compute lead(A) ∧ (k_P = k_Q) directly. At the first Chebyshev order, P receives Q's state through
  the cell with weight w_f, and P's fact neighbours carry no class information.
- **uniform_path_gate (expected near chance):** with w_f = 1, the cells deliver every co-paper of
  both authors; the transport can separate by lead but not by key without the same bilinear
  routing.

**Stated limitation, before running.** The separation for the fact-edge route (and the transport
in uniform_path_gate) is not representational. The sheaf maps of (A, Q) and (P, A) are learned
functions of both endpoints. An edge model can therefore in principle route each co-paper's class
along a key-dependent direction and read it back at P with a key-dependent map. The crosstalk from
the author's other 15 co-papers is of order √(15/d) for a good code. The controls test whether that
route reaches the label under the same budget, not whether it is impossible. A control above 0.60
is a leak by the rule above, whatever the mechanism.

## Attempts
Part A caps this at three designs. If v2 leaks, a third design gets a dated addendum here before it
is run.

## Addendum 2026-11-08: diagnosis of the v2 full-model failure (written before these runs)
Observed before writing this addendum:
- v1 leaked on its first control run.
- v2 controls: 9 runs, all ≤ 0.564, so no leak.
- v2 full seed 42: test 0.504, a FAIL. Train accuracy rose to 0.94 while validation stayed at
  0.51–0.57. w_f means: relevant 0.164, distractor 0.065, all cells 0.090.

The verdict on v2 is unchanged by anything below: it is decided by the pre-registered criterion
on the three full seeds. The diagnosis classifies the failure as a bug downstream of w_f vs.
learnability of w_f.
- **D1, post hoc on the selected checkpoints (no training).** Mean w_f by category for the cells of
  train targets and of test targets, the categories being (lead, key match) = relevant, (non-lead,
  key match) = distractor, (lead, no match) and (non-lead, no match). A train/test gap in the
  relevant-vs-rest separation indicates memorisation of cells. Every paper carries 8 noise
  features, so cells are individually identifiable.
- **D2, oracle gate.** The full arm with w_f replaced by the oracle indicator: 1 on the relevant
  cell (P, L_P, Q\*_P) of every target (train, val and test), 0 on every other cell. Everything else
  is trained as in the full arm (same config, budget and seeds 42, 43, 44); `g3c_v2_oracle.py`.
  **Reading:**
  - oracle test accuracy ≥ 0.90 on every seed: transport, filter and readout carry the bound
    information when w_f selects, so the failure is in learning w_f (learnability or budget), not a
    bug downstream;
  - oracle < 0.90 on any seed: a bug or limitation downstream of w_f, to be localised before
    anything else.

D2 is a diagnostic, not a G3c result: the oracle uses the label rule.
