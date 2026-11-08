# Part K pre-registration (K0, 2026-11-09; written before any Part K code or run)
**Authority:** spec v1.6 (E8–E11, §0 rule 6, R12, §6), task card Part K, and the author's decisions of 2026-11-09:
- (1) gate (ii) by a fixed-β dose-response and a probe;
- (2) two latents;
- (3) `r` never in a search space; calibrate per ℓ_max;
- (4) K2b at n = 2000;
- (5) erratum and the D2 rows;
- (6) `wisp-followup` pushed (done: `origin/wisp-followup` = `bf7f333`).

Branch `wisp-capacity` starts at `bf7f333`. The preconditions were checked: branch `wisp-followup`, no `part-j` branch, only the old `tier0-screen` worktree, `src/` and `tests/` clean.

Nothing here is a performance result. Misses are recorded with the pre-registered reading.

## Optimizer facts used below (VERIFIED-CODE)
- `src/builders.py:44-54`: two groups.
  - The main group: lr = 0.001, AdamW weight_decay 1e-4.
  - The scalar group (`model.scalar_parameters()`, which includes `b` = `path_gate.bias`): lr = 0.001 × `scalar_lr_scale` 10 = **0.01**, weight_decay 0.
- `:69-77`: `CosineAnnealingLR(T_max=epochs)`, stepped once per epoch (`trainer.py:166-172`). One full-batch step per epoch, `clip_grad_norm_(…, 1.0)` (`trainer.py:208-213`).
- **Reach of one scalar in 100 epochs** (DERIVED): Σ_t lr_t = 0.01 · Σ_{t=0}^{99} (1 + cos(πt/100))/2 = 0.01 · (50 + ½ · 1) = **0.505**. The cosines cancel in pairs.

## K1: two latents (E10)
- **Change:**
  - Under `endpoint_edges` and `free_path_maps`, the latent that feeds the maps is computed by a second encoder (same class) from the geometry `[h₀, c]`. `g`'s latent stays on `path_inputs` (`h₀` by default).
  - The second encoder exists only when `g`'s input differs from the geometry, i.e. `geometry_context` on and `path_weight_context` off. Otherwise one shared encoder, as before, keeps the goldens (context off) bit-identical.
  - It is created last, after the filter bank, so every other parameter initialises as before.
  - Pair attention: its path encoder reads the geometry (input width `geometry_dim`).
  - The `full` cell arms are unchanged.
- **Tests (T11-ctx, default context, geometry rows injected through `path_terms(geometry, …, path_input=…)`):**
  - (a) the `endpoint_edges` maps change with an endpoint's geometry row, including its `c` part only, and not with an intermediate's;
  - (b) `g`'s weights do not change when only `c` changes (default), and do under `path_weight_context: true`;
  - (c) the `free_path_maps` maps change with the `c` part of any instance node;
  - (d) the pair-attention cell scores change with `c`.
- **Also:** the goldens are bit-identical (they pin `geometry_context: false`); T-H7 still passes; parameter counts before and after are reported.
- **Expected:**
  - all pass;
  - counts rise by one encoder in `endpoint_edges` and `free_path_maps` at the default context;
  - the pair-attention encoder's first layer widens from `d` to `geometry_dim` per node;
  - unchanged at context off.
- **Mutations:** the map latent fed `h₀` → (a) and (c) fail; `g` fed the geometry by default → (b) and T-H7 fail.
- **Wrong if:** a golden changes, or a test fails after 3 diagnosed attempts (both stops).

## K2: capacity dose-response on S1′ (gate ii)
**Set-up:**
- `h5_synthetic.graph_s1p(n)`, unchanged except n; the split is 50/25/25 (`_split`, seed 1), so n = 500 gives 250 training targets and n = 2000 gives 1,000.
- I3's config: `h5_synthetic.config`, the `acm_h2snn.yaml` model block, `path_source: wisp`, ℓ_max 2, r = 640, `resample_paths: false`, 100 epochs, validation-accuracy selection, `log_every: 1000`.
- **Memory:** if n = 2000 does not fit in 8 GB, the largest n ≤ 2000 that fits is chosen by a one-epoch fit check and written here before any gated run.

**The oracle:**
- `w_f = exp(b + β·o_f)`, with `o_f ∈ {0, 1}` from `relevance_of_cells`.
- **β:** a buffer, a tensor with `requires_grad=True` created on the run's device and registered as a buffer, never given to the optimizer.
- **b:** the existing `path_gate.bias`-style scalar is not used. The oracle registers its own scalar `b` (`nn.Parameter`, b₀ = log 0.5) in the scalar group.
- `g` is not used for weights; everything else trains; nothing is clamped.
- **Max log-weight:** b + β ≤ about −0.69 + 0.5 + log 512 = 6.05 < 30.

**Grid:**
- n = 500: β ∈ {0, log 8, log 64, log 512} × seeds {42, 43, 44};
- n = 2000: β ∈ {0, log 64, log 512} × {42, 43, 44};
- **context arm:** `full` (g learned, no oracle) at n = 2000, seed 42.

**Per-epoch log:**
- train/val accuracy, `b`;
- the median over targets of the relevant cells' share of the target's scalar degree `D` and of the path part's share;
- `∂L_train/∂β` at the run's β, read after each training backward, then reset (β.grad = None);
- for `full`, the mean log-weight of relevant minus irrelevant cells.

**Operator probe** (at initialisation and at the selected checkpoint, oracle installed with its β; it does not train the model):
- `u = A h₀ − h₀` with `A x = apply_operator(operator, x)` from `build_operator` on the evaluation draw, model in eval mode.
- **Features:** `u` restricted to target papers (d = 32), standardised with the training targets' mean and standard deviation (std 0 → 1).
- **Model:** logistic regression with a bias, minimising mean BCE + (λ/2)·‖w‖² with **λ = 1e-3** (the bias is not penalised).
- **Solver:** `torch.optim.LBFGS(lr=1, max_iter=500, history_size=50, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")`, float64, w₀ = 0.
- Fit on the training targets; report accuracy on the test targets (threshold 0.5).
- **Control:** the same probe on `h₀` of the target papers. Solver and λ are fixed here and never tuned.

**Dispersion of the transported carrier states** (reported if the probe is < 0.9):
- For every relevant cell f (target P, relevant cpaper Q\*, label y_f), let `t_f = T_Pᵀ T_{Q*} h₀(Q*)`: Q\*'s state carried into P's stalk by the cell, per unit weight. It is computed with the production `PathTransportOperator` (`apply`, then `apply_T` with only P's position non-zero).
- Let `ŝ_f = (2y_f − 1)·t_f/‖t_f‖`. **Dispersion = 1 − ‖mean_f ŝ_f‖ ∈ [0, 1]**.
  - 0: every relevant cell pushes its target along one class-signed direction, which a linear probe can read.
  - 1: no common direction.
- Reported at initialisation and at the selected checkpoint, with the relevant share of `D`.

**Reachability (rule 6):**
- β is constant, so the selection needs no learned scale.
- `b` moves at most about 0.505 and shifts only the total path mass (all cells by e^{±0.5}); it cannot change the relevant/irrelevant ratio e^β.
- What the expected result needs to learn: the filter (`W_k`), the head, and transports that keep a common class direction across relevant cells. These have many parameters at lr 0.001 (cosine); Part I's `g` (also many parameters) raised log w by about 4.4 in the same schedule. So the change is reachable in principle. Whether it is learned is the test.

**Information-route audit, from `graph_s1p` and the model, per arm:**
- **Graph facts:**
  - target P: 6 bits b_P (±1) + 4 N(0,1) noise; label y independent of b_P;
  - 8 own authors A: 6 bits b_A, no noise;
  - one decoy D per author (in type "author"): the complement bits, writing all of A's 8 cpapers;
  - cpapers Q: class one-hot (2) + 6 bits + 4 noise.
  - Relevant Q\*: bits b_P ⊕ b_A, class y. The other 63 per target: bits ≠ b_P ⊕ b_A, class uniform and independent of y.
  - Facts: A–P (writes), A–Q and D–Q (writes_c).
- **Receptive field:** the filter bank has K = 4 (`x = Σ_{k≤4} T_k(A − I) h₀ W_k`).
- **Routes for y to reach P:**
  1. **Own features:** none. **Memorisation:** P's 4 noise dims + 6 bits are unique per target, against 250 (n = 500) or 1,000 (n = 2000) training targets, so every arm can fit training labels without generalising.
  2. **Facts:** Q's class arrives at 2 applications (P–A–Q), mixed over 64 cpapers (1 with class y, 63 random): a mean statistic has SNR ≈ 1/√63, about 0.55 accuracy. Selecting Q\* by the 6-bit 3-way parity needs pair-conditioned transports to route; Part H/I controls stayed ≤ 0.65. Decoys add b_A's complement at 3 hops (not the label).
  3. **Cells:** P–A–Q cells (all 64 instances per target, coverage 1.0 in I3) couple P and Q in one application, with weights w_f m_f.
     - Oracle: relative weight e^β for Q\*. Idealised SNR ≈ e^β/√63, assuming the head reads the one-application cell channel, transports keep the class direction, and the irrelevant classes are independent.
     - `full`: g reads h₀ of P, A, Q and the relation tokens; it can express the parity only by learning it (not shown in H5/I3).
     - β = 0: the oracle equals uniform weights, with no selection.
- **Leak check:** β = 0 is the arm with no selective route besides learning. If it reaches ≥ 0.9, that is a leak (card), and no gate verdict is given.

**Expected** (card, DERIVED, idealised): accuracy ≈ 0.55 / 0.84 / ≥ 0.99 / ≥ 0.99 at β = 0 / log 8 / log 64 / log 512, for the probe at the checkpoint and for the model at n = 2000. At n = 500 the model may fall short through memorisation (reported, not gated).
- **My prior** (UNVERIFIED): the probe at initialisation may stay low, because random pair-conditioned transports scatter the class direction (dispersion near 1). At the checkpoint, learned maps can align it.
- At n = 2000, I expect the model to pass at log 512 on 3/3 seeds and β = 0 to stay ≤ 0.65. log 64 is less certain.

**Gate (ii)** (card, decision 1): passes iff at n = 2000 the model is ≥ 0.9 on 3/3 seeds at β = log 64 or at β = log 512, and β = 0 is ≤ 0.65 on every seed.

**If it misses, read the probe** (fixed now):
- probe ≥ 0.9 at log 512 while the model is < 0.9 → the information arrives; the limit is learning in the filter or head;
- probe < 0.9 → it does not arrive in one application; report the relevant share of `D` and the dispersion, and stop at the diagnosis.

**`∂L/∂β` reading** (fixed now; not a gate): per β, the fraction of epochs before the best epoch with −∂L/∂β > 0.
- Consistently positive → a learnable scale would move the right way given enough reach.
- Mixed or negative → learning β from the loss is unreliable at this size.

## K2b: S2 (I4 graph) with 1,000 training targets
**Set-up:**
- `i4_s2_unconfounded.graph_s2u(n=2000)` (split 50/25/25 → 1,000 training targets); I4's config and read-out unchanged.
- Arms: `full` × seeds {42, 43, 44}, `uniform` × seed 42. ≤ 100 epochs.
- **`r` before training:** start at I4's r = 200. For each seed's evaluation draw (draw seed = run seed), check σ\* coverage over the 2,000 targets; if any is < 1.0, raise r (300, 400, 600, …) until every draw has coverage 1.0. The chosen r is used for all four runs and written below before training. This is set-up from the draw only.
- **Per-epoch log:** train/val accuracy; the mean log-weight of σ\* minus `a b c e` (evaluation draw, eval mode); the best epoch.

**Reachability:** I4's `full` reached log-weights of about 8–10 (`i4_results.json`), so the range exists. The criterion needs only σ\* − sibling > 0 in mean log-weight: `g` must use the last relation (d vs e) or the end node, both in its input.

**Information-route audit, from `graph_s2u`:**
- **Graph facts:**
  - target T: 4 noise dims;
  - chain nodes M1, M2, M3, N1, N2, N3: 4 noise each, one set per target;
  - carriers: class one-hot + 4 noise. C\*_i (class y), Ce_i and Ck_i (random classes).
  - Facts: a T–M1, b M1–M2, c M2–M3, d M3–C\*, e M3–Ce, f T–N1, g N1–N2, h N2–N3, k N3–Ck, m M2–N3.
- **Routes for y to reach T** (K = 4):
  1. **Own features:** none. **Memorisation:** T's 4 noise dims are unique per target, against 1,000 training targets.
  2. **Facts:** C\* is 4 fact hops from T (a b c d), reached at K = 4. Ce (a b c e) and Ck (f g h k, a b m k) are random-class carriers at the same distance. The route through N3–M2 to C\* is 6 hops, beyond K.
  3. **Cells:**
     - σ\* cells (T…C\*) carry C\*'s class to T in one application.
     - `a b c e` and `a b c` cells reach C\* only through the fact d from M3, in ≥ 2 applications.
     - `f g h k`, `a b m k`, `a b m h`, `f g h m` and the shorter schemas have no route to C\* except through T.
- **Per arm:** `full`: g reads h₀ of every cell node and the relation tokens. σ\* and `a b c e` differ in the last relation (d vs e) and in the end carrier (both carry a class one-hot; relevance is not visible in it). `uniform`: constant w, so all schemas tie.

**Expected (card):** the primary criterion (σ\* strictly highest mean learned log-weight per cell among full-walk schemas) is met on 3/3 `full` seeds; `uniform` is not strictly first.
- **My prior** (UNVERIFIED): better than I4's 1/3, but uncertain.
- **Wrong if** any `full` seed ranks another schema ≥ σ\*. Then C2b's S2 is reported as "not shown at default settings" and moves to the cluster protocol with the same criterion. It is never loosened.

## K3: guards
- **Rules:**
  - (1) a search space naming `wisp_walks_per_anchor` is rejected;
  - (2) a node-classification tuning base config with `wisp_max_length ≠ 4`, `wisp_proposal: adaptive`, or `r` ∉ {0, the dataset's calibrated value at ℓ_max 4} is rejected;
  - (3) `path_weight_context: true` with `endpoint_edges`, `free_path_maps` or `coupling: pair_attention` is rejected;
  - (4) `r` is calibrated per dataset for ℓ_max ∈ {2, 3, 5, 6} by `h6_calibrate_r.py`'s rule (the smallest r whose evaluation-draw incidences, seed 42, reach the schemas-run budget 403,122 / 862,788 / 143,052), stored next to the presets. An unset `r` resolves from the table for its ℓ_max; an ℓ_max without a value raises;
  - (5) NIT: consumers (`dataset.py:143`, `build_model`) raise on an unresolved `path_source == ""`. I choose raising over silent resolution, because a directly built config has no task to resolve against.
- **Decision:** rule (2)'s "calibrated value" is applied in the tuning check only. Spec E11's "a configuration carries 0 or exactly the calibrated value" is read in the protocol context. The synthetic scripts (dataset name `acm`, r = 640 or 200) set their own r and are not tuning configs. Stated in the report.
- **Tests:** one per rule. Each fails under a targeted mutation. `test_wisp.py:249` and `:1202` change (authorised by E11).
- **Expected:** pass. The table is monotone in ℓ_max for each dataset (UNVERIFIED).

## K4, K5
- **K4:** the dated erratum (two items, as carded) appended to the Part I report; the D2-row default statement in AGENTS.md and CLAUDE.md (uncommitted).
- **K5:** the report and the full suite once. Expected: only the 24 pre-existing `test_baselines` failures.
