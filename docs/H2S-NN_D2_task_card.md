# H2S-NN Agent Prompt — Part E: task card D2 (meta-path discovery inside H2S-NN)

Use with **Part A** of `H2S-NN_agent_prompt.md` (role, evidence rules, self-correction loop, report format). Part A's rules apply unchanged.

**Design source:** `H2S-NN_D2_ampss_decision.md` (rev. 4) and `H2S-NN_D2_remaining_limits.md` (Prop. K, Theorem R, node metric).
**Prerequisite:** Part D (D7–D9) units U1–U6 merged. D2 builds on the fact layer, signed transport, the walk sampler, cell identity and `m_f`. If Part D is not done, stop and report.

---

## E.0 How to use this card iteratively

The work runs in units E0–E8 (§E.4). Every unit, and every later revision of the mechanism by any agent, must pass the **Requirements Contract** (§E.2).

- **Before starting a unit:** re-read §E.2 and §E.3. In the pre-registration, list which requirement IDs the unit touches.
- **After a unit:** fill the contract table (`PASS` / `FAIL` / `NOT CHECKED` + evidence) in the report. A unit with any `FAIL` in an A or B row is not done.
- **If you want to change the design** (for example, because S1 fails): first write which requirement the change serves and show that it violates none, including §E.3. If it needs a requirement relaxed, **stop and ask the author**; never relax one yourself.
- **Reports:** `docs/H2S-NN_D2_report_E<k>.md` per unit, in the Part A format, plus a running `docs/H2S-NN_D2_contract.md` with the latest status of every requirement.

---

## E.1 Goal

H2S-NN must find task-relevant meta-paths **without a predefined list**. Discovery is not a stage. It is part of the model's path operator:

```
Π_a      = all (relation, direction) sequences of length 2..ℓ_max valid on the type graph from anchor a's type
           (after the D7 fact layer; a sequence and its reversal are one pattern)
L_P      = Σ_a (1/|Π_a|) Σ_{π∈Π_a} E_{walk ~ p(·|a,π)} [ 1{simple} · L_f(walk) ]          # target operator
estimator: per batch draw m patterns uniformly from Π_a (m = |Π_a| when memory allows), r walks per
           (anchor, pattern), weight 1/(m·r) per walk, identical cells merged by summing weights → m_f
κ_{v,f}  = exp(g(ξ_f)) · softmax_{v∈f}( h(h_v, pos_v, ξ_f) )                              # participation
(L_f x)_v = κ_{v,f} (x_v − T_vᵀ z_f),   z_f = Σ_u κ_{u,f} T_u x_u / Σ_u κ_{u,f}               # O(L), D9 form
x ← x − η M⁻¹ (Δ₀ + Σ_f m_f L_f) x,   M = diag(μ_v D_v),  μ_v = 1 + softplus(q(h_v))
prediction = mean over T fixed-seed draws of the predicted probabilities
```

"Found meta-paths" = patterns ranked by `I_π`, their share of learned path mass, reported with CIs and a permutation null (E6).

---

## E.2 Requirements Contract (binding for every version of the mechanism)

Each row has an ID, the requirement, why it exists, and the check that proves it. Tests named `E-T*` are defined in §E.5.

### A. Operator and theory — violating any of these makes the math wrong
| ID | Requirement | Why | Check |
|---|---|---|---|
| A1 | Every sampled path is simple (no repeated node, endpoints included) | Prop. 1 (forced transport) needs it | E-T3, mutation |
| A2 | Every fact on a path is an edge of the same forward pass's `Δ₀` graph; each batch is a closed subcomplex | Restricted-sheaf lemma; mini-batch correctness | E-T3 |
| A3 | Cell identity = fact sequence in canonical orientation; duplicates merged by **summing** weights | Unbiasedness; no reversal duplicates | Part D T9, E-T4 |
| A4 | Every weight is per (node, cell), positive and finite: `κ_{v,f} > 0`, `m_f > 0`, `μ_v ≥ 1` | Prop. K, metric lemma | E-T6, E-T8 |
| A5 | The operator is symmetric PSD; `ker(Δ₀ + L_P) = ker Δ₀`; eigenvalues of the normalized update operator are real and in `[0, 2]` | Prop. 2, Prop. 3, Prop. K, metric lemma | E-T6, E-T7, E-T8 |
| A6 | `D` is built from exactly the operator's own terms (`m_f`, `κ`, `c_e`) | Stability bound | E-T7, mutation |
| A7 | Backward hops use the adjoint transport (D7) | Functoriality with direction | Part D T2 (re-run) |
| A8 | Link prediction: no held-out fact, in either orientation, on the message graph or on any sampled path, at train and evaluation | Leakage | Part D T11 (re-run with auto patterns) |

### B. Statistics — violating these makes the results uninterpretable
| ID | Requirement | Why | Check |
|---|---|---|---|
| B1 | The target operator is stated in code (docstring) and the estimator is unbiased for it | The model is defined by `L_P`, not by sampling luck | E-T5 (with control) |
| B2 | Inclusion probabilities are known and used: weight `1/(m·r)` per accepted walk; rejected walks contribute 0 | Unbiasedness | E-T5 |
| B3 | Every pattern in `Π_a` has positive sampling probability in every batch (`m/|Π_a| > 0`) | No silent exclusion = no hidden selection | E-T4 |
| B4 | All randomness is derived from the run seed: training draws `seed + epoch`; evaluation averages `T` draws with fixed seeds; there is **no separate selector or materialization seed** | No hidden seed choice ("lucky seeds") | E-T9 |
| B5 | Each seed reruns everything; no artifact (schema list, manifest, cache of selected paths) is reused across seeds | Reported variance includes discovery variance | E-T9, code review |
| B6 | Importance (`I_π`, `I_{v,π}`) is reported only with bootstrap CIs over seeds and a permutation null | Interpretability claims must be testable | E6 |
| B7 | Labels: training labels only enter the loss; validation only for early stopping and choosing `ℓ_max`; test never used for any choice | Leak-free protocol (R4 C7) | Code review + E-T10 |

### C. Discovery semantics — violating these contradicts the paper's thesis
| ID | Requirement | Why | Check |
|---|---|---|---|
| C1 | No predefined pattern list is needed; candidates come from the type graph only | Purpose of AMPSS | E-T2 |
| C2 | Scores see every node on the path (`ξ_f`, `h_v`, positions); no endpoint-only features | Thesis: intermediates matter | E-T11 (intermediate dependence) |
| C3 | One objective: everything is trained by the task loss. No auxiliary probe, no label propagation, no separate search model | v1 Stage I failure | Code review |
| C4 | No discrete selection: no top-K, pruning, thresholds, quotas, budgets per pattern, warm-up or anneal schedules, temperatures | Removes non-determinism and knobs | Code review + config audit |
| C5 | Per-node expressivity via participation `π_{v,f}` (softmax over the cell's nodes) | Author intuition: patterns matter differently per node | E-T6 (nesting), E6 report |

### D. Complexity and knobs
| ID | Requirement | Why | Check |
|---|---|---|---|
| D1 | Knobs are exactly: `ℓ_max` (model choice), `r`, `T` (precision, monotone), `m` (memory budget). Any other numeric constant must be a documented numerical guard, not a tuned value | Reviewer R3 Q3; author requirement | Config audit |
| D2 | Sampling cost `O(|anchors| · m · r · ℓ_max)`; independent of `|Π|` and of node degree; vectorized (no Python loop over nodes or anchors; a loop over the ≤ m drawn patterns is allowed only if measured as < 10% of sampling time) | Scalability (R1, R3, R4) | E-T12 |
| D3 | No dense `[M,d,d]`, `[F,d,d]` or `[F,L+1,d,d]` tensors (inherited from D4 T8) | Memory | D4 T8 (re-run) |

### E. Reporting (per dataset, every run)
`|Π_a|` per anchor type · `m` · `r` · `T` · cells / incidences / merged duplicates per batch · sampling ms per batch · s/epoch · peak memory · parameter count · `I_π` top-10 with CIs · path-mass share `tr(D_P)/tr(D)` · `I_ℓ` (share per length).

---

## E.3 Forbidden designs (do not implement, even as a "quick fix")
1. Scoring patterns by endpoint features, path counts, endpoint diversity, or (propagated) labels.
2. Any auxiliary model fitted separately to choose patterns (linear probes, super-nets, RL agents, LLM scorers).
3. Hard selection of any kind: top-K, greedy addition, pruning, thresholds on weights, per-pattern quotas or budgets.
4. DFS-with-quota or any sampler without a stated distribution.
5. Fixed selector seeds, frozen manifests, or caching discovered patterns across seeds or splits.
6. Schedules on the path term (warm-up, anneal, ramp); a sigmoid gate times a global `γ`.
7. Per-(node, cell) directed weights (`v` listens to `f` but does not speak to it). This breaks symmetry (A5).
8. Free per-incidence maps outside the `path_maps: "free"` ablation.
9. Adding knobs to rescue a failing test. If a test fails, follow §E.6.

---

## E.4 Units (commit each on a new branch `D2_discovery` after its gate passes; do not push)

| Unit | Content | Gate | Contract rows |
|---|---|---|---|
| **E0** | **Audit of v1 AMPSS (evidence only).** In a worktree of tag `v1-submitted`, run `scripts/diagnose_stage1_selector.py` (`seeds`, `permute`, `lp_only`, `random_k`) on DBLP, ACM, IMDB. Report the Jaccard of selected sets across 10 selector seeds, the permuted-label `task_gain` distribution, and the `lp_only` share. Time Stage I preprocessing. No code changes | Report written | — |
| **E1** | **Synthetic benchmark first** (before any model change, so the model cannot be fitted to it). Generator + recovery metric, §E.7 | Generator unit tests; spec frozen and committed **before E3** | — |
| **E2** | **Automatic pattern set** `Π_a`: (relation, direction) sequences on the type graph after the fact layer; reversal merge; counts per dataset | E-T1, E-T2 | C1 |
| **E3** | **Pattern-stratified sampler** extending D8: `m` patterns per batch, `r` walks each, weight `1/(m·r)`; `m = "all"` gives exact stratification | E-T3, E-T4, E-T5, E-T12 | A1–A3, B1–B3, D2 |
| **E4** | **Participation operator** `κ_{v,f}` replacing `γ·w_f`; `D` diagonal `Σ_f m_f κ_{v,f}(1 − κ_{v,f}/S_f)` | E-T6, E-T7, E-T11 | A4–A6, C2, C5 |
| **E5** | **Node metric** `μ_v` in the update | E-T8 | A4, A5 |
| **E6** | **Inference and discovery report:** `T`-draw averaging; evaluation-time stratified estimator of `I_π`, `I_{v,π}`, `I_ℓ`; bootstrap and permutation tooling | E-T9, E-T10 | B4–B7 |
| **E7** | **Theorem R control:** `path_mode: "kron_endpoint"` (each cell replaced by one endpoint edge, transport = holonomy, weight `κ_0κ_L/(κ_0+κ_L)`) | E-T13 | — |
| **E8** | Measurements (§E.8), synthetic S1 run (pre-approved, §E.7), contract file, report | §E.7 pass/fail recorded | all |

### Config (PROPOSAL; verify existing names first and reuse them, e.g. `face_sample_k` for `r`)
- `data.path_patterns: "auto" | "configured"` (default `"auto"`; `"configured"` = literature-list control only).
- `data.max_path_length` (`ℓ_max`).
- `data.pattern_sample_m: int | "all"` (default `"all"`).
- `eval.num_path_draws` (`T`).
- `model.path_weighting: "participation" | "gate"` (`"gate"` = current D4 `γ·w_f`, ablation only).
- `model.node_metric: bool` (default `true`).
- Ablations: `ablation.uniform_participation` (`π` uniform); `ablation.constant_strength` (`w` constant); `path_mode` gains `"kron_endpoint"`.
- None of the ablation keys may appear in a tuning search space (reuse Part D's check).

---

## E.5 Tests (`tests/test_d2_discovery.py`, unittest, float64 unless stated)
| ID | Check | Tolerance / rule |
|---|---|---|
| E-T1 | `Π_a` on hand-built type graphs equals a brute-force enumeration; a sequence and its reversal are one pattern; ACM P→P yields `cite+` and `cite−` as separate hops | exact |
| E-T2 | `path_patterns: "auto"` runs with **no** `MetaPathConfig` entries; `"configured"` reproduces the Part D cells bit-for-bit | exact |
| E-T3 | Every sampled path is simple and every fact is in the batch's `Δ₀`. **Negative control:** disabling rejection is caught | exact |
| E-T4 | Over ≥ 200 batches every pattern with instances is drawn (B3); empirical pattern frequency ≈ `m/|Π_a|` (pre-registered binomial test) | pre-registered |
| E-T5 | Unbiasedness: mean of `L̂_P x` over ≥ 2,000 draws vs the exact target by enumeration with walk probabilities, on small graphs with `m < |Π_a|` and `m = |Π_a|`. **Negative control:** weight `1/r` (missing `|Π|/m` factor) is biased | pre-registered relative error |
| E-T6 | Prop. K on random cells (port `check_kappa.py`): symmetric; PSD; sections in kernel; normalized `λ_max ≤ 2 + 1e-9`; pair identity `E_f = (1/2S) Σ κ_uκ_v ‖T_u x_u − T_v x_v‖²`; uniform `π` reproduces the D4 term up to the factor `n` | ≤ 1e-10 |
| E-T7 | Full model operator: `D` equals the block diagonal of the operator actually applied. **Mutation:** `κ` in `L_P` but not in `D` breaks `λ_max ≤ 2` on some fixture | ≤ 1e-10 |
| E-T8 | Node metric (port `check_schur_metric.py`): eigenvalues of `M⁻¹L` real, in `[0, 2]`; kernel unchanged; `μ ≡ 1` reproduces the no-metric model | ≤ 1e-9 |
| E-T9 | Determinism: same seed gives identical predictions; evaluation identical across calls; training draws differ across epochs; no seed other than the run seed is read (grep and runtime assertion) | exact |
| E-T10 | Protocol: no test-set index reaches the loss, early stopping or `ℓ_max` choice (instrumented run) | exact |
| E-T11 | Intermediate dependence: perturbing an intermediate node's `h` changes `κ` of its cells and the target's output through the path term. **Control:** under `kron_endpoint` the path-term contribution is independent of intermediates | > 1e-8 vs ≤ 1e-12 |
| E-T12 | Scaling: sampling time grows ~linearly in anchors (10× → ≤ 15×) and is flat in `|Π|` at fixed `m` (pre-registered); no Python loop over nodes or anchors (show the code) | pre-registered |
| E-T13 | Theorem R: `kron_endpoint` edge equals the Schur complement of the cell onto its endpoints (holonomy transport; weight `κ_0κ_L/(κ_0+κ_L)`) | ≤ 1e-10 |

**Mutation check** (each must be caught):
- `1/r` instead of `1/(m·r)`;
- duplicates dropped instead of summed;
- softmax over the whole batch instead of within the cell;
- `κ` missing from `D`;
- `μ < 1` allowed;
- endpoints-only participation;
- the evaluation draw reseeded per call;
- a pattern silently excluded when it has few instances.

---

## E.6 Iteration rules when something fails
1. **Diagnose the layer first**, in this order:
   - (a) **Sampler:** are the planted or expected patterns drawn? (coverage counts)
   - (b) **Signal:** with an **oracle** (only the planted pattern, uniform `κ`), does the model learn the task? If not, the problem is the operator or the optimizer, not discovery.
   - (c) **Scoring:** does `κ` move? (gradient norms; scalar parameter group per D4 follow-up)
   - (d) **Precision:** do results stabilise as `r`, `T` grow?
2. **Pre-approved remedies:**
   - increase `r` or `T` (precision);
   - fix optimizer parameter groups;
   - fix bugs found by (a)–(c) with a new regression test.
3. **Not allowed without the author** (§E.3): new knobs, priors over patterns, pruning, label-based scoring, schedules.
4. At most **3 diagnosed attempts** per failing gate, then stop and report: the failure, attempts, the layer identified, and the best hypothesis.
5. **If S1 fails after 3 attempts,** the paper must not claim discovery. Report it; the author decides between narrowing the claim and a design change that is argued against §E.2.

---

## E.7 Synthetic benchmark S1 (spec frozen in E1, before E3)
- **Types:** target `T`; others `A`, `B`, `C`, `X`; ≥ 6 relations, including one self-inverse.
- **Planted signal** (each class-relevant, label noise 10%):
  - **P1** `T–A–T`, homophilic (labels agree);
  - **P2** `T–B–C–T`, contrast (labels differ, heterophilic);
  - **P3** `T–A–X–A–T`, intermediate-dependent: label agreement flips with a binary feature of the `X` node. An endpoint view cannot represent it; Theorem R's control must fail on it.
- **Distractors:** all other patterns carry random labels.
- **Frequency:** vary the walk frequency of P2 so its probability under the walk-frequency measure is 0.5, 0.1 and 0.02 of the target's paths (tests rare-pattern exposure).
- **Pre-registered pass criteria (10 seeds; this run is pre-approved, CPU or single GPU, ≤ 30 min total):**
  1. P1, P2 and P3 each rank above the 95th percentile of the permutation null of `I_π` in ≥ 9/10 seeds, at all three frequencies.
  2. Full model accuracy > `kron_endpoint` control by more than 2 std on the P3-only variant.
  3. Full model ≥ the oracle with the planted list minus 1 std.
- **Real-data multi-seed runs (HGB, H2GB) are not pre-approved.** List them as the next step.

---

## E.8 Measurements (allowed without approval)
For DBLP, ACM, IMDB at d = 32, factorized k = 2:
- `|Π_a|` for `ℓ_max ∈ {2,3,4}`;
- cells, incidences, merged duplicates per batch;
- sampling ms per batch (CPU/GPU);
- one forward + backward: peak memory and wall time (5 repeats after 1 warm-up), parameter count.
- Pre-register the expected memory relative to the Part D numbers.

**Smoke run:** ACM, 1 seed, ≤ 20 epochs. Log loss, η, mean/max `κ`, mean/min `μ`, path-mass share `tr(D_P)/tr(D)`, `η·λ_max`, and `I_ℓ`. **No accuracy claims.**

---

## E.9 Adversarial self-review checklist (D2)
- [ ] Every row of §E.2 marked with evidence in `docs/H2S-NN_D2_contract.md`.
- [ ] Nothing from §E.3 exists in the diff (grep for `topk`, `prune`, `threshold`, `quota`, `warmup`, `anneal`, `selector_seed`, `label_propagation` in new code).
- [ ] Unbiasedness (E-T5) and Prop. K (E-T6) both have working negative controls.
- [ ] The model runs with zero configured meta-paths (E-T2).
- [ ] Intermediate dependence shown, and absent under `kron_endpoint` (E-T11).
- [ ] The S1 spec was committed before E3 (show the commit order).
- [ ] Every new config key validated; ablation keys excluded from tuning spaces.
- [ ] Each test would fail if its property were broken (mutation check, 0 vacuous).

## E.10 Stop conditions
Stop and report instead of continuing if:
- Part D is not merged;
- `|Π_a|` at `ℓ_max = 2` already makes `m = "all"` exceed memory on HGB;
- E-T5, E-T6, E-T7 or E-T8 fails after 3 diagnosed attempts;
- S1 fails after 3 attempts (§E.6);
- a requirement in §E.2 cannot be met without violating another;
- a needed change falls outside this card.
