# H2S-NN — WISP follow-up report (task card Part I, spec v1.4)

**Branch:** `wisp-followup`, starting at `94fc397`, with commits I0 `d2a086c`, I1 `a48889c`, I2 `0197d45`, I3 `1857016`, I4 `e08b51a`, I5 `928787f`, I6 `168fce6`, plus I7. It is not pushed.
`d12-encoder` (`19a9d36`) and `wisp` (`94fc397`) were pushed to `origin` as backups, with no merge. Evidence is in `docs/WISP_evidence/` (`followup_*`, `i1_*`…`i7_*`). Hardware: RTX 4060 8 GB.
Tags: **RUN** VERIFIED-RUN, **CODE** VERIFIED-CODE, **SRC** VERIFIED-SOURCE.
No number here is a performance result. There are no real-data accuracy runs, and the synthetic accuracies are pass/fail functional checks.

## 1. Summary
1. Implemented spec v1.4:
   - `g` reads `h₀` only (E1–E2), with the appendix switch `ablation.path_weight_context`;
   - the default `path_source: wisp` for node classification, with r per dataset (E7);
   - the test gaps the H7 review accepted;
   - overlap sampling: bit-identical, and the exposed sampling is **0.2 / 9.9 / 0.2 %** of epoch time on ACM / DBLP / IMDB, which **passes** the ≤ 25 % gate.
2. **I3, capacity not shown.** The learnable-scale oracle stays at chance on S1′ (0.48 / 0.47 / 0.52). Its relevant cell holds ≤ 0.14 % of its target's degree, and β moved ≤ 0.14. The capacity of the weight channel remains **UNVERIFIED**, so spec §6 WISP gate (ii) is **not met**.
3. **I3, no learned selection.** `full` learns no selection: the relevant − irrelevant log-weight gap is ≤ 0.02. Every arm memorises the 250 training targets by epochs 24–41.
4. **I4.** S2 without the length confound recovered σ\* on **1 of 3** seeds. The read-out finds the planted chain but not reliably its last relation.
5. **Mutations:** 11/11 targeted mutations are caught. **Full suite:** 355 tests; the only failures are the 24 pre-existing `test_baselines` ones (after an I6 regression caught and fixed at I7).

## 2. Files changed
- **Changed:**
  - `src/models/h2snn_model.py`: `path_inputs`; the path encoders read `h₀` (`path_input_dim`).
  - `src/configs/config.py`:
    - `path_weight_context`;
    - `path_source` and `r` resolution by task and preset;
    - validation in `__post_init__`;
    - `overlap_sampling`;
    - the tuning rejections.
  - `src/trainers/trainer.py`: the worker-thread prefetch, `draw_wait_seconds`.
  - `src/baselines/bundles.py`: pins `schemas`.
- **Tests:** `tests/test_wisp.py` (28 → 43 tests).
- **Pinned to `path_source: schemas`:** 15 test sites in 6 test files that pair a config with schema-cell fixtures, and `docs/heads_evidence/v2_golden.py`.
- **Evidence scripts:** `i3_s1p_oracle.py` (the oracle and the no-`h₀` readout live here, not in `src/`), `i4_s2_unconfounded.py`, `i6_overlap_cost.py`, and the mutation checks.
- **Not touched:** the author's uncommitted AGENTS.md/CLAUDE.md D2 rows. They still say "interim default `path_source: schemas`", which is now **stale**.

## 3. Deviations and decisions
- **Decisions fixed in the pre-registration (I0):**
  - **D-I1a:** the path latent reads `h₀` in every path mode, including `endpoint_edges`, `free_path_maps` and pair attention, whose path maps come from the latent.
  - **D-I1b:** `path_weight_context` only with `geometry_context` and `full` / `endpoints`.
  - **D-I2a:** an unset `path_source` resolves by task.
  - **D-I2b:** an unset `r` = 0 resolves to the dataset's calibrated value (42 / 65 / 12).
  - **D-I2c:** `schemas` is rejected in a node-classification tuning base config.
- **Inventory misses**, found by tests and fixed: the direct `path_terms` call in `test_v2_model.py:1132`, and the baseline bundles, whose untied + `dfs` pins conflict with the `wisp` default.
- **I3/I4 settings:**
  - `resample_paths: false` (as in H5, so cell relevance is fixed);
  - `log_every: 1000`, so the diagnostics run at epoch 0 only.
  - I4 kept the pre-registered n = 400 after the first run: its time (139 s) does not scale with graph size (Part H). Runs took 44–51 s (I3) and 109–139 s (I4, over the ~1 min budget).
- **I6:** 4 epochs instead of 2. Epoch 0 runs the logged diagnostics (6–19 s), so a lone epoch 1 would hide the draw behind an atypical epoch. The gate uses epochs 2–3; epoch 1 is reported too.
- **An I6 regression caught at I7.** My I6 affected set (`test_wisp`, `test_trainer`) missed two tests that build trainers by hand. The first full suite failed them (`i7_suite_first_run.txt`):
  - `fit` read `wisp_proposal` without resampling (stub config);
  - `training_draw` needed the new timing list (a trainer built without `__init__`).
  - Fixed in I7 by restoring the old contract: the timing lives in the epoch loop, and `wisp_proposal` is read only when overlapping. The bit-identity test and the I6 mutations were rerun (2/2), and the measured span is unchanged.

## 4. Verification evidence (RUN)
- **I1:**
  - `-k TH7` → OK, 6 rewritten tests (default context; outside nodes; ℓ−1 vs ℓ; appendix; capacity rule).
  - `i1_mutation_check.py` 2/2 caught, and the Part H endpoints mutation retargeted and caught.
  - Affected modules: 132 tests; after one fix of a keyword-only signature, OK (`i1_affected_tests.txt`).
- **I2:**
  - The first suite after the flip had 56 non-baseline failures, all fixture/config mismatches (`i2_affected_tests.txt`).
  - After pinning: 9 modules, **200 OK**, goldens bit-identical (`i2_rerun_tests.txt`).
- **I5:** 6 new tests OK; `i5_mutation_check.py` **6/6 caught**; 9 modules **206 OK**.
- **I6:** the bit-identity test OK; `i6_mutation_check.py` 2/2 caught (wrong seed, dropped proposal); `test_wisp` + `test_trainer` **44 OK**.
- **I7:**
  - first run: 355 tests, with 2 new failures (above);
  - after the fix: `python -m unittest discover -s tests` → `Ran 355 tests … FAILED (failures=11, errors=13)`, all 24 in `test_baselines` (`i7_suite_output.txt`).

## 5. Measurements
**I3: S1′, test accuracy** (`i3_results.json`, `i3_run_output.txt`; 500 targets, 1 relevant instance in 64, coverage 1.0):
| `full` ×3 | oracle `exp(b+β·o_f)` ×3 | `endpoints` | `uniform` | `no_path_cells` | `full_no_h0` |
|---|---|---|---|---|---|
| .480 / .472 / .528 | **.480 / .472 / .520** | .472 | .432 | .400 | .456 |

**The learning mechanism** (per-epoch log, evaluation draw; RUN):
- Every arm passes 0.9 training accuracy by epochs 24–41. The best validation epochs are 0–8.
- **Oracle:** b goes from −0.69 to −0.49…−0.53; β peaks at |β| ≤ 0.14 and ends at +0.007 / +0.139 / −0.022. The relevant cell's share of its target's `D` is ≤ 0.14 % (median), and the path part is 0.5–8 % of `D`.
- **`full`:** the path part peaks at 17–74 % of `D` (per seed) through a uniform rise of log w, up to 5.5. The relevant − irrelevant gap is ≤ 0.02, and the relevant cell's share of `D` is ≤ 1.1 %, about 1/64 of the path part.

**What this means for the weight channel** (DERIVED from these logs):
- To dominate its target, the relevant cell would need a factor of about 700–10,000, i.e. β ≈ 7–9.
- Adam moves a scalar by at most about lr per step (here 0.001 × scalar_lr_scale 10 = 0.01), so about 1 in 100 epochs even with a consistent gradient.
- After memorisation (epoch ≈ 25) the gradient is not consistent.
- So the test as specified cannot reveal capacity at default optimisation. The miss says nothing either way about representational capacity.
- The miss is structural, from the normalisation, `m_f ≈ 1/64` and the optimiser's reach. It does not show the channel to be useless.

**I4: S2, unconfounded** (`i4_results.json`; 11 full schemas, 6 of length 4):
| Seed | σ\* rank, mean log w | σ\* vs the top other | Lift rank | Test acc |
|---|---|---|---|---|
| 42 | 2 | 4.99 vs `f g h k` 5.16 | 2 | .55 |
| 43 | 2 | 8.21 vs `a b c e` 8.24 | 2 | .82 |
| 44 | **1** | 7.13 vs `a b c e` 7.00 | 1 | .91 |

- On seeds 43 and 44, σ\* and its sibling `a b c e` (same chain, other last relation) are lifted ×8.5–11, while every other schema is ≤ ×1. On seed 42 the model never learned the task.
- The `uniform` control ties all schemas (σ\* is rank 11), so the criterion is not vacuous.
- Max log-weight: 9.6 (I4), 5.9 (I3).

**I6: overlap** (`i6_overlap_cost.json`; seed 42; steady state = epochs 2–3):
| | ACM r = 42 | DBLP r = 65 | IMDB r = 12 |
|---|---|---|---|
| Exposed share, on / off | **0.2 %** / 34 % | **9.9 %** / 51 % | **0.2 %** / 33 % |
| Draw in the worker, s | 1.0–1.3 | 1.8–2.0 | 0.45–0.5 |
| Epoch, s (on / off) | 1.9 / 2.8 | 1.9 / 3.65 | 0.88 / 1.15 |

## 6. Self-review checklist (Part H's, re-applied where Part I touched it)
- `endpoints` sees no intermediate: **PASS**, now with no `c` channel for `g` (CODE `h2snn_model.py` `path_inputs`; T-H7 under the default context; mutation caught).
- The evaluation draw ignores `q` and the model: **PASS**, and now tested through `Trainer.fit` (MINOR-5).
- γ is gone under `exp`: **PASS**, also on direct construction (`__post_init__`).
- Every test fails when its property breaks: **PASS** for the 11 targeted mutations. T-H8 and T-H10 are still **NOT CHECKED**.
- Overlap changes no semantics: **PASS** (`torch.equal` on cells, `m_f`, logits and parameters, for both proposals).

## 7. Assumptions imposed for theory and reality checks
- **THEORY-CONSTRAINT** (the normalised operator and the D8 identity): a cell's influence on its target is its share of `D`, and `m_f` spreads a target's path mass over its distinct instances, about 1/64 each in S1′. One decisive instance among many therefore needs `g`'s range to reach about e^7–9.
  - What this may cost on real data: sparse, instance-specific evidence (one decisive co-author) is hard to learn at the default scalar learning rate.
  - The test is a capacity check with a scale-corrected oracle (§8).
- **Reality:** with 250 labelled targets, every arm memorises within about 25 epochs. Memorisation, not propagation, decides the S1′ outcome. The readout without `h₀` memorises too, so G0b's H1 is not the mechanism here.
- **The C2 ladder now isolates the weight channel** as spec v1.4 states: `g` reads instance nodes only (T-H7). Transports and states still carry intermediates in every arm.

## 8. Open items and items for the cluster
- **PROPOSAL (capacity, before spec gate (ii) can be met):** the oracle with `β₀ = log 64` (undoing `m_f`'s 1/64), or a fixed large β, or a larger scalar learning rate for this arm only; and S1′ with more training targets. Each is a new design, not run here.
- I4 with more seeds and longer training; the sibling-schema discrimination.
- The adaptive-`q` variance gate on a fully trained checkpoint (spec E5).
- T-H8 and T-H10 mutations.
- `scripts/profile_model.py` now profiles the `wisp` default.
- **Out of scope:** Amazon-Book LP exceeds 8 GB; the 24 `test_baselines` failures.

## 9. Decisions needed from the author
1. **WISP gate (ii) is not met** (capacity unverified). Choose:
   - (a) a scale-corrected oracle run before any real-data WISP run; or
   - (b) run C2 on the cluster and report the weight channel's capacity as unverified.
2. Update the D2 rows in AGENTS.md and CLAUDE.md: the default is now `wisp`.
3. Push `wisp-followup` after reading this report?
