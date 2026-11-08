# H2S-NN — WISP follow-up report (task card Part I, spec v1.4)

**Branch:** `wisp-followup` from `94fc397`: I0 `d2a086c`, I1 `a48889c`, I2 `0197d45`, I3 `1857016`, I4 `e08b51a`, I5 `928787f`, I6 `168fce6`, I7 `4a0999c` + review. Not pushed.
**Backups:** `d12-encoder` (`19a9d36`) and `wisp` (`94fc397`) are on `origin`, with no merge. Evidence: `docs/WISP_evidence/`. Hardware: RTX 4060 8 GB.
**Tags:** RUN, CODE, SRC, DERIVED, UNVERIFIED, PROPOSAL. There are no real-data accuracy runs, and the synthetic accuracies are pass/fail checks.

## 1. Summary
1. Done:
   - `g` reads `h₀` only (E1–E2);
   - the default `path_source: wisp` with r per dataset (E7);
   - the H7 test gaps;
   - overlap sampling: bit-identical, exposed sampling 0.2 / 9.9 / 0.2 %, which **passes** the ≤ 25 % gate.
2. **I3:** the learnable-scale oracle is at chance (0.48 / 0.47 / 0.52). As run, it **could not** show capacity. **WISP gate (ii) is not met**, and capacity is UNVERIFIED.
3. **I3:** `full` learns no selection (log-weight gap ≤ 0.02). Every arm memorises the 250 training targets by epochs 24–41.
4. **I4:** without the length confound, σ\* is recovered on **1 of 3** seeds. The read-out finds the chain, not reliably its last relation.
5. **Mutations:** 11/11 caught. **Suite:** 355 tests, only the 24 pre-existing `test_baselines` failures. **Independent review:** 0 blockers, 2 MAJOR, both accepted.

## 2. Files changed
- **`src/models/h2snn_model.py`:** `path_inputs`; the path encoders read `h₀`.
- **`src/configs/config.py`:**
  - `ablation.path_weight_context`;
  - `path_source` and `r` resolution by task and preset;
  - validation in `__post_init__`;
  - `data.overlap_sampling`;
  - tuning rejections.
- **`src/trainers/trainer.py`:** the worker-thread prefetch, `draw_wait_seconds`.
- **`src/baselines/bundles.py`:** pins `schemas`.
- **Tests:**
  - `tests/test_wisp.py`, 28 → 43 tests;
  - 14 test sites in 6 files, plus `v2_golden.py`, pinned to `path_source: schemas` (schema-cell fixtures).
- **Evidence scripts:** `i3_s1p_oracle.py` (the oracle and the readout without `h₀` live here, not in `src/`), `i4_s2_unconfounded.py`, `i6_overlap_cost.py`, and the mutation checks.
- **Not touched:** the author's uncommitted D2 rows in AGENTS.md and CLAUDE.md. They still say "interim default `schemas`", which is **stale**.

## 3. Deviations
- **D-I1a (spec deviation, review M2).**
  - In `endpoint_edges` and `free_path_maps`, the path-specific maps are built from `g`'s latent. That latent now reads `h₀` only, so those maps lost `c`. The pair-attention latents lost it too.
  - Spec R12 and §6 keep `[h₀, c]` for maps in every arm.
  - Only the four ladder arms are tested at the default context.
- **The pre-registered settings were not all met:**
  - I4: n = 400 is the script default, which the pre-registration does not name. Seed 43's σ\* coverage is 0.9975 (1 of 400 targets), against "every target".
  - I3/I4 use `resample_paths: false` and `log_every: 1000`.
  - Run times: I3 44–51 s; I4 109–139 s, over the ~1 min budget (per-epoch overhead, not graph size).
- **I6 timing:** 4 epochs instead of 2. Epoch 0's diagnostics take 6–19 s and would hide the draw, so the gate uses epochs 2–3; epoch 1 is reported.
- **Inventory misses**, caught by tests and fixed:
  - the direct `path_terms` call in `test_v2_model.py:1132`;
  - the baseline bundles (untied + `dfs`);
  - an I6 regression in two hand-built trainers. The first I7 suite caught it (`i7_suite_first_run.txt`). The fix: the timing moves to the epoch loop, and `wisp_proposal` is read only when overlapping.
- **Compatibility (review m3):** configs and checkpoints saved before this branch do not reload as they are. They need `path_source: schemas`, `wisp_walks_per_anchor: 0`, and for v1.3 widths `path_weight_context: true`.

## 4. Verification evidence (RUN)
- **I1:** 6 rewritten T-H7 tests at the default context (outside nodes, ℓ−1 vs ℓ, appendix control). 3/3 mutations caught. 132 affected tests OK after fixing one signature.
- **I2:** the first suite after the flip had 56 failures, all fixture/config mismatches. After pinning: 200 OK.
  - The goldens are bit-identical, but they cover only the pinned pre-v1.4 model. No golden covers the new default.
- **I5:** 6 tests; 6/6 mutations caught; 206 affected tests OK.
- **I6:** the bit-identity test (`torch.equal` on cells, `m_f`, logits and parameters, for both proposals) passes, with 2/2 mutations caught.
  - Its power against thread-count-dependent float sums at real-data size is UNVERIFIED.
- **I7:** `python -m unittest discover -s tests` → `Ran 355 tests … failures=11, errors=13`, all in `test_baselines` (`i7_suite_output.txt`).

## 5. Measurements
**I3: S1′** (`i3_results.json`; 500 targets, 1 relevant instance in 64, coverage 1.0), test accuracy:

| `full` ×3 | oracle `exp(b+β·o_f)` ×3 | `endpoints` | `uniform` | `no_path_cells` | readout without `h₀` |
|---|---|---|---|---|---|
| .480 / .472 / .528 | **.480 / .472 / .520** | .472 | .432 | .400 | .456 |

**Mechanism** (RUN, per-epoch log):
- All arms pass 0.9 training accuracy at epochs 24–41. The best validation epochs are 0–8.
- **Oracle:** β peaks at |β| ≤ 0.14 and changes sign (it ends at +0.007 / +0.139 / −0.022). The relevant cell is ≤ 0.14 % of its target's `D`, and the path part is 0.5–8 % of `D`.
- **`full`:** a uniform rise of log w (up to 5.5) takes the path part to 17–74 % of `D`. The relevant − irrelevant gap is ≤ 0.02.

**Why the oracle could not pass** (DERIVED, review M1):
- To dominate, the relevant cell needs about ×700–10,000 (β ≈ 7–9).
- Under the cosine schedule, Adam's total reach for β in 100 epochs is about Σ lr_t ≈ 0.5. So the pre-registered expectation was unreachable from β₀ = 0, which I should have computed at I0.
- β also used ≤ 0.14 of that reach and changed sign. A weak or inconsistent gradient is therefore at least as likely a cause.
- Which limit binds is **UNVERIFIED**. Capacity is neither shown nor refuted.

**I4: S2 without the length confound** (`i4_results.json`; 11 full schemas, 6 of length 4):

| Seed | σ\* rank, mean log w | vs the top other | Lift rank | Test acc |
|---|---|---|---|---|
| 42 | 2 | 4.99 vs `f g h k` 5.16 | 2 | .55 |
| 43 | 2 | 8.21 vs `a b c e` 8.24 | 2 | .82 |
| 44 | **1** | 7.13 vs `a b c e` 7.00 | 1 | .91 |

- On seeds 43 and 44, σ\* and its sibling `a b c e` are lifted ×8.5–11, and every other schema ≤ ×1.
- `uniform` ties every schema (σ\* is rank 11), so the criterion is not vacuous.
- Max log-weight: 9.6 (I4), 5.9 (I3).

**I6: overlap** (`i6_overlap_cost.json`; seed 42, epochs 2–3):
- **Exposed share, overlap on / off:** ACM **0.2 %** / 34 %, DBLP **9.9 %** / 51 %, IMDB **0.2 %** / 33 %.
- **Epoch time:** 1.9 / 1.9 / 0.88 s, against 2.8 / 3.65 / 1.15 s without overlap.
- **With contention included** (on-epoch − (off-epoch − draw); DERIVED, review m2): about 4 / 8 / 10 %. The gate passes either way; these come from a single run.

## 6. Self-review checklist (Part H's, where Part I touched it)
- **`endpoints` sees no intermediate:** PASS. `g` has no `c` channel; T-H7 runs at the default context, and the mutation is caught. The maps of `endpoint_edges` and `free_path_maps` deviate (§3).
- **The evaluation draw ignores `q` and the model:** PASS, now tested through `Trainer.fit`.
- **γ is gone under `exp`:** PASS, also on direct construction.
- **Every test fails when its property breaks:** PASS for 11 targeted mutations. T-H8 and T-H10 are NOT CHECKED.
- **Overlap changes no semantics:** PASS on the fixture, for both proposals.

## 7. Assumptions imposed for theory and reality checks
- **THEORY-CONSTRAINT** (the normalised operator and the D8 identity):
  - A cell's influence is its share of `D`, and `m_f` spreads a target's path mass over its distinct instances, about 1/64 each in S1′.
  - One decisive instance therefore needs `g`'s range to reach about e^7–9.
  - What this may cost on real data: sparse, instance-specific evidence is hard to learn.
- **Reality:**
  - With 250 labelled targets, every arm memorises within about 25 epochs.
  - The readout without `h₀` memorises too, so G0b's H1 is not the mechanism here.
- **The C2 ladder isolates the weight channel** (`g` reads instance nodes only). Transports and states still carry intermediates in every arm.

## 8. Open items and items for the cluster
- **PROPOSAL (capacity):** an oracle check that separates reach from gradient sign:
  - (i) β₀ = log 64, and a fixed large β;
  - (ii) the same with more training targets;
  - (iii) a larger scalar learning rate for the oracle only.
- **PROPOSALs from the review:**
  - two latents for `endpoint_edges` / `free_path_maps` (maps from `[h₀, c]`), and a T11 at the default context;
  - reject `wisp_max_length ≠ 4` and `adaptive` in a tuning base config;
  - limit `path_weight_context` to the ladder;
  - r per ℓ_max for the U3 curve.
- More I4 seeds; the sibling-schema discrimination.
- The adaptive-`q` gate on a fully trained checkpoint (E5).
- T-H8 and T-H10 mutations.
- **Card vs spec:** `r` is exempt from the tuning check by the Part H card but "never tuned" by spec §6.
- **Out of scope:** Amazon-Book LP exceeds 8 GB; the 24 `test_baselines` failures.

## 9. Decisions needed from the author
1. **WISP gate (ii) is not met.** Choose:
   - (a) a capacity check per §8 before any real-data WISP run; or
   - (b) run C2 and report the weight channel's capacity as unverified.
2. **M2:** keep the h₀-only latent for the `endpoint_edges` / `free_path_maps` maps (documented deviation), or implement the two-latent fix?
3. Update the D2 rows (the default is now `wisp`). Resolve `r`'s tuning status (card vs spec).
4. Push `wisp-followup` after reading?
