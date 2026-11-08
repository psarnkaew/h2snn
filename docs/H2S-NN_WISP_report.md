# H2S-NN — WISP report (task card Part H, D2)

`wisp` @ `19a9d36`+H0–H7, not pushed. Evidence in `docs/WISP_evidence/` (cited by file name). RTX 4060 8 GB, Windows, env `h2snn`.
Tags: **RUN** = VERIFIED-RUN, **CODE** = VERIFIED-CODE, **SRC** = VERIFIED-SOURCE.
Real-data accuracy is logged only, the spec §6 WISP gate is closed, and the H5 accuracies are synthetic checks.

## 1. Summary
1. WISP is implemented: an exact-`p` walk sampler, a walk-level IS mixture, `w_f = exp(g)` without γ, the C2 ladder and the read-outs. T-H1…T-H11 pass, and the card's mutations are caught 5/5.
2. T-H7 holds only with the geometry context off, and T-H2 checks a transport-free `L_f`; both deviate from the card (§3).
3. Gates: H3 variance missed (−4.9 % vs −20 %; `uniform` stays); H6 sampling missed (33 / 52 / 30 % on ACM / DBLP / IMDB vs ≤ 25 %); memory passed (1.02×).
4. H5: S1 met (the transport channel). S1′ missed: no selection learned, and my oracle was too weak to test capacity. S2 met as pre-registered but confounded by length, so it says nothing about discovery.
5. The independent review found no BLOCKER and 4 MAJOR items, all accepted (`h7_independent_review.md`, `h7_review_response.md`).

## 2. Files changed
- **New:** `src/data/wisp_sampler.py` (the sampler, IS ratio, merge), `src/analysis/wisp_readouts.py`, and `tests/test_wisp.py` (28 tests).
- **Changed:**
  - `config.py`: the new keys; `uniform_path_gate` removed.
  - `h2snn_model.py`: `exp`, the ladder, per-cell action embeddings.
  - `vertex_path_laplacian.py`: `gamma=None`.
  - `trainer.py`: the adaptive `q`.
  - `dataset.py`, `structures.py`: the `path_source` switch and signature fields.
  - Five test files and `v2_golden.py` pinned to `sigmoid_gamma`.
- **Not committed, as instructed:** the D2 rows of `H2S-NN/AGENTS.md:72` and `CLAUDE.md:734` → "WISP (spec v1.3 §3.2)", with the interim default `path_source: schemas`.

## 3. Deviations and author overrides
**Author (2026-11-08):**
- Reduced H3–H6 settings; the suite after H2, H6 and H7.
- S1 expectation changed; S1′ added; S2 criteria by mean log-weight, then lift.
- Decoys 500 → 4,000, before H5 (`0b77f2c`); the run-size rule.
- The H3/H6 gates stopped being stop conditions *after* the H3 stop (logged in `preregistration.md`, 2026-11-09).

**Mine:**
- **T-H7** perturbs the `[h₀, c]` input. The card's literal h₀ test holds only under `geometry_context: false`, which is tested with a context-on control. With the default context, `endpoints` sees v₁ and v_{ℓ−1} through 1-hop means, so C2 `full` vs `endpoints` is confounded through `c` (CODE `relational_context.py:28-44`).
- **T-H2** checks the estimator on `L_f = I − 11ᵀ/|f|`, not the transported `L̂_P`.
- **H5 settings:** `resample_paths: false`. S2 uses n = 400 instead of 600 (run-size rule, decided before its result); it did not shorten the runs (139–158 s), so the budget was missed.
- **H6:**
  - DBLP and IMDB first ran at ACM's r = 42, against the pre-registered rule. They were corrected after the review to r = 65 and r = 12 (`h6_calibrate_r.json`).
  - The ACM `full`/`schemas`, DBLP and IMDB runs have 3 timed epochs; the other ACM arms have 1.
  - The LP smoke did not run (§5).

## 4. Verification evidence
- `python -m unittest tests.test_wisp -v` → `Ran 28 tests … OK` (RUN `h4_affected_tests.txt`; the reviewer reran it: `Ran 28 tests in 70.893s OK`).
- `python ../docs/WISP_evidence/mutation_check.py` → 5/5 caught (RUN `mutation_output_all.txt`).
- `h4_mutation_check.py` → 5/5 read-out mutations caught (RUN `h4_mutation_output.txt`). The first version missed 2; I fixed it with a partial-mask test and a tested schema mask.
- `python -m unittest discover -s tests` after H6 and after H7 → `Ran 340 tests … FAILED (failures=11, errors=13)`, all in `test_baselines`, the same names as the pre-D7 baseline (RUN `h6_/h7_suite_output.txt`; SRC `H2S-NN_D10_report.md:27`).
- The D12 goldens are bit-identical under `sigmoid_gamma` + `schemas` (RUN, T-H6).

## 5. Measurements
**H3** (RUN `h3_variance_gate.json`; 5-epoch ACM checkpoint, 20 draws, 16 probes): V(adaptive) 963.7 vs V(uniform) 1013.8, −4.9 %. Max IS ratio 1.65.

**H5** (RUN `h5_results.json`, `h5_run_output.txt`; test accuracy; ≤ 100 epochs; every relevant instance drawn):
| Task | `full` ×3 | oracle ×3 | `endpoints` | `schema` | `uniform` | `no_path_cells` | Verdict |
|---|---|---|---|---|---|---|---|
| S1 parity, 4,000 cells | .996 / 1 / 1 | .996 / 1 / .996 | .996 | 1 | .996 | .996 | met |
| S1′ selection, 32,000 cells | .464 ×3 | .464 / .456 / .504 | .440 | .456 | .432 | .400 | **missed**, no leak |

- **S1′** (RUN `h5_epoch_extract.txt`, `h5_s1p_diagnosis.json`):
  - At epoch 99, training accuracy is 0.97–0.99 and validation 0.44–0.49: memorisation.
  - `full` relevant minus irrelevant mean log-weight: +0.007 / −0.022 / −0.010. No selection.
  - The oracle fixes `w_f = 1`. With `m_f ≈ 1/64` and `exp(b)` bypassed, its relevant cell is 0.014–0.050 % of the target's degree. Capacity is **UNVERIFIED**.
- **S2** (RUN `h5_results.json`):
  - σ\* is first by mean log-weight (10.1 / 8.6 / 8.7; the next ≤ 5.7), by lift (6.2–7.0×) and by `I_π` (0.89–0.995). It is 7th by `I_σ`.
  - **Confound (DERIVED, the reviewer's MAJOR-3):** σ\* is the only length-4 schema, and the encoder has a first layer per length. A length-only `g` would rank it first.
  - σ\* holds 1/7 of the initial mass, not the pre-registered "1/48".

**H6** (RUN `h6_smoke_cost.json`; seed 42; peak = `max_memory_allocated` in the training step):
| Run | r | Cells | Incid. | Sigs | Draw s | Train s | Epoch s | Draw share | Peak MB |
|---|---|---|---|---|---|---|---|---|---|
| ACM `schemas` | k = 50 | 134,374 | 403,122 | 2 | 2.05 | 1.70 | 4.25 | 48 % | 1,626 |
| ACM `full` | 42 | 103,103 | 411,257 | 136 | 0.94 | 1.40 | 2.82 | **33 %** | 1,661 |
| ACM `endp.` / `schema` / `unif.` / adaptive `q` | 42 | same | same | same | .94 / 1.24 / 1.10 / 1.24 | 1.39–1.67 | 2.8–3.4 | 33–38 % | 1,604–1,707 |
| DBLP `full` | 65 | 206,698 | 867,420 | 14 | 2.00 | 1.36 | 3.87 | **52 %** | 2,705 |
| IMDB `full` | 12 | 37,614 | 149,530 | 16 | 0.40 | 0.63 | 1.33 | **30 %** | 889 |

- ACM lengths 2 / 3 / 4: 34,055 / 36,148 / 32,900 cells; Σ m_f = 893 / 871 / 785 over 3,025 anchors (the shortfall is the dead-end mass).
- The superseded r = 42 runs (DBLP 48 %, IMDB 61 %) are kept in the JSON.
- **Smoke:** losses finite in every run. GC (MUTAG, `exp`) ran.
- **LP (Amazon-Book) did not run:**
  - The GPU ran out of memory, and so does the pre-Part-H `sigmoid_gamma` control. The CPU run segfaulted with 1.3 GB of RAM free (cause UNVERIFIED; RUN `h6_lp_oom_output.txt`).
  - The suite's LP test trains one step under `exp`. I judged this a resource limit, not a stop.
- **Max log-weight** over every epoch: ≤ 14.2 in H5 (RUN `h5_epoch_extract.txt`), ≤ 1.07 in H6. The stop level is 30.

## 6. Self-review checklist (my verdicts; the reviewer's are in its file)
- Exact `p(ω)` matching enumeration: **PASS** (T-H1: 200k draws, 4 s.e.; CODE `wisp_sampler.py:178-216`).
- Per-walk mixture with ½p + ½q: **PASS** (T-H3, with a per-step control; CODE `:156`, `:241`).
- The evaluation draw ignores `q` and the model, and is shared: **PASS in code** (`dataset.py:226-236`). T-H9's proposal half is vacuous (MINOR-5).
- γ gone under `exp`: **PASS** (no `θ_γ`; a non-default `gamma_init` is rejected, CODE `config.py:779`; `_scaled`, `vertex_path_laplacian.py:258`).
- `endpoints` sees intermediates only through `c`: **PASS** for the channel. The card's T-H7 fails with the context on (§3).
- Read-outs cannot change training: **PASS** (the no-import test; `delta_loss` restores state; CODE `h2snn_model.py:878`).
- Every test fails when its property breaks: **FAIL**. Ten mutations are caught, but there are gaps: MINOR-5, MINOR-6 (far-end off-by-one), T-H8 and T-H10.

## 7. Assumptions imposed for theory and reality checks
- **THEORY-CONSTRAINT:** `w_f > 0` in a symmetric normalised operator (Props. 2, 3 and 5 hold for `e^{±10}`, T-H5). It may cost the asymmetric influence the data suggest. Nothing is clamped.
- **The card's S1 rank argument** covers positive per-edge weights only. Transports carry parity: every S1 arm, `no_path_cells` included, reaches ≥ 0.996.
- **The C2 ladder isolates only the weight channel**, and with the default context the `endpoints` weight also sees intermediates through `c`.
- **C2b's "top `I_σ`" is impossible**, because `I_σ` is a share of pair *weight* (SRC spec `:103`; Prop. K). S2 shows rank 7.
- **Reality checks:** there are no duplicate facts (H0). ACM's draw holds 136 full-walk signatures, against 2 in the interim list.

## 8. Open items
- **Follow-up (author):** overlap sampling with training, i.e. draw the next epoch's cells on the CPU while the GPU trains. It changes no semantics and is not implemented.
- **PROPOSALs from the review:** a trainer-level T-H9; T-H7 at positions ℓ−1 and ℓ; config validation in `__post_init__`; tests for every never-tuned key; a transported T-H2.
- **UNVERIFIED:** S1′ capacity (an oracle `exp(b)·1[relevant]`); S2 discovery (non-leaf distractors). Kendall τ and `ΔL_σ` are tested but not applied.
- **Out of scope:** the host syncs in `build_operator`; Amazon-Book LP exceeds 8 GB.

## 9. Decisions needed from the author
1. **Default flip** (PROPOSAL, not made): `path_source: wisp`, with `wisp_proposal: uniform` and `path_weight: exp`.
   - **For:** an exact measure, all correctness tests pass, 2.2× faster draws than `schemas` on ACM, and equal memory.
   - **Against:** the sampling gate is missed on all three datasets; S1′ shows no learned selection; S2 does not show discovery.
   - **I propose** flipping after the overlap follow-up meets the 25 % gate.
2. C2 `endpoints`: run it with the context off, or accept and state the confound?
3. A follow-up card for the S1′ oracle with scale, a non-leaf S2, and the test gaps above?
4. The C2 and C2b spec wording. Push `wisp`?
