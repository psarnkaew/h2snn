# H2S-NN — WISP report (task card Part H, D2)

**Status (2026-11-09):** H0–H7 done on `wisp` (base `19a9d36`), not pushed. Evidence: `docs/WISP_evidence/`. Hardware: RTX 4060 8 GB, Windows, env `h2snn`.
Real-data accuracy is logged only, the spec §6 WISP gate stays closed, and the H5 accuracies are synthetic functional checks.

## 1. Summary
- WISP (spec v1.3 §3.2) is implemented:
  - a schema-free sampler with an exact target `p` and a walk-level defensive mixture (IS ratio ≤ 2);
  - `w_f = exp(g)` with γ removed, and the C2 ladder;
  - evaluation-only read-outs.
- T-H1…T-H11 pass. All 5 card mutations and all 5 read-out mutations are caught.
- **Gates:**
  - H3 variance gate **missed** (−4.9 % vs −20 %), so `wisp_proposal` stays `uniform`.
  - H6 sampling gate **missed** (33 / 48 / 61 % of epoch time on ACM / DBLP / IMDB vs ≤ 25 %).
  - H6 memory gate **passed** (1.02×).
- **H5:**
  - S1 met: every arm ≥ 0.996, which is the transport channel at work.
  - S1′ **missed**: `full` and the oracle are at chance. `full` learned no selection, and my oracle arm was too weak to test capacity.
  - S2 met on 3/3 seeds.

## 2. Files changed
- `src/data/wisp_sampler.py` (new): the sampler, `p`/`q`, IS ratio, merge, length groups. `structures.py`, `dataset.py`: signature fields and the `path_source` switch.
- `src/configs/config.py`: the new keys, `uniform_path_gate` removed, combination checks, never-tuned rejections.
- `src/models/h2snn_model.py`: `exp` weights, the ladder encoders, per-cell action embeddings. `vertex_path_laplacian.py`: `gamma=None`. `trainer.py`: the adaptive `q`.
- `src/analysis/wisp_readouts.py` (new): the read-outs.
- `tests/test_wisp.py` (new, 28 tests). Five existing test files and `v2_golden.py` are pinned to `sigmoid_gamma`.
- **Not committed, as instructed:** the D2 rows of `H2S-NN/AGENTS.md:72` and `CLAUDE.md:734` now read "WISP (spec v1.3 §3.2)", with the interim default `path_source: schemas` until decision 1.

## 3. Deviations and author overrides
- **Author (2026-11-08):**
  - reduced H3–H6 settings; the full suite after H2, H6 and H7;
  - gates and S1/S1′/S2 expectations are not stop conditions;
  - S1 expectation: `full` and `endpoints` ≥ 0.9; S1′ added;
  - S2 criteria: mean log-weight first, then lift;
  - decoy count 500 → 4,000, corrected before H5 (`0b77f2c`); the run-size rule.
- **Mine:**
  - Branch base `19a9d36` (approved).
  - S1′ design (n = 64, 6-bit parity, decoys).
  - S2 at n = 400 instead of 600, decided from timing before its result (`h5_sizing_note.md`). It did not shorten the runs (139–158 s), so the budget was missed.
  - H6 cost runs use 4 epochs (3 timed).
  - **The LP smoke did not run** (§5).

## 4. Verification evidence
- `python -m unittest tests.test_wisp -v`: 28 OK (`h4_affected_tests.txt`).
- Mutations: card 5/5 (`mutation_output_all.txt`) and read-outs 5/5 (`h4_mutation_output.txt`). Two read-out mutations were missed at first; I added a partial-mask energy test and a tested schema mask.
- Full suite after H6: 340 tests, 11 failures + 13 errors. The only failures are the 24 pre-existing `test_baselines` ones (`h6_suite_output.txt`).
- D12 goldens: bit-identical under `sigmoid_gamma` + `schemas` (T-H6).
- Independent review: a fresh sub-agent that saw only the diff, Part A and Part H. Its findings are kept unedited in `h7_independent_review.md`.

## 5. Measurements
**H3** (5-epoch ACM checkpoint, 20 draws, 16 probes): V(adaptive) 963.7 vs V(uniform) 1013.8, a −4.9 % change. Max IS ratio 1.65.

**H5** (test accuracy; ≤ 100 epochs; validation selection; all relevant instances drawn):
| Task | `full` ×3 | oracle ×3 | `endpoints` | `schema` | `uniform` | `no_path_cells` | Verdict |
|---|---|---|---|---|---|---|---|
| S1 parity (4,000 cells) | .996 / 1 / 1 | .996 / 1 / .996 | .996 | 1 | .996 | .996 | met |
| S1′ selection (32,000 cells) | .464 ×3 | .464 / .456 / .504 | .440 | .456 | .432 | .400 | **missed**; no leak |

- **S1′ diagnosis** (`h5_s1p_diagnosis.json`):
  - Training accuracy reaches 0.98 while validation stays at chance: the model memorises the 250 training targets.
  - `full` selects nothing. Relevant minus irrelevant mean log-weight is +0.007 / −0.022 / −0.010, and the relevant cell holds 1.4 % of the path degree, the uniform 1/64.
  - The oracle fixes `w_f = 1`. With `m_f ≈ 1/64`, and with the learnable scale `exp(b)` bypassed, its relevant cell is only 0.014–0.050 % of the target's degree. **Capacity was not tested.**
- **S2** (n = 400, 7 relation types):
  - σ\* is first by mean log-weight (10.1 / 8.6 / 8.7; the next is ≤ 5.7), by lift (6.9 / 7.0 / 6.2×) and by `I_π` (0.98 / 0.995 / 0.89, from 0.14 at the start).
  - It is **7th by `I_σ`** on every seed.
  - Accuracy 0.97 / 0.90 / 0.91.
  - Caveat: my distractor carriers are leaves, so the draw holds only 5 full schemas. "Rare among many" was not tested.

**H6** (seed 42; means over the timed epochs; peak = `max_memory_allocated` in the training step):
| ACM unless stated | Cells | Incid. | Sigs | Draw s | Train s | Epoch s | Draw share | Peak MB | Max log w |
|---|---|---|---|---|---|---|---|---|---|
| `schemas` (k = 50) | 134,374 | 403,122 | 2 | 2.05 | 1.70 | 4.25 | 48 % | 1,626 | 0.30 |
| `full` (r = 42) | 103,103 | 411,257 | 136 | 0.94 | 1.40 | 2.82 | **33 %** | 1,661 | 0.69 |
| `endpoints` / `schema` / `uniform` | same | same | same | .94 / 1.24 / 1.10 | 1.39 / 1.67 / 1.54 | 2.8 / 3.4 / 3.1 | 33–36 % | 1,707 / 1,659 / 1,604 | ≤ −0.25 |
| `full`, adaptive `q` | same | same | same | 1.24 | 1.52 | 3.24 | 38 % | 1,667 | 0.62 |
| DBLP `full` | 139,493 | 579,082 | 14 | 1.27 | 0.94 | 2.66 | **48 %** | 2,163 | 0.21 |
| IMDB `full` | 118,397 | 479,755 | 16 | 1.58 | 0.65 | 2.60 | **61 %** | 1,440 | 0.26 |

- **ACM lengths 2 / 3 / 4:** 34,055 / 36,148 / 32,900 cells; Σ m_f = 893 / 871 / 785 over 3,025 anchors. The shortfall from 1/3 per length is the discarded dead-end mass.
- **GC** (MUTAG, 1 epoch, `exp`): finite loss, max log w 0.25.
- **LP** (Amazon-Book) **did not run**.
  - The GPU ran out of memory, and so does the pre-Part-H `sigmoid_gamma` control. The CPU run segfaulted with 1.3 GB of RAM free (cause UNVERIFIED; `h6_lp_oom_output.txt`).
  - Substitute: the suite's LP test trains one step under `exp`.
  - I treated this as a resource limit, not a stop.
- **Max log-weight:** ≤ 13.2 in every run, far below 30.

## 6. Self-review checklist
- `p(ω)` uses exact counts and matches enumeration: **PASS** (T-H1: 200k draws, 4 s.e., exact discarded mass).
- The mixture is per walk, with ½p + ½q of the realised walk: **PASS** (T-H3; the per-step control exceeds 2).
- The evaluation draw ignores `q` and the model, and the arms share it: **PASS** (T-H9: 4 arms × 2 proposals; `proposal=None` in `dataset.py`).
- γ is gone under `exp`: **PASS** (T-H6: no `θ_γ`; `gamma_init` rejected; `gamma=None`; no constant multiplier).
- `endpoints` sees intermediates only through `c`: **PASS**. `c_v` is the per-(relation, side) mean of v's 1-hop neighbours' `h₀`, so v₁ and v_{ℓ−1} enter only through that mean. T-H7(b): context off leaves the weight unchanged; the context-on control changes it.
- Read-outs cannot change training: **PASS**. The module imports no training code (tested). `delta_loss` restores the mode and the stats. `last_path_weights` is written only in training mode.
- Every test fails when its property breaks: **PASS** for the 10 mutations. **NOT CHECKED** for T-H8 and T-H10.

## 7. Assumptions imposed for theory and reality checks
- **THEORY-CONSTRAINT:** `w_f > 0` in a symmetric normalised operator; Props. 2, 3 and 5 hold for `exp` over `e^{±10}` (T-H5). It may cost the asymmetric influence the data suggest (plan §10–11). Nothing is clamped.
- **The card's S1 rank argument** covers positive per-edge weights only. Transports carry parity: in S1 every arm, `no_path_cells` included, reaches ≥ 0.996.
- **The C2 ladder isolates only the weight channel.** Intermediates also act through states (`endpoint_edges`, Theorem R) and through transports, which are present in every arm.
- **C2b's "top `I_σ`" is impossible** by Prop. K. S2 confirms it: rank 7 on every seed, while σ\* is first by every frequency-free score. The spec is not edited.
- **Reality checks:**
  - No duplicate facts on ACM, DBLP or IMDB (H0).
  - The ACM draw holds 136 distinct full-walk signatures, against the 2 of the interim list.

## 8. Open items
- **Follow-up (author):** overlap sampling with training, i.e. draw the next epoch's cells on the CPU while the GPU trains the current one. It changes no semantics, is not implemented here, and is motivated by the sampling gate.
- **S1′ capacity is UNVERIFIED.** The check is an oracle with a learnable scale, `w_f = exp(b) · 1[relevant]`.
- S2 with non-leaf distractors: not run.
- Kendall τ and `ΔL_σ` are tested (T-H11) but not applied; they were not pre-registered for a run.
- **Out of scope:** Amazon-Book LP exceeds 8 GB under both weight modes. The 24 `test_baselines` failures.

## 9. Decisions needed from the author
1. **Default flip** (PROPOSAL, not made): `data.path_source: wisp`, with `wisp_proposal: uniform` and `path_weight: exp`.
   - **For:** an exact measure, all correctness tests pass, 2.2× faster draws than `schemas`, equal memory, and the S2 read-out works.
   - **Against:** the sampling gate is missed, and S1′ shows no learned selection.
   - **I propose** flipping once the overlap follow-up meets the 25 % gate. Alternatively, flip now, since the current default misses that gate by more (48 %).
2. A follow-up card for the S1′ learnable-scale oracle and an S2 with non-leaf distractors?
3. The C2 and C2b wording in the spec (yours to discuss with the co-author).
4. Push `wisp` after reading?
