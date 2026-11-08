# Responses to the H7 independent review (`h7_independent_review.md`), 2026-11-09
After the review I changed only the report, the pre-registration log and the evidence. I changed no
code and no tests. Proposed code and test fixes are left for the author.

**MAJOR-1 (DBLP/IMDB `r`): accepted, corrected.**
- I had run both at ACM's r = 42, against the pre-registered rule (preregistration.md:49-52).
- Calibration by the rule (`h6_calibrate_r.py/.json`):
  - DBLP r = 65: 867,420 incidences ≥ 862,788 (r = 64 gives 855,200).
  - IMDB r = 12: 149,530 ≥ 143,052 (r = 11 gives 137,939).
- Both were rerun at these values for 4 epochs (3 timed): `h6_smoke_cost.json` keys `dblp_full_r65` and `imdb_full_r12`.
- These reruns supersede the r = 42 numbers in the report. The r = 42 records stay in the JSON.

**MAJOR-2 (T-H7 under the default context): accepted; declared as a deviation.**
- The card's literal T-H7 holds only under `geometry_context: false`. The test perturbs the `[h0, c]` input instead, and the context-off case is tested with a context-on control.
- The checklist allows `c` as the one channel. Still, with the D12 default (context on), `endpoints` sees v₁ and v_{ℓ−1} through 1-hop means, so C2's `full` vs `endpoints` contrast is confounded through `c`.
- Decision for the author: run C2's `endpoints` arm with the context off, or accept the confound and state it.

**MAJOR-3 (S2 is confounded by length): accepted.**
- σ\* is the only length-4 schema in the draw, and the encoder has a first layer per length (`h2snn_model.py:229`). A length-only `g` would therefore rank σ\* first.
- My pre-registered rarity, "about 1/48", was wrong (DERIVED): σ\* is 1/24 of the walks and 1/7 of the initial mass (I_π_init 0.143).
- S2 is now reported as "met as pre-registered; uninformative about discovery". No control arm was run on S2.

**MAJOR-4 (report format and traceability): accepted.**
- The report is rewritten: a 5-line summary, status tags, commands with decisive output in §4, and a length cut.
- Newly committed evidence:
  - `h7_independent_review.md` (unedited);
  - `h5_epoch_extract.py/.txt`, the per-epoch curves behind "training accuracy 0.98" and the max log-weight across all epochs;
  - `h4_mutation_check.py`, the read-out mutation script;
  - `h7_suite_output.txt`, the full suite after H7.

**MINOR-5 (T-H9's proposal half is vacuous): accepted.**
- The guarantee is in code: the evaluation draw passes no proposal (`dataset.py:226-236`), and training draws are the only consumer of `q` (`trainer.py:56-77`). No test covers it.
- PROPOSAL: a trainer-level test where an adaptive training draw leaves `prepared_data.face_groups` unchanged.

**MINOR-6 (a far-end off-by-one would pass T-H7): accepted (DERIVED).**
- PROPOSAL: perturb positions ℓ−1 and ℓ separately in T-H7.

**MINOR-7 (γ settings): partly accepted.**
- A non-default `gamma_init` is rejected under `exp` (`config.py:779`).
- The default value 1.0 is accepted on purpose, because `save_config` writes every field (the H2 round-trip fix). The report now says "a non-default `gamma_init` is rejected".
- Direct `ModelConfig(...)` construction bypasses the check, and `wisp_*` keys are silently ignored under `schemas`. Both accepted. PROPOSAL: validate in `__post_init__`.

**MINOR-8 (never-tuned rejections are partly tested): accepted.**
- PROPOSAL: tests for every new key, and add `wisp_proposal` to the error message.

**MINOR-9 (two H5 facts were undisclosed): accepted.**
- H5 used `resample_paths: false`, so the S2 read-outs are on the training draw. This is now stated.
- The max log-weight over all H5 epochs is 14.16 (S2, seed 44; `h5_epoch_extract.txt`). The stop level of 30 was never reached. The report's "≤ 13.2" came from last epochs only and is corrected.

**MINOR-10 (T-H2 uses a scalar `L_f`): accepted.**
- T-H2 checks the estimator (the IS ratio and the merge) on the transport-free cell Laplacian, not the transported `L̂_P`. This is declared as a deviation.

**MINOR-11 (stop conditions changed after a result): accepted as a fact.**
- The author lifted the H3/H6-gate stop after the H3 stop (message of 2026-11-08, after `ffb7295`). It is now logged with that order in the pre-registration's dated correction log.

**MINOR-12 (single timed epochs): partly corrected.**
- The DBLP/IMDB reruns have 3 timed epochs.
- The ladder-arm and adaptive-`q` timings remain single-epoch, and the report says so.

**NITs: accepted.**
- The removed γ guard: under `sigmoid_gamma`, the model always builds γ (`h2snn_model.py:555`). A missing γ can therefore arise only from direct construction of `VertexPathSheafLaplacian` (VERIFIED-CODE).
- The host syncs in `build_operator` and the test-only code paths are listed as open items.

**Not verified by the reviewer, now checked:**
- `I_σ` is defined by pair **weight** in spec v1.3 §3.2 (`H2S-NN_methodology_spec.md:103`, VERIFIED-SOURCE), as implemented.
- The 24 `test_baselines` failures are recorded as pre-existing since D10 ("names identical to the pre-D7 baseline", `H2S-NN_D10_report.md:27`; `H2S-NN_D12_report.md:42`), VERIFIED-SOURCE.
