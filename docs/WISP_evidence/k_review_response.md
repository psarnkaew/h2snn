# Responses to the Part K independent review (`k_independent_review.md`), 2026-11-09
After the review I changed only the reports (capacity report; a correction appended under the Part I
erratum) and this file. I changed no code, tests or evidence data.

**MINOR-1 (the erratum says e^β ≈ 10 was "tested"): accepted.**
- K2 did not run e^β ≈ 10 at n = 2000 (grid: 0, log 64, log 512).
- At n = 500, log 8 stayed at chance. At n = 2000, log 64 passed on 2/3 seeds.
- A dated correction is appended under the erratum in `H2S-NN_WISP_followup_report.md` (not a rewrite). It says that K2 found the idealised estimate optimistic, and that only "β ≈ 7–9 is not required" is supported (log 512 = 6.24 passed 3/3).

**MINOR-2 (the summary picks favourable ranges and overreaches): accepted.** The summary now states:
- the gate passes at β = log 512, where the relevant cell holds about 0.87 of its target's `D`, and holds on 2/3 at log 64 (share about 0.45);
- the dispersion and probe ranges are given per β, including log 64 seed 43 (dispersion 0.69, probe 0.72);
- the checkpoint probe tracks the model's own accuracy, so it shows where the information sits in `u` after training, not an independent mechanism.

**MINOR-3 (the K2b route audit missed the relation-specific fact route): accepted.**
- d, e and k are distinct relations, so the fact maps (pair- and role-conditioned, D1/D12) can carry C\*'s class from M3 to T without any cell weight. `uniform` reaching 0.996 confirms that route.
- The report now records the miss and its consequence. S2 at n = 2000 meets the pre-registered criterion: `g` ranks σ\* strictly first on 3/3 seeds, by a weight ratio of only 1.15–1.24× over the sibling. But the task does not require selection.
- So C2b's S2 is "met as pre-registered on a task solvable without the weights". It is not evidence that selection by `g` is needed or useful.
- **PROPOSAL** for the cluster protocol: an S2 variant whose sibling carriers use the same relation type as σ\*'s last hop, so that only cell weights can separate them.

**MINOR-4 (E11's calibrated-r rule is enforced only in the tuning check): accepted as a decision for the author.** The card scopes it to the tuning base config, and the spec states it for every configuration. It is now listed in §9. Enforcing it everywhere would require the synthetic scripts to name a non-calibrated dataset.

**MINOR-5 (Part A format): accepted.** The report is rewritten:
- a 5-line summary;
- status tags;
- §7 with THEORY-CONSTRAINT entries and reality checks;
- the commands for the K1/K2/K2b/K3 measurements;
- the K2b max log-weight (10.3) added.

**NIT-6 (only the pair-attention cell-score latents read `[h₀, c]`; a stale comment):**
- The wording is fixed in the report: the self, edge and node-level scores read raw `h₀`, unchanged (D11 design).
- The stale comment at `src/models/h2snn_model.py:912` is listed as an open item. No code change after the review.

**NIT-7: accepted.** The mutation turns the `ValueError` into a `KeyError`. The test catches it, and it would also catch a silent fallback. Recorded as is.

**NIT-8: explained.** The ACM block appears twice in `k3_calibrate_r_output.txt` because ACM was re-run after the bound check was added, so that its JSON rows carry the bound. The values are identical.

**Could not verify, now addressed:**
- **AGENTS.md / CLAUDE.md:** the D2 line holds the Part H H0 edit plus K4's change. K4 replaced exactly the sentence "Until the H7 default decision: `path_source: schemas` (fixed literature presets) stays the default" with "Default (spec v1.4 E7): `path_source: wisp` for node classification; `path_source: schemas` (fixed literature presets) is the comparison row". It was an exact-string replacement asserted to occur once per file.
- **The 24 `test_baselines` failures:** recorded as pre-existing since D10 ("names identical to the pre-D7 baseline", `H2S-NN_D10_report.md:27`), and in every suite of Parts H and I (`h6_suite_output.txt`, `i7_suite_output.txt`). This is VERIFIED-SOURCE, not re-run at `bf7f333`.
