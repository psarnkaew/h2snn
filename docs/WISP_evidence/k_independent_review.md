# Part K independent review (unedited)
Reviewer: a fresh general-purpose sub-agent, given only the diff `bf7f333..25cab8a` (focused and
full) and Part A + Part K of `docs/H2S-NN_agent_prompt.md` (2026-11-09). Its final report follows
verbatim (only the transport indentation removed). My responses are in `k_review_response.md`.

---

## Independent review: Part K (`wisp-capacity`, bf7f333..25cab8a)

I found no BLOCKER and no MAJOR issue. The code changes, the K2 gate verdict and the K2b criterion all hold up against the evidence. The findings below are about how results are worded and about scope.

### Findings (most severe first)

**MINOR-1: the erratum calls a threshold "tested" that the evidence does not support.** `docs/H2S-NN_WISP_followup_report.md:138` says the idealised estimate "needs only e^β ≈ 10 for 0.9 accuracy. This was tested in Part K, K2".
- K2 never ran e^β ≈ 10 at n = 2000; the grid there was {0, log 64, log 512}.
- At n = 500, β = log 8 gave 0.47–0.52 (`k2_run_output.txt:6-8`).
- At n = 2000, e^β = 64 failed on seed 43 (0.698).
- The K5 report itself says the estimate "undershoots the finite-sample need" (`capacity_report.md:102`).

The part of item 2 that says "7–9 is sufficient, not required" is supported, because log 512 = 6.24 passed 3/3. Failure scenario: a later card takes ≈ 2.3 from the erratum as the log-weight gap `g` must learn. Fix: add that K2 found the idealised estimate optimistic.

**MINOR-2: the summary picks favourable ranges and overreaches.** `capacity_report.md:10` gives "dispersion 0.33–0.52; probe 0.96–0.996".
- The 0.33 is a log 64 value (s42 0.335), while the probe range is the log 512 range.
- log 64 s43 (dispersion 0.691, probe 0.724) is left out.
- The checkpoint probe follows the model's own accuracy (0.724 vs 0.698; 0.956 vs 0.988). It is not independent evidence of "how" the model works.

`:9` says "if a cell's weight is right, the model uses the one decisive instance". That is shown only at e^β = 512, where the relevant share of D is 0.87. At log 64 (share 0.45) it held on only 2/3 seeds.

**MINOR-3: the K2b route audit is incomplete, and the report does not draw the consequence.** `k_preregistration.md:135,140` treat Ce and Ck as random carriers at the same distance as C*. They do not note that d, e and k are distinct relation types. Relation-specific fact maps can therefore carry C*'s class with no cell weight at all. `uniform` reached 0.996 (`k2b_run_output.txt:24`), which confirms that route exists.

The σ*-over-sibling log-weight margin of 0.14–0.21 is a weight ratio of only 1.15–1.24×. The criterion passes as pre-registered: I checked that `strict_rank` and `i4.readout` are unchanged and that the numbers match `k2b_results.json`. But the report (`:11`, `:104`) neither records the audit miss nor states what this means for C2b's S2 status. Failure scenario: S2 is later cited as "g recovers the planted schema" on a task that needs no selection.

**MINOR-4: the E11 calibrated-r rule is enforced more narrowly than the spec states.** `src/configs/config.py:104-116` enforces "0 or the calibrated value" only in the tuning check, and only at ℓ_max 4.
- Spec E11 says "a configuration carries 0 … or exactly the calibrated value" and requires recalibration for every ℓ_max of the U3 curve.
- Example: a U3 config at ℓ_max 3 with an explicit r = 42 (the ℓ_max-4 value) passes `DataConfig.resolved()` silently.
- This matches the card, which scopes the rule to the tuning base config, and it is documented (`:32`, `:110`). It is not listed under §9 "Decisions needed".

**MINOR-5: the report does not fully follow the Part A format.**
- About 1,580 words plus 4 tables, which is over the 2-page limit.
- The Summary runs past 5 lines.
- Claims are not tagged one by one (evidence rule 1).
- §7 contains no THEORY-CONSTRAINT entries or reality checks.
- No commands are given for the K1/K2 measurements.
- The K2b maximum log-weight (10.31) is missing; only K2's 5.8 is reported (`:65`).

**NIT-6:** `:12` says "the pair-attention scores read [h₀, c]". Only the cell-score latents do; the self, edge and node-level cell scores still read raw h₀ (`src/models/pair_attention.py:177-180`). The code follows the card. The comment at `src/models/h2snn_model.py:912` ("attention scores from h0; path latents from the shared encoder") is now out of date.

**NIT-7:** The K3 mutation "uncalibrated l_max does not raise" still raises, just a `KeyError` instead of `ValueError` (`k3_mutation_output.txt:7`). So it does not remove the behaviour it targets. The test (`assertRaises(ValueError)` at acm ℓ_max 7) would also catch the real hazard, a silent fallback to the ℓ_max-4 value.

**NIT-8:** The ACM block appears twice in `k3_calibrate_r_output.txt` (lines 3–7 and 18–22).

**Could not verify:**
- **AGENTS.md / CLAUDE.md:** each differs from HEAD by exactly one D2-row line, and neither is in the branch diff. Whether K4 changed only the default statement, as opposed to the author's earlier uncommitted text, cannot be told without a snapshot.
- **The "24 pre-existing failures":** all 24 are in `test_baselines` (I tallied them). I did not confirm they also fail at bf7f333.
- **The K3 r table:** I did not re-run it, because calibrations are not allowed.

### Units

| Unit | Verdict | Evidence |
|---|---|---|
| K0 | OK | The pre-registration (f73c48f, 05:38) comes before the K1 code (05:51). The K2 script and the n = 2000 addendum (13b322f, 05:52:18) come before the first gated run (metrics.jsonl created 05:52:45). K2b set-up (06:45:21) comes before the first K2b run (06:45:45). Neither script changed after its runs. Reach checks out: scalar lr 0.001×10 = 0.01, `CosineAnnealingLR(T_max=100)`, patience 100 (`config.py:1081`), so Σ lr_t = 0.505. Observed movement of b was ≤ 0.36. Gap: the K2b route audit (MINOR-3). |
| K1 | OK | `path_terms` receives the geometry as `h0` (`h2snn_model.py:919`); `map_encoder` reads it and `g` reads `path_inputs`. The pair-attention latents read the geometry (`:869`). T11-ctx passes, and the golden T1 passes bit for bit. My in-memory mutations were all caught. Parameter counts match the JSON. |
| K2 | OK | β is a buffer and is not in the optimizer. b is in the scalar group. Nothing is clamped. There is one backward pass per epoch (`trainer.py:206-212`), the diagnostics run under `@no_grad`, and the gradient is read and reset in the callback with 0 missing values. The probe matches the pre-registration (standardised features, LBFGS, λ = 1e-3, bias not penalised). There is exactly one relevant cell per target (relevant cells = n), so the per-node sum in `apply_T` equals t_f. Verdict correct: log 512 gives 0.992/0.988/0.996, and β = 0 gives 0.506/0.512/0.472, so there is no leak. Every number in the report table matches the JSON. |
| K2b | ISSUE (minor) | r = 200 was chosen from the draws before training. The criterion is strict and unchanged, and the numbers match. The caveat is underweighted (MINOR-3). |
| K3 | OK / minor | Each of the five rules has its own test, all pass, and the mutation evidence shows 9/9 caught. The table matches the JSON and the config. The DBLP ℓ_max 2 bound is sound: it counts all walks (revisits allowed) over the adjacency (`k3_calibrate_r.py:41-50`), WISP walks are self-avoiding (T-H1), and there are no more cells than walks. 511,887/3 ≈ 170k length-2 walks fits DBLP's size. E11 scope: MINOR-4. |
| K4 | OK / minor | The erratum is appended with no lines deleted. The `filter_bank.py:83,91` citations are correct. Item 2 wording: MINOR-1. |
| K5 | ISSUE (minor) | 365 tests run; all 11 failures and 13 errors are in `test_baselines`. Format and wording issues: MINOR-2, MINOR-5. |

**Rules and stop conditions:** no clamping, and no comparison arm became a default. `link_prediction_dataset.py` is untouched and there is no `part-j` branch. The maximum log-weight was 10.3 (limit 30). No stop condition should have fired.

### Commands run (decisive output)
- `python -m unittest -v tests.test_v2_model.TwoLatentContextTests tests.test_wisp.WispGuardTests tests.test_cooperative.SymmetricIsThePreviousModelTests tests.test_wisp.WispLadderTests tests.test_wisp.WispConfigValidationTests tests.test_wisp.WispSamplerTests` → `Ran 35 tests … OK`. This includes `test_T1_symmetric_reproduces_the_golden_outputs_bit_for_bit ... ok`.
- Scratch script with in-memory mutations; no repository file was touched (`scratchpad/review_k/chk/k1_power*.py`):
  - map latent without c → `failures=4`
  - g sees c by default → `failures=3`
  - pair-attention latents without c → `failures=1`
  - clean → `failures=0`
- Scratch build of `free_path_maps`: both encoders get gradients (1440 and 5536 parameters). `uniform` combined with `endpoint_edges` or `free_path_maps` is rejected by the config, so no encoder is left unused.
- JSON summaries of `k2_results.json` and `k2b_results.json`: 100 epochs per run, no missing dL/dβ values, at n = 500 training accuracy reaches ≥ 0.98 while validation stays ≈ 0.5 (memorisation confirmed), and the K2b ranks and log-weights match the report.
- `git log` and `git show --stat` per commit, plus the creation times of each run's metrics.jsonl: the ordering above holds.
- `git diff -U0 -- H2S-NN/AGENTS.md H2S-NN/CLAUDE.md` → one D2-row line each. `git diff bf7f333 HEAD -- docs/H2S-NN_WISP_followup_report.md` → 0 deleted lines.
- `grep` of the FAIL/ERROR lines in `k5_suite_output.txt` → all 24 in `test_baselines`.
