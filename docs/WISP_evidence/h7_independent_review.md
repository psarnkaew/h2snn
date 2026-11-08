# H7 independent review (unedited)
Reviewer: a fresh general-purpose sub-agent, given only the diff `19a9d36..aeffa07` and Part A + Part H
of `docs/H2S-NN_agent_prompt.md` (2026-11-09). Its final report follows verbatim (only the
transport indentation removed). My responses are in `h7_review_response.md`, not here.

---

**Independent review of `wisp` against base 19a9d36, judged against Part A and Part H as modified by the recorded author overrides**

I found no BLOCKER. I checked the core math by reading the code, and it is correct:
- **Target and counts:** `p(ω)` is built from exact counts: degree minus visited half-edges, `wisp_sampler.py:178-216`.
- **Mixture:** there is one coin per walk (`:156`).
- **IS ratio:** it is `2/(1+q/p)` (`:241-245`), and `m_f = Σ ratio/r` uses the summed merge.
- **Weights:** `w_f = exp(g)` with no γ, and `b` is in the scalar group.
- **Evaluation draw:** it is built with no proposal (`dataset.py:226-236`).
- **Rules:** I found no clamping of `g` or `w_f`, no floors on `q`, and no schema list filtering candidates on the wisp path.

The problems are in the evidence, the test strength and the report.

## Findings (most severe first)

**MAJOR-1. DBLP and IMDB in H6 did not follow the pre-registered rule for `r`, and the report does not say so.**
- The pre-registration says DBLP (862,788 incidences) and IMDB (143,052) "use the same rule in H6 … r and the counts are reported" (patch:6913-6916).
- `h6_smoke_cost.py` overrides only `path_source` (patch:~6182-6183), so both datasets run with `r = 42`, the ACM default (`config.py:327`). No config in `configs/` sets any wisp key.
- Result (`h6_smoke_cost.json`):
  - DBLP: 579,082 incidences, 0.67× the schemas budget.
  - IMDB: 479,755 incidences, 3.35× the schemas budget.
- Report §3 does not list this, and §5 gives no `r` for these runs.
- Consequence: the "gate missed" verdicts (DBLP 48 %, IMDB 61 %) and the "flip now" argument in decision 1 rest on budgets that were not pre-registered. The IMDB draw cost is inflated by the 3.35× budget.

**MAJOR-2. T-H7 as written in the card fails under the default config, and this deviation is not declared.**
- The card's test: perturb an intermediate node's h₀, and the `endpoints` weight must not change.
- `test_wisp.py:426-441` instead perturbs the row of the geometry input `[h0, c]`.
- The author's own control (`test_wisp.py:~466-480`) asserts that the `endpoints` weight **does** change when an intermediate's h₀ moves under `geometry_context: true`, the default.
- Cause: `c` is the per-(relation, side) mean of a node's 1-hop neighbours' h₀ (`relational_context.py:28-44`). So v₁ and v_{ℓ−1} enter the endpoints' `g`.
- The redefinition was the implementer's own choice in the pre-registration (patch:6940-6948), not an author override. It is missing from report §3, and the Summary says "T-H1…T-H11 pass".
- Consequence: C2 `full` vs `endpoints` contrasts on HGB (default context) are confounded through `c`.

**MAJOR-3. S2 does not test schema discovery.**
- In `graph_s2` every distractor carrier is a leaf (patch:5300-5326). So the only length-4 walk that survives is σ*: there are 400 length-4 cells, exactly n, on every seed.
- `PathEncoder` has a separate first layer per length (`h2snn_model.py:229`). A `g` that depends only on length therefore ranks σ* first by mean log-weight.
- σ* is not rare. It holds 1/7 of Σm_f at initialisation (I_π_init 0.143, which matches my derivation of (1/24)/(7/24)). The pre-registration said "about 1/48" (patch:7179).
- No `schema` or `uniform` control arm was run on S2.
- The report's caveat mentions "5 full schemas" but not the length confound. "S2 met 3/3" and "the S2 read-out works" (decision 1) claim more than this shows.

**MAJOR-4. The report breaks the Part A format and has untraceable items.**
- The Summary is 13 lines; the limit is 5.
- Only 3 status tags appear in about 1,650 words, although every non-trivial claim must be tagged.
- §4 gives no command and decisive output per test.
- With two wide tables it is likely over 2 pages.
- It cites `h7_independent_review.md`, which is not in the diff.
- "Training accuracy reaches 0.98" (S1′, patch:2457) is in no committed file. The largest `train_acc_at_best_trainmode` is 0.744, and the per-epoch logs are not committed (patch:4947).
- The read-out mutation script is not committed; only its output is.
- There is no evidence of the full-suite run after H7 that the override requires.

**MINOR-5. The proposal half of T-H9 passes by construction.**
- `test_wisp.py:648-674` sets `wisp_proposal: adaptive`, but `build_node_classification_paths` never reads that key (`dataset.py:143-155`), and the test passes no proposal.
- The real guarantee, `trainer.py:96-99` together with `dataset.py:226-236`, is untested.

**MINOR-6. T-H7 would miss an off-by-one leak at the far end (DERIVED; I did not run it).**
- If `h2snn_model.py:287` used `[:, [0, length-1]]`, the `endpoints` arm would read v_{ℓ−1} instead of v_ℓ.
- That bug passes every T-H7 test: they perturb only position 1 (`test_wisp.py:428`) and position 0 (`:445`).

**MINOR-7. γ-related settings are only partly rejected.**
- `config.py:778` accepts an explicit `gamma_init: 1.0` under `exp` and silently ignores it.
- Constructing `ModelConfig(...)` directly bypasses the check.
- `wisp_*` keys are silently ignored under `path_source: schemas`.
- The report's §6 says "gamma_init rejected".

**MINOR-8. Never-tuned rejection is only partly tested.**
- The base-config rejections of `sigmoid_gamma` and `path_weight_input ≠ full` (`config.py:85-94`) have no test.
- The search-space rejection is tested for 2 of the 6 WISP keys.
- The error message (`:109-111`) omits `wisp_proposal`.

**MINOR-9. Two H5 facts are undisclosed.**
- H5 ran with `resample_paths: False` (patch:5341), so the S2 read-outs are computed on the draw the model trained on.
- H5 records the log-weight only at the last epoch (`log_w_max_last`, patch:5402). So "≤ 13.2 in every run" and the stop condition at 30 were never checked across H5's epochs.

**MINOR-10.** T-H2 uses a scalar stand-in, `L_f = I − 11ᵀ/|f|` (`test_wisp.py:86-95`), not the production `L̂_P` with transports that the card names.

**MINOR-11.** The override that H3/H6 gates are not stop conditions appears only in report §3. There is no dated entry in the pre-registration, although commit `ffb7295` shows the work had already stopped at the H3 gate miss: the stop condition was changed after a result.

**MINOR-12.** The H6 numbers for the ladder arms, adaptive `q`, DBLP and IMDB each come from a single timed epoch.

**NITs**
- The guard that raised when path terms existed without γ was removed (patch:1284-1285). Under `sigmoid_gamma` a missing γ now silently means γ = 1.
- `build_operator` now does two host syncs per forward on every path (`h2snn_model.py:880-883`) and adds new logged keys to existing configs.
- Code used only by test controls ships in production: `log_mix_step` is computed on every adaptive draw, and `merge="drop_duplicates"` is a production branch.

**Not verified**
- "Top I_σ is impossible by Prop. K" holds for the implemented pair-weight share (`wisp_readouts.py:127-140`). I could not check whether the spec defines I_σ by weight or by energy.
- I could not verify that the 24 `test_baselines` failures already existed at the base commit.
- I did not review the uncommitted D2-row edits in `AGENTS.md` and `CLAUDE.md`.

## Checklist

| Item | Verdict | Evidence |
|---|---|---|
| `p(ω)` from exact counts, matches enumeration | PASS | Degree minus seen half-edges (`wisp_sampler.py:178-216`); T-H1 compares log p with an independent enumeration to 1e-12; test passes. |
| Mixture per walk; ratio uses ½p + ½q of the realised walk | PASS | `from_q` is drawn once per walk (`:156`); `_importance_ratio` (`:241`); T-H3 matches the reference to 1e-12; per-step mutation caught (`mutation_output_all.txt`). |
| Evaluation draw ignores `q` and the model; arms share it | PASS (code) / weak test | `dataset.py:226-236` passes no proposal; `trainer.py:98-99` evaluates on `prepared_data`. T-H9's proposal half is vacuous (MINOR-5). |
| γ gone under `exp` | PASS, with a caveat | `theta_gamma` exists only under `sigmoid_gamma`; `_scaled` returns the path term unscaled when γ is None; T-H6. Caveat: `gamma_init: 1.0` is silently accepted (MINOR-7). |
| `endpoints` sees intermediates only through `c` | PASS, but the card's T-H7 fails in the default config (MAJOR-2) | `c` is the 1-hop per-(relation, side) mean (`relational_context.py:28-44`); the encoder reads positions `[0, L]` (`h2snn_model.py:287`). |
| Read-outs cannot change training | PASS | No `src.trainers` or `src.builders` import (tested); `delta_loss` restores mode and stats; `last_path_weights` is written only in training mode (`h2snn_model.py:878`). |
| Every test fails when its property breaks | FAIL | MINOR-5, MINOR-6 and MINOR-10; T-H8 and T-H10 have no mutation check. |

## Command run
`C:/Users/Acer/anaconda3/envs/h2snn/python.exe -m unittest tests.test_wisp` (from `H2S-NN/`) → `Ran 28 tests in 70.893s` / `OK`.

I also read the H6 full-suite output in the diff: `Ran 340 tests … FAILED (failures=11, errors=13)`, and every failure is in `test_baselines`. I ran nothing else.
