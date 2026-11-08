# I7 independent review (unedited)
Reviewer: a fresh general-purpose sub-agent, given only the diff `94fc397..4a0999c` (focused and
full) and Part A + Part I of `docs/H2S-NN_agent_prompt.md` (2026-11-09). Its final report follows
verbatim (only the transport indentation removed). My responses are in `i7_review_response.md`.

---

## Independent review: branch `wisp-followup` (94fc397..4a0999c), Part I

There are no BLOCKER findings. The I1 code is correct for the four ladder arms. The I6 overlap does not change semantics, as far as I can tell from the code. Two unit goals carry MAJOR issues.

### MAJOR

**M1. I3: the oracle could never pass, and the diagnosis claims more than the evidence shows.**
- **The test could not pass.** The pre-registration (`followup_preregistration.md:65`) expects β to grow until the relevant cell, starting from about 0.03 % of `D`, dominates. The report derives that this needs β ≈ 7–9 (report:79). The settings make that impossible:
  - scalar lr = 0.001 × 10 (`builders.py:54`), with no weight decay;
  - `CosineAnnealingLR(T_max=epochs)` (`builders.py:77`; the H5 config uses `scheduler: cosine`);
  - 100 epochs from β₀ = 0.
  - So Adam's total reach is about Σ lr_t ≈ 0.5, not the "about 1" in report:80, which ignores the cosine schedule. The miss was predictable at I0 from numbers already in the pre-registration. The card's goal, testing capacity "with a correct oracle", is not delivered, for the second time after H5.
- **Overstated cause.** Report:83 says "The miss is structural, from the normalisation, m_f ≈ 1/64 and the optimiser's reach."
  - β used at most 0.14 of its reach and reversed (`i3_run_output.txt:1745-1747`): s42 peaked at +0.128 and ended at +0.007; s44 ended at −0.022.
  - So β's gradient was weak or changed sign before memorisation (epochs 24–41). The optimiser's reach is not shown to be the binding limit.
  - "Needs to dominate" is an assumption, not a measurement.
  - No control was run (β₀ = log 64, or a larger scalar lr), so the cause is UNVERIFIED and should be tagged PROPOSAL/UNVERIFIED.
- **Failure scenario:** the author picks §9 option (a) with β₀ = log 64, believing reach is the problem. If the gradient sign is the problem, the run fails again.

**M2. I1 / D-I1a: the `endpoint_edges` maps (and part of `free_path_maps`) no longer read `[h₀, c]`.**
- **What the code does:**
  - `h2snn_model.py:794` builds `latent` from `path_input` (= `h₀`).
  - `:802` builds `EdgePairTerm(self.endpoint_maps(latent), …)`. That arm's maps are now functions of `h₀` only.
  - `:804`: the `free_maps` latent input also lost `c`.
  - `:840-841`: the pair-attention latents also lost `c`.
- **Why it conflicts:** spec R12 (spec:48) says "maps and gates keep `[h₀, c]`". Spec §6 (spec:174) says geometry_context is on "for maps and gates in every arm". The author's decision restricts `g`'s `h₀`-only input to the *ladder* arms.
- **How it is reported:** the report files this under "decisions" (report:37), not deviations.
- **No test at the default context:** T11 runs only at `geometry_context=False` (`test_v2_model.py:1121`), and T-H7 covers only the four ladder arms.
- **Alternative:** two latents (maps from `[h₀, c]`, `g` from `h₀`) would satisfy both clauses. It was not considered.
- **Failure scenario:** a v1.4 rerun of the D11 `endpoint_edges` arm differs from both v1.3 and the spec, which confounds the comparison.

### MINOR

**m1. The never-tuned guard is incomplete, and a new test locks in the gap.**
- Spec:174 lists ℓ_max, r and the proposal as "pre-registered, never tuned".
- `test_wisp.py:1202` asserts that `data.wisp_walks_per_anchor` passes the search-space guard ("r is calibrated").
- The base-config check (`config.py:49-104`) accepts `wisp_walks_per_anchor`, `wisp_max_length ≠ 4` and `wisp_proposal: adaptive`.
- Sweeps are limited by the tuning script's whitelist (`run_h2snn_wandb_tuning.py:59-75`), but its base-config path (`:149`) lets these through.
- `path_weight_context: true` is also accepted with `endpoint_edges` / `free_path_maps` / pair attention (`config.py:954`). That goes beyond the spec's appendix pair.

**m2. The I6 gate metric leaves out thread contention, and the bit-identity test has limited power.**
- **Contention:** with overlap on, the training step slows: IMDB 0.67/0.63 s vs 0.52/0.53 s, ACM 1.49/1.38 vs 1.33/1.37 (`i6_run_output.txt:2421,2430,2457,2466`).
  - Exposed cost measured as on-epoch − (off-epoch − draw) is about **4 / 8 / 10 %** (DERIVED). The reported figure is 0.2 / 9.9 / 0.2 %.
  - The gate still passes either way. Each figure comes from a single run.
- **Bit-identity power:** the test runs on a tiny fixture, below the size where PyTorch splits work across CPU threads. The worker thread's thread count is UNVERIFIED, so a thread-count-dependent float sum (`wisp_sampler.py:301`, adaptive ratios) would go undetected.
  - Real-data logs match to 4 decimals under the uniform proposal (`:2414-2418` vs `:2423-2427`). The adaptive proposal was not checked on real data.
- Measuring 4 epochs instead of the card's 2 is disclosed, and epoch 1 is reported. That is honest.

**m3. Configs saved before this branch break on reload; the report does not say so.**
- `save_config` writes every field. A Part H schema-row config therefore carries `wisp_walks_per_anchor: 42`, which now raises at `config.py:395-403`.
- Configs saved before Part I have no `path_source` key, for example `experiments/d11_screen/d11_acm_full_s42/config.yaml:27-28` (tied, walk). Reloaded by `src/analysis/model_loading.py:21-22`, they resolve to wisp, and the strict `load_state_dict` (`checkpointer.py:70`) fails.
- Checkpoints of runs with `geometry_context` on also fail to load, because the path-encoder width changed (`h2snn_model.py:555`).
- Editing a resolved default config to `schemas` fails unless r is also reset to 0.

**m4. Two I4 report statements are not supported.**
- "Kept the pre-registered n = 400" (report:46): the pre-registration's I4 section names no n.
- The pre-registration requires σ\* to be drawn for every target (prereg:84). Seed 43's coverage is 0.9975 (`i4_run_output.txt:1960`), and the report does not mention it.

**m5. The report does not follow the Part A format.**
- It is 1,803 words plus 3 tables, which is more than 2 pages.
- The Summary runs well past 5 lines.
- It says "15 test sites"; the diff pins 14.
- "Goldens bit-identical" proves nothing about I1 or I2. The golden script pins `geometry_context: false` and `schemas` (`v2_golden.py:82-98`), and no golden covers the new default model.

### NIT
- `DataConfig()` built directly leaves `path_source=""`. Its consumers treat that as schemas (`dataset.py:143`, `h2snn_model.py:1148`).
- The preset r is applied for any `wisp_max_length` (`config.py:509-511`). For the planned accuracy-vs-ℓ_max analysis curve (U3), the incidence budget then no longer matches.

### Rules and stop conditions
- Nothing is clamped in the model; the only clamp is a read-out clamp (`i3_s1p_oracle.py`).
- Max log-weight is 9.6 (I4) and 5.9 (I3), well under 30.
- The comparison arms are not defaults.
- The I6 regression fixed at I7 is inside the card's scope.
- No stop condition should have fired.

### Units

| Unit | Verdict | Evidence |
|---|---|---|
| I0 | OK | `origin/wisp` = 94fc397 and `origin/d12-encoder` = 19a9d36; `wisp-followup` is not on origin; AGENTS.md/CLAUDE.md not committed; inventory and pre-registration present |
| I1 | ISSUE | Ladder code and the T-H7 rewrite are correct and strong (default context, positions ℓ−1 and ℓ, outside nodes, appendix control); M2 |
| I2 | OK | r = 42/65/12 matches `h6_calibrate_r.json`; LP/GC resolve to schemas; the pins are justified by schema fixtures, and no hidden default regression found; m3 |
| I3 | ISSUE | Oracle and readout-without-`h₀` implemented as carded (`i3_s1p_oracle.py`); numbers match the evidence; M1 |
| I4 | OK* | Graph, 6 length-4 schemas and the strict rank match the pre-registration; numbers match `i4_run_output.txt`; m4 |
| I5 | OK | Tests target MINOR-5/7/8/10; the 6 mutations hit real failure modes; m1 |
| I6 | OK* | Seed `cfg.seed + epoch` (`trainer.py:80`); proposal computed on the main thread after the step, from the same state as the sequential path; all RNGs are local generators; no shared mutable state; m2 |
| I7 | ISSUE | m4, m5, m3 undisclosed |

### Commands run
- `C:/Users/Acer/anaconda3/envs/h2snn/python.exe -m unittest tests.test_wisp` (from `H2S-NN/`) → `Ran 43 tests in 102.557s` / `OK`.
- `git branch -r`; `git rev-parse origin/wisp origin/d12-encoder` → `94fc397d…`, `19a9d36f…`; `origin/wisp-followup` is absent.
- Everything else was read-only (grep/sed/Read) on the source, tests, evidence files and spec. One stray empty `python -` heredoc hung, and I stopped it; it touched no file. No files were modified.
