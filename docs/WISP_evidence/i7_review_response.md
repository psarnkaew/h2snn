# Responses to the I7 independent review (`i7_independent_review.md`), 2026-11-09
After the review I changed only the report and this file. I changed no code and no tests. The code
and test fixes the review implies are listed as proposals for the author.

**M1 (I3: the oracle could not pass; the cause is overstated): accepted.**
- **Checked:** the H5 config uses `scheduler: cosine` (`CosineAnnealingLR(T_max=epochs)`, `builders.py:77`). So the scalar learning rate decays from 0.01 to 0, and β's total reach in 100 epochs is about 0.5, not about 1.
- With β₀ = 0, the pre-registered expectation (β ≈ 7–9 needed to dominate) was unreachable by construction. I could have computed this at I0, and did not.
- β used ≤ 0.14 of its reach and changed sign (s42: +0.128, then +0.007; s44: −0.022). So a weak or inconsistent gradient is at least as plausible a cause as the reach. Which one binds is **UNVERIFIED**.
- The report now says "the test as run could not show capacity", gives both candidate causes, and tags the β₀ = log 64 proposal as a PROPOSAL whose outcome is UNVERIFIED. A scale-corrected start would not fix a gradient-sign problem.

**M2 (D-I1a removed `c` from the `endpoint_edges` / `free_path_maps` maps and the pair-attention latents): accepted as a deviation.**
- Spec R12 and §6 keep `[h₀, c]` for maps in every arm. D-I1a, which I pre-registered, followed the shared-latent code structure instead.
- The report now lists it under deviations.
- **PROPOSAL** (needs the author's go): two latents in those arms. Maps would read `[h₀, c]` and `g` would read `h₀`. Add a T11 at the default context.

**m1 (never-tuned guard gaps): accepted.**
- `r` is exempt by Part H's card ("every key above except `wisp_walks_per_anchor`"), but spec §6 lists it as never tuned. The card and the spec disagree; that is for the author.
- **PROPOSALs:**
  - reject `wisp_max_length ≠ 4` and `wisp_proposal: adaptive` in a node-classification tuning base config;
  - limit `path_weight_context: true` to the ladder (no `endpoint_edges`, `free_path_maps` or pair attention).

**m2 (contention; the bit-identity test's power): accepted.**
- A contention-inclusive exposed cost, on-epoch − (off-epoch − draw), is about 4 / 8 / 10 % on ACM / DBLP / IMDB (DERIVED from `i6_run_output.txt`). The gate passes either way. The numbers come from a single run.
- Bit-identity under thread-count-dependent float reductions is **UNVERIFIED** at real-data size for the adaptive proposal. The real-data uniform-proposal logs agree to the printed precision.

**m3 (configs and checkpoints saved before this branch do not reload): accepted. Now disclosed.**
- A schema-row config saved before Part I carries `wisp_walks_per_anchor: 42`, which is now rejected under `schemas`.
- Configs without `path_source` now resolve to `wisp`.
- Checkpoints trained with `geometry_context` on have the v1.3 path-encoder width.
- To reload them, name `path_source: schemas`, `wisp_walks_per_anchor: 0` and, for v1.3 widths, `ablation.path_weight_context: true`. Strict loading, no shims (repo convention).

**m4 (I4 statements): accepted.**
- n = 400 is the script's default, Part H's S2 size. The pre-registration names no n.
- Seed 43's σ\* coverage is 0.9975: 1 of 400 targets has no σ\* cell, against "drawn for every target". Both are now stated.

**m5 (format): accepted.**
- The report is rewritten: a 5-line summary, 14 pinned test sites (not 15), and shorter.
- The goldens cover only the pinned pre-v1.4 model; no golden covers the new default. This is stated.

**NITs: accepted, listed as open items.**
- A directly built `DataConfig()` keeps `path_source=""`, which its consumers read as schemas.
- The preset `r` matches the incidence budget only at ℓ_max = 4. The U3 ℓ_max curve needs `r` recalibrated per ℓ_max.
