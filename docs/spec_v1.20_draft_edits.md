# Spec v1.20 — draft edits (fixed-literature-schema row after Part O; Kendall τ of the read-out), for the author's approval

**Status:** APPLIED 2026-11-13 as spec v1.20 (author approval); backup of v1.19: `docs/spec_backups/H2S-NN_methodology_spec_v1.19.md`. Written against spec **v1.19** on disk (md5 `7922ef0e5886d091c0a8507d7e61f853`). Re-read before applying; every Old string must occur exactly once (checked 2026-11-13: U-1 1×, U-2 1×); back up to `docs/spec_backups/H2S-NN_methodology_spec_v1.19.md`.
**Source:** author decisions of 2026-11-13 on the Part O report (`docs/H2S-NN_partO_report.md`, D-O1–D-O8; `docs/experiment_settings_decisions.md`). **The method is unchanged**; only the comparison row (an ablation of H2S-NN, never used by any baseline) and the read-out statistic are made precise.

**Why (design session, DERIVED from the report and the code, not run):**
- D-O4. As implemented in Part O each listed path's walks carry `1/r_π`, so each path's cells sum to ≈ 1 per target and the row's total cell mass per target is ≈ k (ACM 6, Freebase 7), against ≈ 1 in `full` (WISP: r walks of `1/r`). The row would then differ from `full` in two things, the paths and the scale of the cell weights at initialisation. `1/(k·r_π)` keeps equal shares per listed path and the total of `full`.
- D-O3. Part O draws a listed path with D8's walk (uniform fact of the hop's relation, walks that revisit a node rejected, `path_sampler.sample_walks`); WISP draws a uniform fact to an **unvisited** node (`wisp_sampler.py:8`). Example: author A7 with 3 papers incl. the anchor P1: WISP picks each of the 2 other papers with 1/2; D8 with 1/3 and loses the walk with 1/3. The difference depends on degree, so it is a second difference from `full`. The row uses WISP's hop rule with the action fixed by the list.

---

## U-1 §6 Ablation table — budget, hop rule, cell mass and run directory of the row
**Old:**
> Budget: the same number of walks per anchor as the tuned `r`, split evenly over the listed meta-paths (at least one each), uniform instances within a meta-path; no separate tuning.

**New:**
> Budget: the same number of walks per anchor as the tuned `r`, split evenly over the listed meta-paths in table order (at least one each; `r` below the number of listed paths raises, unreachable in the search space); no separate tuning. Within a meta-path each hop follows WISP's rule with the action fixed by the list: a uniform fact of the listed relation and direction from the current node to an unvisited node; a dead end discards the walk (1.20). A walk of listed path π carries `m = 1/(k·r_π)` (k listed paths, `r_π` walks of π per anchor), so every listed path has the same share and the total cell mass per target equals `full`'s (1.20). Cells use WISP's cell format, so `g` reads a listed cell exactly as it reads a WISP cell (relation embeddings and hop directions). Freebase BB is two book–book hops, as HAN's code in HGB. The row's runs write to their own run directory (the builder adds a suffix to the run name) and never to `full`'s, whose checkpoints the read-out needs.

## U-2 §6 Schema read-out — which Kendall τ
**Old:**
> Kendall τ of the `I_σ` rankings between seeds (C2b)

**New:**
> Kendall τ-b of the `I_σ` rankings of the length-2 sub-path schemas between seeds (C2b; the same set as the top ten; fixed before results, author 2026-11-13)

---

## Proposed changelog entry (prepended above the 1.19 entry; version line → 1.20)
> **Changes from 1.19 (to 1.20; author decisions of 2026-11-13 on the Part O report, `docs/experiment_settings_decisions.md`, draft `docs/spec_v1.20_draft_edits.md`; the method is unchanged):** (U1) the fixed-literature-schema row: WISP's hop rule with the action fixed by the list (not D8's rejection walk), cell mass `1/(k·r_π)` per walk so the total per target equals `full`'s, WISP cell format, Freebase BB as two hops, its own run directory; (U2) the read-out's Kendall τ is over length-2 sub-path schemas.
