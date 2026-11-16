# Handoff: ablation and analysis design session (2026-11-15/16) → main design session

Branch `design-ablation` (worktree `/mnt/data/peeradon/h2snn-ablation`), based on `design-docs` `a463268`; docs only, no code.

## Read first
1. `docs/ablation_analysis_preregistration_draft.md` — approved by the author 2026-11-16: every arm, claim, statistic, analysis, run count and the order of work.
2. `docs/spec_v1.23_draft_edits.md` — X-1 … X-15 against spec v1.22 (md5 `7b44dd2c…a922`), anchors checked. **Wording awaits the author's approval**; apply on `design-docs` only after it (backup v1.22 first).
3. `docs/experiment_settings_decisions.md` — rows 2026-11-15/16 from "Ablation statistics and seeds (B-1)" to "GC heterophily statistics".

## Decisions in one list
Seeds 1–10 paired, Wilcoxon, Holm per claim; four ablation datasets ACM, DBLP, IMDB, Freebase, rule "≥ 3 of 4"; arms at `full`'s configuration + one change, and `endpoints`, `closure_only`, `instance_features`, `no_path_cells`, S2 tuned on their own (claims decided on those); C3 removed (no order claim); new C15 (S2/S0, maps = I), C16 (direction), C17 (composition), `h₀`-only sanity; C2b on `ρ_σ`; C6 within datasets (≥ 6 of 12); edge-removal perturbation study with a density-matched random control (ACM, DBLP, 7 methods); N1–N4b, N6 analyses (N3 corrected to a `g`-input intervention with `endpoints` as the exact zero control); U3 one-at-a-time without E11; C5 on ogbn-mag subgraphs + cost figure + per-dataset table, timing on one GPU type; known-answer tasks after M3b and before tuning; proposal variance before tuning; WISP gate (i) closed; main table Holm per task (NC, LP, GC); GC: no heterophily value, type-mixing ratio reported.

## Cards the main session writes (after spec v1.23)
1. Identity map operator under `anchored` (arms S2, S0); golden unchanged; tests: transports = I, Jacobi diagonal correct, `full` bit-identical.
2. Known-answer runs (S1′-B, S3, S2′) with the updated pre-registration (settings: AdamW, no scheduler/clip, seeds 1–3; reachability recomputed) — the S3 graph generator and fact-input probe audit.
3. Proposal-variance measurement (ACM, seed 1, central configuration).
4. Ablation driver: configurations (a)/(b), seeds 1–10, checkpoint retention, parameter table.
5. Analysis code: N1 (per-node heterophily, class-centred slopes), N2/N2b, N3 (`g`-input swap), N4/N4b, N6 fixed case rule, C2b (`ρ_σ`, permutation null), bootstrap CIs.
6. Edge-removal perturbation generator + random control + MLH/ℋ² measurement (reachable range computed first).
7. ogbn-mag subgraph sampler and timing harness (one GPU type), cost-vs-accuracy figure, per-dataset cost table, GC type-mixing ratio.

## Not done here
No task card, no code, no push to `design-docs`. The `h2snn-b` environment line for the spec is still separate. Merge `design-ablation` into `design-docs` when the author says "merge design-ablation".

## Update after merging `design-docs` `98bc8ed` into `design-ablation` (2026-11-16)
- **Spec v1.23 draft approved by the author.** Apply X-1 … X-15 on `design-docs` together with the main session's v1.23 items (handoff_2026-11-15 §8: PDNS per H2GB's code, H-Pokec SeHGNN N/A / PSHGCN K = 1, env line `h2snn-b`) in one version; spec v1.22 is unchanged on `design-docs`, so the anchors still hold.
- **Slim preparation (author, 2026-11-16) applies to the seven cards above:** keep result-protecting checks only (label-route/leak tests with their mutation, one equivalence test per new mechanism, e.g. maps = I gives transports = I and `full` bit-identical), one-node suite, short reports.
- **Plan item "card 4: ablation driver + G0b":** G0b is closed (WISP gate (i), B-10); card 4 is the ablation driver only. The proposal-variance measurement and the known-answer tasks (S1′-B, S3, S2′) must run before main-table tuning; with the dry-run card running now, place them right after it.
- The dry-run card's evaluation-draw reproduction test also serves the ablation checkpoints (N1–N4 need the draw rebuilt from the base seed).
