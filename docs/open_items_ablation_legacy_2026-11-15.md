# Open design items (2026-11-15, design session B): ablation/analysis gaps and legacy removal

Status: discussion notes for the author, **not decisions**. Nothing here changes the spec or a card until the author decides. Read with spec v1.22 §5–§6 and `docs/known_answer_preregistration_draft.md`. State when written: `design-docs` = `19782cf` (CS7 and CS8 approved, `v3b` ff, M3a card re-pinned).

## A. Legacy removal (v1/v2 code still in `src/`; checked by grep on `design-docs` = `v3b` src, 2026-11-15)
- **Risk:** `ModelConfig` defaults are v2: `propagation: "chebyshev"`, `encoder: "filter_bank"`, `coupling: "cooperative"` (`src/configs/config.py:1167-1177, 1371`). A YAML without `propagation: anchored` silently runs v2. The five `*_v3b.yaml` set it; 12 legacy YAMLs remain (incl. removed datasets amazon, youtube, amazonbook).
- Legacy still present: `filter_bank.py`, `pair_attention.py`, gates/coupling modes, `sigmoid_gamma`, `first_order`, v2 arms (`endpoint_edges`, `holonomy_edges`, `free_path_maps`), samplers `dfs`/`uniform_instance`, `wisp_readouts.delta_loss` (v2 operator), D12 diagnostics; 9 test files reference legacy keys.
- Keep: `PathTerm` (`vertex_path_laplacian.py`, used by `sheaf_energy.py:46`; move it), baseline data pins (`bundles.py:62-68`, `dfs` + `untied`; remove only after showing no baseline output changes), `identity_maps` if the S0/S2 sheaf-vs-no-sheaf arms (B2) are adopted.
- Proposal: an agent card (code is the agent's; the design session writes the card only): (1) fail-closed: v3-B defaults, every legacy value raises; (2) delete what is not kept. Gate: golden and mb0 bit-identical to the CS8 cluster references, full suite green except listed deleted legacy tests, mutation check that a legacy value raises. Estimate 0.5–1 agent day.
- **Author decisions needed:** A1 steps (1) only or (1)+(2); A2 keep `identity_maps`; A3 order relative to M3a (before M3a, or after M3a/M3b).

## B. Ablation and analysis: gaps (round 1, spec as written)
1. Ablation statistics deferred (1.14). Wilcoxon two-sided min p: n=5 0.0625, n=10 0.00195 (scipy, checked); Holm's first threshold 0.05/m is below 0.00195 for m > 25, so the Holm family must be ≤ 25 tests at n=10. Proposal: 10 seeds, paired by seed, Holm per claim over its datasets (m=3), test Macro-F1.
2. Arm tuning (OPEN-5): (a) full's tuned config + one switch; (b) 100-trial tuning per arm; (c) retune lr/wd/dropout. Proposal: (a) for all, (b) for `endpoints` and `closure_only` (kill-rule arms).
3. Ablation datasets unspecified ("3 HGB"); C6 needs `full` vs `endpoints` on all 12 NC datasets.
4. C3 not implemented (`map_family` ∈ {factorized, dense}, no diagonal); synthetic order task not designed; diagonal confounds order with rotation capacity (Thm 4).
5. No expectation/consequence: `free_cell_maps`, `direction_blind_maps`, `schema` rung, `h₀`-only. Proposal: `h₀`-only not beaten on ≥ 2/3 = warning signal.
6. C11, C12, C14 lack "≥ 2 of 3".
7. C6 vague (gain definition, statistic, datasets).
8. C2b refers to "as 1.10"; no Kendall τ threshold.
9. Known-answer: S3 design open; S3×C2 reading table incomplete (kill rule not fired + S3 fails); draft settings (cosine, clip) differ from protocol, to state.
10. WISP gate (i) / G0b still open: precondition for real-data WISP runs.
11. U3: no expectation; ℓ_max 6 outside the space; E11-calibrated r ≠ tuned r.
12. C5: "super-linear" slope undefined; no reading for the mini-batch gap.
13. Parameter counts: spec says ladder within 5 %; code matches only `endpoints`; other arms unstated.

## B2. Author's questions (round 2) and proposals
- **Sheaf vs no sheaf (NSD-vs-GCN style): missing.** S0 `no_path_cells`+`identity_maps`, S1 `no_path_cells`, S2 `full`+`identity_maps` (new: cells with identity transports), S3 `full`. Keep `a_e` learned in S0/S2 (one change only). Expect S1>S0, S3>S2, gap growing with ℋ²; else warning signal.
- **node / node+edge / node+edge+face:** node and node+edge are the same model (Prop. 7, exact elimination); meaningful ladder `h₀`-only → `no_path_cells` → `full`, inside: `hub_A`, `covering_only`. Exists; needs framing.
- **Structured face vs generic path use:** `closure_only` (HAN/SeHGNN), `instance_features` (MAGNN), `hub_A`(+`free_cell_maps`) (hypergraph), fixed-list row exist; `free_cell_maps` needs a claim; order (C3) missing.
- **Non-collapse / class separation:** a GCN-style depth plot is meaningless (anchored solve, Prop. 10, head reads h₀). Proposal: fixed τ ∈ {1,10,100,1000}, retrain, `full` vs S2; linear probe on x* only, within/between-class ratio, t-SNE illustrative; heterophilic + homophilic + synthetic. Outside protocol (τ>50, T_max).
- **Graph clustering:** not recommended as a test (Thm 4 maps come from labels); optional NMI/ARI of k-means on test x* as descriptive.
- **Reviewer concerns still open for v3-B:** parameter-count table for all arms (R4 capacity); controlled heterophily sweep on the same graph (R4); ρ_σ vs per-meta-path H_adj (R4 schemas vs heterophily); real-data case study (R2 c1); cost curves vs r and ℓ_max, sensitivity to r, per-dataset counts table (R1, R3 Q2/Q3, R4).

Next: author decides A1–A3, then B1–B13 and B2; the design session writes one "ablation and analysis pre-registration" draft, then spec v1.23 after approval.
