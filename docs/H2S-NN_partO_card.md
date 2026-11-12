# Task card — Part O: schema read-out and coverage of the main runs, fixed-literature-schema row

**Design authority:** `docs/spec_v1.18_draft_edits.md` **Z-2 and Z-3 as approved by the author** (with O-1 and O-2 as decided), on top of spec v1.17; `H2S-NN/AGENTS.md`/`CLAUDE.md` for environment, coding, testing and evidence tags.
**Precondition:** Part N-fix report read by the author. **Branch:** `part-o`, created from the last commit of `part-nfix` (hash given in the prompt). **Report:** `docs/H2S-NN_partO_report.md`. Evidence: `docs/partO_evidence/`.

## Goal and rule
Make the read-out of spec §3.2 usable on the v3-B main runs and define the comparison row, so that the server runs keep everything the analysis needs. **No performance experiment and no empirical result** (no accuracy, F1, ROC-AUC, MRR, `I_σ` value of a trained model, or comparison is reported); local runs are correctness tests only; no smoke run (author, 2026-11-12). The method is not changed (WISP, energy, maps, solver, readouts of the heads).

## Units

### O0 — Preconditions
Read Z-2, Z-3 and the author's O-1/O-2 decisions; §3.2 "Read-outs"; `src/analysis/wisp_readouts.py`; `src/utils/checkpointer.py`; the scripts that run the main-table seeds (tuning + seeds 0–4) and where they write and delete run directories. Create `part-o`.

### O1 — What the server runs keep
1. Every seed run of the main table (H2S-NN, all tasks) keeps `best.pt` and its resolved configuration (the exact dict the run used, incl. seeds, `ℓ_max`, `r`, base seed) in its run directory; nothing in the campaign deletes them. If a script deletes or overwrites them, change that script only. GC keeps its final weights (its protocol), recorded as such.
2. Reload test: train 2 epochs on a tiny graph, save; reload from `best.pt` + the saved configuration in a fresh process; rebuild the evaluation draw from the base seed; the rebuilt cells (node, fact, sign tensors and `m_f`) and `w_f = exp(g(ξ_f))` equal those computed at the end of training **exactly** (`torch.equal`; same device and dtype). Mutation: a different base seed in the reload → the test fails.

### O2 — Read-out script (evaluation only)
`scripts/schema_readout.py` (imports no training code beyond model loading): from a run directory computes, on the fixed evaluation draw, per length-2 sub-path schema σ: `c_σ` (= `I_σ` with every `w_f = 1`), `I_σ` (learned `w_f`), `ρ_σ = I_σ / c_σ`, mean `g(ξ_f)` over the cells containing σ, and the number of distinct sub-path schemas; for a set of runs (seeds) the Kendall τ between their `I_σ` rankings (`wisp_readouts.kendall_tau`). Schemas are printed with type and relation names (e.g. `paper-author-paper`); the literature list of O-1 is mapped to signatures by name, and a listed meta-path that maps to no signature raises (never silently empty). Output: one JSON per run and one table per dataset; `ΔL_σ` is not computed (v2 operator).
**Tests:** hand-built cells with known `m_f`, `w_f`: `c_σ`, `I_σ`, `ρ_σ` equal hand values; constant `g` → `ρ_σ = 1` for every σ (to 1e-12 in float64); `w_f` from a v3-B anchored model equal `H2SNN._path_weights` on the same draw; the ACM list maps to the expected signatures (PAP, PSP, PTP as typed relation sequences); a misspelled list entry raises. **Mutation:** `I_σ` computed without `m_f` → a test fails.

### O3 — The fixed-literature-schema row (Z-2)
`path_source: schemas` under the v3-B anchored model: lists = O-1 (literature list, source and its per-dataset entries in one table in `src/configs/config.py`, replacing the old presets for this row; the old presets stay only if another code path needs them, stated); budget = O-2 (walks per anchor = the configuration's `r`, split evenly over the listed meta-paths, at least one each, uniform instances within a meta-path); a config builder that takes a tuned `full` configuration and changes only the path source (any other difference raises). Still refused in tuning.
**Tests:** the builder changes only `path_source` (and nothing else; a diff test); every drawn cell matches a listed meta-path; walks per anchor sum to `r` (split rule, incl. `r` not divisible); the evaluation draw is deterministic per base seed; the anchored forward runs (finite) on ACM-sized fixtures.

### O4 — Full suite and report
Full suite once. Report per unit, the reload test result, the read-out tests, any script changed in O1, "Decisions for the author", out-of-scope findings.

## Rules and stop conditions
- One commit per unit, explicit paths, tests with the change; forbidden: loosening tolerances, deleting or skipping tests, silencing exceptions, special-casing inputs, editing the spec.
- Stop: O1's reload is not exact (stop, report the first differing tensor; do not relax to a tolerance); a listed meta-path has no instance on its dataset (stop O3, report); a change would alter the method.
- Not stop conditions: any metric value.
