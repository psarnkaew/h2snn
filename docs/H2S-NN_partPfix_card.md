# Task card — Part P-fix: carry-over fixes from the Part P review

**Design authority:** spec **v1.20** plus the author's decisions of 2026-11-13 in `docs/experiment_settings_decisions.md` (rows "Part P review"; their spec wording is drafted in `docs/spec_v1.21_draft_edits.md` V-1, not yet applied — the decisions bind, never edit the spec). `H2S-NN/AGENTS.md`/`CLAUDE.md` bind for environment, coding, testing and evidence tags.
**Branch:** `part-pfix`, created from `part-p` at the commit given in the prompt. **Report:** `docs/H2S-NN_partPfix_report.md`; evidence `docs/partPfix_evidence/`.

## Goal and rule
Four small fixes found in the Part P review, so that the cluster runs read the right data and the tuner wastes no trial. **Correctness only:** unit tests, mutation checks, the full suite once; no smoke run on real data, no performance experiment, no metric reported as a result, no comparison between methods. HetSheaf (P1), empty vendor clones and the CSNN compiler are **not** in this card (cluster-setup card).

## Units

### F0 — Preconditions (no code)
Read the four "Part P review" decision rows, `docs/H2S-NN_partP_report.md` (D-P6, D-P9, out-of-scope findings 4 and 5), `src/baselines/adapters/h2gformer.py` (grid at l.206, width use l.135-165), `_h2gformer_encoders.py`, `tests/test_shared_protocol.py::OptimizerTests`, `scripts/schema_readout.py` and `src/analysis/schema_readout.py` (`kendall_matrix`), and the data-root paths of H2G-former (`_build_hetero_data`) and Hetero²Net's loader. Create `part-pfix`.

### F1 — AdamW test and the MetaPath2Vec exception (D-P9)
Keep `SparseAdam` for H2GB's MetaPath2Vec pretraining. `test_every_adapter_builds_adamw` checks the optimizer of every **model's** training; the MetaPath2Vec pretraining optimizer is a named exception (one place, with the reason and spec reference), not a broad filter.
**Tests:** every adapter's model optimizer is AdamW (as before); the one exception is exactly MetaPath2Vec pretraining in `_h2gformer_encoders.py`. **Mutation:** a model optimizer switched to `SparseAdam` or Adam → the test fails; a second, unnamed `SparseAdam` site → the test fails.

### F2 — H2G-former width and `dim_pe` (D-P6)
`hidden_dim` ∈ {64, 128, 256, 512}; when `metapath_pe` is on, `dim_pe = min(128, hidden_dim // 2)`; when off, no positional channel (unchanged). No combination of the space raises; the "`hidden_dim ≤ 128` raises" rule is replaced. H2GB datasets keep their config stack; their `(dim_hidden, dim_pe)` must be reachable.
**Tests:** resolved `dim_pe` per width (64→32, 128→64, 256→128, 512→128); every (`hidden_dim`, `metapath_pe`) pair of the space builds the encoder on the tiny ACM-shaped graph and runs one forward (finite); the H2GB configs' pairs (64, 32), (256, 128), (512, 128) are reachable (`SearchSpaceCoverageTests` pattern, values read from the config files); the raw channel width is `hidden_dim − dim_pe > 0`. **Mutation:** `dim_pe` fixed at 128 → the 64/128 forward test fails.

### F3 — Read-out τ over length-2 only (spec v1.20 U2, D-O7)
`scripts/schema_readout.py` reports Kendall τ-b over length-2 sub-path schemas only; the all-spans τ is removed from the table and the JSON (the library may keep the option, not used by the script).
**Tests:** the script's table and JSON on two synthetic runs contain exactly one τ block, over length-2 schemas; its value equals `kendall_matrix(..., length=2)` on the same runs. **Mutation:** all-spans τ in the script → the test fails.

### F4 — Data root from the bundle (H2G-former, Hetero²Net)
H2G-former's NC data path and Hetero²Net's loader read the bundle's `data.root` (thread it through `NodeBundle.metadata` or the adapter call), never `REPO_DATA_ROOT` directly. Every other adapter: list in the report whether it reads the bundle's root (grep `REPO_DATA_ROOT` over `src/baselines`); fix any other site the same way and name it.
**Tests:** with the bundle's `data.root` set to a temp fixture and `REPO_DATA_ROOT` patched to an empty directory, each fixed adapter's `fit_task` runs one epoch on the fixture (finite keys); a grep test asserts no adapter module reads `REPO_DATA_ROOT` except through the bundle helper. **Mutation:** revert one site to `REPO_DATA_ROOT` → its test fails.

### F5 — Full suite and report
Full suite once. Expected: only the four known `test_baselines` vendor/notebook failures and the environment skips; `test_every_adapter_builds_adamw` passes. Report per unit (done, VERIFIED-RUN, deviations), "Decisions for the author", out-of-scope findings.

## Rules and stop conditions
- One commit per unit, explicit paths, tests with the change; never stage line-ending-only diffs, `Claude outputs/`, `data/`, `H2S-NN/paper/`, vendor clones. Forbidden: loosening tolerances, deleting or skipping tests, silencing exceptions, special-casing inputs, editing the spec.
- Stop: a fix would change a baseline's method beyond the decision (stop the unit, report); an adapter's data path cannot take the bundle's root without changing vendor code (report, list, continue).
- Not stop conditions: any metric value in a test.
