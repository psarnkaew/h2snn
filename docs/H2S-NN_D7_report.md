# D7 report (short) — fact layer, walk sampler, defaults flipped

Written for: the H2S-NN author (Part D units U1–U7 were done earlier on `D1_add`; this closes D7 under Part F U1). Evidence: `docs/D7_evidence/` (U1–U7 runs) and `docs/D11_evidence/` (this closing step). Tags per Part A.

## 1. Summary
D7–D9 are implemented and tested (units U1–U7, commits `c0424f6` … `64620a2`). Part F U1 closed the open decision as **option A**: baseline bundles pin `fact_layer: untied`, `path_sampler: dfs`, `resample_paths: false`; the H2S-NN defaults are now `tied` / `walk` / `true`. Baseline data is bit-identical before and after the flip (72 entries, 0 mismatches). The three-mutation check catches 3/3. The U8 measurements of the Part D card (before/after memory per dataset) were **not** done; ACM tied+walk numbers exist only from D10.

## 2. Files changed in the closing step
- `src/configs/config.py`: `DataConfig` defaults `tied` / `walk` / `resample_paths: true` (dataclass and `from_dict`).
- `src/baselines/bundles.py`: `BASELINE_DATA_PINS` merged into every bundle config; overriding a pin raises.
- `tests/test_paths_and_facts.py`: `BaselineBundleIdentityTests` (4 tests).
- `docs/D11_evidence/u1_*`: bundle dump/compare, mutation scripts and outputs.

## 3. Deviations from the spec
None for the closing step. Branch name: the specified `tier0-screen` already exists in another worktree (`.claude/worktrees/h2s-nn-d7-d10-b5b48b`, commit `b17f368` "docs: specs for D7–D10"); work is on `tier0-screen-d11` instead.

## 4. Verification evidence
- **Baseline path-builder check (VERIFIED-CODE):** `grep -rn "face_groups\|FaceGroup" src/baselines scripts notebooks` finds no consumer; every baseline adapter reads `edge_index`/`edge_type`/`x`/labels from the bundle. No baseline consumes path instances from our builders, so none needs the walk sampler. Side effect (not fixed): bundles still run the DFS builder and carry unused `face_groups`.
- **Bundle bit-identity (VERIFIED-RUN):** `u1_bundle_dump.py dump` at pre-flip defaults, again after the flip through `build_bundle` (ACM, DBLP, IMDB: x, y, edges, types, masks, every `FaceGroup` field; MUTAG: per-graph edges/types): `compared 72 entries; mismatched or missing: 0` (`u1_bundle_identity_output.txt`). The unit tests `BaselineBundleIdentityTests` (ACM node bundle vs explicit untied/dfs config; MUTAG graph bundle; pin override rejected; defaults are the model) pass: `Ran 4 tests … OK`.
- **Three-mutation check (VERIFIED-RUN, `u1_d7_mutations_output.txt`):** control 18 tests, 0 failing. M1 (−1 hop uses `Q` instead of `Qᵀ`): caught (T1/T2 tests fail). M2 (duplicates dropped instead of summed): caught (T8, T9). M3 (reverse orientation of a held-out fact not removed): caught (T11, T12). Extra check on the new pins (`u1_mutation_output.txt`): removing each of the three pins is caught 3/3.
- **Earlier D7 evidence (not re-run):** T1 relative error 1.3e-15; 2L transport steps (L=2: 6→4; L=4: 20→8); T8 sum error 0.005 vs drop-duplicates 0.47; fact counts ACM 273,936 / DBLP 119,783 / IMDB 43,321 (`D7_evidence/u1_checks_output.txt`, `u5_checks_output.txt`, `u5_smoke_output.txt`, `inventory_rho_hgb.txt`).
- **Full suite (VERIFIED-RUN):** after the default flip, 6 existing tests failed (`u1_full_suite_before_test_fixes.txt`). Root cause for all six: the tests were written for the old defaults (untied/dfs/no resampling), not model regressions. `test_T10_untied_rejects_minus_hops` and `test_T12_config_rules` now state `fact_layer: untied` / `path_sampler: dfs` explicitly; the two trainer smoke tests (hand-built fixture without a fact layer) set `resample_paths: false`; the graph-classification permutation test pins untied+dfs (walk draws depend on node ids, so invariance holds only in distribution under the defaults); the link-prediction head test checks hop direction with the hop signs (a -1 hop's fact is stored against the path). No tolerance changed, no test deleted. Result (`u1_full_suite_rerun.txt`, clean copy of the tree): `Ran 260 tests ... FAILED (failures=11, errors=13)`, failure names identical to the 24 pre-existing `test_baselines` failures.

## 5. Measurements
Not repeated. Existing: ACM, d=32, k=2, tied+walk: peak 1.49 GiB, 1.05 s per forward+backward (D10 report §5, RTX 4060 Laptop GPU). The per-dataset before/after table of the Part D card remains **UNVERIFIED** (not run).

## 6. Self-review checklist (closing step)
PASS baselines see the pre-D7 data under any H2S-NN default (bit-identity). PASS pins are not overridable (test). PASS each pin removal makes the identity test fail. PASS no new tuning knob (`fact_layer` still rejected in search spaces, `tests/test_paths_and_facts.py` T4). NOT CHECKED: whether any tuning/campaign YAML or notebook relied on the old defaults for H2S-NN runs (grep of `configs/` found no `fact_layer`/`path_sampler` key, so those runs now use tied+walk by design).

## 7. Assumptions imposed for theory and reality checks
- `THEORY-CONSTRAINT: tied facts (one cell per fact) — may cost information carried by the two stored copies of a relation (e.g. a mirror added with different semantics).` ρ is exact edge-set equality; the effect is measured by the `untied` ablation, which stays reachable. Reality check: all 10 HGB relation pairs are exact mirrors (VERIFIED-RUN, `inventory_rho_hgb.txt`).
- `THEORY-CONSTRAINT: baselines receive the untied stored graph — H2S-NN and baselines do not see the same encoding of the facts.` Both receive the same facts; message-passing baselines need both directions. Paper sentence in plan §9.5.

## 8. Open items
UNVERIFIED: LastFM, PubMed, amazon, youtube (files not on disk); per-dataset D7 memory table; bundles run an unused DFS builder (waste, out of scope). Pre-existing: 24 `test_baselines` failures (11 failures, 13 errors), same names as before D7.

## 9. Decisions needed
None for the closing step (option A taken from the Part F card).
