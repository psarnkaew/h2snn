# Model change 1.25 (TD-1) — report (S6, 2026-11-18)

Card `docs/H2S-NN_model_change_td1_card.md`; spec v1.25 (Z-1–Z-38); base `v3b` `9b7ae0b`, merge of `design-docs` `0a9ee3c` (`ca389e5`). Evidence: `docs/model_change_td1_evidence/`. Every job ran by `sbatch`, environment `h2snn-b`; no A6000. Stop conditions: none fired.

## Files changed per unit
| Unit | Commit | Files |
|---|---|---|
| M0 | `29a2488` | `m0_scope.md` (file list + plan), `m0_collect_job.sh`, `m0_collect_17684.txt` |
| M1 | `5b5e8bd` | `src/configs/config.py`, `src/models/h2snn_model.py`, `src/models/sheaf_maps.py`, `src/models/relational_context.py` (deleted), `scripts/ablation/arms.py`, `scripts/known_answer/s3_audit.py`, `scripts/v3b_cost_smoke.py`, `CLAUDE.md`, `tests/test_relational_context.py` (deleted), `tests/test_v2_model.py`, `tests/test_wisp.py` |
| M2 | `07534e8` | `src/configs/config.py`, `src/models/{h2snn_model,map_operators,sheaf_maps,sheaf_energy}.py`, `scripts/ablation/param_table.py`, `CLAUDE.md`, `tests/td1_fixtures.py` (new), `tests/test_{anchored_solve,card4_identity_maps,factorized_maps,partMBfix_functoriality,paths_and_facts,sheaf_energy,v2_model,v3b_theory}.py` |
| M2b | `33c1967` | `src/configs/config.py`, `src/models/h2snn_model.py`, `scripts/ablation/arms.py` (`ego_head`), `CLAUDE.md` |
| M2c | `1c47c9a` | `src/configs/config.py`, `src/models/h2snn_model.py`, `tests/test_v2_model.py` |
| M3 | `59c2a77` | `tests/test_td1_maps.py` (new), `tests/test_card4_ablation_driver.py`, mutation/measure scripts and logs |
| M4 | `2feb496`, `321e017` | `docs/legacy_evidence/v3b_golden.py`, `v3b_golden_cluster.pt`, `docs/partMB_evidence/mb0_reference_cluster.pt`, capture logs |
| M5 | `b630362` | `suite_job.sh`, `m5_suite_17699.txt` |
| M6 | `5c409fd` | `scripts/v3b_cost_smoke.py`, `k3/`, `k4/`, `known_answer_summary.md`, `cost_*.json`, job scripts and logs |

## Tests (`tests/test_td1_maps.py`, float64; measured by job 17693)
| # | Measured | Tolerance |
|---|---|---|
| T1 | ‖QᵀQ−I‖∞ 3.1e-15, \|1−det Q\| 5.3e-15 (fact and path transports, reversed hops); `stored_elements` = E(m²+1)+R(d²+dm) exact (d 8/16, m 2/4) | 1e-12; 1e-10; exact |
| T2 | 1.1e-16 | 1e-14 |
| T3 | 3.6e-15 | 1e-12 |
| T4 | 8.9e-16 (control, distinct roles: change > 1e-3) | 1e-12 |
| T5 | 4.9e-15 | 1e-12 |
| T6 | energy 6.7e-14, ‖LY‖ 3.8e-14; x*−h₀ 4.7e-14 (full state, 31 it.), 6.2e-13 (eliminated, 20 it.); accuracy 1.0; negative control energy 0.0144 | 1e-10; 1e-8; > 1e-3 |
| T7 | gradcheck passes (MLP→exp→apply; MLP weights; W_r→QR→U_r) | default |
| T8 | 5 key/value cases raise "spec 1.25 Z1/Z2" (config, base config, search key); no `relational_context` import; `map_rank` ≠ 4 refused in tuning | — |
| T10 | Q_e − R_r 1.1e-16 under `type_only_maps`; training forward and tuning refuse it | 1e-14 |
| T11 | loss finite; frames, MLP layers 0 and 2 (skew rows too) nonzero; W_r gradient exactly 0 at step 1, nonzero at step 2 | — |
| T12 | three arms one step finite; direction-blind Q = exp(A): 1.9e-15, no frame parameter | 1e-12 |
| T14 | widths d / 2d / 2d; `ego_head` representation = `[x*, h₀]` exactly; LP, GC widths; refused in tuning | exact |
| T15 | |g| up to ~1e3: range 20.0 = 2L, gradients finite; init w_f = exp(L tanh(g/L)) on 4 rungs (uniform: exp(L tanh(b/L))), LP and GC ≤ L | exact / 1e-12 |

## Mutations (job 17691, each on its target test only)
| Mutation | Target | Result |
|---|---|---|
| 1 Drop the antisymmetrisation of C_e | T4 | caught (2.24 > 1e-12) |
| 2 exp(C) instead of exp(C) − I | T3 | caught (2.53 > 1e-12) |

## Golden / mb0 (g2n01, E, 1 thread; job 17698 capture + same-node compare)
| Reference | Old SHA-256 | New SHA-256 | Second run |
|---|---|---|---|
| golden | `6198d18d…68b1687` | `29215150935b6bb1412f36baec52842a78d5edb779ef9d15c3ea59266155a564` (23 configs, 1,238 tensors) | differing: none |
| mb0 | `83f5f57d…8c91700` | `194868fce96fd658123165205d258392f7505b652bc974db26b5b1005dcbda37` (136 tensors) | differing: none |

Golden configurations: dropped `nc_geometry_context_false`, `nc_path_weight_context`; added `nc_ego_head`; `nc_map_family_dense` still skipped (refused key).

## Suite (M5, job 17699, g2n01, RTX 2080 Ti, 1,841 s; peak RSS 7.7 GB)
**751 OK** (no skip). Base `9b7ae0b`: 762 (`unittest` discovery; `pytest` is not installed). Removed: `test_relational_context` 7, `test_factorized_maps` 12, `test_v2_model` 2 (T11-ctx; per-side scale ablation), `test_wisp` 3 (appendix-pair tests). Added: `test_td1_maps` 13. 762 − 24 + 13 = 751.

## Known-answer (card 4 frozen settings; one change: the model). K5 S2′ not rerun: its card 4 result was obtained on the 1.24 model.
| Task | Quantity | Card 4 (1.24) | 1.25 |
|---|---|---|---|
| S3 audit | (i) TV / (ii) probe accuracy / r | 0 / 0.506 0.492 0.500 0.503 on [x,c,x,c,rel] / 2560 | 0 / 0.490 0.515 0.514 0.510 on [x_u, x_v, rel] (band 0.459–0.541) / 2560 — pass |
| S1′-B full + oracle | acc β log512 (s1–3) | 1.000 0.994 0.994 | 0.994 0.992 0.996 |
| | acc β log64 | 0.936 0.698 0.730 | 0.816 0.904 0.870 |
| | acc β 0 (leak if ≥ 0.9) | 0.518 0.512 0.474 | 0.500 0.530 0.528 — no leak |
| | warning signal (< 0.9 at log512) | no | no |
| | C12 known-answer part | not decidable (a `covering_only` run stopped) | not evaluated (`covering_only` not rerun, card) |
| S3 | `full` acc / reading | 0.518 0.530 0.494 / not shown at default settings | 0.494 0.466 0.514 / not shown at default settings |
| | `full` + oracle (log 512) | 1.000 0.996 0.996 | 1.000 0.926 0.982 |
| | `endpoints` / `uniform` | 0.496 0.542 0.496 / 0.490 0.488 0.532 | 0.488 0.496 0.514 / 0.528 0.478 0.532 |
| | `full` log-w gap, length-4 cells | 0.014 0.018 0.011 | 0.016 0.032 −0.002 |

## Cost smoke (job 17733, RTX 3090, ACM d 32, ℓ_max 3, r 16; v3b and 1.25 in one job, 2 reps, median of epochs 1–2)
| Quantity | v3b `9b7ae0b` | 1.25 | Ratio | Predicted | Measured / predicted |
|---|---|---|---|---|---|
| Train step (fwd + bwd) | 0.881 s | 0.480 s | 0.55 | — | — |
| Forward / backward | 0.162 / 0.709 s | 0.167 / 0.308 s | 1.04 / 0.43 | — | — |
| Epoch (draw + step) | 1.189 s | 0.773 s | 0.65 | — | — |
| Forward solve per PCG iteration (operator apply) | 23.1 ms | 24.9 ms | 1.08 | 0.5 | 2.2 |
| Map build (eval) | 19.6 ms | 10.0 ms | 0.51 | 0.2 | 2.6 |
| PCG iterations fwd / adjoint (warm-up 1e-3; at 1e-4) | 6 / 5; 8 / 7 | 6 / 5; 8 / 7 | 1 | — | — |
| Peak GPU memory | 1,851 MB | 1,421 MB | 0.77 | — | — |
| Map storage per fact (floats) | 129 (d 32, k 2) | 17 | 0.13 | 257 → 17 at d 64 (0.13 at d 32) | 1.0 |
| `stored_elements` src / tgt | 17,809,936 / 17,809,936 | 273,936 / 4,661,520 | — | — | — |
| Parameters | 458,342 | 303,596 | 0.66 | — | — |

## Deviations (each one line)
- `tests/test_factorized_maps.py` kept as a helper module (six test files import its fixtures); only its 12 Householder tests were removed (M0 plan said "delete").
- `H2SNN.geometry_inputs()` kept, returning `h₀`, so `scripts/known_answer/common.py`, `scripts/ablation/variance_gate.py` and `src/analysis/model_loading.py` needed no change.
- Files outside the card's list: `tests/td1_fixtures.py` (new fixture), `tests/test_card4_ablation_driver.py` (withdrawn arms absent, `ego_head` present).
- `s3_audit.py` (i) keeps its neighbour-mean definition through a local copy of the deleted `RelationalContext` computation; (ii) reads `[x_u, x_v, rel]`.
- T11: W_r has an exactly zero gradient at C_e = 0 (∂A/∂U = 0); the test checks it at the second step.
- T15: at initialisation only the `uniform` rung has g = b; the gated rungs are checked as w_f = exp(L tanh(g/L)) with g from the gate.
- `v3b_cost_smoke.py`: `--any_r` (r = 16 is not the 1.6 calibrated value) and `--code_root`; host `--mem` 12G had no recorded peak (measured 1.5 GB).
- Void jobs kept as records: 17696 (`/usr/bin/time` missing, no capture), 17703 (forward hook replaced the module output); 17702 ran `covering_only` S1′-B tasks by my index error, cancelled, records set aside in `k3_not_requested_covering_only/`, not used.
- `2feb496` missed the two `.pt` files (NFS attribute cache); recorded in `321e017`.

## Decisions for the author
- Operator apply is 1.08× v3b per PCG iteration at d 32 on ACM (prediction 0.5×): the train step is still 0.55× (backward 0.43×). Accept as is, or ask for a profile at d 64 / on H2GB (TSUBAME)?
- W_r starts with zero gradient (C_e = 0 at init): accept (it moves from step 2), or a design change?
- `free_cell_maps` in the TD-1 family: the free maps of one position vary only inside its 4-dim `U_position`; C17 compares within that family (design-session choice) — confirm.
- Stale wording left untouched (not in the card's files): `AGENTS.md:133` (D1 row, `householder_k`), `src/models/link_readout.py` and `graph_readout.py` docstrings (`[x*, h₀]`), `src/analysis/model_loading.py` docstring (`exp(g)`).
- S3 oracle seed 2 is 0.926 (card 4: 0.996); still the capacity side of the pre-registered reading.

branch=model-td1 sha=<see final message> card=docs/H2S-NN_model_change_td1_card.md base=9b7ae0b66de790a96ad1b6ad6d4b83caa7a4009c
