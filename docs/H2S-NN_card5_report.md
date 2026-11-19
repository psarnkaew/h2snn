# Card 5 report — H2S-NN protocol freeze: B1a on H2GB NC, analysis hooks, Freebase memory probe, TSUBAME cell list

**Card:** `docs/H2S-NN_card5_card.md` (design-docs `6de4000`). **Branch:** `card5` from `v3b` `894fd74` (model
1.25), merged `--no-ff` with design-docs `6de4000` → `ec7494d`. **Binding:** spec v1.26 §6, decisions-log rows
of 2026-11-16/18/19, ablation plan v2. No model component changed; the only real-data run is the U3 probe; no
metric on real data, no tuning.

## Verdict
- U0–U6 done. Every test passes and every mutation is caught. Golden and mb0 differ nowhere.
- **U3:** at the largest configuration, Freebase NC full batch is **OOM on a 48 GB GPU** (the A6000 under the
  author's exception), so it is OOM at 24 GB too. Whether it fits on an H100 94 GB is **not known**; the TSUBAME
  command is below (not a stop of the card).
- U4: the phase `h2snn` gives 24 tuning lines and 24 final-seeds lines. `scripts/tsubame/lines.py` cannot parse
  them yet (D2, for the ops session).

## Files per unit (commits on `card5`)
| Unit | Commit | Files |
|---|---|---|
| U0 | `4129158` | `docs/card5_evidence/u0_scope.md` (worktree, merge, base count 777, golden/mb0 SHA-256, file list) |
| U1 | `762f299` | `src/trainers/minibatch_trainer.py`, `scripts/run_h2snn_wandb_tuning.py`; evidence `h2gb_config_table.py`, `u1_h2gb_config_table.{md,log}`, `jobs/run_cpu.sh` |
| U2 | `a87f0dd` | `src/analysis/mechanism_hooks.py` (new), `src/configs/config.py`, `src/models/h2snn_model.py` |
| U3 | `c33383e` | `scripts/freebase_memory_probe.py` (new); evidence `u3_freebase_probe.{json,log}`, `jobs/run_gpu.sh` |
| U4 | `783fb0a` | `scripts/tsubame/make_cell_list.py` |
| U5 | `1df64c5`, `610c58f`, `f95d151` | `tests/test_card5_protocol_hooks.py` (new, 12 tests); evidence `capture_base.py`, `base_capture_894fd74.json`, `measure_errors.py`, `jobs/{capture_base,run_tests,run_mutations}.sh`, logs of jobs 17806, 17825, 17833, 17834 |
| U6 | `dc1668f` | `docs/card5_evidence/u6_suite_17835.log`, `u6_rerun_17841.log`, `u6_rerun_17842.log` |
| U7 | (this commit) | this report |

Paths of code are under `H2S-NN/`.

### What changed (short)
- **U1 (B1a, `MiniBatchNodeTrainer`).** On datasets in `SUBSET_SIZES` only (`_run` → `_run_b1a`): the subset is
  `validation_subset(dataset, <whole validation target ids, local>, cfg.seed)`; `cfg.seed` is the run seed that
  drew PDNS's split. At every evaluation epoch only the subset is scored; there is no test pass. The best
  checkpoint and `bad_epochs` follow the subset Macro-F1 (any other `checkpoint_metric` is refused). The best
  weights are kept as an in-memory CPU copy (`_cpu_state`) and restored; then the whole validation and test
  splits are scored once. The record has whole-split `val_*`/`test_*`, `dataset`, and `_subset_fields`
  (`val_subset_macro_f1`, `val_subset_size`, `val_subset_fingerprint`), the same functions as the baselines.
  Other datasets run the old loop unchanged. `summarize_tuned_seed` ranks by `sweep_objective` (the end-of-trial
  objective already used it, `src/utils/wandb_campaign.py:474-478`).
- **U2 (hooks, read-only).** `mechanism_hooks.py`: (a) `energy_at_solution` (E, `e_v` with global ids, node-fact,
  node-cell, fact-cell parts, `E_tilde = E/||x*||²`, at the solve state of one evaluation forward, with the
  solve's `D^{-1/2}` scaling); `energy_terms` for any `(x, s, z)`. (b) `knockout(model, nodes)` and the config key
  `evaluation.knockout_nodes` (empty by default; refused in tuning by `reject_never_tuned_settings`, refused by the
  model in a training forward). (c) `transports_per_fact` (`Q_e = F_srcᵀF_tgt / c_e²`, `||Q_e − I||_F`, global
  endpoints, chunked). (d) `cells_per_target` (anchor, node ids, relations, schema key from
  `wisp_readouts.full_schema_keys`, `w_f`, `β_f`). Model: (i) `H2SNN._knockout` in the evaluation forward; (ii) an
  opt-in snapshot of the last evaluation solve (`record_solve_state`, off by default) and `last_cell_betas`
  (evaluation forwards). Both are off by default, so training and default evaluation are unchanged (golden, mb0
  and T4 confirm).
- **U4.** `--phase h2snn` in `make_cell_list.py` (constants `H2SNN_CELLS`, launcher
  `scripts/run_h2snn_wandb_tuning.py`, W&B `h2snn_h2snn_t4`), writing the tuning list and
  `cells_final_seeds_h2snn.txt`. This unit adds a phase; no existing phase or script behaviour changes
  (`baselines-1` output byte-identical, T9).

## H2GB configuration table (U1; job 17802)
| dataset | epochs | early_stopping_patience | eval_period | training_mode | batch / epoch_targets |
|---|---|---|---|---|---|
| ogbn-mag | 500 | 500 | 4 | minibatch | 32 / 8192 |
| mag-year | 500 | 500 | 4 | minibatch | 32 / 8192 |
| oag-cs | 500 | 500 | 2 | minibatch | 32 / 8192 |
| oag-eng | 500 | 500 | 2 | minibatch | 32 / 8192 |
| oag-chem | 500 | 500 | 2 | minibatch | 32 / 8192 |
| ieee-cis-g | 500 | 500 | 4 | minibatch | 32 / 8192 |
| h-pokec | 500 | 500 | 4 | minibatch | 32 / 8192 |
| pdns | 500 | 500 | 4 | minibatch | 32 / 8192 |

None of the eight YAMLs sets `epochs` or the patience; both come from the protocol budget (H2GB NC: 500 epochs,
patience none → 500 = the epoch count). So no early stop can happen, as spec §6 says ("500 epochs … without early
stopping"). Nothing differs from the spec.

## Tests (U5) — measured errors
Targeted run: job 17825 (A6000, env E, 82 tests incl. `test_partMB_trainer`, `test_td1_maps`,
`test_sheaf_energy`, `test_b1a_validation_subset`, `test_search_spaces`, card 4/4b and TSUBAME tests): **OK**.
The file hashes that job printed equal the committed files. Errors measured by `measure_errors.py` (job 17834,
same fixtures, float64).

| # | Test (`tests/test_card5_protocol_hooks.py`) | Tolerance | Measured |
|---|---|---|---|
| T1 | B1a scoring: every training evaluation scores exactly `validation_subset`'s ids; no test id before the end; restored best weights (= `best.pt`) scored once on whole val + test; record keys and fingerprint | exact | pass |
| T2 | PDNS: subset follows the run seed, equals `validation_subset`; missing seed raises | exact | pass |
| T3 | `summarize_tuned_seed` picks the highest `val_subset_macro_f1` (not the whole-split value); subset record without the key raises; HGB uses the whole split | exact | pass |
| T4 | HGB full batch and the ACM mini-batch flag: records equal the capture at `894fd74` (job 17806); `baselines-1` list equal | exact (float.hex) | equal |
| T5 | fixture: hook E vs ½YᵀLY (dense); Σe_v + fact-cell = E; e_v at the Thm 4 section | ≤ 1e-10 | 5.7e-14 (E = 458); 0.0; max e_v 8.9e-30 |
| T5 | model solution: hook E vs ½YᵀL̂Y; Σe_v + fact-cell = E | ≤ 1e-10 | 7.1e-15 (E = 31.5); 3.6e-15 |
| T6 | knockout: empty = identical logits; all nodes = `no_path_cells` logits with the same weights; one node: its cells' β_f = 0, others unchanged; refused in training and tuning configs; full batch and mini-batch | exact / ≤ 1e-12 | 0.0; 0.0; exact |
| T7 | `Q x` vs `F_src⁻¹F_tgt x`; QᵀQ = I; S2 (`fact_maps_identity`) Q = I exactly | ≤ 1e-12 | 6.7e-16; 1.1e-15 (median ‖Q−I‖_F 4.67); exact |
| T8 | node ids, schema keys, `w_f`, `β_f` vs the model's values of the same forward | exact / ≤ 1e-12 | exact; 0.0; 0.0 |
| T9 | 24 tuning + 24 final-seeds lines, order NC (HGB 4, H2GB 8), LP 8, GC 4; each parses with the launcher's parser; configs exist; `baselines-1` byte-identical | exact | pass |

### Mutations (job 17833, CPU, env E; files restored and checked equal to HEAD)
| # | Mutation | Failing tests |
|---|---|---|
| M1 | training epochs score the whole validation split instead of the subset (`minibatch_trainer.py`) | T1 |
| M2 | knockout ignores the node filter and removes every cell (`h2snn_model.py`) | T6 full batch, T6 mini-batch |

## U3 — Freebase NC memory probe (job 17824, g3an01, RTX A6000 47.5 GiB)
`scripts/freebase_memory_probe.py`: `full`, full batch, `d` 64, `ℓ_max` 5, `r` 64, other keys from
`configs/experiments/freebase_h2snn_v3b.yaml`; one training step + one evaluation forward.

| quantity | value |
|---|---|
| status | **OOM in the first training step** (forward/backward): 45.7 GiB allocated, 846 MiB more requested, 0.33 GiB free |
| max allocated / reserved | 45.75 / 47.00 GiB (at the OOM) |
| nodes / facts | 180,098 / 869,706 |
| cells per length (2/3/4/5) | 519,349 / 577,529 / 585,227 / 576,631 (total 2,258,736) |
| node-cell / fact-cell incidences | 10,254,084 / 7,995,348 |
| parameters | 11,805,360 |
| draw time / wall | 35.4 s / 48.6 s; host peak RSS 13.4 GiB |

It does not fit 24 GB or 48 GB. The 94 GB result is unknown, because the peak was not reached. **TSUBAME
command** (from `H2S-NN/` of a TSUBAME checkout that contains this card's probe script, i.e. a `v3b` SHA after
card 5 is merged, with `data/NC/freebase` present; guide §4–5):

```
echo '. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b && python scripts/freebase_memory_probe.py --out freebase_probe_$JOB_ID.json' | qsub -g $(cat ~/.t4_group) -l gpu_1=1 -l h_rt=0:30:00 -cwd -j y -N fb_probe -S /bin/bash
```

The U4 Freebase NC line stays full batch and carries the comment `FREEBASE_NC_NOTE` until this is decided (card
U4 "If U3 is undecided").

## U4 — line count
`--phase h2snn`: **24 tuning lines** (NC acm, dblp, imdb, freebase, ogbn-mag, mag-year, oag-cs, oag-eng,
oag-chem, ieee-cis-g, h-pokec, pdns; LP acm_pa, dblp_ap, imdb_ma, freebase_bb, lastfm, pubmed, ogbn-mag_pp,
h-pokec_uu; GC mutag, enzymes, nci1, proteins) and **24 final-seeds lines** (`--final_seeds 0 1 2 3 4`). Each line:
`scripts/run_h2snn_wandb_tuning.py --config configs/experiments/<ds>_h2snn_v3b.yaml --exp_name h2snn_<task>_<ds>
--num_runs 1 --base_seed 0 --n_trials 100 --output_dir <out>/experiments/h2snn --state_dir <out>/state/<cell>
--wandb_project h2snn_h2snn_t4`. Every flag is the launcher's own (`--state_dir`, `--output_dir`, `--exp_name`,
`--n_trials`, `--final_seeds`, `--wandb_project`); none was missing, so no stop.

## U6 — full suite and references
- **Suite, job 17835** (g3an01 A6000, sbatch, GPU not shared, env E, `--mem 35G`, peak RSS 28.2 GiB, 37 min;
  clean head `f95d151`, in the worktree itself): `Ran 789 tests` = **777 + 12** (`tests/test_card5_protocol_hooks.py`
  12, no other test file added or changed). Result `FAILED (failures=1, errors=4, skipped=4)`. **Every one of the
  nine comes from `vendor/sparse_tools` being absent in this worktree's vendor copy** (copied for card 4, before
  card 1d added `sparse_tools`): `test_baselines` "all declared vendor clones are present" (fail), four PSHGCN
  tests in `test_baselines_large_lp` / `test_freebase_baselines` (`FileNotFoundError: Baseline repository
  'sparse_tools' is not present`), and the four `test_baselines_pshgcn_mag` skips ("sparse_tools not built").
- **Fix (environment only, no tracked file):** the pinned clone (`VENDORS.tsv` row 22, commit `0596f00`, built
  in place) was copied (`cp -a`, not a symlink) from `/mnt/data/peeradon/h2snn/H2S-NN/vendor/sparse_tools`.
- **Rerun of the four affected modules at the same head** (A6000, env E): job 17841 `test_baselines`,
  `test_freebase_baselines`, `test_baselines_large_lp`: `Ran 102 tests … OK` (= 86 + 11 + 5, as in 17835); job
  17842 `test_baselines_pshgcn_mag`: `Ran 4 tests … OK` (no skip).
- So all 789 tests pass at `f95d151`: 683 in 17835 outside these modules, plus 106 in the reruns. The other 683
  were not rerun.
- Compares in 17835: golden `compared 1238 tensors in 23 configurations; differing: none`; mb0
  `compared 136 tensors; differing: none`.
- Logs: `docs/card5_evidence/u6_suite_17835.log`, `u6_rerun_17841.log`, `u6_rerun_17842.log`.

## Golden / mb0
- `docs/legacy_evidence/v3b_golden_cluster.pt` SHA-256 `29215150935b6bb1412f36baec52842a78d5edb779ef9d15c3ea59266155a564`
  (unchanged); compare: `compared 1238 tensors in 23 configurations; differing: none` (jobs 17825, 17835).
- `docs/partMB_evidence/mb0_reference_cluster.pt` SHA-256 `194868fce96fd658123165205d258392f7505b652bc974db26b5b1005dcbda37`
  (unchanged); compare: `compared 136 tensors; differing: none` (jobs 17825, 17835).
- No fixture was re-captured.

## Deviations
1. **GPU.** U3 (17824), the targeted tests (17825), the U6 suite (17835) and its module reruns (17841, 17842) ran
   on the A6000 (g3an01), under the
   author's exceptions of 2026-11-19 (24 GB GPUs were held by other users' 1-day jobs; the 3090/TITAN nodes had
   ~30 GB host memory left, below the suite's 35G). For U3 this tests 47.5 GiB instead of 24 GiB; OOM there
   implies OOM at 24 GB. Mutations (17833) and the error measurement (17834) ran CPU only.
2. **Knockout = cells removed from the solve,** not cells kept with `β_f = 0`. A cell with `β_f = 0` has
   `D_f = 0`, and the Jacobi preconditioner divides by it (`anchored_solve.py` `jacobi_inverse`). With
   `β_f = 0` every term of the cell vanishes and `z_f` decouples, so `x*` equals the solve without the cell.
   `last_cell_betas` reports 0 for knocked cells. T6 checks "all nodes = `no_path_cells` logits" exactly.
3. **Model forward edits** (card allows read-only accessor or knockout mask): the knockout mask, and an opt-in
   evaluation-only snapshot of the solve's inputs and `Y*`. `Y*`'s fact and cell states never leave the forward
   otherwise, and the key must act in both the full-batch and the mini-batch evaluation paths. Both are off by
   default (golden, mb0, T4 unchanged).
4. The T4 reference was captured at `894fd74` in a scratch worktree (`/mnt/data/peeradon/h2snn-card5-base`,
   job 17806; removed with `git worktree remove` after U6, clean, no job using it).
5. **U6 is one suite job plus module reruns,** not one clean suite job: the worktree's vendor copy lacked
   `sparse_tools` (U6 above). The 683 tests outside the four modules were not rerun after the copy.
6. The measured errors come from a separate script on the tests' fixtures (`measure_errors.py`), not from the
   test output; the tests assert the tolerances.

## Decisions for the author
| # | Question | Options | Recommendation and reason |
|---|---|---|---|
| D1 | Freebase NC mode (spec 1.26 "Search range under memory limits") | (a) run the TSUBAME command above; full batch if it fits on the H100 94 GB, else mini-batch (spec), with the mode change in a follow-up prompt; (b) decide mini-batch now | (a): the spec's own check; OOM at 48 GB says nothing about 94 GB. Freebase is not a B1a dataset, so U1 does not touch its mini-batch path |
| D2 | `scripts/tsubame/lines.py` `parse_line` accepts only `run_baseline_tuning.py` lines and the baseline state layout. `sweep_slots.sh` runs each line as `python <line> --resume` (the H2S-NN launcher has `--resume`), but its `lines.py snapshot`/`complete` calls do not recognise `h2snn` lines: provenance gets `before: null`, and a finished final-seeds line is not skipped, so it would run again in a later leg. `status.py` cannot report them | (a) ops session extends `lines.py` (`parse_line`, `snapshot`, `complete`) and `status.py` for the H2S-NN launcher's state layout (TSUBAME card); (b) leave until H2S-NN runs | (a), before the first `h2snn` leg: H2S-NN runs first on TSUBAME |
| D3 | The guide's data copy (§3) lists only `data/NC/{acm,dblp,imdb}` and `data/LP/` | ops session adds `data/NC/freebase`, the eight H2GB datasets and `data/GC` for the `h2snn` phase (and the probe) | needed before any `h2snn` line runs |

branch=card5 sha=dc1668f60049e8d538e4b44d6a41c427975c7a52 card=docs/H2S-NN_card5_card.md base=894fd7421ce8581d38b967906bde5593e881518a
