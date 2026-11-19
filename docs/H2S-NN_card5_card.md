# Task card 5 — H2S-NN protocol freeze: B1a on H2GB NC, analysis hooks, Freebase memory probe, TSUBAME cell list

**Owner (design):** model session. **Executor:** S4, worktree `/mnt/data/peeradon/h2snn-ablprep`.
**Branch:** new branch `card5` from `v3b` `894fd74` (model 1.25). Then `git merge --no-ff` of `design-docs` at the SHA given in the prompt. Merge, never rebase.
**Urgency:** this is the last code card before the H2S-NN freeze. The advisor wants H2S-NN results by the end of November, and the next meeting is 2026-12-01. Keep to the scope below; do not add features.

**Binding (in this order):**
1. Spec **v1.26** (`docs/H2S-NN_methodology_spec.md`), in particular:
   - §6 "Validation during training on the H2GB NC datasets (B1a)";
   - Tuning ("by the validation metric (on the H2GB NC datasets: the B1a subset Macro-F1)");
   - "Search range under memory limits";
   - Freezing.
2. `docs/experiment_settings_decisions.md`:
   - rows of 2026-11-16 "Card 2 review" (B1a);
   - rows of 2026-11-18 "Card 1e F3 + 1f review" and "Card 1f U5 review" (PDNS per run seed, sweep objective);
   - rows of 2026-11-19 (spec 1.26; ablation plan v2).
3. The ablation plan v2 (`docs/ablation_plan_v2_2026-11-19.html`), sections "Face effect", "Two-regime analysis" and "Code". These define the analysis hooks.
4. The baseline implementation of B1a, which is the model to follow:
   - `src/evaluation/validation_subset.py`;
   - `src/baselines/minibatch.py` (`val_subset_loader`, `_subset_fields`, best-state restore);
   - `src/utils/wandb_campaign.py` (`sweep_objective`, `B1A_OBJECTIVE_KEY`).
5. `docs/working_preferences.md`.

**Mode:** protocol and read-only analysis hooks. **No model component changes:** maps, energy, solver, `g`, WISP and heads stay as they are. The only real run is the Freebase memory probe (U3). No metric on real data and no tuning.

## Units

### U0 — Worktree, merge, scope list
- Create the branch and worktree. Merge `design-docs` at the given SHA.
- Record `git rev-parse HEAD`, the suite count at base (777 at `894fd74`, S6 M8, job 17765) and the golden/mb0 SHA-256 at base.
- List every file the units below will touch, with a one-line plan per file. Anything outside this list needs a stop.

### U1 — B1a, PDNS rule and trial selection for H2S-NN on the H2GB NC datasets
Today `MiniBatchNodeTrainer._run` (`src/trainers/minibatch_trainer.py:203-233`) scores the **whole** validation split **and** the test split at every evaluation epoch, and it saves `best_test_metrics` from those passes. Change it as follows. It applies only to datasets in `SUBSET_SIZES`; every other dataset is unchanged.
- **Subset.**
  - The ids are `validation_subset(dataset, <whole validation target ids>, run_seed)`.
  - `run_seed` is the run's seed, the one that drew PDNS's split. It is required for PDNS and ignored elsewhere, as the function does.
  - Use the same function as the baselines; do not write a second one.
- **During training.** At every evaluation epoch, as today (`eval_period`, after the last epoch), score the **subset only**. Run **no test pass during training.**
  - The checkpoint "best" is chosen by the subset Macro-F1.
  - `bad_epochs` and early stopping also follow the subset metric.
  - If the H2GB configurations disable early stopping, as spec §6 says ("500 epochs … without early stopping"), keep that. Check the eight `configs/experiments/<h2gb>_h2snn_v3b.yaml` and report each one's `epochs` and `early_stopping_patience`.
- **After training.** Restore the best subset epoch's weights. The best checkpoint or an in-memory CPU copy are both fine; state which one you use. Then score **once** on the whole validation split and the whole test split.
- **Record.** `best_test_metrics.json` and the trial record carry:
  - the whole-split `val_*` and `test_*` metrics;
  - `val_subset_macro_f1`, `val_subset_size` and `val_subset_fingerprint`, with the same keys and the same functions as `_subset_fields`;
  - `dataset` (the dataset name), which `sweep_objective` reads.
- **Evaluation draw.** It is unchanged: one draw per run from the base seed for every validation and test target. The subset uses those same cells.
- **Trial selection.**
  - In `scripts/run_h2snn_wandb_tuning.py`, the end-of-trial objective and `summarize_tuned_seed` (today `max(records, key=metric_score)`, line 519) rank by `sweep_objective(record, monitor)`, as card 1f U6 does for baselines.
  - A record of a subset dataset without the subset key is refused (the existing guard).
- **What stays.** HGB, LP and GC are unchanged. Any change to their records or golden is a stop.

### U2 — Read-only analysis hooks (ablation plan v2)
All four hooks live in `src/analysis/` (new module `src/analysis/mechanism_hooks.py`, or the existing modules where they fit). Each one:
- reads a trained model and an evaluation batch;
- changes nothing in training;
- is refused in a training or tuning config, wherever a config key is involved.

**(a) Energy per node (R2, R3).** At the solve's state `Y* = (x*, s*, z*)` of an evaluation forward, return:
- `E(Y*)`: every term of the spec §3 energy, with the weights and maps of that forward and the same normalisation the solve uses;
- `e_v` for every node: the sum of the node–fact and node–cell terms that contain `x_v`;
- the fact–cell part separately;
- `Ẽ = E / ‖x*‖²`.

The identity `Σ_v e_v + (fact–cell part) = E` must hold.

**(b) Face knockout at evaluation (F3).** Add `evaluation.knockout_nodes` (empty by default) or an equivalent forward argument.
- In an evaluation forward, every cell whose node list contains a listed node gets `β_f = 0`. Nothing else changes: maps, `a_e`, the solver and the other cells stay the same.
- It is refused in training and tuning configs, like `evaluation.type_only_maps` (S6, T10).
- It must work in full batch (HGB) and in the mini-batch evaluation path.

**(c) Transports per fact (R1).** For a given set of facts, return `Q_e = F_srcᵀ F_tgt / c_e²` (d×d, from `select(rows).materialize()` or the operators' closed form), `‖Q_e − I‖_F`, and the two endpoint node ids. Work in chunks; never materialise all facts at once on H2GB.

**(d) Cells per target (case study).** For an evaluation batch, return for every cell:
- its anchor, node ids in order, relation sequence, schema key (`wisp_readouts`), `w_f` and `β_f`.

First check whether `src/analysis/wisp_readouts.py` or `schema_readout.py` already provides this. Add only what is missing and say which one applied.

### U3 — Freebase memory probe (spec 1.26: full batch if the largest configuration fits on one H100 94 GB)
- Script `scripts/freebase_memory_probe.py`: H2S-NN `full` on HGB Freebase NC, full batch, at the largest configuration (`d` 64, `ℓ_max` 5, `r` 64, other keys at the base config), for one training step and one evaluation forward. It prints `torch.cuda.max_memory_allocated`, `max_memory_reserved`, the cell and incidence counts and wall time.
- **Run once on a 24 GB lab GPU** (3090 or TITAN RTX; never the A6000 or g3an01; sbatch; one GPU not shared).
  - If it fits in 24 GB, it fits on H100 94 GB, and Freebase NC is **full batch**. Report the peak.
  - If it is OOM at 24 GB, do not implement a Freebase mini-batch path. Report the OOM point and the counts, and give the exact one-line command for the author to run the same script on TSUBAME (`qsub -l gpu_1=1`, `docs/tsubame_guide.md`). This is not a stop of the card.

### U4 — TSUBAME cell list for H2S-NN
Add a phase `h2snn` to `scripts/tsubame/make_cell_list.py`, using the existing format, state and output layout of phase `baselines-1` (TSUBAME card T3/T4). It has one tuning line per H2S-NN main-table cell, with launcher `scripts/run_h2snn_wandb_tuning.py`, `--n_trials 100`, the fresh state directory and the W&B project `h2snn_h2snn_t4`, and the final-seeds lines (seeds 0–4) as for baselines.
- **Order:**
  1. the 12 NC datasets (HGB 4, then the 8 H2GB);
  2. the 8 LP tasks;
  3. the 4 GC datasets.
- **Freebase NC:** comes from the configuration as U3 decides it. If U3 is undecided, the line uses full batch and is marked in a comment.
- Read the launcher's arguments from `run_h2snn_wandb_tuning.py`; do not invent flags. If the launcher lacks something the slot scripts need (e.g. `--state_dir`), stop and report.
- The ops session owns the TSUBAME card. State in the report that this unit adds a phase, and that no existing phase or script behaviour changes.

### U5 — Tests (float64 where numeric) and mutations
| # | Test | Tolerance |
|---|---|---|
| T1 | **B1a scoring.** Use a synthetic graph registered as an H2GB subset dataset, in the way of card 3 B3.3's test. During training every evaluation scores exactly `validation_subset(...)`'s ids; **no test id is scored before the end**; after the end the restored best weights are scored once on the whole validation and test splits; the record carries `val_subset_macro_f1`, `val_subset_size`, `val_subset_fingerprint` and `dataset` | exact |
| T2 | **PDNS.** With the dataset registered as `pdns`, the subset changes with the run seed and equals the baselines' function for the same seed; a missing run seed raises | exact |
| T3 | **Selection.** The tuning summary picks the record with the highest `val_subset_macro_f1`, not the highest whole-split value (two crafted records); a subset-dataset record without the key raises | exact |
| T4 | **HGB unchanged.** On an HGB fixture, records and metrics are equal to the base commit | exact |
| T5 | **Energy.** On a fixture, `E` from the hook equals `½ Yᵀ L Y` from a dense `L` built from the energy terms (as `docs/thm4_td1_evidence/check_thm4_td1.py`), `Σ e_v + fact–cell = E`, and every `e_v` is `0` at the Thm 4 section (reuse S6's T6 hook) | ≤ 1e-10 |
| T6 | **Knockout.** An empty set gives identical logits; knocking out every node gives the logits of the same weights with `β_f = 0` for all cells; knocking out one node changes only cells through it (their `β_f` is 0, the others' are unchanged); it is refused in a training config and a tuning config; it works full batch and mini-batch | exact / ≤ 1e-12 |
| T7 | **Transports.** `Q_e` from the hook equals `F_srcᵀF_tgt/c²` from `materialize()`; `QᵀQ = I` | ≤ 1e-12 |
| T8 | **Cells per target.** Node ids, schema keys, `w_f` and `β_f` equal the model's internal values of the same forward | exact / ≤ 1e-12 |
| T9 | **Cell list.** `--phase h2snn` lists 24 tuning cells in the stated order; every line's launcher arguments parse with `run_h2snn_wandb_tuning.py`'s parser; phase `baselines-1` output is byte-identical to the base | exact |

**Mutations** (each must make at least one test fail; report which):
1. Score the whole validation split during training instead of the subset (T1).
2. Let the knockout ignore the node filter, zeroing every cell (T6).

### U6 — Full suite and references
- Run the full suite in the worktree itself, not a copy with a symlinked `vendor/` (S6 job 17756), as an sbatch job with one GPU not shared, environment E and `--mem` 35G (S6 M8 peak 28.2 GiB + 30 %). Expected: everything passes.
- Golden and mb0 compares give `differing: none`, because the hooks are off by default and the B1a change touches only the mini-batch trainer's evaluation schedule.
- If mb0 differs, stop and report which tensors differ and why.
- Report the count against 777, with the tests added per file.

### U7 — Report
- **File:** `docs/H2S-NN_card5_report.md`.
- **Short format** (as the model-change report):
  - files changed per unit;
  - test table with measured errors;
  - mutation results;
  - the H2GB config table (`epochs`, `early_stopping_patience`, `eval_period`);
  - the U3 probe result (peak memory or OOM, counts, TSUBAME command if needed);
  - the U4 line count;
  - suite count;
  - golden/mb0 hashes;
  - deviations;
  - decisions for the author.
- **Last line:** `branch=card5 sha=<full head sha> card=docs/H2S-NN_card5_card.md base=894fd7421ce8581d38b967906bde5593e881518a`. If the report commit cannot contain its own hash, give the evidence commit there and the head in the final message, as S6 did.

## Files
- `src/trainers/minibatch_trainer.py`
- `scripts/run_h2snn_wandb_tuning.py`
- `src/analysis/` (hooks)
- `src/configs/config.py`: only the knockout key and its refusal
- the model's forward, only if the knockout cannot be done from outside; say why
- `scripts/freebase_memory_probe.py`
- `scripts/tsubame/make_cell_list.py`
- tests
- `docs/card5_evidence/`
- the report

**Not touched:**
- `src/baselines/`
- the model components: maps, energy, solver, `g`, WISP and heads
- `validation_subset.py`: use it, do not change it
- the spec, cards and pre-registrations

## Rules and stop conditions
- **Stop and report if:**
  - golden or mb0 changes;
  - an HGB, LP or GC record changes (T4);
  - B1a needs a change in `validation_subset.py` or `sweep_objective`;
  - a hook needs a change to a model component beyond a read-only accessor or the knockout mask;
  - the launcher lacks an argument the TSUBAME slots need;
  - anything in the spec contradicts this card.
- **Not stop conditions:**
  - U3 OOM at 24 GB (report it and give the TSUBAME command);
  - H2GB configs whose early-stopping settings differ from spec §6 (report them in the table; the design session decides).
- **Jobs.** sbatch only; one GPU not shared; `--mem` = recorded peak + 30 %; never the A6000 or g3an01. Job scripts print `git rev-parse HEAD`, hostname and `$SLURM_JOB_ID`.
- **Git.** One commit per unit (U0–U7). Push `card5` after each unit; never push another branch. Never stash, reset, rebase, amend or force-push. Do not merge into `v3b`: the design session reviews and the author approves the fast-forward.
- **Estimate.** U0–U2 ≈ 1 day; U3–U5 ≈ 0.5 day; U6–U7 ≈ 0.5 day plus the suite job (≈ 40 min). The queue for the U3 probe and the suite is the main risk.
