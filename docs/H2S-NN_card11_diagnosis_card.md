# Card 11 — Diagnosis of the first TSUBAME sweeps (LP acm_pa, PCG on PDNS/mag-year, τ on GC)

Written 2026-11-21 by the model design session; approved by the author the same day. Evidence only: **no fix, no change under `src/` or `configs/`**. Report: `docs/H2S-NN_card11_report.md`. Evidence: `docs/card11_evidence/`.

## Why
The first H2S-NN sweeps on TSUBAME (`v3b` `e5bdf10`, H100) show three failures the lab tests did not show (the lab never measured performance, by design):

1. **LP acm_pa does not learn.** 100 trials. Train AUROC stays 0.47 → 0.49 (median first → last epoch) while train loss falls to ≈ ln 2 (constant scores). Validation AUROC is best at epoch 0 (≈ 0.62, the untrained model) and falls below 0.5 with training (median last 0.42). The trial that trains most (`rural-sweep-3`, lr 7.6e-3, d 16, distmult) reaches train 0.76 and validation 0.46. The sweep therefore converged on lr < 3e-4 (92 of 100). On dblp_ap the same code learns (train 0.49 → 0.96, best epoch 39; lastfm 0.51 → 0.95), so the failure is specific to acm_pa.
2. **PCG reaches `T_max` = 100** (`PCGConvergenceError`, `src/models/anchored_solve.py:243`) on PDNS (5 of 5 trials, mini-batch) and on mag-year (epoch 48 of the first trial).
3. **GC: the τ guard fires** (`tau_stop`, "tau > 50", spec 1.16 §6): nci1 τ = 50.3 at epoch 0 (lr 8.6e-3), mutag τ = 50.2 at epoch 6 (lr 1.0e-2; τ = 16.7 already after epoch 0). The one enzymes trial that finished has validation Macro-F1 0.11 (chance 0.17).

The reference trials (config, metrics, log) copied from TSUBAME are in `docs/card11_inputs/` (8 runs: acm_pa `rural-sweep-3` and `absurd-sweep-38`, dblp_ap `daily-sweep-73` (best validation), pdns `balmy-sweep-7`, mag-year `dandy-sweep-1`, nci1 `astral-sweep-1`, mutag `dark-sweep-2`, enzymes `earnest-sweep-3`).

## Preconditions
- New worktree `/mnt/data/peeradon/h2snn-card11`, branch `card11` from `origin/v3b` `e5bdf10432e32c8fe31e8f3faff814951461d4d2`, then `git merge` the design-docs SHA given in the prompt (it holds this card and `docs/card11_inputs/`). The data root is the lab one (`data` → `/mnt/data/peeradon/h2snn/H2S-NN/data`).
- Read: this card; `src/trainers/link_prediction_trainer.py`; `src/data/link_prediction_dataset.py`, `src/data/link_pred_structures.py`, `src/data/hgb_lp_negatives.py`; `src/models/link_readout.py`; `src/models/anchored_solve.py`; `src/trainers/stop_conditions.py`; `src/configs/config.py` (`solver_max_iterations`, `tau_stop`, `ablation.fact_maps_identity`, `ablation.h0_only_readout`); spec §6 "Link prediction"; `docs/lab-gpu-cluster.html`.
- Runs: `sbatch` only, one GPU, `--mem` = measured peak + 30 %. GPU type per the 2026-11-20 lab decision (card 10): prefer 3090 / TITAN RTX; A6000 only if a run does not fit elsewhere, stated in the report. Each run from a config of `docs/card11_inputs/` with **only** the keys this card names changed.

## Units (one commit each; push `card11` after each)

### A — LP acm_pa
- **A0 reproduce.** Train `rural-sweep-3` and `absurd-sweep-38` on the lab GPU, seed 0, 40 epochs, through the same entry point the sweep uses. **Gate:** the curves match TSUBAME in direction (train AUROC flat near 0.5 for `absurd`, validation falling below 0.5 for both). If not, stop and report (environment difference; nothing else runs).
- **A1 data checks (no training), acm_pa and dblp_ap side by side.** From the prepared LP data as the trainer sees it:
  - counts of train / validation / test positives and negatives; head and tail node types;
  - leakage: is any validation or test positive, its reverse copy in the reverse relation, or a duplicate fact present in the message graph (`data.edge_index`)? Count per split;
  - message-graph degree of the head and of the tail for train, validation and test positives and for each split's negatives (quantiles, share with degree 0);
  - whether the tail type has input features (or a learned / one-hot embedding), per node type;
  - a **degree-only score** `log(1+deg_u) + log(1+deg_v)` (message graph): its AUROC on train, validation and test (positives vs that split's negatives).
- **A2 score diagnostics** on the A0 models at epoch 0 and the last epoch (scripts may load checkpoints and call the model's own `encode` / `link_readout`; no `src` change): AUROC per split; Spearman of the score with head and tail degree, separately for positives and negatives; `‖x*‖` and `‖endpoint_projection(x*)‖` against degree; the same for `h₀` (encoder output).
- **A3 controls** (each from `rural-sweep-3` with one change, seed 0, 40 epochs; train and validation AUROC per epoch):
  1. `ablation.fact_maps_identity: true` (S2, maps = I);
  2. `ablation.h0_only_readout: true` (no propagation at the head);
  3. a GCN baseline on acm_pa through the shared baseline LP loop at its central search point (the registry's own function; no new flag);
  4. `daily-sweep-73` (dblp_ap) reproduced, as the positive control.
- **Read-out, fixed now** (the report states which row holds):

| observation | reading | next step (not in this card) |
|---|---|---|
| A1 finds validation/test positives (or their reverse copies) in the message graph, or a split built differently from dblp_ap | task construction bug | fix the acm_pa split (data code) |
| degree-only AUROC < 0.5 on validation but > 0.5 on train, and the trained score follows degree | the model learns a degree shortcut that inverts between train (edges present) and validation (edges held out) | change the LP training graph (e.g. hold the training batch's positives out of the message graph) |
| GCN also fails (validation ≤ 0.5 after training) | task or protocol problem, not H2S-NN | as the first two rows |
| GCN learns, S2 learns, `full` fails | the learned transports break the dot / DistMult decoder on acm_pa | readout question (model session) |
| GCN learns, S2 and h0-only also fail | H2S-NN LP path (encode or readout) | locate in the code |

### B — PCG `T_max` on PDNS and mag-year (mini-batch)
- **B1 instrument** (script under `scripts/diagnostics/card11/`, wrapping `anchored_solve` at run time, no `src` change): per solve, log τ, iterations, final relative residual per scope, number of scopes, the largest scope (rows), the Jacobi preconditioner's min and max diagonal, the largest degree in the scope, `path_log_w` range, and the step / epoch. Verify on one ACM full-batch epoch that the wrapper does not change the loss (bit-identical).
- **B2 PDNS** `balmy-sweep-7`: run until the first `PCGConvergenceError` or 3 h; then the same config with `solver_max_iterations: 300` (config key) for the same wall time: does PCG converge, after how many iterations, at what time per step?
- **B3 mag-year** `dandy-sweep-1`: run with the B1 wrapper until the error or 3 h (it failed at epoch 48 on the H100; if the lab GPU is too slow to reach it, report the trends of τ and iterations per epoch instead).
- **Read-out:** does the iteration count grow with τ (conditioning), with one large scope or hub (preconditioner), or jump at one step (a non-finite or degenerate scope)? State which, with the numbers.

### C — τ on GC
- **C1** nci1 `astral-sweep-1` and mutag `dark-sweep-2` with `tau_stop` raised to `1e6` (config key; the run is a diagnosis, not a result), 100 epochs or the run's own stop: τ, PCG iterations and validation Macro-F1 per epoch. Does τ settle, keep growing, or diverge? Does validation change with τ?
- **C2** the same two configs at lr 1e-3 and 3e-4 (only lr changed, `tau_stop` as in the sweep): the largest τ reached and the epoch; whether the guard fires.
- **Read-out:** τ that settles at a finite value above 50 with no harm to PCG means the guard is too tight for GC (protocol question); τ that keeps growing means the solve tends to the per-graph mean (an over-smoothing optimum of mean pooling; model question); lr dependence only means the GC lr range is the issue.

## Rules
- No file under `src/` or `configs/` changes; diagnostic scripts in `scripts/diagnostics/card11/`, evidence in `docs/card11_evidence/` (logs, TSV, small PNGs). No checkpoints committed.
- Nothing here is a result: no number from this card enters a table or a sweep. Do not tune.
- Every run states GPU type, job id, wall time, peak memory.
- Git: one commit per unit, push `card11`; never stash, reset, rebase or commit others' changes; never merge into `v3b`.

## Stop conditions
- A0 does not reproduce the TSUBAME direction.
- A diagnosis needs a `src/` change to be measured (report what and why).
- A run needs more than 6 h; report the trend instead.

## Report (`docs/H2S-NN_card11_report.md`)
Per part: what ran (job, GPU, time), the tables, which read-out row holds and why, and anything unexpected. End with the evidence line `branch=card11 sha=<sha> card=docs/H2S-NN_card11_diagnosis_card.md base=e5bdf10432e32c8fe31e8f3faff814951461d4d2`.

## Estimate
A ≈ 4–6 h (A0 and A3 are 6 short runs; A1/A2 are scripts), B ≈ 6–8 h wall (mostly waiting), C ≈ 2–3 h. Parts A, B and C are independent: run B's jobs while writing A1/A2. Longer if A0 does not reproduce or the lab queue is busy.
