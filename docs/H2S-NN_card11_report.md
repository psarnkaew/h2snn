# Card 11 report — diagnosis of the first TSUBAME sweeps

Card: `docs/H2S-NN_card11_diagnosis_card.md` (design-docs `785883a`). Branch `card11` = base `e5bdf10` + merge of `785883a`.
Evidence only: no file under `src/` or `configs/` changed (`git diff e5bdf10 -- H2S-NN/src H2S-NN/configs` is empty).
Nothing below is a result: every number is a diagnosis measurement and enters no table or sweep.

Scripts: `H2S-NN/scripts/diagnostics/card11/`. Evidence: `docs/card11_evidence/{A,B,C}/` (per-epoch TSVs, the
trainer's `metrics.jsonl`, the `card11_changes.json` of every run naming the keys changed, and the SLURM logs with
the deprecation-warning lines removed). Run outputs and checkpoints stay outside the repository
(`/mnt/data/peeradon/card11_runs`).

How runs were made. `run_from_config.py` loads a reference trial's resolved `config.yaml` from
`docs/card11_inputs/`, changes only the keys given with `--set` (printed and stored in `card11_changes.json`; the
bookkeeping keys `output_dir`, `exp_name`, `use_wandb: false` do not reach the model) and calls
`seed_everything(cfg.seed)`, `build_experiment_components(cfg)`, `trainer.fit(data, epoch_callback=...)`, which is
what the sweep's `train_trial` does with the same saved config. Every job: `sbatch`, one GPU, conda `h2snn-b`; the
log prints HEAD, host, job id, GPU and the md5 of the card 11 scripts.

Environment notes. (1) The new worktree has no git-ignored `vendor/` clones; the untracked ones were symlinked from
`/mnt/data/peeradon/h2snn/H2S-NN/vendor` (H2GB, sparse_tools, tudataset, …; nothing staged). The first B jobs
18066/18067 failed in 5 s on the missing `vendor/H2GB` and were resubmitted (18068/18069). (2) `--mem`: the first
run of each dataset used a generous request; later runs use the measured host peak + 30 % (acm_pa 2.2 GB → 3 GB;
GC 1.2 GB → 2 GB). Peaks are in the tables.

## A — LP acm_pa

### A0 reproduce — gate holds

| run | job | GPU | wall | peak GPU | host peak |
|---|---|---|---|---|---|
| `rural-sweep-3` (lr 7.6e-3, d 16, distmult) | 18071 | RTX 3090 | 3 min 59 s | 2.99 GiB | 2.2 GB |
| `absurd-sweep-38` (lr 1.1e-4, d 64, dot) | 18072 | RTX 3090 | 2 min 37 s | 4.47 GiB | 1.9 GB |

Seed 0, 40 epochs, no key changed. Per-epoch tables side by side with TSUBAME:
`A/a0/a0_*_vs_tsubame.tsv`.

| run | | train AUROC ep 0 → 39 | val AUROC ep 0 → 39 | best val (epoch) |
|---|---|---|---|---|
| rural | TSUBAME | 0.4725 → 0.7588 | 0.4194 → 0.4578 | 0.4688 (10) |
| rural | lab | 0.4725 → 0.7588 | 0.4194 → 0.4578 | 0.4688 (10) |
| absurd | TSUBAME | 0.4731 → 0.4882 | 0.6290 → 0.4308 | 0.6290 (0) |
| absurd | lab | 0.4708 → 0.4905 | 0.6290 → 0.4306 | 0.6290 (0) |

`rural` matches TSUBAME to four decimals at every epoch (train/val loss, AUROC, τ, PCG iterations); `absurd`
matches in validation to ±0.0007 and in train AUROC to ±0.015 (training-mode noise). Direction as required: train
flat near 0.5 for `absurd`, validation below 0.5 for both. A1–A3 therefore ran.

### A1 data checks (job 18084, CPU, 19 s, host 3.6 GB)

`A/a1/*.tsv`. Prepared data exactly as the trainer sees it (`build_link_prediction_dataset` on the reference configs).

Counts and construction. acm_pa: target paper–rel_2–author, train/val/test positives 8059/895/995, val negatives
895 (one uniform author per positive), test 2-hop 763, test random 995. dblp_ap: author–rel_0–paper,
15913/1768/1964, val negatives 1768, test 2-hop 0 (HGB: no 2-hop candidates), test random 1964. Both splits are
written by the same script with the same settings (`split_meta.json`: seed 42, val 0.09, test 0.1, facts split
once, the reverse relation's rows removed with them). The message graph is the tied fact layer (each fact once, in
its kept relation's orientation: acm_pa keeps author–rel_3–paper for the target pair, dblp_ap author–rel_0–paper).

Leakage (`a1_leakage.tsv`): validation and test positives present in the message graph — stored orientation 0,
reversed 0, in the target pair's relation 0, duplicates 0, overlap with other splits 0, for both tasks. Training
positives are in the message graph (8059/8059 reversed for acm_pa, 15913/15913 stored for dblp_ap), by
construction of the transductive task, the same in both. Known positives among negatives: acm_pa train 4/8059,
val 1/895; dblp_ap train 27/15913, val 2/1768 (HGB does not filter them).

Input features (`a1_features.tsv`): acm_pa authors (tail) have no features (spec F17 removed them) → learned
per-node embedding; **authors have no relation other than the target**: mean degree 1.35, 14 % have degree 0 in the
message graph. dblp_ap's tail (paper) has 4231-d features and term/venue edges (mean degree 8.1, none with degree 0);
its featureless type is the head (author).

Tail degree in the message graph (`a1_degree.tsv`; negatives keep the head, so only the tail can separate):

| task | split | positives: share deg 0 / median / mean | negatives: share deg 0 / median / mean |
|---|---|---|---|
| acm_pa | train | 0.00 / 2 / 3.56 | 0.13 / 1 / 1.36 |
| acm_pa | val | **0.47** / 1 / 2.49 | 0.12 / 1 / 1.34 |
| acm_pa | test | **0.46** / 1 / 2.35 | 2-hop 0.06 / 2 / 3.53; random 0.15 / 1 / 1.33 |
| dblp_ap | train | 0.00 / 8 / 8.57 | 0.00 / 8 / 8.08 |
| dblp_ap | val | 0.00 / 7 / 7.57 | 0.00 / 8 / 8.11 |
| dblp_ap | test | 0.00 / 7 / 7.46 | random 0.00 / 8 / 8.12 |

A training positive's tail always keeps the edge itself; a held-out positive's tail has lost it, and on acm_pa
that leaves half of the validation/test authors isolated.

Degree-only score `log(1+deg_u) + log(1+deg_v)` (`a1_degree_auroc.tsv`, all relations):

| task | train | val | test 2-hop | test random |
|---|---|---|---|---|
| acm_pa | **0.684** | **0.448** | **0.320** | **0.458** |
| dblp_ap | 0.515 | 0.486 | — | 0.481 |

(Target-pair degree only: acm_pa identical in the tail; dblp_ap 0.552 / 0.420 / — / 0.419 — the same inversion
exists on dblp_ap within the target relation, but dblp_ap's tails keep their other relations.)

### A2 score diagnostics (job 18090, RTX 3090, 18 s, host 2.0 GB; extra: job 18094 on the A3 models, CPU, 53 s)

`A/a2/*.tsv`, `A/a2_a3models/*.tsv`. Evaluation mode, one `encode(epoch=-1)` on the evaluation draw, the model's
own `link_readout`; states `init` (before training), `epoch0`, `epoch39` saved by the A0 runs. Train AUROC here is
eval-mode on that epoch's training negatives (the trainer's training-mode value is in `metrics.jsonl`).

AUROC per split:

| run | state | train | val | test 2-hop | test random |
|---|---|---|---|---|---|
| rural | init | 0.468 | 0.520 | 0.544 | 0.527 |
| rural | epoch 0 | 0.542 | 0.420 | 0.402 | 0.427 |
| rural | epoch 39 | 0.756 | 0.458 | 0.323 | 0.468 |
| absurd | init | 0.466 | 0.633 | 0.627 | 0.624 |
| absurd | epoch 39 | 0.524 | 0.431 | 0.431 | 0.459 |

Spearman of the score with the **tail** degree (head degree: |ρ| ≤ 0.25 everywhere):

| run | state | train pos / neg | val pos / neg | test-2hop pos / neg | test-random neg |
|---|---|---|---|---|---|
| rural | init | −0.06 / −0.09 | −0.19 / −0.12 | −0.21 / −0.17 | −0.08 |
| rural | epoch 39 | **0.90 / 0.83** | **0.92 / 0.79** | **0.92 / 0.89** | **0.83** |
| absurd | init | 0.10 / −0.26 | −0.50 / −0.27 | −0.51 / −0.04 | −0.30 |
| absurd | epoch 39 | 0.02 / 0.17 | 0.34 / 0.15 | 0.28 / 0.07 | 0.16 |
| maps = I (A3.1) | epoch 39 | 0.48 / 0.68 | 0.89 / 0.67 | 0.88 / 0.70 | 0.70 |
| dblp_ap (A3.4) | epoch 39 | 0.03 / −0.06 | 0.20 / −0.05 | — | −0.07 |

Norms against tail degree (`a2_norm_degree.tsv`, authors; mean per degree bin 0 / 1 / 2 / 3–5 / 6–10 / 11+):

| run | state | `‖x*‖` | `‖endpoint_projection(x*)‖` | `‖h0‖` |
|---|---|---|---|---|
| rural | init | 2.63 / 1.61 / 1.62 / 1.65 / 1.64 / 1.68 | 0.74 / 0.46 / 0.45 / 0.46 / 0.46 / 0.48 | 2.63 … 2.60 (flat) |
| rural | epoch 39 | 3.77 / 0.61 / 1.57 / 2.04 / 2.72 / 3.94 | 3.91 / 0.39 / 0.60 / 0.83 / 1.20 / 1.89 | 3.77 / 3.60 / 3.55 / 3.52 / 3.40 / 3.52 |
| absurd | init | 7.32 / 4.45 / 4.52 / 4.57 / 4.67 / 5.01 | 2.35 / 1.35 / 1.39 / 1.41 / 1.46 / 1.52 | 7.32 … 7.62 (flat) |

An isolated author has `x* = h0` (no edge term in its row); every connected author is smoothed towards its papers.
So `‖x*‖` (and the projected norm the dot / DistMult score scales with) depends on degree although `‖h0‖` does not.
In the untrained `absurd` model, isolated authors have the largest `‖x*‖` and score highest: half of the
validation positives are isolated, so the untrained model's validation AUROC is 0.63 — the "0.62 at epoch 0" of
the sweep is this inverted degree effect, not a learned signal that training destroys. After training (`rural`
epoch 39) the projected norm grows with degree among connected authors, the score follows tail degree with
ρ ≈ 0.8–0.9 in every split, and that ranks validation positives (half isolated) below negatives.

### A3 controls (seed 0, 40 epochs, train and validation AUROC per epoch)

`A/a3/`. All RTX 3090 (g3n01).

| control | job | wall | peak GPU / host | train AUROC 0 → 39 | val AUROC 0 → 39 | best val |
|---|---|---|---|---|---|---|
| 1. `rural` + `ablation.fact_maps_identity: true` (S2) | 18085 | 2 min 29 s | 1.06 GiB / 2.1 GB | 0.535 → 0.901 | 0.446 → 0.441 | 0.454 (22) |
| 2. `rural` + `ablation.h0_only_readout: true` | 18086 | 2 min 00 s | 0.24 GiB / 1.9 GB | 0.502 → 0.593 | 0.500 → 0.523 | 0.524 (37) |
| 3. GCN, shared LP loop, central point | 18088 | 24 s | 0.80 GiB / 1.3 GB | 0.443 → 0.716 | 0.446 → 0.451 | 0.452 |
| 3'. GCN, lower-middle point (see below) | 18089 | 23 s | 0.77 GiB / 1.3 GB | 0.675 → 0.879 | 0.407 → 0.509 | 0.523 |
| 4. `daily-sweep-73` (dblp_ap), positive control | 18087 | 2 min 55 s | 2.39 GiB / 3.5 GB | 0.489 → 0.986 | 0.515 → 0.833 | 0.833 (39) |

Control 4 reproduces TSUBAME (val 0.5139 → 0.8325 there; `a3_dblp_daily73_vs_tsubame.tsv`).

GCN (control 3). No central-point function exists at `e5bdf10` (card 8's `--grid_point` has only `min` and is
not on this base), so `a3_gcn_control.py` takes the centre of the adapter's own
`get_adapter("gcn").search_space("link_prediction", "acm_pa")` — `values` lists at index `len // 2`, log-uniform
ranges at the geometric midpoint, uniform ranges at the midpoint — and fits through the registry's own
`src.baselines.campaign.fit_cell` (the function every baseline sweep trial calls), seed 0, run 0, the protocol
budget (40 epochs, patience 40). Central point: lr 1e-3, weight decay 5e-4, hidden 128, dropout 0.45, 3 layers,
DistMult. Two keys have two values (`num_layers`, `decoder_type`), so "central" is ambiguous; 3' takes index
`(len − 1) // 2` (2 layers, dot). The loop's `report` gives validation metrics only; the training AUROC is read by a
run-time wrapper of `SharedLinkHead.forward` (training-mode calls), no `src` change. This interpretation of
"central" is mine; the design session may want another point.

### A — read-out

| row | holds? | why |
|---|---|---|
| 1 task construction bug | **no** | 0 validation/test positives in the message graph in either orientation; split built by the same script and settings as dblp_ap |
| 2 degree shortcut that inverts | **yes** | degree-only AUROC 0.684 train vs 0.448 val (0.320 test 2-hop, 0.458 test random); the trained score follows tail degree, ρ 0.90/0.83 (train pos/neg), 0.92/0.79 (val) |
| 3 GCN also fails | **yes** | GCN at the central point: val 0.446 → 0.451 (train 0.72); at the lower-middle point val ≤ 0.523 (train 0.88) |
| 4 GCN learns, S2 learns, full fails | no | GCN does not learn; S2 also fails (val 0.441) |
| 5 GCN learns, S2 and h0-only fail | no | GCN does not learn |

Rows 2 and 3 hold together: a task / protocol problem of acm_pa, not of H2S-NN. Mechanism, from A1/A2: the tail
type (author) has no features and no relation other than the target, so holding out a fact removes the author's
only evidence; 47 % of validation and 46 % of test positive authors are isolated in the message graph against
12–15 % of uniform negatives (and 0 % of training-positive authors, which keep the edge itself). Any model whose
tail representation depends on message-graph degree learns "high tail degree ⇒ link" on the training positives,
and that ranks held-out positives low. dblp_ap has the same inversion inside the target relation (target-pair
degree AUROC 0.552 train / 0.420 val), but its tails (papers) carry features and keep their term/venue edges, so
the score does not follow degree (ρ 0.03) and the model learns. The next step named by rows 2–3 (not in this card):
change the acm_pa training graph or task (e.g. hold the training batch's positives out of the message graph).

Unexpected. (1) The sweep's "best validation at epoch 0 ≈ 0.62" is itself a degree artefact (above). (2) The
h0-only control is the only H2S-NN variant whose validation AUROC ends above 0.5 (0.523): its tail representation
is the per-node embedding alone, which sees degree only through which embeddings get trained (score–tail-degree ρ 0.15–0.36 at epoch 39, `a2_a3models`). (3) τ in `rural` grows steadily (1.08 → 12.3) with PCG
iterations 8 → 43; it never reached a stop condition here.

## B — PCG `T_max` on PDNS and mag-year (mini-batch)

### B1 instrument

`solver_probe.py` replaces, at run time, `src.models.anchored_solve.pcg` (the module global that both
`solve_anchored` and the implicit backward call), `solve_anchored` (marks forward calls) and
`H2SNN._encode_anchored` (keeps epoch, mode, step, message graph, scope index), and calls the originals
unchanged. One row per PCG call (`COLUMNS` in the script): τ, tolerance, T_max, scopes, rows, largest scope, iterations
(max/mean), unconverged scopes, relative residual (max/median), the worst scope (rows, iterations, residual, its
largest node degree, its Jacobi range), the Jacobi diagonal range (all rows and node rows), the batch's largest
degree, `path_log_w` min/max/range, finiteness of b and y, epoch/step/direction, wall time. Opt-in
(`CARD11_PROBE_FAILURE=1`, used only in B2c): for an unconverged scope, its ‖b‖, absolute ‖r‖, ‖y‖ and Lanczos
(60 steps) estimates of the extreme eigenvalues of M and of J^-1/2 M J^-1/2 on that scope, through the same
`apply_m`.

Identity check on one ACM full-batch epoch (`configs/experiments/acm_h2snn_v3b.yaml`, `epochs: 1`), plain twice
and with the probe, in one process from the same seed:

| job | device | result |
|---|---|---|
| 18070 | CPU, pinned kernels E, 1 thread (g3n01, 1 min, 3.1 GB) | plain = plain = probe: train and val loss **bit-identical**, **all parameters equal** |
| 18065 | RTX 3090 (14 s, 1.7 GB) | plain vs plain: losses equal, parameters differ (GPU atomics); probe: train loss differs in the last bits (1.1149368 vs 1.1149367) |
| 18096 | TITAN RTX, 5 plain + 5 probe runs (38 s, 1.7 GB) | one train-loss bit pattern for all 10 runs; parameters differ run to run as in plain vs plain |

The CPU run is the bit-exact proof; on the GPU the backward is non-deterministic with or without the probe, and
the single last-bit difference of 18065 is that run-to-run variation (18096 does not reproduce it).

### B2 PDNS `balmy-sweep-7`

| run | job | GPU | wall | host peak | GPU peak | outcome |
|---|---|---|---|---|---|---|
| B2a T_max 100 (config as is) | 18068 | TITAN RTX | 1 h 47 min | 16.0 GB | 0.48 GiB | `PCGConvergenceError` backward, epoch 1 step 172, rel. residual 7.0e5 |
| B2b `model.solver_max_iterations: 300` | 18095 | TITAN RTX | 1 h 34 min | 14.9 GB | 0.48 GiB | same step, same scope, backward, **300** iterations, rel. residual **1.3e21** |
| B2c T_max 100 + failure analysis (no key changed; env flag) | 18111 | TITAN RTX | 1 h 09 min | 14.7 GB | 0.48 GiB | same step, same scope, 6.975e5 |

The run is deterministic up to the failure: epoch 0 equals TSUBAME (train loss 0.399548, val Macro-F1 0.8726,
`b2_pdns_tmax100_vs_tsubame.tsv`), TSUBAME also failed in epoch 1, and all three lab runs fail at epoch 1 step 172
in scope 27 with τ = 3.0825. Step time ≈ 14 s on the lab (CPU-bound: WISP sampling and scope building, 4 cores at
100 %, GPU ≈ 0 %), ≈ 2.8 s on the H100. B2b ran until the error (1 h 34 min), shorter than the 3 h cap, so "the same
wall time" is the time to the same failure.

Per epoch (`b2_pdns_tmax100_per_epoch.tsv`; 32 scopes per step):

| epoch | dir | solves | iterations max / mean / p95 | τ | largest scope rows | max rel. residual |
|---|---|---|---|---|---|---|
| 0 | fwd / bwd | 256 / 256 | 15 / 10.7 / 14 · 15 / 10.7 / 13 | 0.99 → 2.86 | 497 | 1e-3 |
| 1 (to step 172) | fwd | 173 | 17 / 13.4 / 14 | 2.83 → 3.08 | 364 | 1e-3 |
| 1 (to step 172) | bwd | 173 | **100** / 13.7 / 15 | 2.83 → 3.08 | 364 | **7.0e5** (one solve) |

The failing solve (`b2_pdns_tmax100_top_solves.tsv`, last row): backward of step 172, scope 27 of 32, 252 rows
(the largest of step 172; scopes of up to 497 rows converged in ≤ 15 iterations elsewhere), containing
the batch's largest node (degree 6332); the forward solve of the same step converged in 12 iterations; b and y finite.
Every other training solve took ≤ 17 iterations. Iterations follow τ mildly (Spearman 0.83 fwd / 0.85 bwd over all
training solves; τ only moves 1 → 3) and not the scope size (−0.01) or the largest degree (0.00–0.08).

B2c failure analysis (`b2c_failure_analysis.txt`):

| scope 27 at the failure | value |
|---|---|
| ‖b‖ (backward RHS = dLoss/dY of this scope) | **3.4e-17** (median over the step's 32 scopes 1.7e-3) |
| absolute ‖r‖ after 100 iterations | 2.4e-11 |
| ‖y‖ | 1.5e-11 |
| M on the scope: λ_min / λ_max (Lanczos) | 0.84 / 5.04 (cond ≈ 6, no negative Ritz value) |
| J^-1/2 M J^-1/2: λ_min / λ_max | 0.31 / 1.65 (cond ≈ 5.4) |

So the scope's system is well conditioned and SPD; its right-hand side is ≈ 0 (the scope's target contributes a
vanishing gradient), the stopping test `‖r‖ ≤ 1e-3 ‖b‖` asks for an absolute residual of 3e-20, below the fp32
round-off of `b − M y`, and the relative residual reported is round-off divided by 3e-17. More iterations only
accumulate round-off (1.3e21 at 300).

### B3 mag-year `dandy-sweep-1` (job 18069, TITAN RTX, 3 h 00 min wall limit, host 13.3 GB, GPU 0.36 GiB)

Did not reach the H100's failure point (epoch 49): 35 epochs in 3 h (≈ 300 s/epoch vs ≈ 90–125 s on the H100;
CPU-bound as PDNS). Losses and validation Macro-F1 track TSUBAME (`b3_magyear_vs_tsubame.tsv`). Trend
(`b3_magyear_per_epoch.tsv`, 32 steps/epoch, 256 scopes/step): training iterations max 6–7 at τ ≈ 0.2–0.3 (epochs
0–5), 9–12 at τ ≈ 1.2–1.8 (epochs 20–35); the maximum over the whole run is 12; τ moves 1.0 → 0.20 → 1.8 and
oscillates; largest scope ≈ 3000 rows (one step 5164) without effect on iterations (Spearman 0.00), iterations
follow τ (0.97 fwd / 0.93 bwd). Nothing grows toward 100: a gradual climb from 12 at epoch 35 to 100 by epoch 49 is
not what this trend shows. The TSUBAME log has no solver statistics for mini-batch epochs; it stopped ≈ 2 min into
epoch 49 after a normal epoch-48 evaluation, i.e. abruptly, which is consistent with the PDNS mechanism; that it
**is** the same mechanism on mag-year is inferred, not measured.

### B — read-out

| candidate | holds? | numbers |
|---|---|---|
| iterations grow with τ (conditioning) | no (for the failure) | iterations do follow τ (ρ 0.83–0.97), but stay ≤ 17 (PDNS) / ≤ 12 (mag-year); the failing scope has cond ≈ 6 |
| one large scope or hub (preconditioner) | no | failing scope 252 rows (larger scopes converged), scope size ρ ≈ 0; it holds a degree-6332 hub, but its M is well conditioned under Jacobi (cond 5.4) and the forward solve on it took 12 iterations |
| a jump at one step (non-finite or degenerate scope) | **yes** | one backward solve, one scope, ≤ 17 → 100 iterations; finite, but **degenerate right-hand side**: ‖b‖ = 3.4e-17 vs median 1.7e-3, so the per-scope relative tolerance is unreachable in fp32; T_max 300 does not help (1.3e21) |

Next step (not in this card, a `src` question): the per-scope stopping test (relative to ‖b_scope‖) needs a floor
or an absolute term for scopes whose right-hand side is ≈ 0 (or such a scope counts as converged), in the backward
and forward alike. Unexpected: the failure is in the backward, triggered by a target whose gradient vanishes, not
by conditioning; and lab speed on the H2GB mini-batch path is set by the CPU (step ≈ 5× slower than the H100).

## C — τ on GC

All runs with the B1 probe (one row per PCG call; summaries `C/*_per_epoch.tsv`, `*_drivers.tsv`,
`*_top_solves.tsv`; raw C1 probes committed, raw C2 probes (10 MB) kept in `/mnt/data/peeradon/card11_runs/c2`).
`graph_batch_size: 1` (one graph per step: mutag 150 steps, nci1 ≈ 3300 steps per epoch), so the runs are slow
on any GPU. GPU type: the C1 runs and the two mutag C2 runs on RTX 3090; to avoid ≈ 12 h of serial queueing,
nci1 lr 3e-4 ran on the RTX 2080 Ti and nci1 lr 1e-3 on a TITAN RTX (GPU memory ≈ 0.02 GiB, the runs are CPU-bound;
the τ trajectory does not depend on the GPU type beyond float noise). No A6000 was used.

| run | change | job | GPU | wall | host / GPU peak | outcome |
|---|---|---|---|---|---|---|
| C1 mutag `dark-sweep-2` | `model.tau_stop: 1e6`, `epochs: 100` | 18074 | 3090 | 9 min | 1.2 GB / 0.02 GiB | `PCGConvergenceError` (forward, T_max 100) in epoch 2, τ = 231 |
| C1 nci1 `astral-sweep-1` | same | 18077 | 3090 | 8 min | 1.2 GB / 0.02 GiB | `PCGConvergenceError` (forward) in epoch 0, τ = 135 |
| C2 mutag lr 1e-3 | `optim.lr` only | 18080 | 3090 | 6 h (cap) | 1.5 GB / 0.02 GiB | 143 epochs, guard did not fire |
| C2 mutag lr 3e-4 | `optim.lr` only | 18081 | 3090 | 6 h (cap) | 1.5 GB / 0.02 GiB | 157 epochs, guard did not fire |
| C2 nci1 lr 1e-3 | `optim.lr` only | 18082 | TITAN RTX | 1 h 29 min | 1.5 GB / 0.02 GiB | **τ guard fires** at epoch 3 (τ = 50.02) |
| C2 nci1 lr 3e-4 | `optim.lr` only | 18083 | 2080 Ti | 6 h (cap) | 1.5 GB / 0.02 GiB | 21 epochs, guard did not fire |

(The C2 runs keep the config's 200 epochs; three of them reached the 6 h wall limit, so their tails are trends.
`1e6`, `1e-3` and `3e-4` are parsed by YAML as strings; `TrainConfig` resolves them to floats, as the saved
`config.yaml` of each run shows.)

### C1 — τ with the guard raised (sweep lr: mutag 1.0e-2, nci1 8.6e-3; θ_τ has 10× that lr)

| run | epoch | τ (end of epoch) | PCG iterations max (fwd) | val Macro-F1 |
|---|---|---|---|---|
| mutag | 0 | 18.2 | 36 | 0.333 |
| mutag | 1 | 52.8 | 54 | 0.333 |
| mutag | 2 (110 of 150 steps) | 44 → 231 | 100 (T_max, rel. res. 2.6e-3) | — |
| nci1 | 0 (1018 of ≈ 3300 steps) | 0.92 → 138 | 100 (T_max, rel. res. 1.5e-3) | — |

τ does not settle: it rises every step until PCG fails. Iterations follow τ (Spearman: mutag 0.89 fwd / 0.88 bwd,
nci1 0.77 fwd; the Jacobi maximum 1 + τ·d tracks τ, ρ = 0.89 / 0.77); scope size matters less (nci1 0.41) and degree
not at all (max degree 3–4). The failing solves are well-formed (finite, one scope, 80–270 rows) at τ ≈ 135–231:
here the failure **is** conditioning, unlike B. Validation: mutag Macro-F1 stays 0.333 (one class predicted) while
τ goes 18 → 53; nci1 never finished an epoch.

### C2 — τ at lower lr (guard 50 as in the sweep)

| run | largest τ (epoch) | τ by epoch 0 / 5 / 20 / 50 / 100 / last | iterations max | guard | val Macro-F1 range (best, epoch) |
|---|---|---|---|---|---|
| mutag lr 1e-3 | 22.8 (139) | 1.3 / 7.2 / 12.7 / 13.3 / 15.3 / 21.0 (142) | 48 | no | 0.33–0.83 (0.833, 106) |
| mutag lr 3e-4 | 10.8 (155) | 1.1 / 2.2 / 7.4 / 8.4 / 9.3 / 10.6 (156) | 33 | no | 0.33–0.77 (0.766, 18) |
| nci1 lr 1e-3 | 50.0 (3) | 13.4 / 23.8 / 34.4 / — (epochs 0–2, then > 50) | 60 | **yes, epoch 3** | 0.54–0.66 |
| nci1 lr 3e-4 | 9.2 (7) | 2.6 / 8.5 / 7.1 (epoch 20, last) | 36 | no | 0.49–0.66 (0.659, 15) |

At lower lr, τ rises fast over the first ≈ 5–20 epochs, then levels off (mutag lr 1e-3 ≈ 12–13 over epochs 20–80;
lr 3e-4 ≈ 8.5–9.5 over 30–100; nci1 lr 3e-4 6.4–9.2 over 5–20) and afterwards **keeps creeping up** (mutag
lr 1e-3: 13 → 21 between epochs 80 and 142; lr 3e-4: 9.3 → 10.6 between 100 and 156). Iterations follow τ
(≤ 48). `path_log_w_range` climbs to ≈ 19–20 in every C2 run, i.e. g's log-weights run into both bounds ±10
(`path_log_weight_bound`); in C1 mutag they sit at the lower bound −10. Validation Macro-F1 does not track τ
(mutag's 18-graph validation set moves in steps of ≈ 0.05–0.1 at nearly constant τ).

### C — read-out

| row | holds? | why |
|---|---|---|
| τ settles at a finite value above 50 with no harm to PCG → guard too tight (protocol) | **no** | with the guard removed τ never settles (mutag 231, nci1 138 within < 3 epochs) and PCG fails from conditioning |
| τ keeps growing → solve tends to the per-graph mean (mean-pooling over-smoothing; model question) | **yes, at the sweep's lr** | monotone growth every step until T_max; validation does not improve with it (mutag stuck at one class) |
| lr dependence only → GC lr range is the issue | **partly** | the growth rate scales with lr: at 3e-4 τ stays ≤ 11 for 150+ epochs (mutag) and ≤ 9.2 (nci1); at 1e-3 nci1 still fires the guard at epoch 3, and mutag's τ still creeps (13 → 21) after a plateau |

So the pull on τ is systematic (the second row), and its speed is set by the lr of θ_τ (lr × `scalar_lr_scale` 10),
which is why the sweep's lr ≈ 1e-2 fires the guard within an epoch or a few. Whether the solve indeed tends to the
per-graph mean (x* spread inside a graph shrinking as τ grows) was not measured here; it is the model-session question
the row names. Unexpected: g's log-weights saturating at the ±10 bound in GC (C1 mutag at −10, C2 range ≈ 20).

## Summary of read-outs

- **A (LP acm_pa):** rows 2 and 3 — a degree shortcut that inverts between train and held-out, and GCN fails too:
  a task / protocol problem of acm_pa (authors have no features and no other relation; holding out a fact isolates
  47 % of validation-positive authors), not of H2S-NN.
- **B (PCG T_max, PDNS / mag-year):** a jump at one step in one degenerate scope — the backward right-hand side of
  one scope is ≈ 0 (3.4e-17) and the per-scope relative tolerance becomes unreachable in fp32; not conditioning,
  not scope size; T_max 300 does not help. mag-year: trend only (≤ 12 iterations to epoch 35), same mechanism
  inferred.
- **C (τ on GC):** τ keeps growing at the sweep's lr until PCG fails (row 2); its speed is lr-driven (row 3 in part);
  the guard is not too tight (row 1 no).

Stop conditions of the card: none hit (A0 reproduced; no measurement needed a `src` change; runs over 6 h were
capped by the scripts' own wall limit and reported as trends). Extra runs beyond the card's list, all with no key
changed beyond those named: A2 on the A3 models (18094), the GCN lower-middle point (18089), the B1 GPU repeat
(18096), the B2c failure analysis (18111).
