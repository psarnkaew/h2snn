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
