# H2S-NN card 2 — dry run on real data: report

Card `docs/H2S-NN_dryrun_card.md`; branch `dry-run` (worktree `/mnt/data/peeradon/h2snn-dryrun`), base `v3b`
`6ba6dec` + `design-docs` `726ec84` (merge `f255a63`). Evidence: `docs/dryrun_evidence/`. **No metric value appears
in this report or the evidence; no method is compared with another.** W&B project of the dry runs: `h2snn-dryrun`.
Change of plan by the author (2026-11-16): A6000 retries cancelled; OOM cells measured with one memory axis one grid
step lower on 24 GB GPUs; D4b in a separate prompt.

## Units

| unit | what | commit | jobs |
|---|---|---|---|
| D0 | preconditions and cell list (no runs) | `3bfbbdb` | 16741 step 82 |
| D1 | 17 missing H2S-NN experiment configs (freebase NC, 8 H2GB NC, 5 LP, 3 GC), r and ℓ_max named, in range; test that all 22 cells resolve | `e419370` | 16795 step 7 |
| D2 | dry run, 163 cells; OOM step-down; vendor commits | `5fa5ad0` | arrays 16797, 16860, 16882 (pending), 16889 (pending), 16875 (pending); workers in 16795 |
| D3 | tuning launcher end to end, resume, seeds, read-out, evaluation draw; 3 launcher fixes | `0669e3d` | 16741 steps 97–104 |
| D4 | heterophily statistics: **stopped** (script does not follow M2) | `55aa3f7` | none |
| D5 | full suite on one node (691 OK), ready list, this report | this commit | 16795 steps 15–16 |

## D2 cell table (`dryrun_evidence/cells.md`, summary `d2_summary.md`)

Status at the largest point of each search space on a 24 GB GPU (TITAN RTX), 163 cells (149 HGB / LP / GC, 14 H2GB NC):

| status | H2S-NN | baselines | total |
|---|---|---|---|
| OK | 14 | 104 | 118 |
| OOM (GPU) | 5 | 23 | 28 |
| host OOM | 1 | 2 | 3 |
| stop condition | 2 | – | 2 |
| blocked: baseline code | – | 12 | 12 |

Non-OK rows:
- **Stop condition (recorded, not fixed):** H2S-NN enzymes and nci1, log-weight range > 30 at epoch 0
  (`src/models/h2snn_model.py:740`; values 30.15 and 32.09 at the largest point d 64, ℓ_max 5, r 64, attention pooling).
- **Host OOM:** H2S-NN h-pokec killed at the 30G request (MaxRSS 31.5 GB, 8 min 54 s, during the training epoch);
  h2gformer and hetero2net Freebase NC killed at 24G (MaxRSS 25.2 GB, < 30 s). Retries queued (below).
- **Blocked: baseline code (session 1, card 1d H3b; not fixed here):** HAN NC imdb (`hgb_han.py:153`); HINormer NC
  freebase (`training.py:81` → DGL 0-in-degree); HAN and H2SGNN LP imdb_ma (`_target_block.py:44`); H2SGNN LP freebase_bb
  (CUDA illegal memory access); ChebNetII NC ogbn-mag, mag-year, oag-cs, oag-eng, oag-chem and SeHGNN NC ieee-cis-g
  (`bundles.py:375` → `config.py:1023`, `data.batch_size` missing). **New, not in H3b:** HAN NC freebase — segfault right
  after `vendor/HGB/NC/benchmark/methods/HAN/utils.py:213` (16797_56, FAILED 11:0); HAN LP freebase_bb at heads 4 — segfault
  after `torch_sparse/matmul.py:97` (CSR `torch.sparse.mm`). Both sparse code on Freebase.
- **OOM:** the 28 cells and their step-down results below.

PCG: no solve hit `T_max` in any H2S-NN record; max iterations per solve 7–15 (training fwd/bwd, validation).

### OOM cells: one memory axis one grid step lower, 24 GB (TITAN RTX), for the author's decision

| cell (model task dataset) | max point at 24 GB | axis stepped down | result |
|---|---|---|---|
| H2S-NN NC freebase | OOM > 21.8 GiB + 0.7 | r 64 → 32 | still OOM (needs > 23.2 GiB) |
| H2S-NN LP dblp_ap | OOM > 23.2 + 0.07 | r → 32 | fits with r ≤ 32 (peak 20.2) |
| H2S-NN LP freebase_bb | OOM > 21.4 + 0.8 | r → 32 | still OOM (needs > 22.7 GiB) |
| H2S-NN LP pubmed | OOM > 22.9 + 0.06 | r → 32 | fits with r ≤ 32 (17.4) |
| H2S-NN LP lastfm | OOM > 23.0 + 0.06 | r → 32 | fits with r ≤ 32 (17.2) |
| NSD NC acm | OOM > 18.4 + 3.6 | d 5 → 4 | still OOM (needs > 20.2 + 2.4) |
| NSD NC freebase | OOM > 19.2 + 10.5 | d → 4 | still OOM (> 14.1 + 7.0) |
| NSD LP acm_pa | OOM > 18.4 + 3.6 | d → 4 | still OOM (> 20.1 + 2.4) |
| NSD LP freebase_bb | OOM > 19.1 + 10.1 | d → 4 | still OOM (> 21.2 + 4.0) |
| NSD LP pubmed | OOM > 19.7 + 2.7 | d → 4 | fits with d ≤ 4 (19.7) |
| CSNN NC freebase | OOM > 22.3 + 1.4 | d 4 → 3 | fits with d ≤ 3 (19.3) |
| CSNN LP freebase_bb | OOM > 22.4 + 1.3 | d → 3 | fits with d ≤ 3 (19.4) |
| HINormer NC acm | OOM > 21.1 + 2.0 | len_seq 150 → 100 | still OOM (> 22.4 + 0.9) |
| HINormer NC dblp | OOM > 20.1 + 2.7 | len_seq → 100 | still OOM (> 21.8 + 1.2) |
| HINormer NC imdb | OOM > 20.9 + 3.3 | len_seq → 100 | still OOM (> 21.9 + 1.5) |
| H2GFormer NC acm, dblp, imdb; LP acm_pa, dblp_ap, imdb_ma, lastfm | OOM in the MetaPath2Vec pre-step (+5.19 GiB) | metapath_pe True → False | fits with metapath_pe off (all 7; peak 3.7–13.9) |
| H2GFormer LP freebase_bb | OOM > 19.6 + 2.0 (`gt_layer.py:646`) | hidden_dim 512 → 256 | fits with hidden ≤ 256 (18.8) |
| GAT LP acm_pa | OOM > 17.7 + 4.2 | num_heads 8 → 4 | fits with heads ≤ 4 (11.1) |
| GAT LP freebase_bb | OOM > 14.7 + 12.0 | num_heads → 4 | still OOM (> 16.2 + 6.0) |
| HAN LP freebase_bb | OOM > 22.0 + 1.1 | num_heads → 4 | segfault (see blocked) |
| PSHGCN LP freebase_bb | OOM > 23.2 + 0.2 | K 3 → 2 | still OOM (> 23.2 + 0.2) |
| PSHGCN LP pubmed | OOM > 23.4 + 0.06 | K → 2 | fits with K ≤ 2 (10.6) |

H2GFormer: the MetaPath2Vec width is `min(128, hidden_dim/2)`, so hidden 512 → 256 does not change that step (tried on
lastfm: still OOM, kept as attempt `24GB-stepdown-hidden`); `metapath_pe` is its only grid axis that reaches it.
Summary: 16 fit, 11 still OOM, 1 segfault.

**Pending (queued, not waited for):** A6000 retry at the full point for the 5 H2S-NN OOM cells (array 16889); h-pokec at
120G on g3an01 (16875); HAN / H2GFormer / Hetero2Net Freebase NC at 40G host memory on a 3090 (16882). The A6000 is held by
another user's 2-day job.

## Fixes (D3; H2S-NN `scripts/` only, no `src/` change → golden / mb0 unaffected)

| fix | test | mutation |
|---|---|---|
| `run_baseline_tuning.py`: `--task` not required by argparse (sweep worker is spawned with `--worker_context` only; no baseline trial ran) | `test_baseline_worker_is_spawned_without_task` | caught (16741 step 99) |
| `run_baseline_tuning.py`: `summarize_unit(records, monitor, path)` (the monitor was passed as `summarize_cell`'s task; every sweep raised after its trials) | `test_baseline_unit_summary_selects_by_the_monitor` | caught (step 101) |
| `run_h2snn_wandb_tuning.py`: epoch callback returns without a W&B run (`--final_seeds` raised in `wandb.log`) | `test_final_seeds_run_without_a_wandb_run` | caught (step 101) |

Added: `--dry_run` (epochs 1, tag `dry-run`, refused in a sweep config) in both tuning launchers and the seeds launcher,
3 tests; `scripts/slurm/run_sweep_agents.sbatch`. D3 checks: H2S-NN trials on ACM / acm_pa / MUTAG, resume, seeds launcher,
`--final_seeds`, schema read-out OK; SIGKILL mid-trial then resume: 5/5 trials, 0 failed; **evaluation draw reproduces**
(reload in a fresh process equal, trial dir rebuilds the same cells, seed + 1 negative control detected; 4 groups, 19,606
cells). Details `dryrun_evidence/d3_launcher.md`.

## Full suite (D5)
`dryrun_evidence/d5_suite.txt`: 691 tests, **OK** (job 16795 step 16, g2tn01, TITAN RTX, head `55aa3f7`, pinned kernel environment E as `scripts/slurm/run_tests.sbatch`, 1,211 s). A first run without E (step 15) failed only the 3 bit-exact tests on their environment guard ("bit-exact test needs the pinned kernel environment"); `d5_suite.sh` now sets E. Golden and mb0 compares pass (no `src/` change).

## Heterophily table (D4)
Not produced: `dryrun_evidence/d4_stop.md` (5 differences with file:line; no job run). D4b comes in a separate prompt.

## Baseline cells ready to tune now

Ready = OK at the largest point on 24 GB, and not changed by card 1d (Hetero²Net HGB NC validation route, PSHGCN on MAG,
H3b cells) or card 3 (HetSheaf; Freebase NC of SeHGNN, PSHGCN, H2SGNN): **101 cells** (NC 28, LP 37, GC 36). Full list
with peak memory: `dryrun_evidence/d5_ready_baselines.md`; launcher lines: `dryrun_evidence/d5_ready_baselines.txt`.
Held back although OK: Hetero²Net NC acm, dblp, imdb (card 1d D2). Not in the list: the 16 step-down "fits" cells (they
OOM at the largest point: decision below).

| task | models × datasets |
|---|---|
| NC | chebnetii: acm, dblp, freebase, imdb; csnn: acm, dblp, imdb; gat: acm, dblp, freebase, imdb; gcn: acm, dblp, freebase, imdb; h2sgnn: acm, dblp, imdb; han: acm, dblp; nsd: dblp, imdb; pshgcn: acm, dblp, imdb; sehgnn: acm, dblp, imdb |
| LP | csnn: acm_pa, dblp_ap, imdb_ma, lastfm, pubmed; gat: dblp_ap, imdb_ma, lastfm, pubmed; gcn: acm_pa, dblp_ap, freebase_bb, imdb_ma, lastfm, pubmed; h2gformer: pubmed; h2sgnn: acm_pa, dblp_ap, lastfm, pubmed; han: acm_pa, dblp_ap, lastfm, pubmed; hetero2net: acm_pa, dblp_ap, freebase_bb, imdb_ma, lastfm, pubmed; nsd: dblp_ap, imdb_ma, lastfm; pshgcn: acm_pa, dblp_ap, imdb_ma, lastfm |
| GC | csnn: enzymes, mutag, nci1, proteins; gat: enzymes, mutag, nci1, proteins; gcn: enzymes, mutag, nci1, proteins; h2gformer: enzymes, mutag, nci1, proteins; h2sgnn: enzymes, mutag, nci1, proteins; han: enzymes, mutag, nci1, proteins; hetero2net: enzymes, mutag, nci1, proteins; nsd: enzymes, mutag, nci1, proteins; pshgcn: enzymes, mutag, nci1, proteins |

Commands (W&B project `h2snn_baseline`; spec §6: plain Bayes, 100 trials, seed 0). The D3 launcher fixes are on
`dry-run` only: run from a checkout that has them (this worktree, or `v3b` after `dry-run` is merged).

```
cd /mnt/data/peeradon/h2snn-dryrun/H2S-NN          # or the main checkout once dry-run is merged into v3b
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
mkdir -p logs
sbatch --array=1-101%4 --exclude=g3an01 scripts/slurm/run_sweep_agents.sbatch ../docs/dryrun_evidence/d5_ready_baselines.txt
# resume a line N after a kill or time limit (same command, same line):
sbatch --array=N scripts/slurm/run_sweep_agents.sbatch ../docs/dryrun_evidence/d5_ready_baselines.txt
```
Each line is `scripts/run_baseline_tuning.py --task <t> --models <m> --datasets <d> --num_runs 1 --base_seed 0
--n_trials 100 --output_dir experiments/baselines --wandb_project h2snn_baseline` (`--resume` is added by the sbatch).
`%4` = one agent per free GPU (g2n01 11 GB can run only the cells with peak < 10 GiB; `d5_ready_baselines.md` lists
peaks). The seeds 0–4 runs of the best configuration have no baseline launcher yet (decision below).

## H2S-NN cells: waiting for freeze

| cell | status at the largest point (24 GB) | note |
|---|---|---|
| NC acm, dblp, imdb | OK | |
| NC freebase | OOM | r 32 still OOM; A6000 pending (16889) |
| LP acm_pa, imdb_ma | OK | |
| LP dblp_ap, pubmed, lastfm | OOM | fit at 24 GB with r ≤ 32; A6000 pending (16889) |
| LP freebase_bb | OOM | r 32 still OOM; A6000 pending (16889) |
| GC mutag, proteins | OK | |
| GC enzymes, nci1 | stop condition | log-weight range > 30 at epoch 0 |
| NC ogbn-mag, mag-year, oag-cs, oag-eng, oag-chem, ieee-cis-g, pdns (B 1024) | OK | full validation pass 544 s / 3,394 s / 137 / 155 / 449 / 77 / 79 s |
| NC h-pokec | host OOM (30G) | 120G on g3an01 pending (16875) |
| LP ogbn-mag_pp, h-pokec_uu | not run | no H2S-NN entry point (D0); out of D1–D4 |

## Decisions for the author
1. **OOM cells at 24 GB** (table above): for each, run on the A6000 only, or accept the stepped-down value as the
   largest value tuned on 24 GB GPUs (a range change, so yours), or both. 11 cells do not fit even one step lower
   (H2S-NN Freebase NC and freebase_bb; NSD 4; HINormer 3; GAT and PSHGCN freebase_bb).
2. **enzymes / nci1:** the log-weight-range stop (> 30) fires at epoch 0 at the largest point (d 64, ℓ_max 5, r 64,
   attention). Per spec §6 (1.16) such a trial is "failed" and the search continues; whether the largest point should
   stay in the space for these two datasets is your decision. Nothing was changed.
3. **h-pokec:** H2S-NN needs > 30G host memory (killed at 31.5 GB during the training epoch at B 1024, r 64); only
   g3an01 (225G) can hold it under the 60G rule's exception. 120G retry pending (16875).
4. **H2S-NN sweep has no batch-size axis** (D1): spec 1.17 tunes B ∈ {32, 256, 1024} on H2GB NC for every method.
5. **Validation time on H2GB NC** (not a stop): one full validation pass of H2S-NN is 3,394 s on mag-year (544 s
   ogbn-mag, 449 s oag-chem); with `eval_period` 4 over 500 epochs that is 125 passes per trial.
6. **Seeds 0–4 for baselines:** `run_baseline_tuning.py` tunes per seed unit; there is no step that reruns a cell's best
   configuration on seeds 0–4 (H2S-NN has `--final_seeds`). Needed before the baseline main table.
7. **Large-LP H2S-NN wiring** (D0): `ogbn-mag_pp` and `h-pokec_uu` have no H2S-NN entry point.
8. **D4:** an M2 harness card, or acceptance of Part J's script for HGB NC rows only (`d4_stop.md`).
9. **Session 1:** the 11 H3b cells plus HAN Freebase (NC segfault; LP segfault at heads 4).

branch=dry-run sha=<the commit adding this report; full sha in the final message> card=docs/H2S-NN_dryrun_card.md base=6ba6dec00369ff590b341ce3771da90528fcc1cd
