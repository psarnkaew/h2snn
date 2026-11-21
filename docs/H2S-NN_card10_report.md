# Card 10 report: lab-cluster runs (SLURM): all units done

**Verdict.** All units are done: L1–L4, A2.1–A2.3 and Amendment 3 (A3.1, A3.2, A3.3). The A3.3 job 18119 (CPU only, 16G, no GPU, the `run_tests.sbatch` environment) ran 293 tests, all OK, none skipped, and all 9 card 10 mutations were caught (A3.1/A3.2 job 18116 (CPU only, 16G, no GPU, the `run_tests.sbatch` environment) ran 291 tests, all OK, none skipped, 8 mutations caught). No `src/` change was made: `git diff --stat 2d822a4 -- H2S-NN/src` is empty. No real-data run was made, and the real fit check was **not submitted**; it waits for the author's approval. Base: `v3b` `2d822a4` (card 8), merged as `d9b324c`.

**A bug from before this card, found in A3.2 and fixed in A3.3.** `scripts/run_baseline_published.py` (card 8) has no `--resume`, but both slot runners (`sweep_slots.sh`, `lab_slots.sbatch`) appended `--resume` to every line. So on `2d822a4` **every `published` line fails at start** with `unrecognized arguments: --resume` (exit 2; checked in a CPU job). This includes the TSUBAME `published` phase (38 lines). A3.2 had made `--resume` a no-op in the runner; A3.3 (ops review) removes that dummy flag. The slot scripts now pass `--resume` only to launchers that define it, so a published line runs without it. Any published chain submitted on TSUBAME from `2d822a4` will fail on every line until `v3b` contains A3.3.

## Units

| Unit | Commit | Files | Evidence |
|---|---|---|---|
| L1 lab slot runner | `cd0de1a` | `scripts/slurm/lab_slots.sbatch`, `scripts/slurm/submit_lab.sh`, `tests/test_lab_slots.py` | job 18033: 4 tests OK; `mutation_l1.py` (GPU-type record not checked) caught |
| L2 fit check | `298fd78` | `scripts/slurm/grid_points.py`, `scripts/slurm/fit_check.py`, `--grid_point` in `scripts/run_baseline_tuning.py` and `scripts/run_h2snn_wandb_tuning.py`, `tests/test_fit_check.py` | job 18107: 10 tests OK; `mutation_l2.py` (size axes at their minimum) caught on both launchers |
| A2.2 75 % headroom | `298fd78` (with L2) | `fit_check.py` (`HEADROOM = 0.75`, `peak_fraction`) | job 18107; `mutation_a22.py` (threshold 100 %) caught |
| L3 site split | `9ead523` | `scripts/slurm/site_split.py`, `tests/test_site_split.py` | job 18035: 3 tests OK; `mutation_l3.py` (a line in both lists) caught |
| A2.1 whole-H100 TSUBAME types | `8a8d524` | `scripts/tsubame/submit_chain.sh`, `sweep_slots.sh` (header), `tests/test_tsubame_types.py` | job 18040: 6 tests OK; `mutation_a21.py` (`node_h` S = 4) caught |
| A2.3 failure report | `ca119cd` | `scripts/tsubame/failures.py`, `tests/test_failures_report.py` | job 18041: 1 test OK; `mutation_a23.py` (a failure skipped) caught |
| L4 targeted tests | `50436f4` | `card10_evidence/run_l4.sbatch` | job 18109: 272 tests OK, 0 skipped; all 6 mutations caught; `src/` diff empty |
| A3.1 memory-relevant keys | `b4496c9` | `scripts/slurm/grid_points.py` (`MEMORY_AXES`), both launchers (one trial per grid point, `FIT_GRID`) | job 18116; `mutation_a31.py` (memory keys at their central value) caught on both launchers |
| A3.2 published rows | `685ea8e` | `scripts/run_baseline_published.py` (`--dry_run`, `FIT_PEAK`; its `--resume` no-op removed by A3.3), `fit_check.py` (published probes, grid reading), `site_split.py` (H2GB published → TSUBAME), `tests/test_fit_check.py` | job 18116: 291 tests OK, 0 skipped; `mutation_a32.py` (published lines without a probe) caught; all 8 mutations caught |
| A3.3 `--resume` only where defined | `c45ae79` | `scripts/tsubame/sweep_slots.sh`, `scripts/slurm/lab_slots.sbatch` (`resume_flag`), `scripts/run_baseline_published.py` (dummy `--resume` removed), `tests/test_slot_resume.py`, `tests/test_fit_check.py` | job 18119: 293 tests OK, 0 skipped; `mutation_a33.py` (`--resume` appended unconditionally) caught on both scripts; all 9 mutations caught; `src/` diff empty |

L4's test set:
- every card 10 test;
- every `test_tsubame_*` test, plus `test_card7_tsubame` and `test_card8_phases`;
- every test that touches the two launchers changed in L2: `test_baselines`, `test_baseline_final_seeds`, `test_b1a_validation_subset`, `test_card4_ablation_driver`, `test_card4b_prefreeze`, `test_card5_protocol_hooks`, `test_card7_perturbation`, `test_dry_run_flag`, `test_partM3a_decoder`, `test_partO_reload`, `test_search_spaces`, `test_sweep_batch_size`, `test_paths_and_facts`, `test_wandb_logging`, `test_wandb_and_runs`.

**First L4 attempt (job 18108, log not kept): an environment problem, not a code failure.**
- **Result:** 3 failures and 21 errors.
- **Causes:**
  - the new worktree had no `vendor/` clones and no `data` link;
  - I exported `WANDB_MODE=disabled`;
  - the pinned CPU kernel settings of `run_tests.sbatch` were missing.
- **Fix:** I set the worktree up as guide §3 describes for a checkout: `clone_vendors.sh`, `apply_vendor_patches.sh`, and `data` → the lab data root (untracked, never staged). `run_l4.sbatch` now exports the `run_tests.sbatch` environment. Job 18109 then passed everything.

## L2 and A2.2: what they do

**Launchers (scripts only).** `--grid_point {min,max}` is accepted only with `--dry_run --n_trials 1`, the same guard as card 8, and the test checks it on both launchers.
- **`min`:** unchanged on the baseline launcher (card 8's `smallest_grid_point`). On the H2S-NN launcher it is the same function.
- **`max`:** every **size axis** at its largest value, every other axis at its central value, run as a W&B `grid` of one point. The baseline launcher builds it in the script (`cell_sweep_config`), so `src/baselines/campaign.py` is not changed.
- **Size axes (`grid_points.SIZE_AXES`):**
  - widths: `hidden*`, `emb*`, `embed*`, `attn_vec_dim`
  - layers, hops and order: `*layers*`, `num_gnns`, `K`, `num_hops`, `num_label_hops`, `len_seq`, `n_bases`
  - heads: `num_heads`, `attn_heads`
  - the rest: `d`, `wisp_max_length` (ℓ_max), `wisp_walks_per_anchor` (r), `batch_size`
- **Every search key is classified.** A key that is neither a size axis nor in `OTHER_AXES` is refused, so a new size axis can never silently get its central value. The lists cover every key of every runnable baseline cell (`card10_evidence/search_space_keys.txt`, job 18105) and of H2S-NN.
- **Central value:**
  - numeric list: the lower median;
  - categorical list: the middle entry in listed order;
  - uniform range: the midpoint; log-uniform range: the geometric midpoint.
- **Test:** on every runnable baseline cell and every `*_h2snn_v3b.yaml`, each size axis is at its maximum (H2S-NN: `d` 64, ℓ_max 5, r 64, `batch_size` 1024 on H2GB NC) and every other axis is at its central value.

**Peak line (A2.2).** Every trial, of both launchers and in every run (not only probes), prints one line: `FIT_PEAK {json}`.
- **When:** after training and after the evaluation of the whole validation and test splits.
- **`peak_bytes`:** the allocator's peak reserved memory, plus the device memory held outside the allocator at the end (CUDA context, library workspaces). This is an upper estimate of the device peak.
- **`phases`:**
  - `train`: H2S-NN's epoch callback received training metrics; for baselines, `fit_cell` returned;
  - `val` and `test`: the record holds the validation monitor and a test metric, i.e. the whole-split evaluations. For H2GB (B1a) these come after training, because the epoch callback sees only the validation subset.
- **Test:** faked trials of both launchers print the three phases, and a missing phase is shown as missing.

**`fit_check.py`.** Three subcommands:
- **`build`:** one probe line per cell (a cell's tuning and final-seeds lines share one probe). The cell keeps its token. The line writes only under `--fit-out`, with W&B project `<project>_fitcheck`, and carries `--n_trials 1 --dry_run --grid_point max`. A line carrying an option the probe would drop is refused.
- **`submit`:** runs `submit_lab.sh` for one leg.
- **`read`:** writes `fit_check.tsv` with columns `line_no, cell, gpu_type, peak_mem_mib, peak_fraction, status, line`. Status per cell:

| Status | When |
|---|---|
| `fits` | all three phases measured and peak ≤ 75 % of GPU memory |
| `oom` | an OOM in the log, or all three phases measured and peak > 75 % |
| `error` | anything else, including a missing phase |

The test also feeds this tsv to `site_split.py`: only the `fits` cell goes to the lab.

**Check on the real lists (2026-11-21).** Before A3, job 18112 built the probe lists: published lines were skipped then. After A3, job 18117 built them again with every published line probed; the counts are in the A3.1 cost table. No line was refused for an option the probe would drop.

## A3.1: memory-relevant keys at their most memory-hungry value
`grid_points.MEMORY_AXES` is a third class next to the size axes and the other axes. Under `--grid_point max` each of its keys takes its most memory-hungry value. Where the code does not show which value that is, the probe runs **one trial per value** (a W&B grid), with every size axis at its maximum, and the cell's peak is the largest over those trials. A key in none of the three classes is still refused.

| Key | Most memory-hungry value | Reason (source) |
|---|---|---|
| `feats_type` | **all of 0, 2, 3** | 0 keeps dense attributes, 2/3 give a type a sparse identity as wide as its node count (`src/baselines/bundles.py:202-230`): depends on the dataset |
| `label_feats` (SeHGNN) | True | builds the propagated label features (`src/baselines/adapters/sehgnn.py:146-160,300`) |
| `residual` (SeHGNN) | True | adds `res_fc` (`vendor/SeHGNN/hgb/model.py:142-143,218-219`) |
| `sheaf_model` (NSD) | **BundleSheaf and GeneralSheaf** | Diag learns d values per edge (`disc_models.py:39`). General learns d×d (231) and inverts d×d degree blocks by SVD (`laplacian_builders.py:280-286`); Bundle adds orthogonal-map intermediates per edge (`orthogonal.py:45-64`). Not clear which is larger |
| `sheaf` (HetSheaf) | **all 32 bundle/general pairs** | diag is the smallest family; bundle vs general and the 16 learners are not clear (`src/baselines/adapters/hetsheaf.py:105-123`) |
| `orth` (NSD, CSNN) | cayley | d(d+1)/2 values per map vs d(d−1)/2 (`disc_models.py:146-150`, `Cooperative-Sheaf-NN/model.py:322-326`) and a d×d solve per edge (`orthogonal.py:61-64`); euler is the smallest and exists only for d ≤ 3 |
| `add_lp`, `add_hp` (NSD) | True | each adds one stalk dimension (`neural-sheaf-diffusion/models/sheaf_base.py:21-26`) |
| `conformal` (CSNN) | True | one more learner output per node (`Cooperative-Sheaf-NN/model.py:523`) |
| `gnn_type` (CSNN) | GAT | per-edge attention on top of the per-edge messages GCN/SAGE also build (`model.py:577-583`) |
| `metapath_pe` (H2G-former) | True | adds the Hetero_Metapath channel (`h2gformer.py:20-22,229-231`) |
| `mask_lp` (Hetero2Net) | True | adds the label-embedding input (`hetero2net.py:269-273`) |
| `r` (Hetero2Net) | **0.3 and 0.5** | starts random walks and drops their paths (`Hetero2Net/hetero2net/metapath.py:75-125`): not clear |
| `poly` (H2SGNN) | **all three bases** | different recurrences (`H2SGNN/model.py:80-90,129`): not clear |
| `attn_dropout` (H2G-former) | 0.2 | the attention dropout (`H2GB/layer/gt_layer.py:203`) stores a mask only for p > 0, and the central value is 0.0 |
| `decoder_type` (LP, H2S-NN and baselines) | distmult | per-relation weights and one more product per scored pair (`src/models/link_readout.py:46,65`; `src/baselines/heads.py:65`) |
| `graph_pooling` (H2S-NN GC) | attention | a gate network and per-node scores (`src/models/graph_readout.py:55`) |

**Keys probed once per value** (bold above): `feats_type`, `sheaf_model`, `sheaf`, `r`, `poly`. Two decisions differ from the ops list: `sheaf_model` probes Bundle *and* General (not General alone), and `feats_type` probes 0, 2 and 3 (not 0 alone), for the reasons in the table. Numeric rates, regularisers, temperatures and dropouts whose central value is already > 0 stay at their central value: their size does not change which tensors exist.

**Mechanics.**
- The launchers set the sweep's trial count to the grid size (the command line still says `--n_trials 1`, meaning "one probe") and print `FIT_GRID {"points": k}`.
- `fit_check read` needs k `FIT_PEAK` lines, all with three phases, and takes the largest. Fewer gives `error`; an OOM in any trial gives `oom`.
- The test checks every runnable baseline cell and every H2S-NN config against a decision table written independently in the test, and the grid sizes on both launchers' real `main` paths (NSD ACM 6 trials, GCN ACM 3, H2S-NN ACM LP 1).

**Cost of the grids (job 18117, lists only).**

| Phase | Probe lines | Trials |
|---|---|---|
| `h2snn` | 24 | 24 |
| `baselines-early` | 4 | 70 |
| `baselines-1` | 61 | 296 |
| `baselines-2` | 112 | 556 |
| `published` | 38 | 38 |

HetSheaf is 640 of these trials (32 per cell). Each trial is one epoch, but the trials of one line run one after another on one GPU (`card10_evidence/probe_counts.txt`). If this is too long, a choice for the author: probe HetSheaf's bundle and general families only with their default learner, which would need a reason from the code that the learner does not change memory (not shown so far).

## A3.2: published lines get a fit-check row
- **Probe line:** `fit_check build` writes one probe per `published` line: `run_baseline_published.py --task … --models … --datasets … --final_seeds 0 --dry_run`, under `--fit-out`. The runner trains the published configuration as `published_params` gives it.
- **What `--dry_run` does:** one epoch, through `build_raw_base(..., epochs=1)`, the same campaign epoch override as the tuned `--dry_run` (`src/baselines/campaign.py:272`; no `src/` change). Then comes the record's whole-split evaluation, and the run prints `FIT_PEAK` with the three phases.
- **Guard:** `--dry_run` only with exactly one seed in `--final_seeds`.
- **Same rules as tuned cells:** the same thresholds and the same `fit_check.tsv` row.
- **`site_split`:** `fits` goes to the lab, anything else to TSUBAME, and an H2GB published cell always goes to TSUBAME (range rule). Its dataset list equals `src/configs/config.py` `H2GB_NC_DATASETS`, which the test checks.
- **Test:** published GCN LastFM LP (fits → lab), R-GCN PubMed LP (oom → TSUBAME), PSHGCN IMDB NC (no log → error → TSUBAME), SeHGNN ogbn-mag NC (fits, but H2GB → TSUBAME).
- **Real lists:** every one of the 38 published lines gets a probe (table above). H2GB published lines are probed too, as the amendment says, even though `site_split` sends them to TSUBAME regardless. Leaving them out of the probe list would save lab time; author's choice.

## A3.3: `--resume` only for launchers that define it
- **Rule:** both slot scripts have one `resume_flag` function, which keys on the line's first word.
  - It gives `--resume` to `run_baseline_tuning.py` and `run_h2snn_wandb_tuning.py`, the two tuning launchers, which resume their sweep state.
  - It also gives it to `scripts/tsubame/run_row.py`. Card 7's row lines were written to receive the slot's `--resume`, and `run_row.py` drops it before calling its launcher (`run_row.py:53`), so keeping it there changes nothing for card 7.
  - Every other launcher, including `run_baseline_published.py`, runs exactly as the list line states it.
  - The existing skip stays: a line whose final seeds are all written is not run again (`lines.py complete`).
- **Logs:** the logged `agent:` command shows the call as actually made, with or without `--resume`.
- **Runner:** `run_baseline_published.py` has no `--resume` again. `tests/test_fit_check.py` now checks that it refuses one (exit 2, `unrecognized arguments`).
- **Test (`tests/test_slot_resume.py`):** one list with a baseline tuning line, an H2S-NN line, a `run_row.py` line and a published line runs through each slot script.
  - The scripts run under a fake `python` that records the launcher's argv.
  - Tuning and row lines must end in `--resume`; the published argv must equal the line's words exactly.
  - Mutation `mutation_a33.py` makes `--resume` unconditional, and both tests fail.

## Deviations and points for the author
1. **L1 additions (accepted, Amendment 1):** the GPU-type record is keyed by cell; the job checks the GPU name; `--cpus` defaults to 4; `--max-parallel` is optional.
2. **`FIT_PEAK` is printed by every trial, not only by probes.** The trial worker cannot tell a probe from a real trial (it gets only `--worker_context`), and the line is harmless, but it does appear in real-run logs.
3. **Memory-relevant keys (A3.1)** are decided as in the table above. Two of them differ from the ops list, `sheaf_model` and `feats_type`, because the code does not show one largest value.
4. **Published lines (A3.2)** get a probe and a row. The `--resume` bug above is fixed in the slot scripts (A3.3), not in the runner. `run_row.py` also keeps `--resume` (A3.3), beyond the two tuning launchers the amendment names, because card 7 expects and drops it.
5. **Seen while reading the code, not changed (outside this card):** NSD and CSNN tune `orth` in {householder, cayley, euler} together with `d` up to 5 (NSD) or 4 (CSNN). Euler asserts 2 ≤ d ≤ 3 (`vendor/neural-sheaf-diffusion/models/orthogonal.py:70`; `vendor/Cooperative-Sheaf-NN/model.py:484`), so real sweep trials with euler and a larger d fail (for NSD only with BundleSheaf, the one sheaf model that uses `orth`). `failures.py` will report them as `other`.
6. **`peak_bytes` is an upper estimate.** It is the allocator's peak reserved memory plus memory outside the allocator measured at the end of the trial. It can miss a transient non-allocator peak (e.g. a cuSPARSE workspace) during training; the 25 % headroom is there to cover that.

## How to run (after the author approves this card; nothing below was run)
```
SHA=<approved v3b sha>; RUN=/mnt/data/peeradon/h2snn-runs/$SHA; OUT=/mnt/data/peeradon/h2snn-out-lab/$SHA
git clone git@github.com:psarnkaew/h2snn.git $RUN && cd $RUN && git checkout --detach $SHA
cd H2S-NN && bash scripts/setup/clone_vendors.sh && bash scripts/setup/apply_vendor_patches.sh
ln -s /mnt/data/peeradon/h2snn/H2S-NN/data data            # vendor/ stays a real clone
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
# H2S-NN lines (the GC lines are in --phase h2snn) and a baseline list, written with the lab --out-dir
python scripts/tsubame/make_cell_list.py --phase h2snn       --out $OUT/cells_tuning_h2snn.txt
python scripts/tsubame/make_cell_list.py --phase baselines-2 --out $OUT/cells_tuning.txt
# fit check on the GPU type the list would use (one fit-out per list and GPU type)
python scripts/slurm/fit_check.py build  --list $OUT/cells_tuning_h2snn.txt --fit-out $OUT/fit_h2snn_3090
python scripts/slurm/fit_check.py submit --fit-out $OUT/fit_h2snn_3090 --gpu-type 3090 --mem 48G --time 6:00:00
python scripts/slurm/fit_check.py read   --fit-out $OUT/fit_h2snn_3090 --gpu-type 3090
# split (T4 side: a local mirror of TSUBAME's state/ and experiments/), then submit the lab part
python scripts/slurm/site_split.py --list $OUT/cells_tuning_h2snn.txt --fit-check $OUT/fit_h2snn_3090/fit_check.tsv \
    --gpu-type 3090 --lab-out $OUT --t4-out <t4 mirror>
bash scripts/slurm/submit_lab.sh --list $OUT/cells_tuning_h2snn.lab.txt --out $OUT --gpu-type 3090 --mem 48G --time 72:00:00 --legs 3
# the baseline list the same way (its own fit-out and GPU type), then watch, collect, failures
python scripts/tsubame/status.py   --out $OUT --list $OUT/cells_tuning_h2snn.lab.txt
python scripts/tsubame/collect.py  --out $OUT
python scripts/tsubame/failures.py --out $OUT --list $OUT/cells_tuning_h2snn.lab.txt   # reported to the author; nothing moves
```
- **On TSUBAME:** the `.t4.txt` part goes to `submit_chain.sh` with any whole-H100 type: `gpu_1`, `node_q`, `node_h` or `node_f` (A2.1).
- **Placeholders:** `--mem` and `--time` above are placeholders. Lab nodes have 64 GB RAM (the A6000 node 224 GB), and each run is capped at 72 h.

branch=card10 sha=a821f3f7784dcbb9cce78f0940b461de7013865b card=docs/H2S-NN_card10_lab_runs_card.md base=2d822a4eb48867de4476b97159430a64d8468808
