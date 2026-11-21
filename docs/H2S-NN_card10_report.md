# Card 10 report: lab-cluster runs (SLURM): all units done

**Verdict.** All units are done: L1–L4 and A2.1–A2.3. The L4 job 18109 (CPU only, 16G, no GPU) ran 272 tests, all OK, none skipped, and all 6 card 10 mutations were caught. No `src/` change was made: `git diff --stat 2d822a4 -- H2S-NN/src` is empty. No real-data run was made, and the real fit check was **not submitted**; it waits for the author's approval of this card. Base: `v3b` `2d822a4` (card 8), merged into `card10` as `d9b324c`.

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

**Check on the real lists (2026-11-21).** `fit_check.py build` was run on the lists that `make_cell_list.py` writes at this head: lists only, nothing submitted; `h2snn` on the login node (stdlib), the baseline phases in CPU job 18112, outputs deleted afterwards.

| Phase | Probe lines | Skipped |
|---|---|---|
| `h2snn` | 24 | 0 |
| `baselines-early` | 4 | 0 |
| `baselines-1` | 61 | 0 |
| `baselines-2` | 112 | 0 |
| `published` | 0 | 38 (no search space) |

No line was refused for an option the probe would drop.

## Deviations and points for the author
1. **L1 additions (accepted, Amendment 1):** the GPU-type record is keyed by cell; the job checks the GPU name; `--cpus` defaults to 4; `--max-parallel` is optional.
2. **`FIT_PEAK` is printed by every trial, not only by probes.** The trial worker cannot tell a probe from a real trial (it gets only `--worker_context`), and the line is harmless, but it does appear in real-run logs.
3. **Categorical axes are put at their central value even when they may change memory**, as the card says ("all other keys at their central value"):
   - `feats_type` (0 = all features, the largest; central is 2)
   - `label_feats`, `add_hp`/`add_lp`, `metapath_pe`
   - `sheaf`/`sheaf_model` (`GeneralSheaf` is the largest)
   - `gnn_type`, `graph_pooling`, `decoder_type`

   A cell whose memory peak sits on one of these may pass the check and later OOM on the lab. A2.3's failure report would show that OOM, and the author decides. Alternative: treat these axes as size axes at their most memory-hungry value. **Author's choice.**
4. **Published-config lines (`run_baseline_published.py`, card 8)** have no search space. `fit_check.py build` lists them as skipped and gives them no fit row, so `site_split.py` sends them to TSUBAME.
5. **`peak_bytes` is an upper estimate.** It is the allocator's peak reserved memory plus memory outside the allocator measured at the end of the trial. It can miss a transient non-allocator peak (e.g. a cuSPARSE workspace) during training; the 25 % headroom is there to cover that.

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
