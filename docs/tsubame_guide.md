# TSUBAME4 guide for H2S-NN (only what we use)

Written 2026-11-18 by the design session (ops). Sources: `docs/tsubame_handbook.en.pdf` (TSUBAME4.0 User's Guide, 2026-11-10; read §1–§5, §7.4, §7.6.12; the commercial-application chapters were skipped) and the limits page https://www.t4.cii.isct.ac.jp/en/resource-limit (last update 2025-12-09, read 2026-11-18). Items marked **UNVERIFIED** have not been checked on TSUBAME. Card T0–T6 (`docs/H2S-NN_tsubame_card.md`) checks them. Scripts under `scripts/tsubame/` are written by that card. Until it is approved they do not exist.

## 0. What TSUBAME is for (author, 2026-11-17)
- **Real runs only**: tuning sweeps and final seeds. Development, tests, mutation checks and analysis stay on the lab cluster.
- **Baselines first** (author, 2026-11-18): phase 1 is the baseline cells on HGB NC (ACM, DBLP, IMDB) and the small LP tasks (`acm_pa`, `dblp_ap`, `imdb_ma`, `pubmed`, `lastfm`). Freebase, H2GB NC, the large LP tasks, GC and H2S-NN come later, after their cards are on `v3b` (H2S-NN after the freeze).
- **Every reported number comes from TSUBAME's H100.** Lab-cluster runs (2080 Ti, TITAN RTX, 3090) never enter a table. Results are not bit-identical across GPU types. Using one GPU type for every method keeps the comparison fair.

## 1. The machine in one table
| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA H100 SXM5, **94 GB**, 4 per node | handbook §1.2 |
| Scheduler | **Altair Grid Engine** (`qsub`, `qstat`, `qdel`, `qrsh`), **not SLURM** | §5 |
| Max run time per job | **24 h** (longer only with a node reservation) | §5.1.1; limits page |
| Jobs running at once per user | **30** on weekdays, **100** on weekends (start between Fri 16:00 and Sun 09:00) | limits page |
| Jobs submitted at once per user | 2,000 (running + waiting) | limits page |
| CPU slots at once per user | 6,144 weekdays / 12,288 weekends | limits page |
| Per-user GPU limit | none listed; the job count is the limit | limits page |
| Home + work quota | **125 GiB together** per user | §3.2 |
| Login nodes | `login.t4.gsic.titech.ac.jp` (also `login1`, `login2`); no GPU; 4 GB per process; nothing running longer than 10 min, no Python computations, no servers (VS Code server, Jupyter) | §2.2.2 |
| Compute-node internet | the system is on SINET6 at 100 Gbps; outbound HTTPS from a **compute** node (pip, GitHub, W&B) is **UNVERIFIED** | §1.1 |

### Resource types we use
| Type | CPU cores | Host RAM | GPUs | Local SSD | Use |
|---|---|---|---|---|---|
| `gpu_1` | 8 | 96 GB | 1 | 200 GiB | one sweep agent per job |
| `node_f` | 192 | 768 GB | 4 | 1,660 GiB | **four sweep agents per job** (one per GPU) |
| `cpu_8` | 8 | 18.4 GB | 0 | 40 GiB | environment install, cell lists, collection |

Why `node_f`: the limit counts jobs, not GPUs. With 30 jobs on weekdays, `gpu_1` gives at most 30 GPUs at once and `node_f` gives at most 120. (Slots allow 32 `node_f` jobs, so 30 jobs is the binding limit.) A whole node may wait longer in the queue. `scripts/tsubame/` supports both types, and `node_f` is the default.
Not used: `node_h`, `node_q`, MIG halves (`node_o`, `gpu_h`; half an H100 would change memory limits per cell), the interactive queue (one job, shared), Apptainer, MPI.

## 2. One-time setup (author)
1. **Account and group** (TSUBAME portal): account, SSH public key registered, member of the paying TSUBAME group. Write the group name into `~/.t4_group` on TSUBAME (one line). The scripts read it, so it never enters git.
2. **Login from the lab cluster or the laptop**:
   ```
   ssh <t4-login>@login.t4.gsic.titech.ac.jp -i ~/.ssh/<t4-key>
   ```
3. **Storage layout** (all under the work directory, `WORK=${HOME/home/work}`; home and work share the 125 GiB quota):
   ```
   $WORK/h2snn-runs/<v3b-sha>/      one read-only checkout per approved v3b SHA (code + vendor clones)
   $WORK/h2snn-data/               datasets (data/NC, data/LP, data/GC), shared by every checkout
   $WORK/h2snn-out/<v3b-sha>/      outputs, sweep state, logs of that checkout
   $HOME/miniforge3                conda (env h2snn-b)
   ```
   Budget (measured on the lab cluster 2026-11-18): data 29 GB (NC 27 GB, mostly OAG 9.2 GB and MAG 4.2 GB; LP 1.9 GB; GC 40 MB), env `h2snn-b` 3.7 GB plus Miniforge, outputs small for phase 1. Phase 1 needs only `NC/{acm,dblp,imdb}` and the LP files (< 2 GB). If phase 2 outputs (checkpoints) approach the quota, buy a group disk on `/gs/bs`. Check usage with `t4-user-info disk home`.
4. **Lustre**: many small files are slow and load the metadata server (§3.1). Never untar data into thousands of files when one archive will do. `collect.py` writes one tar per cell.
5. **W&B**: on TSUBAME run `wandb login` once yourself (the key goes to `~/.netrc`). Never write the key into a file in the repo or a job script.

## 3. Code, vendors, data
On the **login node** (git, rsync and tar are allowed there):
```
SHA=<approved v3b sha>
git clone git@github.com:psarnkaew/h2snn.git $WORK/h2snn-runs/$SHA      # or https
cd $WORK/h2snn-runs/$SHA && git checkout --detach $SHA
cd H2S-NN && bash scripts/setup/clone_vendors.sh && bash scripts/setup/apply_vendor_patches.sh
ln -s $WORK/h2snn-data data                                             # one data copy for every checkout
```
From the **lab cluster login node** (or the laptop), copy data and the DGL package:
```
rsync -avL /mnt/data/peeradon/h2snn/H2S-NN/data/ <t4>:<WORK>/h2snn-data/      # all of data/: ≈ 16 GB (2026-11-19)
rsync -av /mnt/home/peeradon/wheels/dgl-2.4.0.th24.cu124-py312_0.tar.bz2 <t4>:~/wheels/
```
Copy **all** of `data/` once (amended 2026-11-19, card 5 D3): HGB NC (acm, dblp, imdb, freebase), the 8 H2GB NC datasets (`NC/mag`, `NC/OAG`, `NC/IEEE-CIS`, `NC/Pokec`, `NC/PDNS`), LP including the large-LP sources, GC, and the perturbed graphs of card 7 (`NC/*_pert`). Sizes on the lab cluster: `NC/OAG` 9.2G, `NC/mag` 4.2G, the rest < 1G each; total ≈ 16G of the 125 GiB home+work quota. After the copy, `du -sh $WORK/h2snn-data` must match the lab's `du -shL data` (state both in the T6 evidence).
**Never make `vendor/` a symlink.** Vendor-module unloading compares resolved module paths with the vendor folders, so a symlinked `vendor/` breaks 86 baseline/vendor tests (decisions log 2026-11-19, row "Model change 1.25 merged"). `vendor/` is always a real clone made by `clone_vendors.sh` inside each checkout; only `data` is a symlink.
Rule: a checkout is never edited and never moved to another SHA. A new approval gets a new directory. Sweep agents re-import the code at every trial, so editing a live checkout would mix code versions inside one sweep.

## 4. Environment (once, in a CPU job, never on the login node)
Miniforge: download the installer on the login node and run `bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3` (extracting files is allowed there). The env is built in a `cpu_8` job, `scripts/tsubame/install_env.sh` (T1). It wraps the existing `scripts/setup/install_env_b.sh` (torch 2.4.1+cu124, DGL 2.4.0 conda package checked by sha256, the PyG wheels, the lock file) and then runs `build_sparse_tools.sh` (CPU extension):
```
qsub -g $(cat ~/.t4_group) -l cpu_8=1 -l h_rt=2:00:00 scripts/tsubame/install_env.sh
```
If compute nodes have no internet (**UNVERIFIED**), the install runs on the login node in steps of under 10 minutes, or the pip wheels are downloaded on the lab cluster and copied over. T6 finds out which.

Then check the GPU stack on an H100 with `scripts/tsubame/env_check.sh` (T1). It prints the driver and CUDA version and runs torch, DGL and PyG CUDA operations, `tests/test_check_env.py`, the CUDA-only test that lab CPU suites skip, and a W&B reachability check. Run it as a **trial run**: no `-g`, free, at most 3 minutes and 2 resource units, priority −5 (§5.2.3). A trial run exists only to check that a program works and is never used for results.
```
qsub -l gpu_1=1 -l h_rt=0:03:00 scripts/tsubame/env_check.sh
```

## 5. Grid Engine: the commands we use
| Purpose | Command |
|---|---|
| Submit (the group goes on the command line, never in the script) | `qsub -g $(cat ~/.t4_group) script.sh args` |
| Script header | `#!/bin/bash` first line; `#$ -cwd`, `#$ -l node_f=1`, `#$ -l h_rt=24:00:00`, `#$ -j y`, `#$ -o <log dir>` |
| Array job | `#$ -t 1-K` (task id in `$SGE_TASK_ID`; job id in `$JOB_ID`) |
| Run after another job ends | `qsub -hold_jid <job id> ...` |
| Status (**not more often than every few minutes**; frequent `qstat` is prohibited) | `qstat`, `qstat -j <id>` |
| Cancel | `qdel <id>`, one array task `qdel <id> -t <task>` |
| Points | `t4-user-info group point` |
Rules from the handbook: run `module purge` before `module load` in a script (§5.2.1.2). `-V` does not pass `LD_LIBRARY_PATH`, so the script activates conda itself. No `-m` mail on many jobs (the mail server may block the university). Priority stays at the default −5 (−4/−3 cost more).
Writing `#$ -t` and `$SGE_TASK_ID` instead of `#SBATCH --array` and `$SLURM_ARRAY_TASK_ID` is the only change from the lab scripts. The lab's `--mem` has no equivalent: memory comes with the resource type.

## 6. Our run flow (phase 1: baselines)
1. **Cell list** (T3): `python scripts/tsubame/make_cell_list.py --phase baselines-1 --out $WORK/h2snn-out/$SHA/cells_tuning.txt` (in a `cpu_8` job, or on the login node if it takes under a minute and no computation). One tuning line per runnable cell, in card 2 D5's form (`run_baseline_tuning.py --task <t> --models <m> --datasets <d> --num_runs 1 --base_seed 0 --n_trials 100 --output_dir <out>/experiments/baselines --state_dir <out>/state --wandb_project h2snn_baseline_t4`). The W&B project is new so that no lab dry-run trial can enter a TSUBAME sweep.
2. **Submit legs** (T1):
   ```
   bash scripts/tsubame/submit_chain.sh --list cells_tuning.txt --type node_f --legs 6
   ```
   This submits an array of `ceil(lines / 4)` node jobs, and then five more arrays, each held (`-hold_jid`) until the previous one ends. Each GPU slot owns fixed lines (slot `s` runs lines `s, s+S, s+2S, …` in order, with `--resume`). A finished line exits at once. A leg that hits 24 h loses at most the trial it was running, and the next leg resumes it. No line ever runs in two places, because lines are assigned to slots statically.
3. **Watch** (T4): `python scripts/tsubame/status.py --out $WORK/h2snn-out/$SHA` reads the sweep state files (finished trials per line, crashed attempts, the 24 h guard). It does not call `qstat`.
4. **Final seeds**: when a line has 100 finished trials, `make_cell_list.py --final-seeds` writes `--final_seeds 0 1 2 3 4` lines (spec §6; the launcher's `run_final_seeds`). Submit them the same way.
5. **Collect** (T4): `python scripts/tsubame/collect.py --out $WORK/h2snn-out/$SHA` writes one CSV (cell, trial or seed, parameters, validation monitor, test metrics, v3b SHA, host, job id, GPU name, wall time), one tar per cell and `SHA256SUMS`. Copy them to the lab cluster:
   `rsync -av <t4>:<WORK>/h2snn-out/$SHA/collected/ /mnt/data/peeradon/h2snn-results/tsubame/$SHA/`
6. **24 h guard** (T2): a trial that is killed by the 24 h limit **twice** stops its line with status `exceeds_24h` and is never retried in a loop. Phase-1 trials are well under an hour (card 2 review: HGB NC/LP ≤ 70 GPU-h per cell at 100 trials), so the guard is only a safety net there. It matters for H2GB NC and H2S-NN (phase 2), which need either epoch-level resume or a node reservation (≤ 96 h, Oct–Mar). That is a separate decision.

## 7. Troubleshooting
| Symptom | Likely cause | Action |
|---|---|---|
| `qsub` error about points or group | missing `-g` or wrong group | `qsub -g $(cat ~/.t4_group) …`; `t4-user-info group point` |
| job state `qw` for hours | `node_f` busy | resubmit the unstarted array tasks as `gpu_1` (`--type gpu_1`); never both for the same lines at once |
| job state `Eqw` | script error before start (path, permissions) | `qstat -j <id>` shows the reason; `qdel`, fix, resubmit |
| `CUDA error: no kernel image` | a wheel built without sm_90 | stop; report (env_check catches it before any sweep) |
| W&B timeout on compute nodes | no outbound HTTPS | stop; this blocks the Bayes sweeps (W&B runs the sweep controller); T6 decides |
| quota exceeded | outputs or checkpoints | `t4-user-info disk home`; tar and move old outputs to the lab cluster |
| line status `exceeds_24h` | one trial longer than a job | report to the design session; no retry |

## 8. Open, checked by T6 on TSUBAME
UNVERIFIED until T6: outbound HTTPS from compute nodes (pip, GitHub, W&B); NVIDIA driver version (must support CUDA 12.4); torch 2.4.1+cu124, DGL 2.4.0 cu124 and the PyG wheels on H100 (sm_90); `node_f` queue wait vs `gpu_1`; whether `~/.netrc` reaches compute nodes (home is Lustre, shared).
