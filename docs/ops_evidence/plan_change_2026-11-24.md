# Plan change 2026-11-24 (author decisions, ops evidence, spec request)

Ops session, 2026-11-24. The author decided these after an ops cost review. **For the model session:** sections 1 and 4 are what the spec (and ablation pre-registration) must state. Section 2 is the evidence.

## 1. Author decisions (2026-11-24)

1. **Hardware.** Every method, including H2S-NN, runs on one TSUBAME `gpu_1`: one H100 SXM5 94 GB with 96 GB host RAM. This replaces "a quarter of a `node_f` node's host memory" (192 GB) in range rule 1. A cell is N/A (OOM) only if it OOMs at the smallest grid point (rule 1 otherwise unchanged).
2. **OOM, two layers.**
   - (1) Estimate memory from the code for each baseline × dataset. If one tensor or dense materialisation alone exceeds 94 GB at the smallest grid point, the cell is reported OOM, and the calculation goes in the appendix.
   - (2) Every remaining doubtful cell gets a 1-epoch probe at the smallest grid point on `gpu_1`.
3. **Trials.**
   - Baselines: **40** trials on every dataset.
   - H2S-NN: **50** on the slow datasets (more than about 2 h per trial: oag-cs, oag-eng, oag-chem, ogbn-mag, PDNS, H-Pokec, H-Pokec U–U) and **100** elsewhere.
   - No multi-worker sweep card.
   - Tuning can be extended later: `--resume --n_trials N` continues the same W&B sweep, because `n_trials` is not in the sweep fingerprint (`src/utils/wandb_campaign.py:582-655`).
4. **Seeds.**
   - **3 seeds (0–2)** for every method and every cell, tuned or published configuration.
   - Seeds 3–4 may be added later (`--final_seeds 3 4`) if time is left.
   - Published-configuration seeds 3–4 that were already running or finished on 2026-11-24 are kept.
   - Ablation is unchanged in scope (ACM, DBLP, IMDB, Freebase; seeds 1–5, on the lab). Its "main-table runs of seeds 1–4 reused" becomes seeds 1–2 reused, and seeds 3–5 are run.
5. **Datasets.**
   - Kept, NC: ACM, DBLP, IMDB, Freebase, ogbn-mag, oag-cs, oag-eng, IEEE-CIS-G, H-Pokec, PDNS.
   - Kept, LP: ACM, DBLP, IMDB, LastFM, PubMed, H-Pokec U–U (the large LP task).
   - Kept, GC: MUTAG, PROTEINS, ENZYMES.
   - **Deferred** (kept in the paper plan, run only if time is left, cut otherwise): **mag-year, oag-chem, Freebase B–B, ogbn-mag P–P, NCI1.**
   - PubMed and LastFM are kept; their heterophily is "n/a (no labels)" (spec 1.8 convention).
6. **Points.** Re-estimate after 2–3 days of measured baseline tuning trials. ABCI is held in reserve and not used now.

## 2. Evidence

### 2.1 Points used and available (TSUBAME)
- Used since 2026-11-20, from `qacct -o <user> -d 40` (wall time × rate; `gpu_1` 0.16/h, `node_q` 0.20/h):
  - finished jobs: 193 points (`gpu_1` 173, `node_q` 20). By day: 11-20 3, 11-21 20, 11-22 48, 11-23 109, 11-24 13.
  - running jobs: about 30 more.
  - Total about 220–230.
- `t4-user-info group point` at 15:16: balance 1,826; deposit 1,217. The deposit is held at submit (`gpu_1` 3.84 per 24 h task) and the unused part is returned at the end.
- The large balance drops seen earlier (for example 1,580 → 955 on 11-22) were deposits, not use.

### 2.2 Measured time per trial (TSUBAME, H2S-NN tuning; prov start/end)

| Dataset | h/trial | Done |
|---|---|---|
| oag-cs | 12.2 | 2 |
| oag-chem | 12.1 | 2 |
| H-Pokec U–U | 11.8 | 2 |
| ogbn-mag | 8.2 | 3 |
| oag-eng | 8.1 | 3 |
| H-Pokec | 4.7 | 5 |
| PDNS | 3.4 | 7 |
| IEEE-CIS-G | 1.2 | 20 |
| mag-year | 1.0 | 24 |
| DBLP A–P | 0.8 | 30 |
| Freebase | 0.47 | 51 |
| Freebase B–B | ~24 | 1 |
| ogbn-mag P–P | — | 0 trials in 23.6 h |

- Trials of one sweep run one after another. At 100 trials, oag-cs or oag-chem take about 51 days of wall time. At 50 trials they take about 26 days; the others take ≤ 17 days.
- Baselines: no tuning trial had finished on TSUBAME. Published-configuration runs, h per seed:
  - oag-cs 4.4–6.0
  - oag-eng 3.6–5.0
  - oag-chem 4.2–5.7
  - PDNS 4.7
  - IEEE-CIS-G 1.9–4.4
  - **mag-year GCN/GAT/HAN: all 15 seeds killed at the 12 h `h_rt`** (job 9530249 tasks 9–23, `failed 44`, exit 137)
- `h2gb_published.py` keeps no mid-training checkpoint (`"train.ckpt_period": "logging"` is ignored, and 0 checkpoint files exist), so killed seeds restart from scratch.

### 2.3 Cost estimates (GPU-h; TSUBAME points at `gpu_1` 0.16/h)

| Plan | GPU-h | Points |
|---|---|---|
| Original (100 trials everywhere, 5 seeds, all datasets) | ~57,000 | 10,000–12,000 |
| Baselines 40 + H2S-NN 50/100, 3 seeds, before deferring mag-year / oag-chem | ~15,600 | ~2,500 |
| Adopted (oag-chem and mag-year also deferred) | lower; re-estimate after measured baseline trials | — |

Baseline per-trial times above are taken from published-configuration runs (full 500 epochs). Tuning trials stop early (patience 40), so the real figure is likely lower (±30–50 %).

### 2.4 Memory (H2S-NN on 96 GB host / 94 GB GPU)
- `qacct` maxrss of the H2S-NN tuning legs, about 24 h each:
  - `node_q` (192 GB host): Freebase NC 16.2 GB, H-Pokec 5.4 GB, H-Pokec U–U 8.2 GB, ogbn-mag P–P 4.1 GB
  - `gpu_1`: 4–14 GB, with no host-memory kill
  - maxvmem (53–147 GB) includes CUDA virtual mappings and is not a limit: PDNS reached 114 GB on `gpu_1` and was not killed.
- A lab dry run of H-Pokec NC was killed at a 30 GB cgroup limit, so maxrss may under-count page cache. The first `gpu_1` legs (submitted 2026-11-24) are the check.
- H2S-NN GPU OOM on the H100 at large configurations (trial-level points, not cell N/A):
  - H-Pokec NC: 2 trials (asked for 18.7 GB more at 77 GB in use)
  - H-Pokec U–U: 1 trial (asked for 33.4 GB more)
  - Other trials of these cells finished.
- Baselines: no GPU OOM on TSUBAME yet; their large-dataset tuning had not started. The 24 GB lab dry runs (`docs/dryrun_evidence/cells.csv` at `0b9873d`) show these cells at risk:
  - NSD: ACM, Freebase, ACM P–A, Freebase B–B, PubMed; OOM even at d = 4
  - H2G-former: most datasets
  - HINormer: ACM, DBLP, IMDB
  - GAT, CSNN, HAN, PSHGCN: Freebase B–B
  - H2G-former and Hetero²Net: Freebase NC (host kill at 24 GB)
  - Layer 1 of the OOM screen (decision 2) is being computed from the code.

### 2.5 Why these datasets (author, with ops numbers)

**OAG.** oag-cs, oag-eng and oag-chem share one source (Open Academic Graph), one schema (paper, author, field, affiliation), one task (journal venue of a paper) and one feature type. They differ in structure:

| | oag-eng | oag-cs | oag-chem |
|---|---|---|---|
| Papers / authors | 371k / 505k | 547k / 511k | 747k / 1.10M |
| Edges (no reverse) | 7.5M | 19.9M | 26.7M |
| Citations per paper | 5.7 | 21.5 | 19.3 |
| Labelled papers | 79 % | 42 % | ~100 % |
| ℋ² / MLH | 0.873 / 0.912 | 0.965 / 0.980 | 0.886 / 0.921 |

- oag-chem is the closest to oag-cs and the most expensive, so it was deferred first.

**H-Pokec and PDNS are kept.**
- With IEEE-CIS-G, they are the only large non-citation graphs (social, DNS security, fraud).
- Their ℋ² is high (0.949, 0.787) although MLH is moderate (0.385, 0.485). IEEE-CIS-G has the lowest MLH (0.135).
- Source: `h2snn-card19/docs/heterophily_evidence/table.md` (card 6 records).

**H-Pokec U–U over ogbn-mag P–P.**
- ogbn-mag P–P has higher ℋ²/MLH, but H2S-NN finished 0 trials in 23.6 h.
- With the 24 h job limit and no epoch checkpoint, a trial cannot finish.
- H-Pokec U–U takes 11.8 h per trial.

### 2.6 ABCI (reserve; facts from docs.abci.ai v3 and the FY2026 tariff page)
- `rt_HG`: 1 × H200 141 GB, 160 GB host, 16 logical cores; 3 points/h standard (1.5 with Development Acceleration Use). 1 point = 220 yen.
- Spot jobs can run up to 168 h; On-demand jobs up to 12 h.
- Scheduler: PBS Pro. Deduction works like TSUBAME: the maximum is held at start and the rest is refunded at the end.
- 1,000 points ≈ 333 GPU-h.
- A different GPU from the spec. Any cell run there must run wholly there, with the GPU capped at 94 GB.

## 3. Ops actions taken (2026-11-24)
- **TSUBAME `t4_auto.py`** (ops, outside git; backups `.bak_pause`, `.bak_magyear`, `.bak_plan1124`):
  - `PAUSED` = {freebase_bb, ogbn-mag_pp, nci1, mag-year, oag-chem}
  - every list runs on `gpu_1` (the `node_q` lists included)
  - trial targets 40 / 50 / 100, applied when a leg is submitted (the lists keep `--n_trials 100`; line identity ignores the trial count)
  - final seeds 0–2
- **Deleted, queued only** (deposits refunded; no running task deleted):
  - b2 node_q 9537282 (11:45); 9523129 tasks 3–4 and 9540881 task 8 (running H2S-NN ogbn-mag P–P / H-Pokec U–U / Freebase B–B legs, 11:45, first pause)
  - 9533068 tasks 1–14
  - 9531542, 9531543, 9537281, 9540094, 9541542
  - 9540095 task 4; 9540881 tasks 1–7 and 9
  - published seeds 3–4: 9533064 tasks 14, 15, 19, 20
  - The tuning legs are resubmitted by `t4_auto` at the new targets on `gpu_1`.
- **Kept running:** H2G-former published seeds 0–4 (9537254 / 9537255, including mag-year and oag-chem, which were already running), 9530256 tasks 27–28, and 9533064 task 12.
- **`hpokec_w2.sh`** now submits H-Pokec H2G-former seeds 1–2 only (list `cells_pub_h2gf_c20_w2_hpokec_s12.txt`).
- **Lab:**
  - Pending H2S-NN / b1 jobs were cancelled to free the cluster for other users: 18444, 18445, 18458, 18459, 18560, 18561, 18562, 18563 (pending tasks only).
  - `lab_final_watch.sh` was stopped. Running tasks continue.
  - Card 19 and other projects' jobs were untouched.
  - Lab resubmission at the new targets waits for the author.

## 4. Spec items for the model session
1. Range rule 1: host memory 96 GB (`gpu_1`) instead of a quarter of `node_f`.
2. OOM layer 1 (calculated, appendix) and layer 2 (1-epoch probe at the smallest grid point).
3. Tuning budget:
   - baselines 40 trials
   - H2S-NN 50 trials on the named slow datasets, 100 elsewhere
   - extension rule (`--resume`)
4. Seeds 0–2 for the main tables and published configurations, with mean ± sample std over 3 seeds. The Welch test on 3 vs 3 seeds is weak; state this.
   - Ablation pre-registration: seeds 1–5 unchanged; reuse of main-table seeds is now 1–2.
5. Dataset list with a "deferred" status: mag-year, oag-chem, Freebase B–B, ogbn-mag P–P, NCI1.
6. Timing runs (C5) stay on one GPU type (H100, TSUBAME).

## 5. OOM decisions (author, 2026-11-24, after OOM layer 1)
- **H2G-former × Freebase NC: N/A (OOM).** Source: layer 1, `docs/ops_evidence/oom_layer1_2026-11-24/oom_layer1.md`.
  - The adapter moves a dense identity feature matrix for each of the 8 node types to the GPU (34.46 GiB) and keeps a second copy (`pristine_x`).
  - At the first forward, 94.18 GiB is live, above the 93.58 GiB device.
  - No grid axis touches these tensors.
  - Not probed (author: report OOM). The tuning leg and its probe were deleted (9545679 task 2, 9545380), and `t4_auto` skips the line (`NA_TOKENS`).
- **MAGNN published configuration × Freebase NC and × LastFM LP: N/A (OOM, host memory).**
  - Job 9533063 on `node_q` (192 GB host) was killed after 113 s and 341 s, with `qacct` maxrss 178.7 and 182.3 GB (exit 137).
  - This is above the 96 GB host of the `gpu_1` rule. Not rerun.
- Remaining layer-2 probes (job 9545379, 1 epoch at the smallest grid point):
  - Hetero²Net × Freebase NC
  - PSHGCN × H-Pokec NC
  - PSHGCN × H-Pokec U–U
  - H2SGNN × H-Pokec U–U
  - Hetero²Net × H-Pokec U–U
- Layer-1 findings for the model session:
  - The PSHGCN H-Pokec override (K = 1, hidden ≤ 64, spec 1.23) is never applied (`pshgcn.py:181-188` returns before `:202`).
  - Time risk: HINormer builds H-Pokec sequences in a Python loop, and ChebNetII recomputes whole-graph bases per batch.

## 6. Loader speed, H-Pokec crash, seeds 3–4 (author, 2026-11-24 evening)
- Measurement (job 9550269, gpu_1, 16 cores): HGT sampling at batch 32 is 95 → 19 ms/batch (PDNS) and 73 → 15 ms/batch (oag-eng) from 0 to 6 workers. A real PDNS GCN published run spends about 24 s of its 28 s epoch sampling, with 0.9 GB peak GPU memory.
- Published GCN, HAN and R-GCN on H-Pokec NC: all 11 seed lines of job 9533064 crashed with `cannot sample n_sample <= 0 samples`. The yaml gives `user` 4 hops and the other types 3, so the sampler reads an undefined size at hop 4. Not rerun.
- Card 21 (`docs/H2S-NN_card21_card.md`, b72f070): loader workers 6 (author: 6, not more) + pad shorter hop lists. Held until it merges: 9545676 (43 tasks) and 9545677.19–27.
- Seeds: keep 3 (0–2) for every cell first. After card 21 merges and points are re-estimated on measured speed, run seeds 3–4 as a second wave if points allow (estimate +80–100 points, ~500–600 GPU-h).

## 7. Card 21 merged; jobs moved to the new code (author, 2026-11-25)

Author decision 2026-11-25: conditional approval of the v3b move. Pending baseline lines move to the new v3b. Void H-Pokec NC lines are stopped and requeued. No in-place checkout update.

- **Review:** diff `f1fc158..5bb3a81 -- H2S-NN/src` shows no problem. `2e3e160` adds docs only on top of `5bb3a81`.
- **Full suite:** lab job 18861 on `5bb3a81` ran 1009 tests: 5 failures and 2 errors. All 7 came from how ops ran the suite, not from card 21:
  - The sbatch did not set the pinned AVX2 kernel settings of `scripts/slurm/run_tests.sbatch`. This failed 4 bit-exact tests: golden, card 5 T4, and 2 mb0 compare.
  - `WANDB_MODE=disabled` failed 3 W&B tests.
  - Rerun of those targets with the `run_tests.sbatch` environment: job 18862, 11 tests OK.
  - No test compares batches against a 0-worker loader.
- **v3b:** fast-forwarded `f1fc158 → 2e3e160`.
- **TSUBAME checkout:** new directory `h2snn-runs/2e3e160…`, with `vendor` copied (diff clean), `data` symlinked and the MetaPath2Vec cache copied. Running jobs keep their checkouts.
- **HINormer cache:** an ops `qsub` wrapper adds `-v H2SNN_HINORMER_SEQ_CACHE=<out>/hinormer_seq_cache` to every job submitted by t4_auto. The setting is confirmed in the job `env_list`.
- **Tuning legs:** t4_auto BASE now runs all baseline lists (early, b1, b2 gpu_1, b2 node_q, H2G-former) on `2e3e160`. H2S-NN lists are unchanged.
  - Not-started legs 9545672, 9545674, 9545676, 9545677 and 9545679 were deleted.
  - Resubmitted on `2e3e160`: 9553074 (early, 4), 9553078 (b1, 20), 9553079 (b2 gpu_1, 43), 9553082 (b2 node_q, 27), 9553085 (H2G-former, 1).
- **Void H-Pokec NC published lines:** these ran with the unequal hop dict.
  - Affected: GCN, HAN and R-GCN (job 9533064) and H2G-former (9537254 and hpokec_w2). None had a finished seed or saved state.
  - Rerun of seeds 0–2 on `2e3e160`:
    - 9553168: `cells_pub_hpokec_c21`, 9 lines.
    - 9553170, then 9553172 held as the second leg: `cells_pub_h2gf_hpokec_c21`, 3 lines.
- **Measurement still to do:** host maxrss and s/epoch with 6 workers on the first H-Pokec NC and ogbn-mag NC jobs. Then re-estimate points, before seeds 3–4.

### Lab, 2026-11-25 (author: stop H2S-NN MUTAG and LastFM, run light baselines instead)

- **Stopped:** H2S-NN MUTAG GC (18458, 6 of 100 trials) and LastFM LP (18444, 9 of 100 trials).
  - Both ran at about 1 CPU core with the GPU at 0–22%.
  - MUTAG took 150–300 s/epoch; LastFM took 24–100 s/epoch.
  - Their tuning state is kept.
- **Submitted:** 26 not-started baseline b1 lab lines, run on v3b 2e3e160 with 40 trials, 28G memory, 72 h per leg, 2 legs.
  - Lists `cells_tuning_b1_light_{2080ti,a6000,3090,titan_rtx}.lab.txt`, jobs:
    - 18863/18864 (2080 Ti, 8 lines)
    - 18865/18866 (A6000, 4 lines)
    - 18867/18868 (3090, 4 lines)
    - 18872/18873 (TITAN RTX, 10 lines)
  - 12 cells with no trial yet moved from 3090 to 2080 Ti or A6000, chosen by their fit-check peak memory. Each cell still runs on one GPU type. Backup: `lab_gpu_types.tsv.bak_c21light`.
- **Not touched:** the Hetero²Net acm_pp and dblp_ap lab sweeps (18560_5, 18562_7). They are at 69 and 86 trials, past the 40-trial target; stopping them awaits the author.

## 8. Card 22 merged; failed H2GB cells reset (author, 2026-11-25)

The author approved the card 22 merge ("อนุมัติ merge card 22 ได้เลย") and, earlier, resetting the failed sweep states after the merge.

- **Card 22:** head `b0f13fb` (code `1b03c39`), report `docs/H2S-NN_card22_report.md` on v3b.
  - The source diff touches only `hetero2net.py` and `minibatch.py`, which the card allows.
- **Full suite:** lab job 18910 on `b0f13fb`, with the `run_tests.sbatch` environment and without `WANDB_MODE`: 1011 tests OK.
- **v3b:** fast-forwarded `2e3e160 → b0f13fb`.
- **Checkouts:** both are new directories for `b0f13fb`. Running jobs keep their checkouts.
  - TSUBAME: `vendor` copied (diff clean), `data` symlinked, `experiments/` (MetaPath2Vec cache) copied.
  - Lab: a worktree, with `vendor` copied.
- **t4_auto:** the baseline lists now run on `b0f13fb`; the qsub wrapper (HINormer cache) is unchanged.
  - Baseline tasks still queued on `2e3e160` (9553079, 9553082) stay there. Card 22 changes nothing for them:
    - The Hetero²Net path changed only for H2GB NC, and all those tasks were moved off that checkout.
    - `hgb_homogeneous_batch` now gives `None` only where a type has no `x`, which happens only on IEEE-CIS-G.
- **Reset:** these four cells had 0 successful and 40 failed trials each. Their `state/<cell>` and `experiments/baselines/<cell>` directories were renamed to `<cell>.void_card22_2e3e160`. Nothing was deleted.
  - Hetero²Net oag-cs NC and IEEE-CIS-G NC.
  - CSNN and NSD IEEE-CIS-G NC.
- **Held Hetero²Net H2GB NC tasks** (all through `_fit_h2gb`, which would fail the same way) were deleted from the old checkout:
  - 9553079.22 (oag-eng), 9553079.38 (pdns), 9553082.14 (freebase), 9553082.23 (h-pokec).
  - Hetero²Net h-pokec_uu LP (9553082.6) uses the unchanged large-LP path and stays.
- **Resubmitted on `b0f13fb`:** the 4 reset cells and the 4 held cells, list `cells_tuning_b2_c22_reset.txt` (8 lines, 40 trials), job 9564510, priority −100 like other tuning legs.
- **SeHGNN IEEE-CIS-G NC:** no reset. The num_hops 4 trials are host OOM with both 6 and 0 workers (card 22 job 18887).
- **Finding for the model session (not changed by ops):** CSNN oag-cs had 1 failed trial (attempt 1, `orth: euler`, `d: 4`).
  - The vendor `orthogonal.py:74` asserts `2 <= d <= 3` for `euler`, in both the CSNN and the NSD vendor copies.
  - The search spaces still pair `euler` with larger d: CSNN `d ∈ {2,3,4}` (`csnn.py:202-205`) and NSD `d ∈ {2,…,5}` (`nsd.py:253-256`).
  - Every such grid point fails as a trial on every dataset, so it uses up a slot in the trial budget.
  - The fix is a search-space decision, owned by the spec.

## 9. Five seeds everywhere (author, advisor, 2026-11-25)

The author set the main-table seeds back to 5 (seeds 0–4), at the advisor's request ("ตั้ง N_SEEDS = 5 แล้วเริ่มเลย", set N_SEEDS = 5 and start now). This replaces "seeds 0–2" of §1.
The spec (1.32 AF) and ablation pre-registration v3 are owned by the model session. v3 reuses main-table seeds 1–2 and runs seeds 3–5, which now overlap main-table seeds 3–4.

- **Points estimate** from measured per-seed wall time on gpu_1 (0.2 points/h; balance 1,781 before submission): about 40–70 points for seeds 3–4 of every cell.
  - Tuned baselines with budget 500/40: 0.01–0.3 h per seed.
  - Published configs: 2–6 h per seed on H2GB; 32 cells already had 5 published seeds.
  - H2S-NN: 0.5–1 h on Freebase, IEEE-CIS-G and DBLP; 3–8 h on OAG, PDNS and H-Pokec; 6–11 h on ogbn-mag.
- **t4_auto:** `N_SEEDS = 5` (backup `.bak_seeds5`). Finished sweeps get per-seed lines for seeds 3–4 on the cell's own out-dir lists, with no retuning. First submission:
  - 9565207 (early, 8 lines), 9565225 (b1, 40 lines), 9565226 (b2 gpu_1, 14 lines).
- **H-Pokec NC published** (the void rerun of §7): seeds 3–4 were added on the same `2e3e160` checkout as seeds 0–2.
  - 9565183: `cells_pub_hpokec_c21_s34`, GCN/HAN/R-GCN, 6 lines.
  - 9565184: `cells_pub_h2gf_hpokec_c21_s34`, H2G-former, 2 lines.
- **Dashboard:** `dash_status.py` and `lab_status.py` count 5 seeds (data v30). A lab cell with 3 of 5 seeds shows as Stopped until the author gives the go on the lab.
- **Not run yet:**
  - Lab seeds 3–4. These need a lab resubmission, which waits for the author's go.
  - H2G-former ogbn-mag published: only seeds 1 and 4 exist. Seeds 0, 2 and 3 were killed at the 12 h job limit (6 lines, job 9537255). This is reported and not rerun.
- **Lab, 2026-11-25 17:48** (author: "ส่ง lab ได้เลย รวม 12 cell ด้วย", send the lab jobs now, the 12 cells included; big models later): watcher `lab_seeds5_watch.sh` (ops).
  - **Tuning:** the 10 cancelled light lines resumed on `b0f13fb`, 40 trials. Jobs 18937/18938 (2080 Ti, 5 lines), 18939/18940 (3090, 1 line), 18941/18942 (TITAN RTX, 4 lines).
  - **Seeds 0–4:** submitted for every lab baseline cell whose 40-trial sweep finished: the 12 cells, plus HAN DBLP, H2G-former PubMed, and Hetero²Net ACM and DBLP.
    - Jobs 18943–18959, on the same GPU type as each cell's tuning, from the checkout that ran its tuning: `2e3e160`, or `0b9873d` for Hetero²Net.
    - Hetero²Net ACM and DBLP ran 100 and 82+ trials before the 40-trial target was set. Their best configuration is selected over all recorded trials.
  - **Later cells:** the remaining sweeps (the 10 above, and HAN LastFM at 34/40) get seeds 0–4 from the watcher when they finish.
  - **Lab cells that already had final seeds:** all have 5 seeds (0–4).
  - **Later (author):** H2S-NN on the lab (LastFM and PubMed LP; MUTAG, ENZYMES and PROTEINS GC), CSNN LastFM, and H2G-former ogbn-mag published seeds 0, 2 and 3.

## 10. Card 23 merged; H2S-NN HGB NC re-tune on the lab (author, 2026-11-26)

Author decision 2026-11-26: conditional approval of the v3b move. If the full suite passes, fast-forward v3b without asking again; then re-tune ACM NC, then DBLP and IMDB NC, with the card 23 regularisation pack.

- **Review:** the `b0f13fb..528aa9b` source diff matches the card.
  - Every pack option is guarded and skipped when off (input dropout, EMA, K-draw averaged evaluation, smoothed checkpoint selection).
  - Off, the code path is that of `b0f13fb`.
  - No source problem found.
- **Full suite:** lab job 19008 on `528aa9b`, with the `run_tests.sbatch` environment and without `WANDB_MODE`: 1029 tests OK.
- **v3b:** fast-forwarded `b0f13fb → 528aa9b`. New lab worktree `h2snn-runs/528aa9b…`, with vendor copied and data linked.
- **Re-tune** (ops `lab_c23_watch.sh`): `<ds>_h2snn_c23.yaml`, `--exp_name <ds>_h2snn_c23`, 100 trials, 72 h × 2 legs, on the GPU type of each original lab cell. When a sweep finishes, the best configuration runs on seeds 0–2 (card text).
  - ACM: 3090, job 19009/19010, W&B sweep `8x9xtrkg`.
  - DBLP: TITAN RTX, job 19011/19012, sweep `ax0y6bj3`.
  - IMDB: 3090, job 19013/19014, sweep `a8pvz3p4`.
  - ACM was submitted first. DBLP and IMDB run in parallel, since the card fixed that they run whatever ACM shows.
- H2S-NN sweeps running on older SHAs are unaffected (their fingerprints are unchanged).

## 11. Card 23 seeds 3–4 on TSUBAME (author, 2026-11-27)

Card 23 asked for the best configuration on seeds 0–2. That ran on the lab and did not match the main table's 5 seeds (§9). The author asked for seeds 3–4 of every card 23 cell, on TSUBAME H100 ("สั่งเลย รัน seed 3–4 ทั้งหมด แบบบน TSUBAME", go, run seeds 3–4 for all of them, on TSUBAME).

- **Deviation from rule 1 (one cell = one GPU type), chosen by the author** when asked: seeds 0–2 ran on the lab (ACM and IMDB on RTX 3090, DBLP on TITAN RTX), seeds 3–4 run on H100.
- **Code:** a new TSUBAME checkout of v3b 528aa9b, the same code as the lab seeds. Only `best_tuned_run0.json` was copied from the lab out dir; the md5 sums match. No retuning.
- **ACM and IMDB:** job 9585871, 4 tasks, one seed per task, gpu_1, 24 h.
- **DBLP:** its sweep had 91/100 trials. `c23_dblp_s34_t4.sh` waits for the lab seeds 0–2 to start, then submits seeds 3–4 the same way.
- **Seeds 0–2 results** (test ×100, mean ± std, for reference only):
  - ACM: Macro-F1 93.32 ± 0.51. The old configuration on the same seeds scored 89.07 ± 0.69.
  - IMDB: Macro-F1 65.59 ± 0.51, against 63.59 ± 0.67.

## 12. Lab seed jobs moved to TSUBAME (author, 2026-11-27)

The author asked to move lab work to TSUBAME step by step ("ทยอยเอางานลง TSUBAME", move the work onto TSUBAME bit by bit). The first step: seven seed jobs for LP cells tuned on the lab's RTX 2080 Ti had waited more than a day in that queue.

- **Cancelled on the lab** (pending, never started): 18943, 18944, 18945, 18963, 18967, 19022, 19024.
- **Cells:** GCN, H2SGNN, R-GCN, CSNN and Hetero²Net on IMDB M–A LP; H2SGNN on LastFM LP; PSHGCN on DBLP A–P LP.
- **Submitted on TSUBAME H100**, one seed per task (seeds 0–4), with no retuning. Each cell's `best_tuned_run0.json` was copied from the lab. Out dir: `h2snn-out/lab_moved`.
  - Job 9587883: checkout 2e3e160 (GCN, H2SGNN).
  - Job 9587886: checkout b0f13fb (the rest).
  - Each cell runs on the checkout its tuning ran on.
- **Mixed hardware, as in §11:** tuning ran on the 2080 Ti, the seeds run on H100.
- **Collecting results:** `pull_t4_seeds.sh` (ops) copies finished `final_seed<N>.json` files back into the lab cell directories and never overwrites existing files. Card 23 seeds 3–4 come back the same way.
- **Card 23 DBLP:** the sweep finished. Seeds 0–2 run on the lab (job 19026); seeds 3–4 run on TSUBAME (job 9586898).
- **Card 23 results, 5 seeds** (test Macro-F1 ×100, mean ± std): ACM 93.30 ± 0.43 (old 88.78 ± 0.64); IMDB 65.78 ± 0.65 (old 64.04 ± 1.07).

## 13. Fixes and reruns (author, 2026-11-28)

**The §12 seed jobs failed and were resubmitted.**
- Jobs 9587883 and 9587886 failed on every task in under a minute. Cause: ops copied only `best_tuned_run0.json`, and `run_final_seeds` also reads `tuning_trials_run0.json`.
- Ops copied `tuning_trials_run0.json` and `tuning_summary.json` from the lab for all 7 cells and resubmitted the same lists: jobs 9602498 (2e3e160) and 9602499 (b0f13fb). The failed logs are kept in `lab_moved/logs/failed_20261127`.

**Card 23 is now the dashboard's main value** for H2S-NN on ACM, DBLP and IMDB NC (author). Five seeds, test Macro-F1 ×100: ACM 93.30 ± 0.43, DBLP 94.47 ± 0.27, IMDB 65.78 ± 0.65. The earlier v3b 0b9873d result stays in each row's note. The paper and the register are unchanged; the model session owns them.

**Seven H-Pokec cells rerun on TSUBAME** (author: "ส่งทั้ง 7 cell ขึ้น TSUBAME เลย", send all 7 cells to TSUBAME now).
- **Cells:** GAT, NSD and CSNN on H-Pokec NC; GCN, GAT, NSD and CSNN on H-Pokec U–U LP.
- **What failed:** all 40 tuning trials of each cell, on checkout 2e3e160, with `AttributeError: 'NodeStorage' object has no attribute 'x'` in `hgb_homogeneous_batch` (`src/baselines/minibatch.py`). H-Pokec has a node type without features.
- **The fix already existed:** card 22 B (featureless types), in b0f13fb. The baseline code in b0f13fb and 528aa9b is identical.
- **Rerun setup**
  - List: `cells_tuning_b2_hpokec_r1128.txt`, with the original lines and a fresh sweep state (`state_r1128`).
  - Checkout b0f13fb, gpu_1. First leg: job 9602950.
  - The failed cell directories moved to `experiments/baselines/_failed_2e3e160`; nothing was deleted.
- **t4_auto changes**
  - It tracks the new list: next legs, then 5 seeds.
  - It ignores these cells' old lines.
  - It does not count their pre-rerun final-seed tries.
- **Mixed code within H-Pokec:** these cells run on b0f13fb. The other H-Pokec baselines started on 2e3e160 and their later legs ran on b0f13fb. The only code difference that matters here is the featureless-type handling.

**Dashboard fixes in `dash_status.py`** (display only; no result changed)
- A Problem stays a Problem. Before, a failed cell whose chain had ended showed "Not started, waiting for the author's go".
- A line whose output directory holds the cell's seeds is preferred over a probe line. Before, PSHGCN H-Pokec and Hetero²Net Freebase showed Not started although their 5 seeds were done.
- Finished TSUBAME cells now fill the Result column from their seed table. 65 cells had shown it empty.

**H2SGNN Freebase NC:** all 40 trials failed on 2e3e160 with `CUBLAS_STATUS_EXECUTION_FAILED` in `loss.backward()`. The H2SGNN adapter is unchanged up to 528aa9b, so this is not a fixed bug. A lab diagnostic (job 19069, one dry-run trial at the min and max grid points, `CUDA_LAUNCH_BLOCKING=1`, no W&B) looks for the cause; it is not a result.
