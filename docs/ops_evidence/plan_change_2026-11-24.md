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
