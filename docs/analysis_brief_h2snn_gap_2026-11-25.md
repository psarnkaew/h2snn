# Analysis brief: why H2S-NN trails other models on some cells (2026-11-25)

The ops session wrote this brief on 2026-11-25. The author asked for a new session ("วิเคราะห์ว่าทำไม H2S-NN ถึงได้ performance น้อยกว่าโมเดลอื่น", analyse why H2S-NN performs worse than other models).
The reader is the new analysis session.

## 1. Goal and limits

**Goal.** Explain, with evidence, where and why H2S-NN scores below the baselines.
For each cell where it trails, say:
- whether the gap is real or within seed noise;
- what in the model, training, tuning or data handling causes it.

The deliverable is a short report, `docs/H2S-NN_gap_analysis_report.md` on the `design-docs` branch, giving:
- for each finding: the evidence (file, number, run id), a confidence level, and what would confirm it;
- a ranked list of possible changes, each with its cost.

**Limits.** These follow from the pre-registration and the freeze (spec 1.31 AE8, 1.32):
- **Analyse only.** Do not change the spec, the pre-registration, `published_number_register.md`, the H2S-NN code, the configs or any result directory.
  - A change to H2S-NN after seeing test results is a deviation. The author decides it, together with the model session (the design-docs session that owns the spec). It must be reported openly.
  - Propose changes; do not apply them.
- **Use validation metrics for diagnosis wherever you can.** Test numbers say where the gap is, not what to tune.
- **TSUBAME:** read files only, through `ssh t4`.
  - On the login node: no qsub, no job control, nothing that runs over 10 minutes.
  - Never print or store the login, group name, keys or the W&B key.
- **Lab cluster:** a diagnostic run (for example, one training with extra logging) needs the author's OK first.
  - Every sbatch sets `--mem` and `--time`, and runs from a checkout under `/mnt/data/peeradon/h2snn-runs/<sha>/`.
  - Never update a checkout in place.
  - Do not touch other sessions' jobs.
- **Git:** stage files by explicit path; never `git add -A`, rebase, reset or bare stash. Commit and push the report as you write it.
- **Public repo and shared pages:** replace absolute `/work/…`, `/gs/…` and `/home/…` paths and user ids in anything you write.

## 2. Current numbers (dashboard data v30, 2026-11-25)

Test metrics are mean ± std over 5 seeds. "Ours" means our pipeline; "published" means numbers copied from papers (`docs/published_number_register.md`, HGB: 5 seeds with HGB hyperparameters).

| Cell | H2S-NN | Others | Reading |
|---|---|---|---|
| ACM NC, Macro-F1 | **88.78 ± 0.64** | published: GCN 92.17, GAT 92.26, R-GCN 91.55, HAN 90.89, NSD 92.07, HetSheaf 94.30 | clear gap of 2–5 points |
| DBLP NC, Macro-F1 | 93.09 ± 0.67 | published: GCN 90.84, GAT 93.83, R-GCN 91.52, HAN 91.67, MAGNN 93.28, HetSheaf 94.93 | middle of the pack |
| IMDB NC, Macro-F1 | **64.04 ± 1.07** | ours: PSHGCN 63.57, H2SGNN 63.57, CSNN 62.51, ChebNetII 59.86; published HetSheaf 64.51 | at the top |
| ACM P–P LP, AUROC (random negatives) | 0.935 ± **0.049** | ours: CSNN 0.953 ± 0.003, HAN 0.953, H2SGNN 0.944, GCN 0.934 | same mean as GCN; **std 15× the baselines'** → seed instability? |
| IMDB M–A LP, AUROC | 0.706 ± 0.024 | baselines still running on the lab | open |

Other notes:
- IMDB NC `test_acc` 0.316 is a multi-label exact-match accuracy. Check that it is computed as the other methods compute it before using it.
- Not comparable yet: every H2GB cell (oag-cs/eng, PDNS, IEEE-CIS-G, H-Pokec, ogbn-mag, Freebase). H2S-NN has only 3–69 of its 50–100 trials there.
  - Published-config baselines there are done. H2G-former leads on oag-cs, oag-eng and PDNS.
  - If the early H2S-NN trials there already look weak on validation, that is useful evidence. Say how many trials it is based on.

Refresh the numbers from the dashboard's data before you rely on them (run-status page `Q9svL8vfunrG1CAY9UgfZs`; ask the author for the link) or from the result files below.

## 3. Where the data is

**Lab out dir:** `/mnt/data/peeradon/h2snn-out/0b9873d0c19455c8df56fa152492188d8918b62d/`
- H2S-NN, per cell: `experiments/h2snn/<cell>/`
  - `final_seed{0-4}.json`, `final_seeds_summary.json`;
  - `best_tuned_run0.json` (the selected configuration);
  - `tuning_trials_run0.json` (every trial's parameters and validation monitor), `tuning_summary.json`, `tuning_failures_run0.json`.
- H2S-NN, per trial: run directories `experiments/h2snn/h2snn_<task>_<dataset>_tune_run0_<wandb-run-name>/`.
- Sweep state: `state/<cell>/tuning_state_run0.json`, which holds the W&B sweep id.
- Baselines: `experiments/baselines/<model>_<task>_<dataset>/`, same layout. Published configs write `published_seed*.json`.
- Logs: `logs/lines/*.log`. The first line has `head=<code sha>`.
- Cell names: `h2snn_node_classification_{acm,dblp,imdb}`, `h2snn_link_prediction_{acm_pp,imdb_ma}`.

**TSUBAME** (read-only): `$W/h2snn-out/<sha>/`, same layout, where `W=${HOME/home/work}`.
- `0b9873d…`: H2S-NN H2GB tuning and the b1/b2 baselines.
- `2d822a4…`: published-config baselines.
- `e52b025…`: the early baselines (ACM/DBLP).

**W&B projects:** `h2snn_h2snn_t4` (H2S-NN sweeps) and `h2snn_baseline_t4` (baselines). The key is already in `~/.netrc`; never copy it.

**Code:** `/mnt/data/peeradon/h2snn-runs/0b9873d0c19455c8df56fa152492188d8918b62d/H2S-NN` is the H2S-NN code that produced these runs.
- `src/models`, `src/trainers`, `src/evaluation`;
- `configs/experiments/<dataset>_h2snn_v3b.yaml`, which holds the search space;
- `scripts/run_h2snn_wandb_tuning.py`.
- Baselines are in `src/baselines/`. The current v3b is `b0f13fb`; its baseline code is unchanged for HGB datasets.

**Documents** (design-docs):
- `docs/H2S-NN_methodology_spec.md` (§6 tuning and seeds, budgets);
- `docs/ablation_analysis_preregistration_v3.md`;
- `docs/published_number_register.md`;
- `docs/ops_evidence/plan_change_2026-11-24.md` (trials 40/50/100, seeds, hardware, 5 seeds from 2026-11-25).

## 4. Questions worth checking first

These are suggestions; follow the evidence.

1. **ACM NC gap.** Is the validation score of the selected configuration also below the baselines, or is it a validation→test drop (overfitting the 6% validation split)?
   - Compare best-trial validation with test for H2S-NN and for our own ACM baselines (e52b025 out dir).
   - Does the search space contain configurations close to what the strong methods use: depth, hidden size, dropout, lr, epochs, early stopping?
2. **ACM LP variance.** Which seed is low? Is it a training failure (loss curve, divergence, a stalled early stop) or a bad split or negative sample? Is the selected trial itself unstable across seeds?
3. **Tuning health.** Look at the 100 trials per cell:
   - fraction failed;
   - spread of validation scores;
   - did the best trial come late (search not converged) or sit at the edge of a range (range too narrow)?
4. **Training budget.** Did runs stop on the epoch cap or on patience? Compare with what the baselines got (budget 500/40 for the tuned baselines).
5. **Model-specific causes.** Examples:
   - the sheaf maps (SO(d)) or the solver hitting tolerances;
   - degree or normalisation on ACM's dense paper–paper relation;
   - the meta-path set;
   - feature handling for featureless types.
   - Tie each claim to a number from the runs.
6. **Where H2S-NN wins** (IMDB NC, DBLP NC is close): what differs about those datasets? A heterophily or homophily measure, if you can compute it cheaply, can make the paper's story precise.

## 5. Handoff

- Report to the author in chat with a short summary and the report link.
- Proposed changes go to the author. The author takes them to the model session if needed.
- Ops (this session's counterpart) runs nothing for you without the author's go.
