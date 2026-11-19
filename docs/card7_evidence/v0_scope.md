# Card 7 V0: worktree, merge, scope list (S6, 2026-11-19)

- Worktree `/mnt/data/peeradon/h2snn-model`, clean before the switch (only the untracked `H2S-NN/data` symlink).
  `model-td1` left at `894fd74` (unchanged).
- Branch `card7` from `v3b` `894fd7421ce8581d38b967906bde5593e881518a`; `git merge --no-ff`
  `design-docs` `10db46398007a23bff54eee78aea8af198c81ce5` -> `235ce3f` (no conflict). `origin/v3b` is still
  `894fd74` (card 5 not on `v3b`): Phase A only.
- Base suite reference: 777 OK (job 17765, model-td1 M8, same `src/` as `894fd74`).

## Files and plan (Phase A; no file under `src/` is touched)
| File | Unit | Plan |
|---|---|---|
| `H2S-NN/scripts/ablation/arms.py` | V1 | `ARM_CHANGES` = exactly the 7 rows (`full`, `no_path_cells`, `S2`, `S0`, `endpoints`, `instance_features`, `literature`); the literature row is built by the existing `fixed_literature_schema_config` (only `data.path_source` changes); `TUNED_ARMS` = the 4 (b) arms; out-of-scope arms removed from every list (they stay in frozen `src/`) |
| `H2S-NN/scripts/ablation/sweeps.py` | V1 | (b) for exactly `endpoints`, `no_path_cells`, `instance_features`, `S2` (100 trials, seed 0, arm-tuning mode); `closure_only` removed |
| `H2S-NN/scripts/ablation/seed_runs.py` | V1 | manifest: 7 rows (a) seeds 1-10 (`full` 5-10, 1-4 looked up), 4 (b) tunes + seeds 1-10; `--freebase_mode` input; checkpoints + `config.yaml` kept per run (launcher behaviour) |
| `H2S-NN/scripts/ablation/{dry_run,param_table}.py` | V1 | follow `ARM_CHANGES` (docstrings only) |
| `H2S-NN/scripts/ablation/stats.py` (new) | V2 | mean ± sd, paired differences, Wilcoxon (scipy), Holm per claim over 4 datasets, median, bootstrap CI; claim rules; 2×2; JSON + Markdown |
| `H2S-NN/scripts/perturbation/make_graphs.py` (new) | V3 | targeted + random graphs at h ∈ {0.2, 0.4, 0.6, 0.8}; h via `compute_heterophily.py` (search: its `homophily_counts`; reported: its `analyse`); HGB files under `data/NC/<DATASET>_pert/<kind>_h<level>/` |
| `H2S-NN/scripts/perturbation/manifest.py` (new) | V3 | 7 methods × graphs × seeds 1-5, S2 on targeted graphs; tuned or `--published_config`; existing fixed-config launchers |
| `H2S-NN/scripts/analysis/q3_schema.py` (new) | V4 | reads `scripts/schema_readout.py` run JSONs: top 10 length-2 by coverage (c ≥ 1 %), τ-b stability + permutation null + Holm, τ-b of c, literature count, Spearman ρ vs schema homophily (reference code), two PDF figures |
| `H2S-NN/scripts/timing/time_methods.py` (new) | V7 | per method and dataset: median epoch time of epochs 2-4 after 1 warm-up, peak GPU memory, parameters; H2S-NN via `v3b_cost_smoke.py`, baselines via the card 2 D6 central-point path |
| `H2S-NN/tests/test_card7_*.py` (new), `tests/test_card4_ablation_driver.py` | all | T1-T5, T8 + the two mutations; the card 4 driver test follows the 7-row list |
| `docs/card7_evidence/` | all | job scripts, logs, reachability table, manifests counts |
| `docs/H2S-NN_card7_report.md` | — | Phase A section |
