# Part O report — schema read-out and coverage of the main runs, fixed-literature-schema row

**Card:** `docs/H2S-NN_partO_card.md` (O0–O4). **Design authority:** spec v1.19 (§3.2 read-outs; §6 Ablation-table bullet "fixed-literature-schema row" and bullet "Schema read-out and coverage"); `docs/predefined_metapaths.md` §2; `docs/experiment_settings_decisions.md`. **Branch:** `part-o` from `v3b` at `af2b4e2`. **Evidence:** `docs/partO_evidence/`.
**Rule kept throughout:** correctness tests only. No training run on real data, no metric, no `I_σ` of a trained model, no comparison between methods. The method (WISP, energy, maps, solver, head readouts) is unchanged.

---

## O0 — Preconditions

**Branch (VERIFIED-RUN).** `git checkout -b part-o af2b4e2` → `part-o`; `git rev-parse v3b` = `af2b4e2bebe8949ec333d0c0abed69f72b7343ba` (v3b is at the given commit).

**Decisions read (VERIFIED-SOURCE, `docs/experiment_settings_decisions.md`).**
- O-1 revised (2026-11-13 03:11): the row uses HGB's published meta-path table, paths starting at the target type, on ACM, DBLP, IMDB, Freebase; it supersedes v1.18's Hetero²Net list. No row on H2GB datasets.
- O-2 (2026-11-13 02:45): walks per anchor = the tuned `r`, split evenly over the listed meta-paths, no separate tuning.
- Z-3 (2026-11-13 02:45): schema read-out and coverage approved. Spec §6 then fixes the reading in advance: `c_σ`, `I_σ`, `ρ_σ`, mean `g`, top ten length-2 sub-path schemas, number of distinct schemas, Kendall τ across seeds. `ΔL_σ` is not reported (v2 operator).
- 2026-11-12 21:03: no smoke runs, correctness tests only. The card's 2-epoch smoke was removed.

**Code read (VERIFIED-CODE).**
- `src/analysis/wisp_readouts.py`. `pair_index` / `_canonical` (l.49–82) build the canonical sub-path signature `(t_0, r_1, d_1+1, t_1, …)`, which is the smaller of the two orientations. `i_sigma` (l.126–134) gives each pair `(i<j)` of cell `f` the weight `w_f m_f / (2|f|)` (Prop. K) and normalises the total to 1. `kendall_tau` (l.266–284) returns τ-b with a permutation p. `delta_loss` (l.230) uses the v2 operator (`build_operator`, `VertexPathSheafLaplacian`) and is not used here.
- `src/utils/checkpointer.py`. `save` (l.46–64) writes `state_dict`, the optimizer, metrics and metadata to `<exp_dir>/checkpoints/<tag>.pt`. `save_if_best` (l.77–97) writes `best.pt`. A strict `load` is at l.66–75.
- `src/analysis/model_loading.py` `load_trained_model` (l.14–36): `config.yaml` → `TrainConfig.from_dict` → `build_dataset` → `build_model` → `Checkpointer.load`.
- Main-table seed runs: `scripts/run_h2snn_wandb_tuning.py`. Tuning runs on seed 0 (`TUNING_SEED`, l.76). `--final_seeds 0 1 2 3 4` calls `run_final_seeds` (l.690–723). That calls `train_trial` (l.384–473) with `output_dir = <output_dir>/<exp>/final_seeds` and `exp_name = <exp>_tune_final_seed{s}_final_seed{s}`, so each seed has its own directory. `train_trial` sets `raw["seed"] = seed` (l.411), builds `TrainConfig.from_dict(raw)` and writes `save_config(cfg, exp_dir/"config.yaml")` (l.420–421) before training. The trainer writes `checkpoints/last.pt` every epoch and `checkpoints/best.pt` (`src/trainers/trainer.py:145-159`). The other drivers (`scripts/run_h2snn_runs.py:128-129`, `scripts/train_h2snn.py:46-47`) also save `config.yaml` first.
- Deletion: `grep -rn "rmtree|unlink|os.remove|.rmdir|shutil|TemporaryDirectory"` over `scripts/` and `src/` (vendored adapters excluded) finds deletions only in `scripts/lp/apply_feature_rule.py:114` and `scripts/lp/write_hgb_negatives.py:70,92`. Both are data-preparation scripts. Nothing deletes a run directory (VERIFIED-RUN, command in O1).
- Evaluation draw: `build_dataset` builds the cells with `seed=config.seed` (`src/data/dataset.py:236-246`). The trainer evaluates on these cells (`trainer.py:141-142`). Training draws use `seed + epoch` (`trainer.py:68-84`).

**Findings that shape O1–O3 (VERIFIED-CODE).**
1. `DataConfig._resolved_node_classification` (`src/configs/config.py:745-790`) fills `metapath_templates` from the old presets (ACM PAP, PSP; DBLP APA, APTPA; IMDB MAM, MDM) for every NC configuration, including `path_source: wisp`. A saved tuned `full` configuration therefore carries those presets. If only `path_source` were changed, the row would sample the old presets. O3 deals with this.
2. `DataConfig.__post_init__` (l.517–525) rejects `wisp_walks_per_anchor ≠ 0` and `wisp_max_length ≠ 4` under `path_source: schemas`. A tuned `full` configuration has both, so the builder could not change only the path source.
3. The model treats schema groups differently from WISP groups. For a schema group, `g`'s hop token is one relation embedding per group, with no direction one-hot (`h2snn_model.py:858-876`), and `face_lengths` come from the draw (`build_model`, l.1419–1431). For example, ACM PcPAP and PrPAP would give `g` identical hop tokens.
4. H2S-NN has no Freebase node-classification loader. `DATASET_TASK_TYPES` (`config.py:399-413`) and `_load_hetero_data` (`dataset.py:26-45`) cover ACM, DBLP and IMDB only. See out-of-scope findings.

**Pre-state of the suite.** No code changed between Part N-fix F5 (`5038744`) and `af2b4e2`: `git diff --stat 5038744 af2b4e2` lists only `docs/` files (VERIFIED-RUN). F5's full suite gave **Ran 583; failures=2, errors=2**, all in `test_baselines` (vendor clones ×3, notebook) (`docs/partNfix_evidence/f5_full_suite_output.txt`). This is the comparison set for O4. The working tree also holds uncommitted edits to `src/baselines/adapters/*` from another session. They are not part of this branch and are never staged.
