# Card 2 D0 — preconditions and cell list (no runs)

Branch `dry-run`: the worktree `/mnt/data/peeradon/h2snn-dryrun` already existed (clean, created at `4bfa5de`, no commits,
never pushed); fast-forwarded to `6ba6dec00369ff590b341ce3771da90528fcc1cd` (`--ff-only`, equal to creating it there), then
`git merge --no-ff 726ec849d2a644b50e4a0ff22d5a37c753ab7075` (merge `f255a63`). `origin/v3b` and `origin/design-docs`
equalled the pinned SHAs after `git fetch`.

Data and vendor: `H2S-NN/data` is a symlink to the main checkout's `H2S-NN/data` (read only; never staged). The vendor
clones were **copied** (`cp -a`, code only, identical files) instead of symlinked: the clones are ignored by directory-only
patterns (`/vendor/H2GB/` in `.gitignore`), so a symlink would appear untracked, and symlinked clones defeat module eviction
in `src/baselines/runtime/vendor_paths.py:88` (Part M3b out-of-scope finding).

## W&B — STOP (card D0, prompt)
Job 16741 step 82 (g2n01, h2snn-b): no `~/.netrc`, no `~/.config/wandb`, no `WANDB_*` variable; `wandb.api.api_key` is
empty (wandb 0.28.0). **No W&B key is configured on the cluster.** The author runs `wandb login` once (key stored under
`/mnt/home/peeradon`, never in the repository). D2 (W&B project `h2snn-dryrun`) and D3 wait for it.

## Documents read
Card `docs/H2S-NN_dryrun_card.md`; `docs/working_preferences.md` (incl. Slim preparation); decisions log rows of
2026-11-16; Part M3b report; baselines-on-H2GB report; Part M card (M2, M10); `H2S-NN/AGENTS.md` (whole), `CLAUDE.md` (not
yet whole: environment and testing sections). Code: `scripts/run_h2snn_wandb_tuning.py` (args, `main`),
`scripts/run_baseline_tuning.py` (args), `scripts/run_h2snn_runs.py` (args), `scripts/train_h2snn.py`,
`src/baselines/registry.py`, `src/baselines/search_grids.py`, `src/configs/config.py` (presets, budgets, `DataConfig`,
`ModelConfig`, `TrainConfig.from_dict`), `src/builders.py`, `src/trainers/minibatch_lp.py`.

## How each dataset gets its configuration today
`configs/experiments/*_h2snn_v3b.yaml` exist for acm, dblp, imdb (NC), acm_pa (LP), mutag (GC). Every other dataset has
only its preset in `src/configs/config.py` (`NODE_CLASSIFICATION_PRESETS`, `LINK_PREDICTION_PRESETS`,
`GRAPH_CLASSIFICATION_PRESETS`). Resolve check of `{'data': {'dataset_name': X}}` (`d0_cells.py`, job 16741 step 83,
`d0_cells.txt`):
- **NC freebase and the 8 H2GB NC datasets do not resolve**: `no calibrated data.wisp_walks_per_anchor ... set r
  explicitly` (`config.py` `WISP_WALKS_PER_ANCHOR` has no entry) -> D1.
- LP dblp_ap, imdb_ma, freebase_bb, pubmed, lastfm and GC proteins, enzymes, nci1 resolve from the preset (LP/GC leave
  r = 0, which a WISP draw refuses: a runnable config must name r) -> D1 writes their YAML as the card lists.
- **Large LP `ogbn-mag_pp`, `h-pokec_uu`: no H2S-NN entry point.** `DATASET_TASK_TYPES` has no entry ("Unknown dataset");
  with `task_type` set, `build_experiment_components` -> `build_link_prediction_dataset` knows local tasks only
  (`link_prediction_dataset.py:878-918`). `LinkEpochLoop`/`LinkScopes` (`src/trainers/minibatch_lp.py`) are tested on
  synthetic graphs but no builder assembles the message graph from the H2GB bundle + `LP/large/<task>` files and no
  trainer calls the loop. Not a configuration: wiring is new code (a builder and a trainer path), outside D1 ("no new
  setting") — **decision for the author / design session** (a follow-up card), not done here.
- PDNS: the card says "7 loaded H2GB NC datasets"; since card 1 (C2, in `v3b` `6ba6dec`) PDNS is loaded too
  (`H2GB_NC_LOADED` has 8). H2S-NN on PDNS is included (8 H2GB NC cells).

## Cell list (`d0_cells.md`, `d0_cells.json`)
| group | cells | state |
|---|---|---|
| H2S-NN ready (config resolves) | 13: NC acm, dblp, imdb; LP acm_pa, dblp_ap, imdb_ma, freebase_bb, pubmed, lastfm; GC mutag, proteins, enzymes, nci1 | D2 (LP/GC presets get a YAML in D1) |
| H2S-NN config missing | 9: NC freebase + 8 H2GB NC | D1 |
| H2S-NN no entry point | 2: ogbn-mag_pp, h-pokec_uu | decision (above) |
| Baselines runnable (registry) | 141: 135 on HGB NC / local LP / GC + 6 on H2GB NC (ChebNetII x 5 MAG/OAG, SeHGNN ieee-cis-g) | D2 |
| Baselines on H2GB NC / large LP not wired | 114 (incl. SeHGNN h-pokec N/A by the author) | waiting for card 1b |
| Baselines without implementation / Freebase not wired | 33 | not cells |
