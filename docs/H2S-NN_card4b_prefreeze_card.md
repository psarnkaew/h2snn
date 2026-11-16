# Task card — Card 4b: card 4 merge and pre-freeze fixes (adaptive proposal, arm-sweep key exclusion, arm switches refused)

**Branch:** continue `ablation-prep` (worktree `/mnt/data/peeradon/h2snn-ablprep`, S4) from its reported head `05025b7e401c35b8ab7193195db46f6c79323bee`. First `git merge --no-ff` of `v3b` `7df909df4487ca8deed03aace71ccce34d5eeb69`, then of the `design-docs` SHA in the prompt. Merge, never rebase. Card 4 and 4b are approved together on 4b's final SHA.
**Binding:** spec **v1.24** (Y1–Y4), `docs/experiment_settings_decisions.md` row "Card 4 review (D1–D7)", card 4, `docs/working_preferences.md` (Slim preparation).
**Mode:** correctness only; no real-data run except what the full suite does; no metric.

## Units
### L0 — Merge
Merge `7df909d` (conflicts only possible in `src/baselines/registry.py`, `notebooks/baselines/`, `tests/test_baselines.py` counts: notebooks are regenerated with the unchanged generator, never edited by hand). Then merge the design-docs SHA.

### L1 — `adaptive` proposal (D1, spec Y1)
- `config.py`: the WISP proposal for H2S-NN runs is `adaptive` (dataclass default ≈ `:789` and the tuning guard ≈ `:255-256`, which now refuses anything other than `adaptive` in a tuning config, with the message citing spec 1.24 Y1). `path_source: schemas` keeps its own rule (≈ `:857-863`; `uniform` there, unchanged).
- Every place that resolves a WISP config for training, tuning, seed runs and the ablation driver ends up with `adaptive` (check `scripts/run_h2snn_*`, `scripts/ablation/*`, `configs/experiments/*_v3b.yaml`; an explicit `uniform` in an experiment config is changed to `adaptive` or removed; list each file).
- **Golden/mb0:** they are bit-exact references of the code path, not of the default. If a golden or mb0 fixture resolves the proposal from the default, pin `wisp_proposal: uniform` explicitly **in that fixture's config** (stated in the report) so the compare stays `differing: none`; do not re-capture any reference. Known-answer scripts (`scripts/known_answer/*`) keep the settings their runs used (state how they get them).
- **Test (one):** a main-table config resolved by the tuning launcher has `adaptive`; a tuning config with `uniform` is refused.

### L2 — Arm-sweep key exclusion (D5, spec Y4)
- Arm-tuning mode only (`tuning.arm`): a fixed table `ARM_INERT_SEARCH_KEYS` maps each arm of card 4 K6 (b) to the search keys that have no effect in that arm (at least `no_path_cells` → `wisp_max_length`, `wisp_walks_per_anchor`; any other entry needs a file:line showing the key is unread in that arm). The launcher drops exactly those keys from that arm's search space and fixes them at the main table's value for that dataset (or any value: they are unread; state which). Main-table tuning drops nothing.
- **Tests:** (1) `no_path_cells` sweep space lacks the two keys and the other arms' spaces equal the main-table space minus their table entry; (2) a main-table sweep refuses the option; (3) mutation: an entry for a key that *is* read in that arm (e.g. `wisp_max_length` for `full`) is refused by the table check.

### L3 — Arm switches refused in a main-table base config (D7, spec Y4)
`ablation.direction_blind_maps: true` and `ablation.h0_only_readout: true` in a main-table tuning base config raise, as the other arm keys do (`ARM_KEYS` pattern ≈ `:205-230`); allowed in arm-tuning mode for that arm only. **Test:** each refused in main-table mode, accepted in its own arm mode.

### L4 — Suite and report
Full suite on one node (sbatch, GPU not shared). golden and mb0 compares `differing: none` (E, 1 thread). Report `docs/H2S-NN_card4b_report.md`: short (units, files changed, tests, golden/mb0, deviations, decisions). Last line `branch=ablation-prep sha=<full head sha> card=docs/H2S-NN_card4b_prefreeze_card.md base=05025b7e401c35b8ab7193195db46f6c79323bee`.

## Files
`src/configs/config.py`, `scripts/run_h2snn_*.py` and `scripts/ablation/*` (only for L1/L2), `configs/experiments/*_v3b.yaml` (L1), golden/mb0 fixture configs (L1, pin only), tests, report, evidence `docs/card4_evidence/` (or the folder card 4 used). Nothing else.

## Rules and stop conditions
- Never re-capture golden/mb0; never change the method, ranges, budgets, tolerances; never edit vendor code, the spec, cards or the pre-registrations.
- Stop and report if: golden/mb0 differ after L1 even with the fixture pin; the proposal reaches training through a path not listed above; a key in `ARM_INERT_SEARCH_KEYS` cannot be shown unread.
- Jobs: sbatch, one GPU not shared (`--exclude=g2n01` unless nothing else is free; no `--overlap` into another session's allocation). Job scripts print `git rev-parse HEAD`, hostname, `$SLURM_JOB_ID`.
- One commit per unit; push `ablation-prep` after each unit; never push another branch; never stash, reset, rebase, amend, force-push.
