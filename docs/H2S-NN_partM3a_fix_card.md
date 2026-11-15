# Task card — Part M3a-fix: `concat_mlp` removed everywhere, notebooks without the decoder switch

**Binding:** `docs/working_preferences.md` (all; in particular "Review, approval and branch flow (2026-11-15)" and "Everything runs on the cluster"); `docs/H2S-NN_partM3a_card.md` and `docs/H2S-NN_partM3a_report.md` (approved at `310e5c7`, "Decisions for the author" 1-3); `docs/H2S-NN_cluster_setup3_report.md` (E, 1 thread); `H2S-NN/AGENTS.md`/`CLAUDE.md`. Spec **v1.22** W2 is the design authority and is not touched: "`concat_mlp` is removed: no baseline has it and HGB uses only dot and DistMult".
**Author decisions (M3a review, 2026-11-15):** (1) the Colab notebooks lose the decoder switch and every mention of `concat_mlp`; (2) `concat_mlp` is refused by config validation as well, and its model code is removed (spec W2 says removed; the M3a card had kept it only to keep M3a small); (3) the tuner keeps `--decoder_type {dot, distmult}` (no change).
**Branch:** `part-m3a-fix`, created from `v3b` at `310e5c7b139ead942205cd8c0f8012f5e3e616d3`, then `git merge --no-ff` of the pinned `design-docs` SHA (given in the prompt; a conflict is a stop). **Report:** `docs/H2S-NN_partM3a_fix_report.md`; evidence `docs/partM3a_fix_evidence/`.

## Rules
- Correctness only; no real data, no metric, no comparison between methods. Environment `h2snn-b`; every computation in a SLURM job; tests through `scripts/slurm/run_tests.sbatch` (E exported); every job script (kept in the evidence folder) prints `git rev-parse HEAD`, `hostname`, `$SLURM_JOB_ID`. Never mutate a file a running job reads.
- **Gate after every unit:** the golden tests (`v2_golden_cluster.pt`) pass and `python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt` reports `differing: none` (exit 0). `dot` and `distmult` outputs and parameters must not change (the `concat_mlp` branch builds its modules only for `concat_mlp`, so removing it changes no RNG draw for the other two; if a gate differs, that assumption is wrong: stop).
- Forbidden: loosening tolerances; deleting or skipping tests other than the `concat_mlp` subcases named below; silencing exceptions; editing the spec, cards or decisions log; overwriting any reference file; touching vendor clones or the author's untracked files (`H2S-NN/scripts/slurm/run_use_dataset.sbatch`, `docs/lab-gpu-cluster.html`, the dirty `vendor/H2GB`).
- Git: one commit per unit, explicit paths, "M3a-fix-<unit>: ..."; push `part-m3a-fix` after each unit; never push or move any other branch; never force-push, rebase, amend or stash.

## Units

### F0 — Preconditions (no code)
`git fetch`; check `origin/v3b` = `310e5c7…` and `origin/design-docs` = the pinned SHA; create the branch and merge. `grep -rn "concat_mlp\|include_decoder_type\|INCLUDE_DECODER_TYPE" H2S-NN --include=*.py --include=*.ipynb --include=*.md --include=*.html --include=*.yaml` (vendor excluded); record every hit with file:line and the unit that handles it.

### F1 — `concat_mlp` refused by config and removed from the model (decision 2)
- `src/configs/config.py`: `DataConfig` validation (LP branch, l.958) accepts only {`dot`, `distmult`}; `concat_mlp` raises `ValueError` with `REMOVED_DECODER_MESSAGE` (where = "data.decoder_type"); the comment at l.672 lists `"dot" | "distmult"`. The M3a refusal in `reject_never_tuned_settings` stays.
- `src/models/link_readout.py`: remove the `concat_mlp` branch in `__init__` and in `forward` and its mentions in the docstring; an unknown decoder still raises. The constructor arguments `hidden_dim`/`dropout` may stay if removing them would change a caller; say which.
- `src/models/h2snn_model.py` docstring l.15 and the comments in `scripts/run_h2snn_runs.py:39`, `scripts/train_h2snn.py:24`, `scripts/run_h2snn_wandb_tuning.py` that say "its model code stays": correct them.
- **Tests:** a config with `decoder_type: concat_mlp` (YAML through the config loader, and `train_h2snn.py --config` / `run_h2snn_runs.py --config` on a synthetic config, without training) raises naming W2; `LinkPredictionReadout(decoder_type="concat_mlp")` raises. Stated test change: `tests/test_task_heads.py:149` loops over (`dot`, `distmult`) only. **Mutation:** the config accepts `concat_mlp` again → the config refusal test fails.

### F2 — Notebooks (decision 1)
In `notebooks/h2snn_{graph_classification,link_prediction,node_classification}_colab.ipynb`: remove `INCLUDE_DECODER_TYPE_IN_TUNING` and the `--include_decoder_type` append; in the LP notebook the decoder comment lists `'dot', 'distmult'` only. Edit cell sources only, keep each file's JSON layout (the NC notebook is one line), no outputs, no execution counts. Search the other `h2snn_*` notebooks too (F0 list).
- **Tests:** extend the notebook test in `tests/test_wandb_and_runs.py` (or add one) so that every `h2snn_*` notebook contains neither `--include_decoder_type`, `INCLUDE_DECODER_TYPE`, nor `concat_mlp`. **Mutation:** the switch restored in one notebook → the test fails.

### F3 — Documentation text
`H2S-NN/CLAUDE.md` l.884 and l.962 and the same text in `AGENTS.md` (this card authorises these lines only): the LP decoder is `dot` | `distmult` (spec v1.22 W2; `concat_mlp` removed). `H2S-NN/docs/h2snn_current_model_documentation.html` l.1553 and l.1590: the same correction (text only).

### F4 — Full suite and report
`sbatch scripts/slurm/run_tests.sbatch`. Expected: as M3a-A4 (706 tests; only `test_notebooks_are_committed_clean` fails, which reads the baseline notebooks, not touched here) plus the new tests green, minus none; mb0 compare `differing: none`. Report: per unit done / VERIFIED-RUN / deviations; the F0 list; stated test changes (old, new, reason); "Decisions for the author"; out-of-scope findings; the report line.

## Stop conditions
F0 hash mismatch or merge conflict; any gate difference; a test of kept code that would need deleting; a caller outside tests that builds `concat_mlp` from code the card does not name; the card wrong or ambiguous. **Not stop conditions:** extra `concat_mlp` mentions in comments or docs (fix them under F3 if in `H2S-NN/` text files outside vendor, else list them).

## Time (estimate)
About 2-3 h including queue.
