# Card 13 — GC epochs 500 (config only)

Written 2026-11-21 by the ops session at the model session's request (author decision relayed by the model session: GC epochs 500, as HetSheaf's sweep files; spec 1.30 records it). Report: `docs/H2S-NN_card13_report.md`.

## Why
HetSheaf's GC sweep files (all 60, design-docs `6964e1e`) train 500 epochs with early stopping on the validation loss (patience 50) and test with the final weights. Our four GC experiment configs say `epochs: 200`. Everything else already matches: the resolved configs give patience 50, early stopping on validation loss and final-weights reporting (`protocol_budget` source `hetsheaf`, `src/configs/config.py:719`).

## Preconditions
- Worktree `/mnt/data/peeradon/h2snn-card13`, branch `card13` from the current `origin/v3b` (state its SHA in the report). If `v3b` moves before approval (card 10, card 12), `git merge origin/v3b` into `card13`; never rebase.
- Read: this card; `configs/experiments/{mutag,proteins,enzymes,nci1}_h2snn_v3b.yaml`; `TrainConfig.from_dict` in `src/configs/config.py` (epochs, `early_stopping_patience`, `protocol_budget`); `tests/test_experiment_configs.py`; `tests/test_gc_protocol.py`.

## Unit E — the change
1. In each of `configs/experiments/mutag_h2snn_v3b.yaml`, `proteins_h2snn_v3b.yaml`, `enzymes_h2snn_v3b.yaml`, `nci1_h2snn_v3b.yaml`: `epochs: 200` → `epochs: 500`. No other key changes.
2. Nothing in `src/` changes. `ProtocolBudget("hetsheaf", 200, 50, ...)` stays: it is used only when a config gives no `epochs`, and these four files give one.

## Tests
- New test (in `tests/test_gc_protocol.py` or a new `tests/test_card13_gc_epochs.py`): for each of the four files, `TrainConfig.from_dict(yaml)` gives `epochs == 500`, `early_stopping_patience == 50`, `checkpoint_metric == "macro_f1"`, `checkpoint_mode == "max"`, and the trainer's early stopping is on the validation loss (assert through the resolved budget's monitor `loss` / mode `min`).
- **Mutations** (each must make a test fail; revert after):
  - one file back to `epochs: 200`;
  - add `early_stopping_patience: 500` to one file.
- Update any existing test that pins these four files' epochs to 200 (name it in the report).
- Full suite on the lab (`run_tests.sbatch`, CPU, as card 10): state the job id, counts and every error. The 4 PSHGCN errors from a checkout without a built `sparse_tools` are known environment errors; name them as such if they appear.

## Done when
- The `git diff origin/v3b...card13` touches only the four YAML files and the tests.
- Report the following in `docs/H2S-NN_card13_report.md`, ending with the line `branch=card13 sha=<head> card=docs/H2S-NN_card13_card.md base=<v3b sha>`:
  - the diff stat;
  - the test and suite results;
  - the mutation results.

## Note
The base-config fingerprint of the four GC cells changes, so a GC tuning state written at an older SHA is refused on `--resume`. GC restarts with new state directories at the new SHA, so nothing is lost.
