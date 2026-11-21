# Card 13 report — GC epochs 500 (config only)

Card: `docs/H2S-NN_card13_card.md` (design-docs `0441668`). Branch `card13` from `origin/v3b` = `2d822a4`
(worktree `/mnt/data/peeradon/h2snn-card13`; `v3b` did not move during the card, so no merge).

## Change (Unit E)
- `configs/experiments/{mutag,proteins,enzymes,nci1}_h2snn_v3b.yaml`: `epochs: 200` → `epochs: 500`. No other key.
- Nothing in `src/` changes; `ProtocolBudget("hetsheaf", 200, 50, ...)` stays (used only when a config has no `epochs`).
- New test `tests/test_card13_gc_epochs.py`: for each of the four files, `TrainConfig.from_dict(yaml)` gives
  `epochs == 500` and `early_stopping_patience == 50`.
- No existing test pinned these four files to 200, so no test was updated. (The `(200, 50)` cases in
  `tests/test_shared_protocol.py` are the GC defaults for a config without `epochs`; unchanged and still pass.)

Commit `d263a11`. Diff stat `git diff --stat origin/v3b...card13` (code commit):

```
 H2S-NN/configs/experiments/enzymes_h2snn_v3b.yaml  |  2 +-
 H2S-NN/configs/experiments/mutag_h2snn_v3b.yaml    |  2 +-
 H2S-NN/configs/experiments/nci1_h2snn_v3b.yaml     |  2 +-
 H2S-NN/configs/experiments/proteins_h2snn_v3b.yaml |  2 +-
 H2S-NN/tests/test_card13_gc_epochs.py              | 32 ++++++++++++++++++++++
 5 files changed, 36 insertions(+), 4 deletions(-)
```
plus this report (committed on top of `d263a11`; the closing line names the code head, as in the card 8 report).

## Targeted tests
One CPU sbatch job, **18141** (g3n01, `--mem 16G`, `--time 1:00:00`, `CUDA_VISIBLE_DEVICES=""`, env `h2snn-b`),
git `d263a11`, clean tree for `src scripts tests configs`. Modules: `tests.test_card13_gc_epochs`,
`tests.test_experiment_configs`, `tests.test_gc_protocol`, `tests.test_shared_protocol`.

**Ran 18 tests: OK (skipped=3).** The three skips are `tests.test_gc_protocol.ProtocolOnMutagTests`
(`test_constant_feature_and_batch_size_one`, `test_node_and_edge_types_are_the_one_hot_argmax`,
`test_split_is_hetsheaf_randperm_seed_42_for_every_training_seed`): "TU data not present" in the fresh worktree.
They load MUTAG data and do not read the experiment YAMLs; the full suite at review runs them where the data is.
No full suite was run (card).

## Mutation
Same job 18141, after the targets: `proteins_h2snn_v3b.yaml` set back to `epochs: 200` (anchor found once), then
`tests.test_card13_gc_epochs` run: **FAILED (failures=1)**, subTest `dataset='proteins'`, `AssertionError: 200 != 500`
→ **M1 CAUGHT (rc=1)**. File restored from the backup; `git diff --stat -- configs` empty afterwards.

## Note
The GC base-config fingerprint changes at this SHA, so an older GC tuning state is refused on `--resume`; GC restarts
with new state directories (card note).

branch=card13 sha=d263a119d5c5ebf409f478324f807e9ee03433d8 card=docs/H2S-NN_card13_card.md base=2d822a4eb48867de4476b97159430a64d8468808
