# Card 4b report — card 4 merge and pre-freeze fixes (L0–L4)

**Card:** `docs/H2S-NN_card4b_prefreeze_card.md`. **Branch:** `ablation-prep` from card 4's head `05025b7`.
**Binding:** spec v1.24 (Y1–Y4), decisions-log row "Card 4 review (D1–D7)". Correctness only; no run
beyond the suite; no metric.

## Verdict
- L0 merged cleanly; L2 and L3 done with their tests; L4 suite and compares below.
- **L1 stopped (stop condition: the proposal and the training paths do not match):** only the full-batch
  NC trainer reads `data.wisp_proposal`; the mini-batch NC, LP and GC training draws are always uniform.
  Spec Y1 says `adaptive` for every H2S-NN run. No L1 change was made; decision D1 below.

## L0 — merge
`git merge --no-ff 7df909d` (`v3b`) → `66090c8`, then `git merge --no-ff 61aa2f6` (`design-docs`) →
`847a3e1`. No conflicts, so no notebook was regenerated. The card-4 snapshot worktree
`/mnt/data/peeradon/h2snn-ablprep-run` was removed (`git worktree remove --force`; it held only untracked
run outputs; no job used it).

## L1 — `adaptive` proposal: stop item, no change
Where `data.wisp_proposal` reaches training (VERIFIED-CODE):

| Path | Reads the proposal? |
|---|---|
| Full-batch NC (`src/trainers/trainer.py:54-66, 116`: q from the previous epoch's draw and w_f; HGB ACM, DBLP, IMDB, Freebase full batch) | yes |
| Mini-batch NC, the eight H2GB datasets (`src/trainers/minibatch_trainer.py:10, 111-115`: `build_wisp_anchor_cells` per batch, documented "`proposal: uniform`") | no: always uniform |
| LP (`src/data/wisp_tasks.py:50-57`: `build_wisp_path_groups` without a proposal; `link_prediction_trainer.py:54-60`) | no: always uniform |
| GC (`src/trainers/graph_classification_trainer.py:220-256`, per-graph draws via `wisp_tasks`) | no: always uniform |

Making Y1 true for every run needs `src/trainers/` and `src/data/` changes (outside card 4b's file list).
Mini-batch `adaptive` also needs a design: q now comes from the previous epoch's whole-graph draw and w_f.
Changing only the config default would make LP, GC and mini-batch configs name a proposal their training
never uses. It would also fail validation wherever `resample_paths` is false (`config.py:1041-1044`).
Known-answer scripts set `wisp_proposal: uniform` explicitly (`scripts/known_answer/common.py:56`;
`resample_paths: false` there), and the K2 gate script does too (`scripts/ablation/variance_gate.py:60`).
Either outcome leaves them unchanged.

## L2 — arm-sweep key exclusion (D5, Y4)
- `src/configs/config.py`: `ARM_INERT_SEARCH_KEYS` (`no_path_cells` → `wisp_max_length`,
  `wisp_walks_per_anchor`; every other arm → none) and `arm_search_space(parameters, tuning_arm)`, which
  refuses an empty arm.
- `scripts/run_h2snn_wandb_tuning.py` (`_build_sweep_for_run`): under `tuning.arm` the arm's inert keys are
  dropped; they keep the **base config's value** (the arm never reads it). Main-table sweeps are unchanged.
- `scripts/ablation/sweeps.py` reads the same table and records that the launcher's space equals the
  restricted one. This closes card 4's K6 stop item.
- Tests (`tests/test_card4b_prefreeze.py`): (1) every arm's launcher space = main-table space minus its
  entry, and `no_path_cells` lacks both keys; (2) the main-table space equals `build_sweep_config`'s, and
  `arm_search_space(…, "")` is refused; (3) table check: each entry is **shown unread**, i.e. the arm's
  evaluation logits are bit-identical (`torch.equal`) for two values of the key (`wisp_max_length` 2/4,
  `wisp_walks_per_anchor` 3/8; shared parameters copied, synthetic ACM-shaped graph). The mutation entry
  `full: (wisp_max_length,)` is refused by the same check.

## L3 — arm switches refused (D7, Y4)
`ARM_KEYS` gains `ablation.direction_blind_maps` and `ablation.h0_only_readout`. Both are refused in a
main-table tuning base config. Each is accepted only in its own arm-tuning mode (`tuning.arm:
direction_blind_maps` / `h0_only`, added to `ARM_TUNING_KEYS`; neither is a configuration-(b) arm of the
pre-registration, and `scripts/ablation/arms.py` `TUNED_ARMS` stays the five). Test: refused in
main-table mode, accepted in its own mode, refused in another arm's mode. Card 4's driver tests were
adjusted (the arm-key list and "the five are a subset of the modes").

## L4 — suite
Full suite at the clean head `b98181b` (job 16898, g3n01, sbatch, GPU not shared; pinned kernel
environment E): **`Ran 719 tests … OK`** (card 4: 706; +13 from the merged `v3b` and card 4b's tests);
`v3b_golden.py compare` → `compared 1365 tensors in 24 configurations; differing: none`;
`mb0_reference.py compare` → `compared 139 tensors; differing: none`
(`docs/card4_evidence/card4b_suite.log`, INFO lines filtered).

## Files changed
`src/configs/config.py`, `scripts/run_h2snn_wandb_tuning.py`, `scripts/ablation/sweeps.py`,
`tests/test_card4b_prefreeze.py` (new), `tests/test_card4_ablation_driver.py`, this report and
`docs/card4_evidence/card4b_suite.log`. No golden or mb0 fixture changed (L1 not done), and no
reference was re-captured.

## Deviations
1. L1 not done (stop item).
2. L2 and L3 were tested together (job 16897, 73 tests OK). The L2 commit `1adc1ec` is the same code
   without L3's lines, and was not run on its own.

## Decisions for the author
| # | Question | Options | Recommendation and reason |
|---|---|---|---|
| D1 | Scope of Y1 (`adaptive`) | (a) Y1 applies to full-batch NC (HGB ACM, DBLP, IMDB, Freebase full batch). The default resolves to `adaptive` there and stays `uniform` for mini-batch NC, LP and GC, and the spec says so. Small `config.py` change in a follow-up prompt, with golden/mb0 pins if needed. (b) Implement `adaptive` in the mini-batch, LP and GC draws (new `src/` design and card; the gate would need re-measuring in those modes) | (a): K2 measured the gate only on full-batch ACM; (b) is new method work before the freeze |

branch=ablation-prep sha=<see final message> card=docs/H2S-NN_card4b_prefreeze_card.md base=05025b7e401c35b8ab7193195db46f6c79323bee
