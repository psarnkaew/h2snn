# Card 17 — IEEE-CIS-G: five H2GB cells as published-configuration runs (weighted cross-entropy)

Written 2026-11-22 by the ops session on the author's decision of the same day (design-docs `54c241a`, option A): the five copied H2GB IEEE-CIS-G numbers (GCN, GAT, R-GCN, HAN, H2G-former) are binary F1 and are not used; each cell runs H2GB's own configuration `vendor/H2GB/configs/IEEE-CIS/IEEE-CIS-<Method>.yaml` **as written** (including its loss and loss weights, schedule, sampler), 5 seeds (0–4), on the H100, and reports Macro-/Micro-F1 like every other cell. Report: `docs/H2S-NN_card17_report.md`.

## Why a card
- The five cells are `filled` in `configs/baselines/paper_filled_cells.tsv` (lines 6, 11, 16, 21, 28), so `cell_status` never makes them `published` and `run_baseline_published.py` refuses them (`scripts/run_baseline_published.py:135`).
- They have no published-parameter file `configs/baselines/<model>/node_classification/ieee-cis-g.yaml`.
- All five IEEE-CIS yamls set `model.loss_fun: weighted_cross_entropy` with `model.loss_fun_weight: [1, 5]`. The shared H2GB loop refuses any loss other than `cross_entropy` (`src/baselines/h2gb_published.py:269-270`). H2GB's loss (`vendor/H2GB/H2GB/loss/weighted_cross_entropy.py`): with a weight list, `F.nll_loss(F.log_softmax(pred, -1), true, weight=tensor(loss_fun_weight))` for multi-class logits.

## Preconditions
- Worktree `/mnt/data/peeradon/h2snn-card17`, branch `card17` from `origin/v3b` `66e9349b79021784a08b37c5f8be0798c0260a04`; copy (not symlink) the git-ignored `vendor/`; `data` → `/mnt/data/peeradon/h2snn/H2S-NN/data`. If `v3b` moves before approval (merge of cards 12 + 14 + 15 + 16), `git merge origin/v3b` into `card17`; never rebase.
- Do not touch files reserved for the parallel cards: `src/models/anchored_solve.py`, `src/data/{path_sampler,wisp_sampler,wisp_tasks,scopes}.py`, `src/trainers/`, `src/configs/config.py` (`PROTOCOL_BUDGETS`, card 16), `src/baselines/budget.py` (card 16).
- Read: `src/baselines/cell_status.py`, `src/baselines/published.py`, `src/baselines/h2gb_published.py` (`EXCLUDED_KEYS`, `RUN_ARGUMENTS`, `MODEL_SECTIONS`, the loss check, the training step that computes the loss), the 25 existing H2GB published files (e.g. `configs/baselines/gcn/node_classification/pdns.yaml`) and how card 8 C1 generated them, the five IEEE-CIS yamls, the register rows (`docs/published_number_register.md`, now `replaced`).

## Steps
1. **Filled rows.** Remove the five `ieee-cis-g` rows from `configs/baselines/paper_filled_cells.tsv`. Nothing else in that file changes.
2. **Published files.** Add `configs/baselines/{gcn,gat,rgcn,han,h2gformer}/node_classification/ieee-cis-g.yaml` in exactly the form of the existing H2GB files (same generator or the same hand rule; `source:` names the IEEE-CIS yaml and the H2GB commit of `vendor/VENDORS.tsv`; `lines:` gives each transcribed key's line). `model.loss_fun` and `model.loss_fun_weight` are transcribed. State any key the generator cannot transcribe and stop if one changes training.
3. **Weighted loss.** In the shared H2GB loop, accept `weighted_cross_entropy` with an explicit `loss_fun_weight` list: the loss is `F.cross_entropy(logits, y, weight=torch.tensor(loss_fun_weight))` on the training targets of the step (equal to H2GB's `nll_loss(log_softmax)` form). `cross_entropy` stays the default and unchanged. `weighted_cross_entropy` **without** a weight list (H2GB's batch-dependent weights) is refused with a clear message (no yaml we run uses it). Evaluation, selection (B1a subset Macro-F1) and metrics are unchanged.
4. **Status.** `cell_status` returns `published` for the five cells; `make_cell_list.py` emits five `run_baseline_published.py` lines for them (state the lines).

## Tests (only these)
- One test: the five cells are `published`, and each file resolves (`published_params_path` exists, `yaml_published_values` round-trips the yaml's transcribed keys, `loss_fun_weight == [1, 5]`).
- One known-answer test: for fixed logits and labels, the loop's weighted loss equals `F.nll_loss(F.log_softmax(logits, -1), y, weight=w)` and differs from the unweighted loss; `cross_entropy` unchanged (bit-identical to before).
- One mutation: drop the weight in step 3; the known-answer test must fail. Revert.
- One smoke on IEEE-CIS-G, GCN only, 1 seed, a few steps (finite loss, parameters move, the record has Macro-/Micro-F1) on one lab GPU.
- Targeted modules only, as one CPU `sbatch` job (`--mem 16G`, `--time 1:00:00`) plus the smoke as one GPU `sbatch` job (`--mem` = measured peak + 30 %). No full suite: ops runs it once at review.

## Done when
- `git diff origin/v3b...card17` touches only `paper_filled_cells.tsv`, the five new yaml files, `src/baselines/h2gb_published.py` (loss only) and the tests.
- Report in `docs/H2S-NN_card17_report.md`: diff stat, the five generated files (keys transcribed and excluded), the five cell-list lines, test and mutation results with job ids, the smoke's peak memory. Last line `branch=card17 sha=<head> card=docs/H2S-NN_card17_card.md base=<v3b sha>`.

## Rules
`sbatch` only on the lab; stage by explicit path; push `card17`; never merge into `v3b`; stop and report if a transcribed key changes training in a way the shared loop cannot run exactly (spec 1.28: such a cell becomes tuned, never approximated).
