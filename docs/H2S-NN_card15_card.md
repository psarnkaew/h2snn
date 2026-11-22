# Card 15 — Replace LP acm_pa with acm_pp (ACM paper–cite–paper)

Written 2026-11-22 by the model design session; approved by the author the same day. Report: `docs/H2S-NN_card15_report.md`.

## Why
acm_pa is ill-posed under our LP protocol (cards 11 A and 12 L): 47 % of validation-positive authors are isolated and featureless, the degree-only AUROC inverts (0.684 train / 0.448 val), GCN fails too. Dropping it alone leaves one homophilic LP task (dblp_ap). acm_pp keeps ACM in the LP table: the target is HGB ACM `paper-cite-paper` (rel 0; stored reverse `paper-ref-paper`, rel 1). Both ends are papers with keyword features and three other relations (author, subject, term), so a held-out citation does not isolate them. No acm_pa baseline has run and the H2S-NN acm_pa sweep is void, so the run count is unchanged.

## Preconditions
- Worktree `/mnt/data/peeradon/h2snn-card15`, branch `card15` from `origin/v3b` `66e9349b79021784a08b37c5f8be0798c0260a04`; `git merge` the design-docs SHA given in the prompt; copy (not symlink) the git-ignored `vendor/`. Cards 12 and 14 are being merged elsewhere (`merge-c12-c14`); do not touch `src/models/anchored_solve.py`, `src/data/{path_sampler,wisp_sampler,wisp_tasks,scopes}.py` or `src/trainers/`.
- Read: `scripts/lp/build_derived_lp.py` (`DERIVED_TASKS`, `exact_reverse`, the F17 feature rule), `src/data/link_prediction_dataset.py`, `src/configs/config.py` (the `acm_pa` entries), `src/baselines/registry.py` (`DERIVED_LP`), every adapter that lists `acm_pa` (`git grep acm_pa -- H2S-NN/src`), the card 12 L script, and how the heterophily records `docs/heterophily_evidence/records/lp__acm_pa__*.json` were made.

## Steps
1. **Build** `acm_pp` with `build_derived_lp.py`: `DerivedTask("acm", 0)`, same split settings as the other derived tasks (seed 42, val 0.09, test 0.1, the reverse relation's rows removed with them; by pair if the builder requires it). The F17 rule should remove nothing (paper features are keywords, not computed from citations): state what it removed.
2. **Degree check** with card 12's L script on acm_pp (dblp_ap alongside as reference). **If acm_pp is flagged by the card 12 rule, stop and report**: the author then drops ACM LP instead.
3. **Wire it in** wherever `acm_pa` is listed: `DERIVED_TASKS`, `link_prediction_dataset.py`, `config.py` (target relation `("paper", "rel_0", "paper")`, task type), `registry.DERIVED_LP`, each adapter's supported-dataset set, and any meta-path choice an adapter makes per LP dataset (use the adapter's own rule for an ACM paper–paper target; state each choice in the report). Remove `acm_pa` from `registry.DERIVED_LP` so no acm_pa line is generated; leave its other entries in place.
4. **H2S-NN config** `configs/experiments/acm_pp_h2snn_v3b.yaml`, a copy of `acm_pa_h2snn_v3b.yaml` with only the task changed.
5. **Heterophily values** for acm_pp with the same scripts and definitions as the acm_pa records.

## Tests (only these)
- The existing LP tests that list the derived tasks include acm_pp and pass.
- One smoke epoch on acm_pp for H2S-NN and for GCN (shared LP loop): finite loss, parameters move.
- The full suite by `sbatch`.

## Rules
`sbatch` only, one GPU, `--mem` = measured peak + 30 %. No tuning, no performance runs. Stage by explicit path, push `card15`, never merge into `v3b`. Stop on: acm_pp flagged; the builder finds no exact reverse for rel 0; a change is needed in a file reserved for cards 12/14.

## Report (`docs/H2S-NN_card15_report.md`)
Split counts, the degree-check row, the files changed, adapter choices, heterophily values, test results. Last line `branch=card15 sha=<sha> card=docs/H2S-NN_card15_card.md base=66e9349b79021784a08b37c5f8be0798c0260a04`.

## Estimate
≈ 1 day (mostly step 3, mechanical).
