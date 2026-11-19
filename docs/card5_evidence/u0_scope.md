# Card 5 U0 — worktree, merge, scope list

- Worktree `/mnt/data/peeradon/h2snn-ablprep` (already a worktree; before the card: branch `ablation-prep` at
  `5ccc792`, no tracked change, untracked only the `H2S-NN/data` symlink, the vendor copies of card 4 and git-ignored
  `experiments/`, `logs/`). `git switch -c card5 894fd7421ce8581d38b967906bde5593e881518a` (`origin/v3b` checked =
  `894fd74`), then `git merge --no-ff 6de4000b8fce2eef9d38d9b4914008dde5cd06ff` (`origin/design-docs` contains it)
  → `ec7494d9804c1f64956b7ab1cfe4d48d66aae860`, no conflict.
- Base suite count: **777** at `894fd74` (S6 M8, job 17765; decisions log 2026-11-19).
- Base references (SHA-256, unchanged by this card): `docs/legacy_evidence/v3b_golden_cluster.pt`
  `29215150935b6bb1412f36baec52842a78d5edb779ef9d15c3ea59266155a564`; `docs/partMB_evidence/mb0_reference_cluster.pt`
  `194868fce96fd658123165205d258392f7505b652bc974db26b5b1005dcbda37`.

## Files this card touches (anything else is a stop)
| File | Unit | Plan |
|---|---|---|
| `H2S-NN/src/trainers/minibatch_trainer.py` | U1 | On `SUBSET_SIZES` datasets only: score the B1a subset (`validation_subset`, run seed = `cfg.seed`, which drew PDNS's split, `src/data/dataset.py:57-58`) at every evaluation epoch, no test pass; best checkpoint and `bad_epochs` from the subset Macro-F1; restore the best weights (in-memory CPU copy) and score once on the whole validation and test splits; record whole-split metrics, `_subset_fields` keys and `dataset`. Other datasets unchanged |
| `H2S-NN/scripts/run_h2snn_wandb_tuning.py` | U1 | `summarize_tuned_seed` ranks by `sweep_objective` (the end-of-trial value already does, `src/utils/wandb_campaign.py:474-478`) |
| `H2S-NN/src/analysis/mechanism_hooks.py` (new) | U2 | (a) energy per node at `Y*`, (c) transports per fact (chunked), (d) cells per target, knockout read-back; read-only |
| `H2S-NN/src/configs/config.py` | U2 | `evaluation.knockout_nodes` (empty default) and its refusal in tuning (`EVALUATION_ONLY_KEYS` pattern) |
| `H2S-NN/src/models/h2snn_model.py` | U2 | Only (i) the knockout mask in the anchored forward (evaluation-only, refused in a training forward like `type_only_maps`) and (ii) an opt-in read-only snapshot of the last solve's operator inputs and `Y*` for hook (a). Why the forward: the key must act in the full-batch and the mini-batch evaluation paths (the full-batch trainer is not in this card's list); `Y*`'s fact and cell states never leave the forward today. Why cells are removed from the solve rather than given `β_f = 0`: a cell row with `β_f = 0` has `D_f = 0`, and the Jacobi preconditioner divides by it (`anchored_solve.py` `jacobi_inverse`, "a cell never has d = 0"); with `β_f = 0` every term of the cell vanishes and `z_f` decouples, so `x*` is the solution without that cell |
| `H2S-NN/scripts/freebase_memory_probe.py` (new) | U3 | full batch, `d` 64, `ℓ_max` 5, `r` 64, one step + one evaluation forward; peaks, counts, time |
| `H2S-NN/scripts/tsubame/make_cell_list.py` | U4 | new phase `h2snn` (24 tuning lines + final-seeds lines); `baselines-1` unchanged |
| `H2S-NN/tests/test_card5_*.py` (new) | U5 | T1–T9, mutations |
| `docs/card5_evidence/`, `docs/H2S-NN_card5_report.md` | all | evidence, job scripts, report |
