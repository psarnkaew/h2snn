# Spec 1.30 — draft edits (model design session, 2026-11-22)

Status: **DRAFT, not applied.** To be applied to `docs/H2S-NN_methodology_spec.md` after:
- the merge of cards 12 and 14 (`merge-c12-c14`) is reviewed;
- card 15 reports (acm_pp flagged or not);
- the author approves.

The method is unchanged. Sources: decisions log rows of 2026-11-21/22; the reports of cards 11, 12, 14 and 15.

## Changelog line (top of the spec)
**Changes from 1.29 (to 1.30; cards 11–15, author decisions of 2026-11-21/22; the method is unchanged):**
- (AD1) PCG zero right-hand-side rule.
- (AD2) The H2S-NN data pipeline runs on the model's device, with a device-side random stream; the seed schedule is unchanged.
- (AD3) LP acm_pa is replaced by acm_pp.
- (AD4) GC epochs 500; GC runs last; NCI1 kept.
- (AD5) Checkpoint and selection signal reaffirmed (1.28 option 1), with the benchmark differences listed in the appendix table.
- (AD6) Freeze moves to the merged `v3b`.
- (AD7) Binary F1 reported on IEEE-CIS-G and PDNS (H4 corrected).

## Edits

**AD1 — §6 "Solver settings", after "per-scope inner products and stopping in mini-batch;"** insert:
> a scope whose right-hand-side norm is at most `ρ = 1e-10` times the largest scope norm of the same call (constant, never tuned) is solved exactly by `y_s = 0` and counts as converged (1.30, card 12). `M` is SPD and block-diagonal over scopes, so `b_s = 0 ⇒ y_s = 0`. Below `ρ` the error is beneath fp32 resolution. The case arises in the backward solve of mini-batch runs, where a scope's `b_sᵀb_s` underflows in fp32 (PDNS). Full batch is one scope and never triggers the rule. The count is logged as `pcg_zero_rhs_scopes`.

**AD2 — §6 "WISP settings"**, append:
> (1.30, card 14) The WISP draw, cells, face groups, scopes, LP endpoint scopes and GC per-graph draws run on the model's device and are batched across targets, chunks and graphs.
> - Each draw uses a counter-based stream (SplitMix64), keyed by the existing seed functions (`training_draw_seed`, `graph_draw_seed`, the evaluation draw seed). A chunk or graph draws what it would draw alone.
> - The draw distribution, cells and scopes are those of 1.29. Only the random stream differs, so every H2S-NN number is re-run on the 1.30 freeze.

**AD3 — §6 "Link prediction"** and every list of derived tasks: replace `acm_pa` by **`acm_pp`** = HGB ACM `paper-cite-paper` (rel 0; exact reverse `paper-ref-paper`, rel 1), built like the other derived tasks.
- Add the reason:
  > acm_pa is not identifiable under F17. 47 % of validation-positive authors are isolated and featureless in the message graph. The degree-only AUROC is 0.684 on train and 0.448 on validation. GCN reaches 0.45. (Cards 11 A and 12 L, by a flag rule fixed before the check.)
- The schema line becomes "acm_pp ACM's paper paths".
- Fill the acm_pp heterophily values from the card 15 report.
- *If card 15 flags acm_pp:* drop ACM LP instead, keeping the same reason sentence.

**AD4 — §6 "Graph classification"**:
- Replace "Epochs (200 or 500): decided at training time" by "500 epochs, as HetSheaf's sweep files (1.30, card 13; early stopping with patience 50 ends most runs earlier)".
- Add:
  > All four datasets are kept (NCI1 included), with 100 trials as everywhere. GC runs after the main tables (priority). A trial stopped by the τ guard is a failed trial ("Failed trials"); the paper reports the failed-trial count per GC dataset. Cause, measured in card 11 C: τ grows at a rate set by `lr × 10` over ≈ 3,300 steps per epoch at batch size 1.

**AD5 — §6 "Evaluation protocol"**, append to the appendix-table examples:
> HGB's NC and LP runners early-stop and checkpoint on validation loss (e.g. `NC/benchmark/methods/GNN/run.py:127`), and HAN on loss and accuracy. H2GB selects the best epoch by its yaml's `metric_best` (accuracy, or binary F1 on PDNS and IEEE-CIS). Ours is the validation metric, unchanged (1.28 option 1, reaffirmed 2026-11-22).

**AD7 — §6 node classification, metrics (author 2026-11-22, card 16):** on the two-class H2GB datasets (IEEE-CIS-G, PDNS) every method we run also reports **binary F1 of class 1** (argmax of `C = 2` logits), and the main table shows binary F1 there, as H2GB's `metric_best: f1` is binary F1 on ≤ 2 classes (`H2GB/graphgym/logger.py:220-226`, `H2GB/logger.py:155`). This replaces spec 1.14 H4 ("H2GB's F1 is Macro-F1 on IEEE-CIS-G"), which was wrong. Selection stays validation Macro-F1 on the B1a subset; the difference goes in the appendix table.

**AD6 — header "Status"**: "frozen with `v3b` `<merged SHA>` (1.30)". The freeze covers this spec and all code that affects training or any reported number, as in 1.23.

## Not changed
- The method and its constants.
- Search spaces; 100 trials.
- Statistics.
- The ablation pre-registration v3.
- The NC protocol.
- The τ guard (50).
- `T_max` (100).
- The tolerance schedule.
