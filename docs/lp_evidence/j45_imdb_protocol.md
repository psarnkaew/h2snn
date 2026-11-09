# J4.5 IMDB protocol checks (spec v1.8 §6, F6): record (2026-11-09)
Read-only checks: no model, trainer or adapter code was changed. The file name sits under `lp_evidence/` with the other J4 records; the subject is node classification.

## Test (`tests/test_imdb_protocol.py`, VERIFIED-RUN: `Ran 6 tests … OK`)
- **Real HGB IMDB** through `build_dataset` (`src/data/dataset.py`), the pipeline used by H2S-NN and by every baseline (`src/baselines/bundles.py:376-385`):
  - the **359 unlabelled movies** (all-zero label rows) are in no train, validation or test mask;
  - the masks cover exactly the 4,573 labelled movies;
  - train + validation = 1,371 (`label.dat`), test = 3,202 (`label.dat.test`), validation share 0.2 of `label.dat` (HGB's 24/6/70).
- **H2S-NN trainer** (`src/trainers/trainer.py:242-273`, `_compute_metrics`), on a fixture where an unlabelled row has logits of +50:
  - the loss equals BCE over the masked (labelled) rows only;
  - macro- and micro-F1 equal sklearn's on `logit > 0` predictions of the masked rows;
  - the default rule differs from oracle top-k on the fixture.
- **Shared baseline metric** (`src/baselines/metrics.py`): `DEFAULT_MULTILABEL_PROTOCOL == "threshold"`, and `classification_metrics` equals the `logit > 0` F1.
- **Mutations** (`j45_mutation_check.py` → `j45_mutation_output.txt`): making `oracle_topk` the default in `src/utils/misc.py` or in `src/baselines/metrics.py` is **caught (2/2)**; the files are restored byte-identical.

## Inventory: every node-classification baseline adapter (VERIFIED-CODE)
| Adapter | Loss (multi-label) | Prediction rule and metric | Checkpoint selection |
|---|---|---|---|
| chebnetii, csnn, h2gformer, h2sgnn, hgb_gnn, hgb_han, hinormer, nsd, pshgcn | `node_classification_criterion` → `BCEWithLogitsLoss` (`training.py:133-145`), on `train_idx` only (`training.py:85`) | `train_full_batch_node_classification` (`training.py:47-130`) → `classification_metrics(…, multilabel_protocol=DEFAULT…)` (`:91-102`) → `threshold` (`metrics.py:39-40`) | validation macro-F1 (`training.py:104-111`) |
| hetero2net | vendor `train_full_batch`: `F.binary_cross_entropy_with_logits` on the canonical train mask (`vendor/Hetero2Net/hetero2net/utils.py:154-157`; mask from `_apply_canonical_split`, `adapters/hetero2net.py:110-112`) | `classification_metrics` with the default (`adapters/hetero2net.py:294-303`) → `threshold`. Their own rule is also `y_pred > 0` (`hetero2net/utils.py:240`) | validation macro-F1 (`:305`) |
| sehgnn | `BCEWithLogitsLoss` (`adapters/sehgnn.py:261-265`), on training batches (`:299`) | `classification_metrics` default (`:281-283`) → `threshold`. Their own rule is `output > 0` (adapter docstring `:25`) | validation macro-F1 (`:305`) |
| latgrl | probe `BCEWithLogitsLoss` (`adapters/latgrl.py:159`), on `train_idx` (`:170`) | `classification_metrics` default (`:176-185`) → `threshold`. Their own `evaluate` uses argmax with cross-entropy; the adapter replaced it (docstring `:25-27`) | validation macro-F1 (`:179`) |

**No adapter uses top-k or a metric of its own.** `oracle_topk` exists only as an option in `src/baselines/metrics.py:43-67` and `src/utils/misc.py:123-126`. No caller in `src/`, `scripts/` or `configs/` passes it (`grep`). **Nothing to report as non-compliant.**
- **Label channel (Hetero²Net `--mask_lp`).** Unlabelled movies get the "no label" token, test labels are hidden, and the training copy also hides validation labels (`adapters/hetero2net.py:115-137`). No label of an unlabelled or held-out movie enters as input.

**Not checked here (out of J4.5):**
- Whether every adapter's validation **macro-F1** is the paired-test statistic. That belongs to the protocol runner, which records `val_macro_f1` through `default_monitor_for_task` = `macro_f1` (`src/utils/run_reports.py:13-15`).
- HetSheaf's own code, which is not an adapter here (spec §6: rerun on HGB's split).
