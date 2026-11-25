# Card 22 report: H2GB tuning failures on v3b `2e3e160`

Card `docs/H2S-NN_card22_card.md` (design-docs `15357f2`). Branch `card22` from v3b `2e3e160`, merge `15357f2` (`e03e89f`), code + tests `674901d`, A's second fix `1b03c39`. Not merged into `v3b`.

**Summary:** none of the three failures is a card 21 side effect. A (two bugs) and B are older bugs that card 21 did not touch; C is OOM at `num_hops` 4.

## A. Hetero²Net H2GB NC: device mismatch

- **Cause** (`hetero2net.py:571` and `:557` at `2e3e160`): `_fit_h2gb` keeps the graph on the CPU and never moves the target's `y` or `y_emb` to the device.
  - Their `main.py:105-108` puts `y` on the device before building the loaders, so `y_emb` (built from `y`) is on the device too.
  - Their step moves only `x` and `edge_index` (`vendor/Hetero2Net/hetero2net/utils.py:25`). So `embedding(y_emb)` (`models.py:114`) and `cross_entropy(y_pred, y_true)` (`utils.py:50`) mix cuda and cpu.
  - This does not depend on the worker count: it fails with `LOADER_WORKERS=0` too (test 1 on `2e3e160`, below).
  - Card 21's `num_workers` is not the cause, and no earlier GPU run of this path existed. The lab successes (acm_pp, dblp_ap) are the HGB path.
- **Fix 1** (`hetero2net.py`, `674901d`):
  - Training: `on_device` (`:568`) moves each batch's `y` and `y_emb` to the device in the main process before their step (`:579`).
  - Evaluation (`:557`): `batch.to(device, "x", "edge_index", "y_emb")`.
  - The graph stays on the CPU, because the loaders' workers sample it.
  - The values reaching the model are the same as in their `main.py`. On a CPU device the moves do nothing, so `LOADER_WORKERS=0` CPU results are unchanged. On GPU the old code could not run at all.
  - The large-LP path (`embed`, `:647`) reads no labels and is unchanged.
- **Second bug, found by test 1 once the device fix was in:**
  - `train_h2gb_epochs` → `_cpu_state(model)` (`minibatch.py:339`) detaches every state entry.
  - Their `EdgeDecoder.lin1 = Linear(-1, …)` (`models.py:14`) runs only with a metapath list and beta > 0, so without metapaths (every H2GB dataset but MAG) it stays lazy and the copy raises `ValueError: Attempted to use an uninitialized parameter`. An edge type's lazy conv (`models.py:49`) is in the same state until a batch holds that type.
  - On TSUBAME this was the next failure after the device error.
  - **Fix** (`hetero2net.py:604-613`, `train_h2gb_epochs` only):
    - The snapshot skips entries still lazy.
    - The restore (`:634`) uses `strict=False` and raises unless the only missing keys are the ones lazy at the snapshot, with no unexpected key.
    - A model without lazy parameters gets `_cpu_state` and a strict load exactly as before (`test_card16_epoch_budget`, `test_b1a_validation_subset` OK).
- **Test 1** (`_fit_h2gb` itself on GPU, `LOADER_WORKERS` 2 and 0; data, plan and split helpers patched):
  - **On `2e3e160`** (job 18886, RTX 2080 Ti, same test file in a temporary worktree): **ERROR** at 2 and 0 workers, `RuntimeError: Expected all tensors to be on the same device … cuda:0 and cpu … wrapper_CUDA__index_select`, the TSUBAME error.
  - **On `674901d`** (device fix only): ERROR at 2 and 0 workers, the lazy-parameter `ValueError` above.
  - **On `1b03c39`'s code** (job 18908; log `job18908_tests_fix2.log`): **OK** at 2 and 0 workers. The same job ran tests 1–2 plus `test_card16_epoch_budget` and `test_b1a_validation_subset`: **11 tests OK**, `WANDB_MODE` unset.

## B. CSNN / NSD IEEE-CIS-G NC: `'NodeStorage' object has no attribute 'x'`

- **Cause** (`minibatch.py:503` at `2e3e160`):
  - `hgb_homogeneous_batch` built `[batch[t].x for t in node_types]`.
  - In H2GB's IEEE-CIS-G only `transaction` has `x` (`vendor/H2GB/H2GB/datasets/ieee_cis_dataset.py:216`); the other types carry `num_nodes` only (`:223`). So every batch raised.
  - `fit_homogeneous_on_hgt` (`:644`) discards that list and takes its inputs from `batch_type_inputs`, which already gives a featureless type its one-hot rows.
  - IEEE-CIS-G is the first H2GB dataset on this path with featureless types, so the path never worked there.
  - Not card 21: `eval_loader` and `val_subset_loader` with workers return `_PassLoader`, which creates a fresh iterator each pass (card 21 Amendment 1.2), so data set after the build is seen. Not card 17's data path either.
- **Fix** (`minibatch.py:504`): `batch[t].get("x")`, which is None for a featureless type (docstring says so). Every type that has `x` is unchanged.
  - The other caller, `large_lp_training.py:242`, also discards the list, so a featureless type no longer crashes there either.
- **Test 2** (CPU, 2 workers):
  - Setup: an IEEE-CIS-G-shaped graph (`transaction` with `x`; `card` and `addr` featureless).
  - Check 1: the real `fit_homogeneous_on_hgt` runs to a record.
  - Check 2 (the card's case): on training and evaluation loaders built before the target's `x` is set, `hgb_homogeneous_batch` gives the late `x` for the target and None for `card`; `batch_type_inputs` gives `card`'s one-hot rows.
  - Job 18888 (lab, CPU, 16G; log `docs/card22_evidence/job18888_test2_cpu.log`): **OK** on the fix. On `2e3e160` (same test file, temporary worktree) it gives **ERROR** `AttributeError: 'NodeStorage' object has no attribute 'x'`, the TSUBAME error.

## C. SeHGNN IEEE-CIS-G NC: killed at the host limit

- **Code:** SeHGNN's H2GB path (`sehgnn.py:408` `_fit_h2gb` → `minibatch.train_index_minibatch`, precomputed features) uses no DataLoader, `num_workers` or `pin_memory`. Card 21 did not touch it, so `H2SNN_LOADER_WORKERS` cannot change its memory. The memory is SeHGNN's propagation (`_sehgnn_h2gb.propagate`): one tensor per metapath up to `num_hops`, with `embed_size`-wide random inputs for the featureless IEEE-CIS-G types.
- **Pattern (ops, W&B sweep `cj1c2xop`):** all 21 killed attempts have `num_hops` 4; all 15 successful ones have 2 or 3, at `embed_size` 256 and 512 alike.
- **Measurement:** job 18887, lab g3an01, CPU, `--mem 180G`, script `docs/card22_evidence/c_sehgnn_memory.py`, log `job18887_c_sehgnn_memory.log`.
  - Configuration: killed attempt 3 (`num_hops` 4, `embed_size` 256, `hidden` 256, `batch_size` 256), the adapter's real `_fit_h2gb`, the plan cut to 1 epoch.
  - | `H2SNN_LOADER_WORKERS` | result |
    |---|---|
    | 6 | OOM-killed at the 180 GB job limit (exit 137) |
    | 0 | OOM-killed at the 180 GB job limit (exit 137) |
  - sacct: 2 oom_kill events, MaxRSS 188,745,220 KiB (180 GiB). Both runs died during the prelude (about 2 min each), before training.
- **Recommendation:** OOM at that grid point (`num_hops` 4 needs more than 180 GB, against 96 GB on gpu_1). Workers play no part. No code change, and no per-cell `H2SNN_LOADER_WORKERS` value is needed.

## For ops

- **Reset and rerun after the merge:**
  - Hetero²Net oag-cs and IEEE-CIS-G NC.
  - The held Hetero²Net H2GB NC tasks (oag-eng, pdns, freebase, h-pokec): every H2GB NC dataset goes through `_fit_h2gb`.
  - CSNN and NSD IEEE-CIS-G NC.
- **Does not need its own reset:** the held h-pokec_uu LP task. It uses the large-LP path, which is unchanged, and can run on either head.
- **SeHGNN IEEE-CIS-G:** no reset needed for C; its killed trials are OOM trials.
- **H2GB cells that succeeded on `2e3e160`:** unaffected. The Hetero²Net change touches only Hetero²Net H2GB NC, which had no success. The `hgb_homogeneous_batch` change alters nothing for a type that has `x`.
- **CSNN oag-cs** (1 failure so far): every oag-cs type has `x` in H2GB, so B does not explain it. Its log was not checked here (TSUBAME).
- **Full suite:** not run (per card); ops runs it at review.

branch=card22 sha=1b03c398100a5ea33a975bcbdfe58be43d846c02 card=docs/H2S-NN_card22_card.md base=2e3e160712953356596c9d4a786b8f1f1d1ba4aa
