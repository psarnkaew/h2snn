# Card 22: H2GB tuning failures on v3b `2e3e160` (Hetero²Net device mismatch, CSNN/NSD IEEE-CIS-G, SeHGNN host memory)

The ops session wrote this card on 2026-11-25, from the first TSUBAME tuning legs that ran on v3b `2e3e160` (card 21 merged). The author asked for it ("เขียน card ให้ model session เลย", write the card for the model session).

Report: `docs/H2S-NN_card22_report.md`.

All evidence comes from TSUBAME gpu_1 (H100 94 GB, 96 GB host, 16 cores), job 9553079 (`cells_tuning_b2_t4_gpu1.leg.202611250127.txt`). Every log is under `logs/lines/` of the `0b9873d` out directory.

## Why

### A. Hetero²Net on H2GB NC: every trial fails with a CPU/GPU device mismatch (VERIFIED-RUN, likely card 21)
- **Failed sweeps:** oag-cs (task 13) and IEEE-CIS-G (task 4), 40 of 40 trials each. The sweep state now records 0 successful and 40 failed trials, so both cells are closed as failed.
- **Two errors, both from the vendor training step** (`vendor/Hetero2Net/hetero2net/utils.py`):
  - `utils.py:42` → `models.py:114` `self.embedding(y_emb)`: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (… argument index in method wrapper_CUDA__index_select)`. 44 times in the oag-cs log.
  - `utils.py:50` `F.cross_entropy(y_pred, y_true)`: the same error, raised from `wrapper_CUDA_nll_loss_forward`. 36 times.
  - Call path: `hetero2net.py:250` `fit_task` → `:576` `_fit_h2gb` → `:596` `train_h2gb_epochs` → `:571` `train_epoch` → `train_mini_batch(... h2gb.epoch_batches(train_loader, ...))`.
- **Why card 21 is the likely cause:** card 21 A2 changed only the loader construction in `_hetero2net_h2gb.py`, `loaders()` and `index_loader()`:
  - it added `num_workers=LOADER_WORKERS`;
  - it added `persistent_workers=shuffle and LOADER_WORKERS > 0`.
  - With 6 workers, the batch tensors the vendor step reads (`y_emb`, `y`) are no longer on the device it expects.
  - Where these tensors are moved to the device was not checked. Find that (`train_mini_batch`, `label_channel`, or the `copy(data)` at `hetero2net.py:~541`).
- **Not affected:** Hetero²Net on HGB data uses another path and runs fine. Lab sweeps on old code `0b9873d`: acm_pp 76/76 trials, dblp_ap 96/96.
- **Held by ops** (pending, not started): the remaining Hetero²Net H2GB tuning tasks.
  - 9553079.22: oag-eng.
  - 9553079.38: pdns.
  - 9553082.6, .14, .23: h-pokec_uu LP, freebase NC, h-pokec NC.

### B. CSNN and NSD on IEEE-CIS-G NC: `'NodeStorage' object has no attribute 'x'` (VERIFIED-RUN, cause not known)
- **Failed sweeps:** CSNN 0 successful of 39 attempted (task 2); NSD 0 of 22 (task 7).
- **Error:** `minibatch.py:651` `fit_homogeneous_on_hgt` → `:449` `train_minibatch_node_classification` → `:644` `forward_seeds` → `:503` `hgb_homogeneous_batch` → `[batch[t].x for t in node_types]`. Some IEEE-CIS-G node type in the batch has no `x`.
- **Same datasets, other models:** CSNN oag-cs has 1 success and 1 failure so far. Other methods on IEEE-CIS-G through other paths succeed (ChebNetII 19/19, PSHGCN 26/26, H2SGNN 21/21).
- **Possible causes:**
  - Card 17 gives IEEE-CIS-G `dim_in None`, so a node type may get its features only after the loader is built.
  - Card 21's worker processes may then miss them: Amendment 3 fixed this only for the training iterator, not for `eval_loader` / `val_subset_loader` passes.
  - Or this path never worked on IEEE-CIS-G: no CSNN/NSD IEEE-CIS-G trial ever ran on old code.
  - Say which.

### C. SeHGNN on IEEE-CIS-G NC: workers killed by host memory (VERIFIED-RUN, cause not known)
- **Failures:** 21 of 36 attempted trials ended with `worker exited with return code -9 before recording a trial` (task 9); 15 succeeded.
- **Memory:** the task's host memory peaked at **maxrss 95.6 GB** (maxvmem 123 GB), at the 96 GB host limit of gpu_1.
- **Open question:** do the failing grid points exceed 96 GB alone, or only with card 21's 6 loader workers (extra processes, pin memory)?
  - If alone, these are OOM trials under the existing rule; no code change needed.
  - If the workers cause it, it is a card 21 side effect to fix.

## Fix
- **A1.** Find where the device mismatch comes from with `LOADER_WORKERS > 0`, and fix it in the adapter (`hetero2net.py` / `_hetero2net_h2gb.py`), not in `vendor/`.
  - Results with `LOADER_WORKERS=0` must stay the same as at `2e3e160`.
  - If the fix needs the loader at 0 workers for Hetero²Net, say so and why.
- **B1.** Find which IEEE-CIS-G node type lacks `x` in the batch, and why. Fix it in `minibatch.py` or the CSNN/NSD adapters.
  - If the cause is card 21 (workers forked before the features exist), fix it for every loader that card 21 gave workers.
  - If the cause is card 17's data path, fix it there and name the file. That file is added to the allowed list only with a note in the report.
- **C1.** On one SeHGNN IEEE-CIS-G configuration that was killed, measure peak host RSS with `H2SNN_LOADER_WORKERS=6` and with `0`.
  - Attempt indices 1–4, 7, 10, … are in `state/sehgnn_node_classification_ieee-cis-g/tuning_state_run0.json`; their parameters are in the W&B sweep.
  - If 6 workers add more than about 10 GB, reduce it in the loader (no pin memory or fewer workers for that path), or say which per-cell `H2SNN_LOADER_WORKERS` value ops should use.
  - If not, report "OOM at that grid point"; no code change.

## Tests (only these; minimal)
1. **A:** one Hetero²Net H2GB NC training step plus one evaluation pass on GPU with `LOADER_WORKERS=2` on a small H2GB-shaped graph. No device error.
2. **B:** the IEEE-CIS-G-shaped case: a node type whose `x` is set after the loader is built. `hgb_homogeneous_batch` gets `x` for every type with workers > 0.
3. **C:** none. Measurement only, in the report.

The full suite runs once at review (ops). Run it with `scripts/slurm/run_tests.sbatch` settings and without `WANDB_MODE`.

## Preconditions
- Branch `card22` from v3b `2e3e160`.
- Run on the lab cluster: GPU tests through `scripts/slurm/run_tests.sbatch`, with `--mem` and `--time` set.
- TSUBAME only through ops.

## Rules
- **Allowed files:**
  - `src/baselines/adapters/hetero2net.py`
  - `src/baselines/adapters/_hetero2net_h2gb.py`
  - `src/baselines/minibatch.py`
  - `src/baselines/adapters/csnn.py`
  - `src/baselines/adapters/nsd.py`
  - `src/baselines/adapters/sehgnn.py`
  - new `tests/test_card22_h2gb_failures.py`
  - Anything else: stop and ask.
- No change to `vendor/`, to search spaces, or to budgets.
- `LOADER_WORKERS=0` behaviour stays the same as at `2e3e160`.
- No sweeps or result directories are touched. Ops resets the failed sweep states after the merge, with the author's approval.

## Report (`docs/H2S-NN_card22_report.md`, short)
- **A, B, C:** the cause with file:line, the fix, and the test result (job id).
- **C:** peak RSS with 6 and 0 workers, and the recommendation.
- **For ops:** which cells need their sweep state reset and rerun:
  - Hetero²Net oag-cs and IEEE-CIS-G NC;
  - CSNN and NSD IEEE-CIS-G NC;
  - SeHGNN IEEE-CIS-G NC if C changes anything.
  - Also say whether any H2GB cell that succeeded on `2e3e160` is affected.

## Estimate
- 0.5–1 day: A about 2–3 h, B about 2–3 h, C about 1–2 h including a lab memory run.
