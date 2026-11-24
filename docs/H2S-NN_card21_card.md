# Card 21 — H2GB loader: worker processes, and the H-Pokec unequal-hop sampler crash

Written 2026-11-24 by the ops session on its measurement of the same day. The author approved writing it ("write the card and fix the H-Pokec sampler crash"). Report: `docs/H2S-NN_card21_report.md`.

## Why

### A. The H2GB loader runs with no worker processes (VERIFIED-RUN, TSUBAME gpu_1, H100, 16 cores, code `0b9873d`)
- `hgt_loader` (`src/baselines/minibatch.py:183-213`) already takes `num_workers` and sets H2GB's `cfg.num_workers` / `cfg.train.persistent_workers`.
  - Every caller leaves the default `0`: `h2gb_published.py:301-304`, `hgb_han.py:451-454`, `rgcn.py:279-282`, `h2sgnn.py:493-501`, `hetsheaf.py:786-790`, `_h2gformer_h2gb.py:190-194`, `minibatch.py:555-561`, and `eval_loader` / `val_subset_loader` through them.
  - `cfg.train.pin_memory` is forced `False`.
  - H2GB's own yamls use 16–18 workers with persistent workers and pin_memory. `h2gb_published.py:55` drops them as "hardware".
- Measured: HGT sampling only, GCN published configuration (batch 32, 256 batches per epoch), 100 batches after 5 warm-up batches (ops job 9550269):

  | dataset | workers 0 | 4 | 6 |
  |---|---|---|---|
  | PDNS | 95 ms/batch | 24 | 19 |
  | oag-eng | 73 ms/batch | 19 | 15 |

- A real run: PDNS GCN published seed 3 (job 9530256 task 25) took 3.9 h for 500 epochs, about 28 s/epoch. About 24 s of that is sampling. Peak GPU memory was 0.9 GB.
- So the H2GB-dataset baselines are bound by CPU sampling. With 6 workers, sampling (about 5 s/epoch) runs alongside the GPU step.

### B. H-Pokec crash: hop lists of unequal length (VERIFIED-RUN + VERIFIED-CODE)
- Published GCN, HAN and R-GCN on H-Pokec NC: all 11 seed lines of job 9533064 (tasks 1–5, 11–13, 16–18) exited 1 after 1–7 epochs:
  - `RuntimeError: cannot sample n_sample <= 0 samples` in `torch.ops.torch_sparse.hgt_sample`;
  - reached from `HGTLoader.collate_fn` → `sample_from_nodes`.
  - Example log: `logs/lines/cells_pub30s_nodef.line001.9533064.1.0.log`.
- The H-Pokec fan-out in every H2GB H-Pokec yaml (`configs/baselines/{gcn,han,rgcn,h2gformer}/node_classification/h-pokec.yaml`, `train.neighbor_sizes_dict`) gives `user` **4** hops (`[2048, 2048, 512, 512]`) and every other node type **3** (`[128, 128, 128]`).
- PyG 2.8.0 `HGTSampler` sets `num_hops = max(len(v) for v in num_samples.values())` = 4.
  - torch_sparse 0.6.18 then reads `num_samples[type][3]` for the 3-element lists: an unchecked read past the end of the list.
  - The value read is arbitrary. When it is ≤ 0 and that type has nodes in its budget at hop 4, the sampler raises.
  - That fits a crash at a random batch after some epochs. Card step B1 confirms it.
- The same dict sits in H2GB's yaml, so H2GB's own code has the same undefined read.
- Scope: every method on H-Pokec NC through `hgt_loader`, published and tuned, including H2G-former. Two H2G-former H-Pokec published seed lines are running now with the same dict.
- H2S-NN does not use this loader.

## Fix
A1. **Workers.**
- Add one module constant in `src/baselines/minibatch.py`, `LOADER_WORKERS`: default **6**, overridable by the environment variable `H2SNN_LOADER_WORKERS` (int ≥ 0), so ops can change it without a card.
- `hgt_loader`'s `num_workers` default becomes `None`, meaning `LOADER_WORKERS`. `eval_loader` and `val_subset_loader` do the same.
- Callers stay unchanged unless one passes `0` explicitly; list any that do.
- `cfg.train.persistent_workers = num_workers > 0` as now. Set `cfg.train.pin_memory = num_workers > 0`.

A2. **Other sampled loaders, same constant.**
- Pass `num_workers=LOADER_WORKERS` and `persistent_workers=LOADER_WORKERS > 0` where the loader is built:
  - `_hetero2net_h2gb.py:99, 111` (`NeighborLoader`);
  - `large_lp_training.py:67, 75` (`NeighborLoader` / `HGTLoader`);
  - `large_lp_minibatch.py:70` (`LinkNeighborLoader`).
- If a loader there is iterated in a way that breaks with workers (for example a generator or closure that cannot be pickled, or a loader rebuilt every step), leave it at 0 and say which one and why.

A3. **Determinism.**
- With workers, a seeded run must still be reproducible. Two runs with the same seed and the same `LOADER_WORKERS` must give the same batches.
- Pass a `torch.Generator` seeded from the run seed to the loader, or confirm that `torch.manual_seed` before loader creation already fixes it. Say which.
- Batches will differ from `LOADER_WORKERS=0`. That is expected; state it in the report.

B1. **Confirm the cause.**
- On CPU, on a small synthetic heterogeneous graph, build `HGTLoader` with one type at 4 hops and another at 3. Show that hop 4 reads an undefined size: either the raise, or a sampled count for the 3-hop type at hop 4.
- Then show that the fix removes it.

B2. **Fix it in `fanout_from_args`** (`h2gb_published.py`; also any tuned path that builds an H-Pokec fan-out dict).
- Pad every shorter hop list to the longest length by repeating its last value. On H-Pokec that gives `[128, 128, 128, 128]` for the 15 non-user types.
- This keeps H2GB's 4-hop `user` sampling and gives the other types a defined size at hop 4.
- Datasets whose lists are already equal must be unchanged; check every H2GB NC yaml and list any other dataset with unequal lengths.
- This is a baseline fix, not a deviation. Record it in the report in one line.

## Tests (only these)
- **One CPU test file** `tests/test_card21_loader.py`:
  1. `LOADER_WORKERS` default 6 and the environment override.
  2. The unequal-hop graph of B1 crashes, or reads an undefined size, before padding and samples cleanly after it. Use 20 batches and a fixed seed.
  3. Same seed and `num_workers=2`: two loader passes give identical first-batch node ids.
- No GPU smoke and no full suite. Ops runs the full suite at review and measures the real speed-up on TSUBAME.

## Preconditions
- Worktree `/mnt/data/peeradon/h2snn-card21`, branch `card21` from `origin/v3b` `f1fc158409ad7d122ee67c42c897208282b9fd65`; `git merge` the design-docs SHA given in the prompt.
- Copy (do not symlink) the git-ignored `vendor/`; `data` → `/mnt/data/peeradon/h2snn/H2S-NN/data`.

## Rules
- `sbatch` only on the lab, CPU job `--mem 16G --time 1:00:00 --cpus-per-task 4`.
- `git diff origin/v3b...card21 -- H2S-NN/src` touches only:
  - `src/baselines/minibatch.py`, `src/baselines/h2gb_published.py`;
  - `src/baselines/adapters/_hetero2net_h2gb.py`;
  - `src/baselines/large_lp_training.py`, `src/baselines/large_lp_minibatch.py`.
- No change to models, losses, schedules or the B1a selection.
- Stage by explicit path; one commit for code + test, one for the report; push `card21`. Never merge into `v3b`; never stash, reset or rebase.
- Stop and report if B1 does not reproduce the undefined read, or if A3 cannot be made deterministic.

## Report (`docs/H2S-NN_card21_report.md`, short)
- The diff.
- A2: one line per loader, workers on or left at 0, and why.
- A3: how determinism is ensured.
- B1: the evidence.
- B2: the list of datasets with unequal hop lengths.
- The test result with job id.
- Last line `branch=card21 sha=<head> card=docs/H2S-NN_card21_card.md base=f1fc158409ad7d122ee67c42c897208282b9fd65`.

## Estimate
About 2–3 h, including the queue.

## Amendment 1 (model design session, author-approved 2026-11-24)
Review of the card against `v3b` `f1fc158`. These points replace the card text where they differ.

1. **B2: pad where the loader is built, not in `fanout_from_args`.**
   - `fanout_from_args` serves only the published path.
   - Tuned cells read the fan-out from `h2gb_sample_sizes`, which returns the same unequal H-Pokec lists. The call sites are `hgb_han.py:447`, `rgcn.py:275`, `h2sgnn.py:485`, `hetsheaf.py:775`, `_h2gformer_h2gb.py:196`, `minibatch.py:540` and `large_lp_training.py:217` (H-Pokec U–U LP).
   - Add one helper `equal_hop_fanout(sizes)` in `minibatch.py`. It pads every shorter list to the longest length by repeating the last value, and returns equal-length input unchanged.
   - Apply the helper in `hgt_loader`, before `cfg.train.neighbor_sizes_dict` is set, and where `large_lp_training.py` builds its `HGTLoader`. Every NC and LP source is then covered.
   - `fanout_from_args` and `h2gb_sample_sizes` stay unchanged.
   - `repeat_fanout` is unaffected: it already refuses lists that are not constant.
2. **Workers per run.**
   - A run keeps four loaders alive (train, val, test, B1a subset; e.g. `hgb_han.py:451-455`, `h2gb_published.py:301-304`). Six persistent workers on each would mean 24 processes on a 16-core node.
   - Use instead:
     - the train loader: `LOADER_WORKERS` workers, persistent;
     - `eval_loader` and `val_subset_loader`: `LOADER_WORKERS` workers, **not** persistent, so they exist only during an evaluation pass;
     - pin_memory: as A1.
   - The A2 loaders follow the same rule (training loaders persistent, evaluation loaders not).
   - Report the peak process count of one run.
3. **Test 2 checks the fix, not the crash.**
   - Reading past the end of a C++ vector is undefined behaviour, so the pre-padding crash may not reproduce on every run.
   - Test 2 asserts two things:
     - after `equal_hop_fanout`, all lists have the same length and keep their first values;
     - 20 batches sample without error on the B1 graph with a fixed seed.
   - The pre-padding crash or undefined read is **evidence for B1** in the report, not a test assertion. The B1 stop condition still applies to that evidence.
4. **Runs already made on H-Pokec NC with the unequal dict are void.** Their hop-4 sizes for non-`user` types were undefined. This covers the two H2G-former published seed lines now running and any H-Pokec NC line that finished.
   - Every H-Pokec NC cell, published and tuned, runs on the card 21 code.
   - Ops records which lines are void in the run register.
5. **Memory (ops, at review, not the agent).**
   - On TSUBAME, measure host maxrss with `LOADER_WORKERS = 6` for one H-Pokec NC run and one ogbn-mag NC run.
   - Compare it with the layer-1 estimates, which assume 0 workers.
   - If a cell would exceed 96 GB, set `H2SNN_LOADER_WORKERS` lower for that cell.
6. **Allowed files are unchanged.** The test file stays `tests/test_card21_loader.py`.

## Amendment 2 (ops session, author-approved 2026-11-24): two more slow spots; PSHGCN unchanged
These are speed-only changes. Any change to a model's outputs, its RNG draws or the B1a selection is out of scope. If a change cannot keep the results identical, leave that code as it is and report it.

C1. **HINormer: per-target Python loop** (`adapters/hinormer.py:100-121` `build_node_sequences_for`, called at `:296`).
- The loop runs over every target with `random.sample`: 1.63M users on H-Pokec, 546k papers on oag-cs, once per trial.
- Allowed: cache the result for a trial with the same (dataset, target ids, `len_seq`, Python `random` state at the call), in memory or on disk under the run's state directory.
- First verify that the `random` state at the call is the same across trials that share a seed. If it is not, the cache is not allowed. A vectorised rewrite is not allowed either, because it changes the draws.
- Report:
  - the measured time of one call on H-Pokec (CPU job, `len_seq` 50);
  - whether the cache applies.

C2. **ChebNetII: whole-graph bases per evaluation batch** (`_chebnetii_h2gb.py:164-190` `ForwardBases`, IEEE-CIS-G and H-Pokec).
- Training needs the recomputation, because the parameters change every step. Evaluation does not: the parameters are fixed for the whole pass.
- Compute the bases once per evaluation pass (val, B1a subset, test) under `torch.no_grad()`, then index the batch rows.
- Equality check, in the test file (test 4):
  - small synthetic graph, CPU;
  - the logits of one evaluation pass before and after the change;
  - equal within `atol 1e-6`.
- If equality fails, revert C2 and report.

C3. **PSHGCN on H-Pokec: no change.**
- The 2026-11-16 override (`search_grids.py:74-80`, `K = 1`, `hidden ≤ 64`) was written for the full-graph search space.
- H-Pokec is now in `H2GB_GENERIC_DATASETS` (`pshgcn.py:94`). Its decoupled mini-batch space (`pshgcn.py:184-190`, `K = 2` fixed, `hidden_x` 256–1024) returns before the override is applied at `:202`.
- That space is the intended one. The override has no effect, and removing it changes nothing. Leave both as they are.
- Ops probe (job 9545379, smallest grid point, 1 trial): GPU peak 0.9 GB, host maxrss 77 GB, 39 min.

Files: add `src/baselines/adapters/hinormer.py` and `src/baselines/adapters/_chebnetii_h2gb.py` to the allowed list. Tests: add test 4 (C2) to `tests/test_card21_loader.py`. Estimate: +1–2 h.

## Amendment 3 (model design session, after the report at `card21` `4a4335a`; 2026-11-24)
**Problem (VERIFIED-CODE).** H2GB's `LoaderWrapper.__init__` calls `iter(dataloader)` (`vendor/H2GB/H2GB/sampler/custom_sampler.py:26`), so the training loader's workers fork when `hgt_loader` returns.
- A persistent DataLoader also caches that iterator in `loader._iterator` and reuses it. The training workers therefore keep the graph as it was when the loader was built.
- Adapters change `data` after building their loaders. In `_h2gformer_h2gb.py`, after `hgt_loader(...)`:
  - `prepare_label_channel` replaces `y`, `train_mask` and `val_mask`;
  - `attach_h2gb_metapath` adds `pestat_Hetero_Metapath`;
  - `configure_from_yaml` sets H2GB's `cfg`.
- With 0 workers the sampler ran in the main process and saw these changes. With workers, the training batches miss them, while the evaluation passes (`_PassLoader`, forked at each pass) see them. Training and evaluation then read different graphs. This is the card 20 failure in another form.

**Fix (in `minibatch.py` only).**
- With `num_workers > 0`, `hgt_loader` returns a training loader that has **no live iterator and no worker process**.
  - Discard H2GB's eager iterator and shut its workers down.
  - Clear the DataLoader's cached `_iterator`.
  - Create the iterator at the first batch.
- H2GB's training semantics stay unchanged: `step` batches per pass, the iterator continued across passes, restarted when exhausted. For example, give the wrapper a lazy iterator object that calls `iter(self.loader)` on its first `__next__`.
- Check the same for `_hetero2net_h2gb.py`'s training `NeighborLoader`, and report it.

**Test 6** (add to `tests/test_card21_loader.py`, CPU, `num_workers = 2`).
- Build the training and evaluation loaders.
- Then add a new node attribute to `data`, and replace one existing tensor such as `y`.
- The first training batch and the first evaluation batch must both carry the new attribute and the new `y`.
- Also check that the test fails on `4d79a11` and passes on the fix; report both.

**Runs.** No run has used card 21 code (it is not merged), so nothing is void. Ops runs the full suite at review as planned.
