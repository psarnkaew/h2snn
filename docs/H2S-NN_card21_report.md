# Card 21 report: loader workers, the H-Pokec unequal-hop fix, HINormer cache, ChebNetII eval bases

Card `docs/H2S-NN_card21_card.md` (design-docs `ba3ecd8`, with Amendments 1 and 2; Amendment 3 from `966230e`). Branch `card21` from `origin/v3b` `f1fc158`, merge `ba3ecd8` (`b7aff39`), code + test `4d79a11`; merge `966230e` (`0181cf5`), Amendment 3 code + test `5bb3a81`. Not merged into `v3b`.

## Diff (`git diff origin/v3b...card21 -- H2S-NN/src H2S-NN/tests`)

| file | change |
|---|---|
| `src/baselines/minibatch.py` (+72/−8) | `LOADER_WORKERS` (default 6, `H2SNN_LOADER_WORKERS` int ≥ 0, read at import). `equal_hop_fanout`. `hgt_loader` / `eval_loader` / `val_subset_loader`: `num_workers=None` → `LOADER_WORKERS`. Fan-out goes through `equal_hop_fanout`. `persistent_workers = workers > 0 and split == "train"`. `pin_memory = workers > 0`. Val/test loaders with workers are returned as `_PassLoader` (see below). |
| `src/baselines/adapters/_hetero2net_h2gb.py` (+9/−2) | A2: `NeighborLoader`s get `LOADER_WORKERS`. The training loader's workers persist; the evaluation and subset loaders' workers are not persistent. |
| `src/baselines/large_lp_training.py` (+15/−5) | A2: `Sampler.endpoints` (evaluation) gets `LOADER_WORKERS`, not persistent. B2: `_hgt` pads the fan-out with `equal_hop_fanout`. |
| `src/baselines/adapters/hinormer.py` (+47/−2) | C1: `sequence_cache_key`, `cached_node_sequences`. The H2GB path calls the cache and records `sequence_cache` (hit / miss / off). |
| `src/baselines/adapters/_chebnetii_h2gb.py` (+38/−1) | C2: `ForwardBases` caches the target block's bases for one evaluation pass. |
| `tests/test_card21_loader.py` (new) | Tests 1–5. |

`large_lp_minibatch.py` is unchanged; see A2. `fanout_from_args` and `h2gb_sample_sizes` are unchanged (Amendment 1.1).

**Callers that pass `num_workers=0` explicitly:** none in `src/`. No caller passes `num_workers` at all, so every one now runs with `LOADER_WORKERS`.

**Eval workers exist only during a pass (Amendment 1.2).**
- H2GB's `LoaderWrapper` calls `iter(dataloader)` in `__init__` and again at the end of every val/test pass. With plain non-persistent workers, each evaluation loader would therefore keep 6 live workers between passes.
- `_PassLoader` fixes this. It keeps the `HGTLoader` (`.loader`, which `_h2gformer_h2gb.py:126` reads) and drops H2GB's eager iterator. Each pass then creates a fresh iterator, yields `min(step, len)` batches and deletes the iterator.
- The batches are the same as with H2GB's wrapper.
- With 0 workers the H2GB wrapper is returned unchanged, so `H2SNN_LOADER_WORKERS=0` gives the old code path.

**Peak process count of one run (derived from the code, not measured):** 1 main process + 6 persistent training workers + 6 workers of the one evaluation loader in use = **13 processes (12 workers)**. Before the change it was 1. Evaluation loaders run one after another, so no more than one set of eval workers is alive at a time.

## A2: per loader

| loader | workers | why |
|---|---|---|
| `_hetero2net_h2gb.loaders` train `NeighborLoader` | `LOADER_WORKERS`, persistent | Iterated across epochs by `epoch_batches`. |
| `_hetero2net_h2gb.loaders` val/test, `index_loader` (B1a subset) | `LOADER_WORKERS`, not persistent | Fully iterated once per evaluation. |
| `large_lp_training.Sampler.endpoints` (`NeighborLoader` / `_hgt`) | `LOADER_WORKERS`, not persistent | Fully iterated once per scored link set. |
| `large_lp_training.Sampler.step` (`link_loader` → `large_lp_minibatch.py:70` `LinkNeighborLoader`; `_hgt`) | **left at 0** | The loader is rebuilt every training step and only `next(iter(...))` is read. Workers would start 6 processes per step for one batch. |

## A3: determinism

- `torch.manual_seed` (through `seed_everything`) before the iterator is created already fixes the batches. No extra `torch.Generator` is passed.
- When the iterator is created, torch draws the shuffle order and the worker base seed from the default generator. Worker *i* runs `torch.manual_seed(base + i)`. torch_sparse's `hgt_sample` and PyG's samplers draw from that worker's default generator, and batches are dealt to workers round-robin and returned in order.
- Test 3 confirms it: same seed with 2 workers → identical first-batch node ids for every type. A 2-worker evaluation loader also gives the 0-worker seed order on two passes.
- **Expected difference:** batches with `LOADER_WORKERS > 0` differ from `LOADER_WORKERS = 0`. Sampling no longer draws from the main process generator, so the main process's later draws (dropout and similar) shift too. Runs are reproducible only for the same `LOADER_WORKERS`.

## B1: evidence (job 18852, `docs/card21_evidence/b1_probe.py`, log `job18852_b1_probe.log`)

- **Setup:** PyG `HGTLoader` (PyG 2.8.0, torch_sparse 0.6.18) on a synthetic graph, `user` [16, 16, 8, 8] and `item` [4, 4, 4]. 10 seeds × 3 passes = 900 batches per setting, run twice.
- **Unpadded:** `item` nodes per batch ranged over 12–187 (run 1) and 12–190 (run 2). 882 and 867 of the 900 batches held more than 12 items, which is more than 3 hops × 4 can give.
  - So hop 4 sampled `item` with a size read past the end of the 3-element list.
  - The value changes from batch to batch and from run to run, so the read is undefined.
  - No `n_sample <= 0` raise occurred in these runs; a garbage value ≤ 0 gives the raise seen on TSUBAME.
- **Padded (`equal_hop_fanout`):** exactly 16 items in every batch (4 hops × 4), 0 raises, both runs.
- The B1 stop condition is not met: the undefined read reproduces.

## B2: datasets with unequal hop lengths

- **Only H-Pokec.** Among the H2GB NC yamls of the methods used (GCN, GAT, HAN, H2Gformer, RGCN), only `Pokec-*.yaml` sets `neighbor_sizes_dict`: `user` [2048, 2048, 512, 512] and the 15 other types [128, 128, 128] → now [128, 128, 128, 128].
- Every other dataset's yaml uses one `neighbor_sizes` list for all types, so its lists are equal-length and unchanged (`equal_hop_fanout` returns the input object).
- Baseline fix, not a deviation: H-Pokec's 3-hop types are padded to 4 hops by repeating their last size, because PyG/torch_sparse otherwise read an undefined size.

## C1: HINormer (job 18856, `c1_hpokec_timing.py`, log `job18856_c1_hpokec.log`)

- **Measured on H-Pokec** (1,731,977 nodes, 51,774,836 edges, 1,632,803 targets, `len_seq` 50, CPU), with the adapter's prelude: one `build_node_sequences_for` call takes **704.8 s**. A cache hit, read from a 327 MB file, takes **4.6 s**.
- **Random state at the call:** two fresh processes with seed 0 had the same `random.getstate()` at the call (hash `58ae6fb4…`) and the same cache key.
  - The second process hit the cache: same sequences (sha `68262de7…`), same post-call state (`6677638…`), same first `random.random()` afterwards (0.6704438895754976).
  - So trials that share a seed share the state, and **the cache applies**.
- **Model-session addition:** the state after the call is stored with the sequences and `random.setstate()` is applied on every hit. Test 5 checks that the first `random.random()` after the call is equal with no cache, on a miss and on a hit.
- **Correctness by construction:** the key holds dataset, edges, target ids, `len_seq` and `random.getstate()`. A different state is a miss.
- **Where the cache lives:** the adapter has no access to the run's state directory, and passing it in would touch files outside the allowed list. The cache is therefore **off unless `H2SNN_HINORMER_SEQ_CACHE` is set**. Ops: set it to `<state_dir>/hinormer_seq_cache` in the job wrapper.
- **Memory:** host maxrss of this prelude on H-Pokec was 33 GB.
- **Job-rule departure:** the 16 GB rule could not hold this measurement. Job 18855 was OOM-killed at 16 GB while building the homogeneous view, so 18856 ran with `--mem 64G --time 2:00:00`.

## C2: ChebNetII

- **When the cache is used:** in eval mode without grad, `ForwardBases` computes all bases once and keeps the rows of the type block that holds the requested rows (the target type).
- **What it is keyed on:** the projection parameters' `(data_ptr, _version)`, the Laplacian and the feature tensors. Optimizer steps and the B1a `load_state_dict` bump `_version`, so they invalidate it. Training mode and grad mode clear the cache and recompute every forward, as before.
- **Test 4:** the logits of an evaluation pass equal the old per-batch recomputation within atol 1e-6, before and after a training step. Equality held, so nothing is reverted.
- **Memory added for one pass:** (K+1) × N_target × width × 4 B. H-Pokec worst grid point (K 10, width 256): about 18 GB on the device.

## C3: PSHGCN unchanged, as amended.

## Tests

- **Job 18854** (lab, CPU, 16G, 4 cores; tree = `4d79a11`): `python -m unittest -v tests.test_card21_loader` → **5 tests OK**.
  - The log shows four ignored `_MultiProcessingDataLoaderIter.__del__` "can only test a child process" tracebacks. They come from a forked worker garbage-collecting a loader iterator left over from an earlier loader in test 3 (a known PyTorch message, not a failure).
- **Not run:** the full suite (per card).
  - Every existing test that builds `hgt_loader` / `eval_loader` now runs with 6 workers by default.
  - None of them compares against stored bit-exact batches (checked by grep). Tests that compare an `hgt_loader` path with a direct 0-worker `HGTLoader` would now see different batches. Ops' full suite will show this; `H2SNN_LOADER_WORKERS=0` restores the old path.
- **Rule note:** job 18852, the first run of the same file, had 2 errors from a misplaced edit (fixed before 18854). Its B1 probe output is the evidence above.

## For ops (review)

- Amendment 1.4: every H-Pokec NC line already run with the unequal dict is void.
- Amendment 1.5: measure host maxrss with 6 workers on TSUBAME.
- Set `H2SNN_HINORMER_SEQ_CACHE` for HINormer H2GB cells.

## Amendment 3: the training loader starts its workers at the first batch

**Fix (`minibatch.py` only)**
- `_drop_eager_iterator`, used for both training and `_PassLoader`, undoes H2GB's `iter(dataloader)` in `LoaderWrapper.__init__`. It sets `iter_loader = None`, clears the DataLoader's cached `_iterator` and calls `_shutdown_workers()` on the dropped iterator.
- With workers, the training wrapper's `iter_loader` becomes `_LazyIterator`, which calls `iter(loader)` at the first `__next__`. So no worker exists when `hgt_loader` returns.
- H2GB's wrapper code is unchanged: `step` batches per pass, the iterator continued across passes and replaced by `iter(loader)` when exhausted. Persistent workers, once started at the first batch, are reused on restarts as before.
- With 0 workers the path is unchanged.

**Hetero2Net:** its training `NeighborLoader` is already lazy. `epoch_batches` calls `iter(loader)` at the first batch (`state["it"]` is missing), and `loaders()` creates no iterator. No change.

**Test 6** (2 workers)
- Setup: training and evaluation loaders are built; then `data["user"].late` is added and `data["user"].y` is replaced.
- Check: the first training batch and the first evaluation batch must carry `late` and the new `y` at their `n_id`.
- Job 18860, log `docs/card21_evidence/job18860_amendment3.log`:
  - **On `4d79a11`**, the new test file in a temporary worktree: **FAIL**, `'late' not found` in the training batch (the training loader is checked first).
  - **On the fix** (tree = `5bb3a81`): the whole file, **6 tests OK**.

No run used card 21 code, so nothing is void.

branch=card21 sha=5bb3a81b6ccdbeb44919f95bae1831d45f44fbb8 card=docs/H2S-NN_card21_card.md base=f1fc158409ad7d122ee67c42c897208282b9fd65
