# Card 14 report: the H2S-NN data pipeline on the GPU

Card: `docs/H2S-NN_card14_card.md`, Amendment 2 (design-docs 47fe21f). Base: `66e9349`. Code commit: `fa68b4e` (branch `card14`).

## Summary

- The WISP draw, cells, face groups, scopes, LP endpoint scopes and GC draws now run on the graph's device. There is no `.cpu()` and no CPU generator left in the pipeline.
- The per-target loop of `build_scope_batch`, the per-graph loop of `build_graph_wisp_path_groups` and the per-chunk loop of the mini-batch draw are now batched tensor operations.
- Speed gate (≥ 5× per step):
  - **met** on mag-year (10.2×) and PDNS (12.4×);
  - **missed** on NCI1 (1.0×), where about 80 % of step time is in `AnchoredSolve` (card 12's file, not in my list);
  - ACM is not slower.
- Tests:
  - (a) equivalence tests pass on CPU and GPU;
  - (b) all four smokes pass;
  - (c) full suite: 986 tests, 2 failures. Both are bit-exact captures of the old random stream, which Amendment 2 says will change.

## What moved to the device

Each item below now runs on the device of the prepared graph.

| Part | File | Change |
|---|---|---|
| Random numbers | `path_sampler.py` | The per-chunk and per-graph CPU `torch.Generator` objects are replaced by a counter-based stream (SplitMix64 in int64 tensor operations). Walk `i` of the stream with seed `s` reads `uniform(key(s), i·C + c)`. The seeds are the existing ones (run seed, `training_draw_seed`, `chunk_seed`, `graph_draw_seed`), so the seed schedule is unchanged and only the random stream differs. CPU and GPU give identical draws (checked in test (a)). |
| Action graph | `wisp_sampler.py` | Built on the device and cached per prepared graph (`cached_action_graph`). The old code rebuilt it on every mini-batch step. |
| WISP draw | `wisp_sampler.py` | All tensors are created on the device. The adaptive proposal's `torch.multinomial` becomes an inverse-CDF lookup (`searchsorted`) on the same distribution. |
| Proposal | `wisp_sampler.py` | `proposal_from_mass` builds on the device of `node_type` instead of copying to the CPU. |
| Incidence index, scope batch | `scopes.py` | Built and cached on the device. |
| Mini-batch NC | `minibatch_trainer.py` | `fit` moves the training and evaluation graphs to the device once. Evaluation logits stay on the device until the end of the split. The per-step `float(loss)` becomes one read per epoch. |
| Large LP | `minibatch_lp.py` | The endpoint graph moves to the model's device on first use. Scopes are built there. |
| HGB LP full batch | `link_prediction_trainer.py` | Per-epoch WISP draws use the device copy of the message graph. The DFS baseline bundles still use the CPU copy. |
| GC | `graph_classification_trainer.py` | The draw runs on the batch's device. The per-batch `loss.item()` becomes one read per epoch. |

The model file was not changed: it had no host round trips in the encode path. The ACM full-batch draw is unchanged. It is built in `src/trainers/trainer.py` and `src/data/dataset.py` on the CPU copy, overlapped by the worker thread. Neither file is on the allowed list, so ACM only gets the device-aware sampler code without running it on the GPU.

## What was vectorised

| Loop (old) | Replacement |
|---|---|
| `build_scope_batch`, one iteration per target (`scope_members`, `_local`, `cells.of(t)`) | One pass. Each cell is joined to its scope(s) with `searchsorted` on the sorted targets, so a repeated target owns the cells twice, as before. Membership is `unique` over the keys `scope·E + fact` and `scope·N + node`. Every local index (batch edges, target rows, cell nodes, cell facts) is a `searchsorted` into those keys. The rows come out in the old order: scope 0 ascending, then scope 1, and so on. It takes `AnchorCells` directly; per-target lists are still accepted for the other callers. |
| Trainers' `[cells.of(int(t)) for t in targets.tolist()]` | Removed. `AnchorCells` is passed to `build_scope_batch`. |
| `sample_wisp_walks_chunked`, one call per chunk | One call. Chunk `k` reads the stream of `chunk_seed(seed, k)` at the anchor's index in the chunk. |
| `build_graph_wisp_path_groups`, one action graph and draw per graph | One action graph for the batch and one draw. Graph `g` reads the stream of `graph_draw_seed(draw_seed, id_g)` at the node's index in its graph, so a graph's cells still do not depend on its batch. The edge-contiguity check is gone because no fact offset is needed any more. |

The remaining Python loops run over hops (≤ ℓ_max), lengths, the visited half-edge columns (≤ ℓ_max·M), schemas and evaluation chunks of `EVAL_BATCH_TARGETS`.

## Speed table

- Setup: lab RTX 3090 (g3n01), 1 warm-up step, then 20 training steps, then one evaluation pass. Same GPU type for old and new.
- Script: `scripts/diagnostics/card14/bench.py`. It runs the trainer's own `fit` with timing wrappers.
- Old code: a detached worktree at `66e9349`. New code: the working tree whose `src/` was committed unchanged as `fa68b4e`.
- Configs: card 11 inputs (mag-year `dandy-sweep-1`, PDNS `balmy-sweep-7`, NCI1 `astral-sweep-1` at lr 3e-4) and ACM `acm_h2snn_v3b.yaml` (full batch).
- s/step is the mean over the 20 steps. GPU utilisation is the `nvidia-smi` mean over those steps.

| Config | s/step old → new | Speed-up | GPU util old → new | Peak GPU (GiB) old → new | Host peak (GB) old → new | Eval pass (s) old → new (draw part) | Jobs (old / new) |
|---|---|---|---|---|---|---|---|
| mag-year | 6.810 → **0.669** | **10.2×** ✅ | 2 % → 23 % | 0.36 → 7.77 | 9.8 → 6.1 | 56.4 (26.8) → 3.0 (0.47) | 18252 / 18260 |
| PDNS | 9.850 → **0.793** | **12.4×** ✅ | 1 % → 16 % | 0.48 → 11.97 | 14.3 → 5.9 | 19.0 (10.2) → 3.8 (0.13) | 18253 / 18261 |
| NCI1 | 0.134 → **0.136** | **1.0×** ❌ | 17 % → 19 % | 0.02 → 0.02 | 1.6 → 1.7 | 29.5 → 30.8 (validation loader) | 18254 / 18262 |
| ACM (full batch) | 1.004 → 0.999 | 1.0× (not slower) ✅ | 45 % → 46 % | 1.70 → 1.71 | 2.0 → 2.0 | 0.28 → 0.29 | 18255 / 18263 |

Peak GPU memory rises because the prepared graph, its action graph and the run's evaluation cells now stay on the GPU. The largest is PDNS at 12.0 GiB at this configuration (B = 32, r = 8, ℓ_max = 4). The H100 rule of spec 1.26 at the largest search point was not measured; this card asked for no tuning runs.

### Profile of what is left on NCI1 (gate missed)

Job 18265 ran `torch.profiler` (CPU + CUDA) over 11 training steps on the new code (`bench.py --profile`).

| Share of self CPU time | Item |
|---|---|
| ≈ 80 % (CPU total) | `AnchoredSolve` forward (0.95 s) + `AnchoredSolveBackward` (1.48 s) out of 3.0 s |
| 23 % | `cudaLaunchKernel`: ≈ 8,300 launches per step, 7.7 µs each |
| 27 % (spread) | `aten::mm`, `bmm`, `add`, `index_select`, `cat`, … under the solver; ≈ 1,600 `mm` per step |

- Self CUDA time is 0.24 s out of 3.0 s, so the GPU is idle about 90 % of the time.
- NCI1 graphs have about 30 nodes, and each step is one graph (`graph_batch_size: 1`, the HetSheaf protocol).
- The step is bound by kernel launches inside the PCG iterations of `src/models/anchored_solve.py`. That file is card 12's and not on this card's list; the data pipeline is a small share.
- Getting 5× on NCI1 needs one of these, all outside this card:
  - CUDA graphs or a fused PCG in `anchored_solve.py`;
  - more than one graph per optimizer step (a protocol change);
  - running GC on the CPU (not measured).

## Tests

| Test | Job | Result |
|---|---|---|
| (a) `tests/test_card14_equivalence.py`: `build_scope_batch` (AnchorCells and per-target-list inputs), `sample_wisp_walks_chunked`, `build_graph_wisp_path_groups` against the 66e9349 loops in `tests/reference/card14_reference.py`, on CPU and CUDA. Every output tensor is exactly equal; no reordering was needed. Random heterogeneous graph with self-loops and a self-inverse relation, lengths 2–4, chunk size 3–4, 3-graph PyG batch. | 18259 (3090) | 3/3 OK. Run with 15 related modules: 155 tests OK. |
| (b) one-epoch smoke, mag-year | 18266 (3090) | loss 1.589 finite, 37/37 parameters moved, 122 s, GPU 7.8 GiB, host 6.1 GB |
| (b) PDNS | 18267 (3090) | loss 0.396 finite, 29/29 moved, 286 s, GPU 12.0 GiB, host 6.0 GB. No `PCGConvergenceError` in epoch 0. |
| (b) NCI1 (lr 3e-4) | 18268 (TITAN RTX); 18271 / 18272 (3090, new / old) | loss 0.684 finite, 158/163 moved. The same 5 tensors (`node_encoder.encoders.33.*`, `node_to_edge.frame_params`) also stay unmoved on the old code (18272), so this is structural (unused), not card 14. |
| (b) ACM full batch | 18269 (TITAN RTX) | loss 1.110 finite, 36/36 moved |
| (c) full suite, W&B tests not under `WANDB_MODE=disabled` | 18270 (3090, 36G, 56 min) | 986 tests, **2 failures, both expected** (below), 0 errors |

The two suite failures both compare bit-exact against captures made with the old random stream:

- `test_v3b_golden`: 870 differing tensors, all from H2S-NN configurations whose WISP draws changed (cells, logits, losses, gradients, solver statistics). It has no new keys.
- `test_card5_protocol_hooks.HgbUnchangedTests.test_T4_records_equal_the_base_commit`: `hgb_full_batch` is a 3-epoch H2S-NN WISP run on a synthetic ACM-shaped graph; its float-hex metrics differ.

Amendment 2 accepts that every number changes. Re-capturing `docs/legacy_evidence/v3b_golden_cluster.pt` and `docs/card5_evidence/base_capture_894fd74.json` is outside this card's files and is the author's call. The sensible time is after card 12 and card 14 are both merged, because card 12 also touches the golden.

## Other things to know

- Existing tests that pass a `torch.Generator` to `sample_wisp_walks` or `sample_walks` keep working: the generator's `initial_seed()` is used as the stream seed.
- Old-vs-new on the 3090 ran two jobs at a time on g3n01 (4 CPUs each). The old mag-year and PDNS jobs overlapped each other; the new ones overlapped the targeted tests.
- `--mem` for these runs came from the card 11 peaks plus 30 %. The measured peaks (in the table) are lower for the new code. For PDNS and mag-year, 8G covers peak + 30 %.
- No A6000 was used. No run reached 1 h.

branch=card14 sha=fa68b4e44497ecfb48104628077dba1be0ba1421 card=docs/H2S-NN_card14_card.md base=66e9349b79021784a08b37c5f8be0798c0260a04
