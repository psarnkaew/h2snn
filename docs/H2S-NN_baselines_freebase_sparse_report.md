# Report — card 1e: Freebase sparse failures of HAN and H2SGNN

Branch `baselines-h2gb-6`. Base: v3b `ae7aef0` for F0–F1; v3b `a8828a8` and design-docs `6da1199` merged before F3 (`28226ce`, `e19d54e`). Session S1. The evidence is in `docs/baselines_h2gb_evidence/p6_*`. No metric was computed anywhere.

## Units and commits

| Unit | Commit | Content |
|---|---|---|
| merge | `457c223` | design-docs `66f0e7a` (card 1e) |
| F0 | `dcc4f92` | diagnosis on real data, one epoch, no metric: `p6_f0_diagnosis.md`, `p6_f0_*.txt`, driver `p6_f0_diag.py` |
| F1 | `6ccdd7c` | stop items with options and costs, no code change: `p6_f1_stop_items.md` |
| F2 | — | suite 17521 cancelled at the author's instruction (2026-11-18): 1e F0–F1 changed no code, and card 1f U4 covers F3 |
| F3.1 | `96d53b8` | H2SGNN row-blocked product (`src/baselines/runtime/blocked_spmm.py`) |
| F3.2 | `a56a801` | HAN Freebase ranges from F0 (`p6_f3_ranges.md`) |

## Per failure

| Cell | Cause (F0) | Where | Source comparison | Outcome |
|---|---|---|---|---|
| H2SGNN LP `freebase_bb` | torch_sparse 0.6.18 `spmm_value_bw_kernel` launches E × 32 threads with an `int` thread index, so it faults for E > 2^26 − 1. Here `A = add(adjs_list, softmax(β))` has E = 166,467,166, and its values need a gradient because β is learnable. | backward of `torch_sparse_matmul(A, x)`, `vendor/H2SGNN/model.py:116` ← `:81`; ours `_h2sgnn_lp.py:179` | The source has no Freebase path (`processing.py:29-128`). The operators are ours, built by the source's rule; the call is the source's. | **F3.1 fixed** (author "a"): row-blocked product, below |
| HAN LP `freebase_bb` | GPU memory above 24 GB at heads 4: 8 meta-path graphs, 202.6M edges, CUDA OOM in GAT 5 of 8 | `GATConv.forward` (DGL) | HAN ships no LP; the graphs are ours | **F3.2: N/A**, reason: "OOM on a 24 GB GPU at num_heads 4, the grid minimum; the width is a secondary memory term (F0 item 3)" |
| HAN NC `freebase` | GPU memory above 24 GB at heads 8: 7 graphs, 140.0M edges. DGL's `_gspmm` max inside `edge_softmax` segfaults instead of raising OOM, with 1.01 GiB free. On CPU one epoch completes: 352 s, 33.5 GiB (17161). | `dgl/sparse.py:709-710` (CUDA edge_softmax) | The same loader, meta-paths and model as the source (`HAN/utils.py:60`, `:175-236`; `model_hetero.py:111-129`). The source runs Freebase on CPU (`README.md:17`). | **F3.2: range `num_heads` {4}** (DERIVED from F0, not measured), `hidden_dim` full |

The earlier "last lines" of the dry run (`utils.py:213`, `matmul.py:97`) were where warnings were printed, not the failing calls. The heads-4 HAN LP segfault was not reproduced, because the OOM came first.

## F3.1 — row-blocked product

- `blocked_matmul(A, x, max_nnz=2**26 - 1)`: if `A` is a `SparseTensor` with nnz > `max_nnz`, it is split greedily into consecutive row blocks of at most `max_nnz` non-zeros (`A.narrow(0, start, length)`). Each block uses the same `torch_sparse.matmul`, and the results are joined with `torch.cat`. Every other call is the plain call. A single row above the limit raises an error (a stop item).
- `_h2sgnn_lp.py`: `H2SGNNLinkEncoder.forward` binds `blocked_matmul` as `torch_sparse_matmul` in the vendor model module's globals (`type(vendor).forward.__globals__`). It uses `bound_attribute`, which restores the previous value on exit, including on an exception. The autograd graph built in the forward carries the blocked products into the backward. Vendor file not edited.
- Every other H2SGNN LP cell is unchanged, because no operator there has nnz > 2^26 − 1.
- Tests (`tests/test_baselines_blocked_spmm.py`):
  - Equivalence on n = 50 rows with uneven degrees, max_nnz 37 (several blocks of different sizes): the forward is `torch.equal`; the value and x gradients match to atol 1e-6. A single row above the limit is refused.
  - No leak: the binding is restored after normal exit and after an exception, and an absent name is removed again.
  - No mutation (Slim).
- **Not measured:** whether the Freebase LP epoch fits on 24 GB with the blocked product. No GPU job ran in F3 (per the prompt); the first TSUBAME trial is the real-data check.

## F3.2 — HAN Freebase ranges

See `p6_f3_ranges.md`. Changes:
- `hgb_han.py`: `search_space(NC, freebase)` sets `num_heads` to [4]; `freebase_bb` is removed from HAN's `SUPPORTED_LP_DATASETS`.
- `registry.py`: `HAN_FREEBASE_LP_NA` is added as an `unsupported` entry on the HAN row.
- `test_baselines.py`: runnable LP cells go from 76 to 75.

No HAN model change and no GPU job. If a heads-4 NC trial OOMs on TSUBAME, record it; any further step-down is a design-session decision.

## Tests

Run in card 1f's U4 suite (CPU job, same branch), reported in `docs/H2S-NN_b1a_baselines_report.md`. The F3.1 test also ran in the U3 job (`docs/b1a_evidence/u3_tests.txt`).

## Notes

- **Resource rule (2026-11-16/17):** 16955 was cancelled and resubmitted at 16G. 17096 and 17115 were cancelled and resubmitted as 17139/17140 at 16G, `--gres=gpu:1 --exclude=g2n01,g3an01`. The only 60G job (17161, CPU) came after a host OOM at 24G (17158). The F3/1f CPU jobs ran at 16G and 32G.
- **No A6000 (author 2026-11-17):** no job requested the A6000 or g3an01. The heads-4 HAN LP OOM is recorded as N/A, not retried on the A6000.
- **Deviation:** the F0 driver `p6_f0_diag.py` changed between runs (sha prefixes `d762…`, `2ecb…`, `e094…` in the F0 job table). Each run's output names the driver version it used.
- The three HAN NC core dumps (about 4.6 GB, from the HAN NC GPU segfault runs) were never committed. They were deleted on 2026-11-18 with the author's approval.
- Combined review with card 1f: the final line is in `docs/H2S-NN_b1a_baselines_report.md`.
