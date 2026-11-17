# Card 1e F0 — diagnosis of the three Freebase sparse failures

Real data, one epoch through `campaign.fit_cell` (the dry run's path) at the dry run's point; no metric is read or
printed. Driver `p6_f0_diag.py` (run as `python -X faulthandler`; it logs every candidate sparse call's operands with a
flush before the call). Env `h2snn-b` (torch 2.4.1+cu124, torch_sparse 0.6.18, DGL 2.4.0). No native debugger exists on
the cluster (gdb, lldb, eu-stack absent), so native frames are named from the Python frame that entered the C call and
from the library source. Every job printed the driver's sha256 first; the driver grew between runs (logging only), so
the committed file is the last version (`e09455527e23f242`); earlier runs used `d7625736eaa8f794` and `2ecb2f5ccfc45b26`.

| Job | Cell | GPU / host | Driver | Output | Outcome |
|---|---|---|---|---|---|
| 17140 | H2SGNN LP freebase_bb, `CUDA_LAUNCH_BLOCKING=1` | RTX 3090, 16G | d762… | `p6_f0_h2sgnn_lp.txt` | CUDA illegal memory access in backward |
| 17156 | HAN LP freebase_bb, heads 4 | RTX 3090, 16G | d762… | `p6_f0_han_lp.txt` | CUDA OOM (forward, GAT 5 of 8) |
| 17139 | HAN NC freebase | RTX 3090, 16G | d762… | `p6_f0_han_nc.txt` | segfault (faulthandler frame cut off) |
| 17157 | HAN NC freebase, `CUDA_LAUNCH_BLOCKING=1` | RTX 3090, 16G | 2ecb… | `p6_f0_han_nc_cuda.txt` | segfault, frame in `edge_softmax` |
| 17160 | same, with GPU memory logged | RTX 3090, 16G | e094… | `p6_f0_han_nc_cuda_mem.txt` | segfault at 1.01 GiB free |
| 17158 | HAN NC freebase on CPU (`P6_DEVICE=cpu`) | — , 24G | 2ecb… | `p6_f0_han_nc_cpu.txt` | host OOM kill at 24.2 GiB |
| 17161 | same at 60G (after the host OOM) | — , 60G | e094… | `p6_f0_han_nc_cpu60.txt` | **one epoch completed**, 352 s, peak RSS 33.5 GiB |

Earlier jobs 16954/16957/17096/17097/17115 failed in the driver itself (an import of `torch_sparse.matmul` that
returns the function; `SparseStorage.col` is a method) or were cancelled under the resource rule; they reached no model
code. Their outputs were overwritten by the reruns.

## 1. H2SGNN LP freebase_bb — CUDA illegal memory access

**Failing call.** `loss.backward()` (`src/baselines/training.py:239`) → autograd backward of torch_sparse `SPMMSum`
for spmm call #6, the last of the K = 6 global products `x = torch_sparse_matmul(A, x)` in `GPR_GPR`
(`vendor/H2SGNN/model.py:116`). Hooks on each spmm output show backward passing through calls #28 … #7 (the
per-meta-path `adj` products) and failing on the first `A` product it reaches. Native kernel: torch_sparse 0.6.18
`csrc/cuda/spmm_cuda.cu`, `spmm_value_bw_kernel`, which runs only when the sparse values require a gradient.

**Operands at the call** (`p6_f0_h2sgnn_lp.txt` lines 63-66):
- `A`: `SparseTensor` 40,402 × 40,402, **nnz = 166,467,166**, on cuda:0. `row`, `col` and `rowptr` are int64; the
  largest col is 40,401 and `rowptr[-1]` is 166,467,166. Values are float32 and require a gradient, because
  `A = add(adjs_list, softmax(beta))` (`model.py:69-70`, `utils.py:9-27`) and `beta` is a learnable parameter
  (`model.py:33`). `A` is coalesced (`utils.py:26`), and the colptr/csr2csc caches exist from call #2 on.
- `x`: dense 40,402 × 128, float32, cuda:0, requires grad.
- **No index or row-pointer value exceeds 2^31 − 1.** The products that do are nnz × width = 21,307,797,248, and
  the kernel's thread count, **E × 32 = 5,326,949,312**.

**Why it faults.** The upstream source of the installed torch_sparse version (0.6.18, `csrc/cuda/spmm_cuda.cu`) shows
three facts:
- `spmm_value_bw_cuda` launches `BLOCKS = (E * 32 + THREADS - 1) / THREADS` blocks of 256 threads.
- The kernel computes `int thread_idx = blockDim.x * blockIdx.x + threadIdx.x; int index_idx = thread_idx >> 5;`
  and guards with `if (index_idx < E)`.
- The kernel takes `int E` as an argument.

For E > 67,108,863 (= 2^31 / 32), `thread_idx` overflows to negative values. A negative `index_idx` still passes the
`< E` guard, and `row_data[index_idx]` reads outside the buffer. That out-of-bounds read is the illegal access.

The forward kernel (`spmm_kernel`) uses one warp per row (40,402 × 32 threads), so the forward runs. The per-meta-path
operators (largest nnz 54,932,925 < 67,108,864) never request a value gradient, so their backward runs too.

**Source comparison.** H2SGNN's own code has no Freebase path: `processing.py:29-128` and `utils.py:55` handle only
IMDB, DBLP, ACM and AMiner. The source therefore never reaches this call on Freebase. The operators are our adapter's:
`_h2sgnn_lp.build_metapath_operators` (`src/baselines/adapters/_h2sgnn_lp.py:91-147`) forms every length-2 product of
relation adjacencies, follows the source's chaining rule, and crops each product to the book block. That yields 8
operators with nnz 52.9M, 4.2M, 14.4M, 37.9M, 54.9M, 35.0M, 0.08M and 3.1M. The vendor's `add` concatenates them and
coalesces them into `A` (166.5M nnz, 10 % of 40,402²).

Each operator is valid: coalesced, int64 indices, on the right device. The call, `utils.add`, `GPR_GPR` and the
learnable β are the source's. The input is ours only in the sense that no source input exists for Freebase.

**Cause:** library at this scale (a 32-bit thread index in torch_sparse's value-gradient kernel). It is not an adapter
defect.

## 2. HAN LP freebase_bb (heads 4, the dry run's step-down) — CUDA OOM

**Failing call.** `_han_lp.py:105` → `model_hetero.py:106` → `HANLayer.forward`, `model_hetero.py:87` → DGL
`GATConv.forward` (`gatconv.py:357`, bias add) for meta-path graph 5 of 8. The error is
`torch.OutOfMemoryError: Tried to allocate 158.00 MiB … 87.81 MiB is free … 23.21 GiB is allocated by PyTorch`
(`p6_f0_han_lp.txt`).

**Operands.**
- **Meta-path graphs:** `dgl.metapath_reachable_graph` over the adapter's operators, all on cuda:0, idtype int64,
  40,402 nodes each. Edge counts per graph: 52,859,170, 4,193,476, 14,405,128, 37,884,185, 54,932,925, 35,046,698,
  84,404 and 3,145,804, about 203M in total.
- **GAT layers:** heads = 4, out = 256, so each layer works on edges × heads values, the largest being 219.7M.
- **Features:** dense 40,402 × 256.
- **Sparse products:** the eight `torch.sparse.mm` products that build the operators run on CPU (COO, int64,
  uncoalesced in, coalesced out) and all complete. The dry run's "last line in the log" (`torch_sparse/matmul.py:97`)
  was where the CSR beta warning was printed, not where the run faulted.
- **No index exceeds 2^31 − 1.**

**Source comparison.** HAN ships no link prediction. The graphs are our adapter's, built from the same operators as
H2SGNN's (`_han_lp` uses `_h2sgnn_lp.build_metapath_operators`).

**Outcome.** On this 24 GB GPU the cell needs more than 24 GB at heads 4. The OOM arrives before the dry run's
segfault could. That segfault (on a TITAN RTX) is consistent with item 3, where DGL segfaults when GPU memory runs out
inside its kernels, but this run did not reproduce it. Per the author's rule (2026-11-17) it is recorded and not
retried on the A6000.

## 3. HAN NC freebase — segfault

**Failing call.** `hgb_han.py:247` → `model_hetero.py:127` (`HAN_freebase.forward`) → `model_hetero.py:87` → DGL
`GATConv.forward` (`gatconv.py:340`) → `edge_softmax` (`dgl/ops/edge_softmax.py:136` →
`dgl/backend/pytorch/sparse.py:1116`, `EdgeSoftmax.forward` `:710`) → `_gspmm(gidx, "copy_rhs", "max", None, score)`
(`dgl/_sparse_ops.py:239`) → native `_CAPI_DGLKernelSpMM`. This is the CUDA branch of edge softmax
(`sparse.py:709-710`) for meta-path graph 7 of 7, `('07', '70')`, and the error is
`Fatal Python error: Segmentation fault` (`p6_f0_han_nc_cuda.txt`, `p6_f0_han_nc_cuda_mem.txt`).

**Operands.**
- **Graph:** 40,402 nodes, 35,046,698 edges, idtype int64, cuda:0. Formats at the call: COO created, CSC created
  by this call.
- **Scores:** dense (35,046,698, 8, 1) float32, 280,373,584 values.
- **Edge counts:** the 7 graphs have 49,275,062, 4,193,476, 4,021,889, 3,145,804, 13,745,798, 30,573,570 and
  35,046,698 edges. Heads 8, out 256.
- **Sparse input:** the input feature is the source's sparse one-hot, COO 40,402 × 40,402, nnz 40,402, int64,
  uncoalesced. Its `F.linear` runs before any graph call and completes; the dry run's "last line"
  (`HAN/utils.py:213`) was the deprecation warning.
- **No index or edge count exceeds 2^31 − 1.** `edge_softmax` on graph 1, the largest (49.3M edges), completed.

**GPU memory before each `edge_softmax`.** Free memory fell steadily on the 23.68 GiB card:

| GAT layer | 1 | 3 | 5 | 6 | 7 |
|---|---|---|---|---|---|
| free GiB | 17.53 | 12.14 | 9.15 | 5.09 | 1.01 |

At the fault 22.24 GiB was allocated. The kernel needs CSC conversion buffers of the order of E × 8 B several times
over. The segfault is therefore an allocation failure inside DGL's native SpMM path, which surfaces as SIGSEGV instead
of a CUDA OOM. This is evidence from the memory trace; with no native debugger, the faulting instruction itself was
not observed.

**Source comparison.**
- **Same call, same operands:** our adapter loads Freebase with the source's own loader (`hgb_han._load_vendor_graph`
  → `HAN/utils.py:175-236`). It builds the same 7 meta-paths (`HAN/utils.py:60`) and the same sparse one-hot feature
  (`:213`), and uses the source's `HAN_freebase` model (`model_hetero.py:111-129`).
- **Two differences:**
  - **Device.** The source's README runs Freebase on CPU (`HAN/README.md:17`,
    `python main.py --dataset Freebase --device cpu --num_epochs 1000`); our adapter runs on CUDA.
  - **Width.** The source's configuration is `hidden_units 8, num_heads [8]` (`HAN/utils.py:97-101`). The dry-run
    point is `hidden_dim 256, num_heads 8` (the maximum of our search space, `hgb_han.py:148-150`), 32 times wider
    per head.
- **Source path on CPU:** run on CPU at the dry-run point, it completes one epoch (17161: forward through all 7 GAT
  layers, backward, eval pass; 352 s; peak RSS 33.5 GiB). At 24G it was OOM-killed at layer 7 (17158).

**Cause:** GPU memory at this scale with our search-space width. DGL turns the out-of-memory in its native SpMM into a
segfault. The operands and code path are the source's. It is not an adapter defect in layout, dtype or coalescing.

## Summary

| Cell | Failing call | > 2^31 − 1? | Same call as the source on Freebase? | Cause |
|---|---|---|---|---|
| H2SGNN LP freebase_bb | torch_sparse `spmm_value_bw_kernel` (backward of `A·x`, `model.py:116`) | thread index E × 32 = 5.33e9 (int) | no Freebase path in the source; operators are ours, the call is the source's | library at this scale (int32 thread index) |
| HAN LP freebase_bb | `GATConv.forward` (bias add), graph 5 of 8 | no | HAN has no LP; graphs are ours | GPU memory > 24 GB at heads 4 |
| HAN NC freebase | DGL `_gspmm` max in `edge_softmax`, graph 7 of 7 | no | yes, same loader, meta-paths and model; source runs on CPU, hidden 8 | GPU memory > 24 GB at hidden 256 × 8 heads; DGL segfaults instead of raising OOM |
