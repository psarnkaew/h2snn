# Card 1e F1 — stop items (no code change)

The F0 diagnosis (`p6_f0_diagnosis.md`) found no adapter defect. The three failures are:
- a torch_sparse kernel limit (H2SGNN);
- GPU memory above 24 GB at Freebase's scale (both HAN cells). In one of them DGL turns the out-of-memory into a
  segfault.

The card's rule therefore applies (F1, second bullet): **stop item, no change**. The options below are for the author
or the design session. Costs are estimates unless a run is cited, and "UNVERIFIED" marks what a single run would
settle. The A6000 is excluded (author, 2026-11-17).

## H2SGNN LP freebase_bb — torch_sparse `spmm_value_bw_kernel` int32 thread index

**Limit.** The kernel launches E × 32 threads and indexes them with `int`, so it fails for nnz E > 67,108,863. The
combined operator `A` has E = 166,467,166.

| # | Option | Equal to the method? | Cost |
|---|---|---|---|
| a | **Row-block product in the adapter.** While the vendor forward runs, the adapter replaces the module attribute `torch_sparse_matmul` in `vendor/H2SGNN/model.py`'s namespace (the file is not edited). For a `SparseTensor` with nnz > 2^26 − 1, the replacement splits it by rows into blocks of ≤ 2^26 − 1 nnz (`narrow` on dim 0), multiplies each block with the same `torch_sparse.matmul`, and concatenates the results. Each output row uses only its own row's entries, and `narrow`'s backward places each block's value gradient into the full value gradient. | yes. The forward is the same per-row sum; whether it is bit-identical to the unsplit kernel is to be tested. | about half a day. One GPU test with a synthetic `A` of nnz > 67,108,863 (about 1.4 GB on the GPU): illegal access before the change, equal outputs and gradients after, plus equality with the unsplit product on a small `A`. One real-data before/after epoch (status, memory, time). Whether the full epoch then fits on 24 GB is **UNVERIFIED**; the forward and the backward through the per-meta-path products already ran. The replacement is a patch of a vendor module attribute, so the author should approve it. |
| b | **CPU for this cell.** | yes | Whether the torch_sparse CPU backward is free of the same limit is **UNVERIFIED**. Time is likely minutes per epoch (12 products of 166M nnz × 128 per epoch, forward and backward), also **UNVERIFIED**. Measuring both takes one CPU job. |
| c | **N/A with a stated reason**, as with SeHGNN on H-Pokec: "torch_sparse 0.6.18 cannot back-propagate into a learnable-weight sum of more than 2^26 non-zeros". | — | none |
| — | Fixing β (no value gradient) | **no**: β is learnable in H2SGNN (`model.py:33`), so this changes the method | not offered |
| — | A newer torch_sparse | nothing may be installed into `h2snn-b`; whether upstream fixed the kernel is UNVERIFIED | not offered |

## HAN LP freebase_bb — CUDA OOM on 24 GB at heads 4

The cell has 8 meta-path graphs with 202.6M edges in total, run at heads 4. HAN ships no link prediction, so the
graphs come from our operators.

| # | Option | Cost |
|---|---|---|
| a | **N/A on 24 GB** (recorded OOM, as the rule says) | none |
| b | **CPU**, as option (a) for HAN NC. This cell has 1.45 × the NC edges at half the heads (810M vs 1,120M edge × head values), so its host memory and time are likely of the NC order (33.5 GiB, about 6 min per epoch). | **UNVERIFIED**; one CPU job at 60G (about 10 min) would measure it |
| c | A further step-down (heads 2) or fewer meta-paths | changes the dry-run Rule / search space; a design-session decision. Whether heads 2 fits is UNVERIFIED |

## HAN NC freebase — GPU memory exhausted inside DGL's `edge_softmax` (segfault)

The cell has 7 meta-path graphs (the source's own, `HAN/utils.py:60`) with 140.0M edges, run at heads 8.

| # | Option | Cost |
|---|---|---|
| a | **CPU, as the source runs Freebase** (`HAN/README.md:17`, `--device cpu`). The adapter would choose CPU for HAN on `freebase`. That matches the source's protocol and leaves the method unchanged. | Measured (17161): **352 s per epoch** (train and eval) at the dry-run point, **peak RSS 33.5 GiB**, so `--mem` 44G. The source runs 1,000 epochs for Freebase (README; the default is 200, `main.py:112`) with patience 100 (`main.py:113`): about 9.8 h for 100 epochs and 98 h for 1,000, which exceeds the 72 h job cap without a checkpoint. Our campaign's epochs and patience come from the protocol (`training.py:21`), so the real cost is protocol epochs × 352 s per trial and seed. That is large for a tuning sweep. |
| b | **Fewer heads on GPU** (heads 4 halves the edge × head tensors) | Whether it fits is **UNVERIFIED**; HAN LP at heads 4 did not fit, with 1.45 × the edges. This narrows the search space, a design-session decision. |
| c | **N/A on 24 GB** with the stated reason | none |

## Recommendation (PROPOSAL)

- **H2SGNN:** (a) if the author accepts a module-attribute patch, (c) otherwise.
- **HAN NC:** (a) if the protocol's epoch budget makes 352 s per epoch affordable, (c) otherwise.
- **HAN LP:** (b) after a one-job CPU measurement, (a) otherwise.
