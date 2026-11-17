# Task card — Baselines on Freebase: sparse-code failures of HAN and H2SGNN (card 1e)

**Branch** `baselines-h2gb-6` from `v3b` `ae7aef026f4a08f50634a2ea154cbfa31e998b92`; merge the pinned `design-docs` SHA first. **Session:** S1. **Report** `docs/H2S-NN_baselines_freebase_sparse_report.md`; evidence `docs/baselines_h2gb_evidence/p6_*`. Slim preparation rules; dry-run Rule of card 2 (no metric; status, memory, time only; 24 GB GPU not shared with another session; job scripts print HEAD, hostname, job id).

## Why
Card 2 (D2, D6.5; `dry-run` `docs/dryrun_evidence/cells.md`, `d6_*`) found three failures, all in sparse code on HGB Freebase (180,098 nodes, 1,057,688 links):
| Cell | Failure (as recorded) |
|---|---|
| HAN NC freebase | segfault right after `vendor/HGB/NC/benchmark/methods/HAN/utils.py:213` (sparse one-hot features `book_num × book_num`) (job 16797_56) |
| HAN LP freebase_bb | OOM at heads 8; at heads 4 segfault after `torch_sparse/matmul.py:97` (CSR `torch.sparse.mm`) |
| H2SGNN LP freebase_bb | CUDA illegal memory access in the backward of `torch_sparse` SPMMSum; forward `vendor/H2SGNN/model.py:116` (`GPR_GPR`) ← `model.py:81`; first file of ours `src/baselines/adapters/_h2sgnn_lp.py:179` (D6.5, jobs 16923_1, 16930_1) |

## Units
### F0 — Diagnose (no fix)
For each failure, on real data (allowed: a diagnosis, no metric), one run with `python -X faulthandler` (and `CUDA_LAUNCH_BLOCKING=1` for CUDA): the exact failing call (Python and native frame), and at that call the shapes, `nnz`, index dtypes, devices, coalesced/sorted state of every sparse operand, and whether any index or `nnz`/row-pointer value exceeds 2^31 − 1. Then answer: **does the source's own code reach the same call with the same operands on Freebase** (compare our adapter's inputs with the source's loader/preprocessing, file:line), or is the input ours (adapter-built, different from the source's)? Record in `p6_f0_diagnosis.md`.
### F1 — Fix or stop item
- **Cause in our adapter** (inputs differ from the source's, wrong dtype/device/layout, uncoalesced, …): fix it to match the source's code, with one test that reproduces the error on a synthetic graph if a synthetic graph can trigger it; if the failure needs Freebase's scale, the evidence is a real-data before/after one-epoch run (status, memory, time only).
- **Cause in vendor code or the library at this scale** (e.g. 32-bit index overflow, a `torch_sparse` kernel bug): **stop item**, no change; give options with cost (e.g. a different sparse layout or op in the adapter that is mathematically equal, an int64 path, CPU for that op, N/A with a stated reason as SeHGNN on H-Pokec) for the author.
- Vendor code is never edited; nothing is installed into `h2snn-b`.
### F2 — Suite and report
Full suite once on one node. Report short: per failure — cause, file:line, source comparison, fix or options; tests; last line `branch=baselines-h2gb-6 sha=<full head> card=docs/H2S-NN_baselines_freebase_sparse_card.md base=ae7aef026f4a08f50634a2ea154cbfa31e998b92`.

### F3 — Fixes (added 2026-11-18 after the F0/F1 review; author decisions "1e: a, a, b"; decisions log row "Card 1e review; F3; 1f merged")
1. **H2SGNN: row-block product** (F1 option a). New module `src/baselines/runtime/blocked_spmm.py` (shared, so card 3 Part B's H2SGNN NC can import it): `blocked_matmul(A, x, max_nnz=2**26 - 1)` — for a `SparseTensor` with nnz > `max_nnz`, split by rows (`A.narrow(0, start, length)`, each block ≤ `max_nnz` nnz; a single row above the limit is a stop item), the same `torch_sparse.matmul` per block, `torch.cat`; otherwise the plain call. `_h2sgnn_lp.py` binds it in place of `torch_sparse_matmul` in the vendor model module's namespace during the vendor forward/backward of that cell only, through a context manager that restores the attribute (vendor file not edited). **Test:** forward and the gradients w.r.t. the sparse values and `x` equal the unblocked product on a synthetic matrix with a small `max_nnz` (several blocks, uneven sizes): forward exact, gradients atol 1e-6 (design session check, n = 50: 0, 0, 4.8e-7); the attribute is restored after the context (no leak). No mutation (Slim).
2. **HAN on Freebase: GPU, ranges from F0 evidence, no ladder run** (author 2026-11-18, second correction): **HAN NC freebase** range `num_heads` = {4} (heads 8 OOM on 24 GB with 1.01 GiB free before the last meta-path graph; heads 4 halves the edge × head tensors; DERIVED, not measured), `hidden_dim` full; **HAN LP freebase_bb** = **N/A**, reason "OOM on a 24 GB GPU at `num_heads` 4, the grid minimum; the width is a secondary memory term (F0 item 3)". Recorded in `p6_f3_ranges.md` and in the registry/grid as the cell's range or N/A entry. No HAN model code change; no GPU job.
3. No suite here: card 1f's U4 suite covers F3 (same branch).

## Files
`src/baselines/**` (incl. the new `src/baselines/runtime/blocked_spmm.py`), baseline tests, evidence `docs/baselines_h2gb_evidence/p6_*`, the report. Nothing else (a needed change elsewhere is a stop item).

## Rules and stop conditions
- One commit per unit; push `baselines-h2gb-6` after each; never push another branch, never `v3b`, never force-push; after the report line, no further push.
- At most 2 GPU jobs at once (S2 and S3 also run). No metric anywhere.
- Stop and report if: the fix would change the method or the source's protocol; the cause is outside our adapter (F1 second bullet); the suite fails.
