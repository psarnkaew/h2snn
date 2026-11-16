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

## Files
`src/baselines/**`, baseline tests, evidence `docs/baselines_h2gb_evidence/p6_*`, the report. Nothing else (a needed change elsewhere is a stop item).

## Rules and stop conditions
- One commit per unit; push `baselines-h2gb-6` after each; never push another branch, never `v3b`, never force-push; after the report line, no further push.
- At most 2 GPU jobs at once (S2 and S3 also run). No metric anywhere.
- Stop and report if: the fix would change the method or the source's protocol; the cause is outside our adapter (F1 second bullet); the suite fails.
