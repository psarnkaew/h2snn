# Card 24 report: H2SGNN Freebase NC row-blocked product (2026-11-28)

Base: v3b `528aa9b`. Author's instruction, 2026-11-28.

**Cause** (ops §13, `plan_change_2026-11-24.md`; lab jobs 19070, 19085, 19086, 19089, 19096; TSUBAME 9604709):
- Freebase NC's combined meta-path operator `A = add(adjs_list, softmax(beta))` has 185 M non-zeros, and its values carry beta's gradient.
- torch_sparse 0.6.18's `SPMMSum` value-gradient backward fails with `illegal memory access` above about 2²⁶ non-zeros: 50 M works, 70 M fails.
- All 40 H2SGNN Freebase NC trials failed in `loss.backward()`.

**Change:** `src/baselines/adapters/h2sgnn.py` only; `vendor/` is unchanged.
- `blocked_vendor_forward` binds the existing `src/baselines/runtime/blocked_spmm.blocked_matmul` with `max_nnz = NC_MAX_NNZ = 50_000_000` as the vendor module's `torch_sparse_matmul`, for the NC forward only (`GPR_GPR` and the other bases).
- This uses the same mechanism as the LP encoder (card 1e F3.1).
- Operators with at most 50 M non-zeros take the plain call, so every NC cell except Freebase is bit-identical.
- Freebase's 185 M operator runs in 4 blocks.

**Evidence of equality** (lab job 19097, RTX 3090):
- At 50 M non-zeros: same output and same value gradients as the plain product; x gradients within 1.4e-6 (summation order).
- At 185 M: 4 blocks, 7.6 GiB peak memory, finite gradients.

**Tests:**
- New: `tests/test_baselines_h2sgnn_nc_blocked.py` (blocked = plain on a small matrix split into ≥ 3 blocks; the binding is undone).
- Run on the lab CPU: the new test, `test_baselines_blocked_spmm`, and the `-k h2sgnn` tests of `test_freebase_baselines`, `test_baselines` and `test_baselines_dryrun_blockers`. 8 tests in total, all OK.

**Not changed:** the LP encoder keeps `MAX_NNZ = 2²⁶ − 1` (Freebase B–B LP, 166 M → 3 blocks). That value is at the measured limit.

**Next:** full suite, then merge into v3b, then a rerun of H2SGNN Freebase NC on TSUBAME (40 trials + 5 seeds).
