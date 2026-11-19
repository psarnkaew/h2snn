# Card 2 D6.5 — H2SGNN LP freebase_bb: failing kernel (no fix; session 1)

Real data, D2's largest point, merged head (`0944edb` code), `CUDA_LAUNCH_BLOCKING=1`; TITAN RTX / RTX 3090, 30G host.
- Job 16923_1 (g2tn01; `CUDA_LAUNCH_BLOCKING=1`): the illegal memory access is raised synchronously inside
  `loss.backward()` (`src/baselines/training.py:239`, `train_link_prediction`), no longer at a later
  `torch.cuda.synchronize()` as in D2 — so the faulting kernel is in the backward pass; the bundle built (136.7 s).
- Job 16930_1 (g3n01; `CUDA_LAUNCH_BLOCKING=1` + `torch.autograd.set_detect_anomaly(True)`, `d6_5_anomaly_trace.txt`):
  **failing kernel: the backward of torch_sparse's `SPMMSum`** (`torch::autograd::CppNode<SPMMSum>`, i.e. the CUDA
  `spmm_sum` backward of a `SparseTensor @ dense` product). Forward call of that node, innermost first:
  `torch_sparse/matmul.py:27` (`spmm_sum`) ← `vendor/H2SGNN/model.py:116` (`GPR_GPR`: `x = torch_sparse_matmul(A, x)`)
  ← `vendor/H2SGNN/model.py:81` (`forward`) ← **first repository file:line `src/baselines/adapters/_h2sgnn_lp.py:179`**
  (`block = self.vendor(metapath_operators, [features], metapath_operators)`) ← `adapters/h2sgnn.py:323, 320, 218`.
Same error on both nodes (not a device fault). Freebase book–book is the largest LP graph; the other H2SGNN LP cells
are OK (D2), so the operator `A` built for freebase_bb (`_h2sgnn_lp.py`, the metapath operators) is the first thing to
check. Not fixed here (src/baselines, session 1).
