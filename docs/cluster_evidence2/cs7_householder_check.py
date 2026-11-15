"""CS7-1d: torch_householder JIT compile (g++ + ninja) against torch 2.4.1, and its known answer.

Run from H2S-NN/ with TORCH_EXTENSIONS_DIR set. Convention (torch_householder/householder.py:59-84,
householder.cpp:28-37): with tau=None each column of `param` is normalised to a unit vector u_j and
the output is H_0 H_1 ... H_{r-1} eye(d, r), H_j = I - 2 u_j u_j^T.
Tolerances fixed by the card: max|Q^T Q - I| < 1e-10 and allclose(atol=1e-10) against plain torch (float64).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

ext_dir = Path(os.environ["TORCH_EXTENSIONS_DIR"])
print("torch", torch.__version__, "TORCH_EXTENSIONS_DIR", ext_dir, "exists before import:", ext_dir.exists())
t0 = time.perf_counter()
from torch_householder import torch_householder_orgqr  # noqa: E402  (JIT-compiles householder.cpp)

print(f"import torch_householder (incl. JIT compile): {time.perf_counter() - t0:.1f} s")
for ninja in sorted(ext_dir.rglob("build.ninja")):
    print(f"--- {ninja}")
    for line in ninja.read_text().splitlines():
        if line.startswith(("cxx =", "cflags =", "post_cflags =", "ldflags =")):
            print("   ", line)
for so in sorted(ext_dir.rglob("*.so")):
    print("built:", so)

torch.manual_seed(0)
param = torch.randn(8, 5, dtype=torch.float64)
q = torch_householder_orgqr(param)

u = param / torch.linalg.norm(param, dim=0, keepdim=True)
ref = torch.eye(8, dtype=torch.float64)
for j in range(5):
    v = u[:, j:j + 1]
    ref = ref @ (torch.eye(8, dtype=torch.float64) - 2.0 * v @ v.T)
ref = ref[:, :5]

orth = float((q.T @ q - torch.eye(5, dtype=torch.float64)).abs().max())
diff = float((q - ref).abs().max())
ok_orth = orth < 1e-10
ok_ref = torch.allclose(q, ref, rtol=0, atol=1e-10)
print(f"dtype {q.dtype} shape {tuple(q.shape)}")
print(f"{'PASS' if ok_orth else 'FAIL'}  orthogonality max|Q^T Q - I| = {orth:.3e} (< 1e-10)")
print(f"{'PASS' if ok_ref else 'FAIL'}  equals plain-torch product of 5 reflectors: max|diff| = {diff:.3e} (atol 1e-10)")
sys.exit(0 if ok_orth and ok_ref else 1)
