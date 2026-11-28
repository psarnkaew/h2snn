"""Card 24: H2SGNN's node-classification forward with the row-blocked product equals the plain one."""

from __future__ import annotations

import unittest

import torch

try:
    import torch_sparse
    from torch_sparse import SparseTensor
except ImportError:  # pragma: no cover - the baseline env has it
    torch_sparse = None

from src.baselines.adapters.h2sgnn import NC_MAX_NNZ, blocked_vendor_forward
from src.baselines.runtime.blocked_spmm import row_blocks

# Module-global name, as in vendor/H2SGNN/model.py (`from torch_sparse import matmul as torch_sparse_matmul`).
torch_sparse_matmul = torch_sparse.matmul if torch_sparse is not None else None


class _VendorLike(torch.nn.Module):
    """Calls `torch_sparse_matmul` by module-global name, like H2SGNN's GPR_GPR."""

    def forward(self, A, x):  # noqa: N803 - vendor naming
        return torch_sparse_matmul(A, x)


@unittest.skipIf(torch_sparse is None, "torch_sparse not installed")
class H2SGNNNodeBlockedForwardTest(unittest.TestCase):
    def test_blocked_forward_equals_plain_and_binding_restored(self) -> None:
        self.assertEqual(NC_MAX_NNZ, 50_000_000)
        g = torch.Generator().manual_seed(0)
        n = 40
        row = torch.repeat_interleave(torch.arange(n), torch.randint(0, 15, (n,), generator=g))
        col = torch.randint(0, n, (row.numel(),), generator=g)
        value = torch.randn(row.numel(), generator=g)
        x0 = torch.randn(n, 5, generator=g)
        max_nnz = 30
        self.assertGreaterEqual(
            len(row_blocks(SparseTensor(row=row, col=col, value=value, sparse_sizes=(n, n)).storage.rowptr(),
                           max_nnz)), 3)

        model = _VendorLike()
        results = []
        for blocked in (False, True):
            v = value.clone().requires_grad_(True)
            A = SparseTensor(row=row, col=col, value=v, sparse_sizes=(n, n))
            out = blocked_vendor_forward(model, A, x0, max_nnz=max_nnz) if blocked else model(A, x0)
            out.square().sum().backward()
            results.append((out.detach(), v.grad))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        torch.testing.assert_close(results[1][1], results[0][1], atol=1e-6, rtol=0)
        self.assertIs(globals()["torch_sparse_matmul"], torch_sparse.matmul)  # binding undone


if __name__ == "__main__":
    unittest.main()
