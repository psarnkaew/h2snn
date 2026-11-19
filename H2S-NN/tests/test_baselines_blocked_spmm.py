"""Card 1e F3.1: the row-blocked torch_sparse product equals the plain one; the binding is undone."""

from __future__ import annotations

import unittest

import torch

try:
    import torch_sparse
    from torch_sparse import SparseTensor
except ImportError:  # pragma: no cover - the baseline env has it
    torch_sparse = None

from src.baselines.runtime.blocked_spmm import blocked_matmul, bound_attribute, row_blocks


@unittest.skipIf(torch_sparse is None, "torch_sparse not installed")
class BlockedMatmulTest(unittest.TestCase):
    def _operator(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        generator = torch.Generator().manual_seed(0)
        n = 50
        # Uneven row degrees (0..19), so the blocks differ in rows and nnz.
        degrees = torch.randint(0, 20, (n,), generator=generator)
        row = torch.repeat_interleave(torch.arange(n), degrees)
        col = torch.randint(0, n, (row.numel(),), generator=generator)
        value = torch.randn(row.numel(), generator=generator)
        return row, col, value, n

    def test_forward_exact_and_gradients_match_unblocked(self) -> None:
        row, col, value, n = self._operator()
        x0 = torch.randn(n, 8, generator=torch.Generator().manual_seed(1))
        max_nnz = 37
        blocks = row_blocks(
            SparseTensor(row=row, col=col, value=value, sparse_sizes=(n, n)).storage.rowptr(),
            max_nnz,
        )
        self.assertGreaterEqual(len(blocks), 3)
        self.assertGreater(len({length for _, length in blocks}), 1)  # uneven blocks

        results = []
        for blocked in (False, True):
            v = value.clone().requires_grad_(True)
            x = x0.clone().requires_grad_(True)
            A = SparseTensor(row=row, col=col, value=v, sparse_sizes=(n, n))
            out = (
                blocked_matmul(A, x, max_nnz=max_nnz)
                if blocked
                else torch_sparse.matmul(A, x)
            )
            weight = torch.randn(out.shape, generator=torch.Generator().manual_seed(2))
            (out * weight).sum().backward()
            results.append((out.detach(), v.grad, x.grad))

        (out_plain, gv_plain, gx_plain), (out_block, gv_block, gx_block) = results
        self.assertTrue(torch.equal(out_plain, out_block))
        torch.testing.assert_close(gv_block, gv_plain, atol=1e-6, rtol=0)
        torch.testing.assert_close(gx_block, gx_plain, atol=1e-6, rtol=0)

        # A single row above the limit cannot be split by rows.
        with self.assertRaises(RuntimeError):
            row_blocks(torch.tensor([0, 2, 10, 11]), 5)

    def test_binding_is_restored(self) -> None:
        sentinel = object()
        namespace = {"torch_sparse_matmul": sentinel}
        with bound_attribute(namespace, "torch_sparse_matmul", blocked_matmul):
            self.assertIs(namespace["torch_sparse_matmul"], blocked_matmul)
        self.assertIs(namespace["torch_sparse_matmul"], sentinel)

        with self.assertRaises(ValueError):
            with bound_attribute(namespace, "torch_sparse_matmul", blocked_matmul):
                raise ValueError("vendor forward failed")
        self.assertIs(namespace["torch_sparse_matmul"], sentinel)

        empty: dict = {}
        with bound_attribute(empty, "torch_sparse_matmul", blocked_matmul):
            pass
        self.assertEqual(empty, {})


if __name__ == "__main__":
    unittest.main()
