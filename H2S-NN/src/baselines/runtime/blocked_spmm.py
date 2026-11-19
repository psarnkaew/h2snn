"""Row-blocked ``torch_sparse.matmul`` for operators above the int32 thread-index limit.

torch_sparse 0.6.18's ``spmm_value_bw_kernel`` (the backward w.r.t. the sparse values of
``SPMMSum``) launches ``nnz * 32`` threads and indexes them with ``int``, so it faults with
an illegal memory access once ``nnz > 2**26 - 1`` (card 1e F0, ``p6_f0_diagnosis.md``).
H2SGNN's combined operator ``A = add(adjs_list, softmax(beta))`` on Freebase LP has 166M
non-zeros and its values carry a gradient (``beta`` is learnable), so it hits the limit.

:func:`blocked_matmul` computes the same product as ``torch_sparse.matmul(A, x)`` by
splitting ``A`` into consecutive row blocks of at most ``max_nnz`` non-zeros
(``SparseTensor.narrow`` on dim 0), multiplying each block with the same
``torch_sparse.matmul`` and concatenating the rows. Every output row reads only its own
row's entries, so the forward is the same per-row reduction; ``narrow`` slices the value
tensor, so autograd routes each block's value gradient into the full value gradient.
Operators at or below the limit go through the plain call unchanged.

:func:`bound_attribute` replaces a name in a vendor module's namespace for the duration
of a ``with`` block and restores it afterwards; the vendor file is not edited.
Card 1e F3.1 (author decision "1e: a").
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, MutableMapping

import torch

#: Largest nnz for which ``nnz * 32`` still fits a signed 32-bit thread index.
MAX_NNZ = 2**26 - 1

_MISSING = object()


def row_blocks(rowptr: torch.Tensor, max_nnz: int) -> list[tuple[int, int]]:
    """Consecutive ``(start, length)`` row ranges, each holding at most ``max_nnz`` nnz.

    Greedy: each block takes as many whole rows as fit. A single row with more than
    ``max_nnz`` entries cannot be split by rows and raises (card 1e F3.1: stop item).
    """

    if max_nnz < 1:
        raise ValueError(f"max_nnz must be positive, got {max_nnz}")
    rowptr = rowptr.detach().to("cpu", torch.int64)
    num_rows = rowptr.numel() - 1
    blocks: list[tuple[int, int]] = []
    start = 0
    while start < num_rows:
        limit = int(rowptr[start]) + max_nnz
        # Last row boundary not beyond the limit.
        end = int(torch.searchsorted(rowptr, torch.tensor(limit), right=True)) - 1
        end = min(end, num_rows)
        if end <= start:
            row_nnz = int(rowptr[start + 1] - rowptr[start])
            raise RuntimeError(
                f"row {start} has {row_nnz} non-zeros, above max_nnz={max_nnz}; "
                "a row-blocked product cannot split it"
            )
        blocks.append((start, end - start))
        start = end
    return blocks


def blocked_matmul(
    src: Any, other: Any, reduce: str = "sum", max_nnz: int = MAX_NNZ
) -> Any:
    """``torch_sparse.matmul(src, other, reduce)``, row-blocked when ``src.nnz() > max_nnz``."""

    from torch_sparse import SparseTensor
    from torch_sparse import matmul

    if (
        not isinstance(src, SparseTensor)
        or not isinstance(other, torch.Tensor)
        or src.nnz() <= max_nnz
    ):
        return matmul(src, other, reduce)

    blocks = row_blocks(src.storage.rowptr(), max_nnz)
    parts = [matmul(src.narrow(0, start, length), other, reduce) for start, length in blocks]
    return torch.cat(parts, dim=0)


@contextlib.contextmanager
def bound_attribute(
    namespace: MutableMapping[str, Any], name: str, replacement: Any
) -> Iterator[None]:
    """Bind ``namespace[name] = replacement`` inside the block; restore it on exit."""

    previous = namespace.get(name, _MISSING)
    namespace[name] = replacement
    try:
        yield
    finally:
        if previous is _MISSING:
            namespace.pop(name, None)
        else:
            namespace[name] = previous


__all__ = ["MAX_NNZ", "blocked_matmul", "bound_attribute", "row_blocks"]
