"""Cluster-setup3 CS8-2: the pinned CPU kernel environment E for the bit-exact tests.

The cluster references (`docs/heads_evidence/v2_golden_cluster.pt`) were captured with
E = ATEN_CPU_CAPABILITY=avx2, MKL_CBWR=AVX2, ONEDNN_MAX_CPU_ISA=AVX2 and 1 thread; CS8-1 showed that
such a capture is bit-identical on the cluster's AVX2 and AVX512 nodes (rule R2). Without E a node
picks its own kernels and the comparison fails silently on small differences, so a bit-exact test
first asserts that E is in effect and then compares with 1 thread (restored after).
"""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from typing import Iterator

import torch

PINNED_MESSAGE = ("bit-exact test needs the pinned kernel environment: "
                  "run through scripts/slurm/run_tests.sbatch")


def assert_kernel_path_pinned(case: unittest.TestCase) -> None:
    capability = torch.backends.cpu.get_cpu_capability()
    case.assertTrue(
        capability == "AVX2" and os.environ.get("MKL_CBWR") == "AVX2",
        f"{PINNED_MESSAGE} (cpu_capability={capability}, MKL_CBWR={os.environ.get('MKL_CBWR')})")


@contextmanager
def one_thread() -> Iterator[None]:
    """Rule R2: the bit-exact comparison runs with 1 thread; the previous count is restored."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
