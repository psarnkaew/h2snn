"""Make DGL importable on this environment's torch build.

DGL 2.2.1 imports GraphBolt at module-import time and looks for a DLL named after the
installed PyTorch version. The `h2snn` environment runs PyTorch 2.6, but the Windows DGL
wheel only ships GraphBolt binaries up to PyTorch 2.3, so a bare ``import dgl`` raises::

    FileNotFoundError: Cannot find DGL C++ graphbolt library at ...
    site-packages/dgl/graphbolt/graphbolt_pytorch_2.6.0.dll

The HGB baselines (GCN, GAT, HAN) and SeHGNN use the graph and convolution APIs, never
GraphBolt, so pre-registering a placeholder module keeps the unused extension from
failing the import. Adapted from the same shim already used by the HGB link-prediction
baselines at HGB/LP/benchmark/methods/baseline/dgl_graphbolt_compat.py.

Import this module *before* any ``import dgl``:

    from src.baselines.runtime import _dgl_compat  # noqa: F401
    import dgl
"""

from __future__ import annotations

import functools
import logging
import sys
import types

logger = logging.getLogger(__name__)


def install() -> None:
    """Stub out ``dgl.graphbolt`` only if DGL cannot be imported without it.

    Unconditionally registering the stub would be wrong on a healthy install: DGL's
    ``__init__`` does ``from . import graphbolt``, and Python would hand it this empty
    module instead of the real one, silently disabling GraphBolt. Colab's
    ``dgl 2.5.0+cu124`` for torch 2.6 is such a healthy install, so the stub must not be
    applied there -- it is only for builds whose GraphBolt binary is missing, like the
    Windows ``dgl 2.2.1`` wheel used for local development.
    """

    if "dgl" in sys.modules:
        return
    try:
        import dgl  # noqa: F401
    except FileNotFoundError:
        # The specific failure this shim exists for: "Cannot find DGL C++ graphbolt
        # library at ...graphbolt_pytorch_<version>.dll".
        sys.modules.setdefault("dgl.graphbolt", types.ModuleType("dgl.graphbolt"))
        logger.info(
            "DGL's GraphBolt extension is missing for this torch build; stubbing it out. "
            "The vendored baselines use DGL's graph/conv APIs, not GraphBolt."
        )
    except Exception:  # noqa: BLE001 - any other import failure is not ours to paper over
        return


install()


def restore_legacy_message_functions() -> None:
    """Re-expose DGL message functions the vendored baselines were written against.

    DGL renamed ``dgl.function.copy_src`` to ``copy_u`` and ``copy_edge`` to ``copy_e``
    and dropped the old names. HINormer targets DGL 0.9.0 and calls ``fn.copy_src``
    (HINormer/model.py:67), so on DGL 2.x its relation encoder raises AttributeError.

    The replacements are the same functions under new names, so aliasing is exact rather
    than an approximation.
    """

    import dgl.function as fn

    for old, new in (("copy_src", "copy_u"), ("copy_edge", "copy_e")):
        if not hasattr(fn, old) and hasattr(fn, new):
            setattr(fn, old, getattr(fn, new))


@functools.lru_cache(maxsize=1)
def dgl_supports_cuda() -> bool:
    """Whether the installed DGL build has CUDA enabled.

    DGL ships separate CPU and CUDA wheels, and there is no CUDA wheel for torch 2.6 on
    Windows, so a local environment can easily end up with a CPU-only build even when
    torch itself has CUDA. Moving a graph then fails with "Device API cuda is not
    enabled" rather than anything that mentions the build.
    """

    import torch

    if not torch.cuda.is_available():
        return False
    try:
        import dgl

        dgl.graph(([0], [0])).to(torch.device("cuda"))
    except Exception:  # noqa: BLE001 - any failure means the build cannot do CUDA
        return False
    return True


def resolve_dgl_device(preferred: "torch.device") -> "torch.device":  # noqa: F821
    """Return ``preferred`` if DGL can use it, else fall back to CPU with a warning.

    The DGL-backed baselines (HGB's GCN/GAT/HAN, SeHGNN) are the only ones affected; the
    rest of the campaign still runs on the GPU.
    """

    import torch

    if preferred.type != "cuda" or dgl_supports_cuda():
        return preferred
    logger.warning(
        "Installed DGL build has no CUDA support; running this baseline on CPU. "
        "Install a CUDA DGL wheel matching your torch build to use the GPU."
    )
    return torch.device("cpu")
