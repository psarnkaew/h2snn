"""Let vendor code written for older PyG run against PyG 2.8.

Several baselines reach into a convolution's parameters by name to apply their own
initialisation. PyG has since renamed those attributes:

* ``GCNConv.weight``  -> ``GCNConv.lin.weight``
* ``GATConv.lin_l``   -> ``GATConv.lin``

HGB's link-prediction models do exactly this
(``HGB/LP/benchmark/methods/GNN/GNN.py:48,86``), so on PyG 2.8 they raise
``AttributeError`` during construction.

Rather than delete their initialisation -- which would silently change the model -- this
module re-exposes the old names as properties pointing at the current parameters, so the
vendor's ``xavier_normal_`` still initialises exactly the tensors it was written to
initialise. The aliases are installed only for the duration of the context.
"""

from __future__ import annotations

import contextlib
import importlib
from typing import Iterator


def ensure_pyg_submodules() -> None:
    """Materialise lazily-loaded ``torch_geometric.nn`` submodules.

    PyG 2.8 gives ``torch_geometric.nn`` a ``__getattr__`` that resolves exported *names*
    but never binds its own submodules, so ``pyg_nn.conv.MessagePassing`` raises
    ``module 'torch_geometric.nn' has no attribute 'conv'``. Importing
    ``torch_geometric.nn.conv`` is not enough on its own -- it lands in ``sys.modules``
    while the attribute stays unbound (verified) -- so the attribute is set explicitly.

    H2GB relies on that attribute access in ``layer/gatedgcn_layer.py:10``, which makes a
    bare ``import H2GB`` fail outright. Binding the names changes no behaviour; it only
    restores the module layout the vendor code was written against.
    """

    import sys

    import torch_geometric.nn as pyg_nn

    for submodule in ("conv", "dense", "pool", "norm"):
        full_name = f"torch_geometric.nn.{submodule}"
        try:
            importlib.import_module(full_name)
        except ImportError:  # pragma: no cover - submodule set varies by PyG version
            continue
        if not hasattr(pyg_nn, submodule):
            setattr(pyg_nn, submodule, sys.modules[full_name])


@contextlib.contextmanager
def legacy_conv_attrs() -> Iterator[None]:
    """Temporarily restore pre-2.x PyG attribute names on GCNConv and GATConv."""

    from torch_geometric.nn import GATConv, GCNConv

    installed: list[tuple[type, str]] = []

    def add_alias(cls: type, old: str, resolve) -> None:
        if hasattr(cls, old):
            return
        setattr(cls, old, property(resolve))
        installed.append((cls, old))

    # GCNConv's learnable transform moved into a Linear submodule.
    add_alias(GCNConv, "weight", lambda self: self.lin.weight)
    # GATConv's source-side transform was `lin_l`; single-`lin` is the current form, with
    # `lin_src` as the alias when separate source/target transforms exist.
    add_alias(
        GATConv,
        "lin_l",
        lambda self: getattr(self, "lin", None) or self.lin_src,
    )

    try:
        yield
    finally:
        for cls, name in installed:
            delattr(cls, name)
