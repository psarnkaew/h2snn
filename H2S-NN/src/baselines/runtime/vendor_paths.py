"""Locate the cloned baseline repositories without installing them.

The baseline authors' code lives in sibling clones at the repository root
(``HGB/``, ``SeHGNN/``, ``PSHGCN/``, ...). None of them is pip-installable in place, and
several use flat intra-package imports (``from model import ...``, ``from utils import
...``) that only resolve when their own directory is on ``sys.path``.

Adapters call :func:`vendor_path` to get a directory and :func:`pushd_sys_path` to add it
for the duration of an import. Paths are added at the *front* so a vendor's ``utils.py``
cannot be shadowed by this project's ``src/utils`` package, and removed afterwards so two
vendors with same-named modules cannot leak into each other.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Directory that must be on sys.path for each vendor's own imports to resolve.
VENDOR_ROOTS: dict[str, Path] = {
    "hgb_nc": PROJECT_ROOT / "HGB" / "NC" / "benchmark",
    "hgb_nc_gnn": PROJECT_ROOT / "HGB" / "NC" / "benchmark" / "methods" / "GNN",
    "hgb_nc_han": PROJECT_ROOT / "HGB" / "NC" / "benchmark" / "methods" / "HAN",
    "hgb_lp": PROJECT_ROOT / "HGB" / "LP" / "benchmark",
    "hgb_lp_gnn": PROJECT_ROOT / "HGB" / "LP" / "benchmark" / "methods" / "GNN",
    "sehgnn": PROJECT_ROOT / "SeHGNN" / "hgb",
    # SeHGNN/hgb/utils.py does a bare `from data_loader import data_loader`, which
    # lives one level up in SeHGNN/data/.
    "sehgnn_data": PROJECT_ROOT / "SeHGNN" / "data",
    "pshgcn": PROJECT_ROOT / "PSHGCN" / "hgb",
    "pshgcn_lp": PROJECT_ROOT / "PSHGCN" / "hgb-lp",
    "h2sgnn": PROJECT_ROOT / "H2SGNN",
    "hetero2net": PROJECT_ROOT / "Hetero2Net",
    "chebnetii": PROJECT_ROOT / "ChebNetII" / "main",
    "nsd": PROJECT_ROOT / "neural-sheaf-diffusion",
    "latgrl": PROJECT_ROOT / "LatGRL" / "LatGRL",
    "h2gb": PROJECT_ROOT / "H2GB",
    # The authors' own HINormer, not H2GB's (whose transformer stack is commented out).
    "hinormer": PROJECT_ROOT / "HINormer",
    # CSNN lives in this repository as the legacy homogeneous baseline.
    "csnn": PROJECT_ROOT,
}

#: Vendors that ship a properly namespaced package rather than flat top-level modules.
#: These cannot collide with another vendor, so they are exempt from the eviction below --
#: and must be, because re-importing them re-runs their registration decorators. H2GB
#: raises ``KeyError: Module with 'GraphormerBias' already defined`` on a second import,
#: which would break any loop that fits the same model twice in one process.
NAMESPACED_VENDORS = frozenset({"h2gb"})


def vendor_path(name: str) -> Path:
    """Return the on-disk root for one vendored baseline, or raise if it is missing."""

    try:
        path = VENDOR_ROOTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(VENDOR_ROOTS))
        raise KeyError(f"Unknown vendor {name!r}. Known vendors: {known}.") from exc
    if not path.is_dir():
        raise FileNotFoundError(
            f"Baseline repository {name!r} is not present at {path}. Clone it at the "
            "repository root before running this adapter."
        )
    return path


def available_vendors() -> dict[str, bool]:
    """Map each vendor name to whether its clone is present. Used by the env-check cell."""

    return {name: path.is_dir() for name, path in VENDOR_ROOTS.items()}


def _module_root(module: object) -> Path | None:
    """The vendor root a loaded module came from, if any."""

    filename = getattr(module, "__file__", None)
    if not filename:
        return None
    try:
        resolved = Path(filename).resolve()
    except (OSError, ValueError):
        return None
    for root in VENDOR_ROOTS.values():
        # `csnn` maps to PROJECT_ROOT, which contains every other vendor; skip it here so
        # it does not claim modules that belong to a more specific root.
        if root == PROJECT_ROOT:
            continue
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return root
    return None


@contextlib.contextmanager
def pushd_sys_path(*names: str) -> Iterator[None]:
    """Temporarily make one or more vendor roots importable.

    Manipulating ``sys.path`` is not enough on its own. These repositories use flat
    top-level module names and several of them collide: PSHGCN and H2SGNN both ship
    ``processing.py``, ``model.py`` and ``dataset_loader.py``, and HGB and SeHGNN both
    ship ``utils``. Because ``sys.modules`` is keyed by bare module name, whichever vendor
    imported first would silently satisfy the second vendor's imports -- H2SGNN would get
    PSHGCN's ``load_dataset`` and fail on an arity mismatch, or worse, succeed with the
    wrong graph.

    So on entry any already-loaded module belonging to a *different* vendor is evicted,
    and on exit modules loaded from the pushed roots are evicted and the originals put
    back. Imports inside the block are therefore always resolved against the requested
    vendor.

    **Construct vendor classes inside the block, not just import them.** PyG's
    ``MessagePassing.__init__`` inspects its subclass's ``message`` signature and resolves
    the annotations through ``sys.modules[cls.__module__]``. If the class is imported here
    but instantiated after the block exits, that lookup raises ``KeyError`` on the evicted
    module name -- which is how ChebNetII's ``ChebnetII_prop`` first failed. NSD and CSNN
    subclass ``MessagePassing`` too.
    """

    roots = [vendor_path(name) for name in names]
    added = [str(root) for root in roots]
    protected = {vendor_path(name) for name in NAMESPACED_VENDORS if name in VENDOR_ROOTS}

    # Evict modules owned by other vendors so their names are free to be re-resolved.
    # Namespaced vendors are left alone: they cannot collide, and re-importing them would
    # re-run their registration side effects.
    displaced: dict[str, object] = {}
    for module_name, module in list(sys.modules.items()):
        owner = _module_root(module)
        if owner is not None and owner not in roots and owner not in protected:
            displaced[module_name] = module
            del sys.modules[module_name]

    before = set(sys.modules)
    for entry in reversed(added):
        sys.path.insert(0, entry)
    try:
        yield
    finally:
        for entry in added:
            with contextlib.suppress(ValueError):
                sys.path.remove(entry)
        # Drop whatever this block imported from the pushed roots, then restore the
        # modules we displaced, leaving sys.modules as we found it.
        for module_name in set(sys.modules) - before:
            owner = _module_root(sys.modules.get(module_name))
            if owner in roots and owner not in protected:
                del sys.modules[module_name]
        sys.modules.update(displaced)
