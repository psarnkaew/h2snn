"""Which optional packages each baseline needs, and whether they are installed.

The campaign's third-party dependencies are split into groups under ``requirements/`` so
that one group failing to install costs only the baselines it backs. DGL, for instance,
has no Windows wheel for torch 2.6 and no PyPI release past 2.1.0; when it cannot be
installed, GCN/GAT/HAN/SeHGNN/HINormer are lost but the remaining eight baselines still
run. The Colab notebooks install one group per cell and use this module to report which
models each failure costs.

Nothing here imports an optional package. Availability is decided with
:func:`importlib.util.find_spec`, which locates a module without executing it -- the right
semantics for DGL in particular, whose Windows build imports only after
:mod:`src.baselines.runtime._dgl_compat` has stubbed out GraphBolt.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.baselines.protocol import LINK_PREDICTION
from src.baselines.runtime.vendor_paths import PROJECT_ROOT

#: Directory holding the per-group pin files that ``requirements.txt`` aggregates.
REQUIREMENTS_DIR: Path = PROJECT_ROOT / "requirements"


def _dgl_cuda_defect() -> str | None:
    """Why the installed DGL cannot use this machine's GPU, if that is both true and fixable.

    DGL ships separate CPU and CUDA builds, and ``find_spec("dgl")`` cannot tell them
    apart -- which is the whole problem this function exists for. A notebook that only
    asks "is dgl importable?" reports ``OK dgl already present``, skips installing the
    ``+cu124`` wheel, and then trains GCN/GAT/HAN/SeHGNN/HINormer on the CPU for hours on
    a machine with an idle GPU. Nothing raises; the numbers are even correct. Only the
    runtime is wrong, and only by a factor nobody notices until the campaign is over.

    Reported as a *defect* -- something a reinstall can fix -- only on Linux, where the
    CUDA wheel actually exists (``dgl-2.5.0+cu124-*-manylinux1_x86_64.whl`` on DGL's
    torch-2.6 index). On Windows no CUDA wheel is published for any torch 2.x, so a
    CPU-only build is the normal state of a local development environment and must not
    make the group look broken; :func:`dgl_cuda_report` still surfaces it there.
    """

    import torch

    if not torch.cuda.is_available():
        return None
    # Imported lazily and only once dgl itself resolves, so this module keeps its
    # promise not to import an optional package at module scope.
    from src.baselines.runtime._dgl_compat import dgl_supports_cuda

    if dgl_supports_cuda():
        return None
    if sys.platform != "linux":
        return None
    return (
        "the installed DGL build has no CUDA support while torch does; reinstall "
        "requirements/baselines-dgl.txt with --force-reinstall to get the +cu124 wheel"
    )


@dataclass(frozen=True, slots=True)
class RequirementGroup:
    """One installable group of optional pins and the import names it provides."""

    name: str
    filename: str
    #: Import names that must resolve for the group to count as installed. These are
    #: import names, not distribution names (``performer_pytorch``, not
    #: ``performer-pytorch``).
    modules: tuple[str, ...]
    summary: str
    #: Extra condition checked only once every module resolves. Returns why the group is
    #: present but unusable, or ``None`` when it is fine. Reserved for defects a
    #: reinstall can repair -- importable-but-wrong-build, not merely suboptimal -- since
    #: a non-``None`` result makes the group count as *not installed*.
    defect_check: Callable[[], str | None] | None = field(default=None, compare=False)

    @property
    def path(self) -> Path:
        return REQUIREMENTS_DIR / self.filename

    def missing_modules(self) -> tuple[str, ...]:
        """Import names in this group that cannot be located."""

        missing: list[str] = []
        for module in self.modules:
            try:
                found = importlib.util.find_spec(module) is not None
            except (ImportError, ValueError):
                # A parent package that is itself broken or absent raises rather than
                # returning None; either way the module is not usable.
                found = False
            if not found:
                missing.append(module)
        return tuple(missing)

    def defect(self) -> str | None:
        """Why this group is importable but unusable, or ``None``.

        Only meaningful once :meth:`missing_modules` is empty; a group that is not there
        at all has nothing to be defective about.
        """

        if self.defect_check is None or self.missing_modules():
            return None
        try:
            return self.defect_check()
        except Exception:  # noqa: BLE001 - a check that cannot run proves nothing
            return None

    def is_installed(self) -> bool:
        """Whether this group is present *and* usable.

        A wrong-build install counts as not installed, so the notebooks' install step
        reinstalls it instead of printing ``already present`` and moving on.
        """

        return not self.missing_modules() and self.defect() is None


#: Every optional group. ``core`` is deliberately absent: it is not optional, and a
#: notebook that cannot install it has nothing to degrade gracefully into.
REQUIREMENT_GROUPS: dict[str, RequirementGroup] = {
    "dgl": RequirementGroup(
        name="dgl",
        filename="baselines-dgl.txt",
        modules=("dgl",),
        summary="DGL graph/conv APIs used by the HGB-derived baselines",
        defect_check=_dgl_cuda_defect,
    ),
    "sheaf": RequirementGroup(
        name="sheaf",
        filename="baselines-sheaf.txt",
        modules=("torch_householder",),
        summary="Householder orthogonal maps used by the two sheaf baselines",
    ),
    "h2gb": RequirementGroup(
        name="h2gb",
        filename="baselines-h2gb.txt",
        modules=(
            "ogb",
            "yacs",
            "dill",
            "pytorch_lightning",
            "torchmetrics",
            "prettytable",
            "gdown",
            "performer_pytorch",
            "seaborn",
        ),
        summary="the H2GB graphgym stack behind H2G-former",
    ),
    "hetero2net": RequirementGroup(
        name="hetero2net",
        filename="baselines-hetero2net.txt",
        modules=("torch_cluster", "texttable", "termcolor"),
        summary="Hetero^2Net's metapath-dropping and logging dependencies",
    ),
}

#: Groups a model needs on every task it supports.
_MODEL_GROUPS: dict[str, tuple[str, ...]] = {
    "gcn": ("dgl",),
    "gat": ("dgl",),
    "han": ("dgl",),
    "sehgnn": ("dgl",),
    "hinormer": ("dgl",),
    "csnn": ("sheaf",),
    "nsd": ("sheaf",),
    "h2gformer": ("h2gb",),
    "hetero2net": ("hetero2net",),
    "chebnetii": (),
    "h2sgnn": (),
    "latgrl": (),
    # PSHGCN's node- and graph-classification paths import no optional package; only its
    # link-prediction vendor file does. See _BY_TASK below.
    "pshgcn": (),
}

#: Extra groups a model needs on one specific task only.
#:
#: PSHGCN/hgb-lp/pshgcn.py opens with `import dgl` -- a leftover, since it never calls
#: DGL -- so the link-prediction vendor import needs the package present even though the
#: model itself does not use it. PSHGCN/hgb/ (node classification) has no such import.
_BY_TASK: dict[tuple[str, str], tuple[str, ...]] = {
    ("pshgcn", LINK_PREDICTION): ("dgl",),
}


def groups_for(model: str, task: str) -> tuple[str, ...]:
    """Names of the requirement groups ``model`` needs in order to run ``task``."""

    if model not in _MODEL_GROUPS:
        known = ", ".join(sorted(_MODEL_GROUPS))
        raise KeyError(f"Unknown baseline {model!r}. Known: {known}.")
    combined = dict.fromkeys(_MODEL_GROUPS[model] + _BY_TASK.get((model, task), ()))
    return tuple(combined)


def missing_groups_for(model: str, task: str) -> tuple[str, ...]:
    """Groups ``model`` needs for ``task`` that are not installed right now."""

    return tuple(
        name
        for name in groups_for(model, task)
        if not REQUIREMENT_GROUPS[name].is_installed()
    )


def unavailable_reason(model: str, task: str) -> str | None:
    """Why ``model`` cannot run ``task`` in this environment, or ``None`` if it can.

    This reports only missing *packages*. A cell can still be unsupported for a modelling
    reason -- ``src.baselines.registry.unsupported_reason`` owns that question.
    """

    missing = missing_groups_for(model, task)
    if not missing:
        return None
    parts: list[str] = []
    for name in missing:
        group = REQUIREMENT_GROUPS[name]
        absent = group.missing_modules()
        # A group can be unavailable for two different reasons, and saying which one
        # saves the reader from checking whether the package is there by hand.
        parts.append(
            f"{name} ({', '.join(absent)})" if absent else f"{name} ({group.defect()})"
        )
    detail = "; ".join(parts)
    files = ", ".join(f"requirements/{REQUIREMENT_GROUPS[n].filename}" for n in missing)
    return f"missing requirement group {detail} -- install {files}"


def models_missing_dependencies(models: tuple[str, ...], task: str) -> dict[str, str]:
    """Map each model that cannot run ``task`` here to the reason, preserving order."""

    reasons: dict[str, str] = {}
    for model in models:
        reason = unavailable_reason(model, task)
        if reason is not None:
            reasons[model] = reason
    return reasons


def installed_groups() -> dict[str, bool]:
    """Map each optional group name to whether it is fully installed."""

    return {name: group.is_installed() for name, group in REQUIREMENT_GROUPS.items()}


def dgl_cuda_report() -> tuple[bool, str]:
    """``(uses_gpu, explanation)`` for the installed DGL build.

    Separate from :func:`installed_groups` because the answer matters on every platform,
    while only Linux treats a CPU-only build as a defect. The baselines this affects --
    GCN, GAT, HAN, SeHGNN, HINormer -- route their device through
    ``src.baselines.runtime._dgl_compat.resolve_dgl_device``, which silently returns CPU
    when DGL cannot reach the GPU. A campaign should decide what to do about that before
    it starts, not discover it in the timings afterwards.
    """

    import torch

    if REQUIREMENT_GROUPS["dgl"].missing_modules():
        return False, "DGL is not installed"
    if not torch.cuda.is_available():
        return False, "torch reports no CUDA device; nothing runs on GPU here"

    from src.baselines.runtime._dgl_compat import dgl_supports_cuda

    if dgl_supports_cuda():
        return True, "DGL has CUDA support"
    if sys.platform == "linux":
        return False, (
            "DGL is installed without CUDA support. The +cu124 wheel exists for this "
            "torch build, so this is a fixable install: reinstall "
            "requirements/baselines-dgl.txt with --force-reinstall."
        )
    return False, (
        "DGL is installed without CUDA support, and no CUDA wheel is published for "
        "torch 2.x on this platform. The DGL-backed baselines will run on CPU."
    )


def refresh() -> None:
    """Forget cached import lookups so a just-finished pip install is visible.

    ``find_spec`` consults :data:`sys.path_importer_cache`, which still holds the
    directory listing from before the install. Notebooks call this between installing a
    group and checking whether it took.
    """

    importlib.invalidate_caches()
