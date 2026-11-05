"""Generate one Colab notebook per (task, requirement group) lane.

A lane is every baseline in one task that needs the same optional dependency group, so
its notebook installs ``requirements/core.txt`` plus **exactly one**
``requirements/baselines-*.txt`` and nothing else. The node-classification DGL lane is
the deliberate exception: it carries its own PyTorch 2.4/CUDA 12.4 installation cell,
because DGL must be obtained from its matching torch-2.4 wheel index. That is the whole
point:

* **Install cost.** The single-notebook-per-task layout paid for every group on every
  run -- ``torch_householder`` compiling from an sdist and the nine-package H2GB stack
  included -- whether or not the models being launched needed them.
* **DGL's ceiling.** DGL publishes no wheel past torch 2.6 (its ``torch-2.7`` index is
  gone), which is exactly the version this project pins. Keeping the DGL-backed
  baselines in their own notebook is what makes it possible to move the other lanes to a
  newer torch later without carrying five models that cannot follow.
* **Session length.** The full campaign is on the order of thirteen thousand model fits.
  Lanes run on separate Colab runtimes concurrently; ``--unit_start``/``--unit_count``
  then partitions a single large lane further.

Splitting does *not* add dependency isolation -- ``src/baselines/runtime/dependencies.py``
and the per-group install already give that, and each notebook still drops the models of
a group that failed rather than raising.

Node classification generates a fixed subset rather than the full matrix; see
:data:`NODE_CLASSIFICATION_SUBSET`.

Usage::

    python scripts/generate_baseline_notebooks.py            # write notebooks/baselines/
    python scripts/generate_baseline_notebooks.py --check    # fail if any file is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.protocol import (
    GRAPH_CLASSIFICATION,
    LINK_PREDICTION,
    NODE_CLASSIFICATION,
)
from src.baselines.registry import DATASETS, models_for_task, unsupported_reason
from src.baselines.runtime.dependencies import REQUIREMENT_GROUPS, groups_for

#: Where the generated notebooks live. Their own directory because there are fourteen of
#: them and they are generated artefacts -- edit this script, never the .ipynb.
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "baselines"

#: The lane a model with no optional dependencies belongs to.
CORE_LANE = "core"

# DGL's torch-2.4/CUDA-12.4 wheels must live in a runtime separate from the project's
# torch-2.6 core. This narrow lane is therefore self-contained rather than reading any
# requirements file. Keep the URL in the generated cell in the exact form DGL documents.
DGL_TORCH24_LANE = f"{NODE_CLASSIFICATION}_dgl"
DGL_TORCH24_WHEEL_INDEX = "https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html"


def uses_dgl_torch24_stack(lane: "Lane") -> bool:
    """Whether *lane* needs the self-contained DGL torch-2.4 runtime."""

    return lane.slug == DGL_TORCH24_LANE

#: Node classification runs a fixed subset, not the full matrix. HINormer is ACM only.
#: Deliberately narrower than ``registry.DATASETS``: the other node-classification
#: baselines are either already covered by published-hyperparameter runs or out of scope
#: for the comparison this subset is for.
NODE_CLASSIFICATION_SUBSET: dict[str, tuple[str, ...]] = {
    "csnn": ("acm", "dblp", "imdb"),
    "nsd": ("acm", "dblp", "imdb"),
    "hinormer": ("acm",),
    "h2gformer": ("acm", "dblp", "imdb"),
    "latgrl": ("acm", "dblp", "imdb"),
}

#: Vendor clone each model reads, for the up-front presence check. Keyed by model because
#: the names differ (``h2gformer`` reads the ``h2gb`` clone) and a missing clone should
#: fail before anything trains rather than midway through a sweep.
MODEL_VENDORS: dict[str, tuple[str, ...]] = {
    "gcn": ("hgb_nc_gnn", "hgb_lp_gnn"),
    "gat": ("hgb_nc_gnn", "hgb_lp_gnn"),
    "han": ("hgb_nc_han",),
    "sehgnn": ("sehgnn", "sehgnn_data"),
    "hinormer": ("hinormer",),
    "csnn": ("csnn",),
    "nsd": ("nsd",),
    "h2gformer": ("h2gb",),
    "hetero2net": ("hetero2net",),
    "chebnetii": ("chebnetii",),
    "h2sgnn": ("h2sgnn",),
    "latgrl": ("latgrl",),
    "pshgcn": ("pshgcn", "pshgcn_lp"),
}

#: One line per model, shown above its launch cell so the reader knows what is running.
MODEL_NOTES: dict[str, str] = {
    "gcn": "HGB's GCN. Node classification goes through DGL; link prediction uses HGB's own PyG implementation.",
    "gat": "HGB's GAT. Same split as GCN: DGL for node classification, PyG for link prediction.",
    "han": "HGB's HAN. DGL on every task, so this one needs a CUDA DGL build throughout.",
    "sehgnn": "Precomputes meta-path features before the model exists, then trains a transformer over them.",
    "hinormer": "The authors' own repository, not H2GB's copy, whose transformer stack is commented out.",
    "csnn": "The legacy in-repo sheaf model. Reads the HGB graph through the shared HomogeneousView plus a per-type input projection.",
    "nsd": "Homogeneous sheaf model, same two adaptations as CSNN. `GeneralSheaf` scales as d^2 per edge -- drop it first if memory bites.",
    "h2gformer": "H2GB's GTModel. Rebuilds graphgym's global cfg per trial so a stale value cannot leak between trials.",
    "hetero2net": "Keeps its own disentanglement regulariser, added to the task loss in every batch as in its own trainer.",
    "chebnetii": "Polynomial spectral filter over HGB's summed, symmetrised graph.",
    "h2sgnn": "Learns per-meta-path polynomial filters with separately optimised temperature parameters.",
    "latgrl": "Trains an unsupervised encoder, then probes the frozen embeddings. Its wall-clock profile is unlike the others.",
    "pshgcn": "Positive-spectrum heterogeneous graph convolution; its link-prediction vendor file carries a leftover `import dgl`.",
}


@dataclass(frozen=True, slots=True)
class Lane:
    """Every cell of one task that shares one optional requirement group."""

    task: str
    group: str
    cells: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def slug(self) -> str:
        return f"{self.task}_{self.group}"

    @property
    def path(self) -> Path:
        return OUTPUT_DIR / f"{self.slug}.ipynb"

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(model for model, _ in self.cells)

    @property
    def num_cells(self) -> int:
        return sum(len(datasets) for _, datasets in self.cells)

    @property
    def vendors(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for model in self.models:
            for vendor in MODEL_VENDORS[model]:
                seen.setdefault(vendor, None)
        return tuple(seen)


def task_cells(task: str) -> dict[str, tuple[str, ...]]:
    """Model -> datasets for one task, honouring the node-classification subset."""

    if task == NODE_CLASSIFICATION:
        requested = NODE_CLASSIFICATION_SUBSET
    else:
        requested = {
            spec.name: DATASETS[task] for spec in models_for_task(task)
        }
    out: dict[str, tuple[str, ...]] = {}
    for model, datasets in requested.items():
        runnable = tuple(
            dataset
            for dataset in datasets
            if not unsupported_reason(model, task, dataset)
        )
        if runnable:
            out[model] = runnable
    return out


def build_lanes() -> list[Lane]:
    """Every (task, group) lane that has at least one runnable cell."""

    lanes: list[Lane] = []
    for task in (NODE_CLASSIFICATION, LINK_PREDICTION, GRAPH_CLASSIFICATION):
        by_group: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
        for model, datasets in task_cells(task).items():
            groups = groups_for(model, task)
            # A model needing two groups would belong to neither lane cleanly; none do
            # today, and this is where that assumption becomes visible if it changes.
            if len(groups) > 1:
                raise ValueError(
                    f"{model}/{task} needs several groups {groups}; the one-group-per-lane "
                    "layout cannot place it. Give it its own lane explicitly."
                )
            by_group.setdefault(groups[0] if groups else CORE_LANE, []).append(
                (model, datasets)
            )
        for group, cells in sorted(by_group.items()):
            lanes.append(Lane(task=task, group=group, cells=tuple(sorted(cells))))
    return lanes


# --------------------------------------------------------------------------------------
# Notebook cells
# --------------------------------------------------------------------------------------


def md(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str) -> list[str]:
    """Notebook sources are line lists, each keeping its trailing newline but the last."""

    body = text.strip("\n")
    parts = body.split("\n")
    return [line + "\n" for line in parts[:-1]] + [parts[-1]]


def dgl_torch24_install_cell() -> str:
    """Install the node-classification DGL runtime without requirements files."""

    return f"""
# This lane is intentionally self-contained. DGL's CUDA 12.4 wheel must match torch
# 2.4, so do not install requirements/core.txt (which pins this project to torch 2.6).
# The DGL command below deliberately uses DGL's torch-2.4/cu124 wheel repository.
import importlib.metadata
import os
import signal
import subprocess
import sys

TORCH_INDEX = 'https://download.pytorch.org/whl/cu124'
PYG_WHEEL_INDEX = 'https://data.pyg.org/whl/torch-2.4.0+cu124.html'
DGL_WHEEL_INDEX = {DGL_TORCH24_WHEEL_INDEX!r}

try:
    installed_torch = importlib.metadata.version('torch')
except importlib.metadata.PackageNotFoundError:
    installed_torch = ''
target_ready = installed_torch.startswith('2.4.') and '+cu124' in installed_torch
print('installed torch distribution:', installed_torch or 'not installed')

commands = [
    [sys.executable, '-m', 'pip', 'install', '--upgrade',
     'tqdm', 'numpy', 'scipy', 'pyyaml', 'scikit-learn', 'wandb', 'pandas',
     'umap-learn', 'matplotlib'],
    [sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall',
     '--no-cache-dir', '--index-url', TORCH_INDEX, 'torch==2.4.*'],
    [sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall',
     '--no-cache-dir', '-f', PYG_WHEEL_INDEX,
     'torch_geometric', 'torch_scatter', 'torch_sparse'],
    # Required exact DGL wheel source for torch 2.4 + CUDA 12.4.
    [sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall',
     '--no-cache-dir', 'dgl', '-f', DGL_WHEEL_INDEX],
]

for command in commands:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError(f'Install failed with exit code {{result.returncode}}: {{" ".join(command)}}')
print('PyTorch 2.4/CUDA 12.4, PyG, DGL, and notebook dependencies installed.')

if not target_ready:
    print('PyTorch was replaced. Restarting; rerun this notebook from the first cell.')
    if IN_COLAB:
        os.kill(os.getpid(), signal.SIGKILL)
    raise RuntimeError('Restart Python, then rerun the notebook from the first cell.')
"""


def dgl_torch24_validation_cell() -> str:
    """Validate the ABI expected by the self-contained DGL lane."""

    return """
# Confirm the compiled extensions after the restart requested by the install cell.
import torch
import torch_geometric
import torch_scatter  # noqa: F401  (proves the compiled torch ABI matches)
import torch_sparse  # noqa: F401

if not torch.__version__.startswith('2.4.'):
    raise RuntimeError(
        f'Expected torch 2.4.*, found {torch.__version__}. Rerun the install cell.'
    )
if torch.version.cuda != '12.4':
    raise RuntimeError(f'Expected the CUDA 12.4 build, found CUDA {torch.version.cuda}.')

from src.baselines.runtime.dependencies import REQUIREMENT_GROUPS

GROUP = REQUIREMENT_GROUPS['dgl']
GROUP_OK = GROUP.is_installed()
if not GROUP_OK:
    raise RuntimeError('DGL did not install from the torch-2.4/cu124 wheel repository.')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch:', torch.__version__, '| pyg:', torch_geometric.__version__, '| device:', DEVICE)
if DEVICE == 'cuda':
    properties = torch.cuda.get_device_properties(0)
    print('GPU:', properties.name, '|', round(properties.total_memory / 1024**3, 1), 'GiB')
else:
    print('\\nWARNING: no GPU on this runtime. Runtime > Change runtime type > GPU.')
"""


def install_cell(lane: Lane) -> str:
    if lane.group == CORE_LANE:
        return """
# This lane needs nothing beyond core.txt, so there is no optional group to install.
# The check still runs: it reports what happens to be present without depending on it.
from src.baselines.runtime.dependencies import installed_groups

for name, present in sorted(installed_groups().items()):
    print(f'{"present" if present else "absent ":<8} {name}')
print('\\nThis lane requires none of them.')
"""
    return f"""
# One optional group, installed on its own. A failure is reported rather than raised;
# the models it backs are dropped further down, before anything launches.
import subprocess
import sys

from src.baselines.runtime.dependencies import REQUIREMENT_GROUPS, refresh

GROUP = REQUIREMENT_GROUPS[{lane.group!r}]


def install_group(group, force: bool = False) -> tuple[bool, str, list[str]]:
    base = [sys.executable, '-m', 'pip', 'install', '--upgrade', '-r', str(group.path)]
    if force:
        base += ['--force-reinstall', '--no-cache-dir']
    result = subprocess.run(base, capture_output=True, text=True)
    refresh()
    if group.is_installed():
        return True, 'installed' + (' (--force-reinstall)' if force else ''), []
    # torch_householder ships only an sdist whose setup.py imports torch at build time;
    # pip's default build isolation hides the installed torch from it.
    retry = subprocess.run(base + ['--no-build-isolation'], capture_output=True, text=True)
    refresh()
    if group.is_installed():
        return True, 'installed (--no-build-isolation)', []
    stderr = retry.stderr or result.stderr
    return False, 'failed', [line for line in stderr.splitlines() if line.strip()][-3:]


if GROUP.is_installed():
    print(f'OK    {{GROUP.name}} already present and usable')
    GROUP_OK = True
else:
    # `is_installed()` is not just an import check: a group that resolves but is the
    # wrong build counts as absent, which is what forces the reinstall below instead of
    # printing 'already present' over a broken install.
    defect = GROUP.defect() if not GROUP.missing_modules() else None
    if defect:
        print(f'DEFECT {{GROUP.name}}: {{defect}}')
        print('       reinstalling from scratch...')
    GROUP_OK, detail, tail = install_group(GROUP, force=bool(defect))
    if GROUP_OK:
        print(f'OK    {{GROUP.name}} {{detail}} - {{GROUP.summary}}')
    else:
        print(f'FAIL  {{GROUP.name}} - {{GROUP.summary}}')
        print(f'      missing: {{", ".join(GROUP.missing_modules()) or GROUP.defect()}}')
        for line in tail:
            print(f'      {{line[:160]}}')
"""


def gate_cell(lane: Lane) -> str:
    """Stop here if the lane cannot do useful work, before a session is spent on it.

    Three shapes, because the lanes genuinely differ: the core lane has no group to
    verify and must not reference one, every other lane needs its group to have
    installed, and the DGL lane additionally needs that group to be the CUDA build.
    """

    if lane.group == CORE_LANE:
        return """
# Nothing to gate: every model in this lane runs on core.txt alone, so there is no
# optional group whose absence could stop it. The install-status variables the other
# lanes check here do not exist in this one, by design.
print('Lane ready: no optional dependency group required.')
"""
    if lane.group != "dgl":
        return """
# Confirm the lane is usable before spending a session on it.
if not GROUP_OK:
    raise RuntimeError(
        f'{GROUP.name} did not install, and every model in this lane needs it. '
        'Fix the pip failure above; there is nothing here that can run without it.'
    )
print('Lane ready.')
"""
    return """
# DGL ships separate CPU and CUDA builds and `import dgl` cannot tell them apart. When
# the CPU build is the one present, `resolve_dgl_device` returns CPU without raising and
# every model in this lane trains on the CPU of a GPU machine -- correct numbers, wrong
# hardware, no error. On Linux the +cu124 wheel exists, so that is a broken install and
# this cell refuses to continue rather than burn a session discovering it in the timings.
import sys

from src.baselines.runtime.dependencies import dgl_cuda_report

if not GROUP_OK:
    raise RuntimeError(
        f'{GROUP.name} did not install, and every model in this lane needs it.'
    )

uses_gpu, explanation = dgl_cuda_report()
print(('OK   ' if uses_gpu else 'WARN '), explanation)
if not uses_gpu and DEVICE == 'cuda' and sys.platform == 'linux':
    raise RuntimeError(
        'Refusing to run the DGL lane on CPU while a GPU is present. '
        + explanation
    )
print('Lane ready.')
"""


def build_notebook(lane: Lane) -> dict[str, Any]:
    """One lane's notebook, from mount to aggregation."""

    specialized_dgl = uses_dgl_torch24_stack(lane)
    group_line = (
        "no optional dependencies"
        if lane.group == CORE_LANE
        else "a self-contained PyTorch 2.4/CUDA 12.4 + DGL installation cell"
        if specialized_dgl
        else f"`requirements/baselines-{lane.group}.txt` -- {REQUIREMENT_GROUPS[lane.group].summary}"
    )
    roster = "\n".join(
        f"- **{model}** on {', '.join(datasets)} -- {MODEL_NOTES[model]}"
        for model, datasets in lane.cells
    )
    cells: list[dict[str, Any]] = [
        md(
            f"""
# Baselines - {lane.task.replace('_', ' ').title()} - `{lane.group}` lane

One lane of the baseline campaign: every {lane.task.replace('_', ' ')} model that needs
{group_line}.

{roster}

**{lane.num_cells} cells.** Lanes are independent, so this notebook installs only its own
group and can run on its own Colab runtime alongside the others. To split this lane
further across machines, keep `NUM_RUNS` and `BASE_SEED` fixed and give each machine a
disjoint `UNIT_START`/`UNIT_COUNT` range; never run the same unit index twice.

Generated by `scripts/generate_baseline_notebooks.py` - edit that, not this file.
"""
        ),
        md("## 0. Environment"),
        code(
            """
from pathlib import Path
import os
import sys

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_DIR = Path('/content/drive/MyDrive/H2S-NN_test')
else:
    PROJECT_DIR = Path.cwd()

os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))
print('Working directory:', PROJECT_DIR)
"""
        ),
        md(
            """
This lane installs its own PyTorch 2.4/CUDA 12.4, PyG, and DGL dependencies directly
in the next cell. It does not use `requirements.txt` or `requirements/core.txt`: those
pin torch 2.6, which is incompatible with the required DGL wheel repository. A fresh
Colab runtime restarts after torch is replaced; **rerun from the first cell once it
restarts**.
"""
            if specialized_dgl
            else """
Core first. On a fresh Colab runtime this replaces Colab's own torch, so the cell kills
the runtime afterwards; **rerun from the first cell once it restarts**. Where the pinned
versions are already present it is a no-op and execution continues.
"""
        ),
        code(
            dgl_torch24_install_cell()
            if specialized_dgl
            else """
import importlib.metadata
import os
import signal
import subprocess
import sys

CORE_REQUIREMENTS = PROJECT_DIR / 'requirements' / 'core.txt'
if not CORE_REQUIREMENTS.exists():
    raise FileNotFoundError(f'Missing pinned requirements: {CORE_REQUIREMENTS}')

try:
    installed_torch = importlib.metadata.version('torch')
except importlib.metadata.PackageNotFoundError:
    installed_torch = ''
target_ready = installed_torch.startswith('2.6.') and '+cu124' in installed_torch
print('installed torch distribution:', installed_torch or 'not installed')

core = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--upgrade', '-r', str(CORE_REQUIREMENTS)],
    capture_output=True,
    text=True,
)
if core.returncode != 0:
    print(core.stdout[-3000:])
    print(core.stderr[-3000:])
    raise RuntimeError(f'Core install failed with exit code {core.returncode}.')
print('Core stack installed.')

if not target_ready:
    print('Pinned Torch/CUDA stack installed. Restarting; rerun from the first cell.')
    if IN_COLAB:
        os.kill(os.getpid(), signal.SIGKILL)
    raise RuntimeError('Restart Python, then rerun the notebook from the first cell.')
"""
        ),
        code(
            dgl_torch24_validation_cell()
            if specialized_dgl
            else """
# The cell above restarts the runtime after replacing Colab's torch. If that rerun did
# not happen, everything below still imports -- and then fails much later, inside a sweep
# worker, with an error about a symbol rather than about the torch version. This is the
# cheap place to catch it, which is why it runs in every lane.
#
# The version strings alone do not prove much: torch_scatter and torch_sparse are
# compiled against one specific torch ABI, and a mismatched build passes a string
# comparison and then dies on import. So import them. Both are in core.txt, so every
# lane has them, and the core lanes genuinely use them -- H2SGNN's link-prediction path
# and the typed graph-classification path both go through torch_sparse.
import torch
import torch_geometric
import torch_scatter  # noqa: F401  (imported to prove the compiled ABI matches)
import torch_sparse  # noqa: F401

if not torch.__version__.startswith('2.6.'):
    raise RuntimeError(
        f'Expected torch 2.6.*, found {torch.__version__}. The install cell above '
        'restarts the runtime; rerun this notebook from the first cell.'
    )
if torch.version.cuda != '12.4':
    raise RuntimeError(f'Expected the CUDA 12.4 build, found CUDA {torch.version.cuda}.')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch:', torch.__version__, '| pyg:', torch_geometric.__version__, '| device:', DEVICE)
if DEVICE == 'cuda':
    properties = torch.cuda.get_device_properties(0)
    print('GPU:', properties.name, '|', round(properties.total_memory / 1024**3, 1), 'GiB')
else:
    # Worth stopping for in any lane: this notebook is about to launch hundreds of fits,
    # and a CPU-only Colab runtime turns that into days. Change the runtime type.
    print()
    print('WARNING: no GPU on this runtime. Runtime > Change runtime type > GPU.')
"""
        ),
        code(
            """# DGL was installed directly in the self-contained environment cell above.
# Its availability and CUDA ABI are checked in the following cell.
"""
            if specialized_dgl
            else install_cell(lane)
        ),
        code(gate_cell(lane)),
        code(
            f"""
# The vendored baselines are cloned under vendor/, not pip-installed. Check the
# ones this lane reads before anything trains.
from src.baselines.runtime import available_vendors

REQUIRED_VENDORS = {list(lane.vendors)!r}
present = available_vendors()
for name in REQUIRED_VENDORS:
    print(('OK      ' if present.get(name) else 'MISSING '), name)
missing = [name for name in REQUIRED_VENDORS if not present.get(name)]
if missing:
    raise RuntimeError(f'Clone these under vendor/ first: {{missing}}')
"""
        ),
        code(
            """
import wandb

wandb_key = None
if IN_COLAB:
    try:
        from google.colab import userdata
        wandb_key = userdata.get('WANDB_API_KEY')
    except Exception:
        wandb_key = None

if wandb_key:
    wandb.login(key=wandb_key, relogin=True)
else:
    wandb.login()
"""
        ),
        md("## 1. Campaign settings"),
        code(
            f"""
TASK = {lane.task!r}
LANE = {lane.group!r}

# (model, datasets) for this lane only.
CELLS = {_format_cells(lane.cells)}

N_TRIALS = 30
NUM_RUNS = 5
BASE_SEED = 42

# Units are seeds. Give each machine a disjoint range; never run one index twice.
UNIT_START = 0
UNIT_COUNT = None

RESUME = False
OUTPUT_DIR = Path('experiments/baselines')
WANDB_PROJECT = 'h2snn_baseline'
WANDB_MODE = 'online'
WANDB_TAGS = ['baselines', TASK, f'lane-{{LANE}}']

# Epochs and patience are protocol, not settings: they come from
# src/baselines/budget.py so every model in a table trains under the same budget.
# Passing --epochs here would silently override that for this lane only.
from src.baselines.budget import default_budget

budget = default_budget(TASK)
print(f'{{sum(len(d) for _, d in CELLS)}} cells x {{NUM_RUNS}} seeds x {{N_TRIALS}} trials')
print(f'budget: {{budget.epochs}} epochs, patience {{budget.patience}}')
"""
        ),
        md(
            """
Drop any model whose requirement group did not install. This happens **before** the
launch cells, so a missing library costs those cells and nothing else, and it is
reported rather than silent.
"""
        ),
        code(
            """
from src.baselines.runtime.dependencies import models_missing_dependencies

requested = [model for model, _ in CELLS]
blocked = models_missing_dependencies(tuple(requested), TASK)
CELLS = [(model, datasets) for model, datasets in CELLS if model not in blocked]

for name, reason in blocked.items():
    print(f'SKIP  {name}: {reason}')
print(f'{len(CELLS)} of {len(requested)} runnable: ' + ', '.join(m for m, _ in CELLS))
if not CELLS:
    raise RuntimeError('No runnable models in this lane; fix the install failure above.')
"""
        ),
        md(
            """
## 2. Scope

Every cell must be runnable, and each is labelled with whether its authors published
settings for that exact dataset. Published cells go through `run_baseline_published.py`
in section 3; the rest are tuned in section 4.
"""
        ),
        code(
            """
from src.baselines.registry import get_adapter, unsupported_reason

PUBLISHED, TUNED = [], []
for model, datasets in CELLS:
    adapter = get_adapter(model)
    for dataset in datasets:
        reason = unsupported_reason(model, TASK, dataset)
        assert reason is None, f'{model}/{dataset} is not runnable: {reason}'
        published = adapter.published_params(TASK, dataset)
        space = adapter.search_space(TASK, dataset)
        (PUBLISHED if published else TUNED).append((model, dataset))
        status = 'published' if published else 'tune'
        print(f'{model:<12}{dataset:<10}{len(space):>3} params  {status}')
print(f'\\n{len(PUBLISHED)} published, {len(TUNED)} to tune')
"""
        ),
        md(
            """
## 3. Published-hyperparameter runs

Cells whose authors published settings for this exact dataset. No sweep: five seeds at
their values, using **their** epoch budget rather than the campaign's, because
reproducing a reported number means reproducing the budget it was reported at. Writes
`published_summary.json`, so it never overwrites a tuned result.
"""
        ),
        code(
            """
import subprocess
import sys

if PUBLISHED:
    by_model = {}
    for model, dataset in PUBLISHED:
        by_model.setdefault(model, []).append(dataset)
    for model, datasets in by_model.items():
        cmd = [
            sys.executable, 'scripts/run_baseline_published.py',
            '--task', TASK,
            '--models', model,
            '--datasets', *datasets,
            '--output_dir', str(OUTPUT_DIR),
            '--num_runs', str(NUM_RUNS),
            '--base_seed', str(BASE_SEED),
        ]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)
else:
    print('No published cells in this lane.')
"""
        ),
        md(
            """
## 4. Tuned runs

`--validate_only` first: it resolves every search space and prints the fit count without
running anything.
"""
        ),
        code(
            """
def tuning_command(model, datasets, validate=False):
    cmd = [
        sys.executable, 'scripts/run_baseline_tuning.py',
        '--task', TASK,
        '--models', model,
        '--datasets', *datasets,
        '--output_dir', str(OUTPUT_DIR),
        '--n_trials', str(N_TRIALS),
        '--num_runs', str(NUM_RUNS),
        '--base_seed', str(BASE_SEED),
        '--unit_start', str(UNIT_START),
        '--wandb_project', WANDB_PROJECT,
        '--wandb_mode', WANDB_MODE,
        '--wandb_tags', *WANDB_TAGS,
    ]
    if UNIT_COUNT is not None:
        cmd += ['--unit_count', str(UNIT_COUNT)]
    if RESUME:
        cmd.append('--resume')
    if validate:
        cmd.append('--validate_only')
    return cmd


TUNE_BY_MODEL = {}
for model, dataset in TUNED:
    TUNE_BY_MODEL.setdefault(model, []).append(dataset)

for model, datasets in TUNE_BY_MODEL.items():
    subprocess.run(tuning_command(model, datasets, validate=True), check=True)
"""
        ),
        md(
            """
One launch cell per model, so a failure costs one model rather than the lane. After a
Colab restart set `RESUME = True` in section 1 and re-run the cell: compatible per-seed
states resume their original sweeps, and incompatible ones are preserved while numbered
continuation sweeps are created.
"""
        ),
    ]

    for model, datasets in lane.cells:
        cells.append(
            md(
                f"### {model} - {', '.join(d.upper() for d in datasets)}\n\n{MODEL_NOTES[model]}"
            )
        )
        cells.append(
            code(
                f"""
model, datasets = {model!r}, TUNE_BY_MODEL.get({model!r}, [])
if datasets:
    cmd = tuning_command(model, datasets)
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)
else:
    print('{model}: nothing to tune in this lane (published cells ran in section 3).')
"""
            )
        )

    cells += [
        md(
            """
## 5. Aggregate

Collects whatever has finished into one CSV for this lane. Cells that have not run are
absent rather than filled in, so a gap stays visibly a gap.
"""
        ),
        code(
            """
import json

import pandas as pd

from src.utils.run_reports import report_metrics_for_task

metrics = report_metrics_for_task(TASK)
wanted = {f'{m}_{TASK}_{d}' for m, ds in CELLS for d in ds}

records = []
for pattern in ('*/tuning_summary.json', '*/published_summary.json'):
    for summary_path in sorted(OUTPUT_DIR.glob(pattern)):
        if summary_path.parent.name not in wanted:
            continue
        summary = json.loads(summary_path.read_text(encoding='utf-8'))['summary']
        model, _, dataset = summary_path.parent.name.partition(f'_{TASK}_')
        row = {
            'model': model,
            'dataset': dataset,
            'source': 'published' if 'published' in summary_path.name else 'tuned',
            'seeds': summary.get('num_units', 0),
        }
        for metric in metrics:
            entry = summary.get('metrics', {}).get(metric, {})
            row[metric] = entry.get('mean')
            row[f'{metric}_std'] = entry.get('std')
        records.append(row)

frame = pd.DataFrame(records)
results_path = Path('results') / f'baselines_{TASK}_{LANE}.csv'
results_path.parent.mkdir(parents=True, exist_ok=True)
frame.to_csv(results_path, index=False)
print(f'{len(frame)} of {len(wanted)} cells finished -> {results_path}')
frame
"""
        ),
        code(
            """
# Manuscript form: mean +/- std on the headline metric this task reports.
if not frame.empty:
    headline = metrics[0]
    shown = frame.assign(
        value=frame.apply(
            lambda r: f"{100 * r[headline]:.2f} +/- {100 * r[f'{headline}_std']:.2f}",
            axis=1,
        )
    )
    print(shown.pivot_table(index='model', columns='dataset', values='value',
                            aggfunc='first').to_string())
else:
    print('Nothing finished yet - run section 3 and 4 first.')
"""
        ),
        md(
            """
## 6. Resume and partition

Per-seed state lives under `OUTPUT_DIR/tuning_state/<cell>/`. Set `RESUME = True` in
section 1 and re-run the relevant launch cell to continue. `pruned` counts trials
Hyperband stopped early, which is a normal outcome and not a failure.
"""
        ),
        code(
            """
state_root = OUTPUT_DIR / 'tuning_state'
for state_path in sorted(state_root.glob('*/tuning_state_*.json')):
    if state_path.parent.name not in wanted or 'worker' in state_path.stem:
        continue
    state = json.loads(state_path.read_text(encoding='utf-8'))
    print(
        f"{state_path.parent.name:<40}{state_path.stem:<26}"
        f"done {len(state.get('records', []))}"
        f" attempted {state.get('attempted_trials', 0)}"
        f" pruned {state.get('pruned_trials', 0)}"
        f" failed {state.get('failed_trials', 0)}"
        f"  sweep {state.get('sweep_id')}"
    )
"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def _format_cells(cells: tuple[tuple[str, tuple[str, ...]], ...]) -> str:
    """The CELLS literal, column-aligned so the lane's scope reads at a glance."""

    width = max(len(repr(model)) for model, _ in cells)
    rows = "\n".join(
        f"    ({model!r:<{width}}, {list(datasets)!r})," for model, datasets in cells
    )
    return "[\n" + rows + "\n]"


def render(lane: Lane) -> str:
    return json.dumps(build_notebook(lane), indent=1, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed notebooks match this generator; write nothing",
    )
    parser.add_argument(
        "--lane",
        help="generate or check one lane slug, e.g. node_classification_dgl",
    )
    args = parser.parse_args()

    lanes = build_lanes()
    if args.lane:
        lanes = [lane for lane in lanes if lane.slug == args.lane]
        if not lanes:
            raise SystemExit(f"Unknown lane: {args.lane}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    for lane in lanes:
        payload = render(lane)
        if args.check:
            current = lane.path.read_text(encoding="utf-8") if lane.path.exists() else ""
            if current != payload:
                stale.append(lane.path.name)
            continue
        lane.path.write_text(payload, encoding="utf-8")
        print(f"{lane.path.relative_to(PROJECT_ROOT)}: {lane.num_cells} cells, "
              f"{', '.join(lane.models)}")

    if args.check:
        if stale:
            raise SystemExit(
                "Stale generated notebooks: "
                + ", ".join(stale)
                + "\nRun: python scripts/generate_baseline_notebooks.py"
            )
        print(f"{len(lanes)} notebooks match the generator.")
        return

    total = sum(lane.num_cells for lane in lanes)
    print(f"\n{len(lanes)} notebooks, {total} cells total.")


if __name__ == "__main__":
    main()
