# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **H2S-NN is mid-revision.** Before any task, read "Revision Status — Read This First" below.
> `src/` holds one model (vertex–2-cell sheaf Laplacian) with node, link and graph heads; v1 is
> on the git tag `v1-submitted`. The agreed design lives in `../docs/H2S-NN_v2_plan.md`.

## Environment — Always Use the `h2snn` Conda Environment

**Never run Python, tests, training, `pip`, or `conda` commands against the `base` conda environment or
the system/Windows-Store Python for this project.** A dedicated environment named `h2snn` (Python
3.12, CUDA 12.4 `torch==2.6.*`, `torch_geometric`, `torch_scatter`, `torch_sparse`,
`wandb`, and `scikit-learn`) already exists and is the only environment with the required libraries
installed. This applies no matter which shell or agent (Claude, GPT, Gemini, ...) is driving.

**Invoke everything through it — this form works identically from PowerShell, cmd, and Git Bash:**
```bash
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts/train_h2snn.py --config <path>
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python -m unittest discover -s tests -v
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python -m pip install <package>
```
In an interactive PowerShell session where conda hooks are initialized, `conda activate h2snn`
first and then call `python`/`pip` directly is equivalent.

Use `conda run` or activate the environment so PyTorch, PyG, CUDA, and compiled dependencies
resolve consistently. The current `src/` model does not depend on `torch_householder`; learned
orthogonal maps use the pure-PyTorch implementation in `src/models/fasthpp.py`.

Sanity-check the environment at any time with:
```bash
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python -c "import torch, torch_geometric, torch_scatter, torch_sparse; print(torch.__version__, torch.cuda.is_available())"
```

## Lab GPU cluster (Linux, SLURM)

The Windows commands above apply to the author's laptop. On the lab cluster (author, 2026-11-14;
`../docs/working_preferences.md`, "Server and GitHub workflow"):

- **Where things are.** Login node `login01` (Debian 12; `ssh peeradon@ai1.net.comp.isct.ac.jp`).
  Repository `/mnt/data/peeradon/h2snn` (project storage, NFS, readable by every lab member, no
  backup): work from `/mnt/data/peeradon/h2snn/H2S-NN`. Home `/mnt/home/peeradon` (50 GB, private,
  no backup): Miniforge, `~/.ssh`, the W&B key. **Secrets live only under home, never under
  `/mnt/data`.** Jobs never run in `$HOME`. `/tmp` on a compute node is fast scratch, deleted at
  job end (and not visible from the login node).
- **No computation on the login node** (tests, training, `pip`/`conda install`, preprocessing). It
  is for git, editing, submitting and monitoring. Compute nodes are reached only through SLURM.
- **Every job sets `--mem` and `--time`** (at most 72 h; longer work needs checkpoints), plus
  `--gres` and `--cpus-per-task`. GPU types: `--gres=gpu:<type>:<n>` with `2080ti` (11 GB),
  `titan_rtx` (24 GB), `3090` (24 GB), `a6000` (48 GB). Driver 550.163.01, CUDA 12.4.
- **Conda: the author's Miniforge, never `/opt/conda`.** Every job and shell runs
  `. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn` (env at `~/.conda/envs/h2snn`,
  Python 3.12). The lab's `/opt/conda` cannot create environments for this user: its `.condarc`
  locks `pkgs_dirs` (`#!final`) to a shared cache owned by other users. Never mix the two in one
  shell or script. SLURM jobs do not read `~/.bashrc`, so each script activates conda itself.
- **Version lock:** `requirements/linux-lock.txt`; `python scripts/setup/check_env.py` (in a job)
  prints every difference against it.
- **Tests:** the whole suite or given unittest targets in one GPU job (log in
  `logs/slurm-<jobid>.out`, git-ignored):
  ```bash
  cd /mnt/data/peeradon/h2snn/H2S-NN
  sbatch scripts/slurm/run_tests.sbatch                       # whole suite
  sbatch scripts/slurm/run_tests.sbatch tests.test_v2_model   # unittest targets
  ```
  Short interactive work: `scripts/slurm/interactive.md` (`srun ... --pty bash`).
- **Push rules (2026-11-14; supersede "do not commit or push" below for agent work branches):**
  the agent pushes its own work branch after each unit (`git push origin <branch>`); never `v3b`,
  never another branch, never force-push. `v3b` is fast-forwarded only after the author approves a
  report. Never commit vendor files without a licence (HGB), `data/`, secrets, `Claude outputs/`,
  `paper/`.

## Coding Style & Conventions for ML/DL Research Projects

---

### 1. Core Philosophy

- **Explicit over implicit.** Every function must be readable from its signature alone — no hidden
  reads from globals, no silent mutations of shared state.
- **No global variables.** Config, model, optimizer, logger — everything is passed explicitly
  through constructors or function arguments.
- **Reproducibility first.** Every experiment run must be fully reconstructable from its saved
  artifacts alone (`config.yaml` + `best.pt`).
- **Thin entry scripts.** `train.py` should only parse args, build objects, and call `trainer.fit()`.
  If it grows beyond ~40 lines, logic has leaked out of the right layer.
- **One responsibility per class.** `model.py` defines architecture. `trainer.py` runs the loop.
  `logger.py` records metrics. Never mix these.

---

### 2. Project Structure

```
project/
├── CLAUDE.md
├── README.md
├── requirements.txt
│
├── configs/                  # YAML config files only — no logic here
│   ├── base.yaml
│   └── experiments/
│       └── resnet_baseline.yaml
│
├── src/
│   ├── models/               # Architecture definitions only
│   │   ├── __init__.py
│   │   └── my_model.py
│   ├── data/                 # Dataset, DataLoader, transforms
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── trainers/             # Training and evaluation loops
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── losses/               # Loss function definitions
│   │   ├── __init__.py
│   │   └── losses.py
│   └── utils/                # Shared utilities
│       ├── __init__.py
│       ├── logger.py
│       ├── checkpointer.py
│       └── misc.py           # seed_everything, save_config, etc.
│
├── scripts/                  # Executable entry points
│   ├── train.py
│   ├── evaluate.py
│   └── sweep.py
│
├── tests/                    # Unit tests mirroring src/ structure
│   ├── test_models.py
│   ├── test_trainer.py
│   └── test_data.py
│
└── experiments/              # Auto-generated, gitignored
    └── resnet_baseline_seed42/
        ├── config.yaml
        ├── run.log
        ├── metrics.jsonl
        └── checkpoints/
            ├── best.pt
            └── last.pt
```

> `experiments/` must be in `.gitignore`. Never commit model weights or logs.

---

### 3. Configuration — Single Source of Truth

#### Rules
- Use a **dataclass** for all configs. Never use raw `dict` or inline magic numbers.
- Config is instantiated **once** in the entry script and passed down explicitly to every component.
- Config is always **saved to disk** at the start of every run.
- Hyperparameters hardcoded anywhere other than the config dataclass are forbidden.

#### `src/configs/config.py`

```python
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_dir: Path = Path("data/")
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    val_split: float = 0.1


@dataclass
class ModelConfig:
    name: str = "my_encoder"
    input_dim: int = 512
    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.1


@dataclass
class OptimConfig:
    name: str = "adam"           # "adam" | "sgd" | "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    scheduler: str = "cosine"    # "cosine" | "step" | "none"


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    epochs: int = 100
    seed: int = 42
    device: str = "cuda"

    exp_name: str = "baseline"
    output_dir: Path = Path("experiments/")

    @property
    def exp_dir(self) -> Path:
        return self.output_dir / self.exp_name
```

---

### 4. Models (`src/models/`)

### Rules
- A model file contains **only** architecture (`nn.Module` subclasses).
- No optimizer, no loss, no DataLoader, no logging inside model files.
- `forward()` takes tensors, returns a typed `NamedTuple` — never a bare tuple or dict.
- Private builder methods are prefixed with `_`.

#### `src/models/my_model.py`

```python
from typing import NamedTuple

import torch
import torch.nn as nn

from src.configs.config import ModelConfig


class ModelOutput(NamedTuple):
    logits: torch.Tensor
    features: torch.Tensor


class MyEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._build()

    def _build(self) -> None:
        self.encoder = nn.Sequential(
            nn.Linear(self.cfg.input_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
        )
        self.head = nn.Linear(self.cfg.hidden_dim, self.cfg.output_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        features = self.encoder(x)
        logits = self.head(features)
        return ModelOutput(logits=logits, features=features)
```

> **Why `NamedTuple` for output?**
> `out[0]` is an invisible bug waiting to happen when you add a third return value later.
> `out.logits` is always unambiguous.

---

### 5. Trainers (`src/trainers/trainer.py`)

### Rules
- `Trainer.__init__()` receives **all** dependencies explicitly — model, optimizer, criterion,
  logger, checkpointer, config.
- `fit()` is the only public method called from outside.
- `_train_epoch()` and `_eval_epoch()` are private. They take a loader and an epoch index.
  They return a `dict[str, float]` of metrics. They do not log, they do not checkpoint — they
  only compute and return.
- Logging and checkpointing happen in `fit()`, not inside the epoch methods.

#### `src/trainers/trainer.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.configs.config import TrainConfig
from src.utils.checkpointer import Checkpointer
from src.utils.logger import ExperimentLogger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        criterion: nn.Module,
        cfg: TrainConfig,
        logger: ExperimentLogger,
        checkpointer: Checkpointer,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.cfg = cfg
        self.logger = logger
        self.checkpointer = checkpointer
        self.device = torch.device(cfg.device)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        for epoch in range(self.cfg.epochs):
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self._eval_epoch(val_loader, epoch)

            # All side effects happen HERE — not buried in epoch methods
            self.logger.log(epoch, train_metrics, val_metrics)
            self.checkpointer.save(self.model, tag="last", meta=val_metrics)
            self.checkpointer.save_if_best(self.model, val_metrics)

            if self.scheduler is not None:
                self.scheduler.step()

    def _train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output.logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.optim.grad_clip,
            )
            self.optimizer.step()
            total_loss += loss.item()

        return {"train_loss": total_loss / len(loader)}

    def _eval_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output.logits, y)
                total_loss += loss.item()
                preds = output.logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return {
            "val_loss": total_loss / len(loader),
            "val_acc": correct / total,
        }
```

---

### 6. Information Keeping

#### 6a. Logger (`src/utils/logger.py`)

##### Rules
- Never use bare `print()` for experiment output.
- One logger class wraps both Python's `logging` (file + console) and optional W&B/TensorBoard.
- Saves `metrics.jsonl` — one JSON object per line. Append-safe and crash-tolerant.
- `metrics.jsonl` is trivially parseable: `pd.read_json("metrics.jsonl", lines=True)`.

```python
import json
import logging
from pathlib import Path


class ExperimentLogger:
    def __init__(self, exp_dir: Path) -> None:
        exp_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = exp_dir / "metrics.jsonl"
        self._setup_file_logger(exp_dir / "run.log")

    def _setup_file_logger(self, log_path: Path) -> None:
        fmt = "%(asctime)s | %(levelname)s | %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(),       # also prints to console
            ],
        )

    def log(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        # Append-safe: one record per line, survives mid-run crashes
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        summary = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
        logging.info(f"Epoch {epoch:04d} | {summary}")

    def info(self, message: str) -> None:
        logging.info(message)
```

#### 6b. Checkpointer (`src/utils/checkpointer.py`)

##### Rules
- Always save `state_dict` + metadata (epoch, metrics) together. Never weights alone.
- `best.pt` tracks the best validation metric. `last.pt` is always the most recent epoch.
- Mode is explicit: `"min"` for loss, `"max"` for accuracy/F1.

```python
from pathlib import Path

import torch
import torch.nn as nn


class Checkpointer:
    def __init__(
        self,
        exp_dir: Path,
        metric: str = "val_loss",
        mode: str = "min",     # "min" | "max"
    ) -> None:
        self.ckpt_dir = exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self._is_better = (
            (lambda new, best: new < best)
            if mode == "min"
            else (lambda new, best: new > best)
        )

    def save(
        self,
        model: nn.Module,
        tag: str,
        meta: dict,
    ) -> Path:
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
        return path

    def save_if_best(
        self,
        model: nn.Module,
        metrics: dict[str, float],
    ) -> bool:
        score = metrics[self.metric]
        if self._is_better(score, self.best_score):
            self.best_score = score
            self.save(model, tag="best", meta=metrics)
            return True
        return False

    def load(self, model: nn.Module, tag: str = "best") -> dict:
        path = self.ckpt_dir / f"{tag}.pt"
        payload = torch.load(path, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        return payload["meta"]
```

---

### 7. Utilities (`src/utils/misc.py`)

```python
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """Seed all RNGs for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config(cfg, path: Path) -> None:
    """Persist the config dataclass as YAML alongside the run artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)
```

---

### 8. Entry Script (`scripts/train.py`)

#### Rules
- **Only four responsibilities:** parse args → build config → instantiate all objects → call `fit()`.
- No training logic. No loss definitions. No dataset logic.
- Config is saved to disk before training starts — so a crash on epoch 1 still leaves a record.

```python
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.configs.config import TrainConfig, DataConfig, ModelConfig, OptimConfig
from src.data.dataset import build_dataloaders
from src.models.my_model import MyEncoder
from src.trainers.trainer import Trainer
from src.utils.checkpointer import Checkpointer
from src.utils.logger import ExperimentLogger
from src.utils.misc import seed_everything, save_config


def build_optimizer(
    model: nn.Module,
    cfg: TrainConfig,
) -> torch.optim.Optimizer:
    if cfg.optim.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
        )
    if cfg.optim.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {cfg.optim.name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if cfg.optim.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.optim.scheduler == "none":
        return None
    raise ValueError(f"Unknown scheduler: {cfg.optim.scheduler}")


def main(cfg: TrainConfig) -> None:
    # 1. Reproducibility
    seed_everything(cfg.seed)

    # 2. Create experiment directory and save config immediately
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg.exp_dir / "config.yaml")

    # 3. Build all components explicitly — nothing is global
    device = torch.device(cfg.device)

    model = MyEncoder(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = build_dataloaders(cfg.data)

    logger = ExperimentLogger(cfg.exp_dir)
    checkpointer = Checkpointer(cfg.exp_dir, metric="val_loss", mode="min")

    logger.info(f"Experiment: {cfg.exp_name}")
    logger.info(f"Output dir: {cfg.exp_dir}")

    # 4. Run
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        cfg=cfg,
        logger=logger,
        checkpointer=checkpointer,
    )
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    cfg = TrainConfig(**raw)
    if args.exp_name:
        cfg.exp_name = args.exp_name

    main(cfg)
```

**Run an experiment:**
```bash
python scripts/train.py --config configs/experiments/resnet_baseline.yaml --exp_name resnet_v1
```

---

### 9. Experiment Output Layout

Every run produces this structure automatically:

```
experiments/
└── resnet_v1/
    ├── config.yaml          ← exact config that produced this run
    ├── run.log              ← full timestamped console output
    ├── metrics.jsonl        ← one JSON record per epoch
    └── checkpoints/
        ├── best.pt          ← best val metric: {state_dict, meta}
        └── last.pt          ← most recent epoch: {state_dict, meta}
```

**Load results for analysis:**
```python
import pandas as pd
df = pd.read_json("experiments/resnet_v1/metrics.jsonl", lines=True)
```

**Reconstruct a run:**
```python
import yaml
from src.configs.config import TrainConfig

with open("experiments/resnet_v1/config.yaml") as f:
    cfg = TrainConfig(**yaml.safe_load(f))
```

---

### 10. Type Annotations

- All functions must have **full type annotations** on arguments and return value.
- Use `torch.Tensor`, not `Any`, for tensor arguments.
- Use `Path`, not `str`, for all file paths.
- Use `dict[str, float]` for metric dictionaries, not bare `dict`.
- Use `NamedTuple` for all multi-tensor model outputs.

```python
# ✅ Correct
def _eval_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]: ...

# ❌ Wrong
def _eval_epoch(self, loader, epoch): ...
```

---

### 11. Naming Conventions

| Concept                   | Convention              | Example                          |
|---------------------------|-------------------------|----------------------------------|
| Config dataclass          | `*Config`               | `TrainConfig`, `ModelConfig`     |
| Model output              | `*Output` (NamedTuple)  | `ModelOutput`                    |
| Private methods           | `_method_name`          | `_train_epoch`, `_build`         |
| Builder functions         | `build_*`               | `build_optimizer`, `build_model` |
| Experiment output dir     | `exp_dir / exp_name`    | `experiments/resnet_v1/`         |
| Checkpoint files          | `best.pt`, `last.pt`    | —                                |
| Metrics log               | `metrics.jsonl`         | one JSON record per line         |
| Config snapshot           | `config.yaml`           | saved at run start               |
| Log file                  | `run.log`               | full timestamped output          |

---

### 12. Absolute Rules — What Claude Must Never Do

| Rule | Reason |
|---|---|
| ❌ Never use `global` variables | Creates invisible dependencies between functions |
| ❌ Never hardcode hyperparameters inline | Cannot reproduce or sweep the experiment |
| ❌ Never put training logic in `model.py` | Violates single responsibility |
| ❌ Never use bare `dict` for config | No type safety, no IDE autocomplete, error-prone |
| ❌ Never use `print()` for experiment output | Use `ExperimentLogger.info()` instead |
| ❌ Never save weights without metadata | Cannot know what epoch or metric the checkpoint came from |
| ❌ Never read from a file path hardcoded as a string | Use `Path` and pass it through config |
| ❌ Never write results to a file inside a model or epoch method | Side effects belong in `fit()` only |
| ❌ Never omit type annotations | Makes code unreadable and untestable |
| ❌ Never commit `experiments/` to git | Use `.gitignore` |

## Revision Status — Read This First

H2S-NN is being revised after critical reviews. **`src/` holds one model: H2S-NN with the
vertex–2-cell sheaf Laplacian (plan §4), shared by three task heads (node classification, link
prediction, graph classification; plan §4.8, decision D6).** The submitted model (v1) and AMPSS
were deleted; they are preserved on the git tag `v1-submitted`. Never describe a planned component
that has no code as what the code does.

As of 2026-11-06:
- D1 + D1a: factorized, matrix-free, conformal-orthogonal restriction maps with one conformal
  scale per edge — `src/models/map_operators.py`, `src/models/sheaf_maps.py`; tests
  `tests/test_factorized_maps.py`, `tests/test_v2_model.py`.
- D4: the model and its operators — `src/models/h2snn_model.py`,
  `src/models/vertex_path_laplacian.py`; tests `tests/test_v2_model.py`. Report:
  `../docs/H2S-NN_D4_report.md`.
- D6: task heads — `src/models/link_readout.py`, `src/models/graph_readout.py`; tests
  `tests/test_task_heads.py` (incl. golden node-classification outputs).
No accuracy result exists yet for any task.

### Sources of truth, in this order

The planning documents live in `../docs/` (from `H2S-NN/`).

1. `CLAUDE.md` and `AGENTS.md` — environment and coding conventions. Binding.
2. `../docs/H2S-NN_v2_plan.md` — agreed decisions D1–D6. Do not reopen an agreed decision; if
   evidence contradicts one, stop and report.
3. `../docs/H2S-NN_fix_spec.md` — verified facts F1–F18, falsifiable hypotheses H1–H12, protocol
   fixes (§6) and writing decisions (§7). Still in force except where the v2 plan supersedes it
   (P2, P3, P5 and P6 are superseded or rejected; see plan §0).
4. `../docs/H2S-NN_agent_prompt.md` — role, evidence rules, the mandatory self-correction loop, and
   task cards. Task cards D1 and D4 are done (reports `../docs/H2S-NN_D1_report.md`,
   `../docs/H2S-NN_D4_report.md`).
5. The code. When a document and the code disagree, the code describes what runs. Report the
   disagreement; do not silently pick one.

Context documents in the same folder: `../docs/critical_reviewer_reviews.md` (the reviews),
`../docs/h2snn_manuscript_paper.pdf` and `../docs/h2snn_manuscript_material.pdf` (the submitted
version). The manuscript source is `paper/h2snn_manuscript.tex`.

**Advisor report (`../docs/advisor_report/`).** A progress report written for the author's advisor,
explaining the revision to a reader who does not work in the code. It is **not** a source of truth
and **must not be edited, regenerated or refreshed unless the author explicitly asks for an update**.
Code, plan or task-report changes are not a reason to touch it. When asked, follow
`../docs/advisor_report/README.md` (audience, structure, number traceability, rendering).

### Evidence tags

Tag every non-trivial claim with exactly one of: `VERIFIED-CODE` (give `file:line`),
`VERIFIED-RUN` (give the command and paste the decisive output), `VERIFIED-SOURCE` (a document you
actually opened), `DERIVED` (show the derivation), `UNVERIFIED` (state the check that would
resolve it), or `PROPOSAL`. Never upgrade a tag without new evidence. Never claim a test passes, a
number improved, or a bug is fixed without showing the run output.

### Decisions at a glance

| ID | Decision (details in `../docs/H2S-NN_v2_plan.md`) | State |
|---|---|---|
| D1 | Factorized, matrix-free, conformal-orthogonal restriction maps `F = c · R_{ψ,side} · H_k(u)`, never materialized at `[M, d, d]`; `model.map_family: factorized \| dense` (dense = comparison family), `model.householder_k`. D1a: one scale `c_e` per edge, shared by both sides | Implemented, unit-tested; accuracy experiments (plan §1.6) not run |
| D2 | WISP (spec v1.3 §3.2): Walk-sampled, Instance-weighted Sheaf Paths replace AMPSS — schema-free self-avoiding walks from a stated target measure, walk-level defensive importance sampling (ratio ≤ 2), `w_f = exp(g(ξ_f))` (γ removed), schemas read out (`I_π`, `I_σ`, `ΔL_σ`), never selected; knobs `ℓ_max` (support), `r` (precision). Default (spec v1.4 E7): `path_source: wisp` for node classification; `path_source: schemas` (fixed literature presets) is the comparison row | Agreed 2026-11-08; implementation: task card Part H, branch `wisp`. "AMPSS" = deleted v1 only |
| D3 | Scalability: normalized Laplacians with full-graph degrees, `η = σ(θ)`, per-batch path cells, `NeighborLoader` mini-batching | Normalization and bounded step implemented (D4); mini-batching not started |
| D4 | Path 2-cells couple to **every** boundary node through the **vertex–2-cell sheaf Laplacian** | Implemented with the §4.6 ablation switches; ACM memory and smoke run only |
| D5 | Baseline provenance (fix spec H8) and IMDB heterophily under fair multilabel definitions (H5); parallel track | Not started |
| D6 | One model; link prediction and graph classification ported to it (heads change only where forced) | Implemented; no accuracy run |

Order of work and gates: plan §6. Dataset ladder (HGB → Freebase / rewiring / IEEE-CIS-G →
ogbn-mag, mag-year, H-Pokec): plan §3.3.

### Rejected — do not build (plan §4.8)

A sheaf-consistency link decoder; per-relation / per-schema energy features for graph
classification; a disjoint message/supervision split for link prediction (the HGB protocol is
kept: training positives are both in the message graph and supervised).

### Hard rules for the revision

- Do not run long training jobs, sweeps or multi-seed experiments without explicit approval. Short
  smoke tests and microbenchmarks are fine.
- Results produced before the revision (v1 runs, old W&B sweeps, manuscript tables) do not
  validate the current model. Never quote them as its evidence.
- Label every baseline number rerun / published† / quoted‡ and trace it to an artifact (D5).
- No design conclusions from untuned cross-model comparisons. Our models are untuned until the
  cluster protocol; baselines (R-GCN, GCN, SimpleHGN, HetSheaf, ...) use published or tuned
  configurations. A gap between H2S-NN and another model (e.g. "feature-free R-GCN beats H2S-NN
  by 5-10 points") is not evidence that any H2S-NN component, design choice or hypothesis is
  wrong, and must not be presented as a deficit, a "real problem", a risk or a reason to change,
  delay or gate the design. Report such numbers only as context, labelled "untuned vs tuned; not
  a design signal". Design conclusions may come only from ablations of the same model under the
  same settings and budget (switch one component on/off, everything else identical), from
  known-answer tests, or from tuned comparisons under the frozen protocol.
- Config loading is strict: every section rejects unknown keys, so old v1/AMPSS configs and
  checkpoints fail loudly. Do not add compatibility shims for them.
- Forbidden fixes: loosening tolerances, deleting or skipping tests, catching and silencing
  exceptions, special-casing test inputs, editing the spec to fit the code. If you believe a
  tolerance or spec is wrong, stop and argue it in your report.
- Stay inside the task card's scope; list out-of-scope problems instead of fixing them.
- Do not commit or push unless asked.

## Project Overview

**H2S-NN** (Heterophilic Heterogeneous Sheaf-Hodge Neural Network) lifts a heterogeneous
information network (typed nodes, typed directed edges) into a 2-complex whose 2-cells are
path instances of meta-path schemas, places a cellular sheaf on it, and diffuses node states
with one normalized vertex–2-cell sheaf Laplacian. One encoder serves node classification, link
prediction and graph classification.

**Two codebases live in this repository — know which one you're touching:**
- **`src/`, `scripts/`, `configs/experiments/`, `tests/`** — H2S-NN.
- **`vendor/`** — every external baseline, never H2S-NN code. Do not add H2S-NN features here.
  - `vendor/csnn/` (`models/`, `exp/`, `utils/`, `gnn_evaluation.py`) — the earlier
    homogeneous-graph sheaf-diffusion baseline (`CSNN`), kept only for benchmark comparisons.
  - `vendor/HINPool/`, `vendor/HINormer/` — vendored code committed to this repository.
  - `vendor/H2GB/`, `SeHGNN/`, `HGB/`, `Hetero2Net/`, `LatGRL/`, `PSHGCN/`, `ChebNetII/`,
    `H2SGNN/`, `neural-sheaf-diffusion/`, `Cooperative-Sheaf-NN/`, `fasth/`, `hgbi/`, `dgl/`,
    `tudataset/` — clone targets for the authors' repositories (empty until cloned; leave them).
  - `src/baselines/runtime/vendor_paths.py` (`VENDOR_DIR`, `VENDOR_ROOTS`) is the single place
    that maps each baseline to its folder. `src/baselines/` holds the adapters and campaign code.

Other folders:
- `docs/` — `h2snn_current_model_documentation.html`, the `sheafcovpool_*` notes and
  `roadA_memory_handoff.md` describe the deleted v1 model (tag `v1-submitted`); they are history,
  not a description of `src/`. `pyg_dataset_eda_homophily_report.html` holds the dataset
  homophily EDA the plan cites.
- `notebooks/` — Colab notebooks that the tests check (`tests/test_baselines.py`,
  `tests/test_wandb_and_runs.py`): `baselines_*.ipynb`, the `h2snn_*_colab.ipynb` runners, and
  `notebooks/baselines/`, which `scripts/generate_baseline_notebooks.py` generates (edit the
  generator, then regenerate; never edit those by hand). Do not delete them or commit outputs.
- `folds/` — TU graph-classification splits, read relative to the working directory by
  `src/data/graph_classification_dataset.py` and the legacy CSNN, so run from the repository root.
- `paper/` — manuscript source, figures and bibliography.
- `references/` — reference PDFs: `HETSHEAF.pdf`, `cooperative_sheaf_neural_network.pdf`, and
  `background/` (heterophily, HGB, NSD, TUDataset, HINPool, pooling), `metapath_search/`,
  `rl_and_search/`, `readout/`, `misc/`.

## Running Experiments

Remember the approval rule above before launching anything longer than a smoke test.

```bash
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts/train_h2snn.py \
    --config configs/experiments/<name>.yaml --exp_name <run_name>
```

Configs in `configs/experiments/` (the task follows from `data.dataset_name`):
- Node classification: `acm`, `dblp`, `imdb` (`*_h2snn.yaml`; meta-path presets in
  `NODE_CLASSIFICATION_PRESETS`, `src/configs/config.py`).
- Link prediction: `lastfm`, `amazonbook`, `pubmed`, `youtube`, `amazon`; decoder per dataset
  via `data.decoder_type` (`dot` | `distmult` | `concat_mlp`).
- Graph classification: `mutag`, `enzymes`, `nci1`, `proteins`.

Config sections: `data`, `model`, `ablation` (plan §4.6 switches, all off by default:
`no_path_cells`, `free_path_maps`, `endpoint_edges`, `holonomy_edges`, `uniform_path_gate`,
`per_side_scale`, `h0_only_readout`), `optim` (`scalar_lr_scale`: learning-rate multiple for the
diffusion scalars `θ_η`, `θ_γ`, which get their own group with no weight decay). Unknown keys are
errors. `orthogonal_map` accepts only `fasthpp` (`src/models/fasthpp.py`); the dense map family is
guarded by `model.node_edge_memory_guard_fraction`.

Validate W&B tuning before creating or resuming a sweep (does not contact W&B or train):

```bash
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts/run_h2snn_wandb_tuning.py \
    --config configs/experiments/acm_h2snn.yaml --num_runs 5 --n_trials 30 \
    --d_values 16 32 --validate_only
```

W&B state is versioned and fingerprints the resolved config and search space. With `--resume`,
compatible per-run states resume their sweeps; incompatible ones are preserved and numbered
continuation sweeps are created. An explicit `--sweep_id` is strict exact-resume mode. To split a
campaign across machines, keep `--num_runs` and `--base_seed` fixed and use `--run_start` plus
`--run_count`; never run the same run index concurrently. Graph-classification repeated CV uses
`scripts/run_graph_classification_repeated_cv_wandb_tuning.py` with `--unit_start`/`--unit_count`
(see fix spec §6.1 for the protocol problems with it).

**Legacy baseline** (`CSNN`, homogeneous graphs, `vendor/csnn/`) — comparison runs only. Run
from the repository root (it reads `folds/` and `tud-data/` relative to the working directory)
with `vendor/csnn` on `PYTHONPATH`:
```bash
PYTHONPATH=vendor/csnn "C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python -m exp.run --dataset=amazon_ratings --task=node_level --epochs=2000 --d=4 --layers=3
PYTHONPATH=vendor/csnn bash vendor/csnn/exp/scripts/run_amazon.sh    # node classification
PYTHONPATH=vendor/csnn bash vendor/csnn/exp/scripts/run_enzymes.sh   # graph classification
```
In PowerShell, set `$env:PYTHONPATH = "vendor\csnn"` first.
Its `--orth` options (`householder`, `matrix_exp`, `cayley`, `euler`) apply only to this baseline.

## Testing

Use `unittest`, not bare `pytest`:

```bash
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python -m unittest discover -s tests -v
```

Do not rely on a remembered test count; record the count and result before your change and compare
after it. Any newly failing existing test blocks progress. `tests/test_task_heads.py` compares
node-classification outputs and gradients bit-for-bit with `../docs/heads_evidence/v2_golden.pt`.

On this Windows machine, PyG-dependent tests can be blocked while importing `xxhash._xxhash` by
Application Control. That is an environment-policy blocker, not an H2S-NN defect: report it, run
the full suite in the clean Colab environment, and do not weaken the policy or add model
workarounds. If a test writes trainer artifacts, close file handlers (`ExperimentLogger.close()`)
before deleting temporary directories on Windows.

## Architecture

### H2S-NN — `H2SNN`, `src/models/h2snn_model.py` (plan §4.2)

```
h₀ = TypeEncoder(x)                                               (src/models/node_encoder.py)
maps F_{v◁e} = c_e R_{ψ,side} H_k(u)                              (src/models/sheaf_maps.py)
repeat K steps:
    z_f = Σ_{v∈f} T_v x_v / |f|                                   # 2-cell state
    x  ← x − η D^{-1/2} [ Δ₀ + γ L_P ] D^{-1/2} x                  (src/models/vertex_path_laplacian.py)
    (L_P x)_v = Σ_{f∋v} w_f T_vᵀ (T_v x_v − z_f)                   # vertex–2-cell coupling
representation = [x_K, h₀]
```
`Δ₀` is the sheaf Laplacian on observed edges with the D1 maps; `D` is the per-node scalar degree
of the operator's own terms (per-incidence sum, so self-loops count twice); `T_v` is ordered
parallel transport along the path (`PathTransportOperator`, matrix-free, orthogonal under D1a);
`η = σ(θ_η)`, `γ = softplus(θ_γ)`, `w_f = σ(g(path latent))`.

| Task head | Module | Reads |
|---|---|---|
| Node classification | `H2SNN.readout` (MLP) | `[x_K, h₀]` of target nodes |
| Link prediction | `LinkPredictionReadout` (`src/models/link_readout.py`): endpoint projection `Linear(2d, d) + GELU`, then `dot` / `distmult` / `concat_mlp` | `[x_K, h₀]` of both endpoints; encoder runs on the train-only message graph and its path cells |
| Graph classification | `GraphReadout` (`src/models/graph_readout.py`) | sum of `[x_K, h₀]` per graph → BatchNorm → MLP; path cells enumerated per batch |

Trainers: `src/trainers/trainer.py` (node), `link_prediction_trainer.py`,
`graph_classification_trainer.py`; each logs `eta` and `gamma` per epoch.

### Legacy baseline (`vendor/csnn/`) — homogeneous-graph `CSNN`

Kept only for comparison runs; do not build new features on it. Paths below are relative to
`vendor/csnn/`.

| Module | File | Role |
|---|---|---|
| `CSNN` | `models/coopshv_model.py` | Encoder, sheaf layers, MLP head |
| `CSNNConv` / `DenseCSNNConv` | `models/CSNNconv.py`, `models/DenseCSNN.py` | Sparse / dense sheaf conv |
| `ConformalSheafLearner` | `models/sheaf_learner.py` | Per-edge `d×d` restriction maps |
| `Orthogonal` | `models/orthogonal.py` | Orthogonal parameterizations |
| `load_heterophilic_data` / `load_tud_data` | `utils/data_utils.py` | Dataset loading |
| `custom_train` / `custom_test` | `utils/exp_utils.py` | Training loop, metrics |

The TU graph-classification splits live in `folds/` at the repository root.

## Dependencies

**Use the `h2snn` conda environment (see "Environment" above) — do not `pip install -r
requirements.txt` into `base` or any other environment.** `h2snn` already has everything in
`requirements.txt` installed: PyTorch with CUDA 12.4 (`torch==2.6.*`), `torch_geometric`,
`torch_scatter`, `torch_sparse`, `wandb`, `scikit-learn`.

To add a new dependency:
```bash
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python -m pip install <package>
```
and record it in the right file under `requirements/`.

`requirements.txt` is an aggregator: the pins live in `requirements/`, split so that one
group failing to install disables only the baselines it backs rather than the whole
campaign. `pip install -r requirements.txt` still installs everything.

| file | backs |
|---|---|
| `requirements/core.txt` | `src/`, `scripts/`, `tests/` and every baseline — **required** |
| `requirements/baselines-dgl.txt` | GCN, GAT, HAN, SeHGNN, HINormer; PSHGCN **on link prediction only** |
| `requirements/baselines-sheaf.txt` | CSNN, NSD (`torch_householder`) |
| `requirements/baselines-h2gb.txt` | H2G-former (the H2GB graphgym stack) |
| `requirements/baselines-hetero2net.txt` | Hetero²Net (`torch_cluster`, `texttable`, `termcolor`) |

ChebNetII, H2SGNN, LatGRL, and PSHGCN (node/graph classification) need nothing beyond core.
`src/baselines/runtime/dependencies.py` maps groups to models and is the single source of
truth for that mapping — update it and `tests/test_baselines.py:RequirementGroupTests`
whenever a baseline gains or loses a dependency. The Colab notebooks install core first,
then one group per cell, and drop the models of any group that failed before launching.

DGL is not on PyPI past 2.1.0 and ships no Windows wheel for any 2.x; it is pinned per
platform against its own wheel index (`dgl==2.5.0+cu124` on Linux for torch 2.6, the
already-installed `2.2.1` on Windows). Do not "simplify" that to a bare `dgl==<version>` —
it cannot resolve.

W&B (Weights & Biases) is used for experiment tracking; set `--wandb=0` to disable.

## LaTeX Research-Paper Writing And Editing

These instructions apply automatically whenever creating, revising, reviewing, or fact-checking
`paper/*.tex`, `paper/references.bib`, paper figures or captions, appendices, supplementary
material, or a submission/rebuttal derived from them. Write as a careful AAAI researcher and
reviewer: the paper must be technically exact, easy to follow, and accessible to an AI or computer
science reader who is not a specialist in heterogeneous sheaves or Hodge theory.

### Required Grounding Before Writing

- Read the surrounding manuscript section and the relevant entries in `paper/references.bib` before
  editing. Preserve the paper's established macros, notation, labels, terminology, and venue format
  unless the task explicitly changes them.
- The manuscript describes **v1**. Ground any description of what v1 does in the code on the git
  tag `v1-submitted` (its `src/` code, tests, experiment configs and completed result artifacts), then in
  `docs/h2snn_current_model_documentation.html` (v1 only). Ground any description of the revised
  model in `../docs/H2S-NN_v2_plan.md` and `../docs/H2S-NN_fix_spec.md` (§7, Claims and writing), and describe
  a v2 component as implemented only once its code and tests exist. Treat the manuscript,
  specifications and plans as claims to verify, not as proof that a feature or result exists.
- Use `references/HETSHEAF.pdf`, `references/metapath_search/HeterSEED.pdf`, and
  `references/metapath_search/Hetero2Net.pdf` as rhetorical style references. Extract their structural habits;
  never copy or closely paraphrase their sentences, and never cite them as evidence for a claim
  merely because they are style references.
- If sources, code, and prose conflict, do not silently choose the stronger claim. State or mark the
  discrepancy and use the narrowest claim supported by the current evidence.

### Adversarial Technical Review Mode

When the user asks to review the paper, switch to an
adversarial technical-review stance. Thinking about writing quality includes where wording hides a technical
ambiguity. Focus on algorithmic novelty, mathematical formulation, optimization design,
technical soundness, evaluation validity, and whether the claimed contribution is actually
supported by the implemented method and evidence. Assume the burden of proof is on the paper:
list every reject-level weakness you can find, explain why a skeptical reviewer would care, and
propose concrete repairs that preserve the overall H2S-NN framework rather than replacing it with a
different project. Prioritize fatal conceptual or formulation problems first, then optimization and
algorithmic gaps, then experimental or ablation gaps. Do not soften the critique with reassurance
unless a specific component is technically solid and important for deciding what to keep.

### Story And Section Architecture

- Build the narrative in the order **context -> concrete problem -> why existing mechanisms are
  insufficient -> key intuition -> method -> evidence -> bounded conclusion**. Each paragraph
  should do one job, open with its main point, and end by resolving that point or motivating the
  next paragraph.
- Follow the useful habits of the three reference papers:
  - **HetSheaf:** motivate the geometry in plain language before formalizing it; connect each new
    construction to a specific limitation; position contributions precisely against adjacent work.
  - **HeterSEED:** make the Background, Motivation, Method, and Contributions beats unmistakable;
    use a concrete observation to motivate each mechanism; explain complementary components before
    giving their technical details.
  - **Hetero2Net:** separate heterogeneity from heterophily, define the challenge and measurements
    before drawing conclusions, and organize empirical claims around explicit questions.
- For the introduction, use a narrowing funnel: real heterogeneous graphs; the joint
  heterogeneity--heterophily problem; the precise representational or operator-level gap; the
  H2S-NN intuition; a compact method preview; then non-overlapping, verifiable contributions.
- Explain a method component in the order **need -> intuition -> mathematical mechanism -> effect
  on information flow -> limitation or scope**. Give a plain-language overview and, when helpful, a
  small typed-path example before dense notation.
- Synthesize related work by mechanism and limitation rather than writing a catalogue of one-sentence
  paper summaries. Compare only along dimensions that matter to the claimed gap, and distinguish
  "not designed for this purpose" from "incapable of it."
- Structure experiments around research questions such as effectiveness, component contribution,
  efficiency, robustness, and failure modes. A results paragraph should state the question, cite the
  relevant table or figure, report the concrete observation, and give a restrained interpretation.

### Academic Style And Accessibility

- Prefer direct, concrete sentences and precise verbs. Define specialized terms at first use and
  distinguish frequently confused concepts explicitly, especially heterogeneity versus heterophily,
  a meta-path schema versus a concrete path instance, and a strict cell/face versus the manuscript's
  "face-like" support.
- Introduce equations with their purpose, define every symbol and index, and explain the operational
  meaning immediately afterward. Include tensor shapes when they prevent ambiguity. Do not make the
  reader reverse-engineer an equation from notation alone.
- Use consistent terminology and notation throughout. Preserve established commands such as
  `\method`, `\ampss`, `\thspool`, `\cF`, `\delta_0`, and `\delta_1`; do not create a synonym merely
  for stylistic variation.
- Prefer one strong claim per sentence. Use signposting and transitions, but avoid repeated previews,
  inflated adjectives ("novel," "powerful," "significant," "comprehensive"), vague references
  ("this," "it," or "the above" without a clear noun), and generic openings that delay the problem.
- Use polished academic English without unnecessary jargon. Keep the grammatical voice and English
  variant consistent within the manuscript; do not globally restyle unrelated passages unless asked.
- Captions must be self-contained enough to identify the setup, compared objects, direction of
  improvement, and meaning of error bars or visual encodings. Do not make a caption assert more than
  the figure or its source data shows.

### Claims, Citations, And Research Integrity

- Cite every substantive external claim close to the clause it supports. Prefer the original paper,
  official proceedings version, or authoritative dataset/documentation source; use surveys for broad
  context, not as a substitute for a primary technical source.
- Verify that a cited source actually supports the exact statement. Do not attach a long citation pile
  to a compound sentence when different sources support different clauses. Split the sentence and map
  citations to claims concretely.
- Never invent or guess a citation key, title, author list, venue, year, DOI, page number, result, or
  BibTeX field. Cross-check each new entry against the source PDF and, when needed, an official
  proceedings page or arXiv record. If verification is unavailable, insert an explicit
  `% TODO(citation): ...` note or report the unresolved claim instead of fabricating support.
- Use the citation commands already established by the manuscript and ensure every cited key exists in
  `paper/references.bib`. Before handoff, audit for undefined citations, uncited new bibliography
  entries when relevant, duplicate entries, and keys whose bibliographic identity does not match the
  prose.
- Attribute comparisons fairly: say what another method does before identifying the remaining gap.
  Avoid universal statements such as "all existing methods" unless the review is demonstrably
  exhaustive. Prefer bounded formulations such as "the compared meta-path methods" or "existing
  node-edge sheaf models in our review."
- Label statements accurately as definitions, intuitions, propositions, empirical observations, or
  hypotheses. Do not present intuition as a theorem, correlation as causation, or a correctness test as
  benchmark validation.
- Report quantitative claims only from traceable tables, logs, or source papers. Include the metric,
  dataset/split, comparison scope, and variability or number of runs when available. Never manufacture
  missing values or use placeholders as completed evidence.
- Keep the implementation boundary explicit.
  - **v1 (submitted model, git tag `v1-submitted`):** closure edges are first-class 1-cells inside `delta0`;
    `delta1` spans the full boundary cycle and `delta1 . delta0 = 0` holds by construction;
    edge-to-face maps are built by transport from one free per-instance map `A_f`; faces are simple
    paths; incidence signs are `+1` because enumeration is forward-only; schema preselection is
    non-differentiable. `SheafCovPool` is v1's single-level graph readout; THS-Pool, Galerkin
    coarsening, spectral pooling and hierarchical variants are superseded historical experiments.
  - **v2 wording rules (plan §4.5), for the revision:** the whole 2-complex carries a genuine
    cellular sheaf (functoriality enforced by transport); states are cochains, not sections, so write
    "diffuses over a cellular sheaf on the 2-complex, including its node→2-cell maps", never "the
    model makes the graph a sheaf". Call the node operator the **vertex–2-cell sheaf Laplacian**, not
    the Hodge Laplacian of the 2-complex (Lemma A′ rules that out). The closure edge belongs to the
    definition of the 2-cell, not the computation. Distinguish the construction from sheaf
    hypergraph models (Duta et al. 2023) and MAGNN as plan §4.5 describes.
  - **Scope of claims:** follow fix spec §7 decision A/B on whether "heterophilic" stays in the
    framing; SheafCovPool and graph classification move to the appendix or a separate paper.
- Never claim broad state of the art, scalability, parameter efficiency, theoretical guarantees, or
  generalization beyond the datasets and evidence actually evaluated. Prefer "best among the compared
  methods under this protocol" over an unqualified "state of the art."

### Safe LaTeX Editing And Verification

- Preserve valid LaTeX structure, cross-references, labels, bibliography style, and the current
  `acmart` setup unless the user requests a venue conversion. Use nonbreaking references such as
  `Table~\ref{...}`, `Figure~\ref{...}`, and `Section~\ref{...}` consistently.
- Edit source files only. Do not hand-edit generated `.aux`, `.bbl`, `.blg`, `.fdb_latexmk`, `.fls`,
  `.log`, `.out`, `.pdf`, or `.synctex.gz` files.
- After a substantive paper edit, perform a claim--evidence pass, a notation/definition pass, a
  grammar/story pass, and a duplication pass. Remove repeated related-work summaries and repeated
  contribution claims unless the repetition serves a clear navigational purpose.
- When the LaTeX toolchain is available, compile from `paper/` with
  `latexmk -pdf -interaction=nonstopmode -halt-on-error h2snn_manuscript.tex`. Inspect the log for
  undefined citations/references, multiply defined labels, missing files, and material layout
  warnings. Report any blocker; do not conceal it by weakening content or editing generated files.
