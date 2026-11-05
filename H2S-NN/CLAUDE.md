# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
| ❌ Never omit type annotations |0 Makes code unreadable and untestable |
| ❌ Never commit `experiments/` to git | Use `.gitignore` |

## Project Overview

**H2S-NN** (Heterophilic Heterogeneous Sheaf-Hodge Neural Network) lifts a heterogeneous
information network (typed nodes, typed directed edges) into a typed 2D cellular complex, learns
relation-aware sheaf restriction maps (node→edge and edge→face), and runs Hodge-style diffusion
over node and edge cochains to classify a target node type. Optional AMPSS discovery uses a
non-differentiable task-aligned Stage-I probe selector followed by task-trained,
schema-conditioned instance weights under protocol `schema_conditioned_v1`.

**Two codebases live in this repository — know which one you're touching:**
- **`src/`, `scripts/train_h2snn.py`, `tests/`** — the current, actively developed H2S-NN
  (heterogeneous, schema-discovery-capable). This is "the project" for almost all new work.
- **`models/`, `exp/`, `utils/`** — an earlier homogeneous-graph sheaf-diffusion baseline (`CSNN`,
  run via `python -m exp.run`), kept only for benchmark comparisons. Do not add new features here
  unless the task is specifically about the baseline comparison.

Key references:
- `docs/h2snn_current_model_documentation.html` — **living document**, treat as more current than
  this file for anything beyond the high-level summary below. Covers the current pipeline,
  file-by-file architecture status, research positioning against related work, and an honest list
  of known limitations/bugs (e.g. closure edges vs. orientation signs in `δ1`, AMPSS
  selection and weighting). Update it whenever a change in `src/` affects a correctness claim; read it before
  making claims about what the model currently does.
- `H2SNN_corrected_algorithm.md` — authoritative mathematical reference with notation table and
  corrected algorithms.
- `H2S-NN_implementation_spec.md` — full implementation specification with pseudocode and tensor
  shapes.
- `cooperative_sheaf_neural_network.pdf` — the research paper.

## Running Experiments

**Current H2S-NN** (heterogeneous, `src/`) — always through the `h2snn` conda environment:
```bash
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts/train_h2snn.py \
    --config configs/experiments/<name>.yaml --exp_name <run_name>
```
Datasets: `acm`, `dblp`, `imdb` (see `src/configs/config.py: DATASET_PRESETS`). Schema discovery
(auto meta-path selection) is controlled by `SchemaDiscoveryConfig` (`discovery.enabled`,
`discovery.mode = manual|auto`) in the same config file.

Active auto-discovery configs must use `protocol_version: schema_conditioned_v1`,
`selector_strategy: probe`, and an `instance_weight_mode` in
`{schema_conditioned, schema_agnostic, uniform}`. Stage I greedily selects schemas by held-out
task gain under hard schema, instance, and incidence budgets. Coverage is stored as sorted sparse
`(schema, unit)` incidence for diagnostics:
post-quota target-node support for node classification, a fixed seeded positive-training-edge
subset for link prediction, and batch-local graph slots for graph classification. Stage II learns
positive weights `2*sigmoid(g_w(z_f || e_schema))` and ramps the upper-Hodge operator from
uniform to learned weights. AMPSS has no schema gate or auxiliary discovery loss; training uses
the task loss only. Selector and materialization seeds are separate from training seeds;
Stage-II ablations must reuse the same selector manifest and materialized-face hash. Legacy
gate/loss keys and legacy checkpoints/tuning states are rejected.

The current `src/` model accepts only `orthogonal_map: fasthpp`, including an exact
non-power-of-two fallback implemented in `src/models/fasthpp.py`. Other orthogonal-map API
choices belong only to the legacy baseline. Node-to-edge map allocation is guarded by
`model.node_edge_memory_guard_fraction` (default `0.20`). W&B/Optuna defaults use `d in {16,32}`
for full-graph tasks and `{16,32,64}` for graph classification.

Node- and link-prediction primary tuning remains full-graph; there is no query-batch-size
tuning parameter. SheafCovPool consumes a fixed raw `3x3` Gram descriptor determined by its
three semantic views. With fixed `data.pooling_type: sheafcov`, graph-classification sweeps may
tune only the post-Gram projection width via `--sheafcov_feat_dims 4 8 16`; do not combine that
flag with `--include_pooling_type`.

Validate W&B tuning before creating or resuming a sweep:

```bash
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts/run_h2snn_wandb_tuning.py \
    --config configs/experiments/acm_h2snn_ampss.yaml --num_runs 5 --n_trials 30 \
    --d_values 16 32 --validate_only
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts/run_h2snn_wandb_tuning.py \
    --config configs/experiments/mutag_h2snn.yaml --num_runs 5 --n_trials 30 \
    --d_values 16 32 64 --sheafcov_feat_dims 4 8 16 --validate_only
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python scripts/run_graph_classification_repeated_cv_wandb_tuning.py \
    --config configs/experiments/mutag_h2snn.yaml --num_repetitions 10 --num_folds 10 \
    --n_trials 30 --unit_start 0 --unit_count 1 --validate_only
```

Graph-classification configs still default to five independent stratified 80/10/10 holdout
runs. The separate repeated-CV script uses unstratified shuffled KFold with 10 folds and 10
repetitions, an inner 10% validation split, and one independent W&B sweep per repetition/fold
pair. A complete campaign therefore creates 100 sweeps and approximately `100 * n_trials`
model fits. Partition it across machines with `--unit_start` and `--unit_count`, keeping all
campaign and seed arguments fixed.

W&B state is versioned and fingerprints the resolved config/search space. With `--resume`,
compatible per-run states resume their original sweeps, missing runs start fresh sweeps, and
incompatible existing runs are preserved while numbered continuation sweeps are created. An
explicit `--sweep_id` remains strict exact-resume mode. Each trial runs in a fresh Python worker
process. To partition one campaign across machines, keep `--num_runs` and `--base_seed` fixed and
use `--run_start` plus `--run_count`; never execute the same run index concurrently.

**Legacy baseline** (`CSNN`, homogeneous graphs, `models/`/`exp/`) — kept for comparison runs only:
```bash
"C:\Users\Acer\anaconda3\Scripts\conda.exe" run -n h2snn python -m exp.run --dataset=amazon_ratings --task=node_level --epochs=2000 --d=4 --layers=3

# Or use preset scripts
bash exp/scripts/run_amazon.sh      # node classification
bash exp/scripts/run_enzymes.sh     # graph classification
```

**Task types**: `--task=node_level` or `--task=graph_level`

**Key baseline model arguments**:
- `--d` : stalk dimension (restriction map size d×d)
- `--layers` : number of sheaf convolutional layers
- `--hidden_channels` : feature embedding dimension
- `--orth` : legacy-baseline-only parameterization (`householder`, `matrix_exp`, `cayley`,
  `euler`); this does not apply to the current `src/` H2SNN, where `matrix_exp` is rejected
- `--gnn_type` : sheaf learner backbone (`SAGE`, `GCN`, `GAT`, `SGC`, `NNConv`, `SumGNN`, `GPS`)
- `--conformal` : enable conformal scaling in restriction maps

## Architecture

> Full stage-by-stage detail, file-level implementation status, and known limitations live in
> `docs/h2snn_current_model_documentation.html` — treat it as more current than this section for
> anything beyond the high-level map below.

### Current model (`src/`) — data flow through `H2SNN` (`src/models/h2snn_model.py`)

```
Typed node features [N, in_channels] + typed edges + meta-path face groups
  → TypeSpecificNodeEncoder                            (src/models/node_encoder.py)
  → NodeToEdgeSheaf: relation-aware src/tgt maps        (src/models/sheaf_maps.py)
  → EdgeInitializer: edge cochain s0                    (src/models/sheaf_maps.py)
  → SequenceEdgeToFaceSheaf: positional-concat MLP → A_f (src/models/sheaf_maps.py)
  → [auto mode] InstanceWeightNetwork: task-trained w_f in (0,2) (src/models/instance_weights.py)
  → CoboundaryOps: matrix-free δ0, δ1, L0, L1            (src/models/coboundary_ops.py)
  → HodgeDiffusion: K-step Euler diffusion               (src/models/diffusion.py)
  → NodeReadout: MLP over [x*, h0, pooled edge signal]   (src/models/readout.py)
  → logits
```

| Module | File | Role |
|---|---|---|
| `H2SNN` | `src/models/h2snn_model.py` | Wires the full forward pass end to end |
| `TypeSpecificNodeEncoder` | `src/models/node_encoder.py` | Per-node-type linear + LayerNorm encoder into shared stalk dim `d` |
| `NodeToEdgeSheaf` | `src/models/sheaf_maps.py` | Learns directed, relation-typed source/target restriction maps |
| `EdgeInitializer` | `src/models/sheaf_maps.py` | Produces initial edge cochain `s0` |
| `SequenceEdgeToFaceSheaf` | `src/models/sheaf_maps.py` | Positional concat MLP (one per path length) over each meta-path instance → the free map `A_f`, from which edge-to-face maps follow by transport. `model.face_encoder='lstm'` restores the superseded LSTM+attention encoder for ablation |
| `InstanceWeightNetwork` | `src/models/instance_weights.py` | Produces task-trained instance weights from ordered path latents and, in the full model, schema embeddings; weights affect only the upper-Hodge operator |
| `CoboundaryOps` | `src/models/coboundary_ops.py` | Matrix-free `δ0`, `δ1`, `L0`, `L1` as operators — never materialized as dense matrices |
| `HodgeDiffusion` | `src/models/diffusion.py` | `K`-step Euler diffusion with (optionally learnable) step sizes |
| `NodeReadout` | `src/models/readout.py` | Final MLP classifier over pooled node/edge state |
| AMPSS Stage I | `src/data/node_stage1.py`, `link_stage1.py`, `graph_classification_dataset.py` | Task-aligned probe selection, sparse task coverage, exact post-quota costs, and immutable selector reports |
| `Trainer` | `src/trainers/trainer.py` | Task-specific supervised loss, metrics, optimization, and checkpointing |

### Legacy baseline (`models/`, `exp/`, `utils/`) — homogeneous-graph `CSNN`

Kept only for comparison against heterophilic/TUD benchmarks; do not build new features on it.

```
Input node features [N, in_channels]
  → Linear encoder → GeLU  [N, hidden_channels × d]
  → ConformalSheafLearner: learns d×d restriction maps per edge
  → CSNNConv × layers: sheaf Hodge diffusion with orthogonal transforms
  → LayerNorm/BatchNorm (optional)
  → MLP classifier → logits
```

| Module | File | Role |
|---|---|---|
| `CSNN` | `models/coopshv_model.py` | Main model — stacks encoder, sheaf layers, MLP head |
| `CSNNConv` | `models/CSNNconv.py` | Sparse sheaf conv layer using `MessagePassing` |
| `DenseCSNNConv` | `models/DenseCSNN.py` | Dense variant using full Laplacian + `torch_sparse.spmm` |
| `ConformalSheafLearner` | `models/sheaf_learner.py` | Learns per-edge restriction maps; supports RWSE/LapPE positional encodings |
| `Orthogonal` | `models/orthogonal.py` | Maps unconstrained params to orthogonal matrices |
| `SheafDiffusion` | `models/sheaf_base.py` | Base class holding shared config |
| `load_heterophilic_data` / `load_tud_data` | `utils/data_utils.py` | Dataset loading for node-level and graph-level tasks |
| `custom_train` / `custom_test` | `utils/exp_utils.py` | Training loop, metrics, loss functions |

`ConformalSheafLearner` computes a d×d matrix per directed edge (u→v):
1. GNN backbone encodes nodes to embeddings
2. Edge embeddings are formed by concatenating endpoint features
3. An MLP maps to d² parameters, reshaped and passed through `Orthogonal`
4. Optional conformal scaling multiplies by a learned scalar exp(log_scale)

### Datasets

- **Current model (`src/`)**: ACM, DBLP, IMDB heterogeneous benchmarks loaded via PyTorch Geometric
  (`src/data/dataset.py`); node classification on one target node type per dataset.
- **Legacy baseline (`models/`/`exp/`)**:
  - Node-level: Cora, PubMed, and heterophilic benchmarks (Squirrel, Chameleon, Actor, Wisconsin, Texas, Cornell, Roman Empire, Amazon Ratings, Tolokers, Questions). Splits stored in `folds/`.
  - Graph-level (TUD benchmarks): ENZYMES, IMDB-BINARY, IMDB-MULTI, NCI1, PROTEINS, REDDIT-BINARY. Preloaded `.pt` files in `tud-data/`.
  - All datasets use 10-fold cross-validation. Evaluation metrics: accuracy (multiclass), ROC-AUC / AP (binary).

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
- Ground descriptions of the current model first in
  `docs/h2snn_current_model_documentation.html`, then in the implemented `src/` code, tests,
  experiment configs, and completed result artifacts. Treat the manuscript, specifications, and
  plans as claims to verify, not as proof that a feature or result exists.
- Use `HETSHEAF.pdf`, `research_paper/HeterSEED.pdf`, and
  `research_paper/Hetero2Net.pdf` as rhetorical style references. Extract their structural habits;
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
- Keep the implementation boundary explicit. As of the Road A refactor: **closure edges are
  first-class 1-cells** inside `delta0` (the separate `closure_L0` correction is deleted), so
  `delta1` spans the **full boundary cycle** (`L` path edges plus the closure edge), and
  **`delta1 . delta0 = 0` holds exactly by construction** — verified at ~1e-7 relative (float32
  roundoff) on ACM/DBLP/IMDB and on the AMPSS path. Edge-to-face maps are therefore **functorial**:
  built by transport from one free per-instance map `A_f`, not emitted freely per (edge, face).
  Faces are **simple paths** (no revisits, endpoints included), enforced in all three enumerators.
  Incidence signs are uniformly `+1`, which is correct *because* enumeration is forward-only —
  asserted by `validate_face_orientation`, not assumed. Still true: schema preselection is
  non-differentiable. No longer true: per-edge attention inside a face (incompatible with the
  telescoping), and the `lambda_c`/`lambda_co` loss terms (both removed — see
  `src/trainers/regularizers.py` for why enforcing consistency would make faces vacuous).
  `SheafCovPool` is the finalized single-level graph readout: invariant Gram descriptors feed
  attention-weighted means and covariances. THS-Pool, Galerkin coarsening, spectral pooling, and
  hierarchical variants are superseded historical experiments; do not describe them as the current readout.
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
